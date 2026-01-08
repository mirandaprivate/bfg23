use modgkr_lib::utils::*;
use modgkr_lib::general_prover::*;
use modgkr_lib::{LayerInfo, LayerInfoDense, ModelExecution};
use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::MultilinearExtension;
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::UniformRand;
use rand::thread_rng;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}, Mutex, OnceLock};
use std::io::{self, Write};

use subroutines::{
    MultilinearKzgPCS,
    PolynomialCommitmentScheme,
};

// 定义常量
const N: usize = 1024; // N 必须是 2 的幂
const LAYERS: usize = 4;

// 简单的内存监测
fn spawn_mem_monitor(sample: Duration) -> (Arc<AtomicBool>, Arc<AtomicU64>, std::thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_cloned = stop.clone();
    let peak_cloned = peak.clone();
    let handle = std::thread::spawn(move || {
        while !stop_cloned.load(Ordering::Relaxed) {
            if let Some(rss) = read_rss_kb() {
                let _ = peak_cloned.fetch_max(rss, Ordering::Relaxed);
            }
            std::thread::sleep(sample);
        }
    });
    (stop, peak, handle)
}

fn read_rss_kb() -> Option<u64> {
    // 每 30 秒打印一次心跳符号
    static LAST_HEARTBEAT: OnceLock<Mutex<Instant>> = OnceLock::new();
    if let Ok(mut t) = LAST_HEARTBEAT.get_or_init(|| Mutex::new(Instant::now())).lock() {
        if t.elapsed() >= Duration::from_secs(30) {
            print!("⌛");
            let _ = io::stdout().flush();
            *t = Instant::now();
        }
    }

    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(val) = parts.get(0) {
                if let Ok(num) = val.parse::<u64>() {
                    return Some(num);
                }
            }
        }
    }
    None
}

fn main() {
    let mut rng = thread_rng();
    let (stop_mem, peak_mem, mem_handle) = spawn_mem_monitor(Duration::from_millis(500));
    
    // 0. 初始化 KZG SRS
    println!("正在初始化 KZG SRS (max_vars={})...", 2 * (N.trailing_zeros() as usize));
    let max_vars = 2 * (N.trailing_zeros() as usize);
    // hyperplonk PCS 提供了测试用 SRS 生成函数
    let params =
        MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, max_vars).unwrap();
    // supported_degree 传 None，num_vars 传 Some(max_vars)
    let (ck, vk) =
        MultilinearKzgPCS::<Bls12_381>::trim(&params, None, Some(max_vars)).unwrap();

    // 1. 生成 MLP 架构数据并提交参数
    println!("正在生成 {} 层 MLP 架构 (n={}, 保持维度为 2 的幂)...", LAYERS, N);
    let mut model_exec: ModelExecution<Fr> = Vec::new();
    let mut kernel_commits = Vec::new();
    let mut kernel_mles = Vec::new();
    let mut commit_time_us: u128 = 0;
    let mut proof_size_bytes: usize = 0;
    let mut proof_size_sumcheck_bytes: usize = 0;
    let mut proof_size_kzg_bytes: usize = 0;
    
    // 初始输入 X: 1 x N，最后一个元素设为 1 用于 bias trick
    let mut current_vec = vec![Fr::rand(&mut rng); N];
    current_vec[N-1] = Fr::one();
    
    for i in 0..LAYERS {
        // 层输入：1通道，1行，N列
        let layer_input = vec![vec![current_vec.clone()]];
        
        // 生成权重矩阵 N x N
        let mut kernel = vec![vec![Fr::rand(&mut rng); N]; N];
        for row in 0..N-1 {
            kernel[row][N-1] = Fr::zero();
        }
        kernel[N-1][N-1] = Fr::one();
        
        // --- Commitment 环节 ---
        // 对权重进行多线性扩展并生成承诺
        let now_commit = Instant::now();
        let (mle_kernel, _, _) = matrix_to_mle(kernel.clone(), (N, N), true); 
        let rc_mle_kernel = Rc::new(mle_kernel.clone());
        let commit = MultilinearKzgPCS::<Bls12_381>::commit(&ck, &rc_mle_kernel).unwrap();
        commit_time_us += now_commit.elapsed().as_micros();
        kernel_commits.push(commit);
        kernel_mles.push(mle_kernel);
        // -----------------------

        // 计算输出: Y = X * W
        let input_arr = from_matrix_to_arr2(vec![current_vec.clone()], (1, N));
        let kernel_arr = from_matrix_to_arr2(kernel.clone(), (N, N));
        let output_arr = input_arr.dot(&kernel_arr);
        let output_matrix = from_arr2_to_matrix(output_arr, (1, N));
        
        model_exec.push(LayerInfo::LID(LayerInfoDense {
            name: format!("MLP_Layer_{}", i),
            input: layer_input,
            kernel: kernel,
            output: output_matrix.clone(),
            dim_input: (1, N),
            dim_kernel: (N, N),
            dim_output: (1, N),
        }));
        
        // 更新下一层的输入
        current_vec = output_matrix[0].clone();
    }
    
    // 2. 准备证明参数
    println!("架构生成完成。准备挑战点...");
    let last_layer_output = match &model_exec.last().unwrap() {
        LayerInfo::LID(l) => l.output.clone(),
        _ => unreachable!(),
    };
    
    let (mle_output, nv0, nv1) = matrix_to_mle(last_layer_output, (1, N), false);
    let mut output_randomness = Vec::new();
    for _ in 0..(nv0 + nv1) {
        output_randomness.push(Fr::rand(&mut rng));
    }
    let model_output_eval = mle_output.evaluate(&output_randomness).unwrap();
    
    // 3. 执行证明器
    println!("开始生成 Sumcheck 证明...");
    let start_prove = Instant::now();
    let prover = GeneralProver::<Bls12_381, Fr>::new(
        model_exec.clone(),
        Vec::new(),
        output_randomness.clone(),
    );
    let mut fs_rng = Blake2s512Rng::setup();
    let (prover_output, _times) = prover.streaming_prove_all_layers(&mut fs_rng);
    let sumcheck_time_ms = start_prove.elapsed().as_millis();
    println!("证明生成完成（Sumcheck），耗时: {} ms", sumcheck_time_ms);

    // 统计 Sumcheck 部分的 Proof 大小
    for output in &prover_output {
        match output {
            ProverOutput::DenseOutput(po) => {
                proof_size_sumcheck_bytes += po.claimed_values.serialized_size();
                for msg in &po.prover_msgs {
                    proof_size_sumcheck_bytes += msg.serialized_size();
                }
            }
            _ => {}
        }
    }
    
    // 4. 执行验证器并打开承诺
    println!("开始验证 Sumcheck 并打开 KZG 承诺...");
    let mut open_time_us: u128 = 0;
    let mut verify_sc_us: u128 = 0;    // Sumcheck 验证时间
    let mut verify_kzg_us: u128 = 0;   // KZG 验证时间（不含 open）
    let mut verifier_fs_rng = Blake2s512Rng::setup();
    
    let mut current_output_eval = model_output_eval;
    let mut current_output_randomness = output_randomness.clone();
    
    let mut commitment_mismatch = false;

    // 逆序遍历每一层（Sumcheck 证明是从输出层向输入层进行的）
    for (i, output) in prover_output.iter().enumerate().rev() {
        // 由于 prover_output 是按照 prove 的顺序排列的（也是逆序的），
        // 这里的 i = 0 对应的是输出层（最后加进去的层）。
        // 原始 model_exec 的层索引应为 LAYERS - 1 - i
        let layer_idx = LAYERS - 1 - i;
        match output {
            ProverOutput::DenseOutput(po) => {
                // 1. 验证本层的 Sumcheck 证明
                let mut layer_verifier = modgkr_lib::layer_verifier::LayerVerifier::new(
                    po.clone(),
                    current_output_eval,
                );
                let now_sc = Instant::now();
                let sc_randomness = layer_verifier.verify_SC(&mut verifier_fs_rng).unwrap();
                verify_sc_us += now_sc.elapsed().as_micros();
                
                // 2. 准备打开承诺的点 (Opening Point)
                // 在 ProverMatMul 中，评估点是 [init_rand_b, sc_randomness]
                // 其中 init_rand_b 是上一层（靠近输出的方向）传下来的挑战点
                let opening_point = [current_output_randomness.clone(), sc_randomness.clone()].concat();
                
                // 3. 执行 Opening Proof
                let rc_mle_kernel = Rc::new(kernel_mles[layer_idx].clone());
                let now_open = Instant::now();
                let (opening_proof, opened_value) = MultilinearKzgPCS::<Bls12_381>::open(&ck, &rc_mle_kernel, &opening_point).unwrap();
                open_time_us += now_open.elapsed().as_micros();
                // 统计 KZG 打开部分的大小（proof + opened value）
                proof_size_kzg_bytes += opening_proof.serialized_size();
                proof_size_kzg_bytes += opened_value.serialized_size();
                
                // 4. 验证打开的一致性
                // po.claimed_values.1 是 Sumcheck 过程中对右矩阵（权重）的评估声明
                if opened_value != po.claimed_values.1 {
                    commitment_mismatch = true;
                    println!(
                        "警告：层 {} 的承诺打开值与 Sumcheck 声明值不一致（继续验证后续层）。",
                        layer_idx
                    );
                }
                
                // 5. 验证 KZG Proof
                let now_kzg = Instant::now();
                let verif_ok = MultilinearKzgPCS::<Bls12_381>::verify(
                    &vk,
                    &kernel_commits[layer_idx],
                    &opening_point,
                    &opened_value,
                    &opening_proof,
                )
                .unwrap();
                verify_kzg_us += now_kzg.elapsed().as_micros();
                if !verif_ok {
                    commitment_mismatch = true;
                    println!(
                        "警告：层 {} 的 KZG 承诺验证失败（继续验证后续层）。",
                        layer_idx
                    );
                }
                
                // 更新状态以验证前一层
                current_output_eval = po.claimed_values.0;
                current_output_randomness = sc_randomness;
            }
            _ => panic!("不支持的层输出类型"),
        }
    }
    
    if commitment_mismatch {
        println!("验证完成，但存在承诺/打开不一致或验证失败，请检查上述警告。");
    } else {
        println!("全部验证通过！{} 层全连接层的参数承诺已成功打开并验证。", LAYERS);
    }
    let commit_time_ms = commit_time_us as f64 / 1000.0;
    let open_time_ms = open_time_us as f64 / 1000.0;           // 打开承诺（计入证明端）
    let prove_sumcheck_ms = sumcheck_time_ms as f64;           // 证明端 Sumcheck
    let prove_total_ms = prove_sumcheck_ms + open_time_ms;     // 证明端 = sumcheck + open（不含 commit）
    let verify_sc_ms = verify_sc_us as f64 / 1000.0;           // 验证 Sumcheck
    let verify_kzg_ms = verify_kzg_us as f64 / 1000.0;         // 验证 KZG（不含 open）
    let verify_total_ms = verify_sc_ms + verify_kzg_ms;
    let total_ms = commit_time_ms + prove_total_ms + verify_total_ms;

    println!("证明耗时（sumcheck + open，不含 commit）: {:.2} ms", prove_total_ms);
    println!("  其中 sumcheck {} ms + open {:.2} ms，commit 另计 {:.2} ms", sumcheck_time_ms, open_time_ms, commit_time_ms);
    println!("验证耗时（Sumcheck + KZG，均不含 open）: {:.2} ms", verify_total_ms);
    println!("  其中 sumcheck {:.2} ms + kzg {:.2} ms", verify_sc_ms, verify_kzg_ms);
    println!(
        "总耗时：commit {:.2} ms + prove {:.2} ms + verify {:.2} ms = {:.2} ms（不含模型生成）",
        commit_time_ms,
        prove_total_ms,
        verify_total_ms,
        total_ms
    );
    proof_size_bytes = proof_size_sumcheck_bytes + proof_size_kzg_bytes;
    println!(
        "Proof 大小：总 {} 字节（约 {:.2} KB） = Sumcheck {} 字节 + KZG 打开 {} 字节",
        proof_size_bytes,
        proof_size_bytes as f64 / 1024.0,
        proof_size_sumcheck_bytes,
        proof_size_kzg_bytes
    );

    // 停止内存监测并打印峰值
    stop_mem.store(true, Ordering::Relaxed);
    let _ = mem_handle.join();
    let peak_kb = peak_mem.load(Ordering::Relaxed);
    println!("峰值内存: {} KB（约 {:.2} MB）", peak_kb, peak_kb as f64 / 1024.0);
}
