//! Comprehensive Cholesky factorization benchmark using GPU profiling
//!
//! Benchmarks blocked Cholesky decomposition on SPD matrices,
//! measuring end-to-end performance and component breakdown.
//!
//! Key metrics:
//! - GFLOP/s (theoretical: n³/3 for Cholesky)
//! - GPU time (median/min)
//! - Memory bandwidth utilization
//! - Scalability across matrix sizes

use cubecl_core::{prelude::*, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_linalg::{cholesky, Triangle, F32Precision};

#[cfg(feature = "cuda")]
type BenchRuntime = cubecl_cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BenchRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
type BenchRuntime = cubecl_cpu::CpuRuntime;

/// Create a diagonally dominant SPD matrix on GPU
///
/// Strategy: A = I*scale + noise
/// This ensures eigenvalues > scale, making it SPD
fn create_spd_matrix(client: &ComputeClient<<BenchRuntime as Runtime>::Server>, n: usize, scale: f32) -> TensorHandle<BenchRuntime> {
    let size = n * n;
    let mut values = vec![0.0_f32; size];

    // Make diagonally dominant: A[i,i] = scale, A[i,j] = 0.1 for i != j
    for i in 0..n {
        values[i * n + i] = scale;
        for j in 0..i {
            let noise = 0.05 * ((i + j) as f32 / n as f32);
            values[i * n + j] = noise;
            values[j * n + i] = noise; // Symmetric
        }
    }

    let handle = client.create_from_slice(f32::as_bytes(&values));
    // TensorHandle::new(handle, shape, strides, storage)
    // For row-major n×n: shape=[n,n], strides=[n, 1]
    TensorHandle::new(handle, vec![n, n], vec![n, 1], f32::as_type_native_unchecked())
}

/// Benchmark Cholesky factorization for a specific matrix size
fn bench_cholesky(n: usize, scale: f32) {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║ Matrix size: {}x{} (scale: {:.1})", n, n, scale);
    println!("╚═══════════════════════════════════════════════════════╝");

    // Create SPD matrix
    let a = create_spd_matrix(&client, n, scale);

    // Memory footprint
    let matrix_mb = (n * n * 4) as f64 / 1_000_000.0;
    println!("  Memory: {:.2} MB", matrix_mb);

    // Theoretical FLOP count for Cholesky: n³/3 + lower order terms
    let flops = (n as f64).powi(3) / 3.0;
    println!("  Theoretical FLOPs: {:.2} GFLOP", flops / 1e9);

    // Warmup
    for _ in 0..3 {
        let _ = cholesky::<BenchRuntime, F32Precision>(&client, a.as_ref(), Triangle::Lower, false);
    }
    future::block_on(client.sync());

    // Benchmark using GPU profiling
    let mut gpu_times = Vec::new();
    let num_runs = if n < 512 { 20 } else { 10 };

    for _ in 0..num_runs {
        future::block_on(client.sync());

        let result = client.profile(|| {
            cholesky::<BenchRuntime, F32Precision>(&client, a.as_ref(), Triangle::Lower, false)
        }, "cholesky");

        match result {
            Ok(profile_duration) => {
                let ticks = future::block_on(profile_duration.resolve());
                let duration = ticks.duration();
                gpu_times.push(duration);
            }
            Err(e) => {
                println!("  ❌ Profile error: {:?}", e);
                return;
            }
        }
    }

    // Statistics
    gpu_times.sort();
    let median = gpu_times[gpu_times.len() / 2];
    let min = gpu_times[0];
    let max = gpu_times[gpu_times.len() - 1];

    let median_ms = median.as_secs_f64() * 1000.0;
    let min_ms = min.as_secs_f64() * 1000.0;
    let max_ms = max.as_secs_f64() * 1000.0;

    // Performance metrics
    let gflops_median = flops / median.as_secs_f64() / 1e9;
    let gflops_min = flops / min.as_secs_f64() / 1e9;

    // Bandwidth (read A once, write L once)
    let bytes_accessed = (n * n * 4 * 2) as f64; // Read + Write
    let bandwidth_gb_s = bytes_accessed / median.as_secs_f64() / 1e9;

    println!("\n  ⏱️  GPU Timing:");
    println!("      Median: {:.3} ms", median_ms);
    println!("      Min:    {:.3} ms", min_ms);
    println!("      Max:    {:.3} ms", max_ms);

    println!("\n  🚀 Performance:");
    println!("      {:.2} GFLOP/s (median)", gflops_median);
    println!("      {:.2} GFLOP/s (peak)", gflops_min);
    println!("      {:.2} GB/s bandwidth", bandwidth_gb_s);

    // Arithmetic intensity: FLOP/byte
    let arithmetic_intensity = flops / bytes_accessed;
    println!("\n  📊 Efficiency:");
    println!("      Arithmetic intensity: {:.2} FLOP/byte", arithmetic_intensity);

    // Verify correctness (optional, just check it ran)
    match cholesky::<BenchRuntime, F32Precision>(&client, a.as_ref(), Triangle::Lower, false) {
        Ok((l, info)) => {
            println!("      Conditioning: {:.2e}", info.condition_estimate.unwrap_or(1.0));
            println!("      ✅ Factorization successful");
        }
        Err(e) => {
            println!("      ❌ Error: {:?}", e);
        }
    }
}

/// Scalability test across different matrix sizes
fn bench_scalability() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║         Cholesky Factorization Benchmark              ║");
    println!("║            GPU Performance Analysis                   ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let device: <BenchRuntime as Runtime>::Device = Default::default();
    println!("\nDevice: {:?}", device);

    // Test sizes covering different scenarios
    let sizes = vec![
        (32, 50.0),    // Small: tests overhead
        (64, 50.0),    // Small-medium: first panel
        (128, 50.0),   // Medium: one full block
        (256, 50.0),   // Medium: multiple blocks
        (512, 50.0),   // Large: block algorithm shines
        (1024, 50.0),  // Very large: GEMM dominates
        (2048, 50.0),  // Huge: full SOTA test
    ];

    for (n, scale) in sizes {
        bench_cholesky(n, scale);
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║                   Benchmark Complete                  ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

/// Conditioning test: how does performance change with condition number?
fn bench_conditioning() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║      Conditioning Impact on Performance               ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let n = 512;

    // Test different diagonal scaling (affects conditioning)
    let scales = vec![
        (10.0, "well-conditioned"),
        (50.0, "moderate"),
        (100.0, "high condition"),
    ];

    for (scale, desc) in scales {
        println!("\n{} (diagonal scale: {:.1})", desc, scale);
        bench_cholesky(n, scale);
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║    CubeCL Linear Algebra - Cholesky Benchmark         ║");
    println!("║              Phase 1: SOTA Performance                ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    // Main scalability benchmark
    bench_scalability();

    // Conditioning impact
    bench_conditioning();

    println!("\n🎉 All benchmarks complete!\n");
}
