//! Comprehensive LU factorization benchmark using GPU profiling
//!
//! Benchmarks blocked LU decomposition with partial pivoting on general matrices,
//! measuring end-to-end performance and component breakdown.
//!
//! Key metrics:
//! - GFLOP/s (theoretical: 2n³/3 for LU)
//! - GPU time (median/min)
//! - Memory bandwidth utilization
//! - Scalability across matrix sizes

use cubecl_core::{prelude::*, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_linalg::{lu_factor, LUConfig, F32Precision};

#[cfg(feature = "cuda")]
type BenchRuntime = cubecl_cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BenchRuntime = cubecl_wgpu::WgpuRuntime;

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
type BenchRuntime = cubecl_cpu::CpuRuntime;

/// Create a random general matrix on GPU
///
/// Strategy: Random values [0,1] to avoid special structure
fn create_test_matrix(client: &ComputeClient<<BenchRuntime as Runtime>::Server>, n: usize) -> TensorHandle<BenchRuntime> {
    let size = n * n;
    let mut values = vec![0.0_f32; size];

    // Fill with pseudo-random values
    for i in 0..n {
        for j in 0..n {
            values[i * n + j] = ((i * 7 + j * 13) % 100) as f32 / 100.0 + 1.0;
        }
        // Make diagonally dominant to avoid singularity
        values[i * n + i] = n as f32 + 10.0;
    }

    let handle = client.create_from_slice(f32::as_bytes(&values));
    TensorHandle::new(handle, vec![n, n], vec![n, 1], f32::as_type_native_unchecked())
}

/// Benchmark LU factorization for a specific matrix size
fn bench_lu(n: usize) {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║ Matrix size: {}x{}", n, n);
    println!("╚═══════════════════════════════════════════════════════╝");

    // Create test matrix
    let a = create_test_matrix(&client, n);

    // Memory footprint
    let matrix_mb = (n * n * 4) as f64 / 1_000_000.0;
    println!("  Memory: {:.2} MB", matrix_mb);

    // Theoretical FLOP count for LU: 2n³/3 + lower order terms
    let flops = 2.0 * (n as f64).powi(3) / 3.0;
    println!("  Theoretical FLOPs: {:.2} GFLOP", flops / 1e9);

    let config = LUConfig::default();

    // Warmup
    for _ in 0..3 {
        let _ = lu_factor::<BenchRuntime, F32Precision>(&client, a.as_ref(), config);
    }
    future::block_on(client.sync());

    // Benchmark using GPU profiling
    let mut gpu_times = Vec::new();
    let num_runs = if n < 512 { 20 } else { 10 };

    for _ in 0..num_runs {
        future::block_on(client.sync());

        let result = client.profile(|| {
            lu_factor::<BenchRuntime, F32Precision>(&client, a.as_ref(), config)
        }, "lu_factor");

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

    // Bandwidth (read A once, write LU once, pivots)
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
    match lu_factor::<BenchRuntime, F32Precision>(&client, a.as_ref(), config) {
        Ok((lu, perm, info)) => {
            println!("      Pivots: {}", perm.len());
            println!("      Quality: {:?}", info.quality);
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
    println!("║         LU Factorization Benchmark                    ║");
    println!("║            GPU Performance Analysis                   ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    let device: <BenchRuntime as Runtime>::Device = Default::default();
    println!("\nDevice: {:?}", device);

    // Test sizes covering different scenarios
    let sizes = vec![
        32,    // Small: tests overhead
        64,    // Small-medium: first panel
        128,   // Medium: one full block
        256,   // Medium: multiple blocks
        512,   // Large: block algorithm shines
        1024,  // Very large: GEMM dominates
        2048,  // Huge: full SOTA test
    ];

    for n in sizes {
        bench_lu(n);
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║                   Benchmark Complete                  ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║    CubeCL Linear Algebra - LU Benchmark               ║");
    println!("╚═══════════════════════════════════════════════════════╝");

    // Main scalability benchmark
    bench_scalability();

    println!("\n🎉 All benchmarks complete!\n");
}
