//! Simple norm benchmarks - manual timing without Benchmark trait
//!
//! This helps us verify if the performance issue is in our kernels
//! or in how we're using the Benchmark framework.

use cubecl_core::{prelude::*, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_random::random_uniform;
use cubecl_linalg::{vector_norm_l2, F32Precision};
use std::time::Instant;

#[cfg(feature = "cuda")]
type BenchRuntime = cubecl_cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BenchRuntime = cubecl_wgpu::WgpuRuntime;

fn bench_size(client: &ComputeClient<<BenchRuntime as Runtime>::Server>, size: usize, name: &str) {
    println!("\n{}", name);
    println!("Size: {} elements ({} MB)", size, size * 4 / 1_000_000);

    // Create input
    let input = TensorHandle::<BenchRuntime, f32>::empty(client, vec![size]);
    random_uniform::<BenchRuntime, f32>(client, f32::from_int(-1), f32::from_int(1), input.as_ref());

    // Warmup
    for _ in 0..3 {
        let _ = vector_norm_l2::<BenchRuntime, F32Precision>(client, input.as_ref()).unwrap();
    }
    future::block_on(client.sync());

    // Benchmark
    let mut times = Vec::new();

    for _ in 0..10 {
        let start = Instant::now();
        let _result = vector_norm_l2::<BenchRuntime, F32Precision>(client, input.as_ref()).unwrap();
        future::block_on(client.sync());
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    // Stats
    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];

    // Bandwidth
    let bytes = (size * 4) as f64;
    let median_s = median.as_secs_f64();
    let gb_per_s = bytes / median_s / 1e9;

    println!("  Min:        {:?}", min);
    println!("  Median:     {:?}", median);
    println!("  Bandwidth:  {:.2} GB/s", gb_per_s);
}

fn main() {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("=== Comprehensive L2 Norm Benchmark ===");
    println!("GPU: {}", BenchRuntime::name(&client));
    println!("Target: >300 GB/s for SOTA\n");
    println!("Note: 2-stage reduction kicks in at 1M elements");

    // Test various sizes to show scaling
    let test_configs = vec![
        (1024, "Tiny (1K) - single stage"),
        (65_536, "Small (64K) - single stage"),
        (1_048_576, "Medium (1M) - threshold"),
        (4_194_304, "Large (4M) - 2-stage"),
        (16_777_216, "Huge (16M) - 2-stage"),
        (67_108_864, "Massive (64M) - 2-stage"),
    ];

    println!("\n{:-<60}", "");

    for (size, name) in test_configs {
        bench_size(&client, size, name);
    }

    println!("\n{:-<60}", "");
    println!("\nPerformance Analysis:");
    println!("  - Small sizes: Lower GB/s due to kernel launch overhead");
    println!("  - Large sizes: Should hit >50% of GPU memory bandwidth");
    println!("  - 2-stage optimization: Critical for SOTA on large vectors");
    println!("\nFor reference:");
    println!("  RTX 3090:  936 GB/s theoretical peak");
    println!("  RTX 4090:  1008 GB/s theoretical peak");
    println!("  A100:      1555 GB/s (HBM2e)");
    println!("  H100:      3350 GB/s (HBM3)");
}
