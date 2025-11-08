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

fn main() {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("=== Simple L2 Norm Benchmark ===\n");

    // Test with 16M elements (same as before)
    let size = 16_777_216;
    println!("Size: {} elements ({} MB)", size, size * 4 / 1_000_000);

    // Create input
    let input = TensorHandle::<BenchRuntime, f32>::empty(&client, vec![size]);
    random_uniform::<BenchRuntime, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let _ = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
    }
    future::block_on(client.sync());

    // Benchmark
    println!("Benchmarking (10 iterations)...\n");
    let mut times = Vec::new();

    for i in 0..10 {
        let start = Instant::now();
        let _result = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
        future::block_on(client.sync());
        let elapsed = start.elapsed();

        times.push(elapsed);
        println!("  Iteration {}: {:?}", i + 1, elapsed);
    }

    // Stats
    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];
    let mean: std::time::Duration = times.iter().sum::<std::time::Duration>() / times.len() as u32;

    println!("\nResults:");
    println!("  Min:    {:?}", min);
    println!("  Median: {:?}", median);
    println!("  Mean:   {:?}", mean);

    // Bandwidth
    let bytes = (size * 4) as f64;
    let median_s = median.as_secs_f64();
    let gb_per_s = bytes / median_s / 1e9;

    println!("\nBandwidth: {:.2} GB/s", gb_per_s);
    println!("(Target: >300 GB/s for SOTA)");
}
