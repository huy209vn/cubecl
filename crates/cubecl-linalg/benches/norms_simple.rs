//! Simple, correct GPU norm benchmark using CubeCL profiling API
//!
//! This uses GPU-side timing to measure only kernel execution,
//! not CPU-GPU transfer overhead.

use cubecl_core::{prelude::*, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_random::random_uniform;
use cubecl_linalg::{vector_norm_l2, F32Precision};

#[cfg(feature = "cuda")]
type BenchRuntime = cubecl_cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BenchRuntime = cubecl_wgpu::WgpuRuntime;

fn bench_norm_gpu(size: usize) {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("\n{} elements ({} MB)", size, size * 4 / 1_000_000);

    // Create input
    let input = TensorHandle::<BenchRuntime, f32>::empty(&client, vec![size]);
    random_uniform::<BenchRuntime, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());

    // Warmup
    for _ in 0..3 {
        let _ = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
    }
    future::block_on(client.sync());

    // Benchmark using GPU profiling
    let mut gpu_times = Vec::new();

    for _ in 0..10 {
        future::block_on(client.sync());

        // Use CubeCL's profiling API - this gives GPU-side timing
        let result = client.profile(|| {
            vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref())
        }, "l2_norm");

        match result {
            Ok(profile_duration) => {
                // Resolve the profile future to get actual timing
                let ticks = future::block_on(profile_duration.resolve());
                let duration = ticks.duration();
                gpu_times.push(duration);
            }
            Err(e) => {
                println!("Profile error: {:?}", e);
                return;
            }
        }
    }

    // Stats
    gpu_times.sort();
    let median = gpu_times[gpu_times.len() / 2];
    let min = gpu_times[0];

    // Bandwidth (input data read once)
    let bytes = (size * 4) as f64;
    let time_s = median.as_secs_f64();
    let gb_per_s = bytes / time_s / 1e9;

    println!("  GPU Time (median): {:.2}µs", time_s * 1e6);
    println!("  GPU Time (min):    {:.2}µs", min.as_secs_f64() * 1e6);
    println!("  Bandwidth:         {:.2} GB/s", gb_per_s);

    // Verify result
    let result = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
    let result_bytes = client.read_one(result.handle.clone());
    let result_value = f32::from_bytes(&result_bytes)[0];
    println!("  Result:            {:.6}", result_value);
}

fn main() {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("=== GPU L2 Norm Benchmark (GPU-side timing) ===");
    println!("GPU: {}", BenchRuntime::name(&client));
    println!("\nUsing CubeCL profiling API for accurate GPU timing");
    println!("(excludes CPU-GPU transfer overhead)\n");

    let sizes = vec![
        1_024,          // 1K
        65_536,         // 64K
        1_048_576,      // 1M (threshold)
        4_194_304,      // 4M
        16_777_216,     // 16M
        67_108_864,     // 64M
    ];

    for size in sizes {
        bench_norm_gpu(size);
    }

    println!("\n{:=<60}", "");
    println!("\nExpected on RTX 3090 (936 GB/s theoretical):");
    println!("  Small (<1M):  10-100 GB/s (overhead limited)");
    println!("  Large (>4M):  400-700 GB/s (50-75% of peak)");
    println!("  Should NEVER exceed 936 GB/s!");
}
