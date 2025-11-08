//! Dead simple benchmark - no fancy profiling, just measure what actually happens

use cubecl_core::{prelude::*, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_random::random_uniform;
use cubecl_linalg::{vector_norm_l2, F32Precision};
use std::time::Instant;

#[cfg(feature = "cuda")]
type BenchRuntime = cubecl_cuda::CudaRuntime;

fn main() {
    let device: <BenchRuntime as Runtime>::Device = Default::default();
    let client = BenchRuntime::client(&device);

    println!("=== DEAD SIMPLE L2 Norm Benchmark ===\n");
    println!("Just measuring wall-clock time + sync. No fancy profiling.\n");

    let sizes = vec![1_048_576, 4_194_304, 16_777_216, 67_108_864];

    for size in sizes {
        println!("\nSize: {} elements ({} MB)", size, size * 4 / 1_000_000);

        // Create input
        let input = TensorHandle::<BenchRuntime, f32>::empty(&client, vec![size]);
        random_uniform::<BenchRuntime, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());
        future::block_on(client.sync());

        // Warmup
        for _ in 0..3 {
            let _ = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
            future::block_on(client.sync());
        }

        // Single timed run
        println!("  Running 1 iteration...");

        let start = Instant::now();
        let result = vector_norm_l2::<BenchRuntime, F32Precision>(&client, input.as_ref()).unwrap();
        future::block_on(client.sync()); // WAIT for GPU
        let elapsed = start.elapsed();

        // Read result (but don't include in timing)
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        // Calculate bandwidth
        let bytes = (size * 4) as f64;
        let time_s = elapsed.as_secs_f64();
        let gb_per_s = bytes / time_s / 1e9;

        println!("  Time:      {:.2}µs", time_s * 1e6);
        println!("  Bandwidth: {:.2} GB/s", gb_per_s);
        println!("  Result:    {:.6}", result_value);

        // Sanity check
        if gb_per_s > 936.0 {
            println!("  ⚠️  WARNING: Exceeds RTX 3090 theoretical peak (936 GB/s)!");
        } else if gb_per_s > 700.0 {
            println!("  ✓ Excellent! >75% of peak");
        } else if gb_per_s > 400.0 {
            println!("  ✓ Good! >40% of peak");
        }
    }

    println!("\n{:=<60}", "");
    println!("\nIf you still see >936 GB/s, the timing is fundamentally broken.");
    println!("If you see 400-700 GB/s, we're SOTA!");
}
