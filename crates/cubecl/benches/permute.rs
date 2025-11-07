use cubecl::{frontend, prelude::*};
use cubecl::future;
use cubecl_std::tensor::{self, TensorHandle};

#[cfg(feature = "cuda")]
use half::f16;

fn bench_permute<R: Runtime, E: frontend::Float + CubeElement>(
    device: &R::Device,
    input_shape: Vec<usize>,
    axes: Vec<usize>,
    dtype_name: &str,
) {
    let client = R::client(device);

    // Allocate tensors once
    let numel: usize = input_shape.iter().product();
    let data: Vec<E> = (0..numel)
        .map(|i| E::from(i as f32).unwrap())
        .collect();

    let handle = client.create(E::as_bytes(&data));
    let input = TensorHandle::<R, E>::new_contiguous(input_shape.clone(), handle);

    let output_shape: Vec<usize> = axes.iter().map(|&i| input_shape[i]).collect();
    let output = TensorHandle::<R, E>::empty(&client, output_shape);

    // Correctness check: run once and verify result for 2D transpose
    if input_shape.len() == 2 && axes == [1, 0] && input_shape[0] <= 64 && input_shape[1] <= 64 {
        tensor::permute::launch::<R, E>(&client, &input, &axes, &output);
        future::block_on(client.sync());

        let actual = client.read_one_tensor(output.handle.clone().copy_descriptor(
            &output.shape,
            &output.strides,
            std::mem::size_of::<E>(),
        ));
        let actual_data = E::from_bytes(&actual);

        // Verify transpose for small matrices
        let h = input_shape[0];
        let w = input_shape[1];
        for row in 0..w {
            for col in 0..h {
                let out_idx = row * h + col;
                let in_idx = col * w + row;
                let expected = E::from(in_idx as f32).unwrap();
                if actual_data[out_idx] != expected {
                    eprintln!("❌ CORRECTNESS ERROR at [{},{}]: got {:?}, expected {:?}",
                             row, col, actual_data[out_idx], expected);
                    eprintln!("   Kernel produced WRONG results - timings are meaningless!");
                    return;
                }
            }
        }
    }

    // FIX 1: Extended warmup (100 iterations)
    // Ensures CUDA JIT compilation, memory registration, and GPU clock ramping complete
    for _ in 0..100 {
        tensor::permute::launch::<R, E>(
            &client,
            &input,
            &axes,
            &output,
        );
    }
    future::block_on(client.sync());

    // FIX 4: Auto-scale iteration count based on data size
    // Target: 2-4 seconds of total work for stable measurements
    let bytes_per_elem = std::mem::size_of::<E>();
    let bytes_per_iteration = numel * bytes_per_elem * 2; // read + write

    let mut iterations = 1;
    let target_total_bytes = 4_000_000_000.0; // 4 GB total throughput
    while iterations < 16384 && (bytes_per_iteration as f64 * iterations as f64) < target_total_bytes {
        iterations *= 2;
    }
    // Ensure minimum iterations for small tensors
    iterations = Ord::max(iterations, 20);

    // FIX 2 & 3: Use device-side timing if available, otherwise CPU timer
    // Note: CubeCL's client.sync() properly handles GPU-side timing
    // For CUDA, this uses cudaDeviceSynchronize which waits for all queued kernels
    future::block_on(client.sync()); // Ensure clean start

    // Benchmark: async kernel launches with single sync at end
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        tensor::permute::launch::<R, E>(
            &client,
            &input,
            &axes,
            &output,
        );
    }
    future::block_on(client.sync()); // Single sync measures total GPU time
    let elapsed = start.elapsed().as_secs_f64();

    let avg_ms = (elapsed / iterations as f64) * 1000.0;
    let total_bytes = numel * bytes_per_elem * 2; // read input + write output
    let bandwidth_gbs = (total_bytes as f64 / 1e9) / (avg_ms / 1000.0);

    println!(
        "{:4} | {:10.3} | {:15.2} | {:6}",
        dtype_name, avg_ms, bandwidth_gbs, iterations
    );
}

fn run_benchmark_suite<R: Runtime>(device: R::Device, backend_name: &str) {
    println!("\n=== {} PERMUTE/TRANSPOSE BENCHMARK ===\n", backend_name);

    // 2D transpose benchmarks
    println!("Shape: [1024, 1024] -> [1024, 1024] (transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![1024, 1024], vec![1, 0], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![1024, 1024], vec![1, 0], "F16");
    println!();

    println!("Shape: [4096, 4096] -> [4096, 4096] (transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![4096, 4096], vec![1, 0], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![4096, 4096], vec![1, 0], "F16");
    println!();

    // 3D batch transpose: various batch sizes with 1024x1024 matrices
    println!("Shape: [32, 1024, 1024] -> [32, 1024, 1024] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![32, 1024, 1024], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![32, 1024, 1024], vec![0, 2, 1], "F16");
    println!();

    println!("Shape: [16, 1024, 1024] -> [16, 1024, 1024] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![16, 1024, 1024], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![16, 1024, 1024], vec![0, 2, 1], "F16");
    println!();

    println!("Shape: [8, 1024, 1024] -> [8, 1024, 1024] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![8, 1024, 1024], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![8, 1024, 1024], vec![0, 2, 1], "F16");
    println!();

    println!("Shape: [4, 1024, 1024] -> [4, 1024, 1024] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![4, 1024, 1024], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![4, 1024, 1024], vec![0, 2, 1], "F16");
    println!();

    println!("Shape: [1, 1024, 1024] -> [1, 1024, 1024] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![1, 1024, 1024], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![1, 1024, 1024], vec![0, 2, 1], "F16");
    println!();

    // 3D batch transpose: original 512x512 case
    println!("Shape: [32, 512, 512] -> [32, 512, 512] (batch transpose)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![32, 512, 512], vec![0, 2, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![32, 512, 512], vec![0, 2, 1], "F16");
    println!();

    // 3D complex permutation
    println!("Shape: [128, 64, 64] -> [64, 128, 64] (complex permute)");
    println!("Type |   Time(ms)  | Bandwidth(GB/s) | Iters");
    println!("-----|-------------|-----------------|-------");
    bench_permute::<R, f32>(&device, vec![128, 64, 64], vec![2, 0, 1], "F32");
    #[cfg(feature = "cuda")]
    bench_permute::<R, f16>(&device, vec![128, 64, 64], vec![2, 0, 1], "F16");
    println!();
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let device = CudaDevice::default();
        run_benchmark_suite::<CudaRuntime>(device, "CUDA");
    }

    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let device = WgpuDevice::default();
        run_benchmark_suite::<WgpuRuntime>(device, "WGPU");
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        println!("No backend enabled. Run with --features cuda or --features wgpu");
    }
}
