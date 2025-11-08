//! Benchmarks for norm operations (L2, L-infinity, Frobenius)
//!
//! These benchmarks test GPU-accelerated norm computations using CubeCL kernels.
//! All operations use actual GPU kernels with #[cube] annotation.

use cubecl_core::{prelude::*, benchmark::{Benchmark, TimingMethod, ProfileDuration}, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_random::random_uniform;
use cubecl_linalg::{vector_norm_l2, vector_norm_inf, frobenius_norm, F32Precision};

// ============================================================================
// L2 Norm Benchmark
// ============================================================================

struct L2NormBench<R: Runtime> {
    size: usize,
    device: R::Device,
    client: ComputeClient<R::Server>,
}

impl<R: Runtime> Benchmark for L2NormBench<R> {
    type Input = TensorHandle<R, f32>;
    type Output = TensorHandle<R, f32>;

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let input = TensorHandle::<R, f32>::empty(&client, vec![self.size]);
        random_uniform::<R, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());
        input
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        vector_norm_l2::<R, F32Precision>(&self.client, input.as_ref())
            .map_err(|e| format!("{:?}", e))
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "l2_norm-{}-{}-{}",
            R::name(&client),
            "f32",
            self.size
        ).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "l2-norm-bench")
            .map_err(|it| format!("{it:?}"))
    }
}

// ============================================================================
// L-infinity Norm Benchmark
// ============================================================================

struct LInfNormBench<R: Runtime> {
    size: usize,
    device: R::Device,
    client: ComputeClient<R::Server>,
}

impl<R: Runtime> Benchmark for LInfNormBench<R> {
    type Input = TensorHandle<R, f32>;
    type Output = TensorHandle<R, f32>;

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let input = TensorHandle::<R, f32>::empty(&client, vec![self.size]);
        random_uniform::<R, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());
        input
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        vector_norm_inf::<R, F32Precision>(&self.client, input.as_ref())
            .map_err(|e| format!("{:?}", e))
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "linf_norm-{}-{}-{}",
            R::name(&client),
            "f32",
            self.size
        ).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "linf-norm-bench")
            .map_err(|it| format!("{it:?}"))
    }
}

// ============================================================================
// Frobenius Norm Benchmark
// ============================================================================

struct FrobeniusNormBench<R: Runtime> {
    rows: usize,
    cols: usize,
    device: R::Device,
    client: ComputeClient<R::Server>,
}

impl<R: Runtime> Benchmark for FrobeniusNormBench<R> {
    type Input = TensorHandle<R, f32>;
    type Output = TensorHandle<R, f32>;

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let input = TensorHandle::<R, f32>::empty(&client, vec![self.rows, self.cols]);
        random_uniform::<R, f32>(&client, f32::from_int(-1), f32::from_int(1), input.as_ref());
        input
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        frobenius_norm::<R, F32Precision>(&self.client, input.as_ref())
            .map_err(|e| format!("{:?}", e))
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "frobenius_norm-{}-{}-{}x{}",
            R::name(&client),
            "f32",
            self.rows,
            self.cols
        ).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "frobenius-norm-bench")
            .map_err(|it| format!("{it:?}"))
    }
}

// ============================================================================
// Benchmark Runners
// ============================================================================

fn run_l2_norm<R: Runtime>(device: R::Device, size: usize) {
    let client = R::client(&device);
    let bench = L2NormBench::<R> {
        size,
        device,
        client,
    };
    println!("{}", bench.name());
    let result = bench.run(TimingMethod::Device).unwrap();
    println!("{}", result);

    // Calculate bandwidth
    // L2 norm: square (read+write) + reduce (read) + sqrt (negligible)
    // Approximate: 3 full passes over data
    let bytes = (size * 4 * 3) as f64; // f32 = 4 bytes, ~3 passes
    let time_s = result.median.as_secs_f64();
    let gb_per_s = bytes / time_s / 1e9;
    println!("  Bandwidth   {:.2} GB/s\n", gb_per_s);
}

fn run_linf_norm<R: Runtime>(device: R::Device, size: usize) {
    let client = R::client(&device);
    let bench = LInfNormBench::<R> {
        size,
        device,
        client,
    };
    println!("{}", bench.name());
    let result = bench.run(TimingMethod::Device).unwrap();
    println!("{}", result);

    // Calculate bandwidth
    // L-inf norm: abs (read+write) + reduce_max (read)
    // Approximate: 2.5 full passes over data
    let bytes = (size * 4 * 2) as f64; // f32 = 4 bytes, ~2 passes
    let time_s = result.median.as_secs_f64();
    let gb_per_s = bytes / time_s / 1e9;
    println!("  Bandwidth   {:.2} GB/s\n", gb_per_s);
}

fn run_frobenius_norm<R: Runtime>(device: R::Device, rows: usize, cols: usize) {
    let client = R::client(&device);
    let bench = FrobeniusNormBench::<R> {
        rows,
        cols,
        device,
        client,
    };
    println!("{}", bench.name());
    let result = bench.run(TimingMethod::Device).unwrap();
    println!("{}", result);

    // Calculate bandwidth
    // Frobenius norm uses L2 pipeline: 3 passes
    let size = rows * cols;
    let bytes = (size * 4 * 3) as f64; // f32 = 4 bytes, ~3 passes
    let time_s = result.median.as_secs_f64();
    let gb_per_s = bytes / time_s / 1e9;
    println!("  Bandwidth   {:.2} GB/s\n", gb_per_s);
}

// ============================================================================
// Main - GPU Benchmarks
// ============================================================================

fn main() {
    println!("=== GPU Norm Benchmarks ===");
    println!("Testing GPU-accelerated norm operations with CubeCL kernels");
    println!("\nBandwidth Analysis:");
    println!("  Modern GPUs: 200-900 GB/s (GDDR6/HBM)");
    println!("  Target: >50% peak bandwidth for large workloads");
    println!("  Note: Multiple kernel launches reduce efficiency\n");

    // Test sizes: small, medium, large, very large
    let sizes = vec![
        1024,       // 1K
        65536,      // 64K
        1048576,    // 1M
        16777216,   // 16M
    ];

    // Matrix sizes for Frobenius norm
    let matrix_sizes = vec![
        (128, 128),     // 16K elements
        (512, 512),     // 256K elements
        (1024, 1024),   // 1M elements
        (2048, 2048),   // 4M elements
    ];

    // Determine which runtime to use
    #[cfg(feature = "cuda")]
    type BenchRuntime = cubecl_cuda::CudaRuntime;

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    type BenchRuntime = cubecl_wgpu::WgpuRuntime;

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    type BenchRuntime = cubecl_cpu::CpuRuntime;

    let device: <BenchRuntime as Runtime>::Device = Default::default();

    println!("\n--- L2 Norm (Euclidean) ---");
    println!("Algorithm: square → reduce_sum → sqrt");
    println!("Kernels: 3 GPU launches per norm\n");
    for &size in &sizes {
        run_l2_norm::<BenchRuntime>(device.clone(), size);
    }

    println!("\n--- L-infinity Norm ---");
    println!("Algorithm: abs → reduce_max");
    println!("Kernels: 2 GPU launches per norm\n");
    for &size in &sizes {
        run_linf_norm::<BenchRuntime>(device.clone(), size);
    }

    println!("\n--- Frobenius Norm (Matrix) ---");
    println!("Algorithm: flatten → L2 norm");
    println!("Kernels: 3 GPU launches per norm\n");
    for &(rows, cols) in &matrix_sizes {
        run_frobenius_norm::<BenchRuntime>(device.clone(), rows, cols);
    }

    println!("\n=== Benchmark Complete ===");

    println!("\n⚠️  Performance Analysis:");
    println!("  If bandwidth is <50 GB/s on modern GPU:");
    println!("    - Multiple kernel launches = high overhead");
    println!("    - Consider fused kernels to reduce launches");
    println!("    - cubecl-reduce may need optimization");
    println!("    - Kernel launch latency dominates small sizes");
    println!("\n  SOTA target: >300 GB/s for large workloads");
    println!("  Current approach: Correctness first, optimize later");

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    println!("\nNote: Running on CPU runtime. For GPU benchmarks, use:");
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    println!("  cargo bench --bench norms --features cuda");
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    println!("  cargo bench --bench norms --features wgpu");
}
