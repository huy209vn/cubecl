//! Benchmarks for norm operations (L2, L-infinity, Frobenius)

use cubecl_core::{prelude::*, benchmark::{Benchmark, TimingMethod, ProfileDuration}, future};
use cubecl_std::tensor::TensorHandle;
use cubecl_random::random_uniform;
use cubecl_linalg::{vector_norm_l2, vector_norm_inf, frobenius_norm, F32Precision};
use std::marker::PhantomData;

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
// CPU Baseline Benchmarks (for comparison)
// ============================================================================

struct CpuL2NormBench {
    size: usize,
}

impl Benchmark for CpuL2NormBench {
    type Input = Vec<f32>;
    type Output = f32;

    fn prepare(&self) -> Self::Input {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..self.size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        Ok(input.iter().map(|x| x * x).sum::<f32>().sqrt())
    }

    fn name(&self) -> String {
        format!("l2_norm-cpu-f32-{}", self.size).to_lowercase()
    }

    fn sync(&self) {
        // No-op for CPU
    }

    fn profile(&self, _args: Self::Input) -> Result<ProfileDuration, String> {
        Err("CPU profiling not supported".to_string())
    }
}

struct CpuLInfNormBench {
    size: usize,
}

impl Benchmark for CpuLInfNormBench {
    type Input = Vec<f32>;
    type Output = f32;

    fn prepare(&self) -> Self::Input {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..self.size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        Ok(input.iter().map(|x| x.abs()).fold(0.0_f32, f32::max))
    }

    fn name(&self) -> String {
        format!("linf_norm-cpu-f32-{}", self.size).to_lowercase()
    }

    fn sync(&self) {
        // No-op for CPU
    }

    fn profile(&self, _args: Self::Input) -> Result<ProfileDuration, String> {
        Err("CPU profiling not supported".to_string())
    }
}

// ============================================================================
// Main benchmark runner
// ============================================================================

fn run_l2_norm<R: Runtime>(device: R::Device, size: usize) {
    let client = R::client(&device);
    let bench = L2NormBench::<R> {
        size,
        device,
        client,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Device).unwrap());
}

fn run_linf_norm<R: Runtime>(device: R::Device, size: usize) {
    let client = R::client(&device);
    let bench = LInfNormBench::<R> {
        size,
        device,
        client,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Device).unwrap());
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
    println!("{}", bench.run(TimingMethod::Device).unwrap());
}

fn run_cpu_l2_norm(size: usize) {
    let bench = CpuL2NormBench { size };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Device).unwrap());
}

fn run_cpu_linf_norm(size: usize) {
    let bench = CpuLInfNormBench { size };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Device).unwrap());
}

fn main() {
    println!("=== Norm Benchmarks ===\n");

    // Test sizes: small, medium, large, very large
    let sizes = vec![1024, 65536, 1048576, 16777216]; // 1K, 64K, 1M, 16M

    // Matrix sizes for Frobenius norm
    let matrix_sizes = vec![
        (128, 128),     // 16K elements
        (512, 512),     // 256K elements
        (1024, 1024),   // 1M elements
        (2048, 2048),   // 4M elements
    ];

    println!("\n--- L2 Norm Benchmarks ---");

    // CPU baseline (pure Rust reference)
    println!("\nCPU Baseline (Pure Rust):");
    for &size in &sizes {
        run_cpu_l2_norm(size);
    }

    // CubeCL CPU Runtime (JIT compiled)
    println!("\nCubeCL CPU Runtime (MLIR JIT):");
    for &size in &sizes {
        run_l2_norm::<cubecl_cpu::CpuRuntime>(Default::default(), size);
    }

    println!("\n--- L-infinity Norm Benchmarks ---");

    // CPU baseline (pure Rust reference)
    println!("\nCPU Baseline (Pure Rust):");
    for &size in &sizes {
        run_cpu_linf_norm(size);
    }

    // CubeCL CPU Runtime (JIT compiled)
    println!("\nCubeCL CPU Runtime (MLIR JIT):");
    for &size in &sizes {
        run_linf_norm::<cubecl_cpu::CpuRuntime>(Default::default(), size);
    }

    println!("\n--- Frobenius Norm Benchmarks ---");

    // CubeCL CPU Runtime (JIT compiled)
    println!("\nCubeCL CPU Runtime (MLIR JIT):");
    for &(rows, cols) in &matrix_sizes {
        run_frobenius_norm::<cubecl_cpu::CpuRuntime>(Default::default(), rows, cols);
    }

    println!("\n=== Benchmark Complete ===");
    println!("\nNote: To benchmark with CUDA or WGPU, add those crates to dev-dependencies");
    println!("and modify main() to include those runtime tests.");
}
