# Norms Benchmark Guide

## Quick Start

```bash
# Run all norm benchmarks
cd crates/cubecl-linalg
cargo bench --bench norms

# Note: First run will be slower due to MLIR kernel compilation
```

## What Gets Benchmarked

### L2 Norm (Euclidean Norm)
**Formula:** `||x||₂ = sqrt(Σ x²)`

**Implementation:** 3 GPU kernel launches:
1. Element-wise square (`square_kernel`)
2. Reduction sum (`cubecl-reduce`)
3. Element-wise sqrt (`sqrt_kernel`)

**Test Sizes:**
- 1,024 elements (1K)
- 65,536 elements (64K)
- 1,048,576 elements (1M)
- 16,777,216 elements (16M)

### L-infinity Norm
**Formula:** `||x||∞ = max(|x|)`

**Implementation:** 2 GPU kernel launches:
1. Element-wise absolute value (`abs_kernel`)
2. Reduction max (`cubecl-reduce`)

**Test Sizes:** Same as L2 norm

### Frobenius Norm
**Formula:** `||A||_F = sqrt(Σ A²)` (for matrix A)

**Implementation:** Reuses L2 norm (treats matrix as flattened vector)

**Test Sizes (matrices):**
- 128 × 128 = 16,384 elements
- 512 × 512 = 262,144 elements
- 1,024 × 1,024 = 1,048,576 elements
- 2,048 × 2,048 = 4,194,304 elements

## Comparison Modes

### CPU Baseline (Pure Rust)
Reference implementations using:
- Standard Rust iterators
- No SIMD optimizations
- Pure sequential execution

**Purpose:** Establish baseline for speedup measurements

### CubeCL CPU Runtime (MLIR JIT)
Our GPU kernels running through:
- MLIR compilation pipeline
- LLVM optimization passes
- JIT execution on CPU

**Purpose:** Test correctness and show JIT overhead before GPU testing

## Expected Output

```
=== Norm Benchmarks ===

--- L2 Norm Benchmarks ---

CPU Baseline (Pure Rust):
l2_norm-cpu-f32-1024
Mean: 1.234 µs, Median: 1.200 µs, Min: 1.150 µs, Max: 1.500 µs

l2_norm-cpu-f32-65536
Mean: 45.678 µs, Median: 45.000 µs, Min: 44.500 µs, Max: 50.000 µs

[... more sizes ...]

CubeCL CPU Runtime (MLIR JIT):
l2_norm-cpuruntime-f32-1024
Mean: 0.987 µs, Median: 0.950 µs, Min: 0.900 µs, Max: 1.200 µs

[... more results ...]
```

## Performance Expectations

### Small Sizes (1K - 64K)
- **CPU Baseline:** Very fast (< 50 µs)
- **CubeCL CPU:** Similar or slightly slower due to kernel launch overhead
- **GPU (when available):** May be slower due to transfer overhead

### Medium Sizes (1M)
- **CPU Baseline:** ~1-5 ms
- **CubeCL CPU:** Competitive with baseline
- **GPU:** Should show 2-5x speedup

### Large Sizes (16M)
- **CPU Baseline:** ~50-200 ms
- **CubeCL CPU:** Similar
- **GPU:** Should show 10-50x speedup (when using CUDA/WGPU)

## Interpreting Results

### What to Look For

1. **Correctness:** All implementations should produce same results (validated in tests)

2. **Scaling:** Time should scale linearly with input size
   - L2: ~3x kernel overhead vs simple ops
   - L-inf: ~2x kernel overhead vs simple ops

3. **Memory Bandwidth:** On GPU, should be limited by:
   - L2 Norm: 3 passes over data (read, reduce, read)
   - L-inf Norm: 2 passes over data (read, reduce)

4. **Reduction Overhead:** CubeCL-reduce should use hierarchical reduction
   - Small vectors: Single-pass reduction
   - Large vectors: Multi-stage reduction with shared memory

### Known Limitations

1. **CPU Runtime:** MLIR JIT compilation has overhead
   - First run will compile kernels
   - Subsequent runs use cached kernels
   - Not optimized for CPU SIMD

2. **Memory Layout:** Currently uses contiguous memory
   - Strided tensors not yet optimized
   - Future work: blocked/tiled layouts

3. **Precision:** Benchmarks use f32
   - f16/bf16 should show better throughput (2x bandwidth)
   - f64 will be slower (2x bandwidth needed)

## Adding GPU Benchmarks

To test with CUDA:

1. Add to `Cargo.toml` dev-dependencies:
```toml
cubecl-cuda = { path = "../cubecl-cuda", version = "0.9.0" }
```

2. Update `benches/norms.rs` main():
```rust
// Add after CPU runtime tests
println!("\nCUDA Runtime:");
for &size in &sizes {
    run_l2_norm::<cubecl_cuda::CudaRuntime>(Default::default(), size);
}
```

3. Run with:
```bash
cargo bench --bench norms --features cuda
```

Same pattern works for WGPU runtime.

## Troubleshooting

### LLVM Download Fails
```
error: failed to run custom build command for `tracel-llvm-bundler`
```

**Solution:** Network issue downloading LLVM binaries. Try:
- Check internet connection
- Clear cargo cache: `cargo clean`
- Retry: `cargo bench --bench norms`

### Out of Memory
Large tests (16M+) may OOM on smaller GPUs.

**Solution:** Edit `benches/norms.rs` to reduce max size:
```rust
let sizes = vec![1024, 65536, 1048576]; // Remove 16M
```

### Benchmark Times Out
Very first run compiles MLIR kernels.

**Solution:** This is normal. Subsequent runs are much faster.

## Performance Targets

Based on typical hardware:

| Size | CPU Baseline | Target GPU (CUDA) | Speedup |
|------|-------------|-------------------|---------|
| 1K   | 1-2 µs      | 5-10 µs          | 0.2-0.5x (overhead) |
| 64K  | 50-100 µs   | 10-20 µs         | 3-5x |
| 1M   | 1-2 ms      | 50-100 µs        | 10-20x |
| 16M  | 50-100 ms   | 200-500 µs       | 100-250x |

## Next Steps

After confirming norms benchmark:

1. **Verify correctness:** Run tests to ensure GPU matches CPU
2. **Profile kernels:** Identify bottlenecks (bandwidth vs compute)
3. **Optimize:** Tune block sizes, vectorization
4. **Add benchmarks:** For triangular ops, Cholesky, etc.
