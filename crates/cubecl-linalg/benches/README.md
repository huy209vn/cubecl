# CubeCL Linalg Benchmarks

This directory contains benchmarks for the cubecl-linalg operations.

## Running Benchmarks

### Basic Run (CPU Runtime)

```bash
cargo bench --bench norms
```

### With CUDA Support

```bash
cargo bench --bench norms --features cuda
```

### With WGPU Support

```bash
cargo bench --bench norms --features wgpu
```

## Benchmark Coverage

### Norm Benchmarks (`norms.rs`)

Tests performance of:
- **L2 Norm (Euclidean)** - `||x||₂ = sqrt(Σx²)`
- **L-infinity Norm** - `||x||∞ = max(|x|)`
- **Frobenius Norm** - `||A||_F = sqrt(Σa²)`

Each benchmark tests multiple sizes:
- **Vectors**: 1K, 64K, 1M, 16M elements
- **Matrices**: 128×128, 512×512, 1024×1024, 2048×2048

### CPU Baseline

The benchmarks include CPU reference implementations for comparison:
- Pure Rust implementations without GPU acceleration
- Useful for measuring GPU speedup

## Output Format

The benchmarks output timing information for each test:
- Operation name
- Backend (CPU/CUDA/WGPU)
- Data type (f32)
- Size
- Timing statistics (mean, median, min, max)

## Expected Performance Characteristics

### L2 Norm
- Complexity: O(n)
- Memory bandwidth bound
- Three GPU kernel launches: square → sum → sqrt
- Should show good speedup on large vectors

### L-infinity Norm
- Complexity: O(n)
- Two GPU kernel launches: abs → max
- Slightly faster than L2 (fewer operations)

### Frobenius Norm
- Complexity: O(mn) for m×n matrix
- Reuses L2 norm implementation
- Treats matrix as flattened vector

## Performance Tips

1. **Warmup**: First run may be slower due to kernel compilation
2. **Size matters**: GPU shows best performance on large datasets (>64K elements)
3. **Memory**: Ensure sufficient GPU memory for largest tests
4. **Precision**: Currently benchmarks use f32; f16/bf16 may show different characteristics

## Adding New Benchmarks

To add a new benchmark:

1. Create a new file in `benches/`
2. Implement the `Benchmark` trait
3. Add `[[bench]]` entry in `Cargo.toml`
4. Follow the pattern from `norms.rs`
