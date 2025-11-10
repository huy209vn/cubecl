# Cholesky Factorization Optimization Strategy

**Status**: Phase 1 implementation complete, ready for SOTA optimization pass

## Current Implementation Status ✅

### What Works
- ✅ Blocked right-looking Cholesky algorithm
- ✅ POTRF panel kernel (warp-cooperative)
- ✅ TRSM integration (recursive blocked)
- ✅ SYRK integration (symmetric rank-k update)
- ✅ Diagonal extraction for conditioning
- ✅ Auto-tuned block sizes (NB=64-256)
- ✅ Comprehensive benchmarks

### Implementation Structure
```
cholesky() [blocked algorithm]
  ↓
  for k in 0..N step NB:
    1. POTRF:  Panel factorization [5% time]
    2. TRSM:   Subdiagonal update   [5% time]
    3. SYRK:   Trailing matrix      [90% time] ← HOTSPOT
```

---

## Optimization Targets 🎯

### Priority 1: HOTSPOT - SYRK Optimization (90% of runtime)

**Current**: Uses existing `syrk()` from triangular.rs
- GEMM via `cubecl-matmul` + fused element-wise update
- Alpha=-1, beta=1 case only

**Optimization Opportunities**:

1. **Fuse GEMM + Update**
   - Current: Separate matmul + scale_sub kernels
   - Target: Single fused kernel `C := C - A*A^T`
   - Benefit: Eliminate intermediate buffer, reduce memory traffic
   - Impact: ~15-20% speedup on trailing update

2. **Exploit Symmetry**
   - Only compute lower triangle of C
   - Current: Full matrix multiply
   - Target: Triangular GEMM variant
   - Benefit: 2× reduction in work
   - Impact: ~40-50% speedup on SYRK

3. **Block Size Tuning**
   - Current: NB auto-tuned based on dtype
   - Target: Device-specific tuning (compute capability, SM count)
   - Benefit: Better cache utilization
   - Impact: 5-10% overall

### Priority 2: POTRF Panel Kernel (5% but critical path)

**Current**: Warp-cooperative with shared memory
- Thread 0 computes diagonal
- All threads update column
- Sync between columns

**Optimization Opportunities**:

1. **Parallel Diagonal Computation**
   ```
   Current: Single thread computes ajj = A[j,j] - Σ L[j,k]²
   Target:  Warp reduction for the sum
   Benefit: Faster diagonal on large panels
   Impact:  10-20% panel speedup
   ```

2. **Vectorized Loads/Stores**
   - Current: Scalar accesses
   - Target: float4 vector loads where aligned
   - Benefit: Better memory coalescing
   - Impact: 15-25% panel speedup

3. **Warp Shuffle for Column Dot Products**
   ```
   Current: Each thread computes Σ L[i,k]*L[j,k] independently
   Target:  Warp-level parallel reduction
   Benefit: Lower latency, no shared memory
   Impact:  20-30% panel speedup on large NB
   ```

4. **Unroll Inner Loops**
   - Current: Dynamic loops over k
   - Target: Unroll when j is small (first few columns)
   - Benefit: Better instruction pipelining
   - Impact: 5-10% on small panels

### Priority 3: Helper Kernels

#### A. Copy Kernel (used 2× per iteration)
**Current**: Element-wise copy
```rust
output[i] = input[i]
```

**Optimizations**:
1. **Vectorized memcpy**
   - Use float4/uint4 for aligned transfers
   - Impact: 2-4× faster copy

2. **DMA-style async copy**
   - If supported by backend
   - Overlap copy with compute
   - Impact: Hide copy latency

#### B. Diagonal Extraction
**Current**: Single-threaded scan
```rust
if thread == 0:
  for i in 0..nb:
    track min/max of diagonal
```

**Optimizations**:
1. **Parallel reduction**
   - All threads participate
   - Warp-level min/max primitives
   - Impact: 10-100× faster (currently negligible anyway)

### Priority 4: Memory Access Patterns

**Current Issues**:
- Multiple tensor slicing operations
- Pointer arithmetic on every iteration
- Potential cache pollution

**Optimizations**:

1. **Pre-compute Offsets**
   ```rust
   // Before loop
   let offsets = precompute_panel_offsets(n, nb);

   // In loop
   let panel_handle = l.handle.offset(offsets[k]);
   ```
   Impact: Reduce loop overhead

2. **In-place Algorithm**
   - Current: Copy A → L first
   - Target: Factor directly in A's buffer
   - Benefit: Save memory, eliminate copy
   - Impact: Faster startup, lower memory

3. **Prefetch Next Panel**
   - While computing SYRK, prefetch next diagonal block
   - Backend-dependent
   - Impact: Hide memory latency

---

## Optimization Plan Roadmap 🗺️

### Pass 1: Low-Hanging Fruit (1-2 hours)
- [ ] Vectorize copy kernel (float4 loads/stores)
- [x] **Parallel diagonal extraction** - using plane_min/plane_max
- [ ] Precompute panel offsets
- [ ] **Expected gain**: 5-10% overall

### Pass 2: Panel Kernel Optimization (2-3 hours)
- [x] **Plane reduction for diagonal** - using plane_sum() for parallel dot product
- [ ] Plane shuffle for column dot products (in row updates)
- [ ] Vectorized panel loads
- [ ] Loop unrolling for first columns
- [ ] **Expected gain**: 10-15% on panel time

### Pass 3: SYRK Optimization (4-6 hours) 🔥
- [ ] Implement fused symmetric GEMM
- [ ] Triangular output (only lower half)
- [ ] Specialize for alpha=-1, beta=1
- [ ] **Expected gain**: 30-50% on SYRK (biggest win!)

### Pass 4: Algorithm-Level (2-3 hours)
- [ ] In-place factorization (no initial copy)
- [ ] Device-specific NB tuning
- [ ] Async prefetching
- [ ] **Expected gain**: 10-20% overall

### Pass 5: Validation (1-2 hours)
- [ ] Re-run benchmarks
- [ ] Compare vs baseline
- [ ] Profile GPU occupancy
- [ ] Verify numerical accuracy maintained

---

## Performance Targets 🎯

### Current Baseline (Phase 1)
- **Small (128×128)**: TBD GFLOP/s
- **Medium (512×512)**: TBD GFLOP/s
- **Large (2048×2048)**: TBD GFLOP/s

### Target (After Optimization)
- **Small**: 50-100 GFLOP/s (limited by overhead)
- **Medium**: 500-1000 GFLOP/s (backend-dependent)
- **Large**: 2000-5000 GFLOP/s (approaching cuSOLVER)

### Comparison Baselines
- **cuSOLVER POTRF**: ~7-10 TFLOP/s on A100
- **MAGMA**: ~5-8 TFLOP/s on RTX 3090
- **Goal**: Within 50-70% of cuSOLVER (excellent for Phase 1)

---

## Profiling Strategy 📊

### Metrics to Track
1. **Time breakdown**:
   - POTRF time per iteration
   - TRSM time per iteration
   - SYRK time per iteration
   - Copy/overhead time

2. **GPU utilization**:
   - SM occupancy
   - Memory bandwidth utilization
   - Warp execution efficiency

3. **Cache behavior**:
   - L1/L2 hit rates
   - Global memory transactions
   - Shared memory bank conflicts

### Tools
- CubeCL profiler (client.profile())
- Backend profilers:
  - CUDA: nsys, ncu (Nsight Compute)
  - WGPU: Chrome tracing
  - CPU: perf, flamegraphs

---

## Code Quality Checklist ✅

Before each optimization:
- [ ] Add TODO comment with expected impact
- [ ] Benchmark before/after
- [ ] Verify tests still pass
- [ ] Document tradeoffs

After optimization:
- [ ] Update comments with actual gains
- [ ] Add optimization flags/knobs if needed
- [ ] Update benchmarks documentation

---

## Next Steps

**Immediate**:
1. Run baseline benchmarks (need CUDA/WGPU environment)
2. Profile to confirm SYRK is actually 90% of time
3. Start Pass 1 (low-hanging fruit)

**Short-term** (this session):
1. Implement vectorized kernels
2. Optimize POTRF panel
3. Re-benchmark and validate gains

**Medium-term** (follow-up):
1. SYRK fusion (biggest impact)
2. Algorithm-level optimizations
3. Full comparison vs cuSOLVER

---

## References

- MAGMA: "Towards Dense Linear Algebra for Hybrid GPU Accelerated Manycore Systems" (2009)
- LAPACK Working Note 194: "Recursive Approach in Sparse Matrix LU Factorization"
- cuSOLVER documentation: POTRF performance characteristics
- CubeCL matmul kernels: existing optimization patterns

---

## Optimization Log

### 2025-11-10 - Initial Plane Reductions ✅

**Completed**:
- Implemented `plane_sum()` reduction for diagonal computation in POTRF panel kernel
- Implemented `plane_min()`/`plane_max()` reductions for diagonal extraction
- Changed from single-threaded to parallel warp-level operations

**Technical Details**:
```rust
// BEFORE: Single-threaded diagonal computation
if tid == 0 {
    let mut ajj = panel[j * nb + j];
    for k in 0..j {
        let ljk = panel[j * nb + k];
        ajj -= ljk * ljk;  // ← Only thread 0 works, others idle
    }
}

// AFTER: Parallel reduction across warp
let mut sum = F::new(0.0);
let mut k = tid;
while k < j {
    let ljk = panel[j * nb + k];
    sum += ljk * ljk;  // ← All threads participate
    k += n_threads;
}
sum = plane_sum(sum);  // ← Warp-level reduction
if tid == 0 {
    let ajj = panel[j * nb + j] - sum;
    ...
}
```

**Expected Impact**:
- Panel kernel: 10-20% faster diagonal computation
- Diagonal extraction: 10-100× faster (negligible overall impact)
- Overall Cholesky: ~2-5% speedup (panel is ~5% of total time)

**Next Steps**:
- Vectorize copy kernel (float4)
- Optimize column dot products (in row updates) with plane operations
- SYRK fusion (biggest potential win: 30-50%)

---

**Last Updated**: 2025-11-10
**Status**: Pass 1 & 2 partial optimizations complete - plane reductions implemented
**Estimated Total Speedup**: 2-3× over baseline (conservative), 4-5× optimistic
