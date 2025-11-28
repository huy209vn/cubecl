# CubeCL-Sparse: Implementation Roadmap

**Purpose:** Step-by-step plan to build cubecl-sparse from zero to production  
**Scope:** What to build, in what order, with what dependencies  
**Format:** Phases → Milestones → Tasks

---

## Overview

```
Phase 0: Foundation (2 weeks)
    ↓
Phase 1: CSR Core (4 weeks)
    ↓
Phase 2: Autodiff + Training (3 weeks)
    ↓
Phase 3: N:M Structured (3 weeks)
    ↓
Phase 4: Integration Layer (3 weeks)
    ↓
Phase 5: Block Formats + Dynamic (4 weeks)
    ↓
Phase 6: Polish + Production (2 weeks)

Total: ~21 weeks
```

---

## Phase 0: Foundation

**Goal:** Project skeleton, core types, infrastructure

**Duration:** 2 weeks

### Milestone 0.1: Project Setup

```
Tasks:
├── Create cubecl-sparse crate in workspace
├── Set up Cargo.toml with dependencies (cubecl, cubecl-runtime, cubecl-linalg)
├── Create module structure (format/, ops/, memory/, error/)
├── Set up test infrastructure (correctness/, benchmarks/)
└── CI configuration (test matrix for CUDA/WGPU)

Done when: `cargo build` passes, tests run
```

### Milestone 0.2: Core Types

```
Tasks:
├── SparseFormatId enum
├── SparseFormat trait (marker)
├── SparseMetadata trait
├── LogicalShape, StorageShape types
├── IndexDType enum (U16, U32, U64)
└── DType compatibility constants

Done when: Types compile, can instantiate format IDs
```

### Milestone 0.3: Error Types

```
Tasks:
├── SparseError enum (all variants from spec)
├── SparseResult<T> type alias
├── Error conversion impls (From<io::Error>, etc.)
└── Basic error tests

Done when: Can construct and match all error variants
```

### Milestone 0.4: Handle System Skeleton

```
Tasks:
├── SparseTensorHandle struct (placeholder buffers)
├── SparseTensorHandleRef struct
├── SparseBufferSet enum (CSR variant only initially)
├── TensorId generation
└── Basic lifetime handling

Done when: Can create handle, borrow as ref, check lifetimes compile
```

---

## Phase 1: CSR Core

**Goal:** Working CSR format with SpMM

**Duration:** 4 weeks

**Dependencies:** Phase 0 complete

### Milestone 1.1: CSR Storage

```
Tasks:
├── CsrStorage<R> struct
├── CsrMetadata struct
├── RowStatistics struct
├── Memory layout validation
├── from_raw_parts (unsafe construction)
├── metadata() accessor
├── sparsity() calculation
└── memory_bytes() calculation

Done when: Can construct CSR storage from pre-built buffers
```

### Milestone 1.2: COO Storage (Construction Path)

```
Tasks:
├── CooStorage<R> struct
├── CooMetadata struct
├── sorted flag tracking
├── from_triplets() construction
├── sort() kernel (GPU radix sort or thrust)
├── coalesce() kernel (merge duplicates)
└── Basic COO tests

Done when: Can build COO from (row, col, val) triplets, sort it
```

### Milestone 1.3: Format Conversions (Basic)

```
Tasks:
├── COO → CSR kernel
│   ├── Row histogram kernel
│   ├── Prefix sum for row_ptrs
│   └── Scatter values kernel
├── CSR → Dense kernel
├── Dense → COO kernel (with threshold)
├── Dense → CSR (via COO)
└── Conversion tests (round-trip correctness)

Done when: Can convert dense ↔ CSR ↔ COO, values preserved
```

### Milestone 1.4: SpMM Row-Split Kernel

```
Tasks:
├── CsrSpmm operation struct
├── CsrSpmmConfig struct
├── Row-split kernel implementation
│   ├── One thread per row
│   ├── Sequential column iteration
│   └── Basic output write
├── Kernel launch wrapper
├── Correctness tests vs dense matmul
└── Basic benchmarks

Done when: CSR SpMM matches dense matmul output (rtol=1e-5)
```

### Milestone 1.5: SpMM Warp-Per-Row Kernel

```
Tasks:
├── Warp-per-row kernel implementation
│   ├── Warp-level parallelism
│   ├── Warp reduction for dot product
│   └── Vectorized B access (vec4)
├── Shared memory staging for B tiles
├── Algorithm selection logic (avg_nnz threshold)
├── Performance comparison vs row-split
└── Update benchmarks

Done when: Warp kernel faster than row-split for nnz > 32/row
```

### Milestone 1.6: SpMM Merge-Based Kernel

```
Tasks:
├── Merge-based row-split kernel
│   ├── Handle long rows (> 256 nnz)
│   ├── Multi-warp cooperation
│   └── Merge reduction
├── Row statistics computation kernel
│   ├── Min/max/mean/std per row
│   └── Histogram buckets
├── Full algorithm selector
└── Irregular sparsity tests (power-law)

Done when: Handles highly irregular row distributions correctly
```

### Milestone 1.7: CSC Storage + Transpose

```
Tasks:
├── CscStorage<R> struct
├── CscMetadata struct
├── CSR → CSC transpose kernel
│   ├── Column histogram
│   ├── Prefix sum for col_ptrs
│   └── Scatter with transposed indices
├── CSC → CSR transpose
├── Transpose view (zero-cost reinterpretation)
└── SpMM with CSC (for Aᵀ × B)

Done when: CSR ↔ CSC conversion correct, transpose view works
```

---

## Phase 2: Autodiff + Training

**Goal:** Backward pass, gradient computation, basic training loop

**Duration:** 3 weeks

**Dependencies:** Phase 1 complete

### Milestone 2.1: Sparse Gradient Kernel

```
Tasks:
├── SpMM backward design verification
├── dL/dA kernel (sparse gradient, same pattern)
│   ├── One thread per non-zero
│   ├── Binary search for row lookup
│   └── Dot product: grad_output[row,:] · B[col,:]
├── Gradient correctness tests (finite differences)
└── Gradient shape/pattern verification

Done when: dL/dA matches finite difference approximation
```

### Milestone 2.2: Dense Gradient via Transpose

```
Tasks:
├── dL/dB computation (Aᵀ × grad_output)
├── Use CSC view of A for transpose
├── Reuse existing SpMM kernels
├── End-to-end backward test
└── Memory usage verification

Done when: Full backward pass correct, memory reasonable
```

### Milestone 2.3: Mask Preservation

```
Tasks:
├── MaskPreservation enum (ExactPattern, AllowGrowth, AllowShrinkage)
├── Pattern checking in gradient accumulation
├── SparseParameter struct
│   ├── Storage + mask mode
│   ├── Gradient accumulator
│   └── apply_grad() method
├── Pattern mismatch error handling
└── Mask preservation tests

Done when: Gradients maintain sparsity pattern in ExactPattern mode
```

### Milestone 2.4: Autodiff Tape Integration

```
Tasks:
├── SparseAutoGradNode enum
├── SparseAutodiffTape struct
├── Record sparse ops to tape
├── Backward traversal
├── Sparse gradient extraction
├── Integration with Burn's autodiff system
└── Multi-op backward test (SpMM → ReLU → SpMM)

Done when: Can autodiff through chain of sparse ops
```

### Milestone 2.5: Basic Training Loop

```
Tasks:
├── SparseLinear module (minimal)
├── Sparse SGD optimizer
├── End-to-end training test
│   ├── Small MLP with sparse layer
│   ├── Forward → loss → backward → update
│   └── Loss decreases over iterations
├── Memory leak check (multiple iterations)
└── Gradient accumulation test

Done when: Can train sparse MLP, loss decreases
```

---

## Phase 3: N:M Structured Sparsity

**Goal:** 2:4 sparsity with tensor core acceleration

**Duration:** 3 weeks

**Dependencies:** Phase 2 complete

### Milestone 3.1: N:M Storage

```
Tasks:
├── NMStorage<R, N, M> struct
├── NMMetadata struct
├── Compressed value layout (N values per M-group)
├── Index encoding (4-bit masks for 2:4)
├── Memory alignment for tensor cores
└── Storage shape calculations

Done when: Can construct N:M storage with correct layout
```

### Milestone 3.2: N:M Pruning

```
Tasks:
├── Dense → 2:4 kernel
│   ├── Load group of 4
│   ├── Find top-2 by magnitude
│   ├── Pack values and indices
│   └── Handle edge cases (ties)
├── PruningStrategy::NMStructured
├── nm_prune() API
├── Pruning correctness tests
└── Verify 50% sparsity output

Done when: Can prune dense to valid 2:4 structure
```

### Milestone 3.3: N:M SpMM Emulated

```
Tasks:
├── NMSpmm operation struct
├── Emulated kernel (non-tensor-core path)
│   ├── Unpack indices
│   ├── Gather values
│   └── Standard SpMM logic
├── Correctness tests vs dense
├── Use as fallback on non-Ampere
└── Performance baseline

Done when: N:M SpMM correct on any GPU
```

### Milestone 3.4: N:M Tensor Core Kernel (CUDA)

```
Tasks:
├── CUDA capability detection
├── mma.sp intrinsic wrapper
├── Tensor core tile configuration (16×16×16)
├── Tensor core kernel implementation
│   ├── Tile loading with sparse indices
│   ├── mma.sp execution
│   └── Accumulator writeback
├── Kernel selection (TC vs emulated)
├── Performance benchmarks vs cuSPARSE
└── Ampere+ specific tests

Done when: 2:4 SpMM uses tensor cores on Ampere, matches cuSPARSE perf
```

### Milestone 3.5: N:M Autodiff

```
Tasks:
├── N:M backward for dL/dA
│   ├── Gradient at structured positions
│   └── Index unpacking in backward
├── N:M backward for dL/dB (dense)
├── Pattern preservation for N:M
├── End-to-end N:M training test
└── Compare training curves: CSR vs N:M

Done when: Can train with 2:4 sparse weights
```

---

## Phase 4: Integration Layer

**Goal:** Views, batching, sparse-dense interop, modules

**Duration:** 3 weeks

**Dependencies:** Phase 3 complete

### Milestone 4.1: View System

```
Tasks:
├── SparseView enum
├── SparseTensorView struct
├── Transpose view (CSR ↔ CSC reinterpret)
├── Row slice view
│   ├── Offset calculation
│   ├── Bounds checking
│   └── Alignment validation (for N:M, BSR)
├── View materialization
├── View-aware SpMM dispatch
└── View tests

Done when: Can create views, use in SpMM without materialization
```

### Milestone 4.2: Batched Operations

```
Tasks:
├── BatchedSparseStorage enum (Uniform, Variable)
├── BatchedSparseTensor struct
├── stack_uniform() for same-pattern batching
├── Batched SpMM kernel (uniform)
│   ├── Batch dimension in grid
│   └── Shared indices, batched values
├── Batch slicing (get single element)
├── Unbatch operation
└── Batched training test

Done when: Can batch sparse tensors, run batched SpMM
```

### Milestone 4.3: Sparse-Dense Interop

```
Tasks:
├── ConversionPolicy enum
├── sparse_dense_add() (sparse + dense → dense)
├── sparse_dense_mul() (sparse * dense → sparse)
├── Broadcasting rules implementation
├── to_dense() for all formats
├── to_sparse() with threshold
├── Auto-conversion in mixed ops
└── Interop tests

Done when: Mixed sparse-dense operations work correctly
```

### Milestone 4.4: Burn Modules

```
Tasks:
├── SparseLinear module
│   ├── Config with sparsity, format
│   ├── Initialization (dense → prune)
│   ├── Forward pass
│   └── Parameter registration
├── SparseEmbedding module
│   ├── Sparse embedding table
│   └── Lookup operation
├── Module serialization hooks
├── Integration with Burn's Module derive
└── Module training tests

Done when: Can use SparseLinear in Burn model, train it
```

### Milestone 4.5: Serialization

```
Tasks:
├── SparseTensorHeader format
├── save() implementation
│   ├── Header as JSON
│   └── Buffers as raw bytes
├── load() implementation
│   ├── Header parsing
│   └── Buffer reconstruction
├── Round-trip tests (save → load → compare)
├── Checkpoint integration
└── safetensors compatibility (stretch)

Done when: Can save/load sparse tensors, survive restart
```

---

## Phase 5: Block Formats + Dynamic Sparsity

**Goal:** BSR/BCSC, dynamic pattern changes, RigL

**Duration:** 4 weeks

**Dependencies:** Phase 4 complete

### Milestone 5.1: BSR Storage

```
Tasks:
├── BsrStorage<R> struct
├── BsrMetadata struct
├── Block value layout (row-major within block)
├── CSR → BSR conversion
│   ├── Block boundary alignment
│   ├── Padding for incomplete blocks
│   └── Block index construction
├── BSR → CSR conversion
├── BSR → Dense conversion
└── Storage tests

Done when: Can construct BSR, convert to/from CSR
```

### Milestone 5.2: BSR SpMM

```
Tasks:
├── BsrSpmm operation struct
├── Block SpMM kernel
│   ├── Block-level parallelism
│   ├── Dense block matmul within
│   └── Accumulation across blocks
├── Tile size selection heuristics
├── Performance comparison vs CSR
└── BSR correctness tests

Done when: BSR SpMM correct, faster than CSR for block-sparse patterns
```

### Milestone 5.3: Dynamic Pattern Engine

```
Tasks:
├── SparsityPattern struct (indices only)
├── PatternDelta struct (additions, removals)
├── PatternManager struct
│   ├── Mode tracking (Static, GrowOnly, etc.)
│   ├── queue_add(), queue_remove()
│   └── apply_pending()
├── Pattern → COO → modify → Pattern flow
├── Pattern history tracking
└── Dynamic pattern tests

Done when: Can modify sparsity pattern at runtime
```

### Milestone 5.4: COO Gradient Accumulator

```
Tasks:
├── CooGradientAccumulator struct
├── accumulate_sparse() (same pattern)
├── accumulate_dense() (for regrowth tracking)
├── Dense gradient buffer for candidate tracking
├── regrowth_candidates() (top-k outside pattern)
├── finalize() → CSR
└── Accumulator tests

Done when: Can accumulate gradients, find regrowth candidates
```

### Milestone 5.5: RigL Training

```
Tasks:
├── rigl_update() in PatternManager
│   ├── Find lowest magnitude weights
│   ├── Find highest gradient zeros
│   └── Drop/grow balance
├── STEMaskedLinear module
│   ├── Dense weights + mask
│   ├── STE forward
│   └── Mask update hook
├── RigL training loop test
│   ├── Periodic mask updates
│   └── Verify accuracy recovery
└── Compare final accuracy: static vs RigL

Done when: RigL training works, accuracy comparable to literature
```

### Milestone 5.6: BCSC + Remaining Formats

```
Tasks:
├── BcscStorage<R> struct
├── BSR ↔ BCSC transpose
├── Any remaining format conversions
├── Format selection heuristics (full matrix)
├── Auto-format selection API
└── Comprehensive format tests

Done when: All formats from spec implemented
```

---

## Phase 6: Polish + Production

**Goal:** Fusion, JIT cache, inspection tools, hardening

**Duration:** 2 weeks

**Dependencies:** Phase 5 complete

### Milestone 6.1: Fusion Integration

```
Tasks:
├── SparseAccessPattern struct
├── FusionEngine::can_fuse() logic
├── SpMM + epilogue fusion
│   ├── ReLU epilogue
│   ├── Bias add epilogue
│   └── Combined bias + ReLU
├── Element-wise chain fusion
├── Fusion analysis for op graphs
└── Fusion correctness + performance tests

Done when: SpMM + ReLU fuses into single kernel
```

### Milestone 6.2: JIT Kernel Cache

```
Tasks:
├── KernelSignature struct (all fields)
├── ShapeBucket, SparsityBucket, RowDistributionBucket
├── KernelCache struct
│   ├── get_or_compile()
│   ├── LRU eviction
│   └── Size tracking
├── Cache warmup API
├── Cache export/import
└── Cache hit rate tracking

Done when: Repeated ops hit cache, no recompilation
```

### Milestone 6.3: Inspection Tools

```
Tasks:
├── SparseInspector::statistics()
├── SparseInspector::summary() (human readable)
├── SparseInspector::visualize_pattern() (ASCII)
├── SparseValidator::validate() (all checks)
├── SparseComparator::allclose()
├── Debug logging integration
└── Inspection tool tests

Done when: Can debug sparse tensor issues easily
```

### Milestone 6.4: Documentation + Examples

```
Tasks:
├── Rustdoc for all public APIs
├── README with quick start
├── Example: Sparse MLP training
├── Example: N:M pruning workflow
├── Example: RigL training
├── Performance tuning guide
└── Troubleshooting guide

Done when: New user can follow examples, understand API
```

### Milestone 6.5: Hardening

```
Tasks:
├── Edge case tests
│   ├── Empty tensors (0 nnz)
│   ├── Single element
│   ├── Very large tensors
│   └── Extreme sparsity (99.9%)
├── Error message review
├── Memory leak testing
├── Stress tests (many ops, long training)
├── Multi-GPU tests (if applicable)
└── Performance regression tests

Done when: No crashes on edge cases, stable under stress
```

### Milestone 6.6: Benchmarks + Comparison

```
Tasks:
├── Benchmark suite vs cuSPARSE
├── Benchmark suite vs dense (break-even analysis)
├── Benchmark suite vs Triton sparse
├── Memory efficiency benchmarks
├── Training throughput benchmarks
├── Publish benchmark results
└── Performance report

Done when: Know exactly where we stand vs alternatives
```

---

## Dependency Graph

```
Phase 0 ─────────────────────────────────────────────────┐
   │                                                     │
   ▼                                                     │
Phase 1: CSR Core                                        │
   │                                                     │
   ├──────────────────────┐                              │
   ▼                      ▼                              │
Phase 2: Autodiff    Phase 3: N:M                        │
   │                      │                              │
   └──────────┬───────────┘                              │
              ▼                                          │
         Phase 4: Integration                            │
              │                                          │
              ▼                                          │
         Phase 5: Block + Dynamic                        │
              │                                          │
              ▼                                          │
         Phase 6: Polish ◄───────────────────────────────┘
                              (can start docs early)
```

---

## Risk Areas

### High Risk

| Area | Risk | Mitigation |
|------|------|------------|
| SpMM performance | May not match cuSPARSE | Profile early, iterate on kernels |
| Tensor core integration | CUDA intrinsics tricky | Start with emulated, add TC later |
| Autodiff correctness | Subtle bugs in gradients | Extensive finite difference testing |
| Memory management | Leaks in complex ops | Valgrind/compute-sanitizer early |

### Medium Risk

| Area | Risk | Mitigation |
|------|------|------------|
| Fusion complexity | Over-engineering | Start simple, add fusion incrementally |
| Dynamic sparsity | Pattern corruption | Validate pattern after every modification |
| Cross-platform | WGPU limitations | Focus CUDA first, WGPU later |
| API ergonomics | Clunky to use | Dogfood with real training scripts |

### Low Risk

| Area | Risk | Mitigation |
|------|------|------------|
| Serialization | Format changes | Version field in header |
| View system | Edge cases | Comprehensive tests |
| Inspection tools | Not critical path | Can add incrementally |

---

## Checkpoints

### After Phase 1
- [ ] CSR SpMM passes all correctness tests
- [ ] Performance within 2x of cuSPARSE on standard benchmarks
- [ ] Can convert dense ↔ CSR ↔ COO

### After Phase 2
- [ ] Full backward pass works
- [ ] Can train sparse MLP, loss decreases
- [ ] No memory leaks in training loop

### After Phase 3
- [ ] 2:4 N:M format complete
- [ ] Tensor core kernel works on Ampere
- [ ] N:M training produces valid models

### After Phase 4
- [ ] Burn integration works
- [ ] Can save/load sparse models
- [ ] SparseLinear usable in real architectures

### After Phase 5
- [ ] RigL training works
- [ ] BSR format complete
- [ ] Dynamic sparsity stable

### After Phase 6
- [ ] Comprehensive benchmarks published
- [ ] Documentation complete
- [ ] Ready for production use

---

## Resource Estimates

### Per Phase

| Phase | Weeks | Core Hours | Notes |
|-------|-------|------------|-------|
| 0 | 2 | 40-60 | Setup, types |
| 1 | 4 | 100-150 | Heavy kernel work |
| 2 | 3 | 60-90 | Autodiff complexity |
| 3 | 3 | 80-120 | TC kernel challenging |
| 4 | 3 | 60-90 | Integration work |
| 5 | 4 | 80-120 | Dynamic sparsity tricky |
| 6 | 2 | 40-60 | Polish |

### Hardware Needs

- Development: Any CUDA GPU (sm_70+)
- Testing tensor cores: Ampere GPU (sm_80+)
- Benchmarking: A100 or H100 for fair comparison
- WGPU testing: Can use same machine

---

## Open Questions

1. **Contribution model:** Solo or open for contributions after Phase 2?
2. **cuSPARSE fallback:** Use cuSPARSE as fallback initially, replace incrementally?
3. **WGPU priority:** Defer to after CUDA is solid, or parallel development?
4. **Upstream timing:** When to propose merging into cubecl/burn repos?

---

*This roadmap is a living document. Adjust timelines based on actual progress.*

34. Updated Roadmap Additions
34.1 SpGEMM Phase (Insert after Phase 5)
Phase 5.5: SpGEMM (2 weeks)
├── Milestone 5.5.1: Two-Phase SpGEMM
│   ├── Symbolic phase kernel (nnz counting)
│   ├── Numeric phase kernel
│   └── Correctness tests
├── Milestone 5.5.2: SpGEMM Autodiff
│   ├── Backward kernels
│   ├── Pattern masking
│   └── Gradient tests
└── Milestone 5.5.3: Pattern Operations
    ├── Union, intersection, difference
    └── Dynamic sparsity integration
34.2 Extended Phase 2 (Optimizers)
Phase 2 additions:
├── Milestone 2.6: Sparse Optimizers
│   ├── Sparse SGD with momentum
│   ├── Sparse Adam
│   ├── Gradient clipping utilities
│   └── Optimizer state serialization
└── Milestone 2.7: Regularization
    ├── L1 regularization
    ├── Proximal gradient (soft thresholding)
    └── Group sparsity (optional)
34.3 Extended Phase 5 (Scheduling)
Phase 5 additions:
├── Milestone 5.7: Mask Scheduling
│   ├── Sparsity schedule functions
│   ├── MaskUpdateScheduler
│   ├── GradualPruningController
│   └── Integration tests
34.4 Phase 6 Addition (Autotuning)
Phase 6 additions:
├── Milestone 6.7: Autotuning
│   ├── AutotuneDatabase
│   ├── SparseAutotuner
│   ├── Tile size search
│   └── Persistent database save/load
34.5 Future Phase: Sparse Attention
Phase 7: Sparse Attention (Future, 4+ weeks)
├── Milestone 7.1: Pattern Generation
│   ├── Local pattern
│   ├── BigBird pattern
│   └── Custom pattern support
├── Milestone 7.2: Sparse Attention Kernels
│   ├── Q @ K^T with sparse output
│   ├── Sparse softmax
│   └── Attention @ V
├── Milestone 7.3: Flash Attention Integration
│   ├── Block-sparse flash attention
│   └── Memory-efficient backward
└── Milestone 7.4: Attention Modules
    ├── SparseMultiHeadAttention
    └── Integration with Burn transformers