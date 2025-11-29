CubeCL-Sparse: Implementation Roadmap (Revised)
Purpose: Step-by-step plan incorporating the complete SpMM architecture
Core Innovation: Adaptive Binned Gather-GEMM with tensor core acceleration
Scope: Foundation through production-ready sparse operations

Overview
Phase 0: Foundation (2 weeks)
    ↓
Phase 1: Storage Formats (3 weeks)
    ↓
Phase 2: Analysis Infrastructure (2 weeks)  ← NEW
    ↓
Phase 3: SpMM Algorithm Suite (5 weeks)     ← EXPANDED
    ↓
Phase 4: Gather-GEMM + Tensor Cores (4 weeks)  ← NEW (our innovation)
    ↓
Phase 5: Autodiff + Training (3 weeks)
    ↓
Phase 6: N:M Structured Sparsity (3 weeks)
    ↓
Phase 7: Integration Layer (3 weeks)
    ↓
Phase 8: Advanced Features (4 weeks)
    ↓
Phase 9: Polish + Production (2 weeks)

Total: ~31 weeks

Phase 0: Foundation
Goal: Project skeleton, core types, infrastructure
Duration: 2 weeks
Milestone 0.1: Project Setup
Tasks:
├── Create cubecl-sparse crate in workspace
├── Set up Cargo.toml with dependencies
│   ├── cubecl-core
│   ├── cubecl-runtime
│   └── cubecl-linalg (for dense GEMM primitives)
├── Create module structure:
│   ├── format/        (storage formats)
│   ├── analysis/      (statistics, classification)
│   ├── ops/           (SpMM, SpGEMM, etc.)
│   ├── kernel/        (GPU kernels)
│   ├── plan/          (execution planning)
│   ├── memory/        (allocation, pooling)
│   └── autodiff/      (gradient computation)
├── Test infrastructure
│   ├── correctness/ (vs dense reference)
│   ├── benchmarks/ (vs cuSPARSE)
│   └── fixtures/ (test matrices)
└── CI configuration

Done when: `cargo build` passes, module structure in place
Milestone 0.2: Core Types
Tasks:
├── Scalar type traits
│   ├── SparseFloat trait (f16, bf16, f32, f64)
│   └── Precision enum
├── Index types
│   ├── IndexDtype enum (U16, U32, U64)
│   └── Index trait for generic indexing
├── Shape types
│   ├── SparseShape { rows, cols, nnz }
│   └── TileShape { tile_m, tile_k }
├── Error types
│   ├── SparseError enum (comprehensive)
│   └── SparseResult<T> alias
└── Device abstraction
    ├── DeviceProperties struct
    │   ├── has_tensor_cores: bool
    │   ├── sm_count: u32
    │   ├── shared_memory_per_sm: u32
    │   ├── memory_bandwidth: f64
    │   └── compute_capability: (u32, u32)
    └── Device capability queries

Done when: Core types compile, can query device properties
Milestone 0.3: Buffer Management
Tasks:
├── SparseBuffer<R> wrapper
│   ├── Typed buffer with runtime
│   ├── Length and byte size tracking
│   └── Slice views
├── BufferPool for scratch allocations
│   ├── Size-bucketed allocation
│   ├── Reuse across operations
│   └── Memory pressure handling
├── Handle types
│   ├── SparseTensorHandle (owned)
│   └── SparseTensorHandleRef (borrowed)
└── Memory statistics tracking

Done when: Can allocate, reuse, and track GPU buffers

Phase 1: Storage Formats
Goal: All sparse storage formats with conversions
Duration: 3 weeks
Milestone 1.1: COO Storage
Tasks:
├── CooStorage<R, F> struct
│   ├── row_indices: Buffer<u32>
│   ├── col_indices: Buffer<u32>
│   ├── values: Buffer<F>
│   ├── shape: SparseShape
│   └── sorted: bool
├── CooMetadata
│   ├── is_sorted()
│   ├── is_coalesced()
│   └── nnz()
├── Construction
│   ├── from_triplets(rows, cols, vals)
│   └── from_dense(tensor, threshold)
├── GPU kernels
│   ├── sort_by_row() - radix sort
│   ├── sort_by_col() - radix sort
│   └── coalesce() - merge duplicates
└── Tests

Done when: Can construct COO, sort it, coalesce duplicates
Milestone 1.2: CSR Storage
Tasks:
├── CsrStorage<R, F> struct
│   ├── row_ptrs: Buffer<u32>    // length: rows + 1
│   ├── col_indices: Buffer<u32> // length: nnz
│   ├── values: Buffer<F>        // length: nnz
│   └── shape: SparseShape
├── CsrMetadata
│   ├── rows(), cols(), nnz()
│   ├── row_length(i) → u32
│   ├── row_range(i) → Range<u32>
│   └── density() → f32
├── Construction
│   ├── from_coo(coo) - GPU kernel
│   │   ├── Row histogram
│   │   ├── Exclusive prefix sum → row_ptrs
│   │   └── Scatter values by row
│   └── from_dense(tensor, threshold)
├── Accessors
│   ├── to_dense() → Tensor
│   ├── row_slice(start, end) → CsrView
│   └── get_row(i) → (cols, vals) on CPU
└── Tests (round-trip with COO and dense)

Done when: CSR ↔ COO ↔ Dense conversions correct
Milestone 1.3: CSC Storage
Tasks:
├── CscStorage<R, F> struct
│   ├── col_ptrs: Buffer<u32>
│   ├── row_indices: Buffer<u32>
│   ├── values: Buffer<F>
│   └── shape: SparseShape
├── CscMetadata (mirrors CSR)
├── Construction
│   ├── from_coo(coo) - sort by column first
│   └── from_csr(csr) - transpose kernel
│       ├── Column histogram
│       ├── Prefix sum → col_ptrs
│       └── Scatter with transposed indices
├── Zero-cost transpose view
│   ├── CSR as "transposed CSC"
│   └── CSC as "transposed CSR"
└── Tests

Done when: CSR ↔ CSC transpose works, view reinterpretation works
Milestone 1.4: BSR Storage
Tasks:
├── BsrStorage<R, F> struct
│   ├── row_ptrs: Buffer<u32>     // block-row pointers
│   ├── col_indices: Buffer<u32>  // block-column indices
│   ├── values: Buffer<F>         // dense blocks, flattened
│   ├── shape: SparseShape
│   ├── block_size: (u32, u32)
│   └── num_blocks: u32
├── BsrMetadata
│   ├── block_rows(), block_cols()
│   ├── blocks_in_row(br) → u32
│   └── block_density() → f32
├── Construction
│   ├── from_csr(csr, block_size)
│   │   ├── Validate alignment
│   │   ├── Group elements into blocks
│   │   ├── Pad incomplete blocks
│   │   └── Build block CSR structure
│   └── from_dense(tensor, block_size, threshold)
├── Block access
│   ├── get_block(br, bc) → Option<DenseBlock>
│   └── block_iterator()
└── Tests

Done when: CSR ↔ BSR conversion correct, blocks properly formed
Milestone 1.5: SELL-C-σ Storage
Tasks:
├── SellCStorage<R, F> struct
│   ├── col_indices: Buffer<u32>  // padded, sliced
│   ├── values: Buffer<F>         // padded, sliced
│   ├── slice_ptrs: Buffer<u32>   // start of each slice
│   ├── max_nnz: Buffer<u32>      // max row length per slice
│   ├── permutation: Buffer<u32>  // original row order
│   ├── shape: SparseShape
│   └── slice_size: u32           // C parameter
├── SellCMetadata
│   ├── num_slices()
│   ├── slice_nnz(s) → u32
│   └── padding_overhead() → f32
├── Construction from CSR
│   ├── Compute row lengths
│   ├── Sort rows by length → permutation
│   ├── Divide into slices of C rows
│   ├── For each slice:
│   │   ├── Find max length
│   │   ├── Pad all rows to max
│   │   └── Pack into slice storage
│   └── Build slice_ptrs and max_nnz
├── Slice size selection heuristic
│   ├── C = 32 for warp alignment
│   ├── C = 64 for more sorting benefit
│   └── Adaptive based on CV
└── Tests

Done when: CSR → SELL-C works, padding tracked correctly
Milestone 1.6: Format Selection
Tasks:
├── FormatHint enum
│   ├── Auto
│   ├── PreferCSR
│   ├── PreferBSR { block_size }
│   ├── PreferSellC { slice_size }
│   └── PreferCSC
├── select_format(stats, operation, hint) → Format
│   ├── If hint != Auto, respect hint
│   ├── If block structure detected → BSR
│   ├── If CV > 2.0 → SELL-C
│   ├── If operation is A^T @ B → CSC
│   └── Default → CSR
├── convert_to_format(sparse, target_format) → Sparse
└── Format conversion cost estimation

Done when: Can auto-select and convert between all formats

Phase 2: Analysis Infrastructure
Goal: Matrix statistics, tile classification, row binning
Duration: 2 weeks
This is critical for our algorithm selection and Gather-GEMM innovation.
Milestone 2.1: Global Statistics
Tasks:
├── MatrixStatistics struct
│   ├── Basic: rows, cols, nnz, density
│   ├── Row metrics:
│   │   ├── avg_nnz_per_row
│   │   ├── median_nnz_per_row
│   │   ├── std_nnz_per_row
│   │   ├── min_nnz_per_row
│   │   ├── max_nnz_per_row
│   │   └── cv (coefficient of variation)
│   ├── Distribution:
│   │   ├── skewness
│   │   └── row_histogram: [u32; 11] (power-of-2 buckets)
│   └── Structure indicators:
│       ├── diagonal_dominance: f32
│       ├── estimated_bandwidth: Option<u32>
│       └── has_empty_rows: bool
├── analyze_csr(csr) → MatrixStatistics
│   ├── Single-pass GPU kernel
│   │   ├── Per-row length computation
│   │   ├── Parallel reduction for sum, sum_sq
│   │   ├── Histogram atomics
│   │   └── Diagonal proximity check
│   └── CPU post-processing for derived stats
├── RowStatistics struct (lightweight, per-use)
│   ├── row_lengths: Buffer<u32>
│   └── Computed on demand
└── Tests with known distributions

Done when: Can compute full statistics for any CSR matrix
Milestone 2.2: Tile Classification
Tasks:
├── TileConfig struct
│   ├── tile_size_m: u32 (default 256)
│   ├── tile_size_k: u32 (default 256)
│   ├── min_elements_for_tiling: u64
│   ├── dense_threshold: f32 (default 0.25)
│   ├── banded_threshold: f32 (default 0.80)
│   └── block_threshold: f32 (default 0.50)
├── TileClass enum
│   ├── Empty
│   ├── Dense
│   ├── Banded { bandwidth, lower_bw, upper_bw }
│   ├── BlockSparse { block_size, block_positions }
│   ├── LowRank { shared_columns }
│   └── Sparse
├── TileInfo struct
│   ├── row_range: Range<u32>
│   ├── col_range: Range<u32>
│   ├── classification: TileClass
│   ├── nnz: u32
│   ├── density: f32
│   └── local_row_stats: TileRowStats
├── TileDecomposition struct
│   ├── tiles: Vec<Vec<TileInfo>>  // [tile_row][tile_col]
│   ├── Summary counts per class
│   └── tile_size_m, tile_size_k
├── Classification kernels
│   ├── analyze_tile() - per tile
│   │   ├── Count nnz
│   │   ├── Build column histogram
│   │   ├── Track diagonal distance
│   │   └── Classify based on thresholds
│   ├── detect_banded()
│   │   ├── Track min/max diagonal offset
│   │   └── Check diagonal_nnz / total_nnz
│   ├── detect_block_sparse()
│   │   ├── Try block sizes [32, 16, 8]
│   │   ├── Count dense blocks per size
│   │   └── Return best if coverage > 70%
│   └── detect_low_rank()
│       ├── Find columns in >50% of rows
│       └── Check count in [4, K/4]
├── decompose_into_tiles(csr, config) → Option<TileDecomposition>
│   ├── Skip if matrix too small
│   ├── Create tile grid
│   ├── Classify each tile
│   └── Return decomposition
└── Tests with synthetic patterned matrices

Done when: Can decompose matrix, correctly classify tile types
Milestone 2.3: Row Binning
Tasks:
├── BinId enum/struct
│   ├── EMPTY (0 nnz)
│   ├── TINY (1-7 nnz)
│   ├── SMALL (8-31 nnz)
│   ├── MEDIUM (32-127 nnz)
│   ├── LARGE (128-511 nnz)
│   └── HUGE (512+ nnz)
├── BinConfig struct
│   ├── boundaries: [u32; 6]
│   ├── enable_tensor_cores: bool
│   └── min_rows_for_gather: u32
├── RowBin<R> struct
│   ├── id: BinId
│   ├── num_rows: u32
│   ├── padded_nnz: u32
│   ├── total_nnz: u32
│   ├── row_indices: Buffer<u32>      // original row numbers
│   ├── gather_cols: Buffer<u32>      // flattened [num_rows × padded_nnz]
│   ├── gather_vals: Buffer<F>        // flattened [num_rows × padded_nnz]
│   └── strategy: BinStrategy
├── BinStrategy enum
│   ├── Skip
│   ├── RowSplit
│   ├── WarpPerRow
│   ├── VectorWarpPerRow { vec_width }
│   ├── GatherGemm { tile_n }
│   ├── GatherTensorCore { tile_m, tile_k, tile_n }
│   └── MergePath
├── RowBinning<R> struct
│   ├── bins: Vec<RowBin<R>>
│   ├── total_rows: u32
│   └── total_nnz: u32
├── create_binning(csr, stats, config) → RowBinning
│   ├── Compute row lengths (or reuse from stats)
│   ├── Count rows per bin
│   ├── Allocate bin buffers
│   ├── Populate bins:
│   │   ├── Gather row indices
│   │   ├── Flatten and pad column indices
│   │   └── Flatten and pad values
│   └── Select strategy per bin
├── Strategy selection logic
│   ├── EMPTY → Skip
│   ├── TINY → RowSplit
│   ├── SMALL → WarpPerRow (or Vector if N large)
│   ├── MEDIUM → GatherGemm (or TC if available)
│   ├── LARGE/HUGE → GatherTensorCore (or GatherGemm fallback)
│   └── Override to MergePath if CV > 1.5 within bin
└── Tests: verify binning preserves all data

Done when: Can bin any CSR, reconstruct original data from bins
Milestone 2.4: Execution Planning
Tasks:
├── SpmmPlan<R> struct
│   ├── m, k, n dimensions
│   ├── stats: MatrixStatistics
│   ├── tiles: Option<TileDecomposition>
│   ├── binning: RowBinning<R>
│   ├── dense_tiles: Vec<DenseTileData<R>>
│   ├── banded_tiles: Vec<BandedTileData>
│   ├── block_tiles: Vec<BlockSparseTileData<R>>
│   └── execution_order: Vec<ExecutionStep>
├── ExecutionStep enum
│   ├── DenseGemm { tile_idx }
│   ├── BandedKernel { tile_idx }
│   ├── BlockSparseKernel { tile_idx }
│   └── BinnedSpMM { bin_idx }
├── DenseTileData<R>
│   ├── row_range, col_range
│   └── extracted_dense: Buffer<F>
├── create_plan(csr, n_output_cols, config) → SpmmPlan
│   ├── Compute statistics
│   ├── Tile decomposition (if large enough)
│   ├── For each tile:
│   │   ├── If DENSE → extract to dense buffer
│   │   ├── If BANDED → record params
│   │   ├── If BLOCK_SPARSE → extract blocks
│   │   └── If SPARSE → handled by binning
│   ├── Create binning for sparse regions
│   └── Build execution order
├── PlanCache
│   ├── cache: HashMap<(MatrixId, N), SpmmPlan>
│   ├── get_or_create()
│   └── invalidate(matrix_id)
└── Tests: plan creation for various matrix types

Done when: Can create execution plan for any matrix

Phase 3: SpMM Algorithm Suite
Goal: Implement all sparse SpMM kernels (non-Gather-GEMM)
Duration: 5 weeks
Milestone 3.1: Row-Split Kernel
Tasks:
├── Algorithm design (from spec):
│   ├── One thread per (row, output_column) pair
│   ├── Sequential iteration over row's non-zeros
│   └── Direct output write
├── Kernel implementation
│   ├── Grid: (num_rows, ceil(N / TILE_N))
│   ├── Block: (BLOCK_SIZE, 1)
│   ├── Thread computes TILE_N output columns
│   └── Vectorized B loads (float4) when possible
├── Variants:
│   ├── row_split_basic - simplest
│   ├── row_split_vectorized - float4 B access
│   └── row_split_tiled - shared memory B staging
├── Launch configuration tuning
│   ├── Block size selection
│   └── Tile size selection
├── Correctness tests vs dense matmul
└── Performance baseline benchmarks

Done when: Row-split correct, ~60% of cuSPARSE perf
Milestone 3.2: Warp-Per-Row Kernel
Tasks:
├── Algorithm design:
│   ├── One warp (32 threads) per row
│   ├── Parallel iteration over non-zeros
│   ├── Warp shuffle reduction
│   └── Lane 0 writes output
├── Warp primitives
│   ├── warp_reduce_sum(val) using shfl_xor
│   ├── warp_broadcast(val, src_lane)
│   └── warp_any(predicate)
├── Kernel implementation
│   ├── Grid: (ceil(M / WARPS_PER_BLOCK), ceil(N / TILE_N))
│   ├── Warp assignment from thread indices
│   ├── Strided nnz iteration (lane L handles nnz L, L+32, ...)
│   └── Per-output-column reduction
├── Variants:
│   ├── warp_per_row_basic
│   ├── warp_per_row_vectorized - each lane handles vec_width cols
│   └── warp_per_row_shared_b - stage B tile in shared memory
├── Multi-warp-per-row extension
│   ├── For rows with nnz > 256
│   ├── Multiple warps cooperate
│   └── Inter-warp reduction via shared memory or atomics
└── Tests and benchmarks

Done when: Warp kernel faster than row-split for nnz > 16
Milestone 3.3: Merge-Path Kernel
Tasks:
├── Merge-path concept implementation:
│   ├── Work items = M (row starts) + nnz (elements)
│   ├── Even distribution across threads
│   └── Binary search for starting position
├── merge_path_search(row_ptrs, M, target) → (row, nnz_offset)
│   ├── Binary search over rows
│   ├── Find row where target falls
│   └── Compute nnz offset within row
├── Kernel implementation
│   ├── Each thread computes work range
│   ├── Find starting (row, nnz) via search
│   ├── Process work items sequentially
│   ├── Track row boundaries
│   └── Atomic add at row boundaries
├── Optimizations:
│   ├── Block-level merge-path (one search per block)
│   ├── Shared memory for row_ptrs segment
│   └── Coalesced B access patterns
├── Analysis: when merge-path wins
│   ├── CV > 1.5 (irregular distributions)
│   ├── Power-law row lengths
│   └── Bimodal distributions
└── Tests with synthetic irregular matrices

Done when: Merge-path handles extreme irregularity correctly
Milestone 3.4: SELL-C Kernel
Tasks:
├── Algorithm design:
│   ├── Process one slice at a time
│   ├── Within slice: ELL-style regular access
│   ├── One warp per row (or one thread for short)
│   └── Perfectly coalesced within slice
├── Kernel implementation
│   ├── Grid: (num_slices, ceil(N / TILE_N))
│   ├── Each block handles one slice
│   ├── Threads assigned to rows within slice
│   └── Regular strided access to col_indices/values
├── Handle permutation
│   ├── Output to permuted row indices
│   └── Or apply inverse permutation after
├── Slice-adaptive strategy
│   ├── Short slices: thread-per-row
│   ├── Medium slices: warp-per-row
│   └── Long slices: multi-warp
└── Tests and comparison vs CSR kernels

Done when: SELL-C kernel works, faster for irregular matrices
Milestone 3.5: BSR Kernel
Tasks:
├── Algorithm design:
│   ├── Process block-row at a time
│   ├── For each block: dense block × B slice
│   ├── Accumulate into output block-row
│   └── Leverage tensor cores for block matmuls
├── Block matmul strategies:
│   ├── 16×16 blocks → direct WMMA
│   ├── 32×32 blocks → tiled WMMA
│   ├── Other sizes → CUDA core dense
├── Kernel implementation
│   ├── Grid: (num_block_rows, ceil(N / TILE_N))
│   ├── Load block values to shared/registers
│   ├── Load B slice to shared memory
│   ├── Dense matmul (use cubecl-linalg or inline)
│   └── Accumulate to output
├── Handle block padding
│   ├── Incomplete blocks at matrix edges
│   └── Mask out padding in output
└── Tests and benchmarks vs CSR

Done when: BSR faster than CSR for block-sparse patterns
Milestone 3.6: Outer Product (CSC) Kernel
Tasks:
├── Algorithm design:
│   ├── Iterate columns of A (via CSC)
│   ├── For each column k: A[:,k] ⊗ B[k,:]
│   ├── Outer product = sparse col × dense row
│   └── Accumulate into C (atomic or partitioned)
├── Parallelization strategies:
│   ├── Column-parallel: one block per column, atomic to C
│   ├── Row-partitioned: partition C rows, each block handles all cols for its rows
│   └── Hybrid: column groups with row partitioning
├── Kernel implementation (column-parallel)
│   ├── Grid: (num_columns, ceil(N / TILE_N))
│   ├── Load B[k, :] to shared memory (one row, coalesced)
│   ├── For each non-zero (row_idx, val) in column:
│   │   └── Atomic add: C[row_idx, :] += val * B[k, :]
│   └── Vectorized atomic adds for efficiency
├── When outer product wins:
│   ├── B row access is perfectly coalesced
│   ├── Good when columns have clustered row indices
│   └── Bad when many atomic conflicts
└── Tests comparing to row-based approaches

Done when: Outer product kernel correct, wins for suitable patterns
Milestone 3.7: Algorithm Dispatcher
Tasks:
├── SpmmAlgorithm enum
│   ├── RowSplit
│   ├── WarpPerRow
│   ├── VectorWarpPerRow { vec_width }
│   ├── MergePath
│   ├── SellC
│   ├── Bsr
│   ├── OuterProduct
│   ├── GatherGemm         (Phase 4)
│   └── GatherTensorCore   (Phase 4)
├── select_algorithm(stats, n, device, format) → SpmmAlgorithm
│   ├── Format-specific paths (BSR → Bsr, SELL-C → SellC)
│   ├── Irregularity check (CV > 1.5 → MergePath)
│   ├── Density checks (per bin in Phase 4)
│   └── Tensor core availability
├── Dispatch function
│   ├── dispatch_spmm(csr, b, algorithm) → c
│   └── Routes to correct kernel
├── Unified API
│   ├── spmm(sparse, dense) → dense
│   ├── Auto-selects format, algorithm
│   └── Uses planning when beneficial
└── Tests covering all dispatch paths

Done when: Can call spmm(), gets routed to best kernel

Phase 4: Gather-GEMM + Tensor Cores
Goal: Implement our core innovation—Adaptive Binned Gather-GEMM
Duration: 4 weeks
This is where we beat cuSPARSE.
Milestone 4.1: Gather Infrastructure
Tasks:
├── Gather pattern analysis
│   ├── Per-warp coalescing requirements
│   ├── Shared memory bank conflict avoidance
│   └── Vectorized gather (float4)
├── Cooperative gather kernel
│   ├── Warp gathers one B row together
│   ├── All lanes access consecutive B columns
│   ├── Store to shared memory
│   └── Handle padding for short rows
├── Shared memory layout design
│   ├── smem[batch_rows][padded_nnz][tile_n]
│   ├── Pad padded_nnz dimension to avoid bank conflicts
│   │   └── Use padded_nnz + 1 for stride
│   └── Alignment for tensor core fragments
├── Double-buffered gather
│   ├── Pipeline: gather tile K+1 while computing tile K
│   ├── Two smem buffers
│   └── Async copy where available (Ampere+)
├── Gather bandwidth measurement
│   ├── Benchmark scattered vs coalesced gather
│   └── Validate efficiency assumptions
└── Tests: gather correctness, bank conflict checks

Done when: Gather achieves >80% of theoretical bandwidth
Milestone 4.2: Gather-GEMM Kernel (CUDA Cores)
Tasks:
├── Kernel design:
│   ├── Process one bin at a time
│   ├── Each block handles batch of rows from bin
│   ├── Phase 1: Cooperative gather B tiles to smem
│   ├── Phase 2: Dense GEMM on gathered data
│   └── Write output
├── Detailed algorithm:
│   For each K tile:
│       // Gather phase
│       for local_row in 0..batch_rows:
│           for k in 0..tile_k (strided by warp):
│               b_row = gather_cols[local_row * padded_nnz + k_offset + k]
│               for j in lane_id..(tile_n) step 32:
│                   smem_B[local_row][k][j] = B[b_row * N + j_offset + j]
│       sync_threads()
│       
│       // Compute phase
│       for local_row in assigned_rows:
│           for j in assigned_cols:
│               for k in 0..tile_k:
│                   acc[local_row][j] += gather_vals[...][k] * smem_B[local_row][k][j]
│       sync_threads()
│   
│   // Write output
│   for local_row, j:
│       C[row_indices[local_row] * N + j_offset + j] = acc[local_row][j]
├── Tile size tuning
│   ├── batch_rows: 16-32 (shared memory limited)
│   ├── tile_k: 32-64 (K dimension)
│   ├── tile_n: 32-64 (output columns)
│   └── Auto-tune based on matrix size
├── Launch configuration
│   ├── Grid: (num_batches, ceil(N / tile_n))
│   ├── Block: 256 threads typical
│   └── Shared memory: request max
└── Tests vs reference SpMM

Done when: Gather-GEMM correct, faster than warp-per-row for nnz > 32
Milestone 4.3: Tensor Core Integration
Tasks:
├── WMMA/CMMA abstraction
│   ├── Use CubeCL's CMMA primitives
│   ├── Fragment types: 16×16×16
│   └── Precision configurations (fp16→fp32, etc.)
├── Fragment loading from gathered data
│   ├── A fragment: load from gather_vals
│   │   └── A_frag[r][c] = gather_vals[batch_row + r][k_base + c]
│   ├── B fragment: load from smem_B
│   │   └── Must match expected layout
│   └── Handle row-major vs col-major
├── Kernel implementation
│   For each K tile (step by 16):
│       // Gather to shared memory (same as CUDA core version)
│       cooperative_gather(...)
│       sync_threads()
│       
│       // Load fragments
│       wmma::load_matrix_sync(a_frag, &gather_vals[...], stride)
│       wmma::load_matrix_sync(b_frag, &smem_B[...], stride)
│       
│       // Tensor core computation
│       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)
│       
│       sync_threads()
│   
│   // Store accumulator
│   wmma::store_matrix_sync(&C[...], c_frag, N, wmma::mem_row_major)
├── Precision support
│   ├── fp16 input, fp16 accumulator (fast)
│   ├── fp16 input, fp32 accumulator (accurate)
│   ├── bf16 input, fp32 accumulator
│   └── tf32 (fp32 with reduced mantissa)
├── Capability detection
│   ├── Check sm_80+ for sparse tensor cores
│   ├── Fallback to CUDA core Gather-GEMM
│   └── Fallback to basic SpMM kernels
└── Benchmarks vs cuSPARSE

Done when: Tensor core kernel works on Ampere+, 2-3× faster than cuSPARSE for medium density
Milestone 4.4: Dense Tile Handler
Tasks:
├── Dense tile extraction kernel
│   ├── Input: CSR + tile bounds
│   ├── Output: Dense buffer (tile_m × tile_k)
│   ├── Scatter from CSR to dense positions
│   └── Zero-fill missing positions
├── Dense GEMM dispatch
│   ├── Use cubecl-linalg GEMM
│   ├── Tensor core path when available
│   └── Accumulate into C output region
├── Integration with tile decomposition
│   ├── For each DENSE tile in plan
│   ├── Extract once during planning
│   └── GEMM during execution
├── Break-even analysis
│   ├── When is extract+GEMM faster than sparse?
│   ├── Track overhead vs benefit
│   └── Tune dense_threshold based on data
└── Tests with partially dense matrices

Done when: Dense tiles correctly extracted and computed via GEMM
Milestone 4.5: Banded Tile Handler
Tasks:
├── Banded kernel design
│   ├── Only load B rows within bandwidth
│   ├── Slide window along diagonal
│   └── Process multiple rows sharing B window
├── Kernel implementation
│   ├── Shared memory: B[bandwidth][tile_n]
│   ├── Load B rows for current diagonal window
│   ├── Compute dot products for rows in window
│   ├── Slide window, update B incrementally
│   └── Write output
├── Window management
│   ├── Lower and upper bandwidth
│   ├── Handle asymmetric bands
│   └── Edge cases at matrix boundaries
└── Tests with banded matrices (tridiagonal, pentadiagonal, etc.)

Done when: Banded kernel correct, faster than general SpMM for banded patterns
Milestone 4.6: Block-Sparse Tile Handler
Tasks:
├── Block extraction
│   ├── From tile classification: list of dense blocks
│   ├── Extract each block to contiguous buffer
│   └── Build block index structure
├── Block-sparse GEMM
│   ├── For each block (br, bc):
│   │   └── C[br*bs : (br+1)*bs, :] += block @ B[bc*bs : (bc+1)*bs, :]
│   ├── Batch blocks with same bc for B reuse
│   └── Use tensor cores per block when bs ≥ 16
├── Integration with BSR format
│   ├── If tile detected as block-sparse, convert to BSR locally
│   └── Use BSR kernel
└── Tests with block-diagonal and checkerboard patterns

Done when: Block-sparse tiles handled efficiently
Milestone 4.7: Full Plan Execution
Tasks:
├── SpmmPlan::execute(csr, b, c)
│   ├── For each step in execution_order:
│   │   ├── DenseGemm → launch dense GEMM
│   │   ├── BandedKernel → launch banded kernel  
│   │   ├── BlockSparseKernel → launch block-sparse
│   │   └── BinnedSpMM → dispatch per-bin kernels
│   └── Synchronize
├── Per-bin dispatch
│   ├── Based on bin.strategy:
│   │   ├── RowSplit → launch row_split kernel
│   │   ├── WarpPerRow → launch warp_per_row kernel
│   │   ├── GatherGemm → launch gather_gemm kernel
│   │   ├── GatherTensorCore → launch gather_tc kernel
│   │   └── MergePath → launch merge_path kernel
│   └── Use bin's prepared gather buffers
├── Multi-stream execution
│   ├── Different bins on different streams
│   ├── Dense tiles can overlap with sparse
│   └── Synchronize at end
├── Memory management
│   ├── Allocate output C
│   ├── Scratch space from pool
│   └── Release scratch after execution
└── End-to-end correctness tests

Done when: Full plan execution matches reference for all matrix types
Milestone 4.8: Performance Validation
Tasks:
├── Benchmark suite setup
│   ├── SuiteSparse matrix collection (subset)
│   ├── Synthetic matrices (random, banded, block, power-law)
│   ├── ML matrices (pruned networks)
│   └── Various sizes (1K to 1M rows)
├── Comparison targets
│   ├── cuSPARSE (NVIDIA)
│   ├── Our basic SpMM (Phase 3)
│   ├── Dense GEMM (upper bound)
│   └── Theoretical peak
├── Metrics
│   ├── Time (ms)
│   ├── GFLOPS (2 × nnz × N / time)
│   ├── Memory bandwidth utilization
│   ├── Tensor core utilization
│   └── Algorithm breakdown (time per component)
├── Analysis
│   ├── Identify winning regimes
│   ├── Find regression cases
│   ├── Tune thresholds based on data
│   └── Document performance model accuracy
└── Publish initial results

Target: 2-4× faster than cuSPARSE for matrices with avg_nnz > 32

Phase 5: Autodiff + Training
Goal: Backward pass, gradient computation, training integration
Duration: 3 weeks
Dependencies: Phase 4 complete (Gather-GEMM for efficient forward)
Milestone 5.1: SpMM Backward Theory
Forward: C = A @ B
Where: A is (M × K) sparse, B is (K × N) dense, C is (M × N) dense

Backward given dL/dC:
    dL/dA: Sparse gradient, same pattern as A
           (dL/dA)[i,k] = Σ_j (dL/dC)[i,j] × B[k,j]
                        = (dL/dC)[i,:] · B[k,:]
           Only compute for (i,k) where A[i,k] ≠ 0
    
    dL/dB: Dense gradient
           dL/dB = A^T @ dL/dC
           Use CSC view of A or transpose

Design verification:
    ├── Derive gradients mathematically
    ├── Verify dimensions match
    └── Identify computational patterns
Milestone 5.2: Sparse Gradient Kernel (dL/dA)
Tasks:
├── Kernel design
│   ├── One thread per non-zero of A
│   ├── For non-zero at (i, k):
│   │   └── grad[nnz_idx] = dot(dL_dC[i, :], B[k, :])
│   ├── Need to find row i from nnz_idx
│   │   └── Binary search in row_ptrs or precompute row_indices
│   └── Dot product: dL_dC row × B row
├── Row index lookup
│   ├── Option A: Binary search row_ptrs
│   ├── Option B: Precompute row_indices array
│   └── Option B preferred (O(1) vs O(log M))
├── Kernel implementation
│   ├── Grid: (ceil(nnz / BLOCK_SIZE), 1)
│   ├── Each thread handles one gradient element
│   ├── Load row index, column index
│   ├── Compute dot product (vectorized)
│   └── Write gradient
├── Optimization: Warp-collaborative dot product
│   ├── If N large, multiple threads per gradient
│   ├── Warp reduction for final sum
│   └── Trade parallelism for efficiency
└── Correctness test: Finite difference verification

Done when: dL/dA matches finite difference within tolerance
Milestone 5.3: Dense Gradient (dL/dB) via Transpose
Tasks:
├── Use existing SpMM with transposed A
│   ├── dL/dB = A^T @ dL/dC
│   ├── A^T is CSC view of CSR A
│   └── Dispatch to appropriate kernel
├── Transpose-free variant
│   ├── Outer product formulation
│   ├── For each row i of A:
│   │   ├── For each non-zero (i, k, val) in row:
│   │   │   └── dL_dB[k, :] += val × dL_dC[i, :]
│   │   └── Atomic accumulation
│   └── May be faster for some sparsity patterns
├── Implementation
│   ├── Primary: CSC view + SpMM
│   ├── Fallback: Explicit transpose if CSC not available
│   └── Select based on operation context
└── Correctness test: Finite difference verification

Done when: dL/dB correct for all input patterns
Milestone 5.4: Backward for Gather-GEMM
Tasks:
├── Analysis: Does Gather-GEMM change gradients?
│   ├── No! Same mathematical operation
│   ├── Forward uses Gather-GEMM for speed
│   └── Backward uses appropriate gradient kernels
├── Potential optimization: Gather-aware backward
│   ├── Reuse bin structure from forward
│   ├── For dL/dA: bin structure gives row grouping
│   │   └── Batch gradient computations by bin
│   ├── For dL/dB: gather structure not directly helpful
│   └── Implement if profiling shows benefit
├── Implementation
│   ├── Use standard gradient kernels initially
│   ├── Profile backward pass
│   └── Optimize hot spots
└── End-to-end backward correctness test

Done when: Backward through Gather-GEMM matches reference
Milestone 5.5: Mask Preservation
Tasks:
├── MaskPreservation enum
│   ├── ExactPattern: Gradient only at existing non-zeros
│   ├── AllowGrowth: Track gradient at all positions (for regrowth)
│   └── AllowShrinkage: Prune zeros during training
├── ExactPattern implementation (default)
│   ├── dL/dA naturally has same pattern (kernel only computes at nnz)
│   ├── Verify pattern match after backward
│   └── Error if pattern mismatch detected
├── AllowGrowth implementation (for RigL etc.)
│   ├── Maintain dense gradient buffer
│   ├── Sparse gradient for existing positions
│   ├── Dense accumulator for candidate positions
│   └── Use for regrowth decisions
├── SparseParameter struct
│   ├── storage: SparseStorage
│   ├── mask_mode: MaskPreservation
│   ├── gradient: Option<SparseGradient>
│   └── dense_gradient_buffer: Option<Buffer> (for AllowGrowth)
└── Tests for each mode

Done when: Mask preservation modes work correctly
Milestone 5.6: Autodiff Integration
Tasks:
├── Autodiff node types
│   ├── SparseMatmulNode { a_id, b_id, c_id }
│   ├── SparseAddNode, SparseMulNode (element-wise)
│   └── SparseToDenseNode, DenseToSparseNode
├── Tape recording
│   ├── Record sparse ops during forward
│   ├── Store necessary intermediates
│   │   └── For SpMM: need A, B for backward
│   └── Handle memory efficiently (don't duplicate large tensors)
├── Backward traversal
│   ├── Topological sort of computation graph
│   ├── Dispatch backward ops
│   ├── Accumulate gradients
│   └── Handle sparse + dense mixed graphs
├── Burn integration
│   ├── Implement Autodiff trait for sparse tensors
│   ├── Register sparse backward operations
│   └── Gradient extraction API
└── Multi-op backward test

Done when: Can autodiff through SpMM → ReLU → SpMM chain
Milestone 5.7: Training Loop
Tasks:
├── SparseLinear module (minimal)
│   ├── weight: SparseParameter
│   ├── bias: Option<DenseParameter>
│   ├── forward(x) → SpMM(weight, x) + bias
│   └── Parameter registration
├── Sparse optimizers
│   ├── SparseSGD
│   │   └── w = w - lr × grad (sparse update)
│   ├── SparseAdam
│   │   ├── Maintain m, v as sparse (same pattern)
│   │   └── Update formula
│   └── Gradient clipping utilities
├── Training loop test
│   ├── Small MLP: Dense → SparseLinear → ReLU → Dense → Loss
│   ├── Forward → Loss → Backward → Optimizer step
│   ├── Verify loss decreases
│   └── Verify weight pattern preserved
├── Memory leak testing
│   ├── Run many iterations
│   ├── Check memory growth
│   └── Fix any leaks
└── Convergence test vs dense baseline

Done when: Can train sparse MLP, achieves similar accuracy to dense

Phase 6: N:M Structured Sparsity
Goal: 2:4 sparsity with tensor core acceleration
Duration: 3 weeks
Milestone 6.1: N:M Storage Format
Tasks:
├── NMStorage<R, F, const N: u32, const M: u32> struct
│   ├── values: Buffer<F>        // [rows × (K/M) × N] compressed values
│   ├── indices: Buffer<u16>     // [rows × (K/M)] packed index masks
│   ├── shape: SparseShape
│   └── Compile-time N, M validation
├── Index encoding for 2:4
│   ├── 4-bit mask: which 2 of 4 positions have values
│   ├── 6 possible patterns: 0011, 0101, 0110, 1001, 1010, 1100
│   ├── Pack 4 masks per u16 for efficiency
│   └── Decode functions
├── Memory layout for tensor cores
│   ├── Align to 16-element boundaries
│   ├── Match mma.sp expected format
│   └── Interleaved vs. planar options
├── Metadata
│   ├── Effective nnz = rows × (K/M) × N
│   ├── Compression ratio = M/N
│   └── density() → f32
└── Construction placeholder (filled in next milestone)

Done when: Can create N:M storage with correct layout
Milestone 6.2: N:M Pruning
Tasks:
├── Pruning algorithm (dense → 2:4)
│   ├── For each group of M=4 consecutive elements:
│   │   ├── Find top N=2 by magnitude
│   │   ├── Encode positions as index mask
│   │   └── Store values and mask
│   └── Handle edge cases (ties, zeros)
├── GPU pruning kernel
│   ├── Grid: (rows, K/M groups)
│   ├── Load group of 4 elements
│   ├── Parallel top-2 selection
│   │   └── Comparison network or sort
│   ├── Encode mask
│   └── Write to output
├── PruningStrategy::NM { n, m }
├── API: nm_prune(dense, n, m) → NMStorage
├── Pruning quality metrics
│   ├── Magnitude sum preserved
│   ├── Pattern distribution
│   └── Per-row statistics
└── Tests: verify 50% sparsity, correct values selected

Done when: Can prune any dense tensor to valid 2:4 format
Milestone 6.3: N:M SpMM Emulated Kernel
Tasks:
├── Algorithm (software path):
│   ├── For each row i, group g:
│   │   ├── Decode index mask → positions [p0, p1]
│   │   ├── Load values v0, v1
│   │   ├── Compute k0 = g*4 + p0, k1 = g*4 + p1
│   │   ├── Gather B[k0, :] and B[k1, :]
│   │   └── Accumulate: C[i, :] += v0×B[k0, :] + v1×B[k1, :]
├── Kernel implementation
│   ├── Grid: (rows, ceil(N / TILE_N))
│   ├── Each thread handles one row, tile of output cols
│   ├── Sequential over groups
│   ├── Decode + gather + accumulate
│   └── Vectorized B access
├── Optimization: Warp-level
│   ├── Warp processes multiple rows
│   ├── Shared memory for B tiles
│   └── Coalesced gather
├── Use as fallback on non-Ampere
└── Correctness tests vs dense matmul

Done when: N:M SpMM correct on any GPU
Milestone 6.4: N:M Tensor Core Kernel
Tasks:
├── CUDA capability detection
│   ├── sm_80+ required for mma.sp
│   ├── Query at runtime
│   └── Fallback path selection
├── mma.sp intrinsic understanding
│   ├── Input format requirements
│   │   ├── A: compressed 2:4 format
│   │   └── B: dense, specific layout
│   ├── Index metadata format
│   ├── Tile sizes: 16×16×16 (M×N×K logical, K/2 physical)
│   └── Output accumulator
├── Kernel implementation
│   ├── Load A fragments (compressed)
│   ├── Load index metadata
│   ├── Load B fragments (dense)
│   ├── Execute mma.sp instruction
│   │   └── Hardware handles sparse gather internally
│   ├── Accumulate across K tiles
│   └── Store output
├── CubeCL CMMA sparse extension
│   ├── cmma_load_sparse_a(...)
│   ├── cmma_mma_sp(...)
│   └── Abstract hardware details
├── Performance comparison vs cuSPARSE 2:4
└── Benchmarks on Ampere/Hopper

Done when: N:M tensor core path works, matches or exceeds cuSPARSE
Milestone 6.5: N:M Backward and Training
Tasks:
├── N:M backward for dL/dA
│   ├── Gradient at structured positions only
│   ├── Decode index to find positions
│   ├── Compute gradient: grad[i, k] = dL_dC[i, :] · B[k, :]
│   └── Store in same N:M format
├── N:M backward for dL/dB (dense)
│   ├── A^T @ dL/dC where A is 2:4
│   ├── Use N:M SpMM with transposed logic
│   └── Output is dense
├── NMParameter struct
│   ├── storage: NMStorage
│   ├── gradient: Option<NMGradient>
│   └── Training methods
├── End-to-end N:M training test
│   ├── Prune dense weights to 2:4
│   ├── Train with N:M sparse ops
│   ├── Compare accuracy to dense
│   └── Compare throughput
└── Document training workflow

Done when: Can train 2:4 sparse models end-to-end

Phase 7: Integration Layer
Goal: Views, batching, interop, Burn modules, serialization
Duration: 3 weeks
Milestone 7.1: View System
Tasks:
├── SparseView enum
│   ├── Full (entire tensor)
│   ├── Transpose (CSR ↔ CSC reinterpret)
│   ├── RowSlice { start, end }
│   └── (ColSlice for CSC)
├── SparseTensorView struct
│   ├── base: &SparseTensor
│   ├── view: SparseView
│   └── Deferred operations
├── Transpose view (zero-cost)
│   ├── CSR viewed as CSC of transpose
│   ├── No data movement
│   └── Affects SpMM dispatch
├── Row slice view
│   ├── Offset into row_ptrs
│   ├── Bounds checking
│   └── Alignment validation for N:M, BSR
├── View materialization
│   ├── materialize() → owned SparseTensor
│   └── Use when view not directly supported
├── View-aware operation dispatch
│   ├── Check view type before kernel selection
│   ├── Use transpose kernel for transpose views
│   └── Slice handling in kernels
└── Tests for all view types

Done when: Views work correctly in operations
Milestone 7.2: Batched Operations
Tasks:
├── BatchedSparseStorage enum
│   ├── Uniform { shared_indices, batched_values }
│   │   └── All batch elements have same sparsity pattern
│   └── Variable { per_element_storage }
│       └── Different patterns per element
├── BatchedSparseTensor struct
│   ├── storage: BatchedSparseStorage
│   ├── batch_size: u32
│   └── element_shape: SparseShape
├── Construction
│   ├── stack_uniform(tensors) → BatchedSparseTensor
│   │   └── Verify identical patterns
│   ├── stack_variable(tensors) → BatchedSparseTensor
│   └── unbatch(batched) → Vec<SparseTensor>
├── Batched SpMM (uniform)
│   ├── Same A pattern, batched values
│   ├── B can be batched or broadcast
│   ├── Kernel: batch dimension in grid
│   └── Reuse gather structure across batch
├── Batched SpMM (variable)
│   ├── Loop over batch elements
│   └── Or: concatenate and use segment info
├── Batch slicing
│   ├── batched[i] → SparseTensor (view)
│   └── Handle both uniform and variable
└── Tests with GNN-style batching

Done when: Batched operations correct and efficient
Milestone 7.3: Sparse-Dense Interop
Tasks:
├── Mixed operations
│   ├── sparse_add_dense(sparse, dense) → dense
│   ├── sparse_mul_dense(sparse, dense) → sparse (Hadamard)
│   ├── sparse_sub_dense, dense_sub_sparse
│   └── Broadcasting support
├── Conversion operations
│   ├── to_dense(sparse) → Tensor
│   │   └── All formats supported
│   ├── to_sparse(dense, format, threshold) → SparseTensor
│   │   └── Threshold-based sparsification
│   └── Conversion cost estimation
├── Auto-conversion policy
│   ├── ConversionPolicy enum
│   │   ├── Never (error on mismatch)
│   │   ├── PreferSparse
│   │   ├── PreferDense
│   │   └── Auto (based on operation)
│   └── Apply during mixed operations
├── Format-specific handling
│   ├── CSR + Dense → Dense
│   ├── BSR + Dense (block-aligned) → optimization
│   └── N:M + Dense → Dense
└── Tests for all combinations

Done when: Mixed sparse-dense operations work seamlessly
Milestone 7.4: Burn Modules
Tasks:
├── SparseLinear module
│   ├── SparseLinearConfig
│   │   ├── input_dim, output_dim
│   │   ├── sparsity: f32
│   │   ├── format: FormatHint
│   │   └── bias: bool
│   ├── Initialization
│   │   ├── Create dense weights
│   │   ├── Prune to target sparsity
│   │   └── Convert to sparse format
│   ├── forward(x) → y
│   │   └── SpMM(weight, x) + bias
│   └── Parameter registration
├── SparseEmbedding module
│   ├── Embedding table as sparse (for sparse vocabulary)
│   ├── Lookup: gather rows
│   └── Gradient: sparse update
├── SparseConv2d (future consideration)
│   └── Sparse filters
├── Module derive integration
│   ├── #[derive(Module)] works with sparse params
│   └── Automatic gradient handling
├── Pretrained weight loading
│   ├── Load dense, prune to sparse
│   └── Load sparse directly
└── Tests: train models with sparse modules

Done when: SparseLinear usable in Burn models
Milestone 7.5: Serialization
Tasks:
├── SparseTensorHeader format (JSON)
│   ├── version: u32
│   ├── format: String (CSR, BSR, etc.)
│   ├── dtype: String
│   ├── index_dtype: String
│   ├── shape: [u64; 3] (rows, cols, nnz)
│   ├── format_specific: Map (block_size, etc.)
│   └── checksum: Option<String>
├── save(sparse, path)
│   ├── Write header as JSON
│   ├── Write buffers as raw bytes
│   │   └── Order defined by format
│   └── Compute checksum
├── load(path) → SparseTensor
│   ├── Read and parse header
│   ├── Validate version compatibility
│   ├── Read buffers
│   ├── Verify checksum
│   └── Construct tensor
├── Checkpoint integration
│   ├── Compatible with Burn checkpointing
│   └── Sparse params saved/loaded correctly
├── safetensors compatibility (stretch)
│   └── Custom format marker for sparse
└── Round-trip tests

Done when: Can save and load sparse tensors across sessions

Phase 8: Advanced Features
Goal: Dynamic sparsity, RigL, SpGEMM, fusion, autotuning
Duration: 4 weeks
Milestone 8.1: Dynamic Sparsity Infrastructure
Tasks:
├── SparsityPattern struct (indices only, no values)
│   ├── Represent pattern separately from values
│   └── Enable pattern comparison and modification
├── PatternDelta struct
│   ├── additions: Vec<(row, col)>
│   ├── removals: Vec<(row, col)>
│   └── Apply to pattern
├── PatternManager struct
│   ├── current_pattern: SparsityPattern
│   ├── mode: PatternMode (Static, GrowOnly, Full)
│   ├── pending_delta: PatternDelta
│   ├── queue_add(row, col)
│   ├── queue_remove(row, col)
│   └── apply_pending() → rebuild storage
├── Pattern modification workflow
│   ├── Accumulate changes during training
│   ├── Apply in batch (expensive)
│   └── Rebuild CSR/BSR from modified pattern
├── Pattern history (optional)
│   └── Track pattern evolution over training
└── Tests for pattern modification

Done when: Can modify sparsity pattern at runtime
Milestone 8.2: RigL Training
Tasks:
├── Gradient tracking for regrowth
│   ├── CooGradientAccumulator
│   │   ├── Sparse gradient (existing positions)
│   │   └── Dense buffer (candidate positions)
│   ├── accumulate_sparse(sparse_grad)
│   ├── accumulate_dense(dense_grad)
│   └── regrowth_candidates(k) → top-k positions
├── rigl_update(params, fraction)
│   ├── Find lowest magnitude weights (drop candidates)
│   ├── Find highest gradient zeros (grow candidates)
│   ├── Drop fraction of weights
│   ├── Grow same number at candidate positions
│   └── Maintain constant sparsity
├── Update scheduling
│   ├── MaskUpdateScheduler
│   │   ├── update_every_n_steps
│   │   ├── cosine_schedule
│   │   └── Custom schedule
│   ├── fraction_schedule (start high, decay)
│   └── Integration with training loop
├── RigL training loop test
│   ├── Initialize with random sparse
│   ├── Train with periodic mask updates
│   ├── Verify accuracy approaches dense
│   └── Compare different schedules
└── Document RigL usage

Done when: RigL training works, matches paper results
Milestone 8.3: SpGEMM (Sparse × Sparse)
Tasks:
├── SpGEMM analysis
│   ├── C = A @ B where both sparse
│   ├── Output pattern unknown a priori
│   └── Two-phase approach: symbolic + numeric
├── Symbolic phase kernel
│   ├── For each row of A:
│   │   ├── Find columns from B that contribute
│   │   └── Union of column sets
│   ├── Count unique columns → output nnz per row
│   ├── Hash table per row (shared memory)
│   └── Output: row_nnz array
├── Numeric phase kernel
│   ├── Allocate output CSR based on symbolic counts
│   ├── Recompute products
│   ├── Accumulate into output positions
│   └── Sort columns within rows (if needed)
├── SpGEMM API
│   └── spgemm(a: &CsrMatrix, b: &CsrMatrix) → CsrMatrix
├── SpGEMM backward
│   ├── dL/dA = dL/dC @ B^T (SpGEMM)
│   ├── dL/dB = A^T @ dL/dC (SpGEMM)
│   └── Pattern masking options
└── Tests with various sparsity patterns

Done when: SpGEMM correct, reasonable performance
Milestone 8.4: Kernel Fusion
Tasks:
├── Fusion opportunities
│   ├── SpMM + bias → single kernel
│   ├── SpMM + ReLU → single kernel
│   ├── SpMM + bias + ReLU → single kernel
│   ├── Element-wise chains on sparse
│   └── SpMM + residual add
├── Epilogue trait
│   ├── trait Epilogue { fn apply(val, row, col, aux) → val }
│   ├── NoOp, BiasAdd, ReLU, BiasReLU, GELU, etc.
│   └── Compile-time dispatch
├── Fused kernel generation
│   ├── SpMM kernels templated on Epilogue
│   ├── Fusion decision at plan creation time
│   └── Specialize kernel compilation
├── Fusion in operation graph
│   ├── Pattern matching in forward graph
│   ├── Identify fuseable sequences
│   └── Replace with fused ops
└── Benchmarks showing fusion benefit

Done when: Fused kernels faster than separate ops
Milestone 8.5: Autotuning
Tasks:
├── Tunable parameters
│   ├── Tile sizes (tile_m, tile_k, tile_n)
│   ├── Block sizes
│   ├── Bin boundaries
│   ├── Algorithm thresholds
│   └── Fusion decisions
├── AutotuneDatabase
│   ├── Store benchmark results
│   ├── Key: (matrix_features, device, operation)
│   ├── Value: best configuration
│   └── Persistence (save/load)
├── SparseAutotuner
│   ├── measure_config(config) → time
│   ├── search_configs(matrix, configs) → best
│   ├── Grid search for tile sizes
│   └── Random search for broader exploration
├── Integration with planning
│   ├── During plan creation, check autotune DB
│   ├── If miss, use heuristics
│   ├── Optional: tune on first use
│   └── Cache tuned configurations
├── Offline tuning script
│   └── Tune on representative matrix set
└── Tests for autotune workflow

Done when: Autotuning improves performance on unseen matrices

Phase 9: Polish + Production
Goal: Hardening, documentation, final benchmarks
Duration: 2 weeks
Milestone 9.1: Edge Cases and Hardening
Tasks:
├── Edge case tests
│   ├── Empty tensors (0 nnz)
│   ├── Single element
│   ├── Single row, single column
│   ├── Extremely sparse (0.001%)
│   ├── Nearly dense (99%)
│   ├── Very large (>100M nnz)
│   └── All-zero rows/columns
├── Error handling review
│   ├── Clear error messages
│   ├── Actionable guidance
│   └── No panics in library code
├── Memory safety
│   ├── No buffer overflows
│   ├── No use-after-free
│   └── Compute sanitizer checks
├── Memory leak testing
│   ├── Long training runs
│   ├── Repeated plan creation/destruction
│   └── Profile memory over time
├── Stress testing
│   ├── Many concurrent operations
│   ├── Rapid format conversions
│   └── Pattern modification stress
└── Fix all issues found

Done when: Library stable under stress
Milestone 9.2: Documentation
Tasks:
├── API documentation
│   ├── Rustdoc for all public types/functions
│   ├── Examples in doc comments
│   └── Module-level overviews
├── User guide
│   ├── Getting started
│   ├── Format selection guide
│   ├── Training with sparse weights
│   ├── N:M sparsity tutorial
│   └── Dynamic sparsity (RigL) tutorial
├── Examples
│   ├── example_sparse_mlp.rs
│   ├── example_nm_pruning.rs
│   ├── example_rigl_training.rs
│   ├── example_format_conversion.rs
│   └── example_benchmark.rs
├── Performance guide
│   ├── When to use sparse
│   ├── Format selection
│   ├── Tuning tips
│   └── Memory optimization
├── Troubleshooting guide
└── Architecture documentation (for contributors)

Done when: New user can learn from docs alone
Milestone 9.3: Final Benchmarks
Tasks:
├── Comprehensive benchmark suite
│   ├── SuiteSparse matrices (representative subset)
│   ├── ML workloads (pruned transformers, GNNs)
│   ├── Scientific computing (FEM, CFD matrices)
│   └── Synthetic (controlled variables)
├── Comparison baselines
│   ├── cuSPARSE (primary comparison)
│   ├── rocSPARSE (AMD)
│   ├── Triton sparse
│   ├── PyTorch sparse
│   └── Dense (break-even analysis)
├── Metrics
│   ├── Throughput (GFLOPS)
│   ├── Time (ms)
│   ├── Memory usage
│   ├── Memory bandwidth utilization
│   └── Tensor core utilization
├── Analysis and visualization
│   ├── Speedup charts
│   ├── Roofline analysis
│   ├── Algorithm breakdown
│   └── Winning regimes identification
├── Publish results
│   └── README, blog post, or paper
└── Performance regression CI

Done when: Clear understanding of where we win and by how much
Milestone 9.4: Release Preparation
Tasks:
├── Version 0.1.0 checklist
│   ├── All Phase 0-9 milestones complete
│   ├── CI passing on all targets
│   ├── Documentation complete
│   ├── Benchmarks published
│   └── CHANGELOG.md
├── API stability review
│   ├── Identify stable vs unstable APIs
│   ├── Mark experimental features
│   └── Plan deprecations if needed
├── Upstream integration
│   ├── Coordinate with cubecl maintainers
│   ├── Integration PR
│   └── Review process
├── Announcement
│   ├── README showcase
│   ├── Burn integration documentation
│   └── Community outreach
└── Post-release monitoring
    └── Issue triage, bug fixes

Done when: cubecl-sparse v0.1.0 released

Dependency Graph (Updated)
Phase 0: Foundation
    │
    ▼
Phase 1: Storage Formats
    │
    ├─────────────────────┐
    ▼                     ▼
Phase 2: Analysis     Phase 3: SpMM Algorithms
    │                     │
    └──────────┬──────────┘
               ▼
    Phase 4: Gather-GEMM + TC ◄── OUR INNOVATION
               │
               ▼
    Phase 5: Autodiff + Training
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Phase 6: N:M          Phase 7: Integration
    │                     │
    └──────────┬──────────┘
               ▼
    Phase 8: Advanced Features
               │
               ▼
    Phase 9: Polish + Production

Critical Path
The critical path to "useful sparse training" is:
Phase 0 (2w) → Phase 1 (3w) → Phase 2 (2w) → Phase 3 (5w) → Phase 4 (4w) → Phase 5 (3w)
                                                                                │
Total: 19 weeks to trainable sparse models                                     ▼
                                                                         Can train!
After Phase 5, you can train sparse models. Phases 6-9 add features and polish.