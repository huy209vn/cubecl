CubeCL-Sparse: Architecture Specification
Version: 0.1.0-draft
Target: GPU-accelerated sparse tensor operations for Burn
Author: Huy
Status: Design Phase
All code is speculative since we are starting out.
Table of Contents

Design Principles & Goals
Format Specification
Memory Architecture
Operation Design
Algorithm Selection System
Autodiff Architecture
CubeCL Fusion Integration
Burn Backend Implementation
File Structure
Implementation Roadmap


1. Design Principles & Goals
1.1 Problem Statement
burn-sparse provides CPU-side sparse tensor abstractions (SparseTensor<B>, SparseParam<B>) with format stubs (COO, CSR, CSC, BlockCSR, N:M). The critical gap: no GPU acceleration. Sparse matmul falls back to dense conversion, making the abstraction useless for real workloads.
Modern sparse neural network training (90%+ sparsity, transformer pruning, RigL, lottery ticket hypothesis) requires GPU-native sparse operations that don't exist in the Burn ecosystem.
1.2 Goals
Primary:
├── GPU-native sparse formats with efficient memory layouts
├── SpMM performance competitive with cuSPARSE/Triton baselines
├── Full autodiff support for training sparse networks
├── Integration with CubeCL's JIT fusion system
└── Clean Burn backend implementation

Secondary:
├── N:M structured sparsity with tensor core acceleration
├── Dynamic sparsity pattern support (RigL, gradual pruning)
├── Multi-platform via CubeCL (CUDA primary, WGPU secondary)
└── Format auto-selection based on sparsity characteristics
1.3 Non-Goals

CPU sparse operations (use existing libraries)
Sparse convolutions (future work, different problem)
Graph neural network primitives (specialized domain)
Sparse attention mechanisms (built on top of this)

1.4 Design Principles
1. GPU-First Design
Every format and operation designed for GPU memory hierarchy. No CPU-side abstractions translated to GPU—native from the start.
2. Composition over Monolith
Small, composable kernels that can fuse. Not mega-kernels that handle every case.
3. Explicit over Magic
Format selection, algorithm choice, and fusion boundaries are explicit. Performance-critical code shouldn't surprise.
4. Static Primary, Dynamic Possible
Optimize for static sparsity patterns (post-training pruning). Support dynamic patterns (RigL) without sacrificing static performance.
5. Correctness then Performance
Autodiff correctness and numerical stability first. Optimize after the foundation is solid.

2. Format Specification
2.1 Core Trait Hierarchy
rust/// Marker trait for sparse storage formats
pub trait SparseFormat: Send + Sync + 'static {
    /// Format identifier for dispatch
    const FORMAT_ID: SparseFormatId;
    
    /// Whether this format supports efficient row access
    const ROW_MAJOR: bool;
    
    /// Whether this format supports efficient column access  
    const COL_MAJOR: bool;
    
    /// Whether sparsity pattern can change after construction
    const DYNAMIC_PATTERN: bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseFormatId {
    Csr,
    Csc,
    Coo,
    NM { n: u8, m: u8 },  // e.g., NM { n: 2, m: 4 } for 2:4
    Bsr { block_rows: u16, block_cols: u16 },
    Bcsc { block_rows: u16, block_cols: u16 },
}

/// Core sparse tensor storage trait
pub trait SparseStorage<R: Runtime>: SparseFormat {
    /// Metadata type (shape, nnz, block info, etc.)
    type Metadata: SparseMetadata;
    
    /// Number of GPU buffers this format requires
    const NUM_BUFFERS: usize;
    
    /// Create storage from dense tensor (sparsification)
    fn from_dense(
        dense: &Tensor<R>,
        threshold: f32,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Self;
    
    /// Convert to dense tensor
    fn to_dense(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Tensor<R>;
    
    /// Get metadata (cheap, no GPU sync)
    fn metadata(&self) -> &Self::Metadata;
    
    /// Actual sparsity ratio (nnz / total_elements)
    fn sparsity(&self) -> f32;
    
    /// Memory footprint in bytes
    fn memory_bytes(&self) -> usize;
}

/// Sparse tensor metadata (CPU-side, no GPU sync needed)
pub trait SparseMetadata: Clone + Send + Sync {
    fn shape(&self) -> &[usize];
    fn nnz(&self) -> usize;
    fn dtype(&self) -> DType;
}
2.2 CSR (Compressed Sparse Row) — Phase 1
The workhorse format. Efficient for row-wise operations, SpMM with dense right-hand side.
Memory Layout
CSR Matrix A (M × K) with NNZ non-zeros:

┌─────────────────────────────────────────────────────────────┐
│  row_ptrs: [u32; M+1]     — Cumulative nnz per row          │
│  col_indices: [u32; NNZ]  — Column index for each value     │
│  values: [T; NNZ]         — Non-zero values                 │
└─────────────────────────────────────────────────────────────┘

Example: 4×5 matrix with 6 non-zeros
     0   1   2   3   4
   ┌───┬───┬───┬───┬───┐
 0 │ 1 │   │ 2 │   │   │  row 0: 2 elements
   ├───┼───┼───┼───┼───┤
 1 │   │   │   │ 3 │   │  row 1: 1 element
   ├───┼───┼───┼───┼───┤
 2 │   │ 4 │   │   │ 5 │  row 2: 2 elements
   ├───┼───┼───┼───┼───┤
 3 │   │   │ 6 │   │   │  row 3: 1 element
   └───┴───┴───┴───┴───┘

row_ptrs:    [0, 2, 3, 5, 6]  (length M+1 = 5)
col_indices: [0, 2, 3, 1, 4, 2]  (length NNZ = 6)
values:      [1, 2, 3, 4, 5, 6]  (length NNZ = 6)

Row i has elements from row_ptrs[i] to row_ptrs[i+1] (exclusive)
Type Definition
rustpub struct CsrStorage<R: Runtime> {
    /// Row pointers buffer [M+1 elements, u32]
    pub row_ptrs: Handle<R::Server>,
    
    /// Column indices buffer [NNZ elements, u32]
    pub col_indices: Handle<R::Server>,
    
    /// Values buffer [NNZ elements, dtype]
    pub values: Handle<R::Server>,
    
    /// CPU-side metadata
    pub meta: CsrMetadata,
}

#[derive(Clone, Debug)]
pub struct CsrMetadata {
    pub rows: usize,      // M
    pub cols: usize,      // K  
    pub nnz: usize,       // Number of non-zeros
    pub dtype: DType,
    
    /// Row distribution statistics (for algorithm selection)
    pub row_stats: Option<RowStatistics>,
}

#[derive(Clone, Debug)]
pub struct RowStatistics {
    pub min_nnz_per_row: usize,
    pub max_nnz_per_row: usize,
    pub avg_nnz_per_row: f32,
    pub std_nnz_per_row: f32,
    
    /// Histogram buckets: [0-8), [8-32), [32-128), [128-512), [512+)
    pub row_length_histogram: [u32; 5],
}

impl SparseFormat for CsrStorage<R> {
    const FORMAT_ID: SparseFormatId = SparseFormatId::Csr;
    const ROW_MAJOR: bool = true;
    const COL_MAJOR: bool = false;
    const DYNAMIC_PATTERN: bool = false;
}
Construction Operations
rustimpl<R: Runtime> CsrStorage<R> {
    /// From COO (sorted by row, then column)
    pub fn from_coo(coo: &CooStorage<R>, client: &ComputeClient<...>) -> Self;
    
    /// From dense with threshold
    pub fn from_dense(dense: &Tensor<R>, threshold: f32, client: &ComputeClient<...>) -> Self;
    
    /// From explicit components (unsafe: caller ensures validity)
    pub unsafe fn from_raw_parts(
        row_ptrs: Handle<R::Server>,
        col_indices: Handle<R::Server>,
        values: Handle<R::Server>,
        meta: CsrMetadata,
    ) -> Self;
    
    /// Transpose to CSC (structural transpose, no value changes)
    pub fn transpose_to_csc(&self, client: &ComputeClient<...>) -> CscStorage<R>;
    
    /// Compute row statistics for algorithm selection
    pub fn compute_row_stats(&mut self, client: &ComputeClient<...>);
}
2.3 CSC (Compressed Sparse Column)
Column-major variant of CSR. Efficient for column-wise operations, SpMM with dense left-hand side (transposed problems).
rustpub struct CscStorage<R: Runtime> {
    /// Column pointers buffer [K+1 elements, u32]
    pub col_ptrs: Handle<R::Server>,
    
    /// Row indices buffer [NNZ elements, u32]
    pub row_indices: Handle<R::Server>,
    
    /// Values buffer [NNZ elements, dtype]
    pub values: Handle<R::Server>,
    
    pub meta: CscMetadata,
}

// Structurally identical to CSR, just transposed interpretation
// CSC of A == CSR of A^T
2.4 N:M Structured Sparsity — Phase 2
Hardware-accelerated on NVIDIA Ampere+ via Sparse Tensor Cores. Exactly N non-zeros in every M consecutive elements.
Memory Layout
2:4 Sparsity (50% structured):
Every group of 4 consecutive elements has exactly 2 non-zeros.

Dense row:    [a, b, c, d, e, f, g, h, i, j, k, l]
              └──group1──┘ └──group2──┘ └──group3──┘

2:4 sparse:   [a, 0, c, 0, e, f, 0, 0, i, 0, 0, l]  
               ✓     ✓     ✓  ✓        ✓        ✓

Storage:
  values:  [a, c, e, f, i, l]  — Only non-zeros, packed
  indices: [0b0101, 0b0011, 0b1001]  — 4-bit mask per group
           (positions 0,2) (positions 0,1) (positions 0,3)
Type Definition
rustpub struct NMStorage<R: Runtime, const N: usize, const M: usize> {
    /// Compressed values [rows × (cols/M) × N elements]
    pub values: Handle<R::Server>,
    
    /// Index metadata [(rows × cols/M) elements, each log2(C(M,N)) bits]
    /// For 2:4: 4 bits per group (encoding which 2 of 4 positions)
    pub indices: Handle<R::Server>,
    
    pub meta: NMMetadata<N, M>,
}

#[derive(Clone, Debug)]
pub struct NMMetadata<const N: usize, const M: usize> {
    pub rows: usize,
    pub cols: usize,  // Must be divisible by M
    pub dtype: DType,
}

impl<R: Runtime, const N: usize, const M: usize> NMStorage<R, N, M> {
    /// Verify N:M constraint at compile time
    const _CHECK: () = assert!(N <= M && M <= 16, "Invalid N:M parameters");
}

// Common instantiations
pub type Sparse2x4<R> = NMStorage<R, 2, 4>;  // 50% sparsity, Ampere tensor cores
pub type Sparse1x4<R> = NMStorage<R, 1, 4>;  // 75% sparsity
pub type Sparse4x8<R> = NMStorage<R, 4, 8>;  // 50% sparsity, larger blocks
Hardware Mapping (CUDA)
Sparse Tensor Core Operation (Ampere+):
  D = A_sparse × B_dense + C

  A_sparse: 2:4 structured, fp16/bf16/tf32
  B_dense:  Dense matrix
  Hardware instruction: mma.sp (sparse matrix multiply-accumulate)

Memory Requirements:
  - Values aligned to 16 bytes
  - Indices packed as 2-bit selectors (for 2:4)
  - Tile sizes: 16×16×16 or 32×16×8 depending on precision
Conversion from Dense
rustimpl<R: Runtime> NMStorage<R, 2, 4> {
    /// Convert dense to 2:4 sparse via magnitude pruning
    /// Keeps top-2 by absolute value in each group of 4
    pub fn from_dense_magnitude(
        dense: &Tensor<R>,
        client: &ComputeClient<...>,
    ) -> Self;
    
    /// Convert dense to 2:4 with custom selection function
    /// selection_fn: for each group of 4, return indices of 2 to keep
    pub fn from_dense_custom<F>(
        dense: &Tensor<R>,
        selection_fn: F,
        client: &ComputeClient<...>,
    ) -> Self
    where
        F: Fn(&[f32; 4]) -> [usize; 2];
    
    /// Convert from unstructured sparse (CSR) to 2:4
    /// May need to add/remove elements to satisfy constraint
    pub fn from_csr(
        csr: &CsrStorage<R>,
        client: &ComputeClient<...>,
    ) -> Self;
}
2.5 BSR/BCSC (Block Sparse) — Phase 3
Blocked formats for better memory coalescing and potential tensor core utilization.
rustpub struct BsrStorage<R: Runtime> {
    /// Block row pointers [(M/br)+1 elements, u32]
    pub block_row_ptrs: Handle<R::Server>,
    
    /// Block column indices [num_blocks elements, u32]
    pub block_col_indices: Handle<R::Server>,
    
    /// Block values [num_blocks × br × bc elements, dtype]
    /// Stored in row-major order within each block
    pub block_values: Handle<R::Server>,
    
    pub meta: BsrMetadata,
}

#[derive(Clone, Debug)]
pub struct BsrMetadata {
    pub rows: usize,           // M (must be divisible by block_rows)
    pub cols: usize,           // K (must be divisible by block_cols)
    pub block_rows: usize,     // br (typically 16, 32, 64)
    pub block_cols: usize,     // bc (typically 16, 32, 64)
    pub num_blocks: usize,     // Number of non-zero blocks
    pub dtype: DType,
}

impl<R: Runtime> BsrStorage<R> {
    /// Optimal block sizes for tensor core utilization
    pub const TENSOR_CORE_BLOCK_SIZES: &'static [(usize, usize)] = &[
        (16, 16),  // fp16/bf16 tensor cores
        (32, 32),  // Larger blocks, better for high sparsity
        (8, 8),    // Finer granularity
    ];
}
2.6 COO (Coordinate Format) — Utility
Flexible format for construction and format conversion. Not optimized for computation.
rustpub struct CooStorage<R: Runtime> {
    /// Row indices [NNZ elements, u32]
    pub row_indices: Handle<R::Server>,
    
    /// Column indices [NNZ elements, u32]  
    pub col_indices: Handle<R::Server>,
    
    /// Values [NNZ elements, dtype]
    pub values: Handle<R::Server>,
    
    /// Whether sorted by (row, col)
    pub sorted: bool,
    
    pub meta: CooMetadata,
}

impl<R: Runtime> CooStorage<R> {
    /// Sort by row then column (required for CSR conversion)
    pub fn sort(&mut self, client: &ComputeClient<...>);
    
    /// Remove duplicate indices (sum values)
    pub fn coalesce(&mut self, client: &ComputeClient<...>);
    
    /// Scatter-add values to existing COO (for gradient accumulation)
    pub fn scatter_add(&mut self, other: &CooStorage<R>, client: &ComputeClient<...>);
}
2.7 Format Conversion Graph
                    ┌─────────┐
                    │   COO   │ ◄── Construction entry point
                    └────┬────┘
                         │ sort + compress
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       ┌─────────┐  ┌─────────┐  ┌─────────┐
       │   CSR   │◄─┤transpose├─►│   CSC   │
       └────┬────┘  └─────────┘  └────┬────┘
            │                         │
            ▼                         ▼
       ┌─────────┐              ┌─────────┐
       │   BSR   │              │  BCSC   │
       └────┬────┘              └─────────┘
            │
            ▼
       ┌─────────┐
       │   N:M   │ ◄── Requires resampling (lossy)
       └─────────┘

Conversion Costs:
  COO → CSR:  O(NNZ) sort + O(M) prefix sum
  CSR ↔ CSC:  O(NNZ) scatter
  CSR → BSR:  O(NNZ) block assignment + possible padding
  CSR → N:M:  O(elements) with pruning/padding (lossy!)
  N:M → CSR:  O(elements) unpack (lossless)
rust/// Format conversion trait
pub trait ConvertFormat<Target: SparseStorage<R>, R: Runtime>: SparseStorage<R> {
    /// Whether conversion is lossless
    const LOSSLESS: bool;
    
    /// Convert to target format
    fn convert(&self, client: &ComputeClient<...>) -> Target;
}

// CSR → CSC: lossless
impl<R: Runtime> ConvertFormat<CscStorage<R>, R> for CsrStorage<R> {
    const LOSSLESS: bool = true;
    fn convert(&self, client: &ComputeClient<...>) -> CscStorage<R> { ... }
}

// CSR → N:M: lossy (must prune or pad)
impl<R: Runtime> ConvertFormat<NMStorage<R, 2, 4>, R> for CsrStorage<R> {
    const LOSSLESS: bool = false;
    fn convert(&self, client: &ComputeClient<...>) -> NMStorage<R, 2, 4> { ... }
}

3. Memory Architecture
3.1 GPU Buffer Organization
Buffer Alignment Requirements:
┌─────────────────────────────────────────────────────────────────┐
│ Format    │ Buffer          │ Element │ Alignment │ Notes       │
├───────────┼─────────────────┼─────────┼───────────┼─────────────┤
│ CSR       │ row_ptrs        │ u32     │ 4 bytes   │             │
│           │ col_indices     │ u32     │ 4 bytes   │             │
│           │ values          │ dtype   │ 16 bytes  │ Vectorized  │
├───────────┼─────────────────┼─────────┼───────────┼─────────────┤
│ N:M (2:4) │ values          │ dtype   │ 16 bytes  │ TC aligned  │
│           │ indices         │ u16     │ 16 bytes  │ Packed 4-bit│
├───────────┼─────────────────┼─────────┼───────────┼─────────────┤
│ BSR       │ block_row_ptrs  │ u32     │ 4 bytes   │             │
│           │ block_col_indices│ u32    │ 4 bytes   │             │
│           │ block_values    │ dtype   │ 128 bytes │ TC tiles    │
└─────────────────────────────────────────────────────────────────┘
3.2 Memory Pool Integration
rust/// Sparse-aware memory pool extension
pub trait SparseMemoryPool<R: Runtime> {
    /// Allocate buffers for sparse storage
    /// Returns handles with appropriate alignment
    fn alloc_sparse<S: SparseStorage<R>>(
        &self,
        meta: &S::Metadata,
    ) -> SparseBufferSet<R, S>;
    
    /// Reallocate for pattern change (dynamic sparsity)
    fn realloc_sparse<S: SparseStorage<R>>(
        &self,
        existing: SparseBufferSet<R, S>,
        new_meta: &S::Metadata,
    ) -> SparseBufferSet<R, S>;
    
    /// Memory pressure callback for sparse eviction policy
    fn register_sparse_eviction_policy(&self, policy: SparseEvictionPolicy);
}

/// Grouped buffer allocation for sparse formats
pub struct SparseBufferSet<R: Runtime, S: SparseStorage<R>> {
    handles: [Handle<R::Server>; S::NUM_BUFFERS],
    layout: BufferLayout,
}

#[derive(Clone, Debug)]
pub struct BufferLayout {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub alignments: Vec<usize>,
    pub total_bytes: usize,
}
3.3 Metadata Caching Strategy
rust/// CPU-side metadata cache to avoid GPU syncs
pub struct SparseMetadataCache {
    /// LRU cache: tensor_id → metadata
    cache: LruCache<TensorId, Arc<dyn SparseMetadata>>,
    
    /// Dirty flags for metadata that needs GPU sync
    dirty: HashSet<TensorId>,
}

impl SparseMetadataCache {
    /// Get metadata without GPU sync (may be stale for dynamic patterns)
    pub fn get(&self, id: TensorId) -> Option<Arc<dyn SparseMetadata>>;
    
    /// Mark metadata as potentially stale (pattern changed)
    pub fn invalidate(&mut self, id: TensorId);
    
    /// Force GPU sync and update cache
    pub async fn sync(&mut self, id: TensorId, client: &ComputeClient<...>);
}
3.4 Tensor Core Alignment
For N:M and BSR formats targeting tensor cores:
rust/// Tensor core tile configurations
#[derive(Clone, Copy, Debug)]
pub struct TensorCoreTile {
    pub m: usize,  // Tile rows
    pub n: usize,  // Tile cols
    pub k: usize,  // Reduction dimension
}

impl TensorCoreTile {
    /// Ampere fp16 sparse tensor core tile
    pub const AMPERE_FP16_SPARSE: Self = Self { m: 16, n: 16, k: 16 };
    
    /// Ampere bf16 sparse tensor core tile  
    pub const AMPERE_BF16_SPARSE: Self = Self { m: 16, n: 16, k: 16 };
    
    /// Ampere tf32 sparse tensor core tile
    pub const AMPERE_TF32_SPARSE: Self = Self { m: 16, n: 16, k: 8 };
}

/// Pad dimensions to tensor core alignment
pub fn pad_to_tile(dim: usize, tile_dim: usize) -> usize {
    (dim + tile_dim - 1) / tile_dim * tile_dim
}

4. Operation Design
4.1 SparseOperation Trait
Integration with CubeCL's operation system for JIT compilation and fusion.
rust/// Sparse operation trait for CubeCL integration
pub trait SparseOperation<R: Runtime>: Send + Sync + 'static {
    /// Input tensor descriptors (sparse and dense)
    type Inputs: SparseInputs;
    
    /// Output tensor descriptors
    type Outputs: SparseOutputs;
    
    /// Operation-specific configuration
    type Config: OperationConfig;
    
    /// Memory access pattern for fusion decisions
    fn memory_access_pattern(&self) -> SparseAccessPattern;
    
    /// Whether this operation can fuse with following element-wise ops
    fn supports_epilogue_fusion(&self) -> bool;
    
    /// Execute the operation
    fn execute(
        &self,
        inputs: Self::Inputs,
        config: Self::Config,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Self::Outputs;
}

/// Memory access pattern descriptor for fusion analysis
#[derive(Clone, Debug)]
pub struct SparseAccessPattern {
    /// Input access patterns
    pub inputs: Vec<AccessPattern>,
    /// Output access pattern
    pub output: AccessPattern,
    /// Whether output sparsity pattern matches an input
    pub output_pattern_source: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum AccessPattern {
    /// Dense sequential access
    DenseSequential { dims: Vec<usize> },
    /// Sparse indexed access
    SparseIndexed { format: SparseFormatId },
    /// Block access (for BSR)
    BlockAccess { block_shape: Vec<usize> },
    /// Structured pattern (for N:M)
    StructuredSparse { n: usize, m: usize },
}
4.2 SpMM (Sparse × Dense Matrix Multiplication)
The critical operation. Multiple algorithm implementations for different sparsity patterns.
Operation Definition
SpMM: A_sparse(M×K) × B_dense(K×N) = C_dense(M×N)

Variants:
  - SpMM_CSR:  CSR × Dense → Dense
  - SpMM_CSC:  Dense × CSC → Dense  (A^T × B via CSC of A)
  - SpMM_NM:   N:M × Dense → Dense  (tensor core accelerated)
  - SpMM_BSR:  BSR × Dense → Dense  (blocked computation)
CSR SpMM Algorithms
rust/// CSR SpMM algorithm variants
#[derive(Clone, Copy, Debug)]
pub enum CsrSpmmAlgorithm {
    /// One thread per row, each thread handles full dot product
    /// Best for: short rows (< 32 elements), uniform row lengths
    RowSplit,
    
    /// One warp per row, warp-level reduction
    /// Best for: medium rows (32-256 elements)
    WarpPerRow,
    
    /// Multiple warps per row with merge
    /// Best for: long rows (> 256 elements), power-law distributions
    MergeBasedRowSplit,
    
    /// Merge-based algorithm (like merge path SpMM)
    /// Best for: highly irregular row lengths
    MergePath,
    
    /// Vectorized row processing
    /// Best for: rows divisible by vector width
    Vectorized { width: usize },
}

/// CSR SpMM operation
pub struct CsrSpmm<R: Runtime> {
    pub algorithm: CsrSpmmAlgorithm,
    pub output_dtype: DType,
}

impl<R: Runtime> SparseOperation<R> for CsrSpmm<R> {
    type Inputs = (CsrStorage<R>, Tensor<R>);  // (A_sparse, B_dense)
    type Outputs = Tensor<R>;                   // C_dense
    type Config = CsrSpmmConfig;
    
    fn memory_access_pattern(&self) -> SparseAccessPattern {
        SparseAccessPattern {
            inputs: vec![
                AccessPattern::SparseIndexed { format: SparseFormatId::Csr },
                AccessPattern::DenseSequential { dims: vec![/* K, N */] },
            ],
            output: AccessPattern::DenseSequential { dims: vec![/* M, N */] },
            output_pattern_source: None,  // Output is dense
        }
    }
    
    fn supports_epilogue_fusion(&self) -> bool {
        true  // Can fuse element-wise ops on output
    }
    
    fn execute(...) -> Tensor<R> { ... }
}

#[derive(Clone, Debug)]
pub struct CsrSpmmConfig {
    /// Thread block dimensions
    pub block_dim: (usize, usize, usize),
    /// Tile sizes for shared memory staging
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    /// Whether to use shared memory for B matrix
    pub stage_b: bool,
}
CSR SpMM Kernel Structure (Row-Split)
rust/// Row-split CSR SpMM kernel for CubeCL
#[cube(launch)]
pub fn csr_spmm_row_split<F: Float>(
    // CSR inputs
    row_ptrs: &Array<u32>,
    col_indices: &Array<u32>,
    values: &Array<F>,
    // Dense input B (K × N, row-major)
    b_matrix: &Tensor<F>,
    // Output C (M × N, row-major)
    c_matrix: &mut Tensor<F>,
    // Dimensions
    #[comptime] m: u32,  // rows of A
    #[comptime] n: u32,  // cols of B (and C)
    #[comptime] k: u32,  // cols of A, rows of B
) {
    let row = ABSOLUTE_POS_X as u32;
    
    if row >= m {
        return;
    }
    
    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1];
    
    // For each output column
    for col in 0..n {
        let mut sum = F::new(0.0);
        
        // Dot product: sparse_row(A) · column(B)
        for idx in row_start..row_end {
            let k_idx = col_indices[idx];
            let a_val = values[idx];
            let b_val = b_matrix[k_idx * n + col];  // B[k_idx, col]
            sum += a_val * b_val;
        }
        
        c_matrix[row * n + col] = sum;
    }
}
CSR SpMM Kernel Structure (Warp-Per-Row)
rust/// Warp-per-row CSR SpMM with vectorized B access
#[cube(launch)]
pub fn csr_spmm_warp_per_row<F: Float>(
    row_ptrs: &Array<u32>,
    col_indices: &Array<u32>,
    values: &Array<F>,
    b_matrix: &Tensor<F>,
    c_matrix: &mut Tensor<F>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] vec_width: u32,  // Vectorization width for B access
) {
    let warp_id = ABSOLUTE_POS_X / WARP_SIZE;
    let lane_id = ABSOLUTE_POS_X % WARP_SIZE;
    let row = warp_id as u32;
    
    if row >= m {
        return;
    }
    
    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1];
    let row_len = row_end - row_start;
    
    // Process output columns in tiles
    for col_tile in range_step(0, n, vec_width) {
        let mut partial = Array::<F>::new(vec_width);
        for i in 0..vec_width {
            partial[i] = F::new(0.0);
        }
        
        // Each lane processes subset of sparse elements
        for idx in range_step(lane_id, row_len, WARP_SIZE) {
            let actual_idx = row_start + idx;
            let k_idx = col_indices[actual_idx];
            let a_val = values[actual_idx];
            
            // Vectorized B access
            for v in 0..vec_width {
                let b_val = b_matrix[k_idx * n + col_tile + v];
                partial[v] += a_val * b_val;
            }
        }
        
        // Warp-level reduction
        for v in 0..vec_width {
            let sum = warp_reduce_sum(partial[v]);
            if lane_id == 0 {
                c_matrix[row * n + col_tile + v] = sum;
            }
        }
    }
}
N:M SpMM (Tensor Core Path)
rust/// 2:4 Sparse Tensor Core SpMM
pub struct NMSpmm<R: Runtime, const N: usize, const M: usize> {
    _phantom: PhantomData<R>,
}

impl<R: Runtime> SparseOperation<R> for NMSpmm<R, 2, 4> {
    type Inputs = (NMStorage<R, 2, 4>, Tensor<R>);
    type Outputs = Tensor<R>;
    type Config = NMSpmmConfig;
    
    fn memory_access_pattern(&self) -> SparseAccessPattern {
        SparseAccessPattern {
            inputs: vec![
                AccessPattern::StructuredSparse { n: 2, m: 4 },
                AccessPattern::DenseSequential { dims: vec![/* K, N */] },
            ],
            output: AccessPattern::DenseSequential { dims: vec![/* M, N */] },
            output_pattern_source: None,
        }
    }
    
    fn supports_epilogue_fusion(&self) -> bool {
        true  // Tensor core ops support epilogue fusion
    }
    
    fn execute(
        &self,
        (a_sparse, b_dense): Self::Inputs,
        config: Self::Config,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Tensor<R> {
        // Dispatch to CUDA sparse tensor core intrinsics (mma.sp)
        // or fall back to emulated path on non-Ampere hardware
        if client.device_features().sparse_tensor_cores() {
            self.execute_tensor_core(a_sparse, b_dense, config, client)
        } else {
            self.execute_emulated(a_sparse, b_dense, config, client)
        }
    }
}

/// Tensor core tile configuration for 2:4 sparse
#[derive(Clone, Debug)]
pub struct NMSpmmConfig {
    /// Tile dimensions matching tensor core shape
    pub tile_m: usize,  // 16 for fp16
    pub tile_n: usize,  // 16 for fp16
    pub tile_k: usize,  // 32 for fp16 2:4 (16 sparse × 2)
    
    /// Number of warps per thread block
    pub warps_m: usize,
    pub warps_n: usize,
    
    /// Pipeline stages for latency hiding
    pub pipeline_stages: usize,
}
4.3 Element-wise Sparse Operations
Operations that preserve sparsity pattern.
rust/// Element-wise operation on sparse tensors
/// Both inputs must have identical sparsity pattern
pub struct SparseElementwise<R: Runtime, Op: ElementwiseOp> {
    pub op: Op,
    _phantom: PhantomData<R>,
}

pub trait ElementwiseOp: Clone + Send + Sync + 'static {
    fn apply<F: Float>(a: F, b: F) -> F;
    fn name() -> &'static str;
}

// Standard ops
pub struct SparseAdd;
pub struct SparseSub;
pub struct SparseMul;  // Hadamard product
pub struct SparseDiv;

impl<R: Runtime, Op: ElementwiseOp> SparseOperation<R> for SparseElementwise<R, Op> {
    type Inputs = (CsrStorage<R>, CsrStorage<R>);
    type Outputs = CsrStorage<R>;
    type Config = ();
    
    fn memory_access_pattern(&self) -> SparseAccessPattern {
        SparseAccessPattern {
            inputs: vec![
                AccessPattern::SparseIndexed { format: SparseFormatId::Csr },
                AccessPattern::SparseIndexed { format: SparseFormatId::Csr },
            ],
            output: AccessPattern::SparseIndexed { format: SparseFormatId::Csr },
            output_pattern_source: Some(0),  // Same pattern as first input
        }
    }
    
    fn supports_epilogue_fusion(&self) -> bool {
        true  // Can chain element-wise ops
    }
    
    fn execute(...) -> CsrStorage<R> { ... }
}

/// Apply dense element-wise op to sparse output (fused epilogue)
/// C_sparse = Op(SpMM(A_sparse, B_dense), bias)
pub struct SparseWithEpilogue<R: Runtime, MainOp, EpilogueOp> {
    pub main_op: MainOp,
    pub epilogue: EpilogueOp,
}
4.4 Reductions & Transpose
rust/// Sparse reduction operations
pub struct SparseReduction<R: Runtime> {
    pub dim: usize,
    pub reduction: ReductionType,
}

#[derive(Clone, Copy, Debug)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    L1Norm,
    L2Norm,
}

impl<R: Runtime> SparseOperation<R> for SparseReduction<R> {
    type Inputs = CsrStorage<R>;
    type Outputs = SparseOrDenseOutput<R>;  // May be dense depending on dim
    type Config = ();
    
    fn execute(&self, input: CsrStorage<R>, ...) -> SparseOrDenseOutput<R> {
        match self.dim {
            // Reduce along rows → dense output (one value per row)
            0 => SparseOrDenseOutput::Dense(self.reduce_rows(input)),
            // Reduce along cols → sparse output (same pattern as input rows)  
            1 => SparseOrDenseOutput::Sparse(self.reduce_cols(input)),
            _ => panic!("Invalid dimension"),
        }
    }
}

/// Sparse transpose (structural, no value changes)
pub struct SparseTranspose<R: Runtime>;

impl<R: Runtime> SparseTranspose<R> {
    /// CSR ↔ CSC transpose (efficient)
    pub fn csr_to_csc(csr: &CsrStorage<R>, client: &ComputeClient<...>) -> CscStorage<R>;
    
    /// CSR → CSR transpose (requires rebuild)
    pub fn csr_to_csr_transposed(csr: &CsrStorage<R>, client: &ComputeClient<...>) -> CsrStorage<R>;
}
4.5 Format Conversion Kernels
rust/// COO to CSR conversion kernel
#[cube(launch)]
pub fn coo_to_csr<F: Float>(
    // COO inputs (assumed sorted by row)
    coo_row_indices: &Array<u32>,
    coo_col_indices: &Array<u32>,
    coo_values: &Array<F>,
    // CSR outputs
    csr_row_ptrs: &mut Array<u32>,
    csr_col_indices: &mut Array<u32>,
    csr_values: &mut Array<F>,
    // Dimensions
    #[comptime] num_rows: u32,
    #[comptime] nnz: u32,
) {
    // Phase 1: Count elements per row (histogram)
    // Phase 2: Prefix sum for row_ptrs
    // Phase 3: Scatter col_indices and values
    // (Three separate kernel launches in practice)
}

/// Dense to CSR sparsification kernel
#[cube(launch)]
pub fn dense_to_csr<F: Float>(
    dense: &Tensor<F>,
    threshold: F,
    // Outputs (pre-allocated based on nnz count)
    row_ptrs: &mut Array<u32>,
    col_indices: &mut Array<u32>,
    values: &mut Array<F>,
    #[comptime] m: u32,
    #[comptime] k: u32,
) {
    // Phase 1: Count nnz per row
    // Phase 2: Prefix sum
    // Phase 3: Compact non-zeros
}

/// Dense to 2:4 structured sparse
#[cube(launch)]
pub fn dense_to_nm_2x4<F: Float>(
    dense: &Tensor<F>,
    // Outputs
    values: &mut Array<F>,      // Compressed values
    indices: &mut Array<u16>,   // 4-bit masks packed into u16
    #[comptime] m: u32,
    #[comptime] k: u32,  // Must be divisible by 4
) {
    let row = ABSOLUTE_POS_X;
    let group = ABSOLUTE_POS_Y;  // Which group of 4 in this row
    
    if row >= m || group * 4 >= k {
        return;
    }
    
    // Load group of 4 elements
    let base_idx = row * k + group * 4;
    let mut vals = [F::new(0.0); 4];
    for i in 0..4 {
        vals[i] = dense[base_idx + i];
    }
    
    // Find top-2 by absolute magnitude
    let (idx0, idx1) = find_top2_magnitude(&vals);
    
    // Store compressed values
    let out_base = row * (k / 4) * 2 + group * 2;
    values[out_base] = vals[idx0];
    values[out_base + 1] = vals[idx1];
    
    // Encode indices as 4-bit mask
    let mask = (1u16 << idx0) | (1u16 << idx1);
    indices[row * (k / 4) + group] = mask;
}

5. Algorithm Selection System
5.1 Selection Criteria
rust/// Algorithm selector based on sparsity characteristics
pub struct AlgorithmSelector {
    /// Cached profiling results
    cache: AlgorithmCache,
    /// Whether to enable runtime autotuning
    autotune: bool,
}

impl AlgorithmSelector {
    /// Select CSR SpMM algorithm based on matrix characteristics
    pub fn select_csr_spmm(
        &self,
        meta: &CsrMetadata,
        n_cols_dense: usize,
    ) -> CsrSpmmAlgorithm {
        let avg_nnz = meta.nnz as f32 / meta.rows as f32;
        let row_stats = meta.row_stats.as_ref();
        
        // Decision tree based on empirical tuning
        match (avg_nnz, n_cols_dense) {
            // Very short rows: one thread per row
            (avg, _) if avg < 8.0 => CsrSpmmAlgorithm::RowSplit,
            
            // Medium rows with small output: warp per row
            (avg, n) if avg < 64.0 && n <= 128 => CsrSpmmAlgorithm::WarpPerRow,
            
            // Medium rows with large output: vectorized
            (avg, n) if avg < 64.0 && n > 128 => {
                CsrSpmmAlgorithm::Vectorized { width: 4 }
            }
            
            // Long rows or highly variable: merge-based
            (avg, _) if avg >= 64.0 => {
                if let Some(stats) = row_stats {
                    let cv = stats.std_nnz_per_row / stats.avg_nnz_per_row;
                    if cv > 1.0 {
                        // High variance: merge path
                        CsrSpmmAlgorithm::MergePath
                    } else {
                        CsrSpmmAlgorithm::MergeBasedRowSplit
                    }
                } else {
                    CsrSpmmAlgorithm::MergeBasedRowSplit
                }
            }
            
            _ => CsrSpmmAlgorithm::WarpPerRow,
        }
    }
    
    /// Select format for SpMM operation
    pub fn select_format_for_spmm(
        &self,
        sparsity: f32,
        has_tensor_cores: bool,
    ) -> SparseFormatId {
        match (sparsity, has_tensor_cores) {
            // 50% sparsity with tensor cores: use 2:4
            (s, true) if (0.45..0.55).contains(&s) => {
                SparseFormatId::NM { n: 2, m: 4 }
            }
            // High sparsity (> 90%): CSR is most memory efficient
            (s, _) if s > 0.9 => SparseFormatId::Csr,
            // Medium sparsity: BSR may help with memory coalescing
            (s, _) if s > 0.7 => SparseFormatId::Bsr { 
                block_rows: 16, 
                block_cols: 16 
            },
            // Lower sparsity: just use dense
            _ => SparseFormatId::Csr,  // Could return "use dense" signal
        }
    }
}
5.2 Format Compatibility Matrix
Operation Compatibility:
┌─────────────┬─────┬─────┬───────┬─────┬──────┐
│ Operation   │ CSR │ CSC │ N:M   │ BSR │ COO  │
├─────────────┼─────┼─────┼───────┼─────┼──────┤
│ SpMM (A×B)  │ ✓✓  │ ○   │ ✓✓✓   │ ✓   │ ✗    │
│ SpMM (Aᵀ×B) │ ○   │ ✓✓  │ ○     │ ○   │ ✗    │
│ SpMV        │ ✓✓  │ ✓   │ ✓     │ ✓   │ ✗    │
│ Elementwise │ ✓✓  │ ✓✓  │ ✓     │ ✓   │ ✓    │
│ Reduction   │ ✓✓  │ ✓   │ ✓     │ ✓   │ ✓    │
│ Transpose   │ ✓   │ ✓   │ ✗     │ ○   │ ✓✓   │
│ Construction│ ○   │ ○   │ ○     │ ○   │ ✓✓   │
└─────────────┴─────┴─────┴───────┴─────┴──────┘

Legend: ✓✓✓=optimal, ✓✓=good, ✓=supported, ○=via conversion, ✗=not supported
5.3 Runtime Profiling Hooks
rust/// Runtime profiling for algorithm tuning
pub struct SparseProfiler {
    /// Operation timing history
    timings: HashMap<OperationKey, Vec<Duration>>,
    /// Best algorithm per configuration
    best_algorithms: HashMap<ConfigKey, AlgorithmChoice>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct OperationKey {
    op_type: OperationType,
    format: SparseFormatId,
    algorithm: AlgorithmId,
    shape_bucket: ShapeBucket,  // Bucketed (M, K, N) for caching
    sparsity_bucket: u8,        // 0-100 quantized sparsity
}

impl SparseProfiler {
    /// Record timing for an operation
    pub fn record(&mut self, key: OperationKey, duration: Duration);
    
    /// Query best algorithm for configuration
    pub fn best_for(&self, op: OperationType, format: SparseFormatId, shape: &Shape) -> Option<AlgorithmId>;
    
    /// Export profiling data for offline analysis
    pub fn export(&self) -> ProfilingData;
    
    /// Import pre-tuned configurations
    pub fn import(&mut self, data: ProfilingData);
}

6. Autodiff Architecture
6.1 Gradient Formulations
Forward:  C = SpMM(A_sparse, B_dense)
          C[i,j] = Σ_k A[i,k] * B[k,j]

Backward: Given dL/dC (dense gradient)

  dL/dA[i,k] = Σ_j dL/dC[i,j] * B[k,j]ᵀ
             = (dL/dC × Bᵀ)[i,k]
             → Sparse output (same pattern as A)

  dL/dB[k,j] = Σ_i A[i,k]ᵀ * dL/dC[i,j]  
             = (Aᵀ × dL/dC)[k,j]
             → Dense output (SpMM with transposed A)
6.2 Backward Operations
rust/// Backward pass for CSR SpMM
pub struct CsrSpmmBackward<R: Runtime>;

impl<R: Runtime> CsrSpmmBackward<R> {
    /// Compute dL/dA (sparse gradient, same pattern as A)
    /// dL/dA = (dL/dC × Bᵀ) masked to pattern of A
    pub fn grad_sparse(
        a_meta: &CsrMetadata,           // Pattern of A
        a_col_indices: &Handle<...>,    // Column indices of A
        grad_output: &Tensor<R>,        // dL/dC (M × N)
        b_dense: &Tensor<R>,            // B (K × N)
        client: &ComputeClient<...>,
    ) -> CsrStorage<R> {
        // For each non-zero position (i, k) in A:
        // grad_a[i,k] = dot(grad_output[i,:], b[k,:])
        // This is row i of grad_output dotted with row k of B
        // 
        // Kernel: one thread per non-zero of A
        //   idx = thread_id
        //   (i, k) = position of A[idx]
        //   grad_a[idx] = dot(grad_output[i,:], b[k,:])
    }
    
    /// Compute dL/dB (dense gradient)
    /// dL/dB = Aᵀ × dL/dC
    pub fn grad_dense(
        a_sparse: &CsrStorage<R>,       // A (M × K) as CSR
        grad_output: &Tensor<R>,        // dL/dC (M × N)
        client: &ComputeClient<...>,
    ) -> Tensor<R> {
        // Convert A to CSC (= CSR of Aᵀ) and run SpMM
        let a_transposed = SparseTranspose::csr_to_csc(a_sparse, client);
        CscSpmm::execute(a_transposed, grad_output, client)
    }
}

/// Backward kernel for sparse gradient (dL/dA)
#[cube(launch)]
pub fn csr_spmm_backward_sparse<F: Float>(
    // Original A structure (pattern only)
    row_ptrs: &Array<u32>,
    col_indices: &Array<u32>,
    // Gradient output dL/dC (M × N)
    grad_output: &Tensor<F>,
    // Dense B (K × N)  
    b_dense: &Tensor<F>,
    // Output: gradient values for A
    grad_values: &mut Array<F>,
    // Dimensions
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] n: u32,
) {
    // One thread per non-zero
    let nnz_idx = ABSOLUTE_POS_X as u32;
    
    // Binary search to find row for this nnz
    let row = binary_search_row_ptrs(row_ptrs, nnz_idx, m);
    let col = col_indices[nnz_idx];  // This is the 'k' index
    
    // Compute dot product: grad_output[row, :] · b[col, :]
    let mut sum = F::new(0.0);
    for j in 0..n {
        let grad_c = grad_output[row * n + j];
        let b_val = b_dense[col * n + j];  // B[k, j]
        sum += grad_c * b_val;
    }
    
    grad_values[nnz_idx] = sum;
}
6.3 Mask Preservation
Critical for training: gradients must maintain the sparsity pattern.
rust/// Sparsity mask preservation modes
#[derive(Clone, Copy, Debug)]
pub enum MaskPreservation {
    /// Gradient has exactly same pattern as forward (standard)
    ExactPattern,
    
    /// Allow gradient to grow (for dynamic sparsity)
    AllowGrowth,
    
    /// Allow gradient to shrink (pruning based on gradient magnitude)
    AllowShrinkage,
}

/// Sparse parameter with mask tracking
pub struct SparseParameter<R: Runtime> {
    /// Current values in sparse format
    pub storage: CsrStorage<R>,
    
    /// Mask preservation mode
    pub mask_mode: MaskPreservation,
    
    /// Original dense shape (for re-densification if needed)
    pub dense_shape: Shape,
    
    /// Gradient accumulator (sparse or dense based on mask_mode)
    pub grad_accumulator: Option<GradAccumulator<R>>,
}

pub enum GradAccumulator<R: Runtime> {
    /// Sparse accumulator (same pattern as parameter)
    Sparse(CsrStorage<R>),
    /// Dense accumulator (for AllowGrowth mode)
    Dense(Tensor<R>),
    /// COO accumulator (for dynamic pattern changes)
    Coo(CooStorage<R>),
}

impl<R: Runtime> SparseParameter<R> {
    /// Accumulate gradient while preserving mask
    pub fn accumulate_grad(
        &mut self,
        grad: &CsrStorage<R>,
        client: &ComputeClient<...>,
    ) {
        match self.mask_mode {
            MaskPreservation::ExactPattern => {
                // Gradient must match pattern exactly
                assert_eq!(grad.meta.nnz, self.storage.meta.nnz);
                // Element-wise add to accumulator
            }
            MaskPreservation::AllowGrowth => {
                // Convert to COO, merge patterns, convert back
            }
            MaskPreservation::AllowShrinkage => {
                // Zero out small gradients, compact pattern
            }
        }
    }
    
    /// Apply accumulated gradient (optimizer step)
    pub fn apply_grad(&mut self, learning_rate: f32, client: &ComputeClient<...>);
    
    /// Re-sparsify after pattern change (for dynamic sparsity)
    pub fn resparsify(&mut self, threshold: f32, client: &ComputeClient<...>);
}
6.4 Autodiff Graph Nodes
rust/// Sparse operation node for autodiff graph
pub enum SparseAutoGradNode<R: Runtime> {
    /// SpMM forward
    SpmmForward {
        inputs: (NodeId, NodeId),        // (A_sparse, B_dense)
        output: NodeId,                   // C_dense
        saved_a_pattern: CsrMetadata,     // Pattern for backward
        saved_b: Option<NodeId>,          // Saved B if needed for backward
    },
    
    /// SpMM backward for sparse input
    SpmmBackwardSparse {
        grad_output: NodeId,
        saved_b: NodeId,
        pattern: CsrMetadata,
        output: NodeId,  // dL/dA (sparse)
    },
    
    /// SpMM backward for dense input
    SpmmBackwardDense {
        grad_output: NodeId,
        saved_a_transposed: NodeId,  // CSC of original A
        output: NodeId,  // dL/dB (dense)
    },
    
    /// Element-wise sparse operation
    ElementwiseForward {
        inputs: Vec<NodeId>,
        output: NodeId,
        op: ElementwiseOp,
    },
}

/// Sparse-aware autodiff tape
pub struct SparseAutodiffTape<R: Runtime> {
    /// Standard dense operations
    dense_tape: AutodiffTape<R>,
    
    /// Sparse operation nodes
    sparse_nodes: Vec<SparseAutoGradNode<R>>,
    
    /// Sparse tensor registry
    sparse_tensors: HashMap<NodeId, SparseTensorRef<R>>,
}

impl<R: Runtime> SparseAutodiffTape<R> {
    /// Record sparse operation
    pub fn record_sparse_op(&mut self, node: SparseAutoGradNode<R>);
    
    /// Execute backward pass
    pub fn backward(&mut self, loss: NodeId, client: &ComputeClient<...>);
    
    /// Get sparse gradient
    pub fn sparse_grad(&self, node: NodeId) -> Option<&CsrStorage<R>>;
}
6.5 Memory Strategy
rust/// Memory-efficient gradient checkpointing for sparse ops
pub struct SparseCheckpointing<R: Runtime> {
    /// Which forward tensors to save vs recompute
    strategy: CheckpointStrategy,
    
    /// Saved activations
    saved: HashMap<NodeId, SavedTensor<R>>,
}

#[derive(Clone, Debug)]
pub enum CheckpointStrategy {
    /// Save all forward tensors (fastest backward, most memory)
    SaveAll,
    
    /// Save only sparse patterns, recompute values
    SavePatternOnly,
    
    /// Checkpoint at regular intervals
    Periodic { interval: usize },
    
    /// Memory budget: save what fits, recompute rest
    MemoryBudget { max_bytes: usize },
}

enum SavedTensor<R: Runtime> {
    /// Full tensor saved
    Full(SparseTensorRef<R>),
    
    /// Only pattern saved (values will be recomputed)
    PatternOnly {
        meta: Box<dyn SparseMetadata>,
        indices: Handle<R::Server>,
    },
}

7. CubeCL Fusion Integration
7.1 Fusion Strategy
Sparse operations have limited fusion potential:

CAN Fuse:
┌─────────────────────────────────────────────────────────────┐
│ SpMM → Element-wise on output                               │
│   C_sparse = A × B                                          │
│   D = relu(C)         # Fuse as epilogue                    │
│   E = D + bias        # Chain element-wise                  │
├─────────────────────────────────────────────────────────────┤
│ Element-wise chain on same sparse pattern                   │
│   B = A * 2                                                 │
│   C = B + 1           # Fuse into single kernel             │
│   D = relu(C)                                               │
└─────────────────────────────────────────────────────────────┘

CANNOT Fuse:
┌─────────────────────────────────────────────────────────────┐
│ Different sparsity patterns                                 │
│   C = A + B   where A, B have different patterns            │
├─────────────────────────────────────────────────────────────┤
│ SpMM chain (reduction boundary)                             │
│   C = A × B                                                 │
│   E = C × D           # Must materialize C first            │
├─────────────────────────────────────────────────────────────┤
│ Format conversions                                          │
│   B = csr_to_csc(A)   # Structural transformation           │
└─────────────────────────────────────────────────────────────┘
7.2 SparseOperation Trait for Fusion
rust/// Extended sparse operation trait with fusion support
pub trait FusibleSparseOperation<R: Runtime>: SparseOperation<R> {
    /// Cube IR representation for fusion analysis
    fn to_cube_ir(&self) -> CubeIR;
    
    /// Memory access pattern for fusion decisions
    fn access_pattern(&self) -> SparseAccessPattern;
    
    /// Whether this op can be an epilogue to another op
    fn can_be_epilogue(&self) -> bool;
    
    /// Whether this op supports epilogue fusion
    fn accepts_epilogue(&self) -> bool;
    
    /// Fuse with epilogue operation
    fn fuse_epilogue<E: FusibleSparseOperation<R>>(
        self,
        epilogue: E,
    ) -> FusedSparseOp<R, Self, E>
    where
        Self: Sized;
}

/// Fusion analysis result
pub struct FusionAnalysis {
    /// Operations that can be fused together
    pub fusion_groups: Vec<FusionGroup>,
    /// Reasons fusion was rejected
    pub rejection_reasons: Vec<(OpId, OpId, FusionRejection)>,
}

#[derive(Clone, Debug)]
pub enum FusionRejection {
    /// Different sparsity patterns
    PatternMismatch,
    /// Memory access pattern prevents fusion
    AccessConflict,
    /// Operation doesn't support fusion
    UnsupportedOp,
    /// Would exceed register pressure
    RegisterPressure,
    /// Reduction boundary
    ReductionBoundary,
}
7.3 Fused Kernel Generation
rust/// Fused SpMM + element-wise epilogue
pub struct SpmmWithEpilogue<R: Runtime, E: ElementwiseOp> {
    pub spmm: CsrSpmm<R>,
    pub epilogue: E,
}

impl<R: Runtime, E: ElementwiseOp> SpmmWithEpilogue<R, E> {
    /// Generate fused kernel
    pub fn compile(&self, client: &ComputeClient<...>) -> CompiledKernel {
        // CubeCL JIT: inline epilogue into SpMM output write
        //
        // Instead of:
        //   c_matrix[row * n + col] = sum;  // SpMM output
        //   // separate kernel
        //   out[i] = epilogue(c[i]);
        //
        // Generate:
        //   c_matrix[row * n + col] = epilogue(sum);  // Fused
    }
}

/// Fused kernel for SpMM with ReLU epilogue
#[cube(launch)]
pub fn csr_spmm_relu_fused<F: Float>(
    row_ptrs: &Array<u32>,
    col_indices: &Array<u32>,
    values: &Array<F>,
    b_matrix: &Tensor<F>,
    c_matrix: &mut Tensor<F>,
    #[comptime] m: u32,
    #[comptime] n: u32,
) {
    let row = ABSOLUTE_POS_X as u32;
    let col = ABSOLUTE_POS_Y as u32;
    
    if row >= m || col >= n {
        return;
    }
    
    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1];
    
    let mut sum = F::new(0.0);
    for idx in row_start..row_end {
        let k = col_indices[idx];
        sum += values[idx] * b_matrix[k * n + col];
    }
    
    // Fused ReLU epilogue
    let output = if sum > F::new(0.0) { sum } else { F::new(0.0) };
    c_matrix[row * n + col] = output;
}
7.4 Comptime vs Runtime
rust/// Sparse-specific comptime parameters
pub struct SparseComptime {
    /// Format is known at compile time
    pub format: Option<SparseFormatId>,
    /// Sparsity level is known (for algorithm selection)
    pub sparsity_bucket: Option<SparsityBucket>,
    /// Block size (for BSR)
    pub block_size: Option<(usize, usize)>,
    /// N:M parameters
    pub nm_params: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug)]
pub enum SparsityBucket {
    Low,      // < 50%
    Medium,   // 50-90%
    High,     // 90-99%
    Extreme,  // > 99%
}

impl SparseComptime {
    /// Whether format specialization can be done at compile time
    pub fn can_specialize_format(&self) -> bool {
        self.format.is_some()
    }
    
    /// Generate specialized kernel for known format
    pub fn specialize<R: Runtime>(
        &self,
        op: &dyn SparseOperation<R>,
    ) -> Option<CompiledKernel> {
        // If format and sparsity are known at compile time,
        // generate fully specialized kernel with no runtime dispatch
    }
}

8. Burn Backend Implementation
8.1 SparseBackend Trait
rust/// Backend trait extension for sparse operations
pub trait SparseBackend: Backend {
    /// CSR storage type
    type CsrStorage: SparseStorage<Self::Runtime>;
    /// CSC storage type  
    type CscStorage: SparseStorage<Self::Runtime>;
    /// N:M storage type
    type NMStorage<const N: usize, const M: usize>: SparseStorage<Self::Runtime>;
    /// BSR storage type
    type BsrStorage: SparseStorage<Self::Runtime>;
    /// COO storage type
    type CooStorage: SparseStorage<Self::Runtime>;
    
    /// Default sparse format for this backend
    fn default_sparse_format() -> SparseFormatId;
    
    /// Whether backend supports N:M tensor cores
    fn supports_nm_tensor_cores() -> bool;
}

/// CubeCL backend implementation
impl<R: Runtime, F: FloatElement, I: IntElement> SparseBackend for JitBackend<R, F, I> {
    type CsrStorage = CsrStorage<R>;
    type CscStorage = CscStorage<R>;
    type NMStorage<const N: usize, const M: usize> = NMStorage<R, N, M>;
    type BsrStorage = BsrStorage<R>;
    type CooStorage = CooStorage<R>;
    
    fn default_sparse_format() -> SparseFormatId {
        SparseFormatId::Csr
    }
    
    fn supports_nm_tensor_cores() -> bool {
        // Check runtime capability
        R::device_features().sparse_tensor_cores()
    }
}
8.2 SparseTensor<B> Realization
rust/// Sparse tensor with generic backend
pub struct SparseTensor<B: SparseBackend> {
    /// Storage discriminant based on format
    storage: SparseStorageEnum<B>,
    /// Tensor metadata
    meta: TensorMetadata,
}

enum SparseStorageEnum<B: SparseBackend> {
    Csr(B::CsrStorage),
    Csc(B::CscStorage),
    NM2x4(B::NMStorage<2, 4>),
    Bsr(B::BsrStorage),
    Coo(B::CooStorage),
}

impl<B: SparseBackend> SparseTensor<B> {
    /// Create from dense tensor via sparsification
    pub fn from_dense(dense: Tensor<B>, threshold: f32) -> Self {
        let csr = B::CsrStorage::from_dense(&dense, threshold, B::client());
        Self {
            storage: SparseStorageEnum::Csr(csr),
            meta: TensorMetadata::from_shape(dense.shape()),
        }
    }
    
    /// Convert to different format
    pub fn to_format(self, format: SparseFormatId) -> Self {
        // Dispatch to format conversion
        match (self.storage, format) {
            (SparseStorageEnum::Csr(csr), SparseFormatId::Csc) => {
                let csc = csr.transpose_to_csc(B::client());
                Self { storage: SparseStorageEnum::Csc(csc), ..self }
            }
            // ... other conversions
        }
    }
    
    /// Convert to dense
    pub fn to_dense(self) -> Tensor<B> {
        match self.storage {
            SparseStorageEnum::Csr(csr) => csr.to_dense(B::client()),
            // ... other formats
        }
    }
    
    /// Get format
    pub fn format(&self) -> SparseFormatId {
        match &self.storage {
            SparseStorageEnum::Csr(_) => SparseFormatId::Csr,
            SparseStorageEnum::Csc(_) => SparseFormatId::Csc,
            SparseStorageEnum::NM2x4(_) => SparseFormatId::NM { n: 2, m: 4 },
            SparseStorageEnum::Bsr(_) => SparseFormatId::Bsr { 
                block_rows: 16, 
                block_cols: 16 
            },
            SparseStorageEnum::Coo(_) => SparseFormatId::Coo,
        }
    }
    
    /// Sparsity ratio
    pub fn sparsity(&self) -> f32 {
        match &self.storage {
            SparseStorageEnum::Csr(s) => s.sparsity(),
            // ... other formats
        }
    }
}
8.3 Operation Dispatch
rust/// Sparse matrix multiplication
pub fn sparse_matmul<B: SparseBackend>(
    a: SparseTensor<B>,
    b: Tensor<B>,
) -> Tensor<B> {
    match a.storage {
        SparseStorageEnum::Csr(csr) => {
            let algorithm = B::algorithm_selector().select_csr_spmm(
                csr.metadata(),
                b.shape().dims[1],
            );
            csr_spmm(csr, b, algorithm, B::client())
        }
        SparseStorageEnum::NM2x4(nm) => {
            if B::supports_nm_tensor_cores() {
                nm_spmm_tensor_core(nm, b, B::client())
            } else {
                nm_spmm_emulated(nm, b, B::client())
            }
        }
        // ... other formats
    }
}

/// Sparse element-wise operations
pub fn sparse_add<B: SparseBackend>(
    a: SparseTensor<B>,
    b: SparseTensor<B>,
) -> SparseTensor<B> {
    // Patterns must match for efficient add
    // If patterns differ, need to union patterns (more expensive)
    assert_eq!(a.format(), b.format(), "Format mismatch");
    
    match (a.storage, b.storage) {
        (SparseStorageEnum::Csr(a_csr), SparseStorageEnum::Csr(b_csr)) => {
            // Check pattern match
            if a_csr.meta.nnz == b_csr.meta.nnz {
                // Assume same pattern, element-wise add
                let result = sparse_elementwise_csr(a_csr, b_csr, SparseAdd, B::client());
                SparseTensor {
                    storage: SparseStorageEnum::Csr(result),
                    meta: a.meta,
                }
            } else {
                // Different patterns, need pattern union
                sparse_add_different_patterns(a_csr, b_csr, B::client())
            }
        }
        // ... other formats
    }
}
8.4 Autodiff Backend
rust/// Autodiff wrapper for sparse backend
#[derive(Clone, Debug)]
pub struct SparseAutodiff<B: SparseBackend> {
    inner: B,
    tape: Arc<Mutex<SparseAutodiffTape<B>>>,
}

impl<B: SparseBackend> SparseBackend for SparseAutodiff<B> {
    type CsrStorage = CsrStorageAutodiff<B>;
    // ... other types with autodiff wrappers
}

/// CSR storage with autodiff tracking
pub struct CsrStorageAutodiff<B: SparseBackend> {
    inner: B::CsrStorage,
    node_id: NodeId,
    requires_grad: bool,
}

/// Autodiff-aware SpMM
pub fn sparse_matmul_autodiff<B: SparseBackend>(
    a: SparseTensor<SparseAutodiff<B>>,
    b: Tensor<SparseAutodiff<B>>,
) -> Tensor<SparseAutodiff<B>> {
    // Forward pass
    let output = sparse_matmul(a.inner(), b.inner());
    
    // Record in tape
    if a.requires_grad() || b.requires_grad() {
        let tape = a.tape();
        tape.lock().unwrap().record_sparse_op(SparseAutoGradNode::SpmmForward {
            inputs: (a.node_id(), b.node_id()),
            output: output.node_id(),
            saved_a_pattern: a.metadata().clone(),
            saved_b: if a.requires_grad() { Some(b.node_id()) } else { None },
        });
    }
    
    output
}

9. File Structure
9.1 Crate Organization
cubecl-sparse/                    # New crate in cubecl workspace
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API exports
│   │
│   ├── format/                   # Sparse format definitions
│   │   ├── mod.rs
│   │   ├── traits.rs             # SparseFormat, SparseStorage, SparseMetadata
│   │   ├── csr.rs                # CsrStorage, CsrMetadata, RowStatistics
│   │   ├── csc.rs                # CscStorage, CscMetadata
│   │   ├── coo.rs                # CooStorage, CooMetadata
│   │   ├── nm.rs                 # NMStorage<N,M>, NMMetadata
│   │   ├── bsr.rs                # BsrStorage, BsrMetadata
│   │   └── bcsc.rs               # BcscStorage
│   │
│   ├── convert/                  # Format conversions
│   │   ├── mod.rs
│   │   ├── dense_to_sparse.rs    # Dense → CSR/COO/N:M
│   │   ├── sparse_to_dense.rs    # CSR/COO/N:M → Dense
│   │   ├── coo_csr.rs            # COO ↔ CSR
│   │   ├── csr_csc.rs            # CSR ↔ CSC (transpose)
│   │   ├── csr_nm.rs             # CSR → N:M (lossy)
│   │   └── csr_bsr.rs            # CSR → BSR
│   │
│   ├── ops/                      # Sparse operations
│   │   ├── mod.rs
│   │   ├── traits.rs             # SparseOperation, FusibleSparseOperation
│   │   ├── spmm/                 # Sparse × Dense matmul
│   │   │   ├── mod.rs
│   │   │   ├── csr_spmm.rs       # CSR SpMM dispatch
│   │   │   ├── csr_row_split.rs  # Row-split kernel
│   │   │   ├── csr_warp_row.rs   # Warp-per-row kernel
│   │   │   ├── csr_merge.rs      # Merge-based kernel
│   │   │   ├── nm_spmm.rs        # N:M SpMM dispatch
│   │   │   ├── nm_tensor_core.rs # Tensor core kernel (Ampere+)
│   │   │   ├── nm_emulated.rs    # Fallback kernel
│   │   │   └── bsr_spmm.rs       # Block SpMM
│   │   │
│   │   ├── spmv.rs               # Sparse × Dense vector
│   │   ├── elementwise.rs        # Sparse element-wise ops
│   │   ├── reduction.rs          # Sparse reductions (sum, mean, norm)
│   │   └── transpose.rs          # Sparse transpose ops
│   │
│   ├── algorithm/                # Algorithm selection
│   │   ├── mod.rs
│   │   ├── selector.rs           # AlgorithmSelector
│   │   ├── profiler.rs           # SparseProfiler, runtime tuning
│   │   └── heuristics.rs         # Sparsity-based heuristics
│   │
│   ├── memory/                   # Memory management
│   │   ├── mod.rs
│   │   ├── pool.rs               # SparseMemoryPool trait
│   │   ├── layout.rs             # BufferLayout, alignment
│   │   └── cache.rs              # SparseMetadataCache
│   │
│   └── fusion/                   # CubeCL fusion integration
│       ├── mod.rs
│       ├── access_pattern.rs     # SparseAccessPattern
│       ├── analysis.rs           # FusionAnalysis, FusionRejection
│       └── epilogue.rs           # SpmmWithEpilogue
│
└── tests/
    ├── correctness/
    │   ├── spmm_csr.rs
    │   ├── spmm_nm.rs
    │   ├── conversions.rs
    │   └── elementwise.rs
    ├── performance/
    │   └── benchmarks.rs
    └── integration/
        └── full_pipeline.rs
9.2 Burn Integration Structure
burn-sparse/                      # Existing crate (to be modified)
├── Cargo.toml                    # Add cubecl-sparse dependency
├── src/
│   ├── lib.rs
│   │
│   ├── tensor/                   # SparseTensor abstraction
│   │   ├── mod.rs
│   │   ├── sparse.rs             # SparseTensor<B> (update existing)
│   │   ├── param.rs              # SparseParam<B> for training
│   │   └── storage.rs            # SparseStorageEnum dispatch
│   │
│   ├── backend/                  # Backend trait
│   │   ├── mod.rs
│   │   ├── traits.rs             # SparseBackend trait
│   │   └── jit.rs                # JitBackend SparseBackend impl
│   │
│   ├── ops/                      # High-level sparse ops
│   │   ├── mod.rs
│   │   ├── matmul.rs             # sparse_matmul()
│   │   ├── elementwise.rs        # sparse_add(), sparse_mul(), etc.
│   │   ├── activation.rs         # Sparse activations
│   │   └── conversion.rs         # to_sparse(), to_dense()
│   │
│   ├── autodiff/                 # Autodiff for sparse
│   │   ├── mod.rs
│   │   ├── tape.rs               # SparseAutodiffTape
│   │   ├── nodes.rs              # SparseAutoGradNode
│   │   ├── backward.rs           # Backward implementations
│   │   └── checkpoint.rs         # SparseCheckpointing
│   │
│   └── optim/                    # Sparse-aware optimizers
│       ├── mod.rs
│       ├── sgd.rs                # Sparse SGD
│       └── adam.rs               # Sparse Adam
│
└── tests/
    └── training/
        ├── mlp_sparse.rs
        └── transformer_sparse.rs
9.3 Module Dependencies
                    ┌─────────────────┐
                    │   burn-sparse   │
                    │  (user-facing)  │
                    └────────┬────────┘
                             │ depends on
                             ▼
                    ┌─────────────────┐
                    │  cubecl-sparse  │
                    │   (GPU ops)     │
                    └────────┬────────┘
                             │ depends on
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  cubecl  │  │cubecl-   │  │cubecl-   │
        │  (core)  │  │  linalg  │  │   nn     │
        └──────────┘  └──────────┘  └──────────┘

Internal cubecl-sparse dependencies:
┌─────────────────────────────────────────────────┐
│                                                 │
│  fusion ──────► ops ──────► format              │
│     │            │            │                 │
│     │            ▼            │                 │
│     │        algorithm        │                 │
│     │            │            │                 │
│     └──────►─────┴─────►──────┘                 │
│                  │                              │
│                  ▼                              │
│               memory                            │
│                  │                              │
│                  ▼                              │
│              convert                            │
│                                                 │
└─────────────────────────────────────────────────┘
9.4 Key File Contents
cubecl-sparse/src/lib.rs
rust//! GPU-accelerated sparse tensor operations for CubeCL
//!
//! This crate provides sparse matrix formats and operations
//! optimized for GPU execution via CubeCL.

pub mod format;
pub mod convert;
pub mod ops;
pub mod algorithm;
pub mod memory;
pub mod fusion;

// Re-exports for convenience
pub use format::{
    SparseFormat, SparseStorage, SparseMetadata, SparseFormatId,
    CsrStorage, CscStorage, CooStorage, NMStorage, BsrStorage,
};
pub use ops::{SparseOperation, FusibleSparseOperation};
pub use ops::spmm::{csr_spmm, nm_spmm, CsrSpmmAlgorithm};
pub use algorithm::AlgorithmSelector;
burn-sparse/src/lib.rs
rust//! Sparse tensor support for Burn
//!
//! Provides SparseTensor<B> abstraction with GPU acceleration
//! via cubecl-sparse backend.

pub mod tensor;
pub mod backend;
pub mod ops;
pub mod autodiff;
pub mod optim;

pub use tensor::{SparseTensor, SparseParam};
pub use backend::SparseBackend;
pub use ops::{sparse_matmul, sparse_add, to_sparse, to_dense};
9.5 Cargo.toml Dependencies
cubecl-sparse/Cargo.toml
toml[package]
name = "cubecl-sparse"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "GPU-accelerated sparse tensor operations for CubeCL"

[dependencies]
cubecl = { path = "../cubecl" }
cubecl-runtime = { path = "../cubecl-runtime" }
cubecl-linalg = { path = "../cubecl-linalg" }

[dev-dependencies]
cubecl = { path = "../cubecl", features = ["cuda"] }
criterion = "0.5"

[features]
default = []
cuda = ["cubecl/cuda"]
wgpu = ["cubecl/wgpu"]

[[bench]]
name = "spmm"
harness = false
burn-sparse/Cargo.toml (additions)
toml[dependencies]
cubecl-sparse = { path = "../cubecl-sparse", optional = true }

[features]
default = ["cubecl-sparse"]
gpu = ["cubecl-sparse"]

10. Implementation Roadmap
10.1 Phase 1: CSR Foundation
Goal: Working CSR SpMM with autodiff
Timeline: 4-6 weeks

Components:
├── Core Types
│   ├── CsrStorage<R> implementation
│   ├── CsrMetadata and RowStatistics
│   └── Basic memory allocation
│
├── SpMM Kernels
│   ├── RowSplit kernel (baseline)
│   ├── WarpPerRow kernel (main workhorse)
│   └── Algorithm selection (sparsity-based)
│
├── Supporting Ops
│   ├── Dense → CSR conversion
│   ├── CSR → Dense conversion
│   ├── Element-wise ops (same pattern)
│   └── CSR ↔ CSC transpose
│
├── Autodiff
│   ├── Backward for sparse gradient (dL/dA)
│   ├── Backward for dense gradient (dL/dB)
│   └── Mask preservation (ExactPattern mode)
│
└── Burn Integration
    ├── SparseBackend trait (CSR only)
    ├── SparseTensor<B> with CSR
    └── Basic API: sparse_matmul, sparse_add

Deliverables:
- Can train sparse MLP/Transformer layers with static CSR weights
- Performance within 2x of cuSPARSE for typical sparsity
- Full test coverage for correctness
10.2 Phase 2: N:M Structured + Tensor Cores
Goal: Hardware-accelerated 2:4 sparsity on Ampere+
Timeline: 4-6 weeks

Components:
├── N:M Storage
│   ├── NMStorage<R, 2, 4> implementation
│   ├── Compressed value layout
│   └── Index metadata packing
│
├── Conversions
│   ├── Dense → 2:4 (magnitude pruning)
│   ├── CSR → 2:4 (pattern adaptation)
│   └── 2:4 → Dense (unpacking)
│
├── SpMM Kernels
│   ├── Tensor core kernel (CUDA mma.sp)
│   ├── Emulated kernel (fallback)
│   └── Hybrid kernel selection
│
├── Autodiff
│   ├── Backward for N:M format
│   └── Pattern-preserving gradient
│
└── Burn Integration
    ├── NMStorage in SparseBackend
    ├── Auto format selection (CSR vs N:M)
    └── API: to_nm_sparse(), nm_matmul()

Deliverables:
- 2:4 sparse matmul using tensor cores
- 1.5-2x speedup over dense at 50% sparsity
- Automatic fallback on non-Ampere hardware
10.3 Phase 3: Block Formats + Dynamic Sparsity
Goal: BSR/BCSC and RigL-style training support
Timeline: 6-8 weeks

Components:
├── Block Formats
│   ├── BsrStorage<R> implementation
│   ├── BcscStorage<R> implementation
│   ├── Block-aware SpMM kernels
│   └── Format selection heuristics
│
├── Dynamic Sparsity
│   ├── AllowGrowth mask mode
│   ├── Pattern union operations
│   ├── COO accumulator for gradients
│   └── Resparsification API
│
├── Advanced Algorithms
│   ├── MergePath SpMM for irregular patterns
│   ├── Runtime autotuning
│   └── Profiling infrastructure
│
├── CubeCL Fusion
│   ├── SparseOperation → CubeIR
│   ├── Epilogue fusion for SpMM
│   └── Element-wise chain fusion
│
└── Full Burn Integration
    ├── Complete SparseBackend trait
    ├── SparseParam<B> for training
    └── Sparse optimizers (SGD, Adam)

Deliverables:
- Full sparse training pipeline
- RigL-style dynamic sparsity training
- Performance competitive with Triton sparse
- Production-ready API

Appendix A: Benchmark Targets
Target Performance (vs cuSPARSE/cuBLAS):

CSR SpMM (90% sparsity):
├── M=4096, K=4096, N=4096
│   cuSPARSE: ~2.1 ms
│   Target: < 2.5 ms (within 1.2x)
│
├── M=16384, K=4096, N=4096  (typical transformer)
│   cuSPARSE: ~8.5 ms
│   Target: < 10 ms (within 1.2x)
│
└── Irregular sparsity (power-law rows)
    Merge path should match cuSPARSE

2:4 Sparse Tensor Core:
├── M=4096, K=4096, N=4096
│   Dense cuBLAS: ~0.8 ms
│   2:4 Target: < 0.5 ms (1.6x speedup)
│
└── M=16384, K=4096, N=4096
    Dense cuBLAS: ~3.2 ms
    2:4 Target: < 2.0 ms (1.6x speedup)

Memory Efficiency:
├── CSR overhead vs dense: < 5% at 90% sparsity
├── 2:4 overhead: ~50% of dense (by design)
└── Gradient memory: same as forward (ExactPattern)
Appendix B: Testing Strategy
Test Categories:

1. Correctness Tests
   ├── SpMM output matches dense matmul (within tolerance)
   ├── Gradient correctness via finite differences
   ├── Format conversion round-trips
   └── Edge cases: empty rows, single element, etc.

2. Performance Tests
   ├── Benchmark suite vs cuSPARSE
   ├── Algorithm selection validation
   ├── Fusion benefit measurement
   └── Memory pressure tests

3. Integration Tests
   ├── Full training loop (sparse MLP)
   ├── Transformer layer with sparse attention
   ├── RigL training (dynamic sparsity)
   └── Mixed precision (fp16/bf16)

4. Stress Tests
   ├── Very large matrices (M > 100k)
   ├── Extreme sparsity (99%+)
   ├── Many small sparse ops (fusion stress)
   └── Memory exhaustion handling

Appendix C: References

cuSPARSE Library (NVIDIA) - Baseline for CSR algorithms
"Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format" - Bell & Garland, 2009
"Merge-Based Sparse Matrix-Vector Multiplication (SpMV) Using the CSR Storage Format" - Merrill & Garland, 2016
"Accelerating Sparse Deep Neural Networks" - NVIDIA, 2021 (N:M sparsity)
"Rigging the Lottery: Making All Tickets Winners" - Evci et al., 2020 (RigL)
"Learning Both Weights and Connections for Efficient Neural Networks" - Han et al., 2015
CubeCL documentation and source
Burn framework architecture docs


CubeCL-Sparse: Architecture Specification — Part 2
Version: 0.1.0-draft
Continuation of: Part 1 (Core GPU Operations)
Focus: Integration Layer, Handle System, Extended Operations

Table of Contents

Handle System
Sparse Views
Batched Operations
DType & Precision Rules
Shape Semantics
Dynamic Pattern Engine
Sparse-Dense Interop
Pruning & Sparsification API
Sparse Modules (burn-nn)
Serialization
Device Placement
Error Model
Fusion Rules (Formalized)
JIT Specialization Signatures
Inspection & Debug Tools
Extended File Structure


11. Handle System
CubeCL uses a three-tier ownership model for tensors. Sparse tensors need the same pattern for consistent kernel interfaces, autodiff tracking, and fusion.
11.1 Handle Hierarchy
┌─────────────────────────────────────────────────────────────────┐
│                     SparseTensor<R>                             │
│  User-facing type. Owns metadata, holds handle reference.       │
│  - shape, dtype, format                                         │
│  - sparsity statistics                                          │
│  - reference to SparseTensorHandle                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ references
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SparseTensorHandle<R>                         │
│  Runtime-owned. Holds actual GPU buffer handles.                │
│  - buffer handles (row_ptrs, col_indices, values, etc.)         │
│  - allocation metadata                                          │
│  - reference count for sharing                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ borrowed by kernels
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SparseTensorHandleRef<'a, R>                   │
│  Borrowed view for kernel execution.                            │
│  - lifetime-bound to handle                                     │
│  - used in kernel signatures                                    │
│  - enables fusion across operations                             │
└─────────────────────────────────────────────────────────────────┘
11.2 Type Definitions
rust/// User-facing sparse tensor
pub struct SparseTensor<R: Runtime> {
    /// Unique tensor ID for autodiff tracking
    id: TensorId,
    
    /// Handle to GPU storage
    handle: Arc<SparseTensorHandle<R>>,
    
    /// Logical shape [rows, cols] or [batch, rows, cols]
    shape: Shape,
    
    /// Data type
    dtype: DType,
    
    /// Sparse format
    format: SparseFormatId,
    
    /// Cached sparsity ratio (avoids GPU sync)
    sparsity: f32,
}

/// Runtime-owned GPU storage
pub struct SparseTensorHandle<R: Runtime> {
    /// Format-specific buffer set
    buffers: SparseBufferSet<R>,
    
    /// Storage metadata (nnz, block counts, etc.)
    storage_meta: StorageMetadata,
    
    /// Memory pool reference for deallocation
    pool: Arc<dyn MemoryPool<R>>,
    
    /// Device this handle lives on
    device: R::Device,
}

/// Format-specific buffer organization
pub enum SparseBufferSet<R: Runtime> {
    Csr {
        row_ptrs: Handle<R::Server>,
        col_indices: Handle<R::Server>,
        values: Handle<R::Server>,
    },
    Csc {
        col_ptrs: Handle<R::Server>,
        row_indices: Handle<R::Server>,
        values: Handle<R::Server>,
    },
    Coo {
        row_indices: Handle<R::Server>,
        col_indices: Handle<R::Server>,
        values: Handle<R::Server>,
    },
    NM {
        values: Handle<R::Server>,
        indices: Handle<R::Server>,
        n: u8,
        m: u8,
    },
    Bsr {
        block_row_ptrs: Handle<R::Server>,
        block_col_indices: Handle<R::Server>,
        block_values: Handle<R::Server>,
        block_rows: u16,
        block_cols: u16,
    },
}

/// Borrowed view for kernel execution
pub struct SparseTensorHandleRef<'a, R: Runtime> {
    /// Borrowed buffer references
    buffers: SparseBufferRefSet<'a, R>,
    
    /// Storage metadata (needed for kernel dispatch)
    storage_meta: &'a StorageMetadata,
    
    /// Format identifier
    format: SparseFormatId,
}

pub enum SparseBufferRefSet<'a, R: Runtime> {
    Csr {
        row_ptrs: &'a Handle<R::Server>,
        col_indices: &'a Handle<R::Server>,
        values: &'a Handle<R::Server>,
    },
    // ... other formats
}
11.3 Handle Operations
rustimpl<R: Runtime> SparseTensorHandle<R> {
    /// Borrow for kernel execution
    pub fn as_ref(&self) -> SparseTensorHandleRef<'_, R> {
        SparseTensorHandleRef {
            buffers: self.buffers.as_ref(),
            storage_meta: &self.storage_meta,
            format: self.buffers.format_id(),
        }
    }
    
    /// Clone handle (shallow, shares buffers)
    pub fn clone_ref(&self) -> Arc<Self> {
        // Increment refcount, share buffers
    }
    
    /// Deep clone (copy GPU buffers)
    pub fn clone_deep(&self, client: &ComputeClient<...>) -> Self {
        // Allocate new buffers, copy data
    }
    
    /// Check if this is the only reference (safe to mutate)
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.pool) == 1
    }
}

impl<R: Runtime> SparseTensor<R> {
    /// Get tensor ID for autodiff
    pub fn id(&self) -> TensorId {
        self.id
    }
    
    /// Borrow handle for operations
    pub fn handle_ref(&self) -> SparseTensorHandleRef<'_, R> {
        self.handle.as_ref()
    }
    
    /// Make tensor contiguous (rebuild if needed)
    pub fn into_contiguous(self, client: &ComputeClient<...>) -> Self {
        // For views, materialize into new storage
    }
    
    /// Ensure unique ownership (clone if shared)
    pub fn into_unique(self, client: &ComputeClient<...>) -> Self {
        if self.handle.is_unique() {
            self
        } else {
            Self {
                handle: Arc::new(self.handle.clone_deep(client)),
                ..self
            }
        }
    }
}
11.4 Kernel Interface
rust/// Kernel that operates on sparse tensors
#[cube(launch)]
pub fn sparse_kernel_example<F: Float>(
    // Takes handle refs, not full tensors
    a: &SparseTensorHandleRef<'_, R>,
    b: &Tensor<F>,  // Dense input
    c: &mut Tensor<F>,  // Dense output
    #[comptime] config: KernelConfig,
) {
    // Access buffers through ref
    match a.buffers {
        SparseBufferRefSet::Csr { row_ptrs, col_indices, values } => {
            // Use buffers directly
        }
        _ => panic!("Unsupported format"),
    }
}

/// Dispatch wrapper that handles tensor → handle_ref conversion
pub fn sparse_matmul<R: Runtime>(
    a: &SparseTensor<R>,
    b: &Tensor<R>,
    client: &ComputeClient<...>,
) -> Tensor<R> {
    let a_ref = a.handle_ref();
    let b_handle = b.as_handle();
    
    // Dispatch to kernel
    sparse_kernel_example::launch(
        client,
        &a_ref,
        &b_handle,
        &mut output_handle,
        config,
    );
    
    Tensor::from_handle(output_handle)
}

12. Sparse Views
Views provide zero-cost structural transforms without copying or rebuilding storage.
12.1 View Types
rust/// Sparse view descriptor
#[derive(Clone, Debug)]
pub enum SparseView {
    /// No transformation (identity)
    Identity,
    
    /// Transpose (reinterpret CSR as CSC or vice versa)
    Transpose,
    
    /// Row slice [start_row, end_row)
    RowSlice { start: usize, end: usize },
    
    /// Column slice [start_col, end_col) — limited support
    ColSlice { start: usize, end: usize },
    
    /// Block interpretation (view CSR as BSR with given block size)
    BlockInterpret { block_rows: usize, block_cols: usize },
    
    /// Batch dimension view (view 2D as slice of 3D)
    BatchSlice { batch_idx: usize },
}

/// Sparse tensor with optional view transformation
pub struct SparseTensorView<R: Runtime> {
    /// Underlying tensor
    base: SparseTensor<R>,
    
    /// View transformation (None = identity)
    view: Option<SparseView>,
    
    /// Effective shape after view
    effective_shape: Shape,
    
    /// Effective format after view
    effective_format: SparseFormatId,
}
12.2 View Rules
View Compatibility Matrix:
┌──────────────┬─────┬─────┬─────┬─────┬─────┐
│ View Type    │ CSR │ CSC │ COO │ N:M │ BSR │
├──────────────┼─────┼─────┼─────┼─────┼─────┤
│ Transpose    │ ✓*  │ ✓*  │ ✓   │ ✗   │ ✓*  │
│ RowSlice     │ ✓   │ ○   │ ✓   │ ✓** │ ✓** │
│ ColSlice     │ ○   │ ✓   │ ✓   │ ✗   │ ✗   │
│ BlockInterp  │ ✓** │ ✓** │ ✗   │ ✗   │ ─   │
│ BatchSlice   │ ✓   │ ✓   │ ✓   │ ✓   │ ✓   │
└──────────────┴─────┴─────┴─────┴─────┴─────┘

Legend:
  ✓   = zero-cost view
  ✓*  = reinterpretation (CSR→CSC, no data change)
  ✓** = requires alignment (slice must align to block/group boundaries)
  ○   = requires materialization (not zero-cost)
  ✗   = not supported
  ─   = not applicable
12.3 View Implementation
rustimpl<R: Runtime> SparseTensorView<R> {
    /// Create transpose view (CSR ↔ CSC reinterpretation)
    pub fn transpose(tensor: SparseTensor<R>) -> Self {
        let effective_format = match tensor.format {
            SparseFormatId::Csr => SparseFormatId::Csc,
            SparseFormatId::Csc => SparseFormatId::Csr,
            SparseFormatId::Coo => SparseFormatId::Coo,  // COO transpose is just swapping interpretation
            f => panic!("Transpose view not supported for {:?}", f),
        };
        
        let effective_shape = Shape::new([
            tensor.shape.dims[1],  // cols become rows
            tensor.shape.dims[0],  // rows become cols
        ]);
        
        Self {
            base: tensor,
            view: Some(SparseView::Transpose),
            effective_shape,
            effective_format,
        }
    }
    
    /// Create row slice view
    pub fn row_slice(tensor: SparseTensor<R>, start: usize, end: usize) -> Result<Self, SparseError> {
        // Validate bounds
        if end > tensor.shape.dims[0] {
            return Err(SparseError::SliceOutOfBounds);
        }
        
        // For N:M and BSR, check alignment
        match tensor.format {
            SparseFormatId::NM { m, .. } => {
                if start % m as usize != 0 || end % m as usize != 0 {
                    return Err(SparseError::SliceAlignmentRequired { 
                        required: m as usize 
                    });
                }
            }
            SparseFormatId::Bsr { block_rows, .. } => {
                if start % block_rows as usize != 0 || end % block_rows as usize != 0 {
                    return Err(SparseError::SliceAlignmentRequired { 
                        required: block_rows as usize 
                    });
                }
            }
            _ => {}
        }
        
        let effective_shape = Shape::new([
            end - start,
            tensor.shape.dims[1],
        ]);
        
        Ok(Self {
            base: tensor,
            view: Some(SparseView::RowSlice { start, end }),
            effective_shape,
            effective_format: tensor.format,
        })
    }
    
    /// Materialize view into concrete tensor (when zero-cost not possible)
    pub fn materialize(self, client: &ComputeClient<...>) -> SparseTensor<R> {
        match self.view {
            None => self.base,
            Some(SparseView::Transpose) => {
                // For transpose, we can keep as view for most ops
                // Only materialize if absolutely needed
                self.base.transpose_to_format(self.effective_format, client)
            }
            Some(SparseView::RowSlice { start, end }) => {
                // Build new CSR with subset of rows
                self.materialize_row_slice(start, end, client)
            }
            Some(SparseView::ColSlice { start, end }) => {
                // More expensive: need to filter and reindex
                self.materialize_col_slice(start, end, client)
            }
            _ => todo!(),
        }
    }
    
    /// Check if view can be used directly (zero-cost) for given operation
    pub fn can_use_directly(&self, op: &dyn SparseOperation<R>) -> bool {
        match (&self.view, op.supported_views()) {
            (None, _) => true,
            (Some(v), supported) => supported.contains(v),
        }
    }
}
12.4 View-Aware Operations
rust/// Extended sparse operation trait with view support
pub trait SparseOperationWithViews<R: Runtime>: SparseOperation<R> {
    /// Which views can be used without materialization
    fn supported_views(&self) -> &[SparseView];
    
    /// Execute with potentially viewed inputs
    fn execute_with_view(
        &self,
        input: SparseTensorView<R>,
        client: &ComputeClient<...>,
    ) -> Self::Outputs {
        if input.can_use_directly(self) {
            self.execute_view(input, client)
        } else {
            self.execute(input.materialize(client), (), client)
        }
    }
    
    /// Execute directly on view (when supported)
    fn execute_view(
        &self,
        input: SparseTensorView<R>,
        client: &ComputeClient<...>,
    ) -> Self::Outputs;
}

// SpMM supports transpose view (use CSC kernel instead of CSR)
impl<R: Runtime> SparseOperationWithViews<R> for CsrSpmm<R> {
    fn supported_views(&self) -> &[SparseView] {
        &[SparseView::Transpose, SparseView::RowSlice { start: 0, end: 0 }]
    }
    
    fn execute_view(
        &self,
        input: SparseTensorView<R>,
        client: &ComputeClient<...>,
    ) -> Tensor<R> {
        match input.view {
            Some(SparseView::Transpose) => {
                // Use CSC SpMM kernel instead
                CscSpmm::execute(input.base, client)
            }
            Some(SparseView::RowSlice { start, end }) => {
                // Adjust row_ptrs offset, execute on slice
                self.execute_row_slice(input.base, start, end, client)
            }
            _ => unreachable!(),
        }
    }
}

13. Batched Operations
Support for batched sparse operations, critical for transformer training.
13.1 Batched Tensor Layout
rust/// Batched sparse tensor storage
pub enum BatchedSparseStorage<R: Runtime> {
    /// Uniform batching: all batch elements have same sparsity pattern
    /// More efficient, single set of index buffers
    Uniform {
        /// Shared index structure
        indices: SparseIndexBuffers<R>,
        /// Batched values [batch_size × nnz]
        values: Handle<R::Server>,
        /// Batch dimension
        batch_size: usize,
    },
    
    /// Variable batching: each element has different pattern
    /// Requires per-element metadata
    Variable {
        /// Per-batch-element storage
        elements: Vec<SparseBufferSet<R>>,
        /// Offsets for each element (for coalesced access)
        offsets: Handle<R::Server>,
    },
}

/// Shared index buffers for uniform batching
pub struct SparseIndexBuffers<R: Runtime> {
    pub format: SparseFormatId,
    pub row_ptrs: Option<Handle<R::Server>>,
    pub col_indices: Option<Handle<R::Server>>,
    pub block_indices: Option<Handle<R::Server>>,
    pub nm_indices: Option<Handle<R::Server>>,
}

/// Batched sparse tensor
pub struct BatchedSparseTensor<R: Runtime> {
    pub storage: BatchedSparseStorage<R>,
    pub shape: Shape,  // [batch, rows, cols]
    pub dtype: DType,
    pub format: SparseFormatId,
}
13.2 Batched SpMM
Batched SpMM: A_sparse[B×M×K] × B_dense[B×K×N] = C_dense[B×M×N]

Two modes:
1. Uniform pattern (all batches share sparsity structure)
   - Single kernel launch, batch dimension in grid
   - Index buffers shared, only values differ per batch
   
2. Variable pattern (each batch has different sparsity)
   - Multiple kernel launches OR
   - Padded to max nnz with masking
rust/// Batched CSR SpMM
pub struct BatchedCsrSpmm<R: Runtime> {
    pub algorithm: CsrSpmmAlgorithm,
}

impl<R: Runtime> BatchedCsrSpmm<R> {
    pub fn execute(
        &self,
        a: &BatchedSparseTensor<R>,
        b: &Tensor<R>,  // [B, K, N]
        client: &ComputeClient<...>,
    ) -> Tensor<R> {
        match &a.storage {
            BatchedSparseStorage::Uniform { indices, values, batch_size } => {
                self.execute_uniform_batch(indices, values, *batch_size, b, client)
            }
            BatchedSparseStorage::Variable { elements, offsets } => {
                self.execute_variable_batch(elements, offsets, b, client)
            }
        }
    }
}

/// Kernel for uniform batched SpMM
#[cube(launch)]
pub fn batched_csr_spmm_uniform<F: Float>(
    // Shared indices (same for all batches)
    row_ptrs: &Array<u32>,
    col_indices: &Array<u32>,
    // Batched values [batch_size × nnz]
    values: &Array<F>,
    // Dense B [batch_size × K × N]
    b_matrix: &Tensor<F>,
    // Output C [batch_size × M × N]
    c_matrix: &mut Tensor<F>,
    #[comptime] batch_size: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] n: u32,
    #[comptime] nnz: u32,
) {
    let batch = ABSOLUTE_POS_Z as u32;
    let row = ABSOLUTE_POS_X as u32;
    let col = ABSOLUTE_POS_Y as u32;
    
    if batch >= batch_size || row >= m || col >= n {
        return;
    }
    
    let row_start = row_ptrs[row];
    let row_end = row_ptrs[row + 1];
    
    let mut sum = F::new(0.0);
    for idx in row_start..row_end {
        let k_idx = col_indices[idx];
        // Values indexed by batch
        let a_val = values[batch * nnz + idx];
        // B indexed by batch
        let b_val = b_matrix[(batch * k + k_idx) * n + col];
        sum += a_val * b_val;
    }
    
    c_matrix[(batch * m + row) * n + col] = sum;
}
13.3 Batch Operations
rustimpl<R: Runtime> BatchedSparseTensor<R> {
    /// Stack multiple sparse tensors with same pattern
    pub fn stack_uniform(tensors: Vec<SparseTensor<R>>, client: &ComputeClient<...>) -> Self {
        // Verify all have same pattern
        let first_meta = tensors[0].storage_metadata();
        for t in &tensors[1..] {
            assert!(t.has_same_pattern(&tensors[0]));
        }
        
        // Concatenate values buffers
        let batch_size = tensors.len();
        let values = concatenate_buffers(
            tensors.iter().map(|t| t.values_buffer()),
            client,
        );
        
        BatchedSparseTensor {
            storage: BatchedSparseStorage::Uniform {
                indices: tensors[0].extract_indices(),
                values,
                batch_size,
            },
            shape: Shape::new([batch_size, first_meta.rows, first_meta.cols]),
            dtype: tensors[0].dtype,
            format: tensors[0].format,
        }
    }
    
    /// Stack sparse tensors with different patterns
    pub fn stack_variable(tensors: Vec<SparseTensor<R>>, client: &ComputeClient<...>) -> Self {
        let batch_size = tensors.len();
        let elements: Vec<_> = tensors.into_iter()
            .map(|t| t.into_buffer_set())
            .collect();
            
        // Compute offsets for coalesced access
        let offsets = compute_batch_offsets(&elements, client);
        
        BatchedSparseTensor {
            storage: BatchedSparseStorage::Variable { elements, offsets },
            // ... shape, dtype, format
        }
    }
    
    /// Index single batch element
    pub fn get(&self, batch_idx: usize) -> SparseTensorView<R> {
        SparseTensorView::batch_slice(self, batch_idx)
    }
    
    /// Unbatch back to individual tensors
    pub fn unbatch(self, client: &ComputeClient<...>) -> Vec<SparseTensor<R>> {
        match self.storage {
            BatchedSparseStorage::Uniform { indices, values, batch_size } => {
                // Split values buffer, share indices
                split_uniform_batch(indices, values, batch_size, client)
            }
            BatchedSparseStorage::Variable { elements, .. } => {
                elements.into_iter()
                    .map(|e| SparseTensor::from_buffer_set(e))
                    .collect()
            }
        }
    }
}

14. DType & Precision Rules
Explicit dtype support matrix for sparse formats and operations.
14.1 Format-DType Compatibility
Format DType Support:
┌─────────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Format  │ f32   │ f16   │ bf16  │ f8    │ i32   │ i8    │
├─────────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ CSR     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │
│ CSC     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │
│ COO     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │ ✓     │
│ N:M     │ ○     │ ✓TC   │ ✓TC   │ ✓TC*  │ ✗     │ ✗     │
│ BSR     │ ✓     │ ✓TC   │ ✓TC   │ ○     │ ✓     │ ○     │
└─────────┴───────┴───────┴───────┴───────┴───────┴───────┘

Legend:
  ✓    = full support
  ✓TC  = tensor core accelerated
  ✓TC* = tensor core on Hopper+ only (fp8)
  ○    = supported but not optimal
  ✗    = not supported
14.2 Index DTypes
rust/// Index element types
pub enum IndexDType {
    /// 32-bit indices (default, supports up to 4B elements)
    U32,
    /// 16-bit indices (for smaller tensors, saves memory)
    U16,
    /// 64-bit indices (for very large tensors)
    U64,
}

impl IndexDType {
    /// Select based on dimension size
    pub fn for_dimension(dim: usize) -> Self {
        match dim {
            d if d <= u16::MAX as usize => IndexDType::U16,
            d if d <= u32::MAX as usize => IndexDType::U32,
            _ => IndexDType::U64,
        }
    }
    
    /// Bytes per index element
    pub fn bytes(&self) -> usize {
        match self {
            IndexDType::U16 => 2,
            IndexDType::U32 => 4,
            IndexDType::U64 => 8,
        }
    }
}
14.3 Mixed Precision SpMM
rust/// Mixed precision configuration for SpMM
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    /// Input A dtype
    pub input_a: DType,
    /// Input B dtype  
    pub input_b: DType,
    /// Accumulation dtype (usually higher precision)
    pub accumulate: DType,
    /// Output dtype
    pub output: DType,
}

impl MixedPrecisionConfig {
    /// Standard fp16 with fp32 accumulation
    pub const FP16_FP32_ACC: Self = Self {
        input_a: DType::F16,
        input_b: DType::F16,
        accumulate: DType::F32,
        output: DType::F16,
    };
    
    /// BF16 with fp32 accumulation (recommended for training)
    pub const BF16_FP32_ACC: Self = Self {
        input_a: DType::BF16,
        input_b: DType::BF16,
        accumulate: DType::F32,
        output: DType::BF16,
    };
    
    /// TF32 (Ampere+)
    pub const TF32: Self = Self {
        input_a: DType::TF32,
        input_b: DType::TF32,
        accumulate: DType::F32,
        output: DType::F32,
    };
}

/// Kernel selection based on dtype
pub fn select_spmm_kernel<R: Runtime>(
    format: SparseFormatId,
    precision: &MixedPrecisionConfig,
    device: &R::Device,
) -> Box<dyn SparseOperation<R>> {
    match (format, precision, device.compute_capability()) {
        // 2:4 + fp16/bf16 + Ampere+ → tensor core kernel
        (SparseFormatId::NM { n: 2, m: 4 }, cfg, cc) 
            if (cfg.input_a == DType::F16 || cfg.input_a == DType::BF16) 
               && cc >= ComputeCapability::SM_80 
        => {
            Box::new(NMSpmmTensorCore::new(precision.clone()))
        }
        
        // BSR + fp16 + Ampere+ → block tensor core kernel
        (SparseFormatId::Bsr { .. }, cfg, cc)
            if cfg.input_a == DType::F16 && cc >= ComputeCapability::SM_80
        => {
            Box::new(BsrSpmmTensorCore::new(precision.clone()))
        }
        
        // Default: CSR with appropriate dtype
        (SparseFormatId::Csr, cfg, _) => {
            Box::new(CsrSpmm::new(cfg.clone()))
        }
        
        _ => panic!("Unsupported format/dtype combination"),
    }
}
14.4 Quantization Support
rust/// Sparse tensor quantization
pub struct SparseQuantization;

impl SparseQuantization {
    /// Quantize sparse tensor to int8
    pub fn quantize_int8<R: Runtime>(
        tensor: SparseTensor<R>,
        scale: f32,
        zero_point: i8,
        client: &ComputeClient<...>,
    ) -> (SparseTensor<R>, QuantizationParams) {
        // Only quantize values buffer, indices unchanged
        let quantized_values = quantize_buffer(
            tensor.values_buffer(),
            scale,
            zero_point,
            client,
        );
        
        (
            tensor.with_values(quantized_values, DType::I8),
            QuantizationParams { scale, zero_point },
        )
    }
    
    /// Dequantize back to float
    pub fn dequantize<R: Runtime>(
        tensor: SparseTensor<R>,
        params: &QuantizationParams,
        target_dtype: DType,
        client: &ComputeClient<...>,
    ) -> SparseTensor<R> {
        let dequantized_values = dequantize_buffer(
            tensor.values_buffer(),
            params,
            target_dtype,
            client,
        );
        
        tensor.with_values(dequantized_values, target_dtype)
    }
}

15. Shape Semantics
Clear distinction between logical shape and storage shape.
15.1 Shape Types
rust/// Logical shape (what the tensor represents mathematically)
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalShape {
    /// Dimensions: [rows, cols] or [batch, rows, cols]
    pub dims: Vec<usize>,
}

/// Storage shape (how data is actually stored)
#[derive(Clone, Debug)]
pub enum StorageShape {
    Csr {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    Csc {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    Coo {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    NM {
        rows: usize,
        cols: usize,  // Must be divisible by M
        n: usize,
        m: usize,
        num_groups: usize,  // cols / m
        values_count: usize,  // rows * num_groups * n
    },
    Bsr {
        rows: usize,  // Must be divisible by block_rows
        cols: usize,  // Must be divisible by block_cols
        block_rows: usize,
        block_cols: usize,
        num_block_rows: usize,  // rows / block_rows
        num_block_cols: usize,  // cols / block_cols
        nnz_blocks: usize,
    },
}

impl StorageShape {
    /// Total elements in values buffer
    pub fn values_count(&self) -> usize {
        match self {
            Self::Csr { nnz, .. } => *nnz,
            Self::Csc { nnz, .. } => *nnz,
            Self::Coo { nnz, .. } => *nnz,
            Self::NM { values_count, .. } => *values_count,
            Self::Bsr { nnz_blocks, block_rows, block_cols, .. } => {
                nnz_blocks * block_rows * block_cols
            }
        }
    }
    
    /// Total bytes for index buffers
    pub fn index_bytes(&self, index_dtype: IndexDType) -> usize {
        let elem_size = index_dtype.bytes();
        match self {
            Self::Csr { rows, nnz, .. } => {
                (rows + 1) * elem_size + nnz * elem_size  // row_ptrs + col_indices
            }
            Self::Csc { cols, nnz, .. } => {
                (cols + 1) * elem_size + nnz * elem_size  // col_ptrs + row_indices
            }
            Self::Coo { nnz, .. } => {
                2 * nnz * elem_size  // row_indices + col_indices
            }
            Self::NM { num_groups, rows, .. } => {
                rows * num_groups * 2  // 4-bit indices packed into u16
            }
            Self::Bsr { num_block_rows, nnz_blocks, .. } => {
                (num_block_rows + 1) * elem_size + nnz_blocks * elem_size
            }
        }
    }
    
    /// Sparsity ratio (proportion of zeros)
    pub fn sparsity(&self) -> f32 {
        let total = self.logical_elements();
        let stored = self.values_count();
        1.0 - (stored as f32 / total as f32)
    }
    
    fn logical_elements(&self) -> usize {
        match self {
            Self::Csr { rows, cols, .. } => rows * cols,
            Self::Csc { rows, cols, .. } => rows * cols,
            Self::Coo { rows, cols, .. } => rows * cols,
            Self::NM { rows, cols, .. } => rows * cols,
            Self::Bsr { rows, cols, .. } => rows * cols,
        }
    }
}
15.2 Shape Compatibility
rust/// Check shape compatibility for sparse operations
pub struct ShapeChecker;

impl ShapeChecker {
    /// Check SpMM compatibility: A[M,K] × B[K,N] → C[M,N]
    pub fn check_spmm(
        a_shape: &LogicalShape,
        b_shape: &LogicalShape,
    ) -> Result<LogicalShape, SparseError> {
        let (m, k_a) = (a_shape.dims[0], a_shape.dims[1]);
        let (k_b, n) = (b_shape.dims[0], b_shape.dims[1]);
        
        if k_a != k_b {
            return Err(SparseError::ShapeMismatch {
                op: "SpMM",
                expected: format!("A.cols ({}) == B.rows", k_a),
                got: k_b,
            });
        }
        
        Ok(LogicalShape { dims: vec![m, n] })
    }
    
    /// Check element-wise compatibility (same logical shape)
    pub fn check_elementwise(
        a_shape: &LogicalShape,
        b_shape: &LogicalShape,
    ) -> Result<LogicalShape, SparseError> {
        if a_shape != b_shape {
            return Err(SparseError::ShapeMismatch {
                op: "Elementwise",
                expected: format!("{:?}", a_shape),
                got: b_shape.dims[0],  // simplified
            });
        }
        Ok(a_shape.clone())
    }
    
    /// Check pattern compatibility (same storage structure)
    pub fn check_pattern_match(
        a_storage: &StorageShape,
        b_storage: &StorageShape,
    ) -> Result<(), SparseError> {
        match (a_storage, b_storage) {
            (StorageShape::Csr { nnz: nnz_a, .. }, StorageShape::Csr { nnz: nnz_b, .. }) => {
                if nnz_a != nnz_b {
                    return Err(SparseError::PatternMismatch {
                        expected_nnz: *nnz_a,
                        got_nnz: *nnz_b,
                    });
                }
            }
            _ => return Err(SparseError::FormatMismatch),
        }
        Ok(())
    }
}

16. Dynamic Pattern Engine
Unified system for managing sparsity patterns that change during training.
16.1 Pattern Manager
rust/// Dynamic sparsity pattern manager
pub struct PatternManager<R: Runtime> {
    /// Current pattern (authoritative)
    current: SparsityPattern<R>,
    
    /// Pending modifications (batched for efficiency)
    pending: PatternDelta,
    
    /// Mode governing pattern changes
    mode: PatternMode,
    
    /// History for debugging/analysis
    history: Option<PatternHistory>,
}

/// Sparsity pattern (structure without values)
pub struct SparsityPattern<R: Runtime> {
    /// Format of the pattern
    pub format: SparseFormatId,
    /// Logical shape
    pub shape: LogicalShape,
    /// Index buffers only (no values)
    pub indices: SparseIndexBuffers<R>,
    /// Statistics
    pub stats: PatternStatistics,
}

#[derive(Clone, Debug)]
pub struct PatternStatistics {
    pub nnz: usize,
    pub sparsity: f32,
    pub row_nnz_min: usize,
    pub row_nnz_max: usize,
    pub row_nnz_mean: f32,
    pub row_nnz_std: f32,
}

/// Pattern modification modes
#[derive(Clone, Copy, Debug)]
pub enum PatternMode {
    /// Pattern is fixed, any modification is an error
    Static,
    
    /// Pattern can grow (add non-zeros)
    GrowOnly,
    
    /// Pattern can shrink (remove non-zeros)
    ShrinkOnly,
    
    /// Pattern can change freely
    Dynamic,
    
    /// Pattern follows N:M constraint (can only swap within groups)
    NMConstrained { n: usize, m: usize },
}
16.2 Pattern Delta (Modifications)
rust/// Batched pattern modifications
#[derive(Default)]
pub struct PatternDelta {
    /// Non-zeros to add: (row, col)
    pub additions: Vec<(usize, usize)>,
    
    /// Non-zeros to remove: (row, col)
    pub removals: Vec<(usize, usize)>,
    
    /// For N:M: swaps within groups (group_idx, old_pos, new_pos)
    pub nm_swaps: Vec<(usize, usize, usize)>,
}

impl PatternDelta {
    pub fn add(&mut self, row: usize, col: usize) {
        self.additions.push((row, col));
    }
    
    pub fn remove(&mut self, row: usize, col: usize) {
        self.removals.push((row, col));
    }
    
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty() && self.nm_swaps.is_empty()
    }
    
    pub fn net_change(&self) -> isize {
        self.additions.len() as isize - self.removals.len() as isize
    }
}
16.3 Pattern Operations
rustimpl<R: Runtime> PatternManager<R> {
    /// Create manager with static pattern
    pub fn static_pattern(pattern: SparsityPattern<R>) -> Self {
        Self {
            current: pattern,
            pending: PatternDelta::default(),
            mode: PatternMode::Static,
            history: None,
        }
    }
    
    /// Create manager for dynamic sparsity (e.g., RigL)
    pub fn dynamic(pattern: SparsityPattern<R>, mode: PatternMode) -> Self {
        Self {
            current: pattern,
            pending: PatternDelta::default(),
            mode,
            history: Some(PatternHistory::new()),
        }
    }
    
    /// Queue addition of non-zero (batched, not applied immediately)
    pub fn queue_add(&mut self, row: usize, col: usize) -> Result<(), SparseError> {
        match self.mode {
            PatternMode::Static => {
                return Err(SparseError::PatternModificationNotAllowed);
            }
            PatternMode::ShrinkOnly => {
                return Err(SparseError::PatternGrowthNotAllowed);
            }
            _ => {}
        }
        self.pending.add(row, col);
        Ok(())
    }
    
    /// Queue removal of non-zero
    pub fn queue_remove(&mut self, row: usize, col: usize) -> Result<(), SparseError> {
        match self.mode {
            PatternMode::Static => {
                return Err(SparseError::PatternModificationNotAllowed);
            }
            PatternMode::GrowOnly => {
                return Err(SparseError::PatternShrinkNotAllowed);
            }
            _ => {}
        }
        self.pending.remove(row, col);
        Ok(())
    }
    
    /// Apply all pending modifications
    pub fn apply_pending(&mut self, client: &ComputeClient<...>) -> Result<(), SparseError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        
        // Convert current pattern to COO for modification
        let mut coo = self.current.to_coo(client);
        
        // Apply removals
        for (row, col) in &self.pending.removals {
            coo.remove(*row, *col);
        }
        
        // Apply additions
        for (row, col) in &self.pending.additions {
            coo.add(*row, *col);
        }
        
        // Sort and coalesce
        coo.sort(client);
        coo.coalesce(client);
        
        // Convert back to target format
        self.current = SparsityPattern::from_coo(coo, self.current.format, client);
        
        // Record history
        if let Some(ref mut history) = self.history {
            history.record(&self.pending, &self.current.stats);
        }
        
        // Clear pending
        self.pending = PatternDelta::default();
        
        Ok(())
    }
    
    /// RigL-style update: drop lowest magnitude, grow highest gradient
    pub fn rigl_update(
        &mut self,
        values: &Handle<R::Server>,
        gradients: &Handle<R::Server>,
        drop_fraction: f32,
        client: &ComputeClient<...>,
    ) -> Result<(), SparseError> {
        if !matches!(self.mode, PatternMode::Dynamic) {
            return Err(SparseError::PatternModificationNotAllowed);
        }
        
        let num_to_drop = (self.current.stats.nnz as f32 * drop_fraction) as usize;
        
        // Find lowest magnitude current weights
        let to_drop = find_lowest_magnitude(values, num_to_drop, client);
        
        // Find highest magnitude gradients at zero positions
        let to_grow = find_highest_gradient_zeros(
            &self.current,
            gradients,
            num_to_drop,  // Grow same amount we drop
            client,
        );
        
        // Queue changes
        for (row, col) in to_drop {
            self.queue_remove(row, col)?;
        }
        for (row, col) in to_grow {
            self.queue_add(row, col)?;
        }
        
        self.apply_pending(client)
    }
}
16.4 COO Accumulator for Gradients
rust/// COO-based gradient accumulator for dynamic sparsity
pub struct CooGradientAccumulator<R: Runtime> {
    /// Accumulated gradients in COO format
    coo: CooStorage<R>,
    
    /// Whether to track positions outside current pattern
    track_dense_positions: bool,
    
    /// Dense gradient buffer (optional, for tracking regrowth candidates)
    dense_grad: Option<Handle<R::Server>>,
}

impl<R: Runtime> CooGradientAccumulator<R> {
    /// Accumulate sparse gradient (same pattern)
    pub fn accumulate_sparse(
        &mut self,
        grad: &CsrStorage<R>,
        client: &ComputeClient<...>,
    ) {
        // Convert to COO and scatter-add
        let grad_coo = grad.to_coo(client);
        self.coo.scatter_add(&grad_coo, client);
    }
    
    /// Accumulate dense gradient (for tracking regrowth candidates)
    pub fn accumulate_dense(
        &mut self,
        grad: &Tensor<R>,
        pattern: &SparsityPattern<R>,
        client: &ComputeClient<...>,
    ) {
        if self.track_dense_positions {
            // Track gradient magnitude at all positions
            self.dense_grad = Some(add_or_create_dense(
                self.dense_grad.take(),
                grad,
                client,
            ));
        }
        
        // Also accumulate at current pattern positions
        let sparse_grad = extract_at_pattern(grad, pattern, client);
        self.accumulate_sparse(&sparse_grad, client);
    }
    
    /// Get regrowth candidates (highest gradient magnitude at zero positions)
    pub fn regrowth_candidates(
        &self,
        pattern: &SparsityPattern<R>,
        k: usize,
        client: &ComputeClient<...>,
    ) -> Vec<(usize, usize)> {
        match &self.dense_grad {
            Some(dense) => {
                // Find top-k positions not in current pattern
                find_topk_outside_pattern(dense, pattern, k, client)
            }
            None => vec![],
        }
    }
    
    /// Extract final gradient and reset
    pub fn finalize(&mut self, client: &ComputeClient<...>) -> CsrStorage<R> {
        self.coo.sort(client);
        self.coo.coalesce(client);
        let result = self.coo.to_csr(client);
        self.coo = CooStorage::empty(self.coo.shape(), client);
        self.dense_grad = None;
        result
    }
}

17. Sparse-Dense Interop
Rules and operations for mixed sparse-dense computation.
17.1 Auto-Conversion Policy
rust/// Policy for automatic sparse ↔ dense conversion
#[derive(Clone, Debug)]
pub enum ConversionPolicy {
    /// Never auto-convert, error on type mismatch
    Strict,
    
    /// Convert sparse to dense if operation requires
    SparseToDenseAllowed,
    
    /// Convert dense to sparse if beneficial (based on threshold)
    DenseToSparseAllowed { sparsity_threshold: f32 },
    
    /// Allow both directions
    Permissive { sparsity_threshold: f32 },
}

impl Default for ConversionPolicy {
    fn default() -> Self {
        // Default: allow sparse→dense, not dense→sparse
        Self::SparseToDenseAllowed
    }
}

/// Check if conversion should happen
pub fn should_convert_to_dense<R: Runtime>(
    sparse: &SparseTensor<R>,
    policy: &ConversionPolicy,
) -> bool {
    match policy {
        ConversionPolicy::Strict => false,
        _ => {
            // Convert to dense if sparsity is low (< 50%)
            sparse.sparsity() < 0.5
        }
    }
}
17.2 Mixed Operations
rust/// Mixed sparse-dense addition: sparse + dense → dense
pub fn sparse_dense_add<R: Runtime>(
    sparse: &SparseTensor<R>,
    dense: &Tensor<R>,
    client: &ComputeClient<...>,
) -> Tensor<R> {
    // Clone dense, then scatter-add sparse values
    let mut result = dense.clone();
    scatter_add_sparse_to_dense(sparse, &mut result, client);
    result
}

/// Mixed sparse-dense multiplication (Hadamard): sparse * dense → sparse
pub fn sparse_dense_mul<R: Runtime>(
    sparse: &SparseTensor<R>,
    dense: &Tensor<R>,
    client: &ComputeClient<...>,
) -> SparseTensor<R> {
    // Result is sparse with same pattern as input sparse
    // Values are element-wise product at sparse positions
    gather_mul_sparse_dense(sparse, dense, client)
}

/// Sparse to dense conversion
pub fn to_dense<R: Runtime>(
    sparse: &SparseTensor<R>,
    client: &ComputeClient<...>,
) -> Tensor<R> {
    match sparse.format() {
        SparseFormatId::Csr => csr_to_dense(sparse.as_csr(), client),
        SparseFormatId::Csc => csc_to_dense(sparse.as_csc(), client),
        SparseFormatId::Coo => coo_to_dense(sparse.as_coo(), client),
        SparseFormatId::NM { .. } => nm_to_dense(sparse.as_nm(), client),
        SparseFormatId::Bsr { .. } => bsr_to_dense(sparse.as_bsr(), client),
    }
}

/// Dense to sparse conversion
pub fn to_sparse<R: Runtime>(
    dense: &Tensor<R>,
    threshold: f32,
    target_format: SparseFormatId,
    client: &ComputeClient<...>,
) -> SparseTensor<R> {
    // First create COO from dense (generic)
    let coo = dense_to_coo(dense, threshold, client);
    
    // Then convert to target format
    match target_format {
        SparseFormatId::Csr => coo.to_csr(client).into(),
        SparseFormatId::Csc => coo.to_csc(client).into(),
        SparseFormatId::Coo => coo.into(),
        SparseFormatId::NM { n, m } => {
            // N:M requires structured sparsification, not threshold
            panic!("Use to_nm_sparse() for N:M format");
        }
        SparseFormatId::Bsr { block_rows, block_cols } => {
            coo.to_bsr(block_rows, block_cols, client).into()
        }
    }
}
17.3 Broadcasting Rules
rust/// Broadcasting rules for sparse-dense operations
#[derive(Clone, Debug)]
pub enum BroadcastResult {
    /// No broadcasting needed, shapes match
    NoOp,
    
    /// Broadcast dense to match sparse (allowed)
    BroadcastDense { 
        from: Shape, 
        to: Shape,
        broadcast_dims: Vec<usize>,
    },
    
    /// Would need to broadcast sparse (not allowed for most ops)
    BroadcastSparseRequired,
    
    /// Shapes incompatible
    Incompatible,
}

pub fn check_broadcast(
    sparse_shape: &LogicalShape,
    dense_shape: &Shape,
) -> BroadcastResult {
    // Sparse broadcasting is generally not supported
    // Dense can broadcast to match sparse
    
    if sparse_shape.dims == dense_shape.dims {
        return BroadcastResult::NoOp;
    }
    
    // Check if dense can broadcast to sparse
    if can_broadcast(&dense_shape.dims, &sparse_shape.dims) {
        let broadcast_dims = find_broadcast_dims(&dense_shape.dims, &sparse_shape.dims);
        return BroadcastResult::BroadcastDense {
            from: dense_shape.clone(),
            to: Shape::new(sparse_shape.dims.clone()),
            broadcast_dims,
        };
    }
    
    // Check if sparse would need broadcasting (not supported)
    if can_broadcast(&sparse_shape.dims, &dense_shape.dims) {
        return BroadcastResult::BroadcastSparseRequired;
    }
    
    BroadcastResult::Incompatible
}

/// Perform broadcast if needed
pub fn broadcast_dense_for_sparse<R: Runtime>(
    dense: Tensor<R>,
    target_shape: &LogicalShape,
    client: &ComputeClient<...>,
) -> Tensor<R> {
    match check_broadcast(target_shape, &dense.shape()) {
        BroadcastResult::NoOp => dense,
        BroadcastResult::BroadcastDense { to, .. } => {
            dense.broadcast(to)
        }
        BroadcastResult::BroadcastSparseRequired => {
            panic!("Sparse broadcasting not supported");
        }
        BroadcastResult::Incompatible => {
            panic!("Incompatible shapes for broadcast");
        }
    }
}

18. Pruning & Sparsification API
User-facing API for creating and modifying sparse tensors.
18.1 Pruning Methods
rust/// Pruning strategy
#[derive(Clone, Debug)]
pub enum PruningStrategy {
    /// Global magnitude pruning (threshold all weights)
    MagnitudeGlobal { 
        /// Target sparsity (0.0-1.0)
        target_sparsity: f32 
    },
    
    /// Per-row/per-column magnitude pruning
    MagnitudeStructured { 
        target_sparsity: f32,
        /// Prune along this dimension
        dim: usize,
    },
    
    /// Random pruning (for initialization)
    Random { 
        target_sparsity: f32,
        seed: u64,
    },
    
    /// Movement pruning (based on weight change during training)
    Movement {
        target_sparsity: f32,
        /// Original weights (for computing movement)
        original: TensorId,
    },
    
    /// N:M structured pruning
    NMStructured { 
        n: usize, 
        m: usize 
    },
    
    /// Block-wise pruning
    Block {
        target_sparsity: f32,
        block_rows: usize,
        block_cols: usize,
    },
}

/// Pruner for creating sparse tensors
pub struct Pruner;

impl Pruner {
    /// Prune dense tensor to sparse
    pub fn prune<R: Runtime>(
        dense: &Tensor<R>,
        strategy: PruningStrategy,
        client: &ComputeClient<...>,
    ) -> SparseTensor<R> {
        match strategy {
            PruningStrategy::MagnitudeGlobal { target_sparsity } => {
                let threshold = compute_magnitude_threshold(dense, target_sparsity, client);
                to_sparse(dense, threshold, SparseFormatId::Csr, client)
            }
            
            PruningStrategy::NMStructured { n, m } => {
                nm_prune(dense, n, m, client)
            }
            
            PruningStrategy::Block { target_sparsity, block_rows, block_cols } => {
                block_prune(dense, target_sparsity, block_rows, block_cols, client)
            }
            
            PruningStrategy::Random { target_sparsity, seed } => {
                random_prune(dense, target_sparsity, seed, client)
            }
            
            _ => todo!(),
        }
    }
    
    /// Gradual pruning (increase sparsity over steps)
    pub fn gradual_prune<R: Runtime>(
        dense: &Tensor<R>,
        initial_sparsity: f32,
        final_sparsity: f32,
        current_step: usize,
        total_steps: usize,
        client: &ComputeClient<...>,
    ) -> SparseTensor<R> {
        // Cubic sparsity schedule (common in literature)
        let t = current_step as f32 / total_steps as f32;
        let current_sparsity = final_sparsity + 
            (initial_sparsity - final_sparsity) * (1.0 - t).powi(3);
        
        Self::prune(dense, PruningStrategy::MagnitudeGlobal { 
            target_sparsity: current_sparsity 
        }, client)
    }
}

/// Compute threshold for target sparsity
fn compute_magnitude_threshold<R: Runtime>(
    tensor: &Tensor<R>,
    target_sparsity: f32,
    client: &ComputeClient<...>,
) -> f32 {
    // Get k-th smallest magnitude where k = target_sparsity * num_elements
    let k = (tensor.shape().num_elements() as f32 * target_sparsity) as usize;
    kth_smallest_magnitude(tensor, k, client)
}
18.2 N:M Pruning
rust/// N:M structured pruning implementation
pub fn nm_prune<R: Runtime>(
    dense: &Tensor<R>,
    n: usize,
    m: usize,
    client: &ComputeClient<...>,
) -> SparseTensor<R> {
    assert!(n <= m, "N must be <= M");
    assert!(dense.shape().dims[1] % m == 0, "Cols must be divisible by M");
    
    // For each group of M elements, keep top N by magnitude
    let (values, indices) = nm_select_topn(dense, n, m, client);
    
    SparseTensor::from_nm_storage(NMStorage {
        values,
        indices,
        meta: NMMetadata {
            rows: dense.shape().dims[0],
            cols: dense.shape().dims[1],
            n,
            m,
            dtype: dense.dtype(),
        },
    })
}

/// N:M kernel: select top-N by magnitude in each group of M
#[cube(launch)]
pub fn nm_select_topn_kernel<F: Float>(
    dense: &Tensor<F>,
    values_out: &mut Array<F>,
    indices_out: &mut Array<u16>,
    #[comptime] rows: u32,
    #[comptime] cols: u32,
    #[comptime] n: u32,
    #[comptime] m: u32,
) {
    let row = ABSOLUTE_POS_X;
    let group = ABSOLUTE_POS_Y;
    let num_groups = cols / m;
    
    if row >= rows || group >= num_groups {
        return;
    }
    
    // Load group of M elements
    let base = row * cols + group * m;
    let mut vals: [F; 8] = [F::new(0.0); 8];  // Assume M <= 8
    let mut mags: [F; 8] = [F::new(0.0); 8];
    
    for i in 0..m {
        vals[i as usize] = dense[base + i];
        mags[i as usize] = abs(vals[i as usize]);
    }
    
    // Find top-N indices (simple selection for small N)
    let mut selected: [u32; 4] = [0; 4];  // Assume N <= 4
    for k in 0..n {
        let mut max_idx = 0u32;
        let mut max_val = F::new(-1.0);
        for i in 0..m {
            if mags[i as usize] > max_val && !contains(&selected[..k as usize], i) {
                max_val = mags[i as usize];
                max_idx = i;
            }
        }
        selected[k as usize] = max_idx;
    }
    
    // Sort selected indices for deterministic output
    sort_indices(&mut selected[..n as usize]);
    
    // Write output values (N per group)
    let out_base = (row * num_groups + group) * n;
    for k in 0..n {
        values_out[out_base + k] = vals[selected[k as usize] as usize];
    }
    
    // Encode indices as bitmask (for 2:4 case)
    let mut mask = 0u16;
    for k in 0..n {
        mask |= 1u16 << selected[k as usize];
    }
    indices_out[row * num_groups + group] = mask;
}
18.3 Straight-Through Estimator (STE)
rust/// Straight-Through Estimator for sparse training
pub struct StraightThroughEstimator;

impl StraightThroughEstimator {
    /// Forward: apply mask. Backward: pass gradient through unchanged.
    pub fn apply<R: Runtime>(
        dense: &Tensor<R>,
        mask: &SparsityPattern<R>,
        client: &ComputeClient<...>,
    ) -> SparseTensor<R> {
        // Forward: zero out non-mask positions
        let values = gather_at_pattern(dense, mask, client);
        SparseTensor::from_pattern_and_values(mask.clone(), values)
    }
    
    /// Backward: expand sparse gradient to dense (implicit in autograd)
    /// dL/d_dense = scatter(dL/d_sparse, pattern)
    /// 
    /// This is handled automatically by the autodiff system.
    /// The key insight: backward is just scatter, which preserves
    /// gradient flow to all original positions.
}

/// Module wrapper for STE in training
pub struct STEMaskedLinear<R: Runtime> {
    /// Dense weights (what we actually optimize)
    pub weight_dense: Tensor<R>,
    /// Current sparsity mask
    pub mask: PatternManager<R>,
    /// Whether to update mask during training
    pub update_mask: bool,
}

impl<R: Runtime> STEMaskedLinear<R> {
    pub fn forward(&self, x: &Tensor<R>, client: &ComputeClient<...>) -> Tensor<R> {
        // Apply mask via STE
        let w_sparse = StraightThroughEstimator::apply(
            &self.weight_dense,
            self.mask.current_pattern(),
            client,
        );
        
        // SpMM
        sparse_matmul(&w_sparse, x, client)
    }
    
    pub fn update_mask(&mut self, gradients: &Tensor<R>, client: &ComputeClient<...>) {
        if self.update_mask {
            // RigL-style mask update
            self.mask.rigl_update(
                &self.weight_dense.as_handle(),
                &gradients.as_handle(),
                0.1,  // drop 10%
                client,
            ).unwrap();
        }
    }
}

19. Sparse Modules (burn-nn)
Pre-built neural network modules with sparse weights.
19.1 SparseLinear
rust/// Sparse linear layer
#[derive(Module, Debug)]
pub struct SparseLinear<B: SparseBackend> {
    /// Sparse weight matrix [out_features × in_features]
    pub weight: SparseParam<B>,
    /// Optional dense bias [out_features]
    pub bias: Option<Param<B>>,
}

#[derive(Config, Debug)]
pub struct SparseLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub sparsity: f32,
    pub sparse_format: SparseFormatId,
}

impl SparseLinearConfig {
    pub fn init<B: SparseBackend>(&self, device: &B::Device) -> SparseLinear<B> {
        // Initialize dense, then prune
        let weight_dense = Tensor::random(
            [self.out_features, self.in_features],
            Distribution::Uniform(-1.0, 1.0),
            device,
        );
        
        let weight_sparse = Pruner::prune(
            &weight_dense,
            PruningStrategy::MagnitudeGlobal { target_sparsity: self.sparsity },
            B::client(),
        );
        
        let bias = if self.bias {
            Some(Param::from(Tensor::zeros([self.out_features], device)))
        } else {
            None
        };
        
        SparseLinear {
            weight: SparseParam::from(weight_sparse),
            bias,
        }
    }
}

impl<B: SparseBackend> SparseLinear<B> {
    /// Forward pass: y = xW^T + b
    pub fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        // x: [batch, in_features]
        // weight: [out_features, in_features] sparse
        // y: [batch, out_features]
        
        // Transpose weight view, then SpMM
        let w_t = self.weight.val().transpose();  // View, not copy
        let mut y = sparse_matmul(&w_t, &x.transpose(), B::client()).transpose();
        
        if let Some(ref bias) = self.bias {
            y = y + bias.val().unsqueeze(0);
        }
        
        y
    }
}
19.2 SparseEmbedding
rust/// Sparse embedding layer (for sparse embedding tables)
#[derive(Module, Debug)]
pub struct SparseEmbedding<B: SparseBackend> {
    /// Embedding table [num_embeddings × embedding_dim]
    /// Stored as CSR where rows are embeddings
    pub weight: SparseParam<B>,
    /// Number of embeddings
    pub num_embeddings: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

#[derive(Config, Debug)]
pub struct SparseEmbeddingConfig {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub sparsity: f32,
}

impl SparseEmbeddingConfig {
    pub fn init<B: SparseBackend>(&self, device: &B::Device) -> SparseEmbedding<B> {
        let weight_dense = Tensor::random(
            [self.num_embeddings, self.embedding_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        
        let weight_sparse = Pruner::prune(
            &weight_dense,
            PruningStrategy::MagnitudeGlobal { target_sparsity: self.sparsity },
            B::client(),
        );
        
        SparseEmbedding {
            weight: SparseParam::from(weight_sparse),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
        }
    }
}

impl<B: SparseBackend> SparseEmbedding<B> {
    /// Lookup embeddings by indices
    pub fn forward(&self, indices: Tensor<B, Int>) -> Tensor<B> {
        // indices: [batch, seq_len] or [batch]
        // output: [batch, seq_len, embedding_dim] or [batch, embedding_dim]
        
        // For sparse embedding, this is row slicing + gather
        sparse_embedding_lookup(&self.weight.val(), &indices, B::client())
    }
}

/// Optimized embedding lookup from sparse table
fn sparse_embedding_lookup<R: Runtime>(
    table: &SparseTensor<R>,
    indices: &Tensor<R, Int>,
    client: &ComputeClient<...>,
) -> Tensor<R> {
    // Use row slicing views for efficiency
    let flat_indices = indices.flatten();
    let num_lookups = flat_indices.shape().num_elements();
    
    // Batch gather rows from sparse table
    sparse_gather_rows(table, &flat_indices, client)
        .reshape(indices.shape().dims.iter().chain(&[table.shape().dims[1]]).copied())
}
19.3 Sparse Attention (Future)
rust/// Sparse attention pattern (for future sparse attention implementation)
#[derive(Clone, Debug)]
pub enum SparseAttentionPattern {
    /// Local window attention
    Local { window_size: usize },
    
    /// Strided attention (every k-th position)
    Strided { stride: usize },
    
    /// Block sparse attention
    BlockSparse { block_size: usize, num_blocks: usize },
    
    /// BigBird-style (local + global + random)
    BigBird {
        local_window: usize,
        num_global_tokens: usize,
        num_random_blocks: usize,
    },
    
    /// Custom pattern from mask
    Custom { pattern: SparsityPattern },
}

// Implementation deferred to future work

20. Serialization
Save and load sparse tensors.
20.1 Serialization Format
rust/// Sparse tensor serialization header
#[derive(Serialize, Deserialize)]
pub struct SparseTensorHeader {
    /// Magic bytes for validation
    pub magic: [u8; 4],  // "SPRS"
    /// Format version
    pub version: u16,
    /// Sparse format
    pub format: SparseFormatId,
    /// Data type
    pub dtype: DType,
    /// Index type
    pub index_dtype: IndexDType,
    /// Logical shape
    pub shape: Vec<usize>,
    /// Storage metadata
    pub storage_meta: StorageMetadataSer,
    /// Checksum of data
    pub checksum: u64,
}

#[derive(Serialize, Deserialize)]
pub struct StorageMetadataSer {
    pub nnz: usize,
    pub block_size: Option<(usize, usize)>,
    pub nm_params: Option<(usize, usize)>,
}

/// File layout:
/// [Header (JSON)] [row_ptrs bytes] [col_indices bytes] [values bytes]
/// 
/// Header is JSON for human readability and debugging.
/// Data sections are raw bytes for efficiency.
20.2 Save/Load API
rust/// Sparse tensor serialization
pub struct SparseTensorIO;

impl SparseTensorIO {
    /// Save sparse tensor to file
    pub fn save<R: Runtime>(
        tensor: &SparseTensor<R>,
        path: impl AsRef<Path>,
        client: &ComputeClient<...>,
    ) -> Result<(), SparseIOError> {
        let path = path.as_ref();
        let mut file = File::create(path)?;
        
        // Create header
        let header = SparseTensorHeader {
            magic: *b"SPRS",
            version: 1,
            format: tensor.format(),
            dtype: tensor.dtype(),
            index_dtype: IndexDType::U32,  // TODO: make configurable
            shape: tensor.shape().dims.clone(),
            storage_meta: tensor.storage_metadata().into(),
            checksum: 0,  // Compute after writing data
        };
        
        // Write header as JSON with length prefix
        let header_json = serde_json::to_vec(&header)?;
        file.write_all(&(header_json.len() as u32).to_le_bytes())?;
        file.write_all(&header_json)?;
        
        // Download buffers from GPU and write
        let buffers = tensor.download_buffers(client)?;
        for buffer in buffers {
            file.write_all(&buffer)?;
        }
        
        Ok(())
    }
    
    /// Load sparse tensor from file
    pub fn load<R: Runtime>(
        path: impl AsRef<Path>,
        device: &R::Device,
        client: &ComputeClient<...>,
    ) -> Result<SparseTensor<R>, SparseIOError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        
        // Read header
        let mut header_len_bytes = [0u8; 4];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;
        
        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json)?;
        let header: SparseTensorHeader = serde_json::from_slice(&header_json)?;
        
        // Validate
        if header.magic != *b"SPRS" {
            return Err(SparseIOError::InvalidMagic);
        }
        if header.version > 1 {
            return Err(SparseIOError::UnsupportedVersion(header.version));
        }
        
        // Read data buffers based on format
        let tensor = match header.format {
            SparseFormatId::Csr => {
                let row_ptrs = read_buffer::<u32>(&mut file, header.shape[0] + 1)?;
                let col_indices = read_buffer::<u32>(&mut file, header.storage_meta.nnz)?;
                let values = read_buffer_dtype(&mut file, header.dtype, header.storage_meta.nnz)?;
                
                SparseTensor::from_csr_buffers(
                    row_ptrs,
                    col_indices,
                    values,
                    header.shape,
                    device,
                    client,
                )
            }
            // ... other formats
            _ => todo!(),
        };
        
        Ok(tensor)
    }
    
    /// Save in safetensors-compatible format
    pub fn save_safetensors<R: Runtime>(
        tensors: &HashMap<String, SparseTensor<R>>,
        path: impl AsRef<Path>,
        client: &ComputeClient<...>,
    ) -> Result<(), SparseIOError> {
        // Safetensors format with sparse metadata in header
        todo!()
    }
}
20.3 Checkpoint Integration
rust/// Checkpoint sparse model state
pub struct SparseCheckpoint;

impl SparseCheckpoint {
    /// Save model with sparse parameters
    pub fn save<R: Runtime>(
        record: &impl Record,
        path: impl AsRef<Path>,
        client: &ComputeClient<...>,
    ) -> Result<(), SparseIOError> {
        // Iterate through record, save sparse tensors separately
        // Dense tensors use standard Burn serialization
        todo!()
    }
    
    /// Load model with sparse parameters
    pub fn load<R: Runtime>(
        path: impl AsRef<Path>,
        device: &R::Device,
        client: &ComputeClient<...>,
    ) -> Result<impl Record, SparseIOError> {
        todo!()
    }
}

21. Device Placement
Moving sparse tensors between devices.
21.1 Device Operations
rustimpl<R: Runtime> SparseTensor<R> {
    /// Move tensor to device
    pub fn to_device(self, device: &R::Device, client: &ComputeClient<...>) -> Self {
        if self.device() == device {
            return self;
        }
        
        // Allocate new buffers on target device
        let new_handle = self.handle.clone_to_device(device, client);
        
        Self {
            handle: Arc::new(new_handle),
            ..self
        }
    }
    
    /// Get current device
    pub fn device(&self) -> &R::Device {
        &self.handle.device
    }
    
    /// Check if tensor is on given device
    pub fn is_on_device(&self, device: &R::Device) -> bool {
        self.handle.device == *device
    }
}

impl<R: Runtime> SparseTensorHandle<R> {
    /// Clone handle to different device
    pub fn clone_to_device(&self, device: &R::Device, client: &ComputeClient<...>) -> Self {
        // Allocate on target device
        let new_buffers = self.buffers.clone_to_device(device, client);
        
        Self {
            buffers: new_buffers,
            storage_meta: self.storage_meta.clone(),
            pool: client.memory_pool(device),
            device: device.clone(),
        }
    }
}
21.2 Multi-Device Operations
rust/// Multi-device sparse operations
pub struct MultiDeviceSparse;

impl MultiDeviceSparse {
    /// Scatter sparse tensor across devices (row-wise partitioning)
    pub fn scatter_rows<R: Runtime>(
        tensor: &SparseTensor<R>,
        devices: &[R::Device],
        client: &ComputeClient<...>,
    ) -> Vec<SparseTensor<R>> {
        let num_devices = devices.len();
        let rows_per_device = tensor.shape().dims[0] / num_devices;
        
        devices.iter().enumerate().map(|(i, device)| {
            let start_row = i * rows_per_device;
            let end_row = if i == num_devices - 1 {
                tensor.shape().dims[0]
            } else {
                (i + 1) * rows_per_device
            };
            
            let slice = SparseTensorView::row_slice(tensor.clone(), start_row, end_row)
                .unwrap()
                .materialize(client);
            
            slice.to_device(device, client)
        }).collect()
    }
    
    /// Gather sparse tensors from multiple devices
    pub fn gather<R: Runtime>(
        tensors: Vec<SparseTensor<R>>,
        target_device: &R::Device,
        client: &ComputeClient<...>,
    ) -> SparseTensor<R> {
        // Move all to target device
        let on_device: Vec<_> = tensors.into_iter()
            .map(|t| t.to_device(target_device, client))
            .collect();
        
        // Concatenate
        SparseTensor::concat_rows(on_device, client)
    }
}

22. Error Model
Comprehensive error types for sparse operations.
22.1 Error Taxonomy
rust/// Sparse operation errors
#[derive(Debug, thiserror::Error)]
pub enum SparseError {
    // Shape errors
    #[error("Shape mismatch in {op}: expected {expected}, got {got}")]
    ShapeMismatch {
        op: &'static str,
        expected: String,
        got: usize,
    },
    
    #[error("Dimension out of bounds: {dim} >= {max}")]
    DimensionOutOfBounds {
        dim: usize,
        max: usize,
    },
    
    // Pattern errors
    #[error("Sparsity pattern mismatch: expected {expected_nnz} non-zeros, got {got_nnz}")]
    PatternMismatch {
        expected_nnz: usize,
        got_nnz: usize,
    },
    
    #[error("Pattern modification not allowed in current mode")]
    PatternModificationNotAllowed,
    
    #[error("Pattern growth not allowed in ShrinkOnly mode")]
    PatternGrowthNotAllowed,
    
    #[error("Pattern shrink not allowed in GrowOnly mode")]
    PatternShrinkNotAllowed,
    
    // Format errors
    #[error("Format mismatch: cannot perform operation between {format_a:?} and {format_b:?}")]
    FormatMismatch {
        format_a: SparseFormatId,
        format_b: SparseFormatId,
    },
    
    #[error("Unsupported format {format:?} for operation {op}")]
    UnsupportedFormat {
        format: SparseFormatId,
        op: &'static str,
    },
    
    #[error("Format conversion from {from:?} to {to:?} is lossy and requires explicit conversion")]
    LossyConversionRequired {
        from: SparseFormatId,
        to: SparseFormatId,
    },
    
    // Block/N:M constraint errors
    #[error("Invalid block size: ({block_rows}, {block_cols}) does not divide ({rows}, {cols})")]
    InvalidBlockSize {
        block_rows: usize,
        block_cols: usize,
        rows: usize,
        cols: usize,
    },
    
    #[error("Invalid N:M constraint: N={n}, M={m}, but cols={cols} is not divisible by M")]
    InvalidNMConstraint {
        n: usize,
        m: usize,
        cols: usize,
    },
    
    #[error("Slice must align to block boundary (required alignment: {required})")]
    SliceAlignmentRequired {
        required: usize,
    },
    
    // Device errors
    #[error("Device {device:?} does not support format {format:?}")]
    DeviceFormatUnsupported {
        device: String,
        format: SparseFormatId,
    },
    
    #[error("Tensors on different devices: {device_a:?} vs {device_b:?}")]
    DeviceMismatch {
        device_a: String,
        device_b: String,
    },
    
    // Memory errors
    #[error("Out of memory: requested {requested} bytes, available {available}")]
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    
    #[error("Memory fragmentation: cannot allocate contiguous block of {size} bytes")]
    MemoryFragmentation {
        size: usize,
    },
    
    // DType errors
    #[error("Unsupported dtype {dtype:?} for format {format:?}")]
    UnsupportedDType {
        dtype: DType,
        format: SparseFormatId,
    },
    
    #[error("DType mismatch: {dtype_a:?} vs {dtype_b:?}")]
    DTypeMismatch {
        dtype_a: DType,
        dtype_b: DType,
    },
    
    // View errors
    #[error("Slice out of bounds: [{start}, {end}) exceeds dimension {dim}")]
    SliceOutOfBounds {
        start: usize,
        end: usize,
        dim: usize,
    },
    
    #[error("View not supported for format {format:?}")]
    ViewNotSupported {
        format: SparseFormatId,
    },
    
    // IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Invalid sparse tensor file: bad magic number")]
    InvalidMagic,
    
    #[error("Unsupported file version: {0}")]
    UnsupportedVersion(u16),
}

/// Result type for sparse operations
pub type SparseResult<T> = Result<T, SparseError>;
22.2 Error Recovery
rust/// Error handling strategies
pub enum ErrorRecovery {
    /// Fail immediately
    Fail,
    
    /// Fall back to dense computation
    FallbackDense,
    
    /// Convert to compatible format and retry
    ConvertAndRetry,
    
    /// Warn and continue with best effort
    WarnAndContinue,
}

/// Context for operation with recovery
pub struct SparseOpContext {
    pub recovery: ErrorRecovery,
    pub warnings: Vec<String>,
}

impl SparseOpContext {
    /// Handle error with configured recovery
    pub fn handle_error<T, R: Runtime>(
        &mut self,
        error: SparseError,
        fallback: impl FnOnce() -> SparseResult<T>,
        client: &ComputeClient<...>,
    ) -> SparseResult<T> {
        match self.recovery {
            ErrorRecovery::Fail => Err(error),
            ErrorRecovery::FallbackDense => {
                self.warnings.push(format!("Falling back to dense: {}", error));
                fallback()
            }
            ErrorRecovery::ConvertAndRetry => {
                self.warnings.push(format!("Converting format: {}", error));
                fallback()
            }
            ErrorRecovery::WarnAndContinue => {
                self.warnings.push(format!("Warning: {}", error));
                fallback()
            }
        }
    }
}

23. Fusion Rules (Formalized)
Explicit rules for when sparse operations can fuse.
23.1 Fusion Rule Table
Fusion Compatibility Matrix:
┌─────────────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Op A → Op B     │ SpMM   │ SpMV   │ ElemWs │ Reduce │ Trans  │ Conv   │
├─────────────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ SpMM            │ ✗ red  │ ✗ red  │ ✓ epi  │ ✓ epi  │ ✗      │ N/A    │
│ SpMV            │ ✗ red  │ ✗ red  │ ✓ epi  │ ✓ epi  │ ✗      │ N/A    │
│ Elementwise     │ ✗      │ ✗      │ ✓ pat  │ ✓      │ ✗      │ N/A    │
│ Reduce          │ ✗      │ ✗      │ ✗ dim  │ ✗      │ ✗      │ N/A    │
│ Transpose       │ ✓ view │ ✓ view │ ✗      │ ✗      │ ✗      │ N/A    │
│ Format Convert  │ ✗      │ ✗      │ ✗      │ ✗      │ ✗      │ N/A    │
└─────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Legend:
  ✓      = can fuse
  ✓ epi  = can fuse as epilogue (element-wise on output)
  ✓ pat  = can fuse if same sparsity pattern
  ✓ view = can fuse as view (no computation)
  ✗      = cannot fuse
  ✗ red  = cannot fuse (reduction boundary)
  ✗ dim  = cannot fuse (dimension change)
  N/A    = not applicable
23.2 Fusion Decision Engine
rust/// Fusion decision engine
pub struct FusionEngine;

impl FusionEngine {
    /// Check if two operations can fuse
    pub fn can_fuse<R: Runtime>(
        op_a: &dyn SparseOperation<R>,
        op_b: &dyn SparseOperation<R>,
    ) -> FusionDecision {
        let pattern_a = op_a.memory_access_pattern();
        let pattern_b = op_b.memory_access_pattern();
        
        // Rule 1: Reduction boundaries break fusion
        if matches!(pattern_a.output, AccessPattern::Reduction { .. }) {
            return FusionDecision::No(FusionRejection::ReductionBoundary);
        }
        
        // Rule 2: Element-wise can fuse if same pattern
        if op_b.is_elementwise() {
            match &pattern_a.output_pattern_source {
                Some(idx) => {
                    // Output has known sparsity pattern
                    return FusionDecision::Yes(FusionStrategy::Epilogue);
                }
                None => {
                    // Output is dense, can always fuse element-wise
                    return FusionDecision::Yes(FusionStrategy::Epilogue);
                }
            }
        }
        
        // Rule 3: Format conversions break fusion
        if op_a.is_format_conversion() || op_b.is_format_conversion() {
            return FusionDecision::No(FusionRejection::FormatConversion);
        }
        
        // Rule 4: Different sparsity patterns cannot fuse element-wise
        if let (Some(pat_a), Some(pat_b)) = (
            pattern_a.output_pattern_source,
            pattern_b.inputs.first().and_then(|p| match p {
                AccessPattern::SparseIndexed { .. } => Some(0),
                _ => None,
            }),
        ) {
            if !patterns_match(&pattern_a, &pattern_b) {
                return FusionDecision::No(FusionRejection::PatternMismatch);
            }
        }
        
        // Default: don't fuse (conservative)
        FusionDecision::No(FusionRejection::UnsupportedOp)
    }
    
    /// Analyze operation graph and find fusion groups
    pub fn analyze_graph<R: Runtime>(
        graph: &SparseOpGraph<R>,
    ) -> FusionAnalysis {
        let mut fusion_groups = Vec::new();
        let mut current_group = FusionGroup::new();
        
        for (i, op) in graph.ops.iter().enumerate() {
            if current_group.is_empty() {
                current_group.add(i);
                continue;
            }
            
            let last_op = &graph.ops[*current_group.ops.last().unwrap()];
            match Self::can_fuse(last_op, op) {
                FusionDecision::Yes(strategy) => {
                    current_group.add_with_strategy(i, strategy);
                }
                FusionDecision::No(reason) => {
                    // Close current group, start new one
                    if current_group.len() > 1 {
                        fusion_groups.push(current_group);
                    }
                    current_group = FusionGroup::new();
                    current_group.add(i);
                }
            }
        }
        
        // Don't forget last group
        if current_group.len() > 1 {
            fusion_groups.push(current_group);
        }
        
        FusionAnalysis { fusion_groups }
    }
}

#[derive(Clone, Debug)]
pub enum FusionDecision {
    Yes(FusionStrategy),
    No(FusionRejection),
}

#[derive(Clone, Debug)]
pub enum FusionStrategy {
    /// Fuse as epilogue (element-wise on output)
    Epilogue,
    /// Fuse as view (structural transform)
    View,
    /// Full kernel fusion
    Full,
}

#[derive(Clone, Debug)]
pub enum FusionRejection {
    ReductionBoundary,
    FormatConversion,
    PatternMismatch,
    AccessConflict,
    UnsupportedOp,
    RegisterPressure,
}

24. JIT Specialization Signatures
Kernel caching and specialization.
24.1 Kernel Signature
rust/// Complete kernel specialization signature
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct KernelSignature {
    /// Operation type
    pub op_type: SparseOpType,
    
    /// Input format(s)
    pub input_formats: Vec<SparseFormatId>,
    
    /// Output format
    pub output_format: Option<SparseFormatId>,
    
    /// Data types
    pub input_dtypes: Vec<DType>,
    pub output_dtype: DType,
    pub accumulate_dtype: DType,
    
    /// Algorithm variant
    pub algorithm: AlgorithmId,
    
    /// Shape bucket (quantized for caching)
    pub shape_bucket: ShapeBucket,
    
    /// Sparsity bucket
    pub sparsity_bucket: SparsityBucket,
    
    /// Row distribution bucket (for CSR)
    pub row_distribution: Option<RowDistributionBucket>,
    
    /// Tile sizes (for blocked algorithms)
    pub tile_config: Option<TileConfig>,
    
    /// Device compute capability
    pub compute_capability: ComputeCapability,
    
    /// Fusion epilogue (if any)
    pub epilogue: Option<EpilogueType>,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ShapeBucket {
    Tiny,      // < 256
    Small,     // 256 - 2K
    Medium,    // 2K - 16K
    Large,     // 16K - 128K
    Huge,      // > 128K
}

impl ShapeBucket {
    pub fn from_dim(dim: usize) -> Self {
        match dim {
            d if d < 256 => Self::Tiny,
            d if d < 2048 => Self::Small,
            d if d < 16384 => Self::Medium,
            d if d < 131072 => Self::Large,
            _ => Self::Huge,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum SparsityBucket {
    Low,      // < 50%
    Medium,   // 50-80%
    High,     // 80-95%
    VeryHigh, // 95-99%
    Extreme,  // > 99%
}

impl SparsityBucket {
    pub fn from_ratio(sparsity: f32) -> Self {
        match sparsity {
            s if s < 0.5 => Self::Low,
            s if s < 0.8 => Self::Medium,
            s if s < 0.95 => Self::High,
            s if s < 0.99 => Self::VeryHigh,
            _ => Self::Extreme,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum RowDistributionBucket {
    Uniform,     // CV < 0.5
    Moderate,    // 0.5 <= CV < 1.0
    Skewed,      // 1.0 <= CV < 2.0
    PowerLaw,    // CV >= 2.0
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct TileConfig {
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum EpilogueType {
    None,
    Relu,
    Gelu,
    BiasAdd,
    BiasAddRelu,
    Custom(u64),  // Hash of custom epilogue
}
24.2 Kernel Cache
rust/// JIT kernel cache
pub struct KernelCache<R: Runtime> {
    /// Compiled kernels by signature
    cache: DashMap<KernelSignature, Arc<CompiledKernel<R>>>,
    
    /// LRU eviction tracking
    lru: Mutex<LruTracker>,
    
    /// Maximum cache size (bytes)
    max_size: usize,
    
    /// Current cache size (bytes)
    current_size: AtomicUsize,
}

impl<R: Runtime> KernelCache<R> {
    /// Get or compile kernel for signature
    pub fn get_or_compile(
        &self,
        signature: KernelSignature,
        compile_fn: impl FnOnce(&KernelSignature) -> CompiledKernel<R>,
    ) -> Arc<CompiledKernel<R>> {
        // Fast path: check cache
        if let Some(kernel) = self.cache.get(&signature) {
            self.lru.lock().unwrap().touch(&signature);
            return kernel.clone();
        }
        
        // Slow path: compile
        let kernel = Arc::new(compile_fn(&signature));
        let kernel_size = kernel.size_bytes();
        
        // Evict if needed
        while self.current_size.load(Ordering::Relaxed) + kernel_size > self.max_size {
            if let Some(victim) = self.lru.lock().unwrap().evict() {
                if let Some((_, removed)) = self.cache.remove(&victim) {
                    self.current_size.fetch_sub(removed.size_bytes(), Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
        
        // Insert
        self.cache.insert(signature.clone(), kernel.clone());
        self.current_size.fetch_add(kernel_size, Ordering::Relaxed);
        self.lru.lock().unwrap().insert(signature);
        
        kernel
    }
    
    /// Precompile kernels for common signatures
    pub fn warmup(&self, signatures: &[KernelSignature], client: &ComputeClient<...>) {
        for sig in signatures {
            self.get_or_compile(sig.clone(), |s| compile_sparse_kernel(s, client));
        }
    }
    
    /// Export cache for serialization
    pub fn export(&self) -> CacheExport {
        CacheExport {
            signatures: self.cache.iter().map(|e| e.key().clone()).collect(),
        }
    }
    
    /// Import and precompile
    pub fn import(&self, export: CacheExport, client: &ComputeClient<...>) {
        self.warmup(&export.signatures, client);
    }
}

25. Inspection & Debug Tools
Tools for debugging and analyzing sparse tensors.
25.1 Statistics & Analysis
rust/// Sparse tensor inspector
pub struct SparseInspector;

impl SparseInspector {
    /// Compute detailed statistics
    pub fn statistics<R: Runtime>(
        tensor: &SparseTensor<R>,
        client: &ComputeClient<...>,
    ) -> SparseStatistics {
        let storage = tensor.storage_metadata();
        
        SparseStatistics {
            // Shape info
            logical_shape: tensor.shape().clone(),
            storage_shape: storage.clone(),
            
            // Sparsity
            nnz: storage.values_count(),
            total_elements: tensor.shape().num_elements(),
            sparsity: tensor.sparsity(),
            
            // Memory
            values_bytes: storage.values_count() * tensor.dtype().bytes(),
            indices_bytes: storage.index_bytes(IndexDType::U32),
            total_bytes: tensor.memory_bytes(),
            dense_equivalent_bytes: tensor.shape().num_elements() * tensor.dtype().bytes(),
            compression_ratio: tensor.shape().num_elements() as f32 * tensor.dtype().bytes() as f32
                / tensor.memory_bytes() as f32,
            
            // Distribution (for CSR)
            row_stats: compute_row_stats(tensor, client),
            
            // Value stats
            value_stats: compute_value_stats(tensor, client),
        }
    }
    
    /// Print human-readable summary
    pub fn summary<R: Runtime>(tensor: &SparseTensor<R>, client: &ComputeClient<...>) -> String {
        let stats = Self::statistics(tensor, client);
        
        format!(
            "SparseTensor [{:?}] {:?}\n\
             Format: {:?}\n\
             Sparsity: {:.2}% ({} / {} elements)\n\
             Memory: {:.2} MB (vs {:.2} MB dense, {:.1}x compression)\n\
             Row nnz: min={}, max={}, mean={:.1}, std={:.1}",
            stats.logical_shape.dims,
            tensor.dtype(),
            tensor.format(),
            stats.sparsity * 100.0,
            stats.nnz,
            stats.total_elements,
            stats.total_bytes as f64 / 1e6,
            stats.dense_equivalent_bytes as f64 / 1e6,
            stats.compression_ratio,
            stats.row_stats.min,
            stats.row_stats.max,
            stats.row_stats.mean,
            stats.row_stats.std,
        )
    }
    
    /// Visualize sparsity pattern (ASCII art for small tensors)
    pub fn visualize_pattern<R: Runtime>(
        tensor: &SparseTensor<R>,
        max_rows: usize,
        max_cols: usize,
        client: &ComputeClient<...>,
    ) -> String {
        let (rows, cols) = (tensor.shape().dims[0], tensor.shape().dims[1]);
        
        // Sample rows/cols if too large
        let row_step = (rows + max_rows - 1) / max_rows;
        let col_step = (cols + max_cols - 1) / max_cols;
        
        let pattern = extract_pattern_sample(tensor, row_step, col_step, client);
        
        let mut output = String::new();
        for row in &pattern {
            for &has_value in row {
                output.push(if has_value { '█' } else { '·' });
            }
            output.push('\n');
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct SparseStatistics {
    pub logical_shape: Shape,
    pub storage_shape: StorageShape,
    pub nnz: usize,
    pub total_elements: usize,
    pub sparsity: f32,
    pub values_bytes: usize,
    pub indices_bytes: usize,
    pub total_bytes: usize,
    pub dense_equivalent_bytes: usize,
    pub compression_ratio: f32,
    pub row_stats: DistributionStats,
    pub value_stats: ValueStats,
}

#[derive(Clone, Debug)]
pub struct DistributionStats {
    pub min: usize,
    pub max: usize,
    pub mean: f32,
    pub std: f32,
    pub histogram: Vec<(usize, usize)>,  // (bucket_start, count)
}

#[derive(Clone, Debug)]
pub struct ValueStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub num_zeros: usize,  // Explicit zeros in values
    pub num_subnormal: usize,
}
25.2 Validation
rust/// Sparse tensor validator
pub struct SparseValidator;

impl SparseValidator {
    /// Validate sparse tensor invariants
    pub fn validate<R: Runtime>(
        tensor: &SparseTensor<R>,
        client: &ComputeClient<...>,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check index bounds
        if let Err(e) = Self::check_index_bounds(tensor, client) {
            errors.push(e);
        }
        
        // Check sorted (for CSR/CSC)
        if let Err(e) = Self::check_sorted(tensor, client) {
            errors.push(e);
        }
        
        // Check for duplicate indices
        if let Some(w) = Self::check_duplicates(tensor, client) {
            warnings.push(w);
        }
        
        // Check for explicit zeros
        if let Some(w) = Self::check_explicit_zeros(tensor, client) {
            warnings.push(w);
        }
        
        // Check N:M constraint
        if let Err(e) = Self::check_nm_constraint(tensor, client) {
            errors.push(e);
        }
        
        ValidationResult { errors, warnings }
    }
    
    fn check_index_bounds<R: Runtime>(
        tensor: &SparseTensor<R>,
        client: &ComputeClient<...>,
    ) -> Result<(), ValidationError> {
        let (rows, cols) = (tensor.shape().dims[0], tensor.shape().dims[1]);
        
        match tensor.format() {
            SparseFormatId::Csr => {
                let max_col = find_max_col_index(tensor, client);
                if max_col >= cols {
                    return Err(ValidationError::IndexOutOfBounds {
                        dim: "column",
                        max_valid: cols - 1,
                        found: max_col,
                    });
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    // ... other validation methods
}

#[derive(Debug)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[derive(Debug)]
pub enum ValidationError {
    IndexOutOfBounds { dim: &'static str, max_valid: usize, found: usize },
    UnsortedIndices { row: usize },
    InvalidRowPointers,
    NMConstraintViolation { group: usize, expected_n: usize, found: usize },
}

#[derive(Debug)]
pub enum ValidationWarning {
    DuplicateIndices { count: usize },
    ExplicitZeros { count: usize },
    HighlySkewedDistribution { cv: f32 },
}
25.3 Comparison & Testing
rust/// Compare sparse tensors for testing
pub struct SparseComparator;

impl SparseComparator {
    /// Check if two sparse tensors are approximately equal
    pub fn allclose<R: Runtime>(
        a: &SparseTensor<R>,
        b: &SparseTensor<R>,
        rtol: f32,
        atol: f32,
        client: &ComputeClient<...>,
    ) -> ComparisonResult {
        // Check shape
        if a.shape() != b.shape() {
            return ComparisonResult::ShapeMismatch {
                a: a.shape().clone(),
                b: b.shape().clone(),
            };
        }
        
        // Convert both to dense for value comparison
        let a_dense = a.to_dense(client);
        let b_dense = b.to_dense(client);
        
        let diff = compute_diff(&a_dense, &b_dense, client);
        
        if diff.max_abs_diff <= atol && diff.max_rel_diff <= rtol {
            ComparisonResult::Equal
        } else {
            ComparisonResult::ValueMismatch {
                max_abs_diff: diff.max_abs_diff,
                max_rel_diff: diff.max_rel_diff,
                num_different: diff.num_different,
                first_diff_idx: diff.first_diff_idx,
            }
        }
    }
    
    /// Check if sparsity patterns match
    pub fn pattern_equals<R: Runtime>(
        a: &SparseTensor<R>,
        b: &SparseTensor<R>,
        client: &ComputeClient<...>,
    ) -> bool {
        if a.storage_metadata().nnz() != b.storage_metadata().nnz() {
            return false;
        }
        
        // Compare index buffers
        compare_indices(a, b, client)
    }
}

#[derive(Debug)]
pub enum ComparisonResult {
    Equal,
    ShapeMismatch { a: Shape, b: Shape },
    ValueMismatch {
        max_abs_diff: f32,
        max_rel_diff: f32,
        num_different: usize,
        first_diff_idx: Option<Vec<usize>>,
    },
}

26. Extended File Structure
Additional modules for Part 2 features.
26.1 Updated cubecl-sparse Structure
cubecl-sparse/src/
├── ... (from Part 1)
│
├── handle/                       # Handle system (Section 11)
│   ├── mod.rs
│   ├── tensor.rs                 # SparseTensor
│   ├── handle.rs                 # SparseTensorHandle
│   └── handle_ref.rs             # SparseTensorHandleRef
│
├── view/                         # View system (Section 12)
│   ├── mod.rs
│   ├── types.rs                  # SparseView enum
│   ├── tensor_view.rs            # SparseTensorView
│   └── ops.rs                    # View-aware operations
│
├── batch/                        # Batched operations (Section 13)
│   ├── mod.rs
│   ├── storage.rs                # BatchedSparseStorage
│   ├── tensor.rs                 # BatchedSparseTensor
│   └── spmm.rs                   # BatchedCsrSpmm
│
├── dtype/                        # DType handling (Section 14)
│   ├── mod.rs
│   ├── compat.rs                 # Compatibility matrix
│   ├── mixed_precision.rs        # MixedPrecisionConfig
│   └── quantize.rs               # Quantization support
│
├── pattern/                      # Dynamic pattern engine (Section 16)
│   ├── mod.rs
│   ├── manager.rs                # PatternManager
│   ├── delta.rs                  # PatternDelta
│   ├── coo_accum.rs              # CooGradientAccumulator
│   └── rigl.rs                   # RigL-specific utilities
│
├── interop/                      # Sparse-dense interop (Section 17)
│   ├── mod.rs
│   ├── mixed_ops.rs              # Mixed operations
│   ├── broadcast.rs              # Broadcasting rules
│   └── convert.rs                # Conversion utilities
│
├── prune/                        # Pruning API (Section 18)
│   ├── mod.rs
│   ├── strategies.rs             # PruningStrategy enum
│   ├── magnitude.rs              # Magnitude pruning
│   ├── nm.rs                     # N:M structured pruning
│   └── ste.rs                    # Straight-through estimator
│
├── io/                           # Serialization (Section 20)
│   ├── mod.rs
│   ├── format.rs                 # File format definitions
│   ├── save.rs                   # Save operations
│   └── load.rs                   # Load operations
│
├── device/                       # Device placement (Section 21)
│   ├── mod.rs
│   └── multi_device.rs           # Multi-device operations
│
├── error/                        # Error types (Section 22)
│   ├── mod.rs
│   └── types.rs                  # SparseError enum
│
├── fusion/                       # Extended fusion (Section 23)
│   ├── ... (from Part 1)
│   ├── rules.rs                  # Formal fusion rules
│   └── engine.rs                 # FusionEngine
│
├── jit/                          # JIT specialization (Section 24)
│   ├── mod.rs
│   ├── signature.rs              # KernelSignature
│   └── cache.rs                  # KernelCache
│
└── inspect/                      # Debug tools (Section 25)
    ├── mod.rs
    ├── statistics.rs             # SparseStatistics
    ├── validate.rs               # SparseValidator
    └── compare.rs                # SparseComparator
26.2 Updated burn-sparse Structure
burn-sparse/src/
├── ... (from Part 1)
│
└── nn/                           # Sparse modules (Section 19)
    ├── mod.rs
    ├── linear.rs                 # SparseLinear
    ├── embedding.rs              # SparseEmbedding
    └── attention.rs              # Sparse attention (future)