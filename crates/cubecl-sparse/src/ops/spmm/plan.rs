//! SpMM execution planning.
//!
//! The plan captures all one-time analysis and preprocessing:
//! - Matrix statistics
//! - Tile decomposition (optional, for large matrices)
//! - Row binning with prepared gather buffers
//! - Execution order

use alloc::collections::BTreeMap;
use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::{ComputeServer, Handle};

use crate::error::SparseResult;
use crate::handle::{SparseTensorHandle, SparseTensorHandleRef};
use crate::ops::spmm::{
    config::SpmmConfig,
    analysis::{
        MatrixStatistics, TileDecomposition, TileInfo, TileClass,
        RowBinning, analyze_csr, decompose_tiles, create_binning,
    },
};

/// A step in the execution plan.
#[derive(Clone, Debug)]
pub enum ExecutionStep {
    /// Execute dense GEMM for a tile.
    DenseGemm { tile_idx: usize },

    /// Execute banded kernel for a tile.
    BandedKernel { tile_idx: usize },

    /// Execute block-sparse kernel for a tile.
    BlockSparseKernel { tile_idx: usize },

    /// Execute binned SpMM for a bin.
    BinnedSpMM { bin_idx: usize },
}

/// Pre-extracted dense tile data.
pub struct DenseTileData

{
    pub tile_info: TileInfo,
    /// Extracted dense data.
    pub dense_buffer: Handle,
    _phantom: core::marker::PhantomData<C>,
}

/// Banded tile execution info.
pub struct BandedTileData {
    pub tile_info: TileInfo,
    pub bandwidth: u32,
    pub lower_bandwidth: u32,
    pub upper_bandwidth: u32,
}

/// Block-sparse tile execution info.
pub struct BlockSparseTileData

{
    pub tile_info: TileInfo,
    pub block_size: u32,
    pub block_data: Handle,
    pub block_indices: Handle,
    pub num_blocks: u32,
    _phantom: core::marker::PhantomData<C>,
}

/// Complete execution plan for SpMM.
///
/// Created once per (matrix, output_cols) pair, reusable across multiple executions.
pub struct SpmmPlan

{
    // --- Dimensions ---
    pub m: u32,
    pub k: u32,
    pub n: u32,

    // --- Analysis results ---
    pub stats: MatrixStatistics,
    pub tiles: Option<TileDecomposition>,

    // --- Binned data ---
    pub binning: RowBinning,

    // --- Extracted tile data ---
    pub dense_tiles: alloc::vec::Vec<DenseTileData>,
    pub banded_tiles: alloc::vec::Vec<BandedTileData>,
    pub block_sparse_tiles: alloc::vec::Vec<BlockSparseTileData>,

    // --- Execution order ---
    pub steps: alloc::vec::Vec<ExecutionStep>,

    // --- Metadata ---
    pub autotuned: bool,
}

impl SpmmPlan

{
    /// Create execution plan for A @ B.
    ///
    /// # Implementation
    /// 1. Analyze matrix (statistics)
    /// 2. Tile decomposition (if enabled and matrix large enough)
    /// 3. Process tiles, extract special cases:
    ///    - Dense tiles → extract to dense buffer
    ///    - Banded tiles → record bandwidth info
    ///    - Block-sparse tiles → extract blocks
    /// 4. Create binning for sparse regions
    /// 5. Build execution steps
    ///
    /// # Algorithm Steps
    /// ```text
    /// Step 1: Analyze matrix
    ///   stats = analyze_csr(sparse, client)
    ///
    /// Step 2: Tile decomposition (if enabled)
    ///   tiles = if config.algorithm.enable_tile_classification {
    ///       decompose_tiles(sparse, &stats, &config.tile, client)
    ///   } else {
    ///       None
    ///   }
    ///
    /// Step 3: Process tiles
    ///   for tile in tiles.non_empty_tiles():
    ///     match tile.classification:
    ///       Dense →
    ///         dense_buffer = extract_dense_tile(sparse, tile, client)
    ///         dense_tiles.push(DenseTileData { tile_info, dense_buffer })
    ///         steps.push(ExecutionStep::DenseGemm { tile_idx })
    ///
    ///       Banded { bandwidth, lower_bandwidth, upper_bandwidth } →
    ///         banded_tiles.push(BandedTileData { tile_info, ... })
    ///         steps.push(ExecutionStep::BandedKernel { tile_idx })
    ///
    ///       BlockSparse { block_size, blocks } →
    ///         (block_data, block_indices, num_blocks) =
    ///           extract_block_sparse_tile(sparse, tile, block_size, blocks, client)
    ///         block_sparse_tiles.push(BlockSparseTileData { ... })
    ///         steps.push(ExecutionStep::BlockSparseKernel { tile_idx })
    ///
    ///       Sparse | LowRank →
    ///         // Handled by binning
    ///
    /// Step 4: Create binning
    ///   binning = create_binning(
    ///       sparse,
    ///       &stats,
    ///       &config.binning,
    ///       n_output_cols,
    ///       client
    ///   )
    ///
    /// Step 5: Add binned execution steps
    ///   for (bin_idx, bin) in binning.bins.enumerate():
    ///     if bin.strategy != Skip:
    ///       steps.push(ExecutionStep::BinnedSpMM { bin_idx })
    ///
    /// Step 6: Return SpmmPlan
    /// ```
    pub fn create(
        sparse: &SparseTensorHandle,
        n_output_cols: u32,
        config: &SpmmConfig,
        client: &ComputeClient,
    ) -> SparseResult<Self> {
        // Extract dimensions
        let meta = sparse.metadata();
        let m = meta.shape[0] as u32;
        let k = meta.shape[1] as u32;
        let n = n_output_cols;

        // Step 1: Analyze matrix
        let stats = analyze_csr(sparse, client)?;

        // Step 2: Tile decomposition (if enabled)
        let tiles = if config.algorithm.enable_tile_classification {
            decompose_tiles(sparse, &stats, &config.tile, client)
        } else {
            None
        };

        // Step 3: Process tiles (TODO: implement tile extraction)
        let dense_tiles = alloc::vec::Vec::new();
        let banded_tiles = alloc::vec::Vec::new();
        let block_sparse_tiles = alloc::vec::Vec::new();
        let mut steps = alloc::vec::Vec::new();

        // Step 4: Create binning
        let binning = create_binning(
            sparse,
            &stats,
            &config.binning,
            n_output_cols,
            client,
        )?;

        // Step 5: Add binned execution steps
        for (bin_idx, bin) in binning.bins.iter().enumerate() {
            if bin.strategy != crate::ops::spmm::analysis::BinStrategy::Skip {
                steps.push(ExecutionStep::BinnedSpMM { bin_idx });
            }
        }

        // Step 6: Return plan
        Ok(SpmmPlan {
            m,
            k,
            n,
            stats,
            tiles,
            binning,
            dense_tiles,
            banded_tiles,
            block_sparse_tiles,
            steps,
            autotuned: false,
        })
    }

    /// Extract dense tile to dense buffer.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Allocate dense tensor for tile: [tile_rows × tile_cols]
    /// 2. Initialize to zero
    /// 3. Launch kernel to extract sparse tile to dense:
    ///    - For each row in tile:
    ///      - For each nnz in row within tile column range:
    ///        - dense_tile[local_row, local_col] = value
    /// 4. Return dense tensor handle
    /// ```
    fn extract_dense_tile(
        _sparse: &SparseTensorHandle,
        _tile: &TileInfo,
        _client: &ComputeClient,
    ) -> SparseResult<Handle> {
        // TODO: Implement dense tile extraction
        todo!("Extract dense tile from sparse matrix")
    }

    /// Extract block-sparse tile.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Allocate buffer for dense blocks: [num_blocks × block_rows × block_cols]
    /// 2. Allocate buffer for block indices: [num_blocks × 2] (row, col pairs)
    /// 3. Launch kernel to extract blocks
    /// 4. Return (block_data, block_indices, num_blocks)
    /// ```
    fn extract_block_sparse_tile(
        _sparse: &SparseTensorHandle,
        _tile: &TileInfo,
        _block_size: u32,
        _blocks: &[(u32, u32)],
        _client: &ComputeClient,
    ) -> SparseResult<(Handle, Handle, u32)> {
        // TODO: Implement block-sparse tile extraction
        todo!("Extract block-sparse tile from sparse matrix")
    }
}

/// Cache for SpMM plans.
///
/// Plans are expensive to create (analysis, binning). Cache and reuse them.
pub struct SpmmPlanCache

{
    cache: BTreeMap<(u64, u32), SpmmPlan>,
}

impl SpmmPlanCache

{
    pub fn new() -> Self {
        Self {
            cache: BTreeMap::new(),
        }
    }

    /// Get cached plan or create new one.
    ///
    /// Key: (matrix_id, n_output_cols)
    ///
    /// # Implementation
    /// ```text
    /// 1. Generate key from matrix_id and n_output_cols
    /// 2. Check cache, return if exists
    /// 3. Otherwise create plan, insert into cache, return
    /// ```
    pub fn get_or_create(
        &mut self,
        sparse: &SparseTensorHandle,
        n_output_cols: u32,
        config: &SpmmConfig,
        client: &ComputeClient,
    ) -> SparseResult<&SpmmPlan> {
        // Generate key
        // TODO: Get proper matrix ID from handle
        let matrix_id = 0u64; // Placeholder
        let key = (matrix_id, n_output_cols);

        // Check cache
        if !self.cache.contains_key(&key) {
            // Create new plan
            let plan = SpmmPlan::create(sparse, n_output_cols, config, client)?;
            self.cache.insert(key, plan);
        }

        Ok(self.cache.get(&key).unwrap())
    }

    /// Invalidate plans for a matrix.
    ///
    /// Call when matrix structure changes.
    pub fn invalidate(&mut self, matrix_id: u64) {
        self.cache.retain(|(id, _), _| *id != matrix_id);
    }

    /// Clear all cached plans.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for SpmmPlanCache

{
    fn default() -> Self {
        Self::new()
    }
}
