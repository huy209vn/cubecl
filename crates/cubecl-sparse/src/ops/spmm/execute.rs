//! SpMM plan execution.
//!
//! Execute the precomputed plan to perform SpMM.

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::{ComputeServer, Handle};

use crate::error::SparseResult;
use crate::handle::{SparseTensorHandle, SparseTensorHandleRef};
use crate::ops::spmm::{
    plan::{SpmmPlan, ExecutionStep, DenseTileData, BandedTileData, BlockSparseTileData},
    analysis::BinStrategy,
    algorithm,
};

impl SpmmPlan

{
    /// Execute the plan: C = A @ B
    ///
    /// # Implementation
    /// 1. Allocate output tensor C (M × N)
    /// 2. For each step in execution order:
    ///    - DenseGemm → execute_dense_tile
    ///    - BandedKernel → execute_banded_tile
    ///    - BlockSparseKernel → execute_block_sparse_tile
    ///    - BinnedSpMM → execute_bin (dispatch to appropriate algorithm)
    /// 3. Return C
    ///
    /// # Algorithm
    /// ```text
    /// Step 1: Allocate output
    ///   let mut c = allocate_zeros(client, [m, n], dtype)
    ///
    /// Step 2: Execute each step in plan
    ///   for step in &self.steps {
    ///     match step {
    ///       ExecutionStep::DenseGemm { tile_idx } =>
    ///         self.execute_dense_tile(*tile_idx, b, b_shape, &mut c, client)?
    ///
    ///       ExecutionStep::BandedKernel { tile_idx } =>
    ///         self.execute_banded_tile(*tile_idx, sparse, b, b_shape, &mut c, client)?
    ///
    ///       ExecutionStep::BlockSparseKernel { tile_idx } =>
    ///         self.execute_block_sparse_tile(*tile_idx, b, b_shape, &mut c, client)?
    ///
    ///       ExecutionStep::BinnedSpMM { bin_idx } =>
    ///         self.execute_bin(*bin_idx, b, b_shape, &mut c, client)?
    ///     }
    ///   }
    ///
    /// Step 3: Return output
    ///   return c
    /// ```
    pub fn execute(
        &self,
        sparse: &SparseTensorHandle,
        b: &Handle,
        b_shape: &[usize],
        client: &ComputeClient<R>,
    ) -> SparseResult<Handle> {
        // Validate dimensions
        assert_eq!(self.k as usize, b_shape[0], "Dimension mismatch: A.cols != B.rows");
        assert_eq!(self.n as usize, b_shape[1], "Dimension mismatch: plan.n != B.cols");

        // Step 1: Allocate output
        let dtype = sparse.dtype();
        let output_size = (self.m as usize) * (self.n as usize);
        let mut c = client.empty(output_size);

        // TODO: Initialize C to zeros
        // launch_zero_kernel(&mut c, output_size, client);

        // Step 2: Execute each step in plan
        for step in &self.steps {
            match step {
                ExecutionStep::DenseGemm { tile_idx } => {
                    self.execute_dense_tile(*tile_idx, b, b_shape, &mut c, client)?;
                }

                ExecutionStep::BandedKernel { tile_idx } => {
                    self.execute_banded_tile(*tile_idx, sparse, b, b_shape, &mut c, client)?;
                }

                ExecutionStep::BlockSparseKernel { tile_idx } => {
                    self.execute_block_sparse_tile(*tile_idx, b, b_shape, &mut c, client)?;
                }

                ExecutionStep::BinnedSpMM { bin_idx } => {
                    self.execute_bin(*bin_idx, b, b_shape, &mut c, client)?;
                }
            }
        }

        // Step 3: Return output
        Ok(c)
    }

    /// Execute dense tile GEMM.
    ///
    /// Use CubeCL's optimized dense GEMM on extracted tile.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Get tile_data = &self.dense_tiles[tile_idx]
    /// 2. Extract corresponding B slice: B[tile.col_range, :]
    /// 3. Call dense GEMM: C_tile = dense_tile @ B_slice
    /// 4. Write result to appropriate C region: C[tile.row_range, :]
    /// 5. Use CubeCL's matmul or custom optimized GEMM
    /// ```
    fn execute_dense_tile(
        &self,
        tile_idx: usize,
        _b: &Handle,
        _b_shape: &[usize],
        _c: &mut Handle,
        _client: &ComputeClient,
    ) -> SparseResult<()> {
        let _tile_data = &self.dense_tiles[tile_idx];

        // TODO: Extract B slice for tile column range
        // TODO: Launch dense GEMM kernel
        // TODO: Write result to C at appropriate offset

        Ok(())
    }

    /// Execute banded tile kernel.
    ///
    /// Specialized kernel for banded structure.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Get tile_data = &self.banded_tiles[tile_idx]
    /// 2. Launch banded kernel:
    ///    - For each row i in tile:
    ///      relevant_cols = [max(0, i-lower_bw), min(K, i+upper_bw+1)]
    ///      C[i, :] = A[i, relevant_cols] @ B[relevant_cols, :]
    /// 3. Optimization:
    ///    - Stage B[diag_region, :] into shared memory
    ///    - Process multiple rows against shared B
    ///    - Slide the diagonal window
    /// ```
    fn execute_banded_tile(
        &self,
        tile_idx: usize,
        _sparse: &SparseTensorHandle,
        _b: &Handle,
        _b_shape: &[usize],
        _c: &mut Handle,
        _client: &ComputeClient,
    ) -> SparseResult<()> {
        let _tile_data = &self.banded_tiles[tile_idx];

        // TODO: Launch banded SpMM kernel
        // TODO: For each row, only process columns within bandwidth

        Ok(())
    }

    /// Execute block-sparse tile kernel.
    ///
    /// Dense GEMM per block.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Get tile_data = &self.block_sparse_tiles[tile_idx]
    /// 2. For each dense block at (br, bc):
    ///    - row_range = [br * block_size, (br+1) * block_size)
    ///    - col_range = [bc * block_size, (bc+1) * block_size)
    ///    - C[row_range, :] += A_block[br, bc] @ B[col_range, :]
    /// 3. Batch blocks with same bc for B reuse
    /// 4. Can use tensor cores per block
    /// ```
    fn execute_block_sparse_tile(
        &self,
        tile_idx: usize,
        _b: &Handle,
        _b_shape: &[usize],
        _c: &mut Handle,
        _client: &ComputeClient,
    ) -> SparseResult<()> {
        let _tile_data = &self.block_sparse_tiles[tile_idx];

        // TODO: For each dense block
        // TODO: Extract block from tile_data.block_data
        // TODO: Extract corresponding B slice
        // TODO: Dense GEMM
        // TODO: Accumulate to C

        Ok(())
    }

    /// Execute binned SpMM for one bin.
    ///
    /// Dispatch to appropriate algorithm based on bin strategy.
    ///
    /// # Algorithm
    /// ```text
    /// bin = &self.binning.bins[bin_idx]
    /// dtype = self.dtype
    ///
    /// match bin.strategy:
    ///   Skip => return  // No-op
    ///
    ///   RowSplit =>
    ///     RowSplitSpMM::execute_bin(bin, b, c, n, dtype, client)
    ///
    ///   WarpPerRow =>
    ///     WarpPerRowSpMM::execute_bin(bin, b, c, n, dtype, client)
    ///
    ///   VectorWarpPerRow { vec_width } =>
    ///     VectorWarpPerRowSpMM::execute_bin(bin, b, c, n, vec_width, dtype, client)
    ///
    ///   GatherGemm { tile_n } =>
    ///     GatherGemmSpMM::execute_bin(bin, b, c, n, k, tile_n, dtype, client)
    ///
    ///   GatherTensorCore { tile_m, tile_k, tile_n } =>
    ///     GatherTensorCoreSpMM::execute_bin(bin, b, c, n, k, tile_m, tile_k, tile_n, dtype, client)
    ///
    ///   MergePath =>
    ///     MergePathSpMM::execute_bin(bin, b, c, n, dtype, client)
    /// ```
    fn execute_bin(
        &self,
        bin_idx: usize,
        b: &Handle,
        b_shape: &[usize],
        c: &mut Handle,
        client: &ComputeClient,
    ) -> SparseResult<()> {
        let bin = &self.binning.bins[bin_idx];
        let n = self.n;
        let k = self.k;
        let dtype = StorageType::F32; // TODO: Get from metadata

        match bin.strategy {
            BinStrategy::Skip => {
                // No-op for empty bins
                Ok(())
            }

            BinStrategy::RowSplit => {
                algorithm::RowSplitSpMM::execute_bin(bin, b, c, n, dtype, client)
            }

            BinStrategy::WarpPerRow => {
                algorithm::WarpPerRowSpMM::execute_bin(bin, b, c, n, dtype, client)
            }

            BinStrategy::VectorWarpPerRow { vec_width } => {
                algorithm::VectorWarpPerRowSpMM::execute_bin(bin, b, c, n, vec_width, dtype, client)
            }

            BinStrategy::GatherGemm { tile_n } => {
                algorithm::GatherGemmSpMM::execute_bin(bin, b, c, n, k, tile_n, dtype, client)
            }

            BinStrategy::GatherTensorCore { tile_m, tile_k, tile_n } => {
                algorithm::GatherTensorCoreSpMM::execute_bin(
                    bin, b, c, n, k, tile_m, tile_k, tile_n, dtype, client,
                )
            }

            BinStrategy::MergePath => {
                algorithm::MergePathSpMM::execute_bin(bin, b, c, n, dtype, client)
            }
        }
    }
}

/// Execute dense GEMM for a dense tile.
///
/// Helper for tile handlers.
///
/// # Algorithm
/// ```text
/// 1. Extract B slice corresponding to tile columns
/// 2. Call dense GEMM (CubeCL matmul or custom)
/// 3. Write to C at appropriate offset
/// ```
pub fn execute_dense_gemm(
    _tile_data: &DenseTileData,
    _b: &Handle,
    _b_shape: &[usize],
    _c: &mut Handle,
    _client: &ComputeClient,
) -> SparseResult<()>

{
    // TODO: Implement
    Ok(())
}

/// Execute banded kernel for a banded tile.
///
/// Helper for tile handlers.
///
/// # Algorithm
/// ```text
/// 1. Launch banded SpMM kernel
/// 2. For each row, only process columns within bandwidth
/// ```
pub fn execute_banded_kernel(
    _sparse: &SparseTensorHandle,
    _tile_data: &BandedTileData,
    _b: &Handle,
    _b_shape: &[usize],
    _c: &mut Handle,
    _client: &ComputeClient,
) -> SparseResult<()>

{
    // TODO: Implement
    Ok(())
}

/// Execute block-sparse GEMM for a block-sparse tile.
///
/// Helper for tile handlers.
///
/// # Algorithm
/// ```text
/// 1. For each dense block:
///    - Extract block from tile_data.block_data
///    - Extract corresponding B slice
///    - Dense GEMM
///    - Accumulate to C
/// ```
pub fn execute_block_sparse_gemm(
    _tile_data: &BlockSparseTileData,
    _b: &Handle,
    _b_shape: &[usize],
    _c: &mut Handle,
    _client: &ComputeClient,
) -> SparseResult<()>

{
    // TODO: Implement
    Ok(())
}
