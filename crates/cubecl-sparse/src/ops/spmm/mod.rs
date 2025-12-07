//! Sparse Matrix-Matrix Multiplication (SpMM).
//!
//! High-performance adaptive SpMM using Gather-GEMM with binning.

pub mod algorithm;
pub mod analysis;
pub mod config;
pub mod execute;
pub mod plan;

// Re-exports
pub use config::{SpmmConfig, AlgorithmConfig, BinningConfig, TileConfig, PrecisionConfig};
pub use plan::{SpmmPlan, SpmmPlanCache};
pub use algorithm::SpmmAlgorithm;

use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::handle::SparseTensorHandle;

/// Execute Sparse Matrix-Matrix Multiplication: C = A @ B
///
/// # Arguments
/// - `sparse`: Sparse matrix A (M × K)
/// - `dense`: Dense matrix B (K × N)
/// - `client`: Compute client
///
/// # Returns
/// Dense matrix C (M × N)
pub fn spmm<R: cubecl_runtime::runtime::Runtime>(
    sparse: &SparseTensorHandle,
    dense: &Handle,
    dense_shape: &[usize],
    client: &ComputeClient<R>,
) -> SparseResult<Handle> {
    let config = SpmmConfig::default();
    spmm_with_config(sparse, dense, dense_shape, &config, client)
}

/// Execute SpMM with custom configuration.
pub fn spmm_with_config<R: cubecl_runtime::runtime::Runtime>(
    sparse: &SparseTensorHandle,
    dense: &Handle,
    dense_shape: &[usize],
    config: &SpmmConfig,
    client: &ComputeClient<R>,
) -> SparseResult<Handle> {
    // Get or create plan
    let n = dense_shape[1] as u32;
    let plan = SpmmPlan::create(sparse, n, config, client)?;

    // Execute plan
    plan.execute(sparse, dense, dense_shape, client)
}

/// Execute SpMM with cached plan.
///
/// Reuses plan for repeated executions with same sparse matrix and output columns.
pub fn spmm_cached<R: cubecl_runtime::runtime::Runtime>(
    sparse: &SparseTensorHandle,
    dense: &Handle,
    dense_shape: &[usize],
    config: &SpmmConfig,
    cache: &mut SpmmPlanCache,
    client: &ComputeClient<R>,
) -> SparseResult<Handle> {
    let n = dense_shape[1] as u32;
    let plan = cache.get_or_create(sparse, n, config, client)?;
    plan.execute(sparse, dense, dense_shape, client)
}
