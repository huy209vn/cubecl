//! Common traits for sparse operations.

use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::handle::SparseTensorHandle;

/// Trait for sparse operations (SpMM, SpMV, etc.)
pub trait SparseOperation {
    /// Configuration type for this operation
    type Config;

    /// Execute the operation
    fn execute<R: cubecl_runtime::runtime::Runtime>(
        sparse: &SparseTensorHandle,
        dense: &Handle,
        config: &Self::Config,
        client: &ComputeClient<R>,
    ) -> SparseResult<Handle>;
}
