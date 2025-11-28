use cubecl_runtime::client::ComputeClient;
use cubecl_core::prelude::Tensor;
use cubecl_runtime::runtime::Runtime;

use cubecl_core::prelude::CubeType;
use cubecl_core::ir::StorageType;
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
pub trait SparseStorage<R: Runtime + CubeType>: SparseFormat {
    /// Metadata type (shape, nnz, block info, etc.)
    type Metadata: SparseMetadata;
    
    /// Number of GPU buffers this format requires
    const NUM_BUFFERS: usize;
    
    /// Create storage from dense tensor (sparsification)
    fn from_dense(
        dense: &Tensor<R>,
        threshold: f32,
        client: &ComputeClient<R>,
    ) -> Self;
    
    /// Convert to dense tensor
    fn to_dense(
        &self,
        client: &ComputeClient<R>,
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
    fn dtype(&self) -> StorageType;
}