//! Core traits for sparse tensor formats

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;

/// Marker trait for sparse storage formats
///
/// All sparse formats (CSR, CSC, COO, N:M, BSR) implement this trait
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

/// Sparse format identifier for runtime dispatch
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SparseFormatId {
    /// Compressed Sparse Row
    Csr,
    /// Compressed Sparse Column
    Csc,
    /// Coordinate (triplet) format
    Coo,
    /// N:M structured sparsity (e.g., 2:4 for 50%)
    NM { n: u8, m: u8 },
    /// Block Sparse Row
    Bsr { block_rows: u16, block_cols: u16 },
    /// Block Sparse Column
    Bcsc { block_rows: u16, block_cols: u16 },
}

/// Core sparse tensor storage trait
pub trait SparseStorage: SparseFormat + Sized {
    /// Metadata type (shape, nnz, block info, etc.)
    type Metadata: SparseMetadata;

    /// Number of GPU buffers this format requires
    const NUM_BUFFERS: usize;

    /// Create storage from dense tensor (sparsification)
    fn from_dense<R: cubecl_runtime::runtime::Runtime>(
        dense: &Handle,
        threshold: f32,
        shape: &[usize],
        client: &ComputeClient<R>,
    ) -> SparseResult<Self>;

    /// Convert to dense tensor
    fn to_dense<R: cubecl_runtime::runtime::Runtime>(&self, client: &ComputeClient<R>) -> SparseResult<Handle>;

    /// Get metadata (cheap, no GPU sync)
    fn metadata(&self) -> &Self::Metadata;

    /// Actual sparsity ratio (nnz / total_elements)
    fn sparsity(&self) -> f32;

    /// Memory footprint in bytes
    fn memory_bytes(&self) -> usize;
}

/// Sparse tensor metadata (CPU-side, no GPU sync needed)
pub trait SparseMetadata: Clone + Send + Sync {
    /// Logical shape of the tensor
    fn shape(&self) -> &[usize];

    /// Number of non-zero elements
    fn nnz(&self) -> usize;

    /// Data type of values
    fn dtype(&self) -> StorageType;
}