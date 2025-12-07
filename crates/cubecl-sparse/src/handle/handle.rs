//! Sparse tensor handle types.

use cubecl_runtime::server::Handle;
use crate::format::{SparseFormatId, SparseStorage};
use crate::error::SparseResult;

/// Handle to a sparse tensor on GPU.
///
/// Owns the GPU buffers for a sparse matrix in a specific format.
pub struct SparseTensorHandle {
    /// Format identifier
    pub format: SparseFormatId,

    /// GPU buffer handles (row_ptrs, col_indices, values, etc.)
    /// Number and meaning depends on format
    pub buffers: alloc::vec::Vec<Handle>,

    /// Shape [rows, cols]
    pub shape: [usize; 2],

    /// Number of non-zeros
    pub nnz: usize,

    /// Element dtype
    pub dtype: cubecl_core::ir::StorageType,
}

impl SparseTensorHandle {
    /// Create from format-specific storage
    pub fn from_storage<S: SparseStorage>(storage: S) -> SparseResult<Self> {
        todo!("Create handle from storage")
    }

    /// Get metadata
    pub fn metadata(&self) -> SparseTensorMetadata {
        SparseTensorMetadata {
            format: self.format,
            shape: self.shape,
            nnz: self.nnz,
            dtype: self.dtype,
        }
    }

    /// Get element dtype
    pub fn dtype(&self) -> cubecl_core::ir::StorageType {
        self.dtype
    }
}

/// Metadata for a sparse tensor
#[derive(Clone, Copy, Debug)]
pub struct SparseTensorMetadata {
    pub format: SparseFormatId,
    pub shape: [usize; 2],
    pub nnz: usize,
    pub dtype: cubecl_core::ir::StorageType,
}
