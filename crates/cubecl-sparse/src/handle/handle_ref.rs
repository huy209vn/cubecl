//! Reference to sparse tensor handle.

use cubecl_runtime::server::Handle;
use crate::format::SparseFormatId;

/// Borrowed reference to a sparse tensor handle.
///
/// Lightweight reference that doesn't own buffers.
pub struct SparseTensorHandleRef<'a> {
    /// Format identifier
    pub format: SparseFormatId,

    /// References to GPU buffer handles
    pub buffers: &'a [Handle],

    /// Shape [rows, cols]
    pub shape: [usize; 2],

    /// Number of non-zeros
    pub nnz: usize,

    /// Element dtype
    pub dtype: cubecl_core::ir::StorageType,
}

impl<'a> SparseTensorHandleRef<'a> {
    /// Get element dtype
    pub fn dtype(&self) -> cubecl_core::ir::StorageType {
        self.dtype
    }

    /// Get shape
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Get number of non-zeros
    pub fn nnz(&self) -> usize {
        self.nnz
    }
}
