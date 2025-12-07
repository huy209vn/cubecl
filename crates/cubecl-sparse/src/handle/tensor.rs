//! High-level sparse tensor type.

use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;
use std::sync::Arc;
use crate::error::SparseResult;
use crate::format::{SparseFormatId, SparseStorage};
use crate::handle::SparseTensorHandle;
use cubecl_core::prelude::StorageType;
pub struct SparseTensor {
    
    /// Handle to GPU storage
    handle: Arc<SparseTensorHandle>,
    
    /// Logical shape [rows, cols]
    shape: [usize;2],
    
    /// Data type
    dtype: StorageType,
    
    /// Sparse format
    format: SparseFormatId,
}

impl SparseTensor {
    /// Create from dense tensor
    pub fn from_dense<R: cubecl_runtime::runtime::Runtime, S: SparseStorage>(
        dense: &Handle,
        threshold: f32,
        shape: &[usize],
        client: &ComputeClient<R>,
    ) -> SparseResult<Self> {
        let storage = S::from_dense(dense, threshold, shape, client)?; // This creates the storage
        let handle = SparseTensorHandle::from_storage(storage)?; // Convert storage to handle
        let metadata = handle.metadata();
        Ok(Self {
            handle: Arc::new(handle),
            shape: metadata.shape,
            dtype: metadata.dtype,
            format: metadata.format,
        })
    }

    /// Convert to dense tensor
    pub fn to_dense<R: cubecl_runtime::runtime::Runtime>(
        &self,
        client: &ComputeClient<R>,
    ) -> SparseResult<Handle> {
        // Access the underlying storage from the handle to convert to dense
        todo!("Implement to_dense using the handle's storage")
    }

    /// Get metadata
    pub fn metadata(&self) -> &SparseTensorMetadata {
        self.handle.metadata()
    }

    /// Get format ID
    pub fn format(&self) -> SparseFormatId {
        self.format
    }

    /// Convert to handle
    pub fn into_handle(self) -> SparseResult<SparseTensorHandle> {
        Ok(Arc::try_unwrap(self.handle).unwrap_or_else(|arc| (*arc).clone()))
    }
}
