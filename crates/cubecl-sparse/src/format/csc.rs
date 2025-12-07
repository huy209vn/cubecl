//! CSC (Compressed Sparse Column) format

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::format::{SparseFormat, SparseFormatId, SparseMetadata, SparseStorage};

pub struct CscStorage {
    pub col_ptrs: Handle,
    pub row_indices: Handle,
    pub values: Handle,
    pub meta: CscMetadata,
}

#[derive(Clone, Debug)]
pub struct CscMetadata {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub dtype: StorageType,
}

impl SparseFormat for CscStorage {
    const FORMAT_ID: SparseFormatId = SparseFormatId::Csc;
    const ROW_MAJOR: bool = false;
    const COL_MAJOR: bool = true;
    const DYNAMIC_PATTERN: bool = false;
}

impl SparseStorage for CscStorage {
    type Metadata = CscMetadata;
    const NUM_BUFFERS: usize = 3;

    fn from_dense<R: cubecl_runtime::runtime::Runtime>(
        _dense: &Handle,
        _threshold: f32,
        _shape: &[usize],
        _client: &ComputeClient<R>,
    ) -> SparseResult<Self> {
        todo!()
    }

    fn to_dense<R: cubecl_runtime::runtime::Runtime>(&self, _client: &ComputeClient<R>) -> SparseResult<Handle> {
        todo!()
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.meta
    }

    fn sparsity(&self) -> f32 {
        let total = self.meta.rows * self.meta.cols;
        if total == 0 { 0.0 } else { 1.0 - (self.meta.nnz as f32 / total as f32) }
    }

    fn memory_bytes(&self) -> usize {
        todo!()
    }
}

impl SparseMetadata for CscMetadata {
    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn dtype(&self) -> StorageType {
        self.dtype
    }
}

impl CscMetadata {
    pub fn new(rows: usize, cols: usize, nnz: usize, dtype: StorageType) -> Self {
        Self { rows, cols, nnz, dtype }
    }
}
