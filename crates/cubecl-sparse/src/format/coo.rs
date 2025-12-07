//! COO (Coordinate) format implementation

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::format::{SparseFormat, SparseFormatId, SparseMetadata, SparseStorage};

/// COO storage
pub struct CooStorage {
    pub row_indices: Handle,
    pub col_indices: Handle,
    pub values: Handle,
    pub sorted: bool,
    pub meta: CooMetadata,
}

#[derive(Clone, Debug)]
pub struct CooMetadata {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub dtype: StorageType,
}

impl SparseFormat for CooStorage {
    const FORMAT_ID: SparseFormatId = SparseFormatId::Coo;
    const ROW_MAJOR: bool = false;
    const COL_MAJOR: bool = false;
    const DYNAMIC_PATTERN: bool = true;
}

impl SparseStorage for CooStorage {
    type Metadata = CooMetadata;
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

impl SparseMetadata for CooMetadata {
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

impl CooStorage {
    pub fn sort<R: cubecl_runtime::runtime::Runtime>(&mut self, _client: &ComputeClient<R>) {
        todo!()
    }

    pub fn coalesce<R: cubecl_runtime::runtime::Runtime>(&mut self, _client: &ComputeClient<R>) {
        todo!()
    }

    pub fn scatter_add<R: cubecl_runtime::runtime::Runtime>(&mut self, _other: &CooStorage, _client: &ComputeClient<R>) {
        todo!()
    }

    pub unsafe fn from_raw_parts(row_indices: Handle, col_indices: Handle, values: Handle, meta: CooMetadata, sorted: bool) -> Self {
        Self {
            row_indices,
            col_indices,
            values,
            sorted,
            meta,
        }
    }
}

impl CooMetadata {
    pub fn new(rows: usize, cols: usize, nnz: usize, dtype: StorageType) -> Self {
        Self { rows, cols, nnz, dtype }
    }
}
