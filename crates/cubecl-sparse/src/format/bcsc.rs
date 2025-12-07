//! BCSC (Block Sparse Column) format

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::format::{SparseFormat, SparseFormatId, SparseMetadata, SparseStorage};

pub struct BcscStorage {
    pub block_col_ptrs: Handle,
    pub block_row_indices: Handle,
    pub block_values: Handle,
    pub meta: BcscMetadata,
}

#[derive(Clone, Debug)]
pub struct BcscMetadata {
    pub rows: usize,
    pub cols: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub num_blocks: usize,
    pub dtype: StorageType,
}

impl SparseFormat for BcscStorage {
    const FORMAT_ID: SparseFormatId = SparseFormatId::Bcsc { block_rows: 16, block_cols: 16 };
    const ROW_MAJOR: bool = false;
    const COL_MAJOR: bool = true;
    const DYNAMIC_PATTERN: bool = false;
}

impl SparseStorage for BcscStorage {
    type Metadata = BcscMetadata;
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
        let total_blocks = (self.meta.rows / self.meta.block_rows) * (self.meta.cols / self.meta.block_cols);
        if total_blocks == 0 { 0.0 } else { 1.0 - (self.meta.num_blocks as f32 / total_blocks as f32) }
    }

    fn memory_bytes(&self) -> usize {
        todo!()
    }
}

impl SparseMetadata for BcscMetadata {
    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn nnz(&self) -> usize {
        self.num_blocks * self.block_rows * self.block_cols
    }

    fn dtype(&self) -> StorageType {
        self.dtype
    }
}

impl BcscMetadata {
    pub fn new(rows: usize, cols: usize, block_rows: usize, block_cols: usize, num_blocks: usize, dtype: StorageType) -> Self {
        Self { rows, cols, block_rows, block_cols, num_blocks, dtype }
    }
}
