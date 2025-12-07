//! N:M structured sparsity format

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::Handle;

use crate::error::SparseResult;
use crate::format::{SparseFormat, SparseFormatId, SparseMetadata, SparseStorage};

/// N:M structured sparse storage
pub struct NMStorage {
    pub values: Handle,
    pub indices: Handle,
    pub meta: NMMetadata,
}

#[derive(Clone, Debug)]
pub struct NMMetadata {
    pub rows: usize,
    pub cols: usize,
    pub n: u8,
    pub m: u8,
    pub dtype: StorageType,
}

impl SparseFormat for NMStorage {
    const FORMAT_ID: SparseFormatId = SparseFormatId::NM { n: 2, m: 4 };
    const ROW_MAJOR: bool = true;
    const COL_MAJOR: bool = false;
    const DYNAMIC_PATTERN: bool = false;
}

impl SparseStorage for NMStorage {
    type Metadata = NMMetadata;
    const NUM_BUFFERS: usize = 2;

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
        1.0 - (self.meta.n as f32 / self.meta.m as f32)
    }

    fn memory_bytes(&self) -> usize {
        todo!()
    }
}

impl SparseMetadata for NMMetadata {
    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn nnz(&self) -> usize {
        let num_groups = self.cols / self.m as usize;
        self.rows * num_groups * self.n as usize
    }

    fn dtype(&self) -> StorageType {
        self.dtype
    }
}

impl NMStorage {
    pub fn from_dense_magnitude<R: cubecl_runtime::runtime::Runtime>(
        _dense: &Handle,
        _n: u8,
        _m: u8,
        _client: &ComputeClient<R>,
    ) -> SparseResult<Self> {
        todo!()
    }

    pub fn verify_constraint(&self) -> bool {
        todo!()
    }
}

impl NMMetadata {
    pub fn new(rows: usize, cols: usize, n: u8, m: u8, dtype: StorageType) -> Self {
        Self { rows, cols, n, m, dtype }
    }
}

pub type Sparse2x4 = NMStorage;
pub type Sparse1x4 = NMStorage;
pub type Sparse4x8 = NMStorage;
