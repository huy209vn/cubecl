use crate::prelude::SparseStorage;
use cubecl_runtime::server::{Handle, ComputeServer};
use core::marker::PhantomData;
use crate::prelude::SparseFormat;
// Fix 1: Use dyn SparseStorage with explicit metadata type
pub trait SparseMemoryPool {
    fn alloc_sparse<S: SparseStorage>(&self, meta: &S::Metadata) -> SparseBufferSet;
    fn realloc_sparse(&self, existing: SparseBufferSet, new_meta: &<S as SparseStorage>::Metadata) -> SparseBufferSet;
    fn register_sparse_eviction_policy(&self, policy: SparseEvictionPolicy);
}

pub struct SparseBufferSet {
    pub handles: [Handle; <SparseStorage as SparseFormat>::NUM_BUFFERS],
    pub layout: BufferLayout,
}
#[derive(Clone, Debug)]
pub struct BufferLayout {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub alignments: Vec<usize>,
    pub total_bytes: usize,
}

/// CPU-side metadata cache to avoid GPU syncs
pub struct SparseMetadataCache {
    /// LRU cache: tensor_id → metadata
    cache: LruCache<TensorId, Arc<dyn SparseMetadata>>,
    
    /// Dirty flags for metadata that needs GPU sync
    dirty: HashSet<TensorId>,
}

impl SparseMetadataCache {
    /// Get metadata without GPU sync (may be stale for dynamic patterns)
    pub fn get(&self, id: TensorId) -> Option<Arc<dyn SparseMetadata>>;
    
    /// Mark metadata as potentially stale (pattern changed)
    pub fn invalidate(&mut self, id: TensorId);
    
    /// Force GPU sync and update cache
    pub async fn sync(&mut self, id: TensorId, client: &ComputeClient<...>);
}