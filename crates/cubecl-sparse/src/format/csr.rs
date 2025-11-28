

pub struct CsrStorage<R: Runtime> {
    /// Row pointers buffer [M+1 elements, u32]
    pub row_ptrs: Handle<R::Server>,
    
    /// Column indices buffer [NNZ elements, u32]
    pub col_indices: Handle<R::Server>,
    
    /// Values buffer [NNZ elements, dtype]
    pub values: Handle<R::Server>,
    
    /// CPU-side metadata
    pub meta: CsrMetadata,
}

#[derive(Clone, Debug)]
pub struct CsrMetadata {
    pub rows: usize,      // M
    pub cols: usize,      // K  
    pub nnz: usize,       // Number of non-zeros
    pub dtype: DType,
    
    /// Row distribution statistics (for algorithm selection)
    pub row_stats: Option<RowStatistics>,
}

#[derive(Clone, Debug)]
pub struct RowStatistics {
    pub min_nnz_per_row: usize,
    pub max_nnz_per_row: usize,
    pub avg_nnz_per_row: f32,
    pub std_nnz_per_row: f32,
    
    /// Histogram buckets: [0-8), [8-32), [32-128), [128-512), [512+)
    pub row_length_histogram: [u32; 5],
}

impl SparseFormat for CsrStorage<R> {
    const FORMAT_ID: SparseFormatId = SparseFormatId::Csr;
    const ROW_MAJOR: bool = true;
    const COL_MAJOR: bool = false;
    const DYNAMIC_PATTERN: bool = false;
}
// Construction Operations
impl<R: Runtime> CsrStorage<R> {
    /// From COO (sorted by row, then column)
    pub fn from_coo(coo: &CooStorage<R>, client: &ComputeClient<...>) -> Self;
    
    /// From dense with threshold
    pub fn from_dense(dense: &Tensor<R>, threshold: f32, client: &ComputeClient<...>) -> Self;
    
    /// From explicit components (unsafe: caller ensures validity)
    pub unsafe fn from_raw_parts(
        row_ptrs: Handle<R::Server>,
        col_indices: Handle<R::Server>,
        values: Handle<R::Server>,
        meta: CsrMetadata,
    ) -> Self;
    
    /// Transpose to CSC (structural transpose, no value changes)
    pub fn transpose_to_csc(&self, client: &ComputeClient<...>) -> CscStorage<R>;
    
    /// Compute row statistics for algorithm selection
    pub fn compute_row_stats(&mut self, client: &ComputeClient<...>);
}