//! Error types for sparse operations

use std::io;

use crate::format::SparseFormatId;
use crate::dtype::DType;

/// Result type for sparse operations
pub type SparseResult<T> = Result<T, SparseError>;

/// Sparse operation errors
#[derive(Debug, thiserror::Error)]
pub enum SparseError {
    // Shape errors
    #[error("Shape mismatch in {op}: expected {expected}, got {got}")]
    ShapeMismatch {
        op: &'static str,
        expected: String,
        got: usize,
    },

    #[error("Dimension out of bounds: {dim} >= {max}")]
    DimensionOutOfBounds { dim: usize, max: usize },

    // Pattern errors
    #[error("Sparsity pattern mismatch: expected {expected_nnz} non-zeros, got {got_nnz}")]
    PatternMismatch {
        expected_nnz: usize,
        got_nnz: usize,
    },

    #[error("Pattern modification not allowed in current mode")]
    PatternModificationNotAllowed,

    #[error("Pattern growth not allowed in ShrinkOnly mode")]
    PatternGrowthNotAllowed,

    #[error("Pattern shrink not allowed in GrowOnly mode")]
    PatternShrinkNotAllowed,

    // Format errors
    #[error("Format mismatch: cannot perform operation")]
    FormatMismatch,

    #[error("Unsupported format {format:?} for operation {op}")]
    UnsupportedFormat {
        format: SparseFormatId,
        op: &'static str,
    },

    #[error("Format conversion from {from:?} to {to:?} is lossy and requires explicit conversion")]
    LossyConversionRequired {
        from: SparseFormatId,
        to: SparseFormatId,
    },

    // Block/N:M constraint errors
    #[error("Invalid block size: ({block_rows}, {block_cols}) does not divide ({rows}, {cols})")]
    InvalidBlockSize {
        block_rows: usize,
        block_cols: usize,
        rows: usize,
        cols: usize,
    },

    #[error("Invalid N:M constraint: N={n}, M={m}")]
    InvalidNMConstraint { n: usize, m: usize },

    #[error("Slice must align to block boundary (required alignment: {required})")]
    SliceAlignmentRequired { required: usize },

    // Device errors
    #[error("Device format unsupported")]
    DeviceFormatUnsupported,

    #[error("Device mismatch")]
    DeviceMismatch,

    // Memory errors
    #[error("Out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("Memory fragmentation: cannot allocate contiguous block of {size} bytes")]
    MemoryFragmentation { size: usize },

    // DType errors
    #[error("Unsupported dtype {dtype:?} for format {format:?}")]
    UnsupportedDType {
        dtype: DType,
        format: SparseFormatId,
    },

    #[error("DType mismatch")]
    DTypeMismatch,

    // View errors
    #[error("Slice out of bounds")]
    SliceOutOfBounds,

    #[error("View not supported for format {format:?}")]
    ViewNotSupported { format: SparseFormatId },

    // IO errors
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid sparse tensor file: bad magic number")]
    InvalidMagic,

    #[error("Unsupported file version: {0}")]
    UnsupportedVersion(u16),
}
