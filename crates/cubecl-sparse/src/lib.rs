//! GPU-accelerated sparse tensor operations for CubeCL
//!
//! This crate provides sparse matrix formats and operations optimized for GPU execution via CubeCL.
//!
//! # Supported Formats
//! - **CSR** (Compressed Sparse Row): Efficient row-wise operations
//! - **CSC** (Compressed Sparse Column): Efficient column-wise operations
//! - **COO** (Coordinate): Flexible construction format
//! - **N:M Structured**: Hardware-accelerated sparse tensor cores (Ampere+)
//! - **BSR/BCSC** (Block Sparse): Block-based formats for better memory coalescing
//!
//! # Example
//! ```ignore
//! use cubecl_sparse::prelude::*;
//!
//! // Create sparse tensor from dense
//! let dense = Tensor::random([1024, 1024], Distribution::Uniform(-1.0, 1.0), &device);
//! let sparse = to_sparse(&dense, 0.1, SparseFormatId::Csr, &client);
//!
//! // Sparse matrix multiplication
//! let result = sparse_matmul(&sparse, &dense_matrix, &client);
//! ```

#![warn(missing_docs)]

extern crate alloc;

pub mod algorithm;
pub mod batch;
pub mod convert;
pub mod device;
pub mod dtype;
pub mod error;
pub mod format;
pub mod fusion;
pub mod handle;
pub mod inspect;
pub mod interop;
pub mod io;
pub mod jit;
pub mod memory;
pub mod ops;
pub mod pattern;
pub mod prune;
pub mod view;

/// Prelude module with commonly used types
pub mod prelude {
    pub use crate::error::{SparseError, SparseResult};
    pub use crate::format::{
        BsrStorage, CooStorage, CscStorage, CsrStorage, NMStorage, SparseFormat, SparseFormatId,
        SparseMetadata, SparseStorage,
    };
    pub use crate::handle::{SparseTensor, SparseTensorHandle, SparseTensorHandleRef};
    pub use crate::ops::spmm::{spmm, spmm_with_config, spmm_cached, SpmmConfig};
    pub use crate::ops::traits::SparseOperation;
    pub use crate::prune::strategies::{Pruner, PruningStrategy};
}
