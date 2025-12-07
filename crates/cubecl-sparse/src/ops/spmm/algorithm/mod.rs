//! SpMM algorithm implementations and selection.
//!
//! This module contains the core kernels:
//! - row_split: Thread-per-row for very sparse (TINY)
//! - warp_row: Warp-per-row for moderate sparsity (SMALL)
//! - gather_gemm: Gather then dense GEMM (MEDIUM, our innovation)
//! - gather_tc: Gather then tensor core GEMM (LARGE/HUGE, our innovation)
//! - merge_path: Perfect load balance for irregular distributions

mod row_split;
mod warp_row;
mod gather_gemm;
mod gather_tc;
mod merge_path;

pub use row_split::RowSplitSpMM;
pub use warp_row::{WarpPerRowSpMM, VectorWarpPerRowSpMM};
pub use gather_gemm::GatherGemmSpMM;
pub use gather_tc::GatherTensorCoreSpMM;
pub use merge_path::MergePathSpMM;

use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::{ComputeServer, Handle};
use cubecl_core::ir::StorageType;

use crate::ops::spmm::analysis::{BinStrategy, RowBin};

/// SpMM algorithm enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpmmAlgorithm {
    /// One thread per row. Best for very sparse rows.
    RowSplit,

    /// One warp per row. Best for moderate sparsity.
    WarpPerRow,

    /// Vectorized warp per row. Best for large N.
    VectorWarpPerRow { vec_width: u32 },

    /// Merge-path. Best for irregular distributions.
    MergePath,

    /// Gather then dense GEMM. Our innovation.
    GatherGemm { tile_n: u32 },

    /// Gather then tensor core GEMM. Our innovation + TC.
    GatherTensorCore { tile_m: u32, tile_k: u32, tile_n: u32 },

    /// Direct dense GEMM (for dense tiles).
    DenseGemm,

    /// Specialized banded kernel.
    Banded { bandwidth: u32 },

    /// Block-sparse GEMM.
    BlockSparse { block_size: u32 },
}
