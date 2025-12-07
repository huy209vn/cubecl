//! Matrix analysis for algorithm selection.
//!
//! This module performs the analysis phase:
//! 1. Global statistics computation
//! 2. Tile classification (optional, for large matrices)
//! 3. Row binning for load-balanced execution

mod statistics;
mod binning;
mod tile;

pub use statistics::{MatrixStatistics, RowStatistics, analyze_csr};
pub use binning::{
    RowBinning, RowBin, BinId, BinStrategy,
    create_binning, select_bin_strategy,
};
pub use tile::{
    TileDecomposition, TileInfo, TileClass, TileRowStats,
    decompose_tiles,
};
