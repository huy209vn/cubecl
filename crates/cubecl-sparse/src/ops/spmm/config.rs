//! SpMM configuration types.

use super::algorithm::SpmmAlgorithm;

/// Top-level SpMM configuration.
#[derive(Clone, Debug)]
pub struct SpmmConfig {
    /// Tile decomposition settings.
    pub tile: TileConfig,

    /// Row binning settings.
    pub binning: BinningConfig,

    /// Algorithm selection overrides.
    pub algorithm: AlgorithmConfig,

    /// Precision settings.
    pub precision: PrecisionConfig,
}

impl Default for SpmmConfig {
    fn default() -> Self {
        Self {
            tile: TileConfig::default(),
            binning: BinningConfig::default(),
            algorithm: AlgorithmConfig::default(),
            precision: PrecisionConfig::default(),
        }
    }
}

/// Tile decomposition configuration.
#[derive(Clone, Debug)]
pub struct TileConfig {
    /// Enable tile classification (disable for small matrices).
    pub enabled: bool,

    /// Tile size in row dimension.
    pub tile_m: u32,

    /// Tile size in column dimension.
    pub tile_k: u32,

    /// Minimum matrix elements to enable tiling.
    /// Default: 256K elements (e.g., 512×512)
    pub min_elements_for_tiling: u64,

    /// Density threshold for classifying tile as dense.
    /// Default: 0.25 (25%)
    pub dense_threshold: f32,

    /// Diagonal fraction threshold for banded classification.
    /// Default: 0.80 (80% of nnz within bandwidth)
    pub banded_threshold: f32,

    /// Block density threshold for block-sparse detection.
    /// Default: 0.50 (50% of block filled)
    pub block_threshold: f32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tile_m: 256,
            tile_k: 256,
            min_elements_for_tiling: 256 * 256 * 4,
            dense_threshold: 0.25,
            banded_threshold: 0.80,
            block_threshold: 0.50,
        }
    }
}

/// Row binning configuration.
#[derive(Clone, Debug)]
pub struct BinningConfig {
    /// Bin boundaries (nnz thresholds).
    /// Default: [0, 8, 32, 128, 512]
    /// Maps to: EMPTY, TINY, SMALL, MEDIUM, LARGE, HUGE
    pub boundaries: Vec<u32>,

    /// Enable tensor core path for eligible bins.
    pub enable_tensor_cores: bool,

    /// Minimum rows in bin to use Gather-GEMM.
    /// Below this, falls back to simpler algorithms.
    pub min_rows_for_gather: u32,

    /// CV (coefficient of variation) threshold to force merge-path within a bin.
    /// High CV indicates irregular distribution needing perfect load balance.
    pub merge_path_cv_threshold: f32,
}

impl Default for BinningConfig {
    fn default() -> Self {
        Self {
            boundaries: vec![0, 8, 32, 128, 512],
            enable_tensor_cores: true,
            min_rows_for_gather: 16,
            merge_path_cv_threshold: 1.5,
        }
    }
}

/// Algorithm selection configuration.
#[derive(Clone, Debug)]
pub struct AlgorithmConfig {
    /// Force a specific algorithm (bypass auto-selection).
    pub force_algorithm: Option<SpmmAlgorithm>,

    /// Enable Gather-GEMM path (core innovation).
    pub enable_gather_gemm: bool,

    /// Enable tile classification.
    pub enable_tile_classification: bool,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            force_algorithm: None,
            enable_gather_gemm: true,
            enable_tile_classification: true,
        }
    }
}

/// Precision configuration.
#[derive(Clone, Debug)]
pub struct PrecisionConfig {
    /// Accumulator precision (may differ from input).
    pub accumulator: AccumulatorPrecision,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            accumulator: AccumulatorPrecision::default(),
        }
    }
}

/// Accumulator precision options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AccumulatorPrecision {
    /// Same as input precision.
    #[default]
    SameAsInput,
    /// Always use f32 accumulator (higher accuracy).
    F32,
    /// Always use f64 accumulator (highest accuracy).
    F64,
}
