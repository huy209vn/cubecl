//! Global policy and auto-tuning cache for linear algebra operations.

use core::fmt;
use cubecl_core::{Runtime, client::ComputeClient};
use once_cell::sync::Lazy;

#[cfg(feature = "std")]
use std::{
    collections::HashMap,
    sync::RwLock,
    string::{String, ToString},
};

#[cfg(not(feature = "std"))]
use alloc::{
    collections::BTreeMap as HashMap,
    string::{String, ToString},
};

#[cfg(not(feature = "std"))]
use spin::RwLock;

use crate::LinalgPrecision;

/// Determinism mode for linear algebra operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterminismMode {
    /// Force deterministic algorithms (may be slower).
    On,
    /// Use deterministic algorithms when performance cost is low.
    Auto,
    /// No determinism guarantees (fastest).
    Off,
}

impl fmt::Display for DeterminismMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeterminismMode::On => write!(f, "on"),
            DeterminismMode::Auto => write!(f, "auto"),
            DeterminismMode::Off => write!(f, "off"),
        }
    }
}

/// Global precision policy for linear algebra operations.
///
/// This controls default behavior for all linalg operations.
/// Can be overridden per-operation if needed.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// Compute dtype ("f32", "f16", "bf16", etc.).
    pub compute_dtype: String,

    /// Accumulation dtype (usually "f32").
    pub accum_dtype: String,

    /// Determinism mode.
    pub determinism: DeterminismMode,

    /// Check conditioning before factorization.
    pub cond_check: bool,

    /// Automatically apply iterative refinement.
    pub iterative_refine: bool,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            compute_dtype: "f32".to_string(),
            accum_dtype: "f32".to_string(),
            determinism: DeterminismMode::Auto,
            cond_check: true,
            iterative_refine: false, // Opt-in for speed
        }
    }
}

static GLOBAL_POLICY: Lazy<RwLock<PrecisionPolicy>> =
    Lazy::new(|| RwLock::new(PrecisionPolicy::default()));

/// Set the global precision policy.
///
/// # Example
///
/// ```ignore
/// use cubecl_linalg::{set_precision_policy, PrecisionPolicy, DeterminismMode};
///
/// let policy = PrecisionPolicy {
///     compute_dtype: "f16".to_string(),
///     accum_dtype: "f32".to_string(),
///     determinism: DeterminismMode::On,
///     cond_check: true,
///     iterative_refine: true,
/// };
/// set_precision_policy(policy);
/// ```
pub fn set_precision_policy(policy: PrecisionPolicy) {
    *GLOBAL_POLICY.write().unwrap() = policy;
}

/// Get the current global precision policy.
pub fn get_precision_policy() -> PrecisionPolicy {
    GLOBAL_POLICY.read().unwrap().clone()
}

/// Block size configuration for factorization algorithms.
///
/// Different block sizes are optimal for different:
/// - Hardware architectures (Tensor Cores, SIMD width)
/// - Precision types (fp16 vs fp32)
/// - Matrix sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockConfig {
    /// Panel factorization block size (NB).
    pub panel_size: usize,

    /// GEMM tile size for trailing matrix updates.
    pub gemm_tile: usize,
}

impl BlockConfig {
    /// Default configuration for fp32 precision.
    pub fn default_fp32() -> Self {
        Self {
            panel_size: 64,
            gemm_tile: 64,
        }
    }

    /// Default configuration for fp16/bf16 precision.
    pub fn default_fp16() -> Self {
        Self {
            panel_size: 128,
            gemm_tile: 128,
        }
    }

    /// Configuration for small matrices (< 512).
    pub fn small() -> Self {
        Self {
            panel_size: 32,
            gemm_tile: 32,
        }
    }

    /// Configuration for large matrices (> 4096).
    pub fn large() -> Self {
        Self {
            panel_size: 256,
            gemm_tile: 256,
        }
    }
}

/// Device-specific tuning cache.
///
/// Caches optimal block configurations per (device, dtype) pair.
#[derive(Default)]
struct TuningCache {
    /// Key: (device_id, dtype_name)
    block_configs: HashMap<(String, String), BlockConfig>,
}

static TUNING_CACHE: Lazy<RwLock<TuningCache>> = Lazy::new(|| RwLock::new(TuningCache::default()));

/// Get the optimal block configuration for a device and precision.
///
/// This function:
/// 1. Checks the cache for existing configuration
/// 2. If not found, selects default based on dtype size
/// 3. Caches the result for future use
///
/// # Example
///
/// ```ignore
/// use cubecl_linalg::{get_block_config, F32Precision};
///
/// let config = get_block_config::<Runtime, F32Precision>(client);
/// println!("Using panel_size: {}", config.panel_size);
/// ```
pub fn get_block_config<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
) -> BlockConfig {
    // Create device fingerprint
    let device_id = format!("{:?}", client.properties());
    let dtype_name = core::any::type_name::<P::EW>();

    // Try to read from cache
    {
        let cache = TUNING_CACHE.read().unwrap();
        if let Some(&config) = cache
            .block_configs
            .get(&(device_id.clone(), dtype_name.to_string()))
        {
            return config;
        }
    }

    // Not in cache - determine default based on dtype size
    let config = if core::mem::size_of::<P::EW>() <= 2 {
        // fp16, bf16: use larger blocks
        BlockConfig::default_fp16()
    } else {
        // fp32, fp64: use smaller blocks
        BlockConfig::default_fp32()
    };

    // Cache it
    {
        let mut cache = TUNING_CACHE.write().unwrap();
        cache
            .block_configs
            .insert((device_id, dtype_name.to_string()), config);
    }

    config
}

/// Manually set the block configuration for a device and precision.
///
/// This overrides auto-tuning. Useful for:
/// - Benchmarking different configurations
/// - Platform-specific optimizations
/// - Testing
///
/// # Example
///
/// ```ignore
/// use cubecl_linalg::{set_block_config, BlockConfig, F32Precision};
///
/// let config = BlockConfig {
///     panel_size: 96,
///     gemm_tile: 96,
/// };
/// set_block_config::<Runtime, F32Precision>(client, config);
/// ```
pub fn set_block_config<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    config: BlockConfig,
) {
    let device_id = format!("{:?}", client.properties());
    let dtype_name = core::any::type_name::<P::EW>();

    let mut cache = TUNING_CACHE.write().unwrap();
    cache
        .block_configs
        .insert((device_id, dtype_name.to_string()), config);
}

/// Clear the tuning cache.
///
/// Useful for testing or when switching between different
/// hardware configurations.
pub fn clear_tuning_cache() {
    let mut cache = TUNING_CACHE.write().unwrap();
    cache.block_configs.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_policy_default() {
        let policy = PrecisionPolicy::default();
        assert_eq!(policy.compute_dtype, "f32");
        assert_eq!(policy.accum_dtype, "f32");
        assert_eq!(policy.determinism, DeterminismMode::Auto);
        assert!(policy.cond_check);
        assert!(!policy.iterative_refine);
    }

    #[test]
    fn test_block_config_defaults() {
        let fp32 = BlockConfig::default_fp32();
        assert_eq!(fp32.panel_size, 64);
        assert_eq!(fp32.gemm_tile, 64);

        let fp16 = BlockConfig::default_fp16();
        assert_eq!(fp16.panel_size, 128);
        assert_eq!(fp16.gemm_tile, 128);
    }

    #[test]
    fn test_determinism_mode_display() {
        assert_eq!(format!("{}", DeterminismMode::On), "on");
        assert_eq!(format!("{}", DeterminismMode::Auto), "auto");
        assert_eq!(format!("{}", DeterminismMode::Off), "off");
    }
}
