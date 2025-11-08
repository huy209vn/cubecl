//! Precision trait system for linear algebra operations.
//!
//! This module defines how different numeric types are used across
//! the memory hierarchy (global memory, working memory, accumulation).

use cubecl_core::prelude::{CubePrimitive, Float, Numeric};

/// Precision specification for linear algebra operations.
///
/// This trait allows configuring different numeric types for:
/// - Global memory (input/output tensors)
/// - Working precision (intermediate computations, factorization)
/// - Accumulation precision (inner products, GEMM accumulation)
///
/// # Example
///
/// ```ignore
/// use cubecl_linalg::{LinalgPrecision, F32Precision};
///
/// fn my_solver<P: LinalgPrecision>() {
///     // Uses P::EG for global memory
///     // Uses P::EW for working computations
///     // Uses P::EA for accumulation
/// }
/// ```
pub trait LinalgPrecision: Send + Sync + 'static {
    /// Global memory type (input/output tensors).
    type EG: Numeric + CubePrimitive;

    /// Working precision (computations, factorization).
    type EW: Float + CubePrimitive;

    /// Accumulation precision (inner products, GEMM accumulation).
    type EA: Float + CubePrimitive;

    /// Maximum allowed condition number before warning.
    const COND_THRESHOLD: f64 = 1e12;

    /// Tolerance for iterative methods.
    const ITERATIVE_TOL: f64 = 1e-6;

    /// Maximum iterations for iterative methods.
    const MAX_ITERS: usize = 100;
}

/// Standard single precision (f32).
#[derive(Debug, Clone, Copy)]
pub struct F32Precision;

impl LinalgPrecision for F32Precision {
    type EG = f32;
    type EW = f32;
    type EA = f32;
}

/// Standard double precision (f64).
#[derive(Debug, Clone, Copy)]
pub struct F64Precision;

impl LinalgPrecision for F64Precision {
    type EG = f64;
    type EW = f64;
    type EA = f64;
    const ITERATIVE_TOL: f64 = 1e-12;
}

/// Mixed precision: fp16 input, fp32 working & accumulation.
///
/// This configuration is ideal for:
/// - Fast factorization using Tensor Cores
/// - High accuracy accumulation
/// - Memory bandwidth savings
#[derive(Debug, Clone, Copy)]
pub struct F16MixedPrecision;

impl LinalgPrecision for F16MixedPrecision {
    type EG = half::f16;
    type EW = f32;
    type EA = f32;
    const ITERATIVE_TOL: f64 = 1e-4; // Relaxed for f16
    const COND_THRESHOLD: f64 = 1e8; // More conservative
}

/// Mixed precision: bf16 input, fp32 working & accumulation.
///
/// Similar to F16MixedPrecision but using bfloat16, which has:
/// - Same exponent range as fp32 (better for gradients)
/// - Less mantissa precision than fp16
#[derive(Debug, Clone, Copy)]
pub struct BF16MixedPrecision;

impl LinalgPrecision for BF16MixedPrecision {
    type EG = half::bf16;
    type EW = f32;
    type EA = f32;
    const ITERATIVE_TOL: f64 = 1e-4;
    const COND_THRESHOLD: f64 = 1e8;
}

/// Pure fp16 precision (for testing/benchmarking).
///
/// Warning: May have limited accuracy for ill-conditioned problems.
#[derive(Debug, Clone, Copy)]
pub struct F16Precision;

impl LinalgPrecision for F16Precision {
    type EG = half::f16;
    type EW = half::f16;
    type EA = half::f16;
    const ITERATIVE_TOL: f64 = 1e-3;
    const COND_THRESHOLD: f64 = 1e6;
}

/// Pure bf16 precision (for testing/benchmarking).
///
/// Warning: May have limited accuracy for ill-conditioned problems.
#[derive(Debug, Clone, Copy)]
pub struct BF16Precision;

impl LinalgPrecision for BF16Precision {
    type EG = half::bf16;
    type EW = half::bf16;
    type EA = half::bf16;
    const ITERATIVE_TOL: f64 = 1e-3;
    const COND_THRESHOLD: f64 = 1e6;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_constants() {
        assert_eq!(F32Precision::ITERATIVE_TOL, 1e-6);
        assert_eq!(F64Precision::ITERATIVE_TOL, 1e-12);
        assert_eq!(F16MixedPrecision::ITERATIVE_TOL, 1e-4);
        assert_eq!(BF16MixedPrecision::ITERATIVE_TOL, 1e-4);
    }

    #[test]
    fn test_condition_thresholds() {
        assert_eq!(F32Precision::COND_THRESHOLD, 1e12);
        assert_eq!(F16MixedPrecision::COND_THRESHOLD, 1e8);
    }
}
