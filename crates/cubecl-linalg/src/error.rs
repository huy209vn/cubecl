//! Error types and diagnostic information for linear algebra operations.

use core::fmt;
use thiserror::Error;

#[cfg(feature = "std")]
use std::string::String;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Errors that can occur during linear algebra operations.
#[derive(Error, Debug, Clone)]
pub enum LinalgError {
    /// Matrix is not symmetric positive definite.
    #[error("Matrix is not symmetric positive definite")]
    NotSPD,

    /// Singular matrix detected during factorization.
    #[error("Singular matrix: pivot at index {index} is zero or too small (value: {value})")]
    SingularPivot {
        /// Index of the singular pivot
        index: usize,
        /// Value of the pivot
        value: f64,
    },

    /// Iterative method did not converge.
    #[error("Iterative method did not converge after {iters} iterations (residual: {residual})")]
    NoConvergence {
        /// Number of iterations attempted
        iters: usize,
        /// Final residual norm
        residual: f64,
    },

    /// Numerical instability detected.
    #[error("Numerical instability detected: condition number {cond} exceeds threshold {threshold}")]
    NumericalInstability {
        /// Estimated condition number
        cond: f64,
        /// Threshold that was exceeded
        threshold: f64,
    },

    /// Invalid tensor shape.
    #[error("Invalid shape: {reason}")]
    InvalidShape {
        /// Description of the shape error
        reason: String,
    },

    /// Incompatible batch dimensions.
    #[error("Incompatible batch dimensions: lhs={lhs:?}, rhs={rhs:?}")]
    BatchMismatch {
        /// Left-hand side batch dimensions
        lhs: Vec<usize>,
        /// Right-hand side batch dimensions
        rhs: Vec<usize>,
    },

    /// Unsupported tensor layout.
    #[error("Unsupported layout: {layout}")]
    UnsupportedLayout {
        /// Description of the unsupported layout
        layout: String,
    },

    /// Matrix multiplication operation failed.
    #[error("Matmul operation failed: {0}")]
    MatmulFailure(String),

    /// Reduction operation failed.
    #[error("Reduction operation failed: {0}")]
    ReduceFailure(String),
}

/// Result type for linear algebra operations.
pub type LinalgResult<T> = Result<T, LinalgError>;

/// Quality indicator for numerical solve results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveQuality {
    /// High confidence in accuracy (condition < 1e6).
    Excellent,
    /// Acceptable, but monitor residuals (condition < 1e10).
    Good,
    /// Numerically challenging, use with caution (condition >= 1e10).
    Marginal,
}

impl fmt::Display for SolveQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveQuality::Excellent => write!(f, "excellent"),
            SolveQuality::Good => write!(f, "good"),
            SolveQuality::Marginal => write!(f, "marginal"),
        }
    }
}

/// Diagnostic information returned with successful solves.
///
/// This structure provides detailed information about the numerical
/// properties and quality of a linear algebra operation.
#[derive(Debug, Clone)]
pub struct SolveInfo {
    /// Estimated condition number (1-norm).
    pub condition_estimate: Option<f64>,

    /// Relative residual norm ||b - Ax|| / ||b||.
    pub relative_residual: Option<f64>,

    /// Number of iterations used (for iterative methods).
    pub iterations: usize,

    /// Whether result was refined using mixed precision.
    pub was_refined: bool,

    /// Numerical quality indicator.
    pub quality: SolveQuality,
}

impl SolveInfo {
    /// Create a new SolveInfo with default values.
    pub fn new() -> Self {
        Self {
            condition_estimate: None,
            relative_residual: None,
            iterations: 0,
            was_refined: false,
            quality: SolveQuality::Good,
        }
    }

    /// Set the condition number estimate and update quality.
    pub fn with_condition(mut self, cond: f64) -> Self {
        self.condition_estimate = Some(cond);
        self.quality = match cond {
            c if c < 1e6 => SolveQuality::Excellent,
            c if c < 1e10 => SolveQuality::Good,
            _ => SolveQuality::Marginal,
        };
        self
    }

    /// Set the relative residual.
    pub fn with_residual(mut self, res: f64) -> Self {
        self.relative_residual = Some(res);
        self
    }

    /// Set the number of iterations.
    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = iters;
        self
    }

    /// Mark that iterative refinement was used.
    pub fn with_refinement(mut self) -> Self {
        self.was_refined = true;
        self
    }

    /// Manually set the quality indicator.
    pub fn with_quality(mut self, quality: SolveQuality) -> Self {
        self.quality = quality;
        self
    }
}

impl Default for SolveInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SolveInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SolveInfo {{ quality: {}", self.quality)?;

        if let Some(cond) = self.condition_estimate {
            write!(f, ", cond: {:.2e}", cond)?;
        }

        if let Some(res) = self.relative_residual {
            write!(f, ", residual: {:.2e}", res)?;
        }

        if self.iterations > 0 {
            write!(f, ", iters: {}", self.iterations)?;
        }

        if self.was_refined {
            write!(f, ", refined")?;
        }

        write!(f, " }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_info_quality() {
        let info = SolveInfo::new().with_condition(1e5);
        assert_eq!(info.quality, SolveQuality::Excellent);

        let info = SolveInfo::new().with_condition(1e8);
        assert_eq!(info.quality, SolveQuality::Good);

        let info = SolveInfo::new().with_condition(1e12);
        assert_eq!(info.quality, SolveQuality::Marginal);
    }

    #[test]
    fn test_solve_info_builder() {
        let info = SolveInfo::new()
            .with_condition(1e7)
            .with_residual(1e-10)
            .with_iterations(5)
            .with_refinement();

        assert_eq!(info.condition_estimate, Some(1e7));
        assert_eq!(info.relative_residual, Some(1e-10));
        assert_eq!(info.iterations, 5);
        assert!(info.was_refined);
        assert_eq!(info.quality, SolveQuality::Good);
    }
}
