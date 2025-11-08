//! Core linear algebra components.
//!
//! This module contains the building blocks for linear algebra operations:
//! - Norms and metrics
//! - Triangular operations
//! - Matrix factorizations (Cholesky, LU)
//! - Iterative refinement
//! - Conditioning and equilibration
//! - Newton-Schulz iterations

pub mod norm;
pub mod triangular;
pub mod cholesky;
pub mod lu;
pub mod iterative;
pub mod conditioning;
pub mod newton_schulz;
