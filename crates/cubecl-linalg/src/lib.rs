//! # CubeCL Linear Algebra
//!
//! GPU-accelerated linear algebra operations for CubeCL.
//!
//! ## Features
//!
//! - **Norms**: L2, L-infinity, Frobenius, spectral norm estimation
//! - **Triangular operations**: TRSM, TRMM with BLAS-compatible API
//! - **Cholesky factorization**: Blocked algorithm for SPD matrices
//! - **LU factorization**: Partial pivoting for general matrices
//! - **Solvers**: High-level solve() and inverse() with auto-dispatch
//! - **Newton-Schulz**: Fast matrix inverse-sqrt and orthogonalization
//! - **Batching**: Native batched operations throughout
//! - **Mixed precision**: Configurable compute/accumulation precision
//!
//! ## Example
//!
//! ```ignore
//! use cubecl_linalg::{solve, F32Precision};
//!
//! // Solve A*x = b using auto-dispatch (Cholesky or LU)
//! let (x, info) = solve::<Runtime, F32Precision>(client, a, b)?;
//! println!("Solved with quality: {:?}", info.quality);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

mod error;
mod precision;
mod policy;

/// Core linear algebra components
pub mod components;

/// Low-level GPU kernels
pub mod kernels;

/// High-level solver wrappers
pub mod solvers;

/// Tests for linear algebra operations
#[cfg(feature = "export_tests")]
pub mod tests;

// Re-export public API
pub use error::*;
pub use precision::*;
pub use policy::*;

// Re-export key components
pub use components::norm::*;
pub use components::triangular::*;
pub use components::cholesky::*;
pub use components::lu::*;
pub use components::newton_schulz::*;
pub use components::conditioning::*;
pub use components::iterative::*;

// Re-export solvers
pub use solvers::solve::*;
pub use solvers::inverse::*;
