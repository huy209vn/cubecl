//! Test infrastructure and utilities.

pub mod test_utils;
pub mod norm_tests;
pub mod cholesky_tests;
pub mod lu_tests;

// Re-export CPU references for use in other tests
pub use norm_tests::{cpu_norm_l2, cpu_norm_inf, cpu_frobenius_norm};
pub use cholesky_tests::{cpu_cholesky, cpu_matmul_at};
pub use lu_tests::{cpu_lu, cpu_matmul, apply_perm_matrix, extract_l, extract_u};
