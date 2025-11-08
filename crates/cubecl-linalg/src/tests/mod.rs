//! Test infrastructure and utilities.

pub mod test_utils;
pub mod norm_tests;

// Re-export CPU references for use in other tests
pub use norm_tests::{cpu_norm_l2, cpu_norm_inf, cpu_frobenius_norm};
