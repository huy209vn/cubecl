//! Tests for Cholesky factorization

use approx::assert_relative_eq;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// CPU reference: Cholesky factorization (unblocked)
///
/// Computes L such that A = L * L^T for SPD matrix A
pub fn cpu_cholesky(a: &[f32], n: usize) -> Result<Vec<f32>, &'static str> {
    let mut l = vec![0.0_f32; n * n];

    // Copy A to L (use lower triangle)
    for i in 0..n {
        for j in 0..=i {
            l[i * n + j] = a[i * n + j];
        }
    }

    // Cholesky-Crout algorithm
    for j in 0..n {
        // Compute L[j,j]
        let mut ajj = l[j * n + j];
        for k in 0..j {
            let ljk = l[j * n + k];
            ajj -= ljk * ljk;
        }

        if ajj <= 0.0 {
            return Err("Matrix is not positive definite");
        }

        let ljj = ajj.sqrt();
        l[j * n + j] = ljj;

        // Compute column j below diagonal
        for i in (j + 1)..n {
            let mut aij = l[i * n + j];
            for k in 0..j {
                aij -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = aij / ljj;
        }
    }

    Ok(l)
}

/// CPU reference: Matrix multiplication C = A * B^T
pub fn cpu_matmul_at(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[j * n + k]; // B^T indexing
            }
            c[i * n + j] = sum;
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::{prelude::*, Runtime};
    use crate::{cholesky, Triangle, F32Precision};

    // Define test runtime
    type TestRuntime = cubecl_cpu::CpuRuntime;

    /// Test Cholesky on 2x2 identity matrix
    fn test_cholesky_2x2_identity_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Identity matrix: [[1, 0], [0, 1]]
        // Expected L: [[1, 0], [0, 1]]
        let values = vec![
            1.0_f32, 0.0_f32,
            0.0_f32, 1.0_f32,
        ];

        // Create input tensor
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[2, 1],   // Row-major strides for 2x2
                &[2, 2],   // Shape
                core::mem::size_of::<f32>(),
            )
        };

        // Compute Cholesky
        let (result, info) = cholesky::<R, F32Precision>(&client, input, Triangle::Lower, false)
            .expect("Cholesky failed");

        // Read result
        let result_bytes = client.read(result.handle.binding());
        let result_values = f32::from_bytes(&result_bytes);

        // Expected: identity (L = I for I = I * I^T)
        assert_relative_eq!(result_values[0], 1.0, epsilon = 1e-5); // L[0,0]
        assert_relative_eq!(result_values[2], 1.0, epsilon = 1e-5); // L[1,1]

        // Off-diagonals should be zero
        assert_relative_eq!(result_values[1].abs(), 0.0, epsilon = 1e-5); // L[0,1] (upper, should be 0)
        assert_relative_eq!(result_values[3].abs(), 0.0, epsilon = 1e-5); // L[1,0]

        println!("✓ 2x2 identity test passed");
    }

    /// Test Cholesky on 2x2 diagonal matrix
    fn test_cholesky_2x2_diagonal_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Diagonal matrix: [[4, 0], [0, 9]]
        // Expected L: [[2, 0], [0, 3]]
        let values = vec![
            4.0_f32, 0.0_f32,
            0.0_f32, 9.0_f32,
        ];

        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[2, 1],
                &[2, 2],
                core::mem::size_of::<f32>(),
            )
        };

        let (result, info) = cholesky::<R, F32Precision>(&client, input, Triangle::Lower, false)
            .expect("Cholesky failed");

        let result_bytes = client.read(result.handle.binding());
        let result_values = f32::from_bytes(&result_bytes);

        // Check diagonal elements
        assert_relative_eq!(result_values[0], 2.0, epsilon = 1e-5); // L[0,0] = sqrt(4)
        assert_relative_eq!(result_values[3], 3.0, epsilon = 1e-5); // L[1,1] = sqrt(9)

        println!("✓ 2x2 diagonal test passed");
    }

    /// Test Cholesky on known 3x3 SPD matrix
    fn test_cholesky_3x3_spd_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // SPD matrix: [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        // This is a well-known test case
        let values = vec![
            4.0_f32, 2.0_f32, 1.0_f32,
            2.0_f32, 5.0_f32, 3.0_f32,
            1.0_f32, 3.0_f32, 6.0_f32,
        ];

        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[3, 1],
                &[3, 3],
                core::mem::size_of::<f32>(),
            )
        };

        let (result, info) = cholesky::<R, F32Precision>(&client, input, Triangle::Lower, false)
            .expect("Cholesky failed");

        let result_bytes = client.read(result.handle.binding());
        let result_values = f32::from_bytes(&result_bytes);

        // Compute CPU reference
        let expected = cpu_cholesky(&values, 3).expect("CPU Cholesky failed");

        // Compare with CPU reference
        for i in 0..3 {
            for j in 0..=i {
                let idx = i * 3 + j;
                assert_relative_eq!(
                    result_values[idx],
                    expected[idx],
                    epsilon = 1e-4,
                    max_relative = 1e-4,
                );
            }
        }

        println!("✓ 3x3 SPD test passed");
    }

    /// Test that L * L^T reconstructs A
    fn test_cholesky_reconstruction_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // 4x4 SPD matrix (constructed as A = B * B^T for random B)
        // Using simple values for reproducibility
        let values = vec![
            10.0_f32,  2.0_f32,  3.0_f32,  1.0_f32,
             2.0_f32,  8.0_f32,  1.0_f32,  2.0_f32,
             3.0_f32,  1.0_f32, 12.0_f32,  4.0_f32,
             1.0_f32,  2.0_f32,  4.0_f32,  7.0_f32,
        ];

        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[4, 1],
                &[4, 4],
                core::mem::size_of::<f32>(),
            )
        };

        let (result, info) = cholesky::<R, F32Precision>(&client, input, Triangle::Lower, false)
            .expect("Cholesky failed");

        let result_bytes = client.read(result.handle.binding());
        let l_values = f32::from_bytes(&result_bytes);

        // Compute L * L^T on CPU
        let reconstructed = cpu_matmul_at(&l_values, &l_values, 4);

        // Compare with original A (only lower triangle is meaningful for SPD)
        for i in 0..4 {
            for j in 0..=i {
                let idx = i * 4 + j;
                assert_relative_eq!(
                    reconstructed[idx],
                    values[idx],
                    epsilon = 1e-3,
                    max_relative = 1e-3,
                );
            }
        }

        println!("✓ 4x4 reconstruction test passed (L*L^T = A)");
    }

    /// Test Cholesky on larger matrix (32x32)
    fn test_cholesky_32x32_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Create 32x32 diagonally dominant SPD matrix
        let n = 32;
        let mut values = vec![0.0_f32; n * n];

        // Make diagonally dominant: A[i,i] = n, A[i,j] = 0.1 for i != j
        for i in 0..n {
            values[i * n + i] = n as f32;
            for j in 0..i {
                values[i * n + j] = 0.1;
                values[j * n + i] = 0.1; // Symmetric
            }
        }

        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[n, 1],
                &[n, n],
                core::mem::size_of::<f32>(),
            )
        };

        let (result, info) = cholesky::<R, F32Precision>(&client, input, Triangle::Lower, false)
            .expect("Cholesky failed");

        let result_bytes = client.read(result.handle.binding());
        let l_values = f32::from_bytes(&result_bytes);

        // Verify reconstruction: L * L^T ≈ A
        let reconstructed = cpu_matmul_at(&l_values, &l_values, n);

        let mut max_error = 0.0_f32;
        for i in 0..n {
            for j in 0..=i {
                let idx = i * n + j;
                let error = (reconstructed[idx] - values[idx]).abs();
                max_error = max_error.max(error);
            }
        }

        // Should be accurate to ~1e-4 for well-conditioned matrix
        assert!(max_error < 1e-3, "Max reconstruction error: {}", max_error);

        println!("✓ 32x32 test passed (max error: {:.2e})", max_error);
    }

    // Actual test functions that run on CPU runtime

    #[test]
    #[cfg(feature = "linalg_tests_cholesky")]
    fn test_cholesky_2x2_identity() {
        let device = Default::default();
        test_cholesky_2x2_identity_impl::<TestRuntime>(&device);
    }

    #[test]
    #[cfg(feature = "linalg_tests_cholesky")]
    fn test_cholesky_2x2_diagonal() {
        let device = Default::default();
        test_cholesky_2x2_diagonal_impl::<TestRuntime>(&device);
    }

    #[test]
    #[cfg(feature = "linalg_tests_cholesky")]
    fn test_cholesky_3x3_spd() {
        let device = Default::default();
        test_cholesky_3x3_spd_impl::<TestRuntime>(&device);
    }

    #[test]
    #[cfg(feature = "linalg_tests_cholesky")]
    fn test_cholesky_reconstruction() {
        let device = Default::default();
        test_cholesky_reconstruction_impl::<TestRuntime>(&device);
    }

    #[test]
    #[cfg(feature = "linalg_tests_cholesky")]
    fn test_cholesky_32x32() {
        let device = Default::default();
        test_cholesky_32x32_impl::<TestRuntime>(&device);
    }
}
