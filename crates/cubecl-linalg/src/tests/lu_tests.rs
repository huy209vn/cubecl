//! Tests for LU factorization with partial pivoting

use approx::assert_relative_eq;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// CPU reference: LU factorization with partial pivoting
///
/// Computes P, L, U such that P*A = L*U
/// Returns (lu, perm) where lu contains L (unit diagonal) in lower triangle
/// and U in upper triangle
pub fn cpu_lu(a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<usize>), &'static str> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = lu[k * n + k].abs();

        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        if max_val < 1e-10 {
            return Err("Matrix is singular");
        }

        // Swap rows in LU and permutation
        if pivot_row != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }

        // Eliminate column k
        let pivot = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor; // Store L factor

            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    Ok((lu, perm))
}

/// Apply permutation to matrix: result = P * A
pub fn apply_perm_matrix(a: &[f32], perm: &[usize], n: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = a[perm[i] * n + j];
        }
    }
    result
}

/// Extract L from LU (with unit diagonal)
pub fn extract_l(lu: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0; // Unit diagonal
        for j in 0..i {
            l[i * n + j] = lu[i * n + j];
        }
    }
    l
}

/// Extract U from LU
pub fn extract_u(lu: &[f32], n: usize) -> Vec<f32> {
    let mut u = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = lu[i * n + j];
        }
    }
    u
}

/// CPU matrix multiplication C = A * B
pub fn cpu_matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
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
    use crate::{lu_factor, LUConfig, F32Precision};

    // Define test runtime
    type TestRuntime = cubecl_cpu::CpuRuntime;

    /// Test LU on 4x4 identity matrix
    fn test_lu_4x4_identity_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);
        let n = 4;

        // Create identity matrix
        let mut a_data = vec![0.0_f32; n * n];
        for i in 0..n {
            a_data[i * n + i] = 1.0;
        }

        // Create tensor
        let a_handle = client.create_from_slice(f32::as_bytes(&a_data));
        let a = cubecl_std::tensor::TensorHandle::new(
            a_handle,
            vec![n, n],
            vec![n, 1],
            f32::as_type_native_unchecked(),
        );

        // Compute LU factorization
        let config = LUConfig::default();
        let (lu, perm, info) = lu_factor::<R, F32Precision>(&client, a.as_ref(), config)
            .expect("LU factorization failed");

        // Read result
        let lu_bytes = client.read_one(lu.handle.clone());
        let lu_data = f32::from_bytes(&lu_bytes);

        // For identity matrix, LU should be identity and perm should be identity
        for i in 0..n {
            assert_eq!(perm[i], i, "Permutation should be identity");
        }

        // Diagonal should be 1.0
        for i in 0..n {
            assert_relative_eq!(lu_data[i * n + i], 1.0, epsilon = 1e-5);
        }

        // Off-diagonal should be ~0
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_relative_eq!(lu_data[i * n + j], 0.0, epsilon = 1e-5);
                }
            }
        }
    }

    /// Test LU on simple 4x4 matrix
    fn test_lu_4x4_simple_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);
        let n = 4;

        // Simple test matrix (from LAPACK test suite style)
        #[rustfmt::skip]
        let a_data = vec![
            2.0, 1.0, 1.0, 0.0,
            4.0, 3.0, 3.0, 1.0,
            8.0, 7.0, 9.0, 5.0,
            6.0, 7.0, 9.0, 8.0,
        ];

        // Create tensor
        let a_handle = client.create_from_slice(f32::as_bytes(&a_data));
        let a = cubecl_std::tensor::TensorHandle::new(
            a_handle,
            vec![n, n],
            vec![n, 1],
            f32::as_type_native_unchecked(),
        );

        // Compute GPU LU factorization
        let config = LUConfig::default();
        let (lu_gpu, perm_gpu, info) = lu_factor::<R, F32Precision>(&client, a.as_ref(), config)
            .expect("LU factorization failed");

        // Read GPU result
        let lu_gpu_bytes = client.read_one(lu_gpu.handle.clone());
        let lu_gpu_data = f32::from_bytes(&lu_gpu_bytes);

        // Compute CPU reference
        let (lu_cpu, perm_cpu) = cpu_lu(&a_data, n).expect("CPU LU failed");

        // Verify P*A = L*U reconstruction
        let pa = apply_perm_matrix(&a_data, &perm_gpu, n);
        let l = extract_l(lu_gpu_data, n);
        let u = extract_u(lu_gpu_data, n);
        let lu_product = cpu_matmul(&l, &u, n);

        // Check that P*A = L*U
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    pa[i * n + j],
                    lu_product[i * n + j],
                    epsilon = 1e-4,
                    max_relative = 1e-3
                );
            }
        }
    }

    /// Test LU on diagonal matrix
    fn test_lu_diagonal_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);
        let n = 8;

        // Diagonal matrix with values 1, 2, 3, ..., 8
        let mut a_data = vec![0.0_f32; n * n];
        for i in 0..n {
            a_data[i * n + i] = (i + 1) as f32;
        }

        // Create tensor
        let a_handle = client.create_from_slice(f32::as_bytes(&a_data));
        let a = cubecl_std::tensor::TensorHandle::new(
            a_handle,
            vec![n, n],
            vec![n, 1],
            f32::as_type_native_unchecked(),
        );

        // Compute LU factorization
        let config = LUConfig::default();
        let (lu, perm, info) = lu_factor::<R, F32Precision>(&client, a.as_ref(), config)
            .expect("LU factorization failed");

        // Read result
        let lu_bytes = client.read_one(lu.handle.clone());
        let lu_data = f32::from_bytes(&lu_bytes);

        // For diagonal matrix:
        // - L should be identity (unit diagonal, zeros below)
        // - U should be the original diagonal
        // - P should select largest diagonal first (so reverse order)

        // Check U diagonal contains the original values (possibly reordered)
        let mut u_diag: Vec<f32> = (0..n).map(|i| lu_data[i * n + i]).collect();
        u_diag.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending

        for i in 0..n {
            let expected = (n - i) as f32; // Reverse order: 8, 7, 6, ..., 1
            assert_relative_eq!(u_diag[i], expected, epsilon = 1e-5);
        }

        // Off-diagonal below should be ~0 (L has unit diagonal, zeros below)
        for i in 1..n {
            for j in 0..i {
                assert_relative_eq!(lu_data[i * n + j], 0.0, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_lu_4x4_identity() {
        test_lu_4x4_identity_impl::<TestRuntime>(&Default::default());
    }

    #[test]
    fn test_lu_4x4_simple() {
        test_lu_4x4_simple_impl::<TestRuntime>(&Default::default());
    }

    #[test]
    fn test_lu_diagonal() {
        test_lu_diagonal_impl::<TestRuntime>(&Default::default());
    }
}
