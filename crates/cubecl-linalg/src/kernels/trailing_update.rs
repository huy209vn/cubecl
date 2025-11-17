//! Trailing matrix updates for blocked LU factorization.
//!
//! After factoring a panel, we need to update the rest of the matrix:
//! 1. TRSM: Update panel to the right (U12 = L11^-1 * A12)
//! 2. GEMM: Update trailing submatrix (A22 -= L21 * U12)

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Update columns to the right of factored panel using triangular solve
///
/// Solves L * U12 = A12 where L is unit lower triangular
/// This is equivalent to forward substitution for each column
///
/// # Arguments
/// * `matrix` - Full matrix [N, N], updated in-place
/// * `n` - Matrix dimension
/// * `k_start` - Panel start row/col
/// * `nb` - Panel size
/// * `n_cols` - Number of columns to update (from k_end to n)
#[cube(launch)]
pub fn trsm_panel_right_kernel<F: Float>(
    matrix: &mut Tensor<F>,
    n: u32,
    k_start: u32,
    nb: u32,
    n_cols: u32,
) {
    // Each thread handles one column in the trailing region
    let col_idx = ABSOLUTE_POS;

    if col_idx < n_cols {
        let global_col = k_start + nb + col_idx;

        // Forward substitution: solve L * x = b for column global_col
        // L is unit lower triangular in [k_start:k_start+nb, k_start:k_start+nb]

        for i in 0..nb {
            let global_row = k_start + i;

            // Subtract contributions from previous rows
            for j in 0..i {
                let prev_row = k_start + j;
                let l_val = matrix[global_row * n + prev_row];
                let x_val = matrix[prev_row * n + global_col];
                matrix[global_row * n + global_col] -= l_val * x_val;
            }

            // L has unit diagonal, so no division needed
        }
    }
}

/// Update trailing submatrix using GEMM: A22 -= L21 * U12
///
/// # Arguments
/// * `matrix` - Full matrix [N, N], updated in-place
/// * `n` - Matrix dimension
/// * `k_start` - Panel start
/// * `nb` - Panel size
/// * `m_rows` - Rows in trailing matrix (n - k_end)
/// * `n_cols` - Cols in trailing matrix (n - k_end)
#[cube(launch)]
pub fn gemm_trailing_kernel<F: Float>(
    matrix: &mut Tensor<F>,
    n: u32,
    k_start: u32,
    nb: u32,
    m_rows: u32,
    n_cols: u32,
) {
    // Each thread computes one element of the trailing matrix
    let idx = ABSOLUTE_POS;
    let total_elems = m_rows * n_cols;

    if idx < total_elems {
        let i = idx / n_cols;  // Row within trailing matrix
        let j = idx % n_cols;  // Col within trailing matrix

        let global_row = k_start + nb + i;
        let global_col = k_start + nb + j;

        // Compute dot product: -sum(L21[i,k] * U12[k,j])
        let mut sum = F::new(0.0);

        for k in 0..nb {
            let l_col = k_start + k;  // Column in L21
            let u_row = k_start + k;  // Row in U12

            let l_val = matrix[global_row * n + l_col];
            let u_val = matrix[u_row * n + global_col];

            sum += l_val * u_val;
        }

        matrix[global_row * n + global_col] -= sum;
    }
}
