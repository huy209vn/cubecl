//! Panel factorization kernels for Cholesky and LU.
//!
//! These kernels handle small on-diagonal blocks during factorization.
//!
//! ## Performance Notes
//!
//! Panel factorization is inherently sequential (column j depends on 0..j-1).
//! The key to SOTA performance is:
//! 1. Use large block sizes (NB=128-256) so panel time is <5% of total
//! 2. Within panel: parallelize row updates, use shared memory
//! 3. Minimize synchronization points
//!
//! This follows MAGMA's design: make the panel fast enough, but focus on
//! making the trailing matrix update (GEMM/SYRK) as fast as possible.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Unblocked Cholesky factorization (POTRF) for small panels.
///
/// Computes L such that A = L * L^T for a small NB x NB panel.
///
/// ## Algorithm (Cholesky-Crout)
///
/// ```text
/// for j = 0 to NB-1:
///   1. L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]²)
///   2. for i = j+1 to NB-1:
///        L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
/// ```
///
/// ## Parallelization Strategy
///
/// - Column loop (j) is sequential due to dependencies
/// - Row updates (i) within column j are parallel across threads
/// - Use shared memory to cache current diagonal value
/// - Sync only between columns, not within column updates
///
/// ## Launch Configuration
///
/// - Use 1D cube with 32-128 threads
/// - Shared memory: minimal (just current diagonal)
/// - One cube per panel
///
/// ## Arguments
///
/// * `panel` - Input/output panel [nb, nb], lower triangle will contain L
/// * `nb` - Panel size (must be ≤ max supported, typically 128-256)
/// * `eps` - Tolerance for zero diagonal check (SPD verification)
/// * `info` - Output: 0 if success, (j+1) if failure at column j
///
/// ## Errors
///
/// If a diagonal element becomes ≤ eps, sets info[0] = j+1 and returns early.
/// This indicates the matrix is not positive definite.
#[cube(launch)]
pub fn potrf_panel_kernel<F: Float>(
    panel: &mut Tensor<F>,
    nb: u32,
    eps: F,
    info: &mut Tensor<u32>,
) {
    let tid = UNIT_POS;
    let n_threads = CUBE_DIM_X;

    // Shared memory for broadcasting diagonal value
    let mut diag_shared = SharedMemory::<F>::new(1);

    // Initialize info to success
    if tid == 0 {
        info[0] = 0;
    }

    // Column-by-column factorization (sequential due to dependencies)
    for j in 0..nb {
        // === Step 1: Compute diagonal element L[j,j] ===
        // L[j,j] = sqrt(A[j,j] - sum_{k=0}^{j-1} L[j,k]^2)

        if tid == 0 {
            let mut ajj = panel[j * nb + j];

            // Subtract contributions from previous columns
            for k in 0..j {
                let ljk = panel[j * nb + k];
                ajj -= ljk * ljk;
            }

            // Check for positive definiteness
            if ajj <= eps {
                info[0] = j + 1; // Error: not SPD at column j
                diag_shared[0] = F::new(0.0);
            } else {
                let ljj = F::sqrt(ajj);
                panel[j * nb + j] = ljj;
                diag_shared[0] = ljj; // Broadcast to other threads
            }
        }

        // Sync to ensure diagonal is computed and broadcasted
        sync_cube();

        // Check if we failed (not SPD)
        let ljj = diag_shared[0];

        // === Step 2: Update column j below diagonal (PARALLEL) ===
        // L[i,j] = (A[i,j] - sum_{k=0}^{j-1} L[i,k]*L[j,k]) / L[j,j]
        //
        // Parallelize over rows i in [j+1, nb), stride by n_threads
        //
        // Only proceed if SPD check passed (ljj > eps)

        if ljj > eps {
            let mut i = j + 1 + tid;
            while i < nb {
                let mut aij = panel[i * nb + j];

                // Subtract contributions from previous columns
                // TODO: Could use warp shuffle reductions for k loop if needed
                for k in 0..j {
                    let lik = panel[i * nb + k];
                    let ljk = panel[j * nb + k];
                    aij -= lik * ljk;
                }

                // Divide by diagonal
                panel[i * nb + j] = aij / ljj;

                i += n_threads;
            }
        }

        // Sync before moving to next column
        sync_cube();
    }
}

/// Extract diagonal elements from a matrix (used for conditioning estimates).
///
/// Computes min and max diagonal values for condition number estimation.
///
/// ## Arguments
///
/// * `matrix` - Input matrix [nb, nb]
/// * `nb` - Matrix size
/// * `diag_min` - Output: minimum diagonal value
/// * `diag_max` - Output: maximum diagonal value
///
/// ## Launch Configuration
///
/// Use single thread or small reduction kernel.
#[cube(launch)]
pub fn extract_diagonal_minmax<F: Float>(
    matrix: &Tensor<F>,
    nb: u32,
    diag_min: &mut Tensor<F>,
    diag_max: &mut Tensor<F>,
) {
    if ABSOLUTE_POS == 0 {
        let mut min_val = matrix[0];
        let mut max_val = matrix[0];

        for i in 1..nb {
            let d = matrix[i * nb + i];
            if d < min_val {
                min_val = d;
            }
            if d > max_val {
                max_val = d;
            }
        }

        diag_min[0] = min_val;
        diag_max[0] = max_val;
    }
}
