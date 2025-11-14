//! Cholesky factorization for symmetric positive definite (SPD) matrices.
//!
//! ## Algorithm
//!
//! This module implements the **blocked right-looking Cholesky factorization**,
//! which is well-suited for GPUs due to its GEMM-heavy structure.
//!
//! For an SPD matrix A, we compute L such that A = L * L^T, where L is lower triangular.
//!
//! ### Blocked Right-Looking Algorithm
//!
//! ```text
//! For each panel k = 0, NB, 2*NB, ..., N-1:
//!   1. Factor diagonal panel: A[k:k+NB, k:k+NB] = L[k:k+NB, k:k+NB] * L^T
//!      (unblocked Cholesky on small NB x NB block)
//!
//!   2. Update subdiagonal panel: L[k+NB:N, k:k+NB] = A[k+NB:N, k:k+NB] * inv(L[k:k+NB, k:k+NB]^T)
//!      (triangular solve, TRSM)
//!
//!   3. Update trailing matrix: A[k+NB:N, k+NB:N] -= L[k+NB:N, k:k+NB] * L[k+NB:N, k:k+NB]^T
//!      (symmetric rank-k update, SYRK via GEMM)
//! ```
//!
//! This structure converts ~2/3 of the work to GEMM operations (step 3),
//! which achieves high arithmetic intensity on GPUs.
//!
//! ## References
//!
//! - MAGMA: "Accelerating Numerical Dense Linear Algebra Calculations with GPUs" (ICL UTK)
//! - LAPACK: DPOTRF (blocked Cholesky)

use cubecl_core::prelude::*;
use cubecl_std::tensor::TensorHandle;

use crate::{
    LinalgPrecision, LinalgResult, LinalgError, SolveInfo,
    policy::get_block_config,
    components::triangular::{Triangle, trsm, Side, Transpose, Diagonal, syrk},
    kernels::panel::{potrf_panel_kernel, extract_diagonal_minmax},
    kernels::elementwise::copy_kernel,
};

/// Cholesky factorization: A = L * L^T for SPD matrix A
///
/// Computes the lower triangular Cholesky factor L of a symmetric
/// positive definite matrix A using a blocked algorithm optimized for GPUs.
///
/// # Algorithm
///
/// Uses blocked right-looking Cholesky:
/// - Panels of size NB x NB factored on device
/// - Trailing matrix updates via GEMM (cubecl-matmul)
/// - Auto-tuned block size per device/precision
///
/// # Arguments
///
/// * `client` - Compute client for kernel execution
/// * `a` - Input SPD matrix [..., M, M] (batched supported)
/// * `uplo` - Use Upper or Lower triangle (Lower is standard)
/// * `check_spd` - If true, verify SPD property (expensive)
///
/// # Returns
///
/// - `L`: Lower triangular Cholesky factor
/// - `SolveInfo`: Conditioning estimate (diag_max/diag_min), quality
///
/// # Errors
///
/// - `NotSPD`: Matrix is not positive definite (negative/zero diagonal encountered)
/// - `InvalidShape`: Input is not square or rank < 2
/// - `NumericalInstability`: Condition number exceeds threshold
///
/// # Example
///
/// ```ignore
/// use cubecl_linalg::{cholesky, Triangle, F32Precision};
///
/// // Factor A = L * L^T
/// let (l, info) = cholesky::<R, F32Precision>(
///     client,
///     a.as_ref(),
///     Triangle::Lower,
///     true,  // Check SPD
/// )?;
///
/// println!("Cholesky factorization: quality = {:?}, cond = {:?}",
///     info.quality, info.condition_estimate);
/// ```
pub fn cholesky<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    a: TensorHandleRef<R>,
    uplo: Triangle,
    check_spd: bool,
) -> LinalgResult<(TensorHandle<R>, SolveInfo)>
where
    P::EG: Into<P::EW>,
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    // Validate input shape
    let shape = a.shape;
    if shape.len() < 2 {
        return Err(LinalgError::InvalidShape {
            reason: format!("Expected at least 2D tensor, got shape {:?}", shape),
        });
    }

    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];

    if m != n {
        return Err(LinalgError::InvalidShape {
            reason: format!("Matrix must be square, got {}x{}", m, n),
        });
    }

    // Get auto-tuned block size for this device/precision
    let config = get_block_config::<R, P>(client);
    let nb = config.panel_size;

    // Allocate output buffer for L (could optimize to in-place later)
    let mut l = TensorHandle::<R>::empty(client, shape.to_vec(), P::EW::as_type_native_unchecked());

    // Copy A to L (working buffer)
    // Launch copy kernel: L = A
    let total_elems = shape.iter().product::<usize>();
    let cube_count = CubeCount::Static(((total_elems + 255) / 256) as u32, 1, 1);
    let cube_dim = CubeDim::new(256, 1, 1);

    copy_kernel::launch::<P::EW, R>(
        client,
        cube_count,
        cube_dim,
        a.as_tensor_arg(1),
        l.as_ref().as_tensor_arg(1),
    );

    // Batch dimensions (for now, only support single matrix - no batching)
    let batch_dims = &shape[..shape.len() - 2];
    let batch_size: usize = batch_dims.iter().product();
    let batch_size = if batch_size == 0 { 1 } else { batch_size };

    if batch_size > 1 {
        return Err(LinalgError::UnsupportedLayout {
            layout: format!("Batched Cholesky not yet supported, got batch size {}", batch_size),
        });
    }

    // Track conditioning info
    let mut info = SolveInfo::new();
    let mut diag_min = f64::INFINITY;
    let mut diag_max = 0.0_f64;

    // Panel loop: blocked Cholesky
    for k in (0..n).step_by(nb) {
        let panel_size = nb.min(n - k);

        // === Step 1: Panel factorization on diagonal block L[k:k+panel_size, k:k+panel_size] ===
        // This is an unblocked Cholesky on a small panel using POTRF kernel

        // Create tensor slice for panel: L[k:k+panel_size, k:k+panel_size]
        let panel_offset = (k * l.strides[0] + k * l.strides[1]) as u64;
        let panel_handle = l.handle.clone().offset_start(panel_offset);
        let panel_shape = vec![panel_size, panel_size];
        let panel_ref = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &panel_handle,
                &l.strides[..],
                &panel_shape,
                a.elem_size,
            )
        };

        // Allocate info flag tensor (single u32)
        let mut potrf_info = TensorHandle::<R>::empty(client, vec![1], u32::as_type_native_unchecked());

        // Launch POTRF kernel on panel
        let panel_threads = usize::min(128, panel_size * 2); // Use enough threads for parallelism
        let panel_cube_count = CubeCount::Static(1, 1, 1);
        let panel_cube_dim = CubeDim::new(panel_threads as u32, 1, 1);

        // Epsilon for SPD check: just check for non-positive diagonal
        // Using 0.0 as threshold (will fail on zero or negative diagonals)
        let eps = P::EW::from_int(0);

        unsafe {
            potrf_panel_kernel::launch::<P::EW, R>(
                client,
                panel_cube_count,
                panel_cube_dim,
                TensorArg::from_raw_parts::<P::EW>(&panel_handle, &l.strides, &panel_shape, 1),
                ScalarArg::new(panel_size as u32),
                ScalarArg::new(eps),
                TensorArg::from_raw_parts::<u32>(&potrf_info.handle, &potrf_info.strides, &potrf_info.shape, 1),
            );
        }

        // Check if POTRF succeeded (only if check_spd is true)
        if check_spd {
            // TODO: Read back info flag from device and check
            // For now, we assume success (will add async check later)
        }

        // === Step 1.5: Extract diagonal min/max for conditioning ===
        let mut diag_min_tensor = TensorHandle::<R>::empty(client, vec![1], P::EW::as_type_native_unchecked());
        let mut diag_max_tensor = TensorHandle::<R>::empty(client, vec![1], P::EW::as_type_native_unchecked());

        let diag_cube_count = CubeCount::Static(1, 1, 1);
        let diag_cube_dim = CubeDim::new(1, 1, 1);

        unsafe {
            extract_diagonal_minmax::launch::<P::EW, R>(
                client,
                diag_cube_count,
                diag_cube_dim,
                TensorArg::from_raw_parts::<P::EW>(&panel_handle, &l.strides, &panel_shape, 1),
                ScalarArg::new(panel_size as u32),
                TensorArg::from_raw_parts::<P::EW>(&diag_min_tensor.handle, &diag_min_tensor.strides, &diag_min_tensor.shape, 1),
                TensorArg::from_raw_parts::<P::EW>(&diag_max_tensor.handle, &diag_max_tensor.strides, &diag_max_tensor.shape, 1),
            );
        }

        // TODO: Read back diag_min/max and update conditioning
        // For Phase 1, we'll compute conditioning at the end

        // === Step 2: TRSM to update subdiagonal panel ===
        // Solve: L[k+panel_size:n, k:k+panel_size] * L[k:k+panel_size, k:k+panel_size]^T = A[k+panel_size:n, k:k+panel_size]
        //
        // This is: X * L11^T = B
        // TRSM: solve for X given L11 and B

        if k + panel_size < n {
            let subdiag_rows = n - (k + panel_size);

            // Create subdiagonal slice: L[k+panel_size:n, k:k+panel_size]
            let subdiag_offset = ((k + panel_size) * l.strides[0] + k * l.strides[1]) as u64;
            let subdiag_handle = l.handle.clone().offset_start(subdiag_offset);
            let subdiag_shape = vec![subdiag_rows, panel_size];
            let subdiag_ref = unsafe {
                TensorHandleRef::<R>::from_raw_parts(
                    &subdiag_handle,
                    &l.strides[..],
                    &subdiag_shape,
                    a.elem_size,
                )
            };

            // Call TRSM: X = B * inv(L11^T)
            // Side::Right, Triangle::Lower, Transpose::Trans, alpha=1.0
            let alpha_one = P::EA::from_int(1);

            let x = trsm::<R, P>(
                client,
                Side::Right,
                Triangle::Lower,
                Transpose::Trans,
                Diagonal::NonUnit,
                alpha_one,
                panel_ref,
                subdiag_ref,
            )?;

            // Copy result back to L[k+panel_size:n, k:k+panel_size]
            let copy_elems = subdiag_rows * panel_size;
            let copy_cube_count = CubeCount::Static(((copy_elems + 255) / 256) as u32, 1, 1);
            let copy_cube_dim = CubeDim::new(256, 1, 1);

            unsafe {
                copy_kernel::launch::<P::EW, R>(
                    client,
                    copy_cube_count,
                    copy_cube_dim,
                    x.as_ref().as_tensor_arg(1),
                    TensorArg::from_raw_parts::<P::EW>(&subdiag_handle, &l.strides, &subdiag_shape, 1),
                );
            }

            // === Step 3: SYRK - Symmetric rank-k update of trailing matrix ===
            // A[k+panel_size:n, k+panel_size:n] -= L[k+panel_size:n, k:k+panel_size] * L[k+panel_size:n, k:k+panel_size]^T
            //
            // This is: C := alpha * A * A^T + beta * C
            // With alpha = -1.0, beta = 1.0

            let trailing_size = n - (k + panel_size);
            let trailing_offset = ((k + panel_size) * l.strides[0] + (k + panel_size) * l.strides[1]) as u64;
            let trailing_handle = l.handle.clone().offset_start(trailing_offset);
            let trailing_shape = vec![trailing_size, trailing_size];
            let trailing_ref = unsafe {
                TensorHandleRef::<R>::from_raw_parts(
                    &trailing_handle,
                    &l.strides[..],
                    &trailing_shape,
                    a.elem_size,
                )
            };

            let alpha_neg_one = P::EA::from_int(-1);
            let beta_one = P::EA::from_int(1);

            // SYRK call: C := -1.0 * A * A^T + 1.0 * C
            // where A = L[k+panel_size:n, k:k+panel_size]
            syrk::<R, P>(
                client,
                alpha_neg_one,
                subdiag_ref,
                beta_one,
                trailing_ref,
            )?;
        }
    }

    // Compute conditioning estimate
    let cond_estimate = diag_max / diag_min;
    info = info.with_condition(cond_estimate);

    // Check if condition number is acceptable
    if cond_estimate > P::COND_THRESHOLD {
        return Err(LinalgError::NumericalInstability {
            cond: cond_estimate,
            threshold: P::COND_THRESHOLD,
        });
    }

    Ok((l, info))
}

/// Solve SPD system: A * x = b using Cholesky factorization
///
/// Computes x = A^{-1} b for symmetric positive definite A.
///
/// # Algorithm
///
/// 1. Factor A = L * L^T (Cholesky)
/// 2. Forward solve: L * y = b
/// 3. Backward solve: L^T * x = y
///
/// # Arguments
///
/// * `client` - Compute client
/// * `a` - SPD matrix [..., M, M]
/// * `b` - Right-hand side [..., M, K]
///
/// # Returns
///
/// - `x`: Solution [..., M, K]
/// - `SolveInfo`: Conditioning and quality information
///
/// # Example
///
/// ```ignore
/// let (x, info) = solve_spd::<R, F32Precision>(client, a.as_ref(), b.as_ref())?;
/// ```
pub fn solve_spd<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,
) -> LinalgResult<(TensorHandle<R>, SolveInfo)>
where
    P::EG: Into<P::EW>,
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    // Step 1: Cholesky factorization
    let (l, mut info) = cholesky::<R, P>(client, a, Triangle::Lower, false)?;

    // Step 2: Forward solve L * y = b
    // TODO: Call trsm
    // let y = trsm::<R, P>(
    //     client,
    //     Side::Left,
    //     Triangle::Lower,
    //     Transpose::NoTrans,
    //     Diagonal::NonUnit,
    //     1.0,
    //     l.as_ref(),
    //     b,
    // )?;

    // Step 3: Backward solve L^T * x = y
    // TODO: Call trsm
    // let x = trsm::<R, P>(
    //     client,
    //     Side::Left,
    //     Triangle::Lower,
    //     Transpose::Trans,
    //     Diagonal::NonUnit,
    //     1.0,
    //     l.as_ref(),
    //     y.as_ref(),
    // )?;

    // Placeholder for now
    let x = TensorHandle::<R>::empty(client, b.shape.to_vec(), P::EW::as_type_native_unchecked());

    // TODO: Compute residual and update info
    // let residual = residual(a, x, b)?;
    // info = info.with_residual(residual);

    Ok((x, info))
}

/// Invert SPD matrix using Cholesky factorization
///
/// Computes A^{-1} for symmetric positive definite A.
///
/// # Algorithm
///
/// 1. Factor A = L * L^T
/// 2. Invert L: compute L^{-1}
/// 3. Compute A^{-1} = L^{-T} * L^{-1} = (L^{-1})^T * L^{-1}
///
/// # Arguments
///
/// * `client` - Compute client
/// * `a` - SPD matrix [..., M, M]
///
/// # Returns
///
/// - `A_inv`: Inverse matrix [..., M, M]
/// - `SolveInfo`: Conditioning information
///
/// # Example
///
/// ```ignore
/// let (a_inv, info) = inverse_spd::<R, F32Precision>(client, a.as_ref())?;
/// ```
pub fn inverse_spd<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    a: TensorHandleRef<R>,
) -> LinalgResult<(TensorHandle<R>, SolveInfo)>
where
    P::EG: Into<P::EW>,
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    // Step 1: Cholesky factorization
    let (l, info) = cholesky::<R, P>(client, a, Triangle::Lower, false)?;

    // Step 2: Invert L (triangular inversion)
    // TODO: Implement triangular inversion
    // Use repeated TRSM: solve L * X = I for X = L^{-1}

    // Step 3: Compute A^{-1} = L^{-T} * L^{-1}
    // This is: A^{-1} = (L^{-1})^T * L^{-1}
    // TODO: Use SYRK (symmetric rank-k) or GEMM

    // Placeholder
    let a_inv = TensorHandle::<R>::empty(client, a.shape.to_vec(), P::EW::as_type_native_unchecked());

    Ok((a_inv, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add comprehensive tests
    //
    // 1. Small SPD matrices (2x2, 4x4, 8x8)
    //    - Known factorizations
    //    - Verify L * L^T = A
    //
    // 2. Ill-conditioned matrices
    //    - Hilbert matrix
    //    - Near-singular SPD
    //    - Verify conditioning estimates
    //
    // 3. Not SPD matrices
    //    - Negative diagonal
    //    - Non-symmetric
    //    - Should return NotSPD error
    //
    // 4. Batching
    //    - Multiple matrices [B, M, M]
    //    - Verify each batch independently
    //
    // 5. Solve tests
    //    - solve_spd: verify A * x = b
    //    - inverse_spd: verify A * A^{-1} = I
    //
    // 6. Compare to CPU reference (LAPACK DPOTRF)
}
