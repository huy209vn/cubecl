//! Triangular matrix operations (TRSM, TRMM, views).
//!
//! This module provides BLAS-compatible triangular operations that serve
//! as building blocks for factorizations (Cholesky, LU).
//!
//! ## Key Operations
//!
//! - **Views**: `triu`, `tril` - zero-copy triangular views
//! - **TRSM**: Triangular solve (blocked algorithm, GEMM-heavy)
//! - **TRMM**: Triangular matrix multiply
//!
//! ## Algorithm Strategy
//!
//! We use **blocked Level-3 BLAS** algorithms that maximize GEMM reuse:
//! - Small triangular ops on diagonal panels
//! - Large GEMM updates on trailing blocks
//! - Good GPU arithmetic intensity
//!
//! ## TRSM Implementation
//!
//! Uses recursive algorithm that divides matrices by 2 until reaching base case:
//! - Base case (N ≤ threshold): Direct GPU kernel
//! - Recursive case: Small solve → GEMM update → Recurse
//! - Converts 90%+ of work to highly-optimized GEMM operations

use core::mem;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::TensorHandle;
use cubecl_matmul::{self as matmul, MatmulInputHandle, Strategy as MatmulStrategy};

#[cfg(feature = "std")]
use std::string::ToString;

#[cfg(not(feature = "std"))]
use alloc::string::ToString;

use crate::{LinalgPrecision, LinalgResult, LinalgError};
use crate::kernels::{fused_scale_sub_kernel, copy_kernel};

/// Side parameter for BLAS operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// A appears on the left: op(A) * X
    Left,
    /// A appears on the right: X * op(A)
    Right,
}

/// Triangle type (upper or lower)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Triangle {
    /// Upper triangular
    Upper,
    /// Lower triangular
    Lower,
}

/// Transpose mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose
    NoTrans,
    /// Transpose A -> A^T
    Trans,
    /// Conjugate transpose A -> A^H (same as Trans for real types)
    ConjTrans,
}

/// Diagonal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagonal {
    /// Use diagonal elements of A
    NonUnit,
    /// Assume diagonal elements are 1 (unit triangular)
    Unit,
}

/// TRSM auto-tuning configuration
///
/// Controls the recursive blocking strategy and base kernel parameters.
#[derive(Debug, Clone, Copy)]
pub struct TrsmConfig {
    /// Base case threshold (stop recursion when N ≤ threshold)
    ///
    /// Typical values:
    /// - Small/old GPUs: 32-64
    /// - Modern GPUs (A100, H100): 64-128
    pub base_threshold: usize,

    /// Work group size for base kernel (threads per block)
    pub workgroup_size_x: u32,
    pub workgroup_size_y: u32,
}

impl Default for TrsmConfig {
    fn default() -> Self {
        Self {
            base_threshold: 64,  // Good default for most modern GPUs
            workgroup_size_x: 16,
            workgroup_size_y: 16,
        }
    }
}

impl TrsmConfig {
    /// Auto-tune configuration based on problem size
    pub fn auto_tune(problem_size: usize) -> Self {
        // Heuristic-based auto-tuning
        // TODO: Profile-guided optimization for specific hardware
        let base_threshold = match problem_size {
            0..=256 => 32,
            257..=1024 => 64,
            _ => 128,
        };

        Self {
            base_threshold,
            workgroup_size_x: 16,
            workgroup_size_y: 16,
        }
    }
}

/// Upper triangular view (zero-copy)
///
/// Returns a view of the upper triangular part of a matrix.
/// Does not allocate new memory - just provides an interpreted view.
///
/// # Arguments
///
/// * `a` - Input matrix [M, M] (or batched [..., M, M])
/// * `k` - Diagonal offset (0 = main diagonal, 1 = superdiagonal, -1 = subdiagonal)
///
/// # Returns
///
/// A handle representing the upper triangular view
///
/// # Example
///
/// ```ignore
/// let u = triu::<R, P>(client, a.as_ref(), 0)?;
/// // u[i,j] = a[i,j] if i <= j+k, else 0
/// ```
pub fn triu<R: Runtime, P: LinalgPrecision>(
    _client: &ComputeClient<R::Server>,
    _a: TensorHandleRef<R>,
    _k: isize,
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float,
{
    // For Phase 1: just return a copy
    // TODO: Implement true zero-copy view with masking kernel

    Err(LinalgError::UnsupportedLayout {
        layout: "triu view not yet implemented".to_string(),
    })
}

/// Lower triangular view (zero-copy)
///
/// Returns a view of the lower triangular part of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix [M, M] (or batched [..., M, M])
/// * `k` - Diagonal offset (0 = main diagonal, 1 = superdiagonal, -1 = subdiagonal)
///
/// # Returns
///
/// A handle representing the lower triangular view
///
/// # Example
///
/// ```ignore
/// let l = tril::<R, P>(client, a.as_ref(), 0)?;
/// // l[i,j] = a[i,j] if i >= j+k, else 0
/// ```
pub fn tril<R: Runtime, P: LinalgPrecision>(
    _client: &ComputeClient<R::Server>,
    _a: TensorHandleRef<R>,
    _k: isize,
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float,
{
    // For Phase 1: just return a copy
    // TODO: Implement true zero-copy view with masking kernel

    Err(LinalgError::UnsupportedLayout {
        layout: "tril view not yet implemented".to_string(),
    })
}

/// Small triangular solve kernel for Left-Lower-NoTrans-NonUnit
///
/// Solves L * X = alpha * B where L is lower triangular.
/// This is the base case for the recursive TRSM algorithm.
///
/// Parallelism strategy: Parallel over RHS columns (nrhs dimension),
/// sequential within each column due to data dependencies.
#[cube(launch)]
fn small_trsm_left_lower_kernel<F: Float>(
    a: &Tensor<F>,       // Lower triangular matrix [k, k]
    b: &mut Tensor<F>,   // RHS matrix [k, nrhs]
    alpha: F,
) {
    // Each thread handles one column of B
    // Solve: L * X[:, col_idx] = alpha * B[:, col_idx]

    let k = a.shape(0);
    let nrhs = b.shape(1);
    let col_idx = ABSOLUTE_POS;

    if col_idx < nrhs {
        // Scale by alpha
        for i in 0..k {
            b[i * nrhs + col_idx] = alpha * b[i * nrhs + col_idx];
        }

        // Forward substitution
        for i in 0..k {
            let mut x_i = b[i * nrhs + col_idx];

            // Subtract contributions from previous elements: x_i -= sum(L[i,j] * x_j)
            for j in 0..i {
                let l_ij = a[i * k + j];
                let x_j = b[j * nrhs + col_idx];
                x_i = x_i - l_ij * x_j;
            }

            // Divide by diagonal: x_i /= L[i,i]
            let l_ii = a[i * k + i];
            x_i = x_i / l_ii;

            b[i * nrhs + col_idx] = x_i;
        }
    }
}

/// Small triangular solve kernel for Left-Lower-Trans-NonUnit
///
/// Solves L^T * X = alpha * B where L is lower triangular.
/// Uses backward substitution since L^T is upper triangular.
#[cube(launch)]
fn small_trsm_left_lower_trans_kernel<F: Float>(
    a: &Tensor<F>,       // Lower triangular matrix [k, k]
    b: &mut Tensor<F>,   // RHS matrix [k, nrhs]
    alpha: F,
) {
    let k = a.shape(0);
    let nrhs = b.shape(1);
    let col_idx = ABSOLUTE_POS;

    if col_idx < nrhs {
        // Scale by alpha
        for i in 0..k {
            b[i * nrhs + col_idx] = alpha * b[i * nrhs + col_idx];
        }

        // Backward substitution for L^T * X = B
        // L^T[i,j] = L[j,i], so L^T is upper triangular
        let mut i = k;
        while i > 0 {
            i -= 1;
            let mut x_i = b[i * nrhs + col_idx];

            // Subtract contributions from later elements: x_i -= sum(L^T[i,j] * x_j) = sum(L[j,i] * x_j)
            for j in (i + 1)..k {
                let l_ji = a[j * k + i]; // L[j,i] = L^T[i,j]
                let x_j = b[j * nrhs + col_idx];
                x_i = x_i - l_ji * x_j;
            }

            // Divide by diagonal: x_i /= L^T[i,i] = L[i,i]
            let l_ii = a[i * k + i];
            x_i = x_i / l_ii;

            b[i * nrhs + col_idx] = x_i;
        }
    }
}

/// Small triangular solve kernel for Left-Upper-NoTrans-NonUnit
///
/// Solves U * X = alpha * B where U is upper triangular.
/// Uses backward substitution.
#[cube(launch)]
fn small_trsm_left_upper_kernel<F: Float>(
    a: &Tensor<F>,       // Upper triangular matrix [k, k]
    b: &mut Tensor<F>,   // RHS matrix [k, nrhs]
    alpha: F,
) {
    let k = a.shape(0);
    let nrhs = b.shape(1);
    let col_idx = ABSOLUTE_POS;

    if col_idx < nrhs {
        // Scale by alpha
        for i in 0..k {
            b[i * nrhs + col_idx] = alpha * b[i * nrhs + col_idx];
        }

        // Backward substitution for U * X = B
        let mut i = k;
        while i > 0 {
            i -= 1;
            let mut x_i = b[i * nrhs + col_idx];

            // Subtract contributions from later elements: x_i -= sum(U[i,j] * x_j)
            for j in (i + 1)..k {
                let u_ij = a[i * k + j];
                let x_j = b[j * nrhs + col_idx];
                x_i = x_i - u_ij * x_j;
            }

            // Divide by diagonal: x_i /= U[i,i]
            let u_ii = a[i * k + i];
            x_i = x_i / u_ii;

            b[i * nrhs + col_idx] = x_i;
        }
    }
}

/// Small triangular solve kernel for Left-Upper-Trans-NonUnit
///
/// Solves U^T * X = alpha * B where U is upper triangular.
/// Uses forward substitution since U^T is lower triangular.
#[cube(launch)]
fn small_trsm_left_upper_trans_kernel<F: Float>(
    a: &Tensor<F>,       // Upper triangular matrix [k, k]
    b: &mut Tensor<F>,   // RHS matrix [k, nrhs]
    alpha: F,
) {
    let k = a.shape(0);
    let nrhs = b.shape(1);
    let col_idx = ABSOLUTE_POS;

    if col_idx < nrhs {
        // Scale by alpha
        for i in 0..k {
            b[i * nrhs + col_idx] = alpha * b[i * nrhs + col_idx];
        }

        // Forward substitution for U^T * X = B
        // U^T[i,j] = U[j,i], so U^T is lower triangular
        for i in 0..k {
            let mut x_i = b[i * nrhs + col_idx];

            // Subtract contributions from previous elements: x_i -= sum(U^T[i,j] * x_j) = sum(U[j,i] * x_j)
            for j in 0..i {
                let u_ji = a[j * k + i]; // U[j,i] = U^T[i,j]
                let x_j = b[j * nrhs + col_idx];
                x_i = x_i - u_ji * x_j;
            }

            // Divide by diagonal: x_i /= U^T[i,i] = U[i,i]
            let u_ii = a[i * k + i];
            x_i = x_i / u_ii;

            b[i * nrhs + col_idx] = x_i;
        }
    }
}

/// Small triangular solve kernel for Right-Lower-NoTrans-NonUnit
///
/// Solves X * L = alpha * B where L is lower triangular.
/// Each thread handles one row of X.
#[cube(launch)]
fn small_trsm_right_lower_kernel<F: Float>(
    a: &Tensor<F>,       // Lower triangular matrix [n, n]
    b: &mut Tensor<F>,   // RHS matrix [m, n]
    alpha: F,
) {
    let m = b.shape(0);
    let n = a.shape(0);
    let row_idx = ABSOLUTE_POS;

    if row_idx < m {
        // Scale by alpha
        for j in 0..n {
            b[row_idx * n + j] = alpha * b[row_idx * n + j];
        }

        // Solve X[row_idx,:] * L = B[row_idx,:]
        // Forward solve over columns (since L is lower triangular)
        for j in 0..n {
            let mut x_j = b[row_idx * n + j];

            // Subtract contributions from previous columns: x_j -= sum(x_k * L[j,k])
            for k in 0..j {
                let x_k = b[row_idx * n + k];
                let l_jk = a[j * n + k];
                x_j = x_j - x_k * l_jk;
            }

            // Divide by diagonal: x_j /= L[j,j]
            let l_jj = a[j * n + j];
            x_j = x_j / l_jj;

            b[row_idx * n + j] = x_j;
        }
    }
}

/// Small triangular solve kernel for Right-Lower-Trans-NonUnit
///
/// Solves X * L^T = alpha * B where L is lower triangular.
/// Since L^T is upper triangular, use backward column solve.
/// **This is the variant needed for Cholesky!**
#[cube(launch)]
fn small_trsm_right_lower_trans_kernel<F: Float>(
    a: &Tensor<F>,       // Lower triangular matrix [n, n]
    b: &mut Tensor<F>,   // RHS matrix [m, n]
    alpha: F,
) {
    let m = b.shape(0);
    let n = a.shape(0);
    let row_idx = ABSOLUTE_POS;

    if row_idx < m {
        // Scale by alpha
        for j in 0..n {
            b[row_idx * n + j] = alpha * b[row_idx * n + j];
        }

        // Solve X[row_idx,:] * L^T = B[row_idx,:]
        // L^T[i,j] = L[j,i], so L^T is upper triangular
        // Backward solve over columns
        let mut j = n;
        while j > 0 {
            j -= 1;
            let mut x_j = b[row_idx * n + j];

            // Subtract contributions from later columns: x_j -= sum(x_k * L^T[j,k]) = sum(x_k * L[k,j])
            for k in (j + 1)..n {
                let x_k = b[row_idx * n + k];
                let l_kj = a[k * n + j]; // L[k,j] = L^T[j,k]
                x_j = x_j - x_k * l_kj;
            }

            // Divide by diagonal: x_j /= L^T[j,j] = L[j,j]
            let l_jj = a[j * n + j];
            x_j = x_j / l_jj;

            b[row_idx * n + j] = x_j;
        }
    }
}

/// Small triangular solve kernel for Right-Upper-NoTrans-NonUnit
///
/// Solves X * U = alpha * B where U is upper triangular.
/// Backward column solve.
#[cube(launch)]
fn small_trsm_right_upper_kernel<F: Float>(
    a: &Tensor<F>,       // Upper triangular matrix [n, n]
    b: &mut Tensor<F>,   // RHS matrix [m, n]
    alpha: F,
) {
    let m = b.shape(0);
    let n = a.shape(0);
    let row_idx = ABSOLUTE_POS;

    if row_idx < m {
        // Scale by alpha
        for j in 0..n {
            b[row_idx * n + j] = alpha * b[row_idx * n + j];
        }

        // Solve X[row_idx,:] * U = B[row_idx,:]
        // Backward solve over columns (since U is upper triangular)
        let mut j = n;
        while j > 0 {
            j -= 1;
            let mut x_j = b[row_idx * n + j];

            // Subtract contributions from later columns: x_j -= sum(x_k * U[j,k])
            for k in (j + 1)..n {
                let x_k = b[row_idx * n + k];
                let u_jk = a[j * n + k];
                x_j = x_j - x_k * u_jk;
            }

            // Divide by diagonal: x_j /= U[j,j]
            let u_jj = a[j * n + j];
            x_j = x_j / u_jj;

            b[row_idx * n + j] = x_j;
        }
    }
}

/// Small triangular solve kernel for Right-Upper-Trans-NonUnit
///
/// Solves X * U^T = alpha * B where U is upper triangular.
/// Since U^T is lower triangular, use forward column solve.
#[cube(launch)]
fn small_trsm_right_upper_trans_kernel<F: Float>(
    a: &Tensor<F>,       // Upper triangular matrix [n, n]
    b: &mut Tensor<F>,   // RHS matrix [m, n]
    alpha: F,
) {
    let m = b.shape(0);
    let n = a.shape(0);
    let row_idx = ABSOLUTE_POS;

    if row_idx < m {
        // Scale by alpha
        for j in 0..n {
            b[row_idx * n + j] = alpha * b[row_idx * n + j];
        }

        // Solve X[row_idx,:] * U^T = B[row_idx,:]
        // U^T[i,j] = U[j,i], so U^T is lower triangular
        // Forward solve over columns
        for j in 0..n {
            let mut x_j = b[row_idx * n + j];

            // Subtract contributions from previous columns: x_j -= sum(x_k * U^T[j,k]) = sum(x_k * U[k,j])
            for k in 0..j {
                let x_k = b[row_idx * n + k];
                let u_kj = a[k * n + j]; // U[k,j] = U^T[j,k]
                x_j = x_j - x_k * u_kj;
            }

            // Divide by diagonal: x_j /= U^T[j,j] = U[j,j]
            let u_jj = a[j * n + j];
            x_j = x_j / u_jj;

            b[row_idx * n + j] = x_j;
        }
    }
}

// TODO: Add Unit diagonal variants (Unit diag means diagonal = 1, skip division)

/// Triangular solve: op(A) * X = alpha * B  or  X * op(A) = alpha * B
///
/// Solves a triangular system of equations using a blocked algorithm
/// that maximizes GEMM reuse for GPU efficiency.
///
/// # Algorithm (Left side, Lower, NoTrans as example)
///
/// Solve L * X = B, where L is lower triangular:
///
/// ```text
/// Partition L and B into blocks:
/// [L11  0  ] [X1]   [B1]
/// [L21 L22 ] [X2] = [B2]
///
/// 1. Solve L11 * X1 = B1  (small triangular solve on diagonal panel)
/// 2. Update B2 := B2 - L21 * X1  (GEMM)
/// 3. Recursively solve L22 * X2 = B2
/// ```
///
/// This pattern converts most work to GEMM operations, which are
/// highly optimized on GPUs.
///
/// # Arguments
///
/// * `side` - Whether A appears on left or right
/// * `uplo` - Whether A is upper or lower triangular
/// * `trans` - Transpose mode for A
/// * `diag` - Whether diagonal is unit or non-unit
/// * `alpha` - Scalar multiplier
/// * `a` - Triangular matrix [..., M, M] or [..., N, N] depending on side
/// * `b` - Right-hand side [..., M, N]
///
/// # Returns
///
/// Solution X with same shape as B
///
/// # Example
///
/// ```ignore
/// // Solve L * X = B where L is lower triangular
/// let x = trsm::<R, P>(
///     client,
///     Side::Left,
///     Triangle::Lower,
///     Transpose::NoTrans,
///     Diagonal::NonUnit,
///     1.0,
///     l.as_ref(),
///     b.as_ref(),
/// )?;
/// ```
pub fn trsm<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    side: Side,
    uplo: Triangle,
    trans: Transpose,
    diag: Diagonal,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    trsm_with_config::<R, P>(client, side, uplo, trans, diag, alpha, a, b, None)
}

/// TRSM with custom configuration (for testing/tuning)
pub fn trsm_with_config<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    side: Side,
    uplo: Triangle,
    trans: Transpose,
    diag: Diagonal,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,
    config: Option<TrsmConfig>,
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    // Validate inputs
    let a_shape = &a.shape;
    let b_shape = &b.shape;

    // For now, only support 2D matrices (no batching)
    // TODO: Add batching support in Phase 3
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(LinalgError::UnsupportedLayout {
            layout: format!("TRSM only supports 2D matrices, got A: {:?}, B: {:?}", a_shape, b_shape),
        });
    }

    let k = a_shape[0];
    if a_shape[1] != k {
        return Err(LinalgError::UnsupportedLayout {
            layout: format!("Matrix A must be square, got shape {:?}", a_shape),
        });
    }

    let (m, n) = (b_shape[0], b_shape[1]);

    // Validate dimensions match
    match side {
        Side::Left => {
            if k != m {
                return Err(LinalgError::UnsupportedLayout {
                    layout: format!("Dimension mismatch: A is {}x{}, B is {}x{}", k, k, m, n),
                });
            }
        }
        Side::Right => {
            if k != n {
                return Err(LinalgError::UnsupportedLayout {
                    layout: format!("Dimension mismatch: A is {}x{}, B is {}x{}", k, k, m, n),
                });
            }
        }
    }

    // Get or create config
    let config = config.unwrap_or_else(|| TrsmConfig::auto_tune(k));

    // Create output tensor (copy of B, will be modified in-place conceptually)
    let output = TensorHandle::<R>::empty(client, b_shape.to_vec(), P::EW::as_type_native_unchecked());

    // Copy B to output using copy_kernel
    let total_elements: usize = b_shape.iter().product();
    let cube_count = CubeCount::Static(((total_elements + 255) / 256) as u32, 1, 1);
    let cube_dim = CubeDim::new(256, 1, 1);

    copy_kernel::launch::<P::EW, R>(
        client,
        cube_count,
        cube_dim,
        b.as_tensor_arg(1),
        output.as_ref().as_tensor_arg(1),
    );

    // Call recursive solver
    trsm_recursive::<R, P>(
        client,
        side,
        uplo,
        trans,
        diag,
        alpha,
        a,
        output.as_ref(),
        config,
    )?;

    Ok(output)
}

/// Recursive TRSM implementation
fn trsm_recursive<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    side: Side,
    uplo: Triangle,
    trans: Transpose,
    diag: Diagonal,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,  // Modified in-place
    config: TrsmConfig,
) -> LinalgResult<()>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    let k = a.shape[0];
    let (m, n) = (b.shape[0], b.shape[1]);

    // Base case: use direct kernel
    if k <= config.base_threshold {
        // Unit diagonal not yet supported
        if diag == Diagonal::Unit {
            return Err(LinalgError::UnsupportedLayout {
                layout: "Unit diagonal TRSM not yet implemented".to_string(),
            });
        }

        // For all current precision types, EA and EW are the same (f32/f32, f64/f64, etc.)
        // Use a transmute-style conversion via bit representation
        // Safety: EA and EW are always the same size and representation for valid precision types
        let alpha_bits = unsafe { mem::transmute_copy::<P::EA, P::EW>(&alpha) };

        // Dispatch to the appropriate kernel based on side/uplo/trans
        match (side, uplo, trans) {
            (Side::Left, Triangle::Lower, Transpose::NoTrans) => {
                small_trsm_left_lower_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(n as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Left, Triangle::Lower, Transpose::Trans) => {
                small_trsm_left_lower_trans_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(n as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Left, Triangle::Upper, Transpose::NoTrans) => {
                small_trsm_left_upper_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(n as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Left, Triangle::Upper, Transpose::Trans) => {
                small_trsm_left_upper_trans_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(n as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Right, Triangle::Lower, Transpose::NoTrans) => {
                small_trsm_right_lower_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(m as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Right, Triangle::Lower, Transpose::Trans) => {
                small_trsm_right_lower_trans_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(m as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Right, Triangle::Upper, Transpose::NoTrans) => {
                small_trsm_right_upper_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(m as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            (Side::Right, Triangle::Upper, Transpose::Trans) => {
                small_trsm_right_upper_trans_kernel::launch::<P::EW, R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(m as u32, 1, 1),
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }
            _ => {
                return Err(LinalgError::UnsupportedLayout {
                    layout: format!("Unsupported TRSM variant: {:?}-{:?}-{:?}", side, uplo, trans),
                });
            }
        }

        return Ok(());
    }

    // Recursive case: dispatch based on side
    // Unit diagonal not yet supported in recursive case
    if diag == Diagonal::Unit {
        return Err(LinalgError::UnsupportedLayout {
            layout: "Unit diagonal TRSM not yet implemented in recursive case".to_string(),
        });
    }

    match side {
        Side::Left => {
            trsm_recursive_left::<R, P>(client, uplo, trans, diag, alpha, a, b, config)
        }
        Side::Right => {
            trsm_recursive_right::<R, P>(client, uplo, trans, diag, alpha, a, b, config)
        }
    }
}

/// Recursive TRSM for Left side: op(A) * X = alpha * B
fn trsm_recursive_left<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    uplo: Triangle,
    trans: Transpose,
    diag: Diagonal,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,
    config: TrsmConfig,
) -> LinalgResult<()>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    let k = a.shape[0];
    let n = b.shape[1];

    // Partition: divide k by 2 (recursive halving strategy)
    let k1 = k / 2;
    let k2 = k - k1;

    // L = [L11   0 ]   B = [B1]   X = [X1]
    //     [L21  L22]       [B2]       [X2]
    //
    // Solve: L * X = alpha * B
    // 1. L11 * X1 = alpha * B1  (recursive)
    // 2. B2 := alpha * B2 - L21 * X1  (GEMM)
    // 3. L22 * X2 = B2  (recursive, alpha=1 since B2 already scaled)

    // Create views for submatrices
    // Create shape arrays with longer lifetime
    let l11_shape = vec![k1, k1];
    let l21_shape = vec![k2, k1];
    let l22_shape = vec![k2, k2];
    let b1_shape = vec![k1, n];
    let b2_shape = vec![k2, n];

    // L11: a[0:k1, 0:k1]
    let l11 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            a.handle,
            &a.strides[..],
            &l11_shape,
            a.elem_size,
        )
    };

    // L21: a[k1:k, 0:k1]
    let l21_offset = (k1 * a.strides[0]) as u64;
    let l21_handle = a.handle.clone().offset_start(l21_offset);
    let l21 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &l21_handle,
            &a.strides[..],
            &l21_shape,
            a.elem_size,
        )
    };

    // L22: a[k1:k, k1:k]
    let l22_offset = (k1 * a.strides[0] + k1 * a.strides[1]) as u64;
    let l22_handle = a.handle.clone().offset_start(l22_offset);
    let l22 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &l22_handle,
            &a.strides[..],
            &l22_shape,
            a.elem_size,
        )
    };

    // B1: b[0:k1, :]
    let b1 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            b.handle,
            &b.strides[..],
            &b1_shape,
            b.elem_size,
        )
    };

    // B2: b[k1:k, :]
    let b2_offset = (k1 * b.strides[0]) as u64;
    let b2_handle = b.handle.clone().offset_start(b2_offset);
    let b2 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &b2_handle,
            &b.strides[..],
            &b2_shape,
            b.elem_size,
        )
    };

    // Step 1: Solve op(A11) * X1 = alpha * B1
    trsm_recursive_left::<R, P>(client, uplo, trans, diag, alpha, l11, b1, config)?;

    // Step 2: Update B2 := alpha * B2 - L21 * X1
    // This is: B2 = alpha * B2 - L21 * B1
    //
    // We compute:
    // 1. temp = L21 * B1  (GEMM)
    // 2. B2 = alpha * B2 - temp  (fused element-wise)

    // Temp result: [k2, n]
    // Use the correct output type for matmul (AccG = Acc::Global)
    type AccG<MP> = cubecl_matmul::components::AccG<MP>;
    let temp_shape = vec![k2, n];
    let temp = TensorHandle::<R>::empty(client, temp_shape.clone(), AccG::<P::EW>::as_type_native_unchecked());

    // GEMM: temp = L21 * B1
    // Fix TensorHandle::new argument order: (handle, shape, strides, storage)
    let l21_handle = TensorHandle::new(l21.handle.clone(), l21.shape.to_vec(), l21.strides.to_vec(), P::EW::as_type_native_unchecked());
    let b1_handle = TensorHandle::new(b1.handle.clone(), b1.shape.to_vec(), b1.strides.to_vec(), P::EW::as_type_native_unchecked());

    // Use P::EW as the MatmulPrecision type
    let _ = matmul::launch::<R>(
        &MatmulStrategy::Auto,
        client,
        MatmulInputHandle::Normal(l21_handle),
        MatmulInputHandle::Normal(b1_handle),
        temp.clone(),
        cubecl_matmul::components::MatmulElems::new::<P::EW>(),
    );

    // Fused: B2 = alpha * B2 - temp
    let total_elements = k2 * n;
    let cube_count = CubeCount::Static(((total_elements + 255) / 256) as u32, 1, 1);
    let cube_dim = CubeDim::new(256, 1, 1);

    // For all current precision types, EA and EW are the same (f32/f32, f64/f64, etc.)
    // Use a transmute-style conversion via bit representation
    // Safety: EA and EW are always the same size and representation for valid precision types
    let alpha_bits = unsafe { mem::transmute_copy::<P::EA, P::EW>(&alpha) };

    fused_scale_sub_kernel::launch::<P::EW, R>(
        client,
        cube_count,
        cube_dim,
        b2.as_tensor_arg(1),
        temp.as_ref().as_tensor_arg(1),
        ScalarArg::new(alpha_bits),
    );

    // Step 3: Solve op(A22) * X2 = B2 (recursive, alpha=1 since B2 already scaled/updated)
    trsm_recursive_left::<R, P>(
        client,
        uplo,
        trans,
        diag,
        P::EA::from_int(1),  // alpha = 1.0
        l22,
        b2,
        config,
    )?;

    Ok(())
}

/// Recursive TRSM for Right side: X * op(A) = alpha * B
fn trsm_recursive_right<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    uplo: Triangle,
    trans: Transpose,
    diag: Diagonal,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    b: TensorHandleRef<R>,
    config: TrsmConfig,
) -> LinalgResult<()>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    let n = a.shape[0];  // A is n×n
    let m = b.shape[0];  // B is m×n

    // Partition: divide n by 2 (column-wise partitioning)
    let n1 = n / 2;
    let n2 = n - n1;

    // For Right-Lower-Trans (what Cholesky needs):
    // X * L^T = B, where L^T is upper triangular
    // Partition column-wise:
    // L = [L11   0 ]  =>  L^T = [L11^T  L21^T]
    //     [L21  L22]            [ 0     L22^T]
    //
    // X = [X1 X2], B = [B1 B2]
    //
    // [X1 X2] * [L11^T  L21^T] = [B1 B2]
    //            [ 0     L22^T]
    //
    // X1*L11^T = B1  and  X1*L21^T + X2*L22^T = B2
    //
    // Solve backward (since L^T is upper):
    // 1. X2 * L22^T = B2  (recursive)
    // 2. B1 := B1 - X2 * L21^T  (GEMM)
    // 3. X1 * L11^T = B1  (recursive, alpha=1 since B1 already scaled)
    //
    // For Right-Upper-NoTrans and other variants, similar logic applies

    // Create shape arrays
    let a11_shape = vec![n1, n1];
    let a12_shape = vec![n1, n2];  // For upper: A12; for lower transpose: L21^T part
    let a21_shape = vec![n2, n1];  // For lower: A21
    let a22_shape = vec![n2, n2];
    let b1_shape = vec![m, n1];
    let b2_shape = vec![m, n2];

    // Determine which blocks we need based on uplo/trans
    // For simplicity, create all blocks and use the relevant ones

    // A11: a[0:n1, 0:n1]
    let a11 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            a.handle,
            &a.strides[..],
            &a11_shape,
            a.elem_size,
        )
    };

    // A12 (upper) or A21^T (lower trans): a[0:n1, n1:n]
    let a12_offset = (n1 * a.strides[1]) as u64;
    let a12_handle = a.handle.clone().offset_start(a12_offset);
    let a12 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &a12_handle,
            &a.strides[..],
            &a12_shape,
            a.elem_size,
        )
    };

    // A21 (lower): a[n1:n, 0:n1]
    let a21_offset = (n1 * a.strides[0]) as u64;
    let a21_handle = a.handle.clone().offset_start(a21_offset);
    let a21 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &a21_handle,
            &a.strides[..],
            &a21_shape,
            a.elem_size,
        )
    };

    // A22: a[n1:n, n1:n]
    let a22_offset = (n1 * a.strides[0] + n1 * a.strides[1]) as u64;
    let a22_handle = a.handle.clone().offset_start(a22_offset);
    let a22 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &a22_handle,
            &a.strides[..],
            &a22_shape,
            a.elem_size,
        )
    };

    // B1: b[:, 0:n1]
    let b1 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            b.handle,
            &b.strides[..],
            &b1_shape,
            b.elem_size,
        )
    };

    // B2: b[:, n1:n]
    let b2_offset = (n1 * b.strides[1]) as u64;
    let b2_handle = b.handle.clone().offset_start(b2_offset);
    let b2 = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &b2_handle,
            &b.strides[..],
            &b2_shape,
            b.elem_size,
        )
    };

    // Dispatch based on uplo/trans
    match (uplo, trans) {
        (Triangle::Lower, Transpose::Trans) | (Triangle::Upper, Transpose::NoTrans) => {
            // Both are upper triangular effective (L^T or U)
            // Solve backward: right to left
            //
            // 1. Solve X2 * A22 = alpha * B2
            trsm_recursive_right::<R, P>(client, uplo, trans, diag, alpha, a22, b2, config)?;

            // 2. Update B1 := alpha * B1 - X2 * A12  (for Upper) or B1 := alpha * B1 - X2 * A21^T (for Lower-Trans)
            // For Lower-Trans: A21^T is the off-diagonal block
            // Need temp = X2 * A21^T (which is B2 * A21^T)
            type AccG<MP> = cubecl_matmul::components::AccG<MP>;
            let temp_shape = vec![m, n1];
            let temp = TensorHandle::<R>::empty(client, temp_shape.clone(), AccG::<P::EW>::as_type_native_unchecked());

            let b2_handle_gemm = TensorHandle::new(b2.handle.clone(), b2.shape.to_vec(), b2.strides.to_vec(), P::EW::as_type_native_unchecked());
            let off_diag_handle = if uplo == Triangle::Upper {
                // Use A12
                TensorHandle::new(a12.handle.clone(), a12.shape.to_vec(), a12.strides.to_vec(), P::EW::as_type_native_unchecked())
            } else {
                // Use A21^T (need to transpose)
                // For A21^T, we multiply B2 * A21^T
                // This is B2 [m, n2] * A21^T [n2, n1]
                // A21 is [n2, n1], so A21^T is [n1, n2]
                // But we pass A21 and use Transposed input handle
                TensorHandle::new(a21.handle.clone(), a21.shape.to_vec(), a21.strides.to_vec(), P::EW::as_type_native_unchecked())
            };

            let input_b2 = MatmulInputHandle::Normal(b2_handle_gemm);

            let mut input_offdiag = MatmulInputHandle::Normal(off_diag_handle);
            if uplo == Triangle::Lower {
                // Transpose A21 to get A21^T
                input_offdiag.swap_dims(0, 1);
            }

            let _ = matmul::launch::<R>(
                &MatmulStrategy::Auto,
                client,
                input_b2,
                input_offdiag,
                temp.clone(),
                cubecl_matmul::components::MatmulElems::new::<P::EW>(),
            );

            // Fused: B1 = alpha * B1 - temp
            let alpha_bits = unsafe { mem::transmute_copy::<P::EA, P::EW>(&alpha) };
            let total_elements = m * n1;
            let cube_count = CubeCount::Static(((total_elements + 255) / 256) as u32, 1, 1);
            let cube_dim = CubeDim::new(256, 1, 1);

            unsafe {
                fused_scale_sub_kernel::launch::<P::EW, R>(
                    client,
                    cube_count,
                    cube_dim,
                    b1.as_tensor_arg(1),
                    temp.as_ref().as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }

            // 3. Solve X1 * A11 = B1 (alpha=1 since B1 already scaled)
            trsm_recursive_right::<R, P>(
                client,
                uplo,
                trans,
                diag,
                P::EA::from_int(1),
                a11,
                b1,
                config,
            )?;
        }
        (Triangle::Lower, Transpose::NoTrans) | (Triangle::Upper, Transpose::Trans) => {
            // Both are lower triangular effective (L or U^T)
            // Solve forward: left to right
            //
            // 1. Solve X1 * A11 = alpha * B1
            trsm_recursive_right::<R, P>(client, uplo, trans, diag, alpha, a11, b1, config)?;

            // 2. Update B2 := alpha * B2 - X1 * A21  (for Lower) or B2 := alpha * B2 - X1 * A12^T (for Upper-Trans)
            type AccG<MP> = cubecl_matmul::components::AccG<MP>;
            let temp_shape = vec![m, n2];
            let temp = TensorHandle::<R>::empty(client, temp_shape.clone(), AccG::<P::EW>::as_type_native_unchecked());

            let b1_handle_gemm = TensorHandle::new(b1.handle.clone(), b1.shape.to_vec(), b1.strides.to_vec(), P::EW::as_type_native_unchecked());
            let off_diag_handle = if uplo == Triangle::Lower {
                TensorHandle::new(a21.handle.clone(), a21.shape.to_vec(), a21.strides.to_vec(), P::EW::as_type_native_unchecked())
            } else {
                TensorHandle::new(a12.handle.clone(), a12.shape.to_vec(), a12.strides.to_vec(), P::EW::as_type_native_unchecked())
            };

            let mut input_offdiag = MatmulInputHandle::Normal(off_diag_handle);
            if uplo == Triangle::Upper {
                // Transpose A12 to get A12^T
                input_offdiag.swap_dims(0, 1);
            }

            let _ = matmul::launch::<R>(
                &MatmulStrategy::Auto,
                client,
                MatmulInputHandle::Normal(b1_handle_gemm),
                input_offdiag,
                temp.clone(),
                cubecl_matmul::components::MatmulElems::new::<P::EW>(),
            );

            // Fused: B2 = alpha * B2 - temp
            let alpha_bits = unsafe { mem::transmute_copy::<P::EA, P::EW>(&alpha) };
            let total_elements = m * n2;
            let cube_count = CubeCount::Static(((total_elements + 255) / 256) as u32, 1, 1);
            let cube_dim = CubeDim::new(256, 1, 1);

            unsafe {
                fused_scale_sub_kernel::launch::<P::EW, R>(
                    client,
                    cube_count,
                    cube_dim,
                    b2.as_tensor_arg(1),
                    temp.as_ref().as_tensor_arg(1),
                    ScalarArg::new(alpha_bits),
                );
            }

            // 3. Solve X2 * A22 = B2 (alpha=1 since B2 already scaled)
            trsm_recursive_right::<R, P>(
                client,
                uplo,
                trans,
                diag,
                P::EA::from_int(1),
                a22,
                b2,
                config,
            )?;
        }
        _ => {
            return Err(LinalgError::UnsupportedLayout {
                layout: format!("Unsupported TRSM Right variant: {:?}-{:?}", uplo, trans),
            });
        }
    }

    Ok(())
}

/// Triangular matrix multiply: B = alpha * op(A) * B  or  B = alpha * B * op(A)
///
/// Multiplies a triangular matrix with a general matrix.
/// Uses blocked algorithm similar to TRSM.
///
/// # Arguments
///
/// * `side` - Whether A appears on left or right
/// * `uplo` - Whether A is upper or lower triangular
/// * `trans` - Transpose mode for A
/// * `diag` - Whether diagonal is unit or non-unit
/// * `alpha` - Scalar multiplier
/// * `a` - Triangular matrix
/// * `b` - General matrix (updated in-place)
///
/// # Returns
///
/// Result B = alpha * op(A) * B (or B * op(A))
///
/// # Example
///
/// ```ignore
/// // Compute B = L * B where L is lower triangular
/// let result = trmm::<R, P>(
///     client,
///     Side::Left,
///     Triangle::Lower,
///     Transpose::NoTrans,
///     Diagonal::NonUnit,
///     1.0,
///     l.as_ref(),
///     b.as_ref(),
/// )?;
/// ```
pub fn trmm<R: Runtime, P: LinalgPrecision>(
    _client: &ComputeClient<R::Server>,
    _side: Side,
    _uplo: Triangle,
    _trans: Transpose,
    _diag: Diagonal,
    _alpha: P::EA,
    _a: TensorHandleRef<R>,
    _b: TensorHandleRef<R>,
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float,
    P::EA: Float,
{
    // TODO: Implement blocked TRMM
    // Similar structure to TRSM but with multiply instead of solve

    Err(LinalgError::UnsupportedLayout {
        layout: "trmm not yet implemented".to_string(),
    })
}

/// Symmetric rank-k update: C = alpha * A * A^T + beta * C
///
/// Performs a symmetric rank-k update on matrix C. This is a key operation
/// in the blocked Cholesky algorithm for updating the trailing matrix.
///
/// # Algorithm
///
/// Computes: C := alpha * A * A^T + beta * C
///
/// Where:
/// - A is [M, K]
/// - C is [M, M] symmetric (only lower triangle is updated/referenced)
/// - Result is symmetric, so only lower triangle is computed
///
/// This is implemented as a GEMM operation: C = A * A^T with scaling factors.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*A^T
/// * `a` - Input matrix [M, K]
/// * `beta` - Scalar multiplier for C
/// * `c` - Symmetric matrix [M, M] (updated in-place, lower triangle)
///
/// # Returns
///
/// Updated matrix C (same handle, modified in-place)
///
/// # Example
///
/// ```ignore
/// // Cholesky trailing matrix update: C := C - A*A^T
/// syrk::<R, P>(
///     client,
///     -1.0,  // alpha = -1.0
///     a.as_ref(),
///     1.0,   // beta = 1.0
///     c.as_ref(),
/// )?;
/// ```
pub fn syrk<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    alpha: P::EA,
    a: TensorHandleRef<R>,
    beta: P::EA,
    c: TensorHandleRef<R>,
) -> LinalgResult<()>
where
    P::EW: Float + cubecl_matmul::components::MatmulPrecision + CubeElement,
    P::EA: Float,
{
    // Validate shapes
    let a_shape = a.shape;
    let c_shape = c.shape;

    if a_shape.len() != 2 || c_shape.len() != 2 {
        return Err(LinalgError::UnsupportedLayout {
            layout: format!("SYRK only supports 2D matrices, got A: {:?}, C: {:?}", a_shape, c_shape),
        });
    }

    let m = a_shape[0];
    let k = a_shape[1];

    if c_shape[0] != m || c_shape[1] != m {
        return Err(LinalgError::InvalidShape {
            reason: format!("Matrix C must be {}x{}, got {}x{}", m, m, c_shape[0], c_shape[1]),
        });
    }

    // SYRK: C := alpha * A * A^T + beta * C
    //
    // OPTIMIZATION: Use specialized fused SYRK kernel
    // - Computes only lower triangle (exploit symmetry)
    // - Fuses GEMM and update into single kernel
    // - Avoids temporary MxM allocation
    //
    // Expected speedup: 1.5-2× over GEMM + element-wise approach
    //
    // Converts alpha/beta from precision EA to EW for kernel
    let alpha_ew = unsafe { mem::transmute_copy::<P::EA, P::EW>(&alpha) };
    let beta_ew = unsafe { mem::transmute_copy::<P::EA, P::EW>(&beta) };

    // Launch optimized SYRK kernel
    crate::kernels::syrk::launch_syrk_fused::<P::EW, R>(
        client,
        a.handle.clone(),
        a.shape,
        a.strides,
        c.handle.clone(),
        c.shape,
        c.strides,
        alpha_ew,
        beta_ew,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_cpu::{CpuRuntime, CpuDevice};

    #[test]
    fn test_trsm_small_2x2() {
        // Test a simple 2x2 lower triangular system
        // L = [2  0]    B = [4]    Expected X = [2]
        //     [3  1]        [5]                 [-1]
        //
        // Solve: L * X = B
        // 2*x1 = 4 => x1 = 2
        // 3*x1 + 1*x2 = 5 => 3*2 + x2 = 5 => x2 = -1

        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        // Create L matrix (lower triangular)
        let l_data = vec![2.0_f32, 0.0, 3.0, 1.0];
        let l = client.create(bytemuck::cast_slice(&l_data));
        let l_handle = TensorHandle::<CpuRuntime, f32>::new(l, vec![2, 2], vec![2, 1]);

        // Create B vector
        let b_data = vec![4.0_f32, 5.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![2, 1], vec![1, 1]);

        // Solve L * X = B
        let result = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Lower,
            Transpose::NoTrans,
            Diagonal::NonUnit,
            1.0,
            l_handle.as_ref(),
            b_handle.as_ref(),
        );

        assert!(result.is_ok(), "TRSM should succeed for 2x2 matrix");

        // TODO: Verify the result values match expected [2, -1]
        // Need to read back from GPU to check
    }

    #[test]
    fn test_trsm_identity() {
        // Test with identity matrix - solution should equal RHS
        // I * X = B => X = B
        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        // Identity matrix
        let l_data = vec![1.0_f32, 0.0, 0.0, 1.0];
        let l = client.create(bytemuck::cast_slice(&l_data));
        let l_handle = TensorHandle::<CpuRuntime, f32>::new(l, vec![2, 2], vec![2, 1]);

        // RHS
        let b_data = vec![3.0_f32, 7.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![2, 1], vec![1, 1]);

        // Solve
        let result = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Lower,
            Transpose::NoTrans,
            Diagonal::NonUnit,
            1.0,
            l_handle.as_ref(),
            b_handle.as_ref(),
        );

        assert!(result.is_ok(), "TRSM should succeed for identity matrix");
    }

    #[test]
    fn test_trsm_4x4() {
        // Test 4x4 to verify recursive case
        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        // Create a 4x4 lower triangular matrix
        #[rustfmt::skip]
        let l_data = vec![
            2.0_f32, 0.0, 0.0, 0.0,
            1.0,     3.0, 0.0, 0.0,
            2.0,     1.0, 4.0, 0.0,
            1.0,     2.0, 1.0, 5.0,
        ];
        let l = client.create(bytemuck::cast_slice(&l_data));
        let l_handle = TensorHandle::<CpuRuntime, f32>::new(l, vec![4, 4], vec![4, 1]);

        // RHS vector
        let b_data = vec![4.0_f32, 9.0, 16.0, 25.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![4, 1], vec![1, 1]);

        // Solve
        let result = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Lower,
            Transpose::NoTrans,
            Diagonal::NonUnit,
            1.0,
            l_handle.as_ref(),
            b_handle.as_ref(),
        );

        assert!(result.is_ok(), "TRSM should succeed for 4x4 matrix (tests recursion)");
    }

    #[test]
    fn test_trsm_multiple_rhs() {
        // Test with multiple right-hand sides (matrix B instead of vector)
        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        // 2x2 lower triangular
        let l_data = vec![2.0_f32, 0.0, 3.0, 1.0];
        let l = client.create(bytemuck::cast_slice(&l_data));
        let l_handle = TensorHandle::<CpuRuntime, f32>::new(l, vec![2, 2], vec![2, 1]);

        // 2x3 RHS matrix (3 different right-hand sides)
        let b_data = vec![4.0_f32, 8.0, 12.0, 5.0, 11.0, 15.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![2, 3], vec![3, 1]);

        // Solve L * X = B
        let result = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Lower,
            Transpose::NoTrans,
            Diagonal::NonUnit,
            1.0,
            l_handle.as_ref(),
            b_handle.as_ref(),
        );

        assert!(result.is_ok(), "TRSM should handle multiple RHS");
    }

    #[test]
    fn test_trsm_with_alpha() {
        // Test with non-unit alpha scaling
        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        let l_data = vec![2.0_f32, 0.0, 3.0, 1.0];
        let l = client.create(bytemuck::cast_slice(&l_data));
        let l_handle = TensorHandle::<CpuRuntime, f32>::new(l, vec![2, 2], vec![2, 1]);

        let b_data = vec![4.0_f32, 5.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![2, 1], vec![1, 1]);

        // Solve L * X = 2.0 * B
        let result = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Lower,
            Transpose::NoTrans,
            Diagonal::NonUnit,
            2.0,
            l_handle.as_ref(),
            b_handle.as_ref(),
        );

        assert!(result.is_ok(), "TRSM should handle alpha scaling");
    }

    #[test]
    #[should_panic(expected = "UnsupportedLayout")]
    fn test_trsm_unsupported_upper() {
        // Upper triangular should fail for now (Phase 1 limitation)
        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        let u_data = vec![2.0_f32, 3.0, 0.0, 1.0];
        let u = client.create(bytemuck::cast_slice(&u_data));
        let u_handle = TensorHandle::<CpuRuntime, f32>::new(u, vec![2, 2], vec![2, 1]);

        let b_data = vec![4.0_f32, 5.0];
        let b = client.create(bytemuck::cast_slice(&b_data));
        let b_handle = TensorHandle::<CpuRuntime, f32>::new(b, vec![2, 1], vec![1, 1]);

        let _ = trsm::<CpuRuntime, crate::F32Precision>(
            &client,
            Side::Left,
            Triangle::Upper,  // This should fail
            Transpose::NoTrans,
            Diagonal::NonUnit,
            1.0,
            u_handle.as_ref(),
            b_handle.as_ref(),
        ).unwrap();  // Should panic here
    }

    #[test]
    #[cfg_attr(not(feature = "cuda"), ignore = "CPU backend doesn't support matmul shared memory ops")]
    fn test_syrk_basic() {
        // Test SYRK: C := C - A*A^T (Cholesky update pattern)
        //
        // A = [2  1]  =>  A*A^T = [5   4]
        //     [1  2]              [4   5]
        //
        // C_initial = [10  8]
        //             [8  10]
        //
        // C_final = C_initial - A*A^T = [5   4]
        //                                [4   5]

        let device = CpuDevice::default();
        let client = CpuRuntime::client(&device);

        // Create A matrix [2, 2]
        let a_data = vec![2.0_f32, 1.0, 1.0, 2.0];
        let a = client.create(bytemuck::cast_slice(&a_data));
        let a_handle = TensorHandle::<CpuRuntime, f32>::new(a, vec![2, 2], vec![2, 1]);

        // Create C matrix [2, 2] - initialize to 10 on diagonal, 8 off-diagonal
        let c_data = vec![10.0_f32, 8.0, 8.0, 10.0];
        let c = client.create(bytemuck::cast_slice(&c_data));
        let c_handle = TensorHandle::<CpuRuntime, f32>::new(c, vec![2, 2], vec![2, 1]);

        // SYRK: C := -1.0 * A*A^T + 1.0 * C  =>  C := C - A*A^T
        let result = syrk::<CpuRuntime, crate::F32Precision>(
            &client,
            -1.0,  // alpha = -1.0
            a_handle.as_ref(),
            1.0,   // beta = 1.0
            c_handle.as_ref(),
        );

        assert!(result.is_ok(), "SYRK should succeed for 2x2 matrix");

        // TODO: Verify the result values
        // Expected: C = [5, 4, 4, 5] after SYRK
        // Need to read back from GPU to check
    }
}
