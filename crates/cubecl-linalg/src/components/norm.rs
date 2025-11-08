//! Norms and metric operations.
//!
//! This module provides various norm computations that are used for:
//! - Convergence checking in iterative methods
//! - Conditioning estimation
//! - Residual computation
//! - Spectral analysis
//!
//! All norms return 0-D device tensors (scalars on GPU) to avoid
//! unnecessary CPU synchronization.

use cubecl_core::prelude::*;
use cubecl_reduce::{reduce};
use cubecl_reduce::instructions::MaxAbs;
use cubecl_std::tensor::TensorHandle;

use crate::{LinalgPrecision, LinalgResult, LinalgError};
use crate::kernels::{sqrt_kernel, SumSquared};

/// L2 norm (Euclidean norm) of a vector: ||x||_2 = sqrt(sum(x_i^2))
///
/// Returns a 0-D tensor containing the norm value on the device.
///
/// **Optimized**: Uses fused SumSquared reducer (2 kernel launches instead of 3).
///
/// # Arguments
///
/// * `client` - Compute client for kernel execution
/// * `x` - Input vector (any rank, flattened)
///
/// # Example
///
/// ```ignore
/// let norm = vector_norm_l2::<R, F32Precision>(client, x.as_ref())?;
/// // norm is a 0-D tensor on device
/// ```
pub fn vector_norm_l2<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    x: TensorHandleRef<R>,
) -> LinalgResult<TensorHandle<R, P::EA>>
where
    P::EW: Float,
    P::EA: Float,
{
    // 1. SumSquared reducer: computes sum(x^2) in a single fused kernel
    // No separate square_kernel needed - squaring is computed inline during reduction
    let sum_shape = vec![1];
    let sum_output = TensorHandle::<R, P::EA>::empty(client, sum_shape.clone());

    reduce::<R, (P::EA, P::EA), P::EA, SumSquared>(
        client,
        x,
        sum_output.as_ref(),
        0, // Reduce along dimension 0
        None, // Auto strategy
        (),
    ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

    // 2. Take square root (still separate kernel, but unavoidable)
    let norm_output = TensorHandle::<R, P::EA>::empty(client, sum_shape);

    sqrt_kernel::launch::<P::EA, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        sum_output.as_arg(1),
        norm_output.as_arg(1),
    );

    Ok(norm_output)
}

/// L-infinity norm of a vector: ||x||_∞ = max(|x_i|)
///
/// Returns a 0-D tensor containing the norm value on the device.
///
/// **Optimized**: Uses fused MaxAbs reducer (1 kernel launch instead of 2).
///
/// # Arguments
///
/// * `client` - Compute client for kernel execution
/// * `x` - Input vector (any rank, flattened)
///
/// # Example
///
/// ```ignore
/// let norm = vector_norm_inf::<R, F32Precision>(client, x.as_ref())?;
/// ```
pub fn vector_norm_inf<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    x: TensorHandleRef<R>,
) -> LinalgResult<TensorHandle<R, P::EA>>
where
    P::EW: Float,
    P::EA: Float,
{
    // MaxAbs reducer: computes max(|x|) in a single fused kernel
    // No separate abs_kernel needed - abs is computed inline during reduction
    let max_shape = vec![1];
    let max_output = TensorHandle::<R, P::EA>::empty(client, max_shape);

    reduce::<R, (P::EA, P::EA), P::EA, MaxAbs>(
        client,
        x,
        max_output.as_ref(),
        0, // Reduce along dimension 0
        None,
        (),
    ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

    Ok(max_output)
}

/// Frobenius norm of a matrix: ||A||_F = sqrt(sum(A_ij^2))
///
/// This is equivalent to treating the matrix as a vector and computing L2 norm.
/// Returns a 0-D tensor containing the norm value on the device.
///
/// # Arguments
///
/// * `client` - Compute client for kernel execution
/// * `a` - Input matrix (2D or higher rank)
///
/// # Example
///
/// ```ignore
/// let norm = frobenius_norm::<R, F32Precision>(client, a.as_ref())?;
/// ```
pub fn frobenius_norm<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    a: TensorHandleRef<R>,
) -> LinalgResult<TensorHandle<R, P::EA>>
where
    P::EW: Float,
    P::EA: Float,
{
    // Frobenius norm is just L2 norm treating matrix as vector
    vector_norm_l2::<R, P>(client, a)
}

/// Estimate the spectral norm (largest singular value) using power iteration.
///
/// Uses the power method: v_{k+1} = A^T A v_k / ||A^T A v_k||
/// Converges to the largest eigenvalue of A^T A, which gives ||A||_2^2.
///
/// Returns a 0-D tensor containing the estimated spectral norm on the device.
///
/// # Arguments
///
/// * `client` - Compute client for kernel execution
/// * `a` - Input matrix [M, N]
/// * `k_iters` - Number of power iterations (default: 10)
///
/// # Algorithm
///
/// 1. Initialize random unit vector v
/// 2. For k iterations:
///    - w = A v
///    - v = A^T w
///    - v = v / ||v||
/// 3. Return ||A v||
///
/// # Example
///
/// ```ignore
/// let spectral = spectral_norm_est::<R, F32Precision>(client, a.as_ref(), 10)?;
/// ```
pub fn spectral_norm_est<R: Runtime, P: LinalgPrecision>(
    _client: &ComputeClient<R::Server>,
    _a: TensorHandleRef<R>,
    _k_iters: usize,
) -> LinalgResult<TensorHandle<R, P::EA>>
where
    P::EW: Float,
    P::EA: Float,
{
    // TODO: Implement full power iteration
    // For now, return a placeholder error
    // This requires:
    // 1. Matrix-vector multiplication (can use cubecl-matmul)
    // 2. Vector normalization
    // 3. Iteration loop on device or CPU-orchestrated

    Err(LinalgError::UnsupportedLayout {
        layout: "spectral_norm_est not yet implemented".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add tests once we have a test runtime setup
}
