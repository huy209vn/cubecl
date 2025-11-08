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

#[cfg(feature = "std")]
use std::string::ToString;

#[cfg(not(feature = "std"))]
use alloc::string::ToString;

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
    use cubecl_reduce::instructions::Sum;

    // Flatten the tensor to 1D for reduction (zero-copy view)
    let total_elements: usize = x.shape.iter().product();
    let flat_stride = vec![1];
    let flat_shape = vec![total_elements];
    let x_flat = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            x.handle,
            &flat_stride,
            &flat_shape,
            x.elem_size,
        )
    };

    // OPTIMIZATION: For large 1D tensors, cubecl-reduce uses only 1 cube when reducing to [1]
    // We reshape to 2D to enable parallel reduction across many cubes

    // Choose a good intermediate size for 2-stage reduction
    // Target: ~4K-8K intermediate elements for good parallelism
    let intermediate_size = if total_elements > 1_000_000 {
        // Large tensor: use 2-stage reduction
        let sqrt_n = (total_elements as f64).sqrt().ceil() as usize;
        // Round to nice number for better alignment
        let chunk_size = (sqrt_n + 255) / 256 * 256;
        total_elements.div_ceil(chunk_size).max(1024).min(8192)
    } else {
        // Small tensor: single-stage is fine
        1
    };

    let intermediate_output;

    if intermediate_size > 1 {
        // Stage 1: Reduce to intermediate size with high parallelism
        // Reshape [N] as [intermediate_size, N/intermediate_size]
        let chunk_size = total_elements / intermediate_size;

        // Create reshaped view (no data copy!)
        let reshaped_shape = vec![intermediate_size, chunk_size];
        let reshaped_strides = vec![chunk_size, 1];
        let x_reshaped = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                x.handle,
                &reshaped_strides,
                &reshaped_shape,
                x.elem_size,
            )
        };

        // Reduce along axis 1: [M, N] → [M, 1] using SumSquared
        let mut stage1_shape = reshaped_shape.clone();
        stage1_shape[1] = 1;
        let stage1_output = TensorHandle::<R, P::EA>::empty(client, stage1_shape.clone());

        reduce::<R, (P::EA, P::EA), P::EA, SumSquared>(
            client,
            x_reshaped,
            stage1_output.as_ref(),
            1, // Reduce along axis 1 (inner dimension)
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        // Stage 2: Reduce [M, 1] → [1, 1] using Sum
        let mut final_shape = stage1_shape.clone();
        final_shape[0] = 1;
        let sum_output = TensorHandle::<R, P::EA>::empty(client, final_shape);

        reduce::<R, (P::EA, P::EA), P::EA, Sum>(
            client,
            stage1_output.as_ref(),
            sum_output.as_ref(),
            0, // Reduce along axis 0
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        intermediate_output = sum_output;
    } else {
        // Single-stage reduction for small tensors
        let sum_shape = vec![1];
        let sum_output = TensorHandle::<R, P::EA>::empty(client, sum_shape.clone());

        reduce::<R, (P::EA, P::EA), P::EA, SumSquared>(
            client,
            x_flat,
            sum_output.as_ref(),
            0,
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        intermediate_output = sum_output;
    }

    // Final step: Take square root
    let norm_output = TensorHandle::<R, P::EA>::empty(client, vec![1]);

    sqrt_kernel::launch::<P::EA, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(1, 1, 1),
        intermediate_output.as_arg(1),
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
    use cubecl_reduce::instructions::Max;

    // Flatten the tensor to 1D for reduction (zero-copy view)
    let total_elements: usize = x.shape.iter().product();
    let flat_stride = vec![1];
    let flat_shape = vec![total_elements];
    let x_flat = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            x.handle,
            &flat_stride,
            &flat_shape,
            x.elem_size,
        )
    };

    // OPTIMIZATION: Use 2-stage reduction for large tensors to enable parallelism

    let intermediate_size = if total_elements > 1_000_000 {
        let sqrt_n = (total_elements as f64).sqrt().ceil() as usize;
        let chunk_size = (sqrt_n + 255) / 256 * 256;
        total_elements.div_ceil(chunk_size).max(1024).min(8192)
    } else {
        1
    };

    let intermediate_output;

    if intermediate_size > 1 {
        // Stage 1: Reduce to intermediate size
        let chunk_size = total_elements / intermediate_size;

        let reshaped_shape = vec![intermediate_size, chunk_size];
        let reshaped_strides = vec![chunk_size, 1];
        let x_reshaped = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                x.handle,
                &reshaped_strides,
                &reshaped_shape,
                x.elem_size,
            )
        };

        // Reduce along axis 1: [M, N] → [M, 1] using MaxAbs
        let mut stage1_shape = reshaped_shape.clone();
        stage1_shape[1] = 1;
        let stage1_output = TensorHandle::<R, P::EA>::empty(client, stage1_shape.clone());

        reduce::<R, (P::EA, P::EA), P::EA, MaxAbs>(
            client,
            x_reshaped,
            stage1_output.as_ref(),
            1,
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        // Stage 2: Reduce [M, 1] → [1, 1] using Max
        let mut final_shape = stage1_shape.clone();
        final_shape[0] = 1;
        let max_output = TensorHandle::<R, P::EA>::empty(client, final_shape);

        reduce::<R, (P::EA, P::EA), P::EA, Max>(
            client,
            stage1_output.as_ref(),
            max_output.as_ref(),
            0,
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        intermediate_output = max_output;
    } else {
        // Single-stage reduction for small tensors
        let max_shape = vec![1];
        let max_output = TensorHandle::<R, P::EA>::empty(client, max_shape);

        reduce::<R, (P::EA, P::EA), P::EA, MaxAbs>(
            client,
            x_flat,
            max_output.as_ref(),
            0,
            None,
            (),
        ).map_err(|e| LinalgError::ReduceFailure(format!("{:?}", e)))?;

        intermediate_output = max_output;
    }

    Ok(intermediate_output)
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
