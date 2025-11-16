//! Pivoting kernels for LU factorization with partial pivoting.
//!
//! This module implements GPU-optimized pivot finding and row swapping using
//! plane (warp/subgroup) operations for maximum performance.
//!
//! ## Key Operations
//!
//! - **Pivot finding**: Find argmax|column[i]| using plane_max reduction
//! - **Row swaps**: Swap two rows efficiently (coalesced memory access)
//! - **Permutation application**: Apply permutation vector to matrix/vector

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::TensorHandle;

use crate::{LinalgPrecision, LinalgResult, LinalgError};

/// Find the pivot element in a column (argmax |column[i]|)
///
/// Simple CPU-side implementation for now.
/// TODO: Implement GPU-accelerated argmax kernel using plane operations.
///
/// # Arguments
/// * `column` - Column vector [N-k] starting from diagonal element
/// * `start_row` - Global row offset (for returning absolute index)
///
/// # Returns
/// * `pivot_idx` - Global row index of pivot element
/// * `pivot_value` - Value of the pivot element
pub fn find_column_pivot<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    column: TensorHandleRef<R>,
    start_row: usize,
) -> LinalgResult<(usize, f32)>
where
    P::EW: Float + CubeElement,
{
    let n = column.shape[0];

    if n == 0 {
        return Err(LinalgError::InvalidShape {
            reason: "Empty column for pivot".to_string(),
        });
    }

    // Simple CPU-side search for max absolute value
    // This is a temporary implementation - will be replaced with GPU kernel
    let data = client.read(column.handle.binding());
    let slice = data.as_slice::<P::EW>();

    let mut max_abs = 0.0f32;
    let mut max_idx = 0;

    for (i, &val) in slice.iter().enumerate() {
        let abs_val = val.abs().to_f32();
        if abs_val > max_abs {
            max_abs = abs_val;
            max_idx = i;
        }
    }

    let pivot_idx = start_row + max_idx;
    Ok((pivot_idx, max_abs))
}

/// Swap two rows of a matrix efficiently
///
/// This kernel performs coalesced row swaps by having each thread
/// swap one column element between the two rows.
///
/// # Arguments
/// * `a` - Matrix to modify [M, N]
/// * `row1`, `row2` - Indices of rows to swap
#[cube(launch)]
pub fn swap_rows_kernel<F: Float>(
    a: &mut Tensor<F>,
    #[comptime] n_cols: u32,
    row1: u32,
    row2: u32,
) {
    let col = ABSOLUTE_POS;

    if col < n_cols {
        let idx1 = row1 * n_cols + col;
        let idx2 = row2 * n_cols + col;

        let tmp = a[idx1];
        a[idx1] = a[idx2];
        a[idx2] = tmp;
    }
}

/// Swap two rows in a matrix (host-side wrapper)
pub fn swap_rows<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    a: &mut TensorHandle<R>,
    row1: usize,
    row2: usize,
) -> LinalgResult<()>
where
    P::EW: Float + CubeElement,
{
    if row1 == row2 {
        return Ok(()); // No-op
    }

    let shape = a.shape.clone();
    let n_rows = shape[shape.len() - 2];
    let n_cols = shape[shape.len() - 1];

    if row1 >= n_rows || row2 >= n_rows {
        return Err(LinalgError::InvalidShape {
            reason: format!("Row indices {}, {} out of bounds for matrix with {} rows",
                          row1, row2, n_rows),
        });
    }

    // Launch kernel with one thread per column
    swap_rows_kernel::launch::<R, P::EW>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new((n_cols as u32 + 255) / 256, 1, 1),
        a.as_ref(),
        n_cols as u32,
        row1 as u32,
        row2 as u32,
    );

    Ok(())
}

/// Apply permutation to a vector: out[i] = vec[perm[i]]
///
/// This is used to permute the RHS vector in solve_lu.
#[cube(launch)]
pub fn apply_permutation_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    perm: &Tensor<u32>,
    #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS;

    if i < n {
        let src_idx = perm[i];
        output[i] = input[src_idx];
    }
}

/// Apply permutation to vector (host-side wrapper)
pub fn apply_permutation<R: Runtime, P: LinalgPrecision>(
    client: &ComputeClient<R::Server>,
    vec: TensorHandleRef<R>,
    perm: &[usize],
) -> LinalgResult<TensorHandle<R>>
where
    P::EW: Float + CubeElement,
{
    let n = vec.shape[0];

    if perm.len() != n {
        return Err(LinalgError::InvalidShape {
            reason: format!("Permutation size {} doesn't match vector size {}",
                          perm.len(), n),
        });
    }

    // Create output tensor
    let output = client.create(vec.handle.binding().elem_size, &vec.shape);

    // Convert permutation to device tensor
    let perm_u32: Vec<u32> = perm.iter().map(|&x| x as u32).collect();
    let perm_handle = client.create_from_slice(&perm_u32);

    // Launch kernel
    apply_permutation_kernel::launch::<R, P::EW>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new((n as u32 + 255) / 256, 1, 1),
        vec,
        output.as_ref(),
        perm_handle.as_ref(),
        n as u32,
    );

    Ok(output)
}

/// Warp-level pivot finding using plane operations (SOTA version)
///
/// This kernel finds the pivot within a warp using plane_max reduction.
/// Much faster than global reduction for small panels.
///
/// Returns (local_index, value) of the maximum absolute value element.
#[cube]
pub fn warp_find_pivot<F: Float>(
    values: &Tensor<F>,
    start_idx: u32,
    end_idx: u32,
    lane_id: u32,
) -> (u32, F) {
    // Each lane loads one element
    let my_idx = start_idx + lane_id;
    let my_val = if my_idx < end_idx {
        values[my_idx]
    } else {
        F::new(0.0)
    };

    let my_abs = my_val.abs();

    // Use plane_max to find maximum across warp
    let max_abs = plane_max(my_abs);

    // Check if this lane has the max
    let is_pivot = my_abs == max_abs && my_idx < end_idx;

    // Use ballot to find which lane has the pivot
    // The first set bit is the pivot lane
    let ballot = plane_ballot(is_pivot);

    // Find first set bit (this gives us the pivot lane)
    // For simplicity, broadcast from all lanes that match and take the first
    let pivot_lane = if is_pivot { lane_id } else { u32::MAX };
    let final_pivot_lane = plane_min(pivot_lane);

    // Broadcast the value from pivot lane
    let pivot_val = plane_broadcast(my_val, final_pivot_lane);

    (final_pivot_lane, pivot_val)
}
