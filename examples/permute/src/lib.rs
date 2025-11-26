

// ================================
// Constants & tuning parameters
// ================================


// ===========================================================
// Host-side utility functions for shape and stride calculations
// ===========================================================



/// Extract (batch, height, width) dimensions for batch transpose kernels.
///
/// Converts 2D or 3D input shapes into a standardized 3D format for transpose kernels.
#[allow(dead_code)]
fn infer_batch_transpose_shape(input_shape: &[usize], _axes: &[usize]) -> (u32, u32, u32) {
    match input_shape.len() {
        2 => {
            // [H, W] → treat as single batch
            (1, input_shape[0] as u32, input_shape[1] as u32)
        }
        3 => {
            // [B, H, W] → batched 2D
            (
                input_shape[0] as u32,
                input_shape[1] as u32,
                input_shape[2] as u32,
            )
        }
        _ => panic!(
            "infer_batch_transpose_shape only supports rank 2 or 3, got rank {}",
            input_shape.len()
        ),
    }
}


// This is the beginning of the permute.rs code
// All the kernels are here
// ...
// ... all the way to the end
// ...

