//! Tests for norm operations

use approx::assert_relative_eq;

/// CPU reference: L2 norm
pub fn cpu_norm_l2(values: &[f32]) -> f32 {
    values.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// CPU reference: L-infinity norm
pub fn cpu_norm_inf(values: &[f32]) -> f32 {
    values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max)
}

/// CPU reference: Frobenius norm (same as L2 for flattened matrix)
pub fn cpu_frobenius_norm(values: &[f32]) -> f32 {
    cpu_norm_l2(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::{prelude::*, Runtime};
    use crate::{vector_norm_l2, vector_norm_inf, frobenius_norm, F32Precision};

    // Define test runtime
    type TestRuntime = cubecl_cpu::CpuRuntime;

    /// Helper to run norm test
    fn test_norm_l2_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Test data: simple vector
        let values = vec![3.0_f32, 4.0_f32]; // Expected norm: 5.0
        let expected = cpu_norm_l2(&values);

        // Create input tensor
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[1],      // stride
                &[2],      // shape
                std::mem::size_of::<f32>(),
            )
        };

        // Compute norm on GPU
        let result = vector_norm_l2::<R, F32Precision>(&client, input)
            .expect("norm_l2 failed");

        // Read result from GPU
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        // Compare
        assert_relative_eq!(result_value, expected, epsilon = 1e-5);
    }

    /// Helper to run inf norm test
    fn test_norm_inf_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Test data: vector with negative values
        let values = vec![-5.0_f32, 3.0_f32, -7.0_f32, 2.0_f32]; // Expected: 7.0
        let expected = cpu_norm_inf(&values);

        // Create input tensor
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[1],      // stride
                &[4],      // shape
                std::mem::size_of::<f32>(),
            )
        };

        // Compute norm on GPU
        let result = vector_norm_inf::<R, F32Precision>(&client, input)
            .expect("norm_inf failed");

        // Read result from GPU
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        // Compare
        assert_relative_eq!(result_value, expected, epsilon = 1e-5);
    }

    /// Helper to run Frobenius norm test
    fn test_frobenius_norm_impl<R: Runtime>(device: &R::Device) {
        let client = R::client(device);

        // Test data: 2x2 matrix
        let values = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
        let expected = cpu_frobenius_norm(&values); // sqrt(1+4+9+16) = sqrt(30)

        // Create input tensor (2x2 matrix)
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &[2, 1],   // row-major stride
                &[2, 2],   // shape
                std::mem::size_of::<f32>(),
            )
        };

        // Compute Frobenius norm on GPU
        let result = frobenius_norm::<R, F32Precision>(&client, input)
            .expect("frobenius_norm failed");

        // Read result from GPU
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        // Compare
        assert_relative_eq!(result_value, expected, epsilon = 1e-5);
    }

    // Actual test functions that will be run
    // These use the test runtime that CubeCL provides

    #[test]
    fn test_norm_l2_f32() {
        test_norm_l2_impl::<TestRuntime>(&Default::default());
    }

    #[test]
    fn test_norm_inf_f32() {
        test_norm_inf_impl::<TestRuntime>(&Default::default());
    }

    #[test]
    fn test_frobenius_norm_f32() {
        test_frobenius_norm_impl::<TestRuntime>(&Default::default());
    }
}
