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

    // Test large vectors that trigger 2-stage reduction
    #[test]
    fn test_norm_l2_large() {
        let device = Default::default();
        let client = TestRuntime::client(&device);

        // Create large vector (2M elements to trigger 2-stage reduction)
        let size = 2_000_000;
        let values: Vec<f32> = (0..size).map(|i| (i % 100) as f32 / 100.0).collect();
        let expected = cpu_norm_l2(&values);

        let input_handle = client.create(f32::as_bytes(&values));
        let strides = vec![1];
        let shape = vec![size];
        let input = unsafe {
            TensorHandleRef::<TestRuntime>::from_raw_parts(
                &input_handle,
                &strides,
                &shape,
                std::mem::size_of::<f32>(),
            )
        };

        let result = vector_norm_l2::<TestRuntime, F32Precision>(&client, input)
            .expect("norm_l2 large failed");

        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        // Slightly higher tolerance for large sums due to floating point accumulation
        assert_relative_eq!(result_value, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_norm_inf_large() {
        let device = Default::default();
        let client = TestRuntime::client(&device);

        let size = 2_000_000;
        let values: Vec<f32> = (0..size).map(|i| {
            let val = (i % 1000) as f32 / 10.0;
            if i % 7 == 0 { -val } else { val }
        }).collect();
        let expected = cpu_norm_inf(&values);

        let input_handle = client.create(f32::as_bytes(&values));
        let strides = vec![1];
        let shape = vec![size];
        let input = unsafe {
            TensorHandleRef::<TestRuntime>::from_raw_parts(
                &input_handle,
                &strides,
                &shape,
                std::mem::size_of::<f32>(),
            )
        };

        let result = vector_norm_inf::<TestRuntime, F32Precision>(&client, input)
            .expect("norm_inf large failed");

        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];

        assert_relative_eq!(result_value, expected, epsilon = 1e-5);
    }

    // Test various sizes to ensure 2-stage threshold works correctly
    #[test]
    fn test_norm_l2_various_sizes() {
        let device = Default::default();
        let client = TestRuntime::client(&device);

        let test_sizes = vec![
            100,        // Small (single-stage)
            10_000,     // Medium (single-stage)
            999_999,    // Just below threshold
            1_000_001,  // Just above threshold (2-stage)
            4_194_304,  // Large power of 2
        ];

        for size in test_sizes {
            let values: Vec<f32> = (0..size).map(|i| ((i * 7) % 101) as f32 / 50.0).collect();
            let expected = cpu_norm_l2(&values);

            let input_handle = client.create(f32::as_bytes(&values));
            let strides = vec![1];
            let shape = vec![size];
            let input = unsafe {
                TensorHandleRef::<TestRuntime>::from_raw_parts(
                    &input_handle,
                    &strides,
                    &shape,
                    std::mem::size_of::<f32>(),
                )
            };

            let result = vector_norm_l2::<TestRuntime, F32Precision>(&client, input)
                .expect(&format!("norm_l2 failed for size {}", size));

            let result_bytes = client.read_one(result.handle.clone());
            let result_value = f32::from_bytes(&result_bytes)[0];

            let diff = (result_value - expected).abs();
            let rel_error = diff / expected.max(1e-6);
            assert!(
                rel_error < 1e-3,
                "Failed for size {}: got {}, expected {}, rel_error = {}",
                size, result_value, expected, rel_error
            );
        }
    }

    // Test edge cases
    #[test]
    fn test_norm_edge_cases() {
        let device = Default::default();
        let client = TestRuntime::client(&device);

        // All zeros
        let values = vec![0.0_f32; 1000];
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<TestRuntime>::from_raw_parts(
                &input_handle,
                &[1],
                &[1000],
                std::mem::size_of::<f32>(),
            )
        };
        let result = vector_norm_l2::<TestRuntime, F32Precision>(&client, input).unwrap();
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];
        assert_relative_eq!(result_value, 0.0, epsilon = 1e-7);

        // All ones
        let values = vec![1.0_f32; 1024];
        let expected = (1024.0_f32).sqrt();
        let input_handle = client.create(f32::as_bytes(&values));
        let input = unsafe {
            TensorHandleRef::<TestRuntime>::from_raw_parts(
                &input_handle,
                &[1],
                &[1024],
                std::mem::size_of::<f32>(),
            )
        };
        let result = vector_norm_l2::<TestRuntime, F32Precision>(&client, input).unwrap();
        let result_bytes = client.read_one(result.handle.clone());
        let result_value = f32::from_bytes(&result_bytes)[0];
        assert_relative_eq!(result_value, expected, epsilon = 1e-5);
    }
}
