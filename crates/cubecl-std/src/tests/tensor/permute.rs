use cubecl_core::{
    CubeElement,
    prelude::{Float, Runtime},
};

use crate::tensor::{self, TensorHandle};

/// Test 2D transpose [H, W] -> [W, H] with axes [1, 0]
pub fn test_permute_2d_transpose<R: Runtime, C: Float + CubeElement>(
    device: &R::Device,
    height: usize,
    width: usize,
) {
    let client = R::client(device);

    // Create input data: [[0,1,2], [3,4,5], ...] for [H,W]
    let numel = height * width;
    let input_data: Vec<C> = (0..numel).map(|i| C::from(i as f32).unwrap()).collect();

    let handle = client.create(C::as_bytes(&input_data));
    let input = TensorHandle::<R, C>::new_contiguous(vec![height, width], handle);
    let output = tensor::permute::launch_alloc(&client, &input, &[1, 0]);

    let actual = client.read_one_tensor(output.handle.clone().copy_descriptor(
        &output.shape,
        &output.strides,
        size_of::<C>(),
    ));
    let actual = C::from_bytes(&actual);

    // Verify output shape
    assert_eq!(output.shape, vec![width, height]);

    // Verify transposed data
    for row in 0..width {
        for col in 0..height {
            let out_idx = row * height + col;
            let in_idx = col * width + row;
            assert_eq!(
                actual[out_idx],
                C::from(in_idx as f32).unwrap(),
                "Mismatch at output[{}, {}]",
                row,
                col
            );
        }
    }
}

/// Test 3D batch transpose [B, H, W] -> [B, W, H] with axes [0, 2, 1]
pub fn test_permute_3d_batch_transpose<R: Runtime, C: Float + CubeElement>(
    device: &R::Device,
    batch: usize,
    height: usize,
    width: usize,
) {
    let client = R::client(device);

    let numel = batch * height * width;
    let input_data: Vec<C> = (0..numel).map(|i| C::from(i as f32).unwrap()).collect();

    let handle = client.create(C::as_bytes(&input_data));
    let input = TensorHandle::<R, C>::new_contiguous(vec![batch, height, width], handle);
    let output = tensor::permute::launch_alloc(&client, &input, &[0, 2, 1]);

    let actual = client.read_one_tensor(output.handle.clone().copy_descriptor(
        &output.shape,
        &output.strides,
        size_of::<C>(),
    ));
    let actual = C::from_bytes(&actual);

    // Verify output shape
    assert_eq!(output.shape, vec![batch, width, height]);

    // Verify each batch is transposed correctly
    for b in 0..batch {
        for row in 0..width {
            for col in 0..height {
                let out_idx = b * width * height + row * height + col;
                let in_idx = b * height * width + col * width + row;
                assert_eq!(
                    actual[out_idx],
                    C::from(in_idx as f32).unwrap(),
                    "Mismatch at batch {} output[{}, {}]",
                    b,
                    row,
                    col
                );
            }
        }
    }
}

/// Test 3D permutation [B, H, W] -> [W, B, H] with axes [2, 0, 1]
pub fn test_permute_3d_complex<R: Runtime, C: Float + CubeElement>(
    device: &R::Device,
    dim0: usize,
    dim1: usize,
    dim2: usize,
) {
    let client = R::client(device);

    let numel = dim0 * dim1 * dim2;
    let input_data: Vec<C> = (0..numel).map(|i| C::from(i as f32).unwrap()).collect();

    let handle = client.create(C::as_bytes(&input_data));
    let input = TensorHandle::<R, C>::new_contiguous(vec![dim0, dim1, dim2], handle);
    let output = tensor::permute::launch_alloc(&client, &input, &[2, 0, 1]);

    let actual = client.read_one_tensor(output.handle.clone().copy_descriptor(
        &output.shape,
        &output.strides,
        size_of::<C>(),
    ));
    let actual = C::from_bytes(&actual);

    // Verify output shape
    assert_eq!(output.shape, vec![dim2, dim0, dim1]);

    // Verify permutation: out[i,j,k] = in[j,k,i]
    for i in 0..dim2 {
        for j in 0..dim0 {
            for k in 0..dim1 {
                let out_idx = i * dim0 * dim1 + j * dim1 + k;
                let in_idx = j * dim1 * dim2 + k * dim2 + i;
                assert_eq!(
                    actual[out_idx],
                    C::from(in_idx as f32).unwrap(),
                    "Mismatch at output[{}, {}, {}]",
                    i,
                    j,
                    k
                );
            }
        }
    }
}

/// Test edge case: empty tensor
pub fn test_permute_empty<R: Runtime, C: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);

    let input = TensorHandle::<R, C>::empty(&client, vec![0, 5]);
    let output = tensor::permute::launch_alloc(&client, &input, &[1, 0]);

    assert_eq!(output.shape, vec![5, 0]);
}

/// Test edge case: single element
pub fn test_permute_single_element<R: Runtime, C: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);

    let handle = client.create(C::as_bytes(&[C::from(42.0).unwrap()]));
    let input = TensorHandle::<R, C>::new_contiguous(vec![1, 1], handle);
    let output = tensor::permute::launch_alloc(&client, &input, &[1, 0]);

    let actual = client.read_one_tensor(output.handle.clone().copy_descriptor(
        &output.shape,
        &output.strides,
        size_of::<C>(),
    ));
    let actual = C::from_bytes(&actual);

    assert_eq!(actual[0], C::from(42.0).unwrap());
}
