//! Basic example demonstrating LU factorization
//!
//! This example shows how to:
//! 1. Create a simple matrix
//! 2. Compute LU factorization with partial pivoting
//! 3. Verify the result by reconstructing the original matrix
//!
//! Run with: cargo run --example lu_basic --features=std

use cubecl_core::prelude::*;
use cubecl_linalg::{lu_factor, LUConfig, F32Precision};

fn main() {
    type Runtime = cubecl_cpu::CpuRuntime;

    let device = Default::default();
    let client = Runtime::client(&device);

    // Create a simple 4x4 test matrix
    let n = 4;
    #[rustfmt::skip]
    let a_data = vec![
        2.0_f32, 1.0, 1.0, 0.0,
        4.0, 3.0, 3.0, 1.0,
        8.0, 7.0, 9.0, 5.0,
        6.0, 7.0, 9.0, 8.0,
    ];

    println!("Original matrix A ({}×{}):", n, n);
    print_matrix(&a_data, n);

    // Create tensor
    let a_handle = client.create_from_slice(f32::as_bytes(&a_data));
    let a = cubecl_std::tensor::TensorHandle::new(
        a_handle,
        vec![n, n],
        vec![n, 1],
        f32::as_type_native_unchecked(),
    );

    // Compute LU factorization
    println!("\nComputing LU factorization with partial pivoting...");
    let config = LUConfig::default();

    match lu_factor::<Runtime, F32Precision>(&client, a.as_ref(), config) {
        Ok((lu, perm, info)) => {
            println!("✓ LU factorization successful!");

            // Read result
            let lu_bytes = client.read_one(lu.handle.clone());
            let lu_data = f32::from_bytes(&lu_bytes);

            println!("\nPermutation vector P:");
            println!("{:?}", perm);

            println!("\nPacked LU matrix (L below diagonal with unit diagonal, U on and above):");
            print_matrix(lu_data, n);

            // Extract and display L and U
            let l = extract_l(lu_data, n);
            let u = extract_u(lu_data, n);

            println!("\nLower triangular L (unit diagonal):");
            print_matrix(&l, n);

            println!("\nUpper triangular U:");
            print_matrix(&u, n);

            // Verify: P*A = L*U
            println!("\nVerifying P*A = L*U...");
            let pa = apply_perm(&a_data, &perm, n);
            let lu_product = matmul(&l, &u, n);

            let max_error = max_abs_diff(&pa, &lu_product);
            println!("Maximum absolute error: {:.2e}", max_error);

            if max_error < 1e-4 {
                println!("✓ Verification passed! P*A ≈ L*U");
            } else {
                println!("✗ Verification failed! Error too large");
            }

            // Display solve info
            println!("\nSolve info:");
            println!("  Quality: {:?}", info.quality);
            if let Some(rcond) = info.condition_estimate {
                println!("  Condition: {:.2e}", rcond);
            }
        }
        Err(e) => {
            println!("✗ LU factorization failed: {:?}", e);
        }
    }
}

fn print_matrix(data: &[f32], n: usize) {
    for i in 0..n {
        print!("  [");
        for j in 0..n {
            print!("{:8.4}", data[i * n + j]);
            if j < n - 1 {
                print!(" ");
            }
        }
        println!("]");
    }
}

fn extract_l(lu: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0; // Unit diagonal
        for j in 0..i {
            l[i * n + j] = lu[i * n + j];
        }
    }
    l
}

fn extract_u(lu: &[f32], n: usize) -> Vec<f32> {
    let mut u = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = lu[i * n + j];
        }
    }
    u
}

fn apply_perm(a: &[f32], perm: &[usize], n: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = a[perm[i] * n + j];
        }
    }
    result
}

fn matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
