//! Element-wise operations for linear algebra.
//!
//! These kernels perform simple element-wise transformations that are
//! used as building blocks for more complex operations.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Element-wise square: out[i] = x[i]^2
#[cube(launch)]
pub fn square_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * input[ABSOLUTE_POS];
    }
}

/// Element-wise absolute value: out[i] = |x[i]|
#[cube(launch)]
pub fn abs_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = F::abs(input[ABSOLUTE_POS]);
    }
}

/// Element-wise square root: out[i] = sqrt(x[i])
#[cube(launch)]
pub fn sqrt_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = F::sqrt(input[ABSOLUTE_POS]);
    }
}

/// Element-wise reciprocal: out[i] = 1 / x[i]
#[cube(launch)]
pub fn recip_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = F::recip(input[ABSOLUTE_POS]);
    }
}

/// Element-wise scale: out[i] = alpha * x[i]
#[cube(launch)]
pub fn scale_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>, alpha: F) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = alpha * input[ABSOLUTE_POS];
    }
}

/// Element-wise add: out[i] = x[i] + y[i]
#[cube(launch)]
pub fn add_kernel<F: Float>(x: &Tensor<F>, y: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// Element-wise subtract: out[i] = x[i] - y[i]
#[cube(launch)]
pub fn sub_kernel<F: Float>(x: &Tensor<F>, y: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = x[ABSOLUTE_POS] - y[ABSOLUTE_POS];
    }
}

/// Element-wise multiply: out[i] = x[i] * y[i]
#[cube(launch)]
pub fn mul_kernel<F: Float>(x: &Tensor<F>, y: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = x[ABSOLUTE_POS] * y[ABSOLUTE_POS];
    }
}

/// Fill tensor with scalar value: out[i] = value
#[cube(launch)]
pub fn fill_kernel<F: Float>(output: &mut Tensor<F>, value: F) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = value;
    }
}

/// Copy tensor: out[i] = input[i]
#[cube(launch)]
pub fn copy_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// Fused AXPY variant: y[i] = alpha * y[i] - x[i]
///
/// Used in TRSM for: B2 = alpha * B2 - L21 * X1
#[cube(launch)]
pub fn fused_scale_sub_kernel<F: Float>(
    y: &mut Tensor<F>,
    x: &Tensor<F>,
    alpha: F,
) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * y[ABSOLUTE_POS] - x[ABSOLUTE_POS];
    }
}
