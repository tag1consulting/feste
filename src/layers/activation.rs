//! Activation Functions
//!
//! This module provides activation functions and their derivatives for
//! backpropagation.
//!
//! ## GELU (Gaussian Error Linear Unit)
//!
//! GELU is used in transformers instead of ReLU because it provides smoother
//! gradients and often performs better in practice.
//!
//! ### Formula
//!
//! ```text
//! GELU(x) = x × Φ(x)
//! ```
//!
//! where Φ(x) is the cumulative distribution function of the standard normal distribution.
//!
//! ### Approximation
//!
//! We use the tanh approximation for efficiency:
//!
//! ```text
//! GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
//! ```
//!
//! This is faster than computing the exact CDF and is accurate enough for neural networks.
//!
//! ### Why GELU?
//!
//! - **Smooth gradients**: Unlike ReLU (which has zero gradient for x<0), GELU has
//!   non-zero gradients everywhere
//! - **Better empirical performance**: Especially in large transformers like GPT-2/BERT
//! - **Probabilistic interpretation**: GELU can be seen as a smooth approximation to
//!   dropout at the neuron level

use crate::tensor::Tensor;
use rayon::prelude::*;

/// GELU activation (forward pass)
///
/// Computes the GELU activation using the tanh approximation.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// Tensor with GELU activation applied element-wise
///
/// # Performance
///
/// Uses parallel computation via Rayon for better performance on multi-core CPUs.
pub fn gelu_forward(x: &Tensor) -> Tensor {
    let result = x
        .data
        .par_iter()
        .map(|&val| {
            0.5 * val
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh())
        })
        .collect();
    Tensor::new(result, x.shape.clone())
}

/// GELU activation derivative (backward pass)
///
/// Computes the gradient of GELU with respect to its input.
///
/// # Arguments
///
/// * `grad_out` - Gradient from next layer
/// * `x` - Original input to GELU (from forward pass)
///
/// # Returns
///
/// Gradient with respect to input: grad_x = grad_out * GELU'(x)
///
/// # Mathematical Derivation
///
/// The derivative involves:
/// 1. Derivative of tanh (sech² term)
/// 2. Derivative of the inner polynomial
/// 3. Product rule application
///
/// This gives us a complex but smooth gradient that helps training.
pub fn gelu_backward(grad_out: &Tensor, x: &Tensor) -> Tensor {
    let grad_data: Vec<f32> = x
        .data
        .par_iter()
        .zip(&grad_out.data)
        .map(|(&x_val, &grad_val)| {
            let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            let inner = sqrt_2_pi * (x_val + 0.044715 * x_val.powi(3));
            let tanh_inner = inner.tanh();
            let sech_sq = 1.0 - tanh_inner * tanh_inner;

            let grad_gelu = 0.5 * (1.0 + tanh_inner)
                + 0.5 * x_val * sech_sq * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x_val.powi(2));

            grad_val * grad_gelu
        })
        .collect();

    Tensor::new(grad_data, x.shape.clone())
}
