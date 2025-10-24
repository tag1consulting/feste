//! Linear Layer (Fully Connected)
//!
//! The linear layer is the fundamental building block of neural networks.
//! It performs an affine transformation: y = x @ W + b
//!
//! ## Forward Pass
//!
//! ```text
//! Input:  x [seq_len, in_features]
//! Weight: W [in_features, out_features]
//! Bias:   b [out_features]
//! Output: y = x @ W + b [seq_len, out_features]
//! ```
//!
//! ## Backward Pass
//!
//! Using the chain rule:
//! ```text
//! grad_W = x^T @ grad_y
//! grad_b = sum(grad_y, axis=0)
//! grad_x = grad_y @ W^T
//! ```
//!
//! ## Why These Gradients?
//!
//! - **grad_W**: Each weight W[i,j] affects output y[*,j] through input x[*,i]
//! - **grad_b**: Each bias b[j] affects all outputs y[*,j] equally
//! - **grad_x**: Needed to backprop to previous layer
//!
//! ## Implementation Notes
//!
//! - Uses He initialization: scale = √(2/in_features)
//! - Bias initialized to zero (common practice)
//! - Caches input x for backward pass

use crate::tensor::Tensor;

/// Helper function for random initialization
///
/// Uses a simple LCG (Linear Congruential Generator) for reproducible initialization.
/// The scale parameter controls the magnitude of initial weights.
pub fn random_init(size: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = seed;
    (0..size)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng / 65536) % 32768) as f32 / 32768.0;
            (val - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Linear layer (fully connected)
///
/// Performs y = x @ W + b where:
/// - W: weight matrix [in_features, out_features]
/// - b: bias vector [out_features]
pub struct TrainableLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl TrainableLinear {
    /// Create a new linear layer with He initialization
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Initialization
    ///
    /// Uses He initialization: scale = √(2/in_features)
    /// This helps prevent vanishing/exploding gradients in deep networks.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let scale = (2.0 / in_features as f32).sqrt();
        Self {
            weight: Tensor::new(
                random_init(in_features * out_features, seed, scale),
                vec![in_features, out_features],
            ),
            bias: Tensor::new(vec![0.0; out_features], vec![out_features]),
        }
    }

    /// Forward pass
    ///
    /// Computes y = x @ W + b and caches x for backward pass
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, in_features]
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache) where:
    /// - output: [seq_len, out_features]
    /// - cache: stores x for backward pass
    pub fn forward(&self, x: &Tensor) -> (Tensor, LinearCache) {
        let y = x.matmul(&self.weight).add(&self.bias);
        let cache = LinearCache { x: x.clone() };
        (y, cache)
    }

    /// Backward pass
    ///
    /// Computes gradients for weights, bias, and input
    ///
    /// # Arguments
    ///
    /// * `grad_out` - Gradient from next layer [seq_len, out_features]
    /// * `cache` - Cached values from forward pass
    ///
    /// # Returns
    ///
    /// Gradients for weight, bias, and input
    pub fn backward(&self, grad_out: &Tensor, cache: &LinearCache) -> LinearGradients {
        // grad_W = x^T @ grad_out
        let grad_weight = cache.x.transpose(-2, -1).matmul(grad_out);

        // grad_b = sum(grad_out) along all dims except last
        let grad_bias_data: Vec<f32> = (0..self.bias.data.len())
            .map(|i| {
                let mut sum = 0.0;
                for row in 0..grad_out.shape[0] {
                    sum += grad_out.data[row * grad_out.shape[1] + i];
                }
                sum
            })
            .collect();
        let grad_bias = Tensor::new(grad_bias_data, self.bias.shape.clone());

        // grad_x = grad_out @ W^T
        let grad_x = grad_out.matmul(&self.weight.transpose(-2, -1));

        LinearGradients {
            weight: grad_weight,
            bias: grad_bias,
            x: grad_x,
        }
    }
}

/// Cache for linear layer backward pass
pub struct LinearCache {
    pub x: Tensor,
}

/// Gradients for linear layer
pub struct LinearGradients {
    pub weight: Tensor,
    pub bias: Tensor,
    pub x: Tensor, // Gradient to pass to previous layer
}
