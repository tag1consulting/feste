//! Multi-Layer Perceptron (MLP)
//!
//! The MLP is a two-layer feedforward network used in each transformer block.
//! It provides the model's capacity to learn complex transformations.
//!
//! ## Architecture
//!
//! ```text
//! x → Linear1 → GELU → Linear2 → y
//! ```
//!
//! ## Expansion Factor
//!
//! GPT-2 uses a 4× expansion:
//! - Input: n_embd
//! - Hidden: n_embd × 4
//! - Output: n_embd
//!
//! This expansion-then-compression pattern is crucial for model capacity.
//!
//! ## Why 4x?
//!
//! The 4× expansion is empirically determined:
//! - Provides enough capacity for complex transformations
//! - Not so large that it dominates parameter count
//! - Standard across many transformer architectures

use super::activation::{gelu_backward, gelu_forward};
use super::dropout::{DropoutCache, TrainableDropout};
use super::linear::{LinearCache, TrainableLinear};
use crate::tensor::Tensor;

/// MLP (feedforward network) with GELU activation
pub struct TrainableMLP {
    pub fc1: TrainableLinear,
    pub fc2: TrainableLinear,
    pub resid_dropout: TrainableDropout,
}

impl TrainableMLP {
    /// Create a new MLP with 4x expansion
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    /// * `dropout_rate` - Dropout probability
    /// * `seed` - Random seed for initialization
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        let hidden = n_embd * 4; // GPT-2 uses 4x expansion
        Self {
            fc1: TrainableLinear::new(n_embd, hidden, seed),
            fc2: TrainableLinear::new(hidden, n_embd, seed + 1000),
            resid_dropout: TrainableDropout::new(dropout_rate),
        }
    }

    /// Forward pass: x → fc1 → GELU → fc2
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, MLPCache) {
        let (h, fc1_cache) = self.fc1.forward(x);
        let h_activated = gelu_forward(&h);
        let (y_proj, fc2_cache) = self.fc2.forward(&h_activated);

        // Apply residual dropout
        let (y, resid_dropout_cache) = self.resid_dropout.forward(&y_proj);

        let cache = MLPCache {
            fc1_cache,
            h, // Save pre-activation for GELU backward
            #[allow(dead_code)]
            h_activated,
            fc2_cache,
            resid_dropout_cache,
        };

        (y, cache)
    }

    /// Backward pass through MLP
    ///
    /// Uses chain rule through fc2, GELU, and fc1
    ///
    /// # Arguments
    ///
    /// * `grad_out` - Gradient from next layer
    /// * `cache` - Cached values from forward pass
    ///
    /// # Returns
    ///
    /// Gradients for all parameters and input
    pub fn backward(&self, grad_out: &Tensor, cache: &MLPCache) -> MLPGradients {
        // Backprop through residual dropout
        let grad_y_proj = self.resid_dropout.backward(grad_out, &cache.resid_dropout_cache);

        // Backprop through fc2
        let fc2_grads = self.fc2.backward(&grad_y_proj, &cache.fc2_cache);

        // Backprop through GELU
        let grad_h = gelu_backward(&fc2_grads.x, &cache.h);

        // Backprop through fc1
        let fc1_grads = self.fc1.backward(&grad_h, &cache.fc1_cache);

        MLPGradients {
            fc1_weight: fc1_grads.weight,
            fc1_bias: fc1_grads.bias,
            fc2_weight: fc2_grads.weight,
            fc2_bias: fc2_grads.bias,
            x: fc1_grads.x,
        }
    }
}

/// Cache for MLP backward pass
pub struct MLPCache {
    pub fc1_cache: LinearCache,
    pub h: Tensor, // Pre-activation (needed for GELU backward)
    #[allow(dead_code)]
    pub h_activated: Tensor,
    pub fc2_cache: LinearCache,
    pub resid_dropout_cache: DropoutCache,
}

/// Gradients for MLP
pub struct MLPGradients {
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
    pub x: Tensor,
}
