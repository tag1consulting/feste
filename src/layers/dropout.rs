//! Dropout Layer
//!
//! Dropout is a regularization technique that randomly zeros out activations
//! during training to prevent overfitting. During inference, it passes values
//! through unchanged.

use crate::tensor::Tensor;

/// Trainable dropout layer
///
/// This layer randomly drops activations during training for regularization.
pub struct TrainableDropout {
    pub rate: f32,
    pub training: bool,
}

impl TrainableDropout {
    /// Create a new dropout layer
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout probability (0.0 = no dropout, 1.0 = drop all)
    pub fn new(rate: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&rate),
            "Dropout rate must be between 0.0 and 1.0"
        );
        Self {
            rate,
            training: true,
        }
    }

    /// Forward pass with caching for backward
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache) where cache stores the dropout mask
    pub fn forward(&self, x: &Tensor) -> (Tensor, DropoutCache) {
        if !self.training || self.rate == 0.0 {
            // No dropout - just pass through
            let cache = DropoutCache {
                mask: None,
                scale: 1.0,
            };
            return (x.clone(), cache);
        }

        if self.rate >= 1.0 {
            // Drop everything
            let cache = DropoutCache {
                mask: Some(vec![false; x.data.len()]),
                scale: 1.0,
            };
            return (Tensor::zeros(x.shape.clone()), cache);
        }

        // Apply dropout with scaling
        let scale = 1.0 / (1.0 - self.rate);
        let mut mask = Vec::with_capacity(x.data.len());
        let mut output = Tensor::zeros(x.shape.clone());

        for i in 0..x.data.len() {
            let keep = rand::random::<f32>() > self.rate;
            mask.push(keep);
            if keep {
                output.data[i] = x.data[i] * scale;
            }
        }

        let cache = DropoutCache {
            mask: Some(mask),
            scale,
        };

        (output, cache)
    }

    /// Backward pass through dropout
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient flowing back from next layer
    /// * `cache` - Cached dropout mask from forward pass
    ///
    /// # Returns
    ///
    /// Gradient with respect to input
    pub fn backward(&self, grad_output: &Tensor, cache: &DropoutCache) -> Tensor {
        if let Some(mask) = &cache.mask {
            // Apply the same mask to gradients
            let mut grad_input = Tensor::zeros(grad_output.shape.clone());
            for (i, &keep) in mask.iter().enumerate() {
                if keep {
                    grad_input.data[i] = grad_output.data[i] * cache.scale;
                }
                // else: gradient is zero (value was dropped)
            }
            grad_input
        } else {
            // No dropout was applied, just pass gradient through
            grad_output.clone()
        }
    }
}

/// Cache for dropout backward pass
pub struct DropoutCache {
    /// Dropout mask (true = kept, false = dropped)
    /// None if dropout was disabled
    pub mask: Option<Vec<bool>>,
    /// Scaling factor applied to kept values
    pub scale: f32,
}
