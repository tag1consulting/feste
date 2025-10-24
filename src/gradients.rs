//! Gradient Utilities
//!
//! This module provides utilities for working with gradients during training.
//! These operations are essential for training stability and monitoring.
//!
//! ## Components
//!
//! - **Gradient Norm Computation**: Measure the magnitude of gradients
//! - **Gradient Clipping**: Prevent gradient explosion by scaling
//!
//! ## Why Gradient Clipping?
//!
//! During training, occasional batches can produce very large gradients that
//! destabilize the model. Gradient clipping prevents this by scaling down
//! gradients when their norm exceeds a threshold.
//!
//! Without clipping:
//! ```text
//! Step 1000: Loss = 3.2
//! Step 1001: Loss = 287.5  (gradient explosion!)
//! Step 1002: Loss = NaN    (training failed)
//! ```
//!
//! With clipping:
//! ```text
//! Step 1000: Loss = 3.2
//! Step 1001: Loss = 3.3  (gradient was clipped)
//! Step 1002: Loss = 3.1  (recovered)
//! ```
//!
//! ## Algorithm
//!
//! ```text
//! norm = √(Σ gradient²)  // Compute L2 norm
//! if norm > max_norm:
//!     gradients *= (max_norm / norm)  // Scale proportionally
//! ```
//!
//! This ensures all gradients are scaled by the same factor, preserving
//! their relative magnitudes while limiting the total update magnitude.
//!
//! ## Example
//!
//! ```rust,no_run
//! use feste::gradients::{compute_grad_norm, clip_gradients};
//! # use feste::gpt2_trainable::GPT2Gradients;
//!
//! # let grads: GPT2Gradients = todo!();
//! // Compute gradient norm for monitoring
//! let norm = compute_grad_norm(&grads);
//! println!("Gradient norm: {:.4}", norm);
//!
//! // Clip if too large
//! let mut grads_clipped = grads;
//! clip_gradients(&mut grads_clipped, 1.0);
//! ```

use crate::gpt2_trainable::GPT2Gradients;
use rayon::prelude::*;

/// Compute the L2 norm of all gradients
///
/// The gradient norm is the square root of the sum of all squared gradient values
/// across all parameters in the model. This gives a single number representing
/// the overall magnitude of the gradient update.
///
/// # Arguments
///
/// * `grads` - Gradients for all model parameters
///
/// # Returns
///
/// The L2 norm: √(Σ g²) for all gradient values g
///
/// # Performance
///
/// Uses parallel computation via Rayon for better performance on multi-core CPUs.
/// The computation is parallelized within each tensor to maximize throughput.
///
/// # Example
///
/// ```rust,no_run
/// # use feste::gradients::compute_grad_norm;
/// # use feste::gpt2_trainable::GPT2Gradients;
/// # let grads: GPT2Gradients = todo!();
/// let norm = compute_grad_norm(&grads);
/// if norm > 5.0 {
///     println!("Warning: Large gradient norm: {:.2}", norm);
/// }
/// ```
pub fn compute_grad_norm(grads: &GPT2Gradients) -> f32 {
    // Helper to compute sum of squares in parallel
    let sum_sq_parallel = |data: &Vec<f32>| -> f32 { data.par_iter().map(|&val| val * val).sum() };

    let mut sum_sq = 0.0;

    // Token and position embeddings
    sum_sq += sum_sq_parallel(&grads.token_embedding.data);
    sum_sq += sum_sq_parallel(&grads.position_embedding.data);

    // All transformer blocks
    for block_grad in &grads.block_grads {
        // LayerNorm 1
        sum_sq += sum_sq_parallel(&block_grad.ln1_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln1_beta.data);

        // Attention
        sum_sq += sum_sq_parallel(&block_grad.attn.q_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.q_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_bias.data);

        // LayerNorm 2
        sum_sq += sum_sq_parallel(&block_grad.ln2_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln2_beta.data);

        // MLP (feedforward network)
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_bias.data);
    }

    // Final layer norm
    sum_sq += sum_sq_parallel(&grads.ln_final_gamma.data);
    sum_sq += sum_sq_parallel(&grads.ln_final_beta.data);

    // Output projection weight
    sum_sq += sum_sq_parallel(&grads.output_weight.data);

    sum_sq.sqrt()
}

/// Clip gradients to a maximum norm
///
/// When the gradient norm exceeds `max_norm`, all gradients are scaled
/// proportionally to bring the norm down to exactly `max_norm`. This prevents
/// gradient explosion while preserving the direction of the gradient update.
///
/// # Arguments
///
/// * `grads` - Gradients to clip (modified in place)
/// * `max_norm` - Maximum allowed gradient norm (typically 1.0)
///
/// # Algorithm
///
/// ```text
/// norm = compute_grad_norm(grads)
/// if norm > max_norm:
///     scale = max_norm / norm
///     for all gradients g:
///         g *= scale
/// ```
///
/// # Performance
///
/// Only performs scaling if clipping is needed. Uses parallel computation via
/// Rayon for better performance on multi-core CPUs.
///
/// # Example
///
/// ```rust,no_run
/// # use feste::gradients::clip_gradients;
/// # use feste::gpt2_trainable::GPT2Gradients;
/// # let mut grads: GPT2Gradients = todo!();
/// // Clip gradients to norm of 1.0 (standard practice)
/// clip_gradients(&mut grads, 1.0);
/// ```
pub fn clip_gradients(grads: &mut GPT2Gradients, max_norm: f32) {
    let norm = compute_grad_norm(grads);

    // Only clip if norm exceeds threshold
    if norm > max_norm {
        let scale = max_norm / norm;

        // Helper to scale tensor data in parallel
        let scale_parallel = |data: &mut Vec<f32>| {
            data.par_iter_mut().for_each(|val| *val *= scale);
        };

        // Scale all gradients by the same factor

        // Token and position embeddings
        scale_parallel(&mut grads.token_embedding.data);
        scale_parallel(&mut grads.position_embedding.data);

        // All transformer blocks
        for block_grad in &mut grads.block_grads {
            // LayerNorm 1
            scale_parallel(&mut block_grad.ln1_gamma.data);
            scale_parallel(&mut block_grad.ln1_beta.data);

            // Attention
            scale_parallel(&mut block_grad.attn.q_weight.data);
            scale_parallel(&mut block_grad.attn.q_bias.data);
            scale_parallel(&mut block_grad.attn.k_weight.data);
            scale_parallel(&mut block_grad.attn.k_bias.data);
            scale_parallel(&mut block_grad.attn.v_weight.data);
            scale_parallel(&mut block_grad.attn.v_bias.data);
            scale_parallel(&mut block_grad.attn.out_weight.data);
            scale_parallel(&mut block_grad.attn.out_bias.data);

            // LayerNorm 2
            scale_parallel(&mut block_grad.ln2_gamma.data);
            scale_parallel(&mut block_grad.ln2_beta.data);

            // MLP (feedforward network)
            scale_parallel(&mut block_grad.mlp.fc1_weight.data);
            scale_parallel(&mut block_grad.mlp.fc1_bias.data);
            scale_parallel(&mut block_grad.mlp.fc2_weight.data);
            scale_parallel(&mut block_grad.mlp.fc2_bias.data);
        }

        // Final layer norm
        scale_parallel(&mut grads.ln_final_gamma.data);
        scale_parallel(&mut grads.ln_final_beta.data);

        // Output projection weight
        scale_parallel(&mut grads.output_weight.data);
    }
}
