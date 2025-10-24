//! Transformer Block
//!
//! A transformer block is the fundamental building block of GPT models.
//! It combines attention and feedforward layers with residual connections
//! and layer normalization.
//!
//! ## Architecture
//!
//! ```text
//! x → LayerNorm → Attention → (+) → LayerNorm → MLP → (+) → output
//! │                             ↑                        ↑
//! └─────────────────────────────┘                        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Pre-Norm vs Post-Norm
//!
//! We use **pre-norm** (LayerNorm before sublayers) rather than post-norm:
//! - More stable training
//! - Better gradient flow
//! - Standard in modern transformers (GPT-2, GPT-3)
//!
//! ## Residual Connections
//!
//! Residual connections are critical for training deep networks:
//! - Allow gradients to flow directly backward
//! - Prevent vanishing gradients
//! - Enable training very deep models (100+ layers)
//!
//! ## Backward Pass
//!
//! The backward pass through residual connections requires careful gradient accumulation.
//! At each residual connection, gradients split into two paths that must be summed.

use super::attention::{AttentionCache, AttentionGradients, TrainableSingleHeadAttention};
use super::layer_norm::{LayerNormCache, TrainableLayerNorm};
use super::mlp::{MLPCache, MLPGradients, TrainableMLP};
use crate::tensor::Tensor;

/// Transformer block combining attention and MLP with residuals
pub struct TrainableTransformerBlock {
    pub ln1: TrainableLayerNorm,
    pub attn: TrainableSingleHeadAttention,
    pub ln2: TrainableLayerNorm,
    pub mlp: TrainableMLP,
}

impl TrainableTransformerBlock {
    /// Create a new transformer block
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    /// * `dropout_rate` - Dropout probability
    /// * `seed` - Random seed for initialization
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        Self {
            ln1: TrainableLayerNorm::new(n_embd),
            attn: TrainableSingleHeadAttention::new(n_embd, dropout_rate, seed),
            ln2: TrainableLayerNorm::new(n_embd),
            mlp: TrainableMLP::new(n_embd, dropout_rate, seed + 1000),
        }
    }

    /// Forward pass: attention + MLP with residual connections
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, BlockCache) {
        // First sub-block: LayerNorm → Attention → Residual
        let (ln1_out, ln1_cache) = self.ln1.forward(x);
        let (attn_out, attn_cache) = self.attn.forward(&ln1_out);
        let x_after_attn = x.add(&attn_out); // Residual connection

        // Second sub-block: LayerNorm → MLP → Residual
        let (ln2_out, ln2_cache) = self.ln2.forward(&x_after_attn);
        let (mlp_out, mlp_cache) = self.mlp.forward(&ln2_out);
        let y = x_after_attn.add(&mlp_out); // Residual connection

        let cache = BlockCache {
            #[allow(dead_code)]
            x: x.clone(),
            ln1_cache,
            attn_cache,
            #[allow(dead_code)]
            x_after_attn,
            ln2_cache,
            mlp_cache,
        };

        (y, cache)
    }

    /// Backward pass through transformer block
    ///
    /// # Arguments
    ///
    /// * `grad_out` - Gradient from next layer
    /// * `cache` - Cached values from forward pass
    ///
    /// # Returns
    ///
    /// Gradients for all parameters and input
    pub fn backward(&self, grad_out: &Tensor, cache: &BlockCache) -> BlockGradients {
        // Backprop through second residual connection
        // Gradient flows to both MLP path and directly to first residual
        let grad_mlp_out = grad_out.clone();
        let mut grad_x_after_attn = grad_out.clone();

        // Backprop through MLP path
        let mlp_grads = self.mlp.backward(&grad_mlp_out, &cache.mlp_cache);
        let ln2_grads = self.ln2.backward(&mlp_grads.x, &cache.ln2_cache);

        // Accumulate gradient from LN2 path
        for i in 0..grad_x_after_attn.data.len() {
            grad_x_after_attn.data[i] += ln2_grads.x.data[i];
        }

        // Backprop through first residual connection
        let grad_attn_out = grad_x_after_attn.clone();
        let mut grad_x = grad_x_after_attn;

        // Backprop through attention path
        let attn_grads = self.attn.backward(&grad_attn_out, &cache.attn_cache);
        let ln1_grads = self.ln1.backward(&attn_grads.x, &cache.ln1_cache);

        // Accumulate gradient from LN1 path
        for i in 0..grad_x.data.len() {
            grad_x.data[i] += ln1_grads.x.data[i];
        }

        BlockGradients {
            ln1_gamma: ln1_grads.gamma,
            ln1_beta: ln1_grads.beta,
            attn: attn_grads,
            ln2_gamma: ln2_grads.gamma,
            ln2_beta: ln2_grads.beta,
            mlp: mlp_grads,
            x: grad_x,
        }
    }
}

/// Cache for transformer block backward pass
pub struct BlockCache {
    #[allow(dead_code)]
    pub x: Tensor,
    pub ln1_cache: LayerNormCache,
    pub attn_cache: AttentionCache,
    #[allow(dead_code)]
    pub x_after_attn: Tensor,
    pub ln2_cache: LayerNormCache,
    pub mlp_cache: MLPCache,
}

/// Gradients for transformer block
pub struct BlockGradients {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: AttentionGradients,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: MLPGradients,
    pub x: Tensor,
}
