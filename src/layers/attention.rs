//! Self-Attention Mechanism
//!
//! Attention is the core innovation of transformers. It allows each position
//! to attend to all previous positions, learning contextual relationships.
//!
//! ## Scaled Dot-Product Attention
//!
//! ```text
//! Q, K, V = x @ W_q, x @ W_k, x @ W_v
//! scores = (Q @ K^T) / √d_k
//! attn_weights = softmax(masked_scores)
//! output = attn_weights @ V
//! ```
//!
//! ## Why Scaling?
//!
//! We divide by √d_k to prevent the dot products from growing too large,
//! which would push softmax into regions with vanishingly small gradients.
//!
//! ## Causal Masking
//!
//! For language modeling, we mask future positions so each token can only
//! attend to itself and previous tokens. This is crucial for autoregressive generation.
//!
//! ## Backward Pass
//!
//! The backward pass through attention involves:
//! 1. Backprop through output projection
//! 2. Backprop through attention-weighted sum (V)
//! 3. Backprop through softmax (with per-row gradients)
//! 4. Backprop through scaled dot-product
//! 5. Backprop through Q, K, V projections
//!
//! The softmax backward is particularly interesting - we need to account for
//! the fact that softmax couples all elements in each row.

use super::dropout::{DropoutCache, TrainableDropout};
use super::linear::{LinearCache, TrainableLinear};
use crate::tensor::Tensor;

/// Single-head self-attention
///
/// This implements one attention head. Multi-head attention would run multiple
/// copies of this in parallel.
pub struct TrainableSingleHeadAttention {
    pub q_proj: TrainableLinear,
    pub k_proj: TrainableLinear,
    pub v_proj: TrainableLinear,
    pub out_proj: TrainableLinear,
    pub attn_dropout: TrainableDropout,
    pub resid_dropout: TrainableDropout,
    pub n_embd: usize,
}

impl TrainableSingleHeadAttention {
    /// Create a new attention layer
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    /// * `dropout_rate` - Dropout probability
    /// * `seed` - Random seed for initialization
    pub fn new(n_embd: usize, dropout_rate: f32, seed: u64) -> Self {
        Self {
            q_proj: TrainableLinear::new(n_embd, n_embd, seed),
            k_proj: TrainableLinear::new(n_embd, n_embd, seed + 1),
            v_proj: TrainableLinear::new(n_embd, n_embd, seed + 2),
            out_proj: TrainableLinear::new(n_embd, n_embd, seed + 3),
            attn_dropout: TrainableDropout::new(dropout_rate),
            resid_dropout: TrainableDropout::new(dropout_rate),
            n_embd,
        }
    }

    /// Forward pass: scaled dot-product attention with causal masking
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache)
    pub fn forward(&self, x: &Tensor) -> (Tensor, AttentionCache) {
        let seq_len = x.shape[0];

        // Project to Q, K, V
        let (q, q_cache) = self.q_proj.forward(x);
        let (k, k_cache) = self.k_proj.forward(x);
        let (v, v_cache) = self.v_proj.forward(x);

        // Scaled dot-product attention
        let scale = (self.n_embd as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / scale);

        // Causal mask: prevent attending to future positions
        let mut mask = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            for j in i + 1..seq_len {
                mask[i * seq_len + j] = 1.0;
            }
        }
        let mask_tensor = Tensor::new(mask, vec![seq_len, seq_len]);
        let masked_scores = scores.masked_fill(&mask_tensor, -1e9);

        // Softmax -> attention weights
        let attn_weights = masked_scores.softmax(-1);

        // Apply dropout to attention weights
        let (attn_weights_dropped, attn_dropout_cache) = self.attn_dropout.forward(&attn_weights);

        // Apply attention to values
        let attn_out = attn_weights_dropped.matmul(&v);

        // Output projection
        let (y_proj, out_cache) = self.out_proj.forward(&attn_out);

        // Apply residual dropout
        let (y, resid_dropout_cache) = self.resid_dropout.forward(&y_proj);

        let cache = AttentionCache {
            x: x.clone(),
            q,
            k,
            v,
            attn_weights,
            #[allow(dead_code)]
            attn_out,
            q_cache,
            k_cache,
            v_cache,
            out_cache,
            attn_dropout_cache,
            resid_dropout_cache,
        };

        (y, cache)
    }

    /// Backward pass through attention
    ///
    /// # Arguments
    ///
    /// * `grad_out` - Gradient from next layer
    /// * `cache` - Cached values from forward pass
    ///
    /// # Returns
    ///
    /// Gradients for all parameters and input
    pub fn backward(&self, grad_out: &Tensor, cache: &AttentionCache) -> AttentionGradients {
        let seq_len = cache.x.shape[0];
        let scale = (self.n_embd as f32).sqrt();

        // Backprop through residual dropout
        let grad_y_proj = self
            .resid_dropout
            .backward(grad_out, &cache.resid_dropout_cache);

        // Backprop through output projection
        let out_grads = self.out_proj.backward(&grad_y_proj, &cache.out_cache);

        // Backprop through attention: grad_v = attn_weights^T @ grad_attn_out
        let grad_v = cache.attn_weights.transpose(-2, -1).matmul(&out_grads.x);

        // grad_attn_weights = grad_attn_out @ v^T
        let grad_attn_weights_dropped = out_grads.x.matmul(&cache.v.transpose(-2, -1));

        // Backprop through attention dropout
        let grad_attn_weights = self
            .attn_dropout
            .backward(&grad_attn_weights_dropped, &cache.attn_dropout_cache);

        // Backprop through softmax (per-row)
        // softmax gradient: grad_scores = attn * (grad_attn - sum(grad_attn * attn))
        let mut grad_scores_data = Vec::new();
        for i in 0..seq_len {
            let start = i * seq_len;
            let end = start + seq_len;
            let attn_row = &cache.attn_weights.data[start..end];
            let grad_attn_row = &grad_attn_weights.data[start..end];

            // Compute dot product for this row
            let dot_product: f32 = attn_row
                .iter()
                .zip(grad_attn_row.iter())
                .map(|(a, g)| a * g)
                .sum();

            // Apply softmax gradient formula
            for j in 0..seq_len {
                let grad_score = attn_row[j] * (grad_attn_row[j] - dot_product);
                grad_scores_data.push(grad_score);
            }
        }
        let grad_scores = Tensor::new(grad_scores_data, vec![seq_len, seq_len]);

        // Backprop through scaled Q @ K^T
        let grad_q = grad_scores.matmul(&cache.k).mul_scalar(1.0 / scale);
        let grad_k = grad_scores
            .transpose(-2, -1)
            .matmul(&cache.q)
            .mul_scalar(1.0 / scale);

        // Backprop through Q, K, V projections
        let q_grads = self.q_proj.backward(&grad_q, &cache.q_cache);
        let k_grads = self.k_proj.backward(&grad_k, &cache.k_cache);
        let v_grads = self.v_proj.backward(&grad_v, &cache.v_cache);

        // Accumulate gradients to input (Q, K, V all connect to same input)
        let mut grad_x_data = vec![0.0; cache.x.data.len()];
        for (i, grad_x_val) in grad_x_data.iter_mut().enumerate() {
            *grad_x_val = q_grads.x.data[i] + k_grads.x.data[i] + v_grads.x.data[i];
        }
        let grad_x = Tensor::new(grad_x_data, cache.x.shape.clone());

        AttentionGradients {
            q_weight: q_grads.weight,
            q_bias: q_grads.bias,
            k_weight: k_grads.weight,
            k_bias: k_grads.bias,
            v_weight: v_grads.weight,
            v_bias: v_grads.bias,
            out_weight: out_grads.weight,
            out_bias: out_grads.bias,
            x: grad_x,
        }
    }
}

/// Cache for attention backward pass
pub struct AttentionCache {
    pub x: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub attn_weights: Tensor,
    #[allow(dead_code)]
    pub attn_out: Tensor,
    pub q_cache: LinearCache,
    pub k_cache: LinearCache,
    pub v_cache: LinearCache,
    pub out_cache: LinearCache,
    pub attn_dropout_cache: DropoutCache,
    pub resid_dropout_cache: DropoutCache,
}

/// Gradients for attention
pub struct AttentionGradients {
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
    pub x: Tensor,
}
