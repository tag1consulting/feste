//! GPT-2 Model Architecture
//!
//! This module implements a complete GPT-2 style transformer model from scratch.
//! The architecture consists of:
//!
//! - **Token and position embeddings**: Convert token IDs to vectors
//! - **Transformer blocks**: Stack of attention + feedforward layers
//! - **Layer normalization**: Stabilize activations
//! - **Linear projection**: Final layer to vocabulary logits
//!
//! ## Architecture Overview
//!
//! ```text
//! Input tokens [batch, seq_len]
//!     ↓
//! Token Embedding [batch, seq_len, n_embd]
//!     + Position Embedding [seq_len, n_embd]
//!     ↓
//! Transformer Block 1 (Attention + MLP)
//!     ↓
//! Transformer Block 2
//!     ↓
//!     ...
//!     ↓
//! Transformer Block N
//!     ↓
//! Layer Norm
//!     ↓
//! Linear → [batch, seq_len, vocab_size]
//! ```
//!
//! ## Forward Pass Only
//!
//! This implementation provides the **forward pass** for inference and understanding.
//! Training (backward pass, gradients, optimization) is not included in Phase 3.
//!
//! ## Example
//!
//! ```rust,no_run
//! use feste::{GPT2, Config};
//!
//! // Create a tiny model configuration
//! let config = Config::tiny(512); // 512 vocab size
//! let model = GPT2::new(&config);
//!
//! // Forward pass: tokens → logits
//! // let logits = model.forward(&token_ids);
//! ```

use crate::tensor::Tensor;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Model configuration
///
/// Defines the architecture hyperparameters for a GPT-2 style model.
///
/// # Fields
///
/// - `vocab_size`: Number of tokens in vocabulary
/// - `n_embd`: Embedding dimension (width of the model)
/// - `n_heads`: Number of attention heads per layer
/// - `n_layers`: Number of transformer blocks
/// - `block_size`: Maximum sequence length (context window)
///
/// # Parameter Count Formula
///
/// Approximate parameters:
/// ```text
/// embeddings ≈ vocab_size × n_embd × 2  (token + position embeddings)
/// per_layer ≈ 12 × n_embd²  (attention + MLP + layer norms)
/// total ≈ embeddings + (n_layers × per_layer)
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub block_size: usize,
    pub dropout_rate: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            n_embd: 768,       // Embedding dimension
            n_heads: 12,       // Number of attention heads
            n_layers: 12,      // Number of transformer blocks
            block_size: 1024,  // Max sequence length
            dropout_rate: 0.1, // Dropout probability
        }
    }
}

impl Config {
    /// Create a tiny config for quick experiments
    ///
    /// **~50K parameters** - Very fast, good for testing (2-5 minutes training)
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary (e.g., from tokenizer)
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 64,        // Very small embedding
            n_heads: 1,        // Single-head attention
            n_layers: 2,       // Shallow
            block_size: 64,    // Short context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    /// Create a small config for experiments
    ///
    /// **~200K parameters** (with small vocab) - Good balance of speed and capability
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 128,       // Small embedding
            n_heads: 1,        // Single-head attention
            n_layers: 3,       // Medium depth
            block_size: 128,   // Medium context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    /// Create a medium config
    ///
    /// **~4M parameters** (with small vocab) - Substantial capacity
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary
    pub fn medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 256,       // Medium embedding
            n_heads: 4,        // Multi-head attention
            n_layers: 4,       // Medium depth
            block_size: 256,   // Medium context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    /// Create GPT-2 Small configuration
    ///
    /// **~163M parameters** (with GPT-2 vocab of 50257 tokens)
    /// **~86M parameters** (with smaller demo vocab of 512 tokens)
    ///
    /// This matches OpenAI's GPT-2 Small architecture:
    /// - 768 dimensional embeddings
    /// - 12 transformer layers
    /// - 12 attention heads
    /// - 1024 token context window
    ///
    /// Note: OpenAI's GPT-2 Small is often quoted as "117-124M parameters" because
    /// they use weight tying (the token embedding matrix is reused as the LM head).
    /// Our implementation uses separate weights for clarity, resulting in ~163M
    /// parameters with the full GPT-2 vocabulary (50257 tokens). With weight tying,
    /// this configuration would be ~124M parameters.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of vocabulary (use 50257 for GPT-2 tokenizer)
    pub fn gpt2_small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 768,       // GPT-2 Small
            n_heads: 12,       // GPT-2 Small
            n_layers: 12,      // GPT-2 Small
            block_size: 1024,  // GPT-2 standard context
            dropout_rate: 0.1, // Dropout probability
        }
    }
}

//
// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================
//

/// GELU (Gaussian Error Linear Unit) activation
///
/// GELU is used in transformers instead of ReLU because it provides smoother
/// gradients and often performs better in practice.
///
/// # Formula
///
/// ```text
/// GELU(x) = x × Φ(x)
/// where Φ(x) is the cumulative distribution function of the standard normal
/// ```
///
/// # Approximation
///
/// We use the tanh approximation for efficiency:
///
/// ```text
/// GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
/// ```
///
/// This is faster than computing the exact CDF and is accurate enough for neural networks.
///
/// # Arguments
///
/// * `x` - Input tensor
///
/// # Returns
///
/// Tensor with GELU activation applied element-wise
pub fn gelu(x: &Tensor) -> Tensor {
    // Constants for the approximation
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let coeff = 0.044715_f32;

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let result: Vec<f32> = x
        .data
        .iter()
        .map(|&val| {
            let x_cubed = val * val * val;
            let inner = sqrt_2_over_pi * (val + coeff * x_cubed);
            0.5 * val * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::new(result, x.shape.clone())
}

//
// ============================================================================
// EMBEDDING LAYER
// ============================================================================
//

/// Token embedding layer
///
/// Converts token IDs to dense vectors. This is a learnable lookup table
/// where each token ID maps to a fixed-size embedding vector.
///
/// # Shape Transformation
///
/// ```text
/// Input:  [batch, seq_len]  (token IDs)
/// Output: [batch, seq_len, n_embd]  (embedding vectors)
/// ```
///
/// # Implementation
///
/// The embedding table has shape `[vocab_size, n_embd]`. For each token ID,
/// we look up the corresponding row in this table.
pub struct Embedding {
    /// Embedding weight matrix: [vocab_size, n_embd]
    pub weight: Tensor,
}

impl Embedding {
    /// Create a new embedding layer with random initialization
    ///
    /// Weights are initialized from N(0, 0.02) following GPT-2.
    /// This uses a normal distribution with mean 0 and standard deviation 0.02,
    /// which helps with gradient flow during training.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Number of tokens in vocabulary
    /// * `n_embd` - Embedding dimension
    pub fn new(vocab_size: usize, n_embd: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();
        let weight_data: Vec<f32> = (0..vocab_size * n_embd)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Self {
            weight: Tensor::new(weight_data, vec![vocab_size, n_embd]),
        }
    }

    /// Forward pass: look up embeddings for token IDs
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Shape [batch, seq_len] with token indices
    ///
    /// # Returns
    ///
    /// Embedding vectors of shape [batch, seq_len, n_embd]
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();
        let n_embd = self.weight.shape[1];

        let mut output = Vec::with_capacity(batch_size * seq_len * n_embd);

        for batch in token_ids {
            for &token_id in batch {
                assert!(
                    token_id < self.weight.shape[0],
                    "Token ID {} out of vocab range (vocab_size = {})",
                    token_id,
                    self.weight.shape[0]
                );
                // Copy the embedding vector for this token
                let start = token_id * n_embd;
                let end = start + n_embd;
                output.extend_from_slice(&self.weight.data[start..end]);
            }
        }

        Tensor::new(output, vec![batch_size, seq_len, n_embd])
    }
}

//
// ============================================================================
// LAYER NORMALIZATION
// ============================================================================
//

/// Layer normalization
///
/// Normalizes activations along the last dimension to have zero mean and unit variance.
/// This stabilizes training and is applied before each sublayer (attention and MLP).
///
/// # Formula
///
/// ```text
/// output = (input - mean) / sqrt(variance + eps) × gamma + beta
/// ```
///
/// where `gamma` and `beta` are learnable parameters.
///
/// # Why Layer Norm?
///
/// Unlike batch normalization (normalizes across batch), layer norm normalizes
/// across features for each sample independently. This works better for:
/// - Variable sequence lengths
/// - Small batch sizes
/// - Recurrent/transformer architectures
pub struct LayerNorm {
    /// Scale parameter (learnable): [n_embd]
    pub gamma: Tensor,
    /// Shift parameter (learnable): [n_embd]
    pub beta: Tensor,
    /// Small constant for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Feature dimension to normalize over
    /// * `eps` - Small constant to prevent division by zero (default: 1e-5)
    pub fn new(n_embd: usize, eps: f32) -> Self {
        // Initialize gamma to 1 (no scaling initially)
        let gamma = Tensor::new(vec![1.0; n_embd], vec![n_embd]);
        // Initialize beta to 0 (no shift initially)
        let beta = Tensor::new(vec![0.0; n_embd], vec![n_embd]);

        Self { gamma, beta, eps }
    }

    /// Forward pass: normalize along last dimension
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor, typically [batch, seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compute mean and variance along last dimension
        let mean = x.mean(-1, true);
        let variance = x.var(-1, true);

        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = x.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());

        // Scale and shift: normalized * gamma + beta
        normalized.mul(&self.gamma).add(&self.beta)
    }
}

//
// ============================================================================
// LINEAR LAYER
// ============================================================================
//

/// Linear (fully connected) layer
///
/// Applies an affine transformation: `y = x @ W + b`
///
/// # Shape Transformation
///
/// ```text
/// Input:  [*, in_features]
/// Output: [*, out_features]
/// ```
///
/// where `*` represents any number of leading dimensions (batch, sequence, etc.)
pub struct Linear {
    /// Weight matrix: [in_features, out_features]
    pub weight: Tensor,
    /// Bias vector: [out_features]
    pub bias: Tensor,
}

impl Linear {
    /// Create a new linear layer with random initialization
    ///
    /// Weights are initialized from N(0, 0.02) following GPT-2.
    /// This uses a normal distribution with mean 0 and standard deviation 0.02,
    /// which helps with gradient flow during training.
    /// Bias is initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| normal.sample(&mut rng))
            .collect();

        let weight = Tensor::new(weight_data, vec![in_features, out_features]);
        let bias = Tensor::zeros(vec![out_features]);

        Self { weight, bias }
    }

    /// Forward pass: y = x @ W + b
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [..., in_features]
    ///
    /// # Returns
    ///
    /// Output tensor [..., out_features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // For simplicity, we handle 3D input [batch, seq, in_features]
        // Reshape to 2D, matmul, then reshape back
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let in_features = x.shape[2];

        // Reshape to [batch * seq, in_features]
        let x_2d = x.reshape(&[batch_size * seq_len, in_features]);

        // Matrix multiply
        let y_2d = x_2d.matmul(&self.weight);

        // Reshape back to [batch, seq, out_features]
        let out_features = self.weight.shape[1];
        let y_3d = y_2d.reshape(&[batch_size, seq_len, out_features]);

        // Add bias (broadcasts automatically)
        y_3d.add(&self.bias)
    }
}

//
// ============================================================================
// ATTENTION MECHANISM
// ============================================================================
//

/// Multi-head self-attention
///
/// The core mechanism that allows the model to focus on different parts of
/// the input sequence when processing each position.
///
/// # Architecture
///
/// 1. **Linear projections**: Q, K, V from input
/// 2. **Split into heads**: Reshape to [batch, n_heads, seq, head_dim]
/// 3. **Scaled dot-product attention**: scores = Q @ K^T / √head_dim
/// 4. **Causal mask**: Prevent attending to future positions
/// 5. **Softmax**: Convert scores to attention weights
/// 6. **Weighted sum**: output = attention @ V
/// 7. **Concatenate heads** and project back
///
/// # Multi-Head Attention
///
/// Instead of one attention operation, we use multiple "heads" in parallel.
/// Each head can learn to attend to different aspects of the sequence.
///
/// For example with n_embd=256 and n_heads=4:
/// - Each head has head_dim = 256/4 = 64
/// - We compute 4 independent attention operations
/// - Concatenate results back to 256 dimensions
///
/// # Causal Masking
///
/// In language modeling, we predict the next token, so position `i` cannot
/// see positions `i+1, i+2, ...` (the future). We enforce this by setting
/// attention scores for future positions to -inf before softmax.
pub struct Attention {
    /// Combined Q, K, V projection: [n_embd, 3 * n_embd]
    pub c_attn: Linear,
    /// Output projection: [n_embd, n_embd]
    pub c_proj: Linear,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head (n_embd / n_heads)
    pub head_dim: usize,
}

impl Attention {
    /// Create a new attention layer
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    /// * `n_heads` - Number of attention heads
    pub fn new(n_embd: usize, n_heads: usize) -> Self {
        assert_eq!(n_embd % n_heads, 0, "n_embd must be divisible by n_heads");

        let head_dim = n_embd / n_heads;

        // Single linear layer computes Q, K, V in one shot
        let c_attn = Linear::new(n_embd, 3 * n_embd);
        // Output projection after concatenating heads
        let c_proj = Linear::new(n_embd, n_embd);

        Self {
            c_attn,
            c_proj,
            n_heads,
            head_dim,
        }
    }

    /// Forward pass: compute multi-head self-attention
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, n_embd] after attention
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let n_embd = x.shape[2];

        // === 1. Compute Q, K, V ===
        // c_attn projects to 3*n_embd (stacked Q, K, V)
        let qkv = self.c_attn.forward(x); // [batch, seq, 3*n_embd]

        // Split into Q, K, V
        let mut q_data = Vec::with_capacity(batch_size * seq_len * n_embd);
        let mut k_data = Vec::with_capacity(batch_size * seq_len * n_embd);
        let mut v_data = Vec::with_capacity(batch_size * seq_len * n_embd);

        for i in 0..batch_size * seq_len {
            let start = i * 3 * n_embd;
            q_data.extend_from_slice(&qkv.data[start..start + n_embd]);
            k_data.extend_from_slice(&qkv.data[start + n_embd..start + 2 * n_embd]);
            v_data.extend_from_slice(&qkv.data[start + 2 * n_embd..start + 3 * n_embd]);
        }

        let q = Tensor::new(q_data, vec![batch_size, seq_len, n_embd]);
        let k = Tensor::new(k_data, vec![batch_size, seq_len, n_embd]);
        let v = Tensor::new(v_data, vec![batch_size, seq_len, n_embd]);

        // === 2. Reshape for multi-head attention ===
        // [batch, seq, n_embd] -> [batch, n_heads, seq, head_dim]
        let q = self.split_heads(&q, batch_size, seq_len);
        let k = self.split_heads(&k, batch_size, seq_len);
        let v = self.split_heads(&v, batch_size, seq_len);

        // === 3. Transpose K for attention scores ===
        // K: [batch, n_heads, seq, head_dim] -> [batch, n_heads, head_dim, seq]
        let k_t = k.transpose(2, 3);

        // === 4. Compute attention scores ===
        // Q @ K^T: [batch, n_heads, seq, head_dim] @ [batch, n_heads, head_dim, seq]
        //       -> [batch, n_heads, seq, seq]
        let scores = q.matmul(&k_t);

        // === 5. Scale by sqrt(head_dim) ===
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = scores.mul_scalar(scale);

        // === 6. Apply causal mask ===
        let mask = self.create_causal_mask(seq_len);
        let scores = scores.masked_fill(&mask, f32::NEG_INFINITY);

        // === 7. Softmax to get attention weights ===
        let attn = scores.softmax(-1); // [batch, n_heads, seq, seq]

        // === 8. Apply attention to values ===
        // attn @ V: [batch, n_heads, seq, seq] @ [batch, n_heads, seq, head_dim]
        //        -> [batch, n_heads, seq, head_dim]
        let out = attn.matmul(&v);

        // === 9. Concatenate heads ===
        let out = self.merge_heads(&out, batch_size, seq_len);

        // === 10. Output projection ===
        self.c_proj.forward(&out)
    }

    /// Split into multiple attention heads
    ///
    /// [batch, seq, n_embd] -> [batch, n_heads, seq, head_dim]
    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // Reshape and transpose
        let mut result = vec![0.0; batch_size * self.n_heads * seq_len * self.head_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        let src_idx = (b * seq_len + s) * (self.n_heads * self.head_dim)
                            + h * self.head_dim
                            + d;
                        let dst_idx = ((b * self.n_heads + h) * seq_len + s) * self.head_dim + d;
                        result[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }

        Tensor::new(
            result,
            vec![batch_size, self.n_heads, seq_len, self.head_dim],
        )
    }

    /// Merge attention heads back
    ///
    /// [batch, n_heads, seq, head_dim] -> [batch, seq, n_embd]
    fn merge_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let n_embd = self.n_heads * self.head_dim;
        let mut result = vec![0.0; batch_size * seq_len * n_embd];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        let src_idx = ((b * self.n_heads + h) * seq_len + s) * self.head_dim + d;
                        let dst_idx = (b * seq_len + s) * n_embd + h * self.head_dim + d;
                        result[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }

        Tensor::new(result, vec![batch_size, seq_len, n_embd])
    }

    /// Create causal attention mask
    ///
    /// Returns a [seq_len, seq_len] mask where:
    /// - 0 = can attend (current or past)
    /// - 1 = cannot attend (future)
    ///
    /// For seq_len=4, mask looks like:
    /// ```text
    /// [0 1 1 1]  position 0 can only see itself
    /// [0 0 1 1]  position 1 can see 0,1
    /// [0 0 0 1]  position 2 can see 0,1,2
    /// [0 0 0 0]  position 3 can see all
    /// ```
    fn create_causal_mask(&self, seq_len: usize) -> Tensor {
        let mut mask_data = vec![0.0; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // j is in the future relative to i
                    mask_data[i * seq_len + j] = 1.0;
                }
            }
        }

        Tensor::new(mask_data, vec![seq_len, seq_len])
    }
}

//
// ============================================================================
// MLP (FEEDFORWARD) LAYER
// ============================================================================
//

/// Multi-layer perceptron (feedforward network)
///
/// Applied after attention in each transformer block. Consists of:
/// 1. Linear layer expanding to 4×n_embd (hidden dimension)
/// 2. GELU activation
/// 3. Linear layer projecting back to n_embd
///
/// # Why 4× expansion?
///
/// GPT-2 uses 4×n_embd as the hidden dimension in the feedforward layer.
/// This provides enough capacity for the network to learn complex transformations
/// while keeping the residual stream (n_embd) smaller for efficiency.
pub struct MLP {
    /// First linear layer: [n_embd, 4*n_embd]
    pub c_fc: Linear,
    /// Second linear layer: [4*n_embd, n_embd]
    pub c_proj: Linear,
}

impl MLP {
    /// Create a new MLP layer
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    pub fn new(n_embd: usize) -> Self {
        let hidden_dim = 4 * n_embd;
        let c_fc = Linear::new(n_embd, hidden_dim);
        let c_proj = Linear::new(hidden_dim, n_embd);

        Self { c_fc, c_proj }
    }

    /// Forward pass: expand → GELU → project
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Expand to hidden dimension
        let h = self.c_fc.forward(x);
        // Apply GELU activation
        let h = gelu(&h);
        // Project back to n_embd
        self.c_proj.forward(&h)
    }
}

//
// ============================================================================
// TRANSFORMER BLOCK
// ============================================================================
//

/// Single transformer block
///
/// Combines attention and feedforward layers with residual connections
/// and layer normalization.
///
/// # Architecture
///
/// ```text
/// Input
///   ↓
///   ├─→ LayerNorm → Attention ─→ + (residual)
///   ↓                            ↓
///   ├─→ LayerNorm → MLP ────────→ + (residual)
///   ↓
/// Output
/// ```
///
/// # Residual Connections
///
/// Each sublayer (attention and MLP) has a residual connection that
/// adds the input back to the output. This helps gradient flow during
/// training and allows building very deep networks.
///
/// # Pre-Norm vs Post-Norm
///
/// GPT-2 uses "pre-norm": layer norm is applied before each sublayer.
/// This is more stable than "post-norm" (after each sublayer) for deep networks.
pub struct Block {
    /// Layer norm before attention
    pub ln_1: LayerNorm,
    /// Multi-head attention
    pub attn: Attention,
    /// Layer norm before MLP
    pub ln_2: LayerNorm,
    /// Feedforward network
    pub mlp: MLP,
}

impl Block {
    /// Create a new transformer block
    ///
    /// # Arguments
    ///
    /// * `n_embd` - Embedding dimension
    /// * `n_heads` - Number of attention heads
    pub fn new(n_embd: usize, n_heads: usize) -> Self {
        Self {
            ln_1: LayerNorm::new(n_embd, 1e-5),
            attn: Attention::new(n_embd, n_heads),
            ln_2: LayerNorm::new(n_embd, 1e-5),
            mlp: MLP::new(n_embd),
        }
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Attention block with residual connection
        let x = x.add(&self.attn.forward(&self.ln_1.forward(x)));

        // MLP block with residual connection
        x.add(&self.mlp.forward(&self.ln_2.forward(&x)))
    }
}

//
// ============================================================================
// GPT-2 MODEL
// ============================================================================
//

/// Complete GPT-2 model
///
/// Combines all components into a full language model that can:
/// - Take token IDs as input
/// - Produce logits over the vocabulary as output
/// - Be used for text generation (with appropriate sampling)
///
/// # Architecture Summary
///
/// - Token embedding + position embedding
/// - N transformer blocks (attention + MLP)
/// - Final layer norm
/// - Linear projection to vocabulary
///
/// # Forward Pass Only
///
/// This implementation provides only the forward pass (inference).
/// Training would require:
/// - Backward pass (gradients through all layers)
/// - Optimizer (Adam with weight decay)
/// - Loss computation (cross-entropy)
/// - Gradient accumulation and clipping
///
/// These are not included in Phase 3 but could be added in future phases.
pub struct GPT2 {
    /// Model configuration
    pub config: Config,
    /// Token embedding layer
    pub token_embedding: Embedding,
    /// Position embedding layer
    pub position_embedding: Embedding,
    /// Stack of transformer blocks
    pub blocks: Vec<Block>,
    /// Final layer normalization
    pub ln_f: LayerNorm,
    /// Output projection to vocabulary (unembedding)
    pub lm_head: Linear,
}

impl GPT2 {
    /// Create a new GPT-2 model with random initialization
    ///
    /// All weights are initialized from N(0, 0.02) following the GPT-2 paper.
    /// Layer norm parameters (gamma, beta) are initialized to 1 and 0 respectively.
    /// Biases are initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// Initialized model ready for forward passes
    pub fn new(config: &Config) -> Self {
        // Create embeddings
        let token_embedding = Embedding::new(config.vocab_size, config.n_embd);
        let position_embedding = Embedding::new(config.block_size, config.n_embd);

        // Create transformer blocks
        let blocks = (0..config.n_layers)
            .map(|_| Block::new(config.n_embd, config.n_heads))
            .collect();

        // Final layer norm
        let ln_f = LayerNorm::new(config.n_embd, 1e-5);

        // Output projection to vocabulary
        let lm_head = Linear::new(config.n_embd, config.vocab_size);

        Self {
            config: config.clone(),
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
        }
    }

    /// Forward pass: tokens → logits
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs [batch_size][seq_len]
    ///
    /// # Returns
    ///
    /// Logits over vocabulary: [batch, seq_len, vocab_size]
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::{GPT2, Config};
    /// let config = Config::tiny(512);
    /// let model = GPT2::new(&config);
    ///
    /// let tokens = vec![vec![1, 2, 3, 4]]; // batch_size=1, seq_len=4
    /// let logits = model.forward(&tokens);
    /// // logits.shape = [1, 4, 512]
    /// ```
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        assert!(
            seq_len <= self.config.block_size,
            "Sequence length {} exceeds block_size {}",
            seq_len,
            self.config.block_size
        );

        // === 1. Token embeddings ===
        let mut x = self.token_embedding.forward(token_ids);

        // === 2. Position embeddings ===
        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions: Vec<Vec<usize>> = vec![(0..seq_len).collect()];
        let pos_emb = self.position_embedding.forward(&positions);

        // Broadcast position embeddings to batch size and add
        // pos_emb: [1, seq_len, n_embd] -> broadcast to [batch, seq_len, n_embd]
        for b in 0..batch_size {
            for s in 0..seq_len {
                for e in 0..self.config.n_embd {
                    let idx = (b * seq_len + s) * self.config.n_embd + e;
                    let pos_idx = s * self.config.n_embd + e;
                    x.data[idx] += pos_emb.data[pos_idx];
                }
            }
        }

        // === 3. Pass through transformer blocks ===
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // === 4. Final layer norm ===
        x = self.ln_f.forward(&x);

        // === 5. Project to vocabulary ===
        self.lm_head.forward(&x)
    }

    /// Count total number of parameters
    ///
    /// Useful for understanding model size and memory requirements
    ///
    /// # Returns
    ///
    /// Total number of learnable parameters
    pub fn count_parameters(&self) -> usize {
        let mut total = 0;

        // Token and position embeddings
        total += self.token_embedding.weight.data.len();
        total += self.position_embedding.weight.data.len();

        // Transformer blocks
        for block in &self.blocks {
            // Attention
            total += block.attn.c_attn.weight.data.len();
            total += block.attn.c_attn.bias.data.len();
            total += block.attn.c_proj.weight.data.len();
            total += block.attn.c_proj.bias.data.len();

            // MLP
            total += block.mlp.c_fc.weight.data.len();
            total += block.mlp.c_fc.bias.data.len();
            total += block.mlp.c_proj.weight.data.len();
            total += block.mlp.c_proj.bias.data.len();

            // Layer norms
            total += block.ln_1.gamma.data.len();
            total += block.ln_1.beta.data.len();
            total += block.ln_2.gamma.data.len();
            total += block.ln_2.beta.data.len();
        }

        // Final layer norm
        total += self.ln_f.gamma.data.len();
        total += self.ln_f.beta.data.len();

        // LM head
        total += self.lm_head.weight.data.len();
        total += self.lm_head.bias.data.len();

        total
    }
}
