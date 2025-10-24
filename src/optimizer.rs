//! AdamW Optimizer Implementation
//!
//! This module implements the AdamW (Adam with decoupled Weight decay) optimizer,
//! the standard optimizer for training transformer models like GPT-2.
//!
//! ## What is AdamW?
//!
//! AdamW improves upon Adam by decoupling weight decay from the gradient-based
//! optimization. It combines:
//! - **Momentum**: Smooths gradient updates using exponential moving average
//! - **RMSProp**: Adapts learning rate per-parameter based on gradient history
//! - **Decoupled weight decay**: L2 regularization applied directly to weights
//!
//! ## Algorithm
//!
//! For each parameter θ with gradient g:
//!
//! ```text
//! # AdamW update:
//! θ = θ * (1 - α * λ)              # Weight decay (if applicable)
//! m = β₁ * m + (1 - β₁) * g        # First moment (momentum)
//! v = β₂ * v + (1 - β₂) * g²       # Second moment (variance)
//! m_hat = m / (1 - β₁^t)           # Bias correction
//! v_hat = v / (1 - β₂^t)           # Bias correction
//! θ = θ - α * m_hat / (√v_hat + ε) # Parameter update
//! ```
//!
//! where:
//! - α (alpha/lr) = learning rate (typically 3e-4 for transformers)
//! - λ (lambda/weight_decay) = 0.1 (typical for transformers)
//! - β₁ (beta1) = 0.9 (momentum decay rate)
//! - β₂ (beta2) = 0.95 (variance decay rate, lower than Adam's 0.999)
//! - ε (epsilon) = 1e-8 (numerical stability)
//! - t = training step number
//!
//! ## Why AdamW over Adam?
//!
//! **vs Adam**:
//! - Better generalization through proper L2 regularization
//! - Weight decay doesn't interact with adaptive learning rates
//! - Standard choice for modern transformer training (GPT, BERT, etc.)
//!
//! **vs SGD**:
//! - Faster convergence (fewer training steps)
//! - Less sensitive to learning rate choice
//! - Adaptive per-parameter learning rates
//!
//! ## Selective Weight Decay
//!
//! Following best practices from modern transformer training, weight decay is
//! applied **only to 2D tensors (weight matrices)**, not to:
//! - Biases (1D tensors)
//! - LayerNorm parameters (1D tensors)
//! - Embeddings (optionally, configurable)
//!
//! This selective application prevents over-regularization of scale/shift
//! parameters while still providing L2 regularization benefits for weight matrices.
//!
//! ## Bias Correction
//!
//! The bias correction terms `(1 - β^t)` are critical. Without them, m and v
//! are biased toward zero during early training steps. The correction ensures
//! the optimizer works well from the first step.
//!
//! ## Implementation Notes
//!
//! This implementation mirrors the GPT-2 model structure exactly:
//! - Separate moment vectors for each parameter
//! - Parallel updates using Rayon for performance
//! - Automatic fallback to sequential for small tensors
//!
//! ## Example
//!
//! ```rust,no_run
//! use feste::optimizer::{AdamWOptimizer, adamw_update};
//! use feste::gpt2_trainable::TrainableGPT2;
//! # use feste::Config;
//!
//! # let config = Config::tiny(512);
//! # let mut model = TrainableGPT2::new(&config);
//! // Initialize optimizer
//! let mut optimizer = AdamWOptimizer::new(&model);
//!
//! // Training loop
//! for step in 0..num_steps {
//!     // ... forward pass, compute gradients ...
//!     # let grads = todo!();
//!
//!     // Update parameters with weight decay
//!     adamw_update(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
//! }
//! ```
//!
//! ## References
//!
//! - Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
//!   https://arxiv.org/abs/1711.05101
//! - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
//!   https://arxiv.org/abs/1412.6980

use crate::gpt2_trainable::{GPT2Gradients, TrainableGPT2};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// AdamW optimizer state
///
/// Maintains first and second moment estimates (m and v) for all model parameters.
/// The structure mirrors `GPT2Gradients` to ensure every parameter has corresponding
/// optimizer state.
///
/// # Fields
///
/// - **m_* (first moment)**: Exponential moving average of gradients (momentum)
/// - **v_* (second moment)**: Exponential moving average of squared gradients (variance)
/// - **beta1**: Momentum decay rate (default: 0.9)
/// - **beta2**: Variance decay rate (default: 0.95, lower than Adam's 0.999)
/// - **epsilon**: Numerical stability constant (default: 1e-8)
/// - **step**: Training step count (for bias correction)
pub struct AdamWOptimizer {
    // First moment (momentum) - matches GPT2Gradients structure
    pub m_token_embedding: Tensor,
    pub m_position_embedding: Tensor,
    pub m_block_states: Vec<BlockAdamState>,
    pub m_ln_final_gamma: Tensor,
    pub m_ln_final_beta: Tensor,
    pub m_output_weight: Tensor,

    // Second moment (variance) - matches GPT2Gradients structure
    pub v_token_embedding: Tensor,
    pub v_position_embedding: Tensor,
    pub v_block_states: Vec<BlockAdamState>,
    pub v_ln_final_gamma: Tensor,
    pub v_ln_final_beta: Tensor,
    pub v_output_weight: Tensor,

    // Hyperparameters
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step: usize,
}

/// Optimizer state for a single transformer block
///
/// Stores moment estimates for all parameters in a transformer block:
/// - Two LayerNorm layers (scale and shift parameters)
/// - Self-attention mechanism (Q, K, V, output projections)
/// - MLP feedforward network (two linear layers)
pub struct BlockAdamState {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: AttentionAdamState,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: MLPAdamState,
}

/// Optimizer state for attention mechanism
///
/// Stores moment estimates for all attention parameters:
/// - Q, K, V projection matrices and biases
/// - Output projection matrix and bias
pub struct AttentionAdamState {
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
}

/// Optimizer state for MLP (feedforward network)
///
/// Stores moment estimates for the two linear layers:
/// - fc1: First projection (n_embd → 4*n_embd)
/// - fc2: Second projection (4*n_embd → n_embd)
pub struct MLPAdamState {
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer for the given model
    ///
    /// Initializes all moment estimates to zero. The optimizer state mirrors
    /// the model structure exactly, ensuring every parameter has optimizer state.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to optimize
    ///
    /// # Returns
    ///
    /// Optimizer with:
    /// - All moment estimates initialized to zero
    /// - Standard hyperparameters (β₁=0.9, β₂=0.95, ε=1e-8)
    /// - Step counter at 0
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::optimizer::AdamWOptimizer;
    /// # use feste::gpt2_trainable::TrainableGPT2;
    /// # use feste::Config;
    /// let config = Config::tiny(512);
    /// let model = TrainableGPT2::new(&config);
    /// let optimizer = AdamWOptimizer::new(&model);
    /// ```
    pub fn new(model: &TrainableGPT2) -> Self {
        // Initialize all momentum and variance tensors to zero
        let m_token_embedding = Tensor::zeros(model.token_embedding.shape.clone());
        let m_position_embedding = Tensor::zeros(model.position_embedding.shape.clone());
        let m_ln_final_gamma = Tensor::zeros(model.ln_final.gamma.shape.clone());
        let m_ln_final_beta = Tensor::zeros(model.ln_final.beta.shape.clone());
        let m_output_weight = Tensor::zeros(model.output_weight.shape.clone());

        let v_token_embedding = Tensor::zeros(model.token_embedding.shape.clone());
        let v_position_embedding = Tensor::zeros(model.position_embedding.shape.clone());
        let v_ln_final_gamma = Tensor::zeros(model.ln_final.gamma.shape.clone());
        let v_ln_final_beta = Tensor::zeros(model.ln_final.beta.shape.clone());
        let v_output_weight = Tensor::zeros(model.output_weight.shape.clone());

        let mut m_block_states = Vec::new();
        let mut v_block_states = Vec::new();

        for block in &model.blocks {
            let m_block = BlockAdamState {
                ln1_gamma: Tensor::zeros(block.ln1.gamma.shape.clone()),
                ln1_beta: Tensor::zeros(block.ln1.beta.shape.clone()),
                attn: AttentionAdamState {
                    q_weight: Tensor::zeros(block.attn.q_proj.weight.shape.clone()),
                    q_bias: Tensor::zeros(block.attn.q_proj.bias.shape.clone()),
                    k_weight: Tensor::zeros(block.attn.k_proj.weight.shape.clone()),
                    k_bias: Tensor::zeros(block.attn.k_proj.bias.shape.clone()),
                    v_weight: Tensor::zeros(block.attn.v_proj.weight.shape.clone()),
                    v_bias: Tensor::zeros(block.attn.v_proj.bias.shape.clone()),
                    out_weight: Tensor::zeros(block.attn.out_proj.weight.shape.clone()),
                    out_bias: Tensor::zeros(block.attn.out_proj.bias.shape.clone()),
                },
                ln2_gamma: Tensor::zeros(block.ln2.gamma.shape.clone()),
                ln2_beta: Tensor::zeros(block.ln2.beta.shape.clone()),
                mlp: MLPAdamState {
                    fc1_weight: Tensor::zeros(block.mlp.fc1.weight.shape.clone()),
                    fc1_bias: Tensor::zeros(block.mlp.fc1.bias.shape.clone()),
                    fc2_weight: Tensor::zeros(block.mlp.fc2.weight.shape.clone()),
                    fc2_bias: Tensor::zeros(block.mlp.fc2.bias.shape.clone()),
                },
            };

            let v_block = BlockAdamState {
                ln1_gamma: Tensor::zeros(block.ln1.gamma.shape.clone()),
                ln1_beta: Tensor::zeros(block.ln1.beta.shape.clone()),
                attn: AttentionAdamState {
                    q_weight: Tensor::zeros(block.attn.q_proj.weight.shape.clone()),
                    q_bias: Tensor::zeros(block.attn.q_proj.bias.shape.clone()),
                    k_weight: Tensor::zeros(block.attn.k_proj.weight.shape.clone()),
                    k_bias: Tensor::zeros(block.attn.k_proj.bias.shape.clone()),
                    v_weight: Tensor::zeros(block.attn.v_proj.weight.shape.clone()),
                    v_bias: Tensor::zeros(block.attn.v_proj.bias.shape.clone()),
                    out_weight: Tensor::zeros(block.attn.out_proj.weight.shape.clone()),
                    out_bias: Tensor::zeros(block.attn.out_proj.bias.shape.clone()),
                },
                ln2_gamma: Tensor::zeros(block.ln2.gamma.shape.clone()),
                ln2_beta: Tensor::zeros(block.ln2.beta.shape.clone()),
                mlp: MLPAdamState {
                    fc1_weight: Tensor::zeros(block.mlp.fc1.weight.shape.clone()),
                    fc1_bias: Tensor::zeros(block.mlp.fc1.bias.shape.clone()),
                    fc2_weight: Tensor::zeros(block.mlp.fc2.weight.shape.clone()),
                    fc2_bias: Tensor::zeros(block.mlp.fc2.bias.shape.clone()),
                },
            };

            m_block_states.push(m_block);
            v_block_states.push(v_block);
        }

        Self {
            m_token_embedding,
            m_position_embedding,
            m_block_states,
            m_ln_final_gamma,
            m_ln_final_beta,
            m_output_weight,
            v_token_embedding,
            v_position_embedding,
            v_block_states,
            v_ln_final_gamma,
            v_ln_final_beta,
            v_output_weight,
            beta1: 0.9,
            beta2: 0.95, // Lower than Adam's 0.999, standard for transformers
            epsilon: 1e-8,
            step: 0,
        }
    }

    /// Create a shallow copy for checkpointing
    ///
    /// Clones all tensors to create an independent copy of the optimizer state.
    /// Used when saving checkpoints to disk.
    ///
    /// # Returns
    ///
    /// New optimizer with cloned state (safe to save/serialize)
    pub fn clone_shallow(&self) -> Self {
        Self {
            m_token_embedding: self.m_token_embedding.clone(),
            m_position_embedding: self.m_position_embedding.clone(),
            m_block_states: self
                .m_block_states
                .iter()
                .map(|b| BlockAdamState {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    attn: AttentionAdamState {
                        q_weight: b.attn.q_weight.clone(),
                        q_bias: b.attn.q_bias.clone(),
                        k_weight: b.attn.k_weight.clone(),
                        k_bias: b.attn.k_bias.clone(),
                        v_weight: b.attn.v_weight.clone(),
                        v_bias: b.attn.v_bias.clone(),
                        out_weight: b.attn.out_weight.clone(),
                        out_bias: b.attn.out_bias.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: MLPAdamState {
                        fc1_weight: b.mlp.fc1_weight.clone(),
                        fc1_bias: b.mlp.fc1_bias.clone(),
                        fc2_weight: b.mlp.fc2_weight.clone(),
                        fc2_bias: b.mlp.fc2_bias.clone(),
                    },
                })
                .collect(),
            m_ln_final_gamma: self.m_ln_final_gamma.clone(),
            m_ln_final_beta: self.m_ln_final_beta.clone(),
            m_output_weight: self.m_output_weight.clone(),
            v_token_embedding: self.v_token_embedding.clone(),
            v_position_embedding: self.v_position_embedding.clone(),
            v_block_states: self
                .v_block_states
                .iter()
                .map(|b| BlockAdamState {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    attn: AttentionAdamState {
                        q_weight: b.attn.q_weight.clone(),
                        q_bias: b.attn.q_bias.clone(),
                        k_weight: b.attn.k_weight.clone(),
                        k_bias: b.attn.k_bias.clone(),
                        v_weight: b.attn.v_weight.clone(),
                        v_bias: b.attn.v_bias.clone(),
                        out_weight: b.attn.out_weight.clone(),
                        out_bias: b.attn.out_bias.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: MLPAdamState {
                        fc1_weight: b.mlp.fc1_weight.clone(),
                        fc1_bias: b.mlp.fc1_bias.clone(),
                        fc2_weight: b.mlp.fc2_weight.clone(),
                        fc2_bias: b.mlp.fc2_bias.clone(),
                    },
                })
                .collect(),
            v_ln_final_gamma: self.v_ln_final_gamma.clone(),
            v_ln_final_beta: self.v_ln_final_beta.clone(),
            v_output_weight: self.v_output_weight.clone(),
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
        }
    }
}

/// AdamW optimizer parameter update
///
/// Implements the complete AdamW algorithm with decoupled weight decay:
/// 1. Applies weight decay directly to weight matrices (not biases/LayerNorm)
/// 2. Updates moment estimates (m and v)
/// 3. Applies bias correction
/// 4. Updates model parameters using adaptive learning rates
///
/// # Arguments
///
/// * `model` - Model to update (modified in place)
/// * `grads` - Gradients from backpropagation
/// * `optimizer` - Optimizer state (moment estimates updated in place)
/// * `lr` - Learning rate (alpha in the AdamW paper)
/// * `weight_decay` - Weight decay coefficient (lambda, typically 0.1)
///
/// # Algorithm
///
/// For each parameter θ with gradient g:
/// ```text
/// θ = θ * (1 - lr * weight_decay)  # Weight decay (2D tensors only)
/// m = β₁*m + (1-β₁)*g              # Update momentum
/// v = β₂*v + (1-β₂)*g²             # Update variance
/// m_hat = m / (1 - β₁^step)        # Bias correction
/// v_hat = v / (1 - β₂^step)        # Bias correction
/// θ -= lr * m_hat / (√v_hat + ε)   # Update parameter
/// ```
///
/// # Selective Weight Decay
///
/// Weight decay is applied only to 2D tensors (weight matrices), not to:
/// - 1D tensors (biases, LayerNorm parameters)
/// - Embeddings (following common practice)
///
/// # Performance
///
/// Uses parallel computation via Rayon for tensors with >1000 elements.
/// Small tensors use sequential updates to avoid parallelization overhead.
///
/// # Example
///
/// ```rust,no_run
/// # use feste::optimizer::{AdamWOptimizer, adamw_update};
/// # use feste::gpt2_trainable::TrainableGPT2;
/// # use feste::Config;
/// # let config = Config::tiny(512);
/// # let mut model = TrainableGPT2::new(&config);
/// # let grads = todo!();
/// let mut optimizer = AdamWOptimizer::new(&model);
///
/// // Training step with weight decay
/// adamw_update(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
/// ```
pub fn adamw_update(
    model: &mut TrainableGPT2,
    grads: &GPT2Gradients,
    optimizer: &mut AdamWOptimizer,
    lr: f32,
    weight_decay: f32,
) {
    optimizer.step += 1;
    let step = optimizer.step as f32;

    // Bias correction factors
    // These correct for initialization bias (m and v start at 0)
    let bias_correction1 = 1.0 - optimizer.beta1.powf(step);
    let bias_correction2 = 1.0 - optimizer.beta2.powf(step);

    let beta1 = optimizer.beta1;
    let beta2 = optimizer.beta2;
    let epsilon = optimizer.epsilon;

    // Helper macro to update a parameter with AdamW
    // Parallelizes for large tensors, sequential for small ones
    // apply_decay: whether to apply weight decay (only for 2D weight matrices)
    macro_rules! adamw_update_param {
        ($param:expr, $grad:expr, $m:expr, $v:expr, $apply_decay:expr) => {
            // Parallelize for large tensors (>1000 elements)
            if $param.data.len() > 1000 {
                $param
                    .data
                    .par_iter_mut()
                    .zip($grad.data.par_iter())
                    .zip($m.data.par_iter_mut().zip($v.data.par_iter_mut()))
                    .for_each(|((param_val, &grad_val), (m_val, v_val))| {
                        // WEIGHT DECAY: Apply before Adam update (decoupled)
                        if $apply_decay {
                            *param_val *= 1.0 - lr * weight_decay;
                        }

                        // Update biased first moment estimate (momentum)
                        *m_val = beta1 * *m_val + (1.0 - beta1) * grad_val;

                        // Update biased second moment estimate (variance)
                        *v_val = beta2 * *v_val + (1.0 - beta2) * grad_val * grad_val;

                        // Compute bias-corrected first moment
                        let m_hat = *m_val / bias_correction1;

                        // Compute bias-corrected second moment
                        let v_hat = *v_val / bias_correction2;

                        // Update parameter (Adam step)
                        *param_val -= lr * m_hat / (v_hat.sqrt() + epsilon);
                    });
            } else {
                // Sequential for small tensors to avoid parallelization overhead
                for i in 0..$param.data.len() {
                    // WEIGHT DECAY: Apply before Adam update (decoupled)
                    if $apply_decay {
                        $param.data[i] *= 1.0 - lr * weight_decay;
                    }

                    let g = $grad.data[i];

                    // Update biased first moment estimate (momentum)
                    $m.data[i] = beta1 * $m.data[i] + (1.0 - beta1) * g;

                    // Update biased second moment estimate (variance)
                    $v.data[i] = beta2 * $v.data[i] + (1.0 - beta2) * g * g;

                    // Compute bias-corrected first moment
                    let m_hat = $m.data[i] / bias_correction1;

                    // Compute bias-corrected second moment
                    let v_hat = $v.data[i] / bias_correction2;

                    // Update parameter (Adam step)
                    $param.data[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        };
    }

    // Update embeddings (no weight decay - common practice)
    adamw_update_param!(
        model.token_embedding,
        grads.token_embedding,
        optimizer.m_token_embedding,
        optimizer.v_token_embedding,
        false // No decay on embeddings
    );
    adamw_update_param!(
        model.position_embedding,
        grads.position_embedding,
        optimizer.m_position_embedding,
        optimizer.v_position_embedding,
        false // No decay on embeddings
    );

    // Update all transformer blocks
    for ((block, block_grads), (m_block, v_block)) in
        model.blocks.iter_mut().zip(&grads.block_grads).zip(
            optimizer
                .m_block_states
                .iter_mut()
                .zip(optimizer.v_block_states.iter_mut()),
        )
    {
        // LayerNorm 1 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln1.gamma,
            block_grads.ln1_gamma,
            m_block.ln1_gamma,
            v_block.ln1_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln1.beta,
            block_grads.ln1_beta,
            m_block.ln1_beta,
            v_block.ln1_beta,
            false // No decay on LayerNorm
        );

        // Self-attention (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.attn.q_proj.weight,
            block_grads.attn.q_weight,
            m_block.attn.q_weight,
            v_block.attn.q_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.q_proj.bias,
            block_grads.attn.q_bias,
            m_block.attn.q_bias,
            v_block.attn.q_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.k_proj.weight,
            block_grads.attn.k_weight,
            m_block.attn.k_weight,
            v_block.attn.k_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.k_proj.bias,
            block_grads.attn.k_bias,
            m_block.attn.k_bias,
            v_block.attn.k_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.v_proj.weight,
            block_grads.attn.v_weight,
            m_block.attn.v_weight,
            v_block.attn.v_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.v_proj.bias,
            block_grads.attn.v_bias,
            m_block.attn.v_bias,
            v_block.attn.v_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.out_proj.weight,
            block_grads.attn.out_weight,
            m_block.attn.out_weight,
            v_block.attn.out_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.out_proj.bias,
            block_grads.attn.out_bias,
            m_block.attn.out_bias,
            v_block.attn.out_bias,
            false // No decay on bias
        );

        // LayerNorm 2 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln2.gamma,
            block_grads.ln2_gamma,
            m_block.ln2_gamma,
            v_block.ln2_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln2.beta,
            block_grads.ln2_beta,
            m_block.ln2_beta,
            v_block.ln2_beta,
            false // No decay on LayerNorm
        );

        // MLP (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.mlp.fc1.weight,
            block_grads.mlp.fc1_weight,
            m_block.mlp.fc1_weight,
            v_block.mlp.fc1_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc1.bias,
            block_grads.mlp.fc1_bias,
            m_block.mlp.fc1_bias,
            v_block.mlp.fc1_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.mlp.fc2.weight,
            block_grads.mlp.fc2_weight,
            m_block.mlp.fc2_weight,
            v_block.mlp.fc2_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc2.bias,
            block_grads.mlp.fc2_bias,
            m_block.mlp.fc2_bias,
            v_block.mlp.fc2_bias,
            false // No decay on bias
        );
    }

    // Final layer norm (no decay on 1D scale/shift parameters)
    adamw_update_param!(
        model.ln_final.gamma,
        grads.ln_final_gamma,
        optimizer.m_ln_final_gamma,
        optimizer.v_ln_final_gamma,
        false // No decay on LayerNorm
    );
    adamw_update_param!(
        model.ln_final.beta,
        grads.ln_final_beta,
        optimizer.m_ln_final_beta,
        optimizer.v_ln_final_beta,
        false // No decay on LayerNorm
    );

    // Output projection weight (decay on 2D weight matrix)
    adamw_update_param!(
        model.output_weight,
        grads.output_weight,
        optimizer.m_output_weight,
        optimizer.v_output_weight,
        true // Decay on weight matrix
    );
}
