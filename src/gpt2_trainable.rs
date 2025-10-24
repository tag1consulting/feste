//! Trainable GPT-2 Implementation
//!
//! This module implements the complete training infrastructure for GPT-2 style
//! transformers, including hand-coded backward passes for all layers.
//!
//! ## Overview
//!
//! Training a neural network requires three things:
//! 1. **Forward pass**: Compute predictions from inputs
//! 2. **Backward pass**: Compute gradients using the chain rule
//! 3. **Optimization**: Update parameters using gradients
//!
//! This module provides all three for a GPT-2 architecture.
//!
//! ## Architecture Components
//!
//! Each component has a trainable version with forward and backward methods:
//!
//! - **TrainableLinear**: Fully connected layer (y = x @ W + b)
//! - **TrainableLayerNorm**: Layer normalization with learnable scale/shift
//! - **TrainableMLP**: Two-layer feedforward network with GELU activation
//! - **TrainableSingleHeadAttention**: Self-attention mechanism
//! - **TrainableTransformerBlock**: Complete transformer block
//! - **TrainableGPT2**: Full model combining all components
//!
//! ## Backpropagation
//!
//! Backpropagation is the chain rule applied recursively:
//!
//! ```text
//! If y = f(g(x)), then dy/dx = (dy/df) * (df/dg) * (dg/dx)
//! ```
//!
//! For each layer, we implement:
//! - `forward()`: Computes output and caches values needed for backward
//! - `backward()`: Computes gradients from downstream gradients and cache
//!
//! ## Example: Linear Layer
//!
//! ```text
//! Forward:  y = x @ W + b
//! Backward:
//!   grad_W = x^T @ grad_y
//!   grad_b = sum(grad_y)
//!   grad_x = grad_y @ W^T
//! ```
//!
//! ## Optimization
//!
//! We implement the Adam optimizer, which maintains:
//! - **First moment (momentum)**: Exponential moving average of gradients
//! - **Second moment (variance)**: Exponential moving average of squared gradients
//! - **Bias correction**: Corrects for initialization bias in early steps
//!
//! Adam is more stable than SGD and requires less hyperparameter tuning.
//!
//! ## Gradient Clipping
//!
//! To prevent gradient explosion during training, we clip gradients to a
//! maximum norm. This is essential for training stability:
//!
//! ```text
//! if ||grad|| > max_norm:
//!     grad = grad * (max_norm / ||grad||)
//! ```
//!
//! ## Model Checkpointing
//!
//! The module includes full save/load functionality for:
//! - Model parameters (weights and biases)
//! - Optimizer state (momentum and variance)
//! - Training metadata (step number, config)
//!
//! This enables resuming training from checkpoints.
//!
//! ## Performance Optimizations
//!
//! Several optimizations keep training fast on CPU:
//! - **Parallel gradient computation**: Uses Rayon for parallel operations
//! - **Efficient memory layout**: Minimizes allocations during backward pass
//! - **Cache blocking**: Reuses matrix multiplication optimizations from tensor module
//!
//! ## Educational Focus
//!
//! Unlike PyTorch's autograd, every gradient is computed explicitly. This makes
//! the code longer but much clearer for learning. You can see exactly how
//! gradients flow through each layer.

use crate::tensor::Tensor;
use crate::tokenizer::BPETokenizer;
use crate::Config;

// Re-export layer types for backward compatibility
pub use crate::layers::{
    gelu_backward, gelu_forward, random_init, AttentionCache, AttentionGradients, BlockCache,
    BlockGradients, LayerNormCache, LinearCache, MLPCache, MLPGradients, TrainableDropout,
    TrainableLayerNorm, TrainableLinear, TrainableMLP, TrainableSingleHeadAttention,
    TrainableTransformerBlock,
};

// Re-export optimizer and gradient utilities for backward compatibility
pub use crate::gradients::{clip_gradients, compute_grad_norm};
pub use crate::optimizer::{
    adamw_update, AdamWOptimizer, AttentionAdamState, BlockAdamState, MLPAdamState,
};

pub struct TrainableGPT2 {
    pub(crate) token_embedding: Tensor,
    pub(crate) position_embedding: Tensor,
    pub(crate) blocks: Vec<TrainableTransformerBlock>,
    pub(crate) ln_final: TrainableLayerNorm,
    pub(crate) output_weight: Tensor,
    pub(crate) config: Config,
}

impl TrainableGPT2 {
    /// Save model weights to a binary file (inference only, for backward compatibility)
    /// For saving with training state, use Checkpoint::save() instead
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            model: self.clone_shallow(),
            optimizer: None,
            tokenizer: None,
            step: 0,
            best_val_loss: f32::INFINITY,
            best_val_step: 0,
        };
        checkpoint.save(path)
    }

    /// Create a shallow copy for saving (just references the tensors)
    fn clone_shallow(&self) -> TrainableGPT2 {
        TrainableGPT2 {
            token_embedding: self.token_embedding.clone(),
            position_embedding: self.position_embedding.clone(),
            blocks: self
                .blocks
                .iter()
                .map(|b| TrainableTransformerBlock {
                    ln1: TrainableLayerNorm {
                        gamma: b.ln1.gamma.clone(),
                        beta: b.ln1.beta.clone(),
                        eps: b.ln1.eps,
                    },
                    attn: TrainableSingleHeadAttention {
                        q_proj: TrainableLinear {
                            weight: b.attn.q_proj.weight.clone(),
                            bias: b.attn.q_proj.bias.clone(),
                        },
                        k_proj: TrainableLinear {
                            weight: b.attn.k_proj.weight.clone(),
                            bias: b.attn.k_proj.bias.clone(),
                        },
                        v_proj: TrainableLinear {
                            weight: b.attn.v_proj.weight.clone(),
                            bias: b.attn.v_proj.bias.clone(),
                        },
                        out_proj: TrainableLinear {
                            weight: b.attn.out_proj.weight.clone(),
                            bias: b.attn.out_proj.bias.clone(),
                        },
                        attn_dropout: TrainableDropout {
                            rate: b.attn.attn_dropout.rate,
                            training: b.attn.attn_dropout.training,
                        },
                        resid_dropout: TrainableDropout {
                            rate: b.attn.resid_dropout.rate,
                            training: b.attn.resid_dropout.training,
                        },
                        n_embd: b.attn.n_embd,
                    },
                    ln2: TrainableLayerNorm {
                        gamma: b.ln2.gamma.clone(),
                        beta: b.ln2.beta.clone(),
                        eps: b.ln2.eps,
                    },
                    mlp: TrainableMLP {
                        fc1: TrainableLinear {
                            weight: b.mlp.fc1.weight.clone(),
                            bias: b.mlp.fc1.bias.clone(),
                        },
                        fc2: TrainableLinear {
                            weight: b.mlp.fc2.weight.clone(),
                            bias: b.mlp.fc2.bias.clone(),
                        },
                        resid_dropout: TrainableDropout {
                            rate: b.mlp.resid_dropout.rate,
                            training: b.mlp.resid_dropout.training,
                        },
                    },
                })
                .collect(),
            ln_final: TrainableLayerNorm {
                gamma: self.ln_final.gamma.clone(),
                beta: self.ln_final.beta.clone(),
                eps: self.ln_final.eps,
            },
            output_weight: self.output_weight.clone(),
            config: self.config.clone(),
        }
    }

    /// Load model weights from a checkpoint file
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let checkpoint = Checkpoint::load(path)?;
        Ok(checkpoint.model)
    }

    pub fn new(config: &Config) -> Self {
        let vocab_size = config.vocab_size;
        let n_embd = config.n_embd;
        let block_size = config.block_size;
        let n_layers = config.n_layers;

        let embedding_scale = (1.0 / (n_embd as f32)).sqrt();
        let token_embedding = Tensor::new(
            random_init(vocab_size * n_embd, 12345, embedding_scale),
            vec![vocab_size, n_embd],
        );
        let position_embedding = Tensor::new(
            random_init(block_size * n_embd, 23456, embedding_scale),
            vec![block_size, n_embd],
        );

        let mut blocks = Vec::new();
        for i in 0..n_layers {
            blocks.push(TrainableTransformerBlock::new(
                n_embd,
                config.dropout_rate,
                10000 * (i as u64 + 1),
            ));
        }

        let weight_scale = (2.0 / (n_embd as f32)).sqrt();
        let output_weight = Tensor::new(
            random_init(n_embd * vocab_size, 78901, weight_scale),
            vec![n_embd, vocab_size],
        );

        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_final: TrainableLayerNorm::new(n_embd),
            output_weight,
            config: config.clone(),
        }
    }

    pub fn forward(&self, input_ids: &[usize]) -> (Tensor, GPT2Cache) {
        let seq_len = input_ids.len();
        let n_embd = self.config.n_embd;
        let vocab_size = self.config.vocab_size;

        // Embed tokens and positions
        let mut embedded = Vec::new();
        for (pos, &token_id) in input_ids.iter().enumerate() {
            let token_id = token_id.min(vocab_size - 1);
            let pos = pos.min(self.config.block_size - 1);

            let token_start = token_id * n_embd;
            let pos_start = pos * n_embd;

            for i in 0..n_embd {
                embedded.push(
                    self.token_embedding.data[token_start + i]
                        + self.position_embedding.data[pos_start + i],
                );
            }
        }

        let mut x = Tensor::new(embedded, vec![seq_len, n_embd]);

        // Forward through all transformer blocks
        let mut block_caches = Vec::new();
        for block in &self.blocks {
            let (x_next, cache) = block.forward(&x);
            block_caches.push(cache);
            x = x_next;
        }

        // Final layer norm
        let (x_normed, ln_final_cache) = self.ln_final.forward(&x);

        // Project to vocabulary
        let logits = x_normed.matmul(&self.output_weight);

        let cache = GPT2Cache {
            input_ids: input_ids.to_vec(),
            block_caches,
            ln_final_cache,
            x_before_final_ln: x,
        };

        (logits, cache)
    }

    pub fn compute_loss(&self, logits: &Tensor, targets: &[usize]) -> f32 {
        let seq_len = targets.len();
        let vocab_size = self.config.vocab_size;
        let mut total_loss = 0.0;

        for (i, &target) in targets.iter().enumerate() {
            let logit_start = i * vocab_size;
            let logits_slice = &logits.data[logit_start..logit_start + vocab_size];

            let max_logit = logits_slice
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = logits_slice.iter().map(|&x| (x - max_logit).exp()).sum();

            let target = target.min(vocab_size - 1);
            let target_logit = logits_slice[target];
            let log_prob = (target_logit - max_logit) - exp_sum.ln();
            total_loss -= log_prob;
        }

        total_loss / seq_len as f32
    }

    pub fn backward(&self, logits: &Tensor, targets: &[usize], cache: &GPT2Cache) -> GPT2Gradients {
        let seq_len = targets.len();
        let vocab_size = self.config.vocab_size;
        let n_embd = self.config.n_embd;

        // 1. Gradient of loss w.r.t logits
        let mut grad_logits = Vec::new();
        for (i, &target_id) in targets.iter().enumerate().take(seq_len) {
            let logit_start = i * vocab_size;
            let logits_slice = &logits.data[logit_start..logit_start + vocab_size];

            let max_logit = logits_slice
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_vals: Vec<f32> = logits_slice
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum: f32 = exp_vals.iter().sum();
            let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

            for (j, &prob) in probs.iter().enumerate() {
                let target = target_id.min(vocab_size - 1);
                let grad = if j == target { prob - 1.0 } else { prob };
                grad_logits.push(grad / seq_len as f32);
            }
        }
        let grad_logits = Tensor::new(grad_logits, vec![seq_len, vocab_size]);

        // 2. Backprop through output projection
        let grad_output_weight = cache
            .x_before_final_ln
            .transpose(-2, -1)
            .matmul(&grad_logits);
        let mut grad_x = grad_logits.matmul(&self.output_weight.transpose(-2, -1));

        // 3. Backprop through final layer norm
        let ln_final_grads = self.ln_final.backward(&grad_x, &cache.ln_final_cache);
        grad_x = ln_final_grads.x;

        // 4. Backprop through transformer blocks (in reverse order)
        let mut block_grads = Vec::new();
        for (block, cache) in self.blocks.iter().zip(&cache.block_caches).rev() {
            let grads = block.backward(&grad_x, cache);
            grad_x = grads.x.clone();
            block_grads.push(grads);
        }
        block_grads.reverse(); // Put back in forward order

        // 5. Backprop to embeddings
        let mut grad_token_embedding = vec![0.0; vocab_size * n_embd];
        let mut grad_position_embedding = vec![0.0; self.config.block_size * n_embd];

        for (pos, &token_id) in cache.input_ids.iter().enumerate() {
            let token_id = token_id.min(vocab_size - 1);
            let pos = pos.min(self.config.block_size - 1);

            for i in 0..n_embd {
                let grad_val = grad_x.data[pos * n_embd + i];
                grad_token_embedding[token_id * n_embd + i] += grad_val;
                grad_position_embedding[pos * n_embd + i] += grad_val;
            }
        }

        GPT2Gradients {
            token_embedding: Tensor::new(grad_token_embedding, vec![vocab_size, n_embd]),
            position_embedding: Tensor::new(
                grad_position_embedding,
                vec![self.config.block_size, n_embd],
            ),
            block_grads,
            ln_final_gamma: ln_final_grads.gamma,
            ln_final_beta: ln_final_grads.beta,
            output_weight: grad_output_weight,
        }
    }

    /// Generate text
    pub fn generate(&self, prompt: &[usize], max_tokens: usize, temperature: f32) -> Vec<usize> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let (logits, _) = self.forward(&tokens);

            // Get last position logits
            let seq_len = tokens.len();
            let vocab_size = self.config.vocab_size;
            let last_pos_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data[last_pos_start..last_pos_start + vocab_size];

            // Sample with temperature
            let scaled_logits: Vec<f32> = last_logits.iter().map(|&x| x / temperature).collect();
            let max_logit = scaled_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_vals: Vec<f32> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum: f32 = exp_vals.iter().sum();
            let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

            let next_token = sample_from_probs(&probs);
            tokens.push(next_token);

            if tokens.len() >= self.config.block_size {
                break;
            }
        }

        tokens
    }
}

pub struct GPT2Cache {
    input_ids: Vec<usize>,
    block_caches: Vec<BlockCache>,
    ln_final_cache: LayerNormCache,
    x_before_final_ln: Tensor,
}

pub struct GPT2Gradients {
    pub token_embedding: Tensor,
    pub position_embedding: Tensor,
    pub block_grads: Vec<BlockGradients>,
    pub ln_final_gamma: Tensor,
    pub ln_final_beta: Tensor,
    pub output_weight: Tensor,
}

//=============================================================================
// CHECKPOINT - For saving/loading training state
//=============================================================================

#[derive(serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub best_val_loss: f32,
    pub best_val_step: usize,
}

pub struct Checkpoint {
    pub model: TrainableGPT2,
    pub optimizer: Option<AdamWOptimizer>,
    pub tokenizer: Option<BPETokenizer>,
    pub step: usize,
    pub best_val_loss: f32,
    pub best_val_step: usize,
}

impl Checkpoint {
    /// Create a checkpoint for inference only (no optimizer state)
    pub fn inference_only(model: TrainableGPT2) -> Self {
        Self {
            model,
            optimizer: None,
            tokenizer: None,
            step: 0,
            best_val_loss: f32::INFINITY,
            best_val_step: 0,
        }
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        println!("💾 Saving checkpoint to {}...", path);

        let mut file = File::create(path)?;

        // Write header and version
        file.write_all(b"FESTE_CKPT")?;
        file.write_all(&[1u8])?; // Version 1

        // Write model config
        let config_json = serde_json::to_string(&self.model.config)?;
        let config_bytes = config_json.as_bytes();
        file.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        file.write_all(config_bytes)?;

        // Helper to write tensor
        let write_tensor = |file: &mut File, tensor: &Tensor| -> std::io::Result<()> {
            file.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
            for &dim in &tensor.shape {
                file.write_all(&(dim as u32).to_le_bytes())?;
            }
            file.write_all(&(tensor.data.len() as u32).to_le_bytes())?;
            for &val in &tensor.data {
                file.write_all(&val.to_le_bytes())?;
            }
            Ok(())
        };

        // Write model weights
        write_tensor(&mut file, &self.model.token_embedding)?;
        write_tensor(&mut file, &self.model.position_embedding)?;

        file.write_all(&(self.model.blocks.len() as u32).to_le_bytes())?;
        for block in &self.model.blocks {
            write_tensor(&mut file, &block.ln1.gamma)?;
            write_tensor(&mut file, &block.ln1.beta)?;
            write_tensor(&mut file, &block.attn.q_proj.weight)?;
            write_tensor(&mut file, &block.attn.q_proj.bias)?;
            write_tensor(&mut file, &block.attn.k_proj.weight)?;
            write_tensor(&mut file, &block.attn.k_proj.bias)?;
            write_tensor(&mut file, &block.attn.v_proj.weight)?;
            write_tensor(&mut file, &block.attn.v_proj.bias)?;
            write_tensor(&mut file, &block.attn.out_proj.weight)?;
            write_tensor(&mut file, &block.attn.out_proj.bias)?;
            write_tensor(&mut file, &block.ln2.gamma)?;
            write_tensor(&mut file, &block.ln2.beta)?;
            write_tensor(&mut file, &block.mlp.fc1.weight)?;
            write_tensor(&mut file, &block.mlp.fc1.bias)?;
            write_tensor(&mut file, &block.mlp.fc2.weight)?;
            write_tensor(&mut file, &block.mlp.fc2.bias)?;
        }

        write_tensor(&mut file, &self.model.ln_final.gamma)?;
        write_tensor(&mut file, &self.model.ln_final.beta)?;
        write_tensor(&mut file, &self.model.output_weight)?;

        // Write optimizer state flag
        let has_optimizer = self.optimizer.is_some();
        file.write_all(&[has_optimizer as u8])?;

        if let Some(opt) = &self.optimizer {
            // Write optimizer metadata
            file.write_all(&opt.step.to_le_bytes())?;
            file.write_all(&opt.beta1.to_le_bytes())?;
            file.write_all(&opt.beta2.to_le_bytes())?;
            file.write_all(&opt.epsilon.to_le_bytes())?;

            // Write optimizer momentum (m) and variance (v) tensors
            write_tensor(&mut file, &opt.m_token_embedding)?;
            write_tensor(&mut file, &opt.m_position_embedding)?;
            write_tensor(&mut file, &opt.v_token_embedding)?;
            write_tensor(&mut file, &opt.v_position_embedding)?;

            for (m_block, v_block) in opt.m_block_states.iter().zip(&opt.v_block_states) {
                write_tensor(&mut file, &m_block.ln1_gamma)?;
                write_tensor(&mut file, &m_block.ln1_beta)?;
                write_tensor(&mut file, &v_block.ln1_gamma)?;
                write_tensor(&mut file, &v_block.ln1_beta)?;

                write_tensor(&mut file, &m_block.attn.q_weight)?;
                write_tensor(&mut file, &m_block.attn.q_bias)?;
                write_tensor(&mut file, &m_block.attn.k_weight)?;
                write_tensor(&mut file, &m_block.attn.k_bias)?;
                write_tensor(&mut file, &m_block.attn.v_weight)?;
                write_tensor(&mut file, &m_block.attn.v_bias)?;
                write_tensor(&mut file, &m_block.attn.out_weight)?;
                write_tensor(&mut file, &m_block.attn.out_bias)?;

                write_tensor(&mut file, &v_block.attn.q_weight)?;
                write_tensor(&mut file, &v_block.attn.q_bias)?;
                write_tensor(&mut file, &v_block.attn.k_weight)?;
                write_tensor(&mut file, &v_block.attn.k_bias)?;
                write_tensor(&mut file, &v_block.attn.v_weight)?;
                write_tensor(&mut file, &v_block.attn.v_bias)?;
                write_tensor(&mut file, &v_block.attn.out_weight)?;
                write_tensor(&mut file, &v_block.attn.out_bias)?;

                write_tensor(&mut file, &m_block.ln2_gamma)?;
                write_tensor(&mut file, &m_block.ln2_beta)?;
                write_tensor(&mut file, &v_block.ln2_gamma)?;
                write_tensor(&mut file, &v_block.ln2_beta)?;

                write_tensor(&mut file, &m_block.mlp.fc1_weight)?;
                write_tensor(&mut file, &m_block.mlp.fc1_bias)?;
                write_tensor(&mut file, &m_block.mlp.fc2_weight)?;
                write_tensor(&mut file, &m_block.mlp.fc2_bias)?;

                write_tensor(&mut file, &v_block.mlp.fc1_weight)?;
                write_tensor(&mut file, &v_block.mlp.fc1_bias)?;
                write_tensor(&mut file, &v_block.mlp.fc2_weight)?;
                write_tensor(&mut file, &v_block.mlp.fc2_bias)?;
            }

            write_tensor(&mut file, &opt.m_ln_final_gamma)?;
            write_tensor(&mut file, &opt.m_ln_final_beta)?;
            write_tensor(&mut file, &opt.m_output_weight)?;
            write_tensor(&mut file, &opt.v_ln_final_gamma)?;
            write_tensor(&mut file, &opt.v_ln_final_beta)?;
            write_tensor(&mut file, &opt.v_output_weight)?;
        }

        // Write tokenizer state flag
        let has_tokenizer = self.tokenizer.is_some();
        file.write_all(&[has_tokenizer as u8])?;

        if let Some(tokenizer) = &self.tokenizer {
            // Serialize tokenizer to JSON
            let tokenizer_json = serde_json::to_string(tokenizer)?;
            let tokenizer_bytes = tokenizer_json.as_bytes();
            file.write_all(&(tokenizer_bytes.len() as u32).to_le_bytes())?;
            file.write_all(tokenizer_bytes)?;
        }

        // Write checkpoint metadata
        let metadata = CheckpointMetadata {
            step: self.step,
            best_val_loss: self.best_val_loss,
            best_val_step: self.best_val_step,
        };
        let metadata_json = serde_json::to_string(&metadata)?;
        let metadata_bytes = metadata_json.as_bytes();
        file.write_all(&(metadata_bytes.len() as u32).to_le_bytes())?;
        file.write_all(metadata_bytes)?;

        let file_size = file.metadata()?.len() as f64 / 1_000_000.0;
        println!("✅ Checkpoint saved successfully!");
        println!("   File size: {:.2} MB", file_size);
        let mut includes = vec!["Model weights"];
        if self.optimizer.is_some() {
            includes.push("Optimizer state");
        }
        if self.tokenizer.is_some() {
            includes.push("Tokenizer");
        }
        includes.push("Training metadata");
        println!("   Includes: {}", includes.join(" + "));

        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::Read;

        println!("📂 Loading checkpoint from {}...", path);

        let mut file = File::open(path)?;

        // Read and verify header
        let mut header = [0u8; 10];
        file.read_exact(&mut header)?;
        if &header != b"FESTE_CKPT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid checkpoint header - expected FESTE_CKPT",
            ));
        }

        // Read version
        let mut version = [0u8; 1];
        file.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported checkpoint version: {}", version[0]),
            ));
        }

        // Read config
        let mut config_len_bytes = [0u8; 4];
        file.read_exact(&mut config_len_bytes)?;
        let config_len = u32::from_le_bytes(config_len_bytes) as usize;

        let mut config_bytes = vec![0u8; config_len];
        file.read_exact(&mut config_bytes)?;
        let config_json = String::from_utf8(config_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Helper to read tensor
        let read_tensor = |file: &mut File| -> std::io::Result<Tensor> {
            let mut shape_len_bytes = [0u8; 4];
            file.read_exact(&mut shape_len_bytes)?;
            let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 4];
                file.read_exact(&mut dim_bytes)?;
                shape.push(u32::from_le_bytes(dim_bytes) as usize);
            }

            let mut data_len_bytes = [0u8; 4];
            file.read_exact(&mut data_len_bytes)?;
            let data_len = u32::from_le_bytes(data_len_bytes) as usize;

            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut val_bytes = [0u8; 4];
                file.read_exact(&mut val_bytes)?;
                data.push(f32::from_le_bytes(val_bytes));
            }

            Ok(Tensor::new(data, shape))
        };

        // Read model weights
        let token_embedding = read_tensor(&mut file)?;
        let position_embedding = read_tensor(&mut file)?;

        let mut num_blocks_bytes = [0u8; 4];
        file.read_exact(&mut num_blocks_bytes)?;
        let num_blocks = u32::from_le_bytes(num_blocks_bytes) as usize;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let ln1_gamma = read_tensor(&mut file)?;
            let ln1_beta = read_tensor(&mut file)?;
            let ln1 = TrainableLayerNorm {
                gamma: ln1_gamma,
                beta: ln1_beta,
                eps: 1e-5,
            };

            let q_weight = read_tensor(&mut file)?;
            let q_bias = read_tensor(&mut file)?;
            let k_weight = read_tensor(&mut file)?;
            let k_bias = read_tensor(&mut file)?;
            let v_weight = read_tensor(&mut file)?;
            let v_bias = read_tensor(&mut file)?;
            let out_weight = read_tensor(&mut file)?;
            let out_bias = read_tensor(&mut file)?;

            let attn = TrainableSingleHeadAttention {
                q_proj: TrainableLinear {
                    weight: q_weight,
                    bias: q_bias,
                },
                k_proj: TrainableLinear {
                    weight: k_weight,
                    bias: k_bias,
                },
                v_proj: TrainableLinear {
                    weight: v_weight,
                    bias: v_bias,
                },
                out_proj: TrainableLinear {
                    weight: out_weight,
                    bias: out_bias,
                },
                attn_dropout: TrainableDropout::new(config.dropout_rate),
                resid_dropout: TrainableDropout::new(config.dropout_rate),
                n_embd: config.n_embd,
            };

            let ln2_gamma = read_tensor(&mut file)?;
            let ln2_beta = read_tensor(&mut file)?;
            let ln2 = TrainableLayerNorm {
                gamma: ln2_gamma,
                beta: ln2_beta,
                eps: 1e-5,
            };

            let fc1_weight = read_tensor(&mut file)?;
            let fc1_bias = read_tensor(&mut file)?;
            let fc2_weight = read_tensor(&mut file)?;
            let fc2_bias = read_tensor(&mut file)?;

            let mlp = TrainableMLP {
                fc1: TrainableLinear {
                    weight: fc1_weight,
                    bias: fc1_bias,
                },
                fc2: TrainableLinear {
                    weight: fc2_weight,
                    bias: fc2_bias,
                },
                resid_dropout: TrainableDropout::new(config.dropout_rate),
            };

            blocks.push(TrainableTransformerBlock {
                ln1,
                attn,
                ln2,
                mlp,
            });
        }

        let ln_final_gamma = read_tensor(&mut file)?;
        let ln_final_beta = read_tensor(&mut file)?;
        let ln_final = TrainableLayerNorm {
            gamma: ln_final_gamma,
            beta: ln_final_beta,
            eps: 1e-5,
        };

        let output_weight = read_tensor(&mut file)?;

        let model = TrainableGPT2 {
            token_embedding,
            position_embedding,
            blocks,
            ln_final,
            output_weight,
            config,
        };

        // Read optimizer state flag
        let mut has_optimizer = [0u8; 1];
        file.read_exact(&mut has_optimizer)?;

        let optimizer = if has_optimizer[0] == 1 {
            // Read optimizer metadata
            let mut step_bytes = [0u8; 8];
            let mut beta1_bytes = [0u8; 4];
            let mut beta2_bytes = [0u8; 4];
            let mut epsilon_bytes = [0u8; 4];

            file.read_exact(&mut step_bytes)?;
            file.read_exact(&mut beta1_bytes)?;
            file.read_exact(&mut beta2_bytes)?;
            file.read_exact(&mut epsilon_bytes)?;

            let step = usize::from_le_bytes(step_bytes);
            let beta1 = f32::from_le_bytes(beta1_bytes);
            let beta2 = f32::from_le_bytes(beta2_bytes);
            let epsilon = f32::from_le_bytes(epsilon_bytes);

            // Read optimizer tensors
            let m_token_embedding = read_tensor(&mut file)?;
            let m_position_embedding = read_tensor(&mut file)?;
            let v_token_embedding = read_tensor(&mut file)?;
            let v_position_embedding = read_tensor(&mut file)?;

            let mut m_block_states = Vec::new();
            let mut v_block_states = Vec::new();

            for _ in 0..model.blocks.len() {
                let m_ln1_gamma = read_tensor(&mut file)?;
                let m_ln1_beta = read_tensor(&mut file)?;
                let v_ln1_gamma = read_tensor(&mut file)?;
                let v_ln1_beta = read_tensor(&mut file)?;

                let m_q_weight = read_tensor(&mut file)?;
                let m_q_bias = read_tensor(&mut file)?;
                let m_k_weight = read_tensor(&mut file)?;
                let m_k_bias = read_tensor(&mut file)?;
                let m_v_weight = read_tensor(&mut file)?;
                let m_v_bias = read_tensor(&mut file)?;
                let m_out_weight = read_tensor(&mut file)?;
                let m_out_bias = read_tensor(&mut file)?;

                let v_q_weight = read_tensor(&mut file)?;
                let v_q_bias = read_tensor(&mut file)?;
                let v_k_weight = read_tensor(&mut file)?;
                let v_k_bias = read_tensor(&mut file)?;
                let v_v_weight = read_tensor(&mut file)?;
                let v_v_bias = read_tensor(&mut file)?;
                let v_out_weight = read_tensor(&mut file)?;
                let v_out_bias = read_tensor(&mut file)?;

                let m_ln2_gamma = read_tensor(&mut file)?;
                let m_ln2_beta = read_tensor(&mut file)?;
                let v_ln2_gamma = read_tensor(&mut file)?;
                let v_ln2_beta = read_tensor(&mut file)?;

                let m_fc1_weight = read_tensor(&mut file)?;
                let m_fc1_bias = read_tensor(&mut file)?;
                let m_fc2_weight = read_tensor(&mut file)?;
                let m_fc2_bias = read_tensor(&mut file)?;

                let v_fc1_weight = read_tensor(&mut file)?;
                let v_fc1_bias = read_tensor(&mut file)?;
                let v_fc2_weight = read_tensor(&mut file)?;
                let v_fc2_bias = read_tensor(&mut file)?;

                m_block_states.push(BlockAdamState {
                    ln1_gamma: m_ln1_gamma,
                    ln1_beta: m_ln1_beta,
                    attn: AttentionAdamState {
                        q_weight: m_q_weight,
                        q_bias: m_q_bias,
                        k_weight: m_k_weight,
                        k_bias: m_k_bias,
                        v_weight: m_v_weight,
                        v_bias: m_v_bias,
                        out_weight: m_out_weight,
                        out_bias: m_out_bias,
                    },
                    ln2_gamma: m_ln2_gamma,
                    ln2_beta: m_ln2_beta,
                    mlp: MLPAdamState {
                        fc1_weight: m_fc1_weight,
                        fc1_bias: m_fc1_bias,
                        fc2_weight: m_fc2_weight,
                        fc2_bias: m_fc2_bias,
                    },
                });

                v_block_states.push(BlockAdamState {
                    ln1_gamma: v_ln1_gamma,
                    ln1_beta: v_ln1_beta,
                    attn: AttentionAdamState {
                        q_weight: v_q_weight,
                        q_bias: v_q_bias,
                        k_weight: v_k_weight,
                        k_bias: v_k_bias,
                        v_weight: v_v_weight,
                        v_bias: v_v_bias,
                        out_weight: v_out_weight,
                        out_bias: v_out_bias,
                    },
                    ln2_gamma: v_ln2_gamma,
                    ln2_beta: v_ln2_beta,
                    mlp: MLPAdamState {
                        fc1_weight: v_fc1_weight,
                        fc1_bias: v_fc1_bias,
                        fc2_weight: v_fc2_weight,
                        fc2_bias: v_fc2_bias,
                    },
                });
            }

            let m_ln_final_gamma = read_tensor(&mut file)?;
            let m_ln_final_beta = read_tensor(&mut file)?;
            let m_output_weight = read_tensor(&mut file)?;
            let v_ln_final_gamma = read_tensor(&mut file)?;
            let v_ln_final_beta = read_tensor(&mut file)?;
            let v_output_weight = read_tensor(&mut file)?;

            Some(AdamWOptimizer {
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
                beta1,
                beta2,
                epsilon,
                step,
            })
        } else {
            None
        };

        // Read tokenizer state flag
        let mut has_tokenizer = [0u8; 1];
        file.read_exact(&mut has_tokenizer)?;

        let tokenizer = if has_tokenizer[0] == 1 {
            // Read tokenizer JSON
            let mut tokenizer_len_bytes = [0u8; 4];
            file.read_exact(&mut tokenizer_len_bytes)?;
            let tokenizer_len = u32::from_le_bytes(tokenizer_len_bytes) as usize;

            let mut tokenizer_bytes = vec![0u8; tokenizer_len];
            file.read_exact(&mut tokenizer_bytes)?;
            let tokenizer_json = String::from_utf8(tokenizer_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let tokenizer: BPETokenizer = serde_json::from_str(&tokenizer_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            Some(tokenizer)
        } else {
            None
        };

        // Read checkpoint metadata
        let mut metadata_len_bytes = [0u8; 4];
        file.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u32::from_le_bytes(metadata_len_bytes) as usize;

        let mut metadata_bytes = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_bytes)?;
        let metadata_json = String::from_utf8(metadata_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("✅ Checkpoint loaded successfully!");
        if optimizer.is_some() {
            println!(
                "   Training state: step {}, best val loss: {:.4}",
                metadata.step, metadata.best_val_loss
            );
        } else {
            println!("   Inference-only model (no optimizer state)");
        }
        if tokenizer.is_some() {
            println!(
                "   Tokenizer included (vocab size: {})",
                tokenizer.as_ref().unwrap().vocab_size()
            );
        }

        Ok(Self {
            model,
            optimizer,
            tokenizer,
            step: metadata.step,
            best_val_loss: metadata.best_val_loss,
            best_val_step: metadata.best_val_step,
        })
    }

    /// Save checkpoint in background thread (non-blocking)
    /// Returns a JoinHandle that can be used to wait for completion
    pub fn save_background(self, path: String) -> std::thread::JoinHandle<std::io::Result<()>> {
        std::thread::spawn(move || self.save(&path))
    }
}

/// Initialize zero gradients matching model structure
fn sample_from_probs(probs: &[f32]) -> usize {
    use std::cell::Cell;
    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(12345) };
    }

    RNG.with(|rng| {
        let mut state = rng.get();
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        rng.set(state);

        let rand_val = ((state / 65536) % 32768) as f32 / 32768.0;

        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return i;
            }
        }
        probs.len() - 1
    })
}

