//! Feste: Educational Language Model Implementation
//!
//! A complete GPT-2 style transformer implemented from scratch in Rust
//! for educational purposes. Named after Shakespeare's witty fool from
//! *Twelfth Night*.
//!
//! # Modules
//!
//! - [`tokenizer`] - Byte Pair Encoding (BPE) tokenization
//! - [`tensor`] - Multi-dimensional arrays and operations
//! - [`model`] - GPT-2 model architecture (forward pass only)
//! - [`gpt2_trainable`] - Trainable GPT-2 with backward pass
//! - [`train`] - Data loading for training
//! - [`training_logger`] - Training metrics and logging
//! - [`instruction_tuning`] - Instruction tuning for chatbots
//!
//! # Example: Tokenization
//!
//! ```rust,no_run
//! use feste::BPETokenizer;
//!
//! // Train a tokenizer
//! let text = std::fs::read_to_string("corpus.txt").unwrap();
//! let mut tokenizer = BPETokenizer::new(1024);
//! tokenizer.train(&text, 1024);
//!
//! // Encode and decode
//! let ids = tokenizer.encode("Hello, world!");
//! let decoded = tokenizer.decode(&ids);
//! assert_eq!(decoded, "Hello, world!");
//! ```
//!
//! # Example: Tensor Operations
//!
//! ```rust
//! use feste::Tensor;
//!
//! // Create a 2x2 matrix
//! let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
//!
//! // Matrix multiplication
//! let c = a.matmul(&b);
//! assert_eq!(c.shape, vec![2, 2]);
//! ```
//!
//! # Example: Model Architecture
//!
//! ```rust
//! use feste::{GPT2, Config};
//!
//! // Create a tiny model
//! let config = Config::tiny(512); // 512 vocab size
//! let model = GPT2::new(&config);
//!
//! // Forward pass: tokens → logits
//! let tokens = vec![vec![1, 2, 3, 4]]; // batch_size=1, seq_len=4
//! let logits = model.forward(&tokens);
//! assert_eq!(logits.shape, vec![1, 4, 512]); // [batch, seq, vocab]
//! ```

pub mod gpt2_trainable;
pub mod gradients;
pub mod instruction_tuning;
pub mod layers;
pub mod model;
pub mod optimizer;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod training_logger;

// Re-export main types for convenience
pub use gradients::{clip_gradients, compute_grad_norm};
pub use model::{gelu, Config, GPT2};
pub use optimizer::{adamw_update, AdamWOptimizer};
pub use tensor::Tensor;
pub use tokenizer::{BPETokenizer, TokenizerStats};
pub use train::{Batch, TextDataLoader, TrainingConfig};
pub use training_logger::{compute_dataset_loss, train_val_split, TrainingLogger};
