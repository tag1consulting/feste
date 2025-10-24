//! Neural Network Layers
//!
//! This module contains all the layer implementations for the trainable GPT-2 model.
//! Each layer provides both forward and backward passes for training.
//!
//! ## Layers
//!
//! - **activation**: GELU activation function (forward and backward)
//! - **linear**: Fully connected layer
//! - **layer_norm**: Layer normalization
//! - **dropout**: Dropout regularization
//! - **mlp**: Multi-layer perceptron (feedforward network)
//! - **attention**: Self-attention mechanism
//! - **block**: Complete transformer block
//!
//! ## Design Pattern
//!
//! Each trainable layer follows a consistent pattern:
//!
//! ```rust,ignore
//! pub struct TrainableLayer {
//!     // Parameters (weights, biases, etc.)
//! }
//!
//! impl TrainableLayer {
//!     pub fn new(...) -> Self { }
//!     pub fn forward(&self, x: &Tensor) -> (Tensor, Cache) { }
//!     pub fn backward(&self, grad: &Tensor, cache: &Cache) -> Gradients { }
//! }
//!
//! pub struct Cache {
//!     // Values needed for backward pass
//! }
//!
//! pub struct Gradients {
//!     // Gradients for parameters and input
//! }
//! ```
//!
//! This pattern makes backpropagation explicit and educational.

pub mod activation;
pub mod attention;
pub mod block;
pub mod dropout;
pub mod layer_norm;
pub mod linear;
pub mod mlp;

// Re-export main types for convenience
pub use activation::{gelu_backward, gelu_forward};
pub use attention::{AttentionCache, AttentionGradients, TrainableSingleHeadAttention};
pub use block::{BlockCache, BlockGradients, TrainableTransformerBlock};
pub use dropout::{DropoutCache, TrainableDropout};
pub use layer_norm::{LayerNormCache, LayerNormGradients, TrainableLayerNorm};
pub use linear::{random_init, LinearCache, LinearGradients, TrainableLinear};
pub use mlp::{MLPCache, MLPGradients, TrainableMLP};
