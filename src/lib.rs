//! Feste: Educational Language Model Implementation
//!
//! A complete GPT-2 style transformer implemented from scratch in Rust
//! for educational purposes. Named after Shakespeare's witty fool from
//! *Twelfth Night*.
//!
//! # Modules
//!
//! - [`tokenizer`] - Byte Pair Encoding (BPE) tokenization
//!
//! # Example
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

pub mod tokenizer;

// Re-export main types for convenience
pub use tokenizer::{BPETokenizer, TokenizerStats};
