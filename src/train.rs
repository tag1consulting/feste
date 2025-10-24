//! Training Data Loading
//!
//! This module provides a simple data loader for training language models on text.
//! It handles tokenization, batching, and sequence generation with a sliding window
//! approach.
//!
//! ## How Sequences Are Generated
//!
//! The data loader uses a sliding window to create training examples:
//!
//! ```text
//! Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
//! Seq length: 4
//! Batch size: 2
//!
//! Batch 1:
//!   Input:  [1, 2, 3, 4]  Target: [2, 3, 4, 5]
//!   Input:  [5, 6, 7, 8]  Target: [6, 7, 8, 9]
//!
//! Batch 2:
//!   Input:  [9, 10, 11, 12]  Target: [10, 11, 12, 13]  (if 13 exists)
//! ```
//!
//! The target is always the input shifted by one position, teaching the model
//! to predict the next token.
//!
//! ## Example
//!
//! ```rust,no_run
//! # use feste::{BPETokenizer, TextDataLoader};
//! # let tokenizer = BPETokenizer::new(512);
//! let text = std::fs::read_to_string("shakespeare.txt")?;
//!
//! let mut loader = TextDataLoader::new(
//!     &text,
//!     &tokenizer,
//!     128,  // sequence length
//!     4     // batch size
//! );
//!
//! while let Some((inputs, targets)) = loader.next_batch() {
//!     // Train on this batch
//!     // inputs: Vec<Vec<usize>> with shape [batch_size, seq_len]
//!     // targets: Vec<Vec<usize>> with shape [batch_size, seq_len]
//! }
//! # Ok::<(), std::io::Error>(())
//! ```

use crate::tokenizer::BPETokenizer;
use std::fs;

/// Type alias for a batch of input/target sequences
/// Each element is a Vec<Vec<usize>> with shape [batch_size][seq_len]
pub type Batch = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Data loader for text datasets
///
/// Loads text, tokenizes it, and provides batches of (input, target) sequence pairs
/// for training language models.
///
/// # Fields
///
/// - `tokens`: All tokenized data
/// - `seq_len`: Length of each training sequence
/// - `batch_size`: Number of sequences per batch
/// - `position`: Current position in the dataset
pub struct TextDataLoader {
    tokens: Vec<usize>,
    seq_len: usize,
    batch_size: usize,
    position: usize,
}

impl TextDataLoader {
    /// Create a data loader from text
    ///
    /// Tokenizes the text immediately and stores all tokens in memory.
    ///
    /// # Arguments
    ///
    /// * `text` - Raw text to train on
    /// * `tokenizer` - Trained tokenizer for encoding
    /// * `seq_len` - Length of each training sequence
    /// * `batch_size` - Number of sequences per batch
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::{BPETokenizer, TextDataLoader};
    /// # let tokenizer = BPETokenizer::new(512);
    /// let text = "To be, or not to be, that is the question.";
    /// let loader = TextDataLoader::new(&text, &tokenizer, 64, 4);
    /// ```
    pub fn new(text: &str, tokenizer: &BPETokenizer, seq_len: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        println!("Loaded {} tokens from text", tokens.len());

        Self {
            tokens,
            seq_len,
            batch_size,
            position: 0,
        }
    }

    /// Create a data loader from a file
    ///
    /// Convenience method that reads the file and creates the loader.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to text file
    /// * `tokenizer` - Trained tokenizer
    /// * `seq_len` - Sequence length
    /// * `batch_size` - Batch size
    ///
    /// # Returns
    ///
    /// Result containing the loader or an IO error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::{BPETokenizer, TextDataLoader};
    /// # let tokenizer = BPETokenizer::new(512);
    /// let loader = TextDataLoader::from_file(
    ///     "shakespeare.txt",
    ///     &tokenizer,
    ///     128,
    ///     4
    /// )?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn from_file(
        path: &str,
        tokenizer: &BPETokenizer,
        seq_len: usize,
        batch_size: usize,
    ) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(Self::new(&text, tokenizer, seq_len, batch_size))
    }

    /// Get the next batch of training data
    ///
    /// Returns a batch of (input, target) sequence pairs. The target is always
    /// the input shifted by one position (next token prediction).
    ///
    /// When the end of the dataset is reached, returns `None` and resets to
    /// the beginning for the next epoch.
    ///
    /// # Returns
    ///
    /// - `Some((inputs, targets))` if a batch is available
    /// - `None` if the epoch is complete (resets position)
    ///
    /// # Shape
    ///
    /// Both inputs and targets have shape `[batch_size, seq_len]`
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::{BPETokenizer, TextDataLoader};
    /// # let tokenizer = BPETokenizer::new(512);
    /// # let mut loader = TextDataLoader::new("text", &tokenizer, 64, 4);
    /// while let Some((inputs, targets)) = loader.next_batch() {
    ///     assert_eq!(inputs.len(), 4);     // batch_size
    ///     assert_eq!(inputs[0].len(), 64); // seq_len
    ///     // Train on this batch...
    /// }
    /// ```
    pub fn next_batch(&mut self) -> Option<Batch> {
        // Check if we have enough tokens left for a full batch
        if self.position + self.batch_size * (self.seq_len + 1) >= self.tokens.len() {
            // Reset to beginning (epoch complete)
            self.position = 0;
            return None;
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        // Build batch by extracting sequences
        for _ in 0..self.batch_size {
            // Ensure we have enough tokens for this sequence
            if self.position + self.seq_len + 1 >= self.tokens.len() {
                break;
            }

            // Extract input sequence: tokens[pos..pos+seq_len]
            let input_seq = self.tokens[self.position..self.position + self.seq_len].to_vec();

            // Extract target sequence: tokens[pos+1..pos+seq_len+1]
            // This is the input shifted by 1 (next token prediction)
            let target_seq =
                self.tokens[self.position + 1..self.position + self.seq_len + 1].to_vec();

            inputs.push(input_seq);
            targets.push(target_seq);

            // Move forward by seq_len (non-overlapping sequences)
            self.position += self.seq_len;
        }

        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }

    /// Reset the data loader to the beginning
    ///
    /// Useful for starting a new epoch without waiting for `next_batch()`
    /// to reach the end.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get the total number of batches in one epoch
    ///
    /// This is an estimate based on the dataset size and batch parameters.
    ///
    /// # Returns
    ///
    /// Number of batches per epoch
    pub fn num_batches(&self) -> usize {
        self.tokens.len() / (self.batch_size * self.seq_len)
    }
}

/// Training configuration
///
/// Hyperparameters for training a language model.
///
/// # Common Configurations
///
/// - **Tiny**: Fast experimentation (minutes)
/// - **Small**: Medium training runs (hours)
/// - **Large**: Full training (overnight)
pub struct TrainingConfig {
    /// Learning rate for optimizer
    pub learning_rate: f32,
    /// Number of passes through the dataset
    pub num_epochs: usize,
    /// Number of sequences per batch
    pub batch_size: usize,
    /// Length of each training sequence
    pub seq_len: usize,
    /// Print metrics every N steps
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 1,
            batch_size: 4,
            seq_len: 64,
            print_every: 100,
        }
    }
}

impl TrainingConfig {
    /// Create a tiny configuration for quick experiments
    ///
    /// Good for:
    /// - Testing code changes
    /// - Quick iterations
    /// - Low-resource environments
    ///
    /// # Returns
    ///
    /// TrainingConfig with small batch size and short sequences
    pub fn tiny() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 3,
            batch_size: 8,
            seq_len: 64,
            print_every: 50,
        }
    }

    /// Create a small configuration for medium experiments
    ///
    /// Good for:
    /// - Prototyping model changes
    /// - Overnight training runs
    /// - Balancing speed and quality
    ///
    /// # Returns
    ///
    /// TrainingConfig with moderate settings
    pub fn small() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 5,
            batch_size: 16,
            seq_len: 128,
            print_every: 100,
        }
    }
}
