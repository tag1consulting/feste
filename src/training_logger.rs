//! Training Logger and Utilities
//!
//! This module provides utilities for tracking and logging training metrics.
//! It includes a CSV logger for detailed tracking and helper functions for
//! computing losses and splitting datasets.
//!
//! ## Components
//!
//! - **TrainingLogger**: Logs metrics to CSV and console with timestamps
//! - **train_val_split**: Splits tokenized data into training and validation sets
//! - **compute_dataset_loss**: Computes average loss over a dataset
//!
//! ## Example
//!
//! ```rust,no_run
//! use feste::TrainingLogger;
//!
//! let mut logger = TrainingLogger::new("training_log.csv")
//!     .expect("Failed to create logger");
//!
//! // Log training step
//! logger.log(100, 0.001, 2.5, 2.8, Some("To be, or not to be"))
//!     .expect("Failed to log");
//! ```
//!
//! ## CSV Format
//!
//! The logger writes CSV files with the following columns:
//! - `step`: Training step number
//! - `elapsed_seconds`: Time since training started
//! - `learning_rate`: Current learning rate
//! - `train_loss`: Training loss (cross-entropy)
//! - `val_loss`: Validation loss
//! - `train_perplexity`: exp(train_loss) - interpretable metric
//! - `val_perplexity`: exp(val_loss) - lower is better
//! - `sample`: Generated text sample
//!
//! ## Perplexity
//!
//! Perplexity measures how "surprised" the model is by the data:
//! ```text
//! perplexity = exp(loss)
//! ```
//!
//! - **Perfect model**: perplexity = 1.0 (loss = 0)
//! - **Random guessing** (vocab=512): perplexity ≈ 512 (loss ≈ 6.2)
//! - **Good model**: perplexity = 10-50 (loss = 2.3-3.9)
//!
//! Lower perplexity means the model makes better predictions.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Training logger for tracking metrics over time
///
/// Logs training metrics to both CSV file and console. The CSV file can be
/// analyzed later for visualization and model comparison.
///
/// # Fields
///
/// - `log_file`: Output CSV file
/// - `start_time`: When training started (for elapsed time calculation)
/// - `last_log_time`: Last log timestamp (for step timing)
pub struct TrainingLogger {
    log_file: File,
    start_time: Instant,
    last_log_time: Instant,
}

impl TrainingLogger {
    /// Create a new training logger
    ///
    /// Creates a CSV file with headers and initializes timing.
    ///
    /// # Arguments
    ///
    /// * `log_path` - Path to CSV file to create
    ///
    /// # Returns
    ///
    /// Result containing the logger or an IO error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::TrainingLogger;
    /// let logger = TrainingLogger::new("training_log.csv")?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn new(log_path: &str) -> std::io::Result<Self> {
        let mut log_file = File::create(log_path)?;

        // Write CSV header
        writeln!(
            log_file,
            "step,elapsed_seconds,learning_rate,train_loss,val_loss,train_perplexity,val_perplexity,sample"
        )?;

        let now = Instant::now();
        Ok(Self {
            log_file,
            start_time: now,
            last_log_time: now,
        })
    }

    /// Log a training step
    ///
    /// Writes metrics to CSV and prints to console with timing information.
    ///
    /// # Arguments
    ///
    /// * `step` - Training step number
    /// * `learning_rate` - Current learning rate
    /// * `train_loss` - Training loss
    /// * `val_loss` - Validation loss
    /// * `sample` - Optional generated text sample
    ///
    /// # Returns
    ///
    /// Result indicating success or IO error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::TrainingLogger;
    /// # let mut logger = TrainingLogger::new("log.csv")?;
    /// logger.log(100, 0.001, 2.5, 2.8, Some("Hello world"))?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn log(
        &mut self,
        step: usize,
        learning_rate: f32,
        train_loss: f32,
        val_loss: f32,
        sample: Option<&str>,
    ) -> std::io::Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Perplexity = exp(loss)
        // This is a more interpretable metric than raw loss
        let train_perplexity = train_loss.exp();
        let val_perplexity = val_loss.exp();

        // Escape quotes in sample text for CSV format
        let sample_escaped = sample.map(|s| s.replace('"', "\"\"")).unwrap_or_default();

        // Write to CSV file
        writeln!(
            self.log_file,
            "{},{:.2},{:.6},{:.4},{:.4},{:.2},{:.2},\"{}\"",
            step,
            elapsed,
            learning_rate,
            train_loss,
            val_loss,
            train_perplexity,
            val_perplexity,
            sample_escaped
        )?;

        // Flush to ensure data is written immediately
        // This is important if training crashes - we don't lose data
        self.log_file.flush()?;

        // Print to console with timing info
        let step_time = self.last_log_time.elapsed().as_secs_f32();
        println!(
            "Step {:4} | Time: {:7.1}s (+{:.1}s) | LR: {:.6} | Train: {:.4} | Val: {:.4} | Perplexity: {:.2}",
            step, elapsed, step_time, learning_rate, train_loss, val_loss, val_perplexity
        );

        if let Some(text) = sample {
            println!("  Sample: \"{}\"", text);
        }

        self.last_log_time = Instant::now();
        Ok(())
    }
}

/// Split tokenized data into training and validation sets
///
/// Performs a simple split at a fixed fraction. The validation set is taken
/// from the end of the data to ensure temporal separation in sequential data.
///
/// # Arguments
///
/// * `tokens` - All tokenized data
/// * `val_fraction` - Fraction to use for validation (e.g., 0.1 for 10%)
///
/// # Returns
///
/// Tuple of (training_tokens, validation_tokens)
///
/// # Example
///
/// ```rust
/// # use feste::train_val_split;
/// let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let (train, val) = train_val_split(&tokens, 0.2);
/// assert_eq!(train.len(), 8);  // 80% for training
/// assert_eq!(val.len(), 2);    // 20% for validation
/// ```
pub fn train_val_split(tokens: &[usize], val_fraction: f32) -> (&[usize], &[usize]) {
    let split_idx = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
    (&tokens[..split_idx], &tokens[split_idx..])
}

/// Compute average loss over a dataset
///
/// Evaluates the model on multiple batches from the dataset and returns
/// the average loss. This is used to compute validation loss during training.
///
/// # Arguments
///
/// * `tokens` - Tokenized dataset
/// * `seq_len` - Sequence length per example
/// * `num_batches` - Number of batches to evaluate (limited by dataset size)
/// * `compute_loss_fn` - Function that computes loss for a single batch
///
/// # Returns
///
/// Average loss across all batches
///
/// # Example
///
/// ```rust,no_run
/// # use feste::compute_dataset_loss;
/// let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let avg_loss = compute_dataset_loss(
///     &tokens,
///     4,    // seq_len
///     2,    // num_batches
///     |input, target| {
///         // Compute loss for this batch
///         0.5  // placeholder
///     }
/// );
/// ```
pub fn compute_dataset_loss<F>(
    tokens: &[usize],
    seq_len: usize,
    num_batches: usize,
    mut compute_loss_fn: F,
) -> f32
where
    F: FnMut(&[usize], &[usize]) -> f32,
{
    // Need at least seq_len + 1 tokens (input + target)
    if tokens.len() < seq_len + 1 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    // Limit num_batches to what's actually available in the dataset
    let max_batches = (tokens.len() - seq_len - 1) / seq_len;
    let num_batches = num_batches.min(max_batches);

    for batch_idx in 0..num_batches {
        // Extract input and target sequences
        // Target is shifted by 1 position (next token prediction)
        let start = (batch_idx * seq_len) % (tokens.len() - seq_len - 1);
        let input_seq = &tokens[start..start + seq_len];
        let target_seq = &tokens[start + 1..start + seq_len + 1];

        let loss = compute_loss_fn(input_seq, target_seq);
        total_loss += loss;
    }

    total_loss / num_batches as f32
}
