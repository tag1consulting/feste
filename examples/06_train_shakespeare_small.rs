//! Train a Small GPT-2 Model on Shakespeare
//!
//! This example demonstrates training a small transformer model:
//! - **Model size**: ~200K parameters (n_embd=128, n_layers=3, n_heads=1)
//! - **Training time**: Approximately 10-20 minutes on modern CPU
//!
//! A good balance between training speed and model capacity for experimentation.
//!
//! ## What You'll Learn
//!
//! - How model size affects learning quality
//! - The relationship between training time and perplexity
//! - Diminishing returns of training (loss plateaus eventually)
//! - How tokenization quality affects generation
//!
//! ## Output
//!
//! All outputs are saved to: `data/shakespeare_small_<timestamp>/`
//! - `training_log.csv` - Step-by-step training metrics
//! - `checkpoint_best.bin` - Model with best validation loss
//! - `checkpoint_step_*.bin` - Periodic checkpoints every 250 steps
//! - `checkpoint_final.bin` - Final model after all steps
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example 06_train_shakespeare_small
//! ```
//!
//! ## Prerequisites
//!
//! Download Shakespeare corpus first:
//! ```bash
//! curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
//! ```
//!
//! ## Default Configuration
//!
//! Model architecture:
//! - Vocabulary size: 1024 tokens
//! - Context length: 128 tokens
//! - Embedding dimension: 128
//! - Number of layers: 3
//! - Number of heads: 1
//!
//! Training hyperparameters:
//! - Training steps: 2000
//! - Learning rate: 0.002
//! - Warmup: 10% of training steps
//! - Gradient clipping: 1.0
//! - Early stopping patience: 5000 steps
//! - Sequence length: 128 tokens (same as block_size)
//! - Gradient accumulation: 8 mini-batches
//!
//! Data configuration:
//! - Tokenizer training data: First 500K characters
//! - Model training data: First 500K characters
//! - Validation split: 10% held out for validation
//!
//! These are starting values for exploration. Try adjusting them to see
//! how they affect training dynamics and final model quality.

use feste::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  Training Small GPT-2 on Shakespeare");
    println!("{}", "=".repeat(70));
    println!();

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/shakespeare_small_{}", timestamp);
    fs::create_dir_all(&run_dir)?;
    println!("📁 Output directory: {}/\n", run_dir);

    // ========================================================================
    // 1. Load Training Data
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("1. Loading Training Data");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string("shakespeare.txt").map_err(|_| {
        "shakespeare.txt not found. Download with:\n  \
         curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt"
    })?;

    println!(
        "Loaded: {} bytes ({:.2} MB)",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );
    println!("Characters: {}", text.chars().count());

    // ========================================================================
    // Data Configuration
    // ========================================================================
    // Configure how much of the corpus to use for different purposes

    // Tokenizer training: How many characters to use for training the BPE tokenizer
    let tokenizer_chars = 500_000;

    // Model training: How many characters to use for training the model
    let model_training_chars = 500_000;

    // Validation split: What fraction to hold out for validation (0.1 = 10%)
    let validation_fraction = 0.1;

    println!("\nData configuration:");
    println!("  Tokenizer training: first {} characters", tokenizer_chars);
    println!(
        "  Model training: first {} characters",
        model_training_chars
    );
    println!("  Validation split: {:.0}%", validation_fraction * 100.0);

    // Extract tokenizer training data
    let tokenizer_text: String = text.chars().take(tokenizer_chars).collect();
    println!("\n✓ Tokenizer corpus: {} bytes", tokenizer_text.len());

    // Extract model training data
    let model_text: String = text.chars().take(model_training_chars).collect();
    println!("✓ Model training corpus: {} bytes", model_text.len());
    println!(
        "  (Will be split {:.0}% train / {:.0}% validation)",
        (1.0 - validation_fraction) * 100.0,
        validation_fraction * 100.0
    );

    // ========================================================================
    // 2. Train Tokenizer
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("2. Training Tokenizer");
    println!("{}", "=".repeat(70));
    println!();

    println!("Training BPE tokenizer (vocab_size=1024)...");
    println!("Using {} characters from corpus", tokenizer_text.len());
    let mut tokenizer = BPETokenizer::new(1024);
    tokenizer.train(&tokenizer_text, 1024);
    println!(
        "✓ Tokenizer trained: {} tokens in vocabulary",
        tokenizer.vocab_size()
    );

    // Analyze vocabulary to show what was learned
    tokenizer.analyze_vocabulary(&tokenizer_text);

    // Save tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", run_dir);
    tokenizer.save(&tokenizer_path)?;
    println!("✓ Tokenizer saved to: {}", tokenizer_path);

    // Show compression on the model training data
    let encoded = tokenizer.encode(&model_text);
    let compression_ratio = model_text.len() as f64 / encoded.len() as f64;
    println!("\nTokenization statistics (on model training data):");
    println!("  Original bytes: {}", model_text.len());
    println!("  Encoded tokens: {}", encoded.len());
    println!("  Compression ratio: {:.2}x", compression_ratio);
    println!("  Bytes per token: {:.2}", 1.0 / compression_ratio);

    // ========================================================================
    // 3. Create Model
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Creating Model");
    println!("{}", "=".repeat(70));
    println!();

    // Small configuration: good balance of speed and quality
    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 128,       // 2x larger than tiny
        n_layers: 3,       // One more layer than tiny
        n_heads: 1,        // Single-head attention (simple)
        block_size: 128,   // Longer context than tiny
        dropout_rate: 0.1, // Dropout probability
    };

    println!("Model configuration:");
    println!("  Vocabulary size: {}", config.vocab_size);
    println!("  Embedding dimension: {}", config.n_embd);
    println!("  Number of layers: {}", config.n_layers);
    println!("  Number of heads: {}", config.n_heads);
    println!("  Context length: {}", config.block_size);

    let mut model = TrainableGPT2::new(&config);
    let num_params = model.num_parameters();
    println!("\nModel size:");
    println!("  Total parameters: {}", num_params);
    println!("  Size: {:.2}M parameters", num_params as f64 / 1_000_000.0);
    println!("  Memory: ~{:.1} MB", num_params as f64 * 4.0 / 1_000_000.0);

    // ========================================================================
    // 4. Train Model
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Training");
    println!("{}", "=".repeat(70));
    println!();

    println!("Training configuration:");

    // Training hyperparameters - now with adaptive LR scheduling!
    let num_steps = 100000; // 🟢 High max limit - adaptive LR + early stopping find convergence
    let learning_rate = 0.002;
    let seq_len = config.block_size; // Use full context window
    let patience = 5000; // Early stopping patience (works with adaptive LR)
    let warmup_fraction = 0.1; // 10% of steps for warmup (10000 steps)
    let gradient_clip_norm = 1.0; // Max gradient norm

    // Note: Adaptive LR scheduling automatically reduces learning rate on plateaus

    println!("  Training steps: {}", num_steps);
    println!(
        "  Learning rate: {} (with warmup and cosine decay)",
        learning_rate
    );
    println!("  Warmup fraction: {}", warmup_fraction);
    println!("  Sequence length: {}", seq_len);
    println!("  Gradient clipping: {}", gradient_clip_norm);
    println!("  Early stopping patience: {}", patience);
    println!("  Gradient accumulation: 8 mini-batches");
    println!("\nExpected time: 10-20 minutes on modern CPU");
    println!("Progress is logged every 50 steps");
    println!("Sample text is generated every 200 steps\n");

    // Weight decay for regularization (tuned for small dataset)
    let weight_decay = 0.01; // Light weight decay for ~200K tokens

    // Run training with configurable validation split
    train_gpt2(
        &mut model,
        &tokenizer,
        &model_text,
        num_steps,
        learning_rate,
        seq_len,
        Some(&run_dir),
        patience,
        warmup_fraction,
        gradient_clip_norm,
        validation_fraction,
        weight_decay,
    );

    // ========================================================================
    // 5. Final Generation - Text Completion
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("5. Text Completion with Trained Model");
    println!("{}", "=".repeat(70));
    println!();

    println!("Testing model with various prompts and temperatures:\n");

    // Test with different prompts and temperatures
    let test_cases = vec![
        ("To be, or not to be", 0.8, 80),
        ("ROMEO.", 0.8, 80),
        ("What is", 0.8, 80),
        ("The king", 0.8, 80),
        ("Love is", 0.8, 80),
        ("O Romeo", 0.8, 80),
        // Same prompt, different temperatures
        ("To be, or not to be", 0.3, 80), // More focused/deterministic
        ("To be, or not to be", 1.2, 80), // More creative/random
    ];

    for (prompt, temperature, max_tokens) in test_cases {
        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, max_tokens, temperature);
        let generated_text = tokenizer.decode(&generated);

        println!("─────────────────────────────────────────────────────────────────────");
        println!("Prompt: \"{}\" (temperature: {})", prompt, temperature);
        println!("─────────────────────────────────────────────────────────────────────");
        println!("{}", generated_text);
        println!();
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));
    println!();

    println!("✅ Training completed successfully!");
    println!();
    println!("Output files in: {}/", run_dir);
    println!("  ├── training_log.csv        Training metrics");
    println!("  ├── tokenizer.json          Trained tokenizer");
    println!("  ├── checkpoint_best.bin     Best model (lowest val loss)");
    println!("  ├── checkpoint_step_*.bin   Periodic checkpoints");
    println!("  └── checkpoint_final.bin    Final model");
    println!();
    println!("Analyze training:");
    println!("  View log: cat {}/training_log.csv", run_dir);
    println!("  Plot loss curves to see learning progress");
    println!("  Compare train vs validation loss for overfitting");
    println!();
    println!("Experiment suggestions:");
    println!("  • Try different learning rates (0.001, 0.0015, 0.0025, 0.003)");
    println!("  • Adjust warmup_fraction (0.05, 0.15, 0.2)");
    println!("  • Change gradient_clip_norm (0.5, 2.0, 5.0)");
    println!("  • Vary patience (1000, 10000, 15000, 20000)");
    println!("  • Train longer (5000, 10000 steps)");
    println!("  • Test different temperatures for generation");
    println!();
    println!("Try other model sizes:");
    println!("  • 05_train_shakespeare_tiny.rs (~50K params)");
    println!("  • 07_train_shakespeare_medium.rs (~4M params)");
    println!("  • 08_train_shakespeare_gpt2.rs (~163M params, GPT-2 Small)");

    Ok(())
}
