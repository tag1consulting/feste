//! Train GPT-2 Small Architecture on Shakespeare
//!
//! This example demonstrates training a full GPT-2 Small architecture:
//! - **Model size**: ~124M parameters (n_embd=768, n_layers=12, n_heads=12, block_size=1024)
//! - **Training time**: Approximately 24-30 hours on modern CPU
//!
//! This uses the same architecture as OpenAI's GPT-2 Small (124M), providing an opportunity
//! to explore how large models behave on smaller datasets and observe the relationship
//! between model capacity, dataset size, and training dynamics.
//!
//! ## Output
//!
//! All outputs are saved to: `data/shakespeare_gpt2_<timestamp>/`
//! - `training_log.csv` - Step-by-step training metrics
//! - `checkpoint_best.bin` - Model with best validation loss
//! - `checkpoint_step_*.bin` - Periodic checkpoints every 250 steps
//! - `checkpoint_final.bin` - Final model after all steps
//!
//! ## Usage
//!
//! Run in background (recommended):
//! ```bash
//! nohup cargo run --release --example 08_train_shakespeare_gpt2 > training.log 2>&1 &
//! ```
//!
//! Monitor progress:
//! ```bash
//! tail -f training.log
//! # or
//! tail -f data/shakespeare_gpt2_*/training_log.csv
//! ```
//!
//! Check if still running:
//! ```bash
//! ps aux | grep 08_train_shakespeare_gpt2
//! ```
//!
//! Kill if needed:
//! ```bash
//! pkill -f 08_train_shakespeare_gpt2
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
//! Model architecture (GPT-2 Small):
//! - Vocabulary size: 20534 tokens
//! - Context length: 1024 tokens
//! - Embedding dimension: 768
//! - Number of layers: 12
//! - Number of heads: 12
//!
//! Training hyperparameters:
//! - Training steps: 5000
//! - Learning rate: 0.0003
//! - Warmup: 10% of training steps
//! - Gradient clipping: 1.0
//! - Early stopping patience: 2000 steps
//! - Sequence length: 512 tokens (same as block_size)
//! - Gradient accumulation: 8 mini-batches
//!
//! Data configuration:
//! - Tokenizer training data: Full corpus
//! - Model training data: Full corpus
//! - Validation split: 10% held out for validation
//!
//! These are starting values for exploration. Try adjusting them to see
//! how they affect training dynamics and final model quality.
//!
//! ## Running Long Training Sessions
//!
//! For multi-day training runs:
//! ```bash
//! # Run in background
//! nohup cargo run --release --example 08_train_shakespeare_gpt2 > training.log 2>&1 &
//!
//! # Monitor progress
//! tail -f training.log
//!
//! # Check if still running
//! ps aux | grep 08_train_shakespeare_gpt2
//! ```

use feste::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  Training GPT-2 Small on Shakespeare");
    println!("{}", "=".repeat(70));
    println!();

    println!("Training GPT-2 Small (163M parameters) on Shakespeare corpus.");
    println!("Expected training time: 24-30 hours on modern CPU.");
    println!("Consider running in background: nohup ... > training.log 2>&1 &\n");

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/shakespeare_gpt2_{}", timestamp);
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
    // For large model, use full corpus for best tokenization
    let tokenizer_chars = text.chars().count();

    // Model training: How many characters to use for training the model
    let model_training_chars = text.chars().count();

    // Validation split: What fraction to hold out for validation (0.1 = 10%)
    let validation_fraction = 0.1;

    println!("\nData configuration:");
    println!(
        "  Tokenizer training: full corpus ({} characters)",
        tokenizer_chars
    );
    println!(
        "  Model training: full corpus ({} characters)",
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

    println!("Training BPE tokenizer (vocab_size=20534)...");
    println!("Using {} characters from corpus", tokenizer_text.len());
    println!("(This will take 3-5 minutes for very large vocabulary)");
    let start_time = SystemTime::now();
    let mut tokenizer = BPETokenizer::new(20534);
    tokenizer.train(&tokenizer_text, 20534);
    let tokenizer_duration = start_time.elapsed()?.as_secs_f64();
    println!(
        "✓ Tokenizer trained in {:.1}s: {} tokens in vocabulary",
        tokenizer_duration,
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
    println!("\n  Very large vocabulary (20534 tokens) = high compression");
    println!("  But also increases model size significantly!");
    println!("  Parameter-to-token ratio: ~117:1 (WAY too high for this dataset)");

    // ========================================================================
    // 3. Create Model
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Creating Model");
    println!("{}", "=".repeat(70));
    println!();

    // GPT-2 Small architecture (768/12/12/1024)
    let config = Config {
        vocab_size: tokenizer.vocab_size(), // Trained on Shakespeare corpus
        n_embd: 768,                        // GPT-2 Small standard
        n_layers: 12,                       // GPT-2 Small standard
        n_heads: 12,                        // GPT-2 Small standard (head_dim=64)
        block_size: 1024,                   // GPT-2 standard context window
        dropout_rate: 0.1,                  // Dropout probability
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
    println!("  Size: {:.1}M parameters", num_params as f64 / 1_000_000.0);
    println!("  Memory: ~{:.1} MB", num_params as f64 * 4.0 / 1_000_000.0);

    println!("\n  ⚠️  This is the GPT-2 Small architecture (~124M params with full vocab)!");
    println!("  BUT it's TOO LARGE for Shakespeare (~1M tokens)");
    println!("  Parameter-to-token ratio is very high (causes overfitting)");
    println!("  Expect validation loss to plateau around 5.1 (perplexity ~161)");
    println!("  Medium model (4M params) achieves BETTER results in 1/15th the time!");

    // ========================================================================
    // 4. Train Model
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Training (This Will Take 24-30 HOURS - Watch For Overfitting!)");
    println!("{}", "=".repeat(70));
    println!();

    println!("Training configuration:");

    // Training hyperparameters - now with adaptive LR scheduling!
    let num_steps = 100000; // 🟢 High max limit - adaptive LR + early stopping find convergence
    let learning_rate = 0.0003;
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
    println!("\n⏱️  Expected time: 24-30 HOURS on modern CPU");
    println!("    Progress logged every 50 steps");
    println!("    Samples generated every 200 steps");
    println!("    Checkpoints saved every 250 steps\n");

    let training_start = SystemTime::now();

    // Weight decay for regularization (moderate for large overfitting model)
    let weight_decay = 0.05; // Moderate weight decay for 163M params on 1M tokens (severe overfit)

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

    let training_duration = training_start.elapsed()?.as_secs();
    let hours = training_duration / 3600;
    let minutes = (training_duration % 3600) / 60;

    // ========================================================================
    // 5. Final Generation
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("5. Final Generation");
    println!("{}", "=".repeat(70));
    println!();

    println!("Generating text samples with various temperatures:\n");

    let test_cases = vec![
        ("To be, or not to be", 0.7, "Low temperature (focused)"),
        ("To be, or not to be", 1.0, "Standard temperature"),
        ("To be, or not to be", 1.2, "High temperature (creative)"),
        ("ROMEO.", 0.8, "Character dialogue"),
        ("But soft, what light", 0.9, "Famous phrase"),
        ("The king", 0.8, "Simple prompt"),
        ("Love is", 1.0, "Abstract concept"),
    ];

    for (prompt, temperature, description) in test_cases {
        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, 100, temperature);
        let generated_text = tokenizer.decode(&generated);

        // Show first 200 characters
        let display_text: String = generated_text.chars().take(200).collect();

        println!("─────────────────────────────────────────────────────────");
        println!("Prompt: \"{}\"", prompt);
        println!("Settings: temperature={} ({})", temperature, description);
        println!("\nGenerated:");
        println!("{}", display_text);
        if generated_text.len() > 200 {
            println!("... (truncated)");
        }
        println!();
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));
    println!();

    println!("✅ Training completed!");
    println!("   Total time: {}h {}m", hours, minutes);
    println!("\n⚠️  Observe the overfitting:");
    println!();
    println!("Output files in: {}/", run_dir);
    println!("  ├── training_log.csv        Complete training history");
    println!("  ├── tokenizer.json          Trained tokenizer (20534 tokens)");
    println!("  ├── checkpoint_best.bin     Best model (lowest val loss) ⭐");
    println!("  ├── checkpoint_step_*.bin   Periodic checkpoints");
    println!("  └── checkpoint_final.bin    Final model");
    println!();
    println!("Analyze training:");
    println!("  View log: cat {}/training_log.csv", run_dir);
    println!("  Plot loss curves in Excel/Python/R");
    println!("  Look for:");
    println!("    • Steady loss decrease over time");
    println!("    • Train/val loss staying close (no overfitting)");
    println!("    • Generated samples improving dramatically");
    println!();
    println!("Actual results (GPT-2 Small, 163M params - OVERFITTED):");
    println!("  • Validation perplexity: 600+ → ~161 (WORSE than Medium!)");
    println!("  • Training perplexity: 600+ → ~130 (better, but overfitting)");
    println!("  • Validation loss: ~6.5 → ~5.1 (plateaus)");
    println!("  • Training loss: ~6.5 → ~4.8 (keeps dropping)");
    println!("  • Train/val gap: ~0.3 loss units (overfitting signal)");
    println!("  • Text quality: Good but NO BETTER than Medium model");
    println!("\n  Compare to Medium model (4M params, OPTIMAL):");
    println!("  • Validation perplexity: ~45 (3.5x BETTER!)");
    println!("  • Training time: 1-2 hours (15x FASTER!)");
    println!("  • No overfitting (minimal train/val gap)");
    println!("\n  Lesson: Bigger is NOT always better! Model sizing matters!");
    println!();
    println!("Load and use the model:");
    println!("  use feste::gpt2_trainable::Checkpoint;");
    println!(
        "  let checkpoint = Checkpoint::load(\"{}/checkpoint_best.bin\")?;",
        run_dir
    );
    println!("  let model = checkpoint.model;");
    println!("  let tokenizer = checkpoint.tokenizer.unwrap();");
    println!("  let generated = model.generate(&prompt_tokens, 200, 0.9);");
    println!();
    println!("What you learned:");
    println!("  • How to identify overfitting (train loss << val loss)");
    println!("  • Why parameter-to-token ratio matters (117:1 is too high)");
    println!("  • That bigger models don't always perform better");
    println!("  • How validation loss plateaus while training loss improves (overfit)");
    println!("\nRecommended next steps:");
    println!("  • Run 07_train_shakespeare_medium.rs for BEST results");
    println!("  • Compare the loss curves side-by-side");
    println!("  • Plot train vs val loss to visualize the overfitting");
    println!("  • Try Medium model on other datasets for optimal results");
    println!("  • Study parameter-to-token ratios for your own projects");

    Ok(())
}
