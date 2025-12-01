//! Train a Tiny GPT-2 Model on Shakespeare (Overfitting Demonstration)
//!
//! This example demonstrates training a minimal transformer model and shows
//! what overfitting looks like in practice:
//! - **Model size**: ~170K parameters (n_embd=64, n_layers=2, n_heads=1)
//! - **Training time**: Approximately 20-40 minutes on modern CPU
//! - **Training data**: Full Shakespeare corpus (~5.5M characters, ~900K tokens)
//! - **Training steps**: 10000 (may still see some overfitting but much less severe)
//!
//! The model now has fewer parameters than training tokens (170K params vs 900K tokens),
//! giving a healthy ~5:1 token-to-parameter ratio, which should reduce overfitting.
//!
//! ## What You'll Learn
//!
//! - Complete training loop (forward pass, loss, backward pass, optimizer)
//! - Gradient accumulation and clipping
//! - Learning rate scheduling (warmup + cosine decay)
//! - Validation loss tracking and early stopping
//! - Model checkpointing
//! - Text generation during training
//! - **MOST IMPORTANTLY**: What overfitting looks like when it happens!
//!
//! ## Expected Behavior (Reduced Overfitting with Full Corpus)
//!
//! With the full corpus, you should see improved training dynamics:
//! 1. **Early training** (steps 0-2000): Both train and val loss decrease together
//! 2. **Middle training** (steps 2000-6000): Continued improvement on both metrics
//! 3. **Late training** (steps 6000-10000): Some plateau but smaller train/val gap
//! 4. **Best checkpoint**: Should occur later in training (around step 5000-8000)
//! 5. **Early stopping**: May still trigger but much later than with small dataset
//!
//! The increased data (900K tokens vs 88K tokens) gives the model more to learn from,
//! reducing memorization and improving generalization.
//!
//! ## Output
//!
//! All outputs are saved to: `data/shakespeare_tiny_<timestamp>/`
//! - `training_log.csv` - Step-by-step training metrics
//! - `checkpoint_best.bin` - Model with best validation loss
//! - `checkpoint_step_*.bin` - Periodic checkpoints every 250 steps
//! - `checkpoint_final.bin` - Final model after all steps
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example 05_train_shakespeare_tiny
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
//! - Vocabulary size: 512 tokens
//! - Context length: 64 tokens
//! - Embedding dimension: 64
//! - Number of layers: 2
//! - Number of heads: 1
//!
//! Training hyperparameters (configured to trigger overfitting):
//! - Training steps: 10000 (intentionally way too many for this dataset!)
//! - Learning rate: 0.003
//! - Warmup: 10% of training steps (1000 steps)
//! - Gradient clipping: 1.0
//! - Early stopping patience: 3000 steps (may trigger if val loss plateaus)
//! - Sequence length: 64 tokens (same as block_size)
//! - Gradient accumulation: 8 mini-batches
//!
//! Data configuration (full corpus for better generalization):
//! - Tokenizer training data: Full Shakespeare corpus (~5.5M characters)
//! - Model training data: Full Shakespeare corpus (~900K tokens)
//! - Validation split: 10% held out for validation
//! - Token-to-parameter ratio: 900K tokens / 170K params ≈ 5:1 (healthier!)
//!
//! With 5:1 tokens:params, the model has enough data to learn patterns without
//! excessive memorization, though this ratio is still on the low side.

use feste::{
    gpt2_trainable::{train_gpt2, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  Training Tiny GPT-2 on Shakespeare");
    println!("  (Full Corpus - Healthier Training)");
    println!("{}", "=".repeat(70));
    println!();
    println!("🎓 EDUCATIONAL NOTE:");
    println!("This example now uses the FULL Shakespeare corpus (~900K tokens).");
    println!("With a 5:1 token-to-parameter ratio, you should see much better");
    println!("generalization compared to the small 88K token subset.");
    println!();

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/shakespeare_tiny_{}", timestamp);
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
    let tokenizer_chars = text.chars().count(); // Use full corpus for tokenizer

    // Model training: How many characters to use for training the model
    let model_training_chars = text.chars().count(); // Use full corpus for training

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

    println!("Training BPE tokenizer (vocab_size=512)...");
    println!("Using {} characters from corpus", tokenizer_text.len());
    let mut tokenizer = BPETokenizer::new(512);
    tokenizer.train(&tokenizer_text, 512);
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

    // Tiny configuration: minimal parameters for fast training
    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        n_embd: 64,        // Very small embedding dimension
        n_layers: 2,       // Just 2 layers
        n_heads: 1,        // Single-head attention
        block_size: 64,    // Short context
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
    println!("  Size: {:.1}K parameters", num_params as f64 / 1_000.0);
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
    // With adaptive LR, we can set high max_steps and let the scheduler handle convergence
    let num_steps = 100000; // 🟢 High max limit - adaptive LR + early stopping find convergence
    let learning_rate = 0.003;
    let seq_len = config.block_size; // Use full context window
    let patience = 5000; // Early stopping works alongside adaptive LR reduction
    let warmup_fraction = 0.1; // 10% of steps for warmup (10000 steps)

    // Note: The library now includes adaptive LR scheduling that will automatically
    // reduce learning rate when validation loss plateaus, extending effective training.
    // This means you can set num_steps high and let the system find the right point!
    let gradient_clip_norm = 1.0; // Max gradient norm

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
    println!("\n✅ Healthier Training Configuration:");
    println!("  • Model parameters: ~170K");
    println!("  • Training tokens: ~900K (token:parameter ratio = 5:1)");
    println!("  • Training for 10000 steps with full corpus");
    println!("  • Expect both train and val loss to improve together for longer");
    println!("  • Best checkpoint should occur later (around step 5000-8000)");
    println!("  • May still see some overfitting but much less severe");
    println!("\nExpected time: 20-40 minutes on modern CPU");
    println!("Progress is logged every 50 steps");
    println!("Sample text is generated every 200 steps\n");

    // Weight decay for regularization
    let weight_decay = 0.01; // Light weight decay for ~900K tokens

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
        ("To be, or not to be", 0.8, 60),
        ("ROMEO.", 0.8, 60),
        ("What is", 0.8, 60),
        ("The king", 0.8, 60),
        ("Love is", 0.8, 60),
        // Same prompt, different temperatures
        ("To be, or not to be", 0.3, 60), // More focused/deterministic
        ("To be, or not to be", 1.2, 60), // More creative/random
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
    println!("\n{}", "=".repeat(70));
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
    println!("  Plot in Python/R/Excel for visualization");
    println!();
    println!("Load trained model:");
    println!(
        "  Use Checkpoint::load(\"{}/checkpoint_best.bin\")",
        run_dir
    );
    println!("  Continue training or generate more text");
    println!();
    println!("Understanding the improved training dynamics:");
    println!("  • Plot train_loss vs val_loss - gap should be smaller than before");
    println!("  • Best model checkpoint should occur later in training");
    println!("  • Validation loss should continue improving for more steps");
    println!("  • Generated text should show better variety and structure");
    println!();
    println!("Next steps to explore:");
    println!("  1. Compare to 88K token version (set model_training_chars = 200_000)");
    println!("  2. Try reducing model size for even better token:param ratio");
    println!("     - Try n_embd=32, n_layers=2 for ~43K params (~21:1 ratio)");
    println!("  3. Experiment with longer training (20K steps) to see if it helps");
    println!("  4. Try different validation splits (5% or 15%)");
    println!();
    println!("Token-to-parameter ratio guidance:");
    println!("  • Current: ~5:1 (better but still low by modern standards)");
    println!("  • Recommended minimum: 10:1 for basic generalization");
    println!("  • Optimal (Chinchilla): 20:1 for compute-efficient training");
    println!();
    println!("Try other model sizes:");
    println!("  • 06_train_shakespeare_small.rs (~874K params)");
    println!("  • 07_train_shakespeare_medium.rs (~4M params)");
    println!("  • 08_train_shakespeare_gpt2.rs (~163M params, GPT-2 Small)");

    Ok(())
}
