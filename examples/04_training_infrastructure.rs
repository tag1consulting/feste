//! Training Infrastructure Demonstration
//!
//! This example demonstrates the training infrastructure without actually training:
//! - Data loading and batching
//! - Training/validation splits
//! - Logging metrics
//!
//! This shows how the pieces fit together before implementing the full training loop.
//!
//! Output is written to: `data/example_training_<timestamp>/`

use feste::{compute_dataset_loss, train_val_split, BPETokenizer, TextDataLoader, TrainingLogger};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Training Infrastructure Demonstration ===\n");

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("data/example_training_{}", timestamp);
    fs::create_dir_all(&run_dir)?;

    println!("Output directory: {}\n", run_dir);

    // Load a small sample of text
    let text = fs::read_to_string("shakespeare.txt")
        .expect("shakespeare.txt not found - download from Project Gutenberg");

    // Use first 100K characters for this demo
    let text: String = text.chars().take(100_000).collect();
    println!("Loaded {} characters of text\n", text.len());

    // Train a small tokenizer
    println!("Training tokenizer (vocab_size=512)...");
    let mut tokenizer = BPETokenizer::new(512);
    tokenizer.train(&text, 512);
    println!(
        "Tokenizer trained: {} tokens in vocabulary\n",
        tokenizer.vocab_size()
    );

    // Analyze vocabulary to show what was learned
    tokenizer.analyze_vocabulary(&text);

    // ========================================================================
    // 1. Data Loading
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("1. Data Loading");
    println!("{}", "=".repeat(70));

    let seq_len = 64;
    let batch_size = 4;

    let mut loader = TextDataLoader::new(&text, &tokenizer, seq_len, batch_size);

    println!("\nCreated data loader:");
    println!("  Sequence length: {}", seq_len);
    println!("  Batch size: {}", batch_size);
    println!("  Estimated batches per epoch: {}\n", loader.num_batches());

    // Get a few batches to demonstrate
    println!("Fetching sample batches:");
    for i in 0..3 {
        if let Some((inputs, targets)) = loader.next_batch() {
            println!(
                "  Batch {}: {} sequences × {} tokens",
                i + 1,
                inputs.len(),
                inputs[0].len()
            );

            // Show first sequence
            if i == 0 {
                println!("\n  First sequence in batch:");
                println!("    Input tokens:  {:?}...", &inputs[0][..10]);
                println!("    Target tokens: {:?}...", &targets[0][..10]);
                println!("    (targets are inputs shifted by 1)");
            }
        }
    }

    // ========================================================================
    // 2. Train/Val Split
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("2. Train/Validation Split");
    println!("{}", "=".repeat(70));

    let all_tokens = tokenizer.encode(&text);
    let (train_tokens, val_tokens) = train_val_split(&all_tokens, 0.1);

    println!("\nSplit data:");
    println!("  Total tokens: {}", all_tokens.len());
    println!("  Training tokens: {} (90%)", train_tokens.len());
    println!("  Validation tokens: {} (10%)", val_tokens.len());

    // ========================================================================
    // 3. Loss Computation
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Computing Dataset Loss");
    println!("{}", "=".repeat(70));

    println!("\nComputing loss on validation set...");

    // Dummy loss function (random baseline)
    let vocab_size = tokenizer.vocab_size();
    let random_loss = (vocab_size as f32).ln(); // Loss for random guessing

    let val_loss = compute_dataset_loss(
        val_tokens,
        seq_len,
        10, // num_batches
        |_input, _target| {
            // In real training, this would be model.compute_loss(input, target)
            // For this demo, we return a "random" baseline loss
            random_loss
        },
    );

    println!("  Validation loss: {:.4}", val_loss);
    println!("  (Random baseline for vocab_size={})", vocab_size);
    println!("  Perplexity: {:.2}", val_loss.exp());

    // ========================================================================
    // 4. Training Logger
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Training Logger");
    println!("{}", "=".repeat(70));

    let log_path = format!("{}/training_log.csv", run_dir);
    println!("\nCreating training logger: {}", log_path);

    let mut logger = TrainingLogger::new(&log_path)?;

    println!("\nSimulating training steps:");
    // Simulate improving loss over 10 steps
    for step in 0..10 {
        let train_loss = random_loss * (1.0 - step as f32 * 0.05); // Fake improvement
        let val_loss = random_loss * (1.0 - step as f32 * 0.04);

        let sample = if step % 3 == 0 {
            Some("To be, or not to be") // Fake sample
        } else {
            None
        };

        logger.log(step * 10, 0.001, train_loss, val_loss, sample)?;
    }

    println!("\n✅ Training log written to: {}", log_path);
    println!("   View it with: cat {}", log_path);
    println!("   Or import into Excel/Python for plotting");
    println!("\nAll outputs saved to: {}/", run_dir);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));

    println!("\n✓ Data loader: Efficiently batches token sequences");
    println!("✓ Train/val split: Separates data for evaluation");
    println!("✓ Loss computation: Evaluates model on dataset");
    println!("✓ Training logger: Records metrics to CSV for analysis");

    Ok(())
}
