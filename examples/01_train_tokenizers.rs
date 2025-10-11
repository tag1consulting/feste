//! Train BPE tokenizers at multiple vocabulary sizes
//!
//! This example demonstrates:
//! - Loading training data (Shakespeare's complete works)
//! - Training tokenizers with different vocabulary sizes
//! - Analyzing vocabulary composition and compression ratios
//! - Saving tokenizers and analysis to timestamped output directories
//!
//! Output is written to: `tokenizer_runs/run_<timestamp>/`
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example 01_train_tokenizers
//! ```
//!
//! # Expected Runtime
//!
//! Varies significantly by hardware (typically 2-10 minutes total).
//! The 1536 and 20534 vocab sizes take the longest.
//!
//! # Prerequisites
//!
//! Download Shakespeare corpus first:
//! ```bash
//! curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
//! ```

use feste::BPETokenizer;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  BPE Tokenizer Training");
    println!("{}", "=".repeat(70));

    // Create timestamped output directory
    // Uses Unix timestamp for simple, dependency-free timestamping
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = format!("tokenizer_runs/run_{}", timestamp);
    fs::create_dir_all(&run_dir)?;

    println!("\nOutput directory: {}\n", run_dir);

    // Load Shakespeare corpus
    // Expects "shakespeare.txt" in the current working directory (usually the project root)
    println!("Loading training data...");
    let text = match fs::read_to_string("shakespeare.txt") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("\nError: shakespeare.txt not found in current directory.");
            eprintln!("Please run from the project root and download the corpus:");
            eprintln!("  curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt\n");
            std::process::exit(1);
        }
    };

    println!(
        "Loaded corpus: {} bytes ({:.2} MB)\n",
        text.len(),
        text.len() as f64 / 1_000_000.0
    );

    // Define vocabulary sizes to train
    // These match the sizes used in our training examples:
    // 256   = byte-level only (no merges) - baseline
    // 512   = tiny model (examples 04, 05)
    // 1024  = small model (example 06)
    // 1536  = medium model (example 07)
    // 20534 = large model (example 08)
    let vocab_sizes = vec![256, 512, 1024, 1536, 20534];

    let mut summary_data = Vec::new();

    // Train tokenizers at each vocabulary size
    for &vocab_size in &vocab_sizes {
        println!("{}", "=".repeat(70));
        println!("Training tokenizer with vocab_size = {}", vocab_size);
        println!("{}", "=".repeat(70));

        let start = SystemTime::now();

        // Create and train tokenizer
        let mut tokenizer = BPETokenizer::new(vocab_size);
        tokenizer.train(&text, vocab_size);

        let duration = start.elapsed()?;

        // Encode the full corpus to measure compression
        println!("Encoding full corpus to measure compression...");
        let encode_start = SystemTime::now();
        let encoded = tokenizer.encode(&text);
        let encode_duration = encode_start.elapsed()?;

        let compression_ratio = text.len() as f64 / encoded.len() as f64;
        let bytes_per_token = 1.0 / compression_ratio;

        // Display results
        println!("\nResults:");
        println!("  Training time: {:.2}s", duration.as_secs_f64());
        println!("  Encoding time: {:.2}s", encode_duration.as_secs_f64());
        println!("  Vocabulary size: {}", tokenizer.vocab_size());
        println!("  Original size: {} bytes", text.len());
        println!("  Encoded length: {} tokens", encoded.len());
        println!("  Compression ratio: {:.2}x", compression_ratio);
        println!("  Bytes per token: {:.2}", bytes_per_token);

        // Test encode/decode round-trip
        println!("\nTesting round-trip encoding...");
        let test_text = "To be, or not to be, that is the question";
        let test_encoded = tokenizer.encode(test_text);
        let test_decoded = tokenizer.decode(&test_encoded);

        if test_text == test_decoded {
            println!("  ✓ Round-trip test PASSED");
        } else {
            println!("  ✗ Round-trip test FAILED");
            println!("    Original: {}", test_text);
            println!("    Decoded:  {}", test_decoded);
            return Err("Round-trip test failed".into());
        }

        // Save tokenizer
        let save_path = format!("{}/tokenizer_{}.json", run_dir, vocab_size);
        tokenizer.save(&save_path)?;
        println!("\nSaved to: {}", save_path);

        // Show vocabulary analysis for non-byte-level tokenizers
        if vocab_size > 256 {
            tokenizer.analyze_vocabulary(&text);
        }

        // Record summary data
        summary_data.push((
            vocab_size,
            duration.as_secs_f64(),
            encode_duration.as_secs_f64(),
            encoded.len(),
            compression_ratio,
        ));

        println!();
    }

    // Write summary file
    println!("{}", "=".repeat(70));
    println!("Creating summary...");
    println!("{}", "=".repeat(70));

    let summary_path = format!("{}/summary.txt", run_dir);
    let mut summary = String::new();

    summary.push_str(&format!("{}\n", "=".repeat(70)));
    summary.push_str("  Tokenizer Training Summary\n");
    summary.push_str(&format!("{}\n\n", "=".repeat(70)));

    summary.push_str(&format!(
        "Corpus: shakespeare.txt ({} bytes, {:.2} MB)\n",
        text.len(),
        text.len() as f64 / 1_000_000.0
    ));
    summary.push_str(&format!("Vocabulary sizes trained: {:?}\n\n", vocab_sizes));

    summary.push_str("Training Results:\n");
    summary.push_str(&format!("{}\n", "-".repeat(70)));
    summary.push_str(&format!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}\n",
        "Vocab", "Train(s)", "Encode(s)", "Tokens", "Compress"
    ));
    summary.push_str(&format!("{}\n", "-".repeat(70)));

    for (vocab_size, train_time, encode_time, token_count, ratio) in &summary_data {
        summary.push_str(&format!(
            "{:<10} {:>12.2} {:>12.2} {:>12} {:>11.2}x\n",
            vocab_size, train_time, encode_time, token_count, ratio
        ));
    }

    summary.push_str(&format!("\n{}\n", "-".repeat(70)));

    summary.push_str("\nKey Observations:\n\n");
    summary.push_str("1. Compression Ratio: Larger vocabularies achieve better compression\n");
    summary.push_str("   (fewer tokens needed to represent the same text)\n\n");
    summary.push_str("2. Training Time: Increases with vocabulary size due to more merges\n");
    summary.push_str("   (but uses sampling optimization for very large vocabs)\n\n");
    summary.push_str("3. Trade-offs:\n");
    summary.push_str("   - Larger vocab = better compression = shorter sequences\n");
    summary.push_str("   - Shorter sequences = faster training and inference for the model\n");
    summary.push_str("   - But: larger embedding tables, more parameters to learn\n\n");

    summary.push_str("Common vocabulary sizes in practice:\n");
    summary.push_str("  - GPT-2: 50,257 tokens\n");
    summary.push_str("  - GPT-3: 50,257 tokens\n");
    summary.push_str("  - Educational models: 512-5000 tokens\n\n");

    summary.push_str(&format!("{}\n", "=".repeat(70)));

    fs::write(&summary_path, summary)?;
    println!("\nSummary written to: {}", summary_path);

    println!("\n{}", "=".repeat(70));
    println!("  Training Complete!");
    println!("{}", "=".repeat(70));
    println!("\nAll tokenizers saved to: {}/", run_dir);
    println!(
        "\nTo use a tokenizer in your code:\n  \
         let tokenizer = BPETokenizer::load(\"{}tokenizer_1024.json\")?;",
        run_dir
    );
    println!();

    Ok(())
}
