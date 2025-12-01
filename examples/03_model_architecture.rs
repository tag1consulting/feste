//! Model Architecture Demonstration
//!
//! This example demonstrates the GPT-2 model architecture:
//! - Creating models of different sizes
//! - Understanding parameter counts
//! - Forward pass through the model
//! - Inspecting intermediate layer outputs
//!
//! This shows the **architecture only** (forward pass). Training would require
//! backpropagation, optimization, and a training loop (not included in Phase 3).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example 03_model_architecture
//! ```
//!
//! # Expected Runtime
//!
//! Less than 5 seconds

use feste::{Config, GPT2};

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  GPT-2 Model Architecture Demonstration");
    println!("{}", "=".repeat(70));

    // ========== Model Configurations ==========
    println!("\n{}", "─".repeat(70));
    println!("1. Model Configurations");
    println!("{}", "─".repeat(70));

    let vocab_size = 512; // Small vocab for demonstration

    let configs = vec![
        ("Tiny", Config::tiny(vocab_size)),
        ("Small", Config::small(vocab_size)),
        ("Medium", Config::medium(vocab_size)),
        ("GPT-2 Small", Config::gpt2_small(vocab_size)),
    ];

    println!(
        "\n{:<10} {:<8} {:<8} {:<8} {:<12} {:<12}",
        "Config", "Vocab", "Embd", "Heads", "Layers", "BlockSize"
    );
    println!("{}", "-".repeat(70));

    for (name, config) in &configs {
        println!(
            "{:<10} {:<8} {:<8} {:<8} {:<12} {:<12}",
            name,
            config.vocab_size,
            config.n_embd,
            config.n_heads,
            config.n_layers,
            config.block_size
        );
    }

    // ========== Parameter Counts ==========
    println!("\n{}", "─".repeat(70));
    println!("2. Parameter Counts");
    println!("{}", "─".repeat(70));

    println!("\nCreating models and counting parameters...\n");

    for (name, config) in &configs {
        let model = GPT2::new(config);
        let params = model.count_parameters();

        println!("{} model:", name);
        println!(
            "  Total parameters: {:>12} ({:.2}M)",
            params,
            params as f32 / 1_000_000.0
        );

        // Estimate memory (4 bytes per f32 parameter)
        let memory_mb = (params * 4) as f32 / 1_000_000.0;
        println!("  Memory (weights): {:>10.1} MB", memory_mb);
        println!();
    }

    // ========== Forward Pass ==========
    println!("{}", "─".repeat(70));
    println!("3. Forward Pass");
    println!("{}", "─".repeat(70));

    // Use tiny model for demonstration
    let config = Config::tiny(vocab_size);
    let model = GPT2::new(&config);

    println!("\nUsing Tiny model for forward pass demonstration");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Embedding dim: {}", config.n_embd);
    println!("  Layers: {}", config.n_layers);

    // Create sample input: batch_size=2, seq_len=8
    let batch_size = 2;
    let seq_len = 8;

    let tokens = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],         // Batch 0
        vec![10, 20, 30, 40, 50, 60, 70, 80], // Batch 1
    ];

    println!("\nInput shape: [batch={}, seq_len={}]", batch_size, seq_len);
    println!("  Batch 0 tokens: {:?}", tokens[0]);
    println!("  Batch 1 tokens: {:?}", tokens[1]);

    println!("\nRunning forward pass...");
    let start = std::time::Instant::now();
    let logits = model.forward(&tokens);
    let elapsed = start.elapsed();

    println!("\nOutput (logits):");
    println!("  Shape: {:?}", logits.shape);
    println!(
        "  Expected: [batch={}, seq_len={}, vocab_size={}]",
        batch_size, seq_len, vocab_size
    );
    println!("  Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    // ========== Performance Benchmarks ==========
    println!("\n{}", "─".repeat(70));
    println!("3b. Performance Benchmarks");
    println!("{}", "─".repeat(70));

    println!("\nBenchmarking all model sizes with 8-token sequences:");
    println!("(Running multiple iterations for accurate measurements)\n");

    let benchmark_configs = vec![
        ("Tiny", Config::tiny(vocab_size)),
        ("Small", Config::small(vocab_size)),
        ("Medium", Config::medium(vocab_size)),
        ("GPT-2 Small", Config::gpt2_small(vocab_size)),
    ];

    // Single token benchmark
    let single_token = vec![vec![42]];

    for (name, config) in &benchmark_configs {
        let model = GPT2::new(config);

        // Warmup run
        let _ = model.forward(&tokens);

        // Benchmark with 8 tokens
        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&tokens);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        // Benchmark with 1 token
        let start_single = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&single_token);
        }
        let elapsed_single = start_single.elapsed();
        let avg_time_single = elapsed_single.as_secs_f64() * 1000.0 / iterations as f64;

        println!("{} model:", name);
        println!("  8 tokens:  {:>8.3} ms/forward", avg_time);
        println!("  1 token:   {:>8.3} ms/forward", avg_time_single);
        println!();
    }

    // Show sample logits for first position
    println!("\nSample logits at position [0, 0, :]:");
    println!("  (predictions for first token in first batch)");
    let sample_size = 10.min(vocab_size);
    print!("  First {} values: [", sample_size);
    for i in 0..sample_size {
        print!("{:.3}", logits.data[i]);
        if i < sample_size - 1 {
            print!(", ");
        }
    }
    println!("]");

    // ========== Architecture Breakdown ==========
    println!("\n{}", "─".repeat(70));
    println!("4. Architecture Breakdown");
    println!("{}", "─".repeat(70));

    println!("\nFor the Tiny model:");
    println!("\n  Token Embedding:");
    println!("    Input: [batch, seq_len] = [2, 8]");
    println!(
        "    Output: [batch, seq_len, n_embd] = [2, 8, {}]",
        config.n_embd
    );
    println!(
        "    Parameters: {} × {} = {}",
        config.vocab_size,
        config.n_embd,
        config.vocab_size * config.n_embd
    );

    println!("\n  Position Embedding:");
    println!("    Positions: 0..{}", seq_len - 1);
    println!(
        "    Output: [1, seq_len, n_embd] = [1, 8, {}]",
        config.n_embd
    );
    println!(
        "    Parameters: {} × {} = {}",
        config.block_size,
        config.n_embd,
        config.block_size * config.n_embd
    );

    println!("\n  Transformer Block (×{}):", config.n_layers);
    println!("    Each block contains:");
    println!("      - Layer Norm 1");
    println!("      - Multi-Head Attention ({} heads)", config.n_heads);
    println!(
        "        • Q, K, V projections: {} → {}",
        config.n_embd,
        config.n_embd * 3
    );
    println!(
        "        • Output projection: {} → {}",
        config.n_embd, config.n_embd
    );
    println!("      - Layer Norm 2");
    println!("      - MLP:");
    println!(
        "        • Expand: {} → {}",
        config.n_embd,
        config.n_embd * 4
    );
    println!("        • GELU activation");
    println!(
        "        • Project: {} → {}",
        config.n_embd * 4,
        config.n_embd
    );

    println!("\n  Final Layer Norm:");
    println!("    Input/Output: [batch, seq_len, n_embd]");

    println!("\n  LM Head (Output Projection):");
    println!(
        "    Input: [batch, seq_len, n_embd] = [2, 8, {}]",
        config.n_embd
    );
    println!(
        "    Output: [batch, seq_len, vocab_size] = [2, 8, {}]",
        config.vocab_size
    );
    println!(
        "    Parameters: {} × {} = {}",
        config.n_embd,
        config.vocab_size,
        config.n_embd * config.vocab_size
    );

    // ========== Multi-Head Attention Explanation ==========
    println!("\n{}", "─".repeat(70));
    println!("5. Multi-Head Attention Details");
    println!("{}", "─".repeat(70));

    println!("\nSingle-head attention (Tiny model):");
    let tiny_config = Config::tiny(vocab_size);
    let tiny_head_dim = tiny_config.n_embd / tiny_config.n_heads;
    println!(
        "  n_embd={}, n_heads={}, head_dim={}",
        tiny_config.n_embd, tiny_config.n_heads, tiny_head_dim
    );
    println!(
        "  The entire {} dimensions attend as one unit",
        tiny_config.n_embd
    );
    println!("  Simple but limited in what patterns it can learn");

    println!("\nMulti-head attention (GPT-2 Small model):");
    let gpt2_config = Config::gpt2_small(vocab_size);
    let gpt2_head_dim = gpt2_config.n_embd / gpt2_config.n_heads;
    println!(
        "  n_embd={}, n_heads={}, head_dim={}",
        gpt2_config.n_embd, gpt2_config.n_heads, gpt2_head_dim
    );
    println!(
        "  Split {} dimensions into {} independent heads of {} dimensions each",
        gpt2_config.n_embd, gpt2_config.n_heads, gpt2_head_dim
    );
    println!("  Each head can learn different attention patterns in parallel");

    println!("\nHow multi-head attention works:");
    println!(
        "  1. Project input to Q, K, V: [batch, seq, {}]",
        gpt2_config.n_embd
    );
    println!(
        "  2. Reshape into heads: [batch, {}, seq, {}]",
        gpt2_config.n_heads, gpt2_head_dim
    );
    println!("  3. Each head computes attention independently");
    println!("     Head 1 might focus on nearby words (local syntax)");
    println!("     Head 2 might focus on sentence structure (long-range dependencies)");
    println!("     Head 3 might focus on semantic relationships");
    println!("     ... and so on for all {} heads", gpt2_config.n_heads);
    println!(
        "  4. Concatenate all heads: [batch, seq, {}]",
        gpt2_config.n_embd
    );
    println!(
        "  5. Output projection: [batch, seq, {}]",
        gpt2_config.n_embd
    );

    // ========== Causal Attention Explanation ==========
    println!("\n{}", "─".repeat(70));
    println!("6. Causal (Autoregressive) Attention");
    println!("{}", "─".repeat(70));

    println!("\nIn language modeling, we predict the NEXT token.");
    println!("Position i cannot see positions i+1, i+2, ... (the future)");

    // Actually create and display the causal mask
    println!("\nCreating causal mask for seq_len=4:");
    let demo_seq_len = 4;
    let mut mask_data = vec![0.0; demo_seq_len * demo_seq_len];
    for i in 0..demo_seq_len {
        for j in 0..demo_seq_len {
            if j > i {
                mask_data[i * demo_seq_len + j] = 1.0; // Mask future positions
            }
        }
    }

    println!("\nMask (1 = masked, 0 = visible):");
    println!("  Pos:  0  1  2  3");
    for i in 0..demo_seq_len {
        print!("    {}: [", i);
        for j in 0..demo_seq_len {
            let val = mask_data[i * demo_seq_len + j];
            if val == 0.0 {
                print!("✓  ");
            } else {
                print!("✗  ");
            }
        }
        println!("]  position {} can see positions 0..={}", i, i);
    }

    println!("\nHow it works:");
    println!("  1. Compute attention scores between all positions");
    println!("  2. Set future positions (where mask=1) to -∞");
    println!("  3. Apply softmax: exp(-∞) = 0, so no attention to future");
    println!("  4. Each position can only attend to itself and past positions");

    // ========== Summary ==========
    println!("\n{}", "=".repeat(70));
    println!("  Summary");
    println!("{}", "=".repeat(70));

    println!("\n✓ GPT-2 architecture implemented from scratch");
    println!("✓ Forward pass working for inference");
    println!("✓ Multiple model sizes available (tiny → GPT-2 Small)");
    println!("✓ All components: embeddings, attention, MLP, layer norm");

    println!("\nKey architectural choices:");
    println!("  • Multi-head self-attention (parallel attention operations)");
    println!("  • Causal masking (prevent seeing future tokens)");
    println!("  • Residual connections (help gradient flow)");
    println!("  • Layer normalization (stabilize activations)");
    println!("  • GELU activation (smooth, works well in practice)");
    println!("  • 4× expansion in MLP (provides representational capacity)");
    println!();
}
