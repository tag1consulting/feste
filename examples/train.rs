//! Configurable GPT-2 Training with Blog Experiment Presets
//!
//! Reproduce any experiment from the blog post "A Witless Fool, Building an
//! LLM From Scratch in Rust" using named presets, or define your own
//! configuration via command-line arguments.
//!
//! ## Usage
//!
//! ```bash
//! # List available presets
//! cargo run --release --example train -- --list-presets
//!
//! # Reproduce blog experiments
//! cargo run --release --example train -- --preset pocket-bard
//! cargo run --release --example train -- --preset spider
//!
//! # Override preset parameters
//! cargo run --release --example train -- --preset pocket-bard --steps 10000
//!
//! # Transfer learning: pre-train on TinyStories, then fine-tune on Shakespeare
//! cargo run --release --example train -- --preset tinystories --data TinyStoriesV2-GPT4-valid.txt
//! cargo run --release --example train -- --preset pocket-bard \
//!     --checkpoint data/tinystories_*/checkpoint_best.bin
//!
//! # Fully custom configuration
//! cargo run --release --example train -- \
//!     --embd 256 --layers 6 --heads 12 --context 448 --vocab 8192
//! ```
//!
//! ## Prerequisites
//!
//! Download Shakespeare corpus:
//! ```bash
//! curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
//! ```

use clap::Parser;
use feste::{
    gpt2_trainable::{train_gpt2, Checkpoint, TrainableGPT2},
    BPETokenizer, Config,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(
    name = "train",
    about = "Configurable GPT-2 training with blog experiment presets"
)]
struct Args {
    /// Named preset from the blog experiments (see --list-presets)
    #[arg(long)]
    preset: Option<String>,

    /// List available presets and exit
    #[arg(long)]
    list_presets: bool,

    // Model architecture (override preset or use standalone)
    /// Embedding dimension
    #[arg(long)]
    embd: Option<usize>,

    /// Number of transformer layers
    #[arg(long)]
    layers: Option<usize>,

    /// Number of attention heads
    #[arg(long)]
    heads: Option<usize>,

    /// Context window length (block_size)
    #[arg(long)]
    context: Option<usize>,

    /// Vocabulary size for BPE tokenizer
    #[arg(long)]
    vocab: Option<usize>,

    // Training parameters
    /// Maximum training steps
    #[arg(long)]
    steps: Option<usize>,

    /// Peak learning rate
    #[arg(long)]
    lr: Option<f32>,

    /// Early stopping patience (steps)
    #[arg(long)]
    patience: Option<usize>,

    /// Warmup fraction of total steps
    #[arg(long, default_value = "0.1")]
    warmup: f32,

    /// Weight decay for regularization
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Gradient clipping max norm
    #[arg(long, default_value = "1.0")]
    grad_clip: f32,

    // Data and checkpoints
    /// Path to training text file
    #[arg(long, default_value = "shakespeare.txt")]
    data: String,

    /// Load model and tokenizer from checkpoint (for fine-tuning)
    #[arg(long)]
    checkpoint: Option<String>,

    /// Load tokenizer from file instead of training one
    #[arg(long)]
    tokenizer: Option<String>,
}

struct Preset {
    name: &'static str,
    embd: usize,
    layers: usize,
    heads: usize,
    context: usize,
    vocab: usize,
    lr: f32,
    steps: usize,
    patience: usize,
    description: &'static str,
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "pocket-bard",
        embd: 256,
        layers: 6,
        heads: 12,
        context: 448,
        vocab: 8192,
        lr: 0.0003,
        steps: 50000,
        patience: 5000,
        description: "Best from-scratch config (~9M params)",
    },
    Preset {
        name: "cyclops",
        embd: 64,
        layers: 2,
        heads: 1,
        context: 64,
        vocab: 8192,
        lr: 0.001,
        steps: 30000,
        patience: 5000,
        description: "Single high-resolution attention head",
    },
    Preset {
        name: "spider",
        embd: 64,
        layers: 2,
        heads: 8,
        context: 64,
        vocab: 8192,
        lr: 0.001,
        steps: 30000,
        patience: 5000,
        description: "Eight low-resolution attention heads",
    },
    Preset {
        name: "wide",
        embd: 256,
        layers: 4,
        heads: 4,
        context: 1024,
        vocab: 8192,
        lr: 0.0003,
        steps: 30000,
        patience: 5000,
        description: "Wide and shallow (256 embd, 4 layers)",
    },
    Preset {
        name: "narrow",
        embd: 128,
        layers: 6,
        heads: 1,
        context: 1024,
        vocab: 8192,
        lr: 0.0003,
        steps: 30000,
        patience: 5000,
        description: "Narrow and deep (128 embd, 6 layers)",
    },
    Preset {
        name: "short-context",
        embd: 256,
        layers: 4,
        heads: 4,
        context: 128,
        vocab: 8192,
        lr: 0.0003,
        steps: 30000,
        patience: 5000,
        description: "Short context window (128 tokens)",
    },
    Preset {
        name: "long-context",
        embd: 256,
        layers: 4,
        heads: 4,
        context: 1024,
        vocab: 8192,
        lr: 0.0003,
        steps: 30000,
        patience: 5000,
        description: "Long context window (1024 tokens)",
    },
    Preset {
        name: "tinystories",
        embd: 256,
        layers: 6,
        heads: 8,
        context: 448,
        vocab: 8192,
        lr: 0.001,
        steps: 50000,
        patience: 2000,
        description: "TinyStories pre-training (use --data TinyStoriesV2-GPT4-valid.txt)",
    },
];

fn print_presets() {
    println!("\nAvailable presets (from the Part 5 blog post):\n");
    println!(
        "  {:<16} {:>5} {:>6} {:>5} {:>7} {:>5} {:>8} {:>6}   {}",
        "NAME", "EMBD", "LAYERS", "HEADS", "CONTEXT", "VOCAB", "LR", "STEPS", "DESCRIPTION"
    );
    println!("  {}", "-".repeat(100));
    for p in PRESETS {
        println!(
            "  {:<16} {:>5} {:>6} {:>5} {:>7} {:>5} {:>8.4} {:>6}   {}",
            p.name, p.embd, p.layers, p.heads, p.context, p.vocab, p.lr, p.steps, p.description
        );
    }
    println!("\nUsage: cargo run --release --example train -- --preset <NAME>");
    println!("Override any parameter: --preset pocket-bard --steps 10000 --lr 0.001");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.list_presets {
        print_presets();
        return Ok(());
    }

    // ========================================================================
    // Resolve configuration from preset + CLI overrides
    // ========================================================================
    let preset = if let Some(ref name) = args.preset {
        Some(
            PRESETS
                .iter()
                .find(|p| p.name == name.as_str())
                .ok_or_else(|| {
                    format!(
                        "Unknown preset '{}'. Use --list-presets to see available options.",
                        name
                    )
                })?,
        )
    } else if args.embd.is_none()
        || args.layers.is_none()
        || args.heads.is_none()
        || args.context.is_none()
    {
        return Err(
            "Provide --preset <NAME> or all model parameters (--embd, --layers, --heads, --context).\n\
             Use --list-presets to see available presets.".into()
        );
    } else {
        None
    };

    let n_embd = args.embd.unwrap_or_else(|| preset.unwrap().embd);
    let n_layers = args.layers.unwrap_or_else(|| preset.unwrap().layers);
    let n_heads = args.heads.unwrap_or_else(|| preset.unwrap().heads);
    let block_size = args.context.unwrap_or_else(|| preset.unwrap().context);
    let vocab_size = args.vocab.unwrap_or_else(|| preset.map(|p| p.vocab).unwrap_or(8192));
    let learning_rate = args.lr.unwrap_or_else(|| preset.map(|p| p.lr).unwrap_or(0.001));
    let num_steps = args.steps.unwrap_or_else(|| preset.map(|p| p.steps).unwrap_or(30000));
    let patience = args.patience.unwrap_or_else(|| preset.map(|p| p.patience).unwrap_or(5000));

    // Validate
    if n_embd % n_heads != 0 {
        return Err(format!(
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            n_embd, n_heads
        )
        .into());
    }

    // ========================================================================
    // Banner
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    if let Some(ref name) = args.preset {
        println!("  Training GPT-2 — preset: {}", name);
    } else {
        println!("  Training GPT-2 — custom configuration");
    }
    println!("{}", "=".repeat(70));
    println!();

    // Create timestamped output directory
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let prefix = args.preset.as_deref().unwrap_or("custom").replace('-', "_");
    let run_dir = format!("data/{}_{}", prefix, timestamp);
    fs::create_dir_all(&run_dir)?;
    println!("Output directory: {}/\n", run_dir);

    // ========================================================================
    // 1. Load Training Data
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("1. Loading Training Data");
    println!("{}", "=".repeat(70));
    println!();

    let text = fs::read_to_string(&args.data).map_err(|_| {
        if args.data == "shakespeare.txt" {
            format!(
                "{} not found. Download with:\n  \
                 curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt",
                args.data
            )
        } else {
            format!("{} not found.", args.data)
        }
    })?;

    println!(
        "Loaded: {} ({:.2} MB, {} characters)",
        args.data,
        text.len() as f64 / 1_000_000.0,
        text.chars().count()
    );

    // ========================================================================
    // 2. Tokenizer
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("2. Tokenizer");
    println!("{}", "=".repeat(70));
    println!();

    let (checkpoint, tokenizer) = if let Some(ref ckpt_path) = args.checkpoint {
        // Load tokenizer from checkpoint
        println!("Loading checkpoint: {}", ckpt_path);
        let checkpoint = Checkpoint::load(ckpt_path)?;
        let tokenizer = checkpoint
            .tokenizer
            .ok_or("Checkpoint does not contain a tokenizer")?;
        println!(
            "Loaded tokenizer ({} tokens) from checkpoint",
            tokenizer.vocab_size()
        );
        (Some(checkpoint.model), tokenizer)
    } else if let Some(ref tok_path) = args.tokenizer {
        // Load tokenizer from file
        println!("Loading tokenizer: {}", tok_path);
        let tokenizer = BPETokenizer::load(tok_path)
            .map_err(|_| format!("Failed to load tokenizer from {}", tok_path))?;
        println!("Loaded: {} tokens", tokenizer.vocab_size());
        (None, tokenizer)
    } else {
        // Train fresh tokenizer
        println!("Training BPE tokenizer (vocab_size={})...", vocab_size);
        let mut tokenizer = BPETokenizer::new(vocab_size);
        tokenizer.train(&text, vocab_size);
        println!("Trained: {} tokens", tokenizer.vocab_size());

        let tokenizer_path = format!("{}/tokenizer.json", run_dir);
        tokenizer.save(&tokenizer_path)?;
        println!("Saved to: {}", tokenizer_path);
        (None, tokenizer)
    };

    let encoded = tokenizer.encode(&text);
    println!("Corpus: {} tokens", encoded.len());

    // ========================================================================
    // 3. Create Model
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("3. Model");
    println!("{}", "=".repeat(70));
    println!();

    let mut model = if let Some(model) = checkpoint {
        let num_params = model.num_parameters();
        println!("  Model loaded from checkpoint");
        println!("  Vocabulary: {}", tokenizer.vocab_size());
        println!(
            "  Parameters: {} ({:.2}M)",
            num_params,
            num_params as f64 / 1_000_000.0
        );
        model
    } else {
        let config = Config {
            vocab_size: tokenizer.vocab_size(),
            n_embd,
            n_layers,
            n_heads,
            block_size,
            dropout_rate: 0.1,
        };
        let model = TrainableGPT2::new(&config);
        let num_params = model.num_parameters();
        println!("  Vocabulary: {}", tokenizer.vocab_size());
        println!("  Embedding: {}", n_embd);
        println!("  Layers: {}", n_layers);
        println!("  Heads: {} ({}d each)", n_heads, n_embd / n_heads);
        println!("  Context: {}", block_size);
        println!(
            "  Parameters: {} ({:.2}M)",
            num_params,
            num_params as f64 / 1_000_000.0
        );
        model
    };

    // ========================================================================
    // 4. Train
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("4. Training");
    println!("{}", "=".repeat(70));
    println!();

    let seq_len = block_size;
    let validation_fraction = 0.1;

    println!("  Steps: {}", num_steps);
    println!("  Learning rate: {}", learning_rate);
    println!("  Warmup: {:.0}%", args.warmup * 100.0);
    println!("  Sequence length: {}", seq_len);
    println!("  Gradient clipping: {}", args.grad_clip);
    println!("  Early stopping patience: {}", patience);
    println!("  Weight decay: {}", args.weight_decay);
    println!("  Validation split: {:.0}%", validation_fraction * 100.0);
    println!();

    train_gpt2(
        &mut model,
        &tokenizer,
        &text,
        num_steps,
        learning_rate,
        seq_len,
        Some(&run_dir),
        patience,
        args.warmup,
        args.grad_clip,
        validation_fraction,
        args.weight_decay,
    );

    // ========================================================================
    // 5. Generate Samples
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("5. Sample Generation");
    println!("{}", "=".repeat(70));
    println!();

    let prompts = vec![
        ("To be, or not to be", 0.8),
        ("ROMEO.", 0.9),
        ("The king", 0.8),
    ];

    for (prompt, temperature) in prompts {
        let prompt_tokens = tokenizer.encode(prompt);
        let generated = model.generate(&prompt_tokens, 60, temperature, 1.0);
        let generated_text = tokenizer.decode(&generated);
        let display: String = generated_text.chars().take(150).collect();

        println!("\"{}\" (t={}):", prompt, temperature);
        println!("  {}", display);
        println!();
    }

    println!("Training complete. Output: {}/", run_dir);

    Ok(())
}
