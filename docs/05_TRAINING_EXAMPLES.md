# Phase 5: Complete Training Examples

This document describes the four progressive training examples that demonstrate training GPT-2 style transformers on Shakespeare's complete works.

## Overview

Phase 5 implements four complete training examples, progressing from tiny (2 minutes) to large (24-30 hours):

| Example | Parameters | Time | Training Data |
|---------|------------|------|---------------|
| 05: Tiny | ~50K | 2-5 min | First 200K chars |
| 06: Small | ~200K | 10-20 min | First 500K chars |
| 07: Medium | ~4M | 1-2 hours | First 2M chars or full corpus |
| 08: GPT-2 Small | ~163M | 24-30 hours | Full corpus |

Each example provides default hyperparameters that can be adjusted to explore training dynamics and model behavior.

## Example 05: Train Shakespeare Tiny

**File**: `examples/05_train_shakespeare_tiny.rs`

### Configuration
```rust
Config {
    vocab_size: 512,    // Minimal vocabulary
    n_embd: 64,         // Small embedding dimension
    n_layers: 2,        // Just 2 layers
    n_heads: 1,         // Single-head attention
    block_size: 64,     // Short context
}
```

### Training Hyperparameters (Defaults)
- Steps: 500
- Learning rate: 0.003
- Warmup fraction: 0.1 (10% of steps)
- Gradient clipping: 1.0
- Early stopping patience: 5000
- Sequence length: 64 (same as block_size)
- Gradient accumulation: 8 mini-batches
- Expected time: 2-5 minutes

### Data Configuration (Defaults)
- Tokenizer training corpus: First 200K characters
- Model training corpus: First 200K characters
- Validation split: 10% of training corpus

All three values are configurable in the example code, allowing you to experiment with different data strategies (e.g., train tokenizer on full corpus but model on subset).

### Purpose
The fastest way to see a model learn. Useful for:
- Testing the training pipeline
- Quick experiments with hyperparameters
- Understanding the training loop
- Rapid iteration

### What You'll Learn
- Complete training loop (forward, backward, optimization)
- Gradient accumulation and clipping
- Learning rate scheduling
- Validation loss tracking
- Model checkpointing
- Text generation during training

### Usage
```bash
cargo run --release --example 05_train_shakespeare_tiny
```

All outputs saved to: `data/shakespeare_tiny_<timestamp>/`

---

## Example 06: Train Shakespeare Small

**File**: `examples/06_train_shakespeare_small.rs`

### Configuration
```rust
Config {
    vocab_size: 1024,   // 2x larger than tiny
    n_embd: 128,        // 2x larger than tiny
    n_layers: 3,        // One more layer
    n_heads: 1,         // Single-head attention
    block_size: 128,    // Longer context
}
```

### Training Hyperparameters (Defaults)
- Steps: 2000
- Learning rate: 0.002
- Warmup fraction: 0.1 (10% of steps)
- Gradient clipping: 1.0
- Early stopping patience: 5000
- Sequence length: 128 (same as block_size)
- Gradient accumulation: 8 mini-batches
- Expected time: 10-20 minutes

### Data Configuration (Defaults)
- Tokenizer training corpus: First 500K characters
- Model training corpus: First 500K characters
- Validation split: 10% of training corpus

All three values are configurable in the example code, allowing you to experiment with different data strategies.

### Purpose
A good balance between training speed and model capacity for experimentation.

### What You'll Learn
- How model size affects learning quality
- The relationship between training time and convergence
- Impact of different hyperparameters on training dynamics
- How tokenization quality affects generation

### Usage
```bash
cargo run --release --example 06_train_shakespeare_small
```

All outputs saved to: `data/shakespeare_small_<timestamp>/`

---

## Example 07: Train Shakespeare Medium

**File**: `examples/07_train_shakespeare_medium.rs`

### Configuration
```rust
Config {
    vocab_size: 1536,
    n_embd: 256,
    n_layers: 4,
    n_heads: 4,         // Multi-head attention (head_dim=64)
    block_size: 256,
}
```

**Total Parameters**: ~4M

### Training Hyperparameters (Defaults)
- Steps: 8000
- Learning rate: 0.0003
- Warmup fraction: 0.1 (10% of steps)
- Gradient clipping: 1.0
- Early stopping patience: 3000
- Sequence length: 256 (same as block_size)
- Gradient accumulation: 8 mini-batches
- Expected time: 1.5-2 hours

### Data Configuration (Defaults)
- Tokenizer training corpus: First 2M characters (or full if smaller)
- Model training corpus: First 2M characters (or full if smaller)
- Validation split: 10% of training corpus

All three values are configurable in the example code, allowing you to experiment with different data strategies.

### Purpose
Demonstrates training with multi-head attention and longer context windows.

### What You'll Learn
- Training with multi-head attention
- Impact of longer context windows
- How model capacity interacts with dataset size
- Monitoring training/validation loss divergence
- When training has converged

### Usage
```bash
# Run in background (recommended for 1+ hour runs)
cargo run --release --example 07_train_shakespeare_medium > training.log 2>&1 &

# Monitor progress
tail -f training.log
# or
tail -f data/shakespeare_medium_*/training_log.csv
```

All outputs saved to: `data/shakespeare_medium_<timestamp>/`

---

## Example 08: Train Shakespeare GPT-2 Small Architecture

**File**: `examples/08_train_shakespeare_gpt2.rs`

### Configuration (GPT-2 Small - 163M Parameters)
```rust
Config {
    vocab_size: 20534,
    n_embd: 768,
    n_layers: 12,
    n_heads: 12,        // head_dim=64
    block_size: 512,
}
```

**Total Parameters**: ~163M (without weight tying)

This is the same architecture as OpenAI's GPT-2 Small (124M with weight tying).

### Training Hyperparameters (Defaults)
- Steps: 5000
- Learning rate: 0.0003
- Warmup fraction: 0.1 (10% of steps)
- Gradient clipping: 1.0
- Early stopping patience: 2000
- Sequence length: 512 (same as block_size)
- Gradient accumulation: 8 mini-batches
- Expected time: **24-30 HOURS**

### Data Configuration (Defaults)
- Tokenizer training corpus: Full Shakespeare corpus
- Model training corpus: Full Shakespeare corpus
- Validation split: 10% of training corpus

All three values are configurable in the example code, allowing you to experiment with different data strategies.

### Purpose
Demonstrates training the full GPT-2 Small architecture, providing an opportunity to explore how very large models behave on smaller datasets.

### Usage
```bash
# Run in background (recommended for multi-day training)
nohup cargo run --release --example 08_train_shakespeare_gpt2 > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check if still running
ps aux | grep 08_train_shakespeare_gpt2

# View current metrics
tail -20 data/shakespeare_gpt2_*/training_log.csv

# Kill if needed
pkill -f 08_train_shakespeare_gpt2
```

All outputs saved to: `data/shakespeare_gpt2_<timestamp>/`

---

## Common Features Across All Examples

### Output Structure
Every example creates a timestamped directory with:
```
data/shakespeare_<size>_<timestamp>/
├── training_log.csv        # Step-by-step metrics
├── tokenizer.json          # Trained tokenizer
├── checkpoint_best.bin     # Best validation loss ⭐
├── checkpoint_step_*.bin   # Periodic checkpoints (every 250 steps)
└── checkpoint_final.bin    # Final model
```

### Training Infrastructure
All examples share common infrastructure:
- **Gradient accumulation**: 8 mini-batches (effective batch size = 8)
- **Gradient clipping**: Configurable (defaults vary by example)
- **Learning rate schedule**: Linear warmup + cosine decay
- **Early stopping**: Configurable patience (defaults vary by example)
- **Checkpointing**: Saves every 250 steps + best + final
- **Background saves**: Checkpoints save in separate threads
- **Validation tracking**: Computes validation loss every 50 steps
- **Sample generation**: Generates text every 200 steps

### Training Metrics Logged
The CSV log contains:
- Step number
- Learning rate (shows schedule)
- Training loss
- Validation loss
- Training perplexity (exp(train_loss))
- Validation perplexity (exp(val_loss))
- Generated text samples (every 200 steps)
- Timestamps

### Loading Trained Models
All checkpoints can be loaded for inference or continued training:

```rust
use feste::gpt2_trainable::Checkpoint;

// Load checkpoint
let checkpoint = Checkpoint::load("data/shakespeare_small_*/checkpoint_best.bin")?;
let model = checkpoint.model;
let tokenizer = checkpoint.tokenizer.unwrap();
let optimizer = checkpoint.optimizer; // Optional: for continued training

// Generate text
let prompt_tokens = tokenizer.encode("To be, or not to be");
let generated = model.generate(&prompt_tokens, 100, 0.8);
let text = tokenizer.decode(&generated);
println!("{}", text);

// Continue training (if optimizer state was saved)
if let Some(mut opt) = optimizer {
    train_gpt2(
        &mut model,
        &tokenizer,
        &text,
        5000,  // Additional steps
        0.001,
        128,
        Some("continued_training_run"),
    );
}
```

---

## Comparison Across All Sizes

### Parameter Counts

| Model | n_embd | n_layers | n_heads | vocab | Parameters |
|-------|--------|----------|---------|-------|------------|
| Tiny | 64 | 2 | 1 | 512 | ~50K |
| Small | 128 | 3 | 1 | 1024 | ~200K |
| Medium | 256 | 4 | 4 | 1536 | ~4M |
| GPT-2 Small | 768 | 12 | 12 | 20534 | ~163M |

### Training Times (on modern CPU)

| Model | Default Steps | Estimated Time | Throughput |
|-------|---------------|----------------|------------|
| Tiny | 500 | 2-5 min | ~2-3 steps/sec |
| Small | 2000 | 10-20 min | ~2 steps/sec |
| Medium | 8000 | 1.5-2 hours | ~1 step/sec |
| GPT-2 Small | 5000 | 24-30 hours | ~0.05 steps/sec |

### Memory Usage

| Model | Parameters | Memory (model) | Memory (optimizer) | Total |
|-------|------------|----------------|--------------------| ------|
| Tiny | 50K | ~0.2 MB | ~0.4 MB | ~0.6 MB |
| Small | 1M | ~4 MB | ~8 MB | ~12 MB |
| Medium | 4M | ~16 MB | ~32 MB | ~48 MB |
| **GPT-2 Small** | **163M** | **~650 MB** | **~1.3 GB** | **~2.0 GB** |

---

## Tips for Running Training Examples

### For Quick Experiments (Tiny, Small)
```bash
# Just run directly
cargo run --release --example 05_train_shakespeare_tiny
cargo run --release --example 06_train_shakespeare_small
```

### For Long Runs (Medium, GPT-2 Small)
```bash
# Use nohup to survive logout
nohup cargo run --release --example 07_train_shakespeare_medium > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check process status
ps aux | grep train_shakespeare

# View current training metrics
tail -20 data/shakespeare_medium_*/training_log.csv
```

### Monitoring Training
Watch for these indicators during training:
- **Loss trajectory**: Observe how training and validation loss change
- **Train vs. Val loss**: Note the gap between them
- **Perplexity**: Often more intuitive than raw loss values
- **Generated samples**: Qualitative assessment of learning progress
- **Numerical stability**: Watch for NaN or Inf values

### Troubleshooting

**Loss is NaN**:
- Learning rate too high (try 0.5x current rate)
- Check for corrupted data or tokenization issues

**Loss not decreasing**:
- Learning rate too low (try 2x current rate)
- Model too small for dataset
- Not enough training steps

**Loss diverging (increasing)**:
- Learning rate too high
- Gradient clipping not working
- Numerical instability in implementation

**Training very slow**:
- Expected for larger models
- Check CPU usage (should be high)
- Ensure `--release` flag is used
- Consider smaller model for faster iteration

---

## Next Steps After Training

Once you've trained models, you can:

1. **Analyze results**: Plot loss curves, compare different configs
2. **Generate text**: Try different temperatures and prompts
3. **Experiment**: Change hyperparameters, try different data
4. **Compare models**: Load multiple checkpoints, compare quality
5. **Understand internals**: Study the training loop implementation
6. **Extend**: Add features like learning rate scheduling, better sampling

## Implementation Details

All training examples use the same core function: `train_gpt2()` defined in `src/gpt2_trainable.rs`. This function handles:

- Data loading and train/val splitting
- Gradient accumulation over mini-batches
- Adam optimizer with momentum
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping (max_norm = 1.0)
- Early stopping based on validation loss
- Periodic checkpointing (every 250 steps)
- Best checkpoint tracking
- Sample generation during training
- Comprehensive logging to CSV

The examples differ only in:
- Model configuration (size, layers, context)
- Tokenizer vocabulary size
- Number of training steps
- Learning rate (tuned for model size)
- Amount of training data used

This design keeps the training infrastructure consistent while allowing easy experimentation with model configurations.

---

## Educational Value

These four examples provide a progression in model scale:

1. **Tiny**: Fast iteration for testing and learning the pipeline
2. **Small**: Quick experiments with moderate capacity
3. **Medium**: Substantial capacity with reasonable training time
4. **GPT-2 Small**: Full GPT-2 Small architecture (163M parameters)

This progression helps explore:
- How model scale affects training dynamics
- The relationship between capacity and dataset size
- Training time vs. model quality trade-offs
- Different hyperparameter configurations
- The role of context length and vocabulary size
- Validation monitoring and checkpoint management

---

**Phase 5 Implementation Status**: ✅ Complete

All four training examples have been implemented, tested, and documented. Each example compiles successfully and follows the patterns established in earlier phases (01-04).
