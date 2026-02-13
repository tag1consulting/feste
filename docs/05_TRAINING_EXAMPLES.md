# Training at Scale: Running the Examples

This document is a companion to the blog post ["A Witless Fool, Building an LLM From Scratch in Rust"](https://www.tag1.com/how-to/part5-witless-fool-building-an-llm-from-scratch/). It covers configuration details and provides guidance for running the training examples in this repository.

## What's Here

Four progressive training examples that train GPT-2 style transformers on Shakespeare's complete works. The blog post covers the experiments, findings, and what the models actually learned. This document is a reference for running the examples yourself.

**Example files:**
- [examples/05_train_shakespeare_tiny.rs](../examples/05_train_shakespeare_tiny.rs) - Tiny model (~50K params)
- [examples/06_train_shakespeare_small.rs](../examples/06_train_shakespeare_small.rs) - Small model (~200K params)
- [examples/07_train_shakespeare_medium.rs](../examples/07_train_shakespeare_medium.rs) - Medium model (~4M params)
- [examples/08_train_shakespeare_gpt2.rs](../examples/08_train_shakespeare_gpt2.rs) - GPT-2 Small architecture (~163M params)

All examples use the `train_gpt2()` function from [src/gpt2_trainable.rs](../src/gpt2_trainable.rs).

## Running the Examples

### Quick experiments (Tiny, Small)
```bash
cargo run --release --example 05_train_shakespeare_tiny
cargo run --release --example 06_train_shakespeare_small
```

### Long runs (Medium, GPT-2 Small)
```bash
nohup cargo run --release --example 07_train_shakespeare_medium > training.log 2>&1 &
tail -f training.log
```

## Example Configurations

### 05: Tiny

```rust
Config {
    vocab_size: 512,
    n_embd: 64,
    n_layers: 2,
    n_heads: 1,
    block_size: 64,
}
```

| Hyperparameter | Default |
|----------------|---------|
| Steps | 500 |
| Learning rate | 0.003 |
| Warmup fraction | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 5000 |
| Gradient accumulation | 8 mini-batches |
| Training data | First 200K characters |
| Expected time | 2-5 minutes |

### 06: Small

```rust
Config {
    vocab_size: 1024,
    n_embd: 128,
    n_layers: 3,
    n_heads: 1,
    block_size: 128,
}
```

| Hyperparameter | Default |
|----------------|---------|
| Steps | 2000 |
| Learning rate | 0.002 |
| Warmup fraction | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 5000 |
| Gradient accumulation | 8 mini-batches |
| Training data | First 500K characters |
| Expected time | 10-20 minutes |

### 07: Medium (~4M params)

```rust
Config {
    vocab_size: 1536,
    n_embd: 256,
    n_layers: 4,
    n_heads: 4,       // head_dim=64
    block_size: 256,
}
```

| Hyperparameter | Default |
|----------------|---------|
| Steps | 8000 |
| Learning rate | 0.0003 |
| Warmup fraction | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 3000 |
| Gradient accumulation | 8 mini-batches |
| Training data | First 2M characters |
| Expected time | 1.5-2 hours |

### 08: GPT-2 Small (~163M params)

```rust
Config {
    vocab_size: 20534,
    n_embd: 768,
    n_layers: 12,
    n_heads: 12,      // head_dim=64
    block_size: 512,
}
```

Same architecture as OpenAI's GPT-2 Small (124M with weight tying).

| Hyperparameter | Default |
|----------------|---------|
| Steps | 5000 |
| Learning rate | 0.0003 |
| Warmup fraction | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping patience | 2000 |
| Gradient accumulation | 8 mini-batches |
| Training data | Full Shakespeare corpus |
| Expected time | **24-30 hours** |

All hyperparameters are configurable in each example's source code. Tokenizer corpus, model training corpus, and validation split (default 10%) can also be adjusted independently.

## Comparison Across Sizes

| Model | n_embd | n_layers | n_heads | vocab | Parameters |
|-------|--------|----------|---------|-------|------------|
| Tiny | 64 | 2 | 1 | 512 | ~50K |
| Small | 128 | 3 | 1 | 1024 | ~200K |
| Medium | 256 | 4 | 4 | 1536 | ~4M |
| GPT-2 Small | 768 | 12 | 12 | 20534 | ~163M |

| Model | Default Steps | Estimated Time | Throughput |
|-------|---------------|----------------|------------|
| Tiny | 500 | 2-5 min | ~2-3 steps/sec |
| Small | 2000 | 10-20 min | ~2 steps/sec |
| Medium | 8000 | 1.5-2 hours | ~1 step/sec |
| GPT-2 Small | 5000 | 24-30 hours | ~0.05 steps/sec |

| Model | Parameters | Memory (model) | Memory (optimizer) | Total |
|-------|------------|----------------|--------------------| ------|
| Tiny | 50K | ~0.2 MB | ~0.4 MB | ~0.6 MB |
| Small | 1M | ~4 MB | ~8 MB | ~12 MB |
| Medium | 4M | ~16 MB | ~32 MB | ~48 MB |
| **GPT-2 Small** | **163M** | **~650 MB** | **~1.3 GB** | **~2.0 GB** |

## Output Structure

Every example creates a timestamped directory:
```
data/shakespeare_<size>_<timestamp>/
├── training_log.csv        # Step-by-step metrics
├── tokenizer.json          # Trained tokenizer
├── checkpoint_best.bin     # Best validation loss
├── checkpoint_step_*.bin   # Periodic checkpoints (every 250 steps)
└── checkpoint_final.bin    # Final model
```

### Training Metrics Logged

The CSV log contains step number, learning rate, training loss, validation loss, training perplexity, validation perplexity, generated text samples (every 200 steps), and timestamps.

### Shared Training Infrastructure

All examples use the same core infrastructure:
- **Gradient accumulation**: 8 mini-batches (effective batch size = 8)
- **Learning rate schedule**: Linear warmup + cosine decay
- **Early stopping**: Configurable patience
- **Checkpointing**: Every 250 steps + best + final (background saves)
- **Validation**: Every 50 steps
- **Sample generation**: Every 200 steps

## Loading Trained Models

```rust
use feste::gpt2_trainable::Checkpoint;

let checkpoint = Checkpoint::load("data/shakespeare_small_*/checkpoint_best.bin")?;
let model = checkpoint.model;
let tokenizer = checkpoint.tokenizer.unwrap();

// Generate text
let prompt_tokens = tokenizer.encode("To be, or not to be");
let generated = model.generate(&prompt_tokens, 100, 0.8);
let text = tokenizer.decode(&generated);
println!("{}", text);
```

## Monitoring and Troubleshooting

### What to Watch

- **Loss trajectory**: Should decrease from baseline (ln(vocab_size))
- **Train vs. val loss gap**: Small gap is normal; divergence means overfitting
- **Generated samples**: More informative than loss numbers alone (see the blog post for why)

### Common Problems

**Loss is NaN**: Learning rate too high (try 0.5x) or corrupted data.

**Loss not decreasing**: Learning rate too low (try 2x), model too small, or not enough steps.

**Loss diverging**: Learning rate too high or numerical instability.

**Training very slow**: Ensure `--release` flag is used. Larger models are expected to be slow.

## Reproducing Blog Experiments

The configurable training example ([examples/train.rs](../examples/train.rs)) lets you reproduce any experiment from the blog post using named presets:

```bash
cargo run --release --example train -- --list-presets
cargo run --release --example train -- --preset pocket-bard
cargo run --release --example train -- --preset spider
```

### Available Presets

| Preset | embd | layers | heads | context | Blog section |
|--------|------|--------|-------|---------|--------------|
| `pocket-bard` | 256 | 6 | 12 | 448 | Pocket Bard (~9M params) |
| `cyclops` | 64 | 2 | 1 | 64 | Cyclops vs Spider |
| `spider` | 64 | 2 | 8 | 64 | Cyclops vs Spider |
| `wide` | 256 | 4 | 4 | 1024 | Wide and Shallow |
| `narrow` | 128 | 6 | 1 | 1024 | Narrow and Deep |
| `short-context` | 256 | 4 | 4 | 128 | Working Memory |
| `long-context` | 256 | 4 | 4 | 1024 | Working Memory |
| `tinystories` | 256 | 6 | 8 | 448 | TinyStories Base |

All presets use vocab 8192 for comparable perplexity. Any parameter can be overridden:

```bash
cargo run --release --example train -- --preset pocket-bard --steps 10000
```

### Transfer Learning

Pre-train on TinyStories, then fine-tune on Shakespeare:

```bash
# Step 1: Pre-train
cargo run --release --example train -- --preset tinystories \
    --data TinyStoriesV2-GPT4-valid.txt

# Step 2: Fine-tune
cargo run --release --example train -- --preset pocket-bard \
    --checkpoint data/tinystories_*/checkpoint_best.bin
```

### Custom Configurations

Skip presets entirely and specify all parameters:

```bash
cargo run --release --example train -- \
    --embd 256 --layers 6 --heads 12 --context 448 --vocab 8192 \
    --steps 50000 --lr 0.0003
```
