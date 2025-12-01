# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

A complete trainable transformer that demonstrates how language models work by implementing every component from basic operations. No deep learning frameworks are used.

The implementation trains on Shakespeare's works and generates text in similar style, showing clear perplexity improvements as training progresses.

## Quick Start

```bash
# Get training data
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt

# Train a small model (10-15 minutes)
cargo run --release --example 06_train_shakespeare_small

# Generate text with different temperatures
cargo run --release --example 09_temperature_sampling
```

## Training Examples

Models of different sizes for different compute budgets:

- `05_train_shakespeare_tiny` - 50K parameters, 2-5 minutes
- `06_train_shakespeare_small` - 200K parameters, 10-20 minutes
- `07_train_shakespeare_medium` - 4M parameters, 1-2 hours
- `08_train_shakespeare_gpt2` - **163M parameters (GPT-2 Small), 24-30 HOURS**

All examples expose hyperparameters (learning rate, warmup, patience, gradient clipping) for experimentation.

## Text Generation Examples

- `09_temperature_sampling` - How temperature affects creativity (30 seconds)
- `10_prompt_engineering` - How prompts influence output (1-2 minutes)
- `11_generation_speed_benchmark` - Measuring O(n²) behavior (2-3 minutes)

See [`docs/06_GENERATION.md`](docs/06_GENERATION.md) for detailed generation documentation.

## Foundation Examples

- `01_train_tokenizers` - BPE tokenization at multiple vocab sizes
- `02_tensor_operations` - Matrix multiplication and operations
- `03_model_architecture` - Transformer architecture exploration
- `04_training_infrastructure` - Training loop components

## License

Apache 2.0
