# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

An educational implementation demonstrating how language models work by building every component from basic operations. No deep learning frameworks are used.

## Current Status

**Phases 1-4 Complete: Training Infrastructure Ready**

The project has tokenization, tensor operations, model architecture, and complete training infrastructure including backpropagation through all layers, Adam optimization, and gradient clipping.

Training examples will be added in Phase 5.

### Running Examples

```bash
# Train tokenizers (30-60 seconds)
cargo run --release --example 01_train_tokenizers

# Tensor operations (< 1 second)
cargo run --release --example 02_tensor_operations

# Model architecture (< 5 seconds)
cargo run --release --example 03_model_architecture

# Training infrastructure demo
cargo run --release --example 04_training_infrastructure
```

## Getting Training Data

Download Shakespeare's complete works:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

## License

Apache 2.0
