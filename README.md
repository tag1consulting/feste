# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

An educational implementation demonstrating how language models work by building every component from basic operations. No deep learning frameworks are used.

## Current Status

**Phases 1-3 Complete: Foundation Ready**

The project now has tokenization, tensor operations, and the complete transformer architecture. The model can run forward passes and produce predictions, though weights are random until training is implemented.

### Running Examples

```bash
# Train tokenizers (30-60 seconds)
cargo run --release --example 01_train_tokenizers

# See tensor operations (< 1 second)
cargo run --release --example 02_tensor_operations

# Explore model architecture (< 5 seconds)
cargo run --release --example 03_model_architecture
```

The model architecture example shows parameter counts for different model sizes and demonstrates a complete forward pass.

## Getting Training Data

Download Shakespeare's complete works:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

## License

Apache 2.0
