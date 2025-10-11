# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

An educational implementation demonstrating how language models work by building every component from basic operations. No deep learning frameworks are used.

## Current Status

**Phase 1 Complete: Tokenization**
**Phase 2 Complete: Tensor Operations**

The project now has a working tokenizer and tensor library. The tensor implementation includes matrix multiplication with automatic parallelization, broadcasting, numerically stable softmax, and all operations needed for transformers.

### Running Examples

```bash
# Train tokenizers at multiple vocabulary sizes (30-60 seconds)
cargo run --release --example 01_train_tokenizers

# See tensor operations in action (< 1 second)
cargo run --release --example 02_tensor_operations
```

The tensor example demonstrates matrix multiplication performance, numerical stability with large values, and broadcasting behavior.

## Getting Training Data

Download Shakespeare's complete works from Project Gutenberg:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

## License

Apache 2.0
