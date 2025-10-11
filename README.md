# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

An educational implementation demonstrating how language models work by building every component from basic operations. No deep learning frameworks are used.

## Current Status

**Phase 1 Complete: Tokenization**

The BPE tokenizer is fully implemented and working. It can train on text corpora, encode text to token IDs, and decode back to text with perfect round-trip accuracy.

Example:
```bash
cargo run --release --example 01_train_tokenizers
```

This trains tokenizers at vocabulary sizes 256, 512, 1024, 2048, and 5000 on Shakespeare's complete works. Training takes about 30-60 seconds and demonstrates compression ratios improving with vocabulary size.

## Getting Training Data

Download Shakespeare's complete works from Project Gutenberg:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

This 5.5MB file is public domain and serves as the training corpus.

## License

Apache 2.0
