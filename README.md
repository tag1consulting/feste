# Feste

A GPT-2 style transformer language model implemented from scratch in Rust for educational purposes. Companion code for the blog series *Building an LLM From Scratch in Rust*.

## Why Feste?

Feste is the fool in Shakespeare's Twelfth Night, known for his wordplay and wit. The model trains on Shakespeare's complete works and generates text in his style, making the name a natural fit.

## What This Is

A complete trainable transformer that demonstrates how language models work by implementing every component from basic operations. No deep learning frameworks are used.

The implementation trains on Shakespeare's works and generates text in similar style, showing clear perplexity improvements as training progresses.

## Blog Series

Each part of the blog has a companion doc with configuration details and implementation reference:

| Part | Blog Post | Code Reference |
|------|-----------|----------------|
| 1 | [Tokenization](https://www.tag1.com/how-to/part1-tokenization-building-an-llm-from-scratch-in-rust/) | [`docs/01_TOKENIZATION.md`](docs/01_TOKENIZATION.md) |
| 2 | [Tensor Operations](https://www.tag1.com/how-to/part2-tensor-operations-building-an-llm-from-scratch/) | [`docs/02_TENSOR_OPERATIONS.md`](docs/02_TENSOR_OPERATIONS.md) |
| 3 | [Model Architecture](https://www.tag1.com/how-to/part3-model-architecture-building-an-llm-from-scratch/) | [`docs/03_MODEL_ARCHITECTURE.md`](docs/03_MODEL_ARCHITECTURE.md) |
| 4 | [Training Infrastructure](https://www.tag1.com/how-to/part4-training-infrastructure-building-an-llm-from-scratch/) | [`docs/04_TRAINING.md`](docs/04_TRAINING.md) |
| 5 | [A Witless Fool](https://www.tag1.com/how-to/part5-witless-fool-building-an-llm-from-scratch/) | [`docs/05_TRAINING_EXAMPLES.md`](docs/05_TRAINING_EXAMPLES.md) |

## Quick Start

```bash
# Get training data
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt

# Train a small model (10-15 minutes)
cargo run --release --example 06_train_shakespeare_small
```

## Reproducing Blog Experiments

The configurable training example lets you reproduce any experiment from the Part 5 blog post using named presets:

```bash
# List available presets
cargo run --release --example train -- --list-presets

# Run a preset
cargo run --release --example train -- --preset pocket-bard

# Override parameters
cargo run --release --example train -- --preset spider --steps 10000

# Fully custom configuration
cargo run --release --example train -- \
    --embd 256 --layers 6 --heads 12 --context 448 --vocab 8192
```

See [`docs/05_TRAINING_EXAMPLES.md`](docs/05_TRAINING_EXAMPLES.md) for the full preset table, transfer learning instructions, and details on all training examples.

## Examples

### Foundation (Parts 1-4)

- `01_train_tokenizers` - BPE tokenization at multiple vocab sizes
- `02_tensor_operations` - Matrix multiplication and operations
- `03_model_architecture` - Transformer architecture exploration
- `04_training_infrastructure` - Training loop components

### Training (Part 5)

- `05_train_shakespeare_tiny` - 50K parameters, 2-5 minutes
- `06_train_shakespeare_small` - 200K parameters, 10-20 minutes
- `07_train_shakespeare_medium` - 4M parameters, 1-2 hours
- `08_train_shakespeare_gpt2` - 163M parameters (GPT-2 Small), 24-30 hours
- `train` - Configurable training with blog experiment presets

## License

Apache 2.0
