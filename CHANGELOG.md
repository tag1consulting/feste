# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2025-10-12

### Added

**Generation Fundamentals (Examples 09-11):**
- Autoregressive text generation implementation
- Temperature-based sampling for controlling randomness
- Example 09: `temperature_sampling.rs` - Demonstrates temperature effects
- Example 10: `prompt_engineering.rs` - Explores prompt effects on generation
- Example 11: `generation_speed_benchmark.rs` - Measures O(n²) performance characteristics
- In-distribution vs out-of-distribution prompt analysis

**KV Caching Infrastructure:**
- KV cache implementation for 30-150x generation speedup
- `KVCache` structure for storing attention keys and values
- `forward_with_cache()` method for incremental forward passes
- `generate_with_callback()` for streaming generation with callbacks
- `generate_until()` for early stopping at special tokens
- Reduces generation complexity from O(n³) to O(n²)

**Instruction Tuning (Examples 12-13):**
- Example 12: `instruction_tuning_chatbot.rs` - Transform models into conversational chatbots
- Example 13: `test_chatbot.rs` - Interactive chatbot REPL with streaming responses
- Instruction tuning module (`src/instruction_tuning.rs`)
- Loss masking for training only on assistant responses
- Catastrophic forgetting prevention (Shakespeare text mixing)
- Multi-turn conversation support with history management
- Interactive debugging and context window tracking

**New Dependencies:**
- `serde_json` for instruction dataset loading

**Documentation:**
- Comprehensive generation guide in `docs/06_GENERATION.md` (merged and expanded)
- Part 1: Generation fundamentals (temperature, prompts, performance)
- Part 2: Building applications (KV caching, instruction tuning, chatbots)
- Part 3: Reference (advanced sampling strategies, examples guide)

## [0.1.5] - 2025-10-11

### Added
- Four complete training examples with progressive complexity
- Example 05: Tiny model (~50K params, 2-5 minutes training)
- Example 06: Small model (~1M params, 10-20 minutes training)
- Example 07: Medium model (~4M params, 1-2 hours training)
- Example 08: Large model (TRUE GPT-2 Large: 117M params, 768d/12L/12H, 5-7 days training)
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warmup and cosine decay
- Early stopping based on validation loss
- Best model checkpoint tracking
- Periodic checkpointing every 250 steps
- Background checkpoint saving to not block training
- Sample text generation during training
- Complete Shakespeare training pipeline
- Comprehensive training documentation

## [0.1.4] - 2025-10-11

### Added
- Complete training infrastructure with forward and backward passes
- Trainable GPT-2 model with full backpropagation through all layers
- Adam optimizer with momentum and variance adaptation
- Gradient clipping for training stability
- TextDataLoader for efficient batching and data iteration
- TrainingLogger with CSV output and perplexity tracking
- Model checkpointing (save/load) with optimizer state
- Checkpoint metadata and versioning
- Training/validation data splitting utilities
- Dataset loss computation
- Modular gradient system for extensibility
- Example: `04_training_infrastructure.rs` demonstrating training setup

## [0.1.3] - 2025-10-11

### Added
- Complete GPT-2 transformer model architecture (forward pass)
- Token and position embeddings
- Layer normalization with learnable scale and shift
- Multi-head self-attention with causal masking
- MLP (feedforward) layers with GELU activation
- Transformer blocks with residual connections
- Configurable model sizes (tiny, small, medium, large)
- Parameter counting utilities
- Example: `03_model_architecture.rs` demonstrating model creation and forward pass

## [0.1.2] - 2025-10-11

### Added
- Tensor type for multi-dimensional arrays with shape and stride support
- Matrix multiplication with automatic sequential/parallel selection
- Cache-blocked parallel matrix multiplication for large matrices (8×8 tiles)
- Element-wise operations: add, subtract, multiply, divide
- Scalar operations on tensors
- Broadcasting support for operations on different shapes
- Numerically stable softmax with max subtraction
- Reshape and transpose operations
- Statistical operations: mean, variance along axes
- Masked fill operation for attention masks
- Example: `02_tensor_operations.rs` demonstrating all tensor operations

## [0.1.1] - 2025-10-11

### Added
- BPE (Byte Pair Encoding) tokenizer implementation
- Tokenizer training with configurable vocabulary size
- Text encoding and decoding with round-trip guarantees
- Parallel pair counting for faster training
- Parallel encoding for large texts (>200KB)
- Training optimization using samples for large vocabularies
- Vocabulary analysis and statistics
- Example: `01_train_tokenizers.rs` demonstrating tokenizer training on Shakespeare corpus

## [0.1.0] - 2025-10-11

### Added
- Initial project structure with src/lib.rs
- README documenting project goals and educational approach
- CHANGELOG for version tracking
- DATA.md documenting Shakespeare corpus source and licensing
- Apache 2.0 LICENSE
- Cargo.toml with proper metadata and dependencies
- .gitignore for Rust project
