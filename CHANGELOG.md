# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
