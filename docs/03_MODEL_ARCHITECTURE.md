# Model Architecture: Building GPT-2 from Scratch

This document is a companion to the blog post ["Building an LLM From Scratch in Rust, Part 3: Model Architecture"](https://www.tag1.com/how-to/part3-model-architecture-building-an-llm-from-scratch/). It covers implementation details and provides guidance for working with the model architecture code in this repository.

## What's Here

This phase implements a complete GPT-2 style transformer model. The model can perform forward passes (inference) but does not yet include training capabilities. Weights are random, so predictions are meaningless until training.

**Implementation files:**
- [src/model.rs](../src/model.rs) - Core model components (Embedding, LayerNorm, MLP, Attention, Block, GPT)
- [examples/03_model_architecture.rs](../examples/03_model_architecture.rs) - Model creation and forward pass demonstration

## Running the Example

```bash
cargo run --release --example 03_model_architecture
```

This creates models of various sizes and runs forward passes through the architecture.

## Expected Output

The example creates progressively larger models:

```
======================================================================
  GPT-2 Model Architecture Demonstration
======================================================================

Creating Tiny model (for testing)...
  vocab_size: 512, n_embd: 128, n_heads: 4, n_layers: 3, block_size: 128
  Parameters: ~847,872 (0.85M)
  ✓ Model created

Creating Small model (for experiments)...
  vocab_size: 512, n_embd: 256, n_heads: 4, n_layers: 4, block_size: 256
  Parameters: ~3,473,408 (3.47M)
  ✓ Model created

Creating GPT-2 Small (standard size)...
  vocab_size: 50257, n_embd: 768, n_heads: 12, n_layers: 12, block_size: 1024
  Parameters: ~163,037,184 (163M)
  ✓ Model created

Forward pass test (Tiny model)...
  Input: batch_size=1, seq_len=16
  Output: [1, 16, 512] (logits over vocabulary)
  Time: 8.234ms
  ✓ Forward pass successful

All tests completed successfully!
```

The key indicator is that forward passes complete without errors and produce output of the expected shape `[batch, seq_len, vocab_size]`.

## Understanding the Results

### Model Sizes

Four configurations are available, each suited to different purposes:

**Tiny (0.8M parameters)**
```rust
vocab_size: 512, n_embd: 128, n_heads: 4, n_layers: 3, block_size: 128
```
Very fast, good for debugging and testing changes. Forward pass completes in under 10ms.

**Small (3.5M parameters)**
```rust
vocab_size: 512, n_embd: 256, n_heads: 4, n_layers: 4, block_size: 256
```
Good balance for experimentation. Large enough to learn patterns, small enough to iterate quickly.

**Medium (20-40M parameters)**
```rust
vocab_size: 512, n_embd: 384, n_heads: 6, n_layers: 6, block_size: 256
```
Larger capacity, still manageable on CPU. Expect forward passes around 200ms.

**Large / GPT-2 Small (163M parameters)**
```rust
vocab_size: 50257, n_embd: 768, n_heads: 12, n_layers: 12, block_size: 1024
```
Full GPT-2 Small architecture. Forward passes take 2-3 seconds on CPU. This is the standard benchmark size.

### Parameter Count Breakdown

Understanding where parameters live helps estimate memory requirements:

```
Embeddings:
  Token: vocab_size × n_embd
  Position: block_size × n_embd

Per Transformer Block:
  Attention QKV: n_embd × (3 × n_embd) + 3×n_embd (bias)
  Attention output: n_embd × n_embd + n_embd (bias)
  Attention layer norm: 2 × n_embd (gamma + beta)
  MLP expand: n_embd × (4 × n_embd) + 4×n_embd (bias)
  MLP project: (4 × n_embd) × n_embd + n_embd (bias)
  MLP layer norm: 2 × n_embd

Final:
  Layer norm: 2 × n_embd
  LM head: n_embd × vocab_size + vocab_size (bias)
```

**Example for Small model** (n_embd=256, n_layers=4, vocab=512, block_size=256):
```
Token embeddings: 512 × 256 = 131,072
Position embeddings: 256 × 256 = 65,536
Per block: ~788,480
4 blocks: 3,153,920
Final layer norm + LM head: ~131,584
Total: ~3,482,112 parameters
```

At 4 bytes per float32, that's about 14 MB for weights alone.

## Implementation Details

### Core Data Structures

```rust
pub struct GPT {
    pub wte: Embedding,        // token embeddings [vocab_size, n_embd]
    pub wpe: Embedding,        // position embeddings [block_size, n_embd]
    pub blocks: Vec<Block>,    // n_layers transformer blocks
    pub ln_f: LayerNorm,       // final layer norm
    pub lm_head: Linear,       // output projection [n_embd, vocab_size]
}

pub struct Block {
    pub ln_1: LayerNorm,       // before attention
    pub attn: MultiHeadAttention,
    pub ln_2: LayerNorm,       // before MLP
    pub mlp: MLP,
}

pub struct MultiHeadAttention {
    pub c_attn: Linear,        // [n_embd, 3*n_embd] - Q, K, V projection
    pub c_proj: Linear,        // [n_embd, n_embd] - output projection
    pub n_heads: usize,
    pub head_dim: usize,       // n_embd / n_heads
}

pub struct MLP {
    pub c_fc: Linear,          // [n_embd, 4*n_embd] - expansion
    pub c_proj: Linear,        // [4*n_embd, n_embd] - projection
}
```

### Initialization Scheme

Following GPT-2's initialization:

**Embeddings and Linear weights:** Normal distribution N(0, 0.02)
```rust
Tensor::randn(&[vocab_size, n_embd], 0.0, 0.02)
```

**Linear biases:** Zeros
```rust
Tensor::zeros(vec![out_features])
```

**Layer norm:** gamma=1, beta=0 (identity transformation initially)
```rust
gamma: Tensor::ones(vec![n_embd])
beta: Tensor::zeros(vec![n_embd])
```

The 0.02 standard deviation is small enough to prevent exploding activations at initialization but large enough to break symmetry between neurons.

### Attention Implementation Tricks

**Single projection for Q, K, V:**

Instead of three separate matrix multiplications:
```rust
// Naive approach (not what we do)
let q = input.matmul(&w_q);
let k = input.matmul(&w_k);
let v = input.matmul(&w_v);
```

We use one large matrix that produces all three at once:
```rust
// Actual implementation
let qkv = input.matmul(&c_attn.weight);  // [batch, seq, 3*n_embd]
// Split into Q, K, V
```

One matrix multiply is faster than three, even though it's 3x larger. Memory access patterns are better and there's less kernel launch overhead.

**Head dimension constraint:**

`n_heads` must evenly divide `n_embd`. The head dimension is:
```rust
head_dim = n_embd / n_heads
```

Common configurations use head_dim=64:
- n_embd=768, n_heads=12 → head_dim=64
- n_embd=256, n_heads=4 → head_dim=64

This isn't arbitrary. Attention scores are scaled by 1/√head_dim = 1/8, which keeps values in a numerically stable range.

**Reshaping for parallel heads:**

The implementation reshapes tensors to process all heads in parallel:
```rust
// [batch, seq, n_embd] → [batch, n_heads, seq, head_dim]
let q = q.reshape(&[batch, seq, n_heads, head_dim]).transpose(1, 2);
```

Each head then computes attention independently. After attention, we reshape back:
```rust
// [batch, n_heads, seq, head_dim] → [batch, seq, n_embd]
let out = out.transpose(1, 2).reshape(&[batch, seq, n_embd]);
```

### GELU Approximation

We use the tanh approximation for GELU activation:

```rust
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}
```

This matches GPT-2's implementation. The exact GELU uses the error function (erf), but the tanh approximation is faster and the difference is negligible for model quality.

### Memory Layout

All tensors use row-major (C-style) layout:
- Shape `[batch, seq, embd]`
- Last dimension is contiguous in memory
- Walking along the embedding dimension has perfect cache locality

This matters for performance. Matrix multiplications access the last dimension repeatedly, so having it contiguous means fewer cache misses.

## What We Didn't Implement

**Weight tying:** GPT-2 shares weights between token embeddings and the output projection (lm_head). This reduces parameters significantly (38M fewer for GPT-2 Small). We keep them separate for clarity. The math works the same either way.

**Dropout:** Regularization technique that randomly zeros activations during training. Not needed for forward-only inference. Will be added in Part 4.

**Flash Attention:** Memory-efficient attention that avoids materializing the full attention matrix. Our implementation creates the full [seq, seq] matrix, which limits sequence length. Flash attention would allow longer contexts but adds significant complexity.

**KV Cache:** During generation, we recompute attention for all previous tokens on every step. Caching key and value tensors would speed up generation significantly. Covered in Part 6.

## Common Issues

### Shape mismatch in attention

```
thread 'main' panicked at 'Attention dimension mismatch'
```

Check that n_heads divides n_embd evenly:
- n_embd=768, n_heads=12 → head_dim=64 ✓
- n_embd=768, n_heads=10 → head_dim=76.8 ✗

### Out of memory

```
memory allocation of ... bytes failed
```

Large models consume significant memory. GPT-2 Small needs ~650 MB for weights plus several GB for activations. Try smaller configurations:
- Reduce n_embd: 768 → 384 → 256
- Reduce n_layers: 12 → 6 → 4
- Reduce block_size: 1024 → 256 → 128
- Reduce vocab_size: 50257 → 512

### Slow forward passes

Make sure you're compiling with `--release`:
```bash
cargo run --release --example 03_model_architecture
```

Debug builds are 10-100x slower. If still slow, check your model size. GPT-2 Small (163M params) takes 2-3 seconds per forward pass on CPU. That's expected.

### NaN in output

With random initialization, very deep networks can produce NaN values due to numerical instability. This isn't a problem for the architecture demonstration. During training, proper gradient clipping prevents this.

If you see NaN consistently:
1. Check that layer norm epsilon is positive (typically 1e-5)
2. Verify softmax has the max-subtraction stability trick
3. Try smaller initialization (0.01 instead of 0.02)

## Benchmarking

Forward pass times on M1 MacBook Pro (8 performance cores), batch_size=1:

| Model | Parameters | seq_len=16 | seq_len=64 | seq_len=256 |
|-------|------------|------------|------------|-------------|
| Tiny | 0.8M | ~8ms | ~25ms | ~150ms |
| Small | 3.5M | ~15ms | ~50ms | ~300ms |
| Medium | 20-40M | ~80ms | ~200ms | ~1.2s |
| GPT-2 Small | 163M | ~400ms | ~1.5s | ~8s |

Times scale with:
- **Parameters:** More layers and larger n_embd means more computation
- **Sequence length:** Attention is O(n²), so doubling sequence length roughly quadruples attention time
- **Batch size:** Linear scaling (2x batch = 2x time)

Your results will vary based on CPU architecture and core count.

## Next Steps

After running the architecture example:

1. **Trace a forward pass** - Add print statements to see tensor shapes at each stage
2. **Experiment with configurations** - Try different n_embd, n_heads, n_layers combinations
3. **Profile memory usage** - Watch system memory while creating large models
4. **Break things intentionally** - Try invalid configurations to understand the error messages

Move on to Part 4: Training Infrastructure to learn how backpropagation and optimization turn random weights into a working language model.

## Further Reading

- **Attention Is All You Need** (Vaswani et al., 2017) - Original transformer paper: https://arxiv.org/abs/1706.03762

- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

- **The Illustrated Transformer** (Jay Alammar) - Visual guide with diagrams: https://jalammar.github.io/illustrated-transformer/

- **nanoGPT** (Andrej Karpathy) - Minimal PyTorch GPT that influenced this project: https://github.com/karpathy/nanoGPT
