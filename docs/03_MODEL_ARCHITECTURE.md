# Model Architecture: Building GPT-2 from Scratch

This document is a companion to the blog post "Building an LLM From Scratch in Rust, Part 3: Model Architecture". It covers implementation details and provides guidance for working with the model architecture code in this repository.

## What's Here

This phase implements a complete GPT-2 style transformer model with all core components. The model can perform **forward passes** (inference) but does not yet include training capabilities (backward pass, optimization).

By the end of this phase, you'll have a working GPT-2 Small architecture transformer (768 dimensions, 12 layers, 12 heads, 1024 token context) that can be trained on a CPU. With the full GPT-2 vocabulary (50,257 tokens), this reaches 163 million parameters, or 124M if we implemented weight tying. For demonstration purposes, we use a smaller vocabulary, resulting in 87M parameters. The weights are random, so predictions are meaningless until training.

**Implementation files:**
- [src/model.rs](../src/model.rs) - Core model components (Embedding, LayerNorm, MLP, Attention, Block, GPT)
- [examples/03_model_architecture.rs](../examples/03_model_architecture.rs) - Model creation and forward pass demonstration

## Running the Example

```bash
cargo run --release --example 03_model_architecture
```

This demonstrates model creation, parameter counting, and forward passes through the architecture.

## Expected Output

The example creates models of various sizes and shows parameter counts:

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

## Understanding the Architecture

### What is a Transformer?

The transformer is the architecture that powers modern language models. It processes all tokens in a sequence simultaneously using attention mechanisms, unlike RNNs which process sequentially. This parallelization makes training orders of magnitude faster and allows models to handle much longer context.

The architecture involves two key operations stacked in multiple layers:
1. **Attention mechanisms** - Let each token look at other tokens and decide which are relevant
2. **Feedforward networks** - Process what the attention mechanisms found

GPT-2 uses 12 of these layers stacked together. GPT-3 uses 96.

### The Forward Pass

The forward pass flows through several stages:

```text
Token IDs [batch, seq]
  ↓
Token Embeddings [batch, seq, n_embd]
  + Position Embeddings
  ↓
Transformer Block 1: Attention + MLP
  ↓
Transformer Block 2: Attention + MLP
  ↓
  ... (n_layers blocks)
  ↓
Transformer Block N: Attention + MLP
  ↓
Final LayerNorm
  ↓
Linear Projection → [batch, seq, vocab_size]
  ↓
Logits (scores for each token)
```

Each stage transforms the data. Token IDs become rich vector representations. These flow through transformer layers of attention and feedforward processing. The final layer maps back to vocabulary space, producing a score for every possible next token.

## Core Components

### 1. Embeddings: Token and Position

**Token Embeddings** convert discrete token IDs into continuous vectors.

```rust
pub struct Embedding {
    pub weight: Tensor,  // [vocab_size, n_embd]
}
```

For a vocabulary of 50,257 tokens with 768-dimensional embeddings, this matrix contains 38 million parameters. Each token ID gets its own row in this matrix. Looking up an embedding is just indexing into the matrix and copying those n_embd values.

**Position Embeddings** add positional information since transformers process all tokens in parallel and have no inherent sense of order.

```rust
pub wpe: Embedding,  // [block_size, n_embd]
```

Position 0, 1, 2, ... each get a learned vector. These are added element-wise to token embeddings. The block_size (1,024 for GPT-2) is the maximum sequence length, also known as the context window.

When processing a token at position i:
```text
Token embedding:    [0.1,  -0.3,  0.2,  ...] (n_embd values)
Position embedding: [0.05, -0.02, 0.01, ...] (n_embd values)
Combined:           [0.15, -0.32, 0.21, ...] (n_embd values)
```

The result captures both what the token is and where it appears in the sequence.

### 2. Layer Normalization

**Purpose:** Stabilize activations and improve training.

```rust
pub struct LayerNorm {
    pub gamma: Tensor,  // [n_embd] - scale parameter
    pub beta: Tensor,   // [n_embd] - shift parameter
    pub eps: f32,       // typically 1e-5
}
```

**Formula:**
```text
output = ((input - mean) / sqrt(variance + eps)) × gamma + beta
```

For each token in the sequence, we compute mean and variance across the n_embd dimensions, normalize to mean=0 and variance=1, then scale and shift with learned parameters.

Layer normalization happens twice per transformer block (before attention and before MLP) and once after all blocks (final layer norm).

**Why layer norm (not batch norm)?**
- Works with variable sequence lengths
- Normalizes per-sample, doesn't depend on batch statistics
- More stable for small batches
- Standard for transformers

**Why pre-norm?**

GPT-2 uses pre-norm (normalize before sublayers) rather than post-norm (normalize after). Pre-norm is more stable for very deep networks because it prevents the residual path from accumulating unnormalized values through many layers.

```text
Pre-norm:  x = x + Attention(LayerNorm(x))
Post-norm: x = LayerNorm(x + Attention(x))
```

### 3. Multi-Head Self-Attention

**Purpose:** Allow the model to focus on different parts of the input simultaneously.

Attention uses three projections of the input: queries (Q), keys (K), and values (V). Think of it like a database lookup. The query is what you're searching for. The keys are labels on every piece of information. The values are the actual information.

**Single-head attention process:**
```text
1. Project input to Q, K, V: [batch, seq, n_embd] → [batch, seq, head_dim]
2. Compute scores: Q @ K^T / sqrt(head_dim) → [batch, seq, seq]
3. Apply causal mask (set future positions to -inf)
4. Softmax to get attention weights → [batch, seq, seq]
5. Apply to values: attention_weights @ V → [batch, seq, head_dim]
```

The division by sqrt(head_dim) is a scaling factor. Without it, dot products grow large as dimensions increase, pushing softmax into regions where gradients are tiny.

**Multi-head attention** runs several attention operations in parallel. Each "head" has its own Q, K, V projections and learns to attend to different patterns.

```text
Example with n_embd=768, n_heads=12:
- Split into 12 heads of dimension 64 each
- Each head computes attention independently
- Concatenate: [head1, head2, ..., head12] → 768 dimensions
- Final projection back to 768 dimensions
```

One head might focus on syntactic relationships (nouns attending to verbs). Another might track thematic connections (food words attending to each other). A third might focus on local context (nearby words). The model learns what each head should focus on during training.

**Causal Masking:**

When predicting the next token, the model shouldn't peek at future tokens. We enforce this by setting attention scores for future positions to negative infinity before softmax:

```text
Attention mask for seq_len=4:
[✓ ✗ ✗ ✗]  position 0 can't see future
[✓ ✓ ✗ ✗]  position 1 can't see positions 2, 3
[✓ ✓ ✓ ✗]  position 2 can't see position 3
[✓ ✓ ✓ ✓]  position 3 can see all past
```

Since exp(-infinity) = 0, masked positions contribute nothing after softmax. This is called causal or autoregressive attention.

**Implementation:**
```rust
pub struct MultiHeadAttention {
    pub c_attn: Linear,    // [n_embd, 3*n_embd] - projects to Q, K, V all at once
    pub c_proj: Linear,    // [n_embd, n_embd] - output projection
    pub n_heads: usize,
    pub head_dim: usize,   // n_embd / n_heads
}
```

The implementation uses a clever trick: one large matrix produces all heads' Q, K, V at once, then we reshape to separate the heads. More efficient than 12 separate matrix multiplications.

### 4. MLP (Feedforward Network)

**Purpose:** Process each position independently after attention mixes information across positions.

```rust
pub struct MLP {
    pub c_fc: Linear,    // [n_embd, 4*n_embd] - expansion
    pub c_proj: Linear,  // [4*n_embd, n_embd] - projection
}
```

**Architecture:**
```text
Input [batch, seq, n_embd]
  ↓
Linear: n_embd → 4×n_embd (expansion)
  ↓
GELU activation
  ↓
Linear: 4×n_embd → n_embd (projection)
  ↓
Output [batch, seq, n_embd]
```

For GPT-2 Small with n_embd=768, the intermediate dimension is 3,072. Every token's 768 numbers get expanded to 3,072, transformed, and compressed back to 768.

**Why 4× expansion?**

The original "Attention is All You Need" paper used 4×, and it worked well, so it stuck. Provides substantial capacity without making the model impractically expensive.

**GELU Activation:**

GELU (Gaussian Error Linear Unit) is a smooth activation that works better than ReLU for transformers.

```text
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

This approximation is fast and accurate. For negative inputs, GELU outputs small negative values rather than zero (like ReLU). The smooth curve helps gradients flow during training.

### 5. Transformer Block

**Purpose:** Combine attention and MLP with residual connections.

```rust
pub struct Block {
    pub ln_1: LayerNorm,  // before attention
    pub attn: MultiHeadAttention,
    pub ln_2: LayerNorm,  // before MLP
    pub mlp: MLP,
}
```

**Architecture:**
```text
Input
  ↓
  ├─→ LayerNorm → Attention → + (residual)
  ↓                            ↓
  ├─→ LayerNorm → MLP ────────→ + (residual)
  ↓
Output
```

**Residual connections** are critical for training deep networks:
```text
output = input + sublayer(input)
```

Instead of replacing the input, we add the sublayer's output to the input. This creates a direct path for gradients to flow backward during training. Even if the gradient through the sublayer vanishes, the gradient through the residual connection remains. This "gradient highway" lets us stack 12, 96, or more layers.

### 6. Complete GPT Model

```rust
pub struct GPT {
    pub wte: Embedding,        // token embeddings [vocab_size, n_embd]
    pub wpe: Embedding,        // position embeddings [block_size, n_embd]
    pub blocks: Vec<Block>,    // n_layers transformer blocks
    pub ln_f: LayerNorm,       // final layer norm
    pub lm_head: Linear,       // output projection [n_embd, vocab_size]
}
```

The forward pass:
1. Look up token embeddings
2. Add position embeddings
3. Pass through all transformer blocks
4. Apply final layer normalization
5. Project to vocabulary space

The final output has shape `[batch, seq_len, vocab_size]`. Each position gets a score for every token in the vocabulary. These scores (logits) represent the model's belief about what token should come next.

## Model Configurations

Different configurations trade off between size, speed, and capacity:

### Tiny (for testing)
```rust
vocab_size: 512, n_embd: 128, n_heads: 4, n_layers: 3, block_size: 128
```
**~0.8M parameters** - Very fast, good for debugging

### Small (for experiments)
```rust
vocab_size: 512, n_embd: 256, n_heads: 4, n_layers: 4, block_size: 256
```
**~3.5M parameters** - Good balance for experimentation

### Medium
```rust
vocab_size: 512, n_embd: 384, n_heads: 6, n_layers: 6, block_size: 256
```
**~20-40M parameters** - Larger but manageable on CPU

### Large (GPT-2 Small)
```rust
vocab_size: 50257, n_embd: 768, n_heads: 12, n_layers: 12, block_size: 1024
```
**~163M parameters** - Full GPT-2 Small size (without weight tying)

## Parameter Count Formula

Understanding where parameters live helps estimate memory requirements:

```text
Embeddings:
  Token: vocab_size × n_embd
  Position: block_size × n_embd

Per Transformer Block:
  Attention:
    QKV projection: n_embd × (3 × n_embd)
    Output projection: n_embd × n_embd
    Layer norm: 2 × n_embd (gamma + beta)
  MLP:
    Expand: n_embd × (4 × n_embd)
    Project: (4 × n_embd) × n_embd
    Layer norm: 2 × n_embd

Final:
  Layer norm: n_embd (gamma + beta)
  LM head: n_embd × vocab_size

Total ≈ embeddings + (n_layers × per_layer) + final
```

**Example calculation for Small model** (n_embd=256, n_layers=4, vocab=512):
```text
Embeddings: 512×256 + 256×256 = 196,608
Per block:
  Attention: 256×768 + 256×256 + 512 = 263,168
  MLP: 256×1024 + 1024×256 + 512 = 525,312
  Total per block: ~788,480
4 blocks: 3,153,920
Final: 512 + 256×512 = 131,584
Total: ~3,482,112 parameters
```

At 4 bytes per float32, that's about 14 MB of memory just for weights. Add activations during the forward pass and you need significantly more. GPT-2 Small with 163M parameters needs about 650 MB just for weights.

## Implementation Details

### Memory Layout

All tensors use **row-major (C-style)** layout:
- Shape `[batch, seq, embd]`
- Data order: `[batch0_seq0_embd0, batch0_seq0_embd1, ..., batch0_seq1_embd0, ...]`

The last dimension is most tightly packed. Values next to each other in the last dimension sit next to each other in memory. Perfect for cache efficiency.

### Initialization

Following GPT-2 initialization scheme:

**Embeddings:** N(0, 0.02) - small random values
```rust
Tensor::randn(&[vocab_size, n_embd], 0.0, 0.02)
```

**Linear layers:** N(0, 0.02) for weights, zeros for biases
```rust
weight: Tensor::randn(&[out_features, in_features], 0.0, 0.02)
bias: Tensor::zeros(vec![out_features])
```

**Layer norm:** gamma=1, beta=0 (identity transformation initially)
```rust
gamma: Tensor::ones(vec![n_embd])
beta: Tensor::zeros(vec![n_embd])
```

### Attention Implementation

**Single projection for Q, K, V:**
- One Linear(n_embd, 3×n_embd) instead of three separate layers
- Split output into Q, K, V
- More efficient (one matrix multiply instead of three)

**Head splitting:**
- Reshape `[batch, seq, n_embd]` → `[batch, n_heads, seq, head_dim]`
- Each head operates independently
- Reshape back: `[batch, n_heads, seq, head_dim]` → `[batch, seq, n_embd]`

**Why n_heads must divide n_embd:**

head_dim = n_embd / n_heads must be an integer. Common configurations:
- n_embd=768, n_heads=12 → head_dim=64
- n_embd=1024, n_heads=16 → head_dim=64
- n_embd=256, n_heads=4 → head_dim=64

head_dim=64 is common because attention scores scale by 1/√head_dim = 1/8, a nice factor for numerical stability.

## What's Not Included (Yet)

This phase implements **forward pass only**. Training requires:

### Backward Pass (Backpropagation)
- Compute gradients for all parameters
- Chain rule through all layers
- Cache activations for gradient computation

### Optimizer (Adam)
- First moment (momentum)
- Second moment (RMSprop-style adaptive learning rates)
- Weight decay
- Learning rate schedule

### Training Loop
- Load batches of data
- Forward pass
- Loss computation (cross-entropy)
- Backward pass
- Optimizer step
- Gradient clipping (prevent exploding gradients)

### Additional Training Features
- Dropout (regularization)
- Gradient accumulation (simulate larger batches)
- Checkpointing (save/load models)
- Learning rate warmup
- Mixed precision training

These will be covered in Part 4: Training Infrastructure.

## Common Issues

### Shape mismatch in attention

```
thread 'main' panicked at 'Attention dimension mismatch'
```

Check that n_heads divides n_embd evenly. If n_embd=768 and n_heads=12, head_dim=64 works. If n_embd=768 and n_heads=10, head_dim=76.8 doesn't work.

### Out of memory

```
memory allocation of ... bytes failed
```

Large models consume significant memory. GPT-2 Small (163M parameters) needs ~650 MB just for weights, plus several GB for activations during forward pass. Try smaller configurations:
- Reduce n_embd (768 → 384 → 256)
- Reduce n_layers (12 → 6 → 4)
- Reduce block_size (1024 → 256 → 128)
- Reduce vocab_size (50257 → 512)

### Slow forward passes

Make sure you're compiling with `--release`:
```bash
cargo run --release --example 03_model_architecture
```

Debug builds are 10-100x slower than release builds.

### NaN in output

With random initialization, the model sometimes produces NaN (not a number) values, especially for very deep networks. This indicates numerical instability. Not a problem for this demonstration since we're just verifying the architecture works. During training, proper initialization and gradient clipping will prevent this.

## Benchmarking

Forward pass times on M1 MacBook Pro (8 performance cores):

- **Tiny model** (0.8M params): ~8ms for batch_size=1, seq_len=16
- **Small model** (3.5M params): ~35ms for batch_size=1, seq_len=32
- **Medium model** (20-40M params): ~200ms for batch_size=1, seq_len=64
- **GPT-2 Small** (163M params): ~2-3 seconds for batch_size=1, seq_len=128

Times scale roughly with:
- Number of parameters (more layers, larger n_embd)
- Sequence length (attention is O(n²) in sequence length)
- Batch size (linear scaling)

Your results will vary based on CPU architecture and core count.

## Key Design Decisions

### Why Multi-Head Attention?

**Single head:** Model can only attend one way

**Multiple heads:** Different heads learn different patterns
- Syntax head: noun-verb agreement
- Semantic head: related concepts
- Position head: nearby tokens
- Reference head: pronouns to referents

The model learns what each head should focus on. We don't specify it. The learning process naturally leads to specialization.

### Why Causal Masking?

Language modeling is **autoregressive:** predict token N+1 given tokens 0..N.

```text
Sentence: "The cat sat on the"
Position 0: [] → predict "The"
Position 1: ["The"] → predict "cat"
Position 2: ["The", "cat"] → predict "sat"
...
```

Position i uses tokens 0..i to predict token i+1, so it *cannot see* i+1..N. Masking enforces this.

### Why Residual Connections?

**Without residuals:** Deep networks have vanishing gradients

**With residuals:** Gradients can flow directly through skip connections

```text
output = input + sublayer(input)
gradient: ∂L/∂input = ∂L/∂output × (1 + ∂sublayer/∂input)
```

The "1 +" term ensures gradients flow even if sublayer gradients vanish. This is why we can stack 12, 96, or more layers.

### Why is Attention O(n²)?

For sequence length n:
- Q @ K^T creates [n, n] matrix (every position attends to every position)
- Storage: O(n²)
- Computation: O(n² × d) where d is head dimension

This quadratic scaling limits context length. GPT-2's 1,024 tokens is manageable. GPT-4's 128K context requires substantial memory. This is why there's active research into efficient attention mechanisms (sparse attention, linear attention, flash attention).

## Next Steps

After understanding the model architecture:

1. **Run the example** - See model creation and forward passes with different sizes
2. **Experiment with configurations** - Try different n_embd, n_heads, n_layers
3. **Trace a forward pass** - Follow data through embeddings → blocks → output
4. **Count parameters** - Understand where memory goes

To actually use the model:
- **Part 4: Training Infrastructure** - Implement backpropagation, optimizer, training loop
- **Part 6: Text Generation** - Sample from predictions to generate text

## Further Reading

- **Attention Is All You Need** (Vaswani et al., 2017) - Original transformer paper introducing multi-head attention and the architecture we're implementing

- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2 paper describing the architecture details and training approach

- **The Illustrated Transformer** (Jay Alammar) - Visual guide to transformer architecture with diagrams and animations

- **GPT-2 Source Code** (OpenAI) - Original TensorFlow implementation for reference: https://github.com/openai/gpt-2

- **nanoGPT** (Andrej Karpathy) - Minimal PyTorch implementation that heavily influenced this project: https://github.com/karpathy/nanoGPT
