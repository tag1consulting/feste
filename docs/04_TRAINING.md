# Training Infrastructure: Teaching the Model

This document is a companion to the blog post ["Building an LLM From Scratch in Rust, Part 4: Training Infrastructure"](https://www.tag1.com/how-to/part4-training-infrastructure-building-an-llm-from-scratch/). It covers implementation details and provides guidance for working with the training code in this repository.

## What's Here

Training turns random weights into a working language model. This phase implements backpropagation through every layer, the AdamW optimizer, gradient clipping, data loading, and training metrics.

**Implementation files:**
- [src/gpt2_trainable.rs](../src/gpt2_trainable.rs) - Trainable model with forward/backward passes
- [src/train.rs](../src/train.rs) - Data loading and training utilities
- [src/training_logger.rs](../src/training_logger.rs) - Metrics tracking and logging
- [examples/04_training_infrastructure.rs](../examples/04_training_infrastructure.rs) - Training infrastructure demonstration

## Running the Example

```bash
cargo run --release --example 04_training_infrastructure
```

This demonstrates the training infrastructure without running a full training loop. It shows data loading, batch creation, loss computation, and simulated training progress.

## Expected Output

```
======================================================================
  Training Infrastructure Demonstration
======================================================================

Training BPE tokenizer on Shakespeare...
  Initial vocab: 256 (byte-level)
  Target vocab: 512
  Training with 200 merges...
  ✓ Tokenizer trained with 512 tokens

Creating data loader...
  Total tokens: 338,025
  Sequence length: 64
  Batch size: 16
  Training examples: 5,281
  Validation examples: 587
  ✓ Data loader created

Computing baseline loss...
  Random predictions on vocab_size=512
  Expected loss: ~6.24 (ln(512))
  Actual loss: 6.2385
  Perplexity: 512.0
  ✓ Baseline computed correctly

Simulated training (showing metric evolution):
Step    0 | Time:     0.0s | LR: 0.000000 | Loss: 6.24 | Val: 6.24 | PPL: 512.0
Step  200 | Time:  5400.0s | LR: 0.000060 | Loss: 5.89 | Val: 5.91 | PPL: 369.4
Step  400 | Time: 10800.0s | LR: 0.000120 | Loss: 5.21 | Val: 5.28 | PPL: 196.8
Step  600 | Time: 16200.0s | LR: 0.000180 | Loss: 4.67 | Val: 4.79 | PPL: 120.3
Step  800 | Time: 21600.0s | LR: 0.000240 | Loss: 4.21 | Val: 4.38 | PPL:  80.1
Step 1000 | Time: 27000.0s | LR: 0.000300 | Loss: 3.82 | Val: 4.05 | PPL:  57.4

Training log saved to: data/training_20250101_120000/training_log.csv
✓ All infrastructure components working
```

Key indicators:
- Tokenizer trains successfully on Shakespeare text
- Data loader creates proper train/validation splits (90/10)
- Baseline loss matches theoretical value (ln(vocab_size) ≈ 6.24 for vocab_size=512)
- Simulated training shows realistic metric progression

## Understanding the Results

### Baseline Loss

With random weights, the model assigns roughly equal probability to all tokens. The expected loss is:

```
loss = -log(1/vocab_size) = log(vocab_size)
```

For vocab_size=512: `log(512) ≈ 6.24`

If your initial loss differs significantly from this, something is wrong with loss computation or model initialization.

### Training Progression

A healthy training run shows:
- **Loss decreasing** from baseline toward 2.0-4.0
- **Validation loss tracking training loss** (with a small gap)
- **Perplexity dropping** from vocab_size toward 20-100

Warning signs:
- Loss stuck at baseline → learning rate too low or gradients broken
- Loss exploding to NaN → learning rate too high
- Val loss increasing while train loss drops → overfitting

## Implementation Details

### Core Data Structures

```rust
pub struct TextDataLoader {
    tokens: Vec<usize>,           // Full tokenized corpus
    train_end: usize,             // Index where training data ends
    seq_len: usize,               // Sequence length per example
    batch_size: usize,            // Examples per batch
    current_pos: usize,           // Current position in corpus
    is_training: bool,            // Training or validation mode
}

pub struct TrainingLogger {
    file: BufWriter<File>,        // CSV output
    start_time: Instant,          // For elapsed time tracking
}

pub struct AdamWOptimizer {
    m: HashMap<String, Tensor>,   // First moment (momentum)
    v: HashMap<String, Tensor>,   // Second moment (adaptive LR)
    step: usize,                  // For bias correction
}
```

### Data Loading

The loader tokenizes text once upfront and generates batches on demand:

```rust
let mut loader = TextDataLoader::new(&text, &tokenizer, seq_len, batch_size);

while let Some((inputs, targets)) = loader.next_batch() {
    // inputs: [batch_size, seq_len] - token IDs
    // targets: [batch_size, seq_len] - next tokens (shifted by 1)
}
```

**Train/validation split:** First 90% is training, last 10% is validation. Call `loader.set_validation(true)` to switch modes.

**Sequence generation:** Uses sliding window with stride equal to seq_len (non-overlapping sequences). Each target is the input shifted by one position.

### Loss Computation

Cross-entropy with numerical stability:

```rust
pub fn compute_loss(&self, logits: &Tensor, targets: &[usize]) -> f32 {
    let mut total_loss = 0.0;

    for (i, &target) in targets.iter().enumerate() {
        let logits_slice = &logits.data[i * vocab_size..(i + 1) * vocab_size];

        // Max subtraction for stability (same trick as softmax)
        let max_logit = logits_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits_slice.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_prob = (logits_slice[target] - max_logit) - exp_sum.ln();

        total_loss -= log_prob;
    }

    total_loss / targets.len() as f32
}
```

The max subtraction prevents overflow when logits are large. Without it, `exp(300)` overflows to infinity.

### Gradient Clipping

Prevents exploding gradients by scaling all gradients when their total norm exceeds a threshold:

```rust
pub fn clip_gradients(grads: &mut GPT2Gradients, max_norm: f32) {
    let norm = compute_grad_norm(grads);

    if norm > max_norm {
        let scale = max_norm / norm;
        scale_all_gradients(grads, scale);
    }
}
```

Standard threshold is 1.0. Clipping preserves gradient direction while limiting magnitude.

### AdamW Update

```rust
pub fn adamw_step(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f32,
    beta1: f32,      // 0.9
    beta2: f32,      // 0.95
    eps: f32,        // 1e-8
    weight_decay: f32,  // 0.1
    step: usize,
) {
    // Update moments
    // m = beta1 * m + (1 - beta1) * grad
    // v = beta2 * v + (1 - beta2) * grad^2

    // Bias correction
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);

    // Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
    // Weight decay: param *= (1 - lr * weight_decay)
}
```

**Bias correction** is critical early in training. Without it, the first few updates are much smaller than they should be because m and v start at zero.

**Weight decay** is applied separately from gradient updates (the "W" in AdamW). Only applied to weight matrices, not biases or layer norm parameters.

### Caching for Backward Pass

The forward pass caches activations needed for gradient computation:

```rust
pub struct BlockCache {
    ln1_cache: LayerNormCache,    // Input to attention
    attn_cache: AttentionCache,   // Q, K, V, attention weights
    ln2_cache: LayerNormCache,    // Input to MLP
    mlp_cache: MLPCache,          // Intermediate activations
    residual1: Tensor,            // For residual gradient
    residual2: Tensor,            // For residual gradient
}
```

Each layer's cache stores what that layer needs to compute its gradients. Without caching, we'd need to recompute the forward pass during backprop.

### Training Metrics

The logger tracks:

| Metric | Meaning | Good Values |
|--------|---------|-------------|
| `train_loss` | Cross-entropy on training data | Decreasing toward 2-4 |
| `val_loss` | Cross-entropy on held-out data | Close to train_loss |
| `perplexity` | exp(loss) | Decreasing toward 20-100 |
| `learning_rate` | Current LR | Follows schedule |
| `grad_norm` | Gradient magnitude | Stable, not exploding |

**CSV format:**
```csv
step,elapsed_seconds,learning_rate,train_loss,val_loss,train_perplexity,val_perplexity,sample
0,62.69,0.000000,7.6516,7.7107,2104.09,2232.13,"begeown..."
200,5228.69,0.000030,6.6899,6.6699,804.22,788.31,"ssFfor y..."
```

### Checkpointing

Two checkpoint types:

**Inference-only** (smaller):
```rust
checkpoint.save_inference("model.bin")?;
// Contains: model weights, config
```

**Training** (larger):
```rust
checkpoint.save_training("checkpoint.bin")?;
// Contains: model weights, config, optimizer state (m, v), step count
```

Training checkpoints are ~3x larger because they include optimizer state. You need the optimizer state to resume training without losing momentum.

**Format:** Custom binary with header "FESTE_CKPT", version number, JSON config, then length-prefixed f32 arrays for all tensors.

## What We Didn't Implement

**Dropout:** Regularization that randomly zeros activations during training. Would reduce overfitting but adds complexity. GPT-2 used dropout=0.1.

**Mixed precision:** Using float16 for most operations. Significant speedup on GPUs with tensor cores, minimal benefit on CPU.

**Gradient accumulation:** Simulating larger batch sizes by accumulating gradients across multiple forward passes before updating. Useful when batch size is memory-limited.

**Learning rate schedules:** We use constant LR after warmup. Cosine annealing or other schedules can improve final performance.

**Distributed training:** Training across multiple machines. Our models are small enough for single-machine training.

## Common Issues

### Loss is NaN

```
Step  142 | Loss: NaN | Val: NaN
```

**Causes and solutions:**
1. Learning rate too high → reduce by 10x (try 3e-5)
2. Gradient explosion → enable clipping with max_norm=1.0
3. Bad data (NaN in inputs) → check tokenization
4. Softmax overflow → verify max subtraction is working

### Loss not decreasing

```
Step    0 | Loss: 6.24
Step  500 | Loss: 6.22
Step 1000 | Loss: 6.21
```

**Causes and solutions:**
1. Learning rate too low → increase by 2-3x
2. Gradients are zero → check backward pass implementation
3. Model too small → use more layers or wider embeddings
4. Data not shuffled → verify loader is working

Debug by printing gradient norm:
```rust
println!("Grad norm: {}", compute_grad_norm(&grads));
```

If grad norm is zero or tiny, backprop is broken.

### Training too slow

**Expected times per step** (M1 MacBook Pro, batch=8, seq_len=128):
- Tiny (0.8M): ~25ms
- Small (3.5M): ~90ms
- Medium (20M): ~450ms

If much slower:
1. Use `--release` flag (10-100x faster than debug)
2. Reduce batch_size or seq_len
3. Check Rayon is using all cores

### Out of memory

Memory usage scales with:
- `batch_size` (linear)
- `seq_len` (quadratic due to attention)
- `n_embd` and `n_layers` (model size)

Solutions:
1. Halve batch_size (cuts memory ~50%)
2. Reduce seq_len (256 → 128 cuts memory ~75%)
3. Use smaller model

### Overfitting

```
Step  800 | Train: 3.45 | Val: 4.23
Step 1000 | Train: 3.12 | Val: 4.38  <- val going up!
```

**Solutions:**
1. Stop earlier (use best validation checkpoint)
2. Use smaller model
3. Get more training data
4. Add dropout (requires code changes)

## Training Tips

### Hyperparameter Starting Points

| Model Size | Learning Rate | Batch Size | Seq Len |
|------------|---------------|------------|---------|
| < 1M params | 3e-4 | 16-32 | 64 |
| 1-10M params | 3e-4 | 8-16 | 128 |
| 10-50M params | 1e-4 | 4-8 | 128-256 |
| > 50M params | 6e-5 | 2-4 | 256 |

### Learning Rate Tuning

Start with 3e-4. If:
- Loss explodes → reduce by 10x
- Loss barely moves → increase by 2-3x
- Loss decreases then plateaus → try warmup

### When to Stop

Stop when validation loss stops improving for several hundred steps. Use the checkpoint with best validation loss, not the final one.

### Monitoring

Check every 50-100 steps:
- Is train loss decreasing?
- Is val loss tracking train loss?
- Is grad norm stable (not exploding)?

Generate sample text every few hundred steps to see qualitative progress.

## Benchmarking

Per-step times on M1 MacBook Pro (8 cores):

| Model | Params | batch=16, seq=64 | batch=8, seq=128 | batch=4, seq=256 |
|-------|--------|------------------|------------------|------------------|
| Tiny | 0.8M | ~25ms | ~45ms | ~120ms |
| Small | 3.5M | ~50ms | ~90ms | ~250ms |
| Medium | 20M | ~200ms | ~450ms | ~1.2s |

**Scaling:**
- Batch size: linear (2x batch ≈ 2x time)
- Sequence length: quadratic (2x seq ≈ 4x time for attention)
- Parameters: roughly linear

## Next Steps

After running the infrastructure example:

1. **Verify gradients** - Print grad norms to confirm backprop works
2. **Test with tiny model** - Quick iterations to validate the loop
3. **Watch metrics** - Understand what healthy training looks like
4. **Try breaking things** - Set LR too high, see what happens

Move on to Part 5: Training Examples to train actual models at multiple scales and watch them learn to generate text.

## Further Reading

- **AdamW Paper**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019) - https://arxiv.org/abs/1711.05101

- **Original Adam**: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014) - https://arxiv.org/abs/1412.6980

- **Yes you should understand backprop** (Karpathy) - https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b

- **Layer Normalization** (Ba et al., 2016) - https://arxiv.org/abs/1607.06450
