# Tensor Operations: The Foundation of Neural Networks

This document is a companion to the blog post "Building an LLM From Scratch in Rust, Part 2: Tensor Operations". It covers implementation details and provides guidance for working with the tensor code in this repository.

## What's Here

Neural networks manipulate data as tensors. Every operation in a language model reduces to tensor operations. Attention is matrix multiplication. Layer normalization is statistics on tensor dimensions. Token embeddings are table lookups that produce tensors.

This implementation builds the tensor library that powers everything else in Feste.

**Implementation files:**
- [src/tensor.rs](../src/tensor.rs) - Core tensor implementation
- [examples/02_tensor_operations.rs](../examples/02_tensor_operations.rs) - Operations demonstration

## Running the Example

```bash
cargo run --release --example 02_tensor_operations
```

This demonstrates every operation we've implemented. Creating tensors with different patterns. Matrix multiplication with both simple and optimized parallel versions. Element-wise operations. Broadcasting. Softmax with numerical stability. Reshaping and transposing. Statistical operations for normalization. Masked fill for attention.

## Expected Output

The example runs in under a second on modern hardware:

```
======================================================================
  Tensor Operations Demonstration
======================================================================

Creating tensors...
✓ Created 2×3 matrix from data
✓ Created 3×4 zeros tensor
✓ Created range tensor [0.0, 1.0, ..., 9.0]

Matrix multiplication (small - sequential)...
  2×3 @ 3×2 = 2×2
  Time: 0.001ms

Matrix multiplication (large - parallel cache-blocked)...
  64×64 @ 64×64 = 64×64
  Time: 0.843ms

Element-wise operations...
  Addition: [2, 2] + [2, 2] = [2, 2]
  Scalar multiplication: [2, 2] * 2.0
  Time: 0.002ms

Softmax (numerical stability test)...
  Large logits: [100.0, 200.0, 300.0]
  Softmax (stable): [0.0, 0.0, 1.0]
  Sum: 1.000000 ✓
  Time: 0.004ms

Broadcasting...
  [2, 3] + [3] (bias addition)
  Result shape: [2, 3]
  Time: 0.003ms

Masked fill (causal masking)...
  Applied causal mask to 2×2 scores
  Future positions set to -inf
  Time: 0.002ms

All operations completed successfully!
```

The key performance indicator is the 64×64 matrix multiplication. On an M1 MacBook Pro, that completes in under a millisecond. The cache-blocked parallelization is working.

The softmax test with large logits `[100.0, 200.0, 300.0]` produces valid probabilities that sum to 1.0. Without the max subtraction trick, `exp(300.0)` would overflow to infinity and break everything. The numerical stability works.

## Understanding the Operations

### Creating Tensors

Three basic ways to make tensors:

```rust
// From explicit data and shape
let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

// Filled with zeros
let zeros = Tensor::zeros(vec![3, 4]);

// Sequential numbers
let range = Tensor::arange(0, 10);
```

The `new()` constructor verifies that the data size matches the shape. Four values with shape `[2, 3]` would panic. Better to fail immediately than corrupt data silently.

Zeros are common for initialization. Many tensors start at zero and get filled during computation.

Sequential numbers are useful for testing. When you want to verify an operation is working correctly, having predictable input values makes debugging easier.

### Matrix Multiplication

The single most important operation in neural networks. Every major transformer operation uses matrix multiplication.

Linear layers multiply input by weight matrices. Attention computes `Q @ K^T` to get scores, then multiplies those scores by values. Feedforward layers are matrix multiplications with nonlinearity in between.

Our implementation automatically chooses between two versions based on workload size.

**Sequential version** for small matrices where `m * n * k < 1000`:
```rust
for i in 0..m {
    for j in 0..n {
        let mut sum = 0.0;
        for l in 0..k {
            sum += a[i * k + l] * b[l * n + j];
        }
        result[i * n + j] = sum;
    }
}
```

Three nested loops. Outer loop picks a row from the first matrix. Middle loop picks a column from the second matrix. Inner loop does the multiply-and-add across all elements. For tiny matrices, this is faster than spawning threads.

**Parallel cache-blocked version** for large matrices:

Divides work into 8×8 tiles that fit in L1 cache. Each tile is 64 floats or 256 bytes. L1 cache is typically 32-64KB per core, so tiles fit comfortably with room for multiple tiles at once.

Distributes row blocks across CPU cores using Rayon. On an 8-core machine, Rayon might spawn 8 threads and give each thread different rows to compute. One thread gets rows 0-7, another gets rows 8-15, and so on. No locks or synchronization needed because each thread writes to different memory locations.

The inner loop is ordered to access memory sequentially:
```rust
for i in block_start..block_end {
    for k in k_block_start..k_block_end {
        let a_val = A[i, k];
        for j in j_block_start..j_block_end {
            C[i, j] += a_val * B[k, j];
        }
    }
}
```

Walking along rows means the CPU loads a cache line and uses all of it before moving to the next. Cache hit rate goes way up.

Typical speedup is 2-4x on multi-core CPUs. Not 8x on an 8-core machine because memory bandwidth becomes the bottleneck. All cores compete for RAM access, and memory can only deliver data so fast.

### Batched Matrix Multiplication

Sometimes we need many matrix multiplications at once, all with the same structure but different data. Processing multiple sentences simultaneously, where each sentence needs the same calculations applied independently.

This creates 4D tensors where we parallelize across the independent calculations:
```rust
result.par_chunks_mut(seq1 * seq2)
    .enumerate()
    .for_each(|(bh_idx, chunk)| {
        let b = bh_idx / n_heads;
        let h = bh_idx % n_heads;
        // Compute 2D matmul for this batch/head
    }
);
```

When we build the transformer in Part 3, this gets used for attention heads. Each attention head looks at the input differently, focusing on different patterns. They all run the same matrix multiplication operations, just with different learned weights. Processing them in parallel is a natural fit.

### Element-wise Operations

Operations that work position by position. Add two tensors by taking position [0,0] from the first, adding it to position [0,0] from the second. Then position [0,1] plus position [0,1]. And so on through the entire tensor.

```rust
pub fn add(&self, other: &Tensor) -> Tensor {
    if self.shape == other.shape {
        let result = self.data.par_iter()
                              .zip(&other.data)
                              .map(|(a, b)| a + b)
                              .collect();
        return Tensor::new(result, self.shape.clone());
    }
    // Broadcasting cases...
}
```

The `par_iter()` tells Rayon to process elements in parallel. It splits the data across threads, each thread processes its chunk, and the results get collected into a new vector.

Multiplication, subtraction, and division follow the same pattern.

Scalar operations multiply every element by the same value:
```rust
pub fn mul_scalar(&self, scalar: f32) -> Tensor {
    let result = self.data.par_iter().map(|&x| x * scalar).collect();
    Tensor::new(result, self.shape.clone())
}
```

For large tensors with thousands or millions of elements, parallelism provides significant speedup. For tiny tensors with dozens of elements, the threading overhead costs more than it saves. Rayon looks at the workload size and decides whether spawning threads is worthwhile or if sequential processing would be faster.

### Softmax and Numerical Stability

Softmax converts scores into percentages that sum to 100%. The model produces raw scores for every word in the vocabulary. Softmax turns those into probabilities showing how confident the model is about each choice.

The math uses the exponential function:
```rust
let exp_values: Vec<f32> = row.iter().map(|&x| x.exp()).collect();
let sum: f32 = exp_values.iter().sum();
let normalized: Vec<f32> = exp_values.iter().map(|&x| x / sum).collect();
```

This breaks when scores get large. `exp(300.0)` overflows to infinity. Language models regularly produce scores in the hundreds. Direct implementation would overflow immediately.

The fix is subtracting the maximum score first:
```rust
let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
let exp_values: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
let sum: f32 = exp_values.iter().sum();
let normalized: Vec<f32> = exp_values.iter().map(|&x| x / sum).collect();
```

Even if original scores were huge like `[302.0, 301.0, 300.1]`, after subtracting the maximum (302.0) we'd have `[0.0, -1.0, -1.9]`. Small numbers that won't overflow. The exponentials work fine, and we get the same relative percentages.

This works even with big spreads. Scores like `[302.0, 301.0, 0.1]` become `[0.0, -1.0, -301.9]` after subtracting the max. That large negative number is fine because `exp(-301.9)` equals approximately zero, which computers can represent just fine. Large positive exponentials cause overflow, but large negative exponentials just approach zero.

Subtracting the max ensures the largest exponential is always `exp(0) = 1.0`, and all others are smaller. No overflow possible. Mathematically equivalent, numerically stable.

We compute softmax per row in parallel since each row is independent. Rayon distributes the work across cores automatically.

### Broadcasting

When you add two tensors together, they normally need the same shape. A 2×3 matrix plus another 2×3 matrix makes sense. But what if you want to add a single row of values to every row in a matrix?

Say you have a 2×3 matrix:
```
[1.0  2.0  3.0]
[4.0  5.0  6.0]
```

And you want to add the values `[0.1, 0.2, 0.3]` to every row. These adjustment values (called biases in neural networks) shift the results slightly. Without broadcasting, you'd need to manually duplicate these values into a full 2×3 matrix, then add them. Wastes memory storing duplicates and wastes time copying data around.

Broadcasting does this automatically:
```rust
let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
let bias = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);
let result = matrix.add(&bias);
```

The result is `[2, 3]` with the bias added to each row. Efficient in both memory and speed.

This pattern appears constantly in neural networks. When a layer transforms its input, it multiplies by a weight matrix and then adds a bias. The bias is one value per output dimension, but you're processing many inputs at once. Broadcasting adds it efficiently without duplicating.

We also handle batch broadcasting. When processing multiple examples at once, you often want to apply the same operation to each one. A `[4, 2, 3]` tensor represents 4 batch items, each with shape `[2, 3]`. Broadcasting lets you add a single `[2, 3]` tensor to all 4 batch items.

Tensor libraries like NumPy support broadcasting across arbitrary dimension combinations with complex rules. Feste implements only what transformers actually use. Adding vectors to matrix rows, and applying operations across batches. Those two patterns cover everything the model needs. Simpler code, fewer edge cases, easier to understand.

### Reshaping and Transposing

Reshaping reinterprets the same data with different dimensions. The actual numbers don't move or change, you just tell the tensor to organize them differently.

Take six values stored as `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`. You could interpret this as a 2×3 matrix or a 3×2 matrix. Same six numbers in the same order in memory. Just different shapes.

```rust
let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
let reshaped = matrix.reshape(&[3, 2]);
```

The total number of elements must stay the same. You can't reshape 6 values into a 2×4 matrix because that would need 8 values.

The model uses reshape when it needs data in a different arrangement for an operation. Flatten a multi-dimensional tensor into a vector before a linear layer. Split or merge dimensions for attention heads.

Transposing is different. It actually rearranges the data:
```rust
let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let transposed = matrix.transpose(0, 1);
```

For a 2×2 matrix, transpose swaps rows and columns. The data actually moves. Position [0,1] becomes position [1,0]. Standard matrix transpose from linear algebra.

For higher dimensional tensors, transpose can swap any two dimensions. You specify which two dimensions to swap and it rearranges the data accordingly. Transformers use this when computing attention. The details of how and why will become clear in Part 3 when we build the model architecture.

### Statistical Operations

Neural networks work better when data stays in a reasonable numeric range. If values get too large or too small, training becomes unstable. Layer normalization keeps values centered and scaled appropriately.

The math requires computing the mean (average) and variance (spread) of values. For a row of numbers `[1.0, 2.0, 3.0, 4.0, 5.0]`, the mean is 3.0. The variance measures how spread out the values are from that average.

We compute these statistics along specific dimensions of a tensor. For a 2×3 matrix, you might want the mean of each row separately:
```rust
let mean = tensor.mean(-1, true);
```

The `-1` tells it which direction to compute. For a 2×3 matrix, `-1` means "work across each row." So we get one mean per row. If we used `0` instead, it would work down each column, giving us one mean per column.

The `true` tells it to keep the result as a matrix instead of flattening to a simple list. Getting `[[2.0], [5.0]]` (a 2×1 matrix) instead of `[2.0, 5.0]` (a flat list) makes broadcasting work automatically when we use these means in the next calculation.

Variance works the same way:
```rust
let var = tensor.var(-1, true);
```

The implementation splits the tensor into slices along the target dimension, computes statistics for each slice in parallel, then collects the results.

Layer normalization uses these statistics to standardize values:
```rust
let mean = x.mean(-1, true);
let var = x.var(-1, true);
let normalized = x.sub(&mean).div(&var.add_scalar(eps).sqrt());
let scaled = normalized.mul(&gamma).add(&beta);
```

First subtract the mean from every value in each row. Then divide by the standard deviation (square root of variance). We add a tiny value (epsilon) to the variance before taking the square root so we never divide by zero if all values happen to be identical. Typically something like 0.00001.

At this point all values are standardized, but forcing everything to have mean 0 and variance 1 might be too restrictive. So we scale by multiplying with `gamma` (one learned value per dimension) and shift by adding `beta` (another learned value per dimension). These are parameters the model learns during training, letting it decide how much normalization is actually helpful.

The details of why this helps training will make more sense in Part 3 when we build the transformer layers. For now, know that we can compute statistics along tensor dimensions and use them to normalize data.

### Masked Fill

When predicting the next token, the model shouldn't be able to peek at future tokens. That would be cheating. If the model is trying to predict what comes after "To be or", it should only see those three tokens, not the rest of the sentence.

Feste enforces this with masking:
```rust
let scores = query.matmul(&key_transposed);
let masked = scores.masked_fill(&mask, f32::NEG_INFINITY);
let attention = masked.softmax(-1);
```

The mask is a grid of 0s and 1s. Where the mask is 1, we replace that score with negative infinity. Remember from softmax that `exp(-infinity) = 0`. So after softmax, those positions contribute nothing. The model effectively can't see them.

For predicting the next token, we use a causal mask (also called an upper triangular mask):
```
[0 1 1]
[0 0 1]
[0 0 0]
```

Position 0 can only look at itself (everything else is masked with 1s). Position 1 can look at positions 0 and 1 (position 2 is masked). Position 2 can look at all previous positions (nothing is masked). Each position only sees tokens that came before it, never tokens that come after.

The details of how attention works and why masking matters will become clear in Part 3.

## Implementation Details

### Core Data Structure

```rust
pub struct Tensor {
    pub data: Vec<f32>,      // Flat array of values
    pub shape: Vec<usize>,   // Dimensions like [2, 3, 4]
    pub strides: Vec<usize>, // Step sizes for indexing
}
```

All tensor data is stored as a single flat list of numbers, regardless of how many dimensions the tensor has. Even though you think of a matrix as having rows and columns, the computer stores it as one long sequence of numbers.

The shape tells you how to interpret those flat data values. The shape `[2, 3]` means "two rows, three columns."

The strides tell us how to navigate this flat array as if it were multi-dimensional. For shape `[2, 3]`, strides are `[3, 1]`. Moving one step in the first dimension (rows) advances 3 positions in the flat array. Moving one step in the second dimension (columns) advances 1 position.

This layout (called row-major) means the last dimension is most tightly packed. Values next to each other in the last dimension sit next to each other in memory. When the CPU loads data, it grabs chunks (cache lines) of nearby memory. If you're accessing elements sequentially in the last dimension, every value you need is already loaded. Perfect cache efficiency.

### Performance Thresholds

```rust
const WORK_THRESHOLD: usize = 1_000;  // m * n * k
const BLOCK_SIZE: usize = 8;          // 8×8 tiles
```

**Work threshold** determines when to use parallel matrix multiplication. Below 1,000 multiply-add operations, sequential is faster (avoid thread overhead). Above that, parallel provides speedup. Tuned empirically for typical transformer workloads.

**Block size** determines cache tile dimensions. 8×8 tiles are 64 floats or 256 bytes. Fits comfortably in L1 cache (typically 32-64KB). Provides good parallelism for typical matrix sizes. A 128×128 matrix has 256 blocks, enough to keep multiple cores busy.

### Parallel Strategy

```rust
result.par_chunks_mut(BLOCK_SIZE * n)
      .enumerate()
      .for_each(|(block_i, result_block)| {
          // Each thread processes BLOCK_SIZE rows independently
      });
```

Parallelizing over row blocks means each thread works on different rows. They never write to the same memory locations. No locks or synchronization needed. This is why matrix multiplication parallelizes well.

Good load balancing because all threads get equal work. Cache-friendly because each thread works on contiguous memory.

## What We Haven't Implemented

Our tensor library works. It handles everything GPT-2 needs. But production libraries like PyTorch do much more.

**Memory management.** Feste allocates a new vector for every operation. Production libraries reuse buffers. They maintain memory pools and carefully recycle allocations to avoid the overhead of constantly asking the operating system for memory. Complex bookkeeping involving lifetime tracking and buffer management.

**GPU support.** Feste runs on CPU cores. Modern training uses GPUs with thousands of cores working in parallel. But GPU programming is a different world. CUDA kernels, memory coalescing, warp-level parallelism. Adding GPU support would triple the codebase size and require understanding an entirely different execution model.

**Automatic differentiation.** Feste's tensors don't track gradients. Training requires computing derivatives so the model knows how to adjust its weights. Part 4 will add this, but it's a substantial feature requiring computation graphs and backpropagation through every operation.

**More operations.** Feste implements what GPT-2 needs. Production libraries support hundreds of operations for different model architectures. Convolutions for computer vision. Recurrent operations for sequence models. Specialized activations. Each operation multiplies the testing surface and adds complexity. Feste implements the essential operations and nothing more.

**SIMD vectorization.** Modern CPUs have SIMD (Single Instruction Multiple Data) instructions that operate on multiple values simultaneously. The Rust compiler auto-vectorizes some of our loops, but hand-written SIMD using intrinsics would be faster. Libraries like Intel MKL are heavily optimized with hand-tuned assembly for specific CPU architectures. Specialist optimization work that requires deep CPU architecture knowledge.

**Sparse tensors.** Dense tensors store every element even if most are zero. Sparse tensors store only non-zero values with their positions, saving massive amounts of memory for mostly-empty data. Requires different algorithms and careful index management. Feste doesn't need it for GPT-2.

Feste chooses simplicity over performance. The operations work correctly and run fast enough to train small models in reasonable time. You can experiment, understand what's happening, and see results. Not fast enough to train large production models, but that's not the goal. We're building Feste to learn how transformers work from the ground up.

## Common Issues

### Shape mismatch errors

```
thread 'main' panicked at 'Matrix multiplication dimension mismatch'
```

Check that your matrix dimensions are compatible. For `A @ B`, the number of columns in A must equal the number of rows in B. A `[2, 3]` matrix can multiply a `[3, 4]` matrix (result is `[2, 4]`), but not a `[2, 3]` matrix.

### Reshape errors

```
thread 'main' panicked at 'Cannot reshape: total elements do not match'
```

The total number of elements must stay the same when reshaping. You can reshape 6 elements as `[2, 3]` or `[3, 2]` or `[6]`, but not `[2, 4]` (needs 8 elements).

### Broadcasting errors

```
thread 'main' panicked at 'Broadcasting not supported for these shapes'
```

Broadcasting has specific rules. Feste implements only what transformers actually use. Adding a `[3]` vector to a `[2, 3]` matrix works. Adding a `[2]` vector to a `[2, 3]` matrix doesn't.

### Slow performance

Make sure you're compiling with `--release`. Debug builds are much slower:
```bash
cargo run --release --example 02_tensor_operations
```

For large matrix multiplications (64×64 and above), the parallel version should be significantly faster than sequential. If not, check that Rayon is properly detecting your CPU cores.

## Benchmarking

The example includes timing for major operations. On an M1 MacBook Pro (8 performance cores):

- 64×64 matrix multiplication: ~0.8ms (parallel)
- Element-wise operations on 10K elements: ~0.01ms
- Softmax on 1K values: ~0.005ms
- Broadcasting (2×3 + 3): ~0.003ms

Your results will vary based on CPU architecture and core count. The important thing is seeing the parallel version of matrix multiplication outperform the sequential version for large matrices.

## Next Steps

After understanding tensor operations, move on to Chapter 3: Transformer Architecture. We'll use these operations to build actual neural network components.

Token embeddings that convert token IDs into vectors. Position embeddings that encode where each token sits in the sequence. Multi-head self-attention using the matrix multiplication and softmax we built here. Feedforward layers. Layer normalization using the mean and variance operations. Residual connections.

The architecture will be complete. A working forward pass that takes token IDs as input and produces logits as output. Feed it "To be or not" and it produces scores for every token in the vocabulary. Wrong scores from untrained weights, but the right structure to eventually learn the right scores.

## Further Reading

- **Matrix Multiplication Optimization**: [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm) - Theoretical foundation for cache-blocking techniques. Explains why processing data in tiles improves cache performance.

- **Numerical Stability**: [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) - Classic paper on floating-point computation. Explains why operations like softmax need careful handling.

- **Broadcasting Semantics**: [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - Comprehensive guide to broadcasting rules. Our implementation uses a subset of these patterns.

- **Rayon Parallelism**: [Rayon Documentation](https://docs.rs/rayon/latest/rayon/) - Details on how Rayon distributes work across CPU cores. Understanding work-stealing and parallel iterators.
