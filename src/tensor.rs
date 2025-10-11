//! Tensor Operations for Neural Networks
//!
//! This module provides a minimal tensor library optimized for transformer models.
//! Tensors store multi-dimensional arrays with shape and stride information for
//! efficient indexing and memory layout.
//!
//! ## Core Concepts
//!
//! - **Data**: Flat `Vec<f32>` storing all elements in row-major order
//! - **Shape**: Dimensions of the tensor (e.g., `[batch, seq, dim]`)
//! - **Strides**: Step sizes for each dimension to compute flat indices
//!
//! ## Example
//!
//! ```rust
//! use feste::Tensor;
//!
//! // Create a 2x3 matrix
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::new(data, vec![2, 3]);
//!
//! // Matrix multiplication
//! let other = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
//! let result = tensor.matmul(&other);
//! assert_eq!(result.shape, vec![2, 2]);
//! ```
//!
//! ## Performance Optimizations
//!
//! Several operations use parallel processing via Rayon for performance:
//!
//! - **Matrix multiplication**: Cache-blocked algorithm with parallel row processing
//! - **Element-wise operations**: Parallel iteration over data
//! - **Softmax**: Parallel computation per row
//!
//! These optimizations provide 2-4x speedup on multi-core CPUs while keeping
//! the code educational and understandable. All optimizations are clearly marked
//! in the code with comments explaining the approach.

use rayon::prelude::*;

/// A multi-dimensional array for neural network computations
///
/// Tensors store data in a contiguous `Vec<f32>` with shape and stride information
/// for efficient multi-dimensional indexing. All operations use row-major (C-style)
/// memory layout.
///
/// # Fields
///
/// - `data`: Flat array of f32 values
/// - `shape`: Dimensions (e.g., `[2, 3]` for a 2x3 matrix)
/// - `strides`: Step sizes for each dimension (computed from shape)
///
/// # Memory Layout
///
/// For shape `[2, 3]`, data is stored as: `[row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]`
///
/// Strides would be `[3, 1]` meaning:
/// - Moving one step in dimension 0 (rows) advances 3 positions in data
/// - Moving one step in dimension 1 (cols) advances 1 position in data
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Flat storage of all tensor elements
    pub data: Vec<f32>,
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Strides for each dimension (computed from shape)
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with given data and shape
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of values
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Panics
    ///
    /// Panics if the product of shape dimensions doesn't equal data length
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// assert_eq!(tensor.shape, vec![2, 2]);
    /// ```
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_size,
            "Data length ({}) doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_size
        );

        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::zeros(vec![3, 4]);
    /// assert_eq!(tensor.data.len(), 12);
    /// assert!(tensor.data.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Self::new(data, shape)
    }

    /// Compute strides from shape (row-major layout)
    ///
    /// For shape `[d0, d1, d2]`, strides are `[d1*d2, d2, 1]`
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor dimensions
    ///
    /// # Returns
    ///
    /// Vector of stride values for each dimension
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Matrix multiplication
    ///
    /// Supports:
    /// - 2D × 2D: Standard matrix multiplication
    /// - 4D × 4D: Batched matmul for attention (see below)
    ///
    /// # 2D Matrix Multiplication
    ///
    /// For `A @ B` where `A` is `[m, k]` and `B` is `[k, n]`:
    /// - Result shape: `[m, n]`
    /// - Each element `C[i,j] = sum(A[i,k] * B[k,j])` for all k
    ///
    /// # Performance
    ///
    /// - **Small matrices** (< 1K ops): Sequential computation
    /// - **Large matrices** (≥ 1K ops): Parallel cache-blocked algorithm
    ///
    /// The parallel version uses 8×8 blocks for cache efficiency and
    /// parallelizes across output rows, providing 2-4x speedup on typical
    /// multi-core CPUs.
    ///
    /// # 4D Batched Matrix Multiplication
    ///
    /// For attention computations with shape `[batch, n_heads, seq, head_dim]`
    /// Processes each (batch, head) pair independently in parallel.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are incompatible or unsupported
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.shape, vec![2, 2]);
    /// ```
    ///
    /// SIMD-optimized inner loop for matrix multiplication
    /// Computes: result[j] += a_val * b[j] for all j
    /// Structured for auto-vectorization on ARM NEON (Apple Silicon)
    #[inline(always)]
    fn matmul_inner_simd(a_val: f32, b: &[f32], result: &mut [f32]) {
        // Simple loop that LLVM can auto-vectorize
        // On Apple Silicon this will use ARM NEON SIMD instructions
        for (r, &b_val) in result.iter_mut().zip(b.iter()) {
            *r += a_val * b_val;
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // === 2D MATRIX MULTIPLICATION ===
        if self.shape.len() == 2 && other.shape.len() == 2 {
            assert_eq!(
                self.shape[1], other.shape[0],
                "Matrix dimensions incompatible: [{}, {}] @ [{}, {}]",
                self.shape[0], self.shape[1], other.shape[0], other.shape[1]
            );

            let m = self.shape[0];
            let n = other.shape[1];
            let k = self.shape[1];

            // Use parallel version for larger matrices (work threshold: 1000 operations)
            // This threshold balances parallel overhead against performance gains
            if m * n * k >= 1_000 {
                return self.matmul_parallel_blocked(other, m, n, k);
            }

            // Sequential version for small matrices (avoids parallel overhead)
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.data[i * k + l] * other.data[l * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }

            return Tensor::new(result, vec![m, n]);
        }

        // === 4D BATCHED MATRIX MULTIPLICATION (for attention) ===
        // Shape: [batch, n_heads, seq, head_dim] @ [batch, n_heads, head_dim, seq]
        if self.shape.len() == 4 && other.shape.len() == 4 {
            let batch = self.shape[0];
            let n_heads = self.shape[1];
            let seq1 = self.shape[2];
            let inner_dim = self.shape[3];
            let seq2 = other.shape[3];

            assert_eq!(
                other.shape[2], inner_dim,
                "Inner dimensions must match for batched matmul"
            );

            let total_size = batch * n_heads * seq1 * seq2;
            let mut result = vec![0.0; total_size];

            // Parallelize over (batch, head) combinations
            // Each (batch, head) pair computes an independent seq1×seq2 matrix multiplication
            result
                .par_chunks_mut(seq1 * seq2)
                .enumerate()
                .for_each(|(bh_idx, chunk)| {
                    let b = bh_idx / n_heads;
                    let h = bh_idx % n_heads;

                    // Compute 2D matmul for this batch/head
                    for i in 0..seq1 {
                        for j in 0..seq2 {
                            let mut sum = 0.0;
                            for l in 0..inner_dim {
                                let self_idx =
                                    ((b * n_heads + h) * seq1 + i) * inner_dim + l;
                                let other_idx =
                                    ((b * n_heads + h) * inner_dim + l) * seq2 + j;
                                sum += self.data[self_idx] * other.data[other_idx];
                            }
                            chunk[i * seq2 + j] = sum;
                        }
                    }
                });

            return Tensor::new(result, vec![batch, n_heads, seq1, seq2]);
        }

        panic!(
            "Unsupported matmul shapes: {:?} @ {:?}",
            self.shape, other.shape
        );
    }

    /// Parallel cache-blocked matrix multiplication
    ///
    /// This optimization provides significant speedup for large matrices through:
    ///
    /// 1. **Cache blocking**: Processes data in 8×8 blocks that fit in L1 cache
    /// 2. **Parallel processing**: Distributes row blocks across CPU cores via Rayon
    /// 3. **Memory locality**: Inner loops access memory sequentially
    ///
    /// The 8×8 block size (256 bytes per block) balances cache efficiency with
    /// parallelism opportunities. Smaller blocks would improve cache hit rate but
    /// reduce available parallelism; larger blocks do the opposite.
    ///
    /// # Performance
    ///
    /// Typically 2-4x faster than naive implementation on multi-core CPUs,
    /// with speedup increasing for larger matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand matrix
    /// * `m` - Rows in self
    /// * `n` - Columns in other
    /// * `k` - Inner dimension
    ///
    /// # Returns
    ///
    /// Result tensor of shape `[m, n]`
    fn matmul_parallel_blocked(
        &self,
        other: &Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Tensor {
        // Block size for cache optimization
        // 8×8 blocks = 256 bytes (fits well in L1 cache: typically 32-64KB)
        const BLOCK_SIZE: usize = 8;

        let mut result = vec![0.0; m * n];

        // Parallelize over output row blocks
        // Each thread processes BLOCK_SIZE rows independently
        result
            .par_chunks_mut(BLOCK_SIZE * n)
            .enumerate()
            .for_each(|(block_i, result_block)| {
                let i_start = block_i * BLOCK_SIZE;
                let i_end = (i_start + BLOCK_SIZE).min(m);

                // Iterate over column blocks
                for j_start in (0..n).step_by(BLOCK_SIZE) {
                    let j_end = (j_start + BLOCK_SIZE).min(n);

                    // Iterate over inner dimension blocks
                    for k_start in (0..k).step_by(BLOCK_SIZE) {
                        let k_end = (k_start + BLOCK_SIZE).min(k);

                        // Compute this block (cache-friendly inner loops)
                        for i in i_start..i_end {
                            let row_offset = (i - i_start) * n;
                            for k_idx in k_start..k_end {
                                let a_val = self.data[i * k + k_idx];

                                // SIMD-optimized innermost loop
                                Self::matmul_inner_simd(
                                    a_val,
                                    &other.data[k_idx * n + j_start..k_idx * n + j_end],
                                    &mut result_block[row_offset + j_start..row_offset + j_end],
                                );
                            }
                        }
                    }
                }
            });

        Tensor::new(result, vec![m, n])
    }

    /// Softmax activation
    ///
    /// Computes softmax along the specified axis:
    ///
    /// ```text
    /// softmax(x)[i] = exp(x[i]) / sum(exp(x[j])) for all j
    /// ```
    ///
    /// # Numerical Stability
    ///
    /// Uses the numerically stable version:
    ///
    /// ```text
    /// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    /// ```
    ///
    /// Subtracting the maximum prevents overflow in exp() while producing
    /// the same result (since the max factors cancel out).
    ///
    /// # Arguments
    ///
    /// * `axis` - Axis along which to compute softmax (use -1 for last axis)
    ///
    /// # Performance
    ///
    /// For 2D tensors with axis=-1, softmax is computed per row in parallel.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    /// let result = tensor.softmax(-1);
    /// // Result sums to 1.0 along last dimension
    /// ```
    pub fn softmax(&self, axis: isize) -> Tensor {
        // Convert negative axis to positive
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // === 2D SOFTMAX PER ROW (common case for attention) ===
        if self.shape.len() == 2 && axis_pos == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            // Parallel softmax computation per row
            let result: Vec<f32> = (0..rows)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    let row = &self.data[start..end];

                    // Find max for numerical stability
                    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Compute exp(x - max)
                    let exp_values: Vec<f32> =
                        row.iter().map(|&x| (x - max).exp()).collect();

                    // Normalize
                    let sum: f32 = exp_values.iter().sum();
                    exp_values.into_iter().map(move |val| val / sum)
                })
                .collect();

            return Tensor::new(result, self.shape.clone());
        }

        // === FALLBACK: GLOBAL SOFTMAX ===
        // Less common, but included for completeness
        let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> =
            self.data.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        let result = exp_values.iter().map(|&x| x / sum).collect();

        Tensor::new(result, self.shape.clone())
    }

    /// Element-wise addition with broadcasting support
    ///
    /// Supports several broadcasting patterns common in transformers:
    ///
    /// 1. **Exact match**: Same shape
    /// 2. **Broadcast last dim**: `[*, n] + [n]` (e.g., adding bias)
    /// 3. **Broadcast batch**: `[batch, seq, dim] + [seq, dim]`
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to add (may have different shape if broadcasting applies)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let b = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    /// let c = a.add(&b);
    /// assert_eq!(c.data, vec![2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn add(&self, other: &Tensor) -> Tensor {
        // === EXACT MATCH: Same shape ===
        if self.shape == other.shape {
            let result = self
                .data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // === BROADCAST BATCH: [batch, seq, dim] + [seq, dim] ===
        if self.shape.len() == 3 && other.shape.len() == 2 {
            let batch_size = self.shape[0];
            let seq_len = self.shape[1];
            let dim = self.shape[2];

            assert_eq!(
                other.shape[0], seq_len,
                "Sequence length must match for broadcasting"
            );
            assert_eq!(
                other.shape[1], dim,
                "Dimension must match for broadcasting"
            );

            let result: Vec<f32> = (0..batch_size * seq_len * dim)
                .into_par_iter()
                .map(|i| {
                    let s = (i / dim) % seq_len;
                    let d = i % dim;
                    let other_idx = s * dim + d;
                    self.data[i] + other.data[other_idx]
                })
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // === BROADCAST LAST DIM: [*, n] + [n] (e.g., bias addition) ===
        if self.shape.len() > other.shape.len() {
            let last_dim = *self.shape.last().unwrap();
            if other.data.len() == last_dim {
                let result: Vec<f32> = (0..self.data.len())
                    .into_par_iter()
                    .map(|i| {
                        let other_idx = i % last_dim;
                        self.data[i] + other.data[other_idx]
                    })
                    .collect();
                return Tensor::new(result, self.shape.clone());
            }
        }

        panic!(
            "Unsupported broadcast for add: {:?} + {:?}",
            self.shape, other.shape
        );
    }

    /// Element-wise multiplication with broadcasting
    ///
    /// See `add()` for broadcasting patterns.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        // Exact match
        if self.shape == other.shape {
            let result = self
                .data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // Broadcast last dimension
        if self.shape.len() > other.shape.len() {
            let last_dim = *self.shape.last().unwrap();
            if other.data.len() == last_dim {
                let result: Vec<f32> = (0..self.data.len())
                    .into_par_iter()
                    .map(|i| {
                        let other_idx = i % last_dim;
                        self.data[i] * other.data[other_idx]
                    })
                    .collect();
                return Tensor::new(result, self.shape.clone());
            }
        }

        panic!(
            "Unsupported broadcast for mul: {:?} * {:?}",
            self.shape, other.shape
        );
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Shapes must match for subtraction"
        );
        let result = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for division");
        let result = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(a, b)| a / b)
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Add scalar to all elements
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x + scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Multiply all elements by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x * scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Divide all elements by scalar
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x / scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor {
        let result = self.data.par_iter().map(|&x| x.sqrt()).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Reshape tensor to new shape
    ///
    /// Total number of elements must remain the same.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let reshaped = tensor.reshape(&[3, 2]);
    /// assert_eq!(reshaped.shape, vec![3, 2]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_size,
            "Cannot reshape: element count mismatch"
        );
        Tensor::new(self.data.clone(), new_shape.to_vec())
    }

    /// Transpose two dimensions
    ///
    /// # Arguments
    ///
    /// * `dim1` - First dimension to swap (supports negative indexing)
    /// * `dim2` - Second dimension to swap (supports negative indexing)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let transposed = tensor.transpose(0, 1);
    /// assert_eq!(transposed.shape, vec![2, 2]);
    /// ```
    pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
        let ndim = self.shape.len() as isize;

        // Convert negative indices
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;
        let d2 = if dim2 < 0 { ndim + dim2 } else { dim2 } as usize;

        // Create new shape with swapped dimensions
        let mut new_shape = self.shape.clone();
        new_shape.swap(d1, d2);

        // For 2D matrices, we can use a simple transpose
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = vec![0.0; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = self.data[i * cols + j];
                }
            }

            return Tensor::new(result, new_shape);
        }

        // For higher dimensions, do full transpose with stride remapping
        let old_strides = &self.strides;
        let mut new_strides = old_strides.clone();
        new_strides.swap(d1, d2);

        let total_size = self.data.len();
        let mut result = vec![0.0; total_size];

        for (i, item) in result.iter_mut().enumerate().take(total_size) {
            // Compute old multi-index from flat index
            let mut old_idx = 0;
            let mut remaining = i;

            for (dim_idx, &stride) in new_strides.iter().enumerate() {
                let coord = remaining / stride;
                remaining %= stride;
                old_idx += coord * old_strides[dim_idx];
            }

            *item = self.data[old_idx];
        }

        Tensor::new(result, new_shape)
    }

    /// Replace values where mask is true with given value
    ///
    /// Used for causal masking in attention (setting future positions to -inf)
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean mask (non-zero = true)
    /// * `value` - Value to fill where mask is true
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        assert_eq!(
            self.shape, mask.shape,
            "Mask shape must match tensor shape"
        );
        let result = self
            .data
            .par_iter()
            .zip(&mask.data)
            .map(|(&x, &m)| if m != 0.0 { value } else { x })
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Compute mean along an axis
    ///
    /// # Arguments
    ///
    /// * `axis` - Axis along which to compute mean (use -1 for last axis)
    /// * `keepdim` - Whether to keep the reduced dimension (size 1)
    pub fn mean(&self, axis: isize, keepdim: bool) -> Tensor {
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // For 2D tensor, compute mean along specified axis
        if self.shape.len() == 2 && axis_pos == 1 {
            // Mean along columns (result has shape [rows, 1] or [rows])
            let rows = self.shape[0];
            let cols = self.shape[1];

            let result: Vec<f32> = (0..rows)
                .into_par_iter()
                .map(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    let sum: f32 = self.data[start..end].iter().sum();
                    sum / cols as f32
                })
                .collect();

            let new_shape = if keepdim {
                vec![rows, 1]
            } else {
                vec![rows]
            };
            return Tensor::new(result, new_shape);
        }

        // For 3D tensor [batch, seq, dim], compute mean along last axis
        if self.shape.len() == 3 && axis_pos == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];

            let result: Vec<f32> = (0..batch * seq)
                .into_par_iter()
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    let sum: f32 = self.data[start..end].iter().sum();
                    sum / dim as f32
                })
                .collect();

            let new_shape = if keepdim {
                vec![batch, seq, 1]
            } else {
                vec![batch, seq]
            };
            return Tensor::new(result, new_shape);
        }

        panic!("Unsupported mean operation for shape {:?}", self.shape);
    }

    /// Compute variance along an axis
    ///
    /// # Arguments
    ///
    /// * `axis` - Axis along which to compute variance
    /// * `keepdim` - Whether to keep the reduced dimension
    pub fn var(&self, axis: isize, keepdim: bool) -> Tensor {
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // For 3D tensor [batch, seq, dim], compute variance along last axis
        if self.shape.len() == 3 && axis_pos == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];

            let result: Vec<f32> = (0..batch * seq)
                .into_par_iter()
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    let slice = &self.data[start..end];

                    // Compute mean
                    let mean: f32 = slice.iter().sum::<f32>() / dim as f32;

                    // Compute variance
                    let variance: f32 = slice
                        .iter()
                        .map(|&x| {
                            let diff = x - mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        / dim as f32;

                    variance
                })
                .collect();

            let new_shape = if keepdim {
                vec![batch, seq, 1]
            } else {
                vec![batch, seq]
            };
            return Tensor::new(result, new_shape);
        }

        panic!("Unsupported var operation for shape {:?}", self.shape);
    }

    /// Create a tensor with sequential integers
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value (inclusive)
    /// * `end` - Ending value (exclusive)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::Tensor;
    /// let tensor = Tensor::arange(0, 5);
    /// assert_eq!(tensor.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn arange(start: usize, end: usize) -> Tensor {
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let len = data.len();
        Tensor::new(data, vec![len])
    }
}
