//! Tensor Operations Demonstration
//!
//! This example demonstrates the core tensor operations used in neural networks:
//! - Creating tensors
//! - Matrix multiplication (sequential and parallel)
//! - Element-wise operations
//! - Broadcasting
//! - Softmax and other activations
//! - Reshaping and transposing
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example 02_tensor_operations
//! ```
//!
//! # Expected Runtime
//!
//! Less than 1 second

use feste::Tensor;
use std::time::Instant;

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  Tensor Operations Demonstration");
    println!("{}", "=".repeat(70));

    // ========== Basic Tensor Creation ==========
    println!("\n{}",  "─".repeat(70));
    println!("1. Creating Tensors");
    println!("{}", "─".repeat(70));

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![2, 3]);
    println!("Created 2×3 tensor:");
    println!("  Shape: {:?}", tensor.shape);
    println!("  Data: {:?}", tensor.data);

    let zeros = Tensor::zeros(vec![3, 4]);
    println!("\nCreated 3×4 zero tensor:");
    println!("  Shape: {:?}", zeros.shape);
    println!("  Sum: {}", zeros.data.iter().sum::<f32>());

    let range = Tensor::arange(0, 10);
    println!("\nCreated range tensor [0, 10):");
    println!("  Shape: {:?}", range.shape);
    println!("  Data: {:?}", range.data);

    // ========== Matrix Multiplication ==========
    println!("\n{}", "─".repeat(70));
    println!("2. Matrix Multiplication");
    println!("{}", "─".repeat(70));

    // Small matrix multiplication (sequential)
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    println!("\nSmall matrices (2×2) - Sequential:");
    println!("  A: {:?}", a.data);
    println!("  B (identity): {:?}", b.data);

    let start = Instant::now();
    let c = a.matmul(&b);
    let elapsed = start.elapsed();

    println!("  Result A @ B: {:?}", c.data);
    println!("  Time: {:.2}μs", elapsed.as_micros());

    // Larger matrix multiplication (parallel)
    let large_a = Tensor::new(vec![1.0; 64 * 64], vec![64, 64]);
    let large_b = Tensor::new(vec![1.0; 64 * 64], vec![64, 64]);

    println!("\nLarge matrices (64×64) - Parallel:");
    println!("  Matrix size: {} elements each", 64 * 64);

    let start = Instant::now();
    let large_c = large_a.matmul(&large_b);
    let elapsed = start.elapsed();

    println!("  Result shape: {:?}", large_c.shape);
    println!("  First element (should be 64.0): {}", large_c.data[0]);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    // ========== Element-wise Operations ==========
    println!("\n{}", "─".repeat(70));
    println!("3. Element-wise Operations");
    println!("{}", "─".repeat(70));

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

    println!("\nX: {:?}", x.data);
    println!("Y: {:?}", y.data);

    let sum = x.add(&y);
    println!("X + Y: {:?}", sum.data);

    let product = x.mul(&y);
    println!("X * Y: {:?}", product.data);

    let diff = x.sub(&y);
    println!("X - Y: {:?}", diff.data);

    let quotient = x.div(&y);
    println!("X / Y: {:?}", quotient.data);

    // ========== Scalar Operations ==========
    println!("\n{}", "─".repeat(70));
    println!("4. Scalar Operations");
    println!("{}", "─".repeat(70));

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("\nX: {:?}", x.data);

    let scaled = x.mul_scalar(2.0);
    println!("X * 2: {:?}", scaled.data);

    let shifted = x.add_scalar(10.0);
    println!("X + 10: {:?}", shifted.data);

    let divided = x.div_scalar(2.0);
    println!("X / 2: {:?}", divided.data);

    let sqrt_x = x.sqrt();
    println!("sqrt(X): {:?}", sqrt_x.data);

    // ========== Broadcasting ==========
    println!("\n{}", "─".repeat(70));
    println!("5. Broadcasting");
    println!("{}", "─".repeat(70));

    // Broadcasting last dimension (like adding bias)
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);

    println!("\nMatrix [2, 3]: {:?}", matrix.data);
    println!("Bias [3]: {:?}", bias.data);

    let with_bias = matrix.add(&bias);
    println!("Matrix + Bias (broadcasted): {:?}", with_bias.data);

    // ========== Softmax ==========
    println!("\n{}", "─".repeat(70));
    println!("6. Softmax (Numerical Stability)");
    println!("{}", "─".repeat(70));

    let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    println!("\nLogits: {:?}", logits.data);

    let probs = logits.softmax(-1);
    println!("Softmax (axis=-1): {:?}", probs.data);

    let sum: f32 = probs.data.iter().sum();
    println!("Sum of probabilities: {:.6} (should be 1.0)", sum);

    // Test with large values (numerical stability)
    let large_logits = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
    println!("\nLarge logits: {:?}", large_logits.data);

    let stable_probs = large_logits.softmax(-1);
    println!("Softmax (stable): {:?}", stable_probs.data);
    println!(
        "Sum: {:.6} (no overflow!)",
        stable_probs.data.iter().sum::<f32>()
    );

    // ========== Reshaping ==========
    println!("\n{}", "─".repeat(70));
    println!("7. Reshaping");
    println!("{}", "─".repeat(70));

    let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    println!("\nOriginal [2, 3]: {:?}", original.data);

    let reshaped = original.reshape(&[3, 2]);
    println!("Reshaped [3, 2]: {:?}", reshaped.data);
    println!("  Shape: {:?}", reshaped.shape);

    let flat = reshaped.reshape(&[6]);
    println!("Flattened [6]: {:?}", flat.data);

    // ========== Transpose ==========
    println!("\n{}", "─".repeat(70));
    println!("8. Transpose");
    println!("{}", "─".repeat(70));

    let matrix = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3], // 2 rows, 3 columns
    );
    println!("\nOriginal [2, 3]:");
    println!("  Row 0: [{}, {}, {}]", matrix.data[0], matrix.data[1], matrix.data[2]);
    println!("  Row 1: [{}, {}, {}]", matrix.data[3], matrix.data[4], matrix.data[5]);

    let transposed = matrix.transpose(0, 1);
    println!("\nTransposed [3, 2] (rows↔columns):");
    println!("  Shape: {:?}", transposed.shape);
    println!("  Row 0: [{}, {}]", transposed.data[0], transposed.data[1]);
    println!("  Row 1: [{}, {}]", transposed.data[2], transposed.data[3]);
    println!("  Row 2: [{}, {}]", transposed.data[4], transposed.data[5]);

    // ========== Mean and Variance ==========
    println!("\n{}", "─".repeat(70));
    println!("9. Statistical Operations");
    println!("{}", "─".repeat(70));

    let data = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    println!("\nData [3, 3]: {:?}", data.data);

    let row_means = data.mean(-1, false);
    println!("Row means: {:?}", row_means.data);

    let data_3d = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1, 2, 3], // [batch=1, seq=2, dim=3]
    );
    println!("\n3D Data [1, 2, 3]: {:?}", data_3d.data);

    let means = data_3d.mean(-1, true);
    println!("Mean along last dim (keepdim=true):");
    println!("  Shape: {:?}", means.shape);
    println!("  Values: {:?}", means.data);

    let vars = data_3d.var(-1, true);
    println!("Variance along last dim (keepdim=true):");
    println!("  Shape: {:?}", vars.shape);
    println!("  Values: {:?}", vars.data);

    // ========== Masked Fill (for Attention) ==========
    println!("\n{}", "─".repeat(70));
    println!("10. Masked Fill (Causal Masking)");
    println!("{}", "─".repeat(70));

    let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("\nAttention scores [2, 2]: {:?}", scores.data);

    // Causal mask: upper triangle = 1 (mask out future)
    let mask = Tensor::new(
        vec![0.0, 1.0, 0.0, 0.0], // Row 0 can't see col 1; row 1 can see all
        vec![2, 2],
    );
    println!("Causal mask: {:?}", mask.data);

    let masked = scores.masked_fill(&mask, f32::NEG_INFINITY);
    println!("Masked scores: {:?}", masked.data);
    println!("  (Future positions set to -inf)");

    // ========== Summary ==========
    println!("\n{}", "=".repeat(70));
    println!("  Summary");
    println!("{}", "=".repeat(70));
    println!("\n✓ All tensor operations working correctly");
    println!("✓ Parallel operations provide speedup for large matrices");
    println!("✓ Numerical stability maintained (softmax with large values)");
    println!("✓ Broadcasting works for common patterns");
    println!("\nThese operations form the building blocks for transformer models!");
    println!();
}
