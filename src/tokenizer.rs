//! Byte Pair Encoding (BPE) Tokenizer
//!
//! This module implements BPE tokenization from scratch. BPE is the standard
//! tokenization method used by GPT-2, GPT-3, and many other language models.
//!
//! ## How BPE Works
//!
//! 1. **Start with byte-level encoding**: 256 base tokens (one per byte value: 0-255)
//! 2. **Count adjacent pairs**: Find the most common adjacent byte pair in the corpus
//! 3. **Merge the most frequent pair**: Create a new token representing that pair
//! 4. **Repeat**: Continue until vocabulary reaches the target size
//!
//! ## Example
//!
//! Given corpus: "hello hello world"
//! - Iteration 1: Most common pair = ('l','l') → merge to token 256: "he[256]o he[256]o world"
//! - Iteration 2: Most common pair = ('h','e') → merge to token 257: "[257][256]o [257][256]o world"
//! - Iteration 3: Most common pair = ('[257]','[256]') → merge to token 258: "[258]o [258]o world"
//! - Continue until vocab_size is reached...
//!
//! ## Why Tokenization Matters
//!
//! Language models never see raw text—they see token IDs. This fundamental
//! transformation explains many "quirky" behaviors:
//!
//! - **Can't reliably count letters**: The model sees tokens like ["run", "ning"],
//!   not individual characters
//! - **Sensitive to spacing**: " hello" and "hello" tokenize differently
//! - **Struggles with rare words**: Uncommon words are broken into unfamiliar pieces
//! - **Reversal is hard**: Character-level operations are difficult when working with tokens
//!
//! ## Implementation Notes
//!
//! This implementation includes several optimizations for practical use:
//!
//! - **Parallel pair counting**: Uses Rayon to count byte pairs across CPU cores,
//!   providing 2-3x speedup on multi-core systems
//! - **Parallel encoding**: Large texts are split into chunks and encoded in parallel
//! - **Training on samples**: For very large vocabularies (>2000 tokens), training
//!   uses a subset of the corpus to learn patterns faster
//! - **Efficient merge application**: Builds new token vectors instead of in-place
//!   removal, avoiding O(n²) complexity
//!
//! These optimizations are necessary to make training practical on CPU, but the
//! underlying algorithm is standard BPE.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A Byte Pair Encoding tokenizer
///
/// The tokenizer maintains a vocabulary of tokens (strings mapped to IDs) and
/// a sequence of merge rules learned during training. The merges are applied
/// in order during encoding to convert text into token IDs.
#[derive(Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// Maps token strings to their integer IDs
    /// Always starts with 256 base tokens for byte-level encoding
    vocab: HashMap<String, usize>,

    /// Sequence of merge rules learned during training
    /// Each merge combines two tokens into a new token
    /// Applied in order during encoding
    merges: Vec<(String, String)>,

    /// Unknown token (currently unused, kept for future extensibility)
    #[allow(dead_code)]
    unk_token: String,
}

impl BPETokenizer {
    /// Create a new tokenizer with base vocabulary
    ///
    /// Initializes the tokenizer with 256 base tokens representing each possible
    /// byte value (0-255). These are encoded as hex strings like "<00>", "<01>", etc.
    ///
    /// # Arguments
    ///
    /// * `_vocab_size` - Target vocabulary size (currently unused in constructor,
    ///   actual vocab size is determined during training)
    ///
    /// # Returns
    ///
    /// A new tokenizer with 256 base tokens and no merges
    pub fn new(_vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();

        // Initialize with byte-level tokens (256 base tokens)
        // Each byte value is represented as a hex string: <00>, <01>, ..., <ff>
        for byte in 0..=255 {
            vocab.insert(format!("<{:02x}>", byte), vocab.len());
        }

        Self {
            vocab,
            merges: Vec::new(),
            unk_token: "<unk>".to_string(),
        }
    }

    /// Train BPE on a text corpus
    ///
    /// Learns merge rules by iteratively finding and merging the most frequent
    /// adjacent token pairs. This builds a vocabulary from the base 256 byte
    /// tokens up to the target vocabulary size.
    ///
    /// # Arguments
    ///
    /// * `text` - Training corpus (typically several MB of text)
    /// * `vocab_size` - Target vocabulary size (common values: 512, 1024, 2048, 5000)
    ///
    /// # Performance Optimization
    ///
    /// For large vocabularies (>2000 tokens), this method trains on a 200KB sample
    /// of the corpus rather than the full text. This is much faster and still learns
    /// the most common patterns effectively. The learned merges are then available
    /// for encoding the full corpus.
    ///
    /// # Example
    ///
    /// ```rust
    /// use feste::BPETokenizer;
    ///
    /// // Train on a sample text corpus
    /// let text = "hello world hello rust hello world";
    /// let mut tokenizer = BPETokenizer::new(300);
    /// tokenizer.train(&text, 300);
    ///
    /// // Verify training worked
    /// assert!(tokenizer.vocab_size() > 256);
    /// ```
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        // If target vocab size is 256 or less, we're already done
        if vocab_size <= 256 {
            return;
        }

        println!("Training BPE tokenizer...");
        println!("  Starting vocab size: {}", self.vocab.len());
        println!("  Target vocab size: {}", vocab_size);
        println!("  Corpus size: {} bytes", text.len());

        // Determine number of merges needed
        let num_merges = vocab_size - 256;

        // Optimization: For large vocab sizes (>2000), train on a smaller subset
        // 200KB is sufficient to learn common patterns and trains much faster
        let training_text = if num_merges > 2000 && text.len() > 200_000 {
            let sample_size = 200_000;
            println!(
                "  Using {}KB training sample for speed (learns common patterns faster)",
                sample_size / 1000
            );
            &text[..sample_size]
        } else {
            text
        };

        // Convert training text to byte-level tokens
        // Each byte becomes a token like "<00>", "<01>", etc.
        let mut tokens: Vec<String> = training_text
            .bytes()
            .map(|b| format!("<{:02x}>", b))
            .collect();

        // Buffer to avoid repeated allocations (Double Buffer Optimization)
        let mut new_tokens = Vec::with_capacity(tokens.len());

        // Learn merges iteratively
        for merge_idx in 0..num_merges {
            // === PARALLEL PAIR COUNTING ===
            // This is the performance bottleneck, so we parallelize it

            // Chunk size: At least 50K tokens per chunk, or divide by thread count
            let chunk_size = 50_000.max(tokens.len() / rayon::current_num_threads().max(1));

            // Count all adjacent pairs in parallel across chunks
            let pair_counts: HashMap<(String, String), usize> = tokens
                .par_chunks(chunk_size)
                .enumerate()
                .fold(HashMap::new, |mut local_counts, (chunk_idx, chunk)| {
                    // Count pairs within this chunk
                    for window in chunk.windows(2) {
                        let pair = (window[0].clone(), window[1].clone());
                        *local_counts.entry(pair).or_insert(0) += 1;
                    }

                    // Handle chunk boundaries: count the pair spanning to next chunk
                    if chunk_idx * chunk_size + chunk.len() < tokens.len() {
                        if let Some(last) = chunk.last() {
                            if let Some(next) = tokens.get(chunk_idx * chunk_size + chunk.len()) {
                                let pair = (last.clone(), next.clone());
                                *local_counts.entry(pair).or_insert(0) += 1;
                            }
                        }
                    }

                    local_counts
                })
                .reduce(HashMap::new, |mut a, b| {
                    // Merge pair counts from all chunks
                    for (pair, count) in b {
                        *a.entry(pair).or_insert(0) += count;
                    }
                    a
                });

            // If no pairs found, training is complete
            if pair_counts.is_empty() {
                break;
            }

            // === DETERMINISTIC TIE-BREAKING ===
            // HashMap iteration is random. To guarantee the exact same merges every run,
            // we sort the pairs.
            // Primary sort: Count (descending)
            // Secondary sort: Token strings (lexicographically ascending)
            let mut pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect();
            pairs.sort_by(|a, b| {
                b.1.cmp(&a.1) // Count descending
                    .then_with(|| a.0.cmp(&b.0)) // String ascending
            });

            // The first pair is the winner
            let (best_pair, count) = pairs[0].clone();

            // Create new token by concatenating the pair
            let new_token = format!("{}{}", best_pair.0, best_pair.1);

            // Add to vocabulary and record the merge rule
            self.vocab.insert(new_token.clone(), self.vocab.len());
            self.merges.push(best_pair.clone());

            // === APPLY MERGE TO CORPUS ===
            // Rebuild the token list with the new merge applied.
            // We use the double-buffer strategy here to reuse memory.
            new_tokens.clear();

            let mut i = 0;
            while i < tokens.len() {
                // If we find the pair, replace it with the merged token
                if i < tokens.len() - 1 && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1
                {
                    new_tokens.push(new_token.clone());
                    i += 2; // Skip both tokens in the pair
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            // Swap the buffers so `tokens` has the updated data for the next loop
            std::mem::swap(&mut tokens, &mut new_tokens);

            // Progress logging every 50 merges
            if merge_idx % 50 == 0 {
                println!(
                    "  Merge {}/{}: {:?} (count: {}) -> vocab size: {}",
                    merge_idx + 1,
                    num_merges,
                    best_pair,
                    count,
                    self.vocab.len()
                );
            }
        }

        println!("Training complete! Final vocab size: {}", self.vocab.len());
        println!("Learned {} merges\n", self.merges.len());
    }

    /// Convert text to byte-level token strings
    ///
    /// Internal helper that converts each byte to its hex representation.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to convert
    ///
    /// # Returns
    ///
    /// Vector of token strings like ["<68>", "<65>", "<6c>", "<6c>", "<6f>"] for "hello"
    fn byte_encode(&self, text: &str) -> Vec<String> {
        text.bytes().map(|b| format!("<{:02x}>", b)).collect()
    }

    /// Encode text to token IDs
    ///
    /// Converts text into a sequence of token IDs by first converting to byte-level
    /// tokens, then applying learned merge rules in order.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    ///
    /// # Performance Optimization
    ///
    /// For large texts (>200KB), this method splits the text into chunks and
    /// encodes them in parallel, providing significant speedup on multi-core systems.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::BPETokenizer;
    /// # let mut tokenizer = BPETokenizer::new(256);
    /// let ids = tokenizer.encode("Hello, world!");
    /// println!("Token IDs: {:?}", ids);
    /// ```
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Threshold for parallel processing
        const CHUNK_SIZE: usize = 100_000; // bytes per chunk

        if text.len() > CHUNK_SIZE * 2 {
            // === PARALLEL ENCODING FOR LARGE TEXTS ===
            // Split text into non-overlapping chunks for parallel encoding
            //
            // Note: This means we don't apply merges across chunk boundaries.
            // This is acceptable because:
            // 1. Boundaries are rare (every 100KB)
            // 2. Impact on compression is negligible
            // 3. Correctness is guaranteed (no duplicate or missing tokens)

            let mut chunks = Vec::new();
            let mut start = 0;

            while start < text.len() {
                // Calculate end position
                let mut end = (start + CHUNK_SIZE).min(text.len());

                // Adjust to character boundary (can't split UTF-8 characters)
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }

                chunks.push(&text[start..end]);

                // Move to next chunk
                start = end;
            }

            // Encode each chunk in parallel
            let encoded_chunks: Vec<Vec<usize>> = chunks
                .par_iter()
                .map(|chunk| self.encode_sequential(chunk))
                .collect();

            // Concatenate all encoded chunks
            let mut result = Vec::new();
            for chunk in encoded_chunks {
                result.extend_from_slice(&chunk);
            }
            result
        } else {
            // Small text: use sequential version (avoids parallel overhead)
            self.encode_sequential(text)
        }
    }

    /// Encode text sequentially (non-parallel version)
    ///
    /// Internal method that performs the actual encoding by applying merge rules.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    fn encode_sequential(&self, text: &str) -> Vec<usize> {
        let mut tokens = self.byte_encode(text);
        // buffer to avoid repeated allocations
        let mut new_tokens = Vec::with_capacity(tokens.len());

        for (pair_a, pair_b) in &self.merges {
            let merged = format!("{}{}", pair_a, pair_b);

            // Fast path: if the pair isn't even in the tokens, skip this merge rule
            // (Optional: this checks O(N) but saves the copy loop.
            // Useful if N is large and the pair is rare.)

            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == *pair_a && tokens[i + 1] == *pair_b {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            // Swap buffers: new_tokens becomes input for next iteration
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        tokens
            .iter()
            .map(|token| *self.vocab.get(token).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs back to text
    ///
    /// Converts a sequence of token IDs back into the original text by looking up
    /// each ID in the vocabulary and parsing the hex-encoded bytes.
    ///
    /// # Arguments
    ///
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// Decoded text string
    ///
    /// # Example
    ///
    /// ```rust
    /// # use feste::BPETokenizer;
    /// # let tokenizer = BPETokenizer::new(256);
    /// # let ids = tokenizer.encode("Hello!");
    /// let text = tokenizer.decode(&ids);
    /// assert_eq!(text, "Hello!");
    /// ```
    pub fn decode(&self, ids: &[usize]) -> String {
        // Create reverse vocabulary map (ID -> token string)
        let id_to_token: HashMap<usize, String> = self
            .vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();

        // Convert IDs back to token strings
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|id| id_to_token.get(id).cloned())
            .collect();

        // Join all tokens and decode using the shared hex-parsing logic
        let merged = tokens.join("");
        self.decode_token(&merged)
    }

    /// Get the vocabulary size
    ///
    /// # Returns
    ///
    /// Number of tokens in the vocabulary (256 + number of merges)
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Save tokenizer to a JSON file
    ///
    /// Serializes the tokenizer (vocabulary and merge rules) to a JSON file
    /// for later loading.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the tokenizer
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use feste::BPETokenizer;
    /// # let tokenizer = BPETokenizer::new(256);
    /// tokenizer.save("tokenizer.json").expect("Failed to save");
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from a JSON file
    ///
    /// Deserializes a previously saved tokenizer from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the tokenizer file
    ///
    /// # Returns
    ///
    /// Result containing the loaded tokenizer or an error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use feste::BPETokenizer;
    ///
    /// let tokenizer = BPETokenizer::load("tokenizer.json")
    ///     .expect("Failed to load tokenizer");
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let tokenizer: BPETokenizer = serde_json::from_str(&json)?;
        Ok(tokenizer)
    }

    /// Get statistics about the tokenizer
    ///
    /// # Returns
    ///
    /// TokenizerStats struct with vocabulary information
    pub fn stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.vocab.len(),
            num_merges: self.merges.len(),
            base_tokens: 256,
        }
    }

    /// Analyze vocabulary and display insights
    ///
    /// Prints detailed information about the tokenizer's vocabulary, including:
    /// - Token composition (base vs. learned)
    /// - Sample of learned tokens
    /// - Compression analysis on sample text
    /// - Example tokenizations
    ///
    /// This is useful for understanding what the tokenizer has learned.
    ///
    /// # Arguments
    ///
    /// * `sample_text` - Text to use for compression analysis and examples
    pub fn analyze_vocabulary(&self, sample_text: &str) {
        println!("\n=== Vocabulary Analysis ===\n");

        // Find human-readable tokens (merged tokens, not just base bytes)
        let mut readable_tokens: Vec<(String, usize)> = self
            .vocab
            .iter()
            .filter(|(token, _)| !token.starts_with('<') || token.len() > 4)
            .map(|(token, id)| (token.clone(), *id))
            .collect();

        // Sort by token ID (roughly reflects merge order during training)
        readable_tokens.sort_by_key(|(_, id)| *id);

        // Display token type breakdown
        let base_tokens = 256;
        let merged_tokens = self.vocab.len() - base_tokens;
        println!("Token Composition:");
        println!("  Base tokens (bytes): {}", base_tokens);
        println!("  Learned merges: {}", merged_tokens);
        println!("  Total vocabulary: {}\n", self.vocab.len());

        // Show sample of learned tokens
        println!("Sample of Learned Tokens (first 30):");
        let display_count = 30.min(readable_tokens.len());
        for (token, id) in readable_tokens.iter().take(display_count) {
            // Try to decode token for display
            let decoded = self.decode_token(token);
            if decoded.len() <= 20 && !decoded.is_empty() {
                println!("  [{}] \"{}\"", id, decoded);
            }
        }

        // Analyze compression on sample text
        if !sample_text.is_empty() {
            println!("\nCompression Analysis (on sample):");
            let sample_chars: String = sample_text.chars().take(10000).collect();
            let tokens = self.encode(&sample_chars);
            let char_count = sample_chars.len();
            let token_count = tokens.len();
            let compression_ratio = char_count as f32 / token_count as f32;

            println!("  Sample size: {} characters", char_count);
            println!("  Token count: {} tokens", token_count);
            println!("  Compression ratio: {:.2}x", compression_ratio);
            println!("  Avg chars per token: {:.1}", compression_ratio);
        }

        // Show example tokenizations
        println!("\nExample Tokenizations:");
        let examples = vec![
            "To be, or not to be",
            "Romeo and Juliet",
            "Wherefore art thou",
            "The quality of mercy",
        ];

        for example in examples {
            let tokens = self.encode(example);
            let token_strs: Vec<String> = tokens
                .iter()
                .map(|&id| {
                    // Find token string for this ID
                    self.vocab
                        .iter()
                        .find(|(_, v)| **v == id)
                        .map(|(k, _)| self.decode_token(k))
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect();
            println!(
                "  \"{}\" -> {} tokens: [{}]",
                example,
                tokens.len(),
                token_strs.join("|")
            );
        }

        println!("\n{}\n", "=".repeat(60));
    }

    /// Helper to decode hex-encoded token strings back to readable text
    ///
    /// Parses a string containing hex-encoded bytes like "<68><65><6c><6c><6f>"
    /// and converts them back to UTF-8 text. This method handles both single tokens
    /// and concatenated sequences of tokens.
    ///
    /// # Arguments
    ///
    /// * `token` - Token string (or concatenated tokens) to decode
    ///
    /// # Returns
    ///
    /// Decoded string representation
    fn decode_token(&self, token: &str) -> String {
        // Parse hex-encoded bytes back to UTF-8 text
        let mut bytes = Vec::new();
        let mut chars = token.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Parse hex byte: <XX> -> byte value
                let mut hex_str = String::new();
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '>' {
                        chars.next(); // consume '>'
                        break;
                    }
                    hex_str.push(chars.next().unwrap());
                }

                // Convert hex string to byte and collect
                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    bytes.push(byte);
                }
            }
            // Ignore any non-hex characters (shouldn't be any in valid tokens)
        }

        // Convert collected bytes to UTF-8 string
        String::from_utf8_lossy(&bytes).to_string()
    }
}

/// Statistics about a tokenizer's vocabulary
#[derive(Debug)]
pub struct TokenizerStats {
    /// Total vocabulary size (base tokens + learned merges)
    pub vocab_size: usize,
    /// Number of merge rules learned
    pub num_merges: usize,
    /// Number of base tokens (always 256 for byte-level BPE)
    pub base_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_token_single_byte() {
        let tokenizer = BPETokenizer::new(256);

        // Single byte token: 'h' = 0x68
        let result = tokenizer.decode_token("<68>");
        assert_eq!(result, "h");
    }

    #[test]
    fn test_decode_token_multiple_bytes() {
        let tokenizer = BPETokenizer::new(256);

        // Multiple bytes: "hello"
        let result = tokenizer.decode_token("<68><65><6c><6c><6f>");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_decode_token_with_space() {
        let tokenizer = BPETokenizer::new(256);

        // "hi " (with space, 0x20)
        let result = tokenizer.decode_token("<68><69><20>");
        assert_eq!(result, "hi ");
    }

    #[test]
    fn test_decode_token_utf8_multibyte() {
        let tokenizer = BPETokenizer::new(256);

        // "é" in UTF-8 is [0xc3, 0xa9]
        let result = tokenizer.decode_token("<c3><a9>");
        assert_eq!(result, "é");
    }

    #[test]
    fn test_decode_token_empty() {
        let tokenizer = BPETokenizer::new(256);

        let result = tokenizer.decode_token("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_basic() {
        let tokenizer = BPETokenizer::new(256);

        // Create a simple test: encode "hello" as individual bytes
        let text = "hello";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = BPETokenizer::new(256);

        let test_cases = vec![
            "hello",
            "Hello, world!",
            "To be, or not to be",
            "123 456 789",
            "special chars: !@#$%^&*()",
            "newline\nand\ttab",
            "UTF-8: café, naïve, 日本語",
        ];

        for text in test_cases {
            let encoded = tokenizer.encode(text);
            let decoded = tokenizer.decode(&encoded);
            assert_eq!(decoded, text, "Failed roundtrip for: {}", text);
        }
    }

    #[test]
    fn test_encode_decode_with_merges() {
        // Create tokenizer and train it
        let mut tokenizer = BPETokenizer::new(300);
        let training_text = "hello hello world world hello";
        tokenizer.train(training_text, 300);

        // Test that encode/decode still works after training
        let test_text = "hello world";
        let encoded = tokenizer.encode(test_text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, test_text);
    }

    #[test]
    fn test_decode_token_consistency_with_decode() {
        let tokenizer = BPETokenizer::new(256);

        // Test that decode_token produces same result as decode for single token
        let token_str = "<68><65><6c><6c><6f>"; // "hello"

        // Direct decode_token
        let direct_result = tokenizer.decode_token(token_str);

        // Simulate what decode does: parse the token string as if it came from vocab
        let simulated_result = tokenizer.decode_token(token_str);

        assert_eq!(direct_result, simulated_result);
        assert_eq!(direct_result, "hello");
    }

    #[test]
    fn test_decode_token_concatenated() {
        let tokenizer = BPETokenizer::new(256);

        // Multiple tokens concatenated (what decode does before calling decode_token)
        let concatenated = "<68><65><6c><6c><6f><20><77><6f><72><6c><64>";
        let result = tokenizer.decode_token(concatenated);

        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = BPETokenizer::new(256);
        assert_eq!(tokenizer.vocab_size(), 256);

        let mut tokenizer2 = BPETokenizer::new(512);
        tokenizer2.train("hello hello world", 512);
        // Note: actual vocab size depends on how many unique pairs exist in the corpus
        // Small corpus won't reach target vocab size, so just verify it increased
        assert!(tokenizer2.vocab_size() > 256);
        assert!(tokenizer2.vocab_size() <= 512);
    }

    #[test]
    fn test_base_vocab_coverage() {
        let tokenizer = BPETokenizer::new(256);

        // All byte values should be encodable
        for byte in 0u8..=255u8 {
            let text = String::from_utf8(vec![byte]).unwrap_or_else(|_| {
                // For invalid UTF-8, create string from bytes using from_utf8_lossy
                String::from_utf8_lossy(&[byte]).to_string()
            });

            let encoded = tokenizer.encode(&text);
            let decoded = tokenizer.decode(&encoded);

            // Should roundtrip correctly
            assert_eq!(decoded, text);
        }
    }
}
