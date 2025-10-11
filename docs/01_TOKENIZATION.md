# Tokenization: Text to Numbers

This document is a companion to the blog post "Building an LLM From Scratch in Rust, Part 1: Tokenization". It covers implementation details and provides guidance for working with the tokenization code in this repository.

## What's Here

Language models don't read text. They read numbers. The transformation from text to numbers happens in the tokenizer. This implementation uses Byte Pair Encoding (BPE), the same algorithm used by GPT-2 and GPT-3.

**Implementation files:**
- [src/tokenizer.rs](../src/tokenizer.rs) - Core BPE implementation
- [examples/01_train_tokenizers.rs](../examples/01_train_tokenizers.rs) - Training example

## Running the Example

```bash
# Download Shakespeare corpus
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt

# Train tokenizers at multiple vocabulary sizes
cargo run --release --example 01_train_tokenizers
```

This trains tokenizers with vocabulary sizes of `[256, 512, 1024, 1536, 20534]` and saves results to `data/example_tokenizer_<timestamp>/`.

The vocabulary sizes align with the training examples we'll use in later chapters:
- 256 = Byte-level only (no merges learned)
- 512 = Tiny model baseline
- 1024 = Small model
- 1536 = Medium model
- 20534 = Large model approaching production scale

## Expected Output

Training all five tokenizers takes about 10 minutes on a modern CPU:

```
======================================================================
  BPE Tokenizer Training
======================================================================

Output directory: data/example_tokenizer_1234567890

Loading training data...
Loaded corpus: 5422721 bytes (5.42 MB)

======================================================================
Training tokenizer with vocab_size = 256
======================================================================
  Starting vocab size: 256
  Target vocab size: 256
  Corpus size: 5422721 bytes
Training complete! Final vocab size: 256
Learned 0 merges

Encoding full corpus to measure compression...

Results:
  Training time: 0.00s
  Encoding time: 0.14s
  Vocabulary size: 256
  Original size: 5422721 bytes
  Encoded length: 5422721 tokens
  Compression ratio: 1.00x
  Bytes per token: 1.00

Testing round-trip encoding...
  ✓ Round-trip test PASSED

Saved to: data/example_tokenizer_1234567890/tokenizer_256.json

[... continues for each vocab size ...]
```

The final vocabulary (20534) takes about 5 minutes to train because of the sampling optimization. Without sampling, it would take hours.

## Understanding the Results

### Vocabulary Size 256 (Byte-level Only)

No merges learned. Every byte gets its own token. Compression ratio is 1.00x because we're not compressing anything. This is the baseline before BPE does any work.

### Vocabulary Size 512

With 256 merges, compression jumps to about 1.96x. The tokenizer learns common patterns like "e ", "th", "t ", "and ", "you". These are the building blocks of English text.

Training takes about 64 seconds. Encoding takes about 11 seconds.

Example tokenization:
```
"To be, or not to be" -> 6 tokens: [To |be |or |not |to |be]
```

### Vocabulary Size 1024

With 768 merges, compression improves to 2.48x. Common words like "be", "or", "not", "and" get their own tokens.

Training takes about 167 seconds. Encoding takes about 21 seconds.

Example tokenization:
```
"Wherefore art thou" -> 5 tokens: [Wh|ere|fore |art |thou]
```

Notice "Wherefore" splits into 3 pieces. The tokenizer learned "ere" and "fore" as common patterns.

### Vocabulary Size 1536

With 1,280 merges, compression reaches 2.78x. The tokenizer now knows longer patterns and more domain-specific vocabulary from Shakespeare.

Training takes about 281 seconds. Encoding takes about 31 seconds.

Example tokenization:
```
"Wherefore art thou" -> 4 tokens: [Wh|erefore |art |thou]
```

Now "Wherefore" is just 2 pieces. The archaic word pattern has been learned.

### Vocabulary Size 20,534

With 20,278 merges, compression hits 3.66x. Many common phrases, character names, and domain-specific terms get their own tokens.

Training takes about 286 seconds (sampling optimization prevents exponential growth). Encoding takes about 242 seconds.

Example tokenization:
```
"To be, or not to be" -> 5 tokens: [To |be |or |not to |be]
```

Finally we see phrase-level tokens. "not to" appears as a single token because it's extremely common in Shakespeare and English generally.

## Trade-offs

Larger vocabularies compress better but have costs:

**Benefits:**
- Better compression (shorter sequences)
- Faster training and inference (fewer tokens to process)
- More precise representation of common words

**Costs:**
- Larger embedding tables (more parameters)
- More parameters to learn from data
- Slower encoding (more merge rules to check)
- Rare tokens may not be learned well

The encoding time scaling is dramatic. At vocabulary size 20534, encoding takes 242 seconds vs 31 seconds for vocabulary size 1536. That's nearly 8x slower. This is the hidden cost of large vocabularies.

For production models training on billions of tokens, the slower encoding is worth it. The cost is paid once during data preprocessing, but the model benefits from shorter sequences throughout training and deployment.

For educational purposes or rapid experimentation, smaller vocabularies (512-1536) offer the best balance.

## Implementation Details

### Core API

```rust
// Create new tokenizer with base vocabulary
let mut tokenizer = BPETokenizer::new(vocab_size);

// Train on corpus (learns merge rules)
tokenizer.train(&text, vocab_size);

// Convert text to token IDs
let ids: Vec<usize> = tokenizer.encode(text);

// Convert token IDs back to text
let text: String = tokenizer.decode(&ids);

// Save/load for later use
tokenizer.save("tokenizer.json")?;
let tokenizer = BPETokenizer::load("tokenizer.json")?;
```

### Key Data Structures

```rust
pub struct BPETokenizer {
    vocab: HashMap<String, usize>,    // Maps token strings to IDs
    merges: Vec<(String, String)>,    // Merge rules in order
    unk_token: String,                // Unknown token (unused)
}
```

The vocabulary starts with 256 entries mapping hex-encoded bytes (`<00>` through `<ff>`) to IDs 0-255. During training, we add new entries for each merge. The merge rules are stored in a vector because order matters during encoding.

Merge order matters because merges build on each other. If merge 1 learns that `<74><68>` (bytes for "th") becomes token 256, and merge 50 learns that token 256 + `<65>` ("th" + "e") becomes token 300, you must apply merge 1 first. Apply the merges out of order and you won't find the patterns.

Decoding is simpler. You just look up each token ID in the vocabulary and convert it back to its byte sequence.

### Hex Encoding Convention

We encode the first 256 base tokens as hex strings like `<41>` for readability. Byte 65 (decimal) is `<41>` in hex, which happens to be the letter "A" in UTF-8. Byte 32 is `<20>`, which is a space.

This is a formatting choice that makes token boundaries visually distinct when debugging. In the implementation, everything is stored consistently as token IDs that map to these hex strings in the vocabulary.

### Optimizations Included

**Parallel Pair Counting**

The core training loop counts adjacent pairs in the corpus. With millions of bytes and potentially thousands of merges to learn, a single-threaded implementation would take hours.

We use Rayon to parallelize pair counting across CPU cores. On a modern multi-core system, this gives roughly 2-3x speedup. Why not 8x on an 8-core machine? Because only the counting phase is parallelized. More fundamentally, all cores compete for memory bandwidth. They're reading from the same RAM, and memory can only deliver data so fast.

**Chunk Boundaries**

When splitting the corpus into chunks for parallel processing, what about pairs that span chunk boundaries? If chunk 1 ends with "th" and chunk 2 starts with "e", we'd miss counting "the". We handle this explicitly by having each thread check its chunk boundary and count the pair between its last token and the next chunk's first token.

**Sampling for Large Vocabularies**

For vocabulary sizes over 2,000 tokens, we train on a 200KB sample instead of processing the entire 5.5MB corpus for every merge iteration.

This works because the most frequent pairs appear with similar relative frequencies in any reasonably-sized sample. Testing on a 1024-token vocabulary confirms this: 80% of the top-10 merges and 88% of the top-50 merges are identical between full corpus and sample training.

The practical benefit is significant: training on the sample is 18x faster (9 seconds vs 165 seconds). More importantly, the resulting tokenizer performs nearly identically, achieving just 0.34% different compression on test data.

**Efficient Merge Application**

During encoding, we apply merge rules in order. The naive approach would modify the token list in place: find a pair, delete both tokens, insert the merged token, continue. But deletion from the middle of a vector is expensive. When you delete using `remove()`, everything after that position shifts down. Do that repeatedly and you're shifting the same elements over and over.

The key optimization is building a new vector rather than modifying the existing one:

```rust
let mut new_tokens = Vec::with_capacity(tokens.len());
let mut i = 0;
while i < tokens.len() {
    if i < tokens.len() - 1 && tokens[i] == *pair_a && tokens[i + 1] == *pair_b {
        new_tokens.push(merged.clone());
        i += 2;  // Skip both tokens in the pair
    } else {
        new_tokens.push(tokens[i].clone());
        i += 1;
    }
}
tokens = new_tokens;
```

This trades memory for speed. We temporarily use twice the memory while building the new vector, but modern systems have plenty of RAM and the allocation is short-lived.

### What We Didn't Optimize

Our implementation prioritizes clarity and educational value over raw performance. Production tokenizers use additional optimizations:

**Trie-based merge lookup** uses a tree structure for O(log n) lookup instead of checking every merge rule sequentially.

**Cached encoding** stores the tokenization result for sequences you've already seen (memoization).

**Integer pair representation** stores merges as three numbers instead of variable-length strings.

**Minimal perfect hashing** for vocabulary lookups, guaranteeing zero collisions with minimal memory.

**Vocabulary pruning** keeps only frequently-used tokens, replacing rare tokens with more useful candidates.

**Special token handling** for `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>` tokens used in various model architectures.

We're not using these because the goal is understanding, not optimization. Our implementation shows every step explicitly. The code looks like the pseudocode you'd write on a whiteboard. The performance characteristics we see in benchmarks directly reflect algorithmic choices.

For production use, you'd use a highly optimized library like [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) or [tokenizers](https://github.com/huggingface/tokenizers).

## Common Issues

### shakespeare.txt not found

Download the corpus:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

### Training is slow

Make sure you're compiling with `--release`. Debug builds are much slower:
```bash
cargo run --release --example 01_train_tokenizers
```

For vocabulary sizes over 2000, the implementation automatically uses sampling optimization.

### Round-trip test fails

If the encode/decode round-trip doesn't match the original text, something is broken. This should never happen with our implementation. If it does:

1. Check that you're using the correct tokenizer file
2. Verify the file wasn't corrupted
3. Try retraining the tokenizer

### Poor compression ratio

If compression ratio is lower than expected:

1. Check that vocabulary size actually increased (run `tokenizer.vocab_size()`)
2. Ensure training completed (check for "Training complete!" message)
3. Try a larger vocabulary size
4. Some corpora compress better than others (code compresses very well, poetry less so)

## Analyzing Vocabulary

The trained tokenizers include analysis showing what was learned:

```
=== Vocabulary Analysis ===

Token Composition:
  Base tokens (bytes): 256
  Learned merges: 768
  Total vocabulary: 1024

Sample of Learned Tokens (first 30):
  [256] "<74><68>" (th)
  [257] "<65><20>" (e )
  [258] "<6f><75>" (ou)
  ...

Example Tokenizations:
  "To be, or not to be" -> [To| |be| |or| |not| |to| |be]
  "Romeo and Juliet" -> [Rom|eo| |and| |Jul|iet]
```

This shows:
- How many base tokens vs. learned merges
- What common patterns were learned
- How specific phrases get split into tokens

## Next Steps

After understanding tokenization, move on to Chapter 2: Tensor Operations. We'll build the numerical foundation for neural networks using matrix multiplication, element-wise operations, broadcasting, and activation functions.

The tokenizer we built here will be used throughout the remaining chapters to convert text into token IDs that the model can process.

## Further Reading

- **Original BPE paper**: Sennrich et al., ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) (2016) - The foundational paper introducing Byte Pair Encoding for NLP. Describes the algorithm we implemented and demonstrates its effectiveness on machine translation tasks.

- **GPT-2 tokenizer**: [OpenAI's BPE implementation with 50,257 tokens](https://github.com/openai/gpt-2/blob/master/src/encoder.py) - Production-grade tokenizer code from GPT-2. Shows how the algorithm scales to large vocabularies and handles special tokens.

- **SentencePiece**: [Google's tokenization library](https://github.com/google/sentencepiece) - Language-agnostic tokenizer implementing BPE, unigram language models, and other subword algorithms. Used by many production models including T5 and BERT variants.

- **Byte-level BPE**: [HuggingFace's explanation](https://huggingface.co/docs/transformers/tokenizer_summary#bytelevel-bpe) - Details why byte-level encoding (what we implemented) handles any Unicode text without preprocessing. Contrasts with character-level approaches that require explicit Unicode handling.
