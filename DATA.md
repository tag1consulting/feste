# Training Data

## Shakespeare's Complete Works

**Source:** Project Gutenberg
**URL:** https://www.gutenberg.org/files/100/100-0.txt
**License:** Public Domain
**Size:** ~5.5MB plain text
**Content:** 40+ plays, sonnets, and poems

### Obtaining the Data

Download directly from Project Gutenberg:
```bash
curl -o shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
```

Alternatively, the file may be included in this repository (it is public domain).

### Legal Status

This text is in the public domain worldwide. Project Gutenberg's files are free to use for any purpose with no restrictions beyond preserving the Project Gutenberg trademark notice (which we remove during preprocessing).

### Data Preprocessing

We perform minimal preprocessing:
1. Remove Project Gutenberg header and footer (legal notices)
2. Keep all text as-is (preserve Shakespeare's original spelling and punctuation)
3. The tokenizer handles raw text directly

### Why Shakespeare?

Shakespeare's complete works provide an ideal training corpus for an educational LLM:

1. **Public domain:** No licensing concerns, can be freely distributed
2. **Appropriate size:** 5.5MB is large enough to train interesting models but small enough to process in reasonable time
3. **Distinctive style:** Elizabethan English has recognizable patterns that make learning visible
4. **Cultural significance:** Familiar to many readers, making outputs easier to evaluate
5. **Rich language:** Complex vocabulary and sentence structures provide good learning challenges
6. **Established benchmark:** Many educational LLM projects use Shakespeare, enabling comparison

### Alternative Corpora

The implementation works with any plain text file. To train on different data:

```rust
let text = std::fs::read_to_string("your_corpus.txt")?;
let tokenizer = Tokenizer::train(&text, vocab_size);
// ... proceed with training
```

Suitable alternatives:
- **Other classic literature** (check copyright status)
- **Wikipedia dumps** (CC-BY-SA licensed)
- **Your own writing** (useful for understanding how models learn from specific domains)
- **Code repositories** (demonstrates language models work on formal languages)

### Data Not Included

We do not use:
- Copyrighted modern texts without permission
- Personal data or private information
- Data requiring scraping or terms-of-service violations
- Extremely large corpora requiring preprocessing infrastructure (this is an educational project)
