# Late Interaction (ColBERT)

Implement fine-grained semantic retrieval using multi-vector embeddings and MaxSim scoring for high-quality document search and RAG pipelines.

!!! info "Feature Gate"
    This feature requires the `late-interaction` feature flag.

    ```toml
    [dependencies]
    mullama = { version = "0.1", features = ["late-interaction"] }
    ```

    For parallel scoring of large document collections, also enable `parallel`:

    ```toml
    mullama = { version = "0.1", features = ["late-interaction", "parallel"] }
    ```

## Overview

Late interaction (also known as ColBERT-style retrieval) provides:

- **MultiVectorGenerator** for per-token embedding generation
- **MultiVectorEmbedding** representation with per-token vectors
- **MultiVectorConfig** for embedding options
- **LateInteractionScorer** with MaxSim scoring algorithm
- **Top-K retrieval** and document ranking
- **Parallel scoring** for large collections

---

## What Is Late Interaction?

Traditional embedding models produce a **single vector** per document. Late interaction produces a **vector per token**, enabling more nuanced matching:

```
Single-Vector Retrieval:
  Query:    "machine learning" -> [0.2, 0.5, 0.1, ...]  (1 vector)
  Document: "deep learning AI" -> [0.3, 0.4, 0.2, ...]  (1 vector)
  Score:    cosine(query_vec, doc_vec) = 0.85

Late Interaction (Multi-Vector):
  Query:    "machine"  -> [0.2, 0.5, 0.1, ...]
            "learning" -> [0.1, 0.8, 0.3, ...]     (N vectors)

  Document: "deep"     -> [0.1, 0.3, 0.2, ...]
            "learning" -> [0.1, 0.9, 0.3, ...]
            "AI"       -> [0.4, 0.2, 0.6, ...]     (M vectors)

  Score:    MaxSim = sum of max similarities per query token
          = max_sim("machine", doc_tokens) + max_sim("learning", doc_tokens)
```

### Comparison with Dense Embeddings

| Approach | Granularity | Quality | Speed | Storage |
|----------|-------------|---------|-------|---------|
| Single-vector (dense) | Document-level | Good | Fast | Low |
| Late interaction | Token-level | Excellent | Medium | Higher |
| Cross-encoder | Full attention | Best | Slow | N/A |

Late interaction offers a strong quality-speed tradeoff: much better retrieval quality than single-vector approaches, with much faster scoring than cross-encoders.

---

## MultiVectorGenerator

Generate per-token embeddings from text inputs.

=== "Node.js"

    ```javascript
    const { MultiVectorGenerator } = require('mullama');

    const generator = new MultiVectorGenerator({
      model: 'embedding-model.gguf',
      normalize: true,
      skipSpecialTokens: true,
      batchSize: 32
    });

    // Generate multi-vector embedding
    const embedding = await generator.embedText('What is machine learning?');
    console.log(`Tokens: ${embedding.length}, Dimension: ${embedding.dimension}`);

    // Batch embedding
    const embeddings = await generator.embedBatch([
      'Machine learning algorithms',
      'Neural network architectures',
      'Natural language processing'
    ]);
    ```

=== "Python"

    ```python
    from mullama import MultiVectorGenerator, MultiVectorConfig

    config = MultiVectorConfig(
        normalize=True,
        skip_special_tokens=True,
        batch_size=32
    )

    generator = MultiVectorGenerator(model="embedding-model.gguf", config=config)

    # Generate multi-vector embedding
    embedding = generator.embed_text("What is machine learning?")
    print(f"Tokens: {len(embedding)}, Dimension: {embedding.dimension}")

    # Batch embedding
    embeddings = generator.embed_batch([
        "Machine learning algorithms",
        "Neural network architectures",
        "Natural language processing"
    ])
    ```

=== "Rust"

    ```rust
    use mullama::late_interaction::{MultiVectorGenerator, MultiVectorConfig};
    use mullama::Model;
    use std::sync::Arc;

    let model = Arc::new(Model::load("embedding-model.gguf")?);

    let config = MultiVectorConfig::new()
        .normalize(true)
        .skip_special_tokens(true)
        .batch_size(32);

    let mut generator = MultiVectorGenerator::new(model, config)?;

    // Generate multi-vector embedding
    let embedding = generator.embed_text("What is machine learning?")?;
    println!("Tokens: {}, Dimension: {}", embedding.len(), embedding.dimension());

    // Batch embedding
    let embeddings = generator.embed_batch(&[
        "Machine learning algorithms",
        "Neural network architectures",
        "Natural language processing",
    ])?;
    ```

### MultiVectorConfig

| Parameter | Description | Default |
|-----------|-------------|---------|
| `normalize` | L2 normalize each token embedding | `true` |
| `skip_special_tokens` | Skip BOS/EOS/PAD token embeddings | `true` |
| `store_token_ids` | Store token IDs for debugging | `false` |
| `batch_size` | Batch size for `embed_batch` | 32 |
| `max_seq_len` | Maximum sequence length (0 = model default) | 0 |

---

## LateInteractionScorer with MaxSim

Score query-document pairs using the MaxSim algorithm: sum of maximum similarities for each query token against all document tokens.

=== "Node.js"

    ```javascript
    const { LateInteractionScorer } = require('mullama');

    const query = await generator.embedText('What is deep learning?');
    const document = await generator.embedText(
      'Deep learning is a subset of machine learning that uses neural networks.'
    );

    // Basic MaxSim score
    const score = LateInteractionScorer.maxSim(query, document);
    console.log(`MaxSim score: ${score.toFixed(4)}`);

    // Normalized by query length
    const normScore = LateInteractionScorer.maxSimNormalized(query, document);
    console.log(`Normalized: ${normScore.toFixed(4)}`);
    ```

=== "Python"

    ```python
    from mullama import LateInteractionScorer

    query = generator.embed_text("What is deep learning?")
    document = generator.embed_text(
        "Deep learning is a subset of machine learning that uses neural networks."
    )

    # Basic MaxSim score
    score = LateInteractionScorer.max_sim(query, document)
    print(f"MaxSim score: {score:.4f}")

    # Normalized by query length
    norm_score = LateInteractionScorer.max_sim_normalized(query, document)
    print(f"Normalized: {norm_score:.4f}")
    ```

=== "Rust"

    ```rust
    use mullama::late_interaction::LateInteractionScorer;

    let query = generator.embed_text("What is deep learning?")?;
    let document = generator.embed_text(
        "Deep learning is a subset of machine learning that uses neural networks."
    )?;

    // Basic MaxSim score
    let score = LateInteractionScorer::max_sim(&query, &document);
    println!("MaxSim score: {:.4}", score);

    // Normalized by query length
    let norm_score = LateInteractionScorer::max_sim_normalized(&query, &document);
    println!("Normalized MaxSim: {:.4}", norm_score);

    // Symmetric MaxSim (average of both directions)
    let sym_score = LateInteractionScorer::max_sim_symmetric(&query, &document);
    println!("Symmetric MaxSim: {:.4}", sym_score);
    ```

---

## Scoring Queries Against Documents

### Top-K Retrieval

Find the most relevant documents for a query.

=== "Node.js"

    ```javascript
    // Pre-compute document embeddings
    const docEmbeddings = await generator.embedBatch(corpus);

    const query = await generator.embedText('How do neural networks learn?');
    const topK = LateInteractionScorer.findTopK(query, docEmbeddings, 10);

    console.log('Top 10 results:');
    topK.forEach(([docIdx, score], rank) => {
      console.log(`  ${rank + 1}. [${score.toFixed(4)}] ${corpus[docIdx]}`);
    });
    ```

=== "Python"

    ```python
    # Pre-compute document embeddings
    doc_embeddings = generator.embed_batch(corpus)

    query = generator.embed_text("How do neural networks learn?")
    top_k = LateInteractionScorer.find_top_k(query, doc_embeddings, 10)

    print("Top 10 results:")
    for rank, (doc_idx, score) in enumerate(top_k):
        print(f"  {rank + 1}. [{score:.4f}] {corpus[doc_idx]}")
    ```

=== "Rust"

    ```rust
    // Pre-compute document embeddings
    let doc_embeddings: Vec<_> = corpus.iter()
        .map(|text| generator.embed_text(text))
        .collect::<Result<Vec<_>, _>>()?;

    let query = generator.embed_text("How do neural networks learn?")?;
    let top_k = LateInteractionScorer::find_top_k(&query, &doc_embeddings, 10);

    println!("Top 10 results:");
    for (rank, (doc_idx, score)) in top_k.iter().enumerate() {
        println!("  {}. [{:.4}] {}", rank + 1, score, corpus[*doc_idx]);
    }
    ```

### Batch Scoring

Score multiple queries against multiple documents efficiently.

```rust
let queries: Vec<_> = query_texts.iter()
    .map(|q| generator.embed_text(q))
    .collect::<Result<Vec<_>, _>>()?;

// Returns a score matrix [num_queries x num_documents]
let score_matrix = LateInteractionScorer::batch_score(&queries, &doc_embeddings);

for (i, query_scores) in score_matrix.iter().enumerate() {
    let best = query_scores.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("Query {}: best match is doc {} (score: {:.4})", i, best.0, best.1);
}
```

---

## Parallel Scoring

!!! info "Requires `parallel` Feature"
    Parallel scoring methods require both `late-interaction` and `parallel` features.

For large document collections, use parallel variants for significant speedup.

=== "Node.js"

    ```javascript
    // Parallel top-k search
    const topK = LateInteractionScorer.findTopKParallel(query, docEmbeddings, 10);

    // Parallel batch scoring
    const scores = LateInteractionScorer.batchScoreParallel(queries, docEmbeddings);
    ```

=== "Python"

    ```python
    # Parallel top-k search
    top_k = LateInteractionScorer.find_top_k_parallel(query, doc_embeddings, 10)

    # Parallel batch scoring
    scores = LateInteractionScorer.batch_score_parallel(queries, doc_embeddings)
    ```

=== "Rust"

    ```rust
    // Parallel top-k search (uses Rayon work-stealing)
    let top_k = LateInteractionScorer::find_top_k_parallel(&query, &documents, 10);

    // Parallel batch scoring
    let scores = LateInteractionScorer::batch_score_parallel(&queries, &documents);

    // Parallel document ranking
    let rankings = LateInteractionScorer::rank_documents_parallel(&query, &documents);
    ```

### Performance Comparison

| Documents | Sequential | Parallel (8 threads) | Speedup |
|-----------|-----------|---------------------|---------|
| 1,000 | 50ms | 10ms | 5x |
| 10,000 | 500ms | 80ms | 6.25x |
| 100,000 | 5s | 750ms | 6.7x |
| 1,000,000 | 50s | 7s | 7.1x |

---

## Use Case: High-Precision Semantic Search

Late interaction excels when you need:

- **Multi-topic documents** - Each token matches independently
- **Long queries** - More query tokens means more matching opportunities
- **Partial matching** - Documents matching some query terms still score well
- **Fine-grained ranking** - Token-level interactions capture nuanced relevance

---

## Building a ColBERT Retrieval Pipeline

=== "Node.js"

    ```javascript
    const { MultiVectorGenerator, LateInteractionScorer } = require('mullama');

    class DocumentIndex {
      constructor(generator) {
        this.generator = generator;
        this.documents = [];
        this.embeddings = [];
      }

      async addDocuments(texts) {
        const newEmbeddings = await this.generator.embedBatch(texts);
        this.documents.push(...texts);
        this.embeddings.push(...newEmbeddings);
      }

      async search(queryText, topK = 5) {
        const query = await this.generator.embedText(queryText);
        const results = LateInteractionScorer.findTopK(query, this.embeddings, topK);
        return results.map(([idx, score]) => ({
          text: this.documents[idx],
          score
        }));
      }
    }

    const generator = new MultiVectorGenerator({ model: 'colbert-model.gguf' });
    const index = new DocumentIndex(generator);

    await index.addDocuments(corpus);
    const results = await index.search('What programming language is best for AI?', 3);
    results.forEach((r, i) => console.log(`${i + 1}. [${r.score.toFixed(4)}] ${r.text}`));
    ```

=== "Python"

    ```python
    from mullama import MultiVectorGenerator, MultiVectorConfig, LateInteractionScorer

    class DocumentIndex:
        def __init__(self, generator):
            self.generator = generator
            self.documents = []
            self.embeddings = []

        def add_documents(self, texts):
            new_embeddings = self.generator.embed_batch(texts)
            self.documents.extend(texts)
            self.embeddings.extend(new_embeddings)

        def search(self, query_text, top_k=5):
            query = self.generator.embed_text(query_text)
            results = LateInteractionScorer.find_top_k(query, self.embeddings, top_k)
            return [(self.documents[idx], score) for idx, score in results]

    config = MultiVectorConfig(normalize=True, skip_special_tokens=True)
    generator = MultiVectorGenerator(model="colbert-model.gguf", config=config)
    index = DocumentIndex(generator)

    index.add_documents(corpus)
    results = index.search("What programming language is best for AI?", top_k=3)
    for rank, (text, score) in enumerate(results):
        print(f"{rank + 1}. [{score:.4f}] {text}")
    ```

=== "Rust"

    ```rust
    use mullama::late_interaction::{
        MultiVectorGenerator, MultiVectorConfig,
        LateInteractionScorer, MultiVectorEmbedding,
    };
    use mullama::Model;
    use std::sync::Arc;

    struct DocumentIndex {
        documents: Vec<String>,
        embeddings: Vec<MultiVectorEmbedding>,
    }

    impl DocumentIndex {
        fn new() -> Self {
            Self { documents: Vec::new(), embeddings: Vec::new() }
        }

        fn add_documents(
            &mut self,
            texts: &[&str],
            generator: &mut MultiVectorGenerator,
        ) -> Result<(), mullama::MullamaError> {
            let new_embeddings = generator.embed_batch(texts)?;
            for (text, emb) in texts.iter().zip(new_embeddings) {
                self.documents.push(text.to_string());
                self.embeddings.push(emb);
            }
            Ok(())
        }

        fn search(&self, query: &MultiVectorEmbedding, top_k: usize) -> Vec<(f32, &str)> {
            let results = LateInteractionScorer::find_top_k(query, &self.embeddings, top_k);
            results.into_iter()
                .map(|(idx, score)| (score, self.documents[idx].as_str()))
                .collect()
        }
    }

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let model = Arc::new(Model::load("colbert-model.gguf")?);
        let config = MultiVectorConfig::new().normalize(true).skip_special_tokens(true);
        let mut generator = MultiVectorGenerator::new(model, config)?;

        let mut index = DocumentIndex::new();
        index.add_documents(&corpus, &mut generator)?;

        let query = generator.embed_text("What programming language is best for AI?")?;
        let results = index.search(&query, 3);

        for (rank, (score, text)) in results.iter().enumerate() {
            println!("{}. [{:.4}] {}", rank + 1, score, text);
        }

        Ok(())
    }
    ```

---

## Performance Considerations

### Storage Requirements

Multi-vector embeddings require more storage than single-vector:

| Documents | Avg Tokens/Doc | Dimension | Single-Vector | Multi-Vector |
|-----------|---------------|-----------|---------------|-------------|
| 10,000 | 50 | 768 | 29 MB | 1.4 GB |
| 100,000 | 50 | 768 | 290 MB | 14 GB |
| 1,000,000 | 50 | 768 | 2.9 GB | 140 GB |

!!! tip "Storage Optimization"
    For production deployments with large collections, consider:

    - Reducing embedding dimension via projection
    - Quantizing embeddings to int8
    - Using a first-stage retriever (BM25 or dense) followed by late interaction re-ranking

---

## See Also

- [Embeddings Guide](../guide/embeddings.md) - Single-vector embedding basics
- [Parallel Processing](parallel.md) - Accelerate scoring with parallelism
- [SIMD Optimizations](simd.md) - Hardware-accelerated operations
