# Embeddings

Generate text embeddings for semantic search, similarity, and RAG applications.

## Basic Embeddings

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("embedding-model.gguf")?);
let params = ContextParams {
    embeddings: true,  // Enable embedding mode
    ..Default::default()
};
let mut context = Context::new(model, params)?;

let embedding = context.get_embeddings("Hello, world!")?;
println!("Embedding dimension: {}", embedding.len());
```

## Embedding Models

Recommended models for embeddings:

| Model | Dimension | Use Case |
|-------|-----------|----------|
| nomic-embed-text | 768 | General purpose |
| bge-base | 768 | Search/retrieval |
| bge-large | 1024 | High accuracy |
| e5-base | 768 | Multilingual |
| all-MiniLM-L6 | 384 | Fast/lightweight |

## Batch Embeddings

Embed multiple texts efficiently:

```rust
let texts = vec![
    "First document",
    "Second document",
    "Third document",
];

let embeddings: Vec<Vec<f32>> = texts
    .iter()
    .map(|text| context.get_embeddings(text))
    .collect::<Result<_, _>>()?;
```

## Similarity Search

Compute cosine similarity between embeddings:

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

let query_embedding = context.get_embeddings("What is machine learning?")?;

let documents = vec![
    "Machine learning is a subset of AI",
    "The weather is nice today",
    "Neural networks learn from data",
];

let mut results: Vec<(usize, f32)> = documents
    .iter()
    .enumerate()
    .map(|(i, doc)| {
        let emb = context.get_embeddings(doc).unwrap();
        let sim = cosine_similarity(&query_embedding, &emb);
        (i, sim)
    })
    .collect();

results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("Most relevant: {}", documents[results[0].0]);
```

## Semantic Search

Build a simple semantic search engine:

```rust
struct Document {
    id: usize,
    text: String,
    embedding: Vec<f32>,
}

struct SearchIndex {
    documents: Vec<Document>,
    context: Context,
}

impl SearchIndex {
    fn add_document(&mut self, text: &str) -> Result<usize, mullama::MullamaError> {
        let embedding = self.context.get_embeddings(text)?;
        let id = self.documents.len();

        self.documents.push(Document {
            id,
            text: text.to_string(),
            embedding,
        });

        Ok(id)
    }

    fn search(&self, query: &str, top_k: usize) -> Result<Vec<&Document>, mullama::MullamaError> {
        let query_emb = self.context.get_embeddings(query)?;

        let mut scored: Vec<_> = self.documents
            .iter()
            .map(|doc| {
                let sim = cosine_similarity(&query_emb, &doc.embedding);
                (doc, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored.into_iter().take(top_k).map(|(doc, _)| doc).collect())
    }
}
```

## RAG (Retrieval-Augmented Generation)

Combine embeddings with generation:

```rust
struct RagSystem {
    embedding_ctx: Context,
    generation_ctx: Context,
    documents: Vec<Document>,
}

impl RagSystem {
    fn answer(&mut self, question: &str) -> Result<String, mullama::MullamaError> {
        // 1. Find relevant documents
        let query_emb = self.embedding_ctx.get_embeddings(question)?;
        let relevant = self.find_relevant(&query_emb, 3);

        // 2. Build context
        let context = relevant
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        // 3. Generate answer
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, question
        );

        self.generation_ctx.generate(&prompt, 200)
    }

    fn find_relevant(&self, query_emb: &[f32], top_k: usize) -> Vec<&Document> {
        let mut scored: Vec<_> = self.documents
            .iter()
            .map(|doc| (doc, cosine_similarity(query_emb, &doc.embedding)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(top_k).map(|(d, _)| d).collect()
    }
}
```

## Normalization

Normalize embeddings for consistent similarity:

```rust
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

let mut embedding = context.get_embeddings("text")?;
normalize(&mut embedding);
```

## Pooling Strategies

Different pooling for different models:

```rust
enum PoolingStrategy {
    Mean,      // Average all token embeddings
    Max,       // Max pooling
    Cls,       // Use [CLS] token only
    Last,      // Use last token
}

fn pool_embeddings(token_embeddings: &[Vec<f32>], strategy: PoolingStrategy) -> Vec<f32> {
    match strategy {
        PoolingStrategy::Mean => {
            let dim = token_embeddings[0].len();
            let mut result = vec![0.0; dim];

            for emb in token_embeddings {
                for (i, v) in emb.iter().enumerate() {
                    result[i] += v;
                }
            }

            let n = token_embeddings.len() as f32;
            result.iter_mut().for_each(|v| *v /= n);
            result
        }
        PoolingStrategy::Cls => token_embeddings[0].clone(),
        PoolingStrategy::Last => token_embeddings.last().unwrap().clone(),
        PoolingStrategy::Max => {
            let dim = token_embeddings[0].len();
            let mut result = vec![f32::MIN; dim];

            for emb in token_embeddings {
                for (i, v) in emb.iter().enumerate() {
                    if *v > result[i] {
                        result[i] = *v;
                    }
                }
            }
            result
        }
    }
}
```

## Caching Embeddings

Cache embeddings for performance:

```rust
use std::collections::HashMap;

struct EmbeddingCache {
    cache: HashMap<String, Vec<f32>>,
    context: Context,
}

impl EmbeddingCache {
    fn get_or_compute(&mut self, text: &str) -> Result<&Vec<f32>, mullama::MullamaError> {
        if !self.cache.contains_key(text) {
            let embedding = self.context.get_embeddings(text)?;
            self.cache.insert(text.to_string(), embedding);
        }
        Ok(self.cache.get(text).unwrap())
    }
}
```

## Best Practices

1. **Use embedding models** - Not all LLMs are good for embeddings
2. **Normalize embeddings** - For consistent cosine similarity
3. **Batch operations** - Process multiple texts together
4. **Cache results** - Embeddings are deterministic
5. **Choose appropriate dimensions** - Balance accuracy vs speed
