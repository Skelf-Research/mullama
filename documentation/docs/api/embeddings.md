---
title: Embeddings API
description: Text embedding generation, similarity computation, and pooling strategies
---

# Embeddings API

The embeddings module provides comprehensive support for generating text embeddings, computing similarities, and applying various pooling strategies. Embeddings are vector representations of text useful for semantic search, clustering, classification, and retrieval-augmented generation (RAG).

## Embeddings Struct

Holds generated embedding vectors with dimension metadata.

```rust
#[derive(Debug, Clone)]
pub struct Embeddings {
    pub data: Vec<f32>,
    pub dimension: usize,
}
```

### Fields

| Name | Type | Description |
|------|------|-------------|
| `data` | `Vec<f32>` | Flat array of embedding values (all vectors concatenated) |
| `dimension` | `usize` | Embedding dimension per vector |

### Methods

#### `new`

```rust
pub fn new(data: Vec<f32>, dimension: usize) -> Self
```

Create embeddings from raw data and dimension. The data length must be a multiple of dimension.

#### `get`

```rust
pub fn get(&self, index: usize) -> Option<&[f32]>
```

Get the embedding vector at the specified index. Returns `None` if index is out of bounds.

| Name | Type | Description |
|------|------|-------------|
| `index` | `usize` | Embedding index (for batch results containing multiple vectors) |

#### `len`

```rust
pub fn len(&self) -> usize
```

Returns the number of embedding vectors stored (data.len() / dimension).

#### `is_empty`

```rust
pub fn is_empty(&self) -> bool
```

Check if there are no embeddings.

#### `as_slice`

```rust
pub fn as_slice(&self) -> &[f32]
```

Get all embedding data as a flat slice.

#### `to_vecs`

```rust
pub fn to_vecs(&self) -> Vec<Vec<f32>>
```

Convert to a list of individual embedding vectors.

**Example:**

```rust
use mullama::embedding::Embeddings;

let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
let embeddings = Embeddings::new(data, 3); // 2 embeddings of dimension 3

assert_eq!(embeddings.len(), 2);
assert_eq!(embeddings.get(0), Some(&[0.1, 0.2, 0.3][..]));
assert_eq!(embeddings.get(1), Some(&[0.4, 0.5, 0.6][..]));

let vecs = embeddings.to_vecs();
assert_eq!(vecs.len(), 2);
assert_eq!(vecs[0], vec![0.1, 0.2, 0.3]);
```

## PoolingStrategy

Controls how individual token embeddings are combined into a single sequence-level embedding vector.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolingStrategy {
    /// Use the last token's embedding (default for causal/GPT-style models)
    Last,
    /// Average all token embeddings (best for sentence similarity)
    Mean,
    /// Use the first token's embedding (CLS token for BERT-like models)
    First,
    /// Max pooling across all dimensions (captures strongest features)
    Max,
    /// Use llama.cpp's native pooling (based on model/context settings)
    Native,
}
```

### Strategy Comparison

| Strategy | Best For | Description | Models |
|----------|----------|-------------|--------|
| `Last` | Causal LMs | Uses the final token's hidden state | GPT, Llama |
| `Mean` | Sentence similarity | Averages all token embeddings equally | Sentence transformers |
| `First` | BERT-style models | Uses the [CLS] token position (index 0) | BERT, RoBERTa |
| `Max` | Feature detection | Takes the maximum value per dimension | Classification tasks |
| `Native` | Model-specific | Defers to llama.cpp's built-in pooling logic | Any model |

## EmbeddingConfig

Configuration for the embedding generation process.

```rust
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub pooling: PoolingStrategy,
    pub normalize: bool,
    pub batch_size: usize,
}
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pooling` | `PoolingStrategy` | `Native` | Pooling strategy for combining token embeddings into sequence embedding |
| `normalize` | `bool` | `true` | L2-normalize embeddings to unit length (required for cosine similarity) |
| `batch_size` | `usize` | `32` | Number of texts to process in each batch for `embed_batch` |

## EmbeddingGenerator

The primary interface for generating embeddings from text. Manages its own context configured for embedding mode.

```rust
pub struct EmbeddingGenerator {
    model: Arc<Model>,
    context: Context,
    config: EmbeddingConfig,
}
```

### `EmbeddingGenerator::new`

Create a new embedding generator. Automatically configures the context with `embeddings: true`.

```rust
pub fn new(model: Arc<Model>, config: EmbeddingConfig) -> Result<Self, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Arc<Model>` | -- | Model to use for embedding generation |
| `config` | `EmbeddingConfig` | -- | Configuration for embedding behavior |

**Errors:** `MullamaError::ContextError` -- Failed to create embedding context.

**Example:**

```rust
use mullama::{Model, embedding::{EmbeddingGenerator, EmbeddingConfig, PoolingStrategy}};
use std::sync::Arc;

let model = Arc::new(Model::load("nomic-embed.gguf")?);
let config = EmbeddingConfig {
    pooling: PoolingStrategy::Mean,
    normalize: true,
    batch_size: 32,
};
let mut generator = EmbeddingGenerator::new(model, config)?;
```

### `embed_text`

Generate an embedding vector for a single text input.

```rust
pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `&str` | -- | Input text to embed |

**Returns:** `Result<Vec<f32>, MullamaError>` -- The embedding vector with length equal to model's `n_embd`.

**Errors:**

- `MullamaError::EmbeddingError` -- Embedding generation failed
- `MullamaError::TokenizationError` -- Text tokenization failed

**Example:**

```rust
let embedding = generator.embed_text("Hello, world!")?;
println!("Embedding dimension: {}", embedding.len());
println!("First 5 values: {:?}", &embedding[..5]);
```

### `embed_batch`

Generate embeddings for multiple texts efficiently. Processes texts in batches according to `config.batch_size`.

```rust
pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MullamaError>
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `texts` | `&[&str]` | -- | Slice of texts to embed |

**Returns:** `Result<Vec<Vec<f32>>, MullamaError>` -- One embedding vector per input text.

**Example:**

```rust
let texts = &["Hello", "World", "Rust is great"];
let embeddings = generator.embed_batch(texts)?;

for (text, emb) in texts.iter().zip(embeddings.iter()) {
    println!("{}: {} dimensions", text, emb.len());
}
```

## Utility Functions

### `normalize`

L2-normalize an embedding vector to unit length. Required for meaningful cosine similarity comparisons.

```rust
pub fn normalize(embedding: &[f32]) -> Vec<f32>
```

**Example:**

```rust
use mullama::embedding::normalize;

let raw = vec![3.0, 4.0]; // magnitude = 5.0
let normalized = normalize(&raw);
// normalized = [0.6, 0.8], magnitude = 1.0

let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
assert!((magnitude - 1.0).abs() < 1e-6);
```

### `cosine_similarity`

Compute the cosine similarity between two embedding vectors.

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `a` | `&[f32]` | -- | First embedding vector |
| `b` | `&[f32]` | -- | Second embedding vector |

**Returns:** `f32` -- Similarity score from -1.0 (opposite meaning) to 1.0 (identical meaning).

!!! note "Pre-normalization"
    If embeddings are already L2-normalized (the default when `config.normalize = true`), cosine similarity reduces to a simple dot product, which is significantly faster.

**Example:**

```rust
use mullama::embedding::cosine_similarity;

let a = generator.embed_text("The cat sat on the mat")?;
let b = generator.embed_text("A feline rested on the rug")?;
let c = generator.embed_text("Quantum computing research")?;

let sim_ab = cosine_similarity(&a, &b);
let sim_ac = cosine_similarity(&a, &c);

println!("Similar sentences: {:.4}", sim_ab);   // High similarity (~0.8+)
println!("Different topics: {:.4}", sim_ac);     // Low similarity (~0.2)
```

### `dot_product`

Compute the dot product between two vectors. For normalized vectors, this equals cosine similarity.

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32
```

## Context Configuration for Embedding Mode

To generate embeddings directly (without `EmbeddingGenerator`), configure the context with `embeddings: true`:

```rust
use mullama::{Model, Context, ContextParams};
use std::sync::Arc;

let model = Arc::new(Model::load("embedding-model.gguf")?);

let params = ContextParams {
    embeddings: true,  // Required for embedding output
    n_ctx: 512,        // Shorter context is fine for embeddings
    ..Default::default()
};

let mut ctx = Context::new(model.clone(), params)?;

// Tokenize and decode
let tokens = model.tokenize("text to embed", true, false)?;
ctx.decode(&tokens)?;

// Access raw embeddings from context (model-dependent format)
```

!!! note "Embedding Models"
    Not all models produce useful embeddings. Use models specifically trained for embedding tasks (e.g., nomic-embed-text, all-MiniLM, BGE, E5) for best results. General-purpose chat models may produce embeddings but with lower quality for similarity tasks.

## Batch Embedding with Normalization

```rust
use mullama::{Model, embedding::{EmbeddingGenerator, EmbeddingConfig, PoolingStrategy, normalize}};
use std::sync::Arc;

let model = Arc::new(Model::load("nomic-embed.gguf")?);
let config = EmbeddingConfig {
    pooling: PoolingStrategy::Mean,
    normalize: true,   // Embeddings come pre-normalized
    batch_size: 64,    // Process 64 texts at a time
};
let mut generator = EmbeddingGenerator::new(model, config)?;

// Generate embeddings for a corpus
let corpus = vec![
    "Rust programming language",
    "Python data science",
    "JavaScript web development",
    "Go cloud infrastructure",
];

let embeddings = generator.embed_batch(
    &corpus.iter().map(|s| *s).collect::<Vec<_>>()
)?;

// All embeddings are unit-length when normalize=true
for emb in &embeddings {
    let mag: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mag - 1.0).abs() < 1e-5);
}
```

## Complete Example: Semantic Search

```rust
use mullama::{Model, embedding::{EmbeddingGenerator, EmbeddingConfig, PoolingStrategy, cosine_similarity}};
use std::sync::Arc;

fn main() -> Result<(), mullama::MullamaError> {
    let model = Arc::new(Model::load("nomic-embed.gguf")?);
    let config = EmbeddingConfig {
        pooling: PoolingStrategy::Mean,
        normalize: true,
        batch_size: 32,
    };
    let mut generator = EmbeddingGenerator::new(model, config)?;

    // Index documents
    let documents = vec![
        "Rust is a systems programming language focused on safety",
        "Python excels at data science and machine learning",
        "JavaScript powers interactive web applications",
        "Go is designed for scalable cloud services",
    ];

    let doc_embeddings = generator.embed_batch(
        &documents.iter().map(|s| *s).collect::<Vec<_>>()
    )?;

    // Search query
    let query = "What language is best for systems programming?";
    let query_embedding = generator.embed_text(query)?;

    // Rank by similarity
    let mut results: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_similarity(&query_embedding, doc_emb)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Query: {}", query);
    println!("Results:");
    for (idx, score) in results {
        println!("  {:.4}: {}", score, documents[idx]);
    }

    Ok(())
}
```
