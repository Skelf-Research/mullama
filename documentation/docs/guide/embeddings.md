# Embeddings

Generate vector representations of text for semantic search, similarity calculations, clustering, and Retrieval-Augmented Generation (RAG) pipelines.

## What Are Embeddings?

Embeddings are dense vector representations that capture the semantic meaning of text. Similar texts produce vectors that are close together in the embedding space, enabling applications like:

- **Semantic search** -- Find documents by meaning, not just keywords
- **RAG pipelines** -- Retrieve relevant context for LLM generation
- **Text clustering** -- Group similar documents together
- **Duplicate detection** -- Identify semantically similar content
- **Recommendation systems** -- Find related items based on descriptions

## Basic Embedding Generation

Generate embeddings from text:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';

    const model = await Model.load('./embedding-model.gguf');
    const context = new Context(model, { embeddings: true });

    const embedding = await context.getEmbedding("Hello, world!");
    console.log(`Dimension: ${embedding.dimension}`);
    console.log(`Vector (first 5): ${embedding.data.slice(0, 5)}`);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams

    model = Model.load("./embedding-model.gguf")
    context = Context(model, ContextParams(embeddings=True))

    embedding = context.get_embedding("Hello, world!")
    print(f"Dimension: {embedding.dimension}")
    print(f"Vector (first 5): {embedding.data[:5]}")
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};
    use std::sync::Arc;

    let model = Arc::new(Model::load("embedding-model.gguf")?);
    let params = ContextParams { embeddings: true, ..Default::default() };
    let mut context = Context::new(model, params)?;

    let embedding = context.get_embedding("Hello, world!")?;
    println!("Dimension: {}", embedding.dimension);
    println!("Vector (first 5): {:?}", &embedding.data[..5]);
    ```

=== "CLI"

    ```bash
    mullama embed llama3.2:1b "Hello, world!"
    ```

!!! info "Embedding Models"
    While general-purpose LLMs can produce embeddings, purpose-built embedding models (like `nomic-embed-text` or `mxbai-embed-large`) produce higher quality vectors for retrieval tasks.

## Pooling Strategies

Pooling determines how token-level embeddings are combined into a single document-level vector:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `Mean` | Average of all token embeddings | General purpose, most common |
| `Last` | Last token's embedding only | Autoregressive models |
| `First` | First token's (CLS) embedding | BERT-style models |
| `Max` | Element-wise max across tokens | Capturing dominant features |
| `Native` | Use model's built-in pooling | Purpose-built embedding models |

=== "Node.js"

    ```javascript
    import { Model, Context, PoolingStrategy } from 'mullama';

    const model = await Model.load('./embedding-model.gguf');
    const context = new Context(model, { embeddings: true });

    // Use mean pooling (default)
    const meanEmbed = await context.getEmbedding("Hello!", {
      pooling: PoolingStrategy.Mean,
    });

    // Use last token pooling (for autoregressive models)
    const lastEmbed = await context.getEmbedding("Hello!", {
      pooling: PoolingStrategy.Last,
    });
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams, PoolingStrategy

    model = Model.load("./embedding-model.gguf")
    context = Context(model, ContextParams(embeddings=True))

    # Use mean pooling (default)
    mean_embed = context.get_embedding("Hello!",
        pooling=PoolingStrategy.Mean)

    # Use last token pooling (for autoregressive models)
    last_embed = context.get_embedding("Hello!",
        pooling=PoolingStrategy.Last)
    ```

=== "Rust"

    ```rust
    use mullama::{Context, ContextParams, PoolingStrategy};

    let params = ContextParams { embeddings: true, ..Default::default() };
    let mut context = Context::new(model, params)?;

    // Use mean pooling (default)
    let mean_embed = context.get_embedding_with_pooling(
        "Hello!", PoolingStrategy::Mean
    )?;

    // Use last token pooling (for autoregressive models)
    let last_embed = context.get_embedding_with_pooling(
        "Hello!", PoolingStrategy::Last
    )?;
    ```

=== "CLI"

    ```bash
    # Use mean pooling (default)
    mullama embed llama3.2:1b "Hello!" --pooling mean

    # Use last token pooling
    mullama embed llama3.2:1b "Hello!" --pooling last
    ```

!!! tip "Choosing a Pooling Strategy"
    - Use **Mean** for most embedding models and general-purpose tasks
    - Use **Last** when using an autoregressive (decoder-only) model for embeddings
    - Use **Native** when the model was specifically trained with a built-in pooling method

## Batch Embedding

Generate embeddings for multiple texts efficiently in a single call:

=== "Node.js"

    ```javascript
    const texts = [
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is a subset of artificial intelligence",
      "Rust is a systems programming language",
      "Python is popular for data science",
    ];

    const embeddings = await context.getEmbeddingBatch(texts);

    for (let i = 0; i < texts.length; i++) {
      console.log(`"${texts[i].substring(0, 30)}..." -> dim ${embeddings[i].dimension}`);
    }
    ```

=== "Python"

    ```python
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Rust is a systems programming language",
        "Python is popular for data science",
    ]

    embeddings = context.get_embedding_batch(texts)

    for text, emb in zip(texts, embeddings):
        print(f'"{text[:30]}..." -> dim {emb.dimension}')
    ```

=== "Rust"

    ```rust
    let texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Rust is a systems programming language",
        "Python is popular for data science",
    ];

    let embeddings = context.get_embedding_batch(&texts)?;

    for (text, emb) in texts.iter().zip(embeddings.iter()) {
        println!("\"{}...\" -> dim {}", &text[..30], emb.dimension);
    }
    ```

=== "CLI"

    ```bash
    # Batch embedding via REST API
    curl -X POST http://localhost:8080/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "model": "nomic-embed-text",
        "input": [
          "The quick brown fox",
          "Machine learning is AI",
          "Rust is a language"
        ]
      }'
    ```

!!! tip "Batch Performance"
    Batch embedding is significantly faster than processing texts individually. The overhead of context setup is amortized across all inputs. For large datasets, use batch sizes of 32-256 for optimal throughput.

## Similarity Calculations

Compare embeddings to find semantically similar texts:

=== "Node.js"

    ```javascript
    import { cosineSimilarity, dotProduct } from 'mullama';

    const embed1 = await context.getEmbedding("How do I cook pasta?");
    const embed2 = await context.getEmbedding("What is the recipe for spaghetti?");
    const embed3 = await context.getEmbedding("Tell me about quantum physics");

    // Cosine similarity (most common, range: -1 to 1)
    const sim12 = cosineSimilarity(embed1.data, embed2.data);
    const sim13 = cosineSimilarity(embed1.data, embed3.data);

    console.log(`Pasta vs Spaghetti: ${sim12.toFixed(4)}`);  // High similarity
    console.log(`Pasta vs Quantum: ${sim13.toFixed(4)}`);    // Low similarity

    // Dot product (for models trained with dot product similarity)
    const dot12 = dotProduct(embed1.data, embed2.data);
    console.log(`Dot product: ${dot12.toFixed(4)}`);
    ```

=== "Python"

    ```python
    import numpy as np
    from mullama import cosine_similarity, dot_product

    embed1 = context.get_embedding("How do I cook pasta?")
    embed2 = context.get_embedding("What is the recipe for spaghetti?")
    embed3 = context.get_embedding("Tell me about quantum physics")

    # Cosine similarity (most common, range: -1 to 1)
    sim12 = cosine_similarity(embed1.data, embed2.data)
    sim13 = cosine_similarity(embed1.data, embed3.data)

    print(f"Pasta vs Spaghetti: {sim12:.4f}")  # High similarity
    print(f"Pasta vs Quantum: {sim13:.4f}")    # Low similarity

    # Or use numpy directly
    vec1 = np.array(embed1.data)
    vec2 = np.array(embed2.data)
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    ```

=== "Rust"

    ```rust
    use mullama::embedding::{cosine_similarity, dot_product};

    let embed1 = context.get_embedding("How do I cook pasta?")?;
    let embed2 = context.get_embedding("What is the recipe for spaghetti?")?;
    let embed3 = context.get_embedding("Tell me about quantum physics")?;

    // Cosine similarity (most common, range: -1 to 1)
    let sim12 = cosine_similarity(&embed1.data, &embed2.data);
    let sim13 = cosine_similarity(&embed1.data, &embed3.data);

    println!("Pasta vs Spaghetti: {:.4}", sim12);  // High similarity
    println!("Pasta vs Quantum: {:.4}", sim13);    // Low similarity

    // Dot product
    let dot12 = dot_product(&embed1.data, &embed2.data);
    println!("Dot product: {:.4}", dot12);
    ```

=== "CLI"

    ```bash
    # Compare two texts
    mullama embed llama3.2:1b --compare \
      "How do I cook pasta?" \
      "What is the recipe for spaghetti?"
    ```

## Normalization

Normalize embeddings to unit length for efficient cosine similarity computation:

=== "Node.js"

    ```javascript
    import { normalize } from 'mullama';

    const embedding = await context.getEmbedding("Hello!");

    // Normalize to unit vector (L2 norm = 1)
    const normalized = normalize(embedding.data);

    // After normalization, dot product equals cosine similarity
    const sim = dotProduct(normalizedA, normalizedB);
    ```

=== "Python"

    ```python
    import numpy as np

    embedding = context.get_embedding("Hello!")

    # Normalize to unit vector (L2 norm = 1)
    vec = np.array(embedding.data)
    normalized = vec / np.linalg.norm(vec)

    # After normalization, dot product equals cosine similarity
    sim = np.dot(normalized_a, normalized_b)
    ```

=== "Rust"

    ```rust
    use mullama::embedding::normalize;

    let embedding = context.get_embedding("Hello!")?;

    // Normalize to unit vector (L2 norm = 1)
    let normalized = normalize(&embedding.data);

    // After normalization, dot product equals cosine similarity
    let sim = dot_product(&normalized_a, &normalized_b);
    ```

=== "CLI"

    ```bash
    # Normalize output
    mullama embed llama3.2:1b "Hello!" --normalize
    ```

!!! info "When to Normalize"
    Pre-normalizing embeddings before storage makes similarity calculations faster (simple dot product instead of full cosine similarity). Most vector databases can normalize on ingestion.

## Use with Vector Databases

Store and query embeddings with popular vector databases:

=== "Node.js"

    ```javascript
    import { Model, Context } from 'mullama';
    import { ChromaClient } from 'chromadb';

    // Generate embeddings
    const model = await Model.load('./nomic-embed-text.gguf');
    const context = new Context(model, { embeddings: true });

    // Store in ChromaDB
    const chroma = new ChromaClient();
    const collection = await chroma.createCollection({ name: 'docs' });

    const documents = [
      "Rust is a systems programming language",
      "Python is great for data science",
      "JavaScript runs in the browser",
    ];

    const embeddings = await context.getEmbeddingBatch(documents);

    await collection.add({
      ids: documents.map((_, i) => `doc_${i}`),
      embeddings: embeddings.map(e => e.data),
      documents: documents,
    });

    // Query
    const queryEmbed = await context.getEmbedding("What language for web?");
    const results = await collection.query({
      queryEmbeddings: [queryEmbed.data],
      nResults: 2,
    });
    console.log(results.documents);
    ```

=== "Python"

    ```python
    from mullama import Model, Context, ContextParams
    import chromadb

    # Generate embeddings
    model = Model.load("./nomic-embed-text.gguf")
    context = Context(model, ContextParams(embeddings=True))

    # Store in ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("docs")

    documents = [
        "Rust is a systems programming language",
        "Python is great for data science",
        "JavaScript runs in the browser",
    ]

    embeddings = context.get_embedding_batch(documents)

    collection.add(
        ids=[f"doc_{i}" for i in range(len(documents))],
        embeddings=[e.data for e in embeddings],
        documents=documents,
    )

    # Query
    query_embed = context.get_embedding("What language for web?")
    results = collection.query(
        query_embeddings=[query_embed.data],
        n_results=2,
    )
    print(results["documents"])
    ```

=== "Rust"

    ```rust
    use mullama::{Model, Context, ContextParams};

    let model = Arc::new(Model::load("nomic-embed-text.gguf")?);
    let params = ContextParams { embeddings: true, ..Default::default() };
    let mut context = Context::new(model, params)?;

    let documents = vec![
        "Rust is a systems programming language",
        "Python is great for data science",
        "JavaScript runs in the browser",
    ];

    let embeddings = context.get_embedding_batch(&documents)?;

    // Store in your vector database of choice
    for (doc, emb) in documents.iter().zip(embeddings.iter()) {
        vector_db.insert(doc, &emb.data)?;
    }

    // Query
    let query_embed = context.get_embedding("What language for web?")?;
    let results = vector_db.search(&query_embed.data, 2)?;
    ```

=== "CLI"

    ```bash
    # Generate embeddings as JSON for piping to a database
    mullama embed nomic-embed-text "Rust is a systems language" --format json
    ```

## Dimension and Model Compatibility

Different embedding models produce vectors of different dimensions:

| Model | Dimensions | Quality | Use Case |
|-------|-----------|---------|----------|
| nomic-embed-text | 768 | Excellent | General text |
| mxbai-embed-large | 1024 | Excellent | High-quality retrieval |
| all-minilm | 384 | Good | Lightweight, fast |
| snowflake-arctic | 1024 | Excellent | Code + text |

!!! warning "Dimension Mismatch"
    Embeddings from different models are not compatible. Always use the same model for both indexing and querying. Mixing models will produce meaningless similarity scores.

## Performance: Batch Size vs Throughput

Batch processing significantly improves embedding throughput:

| Batch Size | Texts/Second (CPU) | Texts/Second (GPU) |
|-----------|-------------------|-------------------|
| 1 | ~10 | ~50 |
| 8 | ~60 | ~300 |
| 32 | ~150 | ~800 |
| 128 | ~200 | ~1200 |
| 256 | ~210 | ~1300 |

!!! tip "Optimal Batch Size"
    Throughput plateaus beyond batch size 128-256. Larger batches consume more memory without meaningful speed gains. Start with 32 and increase if you need more throughput and have available memory.

## See Also

- [Text Generation](generation.md) -- Using the same models for text generation
- [Tutorials: RAG Pipeline](../examples/rag.md) -- End-to-end RAG implementation
- [Tutorials: Semantic Search](../examples/semantic-search.md) -- Building a search engine
- [API Reference: Embeddings](../api/embeddings.md) -- Complete Embeddings API documentation
