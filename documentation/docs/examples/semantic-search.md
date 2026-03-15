---
title: "Tutorial: Semantic Search"
description: Build a semantic search engine with embedding-based retrieval, similarity ranking, and hybrid search using Mullama.
---

# Semantic Search

Build a semantic search engine from scratch. Unlike keyword search, semantic search understands meaning -- "automobile" matches "car," and "happy" matches "joyful." This tutorial covers document ingestion, chunking strategies, embedding generation, similarity ranking, and hybrid search.

---

## What You'll Build

A semantic search system that:

- Ingests documents with multiple chunking strategies
- Generates embeddings in batch for performance
- Stores vectors in an in-memory index
- Ranks results by cosine similarity with configurable thresholds
- Supports query-time filtering
- Extends to hybrid search (keyword + semantic)

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- An embedding-capable GGUF model (e.g., `nomic-embed-text`)
- Node.js 16+ or Python 3.8+ (with NumPy)

```bash
mullama pull nomic-embed-text
```

---

## Architecture

```
Indexing Phase:
Documents --> Chunking --> Batch Embed --> Store (vectors + metadata)

Query Phase:
Query --> Embed --> Similarity Search --> Rank --> Filter --> Results
```

---

## Step 1: Document Ingestion

Load and prepare documents with metadata for later filtering.

=== "Node.js"
    ```javascript
    const { JsModel, JsEmbeddingGenerator, cosineSimilarity } = require('mullama');

    // Sample document corpus
    const documents = [
        {
            id: 'doc1', title: 'Introduction to Rust',
            content: 'Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It is designed for performance and safety, particularly safe concurrency.',
            category: 'programming'
        },
        {
            id: 'doc2', title: 'Python for Data Science',
            content: 'Python has become the dominant language for data science and machine learning. Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools for data manipulation, analysis, and predictive modeling.',
            category: 'programming'
        },
        {
            id: 'doc3', title: 'Neural Network Basics',
            content: 'Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information using weighted connections. Training adjusts these weights to minimize prediction errors.',
            category: 'ai'
        },
        {
            id: 'doc4', title: 'Cloud Computing Overview',
            content: 'Cloud computing delivers computing services over the internet including servers, storage, databases, networking, and software. Major providers include AWS, Azure, and Google Cloud Platform.',
            category: 'infrastructure'
        },
        {
            id: 'doc5', title: 'Transformer Architecture',
            content: 'The transformer architecture uses self-attention mechanisms to process sequences in parallel. Unlike RNNs, transformers can attend to all positions simultaneously, enabling better modeling of long-range dependencies in language.',
            category: 'ai'
        },
        {
            id: 'doc6', title: 'Database Design Principles',
            content: 'Relational databases organize data into tables with rows and columns. Normalization reduces redundancy, while indexes speed up queries. ACID properties ensure transaction reliability in concurrent environments.',
            category: 'infrastructure'
        },
    ];
    ```

=== "Python"
    ```python
    from mullama import Model, EmbeddingGenerator, cosine_similarity
    import numpy as np
    import time

    # Sample document corpus
    documents = [
        {
            "id": "doc1", "title": "Introduction to Rust",
            "content": "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It is designed for performance and safety, particularly safe concurrency.",
            "category": "programming"
        },
        {
            "id": "doc2", "title": "Python for Data Science",
            "content": "Python has become the dominant language for data science and machine learning. Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools for data manipulation, analysis, and predictive modeling.",
            "category": "programming"
        },
        {
            "id": "doc3", "title": "Neural Network Basics",
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information using weighted connections. Training adjusts these weights to minimize prediction errors.",
            "category": "ai"
        },
        {
            "id": "doc4", "title": "Cloud Computing Overview",
            "content": "Cloud computing delivers computing services over the internet including servers, storage, databases, networking, and software. Major providers include AWS, Azure, and Google Cloud Platform.",
            "category": "infrastructure"
        },
        {
            "id": "doc5", "title": "Transformer Architecture",
            "content": "The transformer architecture uses self-attention mechanisms to process sequences in parallel. Unlike RNNs, transformers can attend to all positions simultaneously, enabling better modeling of long-range dependencies in language.",
            "category": "ai"
        },
        {
            "id": "doc6", "title": "Database Design Principles",
            "content": "Relational databases organize data into tables with rows and columns. Normalization reduces redundancy, while indexes speed up queries. ACID properties ensure transaction reliability in concurrent environments.",
            "category": "infrastructure"
        },
    ]
    ```

---

## Step 2: Chunking Strategies

Choose a chunking strategy based on your document type.

=== "Node.js"
    ```javascript
    // Strategy 1: Fixed-size word chunks with overlap
    function chunkByWords(text, chunkSize = 100, overlap = 20) {
        const words = text.split(/\s+/);
        const chunks = [];
        for (let i = 0; i < words.length; i += chunkSize - overlap) {
            const chunk = words.slice(i, i + chunkSize).join(' ');
            if (chunk.trim()) chunks.push(chunk);
        }
        return chunks;
    }

    // Strategy 2: Sentence-based chunking
    function chunkBySentences(text, maxSentences = 3) {
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const chunks = [];
        for (let i = 0; i < sentences.length; i += maxSentences) {
            chunks.push(sentences.slice(i, i + maxSentences).join(' ').trim());
        }
        return chunks;
    }

    // Strategy 3: Paragraph-based chunking
    function chunkByParagraphs(text) {
        return text.split(/\n\n+/).filter(p => p.trim().length > 20);
    }

    // Apply chunking to documents
    const chunks = [];
    for (const doc of documents) {
        const docChunks = chunkBySentences(doc.content, 2);
        for (const chunk of docChunks) {
            chunks.push({
                id: `${doc.id}_${chunks.length}`,
                content: chunk,
                title: doc.title,
                category: doc.category,
                docId: doc.id,
            });
        }
    }
    console.log(`Created ${chunks.length} chunks from ${documents.length} documents`);
    ```

=== "Python"
    ```python
    import re

    # Strategy 1: Fixed-size word chunks with overlap
    def chunk_by_words(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # Strategy 2: Sentence-based chunking
    def chunk_by_sentences(text: str, max_sentences: int = 3) -> list[str]:
        sentences = re.findall(r'[^.!?]+[.!?]+', text) or [text]
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    # Strategy 3: Paragraph-based chunking
    def chunk_by_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

    # Apply chunking to documents
    chunks = []
    for doc in documents:
        doc_chunks = chunk_by_sentences(doc["content"], max_sentences=2)
        for chunk in doc_chunks:
            chunks.append({
                "id": f'{doc["id"]}_{len(chunks)}',
                "content": chunk,
                "title": doc["title"],
                "category": doc["category"],
                "doc_id": doc["id"],
            })
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    ```

---

## Step 3: Batch Embedding Generation

Embed all chunks efficiently using batch processing.

=== "Node.js"
    ```javascript
    // Load embedding model
    const embedModel = JsModel.load('./nomic-embed-text.Q4_K_M.gguf');
    const embedGen = new JsEmbeddingGenerator(embedModel, 512, true);

    console.log(`Embedding model loaded (dim=${embedGen.nEmbd})`);

    // Batch embed all chunks
    const startTime = Date.now();
    const texts = chunks.map(c => c.content);
    const embeddings = embedGen.embedBatch(texts);
    const elapsed = (Date.now() - startTime) / 1000;

    console.log(`Embedded ${chunks.length} chunks in ${elapsed.toFixed(2)}s`);
    console.log(`Throughput: ${(chunks.length / elapsed).toFixed(1)} chunks/sec`);
    ```

=== "Python"
    ```python
    # Load embedding model
    embed_model = Model.load("./nomic-embed-text.Q4_K_M.gguf")
    embed_gen = EmbeddingGenerator(embed_model, n_ctx=512, normalize=True)

    print(f"Embedding model loaded (dim={embed_gen.n_embd})")

    # Batch embed all chunks
    start_time = time.time()
    texts = [c["content"] for c in chunks]
    embeddings = embed_gen.embed_batch(texts)
    elapsed = time.time() - start_time

    print(f"Embedded {len(chunks)} chunks in {elapsed:.2f}s")
    print(f"Throughput: {len(chunks) / elapsed:.1f} chunks/sec")
    ```

---

## Step 4: Search Index

Build a search index with metadata filtering.

=== "Node.js"
    ```javascript
    class SearchIndex {
        constructor() {
            this.entries = [];  // { chunk, embedding }
        }

        addBatch(chunks, embeddings) {
            for (let i = 0; i < chunks.length; i++) {
                this.entries.push({ chunk: chunks[i], embedding: embeddings[i] });
            }
        }

        search(queryEmbedding, options = {}) {
            const { topK = 5, threshold = 0.0, category = null } = options;

            let results = this.entries.map(entry => ({
                chunk: entry.chunk,
                score: cosineSimilarity(queryEmbedding, entry.embedding)
            }));

            // Apply category filter
            if (category) {
                results = results.filter(r => r.chunk.category === category);
            }

            // Apply threshold and sort
            return results
                .filter(r => r.score >= threshold)
                .sort((a, b) => b.score - a.score)
                .slice(0, topK);
        }

        get size() { return this.entries.length; }
    }

    // Build index
    const index = new SearchIndex();
    index.addBatch(chunks, embeddings);
    console.log(`Index built: ${index.size} entries`);
    ```

=== "Python"
    ```python
    class SearchIndex:
        def __init__(self):
            self.entries = []  # list of {"chunk": ..., "embedding": ...}

        def add_batch(self, chunks: list, embeddings: list):
            for chunk, emb in zip(chunks, embeddings):
                self.entries.append({"chunk": chunk, "embedding": emb})

        def search(self, query_embedding, top_k=5, threshold=0.0, category=None):
            results = [
                {"chunk": e["chunk"], "score": cosine_similarity(query_embedding, e["embedding"])}
                for e in self.entries
            ]

            # Apply category filter
            if category:
                results = [r for r in results if r["chunk"]["category"] == category]

            # Apply threshold and sort
            results = [r for r in results if r["score"] >= threshold]
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        @property
        def size(self):
            return len(self.entries)

    # Build index
    index = SearchIndex()
    index.add_batch(chunks, embeddings)
    print(f"Index built: {index.size} entries")
    ```

---

## Step 5: Query and Rank

Search the index with natural language queries.

=== "Node.js"
    ```javascript
    function search(query, options = {}) {
        const queryEmbedding = embedGen.embed(query);
        const results = index.search(queryEmbedding, options);

        console.log(`\nQuery: "${query}"`);
        if (options.category) console.log(`Filter: category=${options.category}`);
        console.log(`Results (${results.length}):`);

        for (const result of results) {
            console.log(`  [${result.score.toFixed(4)}] ${result.chunk.title}`);
            console.log(`    "${result.chunk.content.slice(0, 80)}..."`);
        }

        return results;
    }

    // Search examples
    search('memory safety in programming');
    search('how do neural networks learn');
    search('cloud services and infrastructure');
    search('deep learning architectures', { category: 'ai' });
    search('fast programming language', { topK: 3, threshold: 0.4 });
    ```

=== "Python"
    ```python
    def search(query: str, **options):
        query_embedding = embed_gen.embed(query)
        results = index.search(query_embedding, **options)

        print(f'\nQuery: "{query}"')
        if options.get("category"):
            print(f"Filter: category={options['category']}")
        print(f"Results ({len(results)}):")

        for result in results:
            print(f'  [{result["score"]:.4f}] {result["chunk"]["title"]}')
            print(f'    "{result["chunk"]["content"][:80]}..."')

        return results

    # Search examples
    search("memory safety in programming")
    search("how do neural networks learn")
    search("cloud services and infrastructure")
    search("deep learning architectures", category="ai")
    search("fast programming language", top_k=3, threshold=0.4)
    ```

---

## Complete Working Example

=== "Node.js"
    ```javascript
    const { JsModel, JsEmbeddingGenerator, cosineSimilarity } = require('mullama');

    // --- Configuration ---
    const MODEL_PATH = process.env.EMBED_MODEL || './nomic-embed-text.Q4_K_M.gguf';

    // --- Documents ---
    const documents = [
        { id: '1', title: 'Rust', content: 'Rust is a systems language focused on safety...', category: 'lang' },
        { id: '2', title: 'Python', content: 'Python is popular for data science and ML...', category: 'lang' },
        { id: '3', title: 'Transformers', content: 'Transformer architecture uses self-attention...', category: 'ai' },
        { id: '4', title: 'Databases', content: 'Relational databases organize data into tables...', category: 'infra' },
    ];

    // --- Chunk & Embed ---
    const model = JsModel.load(MODEL_PATH);
    const gen = new JsEmbeddingGenerator(model, 512, true);

    const chunks = documents.map(d => ({ ...d, text: d.content }));
    const embeddings = gen.embedBatch(chunks.map(c => c.content));
    console.log(`Indexed ${chunks.length} documents (dim=${gen.nEmbd})`);

    // --- Search ---
    class Index {
        constructor(docs, embs) { this.docs = docs; this.embs = embs; }
        search(qEmb, k = 3) {
            return this.embs
                .map((e, i) => ({ doc: this.docs[i], score: cosineSimilarity(qEmb, e) }))
                .sort((a, b) => b.score - a.score)
                .slice(0, k);
        }
    }

    const idx = new Index(chunks, embeddings);

    function query(text) {
        const qEmb = gen.embed(text);
        const results = idx.search(qEmb);
        console.log(`\n"${text}"`);
        results.forEach(r => console.log(`  [${r.score.toFixed(3)}] ${r.doc.title}: ${r.doc.content.slice(0, 60)}`));
    }

    query('memory safe programming language');
    query('machine learning tools');
    query('attention mechanisms in AI');
    query('storing structured data');
    ```

=== "Python"
    ```python
    from mullama import Model, EmbeddingGenerator, cosine_similarity
    import numpy as np

    # --- Configuration ---
    MODEL_PATH = "./nomic-embed-text.Q4_K_M.gguf"

    # --- Documents ---
    documents = [
        {"id": "1", "title": "Rust", "content": "Rust is a systems language focused on safety...", "category": "lang"},
        {"id": "2", "title": "Python", "content": "Python is popular for data science and ML...", "category": "lang"},
        {"id": "3", "title": "Transformers", "content": "Transformer architecture uses self-attention...", "category": "ai"},
        {"id": "4", "title": "Databases", "content": "Relational databases organize data into tables...", "category": "infra"},
    ]

    # --- Chunk & Embed ---
    model = Model.load(MODEL_PATH)
    gen = EmbeddingGenerator(model, n_ctx=512, normalize=True)

    embeddings = gen.embed_batch([d["content"] for d in documents])
    print(f"Indexed {len(documents)} documents (dim={gen.n_embd})")

    # --- Search ---
    class Index:
        def __init__(self, docs, embs):
            self.docs, self.embs = docs, embs

        def search(self, q_emb, k=3):
            scores = [(i, cosine_similarity(q_emb, e)) for i, e in enumerate(self.embs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [{"doc": self.docs[i], "score": s} for i, s in scores[:k]]

    idx = Index(documents, embeddings)

    def query(text: str):
        q_emb = gen.embed(text)
        results = idx.search(q_emb)
        print(f'\n"{text}"')
        for r in results:
            print(f'  [{r["score"]:.3f}] {r["doc"]["title"]}: {r["doc"]["content"][:60]}')

    query("memory safe programming language")
    query("machine learning tools")
    query("attention mechanisms in AI")
    query("storing structured data")
    ```

---

## Extension: Hybrid Search

Combine keyword matching with semantic search for better recall.

=== "Node.js"
    ```javascript
    function hybridSearch(query, index, embedGen, options = {}) {
        const { topK = 5, semanticWeight = 0.7, keywordWeight = 0.3 } = options;

        // Semantic search
        const qEmb = embedGen.embed(query);
        const semanticResults = index.search(qEmb, { topK: topK * 2 });

        // Keyword search (simple TF-IDF approximation)
        const queryTerms = query.toLowerCase().split(/\s+/);

        const hybridScores = semanticResults.map(result => {
            const content = result.chunk.content.toLowerCase();
            const keywordHits = queryTerms.filter(term => content.includes(term)).length;
            const keywordScore = keywordHits / queryTerms.length;

            return {
                ...result,
                semanticScore: result.score,
                keywordScore,
                hybridScore: result.score * semanticWeight + keywordScore * keywordWeight,
            };
        });

        return hybridScores
            .sort((a, b) => b.hybridScore - a.hybridScore)
            .slice(0, topK);
    }

    // Usage
    const results = hybridSearch('Rust memory safety', index, embedGen);
    results.forEach(r => {
        console.log(`[hybrid=${r.hybridScore.toFixed(3)} sem=${r.semanticScore.toFixed(3)} ` +
                    `kw=${r.keywordScore.toFixed(3)}] ${r.chunk.title}`);
    });
    ```

=== "Python"
    ```python
    def hybrid_search(query: str, index, embed_gen, top_k=5,
                      semantic_weight=0.7, keyword_weight=0.3):
        # Semantic search
        q_emb = embed_gen.embed(query)
        semantic_results = index.search(q_emb, top_k=top_k * 2)

        # Keyword search (simple TF-IDF approximation)
        query_terms = query.lower().split()

        hybrid_scores = []
        for result in semantic_results:
            content = result["chunk"]["content"].lower()
            keyword_hits = sum(1 for term in query_terms if term in content)
            keyword_score = keyword_hits / len(query_terms) if query_terms else 0

            hybrid_scores.append({
                **result,
                "semantic_score": result["score"],
                "keyword_score": keyword_score,
                "hybrid_score": result["score"] * semantic_weight + keyword_score * keyword_weight,
            })

        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_scores[:top_k]

    # Usage
    results = hybrid_search("Rust memory safety", index, embed_gen)
    for r in results:
        print(f'[hybrid={r["hybrid_score"]:.3f} sem={r["semantic_score"]:.3f} '
              f'kw={r["keyword_score"]:.3f}] {r["chunk"]["title"]}')
    ```

---

## Extension: ColBERT for Passage Retrieval

ColBERT uses per-token embeddings for fine-grained matching.

=== "Node.js"
    ```javascript
    function colbertRerank(query, candidates, embedGen, model) {
        // Tokenize query into individual terms
        const queryTokens = query.split(/\s+/).filter(t => t.length > 2);
        const queryEmbeddings = embedGen.embedBatch(queryTokens);

        return candidates.map(candidate => {
            // Tokenize document
            const docTokens = candidate.chunk.content.split(/\s+/).filter(t => t.length > 2);
            const docEmbeddings = embedGen.embedBatch(docTokens.slice(0, 50)); // Limit for speed

            // MaxSim scoring
            let totalSim = 0;
            for (const qEmb of queryEmbeddings) {
                let maxSim = -Infinity;
                for (const dEmb of docEmbeddings) {
                    const sim = cosineSimilarity(qEmb, dEmb);
                    if (sim > maxSim) maxSim = sim;
                }
                totalSim += maxSim;
            }

            return { ...candidate, colbertScore: totalSim / queryEmbeddings.length };
        }).sort((a, b) => b.colbertScore - a.colbertScore);
    }
    ```

=== "Python"
    ```python
    def colbert_rerank(query: str, candidates: list, embed_gen) -> list:
        """Rerank candidates using ColBERT-style MaxSim scoring."""
        query_tokens = [t for t in query.split() if len(t) > 2]
        query_embeddings = embed_gen.embed_batch(query_tokens)

        reranked = []
        for candidate in candidates:
            doc_tokens = [t for t in candidate["chunk"]["content"].split() if len(t) > 2]
            doc_embeddings = embed_gen.embed_batch(doc_tokens[:50])

            # MaxSim scoring
            total_sim = 0.0
            for q_emb in query_embeddings:
                max_sim = max(cosine_similarity(q_emb, d_emb) for d_emb in doc_embeddings)
                total_sim += max_sim

            reranked.append({**candidate, "colbert_score": total_sim / len(query_embeddings)})

        reranked.sort(key=lambda x: x["colbert_score"], reverse=True)
        return reranked
    ```

---

## Performance Benchmarks

Typical embedding throughput on different hardware:

| Hardware | Model | Throughput | Latency (single) |
|----------|-------|-----------|-------------------|
| Apple M2 | nomic-embed-text Q4 | ~120 texts/sec | ~8ms |
| RTX 4090 | nomic-embed-text Q4 | ~400 texts/sec | ~2.5ms |
| Intel i7-12700 | nomic-embed-text Q4 | ~60 texts/sec | ~16ms |
| Raspberry Pi 5 | nomic-embed-text Q4 | ~8 texts/sec | ~125ms |

!!! tip "Production Vector Stores"
    For production workloads with >10,000 documents, use a dedicated vector database:

    - **pgvector** -- PostgreSQL extension, great for existing Postgres users
    - **Qdrant** -- Open-source, rich filtering, gRPC API
    - **Pinecone** -- Managed service, zero ops
    - **Weaviate** -- GraphQL API, automatic vectorization

---

## What's Next

- [RAG Pipeline](rag.md) -- Add generation to your search results
- [Batch Processing](batch.md) -- Optimize embedding large collections
- [API Server](api-server.md) -- Serve search over HTTP
- [Edge Deployment](edge-deployment.md) -- Run search on constrained devices
