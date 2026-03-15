---
title: "Tutorial: RAG Pipeline"
description: Build a Retrieval-Augmented Generation pipeline with document embeddings, vector search, and grounded answers using Mullama.
---

# RAG Pipeline

Build a Retrieval-Augmented Generation system that answers questions using your own documents. The pipeline embeds documents, stores vectors, retrieves relevant context, and generates grounded answers.

---

## What You'll Build

A complete RAG system that:

- Ingests and chunks documents for embedding
- Generates embeddings with batch processing for efficiency
- Stores vectors in a simple in-memory vector store
- Retrieves relevant documents using cosine similarity
- Assembles context-aware prompts
- Generates answers grounded in retrieved documents

---

## Prerequisites

- Mullama installed (`npm install mullama` or `pip install mullama`)
- A GGUF model with embedding support (e.g., `nomic-embed-text` for embeddings, any instruct model for generation)
- Node.js 16+ or Python 3.8+ (with NumPy for Python)

```bash
mullama pull nomic-embed-text    # For embeddings
mullama pull llama3.2:1b         # For generation
```

---

## Architecture Overview

```
Documents --> Chunk --> Embed (batch) --> Vector Store
                                              |
User Query --> Embed -----------------------> Similarity Search --> Top-K Results
                                                                        |
                                                                        v
                                                                Context Assembly
                                                                        |
                                                                        v
                                                                LLM Generation --> Answer
```

---

## Step 1: Document Ingestion and Chunking

Split documents into overlapping chunks for better retrieval granularity.

=== "Node.js"
    ```javascript
    const { JsModel, JsEmbeddingGenerator } = require('mullama');

    function chunkText(text, chunkSize = 500, overlap = 50) {
        const words = text.split(/\s+/);
        const chunks = [];

        for (let i = 0; i < words.length; i += chunkSize - overlap) {
            const chunk = words.slice(i, i + chunkSize).join(' ');
            if (chunk.trim().length > 0) {
                chunks.push(chunk);
            }
        }
        return chunks;
    }

    // Sample documents
    const documents = [
        {
            title: "Rust Programming",
            content: `Rust is a systems programming language focused on safety,
                concurrency, and performance. It achieves memory safety without
                garbage collection through its ownership system. The borrow checker
                enforces strict rules about references and lifetimes at compile time.
                Rust's zero-cost abstractions allow high-level code to compile down
                to efficient machine code comparable to C and C++.`
        },
        {
            title: "Machine Learning Basics",
            content: `Machine learning is a subset of artificial intelligence that
                enables systems to learn from data without explicit programming.
                Supervised learning uses labeled datasets to train models for
                classification and regression. Unsupervised learning finds hidden
                patterns in unlabeled data through clustering and dimensionality
                reduction. Deep learning uses neural networks with many layers
                to learn complex representations of data.`
        },
        {
            title: "Large Language Models",
            content: `Large language models (LLMs) are neural networks trained on
                vast amounts of text data. They use the transformer architecture
                with self-attention mechanisms to capture long-range dependencies.
                LLMs can generate text, answer questions, summarize documents,
                and translate languages. Local inference with quantized models
                enables private, offline use of LLMs on consumer hardware.`
        },
    ];

    // Chunk all documents
    const chunks = [];
    for (const doc of documents) {
        const docChunks = chunkText(doc.content, 100, 20);
        for (const chunk of docChunks) {
            chunks.push({ title: doc.title, content: chunk });
        }
    }
    console.log(`Created ${chunks.length} chunks from ${documents.length} documents`);
    ```

=== "Python"
    ```python
    from mullama import Model, Context, SamplerParams, EmbeddingGenerator, cosine_similarity
    import numpy as np

    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # Sample documents
    documents = [
        {
            "title": "Rust Programming",
            "content": """Rust is a systems programming language focused on safety,
                concurrency, and performance. It achieves memory safety without
                garbage collection through its ownership system. The borrow checker
                enforces strict rules about references and lifetimes at compile time.
                Rust's zero-cost abstractions allow high-level code to compile down
                to efficient machine code comparable to C and C++."""
        },
        {
            "title": "Machine Learning Basics",
            "content": """Machine learning is a subset of artificial intelligence that
                enables systems to learn from data without explicit programming.
                Supervised learning uses labeled datasets to train models for
                classification and regression. Unsupervised learning finds hidden
                patterns in unlabeled data through clustering and dimensionality
                reduction. Deep learning uses neural networks with many layers
                to learn complex representations of data."""
        },
        {
            "title": "Large Language Models",
            "content": """Large language models (LLMs) are neural networks trained on
                vast amounts of text data. They use the transformer architecture
                with self-attention mechanisms to capture long-range dependencies.
                LLMs can generate text, answer questions, summarize documents,
                and translate languages. Local inference with quantized models
                enables private, offline use of LLMs on consumer hardware."""
        },
    ]

    # Chunk all documents
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["content"], chunk_size=100, overlap=20)
        for chunk in doc_chunks:
            chunks.append({"title": doc["title"], "content": chunk})

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    ```

---

## Step 2: Generate Embeddings (Batch)

Use the embedding generator to create vectors for all chunks efficiently.

=== "Node.js"
    ```javascript
    // Load embedding model
    const embedModel = JsModel.load('./nomic-embed-text.Q4_K_M.gguf');
    const embedGen = new JsEmbeddingGenerator(embedModel, 512, true); // normalize=true

    console.log(`Embedding dimension: ${embedGen.nEmbd}`);

    // Batch embed all chunks for efficiency
    const texts = chunks.map(c => c.content);
    console.log(`Embedding ${texts.length} chunks...`);

    const startTime = Date.now();
    const embeddings = embedGen.embedBatch(texts);
    const elapsed = (Date.now() - startTime) / 1000;

    console.log(`Embedded ${embeddings.length} chunks in ${elapsed.toFixed(2)}s`);
    console.log(`Throughput: ${(embeddings.length / elapsed).toFixed(1)} chunks/sec`);
    ```

=== "Python"
    ```python
    # Load embedding model
    embed_model = Model.load("./nomic-embed-text.Q4_K_M.gguf")
    embed_gen = EmbeddingGenerator(embed_model, n_ctx=512, normalize=True)

    print(f"Embedding dimension: {embed_gen.n_embd}")

    # Batch embed all chunks for efficiency
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    import time
    start = time.time()
    embeddings = embed_gen.embed_batch(texts)
    elapsed = time.time() - start

    print(f"Embedded {len(embeddings)} chunks in {elapsed:.2f}s")
    print(f"Throughput: {len(embeddings) / elapsed:.1f} chunks/sec")
    ```

---

## Step 3: Vector Store

Build a simple in-memory vector store with cosine similarity search.

=== "Node.js"
    ```javascript
    const { cosineSimilarity } = require('mullama');

    class VectorStore {
        constructor() {
            this.documents = [];  // { title, content }
            this.embeddings = []; // number[][]
        }

        add(document, embedding) {
            this.documents.push(document);
            this.embeddings.push(embedding);
        }

        addBatch(documents, embeddings) {
            for (let i = 0; i < documents.length; i++) {
                this.add(documents[i], embeddings[i]);
            }
        }

        search(queryEmbedding, topK = 3, threshold = 0.0) {
            const scores = this.embeddings.map((emb, idx) => ({
                index: idx,
                score: cosineSimilarity(queryEmbedding, emb)
            }));

            return scores
                .filter(s => s.score >= threshold)
                .sort((a, b) => b.score - a.score)
                .slice(0, topK)
                .map(s => ({
                    document: this.documents[s.index],
                    score: s.score
                }));
        }

        get size() { return this.documents.length; }
    }

    // Index all chunks
    const store = new VectorStore();
    store.addBatch(chunks, embeddings);
    console.log(`Vector store: ${store.size} documents indexed`);
    ```

=== "Python"
    ```python
    class VectorStore:
        def __init__(self):
            self.documents = []   # list of dicts
            self.embeddings = []  # list of numpy arrays

        def add(self, document: dict, embedding: np.ndarray):
            self.documents.append(document)
            self.embeddings.append(embedding)

        def add_batch(self, documents: list, embeddings: list):
            for doc, emb in zip(documents, embeddings):
                self.add(doc, emb)

        def search(self, query_embedding: np.ndarray, top_k: int = 3,
                   threshold: float = 0.0) -> list[dict]:
            scores = [
                {"index": i, "score": cosine_similarity(query_embedding, emb)}
                for i, emb in enumerate(self.embeddings)
            ]

            results = [s for s in scores if s["score"] >= threshold]
            results.sort(key=lambda x: x["score"], reverse=True)

            return [
                {"document": self.documents[s["index"]], "score": s["score"]}
                for s in results[:top_k]
            ]

        @property
        def size(self):
            return len(self.documents)

    # Index all chunks
    store = VectorStore()
    store.add_batch(chunks, embeddings)
    print(f"Vector store: {store.size} documents indexed")
    ```

---

## Step 4: Query and Retrieve

Embed the user's query and find the most relevant chunks.

=== "Node.js"
    ```javascript
    function retrieveContext(query, store, embedGen, topK = 3) {
        // Embed the query
        const queryEmbedding = embedGen.embed(query);

        // Search for relevant chunks
        const results = store.search(queryEmbedding, topK, 0.3);

        console.log(`\nQuery: "${query}"`);
        console.log(`Found ${results.length} relevant chunks:`);
        for (const result of results) {
            console.log(`  [${result.score.toFixed(4)}] ${result.document.title}`);
        }

        return results;
    }

    // Example query
    const query = "How does Rust achieve memory safety?";
    const results = retrieveContext(query, store, embedGen);
    ```

=== "Python"
    ```python
    def retrieve_context(query: str, store: VectorStore,
                         embed_gen: EmbeddingGenerator, top_k: int = 3) -> list[dict]:
        # Embed the query
        query_embedding = embed_gen.embed(query)

        # Search for relevant chunks
        results = store.search(query_embedding, top_k=top_k, threshold=0.3)

        print(f'\nQuery: "{query}"')
        print(f"Found {len(results)} relevant chunks:")
        for result in results:
            print(f'  [{result["score"]:.4f}] {result["document"]["title"]}')

        return results

    # Example query
    query = "How does Rust achieve memory safety?"
    results = retrieve_context(query, store, embed_gen)
    ```

---

## Step 5: Generate Grounded Answer

Assemble the retrieved context into a prompt and generate an answer.

=== "Node.js"
    ```javascript
    function assembleRAGPrompt(query, results, maxContextChars = 2000) {
        let context = '';
        for (const result of results) {
            const docText = `[Source: ${result.document.title}]\n${result.document.content}\n\n`;
            if (context.length + docText.length > maxContextChars) break;
            context += docText;
        }

        return `Use the following documents to answer the question accurately.
    If the answer cannot be found in the documents, say "I don't have enough information."
    Cite the source document when possible.

    --- Documents ---
    ${context}
    --- End Documents ---

    Question: ${query}

    Answer:`;
    }

    function generateAnswer(query, results, genModel, genCtx) {
        const prompt = assembleRAGPrompt(query, results);

        // Use low temperature for factual responses
        const params = { temperature: 0.3, topK: 20, topP: 0.85 };

        console.log('\nGenerating answer...');
        const pieces = genCtx.generateStream(prompt, 300, params);
        let answer = '';
        process.stdout.write('Answer: ');
        for (const piece of pieces) {
            process.stdout.write(piece);
            answer += piece;
        }
        console.log('\n');
        return answer.trim();
    }

    // Load generation model
    const genModel = JsModel.load('./llama3.2-1b-instruct.Q4_K_M.gguf', { nGpuLayers: -1 });
    const genCtx = new JsContext(genModel, { nCtx: 4096, nBatch: 512 });

    const answer = generateAnswer(query, results, genModel, genCtx);
    ```

=== "Python"
    ```python
    def assemble_rag_prompt(query: str, results: list[dict],
                            max_context_chars: int = 2000) -> str:
        context = ""
        for result in results:
            doc_text = f'[Source: {result["document"]["title"]}]\n{result["document"]["content"]}\n\n'
            if len(context) + len(doc_text) > max_context_chars:
                break
            context += doc_text

        return f"""Use the following documents to answer the question accurately.
    If the answer cannot be found in the documents, say "I don't have enough information."
    Cite the source document when possible.

    --- Documents ---
    {context}
    --- End Documents ---

    Question: {query}

    Answer:"""

    def generate_answer(query: str, results: list[dict], gen_model, gen_ctx) -> str:
        prompt = assemble_rag_prompt(query, results)

        # Use low temperature for factual responses
        params = SamplerParams(temperature=0.3, top_k=20, top_p=0.85)

        print("\nGenerating answer...")
        pieces = gen_ctx.generate_stream(prompt, max_tokens=300, params=params)
        answer = ""
        print("Answer: ", end="", flush=True)
        for piece in pieces:
            print(piece, end="", flush=True)
            answer += piece
        print("\n")
        return answer.strip()

    # Load generation model
    gen_model = Model.load("./llama3.2-1b-instruct.Q4_K_M.gguf", n_gpu_layers=-1)
    gen_ctx = Context(gen_model, n_ctx=4096, n_batch=512)

    answer = generate_answer(query, results, gen_model, gen_ctx)
    ```

---

## Complete Working Example

=== "Node.js"
    ```javascript
    const { JsModel, JsContext, JsEmbeddingGenerator, cosineSimilarity } = require('mullama');

    // --- Configuration ---
    const EMBED_MODEL = process.env.EMBED_MODEL || './nomic-embed-text.Q4_K_M.gguf';
    const GEN_MODEL = process.env.GEN_MODEL || './llama3.2-1b-instruct.Q4_K_M.gguf';

    // --- Documents ---
    const documents = [
        { title: "Rust", content: "Rust achieves memory safety through ownership and borrowing..." },
        { title: "ML", content: "Machine learning enables systems to learn from data..." },
        { title: "LLMs", content: "Large language models use transformer architecture..." },
    ];

    // --- Chunk ---
    function chunkText(text, size = 100, overlap = 20) {
        const words = text.split(/\s+/);
        const chunks = [];
        for (let i = 0; i < words.length; i += size - overlap) {
            chunks.push(words.slice(i, i + size).join(' '));
        }
        return chunks.filter(c => c.trim());
    }

    const chunks = documents.flatMap(doc =>
        chunkText(doc.content).map(c => ({ title: doc.title, content: c }))
    );

    // --- Embed ---
    console.log('Loading embedding model...');
    const embedModel = JsModel.load(EMBED_MODEL);
    const embedGen = new JsEmbeddingGenerator(embedModel, 512, true);
    const embeddings = embedGen.embedBatch(chunks.map(c => c.content));
    console.log(`Indexed ${chunks.length} chunks (dim=${embedGen.nEmbd})`);

    // --- Store ---
    class VectorStore {
        constructor() { this.docs = []; this.embs = []; }
        addBatch(docs, embs) { this.docs.push(...docs); this.embs.push(...embs); }
        search(qEmb, k = 3) {
            return this.embs
                .map((e, i) => ({ doc: this.docs[i], score: cosineSimilarity(qEmb, e) }))
                .sort((a, b) => b.score - a.score)
                .slice(0, k);
        }
    }
    const store = new VectorStore();
    store.addBatch(chunks, embeddings);

    // --- Query ---
    console.log('\nLoading generation model...');
    const genModel = JsModel.load(GEN_MODEL, { nGpuLayers: -1 });
    const genCtx = new JsContext(genModel, { nCtx: 4096 });

    function rag(query) {
        const qEmb = embedGen.embed(query);
        const results = store.search(qEmb, 3);
        console.log(`\nQuery: "${query}"`);
        results.forEach(r => console.log(`  [${r.score.toFixed(3)}] ${r.doc.title}`));

        const context = results.map(r => `[${r.doc.title}] ${r.doc.content}`).join('\n\n');
        const prompt = `Documents:\n${context}\n\nQuestion: ${query}\nAnswer:`;

        const pieces = genCtx.generateStream(prompt, 200, { temperature: 0.3 });
        process.stdout.write('Answer: ');
        for (const p of pieces) process.stdout.write(p);
        console.log('\n');
        genCtx.clearCache();
    }

    rag("How does Rust handle memory safety?");
    rag("What is deep learning?");
    ```

=== "Python"
    ```python
    from mullama import Model, Context, SamplerParams, EmbeddingGenerator, cosine_similarity
    import numpy as np

    # --- Configuration ---
    EMBED_MODEL = "./nomic-embed-text.Q4_K_M.gguf"
    GEN_MODEL = "./llama3.2-1b-instruct.Q4_K_M.gguf"

    # --- Documents ---
    documents = [
        {"title": "Rust", "content": "Rust achieves memory safety through ownership and borrowing..."},
        {"title": "ML", "content": "Machine learning enables systems to learn from data..."},
        {"title": "LLMs", "content": "Large language models use transformer architecture..."},
    ]

    # --- Chunk ---
    def chunk_text(text, size=100, overlap=20):
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size - overlap) if words[i:i+size]]

    chunks = [
        {"title": doc["title"], "content": chunk}
        for doc in documents
        for chunk in chunk_text(doc["content"])
    ]

    # --- Embed ---
    print("Loading embedding model...")
    embed_model = Model.load(EMBED_MODEL)
    embed_gen = EmbeddingGenerator(embed_model, n_ctx=512, normalize=True)
    embeddings = embed_gen.embed_batch([c["content"] for c in chunks])
    print(f"Indexed {len(chunks)} chunks (dim={embed_gen.n_embd})")

    # --- Store ---
    class VectorStore:
        def __init__(self):
            self.docs, self.embs = [], []

        def add_batch(self, docs, embs):
            self.docs.extend(docs)
            self.embs.extend(embs)

        def search(self, q_emb, k=3):
            scores = [(i, cosine_similarity(q_emb, e)) for i, e in enumerate(self.embs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [{"doc": self.docs[i], "score": s} for i, s in scores[:k]]

    store = VectorStore()
    store.add_batch(chunks, embeddings)

    # --- Query ---
    print("\nLoading generation model...")
    gen_model = Model.load(GEN_MODEL, n_gpu_layers=-1)
    gen_ctx = Context(gen_model, n_ctx=4096)

    def rag(query: str):
        q_emb = embed_gen.embed(query)
        results = store.search(q_emb, k=3)
        print(f'\nQuery: "{query}"')
        for r in results:
            print(f'  [{r["score"]:.3f}] {r["doc"]["title"]}')

        context = "\n\n".join(f'[{r["doc"]["title"]}] {r["doc"]["content"]}' for r in results)
        prompt = f"Documents:\n{context}\n\nQuestion: {query}\nAnswer:"

        params = SamplerParams(temperature=0.3)
        pieces = gen_ctx.generate_stream(prompt, max_tokens=200, params=params)
        print("Answer: ", end="", flush=True)
        for p in pieces:
            print(p, end="", flush=True)
        print("\n")
        gen_ctx.clear_cache()

    rag("How does Rust handle memory safety?")
    rag("What is deep learning?")
    ```

---

## Extension: ColBERT Late Interaction

For better retrieval quality, use ColBERT-style late interaction scoring. Instead of a single vector per document, ColBERT uses per-token embeddings and computes MaxSim scores.

=== "Node.js"
    ```javascript
    // ColBERT-style: embed query tokens individually, score against doc tokens
    function colbertScore(queryTokenEmbs, docTokenEmbs) {
        let totalScore = 0;
        for (const qEmb of queryTokenEmbs) {
            let maxSim = -Infinity;
            for (const dEmb of docTokenEmbs) {
                const sim = cosineSimilarity(qEmb, dEmb);
                if (sim > maxSim) maxSim = sim;
            }
            totalScore += maxSim;
        }
        return totalScore / queryTokenEmbs.length;
    }
    ```

=== "Python"
    ```python
    def colbert_score(query_token_embs: list, doc_token_embs: list) -> float:
        """ColBERT MaxSim: for each query token, find max similarity with any doc token."""
        total_score = 0.0
        for q_emb in query_token_embs:
            max_sim = max(cosine_similarity(q_emb, d_emb) for d_emb in doc_token_embs)
            total_score += max_sim
        return total_score / len(query_token_embs)
    ```

---

## Extension: Structured Output for Citations

Generate answers with structured citations back to source documents.

=== "Node.js"
    ```javascript
    function ragWithCitations(query, results) {
        const prompt = `Answer the question using ONLY the provided sources.
    Format your answer as JSON: {"answer": "...", "sources": ["Source 1", "Source 2"]}

    Sources:
    ${results.map(r => `[${r.doc.title}]: ${r.doc.content}`).join('\n')}

    Question: ${query}
    JSON Response:`;

        const response = genCtx.generate(prompt, 300, { temperature: 0.1 });
        try {
            return JSON.parse(response.trim());
        } catch {
            return { answer: response.trim(), sources: [] };
        }
    }
    ```

=== "Python"
    ```python
    import json

    def rag_with_citations(query: str, results: list[dict]) -> dict:
        sources = "\n".join(f'[{r["doc"]["title"]}]: {r["doc"]["content"]}' for r in results)
        prompt = f"""Answer the question using ONLY the provided sources.
    Format your answer as JSON: {{"answer": "...", "sources": ["Source 1", "Source 2"]}}

    Sources:
    {sources}

    Question: {query}
    JSON Response:"""

        response = gen_ctx.generate(prompt, max_tokens=300,
                                    params=SamplerParams(temperature=0.1))
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"answer": response.strip(), "sources": []}
    ```

---

## Production Tips

!!! tip "Scaling Beyond In-Memory"
    For production workloads, replace the in-memory vector store with:

    - **pgvector** -- PostgreSQL extension for vector similarity search
    - **Pinecone** -- Managed vector database with filtering
    - **Qdrant** -- Open-source vector database with rich query API
    - **ChromaDB** -- Lightweight embedding database for prototyping

!!! tip "Chunking Strategies"
    - **Fixed-size**: Simple but may split sentences (use overlap to mitigate)
    - **Sentence-based**: Split on sentence boundaries for coherent chunks
    - **Paragraph-based**: Natural document structure
    - **Semantic**: Use embedding similarity to find natural break points
    - Typical chunk sizes: 200-500 tokens with 10-20% overlap

!!! note "Embedding Models"
    For best retrieval quality, use a dedicated embedding model rather than a general-purpose LLM. Models like `nomic-embed-text` or `bge-small-en` produce higher-quality embeddings than using an instruct model's hidden states.

---

## What's Next

- [Semantic Search](semantic-search.md) -- Deep dive into search ranking and hybrid approaches
- [Batch Processing](batch.md) -- Embed large document collections efficiently
- [API Server](api-server.md) -- Expose your RAG pipeline as a REST API
- [Edge Deployment](edge-deployment.md) -- Run RAG on resource-constrained devices
