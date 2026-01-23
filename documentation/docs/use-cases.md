---
title: Use Cases
description: Real-world applications and deployment patterns for Mullama
---

# Use Cases

Mullama's modular feature system enables a wide range of AI-powered applications. This guide covers common deployment patterns, each with a problem statement, solution architecture, code snippet, required features, and links to related tutorials.

---

## 1. Conversational AI

**Problem:** Build a responsive chatbot with streaming responses that maintains conversation context across turns, with real-time token delivery for a natural chat experience.

**Solution:** Use Mullama's streaming generation with KV-cache management for multi-turn conversations. Chat templates handle proper prompt formatting for instruction-tuned models.

**Architecture:**

```rust
use mullama::{Model, Context, ContextParams, SamplerParams, ChatMessage};
use mullama::streaming::StreamConfig;
use std::sync::Arc;

let model = Arc::new(Model::load("llama-3-8b-instruct.gguf")?);
let mut ctx = Context::new(model.clone(), ContextParams {
    n_ctx: 4096,
    ..Default::default()
})?;

let mut history: Vec<ChatMessage> = vec![
    ChatMessage { role: "system".into(), content: "You are a helpful assistant.".into() },
];

// Each turn: format prompt, tokenize, decode, stream response
loop {
    let user_input = get_user_input();
    history.push(ChatMessage { role: "user".into(), content: user_input });

    let prompt = model.apply_chat_template(&history, true)?;
    let tokens = model.tokenize(&prompt, false, true)?;

    ctx.clear_cache(); // Reset for new conversation context
    ctx.decode(&tokens)?;

    let mut response = String::new();
    let mut sampler = SamplerParams::default().build_chain(model.clone())?;
    for _ in 0..512 {
        let token = sampler.sample(&mut ctx, -1);
        sampler.accept(token);
        if model.token_is_eog(token) { break; }
        let text = model.token_to_str(token, 0, false)?;
        print!("{}", text); // Stream to user
        response.push_str(&text);
        ctx.decode_single(token)?;
    }

    history.push(ChatMessage { role: "assistant".into(), content: response });
}
```

**Features used:** Core (no additional features required for basic chat; add `streaming` for async streaming, `web` for HTTP API)

**Related:** [Streaming Guide](guide/streaming.md) | [Chatbot Example](examples/chatbot.md)

---

## 2. Document Q&A (RAG)

**Problem:** Answer questions about a large document corpus by finding relevant passages and generating grounded answers, avoiding hallucination by providing source context.

**Solution:** Use Mullama's embedding API to index documents, find similar passages via cosine similarity, then generate answers with retrieved context inserted into the prompt.

**Architecture:**

```rust
use mullama::{Model, Context, ContextParams};
use mullama::embedding::{EmbeddingGenerator, EmbeddingConfig, PoolingStrategy, cosine_similarity};
use std::sync::Arc;

// Step 1: Index documents with embedding model
let embed_model = Arc::new(Model::load("nomic-embed.gguf")?);
let config = EmbeddingConfig {
    pooling: PoolingStrategy::Mean,
    normalize: true,
    batch_size: 32,
};
let mut embedder = EmbeddingGenerator::new(embed_model, config)?;

let documents = load_documents("./docs/")?;
let doc_embeddings: Vec<Vec<f32>> = documents.iter()
    .map(|doc| embedder.embed_text(doc))
    .collect::<Result<_, _>>()?;

// Step 2: Query with embedding similarity
let query = "How does memory management work?";
let query_embedding = embedder.embed_text(query)?;

let mut ranked: Vec<(usize, f32)> = doc_embeddings.iter()
    .enumerate()
    .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
    .collect();
ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

// Step 3: Generate answer with top-3 context
let context_docs: String = ranked.iter().take(3)
    .map(|(i, _)| documents[*i].as_str())
    .collect::<Vec<_>>()
    .join("\n---\n");

let gen_model = Arc::new(Model::load("llama-3-8b-instruct.gguf")?);
let mut ctx = Context::new(gen_model.clone(), ContextParams::default())?;

let prompt = format!(
    "Based on the following context, answer the question.\n\n\
     Context:\n{}\n\nQuestion: {}\n\nAnswer:",
    context_docs, query
);
let tokens = gen_model.tokenize(&prompt, true, false)?;
let answer = ctx.generate(&tokens, 300)?;
```

**Features used:** Core (embeddings are in the core module)

**Related:** [Embeddings Guide](guide/embeddings.md) | [RAG Example](examples/rag.md)

---

## 3. Voice Interface

**Problem:** Build a voice-activated AI assistant that listens for speech, processes audio input, and generates spoken or text responses in real time.

**Solution:** Use Mullama's streaming audio processor for real-time voice capture with voice activity detection, then feed audio features into a multimodal model for transcription and response generation.

**Architecture:**

```rust
use mullama::streaming_audio::{StreamingAudioProcessor, AudioStreamConfig};
use mullama::multimodal::{MultimodalProcessor, AudioInput};
use mullama::{AsyncModel, StreamConfig};

// Configure audio capture
let audio_config = AudioStreamConfig {
    sample_rate: 16000,
    channels: 1,
    buffer_size: 4096,
    vad_enabled: true,
    noise_reduction: true,
    ..Default::default()
};

let mut audio_processor = StreamingAudioProcessor::new(audio_config)?;
let model = AsyncModel::load("whisper-llama.gguf").await?;

// Listen for speech
loop {
    let audio_chunk = audio_processor.capture_utterance().await?;

    // Process audio to text
    let audio_input = AudioInput::from_samples(audio_chunk.samples, 16000);
    let transcript = model.generate(&format!(
        "Transcribe: <audio>{}</audio>", audio_input.to_features()?
    ), 200).await?;

    println!("You said: {}", transcript);

    // Generate response
    let response = model.generate(&transcript, 200).await?;
    println!("Assistant: {}", response);
}
```

**Features used:** `multimodal`, `streaming-audio`, `async`

**Related:** [Multimodal Guide](guide/multimodal.md) | [Voice Assistant Example](examples/voice-assistant.md)

---

## 4. Local API Server

**Problem:** Replace cloud-based LLM APIs (OpenAI, Anthropic) with a local server that offers the same REST interface but runs entirely on-premise, ensuring data privacy and eliminating per-token costs.

**Solution:** Use Mullama's web integration to serve an OpenAI-compatible API with Axum, supporting both synchronous and streaming completions.

**Architecture:**

```rust
use mullama::{AsyncModel, AsyncConfig};
use mullama::web::{AppState, RouterBuilder, GenerateRequest, GenerateResponse};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), mullama::MullamaError> {
    let model = AsyncModel::load_with_config(AsyncConfig {
        model_path: "llama-3-8b-instruct.gguf".to_string(),
        gpu_layers: 32,
        context_size: 4096,
        max_concurrent: 4,
        ..Default::default()
    }).await?;

    let state = AppState::new(Arc::new(model));

    let app = RouterBuilder::new(state)
        .add_completions()          // POST /v1/completions
        .add_chat_completions()     // POST /v1/chat/completions
        .add_embeddings()           // POST /v1/embeddings
        .add_models()               // GET /v1/models
        .add_health()               // GET /health
        .cors_permissive()
        .build();

    println!("Serving on http://127.0.0.1:8080");
    axum::serve(
        tokio::net::TcpListener::bind("127.0.0.1:8080").await?,
        app,
    ).await?;

    Ok(())
}
```

**Features used:** `async`, `web`, `streaming`

**Related:** [API Server Example](examples/api-server.md) | [Streaming API](api/streaming.md)

---

## 5. Edge AI (Raspberry Pi / Embedded)

**Problem:** Deploy an AI model on resource-constrained edge devices (Raspberry Pi, Jetson Nano) for offline inference with minimal memory footprint and no cloud dependency.

**Solution:** Use Mullama's CPU-only mode with aggressive quantization (Q4_0 model, Q4_0 KV-cache) and minimal thread count to fit within device constraints.

**Architecture:**

```rust
use mullama::{Model, Context, ContextParams, ModelParams, SamplerParams, KvCacheType};
use std::sync::Arc;

// Load with minimal resource usage
let params = ModelParams {
    n_gpu_layers: 0,       // CPU only
    use_mmap: true,        // Memory-mapped for low RSS
    use_mlock: false,      // Don't lock RAM
    ..Default::default()
};

let model = Arc::new(Model::load_with_params("tinyllama-1.1b-q4_0.gguf", params)?);

// Minimal context for edge deployment
let ctx_params = ContextParams {
    n_ctx: 512,            // Short context saves memory
    n_batch: 128,          // Small batch for low latency
    n_threads: 4,          // Match Pi's core count
    type_k: KvCacheType::Q4_0,  // 75% KV-cache savings
    type_v: KvCacheType::Q4_0,
    ..Default::default()
};

let mut ctx = Context::new(model.clone(), ctx_params)?;

// Generate with constrained resources
let tokens = model.tokenize("Sensor reading: 23.5C. Status:", true, false)?;
let sampler = SamplerParams::precise(); // Low temperature for factual output
let output = ctx.generate_with_params(&tokens, 50, &sampler)?;
println!("AI: {}", output);
```

**Features used:** Core only (no additional features for minimal binary size)

**Related:** [Models Guide](guide/models.md) | [Platform Setup](getting-started/platform-setup.md)

---

## 6. Content Generation (Batch Processing)

**Problem:** Generate large volumes of content (product descriptions, article summaries, translations) efficiently by processing many prompts in parallel.

**Solution:** Use Mullama's multi-threaded architecture with shared model and per-thread contexts to process prompts concurrently, maximizing throughput.

**Architecture:**

```rust
use mullama::{Model, Context, ContextParams, SamplerParams};
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::load("llama-3-8b-instruct.gguf")?);

let prompts = vec![
    "Write a product description for: wireless headphones",
    "Write a product description for: mechanical keyboard",
    "Write a product description for: ergonomic mouse",
    "Write a product description for: USB-C hub",
    // ... hundreds more
];

// Process in parallel using thread pool
let chunk_size = prompts.len() / num_cpus::get();
let handles: Vec<_> = prompts.chunks(chunk_size)
    .map(|chunk| {
        let model = model.clone();
        let chunk = chunk.to_vec();
        thread::spawn(move || {
            let mut ctx = Context::new(model.clone(), ContextParams {
                n_ctx: 1024,
                n_threads: 2, // Fewer threads per context when parallelizing
                ..Default::default()
            }).unwrap();

            let sampler = SamplerParams {
                temperature: 0.7,
                penalty_repeat: 1.15,
                ..Default::default()
            };

            chunk.iter().map(|prompt| {
                ctx.clear_cache();
                let tokens = model.tokenize(prompt, true, false).unwrap();
                ctx.generate_with_params(&tokens, 200, &sampler).unwrap()
            }).collect::<Vec<String>>()
        })
    })
    .collect();

let results: Vec<String> = handles.into_iter()
    .flat_map(|h| h.join().unwrap())
    .collect();

println!("Generated {} descriptions", results.len());
```

**Features used:** Core (multi-threading via standard library; add `parallel` for Rayon integration)

**Related:** [Batch Example](examples/batch.md) | [Generation Guide](guide/generation.md)

---

## 7. Semantic Search (Embeddings + ColBERT)

**Problem:** Build a high-quality semantic search engine that goes beyond keyword matching, using dense retrieval with optional late-interaction (ColBERT-style) for better ranking precision.

**Solution:** Use Mullama's embedding API with Mean pooling for passage retrieval, and late-interaction multi-vector embeddings for re-ranking the top candidates.

**Architecture:**

```rust
use mullama::{Model, embedding::{EmbeddingGenerator, EmbeddingConfig, PoolingStrategy, cosine_similarity}};
use std::sync::Arc;

// Dense retrieval with single-vector embeddings
let embed_model = Arc::new(Model::load("nomic-embed.gguf")?);
let config = EmbeddingConfig {
    pooling: PoolingStrategy::Mean,
    normalize: true,
    batch_size: 64,
};
let mut embedder = EmbeddingGenerator::new(embed_model.clone(), config)?;

// Index corpus
let corpus = load_corpus("./data/documents.jsonl")?;
let corpus_embeddings = embedder.embed_batch(
    &corpus.iter().map(|s| s.as_str()).collect::<Vec<_>>()
)?;

// Search function
fn search(
    query: &str,
    embedder: &mut EmbeddingGenerator,
    corpus: &[String],
    corpus_embeddings: &[Vec<f32>],
    top_k: usize,
) -> Vec<(usize, f32)> {
    let query_emb = embedder.embed_text(query).unwrap();

    let mut scores: Vec<(usize, f32)> = corpus_embeddings.iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_similarity(&query_emb, doc_emb)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}

// Execute search
let results = search("rust memory safety", &mut embedder, &corpus, &corpus_embeddings, 10);
for (idx, score) in &results {
    println!("{:.4}: {}", score, &corpus[*idx][..80]);
}
```

**Features used:** Core (embeddings); add `late-interaction` for ColBERT-style multi-vector retrieval

**Related:** [Embeddings Guide](guide/embeddings.md) | [Embeddings API](api/embeddings.md)

---

## 8. Code Assistant (Structured Output)

**Problem:** Build a code generation assistant that produces syntactically valid, structured output (JSON, SQL, code) by constraining the model's generation to a formal grammar.

**Solution:** Use Mullama's grammar-constrained sampling with GBNF grammars to guarantee output conformance, combined with code-specific models and infill support.

**Architecture:**

```rust
use mullama::{Model, Context, ContextParams, SamplerParams};
use mullama::sampling::{SamplerChain, Sampler};
use std::sync::Arc;

let model = Arc::new(Model::load("codellama-7b-instruct.gguf")?);
let mut ctx = Context::new(model.clone(), ContextParams {
    n_ctx: 4096,
    ..Default::default()
})?;

// Define JSON schema grammar
let json_schema_grammar = r#"
root        ::= object
object      ::= "{" ws pairs ws "}"
pairs       ::= pair ("," ws pair)*
pair        ::= string ":" ws value
value       ::= string | number | object | array | "true" | "false" | "null"
array       ::= "[" ws (value ("," ws value)*)? ws "]"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
number      ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws          ::= [ \t\n]*
"#;

// Build sampler chain with grammar constraint
let mut chain = SamplerChain::with_defaults();
chain.add(Sampler::grammar(model.clone(), json_schema_grammar, "root")?);
chain.add(Sampler::temperature(0.3)?); // Low temp for precise code
chain.add(Sampler::dist(42)?);

// Generate structured output
let prompt = "Generate a JSON config for a web server with host, port, and routes:";
let tokens = model.tokenize(prompt, true, false)?;
ctx.decode(&tokens)?;

let mut output = String::new();
for _ in 0..500 {
    let token = chain.sample(&mut ctx, -1);
    chain.accept(token);
    if model.token_is_eog(token) { break; }
    let text = model.token_to_str(token, 0, false)?;
    output.push_str(&text);
    ctx.decode_single(token)?;
}

// Output is guaranteed to be valid JSON
let parsed: serde_json::Value = serde_json::from_str(&output)?;
println!("{}", serde_json::to_string_pretty(&parsed)?);
```

**Features used:** Core (grammar sampling is part of the sampling module)

**Related:** [Structured Output Guide](guide/structured-output.md) | [Sampling API](api/sampling.md)

---

## Feature Decision Diagram

Use this guide to determine which features to enable based on your use case:

| What You Need | Feature to Enable | Key Types |
|---------------|-------------------|-----------|
| Basic text generation | (none - core) | `Model`, `Context`, `SamplerParams` |
| Text embeddings | (none - core) | `EmbeddingGenerator`, `PoolingStrategy` |
| Grammar-constrained output | (none - core) | `Sampler::grammar` |
| Non-blocking inference | `async` | `AsyncModel`, `AsyncContext` |
| Real-time token streaming | `streaming` | `TokenStream`, `StreamConfig` |
| Image understanding | `multimodal` | `VisionEncoder`, `MultimodalProcessor` |
| Audio processing | `multimodal` | `AudioInput`, `AudioFeatures` |
| Live microphone input | `streaming-audio` | `StreamingAudioProcessor` |
| REST API server | `web` | `AppState`, `RouterBuilder` |
| WebSocket connections | `websockets` | `WebSocketServer` |
| Parallel batch processing | `parallel` | Rayon-based operations |
| ColBERT retrieval | `late-interaction` | Multi-vector embeddings |
| Background model daemon | `daemon` | Daemon CLI commands |
| All features | `full` | Everything above |

### Quick Start by Use Case

```toml
# Chatbot
mullama = { version = "0.1", features = ["streaming"] }

# RAG / Semantic search
mullama = { version = "0.1" }  # Core is sufficient

# Voice assistant
mullama = { version = "0.1", features = ["multimodal", "streaming-audio"] }

# API server
mullama = { version = "0.1", features = ["web", "streaming"] }

# Edge deployment (minimal binary)
mullama = { version = "0.1", default-features = false }

# Batch processing
mullama = { version = "0.1", features = ["parallel"] }

# Everything
mullama = { version = "0.1", features = ["full"] }
```
