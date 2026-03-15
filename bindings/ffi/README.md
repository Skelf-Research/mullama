# Mullama FFI

C ABI layer for Mullama - enables language bindings for Node.js, Python, PHP, Go, and more.

## Overview

This crate provides a stable C ABI for the Mullama LLM library. It generates both a shared library (`.so`/`.dylib`/`.dll`) and a static library (`.a`/`.lib`), along with a C header file for integration.

## Build

```bash
# Build release version
cargo build -p mullama-ffi --release

# Run tests
cargo test -p mullama-ffi
```

## Output Files

After building, you'll find:

- **Shared library**: `target/release/libmullama_ffi.so` (Linux), `.dylib` (macOS), `.dll` (Windows)
- **Static library**: `target/release/libmullama_ffi.a` (Unix), `.lib` (Windows)
- **C header**: `bindings/ffi/include/mullama.h`

## API Overview

### Initialization

```c
#include <mullama.h>

// Initialize backend (call once at startup)
mullama_backend_init();

// ... use library ...

// Cleanup (call once at shutdown)
mullama_backend_free();
```

### Model Loading

```c
// Load with default parameters
MullamaModel* model = mullama_model_load("model.gguf", NULL);

// Load with custom parameters
MullamaModelParams params = mullama_model_default_params();
params.n_gpu_layers = 32;
MullamaModel* model = mullama_model_load("model.gguf", &params);

// Check for errors
if (!model) {
    printf("Error: %s\n", mullama_get_last_error());
}

// Free when done
mullama_model_free(model);
```

### Text Generation

```c
// Create context
MullamaContext* ctx = mullama_context_new(model, NULL);

// Tokenize input
int32_t tokens[1024];
int n_tokens = mullama_tokenize(model, "Hello, AI!", tokens, 1024, true, false);

// Generate with default sampling
char output[4096];
int result = mullama_generate(ctx, tokens, n_tokens, 100, NULL, output, 4096);

printf("Generated: %s\n", output);

// Free context
mullama_context_free(ctx);
```

### Streaming Generation

```c
// Callback function
bool on_token(const char* token, void* user_data) {
    printf("%s", token);
    fflush(stdout);
    return true;  // Continue generation
}

// Stream tokens
mullama_generate_streaming(ctx, tokens, n_tokens, 100, NULL, on_token, NULL);
```

### Embeddings

```c
// Create embedding generator
MullamaEmbeddingGenerator* gen = mullama_embedding_generator_new(model, NULL);

// Generate embeddings
int n_embd = mullama_embedding_generator_n_embd(gen);
float* embeddings = malloc(n_embd * sizeof(float));
mullama_embed_text(gen, "Hello world", embeddings, n_embd);

// Compute similarity
float sim = mullama_embedding_cosine_similarity(emb1, emb2, n_embd);

// Free
mullama_embedding_generator_free(gen);
```

## Error Handling

All functions that can fail return an error code (negative value) or NULL. Use `mullama_get_last_error()` to get a detailed error message:

```c
MullamaModel* model = mullama_model_load("nonexistent.gguf", NULL);
if (!model) {
    const char* error = mullama_get_last_error();
    fprintf(stderr, "Failed to load model: %s\n", error);
}
```

## Thread Safety

- Model handles are thread-safe and can be shared across threads
- Context handles use internal locking for safe concurrent access
- Error messages are thread-local

## Memory Management

- All handles must be freed using their corresponding `*_free()` function
- Handles can be cloned using `*_clone()` functions to share ownership
- Output buffers must be provided by the caller

## Functions Reference

### Backend
- `mullama_backend_init()` - Initialize backend
- `mullama_backend_free()` - Free backend resources
- `mullama_supports_gpu_offload()` - Check GPU support
- `mullama_system_info()` - Get system information

### Model
- `mullama_model_load()` - Load a model
- `mullama_model_free()` - Free a model
- `mullama_tokenize()` - Tokenize text
- `mullama_detokenize()` - Convert tokens to text
- `mullama_model_n_ctx_train()` - Get training context size
- `mullama_model_n_embd()` - Get embedding dimension
- `mullama_model_n_vocab()` - Get vocabulary size
- `mullama_model_apply_chat_template()` - Format chat messages

### Context
- `mullama_context_new()` - Create a context
- `mullama_context_free()` - Free a context
- `mullama_decode()` - Process tokens
- `mullama_generate()` - Generate text
- `mullama_context_kv_cache_clear()` - Clear KV cache
- `mullama_context_get_logits()` - Get model logits
- `mullama_context_get_embeddings()` - Get embeddings

### Streaming
- `mullama_generate_streaming()` - Stream generated tokens
- `mullama_generate_streaming_cancellable()` - Streaming with cancellation
- `mullama_cancel_token_new()` - Create cancellation token
- `mullama_cancel_token_cancel()` - Cancel operation

### Embeddings
- `mullama_embedding_generator_new()` - Create embedding generator
- `mullama_embedding_generator_free()` - Free generator
- `mullama_embed_text()` - Embed single text
- `mullama_embed_batch()` - Embed multiple texts
- `mullama_embedding_cosine_similarity()` - Compute similarity
- `mullama_embedding_normalize()` - Normalize vector

### Sampling
- `mullama_sampler_default_params()` - Get default parameters
- `mullama_sampler_greedy_params()` - Deterministic sampling
- `mullama_sampler_creative_params()` - High randomness
- `mullama_sampler_precise_params()` - Low randomness

## License

MIT OR Apache-2.0
