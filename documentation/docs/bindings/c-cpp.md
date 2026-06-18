---
title: C/C++ Bindings
description: Direct access to the Mullama FFI layer through a stable C ABI, with RAII wrappers for C++, CMake integration, and cancellation support.
---

# C/C++ Bindings

The C/C++ bindings provide direct access to the Mullama FFI layer through a stable C ABI. This is the foundation layer that all other language bindings (Python, Node.js, Go, PHP) are built upon. The C API provides handle-based memory management and callback-driven streaming, while the C++ examples demonstrate RAII wrappers using `unique_ptr` with custom deleters.

## Building

```bash
# Build the shared library
cargo build --release -p mullama-ffi

# Output files:
# Linux:   target/release/libmullama_ffi.so
# macOS:   target/release/libmullama_ffi.dylib
# Windows: target/release/mullama_ffi.dll
```

The header file is located at `bindings/ffi/include/mullama.h`.

### Install System-Wide (Linux)

```bash
sudo cp target/release/libmullama_ffi.so /usr/local/lib/
sudo cp bindings/ffi/include/mullama.h /usr/local/include/
sudo ldconfig
```

### Install System-Wide (macOS)

```bash
sudo cp target/release/libmullama_ffi.dylib /usr/local/lib/
sudo cp bindings/ffi/include/mullama.h /usr/local/include/
```

---

## Header File Reference

### Types

```c
// Opaque handle types
typedef struct MullamaMullamaModel MullamaMullamaModel;
typedef struct MullamaMullamaContext MullamaMullamaContext;
typedef struct MullamaMullamaEmbeddingGenerator MullamaMullamaEmbeddingGenerator;
typedef struct MullamaCancelToken MullamaCancelToken;

// Model loading parameters
typedef struct {
    int n_gpu_layers;      // GPU layers (0 = CPU, -1 = all)
    int main_gpu;          // Main GPU device index
    bool use_mmap;         // Memory mapping
    bool use_mlock;        // Lock in memory
    bool vocab_only;       // Vocabulary only
    bool check_tensors;    // Check tensor integrity
} MullamaMullamaModelParams;

// Context creation parameters
typedef struct {
    uint32_t n_ctx;        // Context size (0 = model default)
    uint32_t n_batch;      // Logical batch size
    uint32_t n_ubatch;     // Physical batch size
    uint32_t n_seq_max;    // Max sequences
    int n_threads;         // Threads for generation
    int n_threads_batch;   // Threads for batch processing
    bool embeddings;       // Enable embeddings
    bool offload_kqv;      // Offload KQV to GPU
    int flash_attn;        // Flash attention (0=auto, 1=off, 2=on)
} MullamaMullamaContextParams;

// Sampler parameters
typedef struct {
    float temperature;     // 0.0 = deterministic
    int top_k;             // 0 = disabled
    float top_p;           // 1.0 = disabled
    float min_p;           // 0.0 = disabled
    float typical_p;       // 1.0 = disabled
    float penalty_repeat;  // 1.0 = disabled
    float penalty_freq;    // 0.0 = disabled
    float penalty_present; // 0.0 = disabled
    int penalty_last_n;    // Token window for penalties
    bool penalize_nl;      // Penalize newlines
    bool ignore_eos;       // Ignore EOS tokens
    uint32_t seed;         // Random seed (0 = random)
} MullamaMullamaSamplerParams;

// Embedding configuration
typedef struct {
    uint32_t n_ctx;        // Context size
    uint32_t n_batch;      // Batch size
    int n_threads;         // Thread count
    int pooling_type;      // 0=none, 1=mean, 2=cls, 3=last
    bool normalize;        // Normalize embeddings
} MullamaMullamaEmbeddingConfig;

// Chat message for template formatting
typedef struct {
    const char* role;      // Message role ("system", "user", "assistant")
    const char* content;   // Message content
} MullamaChatMessage;

// Streaming callback type
typedef bool (*MullamaStreamCallback)(const char* token, void* user_data);
```

### Error Codes

```c
enum MullamaErrorCode {
    MULLAMA_OK              =  0,   // Success
    MULLAMA_ERR_NULL        = -1,   // Null pointer
    MULLAMA_ERR_MODEL_LOAD  = -2,   // Model load failed
    MULLAMA_ERR_CONTEXT     = -3,   // Context error
    MULLAMA_ERR_TOKENIZE    = -4,   // Tokenization failed
    MULLAMA_ERR_GENERATE    = -5,   // Generation failed
    MULLAMA_ERR_SAMPLER     = -6,   // Sampler error
    MULLAMA_ERR_EMBEDDING   = -7,   // Embedding failed
    MULLAMA_ERR_INPUT       = -8,   // Invalid input
    MULLAMA_ERR_BUFFER      = -9,   // Buffer too small
    MULLAMA_ERR_BACKEND     = -10,  // Backend init failed
    MULLAMA_ERR_CANCELLED   = -11,  // Cancelled
    MULLAMA_ERR_INTERNAL    = -12,  // Internal error
    MULLAMA_ERR_UNAVAIL     = -13,  // Feature not available
    MULLAMA_ERR_UTF8        = -14,  // UTF-8 error
    MULLAMA_ERR_LOCK        = -15,  // Lock failed
};
```

---

### Backend Functions

```c
// Initialize the backend (call once at startup)
void mullama_backend_init(void);

// Free backend resources (call once at shutdown)
void mullama_backend_free(void);

// System capabilities
bool mullama_supports_gpu_offload(void);
bool mullama_supports_mmap(void);
bool mullama_supports_mlock(void);
size_t mullama_max_devices(void);

// System information (writes to buffer, returns bytes written or -required_size)
int mullama_system_info(char* output, size_t max_output);

// Version
const char* mullama_version(void);
uint32_t mullama_version_major(void);
uint32_t mullama_version_minor(void);
uint32_t mullama_version_patch(void);

// Timing
int64_t mullama_time_us(void);
```

---

### Error Handling

```c
// Get last error message (NULL if no error, thread-local)
const char* mullama_get_last_error(void);

// Clear the last error
void mullama_clear_error(void);

// Get human-readable description of an error code
const char* mullama_error_code_description(int code);
```

---

### Model Functions

```c
// Load a model (returns NULL on failure)
MullamaMullamaModel* mullama_model_load(const char* path, const MullamaMullamaModelParams* params);

// Free a model
void mullama_model_free(MullamaMullamaModel* model);

// Clone model handle (increments reference count)
MullamaMullamaModel* mullama_model_clone(const MullamaMullamaModel* model);

// Get default parameters
MullamaMullamaModelParams mullama_model_default_params(void);

// Tokenize text (returns token count, or -required_size if buffer too small)
int mullama_tokenize(
    const MullamaMullamaModel* model,
    const char* text,
    int* tokens,
    int max_tokens,
    bool add_bos,
    bool special
);

// Detokenize tokens (returns bytes written, or -required_size)
int mullama_detokenize(
    const MullamaMullamaModel* model,
    const int* tokens,
    int n_tokens,
    char* output,
    int max_output
);

// Model information
int mullama_model_n_ctx_train(const MullamaMullamaModel* model);
int mullama_model_n_embd(const MullamaMullamaModel* model);
int mullama_model_n_vocab(const MullamaMullamaModel* model);
int mullama_model_n_layer(const MullamaMullamaModel* model);
int mullama_model_n_head(const MullamaMullamaModel* model);
int mullama_model_token_bos(const MullamaMullamaModel* model);
int mullama_model_token_eos(const MullamaMullamaModel* model);
int mullama_model_token_nl(const MullamaMullamaModel* model);
bool mullama_model_token_is_eog(const MullamaMullamaModel* model, int token);
uint64_t mullama_model_size(const MullamaMullamaModel* model);
uint64_t mullama_model_n_params(const MullamaMullamaModel* model);
bool mullama_model_has_encoder(const MullamaMullamaModel* model);
bool mullama_model_has_decoder(const MullamaMullamaModel* model);

// Model description (returns bytes written or -required_size)
int mullama_model_desc(const MullamaMullamaModel* model, char* output, size_t max_output);

// Metadata access
int mullama_model_meta_val(
    const MullamaMullamaModel* model,
    const char* key,
    char* output,
    size_t max_output
);

// Chat template
int mullama_model_apply_chat_template(
    const MullamaMullamaModel* model,
    const MullamaChatMessage* messages,
    int n_messages,
    bool add_generation_prompt,
    char* output,
    size_t max_output
);
```

---

### Context Functions

```c
// Create context (returns NULL on failure)
MullamaMullamaContext* mullama_context_new(
    const MullamaMullamaModel* model,
    const MullamaMullamaContextParams* params
);

// Free context
void mullama_context_free(MullamaMullamaContext* ctx);

// Get default parameters
MullamaMullamaContextParams mullama_context_default_params(void);

// Decode tokens (returns 0 on success, negative on error)
int mullama_decode(MullamaMullamaContext* ctx, const int* tokens, int n_tokens);

// Generate text (returns bytes written or negative error code)
int mullama_generate(
    MullamaMullamaContext* ctx,
    const int* tokens,
    int n_tokens,
    int max_tokens,
    const MullamaMullamaSamplerParams* params,
    char* output,
    size_t max_output
);

// Context properties
uint32_t mullama_context_n_ctx(const MullamaMullamaContext* ctx);
uint32_t mullama_context_n_batch(const MullamaMullamaContext* ctx);
int mullama_context_n_threads(const MullamaMullamaContext* ctx);

// Thread management
int mullama_context_set_n_threads(MullamaMullamaContext* ctx, int n_threads, int n_threads_batch);

// KV cache management
int mullama_context_kv_cache_clear(MullamaMullamaContext* ctx);
int mullama_context_kv_cache_seq_rm(MullamaMullamaContext* ctx, int seq_id, int p0, int p1);

// Logits and embeddings
int mullama_context_get_logits(const MullamaMullamaContext* ctx, float* output, size_t max_output);
int mullama_context_get_embeddings(const MullamaMullamaContext* ctx, float* output, size_t max_output);

// State save/load
int mullama_context_save_state(const MullamaMullamaContext* ctx, uint8_t* output, size_t max_output);
int mullama_context_load_state(MullamaMullamaContext* ctx, const uint8_t* data, size_t data_size);
```

---

### Sampler Functions

```c
// Get preset sampler parameters
MullamaMullamaSamplerParams mullama_sampler_default_params(void);
MullamaMullamaSamplerParams mullama_sampler_greedy_params(void);
MullamaMullamaSamplerParams mullama_sampler_creative_params(void);
MullamaMullamaSamplerParams mullama_sampler_precise_params(void);
```

---

### Streaming Functions

```c
// Streaming generation with callback
int mullama_generate_streaming(
    MullamaMullamaContext* ctx,
    const int* tokens,
    int n_tokens,
    int max_tokens,
    const MullamaMullamaSamplerParams* params,
    MullamaStreamCallback callback,
    void* user_data
);

// Streaming with cancellation support
int mullama_generate_streaming_cancellable(
    MullamaMullamaContext* ctx,
    const int* tokens,
    int n_tokens,
    int max_tokens,
    const MullamaMullamaSamplerParams* params,
    MullamaStreamCallback callback,
    void* user_data,
    const MullamaCancelToken* cancel_token
);

// Cancellation token management
MullamaCancelToken* mullama_cancel_token_new(void);
void mullama_cancel_token_cancel(MullamaCancelToken* token);
bool mullama_cancel_token_is_cancelled(const MullamaCancelToken* token);
void mullama_cancel_token_free(MullamaCancelToken* token);
```

---

### Embedding Functions

```c
// Create embedding generator (returns NULL on failure)
MullamaMullamaEmbeddingGenerator* mullama_embedding_generator_new(
    const MullamaMullamaModel* model,
    const MullamaMullamaEmbeddingConfig* config
);

// Free embedding generator
void mullama_embedding_generator_free(MullamaMullamaEmbeddingGenerator* gen);

// Get default config
MullamaMullamaEmbeddingConfig mullama_embedding_default_config(void);

// Generate embedding for single text (returns floats written or negative error)
int mullama_embed_text(
    MullamaMullamaEmbeddingGenerator* gen,
    const char* text,
    float* output,
    size_t max_output
);

// Batch embedding (output is flattened: n_texts * n_embd)
int mullama_embed_batch(
    MullamaMullamaEmbeddingGenerator* gen,
    const char** texts,
    int n_texts,
    float* output,
    size_t max_output
);

// Get embedding dimension
int mullama_embedding_generator_n_embd(const MullamaMullamaEmbeddingGenerator* gen);

// Cosine similarity between two vectors
float mullama_embedding_cosine_similarity(const float* a, const float* b, size_t n);

// Normalize an embedding vector in-place
void mullama_embedding_normalize(float* embedding, size_t n);
```

---

## Memory Management

The C API uses **handle-based** memory management:

1. **Creation functions** (`mullama_model_load`, `mullama_context_new`, etc.) return opaque pointers.
2. **Free functions** (`mullama_model_free`, `mullama_context_free`, etc.) release resources.
3. Every successful creation must be paired with exactly one free call.
4. After freeing, the pointer is invalid and must not be used.

```c
// Correct usage pattern
MullamaMullamaModel* model = mullama_model_load("model.gguf", NULL);
if (!model) {
    fprintf(stderr, "Error: %s\n", mullama_get_last_error());
    return 1;
}

// ... use model ...

mullama_model_free(model);  // Always free
model = NULL;               // Good practice: null out freed pointers
```

---

## Complete C Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mullama.h"

// Streaming callback
bool on_token(const char* token, void* user_data) {
    printf("%s", token);
    fflush(stdout);
    int* count = (int*)user_data;
    (*count)++;
    return true;  // continue generating
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "model.gguf";
    const char* prompt = argc > 2 ? argv[2] : "Hello, world!";

    // Initialize backend
    mullama_backend_init();

    // Check GPU support
    if (mullama_supports_gpu_offload()) {
        printf("GPU offloading is available\n");
    }

    // Load model
    MullamaMullamaModelParams model_params = mullama_model_default_params();
    model_params.n_gpu_layers = -1;  // Use all GPU layers

    MullamaMullamaModel* model = mullama_model_load(model_path, &model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", mullama_get_last_error());
        mullama_backend_free();
        return 1;
    }

    // Print model info
    char desc[256];
    mullama_model_desc(model, desc, sizeof(desc));
    printf("Model: %s\n", desc);
    printf("Parameters: %llu\n", (unsigned long long)mullama_model_n_params(model));
    printf("Layers: %d\n", mullama_model_n_layer(model));

    // Create context
    MullamaMullamaContextParams ctx_params = mullama_context_default_params();
    ctx_params.n_ctx = 2048;

    MullamaMullamaContext* ctx = mullama_context_new(model, &ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context: %s\n", mullama_get_last_error());
        mullama_model_free(model);
        mullama_backend_free();
        return 1;
    }

    // Tokenize prompt
    int tokens[1024];
    int n_tokens = mullama_tokenize(model, prompt, tokens, 1024, true, false);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed: %s\n", mullama_get_last_error());
        mullama_context_free(ctx);
        mullama_model_free(model);
        mullama_backend_free();
        return 1;
    }
    printf("Prompt tokens: %d\n", n_tokens);

    // Generate with streaming
    printf("\nResponse: ");
    int token_count = 0;
    MullamaMullamaSamplerParams sampler = mullama_sampler_default_params();
    sampler.temperature = 0.7;

    int result = mullama_generate_streaming(
        ctx, tokens, n_tokens, 200, &sampler, on_token, &token_count
    );

    if (result < 0) {
        fprintf(stderr, "\nGeneration error: %s\n", mullama_get_last_error());
    } else {
        printf("\n\n[Generated %d tokens]\n", token_count);
    }

    // Cleanup
    mullama_context_free(ctx);
    mullama_model_free(model);
    mullama_backend_free();

    return 0;
}
```

### Compiling the C Example

```bash
gcc -o example example.c -I/usr/local/include -L/usr/local/lib -lmullama_ffi -lm -lpthread
# or with the build tree:
gcc -o example example.c \
    -I./bindings/ffi/include \
    -L./target/release \
    -lmullama_ffi -lm -lpthread -lstdc++
```

---

## Complete C++ Example (RAII Wrapper)

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <functional>
#include "mullama.h"

namespace mullama {

// Custom deleter for RAII handles
struct ModelDeleter {
    void operator()(MullamaMullamaModel* p) const {
        if (p) mullama_model_free(p);
    }
};

struct ContextDeleter {
    void operator()(MullamaMullamaContext* p) const {
        if (p) mullama_context_free(p);
    }
};

struct EmbeddingGenDeleter {
    void operator()(MullamaMullamaEmbeddingGenerator* p) const {
        if (p) mullama_embedding_generator_free(p);
    }
};

using ModelPtr = std::unique_ptr<MullamaMullamaModel, ModelDeleter>;
using ContextPtr = std::unique_ptr<MullamaMullamaContext, ContextDeleter>;
using EmbeddingGenPtr = std::unique_ptr<MullamaMullamaEmbeddingGenerator, EmbeddingGenDeleter>;

// Backend RAII guard
class Backend {
public:
    Backend() { mullama_backend_init(); }
    ~Backend() { mullama_backend_free(); }
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    static bool supports_gpu() { return mullama_supports_gpu_offload(); }
    static std::string system_info() {
        char buf[4096];
        int n = mullama_system_info(buf, sizeof(buf));
        return n > 0 ? std::string(buf) : "";
    }
};

// Model wrapper
class Model {
public:
    explicit Model(const std::string& path, const MullamaMullamaModelParams& params = {})
    {
        auto* raw = mullama_model_load(path.c_str(), &params);
        if (!raw) {
            throw std::runtime_error(
                std::string("Failed to load model: ") + mullama_get_last_error()
            );
        }
        ptr_.reset(raw);
    }

    std::vector<int> tokenize(const std::string& text, bool add_bos = true) const {
        std::vector<int> tokens(text.size() * 2 + 10);
        int n = mullama_tokenize(
            ptr_.get(), text.c_str(), tokens.data(),
            static_cast<int>(tokens.size()), add_bos, false
        );
        if (n < 0) {
            throw std::runtime_error("Tokenization failed: " +
                std::string(mullama_get_last_error()));
        }
        tokens.resize(n);
        return tokens;
    }

    std::string detokenize(const std::vector<int>& tokens) const {
        std::vector<char> buf(tokens.size() * 32);
        int n = mullama_detokenize(
            ptr_.get(), tokens.data(), static_cast<int>(tokens.size()),
            buf.data(), static_cast<int>(buf.size())
        );
        if (n < 0) {
            throw std::runtime_error("Detokenization failed");
        }
        return std::string(buf.data());
    }

    int n_ctx_train() const { return mullama_model_n_ctx_train(ptr_.get()); }
    int n_embd() const { return mullama_model_n_embd(ptr_.get()); }
    int n_vocab() const { return mullama_model_n_vocab(ptr_.get()); }
    int n_layer() const { return mullama_model_n_layer(ptr_.get()); }
    uint64_t size() const { return mullama_model_size(ptr_.get()); }
    uint64_t n_params() const { return mullama_model_n_params(ptr_.get()); }

    std::string description() const {
        char buf[256];
        mullama_model_desc(ptr_.get(), buf, sizeof(buf));
        return std::string(buf);
    }

    MullamaMullamaModel* get() const { return ptr_.get(); }

private:
    ModelPtr ptr_;
};

// Context wrapper
class Context {
public:
    Context(const Model& model, const MullamaMullamaContextParams& params = {})
    {
        auto* raw = mullama_context_new(model.get(), &params);
        if (!raw) {
            throw std::runtime_error(
                std::string("Failed to create context: ") + mullama_get_last_error()
            );
        }
        ptr_.reset(raw);
    }

    std::string generate(const std::vector<int>& tokens, int max_tokens,
                          const MullamaMullamaSamplerParams& params = {}) {
        std::vector<char> buf(max_tokens * 32);
        int n = mullama_generate(
            ptr_.get(), tokens.data(), static_cast<int>(tokens.size()),
            max_tokens, &params, buf.data(), buf.size()
        );
        if (n < 0) {
            throw std::runtime_error(
                std::string("Generation failed: ") + mullama_get_last_error()
            );
        }
        return std::string(buf.data());
    }

    std::string generate(const Model& model, const std::string& prompt,
                          int max_tokens,
                          const MullamaMullamaSamplerParams& params = {}) {
        auto tokens = model.tokenize(prompt);
        return generate(tokens, max_tokens, params);
    }

    void generate_streaming(const std::vector<int>& tokens, int max_tokens,
                             const MullamaMullamaSamplerParams& params,
                             std::function<bool(const std::string&)> callback) {
        struct CallbackData {
            std::function<bool(const std::string&)>* fn;
        };
        CallbackData data{&callback};

        auto c_callback = [](const char* token, void* user_data) -> bool {
            auto* d = static_cast<CallbackData*>(user_data);
            return (*d->fn)(std::string(token));
        };

        int result = mullama_generate_streaming(
            ptr_.get(), tokens.data(), static_cast<int>(tokens.size()),
            max_tokens, &params, c_callback, &data
        );
        if (result < 0) {
            throw std::runtime_error(
                std::string("Streaming failed: ") + mullama_get_last_error()
            );
        }
    }

    void clear_cache() { mullama_context_kv_cache_clear(ptr_.get()); }
    uint32_t n_ctx() const { return mullama_context_n_ctx(ptr_.get()); }
    uint32_t n_batch() const { return mullama_context_n_batch(ptr_.get()); }

private:
    ContextPtr ptr_;
};

// EmbeddingGenerator wrapper
class EmbeddingGenerator {
public:
    EmbeddingGenerator(const Model& model, uint32_t n_ctx = 512, bool normalize = true)
        : n_embd_(model.n_embd())
    {
        MullamaMullamaEmbeddingConfig config = mullama_embedding_default_config();
        config.n_ctx = n_ctx;
        config.normalize = normalize;

        auto* raw = mullama_embedding_generator_new(model.get(), &config);
        if (!raw) {
            throw std::runtime_error(
                std::string("Failed to create embedding generator: ") +
                mullama_get_last_error()
            );
        }
        ptr_.reset(raw);
    }

    std::vector<float> embed(const std::string& text) {
        std::vector<float> result(n_embd_);
        int n = mullama_embed_text(
            ptr_.get(), text.c_str(), result.data(), result.size()
        );
        if (n < 0) {
            throw std::runtime_error(
                std::string("Embedding failed: ") + mullama_get_last_error()
            );
        }
        result.resize(n);
        return result;
    }

    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(embed(text));
        }
        return results;
    }

    int n_embd() const { return n_embd_; }

    static float cosine_similarity(const std::vector<float>& a,
                                    const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }
        return mullama_embedding_cosine_similarity(a.data(), b.data(), a.size());
    }

private:
    EmbeddingGenPtr ptr_;
    int n_embd_;
};

} // namespace mullama

// Usage example
int main() {
    mullama::Backend backend;

    std::cout << "GPU support: " << (mullama::Backend::supports_gpu() ? "yes" : "no") << "\n";

    // Load model
    MullamaMullamaModelParams model_params = mullama_model_default_params();
    model_params.n_gpu_layers = -1;

    mullama::Model model("./model.gguf", model_params);
    std::cout << "Model: " << model.description() << "\n";
    std::cout << "Parameters: " << model.n_params() << "\n";

    // Create context
    MullamaMullamaContextParams ctx_params = mullama_context_default_params();
    ctx_params.n_ctx = 2048;

    mullama::Context ctx(model, ctx_params);

    // Generate with streaming
    auto tokens = model.tokenize("Explain C++ RAII:");
    auto params = mullama_sampler_precise_params();

    std::cout << "\nResponse: ";
    ctx.generate_streaming(tokens, 200, params, [](const std::string& token) {
        std::cout << token << std::flush;
        return true;
    });
    std::cout << "\n";

    // Embeddings
    mullama::EmbeddingGenerator gen(model);
    auto emb1 = gen.embed("Hello, world!");
    auto emb2 = gen.embed("Hi there!");
    float sim = mullama::EmbeddingGenerator::cosine_similarity(emb1, emb2);
    std::cout << "Similarity: " << sim << "\n";

    return 0;
}
```

### Compiling the C++ Example

```bash
g++ -std=c++17 -o example example.cpp \
    -I./bindings/ffi/include \
    -L./target/release \
    -lmullama_ffi -lm -lpthread -lstdc++
```

---

## CMake Integration

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app)

set(CMAKE_CXX_STANDARD 17)

# Find the mullama library
find_library(MULLAMA_LIB mullama_ffi
    HINTS
        ${CMAKE_SOURCE_DIR}/../target/release
        /usr/local/lib
        /usr/lib
)

find_path(MULLAMA_INCLUDE mullama.h
    HINTS
        ${CMAKE_SOURCE_DIR}/../bindings/ffi/include
        /usr/local/include
        /usr/include
)

if(NOT MULLAMA_LIB OR NOT MULLAMA_INCLUDE)
    message(FATAL_ERROR "Could not find mullama library or headers")
endif()

# Create your executable
add_executable(my_app main.cpp)

target_include_directories(my_app PRIVATE ${MULLAMA_INCLUDE})
target_link_libraries(my_app ${MULLAMA_LIB} m pthread stdc++)

# Set RPATH for finding the library at runtime
set_target_properties(my_app PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
```

### Using CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./my_app
```

---

## Meson Integration

```meson
project('my_app', 'cpp',
    version: '0.1.0',
    default_options: ['cpp_std=c++17'])

# Find mullama library
mullama_dep = declare_dependency(
    include_directories: include_directories('../bindings/ffi/include'),
    dependencies: [
        meson.get_compiler('cpp').find_library('mullama_ffi',
            dirs: [meson.source_root() / '../target/release',
                   '/usr/local/lib']),
        dependency('threads'),
    ]
)

executable('my_app',
    'main.cpp',
    dependencies: [mullama_dep],
    link_args: ['-lm', '-lstdc++'])
```

### Using Meson

```bash
meson setup build
meson compile -C build
./build/my_app
```

---

## Cancellation Support

The C API provides cancellation tokens for interrupting long-running generation:

```c
#include <pthread.h>
#include "mullama.h"

MullamaCancelToken* cancel_token = NULL;

// Signal handler or another thread can cancel
void* cancel_thread(void* arg) {
    sleep(5);  // Cancel after 5 seconds
    mullama_cancel_token_cancel(cancel_token);
    return NULL;
}

bool on_token(const char* token, void* user_data) {
    printf("%s", token);
    return true;
}

int main() {
    mullama_backend_init();

    MullamaMullamaModel* model = mullama_model_load("model.gguf", NULL);
    MullamaMullamaContext* ctx = mullama_context_new(model, NULL);

    int tokens[1024];
    int n = mullama_tokenize(model, "Write a very long essay:", tokens, 1024, true, false);

    // Create cancellation token
    cancel_token = mullama_cancel_token_new();

    // Start cancel timer in another thread
    pthread_t tid;
    pthread_create(&tid, NULL, cancel_thread, NULL);

    // Generate with cancellation support
    MullamaMullamaSamplerParams params = mullama_sampler_default_params();
    int result = mullama_generate_streaming_cancellable(
        ctx, tokens, n, 10000, &params, on_token, NULL, cancel_token
    );

    if (result == -11) {  // MULLAMA_ERR_CANCELLED
        printf("\n[Generation cancelled]\n");
    }

    pthread_join(tid, NULL);
    mullama_cancel_token_free(cancel_token);
    mullama_context_free(ctx);
    mullama_model_free(model);
    mullama_backend_free();
    return 0;
}
```

---

## Thread Safety

- **Model handles** can be shared across threads (they use `Arc` internally).
- **Context handles** are NOT thread-safe. Each thread must have its own context.
- **Error messages** are thread-local, so concurrent FFI calls will not overwrite each other's errors.
- **Backend init/free** should be called from a single thread (init is internally synchronized).

```c
// Correct multi-threaded usage:
// 1. Initialize backend on main thread
mullama_backend_init();

// 2. Load model on any thread
MullamaMullamaModel* model = mullama_model_load("model.gguf", NULL);

// 3. Each thread creates its own context
// Thread 1:
MullamaMullamaContext* ctx1 = mullama_context_new(model, NULL);
// Thread 2:
MullamaMullamaContext* ctx2 = mullama_context_new(model, NULL);

// 4. Each thread uses its own context
// Thread 1: mullama_generate(ctx1, ...);
// Thread 2: mullama_generate(ctx2, ...);

// 5. Cleanup (after all threads are done)
mullama_context_free(ctx1);
mullama_context_free(ctx2);
mullama_model_free(model);
mullama_backend_free();
```

---

## Performance Tips

1. **GPU offloading** -- set `n_gpu_layers = -1` for maximum performance when a GPU is available.

2. **Reuse contexts** -- context creation has overhead. Clear the cache between generations rather than recreating.

3. **Buffer sizing** -- for `mullama_generate`, allocate `max_tokens * 32` bytes as a safe upper bound for the output buffer.

4. **Batch embeddings** -- use `mullama_embed_batch` for multiple texts rather than calling `mullama_embed_text` in a loop.

5. **Memory mapping** -- keep `use_mmap = true` for faster loading and lower memory usage.

6. **Thread count** -- the default auto-detection works well. Override `n_threads` only for specific tuning.

7. **State save/load** -- use `mullama_context_save_state` / `mullama_context_load_state` to cache and restore context state for repeated prompts.
