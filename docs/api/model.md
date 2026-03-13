# Model API Reference

The `Model` struct represents a loaded llama.cpp model and provides methods for tokenization, model introspection, and text processing.

## Loading Models

### `Model::from_file`

Load a model from a GGUF file with default parameters.

```rust
use mullama::Model;

let model = Model::from_file("path/to/model.gguf")?;
```

**Parameters:**
- `path`: Path to the GGUF model file

**Returns:** `Result<Model, MullamaError>`

### `Model::from_file_with_params`

Load a model with custom parameters.

```rust
use mullama::{Model, ModelParams};

let params = ModelParams::default()
    .with_n_gpu_layers(35)
    .with_vocab_only(false);

let model = Model::from_file_with_params("path/to/model.gguf", params)?;
```

**Parameters:**
- `path`: Path to the GGUF model file
- `params`: Model loading parameters

**Returns:** `Result<Model, MullamaError>`

## Model Information

### Basic Properties

```rust
// Vocabulary size
let vocab_size = model.vocab_size();

// Training context size
let ctx_size = model.n_ctx_train();

// Embedding dimensions
let n_embd = model.n_embd();

// Number of layers
let n_layer = model.n_layer();
```

### Model Metadata

```rust
// Model description
let desc = model.desc();

// Model size in bytes
let size = model.size();

// Parameter count
let n_params = model.n_params();
```

## Tokenization

### `tokenize`

Convert text to tokens.

```rust
let tokens = model.tokenize("Hello, world!", true)?;
```

**Parameters:**
- `text`: Input text to tokenize
- `add_special`: Whether to add special tokens (BOS, etc.)

**Returns:** `Result<Vec<Token>, MullamaError>`

### `token_to_str`

Convert a token back to text.

```rust
let text = model.token_to_str(token)?;
```

**Parameters:**
- `token`: Token to convert

**Returns:** `Result<String, MullamaError>`

### `detokenize`

Convert multiple tokens to text.

```rust
let tokens = vec![1, 2, 3, 4];
let text = model.detokenize(&tokens)?;
```

**Parameters:**
- `tokens`: Slice of tokens to convert

**Returns:** `Result<String, MullamaError>`

## Special Tokens

### Token Types

```rust
// Beginning of sequence
let bos = model.token_bos();

// End of sequence
let eos = model.token_eos();

// Newline token
let nl = model.token_nl();

// Unknown token
let unk = model.token_unk();
```

### Token Classification

```rust
// Check if token is end-of-generation
if model.is_eog_token(token) {
    println!("Generation should stop");
}

// Check if token is beginning-of-sequence
if model.is_bos_token(token) {
    println!("This is the start token");
}

// Check if token is end-of-sequence
if model.is_eos_token(token) {
    println!("This is the end token");
}
```

## Token Attributes

### `token_get_attr`

Get token attributes.

```rust
use mullama::TokenAttr;

let attrs = model.token_get_attr(token);

if attrs.contains(TokenAttr::NORMAL) {
    println!("Normal token");
}

if attrs.contains(TokenAttr::CONTROL) {
    println!("Control token");
}
```

### `token_get_type`

Get token type.

```rust
use mullama::TokenType;

match model.token_get_type(token) {
    TokenType::Normal => println!("Normal token"),
    TokenType::Control => println!("Control token"),
    TokenType::Unknown => println!("Unknown token"),
    TokenType::Unused => println!("Unused token"),
    _ => println!("Other token type"),
}
```

## Vocabulary Introspection

### `token_get_text`

Get the text representation of a token.

```rust
let text = model.token_get_text(token);
println!("Token {} = '{}'", token, text);
```

### `token_get_score`

Get the score/probability of a token.

```rust
let score = model.token_get_score(token);
println!("Token {} score: {}", token, score);
```

## Model Architecture

### `rope_freq_scale_train`

Get the RoPE frequency scale used during training.

```rust
let rope_scale = model.rope_freq_scale_train();
```

### `vocab_type`

Get the vocabulary type.

```rust
use mullama::VocabType;

match model.vocab_type() {
    VocabType::Spm => println!("SentencePiece model"),
    VocabType::Bpe => println!("Byte-pair encoding"),
    VocabType::Wpm => println!("WordPiece model"),
    _ => println!("Other vocab type"),
}
```

## Advanced Features

### KV Overrides

Apply key-value overrides to modify model behavior.

```rust
use mullama::KvOverride;
use std::collections::HashMap;

let mut overrides = HashMap::new();
overrides.insert("max_seq_len".to_string(), KvOverride::Int(4096));
overrides.insert("rope_freq_base".to_string(), KvOverride::Float(10000.0));

let model = Model::from_file_with_overrides("model.gguf", overrides)?;
```

### Embeddings

Get embedding vector for tokens.

```rust
// This requires a context for evaluation
let embeddings = model.get_embeddings(&context)?;
println!("Embedding dimensions: {}", embeddings.len());
```

## Error Handling

All model operations return `Result<T, MullamaError>`. Common error types:

```rust
use mullama::MullamaError;

match model.tokenize(text, true) {
    Ok(tokens) => { /* success */ },
    Err(MullamaError::ModelError(msg)) => {
        eprintln!("Model error: {}", msg);
    },
    Err(MullamaError::TokenizationError(msg)) => {
        eprintln!("Tokenization failed: {}", msg);
    },
    Err(MullamaError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Thread Safety

The `Model` struct is `Send + Sync` and can be safely shared across threads:

```rust
use std::sync::Arc;
use std::thread;

let model = Arc::new(Model::from_file("model.gguf")?);

let handles: Vec<_> = (0..4).map(|i| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        let tokens = model.tokenize(&format!("Thread {}", i), true).unwrap();
        println!("Thread {}: {} tokens", i, tokens.len());
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## Best Practices

### Memory Management
- Models are automatically freed when dropped
- Use `Arc<Model>` for sharing across threads
- Consider `with_use_mmap(true)` for large models

### Performance
- Load the model once and reuse it
- Use appropriate `n_gpu_layers` for your hardware
- Enable vocabulary-only loading for tokenization-only use cases

### Error Handling
- Always handle model loading errors gracefully
- Check for file existence before loading
- Validate model compatibility with your use case

## Examples

### Complete Tokenization Example

```rust
use mullama::{Model, MullamaError};

fn analyze_text(model: &Model, text: &str) -> Result<(), MullamaError> {
    println!("Analyzing: '{}'", text);

    let tokens = model.tokenize(text, true)?;
    println!("Tokens: {:?}", tokens);

    for (i, &token) in tokens.iter().enumerate() {
        let text = model.token_to_str(token)?;
        let score = model.token_get_score(token);
        let token_type = model.token_get_type(token);

        println!("  {}: {} -> '{}' (score: {:.3}, type: {:?})",
                 i, token, text, score, token_type);
    }

    let reconstructed = model.detokenize(&tokens)?;
    println!("Reconstructed: '{}'", reconstructed);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("model.gguf")?;
    analyze_text(&model, "Hello, world! How are you today?")?;
    Ok(())
}
```

### Model Information Example

```rust
use mullama::Model;

fn print_model_info(model: &Model) {
    println!("Model Information:");
    println!("  Description: {}", model.desc());
    println!("  Vocabulary size: {}", model.vocab_size());
    println!("  Context size: {}", model.n_ctx_train());
    println!("  Embedding dimensions: {}", model.n_embd());
    println!("  Layers: {}", model.n_layer());
    println!("  Parameters: {}", model.n_params());
    println!("  Size: {} bytes", model.size());
    println!("  Vocab type: {:?}", model.vocab_type());

    println!("Special tokens:");
    println!("  BOS: {}", model.token_bos());
    println!("  EOS: {}", model.token_eos());
    println!("  UNK: {}", model.token_unk());
    println!("  NL: {}", model.token_nl());
}
```

This comprehensive API provides everything needed for model loading, tokenization, and introspection in a memory-safe, ergonomic way.