# Multimodal API

APIs for vision-language and audio-language models.

## MtmdContext

Main interface for multimodal processing.

### `MtmdContext::new`

Create a multimodal context.

```rust
pub fn new(
    mmproj_path: &str,
    model: &Arc<Model>,
    params: MtmdParams
) -> Result<Self, MullamaError>
```

**Parameters:**
- `mmproj_path` - Path to the multimodal projector file
- `model` - The text model
- `params` - Configuration parameters

**Example:**
```rust
let model = Arc::new(Model::load("llava-model.gguf")?);
let mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;
```

---

### `supports_vision`

Check if vision (image) input is supported.

```rust
pub fn supports_vision(&self) -> bool
```

---

### `supports_audio`

Check if audio input is supported.

```rust
pub fn supports_audio(&self) -> bool
```

---

### `audio_bitrate`

Get the audio sample rate if audio is supported.

```rust
pub fn audio_bitrate(&self) -> Option<i32>
```

---

### `bitmap_from_file`

Load an image or audio file as a bitmap.

```rust
pub fn bitmap_from_file(&self, path: &str) -> Result<Bitmap, MullamaError>
```

**Supported formats:**
- Images: JPEG, PNG, BMP, GIF, WebP
- Audio: WAV, MP3, FLAC

**Example:**
```rust
let image = mtmd.bitmap_from_file("photo.jpg")?;
```

---

### `bitmap_from_buffer`

Load from a byte buffer.

```rust
pub fn bitmap_from_buffer(&self, data: &[u8]) -> Result<Bitmap, MullamaError>
```

**Example:**
```rust
let data = std::fs::read("image.png")?;
let image = mtmd.bitmap_from_buffer(&data)?;
```

---

### `tokenize`

Tokenize text with media, replacing markers with bitmaps.

```rust
pub fn tokenize(
    &mut self,
    text: &str,
    bitmaps: &[&Bitmap]
) -> Result<InputChunks, MullamaError>
```

**Parameters:**
- `text` - Prompt with `<__media__>` markers
- `bitmaps` - Bitmaps to substitute for each marker

**Example:**
```rust
let image = mtmd.bitmap_from_file("photo.jpg")?;
let chunks = mtmd.tokenize(
    "What's in this image? <__media__>",
    &[&image]
)?;
```

---

### `eval_chunks`

Evaluate multimodal chunks in a context.

```rust
pub fn eval_chunks(
    &mut self,
    context: &mut Context,
    chunks: &InputChunks,
    n_past: i32,
    seq_id: i32,
    n_batch: i32,
    logits_last: bool
) -> Result<i32, MullamaError>
```

**Parameters:**
- `context` - The llama context
- `chunks` - Tokenized input chunks
- `n_past` - Current position
- `seq_id` - Sequence ID (usually 0)
- `n_batch` - Batch size
- `logits_last` - Compute logits for last token

**Returns:** New position after evaluation

**Example:**
```rust
let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;
```

---

### `encode_chunk`

Encode a single chunk.

```rust
pub fn encode_chunk(&mut self, chunk: &InputChunk) -> Result<(), MullamaError>
```

---

### `get_output_embeddings`

Get embeddings from the last encode operation.

```rust
pub fn get_output_embeddings(&self, chunk: &InputChunk) -> Option<&[f32]>
```

---

## MtmdParams

Configuration for multimodal context.

```rust
pub struct MtmdParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: i32,
    pub media_marker: Option<String>,
    pub warmup: bool,
    pub image_min_tokens: Option<i32>,
    pub image_max_tokens: Option<i32>,
}
```

### Fields

| Field | Default | Description |
|-------|---------|-------------|
| `use_gpu` | `true` | Use GPU for vision encoder |
| `print_timings` | `false` | Print processing times |
| `n_threads` | `4` | CPU threads |
| `media_marker` | `None` | Custom marker (default: `<__media__>`) |
| `warmup` | `true` | Warmup encode on init |
| `image_min_tokens` | `None` | Min tokens per image |
| `image_max_tokens` | `None` | Max tokens per image |

**Example:**
```rust
let params = MtmdParams {
    use_gpu: true,
    print_timings: true,
    media_marker: Some("<image>".to_string()),
    ..Default::default()
};
```

---

## Bitmap

Represents image or audio data.

### `width`

Get image width (1 for audio).

```rust
pub fn width(&self) -> u32
```

---

### `height`

Get image height (1 for audio).

```rust
pub fn height(&self) -> u32
```

---

### `from_rgb`

Create bitmap from raw RGB pixels.

```rust
pub fn from_rgb(width: u32, height: u32, data: &[u8]) -> Result<Self, MullamaError>
```

**Example:**
```rust
let pixels: Vec<u8> = vec![0; width * height * 3];
let bitmap = Bitmap::from_rgb(width, height, &pixels)?;
```

---

## InputChunks

Collection of tokenized chunks.

### `len`

Get number of chunks.

```rust
pub fn len(&self) -> usize
```

---

### `iter`

Iterate over chunks.

```rust
pub fn iter(&self) -> impl Iterator<Item = InputChunk>
```

**Example:**
```rust
for chunk in chunks.iter() {
    println!("Type: {:?}, Tokens: {}", chunk.chunk_type(), chunk.n_tokens());
}
```

---

## InputChunk

A single tokenized chunk.

### `chunk_type`

Get the chunk type.

```rust
pub fn chunk_type(&self) -> ChunkType
```

---

### `n_tokens`

Get the number of tokens.

```rust
pub fn n_tokens(&self) -> i32
```

---

## ChunkType

Enum for chunk types.

```rust
pub enum ChunkType {
    Text,
    Image,
    Audio,
}
```

---

## Complete Example

```rust
use mullama::{
    Model, Context, ContextParams,
    MtmdContext, MtmdParams, ChunkType
};
use std::sync::Arc;

fn describe_image(image_path: &str) -> Result<String, mullama::MullamaError> {
    // Load model
    let model = Arc::new(Model::load("llava-v1.5-7b-q4.gguf")?);
    let mut ctx = Context::new(model.clone(), ContextParams::default())?;

    // Create multimodal context
    let mut mtmd = MtmdContext::new(
        "llava-v1.5-7b-mmproj-f16.gguf",
        &model,
        MtmdParams::default()
    )?;

    // Load image
    let image = mtmd.bitmap_from_file(image_path)?;
    println!("Image: {}x{}", image.width(), image.height());

    // Tokenize
    let chunks = mtmd.tokenize(
        "Describe this image: <__media__>",
        &[&image]
    )?;

    // Print chunk info
    for (i, chunk) in chunks.iter().enumerate() {
        let type_name = match chunk.chunk_type() {
            ChunkType::Text => "text",
            ChunkType::Image => "image",
            ChunkType::Audio => "audio",
        };
        println!("Chunk {}: {} ({} tokens)", i, type_name, chunk.n_tokens());
    }

    // Evaluate
    let n_past = mtmd.eval_chunks(&mut ctx, &chunks, 0, 0, 512, true)?;

    // Generate
    ctx.generate_continue(n_past, 256)
}
```

---

## Error Handling

```rust
use mullama::MullamaError;

match mtmd.bitmap_from_file(path) {
    Ok(bitmap) => { /* use bitmap */ }
    Err(MullamaError::MultimodalError(msg)) => {
        eprintln!("Failed to load: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}

match mtmd.tokenize(text, &bitmaps) {
    Ok(chunks) => { /* use chunks */ }
    Err(MullamaError::InvalidInput(msg)) => {
        // Wrong number of bitmaps for markers
        eprintln!("Invalid input: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```
