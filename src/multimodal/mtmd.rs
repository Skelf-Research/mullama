//! MtmdContext FFI wrapper and related types for llama.cpp multimodal binding

use std::ffi::{CStr, CString};
use std::sync::Arc;

use crate::error::MullamaError;
use crate::sys;

/// Bitmap wrapper for mtmd_bitmap (holds image or audio data)
///
/// This is the core data structure for passing media to the multimodal processor.
/// It can hold either image data (RGB pixels) or audio data (f32 samples).
pub struct Bitmap {
    pub(crate) ptr: *mut sys::mtmd_bitmap,
}

// SAFETY: mtmd_bitmap is read-only after creation
unsafe impl Send for Bitmap {}
unsafe impl Sync for Bitmap {}

impl Bitmap {
    /// Create a bitmap from RGB image data
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - RGB pixel data (length must be width * height * 3)
    ///
    /// # Example
    /// ```rust
    /// let rgb_data = vec![255u8; 224 * 224 * 3]; // White 224x224 image
    /// let bitmap = Bitmap::from_image(224, 224, &rgb_data)?;
    /// ```
    pub fn from_image(width: u32, height: u32, data: &[u8]) -> Result<Self, MullamaError> {
        let expected_len = (width * height * 3) as usize;
        if data.len() != expected_len {
            return Err(MullamaError::InvalidInput(format!(
                "Image data length {} doesn't match expected {} ({}x{}x3)",
                data.len(),
                expected_len,
                width,
                height
            )));
        }

        let ptr = unsafe { sys::mtmd_bitmap_init(width, height, data.as_ptr()) };
        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create image bitmap".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Create a bitmap from audio samples
    ///
    /// # Arguments
    /// * `samples` - Audio samples as f32 (PCM format, -1.0 to 1.0)
    ///
    /// # Example
    /// ```rust
    /// let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
    /// let bitmap = Bitmap::from_audio(&samples)?;
    /// ```
    pub fn from_audio(samples: &[f32]) -> Result<Self, MullamaError> {
        if samples.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Audio samples cannot be empty".to_string(),
            ));
        }

        let ptr = unsafe { sys::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr()) };
        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create audio bitmap".to_string(),
            ));
        }

        Ok(Self { ptr })
    }

    /// Get the width of the bitmap (for images)
    pub fn width(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.ptr) }
    }

    /// Get the height of the bitmap (for images)
    pub fn height(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.ptr) }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        unsafe { sys::mtmd_bitmap_get_n_bytes(self.ptr) }
    }

    /// Check if this bitmap contains audio data
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.ptr) }
    }

    /// Get the bitmap ID (if set)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_bitmap_get_id(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
        }
    }

    /// Set the bitmap ID (useful for KV cache tracking)
    pub fn set_id(&mut self, id: &str) -> Result<(), MullamaError> {
        let id_c = CString::new(id)
            .map_err(|_| MullamaError::InvalidInput("ID contains null byte".to_string()))?;
        unsafe { sys::mtmd_bitmap_set_id(self.ptr, id_c.as_ptr()) };
        Ok(())
    }

    /// Get the raw pointer (for FFI calls)
    pub(crate) fn as_ptr(&self) -> *const sys::mtmd_bitmap {
        self.ptr
    }
}

impl Drop for Bitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_bitmap_free(self.ptr) };
        }
    }
}

/// Input chunks - a collection of tokenized multimodal input
///
/// This holds the result of tokenizing text with embedded media markers.
/// Each chunk is either text tokens or media (image/audio) data.
pub struct InputChunks {
    ptr: *mut sys::mtmd_input_chunks,
}

unsafe impl Send for InputChunks {}
unsafe impl Sync for InputChunks {}

impl InputChunks {
    /// Create a new empty input chunks container
    pub fn new() -> Self {
        let ptr = unsafe { sys::mtmd_input_chunks_init() };
        Self { ptr }
    }

    /// Get the number of chunks
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.ptr) }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index
    pub fn get(&self, idx: usize) -> Option<InputChunk> {
        if idx >= self.len() {
            return None;
        }
        let chunk_ptr = unsafe { sys::mtmd_input_chunks_get(self.ptr, idx) };
        if chunk_ptr.is_null() {
            None
        } else {
            Some(InputChunk {
                ptr: chunk_ptr,
                owned: false,
            })
        }
    }

    /// Get total number of tokens across all chunks
    pub fn total_tokens(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_tokens(self.ptr) }
    }

    /// Get total position count (for M-RoPE models this differs from token count)
    pub fn total_positions(&self) -> i32 {
        unsafe { sys::mtmd_helper_get_n_pos(self.ptr) }
    }

    /// Iterate over chunks
    pub fn iter(&self) -> impl Iterator<Item = InputChunk> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// Get the raw pointer (for FFI calls)
    pub(crate) fn as_ptr(&self) -> *mut sys::mtmd_input_chunks {
        self.ptr
    }
}

impl Default for InputChunks {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InputChunks {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_input_chunks_free(self.ptr) };
        }
    }
}

/// Type of input chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// Text tokens
    Text,
    /// Image data
    Image,
    /// Audio data
    Audio,
}

impl From<sys::mtmd_input_chunk_type> for ChunkType {
    fn from(t: sys::mtmd_input_chunk_type) -> Self {
        match t {
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_TEXT => ChunkType::Text,
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_IMAGE => ChunkType::Image,
            sys::mtmd_input_chunk_type::MTMD_INPUT_CHUNK_TYPE_AUDIO => ChunkType::Audio,
        }
    }
}

/// A single input chunk (text or media)
pub struct InputChunk {
    ptr: *const sys::mtmd_input_chunk,
    owned: bool,
}

impl InputChunk {
    /// Get the type of this chunk
    pub fn chunk_type(&self) -> ChunkType {
        let t = unsafe { sys::mtmd_input_chunk_get_type(self.ptr) };
        ChunkType::from(t)
    }

    /// Get the number of tokens in this chunk
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr) }
    }

    /// Get the number of positions (for M-RoPE)
    pub fn n_positions(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr) }
    }

    /// Get the chunk ID (for media chunks)
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { sys::mtmd_input_chunk_get_id(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
        }
    }

    /// Get text tokens (only valid for text chunks)
    pub fn text_tokens(&self) -> Option<Vec<i32>> {
        if self.chunk_type() != ChunkType::Text {
            return None;
        }

        let mut n_tokens: usize = 0;
        let tokens_ptr = unsafe { sys::mtmd_input_chunk_get_tokens_text(self.ptr, &mut n_tokens) };

        if tokens_ptr.is_null() || n_tokens == 0 {
            return None;
        }

        let tokens = unsafe { std::slice::from_raw_parts(tokens_ptr, n_tokens) };
        Some(tokens.to_vec())
    }

    /// Copy this chunk (takes ownership)
    pub fn copy(&self) -> Self {
        let ptr = unsafe { sys::mtmd_input_chunk_copy(self.ptr) };
        Self { ptr, owned: true }
    }
}

impl Drop for InputChunk {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { sys::mtmd_input_chunk_free(self.ptr as *mut _) };
        }
    }
}

/// Multimodal context parameters
#[derive(Debug, Clone)]
pub struct MtmdParams {
    /// Whether to use GPU for encoding
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use
    pub n_threads: i32,
    /// Media marker in prompts (default: "<__media__>")
    pub media_marker: Option<String>,
    /// Flash attention setting
    pub flash_attn_type: sys::llama_flash_attn_type,
    /// Whether to run warmup encode after initialization
    pub warmup: bool,
    /// Minimum image tokens (for dynamic resolution models)
    pub image_min_tokens: Option<i32>,
    /// Maximum image tokens (for dynamic resolution models)
    pub image_max_tokens: Option<i32>,
}

impl Default for MtmdParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            print_timings: false,
            n_threads: 4,
            media_marker: None,
            flash_attn_type: sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO,
            warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }
}

impl MtmdParams {
    /// Convert to FFI params, using defaults for unspecified values
    fn to_sys(&self) -> sys::mtmd_context_params {
        let default = unsafe { sys::mtmd_context_params_default() };

        sys::mtmd_context_params {
            use_gpu: self.use_gpu,
            print_timings: self.print_timings,
            n_threads: self.n_threads,
            image_marker: default.image_marker, // deprecated, keep default
            media_marker: default.media_marker, // keep default, custom set later
            flash_attn_type: self.flash_attn_type,
            warmup: self.warmup,
            image_min_tokens: self.image_min_tokens.unwrap_or(default.image_min_tokens),
            image_max_tokens: self.image_max_tokens.unwrap_or(default.image_max_tokens),
        }
    }
}

/// Multimodal context for vision-language and audio-language models
///
/// This is the main interface for processing multimodal inputs with llama.cpp models.
///
/// # Example
/// ```rust
/// use mullama::{Model, MtmdContext, Bitmap, MtmdParams};
/// use std::sync::Arc;
///
/// let model = Arc::new(Model::load("llava-model.gguf")?);
/// let mut mtmd = MtmdContext::new("llava-mmproj.gguf", &model, MtmdParams::default())?;
///
/// // Load an image
/// let image = Bitmap::from_file(&mtmd, "test.jpg")?;
///
/// // Tokenize prompt with image
/// let chunks = mtmd.tokenize("Describe this image: <__media__>", &[&image])?;
///
/// // Evaluate in context and generate...
/// ```
pub struct MtmdContext {
    ptr: *mut sys::mtmd_context,
    model: Arc<crate::Model>,
    // Keep CStrings alive
    _media_marker: Option<CString>,
}

unsafe impl Send for MtmdContext {}
// Safety: Access is synchronized through RwLock in the daemon
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// Create a new multimodal context
    ///
    /// # Arguments
    /// * `mmproj_path` - Path to the multimodal projector file (e.g., mmproj.gguf)
    /// * `model` - The text model to use
    /// * `params` - Configuration parameters
    pub fn new(
        mmproj_path: &str,
        model: &Arc<crate::Model>,
        params: MtmdParams,
    ) -> Result<Self, MullamaError> {
        let path_c = CString::new(mmproj_path)
            .map_err(|_| MullamaError::InvalidInput("Path contains null byte".to_string()))?;

        let mut sys_params = params.to_sys();

        // Handle custom media marker
        let media_marker_c = if let Some(ref marker) = params.media_marker {
            let c = CString::new(marker.as_str())
                .map_err(|_| MullamaError::InvalidInput("Marker contains null byte".to_string()))?;
            sys_params.media_marker = c.as_ptr();
            Some(c)
        } else {
            None
        };

        let ptr = unsafe { sys::mtmd_init_from_file(path_c.as_ptr(), model.as_ptr(), sys_params) };

        if ptr.is_null() {
            return Err(MullamaError::ModelLoadError(format!(
                "Failed to load multimodal projector from: {}",
                mmproj_path
            )));
        }

        Ok(Self {
            ptr,
            model: model.clone(),
            _media_marker: media_marker_c,
        })
    }

    /// Check if this context supports vision (image) input
    pub fn supports_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.ptr) }
    }

    /// Check if this context supports audio input
    pub fn supports_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.ptr) }
    }

    /// Get the audio bitrate (sample rate) if audio is supported
    pub fn audio_bitrate(&self) -> Option<i32> {
        let rate = unsafe { sys::mtmd_get_audio_bitrate(self.ptr) };
        if rate < 0 {
            None
        } else {
            Some(rate)
        }
    }

    /// Check if non-causal attention is needed for decoding
    pub fn needs_non_causal(&self) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.ptr) }
    }

    /// Check if M-RoPE is used
    pub fn uses_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.ptr) }
    }

    /// Load a bitmap from a file (image or audio)
    ///
    /// Supported formats:
    /// - Images: jpg, png, bmp, gif (via stb_image)
    /// - Audio: wav, mp3, flac (via miniaudio)
    pub fn bitmap_from_file(&self, path: &str) -> Result<Bitmap, MullamaError> {
        let path_c = CString::new(path)
            .map_err(|_| MullamaError::InvalidInput("Path contains null byte".to_string()))?;

        let ptr = unsafe { sys::mtmd_helper_bitmap_init_from_file(self.ptr, path_c.as_ptr()) };

        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to load media from file: {}",
                path
            )));
        }

        Ok(Bitmap { ptr })
    }

    /// Load a bitmap from a buffer (image or audio file data)
    pub fn bitmap_from_buffer(&self, data: &[u8]) -> Result<Bitmap, MullamaError> {
        let ptr =
            unsafe { sys::mtmd_helper_bitmap_init_from_buf(self.ptr, data.as_ptr(), data.len()) };

        if ptr.is_null() {
            return Err(MullamaError::MultimodalError(
                "Failed to create bitmap from buffer".to_string(),
            ));
        }

        Ok(Bitmap { ptr })
    }

    /// Tokenize text with media markers, replacing markers with bitmaps
    ///
    /// # Arguments
    /// * `text` - Text prompt containing media markers (default: "<__media__>")
    /// * `bitmaps` - Bitmaps to substitute for each marker
    ///
    /// # Returns
    /// Input chunks ready for evaluation
    ///
    /// # Errors
    /// - Returns error if number of bitmaps doesn't match markers
    /// - Returns error if preprocessing fails
    pub fn tokenize(
        &mut self,
        text: &str,
        bitmaps: &[&Bitmap],
    ) -> Result<InputChunks, MullamaError> {
        let text_c = CString::new(text)
            .map_err(|_| MullamaError::InvalidInput("Text contains null byte".to_string()))?;

        let input_text = sys::mtmd_input_text {
            text: text_c.as_ptr(),
            add_special: true,
            parse_special: true,
        };

        let chunks = InputChunks::new();

        // Create array of bitmap pointers
        let bitmap_ptrs: Vec<*const sys::mtmd_bitmap> =
            bitmaps.iter().map(|b| b.as_ptr()).collect();

        let result = unsafe {
            sys::mtmd_tokenize(
                self.ptr,
                chunks.ptr,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };

        match result {
            0 => Ok(chunks),
            1 => Err(MullamaError::InvalidInput(format!(
                "Number of bitmaps ({}) doesn't match markers in text",
                bitmaps.len()
            ))),
            2 => Err(MullamaError::MultimodalError(
                "Image preprocessing failed".to_string(),
            )),
            _ => Err(MullamaError::MultimodalError(format!(
                "Tokenization failed with code: {}",
                result
            ))),
        }
    }

    /// Evaluate chunks in a llama context
    ///
    /// This is the main function for processing multimodal input.
    /// It handles both text and media chunks automatically.
    ///
    /// # Arguments
    /// * `context` - The llama context to evaluate in
    /// * `chunks` - The tokenized input chunks
    /// * `n_past` - Current position in the context
    /// * `seq_id` - Sequence ID (usually 0)
    /// * `n_batch` - Batch size for processing
    /// * `logits_last` - Whether to compute logits for the last token
    ///
    /// # Returns
    /// New position after evaluation
    pub fn eval_chunks(
        &mut self,
        context: &mut crate::Context,
        chunks: &InputChunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32, MullamaError> {
        let mut new_n_past: i32 = 0;

        let result = unsafe {
            sys::mtmd_helper_eval_chunks(
                self.ptr,
                context.as_ptr(),
                chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &mut new_n_past,
            )
        };

        if result != 0 {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to evaluate multimodal chunks: error code {}",
                result
            )));
        }

        Ok(new_n_past)
    }

    /// Encode a single chunk and get the embeddings
    pub fn encode_chunk(&mut self, chunk: &InputChunk) -> Result<(), MullamaError> {
        let result = unsafe { sys::mtmd_encode_chunk(self.ptr, chunk.ptr) };

        if result != 0 {
            return Err(MullamaError::MultimodalError(format!(
                "Failed to encode chunk: error code {}",
                result
            )));
        }

        Ok(())
    }

    /// Get the output embeddings from the last encode operation
    ///
    /// The size depends on the chunk that was encoded.
    pub fn get_output_embeddings(&self, chunk: &InputChunk) -> Option<&[f32]> {
        let ptr = unsafe { sys::mtmd_get_output_embd(self.ptr) };
        if ptr.is_null() {
            return None;
        }

        let n_tokens = chunk.n_tokens();
        let n_embd = self.model.n_embd() as usize;
        let size = n_tokens * n_embd;

        Some(unsafe { std::slice::from_raw_parts(ptr, size) })
    }

    /// Get the default media marker string
    pub fn default_marker() -> String {
        let ptr = unsafe { sys::mtmd_default_marker() };
        if ptr.is_null() {
            "<__media__>".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
        }
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::mtmd_free(self.ptr) };
        }
    }
}
