//! Streaming interfaces for real-time token generation
//!
//! This module provides streaming capabilities for real-time text generation,
//! allowing tokens to be processed as they are generated rather than waiting
//! for the entire response.
//!
//! ## Features
//!
//! - **Real-time streaming**: Process tokens as they are generated
//! - **Backpressure handling**: Automatically handles flow control
//! - **Error recovery**: Graceful error handling in streams
//! - **Cancellation**: Streams can be cancelled at any time
//! - **Multiple output formats**: Text, tokens, or structured data
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::streaming::{TokenStream, StreamConfig};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mullama::MullamaError> {
//!     let model = mullama::async_support::AsyncModel::load("model.gguf").await?;
//!     let config = StreamConfig::default().max_tokens(100);
//!
//!     let mut stream = TokenStream::new(model, "Hello, world!", config).await?;
//!
//!     while let Some(result) = stream.next().await {
//!         match result {
//!             Ok(token_data) => print!("{}", token_data.text),
//!             Err(e) => eprintln!("Stream error: {}", e),
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

#[cfg(feature = "streaming")]
use async_stream::stream;
#[cfg(feature = "streaming")]
use futures::{Stream, StreamExt};
#[cfg(feature = "streaming")]
use std::pin::Pin;
#[cfg(feature = "streaming")]
use std::task::{Context as TaskContext, Poll};
#[cfg(feature = "streaming")]
use tokio::sync::mpsc;

#[cfg(feature = "async")]
use crate::async_support::AsyncModel;
use crate::{ContextParams, MullamaError, SamplerParams, TokenId};

/// Data emitted by token streams
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
pub struct TokenData {
    /// The generated token ID
    pub token_id: TokenId,
    /// The text representation of the token
    pub text: String,
    /// Position in the generation sequence
    pub position: usize,
    /// Whether this is the final token
    pub is_final: bool,
    /// Generation probability (if available)
    pub probability: Option<f32>,
}

/// Configuration for streaming operations
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampling parameters
    pub sampler_params: SamplerParams,
    /// Context parameters
    pub context_params: ContextParams,
    /// Buffer size for the internal channel
    pub buffer_size: usize,
    /// Whether to include probabilities in the output
    pub include_probabilities: bool,
}

#[cfg(feature = "streaming")]
impl Default for StreamConfig {
    fn default() -> Self {
        let mut sampler_params = SamplerParams::default();
        sampler_params.temperature = 0.7;
        sampler_params.top_k = 40;
        sampler_params.top_p = 0.9;

        let mut context_params = ContextParams::default();
        context_params.n_ctx = 2048;

        Self {
            max_tokens: 100,
            sampler_params,
            context_params,
            buffer_size: 32,
            include_probabilities: false,
        }
    }
}

#[cfg(feature = "streaming")]
impl StreamConfig {
    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature for sampling
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.sampler_params.temperature = temperature;
        self
    }

    /// Set top-k sampling parameter
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.sampler_params.top_k = top_k;
        self
    }

    /// Set top-p sampling parameter
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.sampler_params.top_p = top_p;
        self
    }

    /// Set context size
    pub fn context_size(mut self, n_ctx: u32) -> Self {
        self.context_params.n_ctx = n_ctx;
        self
    }

    /// Enable probability reporting
    pub fn include_probabilities(mut self, include: bool) -> Self {
        self.include_probabilities = include;
        self
    }

    /// Set internal buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

/// Token stream for real-time generation
#[cfg(feature = "streaming")]
pub struct TokenStream {
    receiver: mpsc::Receiver<Result<TokenData, MullamaError>>,
    _handle: tokio::task::JoinHandle<()>,
}

#[cfg(feature = "streaming")]
impl TokenStream {
    /// Create a new token stream
    ///
    /// # Arguments
    ///
    /// * `model` - The async model to use for generation
    /// * `prompt` - Input prompt text
    /// * `config` - Stream configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mullama::streaming::{TokenStream, StreamConfig};
    /// use futures::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), mullama::MullamaError> {
    ///     let model = mullama::async_support::AsyncModel::load("model.gguf").await?;
    ///     let config = StreamConfig::default()
    ///         .max_tokens(50)
    ///         .temperature(0.8);
    ///
    ///     let mut stream = TokenStream::new(model, "Once upon a time", config).await?;
    ///
    ///     while let Some(result) = stream.next().await {
    ///         match result {
    ///             Ok(token_data) => {
    ///                 print!("{}", token_data.text);
    ///                 if token_data.is_final {
    ///                     println!("\nGeneration complete!");
    ///                 }
    ///             }
    ///             Err(e) => eprintln!("Error: {}", e),
    ///         }
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn new(
        model: AsyncModel,
        prompt: impl Into<String>,
        config: StreamConfig,
    ) -> Result<Self, MullamaError> {
        let prompt = prompt.into();
        let (sender, receiver) = mpsc::channel(config.buffer_size);

        // Tokenize prompt on current thread (fast, no blocking)
        let tokens = model.model().tokenize(&prompt, true, false)?;
        if tokens.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Prompt produced no tokens".to_string(),
            ));
        }

        // Decode prompt and create context in a blocking task
        let model_arc = model.model().clone();
        let context = model.create_context_async(config.context_params.clone()).await?;
        let sampler = config.sampler_params.build_chain(model.model().clone())?;
        let max_tokens = config.max_tokens;
        let include_probabilities = config.include_probabilities;

        // Run the entire generation loop in spawn_blocking to avoid
        // blocking the tokio runtime. Bridge tokens via std::sync::mpsc.
        let (blocking_tx, blocking_rx) = std::sync::mpsc::channel::<Result<TokenData, MullamaError>>();

        let handle = tokio::spawn(async move {
            // Forward tokens from the blocking task's std channel to the async channel
            while let Ok(item) = blocking_rx.recv() {
                if sender.send(item).await.is_err() {
                    break;
                }
            }
        });

        // Move all blocking state into the spawn_blocking task
        tokio::task::spawn_blocking(move || {
            let mut ctx = context.into_inner();

            // Decode the prompt to establish KV cache context
            if let Err(e) = ctx.decode(&tokens) {
                let _ = blocking_tx.send(Err(e));
                return;
            }

            let mut sampler = sampler;
            let mut position: usize = 0;

            for token_pos in 0..max_tokens {
                // Sample next token from the context's logits
                let next_token = sampler.sample(&mut ctx, -1);
                sampler.accept(next_token);

                // Compute probability if requested
                let token_prob = if include_probabilities {
                    let logits = ctx.get_logits_ith(-1);
                    if !logits.is_empty() && (next_token as usize) < logits.len() {
                        let token_logit = logits[next_token as usize];
                        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
                        Some((token_logit - max_logit).exp() / exp_sum)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Check for end of generation using proper EOS detection
                let is_eog = model_arc.token_is_eog(next_token);
                let is_final = is_eog || token_pos == max_tokens - 1;

                // Convert token to text
                let text = match model_arc.token_to_str(next_token, 0, false) {
                    Ok(text) => text,
                    Err(e) => {
                        let _ = blocking_tx.send(Err(e));
                        return;
                    }
                };

                // Create token data
                let token_data = TokenData {
                    token_id: next_token,
                    text,
                    position,
                    is_final,
                    probability: token_prob,
                };

                // Send token data via channel
                if blocking_tx.send(Ok(token_data)).is_err() {
                    // Receiver dropped, stop generation
                    return;
                }

                // Stop if end of generation (don't decode EOS tokens)
                if is_eog {
                    return;
                }

                // Decode the generated token to update KV cache for next iteration
                if let Err(e) = ctx.decode(std::slice::from_ref(&next_token)) {
                    let _ = blocking_tx.send(Err(e));
                    return;
                }

                position += 1;
            }
        });

        Ok(TokenStream {
            receiver,
            _handle: handle,
        })
    }

    /// Create a text-only stream that yields just the text content
    ///
    /// This is a convenience method for when you only need the generated text.
    pub async fn text_only(
        model: AsyncModel,
        prompt: impl Into<String>,
        config: StreamConfig,
    ) -> Result<impl Stream<Item = Result<String, MullamaError>>, MullamaError> {
        let stream = Self::new(model, prompt, config).await?;
        Ok(stream.map(|result| result.map(|token_data| token_data.text)))
    }

    /// Create a stream that yields complete words instead of individual tokens
    ///
    /// This buffers tokens until word boundaries are detected.
    pub async fn word_stream(
        model: AsyncModel,
        prompt: impl Into<String>,
        config: StreamConfig,
    ) -> Result<impl Stream<Item = Result<String, MullamaError>>, MullamaError> {
        let stream = Self::new(model, prompt, config).await?;

        Ok(stream! {
            let mut word_buffer = String::new();
            let mut token_stream = stream;

            while let Some(result) = token_stream.next().await {
                match result {
                    Ok(token_data) => {
                        word_buffer.push_str(&token_data.text);

                        // Check for word boundaries (simplified)
                        if token_data.text.contains(' ') || token_data.text.contains('\n') || token_data.is_final {
                            if !word_buffer.trim().is_empty() {
                                yield Ok(word_buffer.clone());
                                word_buffer.clear();
                            }
                        }
                    }
                    Err(e) => yield Err(e),
                }
            }

            // Yield any remaining content
            if !word_buffer.trim().is_empty() {
                yield Ok(word_buffer);
            }
        })
    }
}

#[cfg(feature = "streaming")]
impl Stream for TokenStream {
    type Item = Result<TokenData, MullamaError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

/// Utility functions for streaming
#[cfg(feature = "streaming")]
pub mod utils {
    use super::*;

    /// Collect a stream into a complete string
    ///
    /// This is useful when you want to use the streaming interface but
    /// need the complete result.
    pub async fn collect_to_string(
        mut stream: impl Stream<Item = Result<TokenData, MullamaError>> + Unpin,
    ) -> Result<String, MullamaError> {
        let mut result = String::new();

        while let Some(token_result) = stream.next().await {
            let token_data = token_result?;
            result.push_str(&token_data.text);
        }

        Ok(result)
    }

    /// Collect stream with metadata
    ///
    /// Returns both the complete text and generation metadata.
    pub async fn collect_with_metadata(
        mut stream: impl Stream<Item = Result<TokenData, MullamaError>> + Unpin,
    ) -> Result<GenerationResult, MullamaError> {
        let mut result = String::new();
        let mut token_count = 0;
        let mut tokens = Vec::new();

        while let Some(token_result) = stream.next().await {
            let token_data = token_result?;
            result.push_str(&token_data.text);
            tokens.push(token_data.token_id);
            token_count += 1;
        }

        Ok(GenerationResult {
            text: result,
            token_count,
            tokens,
        })
    }

    /// Generation result with metadata
    #[derive(Debug, Clone)]
    pub struct GenerationResult {
        pub text: String,
        pub token_count: usize,
        pub tokens: Vec<TokenId>,
    }
}

#[cfg(not(feature = "streaming"))]
compile_error!("Streaming support requires the 'streaming' feature to be enabled");
