//! Daemon server implementation
//!
//! Core daemon that manages models and handles requests from IPC and HTTP.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::mpsc;

use super::models::{ModelLoadConfig, ModelManager, RequestGuard, DEFAULT_CONTEXT_POOL_SIZE};
use super::protocol::*;
use crate::embedding::{EmbeddingConfig, EmbeddingGenerator};
use crate::memory_monitor::{MemoryConfig, MemoryMonitor, MemoryPressure, RecoveryManager};
use crate::{MullamaError, SamplerParams};

/// Daemon server configuration
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// IPC socket address
    pub ipc_addr: String,
    /// HTTP port (None to disable)
    pub http_port: Option<u16>,
    /// HTTP bind address
    pub http_addr: String,
    /// Default context size for new models
    pub default_context_size: u32,
    /// Default GPU layers
    pub default_gpu_layers: i32,
    /// Number of contexts in each model's context pool
    pub default_context_pool_size: usize,
    /// Number of threads per model
    pub threads_per_model: i32,
    /// HTTP API key (Bearer token / x-api-key)
    pub http_api_key: Option<String>,
    /// Enforce API key authentication for HTTP endpoints
    pub enforce_http_api_key: bool,
    /// Hard cap on max_tokens for generation requests
    pub max_tokens_per_request: u32,
    /// Maximum accepted HTTP request body size in bytes
    pub max_request_body_bytes: usize,
    /// Maximum concurrent in-flight HTTP requests
    pub max_concurrent_http_requests: usize,
    /// Maximum requests per second for HTTP endpoints
    pub max_requests_per_second: u64,
    /// Memory monitoring configuration
    pub memory_config: MemoryConfig,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            ipc_addr: super::DEFAULT_SOCKET.to_string(),
            http_port: Some(super::DEFAULT_HTTP_PORT),
            http_addr: "127.0.0.1".to_string(),
            default_context_size: 4096,
            default_gpu_layers: 0,
            default_context_pool_size: DEFAULT_CONTEXT_POOL_SIZE,
            threads_per_model: (num_cpus::get() / 2).max(1) as i32,
            http_api_key: None,
            enforce_http_api_key: false,
            max_tokens_per_request: 4096,
            max_request_body_bytes: 2 * 1024 * 1024,
            max_concurrent_http_requests: 64,
            max_requests_per_second: 200,
            memory_config: MemoryConfig::default(),
            enable_memory_monitoring: true,
        }
    }
}

#[inline]
fn find_stop_in_recent_window(
    generated: &str,
    previous_len: usize,
    stop_sequences: &[String],
    max_stop_len: usize,
) -> Option<usize> {
    if stop_sequences.is_empty() || max_stop_len == 0 {
        return None;
    }

    let mut start = previous_len.saturating_sub(max_stop_len.saturating_sub(1));
    while start > 0 && !generated.is_char_boundary(start) {
        start -= 1;
    }

    let window = &generated[start..];
    for stop in stop_sequences {
        if let Some(relative_pos) = window.find(stop) {
            return Some(start + relative_pos);
        }
    }

    None
}

fn merge_stop_sequences(base: Vec<String>, additional: Vec<String>) -> Vec<String> {
    let mut merged = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for stop in base.into_iter().chain(additional.into_iter()) {
        if stop.is_empty() {
            continue;
        }
        if seen.insert(stop.clone()) {
            merged.push(stop);
        }
    }

    merged
}

fn model_config_from_ollama_model(
    model: &super::ollama::OllamaModel,
) -> super::models::ModelConfig {
    super::models::ModelConfig {
        stop_sequences: model.get_stop_sequences(),
        system_prompt: model.system_prompt.clone(),
        temperature: model.parameters.temperature,
        top_p: model.parameters.top_p,
        top_k: model.parameters.top_k,
        context_size: model.parameters.num_ctx,
    }
}

fn infer_ollama_model_config(path: &str) -> Option<super::models::ModelConfig> {
    let target = std::fs::canonicalize(path).ok()?;
    let client = super::ollama::OllamaClient::new().ok()?;

    for model in client.list_cached() {
        let Ok(cached_path) = std::fs::canonicalize(&model.gguf_path) else {
            continue;
        };
        if cached_path == target {
            return Some(model_config_from_ollama_model(&model));
        }
    }

    None
}

/// The daemon server
pub struct Daemon {
    pub config: DaemonConfig,
    pub models: Arc<ModelManager>,
    pub start_time: Instant,
    pub shutdown: Arc<AtomicBool>,
    pub active_requests: Arc<AtomicU32>,
    pub total_requests: AtomicU64,
    /// Cancellation flags for streaming requests (request_id -> cancel flag)
    pub cancellations: Arc<DashMap<String, Arc<AtomicBool>>>,
    /// Memory monitor for tracking system and GPU memory
    pub memory_monitor: Option<Arc<MemoryMonitor>>,
    /// Recovery manager for handling OOM situations
    pub recovery_manager: RecoveryManager,
}

impl Daemon {
    /// Create a new daemon
    pub fn new(config: DaemonConfig) -> Self {
        // Initialize memory monitor if enabled
        let memory_monitor = if config.enable_memory_monitoring {
            let monitor = MemoryMonitor::new(config.memory_config.clone());
            monitor.start(); // Start background monitoring
            Some(monitor)
        } else {
            None
        };

        // Create recovery manager with monitor if available
        let recovery_manager = if let Some(ref monitor) = memory_monitor {
            RecoveryManager::new().with_monitor(Arc::clone(monitor))
        } else {
            RecoveryManager::new()
        };

        Self {
            config,
            models: Arc::new(ModelManager::new()),
            start_time: Instant::now(),
            shutdown: Arc::new(AtomicBool::new(false)),
            active_requests: Arc::new(AtomicU32::new(0)),
            total_requests: AtomicU64::new(0),
            cancellations: Arc::new(DashMap::new()),
            memory_monitor,
            recovery_manager,
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_max_tokens(&self, max_tokens: u32) -> Result<(), Response> {
        if max_tokens == 0 {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "max_tokens must be greater than 0",
            ));
        }

        if max_tokens > self.config.max_tokens_per_request {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                format!(
                    "max_tokens {} exceeds server limit {}",
                    max_tokens, self.config.max_tokens_per_request
                ),
            ));
        }

        Ok(())
    }

    fn register_cancellation(&self, request_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.cancellations
            .insert(request_id.to_string(), Arc::clone(&flag));
        flag
    }

    pub fn cancel_request(&self, request_id: &str) -> bool {
        if let Some(flag) = self.cancellations.get(request_id) {
            flag.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Handle a request
    pub async fn handle_request(&self, request: Request) -> Response {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match request {
            Request::Ping => Response::Pong {
                uptime_secs: self.start_time.elapsed().as_secs(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },

            Request::Status => self.handle_status().await,
            Request::ListModels => self.handle_list_models().await,

            Request::LoadModel {
                alias,
                path,
                gpu_layers,
                context_size,
            } => {
                self.handle_load_model(alias, path, gpu_layers, context_size)
                    .await
            }

            Request::UnloadModel { alias } => self.handle_unload_model(&alias).await,
            Request::SetDefaultModel { alias } => self.handle_set_default(&alias).await,

            Request::ChatCompletion {
                model,
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                stream,
                stop,
                response_format,
                tools: _,
                tool_choice: _,
                thinking: _,
            } => {
                self.handle_chat_completion(
                    model,
                    messages,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    presence_penalty,
                    stream,
                    stop,
                    response_format,
                )
                .await
            }

            Request::Completion {
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                stream,
                stop,
            } => {
                self.handle_completion(
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    presence_penalty,
                    stream,
                    stop,
                )
                .await
            }

            Request::Embeddings { model, input } => self.handle_embeddings(model, input).await,

            Request::Tokenize { model, text } => self.handle_tokenize(model, &text).await,

            Request::Cancel { request_id } => {
                if self.cancel_request(&request_id) {
                    Response::Cancelled { request_id }
                } else {
                    Response::error(
                        ErrorCode::InvalidRequest,
                        format!("No active request found with id '{}'", request_id),
                    )
                }
            }

            Request::Shutdown => {
                self.shutdown.store(true, Ordering::SeqCst);
                Response::ShuttingDown
            }
        }
    }

    async fn handle_status(&self) -> Response {
        let default_model = self.models.default_alias();

        // Get memory stats from monitor if available
        let memory_used_mb = self
            .memory_monitor
            .as_ref()
            .map(|m| {
                let stats = m.stats();
                // Use GPU memory if available, otherwise system memory
                let used = if stats.gpu_total > 0 {
                    stats.gpu_used
                } else {
                    stats.system_used
                };
                used / (1024 * 1024)
            })
            .unwrap_or(0);

        Response::Status(DaemonStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            models_loaded: self.models.count(),
            default_model,
            http_endpoint: self
                .config
                .http_port
                .map(|p| format!("http://{}:{}", self.config.http_addr, p)),
            ipc_endpoint: self.config.ipc_addr.clone(),
            stats: DaemonStats {
                requests_total: self.total_requests.load(Ordering::Relaxed),
                tokens_generated: self.models.total_tokens(),
                active_requests: self.active_requests.load(Ordering::Relaxed),
                memory_used_mb,
                gpu_available: crate::supports_gpu_offload(),
            },
        })
    }

    async fn handle_list_models(&self) -> Response {
        let models = self.models.list();
        Response::Models(
            models
                .into_iter()
                .map(|(alias, info, is_default, active)| ModelStatus {
                    alias,
                    info,
                    is_default,
                    active_requests: active,
                })
                .collect(),
        )
    }

    async fn handle_load_model(
        &self,
        alias: String,
        path: String,
        gpu_layers: i32,
        context_size: u32,
    ) -> Response {
        let mut resolved_context_size = if context_size == 0 {
            self.config.default_context_size
        } else {
            context_size
        };

        let mut config = ModelLoadConfig::new(&alias, &path)
            .gpu_layers(if gpu_layers == 0 {
                self.config.default_gpu_layers
            } else {
                gpu_layers
            })
            .context_size(resolved_context_size)
            .context_pool_size(self.config.default_context_pool_size)
            .threads(self.config.threads_per_model);

        if let Some(ollama_config) = infer_ollama_model_config(&path) {
            if context_size == 0 {
                if let Some(ctx) = ollama_config.context_size {
                    resolved_context_size = ctx;
                }
                config = config.context_size(resolved_context_size);
            }
            config = config.with_config(ollama_config);
        }

        match self.models.load(config).await {
            Ok(info) => Response::ModelLoaded { alias, info },
            Err(e) => Response::error(ErrorCode::ModelLoadFailed, e.to_string()),
        }
    }

    async fn handle_unload_model(&self, alias: &str) -> Response {
        match self.models.unload(alias).await {
            Ok(()) => Response::ModelUnloaded {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    async fn handle_set_default(&self, alias: &str) -> Response {
        match self.models.set_default(alias).await {
            Ok(()) => Response::DefaultModelSet {
                alias: alias.to_string(),
            },
            Err(e) => Response::error(ErrorCode::ModelNotFound, e.to_string()),
        }
    }

    fn apply_default_system_prompt(
        &self,
        messages: Vec<ChatMessage>,
        system_prompt: Option<&str>,
    ) -> Vec<ChatMessage> {
        let Some(system_prompt) = system_prompt else {
            return messages;
        };
        if system_prompt.trim().is_empty() {
            return messages;
        }
        if messages
            .iter()
            .any(|m| m.role.eq_ignore_ascii_case("system"))
        {
            return messages;
        }

        let mut with_system = Vec::with_capacity(messages.len() + 1);
        with_system.push(ChatMessage {
            role: "system".to_string(),
            content: system_prompt.to_string().into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
        with_system.extend(messages);
        with_system
    }

    #[allow(clippy::too_many_arguments)]
    fn build_sampler_params(
        &self,
        loaded: &super::models::LoadedModel,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        default_temperature: f32,
    ) -> SamplerParams {
        let mut sampler = SamplerParams::default();
        sampler.temperature = temperature
            .or(loaded.config.temperature)
            .unwrap_or(default_temperature);
        sampler.top_p = top_p.or(loaded.config.top_p).unwrap_or(sampler.top_p);
        sampler.top_k = top_k.or(loaded.config.top_k).unwrap_or(sampler.top_k);
        if let Some(v) = frequency_penalty {
            sampler.penalty_freq = v;
        }
        if let Some(v) = presence_penalty {
            sampler.penalty_present = v;
        }
        sampler
    }

    /// Handle streaming chat completion - returns receiver for SSE
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_chat_completion_streaming(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(max_tokens)?;

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());
        // Build prompt from messages using model's chat template
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();

        // Get stop sequences - prefer model config (from Ollama/Modelfile) over architecture detection
        let default_stops = if !loaded.config.stop_sequences.is_empty() {
            loaded.config.stop_sequences.clone()
        } else {
            loaded.model.get_chat_stop_sequences()
        };
        let all_stops = merge_stop_sequences(default_stops, stop);
        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );

        // Start streaming generation
        match self
            .generate_text_streaming(loaded, prompt, max_tokens, sampler_params, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_chat_completion(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stream: bool,
        stop: Vec<String>,
        response_format: Option<ResponseFormat>,
    ) -> Response {
        if stream {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Streaming chat over IPC Request::ChatCompletion is not supported; use streaming HTTP endpoints",
            );
        }
        if let Err(resp) = self.validate_max_tokens(max_tokens) {
            return resp;
        }

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());
        // Build prompt from messages using model's chat template
        let prompt = self.build_chat_prompt(&loaded.model, &messages);

        // Get stop sequences - prefer model config (from Ollama/Modelfile) over architecture detection
        let default_stops = if !loaded.config.stop_sequences.is_empty() {
            loaded.config.stop_sequences.clone()
        } else {
            loaded.model.get_chat_stop_sequences()
        };
        let all_stops = merge_stop_sequences(default_stops, stop);
        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );

        // Generate
        let result = self
            .generate_text(
                &loaded,
                &prompt,
                max_tokens,
                sampler_params,
                &all_stops,
                response_format.as_ref(),
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::ChatCompletion(ChatCompletionResponse {
                    id: generate_completion_id(),
                    object: "chat.completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text.into(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    thinking: None,
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_completion(
        &self,
        model: Option<String>,
        prompt: String,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stream: bool,
        stop: Vec<String>,
    ) -> Response {
        if stream {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Streaming completion over IPC Request::Completion is not supported; use /v1/completions with stream=true",
            );
        }
        if let Err(resp) = self.validate_max_tokens(max_tokens) {
            return resp;
        }

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );
        let default_stops = loaded.config.stop_sequences.clone();
        let all_stops = merge_stop_sequences(default_stops, stop);
        let result = self
            .generate_text(
                &loaded,
                &prompt,
                max_tokens,
                sampler_params,
                &all_stops,
                None,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::Completion(CompletionResponse {
                    id: generate_completion_id(),
                    object: "text_completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![CompletionChoice {
                        index: 0,
                        text,
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    /// Handle streaming text completion - returns receiver for SSE
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_completion_streaming(
        &self,
        model: Option<String>,
        prompt: String,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(max_tokens)?;

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let model_alias = loaded.alias.clone();
        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );
        let default_stops = loaded.config.stop_sequences.clone();
        let all_stops = merge_stop_sequences(default_stops, stop);

        match self
            .generate_text_streaming(loaded, prompt, max_tokens, sampler_params, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    /// Handle vision chat completion (images + text)
    #[cfg(feature = "multimodal")]
    pub async fn handle_vision_chat_completion(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Vec<String>,
    ) -> Response {
        if let Err(resp) = self.validate_max_tokens(max_tokens) {
            return resp;
        }

        use crate::{Bitmap, MtmdContext};
        use base64::Engine;

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        // Check if model has multimodal support
        if !loaded.has_multimodal() {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            );
        }

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());

        // Extract images from messages and decode base64
        let mut bitmaps: Vec<Bitmap> = Vec::new();
        let mtmd_guard = loaded.mtmd_context.as_ref().unwrap().read().await;

        for msg in &messages {
            for img_url in msg.content.images() {
                // Parse data URI: data:image/jpeg;base64,/9j/4AAQ...
                let url = &img_url.url;
                if let Some(base64_data) = url.strip_prefix("data:").and_then(|s| {
                    // Find the base64 part after the comma
                    s.split_once(',').map(|(_, data)| data)
                }) {
                    // Decode base64
                    match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                        Ok(image_bytes) => {
                            // Create bitmap from decoded image data
                            match mtmd_guard.bitmap_from_buffer(&image_bytes) {
                                Ok(bitmap) => bitmaps.push(bitmap),
                                Err(e) => {
                                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                                    return Response::error(
                                        ErrorCode::InvalidRequest,
                                        format!("Failed to load image: {}", e),
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            self.active_requests.fetch_sub(1, Ordering::Relaxed);
                            return Response::error(
                                ErrorCode::InvalidRequest,
                                format!("Invalid base64 image data: {}", e),
                            );
                        }
                    }
                } else {
                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                    return Response::error(
                        ErrorCode::InvalidRequest,
                        "Image URL must be a base64 data URI (data:image/...;base64,...)",
                    );
                }
            }
        }

        drop(mtmd_guard);

        // Build prompt with <__media__> markers for images
        // For VLMs, we need to place image markers where images should be processed
        let prompt = self.build_vision_prompt(&loaded.model, &messages);

        // Get stop sequences - prefer model config (from Ollama/Modelfile) over architecture detection
        let default_stops = if !loaded.config.stop_sequences.is_empty() {
            loaded.config.stop_sequences.clone()
        } else {
            loaded.model.get_chat_stop_sequences()
        };
        let all_stops = merge_stop_sequences(default_stops, stop);
        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );

        // Process with multimodal context
        let result = self
            .generate_vision_text(
                &loaded,
                &prompt,
                &bitmaps,
                max_tokens,
                sampler_params,
                &all_stops,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Response::ChatCompletion(ChatCompletionResponse {
                    id: generate_completion_id(),
                    object: "chat.completion".to_string(),
                    created,
                    model: loaded.alias.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text.into(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    thinking: None,
                })
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    /// Build prompt for vision models with image markers
    #[cfg(feature = "multimodal")]
    fn build_vision_prompt(&self, model: &crate::Model, messages: &[ChatMessage]) -> String {
        // Extract text content, replacing images with <__media__> markers
        let mut processed_messages: Vec<(String, String)> = Vec::new();

        for msg in messages {
            let mut content = String::new();
            match &msg.content {
                MessageContent::Text(s) => content = s.clone(),
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => content.push_str(text),
                            ContentPart::ImageUrl { .. } => {
                                // Insert media marker where image should go
                                content.push_str("<__media__>");
                            }
                        }
                    }
                }
            }
            processed_messages.push((msg.role.clone(), content));
        }

        // Try to use the model's built-in chat template
        let msg_tuples: Vec<(&str, &str)> = processed_messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();

        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(_) => {
                // Fallback to simple format
                let mut prompt = String::new();
                for (role, content) in &processed_messages {
                    match role.as_str() {
                        "system" => prompt.push_str(&format!("System: {}\n\n", content)),
                        "user" => prompt.push_str(&format!("User: {}\n\n", content)),
                        "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", content)),
                        _ => prompt.push_str(&format!("{}: {}\n\n", role, content)),
                    }
                }
                prompt.push_str("Assistant:");
                prompt
            }
        }
    }

    /// Generate text with vision input
    #[cfg(feature = "multimodal")]
    async fn generate_vision_text(
        &self,
        loaded: &super::models::LoadedModel,
        prompt: &str,
        bitmaps: &[crate::Bitmap],
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
    ) -> Result<(String, u32, u32), MullamaError> {
        // Get locks on context and mtmd_context (uses context pool for concurrent requests)
        let mut ctx_guard = loaded.acquire_context().await;
        let mut mtmd_guard = loaded.mtmd_context.as_ref().unwrap().write().await;

        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        // Run CPU-bound generation in blocking context
        let (generated, prompt_tokens, completion_tokens) = tokio::task::block_in_place(|| {
            // Clear KV cache
            ctx_guard.kv_cache_clear();

            // Create bitmap references for tokenize
            let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();

            // Tokenize the prompt with images
            let chunks = mtmd_guard.tokenize(prompt, &bitmap_refs)?;

            // Evaluate chunks (processes both text and images)
            // Use a reasonable default batch size
            let n_batch = 512;
            let n_past = mtmd_guard.eval_chunks(&mut ctx_guard, &chunks, 0, 0, n_batch, true)?;

            let prompt_tokens = n_past as u32;

            // Set up sampler
            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Generate tokens
            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                // Sample next token
                let next_token = sampler.sample(&mut *ctx_guard, -1);

                // Check for end of generation
                if model.vocab_is_eog(next_token) {
                    break;
                }

                // Get token text
                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    let previous_len = generated.len();
                    generated.push_str(&text);

                    if let Some(pos) = find_stop_in_recent_window(
                        &generated,
                        previous_len,
                        &stop_sequences,
                        max_stop_len,
                    ) {
                        generated.truncate(pos);
                        return Ok((generated, prompt_tokens, completion_tokens));
                    }
                }

                // Accept the token and evaluate
                sampler.accept(next_token);
                ctx_guard.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, prompt_tokens, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Handle streaming vision chat completion
    #[cfg(feature = "multimodal")]
    pub async fn handle_vision_chat_completion_streaming(
        &self,
        model: Option<String>,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(max_tokens)?;

        use crate::Bitmap;
        use base64::Engine;

        // Get model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        // Check if model has multimodal support
        if !loaded.has_multimodal() {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            ));
        }

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());

        // Extract and decode images
        let mut bitmaps: Vec<Bitmap> = Vec::new();
        {
            let mtmd_guard = loaded.mtmd_context.as_ref().unwrap().read().await;

            for msg in &messages {
                for img_url in msg.content.images() {
                    let url = &img_url.url;
                    if let Some(base64_data) = url
                        .strip_prefix("data:")
                        .and_then(|s| s.split_once(',').map(|(_, data)| data))
                    {
                        match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                            Ok(image_bytes) => match mtmd_guard.bitmap_from_buffer(&image_bytes) {
                                Ok(bitmap) => bitmaps.push(bitmap),
                                Err(e) => {
                                    return Err(Response::error(
                                        ErrorCode::InvalidRequest,
                                        format!("Failed to load image: {}", e),
                                    ));
                                }
                            },
                            Err(e) => {
                                return Err(Response::error(
                                    ErrorCode::InvalidRequest,
                                    format!("Invalid base64 image data: {}", e),
                                ));
                            }
                        }
                    } else {
                        return Err(Response::error(
                            ErrorCode::InvalidRequest,
                            "Image URL must be a base64 data URI",
                        ));
                    }
                }
            }
        }

        // Build prompt with image markers
        let prompt = self.build_vision_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();

        // Get stop sequences - prefer model config (from Ollama/Modelfile) over architecture detection
        let default_stops = if !loaded.config.stop_sequences.is_empty() {
            loaded.config.stop_sequences.clone()
        } else {
            loaded.model.get_chat_stop_sequences()
        };
        let all_stops = merge_stop_sequences(default_stops, stop);
        let sampler_params = self.build_sampler_params(
            &loaded,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            0.7,
        );

        // Start streaming generation with vision
        match self
            .generate_vision_text_streaming(
                loaded,
                prompt,
                bitmaps,
                max_tokens,
                sampler_params,
                all_stops,
            )
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    /// Generate streaming text with vision input
    #[cfg(feature = "multimodal")]
    async fn generate_vision_text_streaming(
        &self,
        loaded: std::sync::Arc<super::models::LoadedModel>,
        prompt: String,
        bitmaps: Vec<crate::Bitmap>,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        // Generate request ID - use Arc<str> for zero-copy sharing across all chunks
        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let models_ref = self.models.clone();
        let cancellations = Arc::clone(&self.cancellations);
        let active_requests = Arc::clone(&self.active_requests);
        let request_id_cleanup = request_id.clone();
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Process image chunks first, then spawn streaming task
        // We need to get both locks, process images, then stream text generation

        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            let mut context = loaded.acquire_context().await;
            let mut mtmd_context = loaded.mtmd_context.as_ref().unwrap().write().await;

            let result = tokio::task::block_in_place(|| {
                // Clear KV cache
                context.kv_cache_clear();

                // Create bitmap references
                let bitmap_refs: Vec<&crate::Bitmap> = bitmaps.iter().collect();

                // Tokenize and evaluate chunks (processes images)
                let chunks = mtmd_context.tokenize(&prompt, &bitmap_refs)?;
                let n_batch = 512;
                let _n_past =
                    mtmd_context.eval_chunks(&mut context, &chunks, 0, 0, n_batch, true)?;

                // Set up sampler
                let mut sampler = sampler_params.build_chain(model.clone())?;

                let mut generated = String::new();
                let mut index = 0u32;
                let mut sent_len = 0usize;
                let hold_back = max_stop_len.saturating_sub(1);
                let mut tokens_generated = 0u32;
                let mut last_token_id = 0i32;

                for _ in 0..max_tokens {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let next_token = sampler.sample(&mut *context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        tokens_generated += 1;
                        last_token_id = next_token;
                        let previous_len = generated.len();
                        generated.push_str(&text);

                        if let Some(pos) = find_stop_in_recent_window(
                            &generated,
                            previous_len,
                            &stop_sequences,
                            max_stop_len,
                        ) {
                            if pos > sent_len {
                                let partial = &generated[sent_len..pos];
                                let chunk = StreamChunk {
                                    request_id: request_id_arc.clone(),
                                    index,
                                    delta: partial.to_string(),
                                    token_id: next_token,
                                    thinking: None,
                                    tool_calls: None,
                                };
                                let _ = tx.blocking_send(chunk);
                            }
                            return Ok::<_, MullamaError>(tokens_generated);
                        }

                        // Keep a small tail to avoid leaking a stop prefix split across tokens.
                        let mut flush_end = generated.len().saturating_sub(hold_back);
                        while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                            flush_end -= 1;
                        }
                        if flush_end > sent_len {
                            let chunk = StreamChunk {
                                request_id: request_id_arc.clone(),
                                index,
                                delta: generated[sent_len..flush_end].to_string(),
                                token_id: next_token,
                                thinking: None,
                                tool_calls: None,
                            };
                            if tx.blocking_send(chunk).is_err() {
                                break;
                            }
                            sent_len = flush_end;
                            index += 1;
                        }
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                if sent_len < generated.len() {
                    let chunk = StreamChunk {
                        request_id: request_id_arc.clone(),
                        index,
                        delta: generated[sent_len..].to_string(),
                        token_id: last_token_id,
                        thinking: None,
                        tool_calls: None,
                    };
                    let _ = tx.blocking_send(chunk);
                }

                Ok::<_, MullamaError>(tokens_generated)
            });

            if let Ok(tokens) = result {
                models_ref.add_tokens(tokens as u64);
            }

            cancellations.remove(&request_id_cleanup);
            active_requests.fetch_sub(1, Ordering::Relaxed);
        });

        // Return prompt_tokens as 0 for now since we process asynchronously
        // The actual prompt token count will be computed inside the task
        Ok((rx, 0, request_id))
    }

    pub async fn handle_embeddings(
        &self,
        model: Option<String>,
        input: EmbeddingInput,
    ) -> Response {
        // Get the model
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        // Create an EmbeddingGenerator with default config (Mean pooling, normalized)
        let config = EmbeddingConfig::default();
        let mut generator = match EmbeddingGenerator::new(loaded.model.clone(), config) {
            Ok(g) => g,
            Err(e) => {
                return Response::error(
                    ErrorCode::Internal,
                    format!("Failed to create embedding generator: {}", e),
                )
            }
        };

        // Collect texts and generate embeddings
        let texts: Vec<String> = match &input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Multiple(texts) => texts.clone(),
        };

        // Count tokens for usage stats
        let mut total_tokens = 0usize;
        for text in &texts {
            if let Ok(tokens) = loaded.model.tokenize(text, true, false) {
                total_tokens += tokens.len();
            }
        }

        // Generate embeddings
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = match generator.embed_batch(&text_refs) {
            Ok(emb) => emb,
            Err(e) => {
                return Response::error(
                    ErrorCode::GenerationFailed,
                    format!("Failed to generate embeddings: {}", e),
                )
            }
        };

        // Build response
        let data: Vec<EmbeddingData> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, embedding)| EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: i as u32,
            })
            .collect();

        Response::Embeddings(EmbeddingsResponse {
            object: "list".to_string(),
            data,
            model: loaded.alias.clone(),
            usage: Usage {
                prompt_tokens: total_tokens as u32,
                completion_tokens: 0,
                total_tokens: total_tokens as u32,
            },
        })
    }

    async fn handle_tokenize(&self, model: Option<String>, text: &str) -> Response {
        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        match loaded.model.tokenize(text, false, false) {
            Ok(tokens) => {
                let count = tokens.len();
                Response::Tokens { tokens, count }
            }
            Err(e) => Response::error(ErrorCode::Internal, e.to_string()),
        }
    }

    pub fn build_chat_prompt(&self, model: &crate::Model, messages: &[ChatMessage]) -> String {
        // Convert ChatMessage to the format expected by apply_chat_template
        // Extract text content from each message (ignoring images for the prompt template)
        let text_contents: Vec<String> = messages.iter().map(|m| m.content.text()).collect();
        let msg_tuples: Vec<(&str, &str)> = messages
            .iter()
            .zip(text_contents.iter())
            .map(|(m, content)| (m.role.as_str(), content.as_str()))
            .collect();

        // Try to use the model's built-in chat template
        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(e) => {
                // Log warning about template fallback
                eprintln!(
                    "[WARN] Chat template failed: {}. Using generic format. \
                    Model may produce suboptimal output.",
                    e
                );

                // Fallback to simple format if template fails
                let mut prompt = String::new();

                for (msg, content) in messages.iter().zip(text_contents.iter()) {
                    match msg.role.as_str() {
                        "system" => {
                            prompt.push_str(&format!("System: {}\n\n", content));
                        }
                        "user" => {
                            prompt.push_str(&format!("User: {}\n\n", content));
                        }
                        "assistant" => {
                            prompt.push_str(&format!("Assistant: {}\n\n", content));
                        }
                        _ => {
                            prompt.push_str(&format!("{}: {}\n\n", msg.role, content));
                        }
                    }
                }

                prompt.push_str("Assistant:");
                prompt
            }
        }
    }

    pub async fn generate_text(
        &self,
        loaded: &super::models::LoadedModel,
        prompt: &str,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: &[String],
        response_format: Option<&ResponseFormat>,
    ) -> Result<(String, u32, u32), crate::MullamaError> {
        // Tokenize - respect model's BOS token setting to avoid double BOS
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        // Convert response_format to grammar string if needed
        let grammar_gbnf = match response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                // Convert JSON Schema to GBNF grammar
                match crate::structured_output::JsonSchemaConverter::convert(&json_schema.schema) {
                    Ok(grammar) => Some(grammar.to_gbnf()),
                    Err(e) => {
                        eprintln!("[WARN] Failed to convert JSON schema to grammar: {}", e);
                        None
                    }
                }
            }
            Some(ResponseFormat::JsonObject) => {
                // Use generic JSON grammar
                match crate::grammar::presets::json() {
                    Ok(grammar) => Some(grammar.to_gbnf()),
                    Err(e) => {
                        eprintln!("[WARN] Failed to create JSON grammar: {}", e);
                        None
                    }
                }
            }
            Some(ResponseFormat::Text) | None => None,
        };

        // Acquire context from pool (allows concurrent requests via round-robin selection)
        let mut context = loaded.acquire_context().await;

        // Run CPU-bound generation in a blocking context to not block the async runtime
        // block_in_place allows blocking while keeping the current task context
        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let (generated, completion_tokens) = tokio::task::block_in_place(|| {
            // Clear KV cache to start fresh for each request
            context.kv_cache_clear();

            // Setup sampler
            let mut sampler = sampler_params.build_chain(model.clone())?;

            // Add grammar sampler if response_format requires it
            if let Some(gbnf) = &grammar_gbnf {
                let grammar_sampler =
                    crate::sampling::Sampler::grammar(model.clone(), gbnf, "root")?;
                sampler.add(grammar_sampler);
            }

            // Decode prompt
            context.decode(&tokens)?;

            // Generate tokens - pre-allocate with estimated capacity
            let mut generated = String::with_capacity((max_tokens as usize) * 6);
            let mut completion_tokens = 0u32;

            for _ in 0..max_tokens {
                // Use -1 to sample from the last token's logits
                let next_token = sampler.sample(&mut context, -1);

                if model.vocab_is_eog(next_token) {
                    break;
                }

                if let Ok(text) = model.token_to_str(next_token, 0, false) {
                    let previous_len = generated.len();
                    generated.push_str(&text);

                    if let Some(pos) = find_stop_in_recent_window(
                        &generated,
                        previous_len,
                        &stop_sequences,
                        max_stop_len,
                    ) {
                        generated.truncate(pos);
                        return Ok((generated, completion_tokens));
                    }
                }

                // Accept the token to update sampler state (grammar, repetition, etc.)
                sampler.accept(next_token);
                context.decode_single(next_token)?;
                completion_tokens += 1;
            }

            Ok::<_, MullamaError>((generated, completion_tokens))
        })?;

        self.models.add_tokens(completion_tokens as u64);

        Ok((generated, prompt_tokens, completion_tokens))
    }

    /// Generate text with streaming - yields tokens as they're generated
    pub async fn generate_text_streaming(
        &self,
        loaded: Arc<super::models::LoadedModel>,
        prompt: String,
        max_tokens: u32,
        sampler_params: SamplerParams,
        stop_sequences: Vec<String>,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String), MullamaError> {
        // Tokenize - respect model's BOS token setting
        let add_bos = loaded.model.add_bos_token();
        let tokens = loaded.model.tokenize(&prompt, add_bos, false)?;
        let prompt_tokens = tokens.len() as u32;

        // Generate request ID - use Arc<str> for zero-copy sharing across all chunks
        // This is a Rust-exclusive optimization: in Go, each chunk would clone the string
        let request_id = generate_completion_id();
        let request_id_arc: Arc<str> = Arc::from(request_id.as_str());
        let cancel_flag = self.register_cancellation(&request_id);

        // Create channel for streaming chunks
        let (tx, rx) = mpsc::channel::<StreamChunk>(32);

        // Clone what we need for the spawned task
        let model = loaded.model.clone();
        let stop_sequences: Vec<String> = stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let models_ref = self.models.clone();
        let cancellations = Arc::clone(&self.cancellations);
        let active_requests = Arc::clone(&self.active_requests);
        let request_id_cleanup = request_id.clone();
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Spawn the generation task
        tokio::spawn(async move {
            let _guard = RequestGuard::new(loaded.clone());
            // Acquire context from pool (enables concurrent streaming to same model)
            let mut context = loaded.acquire_context().await;

            // Run CPU-bound generation in blocking context
            let result = tokio::task::block_in_place(|| {
                context.kv_cache_clear();

                let mut sampler = sampler_params.build_chain(model.clone())?;

                context.decode(&tokens)?;

                let mut generated = String::new();
                let mut index = 0u32;
                let mut sent_len = 0usize;
                let hold_back = max_stop_len.saturating_sub(1);
                let mut tokens_generated = 0u32;
                let mut last_token_id = 0i32;

                for _ in 0..max_tokens {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let next_token = sampler.sample(&mut context, -1);

                    if model.vocab_is_eog(next_token) {
                        break;
                    }

                    if let Ok(text) = model.token_to_str(next_token, 0, false) {
                        tokens_generated += 1;
                        last_token_id = next_token;
                        let previous_len = generated.len();
                        generated.push_str(&text);

                        if let Some(pos) = find_stop_in_recent_window(
                            &generated,
                            previous_len,
                            &stop_sequences,
                            max_stop_len,
                        ) {
                            if pos > sent_len {
                                let partial = &generated[sent_len..pos];
                                let chunk = StreamChunk {
                                    request_id: request_id_arc.clone(),
                                    index,
                                    delta: partial.to_string(),
                                    token_id: next_token,
                                    thinking: None,
                                    tool_calls: None,
                                };
                                let _ = tx.blocking_send(chunk);
                            }
                            return Ok::<_, MullamaError>(tokens_generated);
                        }

                        // Keep a small tail to avoid leaking a stop prefix split across tokens.
                        let mut flush_end = generated.len().saturating_sub(hold_back);
                        while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                            flush_end -= 1;
                        }
                        if flush_end > sent_len {
                            let chunk = StreamChunk {
                                request_id: request_id_arc.clone(),
                                index,
                                delta: generated[sent_len..flush_end].to_string(),
                                token_id: next_token,
                                thinking: None,
                                tool_calls: None,
                            };

                            // If receiver dropped, stop generation
                            if tx.blocking_send(chunk).is_err() {
                                break;
                            }

                            sent_len = flush_end;
                            index += 1;
                        }
                    }

                    sampler.accept(next_token);
                    context.decode_single(next_token)?;
                }

                if sent_len < generated.len() {
                    let chunk = StreamChunk {
                        request_id: request_id_arc.clone(),
                        index,
                        delta: generated[sent_len..].to_string(),
                        token_id: last_token_id,
                        thinking: None,
                        tool_calls: None,
                    };
                    let _ = tx.blocking_send(chunk);
                }

                Ok::<_, MullamaError>(tokens_generated)
            });

            if let Ok(tokens_generated) = result {
                models_ref.add_tokens(tokens_generated as u64);
            }

            cancellations.remove(&request_id_cleanup);
            active_requests.fetch_sub(1, Ordering::Relaxed);
        });

        Ok((rx, prompt_tokens, request_id))
    }

    /// Get current memory pressure level
    pub fn memory_pressure(&self) -> MemoryPressure {
        self.memory_monitor
            .as_ref()
            .map(|m| m.pressure())
            .unwrap_or(MemoryPressure::Normal)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Option<crate::memory_monitor::MemoryStats> {
        self.memory_monitor.as_ref().map(|m| m.stats())
    }

    /// Check if memory recovery is needed
    pub fn needs_memory_recovery(&self) -> bool {
        self.recovery_manager.needs_recovery()
    }

    /// Log memory warning if pressure is elevated
    #[allow(dead_code)]
    fn log_memory_pressure(&self) {
        if let Some(monitor) = &self.memory_monitor {
            let pressure = monitor.pressure();
            let stats = monitor.stats();

            match pressure {
                MemoryPressure::Warning => {
                    eprintln!(
                        "[WARN] Memory pressure elevated: {:.1}% GPU, {:.1}% system",
                        stats.gpu_usage() * 100.0,
                        stats.system_usage() * 100.0
                    );
                }
                MemoryPressure::Critical => {
                    eprintln!(
                        "[ERROR] Memory pressure CRITICAL: {:.1}% GPU, {:.1}% system",
                        stats.gpu_usage() * 100.0,
                        stats.system_usage() * 100.0
                    );
                }
                MemoryPressure::Emergency => {
                    eprintln!(
                        "[ERROR] Memory EMERGENCY: {:.1}% GPU, {:.1}% system - recovery needed",
                        stats.gpu_usage() * 100.0,
                        stats.system_usage() * 100.0
                    );
                }
                MemoryPressure::Normal => {}
            }
        }
    }

    /// Check if shutdown was requested
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

/// Builder for daemon configuration
pub struct DaemonBuilder {
    config: DaemonConfig,
    initial_models: Vec<ModelLoadConfig>,
}

impl DaemonBuilder {
    pub fn new() -> Self {
        Self {
            config: DaemonConfig::default(),
            initial_models: Vec::new(),
        }
    }

    pub fn ipc_socket(mut self, addr: impl Into<String>) -> Self {
        self.config.ipc_addr = addr.into();
        self
    }

    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = Some(port);
        self
    }

    pub fn disable_http(mut self) -> Self {
        self.config.http_port = None;
        self
    }

    pub fn http_addr(mut self, addr: impl Into<String>) -> Self {
        self.config.http_addr = addr.into();
        self
    }

    pub fn http_api_key(mut self, api_key: Option<String>) -> Self {
        self.config.http_api_key = api_key;
        self
    }

    pub fn enforce_http_api_key(mut self, enforce: bool) -> Self {
        self.config.enforce_http_api_key = enforce;
        self
    }

    pub fn max_tokens_per_request(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens_per_request = max_tokens;
        self
    }

    pub fn max_request_body_bytes(mut self, bytes: usize) -> Self {
        self.config.max_request_body_bytes = bytes;
        self
    }

    pub fn max_concurrent_http_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_http_requests = max;
        self
    }

    pub fn max_requests_per_second(mut self, max: u64) -> Self {
        self.config.max_requests_per_second = max;
        self
    }

    pub fn default_context_size(mut self, size: u32) -> Self {
        self.config.default_context_size = size;
        self
    }

    pub fn default_gpu_layers(mut self, layers: i32) -> Self {
        self.config.default_gpu_layers = layers;
        self
    }

    pub fn default_context_pool_size(mut self, size: usize) -> Self {
        self.config.default_context_pool_size = size.max(1);
        self
    }

    pub fn threads_per_model(mut self, threads: i32) -> Self {
        self.config.threads_per_model = threads;
        self
    }

    /// Configure memory monitoring
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory_config = config;
        self
    }

    /// Enable or disable memory monitoring
    pub fn enable_memory_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_memory_monitoring = enable;
        self
    }

    /// Add a model to load on startup (format: "alias:path" or just "path")
    pub fn model(mut self, spec: impl Into<String>) -> Self {
        let spec = spec.into();
        let (alias, path) = if let Some(pos) = spec.find(':') {
            (spec[..pos].to_string(), spec[pos + 1..].to_string())
        } else {
            // Use filename without extension as alias
            let path = std::path::Path::new(&spec);
            let alias = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".to_string());
            (alias, spec)
        };

        self.initial_models.push(
            ModelLoadConfig::new(alias, path)
                .gpu_layers(self.config.default_gpu_layers)
                .context_size(self.config.default_context_size)
                .context_pool_size(self.config.default_context_pool_size)
                .threads(self.config.threads_per_model),
        );
        self
    }

    pub fn build(self) -> (Daemon, Vec<ModelLoadConfig>) {
        (Daemon::new(self.config), self.initial_models)
    }
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_daemon() -> Daemon {
        let config = DaemonConfig {
            enable_memory_monitoring: false,
            ..DaemonConfig::default()
        };
        Daemon::new(config)
    }

    #[test]
    fn merge_stop_sequences_deduplicates_and_filters_empty() {
        let merged = merge_stop_sequences(
            vec!["</s>".to_string(), "".to_string()],
            vec!["<|eot_id|>".to_string(), "</s>".to_string()],
        );
        assert_eq!(merged, vec!["</s>", "<|eot_id|>"]);
    }

    #[test]
    fn find_stop_in_recent_window_detects_cross_token_boundary() {
        let generated = "hello<|eot_id|>";
        let previous_len = "hello<|eo".len();
        let stop_sequences = vec!["<|eot_id|>".to_string()];
        let pos = find_stop_in_recent_window(generated, previous_len, &stop_sequences, 10);
        assert_eq!(pos, Some("hello".len()));
    }

    #[test]
    fn apply_default_system_prompt_only_when_missing() {
        let daemon = test_daemon();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string().into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let with_system =
            daemon.apply_default_system_prompt(messages.clone(), Some("You are helpful."));
        assert_eq!(with_system.len(), 2);
        assert_eq!(with_system[0].role, "system");

        let with_existing = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "existing".to_string().into(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            messages[0].clone(),
        ];
        let unchanged = daemon.apply_default_system_prompt(with_existing.clone(), Some("ignored"));
        assert_eq!(unchanged.len(), with_existing.len());
        assert_eq!(unchanged[0].content.text(), "existing");
    }
}
