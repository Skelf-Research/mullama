//! Ollama-compatible REST API endpoints
//!
//! Provides Ollama-native endpoints that use NDJSON streaming
//! (newline-delimited JSON) instead of SSE, matching Ollama's API format.

use axum::{
    body::Body,
    extract::{Json, State},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use futures::stream::StreamExt as _;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::openai::AppState;
use super::server::Daemon;

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub num_predict: Option<i32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<OllamaChatMessage>,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaShowRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaCopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaDeleteRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaPsResponse {
    pub models: Vec<OllamaRunningModel>,
}

#[derive(Debug, Serialize)]
pub struct OllamaRunningModel {
    pub name: String,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OllamaVersionResponse {
    pub version: String,
}

// ============================================================================
// Router
// ============================================================================

/// Create the Ollama-compatible API router
pub fn create_ollama_router(daemon: Arc<Daemon>) -> Router {
    let mut router = Router::new()
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/copy", post(ollama_copy))
        .route("/api/delete", delete(ollama_delete))
        .route("/api/embeddings", post(ollama_embeddings))
        .route("/api/ps", get(ollama_ps))
        .route("/api/version", get(ollama_version))
        .with_state(daemon.clone());

    if daemon.config.http.enforce_api_key {
        if let Some(api_key) = daemon.config.http.api_key.as_deref() {
            let auth = crate::daemon::openai::middleware::HttpAuthState {
                api_key: Arc::<str>::from(api_key),
            };
            router = router.layer(axum::middleware::from_fn_with_state(
                auth,
                crate::daemon::openai::middleware::require_api_key,
            ));
        }
    }

    if daemon.config.http.max_requests_per_second > 0 {
        let rate_limit_state = crate::daemon::openai::middleware::HttpRateLimitState {
            limit: daemon.config.http.max_requests_per_second,
            second: Arc::new(AtomicU64::new(
                crate::daemon::openai::types::unix_timestamp_secs(),
            )),
            count: Arc::new(AtomicU64::new(0)),
        };
        router = router.layer(axum::middleware::from_fn_with_state(
            rate_limit_state,
            crate::daemon::openai::middleware::enforce_rate_limit,
        ));
    }

    router
}

// ============================================================================
// Handlers
// ============================================================================

async fn ollama_generate(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Response {
    let model_alias = req.model.clone();
    let max_tokens = req
        .options
        .as_ref()
        .and_then(|o| o.num_predict)
        .unwrap_or(128) as u32;

    let should_stream = req.stream.unwrap_or(false);

    let loaded = match daemon.models.get(Some(&model_alias)).await {
        Ok(m) => m,
        Err(_) => {
            let resp = serde_json::json!({
                "error": format!("model '{}' not found", req.model)
            });
            return (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response();
        }
    };

    let sampler_params = options_to_sampler_params(&req.options);
    let stop_sequences = loaded.config.stop_sequences.clone();

    if should_stream {
        let params = crate::daemon::protocol::CompletionParams {
            model: Some(model_alias.clone()),
            prompt: req.prompt,
            max_tokens,
            temperature: Some(sampler_params.temperature),
            top_p: Some(sampler_params.top_p),
            top_k: Some(sampler_params.top_k),
            frequency_penalty: Some(sampler_params.penalty_freq),
            presence_penalty: Some(sampler_params.penalty_present),
            stream: false,
            stop: stop_sequences,
        };

        match daemon.handle_completion_streaming(params).await {
            Ok((rx, prompt_tokens, _request_id, model_name)) => {
                let model_name = Arc::new(model_name);
                let model_alias = Arc::new(model_alias);
                let ndjson_body = futures::stream::unfold((ReceiverStream::new(rx), false, model_name, model_alias, prompt_tokens, 0u32), |(mut stream, finished, model_name, model_alias, prompt_tokens, mut token_count)| async move {
                    if finished {
                        return None;
                    }
                    if let Some(chunk) = stream.next().await {
                        token_count += 1;
                        let resp = OllamaGenerateResponse {
                            model: model_name.to_string(),
                            response: chunk.delta,
                            done: false,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: None,
                        };
                        Some((format!("{}\n", serde_json::to_string(&resp).unwrap_or_default()), (stream, false, model_name, model_alias, prompt_tokens, token_count)))
                    } else {
                        let final_resp = OllamaGenerateResponse {
                            model: model_alias.to_string(),
                            response: String::new(),
                            done: true,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: Some(token_count),
                        };
                        Some((format!("{}\n", serde_json::to_string(&final_resp).unwrap_or_default()), (stream, true, model_name, model_alias, prompt_tokens, token_count)))
                    }
                }).map(Ok::<_, std::convert::Infallible>);

                return Response::builder()
                    .header("content-type", "application/x-ndjson")
                    .body(Body::from_stream(ndjson_body))
                    .unwrap()
                    .into_response();
            }
            Err(e) => {
                let error_msg = if let crate::daemon::protocol::Response::Error { message, .. } = e {
                    message
                } else {
                    "unknown error".to_string()
                };
                let resp = serde_json::json!({"error": format!("streaming error: {}", error_msg)});
                return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response();
            }
        }
    }

    // Non-streaming path: use daemon.generate_text() for accurate token counts
    let start = std::time::Instant::now();

    let result = match daemon.generate_text(
        &loaded,
        &req.prompt,
        max_tokens,
        sampler_params,
        &stop_sequences,
        None,
        None,
    ).await {
        Ok((text, prompt_tokens, completion_tokens, _timings, _new_cached)) => {
            let elapsed = start.elapsed();
            OllamaGenerateResponse {
                model: model_alias,
                response: text,
                done: true,
                total_duration: Some(elapsed.as_nanos() as u64),
                load_duration: Some(0),
                prompt_eval_count: Some(prompt_tokens),
                eval_count: Some(completion_tokens),
            }
        }
        Err(e) => {
            let resp = serde_json::json!({"error": format!("generation error: {}", e)});
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response();
        }
    };

    Json(result).into_response()
}

async fn ollama_chat(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> Response {
    let model_alias = req.model.clone();
    let max_tokens = req
        .options
        .as_ref()
        .and_then(|o| o.num_predict)
        .unwrap_or(128) as u32;

    let should_stream = req.stream.unwrap_or(false);

    let loaded = match daemon.models.get(Some(&model_alias)).await {
        Ok(m) => m,
        Err(_) => {
            let resp = serde_json::json!({
                "error": format!("model '{}' not found", req.model)
            });
            return (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response();
        }
    };

    let sampler_params = options_to_sampler_params(&req.options);
    let stop_sequences = loaded.config.stop_sequences.clone();

    // Format messages as prompt using chat template
    let model = loaded.model.clone();
    let msg_tuples: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();

    let prompt = match model.apply_chat_template(None, &msg_tuples, true) {
        Ok(p) => p,
        Err(_) => {
            req.messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
                + "\nassistant: "
        }
    };

    if should_stream {
        let params = crate::daemon::protocol::CompletionParams {
            model: Some(model_alias.clone()),
            prompt,
            max_tokens,
            temperature: Some(sampler_params.temperature),
            top_p: Some(sampler_params.top_p),
            top_k: Some(sampler_params.top_k),
            frequency_penalty: Some(sampler_params.penalty_freq),
            presence_penalty: Some(sampler_params.penalty_present),
            stream: false,
            stop: stop_sequences,
        };

        match daemon.handle_completion_streaming(params).await {
            Ok((rx, prompt_tokens, _request_id, model_name)) => {
                let model_name = Arc::new(model_name);
                let model_alias = Arc::new(model_alias);
                let ndjson_body = futures::stream::unfold((ReceiverStream::new(rx), false, model_name, model_alias, prompt_tokens, 0u32), |(mut stream, finished, model_name, model_alias, prompt_tokens, mut token_count)| async move {
                    if finished {
                        return None;
                    }
                    if let Some(chunk) = stream.next().await {
                        token_count += 1;
                        let resp = OllamaChatResponse {
                            model: model_name.to_string(),
                            message: Some(OllamaChatMessage {
                                role: "assistant".to_string(),
                                content: chunk.delta,
                            }),
                            done: false,
                            total_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: None,
                        };
                        Some((format!("{}\n", serde_json::to_string(&resp).unwrap_or_default()), (stream, false, model_name, model_alias, prompt_tokens, token_count)))
                    } else {
                        let final_resp = OllamaChatResponse {
                            model: model_alias.to_string(),
                            message: Some(OllamaChatMessage {
                                role: "assistant".to_string(),
                                content: String::new(),
                            }),
                            done: true,
                            total_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: Some(token_count),
                        };
                        Some((format!("{}\n", serde_json::to_string(&final_resp).unwrap_or_default()), (stream, true, model_name, model_alias, prompt_tokens, token_count)))
                    }
                }).map(Ok::<_, std::convert::Infallible>);

                return Response::builder()
                    .header("content-type", "application/x-ndjson")
                    .body(Body::from_stream(ndjson_body))
                    .unwrap()
                    .into_response();
            }
            Err(e) => {
                let error_msg = if let crate::daemon::protocol::Response::Error { message, .. } = e {
                    message
                } else {
                    "unknown error".to_string()
                };
                let resp = serde_json::json!({"error": format!("streaming error: {}", error_msg)});
                return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response();
            }
        }
    }

    // Non-streaming path: use daemon.generate_text() for accurate token counts
    let start = std::time::Instant::now();

    let result = match daemon.generate_text(
        &loaded,
        &prompt,
        max_tokens,
        sampler_params,
        &stop_sequences,
        None,
        None,
    ).await {
        Ok((text, prompt_tokens, completion_tokens, _timings, _new_cached)) => {
            let elapsed = start.elapsed();
            OllamaChatResponse {
                model: model_alias,
                message: Some(OllamaChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                }),
                done: true,
                total_duration: Some(elapsed.as_nanos() as u64),
                prompt_eval_count: Some(prompt_tokens),
                eval_count: Some(completion_tokens),
            }
        }
        Err(e) => {
            let resp = serde_json::json!({"error": format!("generation error: {}", e)});
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response();
        }
    };

    Json(result).into_response()
}

async fn ollama_tags(State(daemon): State<AppState>) -> impl IntoResponse {
    let models_info = daemon.models.list();
    let models: Vec<OllamaModelInfo> = tokio::task::spawn_blocking(move || {
        models_info
            .into_iter()
            .map(|(alias, info, _, _)| {
                let file_size = std::fs::metadata(&info.path)
                    .map(|m| m.len())
                    .unwrap_or(info.parameters);
                OllamaModelInfo {
                    name: alias,
                    size: Some(file_size),
                    digest: None,
                    modified_at: None,
                }
            })
            .collect()
    })
    .await
    .unwrap_or_default();

    Json(OllamaTagsResponse { models })
}

async fn ollama_show(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> impl IntoResponse {
    match daemon.models.get(Some(&req.name)).await {
        Ok(loaded) => {
            let desc = loaded.model.desc();
            Json(OllamaShowResponse {
                modelfile: format!("# Modelfile for {}\nFROM {}", req.name, desc),
                parameters: Some(desc),
                template: None,
            })
            .into_response()
        }
        Err(_) => {
            let resp = serde_json::json!({"error": format!("model '{}' not found", req.name)});
            (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response()
        }
    }
}

async fn ollama_copy(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaCopyRequest>,
) -> impl IntoResponse {
    let source_model = match daemon.models.get(Some(&req.source)).await {
        Ok(m) => m,
        Err(_) => {
            let resp = serde_json::json!({
                "error": format!("model '{}' not found", req.source)
            });
            return (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response();
        }
    };

    let source_info = &source_model.info;
    let source_path = source_info.path.clone();
    let source_config = source_model.config.clone();

    let mut load_config = crate::daemon::models::ModelLoadConfig::new(&req.destination, source_path);
    load_config.gpu_layers = source_info.gpu_layers;
    load_config.context_size = source_info.context_size;
    load_config.model_config = Some(source_config);

    match daemon.models.load(load_config).await {
        Ok(_) => {
            let resp = serde_json::json!({
                "status": format!("copied {} to {}", req.source, req.destination)
            });
            Json(resp).into_response()
        }
        Err(e) => {
            let resp = serde_json::json!({
                "error": format!("failed to copy model: {}", e)
            });
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response()
        }
    }
}

async fn ollama_delete(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaDeleteRequest>,
) -> Response {
    // Unload from memory first (not found is acceptable)
    let model_alias = req.name.clone();
    match daemon.models.unload(&model_alias).await {
        Ok(_) => {
            let resp = serde_json::json!({
                "status": format!("deleted {}", req.name)
            });
            Json(resp).into_response()
        }
        Err(e) => {
            let resp = serde_json::json!({
                "error": format!("failed to delete model: {}", e)
            });
            (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response()
        }
    }
}

async fn ollama_embeddings(
    State(daemon): State<AppState>,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> impl IntoResponse {
    match daemon.models.get(Some(&req.model)).await {
        Ok(loaded) => {
            let model = loaded.model.clone();
            let mut ctx = loaded.acquire_context().await;

            let result = tokio::task::block_in_place(|| -> Result<Vec<f32>, crate::MullamaError> {
                let tokens = model.tokenize(&req.prompt, true, false)?;
                ctx.decode(&tokens)?;
                ctx.get_embeddings()
                    .map(|emb| emb.to_vec())
                    .ok_or_else(|| crate::MullamaError::OperationFailed(
                        "embeddings not available for this model".to_string(),
                    ))
            });

            match result {
                Ok(embedding) => Json(OllamaEmbeddingsResponse { embedding }).into_response(),
                Err(e) => {
                    let resp = serde_json::json!({"error": e.to_string()});
                    (axum::http::StatusCode::BAD_REQUEST, Json(resp)).into_response()
                }
            }
        }
        Err(_) => {
            let resp = serde_json::json!({"error": format!("model '{}' not found", req.model)});
            (axum::http::StatusCode::NOT_FOUND, Json(resp)).into_response()
        }
    }
}

async fn ollama_ps(State(daemon): State<AppState>) -> impl IntoResponse {
    let models_info = daemon.models.list();
    let models: Vec<OllamaRunningModel> = tokio::task::spawn_blocking(move || {
        models_info
            .into_iter()
            .map(|(alias, info, _, _)| {
                let vram_size = std::fs::metadata(&info.path)
                    .map(|m| m.len())
                    .unwrap_or(info.parameters);
                OllamaRunningModel {
                    name: alias,
                    size: vram_size,
                    digest: None,
                    expires_at: None,
                }
            })
            .collect()
    })
    .await
    .unwrap_or_default();

    Json(OllamaPsResponse { models })
}

async fn ollama_version() -> impl IntoResponse {
    Json(OllamaVersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// ============================================================================
// Helpers
// ============================================================================

fn options_to_sampler_params(options: &Option<OllamaOptions>) -> crate::SamplerParams {
    let mut params = crate::SamplerParams::default();
    if let Some(opts) = options {
        if let Some(temp) = opts.temperature {
            params.temperature = temp;
        }
        if let Some(top_k) = opts.top_k {
            params.top_k = top_k;
        }
        if let Some(top_p) = opts.top_p {
            params.top_p = top_p;
        }
        if let Some(seed) = opts.seed {
            params.seed = seed;
        }
        if let Some(repeat_penalty) = opts.repeat_penalty {
            params.penalty_repeat = repeat_penalty;
        }
    }
    params
}
