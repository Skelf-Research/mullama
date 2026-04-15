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
    Router::new()
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/copy", post(ollama_copy))
        .route("/api/delete", delete(ollama_delete))
        .route("/api/embeddings", post(ollama_embeddings))
        .route("/api/ps", get(ollama_ps))
        .route("/api/version", get(ollama_version))
        .with_state(daemon)
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
            return Json(OllamaGenerateResponse {
                model: model_alias,
                response: format!("model '{}' not found", req.model),
                done: true,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                eval_count: None,
            })
            .into_response();
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
                let _created = crate::daemon::protocol::unix_timestamp_secs();
                let model_name = Arc::new(model_name);
                let model_alias = Arc::new(model_alias);
                let ndjson_body = futures::stream::unfold((ReceiverStream::new(rx), false, model_name, model_alias, prompt_tokens), |(mut stream, finished, model_name, model_alias, prompt_tokens)| async move {
                    if finished {
                        return None;
                    }
                    if let Some(chunk) = stream.next().await {
                        let resp = OllamaGenerateResponse {
                            model: model_name.to_string(),
                            response: chunk.delta,
                            done: false,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: Some(1),
                        };
                        Some((format!("{}\n", serde_json::to_string(&resp).unwrap_or_default()), (stream, false, model_name, model_alias, prompt_tokens)))
                    } else {
                        let final_resp = OllamaGenerateResponse {
                            model: model_alias.to_string(),
                            response: String::new(),
                            done: true,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: None,
                        };
                        Some((format!("{}\n", serde_json::to_string(&final_resp).unwrap_or_default()), (stream, true, model_name, model_alias, prompt_tokens)))
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
                return Json(OllamaGenerateResponse {
                    model: model_alias,
                    response: format!("streaming error: {}", error_msg),
                    done: true,
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_count: None,
                    eval_count: None,
                })
                .into_response();
            }
        }
    }

    let start = std::time::Instant::now();

    let result = {
        let mut ctx = loaded.acquire_context().await;
        let model = loaded.model.clone();
        let tokens = match model.tokenize(&req.prompt, true, false) {
            Ok(t) => t,
            Err(e) => {
                return Json(OllamaGenerateResponse {
                    model: model_alias,
                    response: format!("tokenization error: {}", e),
                    done: true,
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_count: None,
                    eval_count: None,
                })
                .into_response();
            }
        };

        let prompt_tokens = tokens.len() as u32;
        let gen_result = tokio::task::block_in_place(|| {
            ctx.generate_with_params(&tokens, max_tokens as usize, &sampler_params)
        });

        match gen_result {
            Ok(text) => {
                let elapsed = start.elapsed();
                OllamaGenerateResponse {
                    model: model_alias,
                    response: text,
                    done: true,
                    total_duration: Some(elapsed.as_nanos() as u64),
                    load_duration: Some(0),
                    prompt_eval_count: Some(prompt_tokens),
                    eval_count: Some(max_tokens),
                }
            }
            Err(e) => OllamaGenerateResponse {
                model: model_alias,
                response: format!("generation error: {}", e),
                done: true,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                eval_count: None,
            },
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
            return Json(OllamaChatResponse {
                model: model_alias,
                message: Some(OllamaChatMessage {
                    role: "assistant".to_string(),
                    content: format!("model '{}' not found", req.model),
                }),
                done: true,
                total_duration: None,
                prompt_eval_count: None,
                eval_count: None,
            })
            .into_response();
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
                let ndjson_body = futures::stream::unfold((ReceiverStream::new(rx), false, model_name, model_alias, prompt_tokens), |(mut stream, finished, model_name, model_alias, prompt_tokens)| async move {
                    if finished {
                        return None;
                    }
                    if let Some(chunk) = stream.next().await {
                        let resp = OllamaChatResponse {
                            model: model_name.to_string(),
                            message: Some(OllamaChatMessage {
                                role: "assistant".to_string(),
                                content: chunk.delta,
                            }),
                            done: false,
                            total_duration: None,
                            prompt_eval_count: Some(prompt_tokens),
                            eval_count: Some(1),
                        };
                        Some((format!("{}\n", serde_json::to_string(&resp).unwrap_or_default()), (stream, false, model_name, model_alias, prompt_tokens)))
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
                            eval_count: None,
                        };
                        Some((format!("{}\n", serde_json::to_string(&final_resp).unwrap_or_default()), (stream, true, model_name, model_alias, prompt_tokens)))
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
                return Json(OllamaChatResponse {
                    model: model_alias,
                    message: Some(OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: format!("streaming error: {}", error_msg),
                    }),
                    done: true,
                    total_duration: None,
                    prompt_eval_count: None,
                    eval_count: None,
                })
                .into_response();
            }
        }
    }

    // Non-streaming path
    let start = std::time::Instant::now();

    let result = {
        let mut ctx = loaded.acquire_context().await;
        let tokens = match model.tokenize(&prompt, true, false) {
            Ok(t) => t,
            Err(e) => {
                return Json(OllamaChatResponse {
                    model: model_alias,
                    message: Some(OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: format!("tokenization error: {}", e),
                    }),
                    done: true,
                    total_duration: None,
                    prompt_eval_count: None,
                    eval_count: None,
                })
                .into_response();
            }
        };

        let prompt_tokens = tokens.len() as u32;
        let gen_result = tokio::task::block_in_place(|| {
            ctx.generate_with_params(&tokens, max_tokens as usize, &sampler_params)
        });

        match gen_result {
            Ok(text) => {
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
                    eval_count: Some(max_tokens),
                }
            }
            Err(e) => OllamaChatResponse {
                model: model_alias,
                message: Some(OllamaChatMessage {
                    role: "assistant".to_string(),
                    content: format!("generation error: {}", e),
                }),
                done: true,
                total_duration: None,
                prompt_eval_count: None,
                eval_count: None,
            },
        }
    };

    Json(result).into_response()
}

async fn ollama_tags(State(daemon): State<AppState>) -> impl IntoResponse {
    let models: Vec<OllamaModelInfo> = daemon
        .models
        .list()
        .into_iter()
        .map(|(alias, info, _, _)| OllamaModelInfo {
            name: alias,
            size: Some(info.parameters),
            digest: None,
            modified_at: None,
        })
        .collect();

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
) -> impl IntoResponse {
    match daemon.models.unload(&req.name).await {
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

            let tokens = match model.tokenize(&req.prompt, true, false) {
                Ok(t) => t,
                Err(e) => {
                    let resp = serde_json::json!({"error": format!("tokenization error: {}", e)});
                    return (axum::http::StatusCode::BAD_REQUEST, Json(resp)).into_response();
                }
            };

            let decode_result = tokio::task::block_in_place(|| ctx.decode(&tokens));
            if let Err(e) = decode_result {
                let resp = serde_json::json!({"error": format!("decode error: {}", e)});
                return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response();
            }

            match ctx.get_embeddings() {
                Some(emb) => Json(OllamaEmbeddingsResponse {
                    embedding: emb.to_vec(),
                })
                .into_response(),
                None => {
                    let resp =
                        serde_json::json!({"error": "embeddings not available for this model"});
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
    let models: Vec<OllamaRunningModel> = daemon
        .models
        .list()
        .into_iter()
        .map(|(alias, info, _, _)| OllamaRunningModel {
            name: alias,
            size: info.parameters,
            digest: None,
            expires_at: None,
        })
        .collect();

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
