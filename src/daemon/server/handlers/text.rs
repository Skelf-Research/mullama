use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::super::{prompt::merge_stop_sequences, Daemon};
use crate::daemon::models::RequestGuard;
use crate::daemon::protocol::{
    generate_completion_id, ChatChoice, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionResponse, ErrorCode, Response, ResponseFormat, StreamChunk, Usage,
};

impl Daemon {
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

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();

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

        match self
            .generate_text_streaming(loaded, prompt, max_tokens, sampler_params, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn handle_chat_completion(
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

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());
        let prompt = self.build_chat_prompt(&loaded.model, &messages);

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
    pub(crate) async fn handle_completion(
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
}
