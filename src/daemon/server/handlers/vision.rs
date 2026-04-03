use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::super::{prompt::merge_stop_sequences, Daemon};
use crate::daemon::models::RequestGuard;
use crate::daemon::protocol::{
    generate_completion_id, ChatChoice, ChatCompletionResponse, ChatMessage, ErrorCode, Response,
    StreamChunk, Usage,
};

impl Daemon {
    #[allow(clippy::too_many_arguments)]
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

        use crate::Bitmap;
        use base64::Engine;

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

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

        let mut bitmaps: Vec<Bitmap> = Vec::new();
        let mtmd_ref = match loaded.mtmd_context.as_ref() {
            Some(r) => r,
            None => {
                return Response::error(
                    ErrorCode::InvalidRequest,
                    "No multimodal context available. Load with --mmproj to enable vision.",
                );
            }
        };
        let mtmd_guard = mtmd_ref.read().await;

        for msg in &messages {
            for img_url in msg.content.images() {
                let url = &img_url.url;
                if let Some(base64_data) = url.strip_prefix("data:").and_then(|s| {
                    s.split_once(',').map(|(_, data)| data)
                }) {
                    match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                        Ok(image_bytes) => match mtmd_guard.bitmap_from_buffer(&image_bytes) {
                            Ok(bitmap) => bitmaps.push(bitmap),
                            Err(e) => {
                                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                                return Response::error(
                                    ErrorCode::InvalidRequest,
                                    format!("Failed to load image: {}", e),
                                );
                            }
                        },
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

        let prompt = self.build_vision_prompt(&loaded.model, &messages);

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

    #[allow(clippy::too_many_arguments)]
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

        let loaded = match self.models.get(model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        if !loaded.has_multimodal() {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            ));
        }

        let messages =
            self.apply_default_system_prompt(messages, loaded.config.system_prompt.as_deref());

        let mut bitmaps: Vec<Bitmap> = Vec::new();
        {
            let mtmd_ref = loaded.mtmd_context.as_ref().ok_or_else(|| {
                Response::error(
                    ErrorCode::InvalidRequest,
                    "No multimodal context available. Load with --mmproj to enable vision.",
                )
            })?;
            let mtmd_guard = mtmd_ref.read().await;

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

        let prompt = self.build_vision_prompt(&loaded.model, &messages);
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
}
