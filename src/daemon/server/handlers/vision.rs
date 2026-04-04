use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::super::{prompt::resolve_chat_stop_sequences, Daemon};
use crate::daemon::models::RequestGuard;
use crate::daemon::protocol::{
    ChatCompletionParams, ChatMessage, ErrorCode, Response, StreamChunk,
};

/// Extract bitmaps from chat messages containing base64 image data URIs.
///
/// Shared by both streaming and non-streaming vision handlers.
fn extract_bitmaps_from_messages(
    messages: &[ChatMessage],
    mtmd: &crate::MtmdContext,
) -> Result<Vec<crate::Bitmap>, String> {
    use base64::Engine;

    let mut bitmaps = Vec::new();
    for msg in messages {
        for img_url in msg.content.images() {
            let url = &img_url.url;
            if let Some(base64_data) = url
                .strip_prefix("data:")
                .and_then(|s| s.split_once(',').map(|(_, data)| data))
            {
                let image_bytes = base64::engine::general_purpose::STANDARD
                    .decode(base64_data)
                    .map_err(|e| format!("Invalid base64 image data: {}", e))?;
                let bitmap = mtmd
                    .bitmap_from_buffer(&image_bytes)
                    .map_err(|e| format!("Failed to load image: {}", e))?;
                bitmaps.push(bitmap);
            } else {
                return Err(
                    "Image URL must be a base64 data URI (data:image/...;base64,...)".to_string(),
                );
            }
        }
    }
    Ok(bitmaps)
}

impl Daemon {
    pub async fn handle_vision_chat_completion(
        &self,
        params: ChatCompletionParams,
    ) -> Response {
        if let Err(resp) = self.validate_max_tokens(params.max_tokens) {
            return resp;
        }

        let loaded = match self.models.get(params.model.as_deref()).await {
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

        let sampler_params = self.build_chat_sampler(&loaded, &params);
        let messages =
            self.apply_default_system_prompt(params.messages, loaded.config.system_prompt.as_deref());

        let mtmd_ref = match loaded.mtmd_context.as_ref() {
            Some(r) => r,
            None => {
                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                return Response::error(
                    ErrorCode::InvalidRequest,
                    "No multimodal context available. Load with --mmproj to enable vision.",
                );
            }
        };
        let bitmaps = {
            let mtmd_guard = mtmd_ref.read().await;
            match extract_bitmaps_from_messages(&messages, &mtmd_guard) {
                Ok(b) => b,
                Err(msg) => {
                    self.active_requests.fetch_sub(1, Ordering::Relaxed);
                    return Response::error(ErrorCode::InvalidRequest, msg);
                }
            }
        };

        let prompt = self.build_vision_prompt(&loaded.model, &messages);
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        let result = self
            .generate_vision_text(
                &loaded,
                &prompt,
                &bitmaps,
                params.max_tokens,
                sampler_params,
                &all_stops,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                super::build_chat_completion_response(&loaded.alias, text, prompt_tokens, completion_tokens)
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    pub async fn handle_vision_chat_completion_streaming(
        &self,
        params: ChatCompletionParams,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(params.max_tokens)?;

        let loaded = match self.models.get(params.model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        if !loaded.has_multimodal() {
            return Err(Response::error(
                ErrorCode::InvalidRequest,
                "Model does not have multimodal support. Load with --mmproj to enable vision.",
            ));
        }

        let sampler_params = self.build_chat_sampler(&loaded, &params);
        let messages =
            self.apply_default_system_prompt(params.messages, loaded.config.system_prompt.as_deref());

        let bitmaps = {
            let mtmd_ref = loaded.mtmd_context.as_ref().ok_or_else(|| {
                Response::error(
                    ErrorCode::InvalidRequest,
                    "No multimodal context available. Load with --mmproj to enable vision.",
                )
            })?;
            let mtmd_guard = mtmd_ref.read().await;
            extract_bitmaps_from_messages(&messages, &mtmd_guard).map_err(|msg| {
                Response::error(ErrorCode::InvalidRequest, msg)
            })?
        };

        let prompt = self.build_vision_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        match self
            .generate_vision_text_streaming(
                loaded,
                prompt,
                bitmaps,
                params.max_tokens,
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
