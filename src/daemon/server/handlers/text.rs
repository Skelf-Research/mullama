use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::super::{prompt::{merge_stop_sequences, resolve_chat_stop_sequences}, Daemon};
use crate::daemon::models::RequestGuard;
use crate::daemon::protocol::{
    ChatCompletionParams, CompletionParams, ErrorCode, Response, StreamChunk,
};

impl Daemon {
    pub async fn handle_chat_completion_streaming(
        &self,
        params: ChatCompletionParams,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(params.max_tokens)?;

        let loaded = match self.models.get(params.model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let sampler_params = self.build_chat_sampler(&loaded, &params);
        let messages =
            self.apply_default_system_prompt(params.messages, loaded.config.system_prompt.as_deref());
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        match self
            .generate_text_streaming(loaded, prompt, params.max_tokens, sampler_params, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }

    pub(crate) async fn handle_chat_completion(&self, params: ChatCompletionParams) -> Response {
        if params.stream {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Streaming chat over IPC Request::ChatCompletion is not supported; use streaming HTTP endpoints",
            );
        }
        if let Err(resp) = self.validate_max_tokens(params.max_tokens) {
            return resp;
        }

        let loaded = match self.models.get(params.model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let sampler_params = self.build_chat_sampler(&loaded, &params);
        let messages = self
            .apply_default_system_prompt(params.messages, loaded.config.system_prompt.as_deref());
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        let result = self
            .generate_text(
                &loaded,
                &prompt,
                params.max_tokens,
                sampler_params,
                &all_stops,
                params.response_format.as_ref(),
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                self.store.update_model_stats(
                    &loaded.alias,
                    1,
                    completion_tokens as u64,
                    prompt_tokens as u64,
                    0,
                );
                super::build_chat_completion_response(&loaded.alias, text, prompt_tokens, completion_tokens)
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    pub(crate) async fn handle_completion(&self, params: CompletionParams) -> Response {
        if params.stream {
            return Response::error(
                ErrorCode::InvalidRequest,
                "Streaming completion over IPC Request::Completion is not supported; use /v1/completions with stream=true",
            );
        }
        if let Err(resp) = self.validate_max_tokens(params.max_tokens) {
            return resp;
        }

        let loaded = match self.models.get(params.model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Response::error(ErrorCode::ModelNotFound, e.to_string()),
        };

        let _guard = RequestGuard::new(loaded.clone());
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let sampler_params = self.build_completion_sampler(&loaded, &params);
        let all_stops = merge_stop_sequences(loaded.config.stop_sequences.clone(), params.stop);
        let result = self
            .generate_text(
                &loaded,
                &params.prompt,
                params.max_tokens,
                sampler_params,
                &all_stops,
                None,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens)) => {
                self.store.update_model_stats(
                    &loaded.alias,
                    1,
                    completion_tokens as u64,
                    prompt_tokens as u64,
                    0,
                );
                super::build_completion_response(&loaded.alias, text, prompt_tokens, completion_tokens)
            }
            Err(e) => Response::error(ErrorCode::GenerationFailed, e.to_string()),
        }
    }

    pub async fn handle_completion_streaming(
        &self,
        params: CompletionParams,
    ) -> Result<(mpsc::Receiver<StreamChunk>, u32, String, String), Response> {
        self.validate_max_tokens(params.max_tokens)?;

        let loaded = match self.models.get(params.model.as_deref()).await {
            Ok(m) => m,
            Err(e) => return Err(Response::error(ErrorCode::ModelNotFound, e.to_string())),
        };

        let model_alias = loaded.alias.clone();
        let sampler_params = self.build_completion_sampler(&loaded, &params);
        let all_stops = merge_stop_sequences(loaded.config.stop_sequences.clone(), params.stop);

        match self
            .generate_text_streaming(loaded, params.prompt, params.max_tokens, sampler_params, all_stops)
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }
}
