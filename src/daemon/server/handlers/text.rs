use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::super::{
    prompt::{merge_stop_sequences, resolve_chat_stop_sequences},
    Daemon,
};
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
        let messages = self
            .apply_default_system_prompt(params.messages, loaded.config.system_prompt.as_deref());
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let model_alias = loaded.alias.clone();
        let grammar = crate::daemon::server::generation::resolve_chat_grammar(
            params.response_format.as_ref(),
            params.tools.as_deref(),
            params.tool_choice.as_ref(),
        );
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        // Phase-C streaming route: dispatch through the batched scheduler
        // when present (concurrent streams share one decode call). Falls
        // back to the legacy pool path on miss — same surface, identical
        // mpsc<StreamChunk> shape.
        // Streaming sessions get prefix-reuse too: extract session_id from
        // params and pass it to the batched scheduler so the slot's cached
        // tokens are matched. Durable restore via the batched path is also
        // wired: we look up the session in the session store and pass any
        // saved KV blob through so cold-restart streams can hydrate.
        let session_id = params.session.clone().filter(|s| !s.is_empty());
        let kv_reuse = session_id.as_ref().map(|id| {
            let lookup = self
                .sessions
                .get(id, &loaded.alias, loaded.pool_size(), &loaded.kv_compat);
            crate::daemon::server::generation::KvReuse {
                slot: lookup.slot,
                cached_tokens: lookup.cached_tokens,
                restore: lookup.restore,
            }
        });
        let batched_available = loaded.batcher.read().await.is_some();
        let stream_result = if batched_available {
            self.generate_text_streaming_batched(
                loaded,
                prompt,
                params.max_tokens,
                sampler_params,
                all_stops,
                grammar,
                session_id,
                kv_reuse,
            )
            .await
        } else {
            self.generate_text_streaming(
                loaded,
                prompt,
                params.max_tokens,
                sampler_params,
                all_stops,
                grammar,
            )
            .await
        };
        match stream_result {
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
        // Sliding-window pruning: when a session is active and a turn bound is
        // requested, drop everything older than the last N user turns before
        // rendering the prompt. This bounds both the prompt and the pinned KV
        // so a long session can't overflow `n_ctx`. Stateless requests keep
        // their full history — pruning only helps the reuse path.
        let session_id = params.session.clone().filter(|s| !s.is_empty());
        let messages = match (session_id.as_ref(), params.session_keep_turns) {
            (Some(_), Some(n)) if n > 0 => {
                super::super::prompt::trim_to_last_n_user_turns(&messages, n)
            }
            _ => messages,
        };
        let prompt = self.build_chat_prompt(&loaded.model, &messages);
        let all_stops = resolve_chat_stop_sequences(&loaded, params.stop);

        // Agent file-access prefetch: record source paths mentioned in this
        // turn's content so the idle hydrator can predict the agent's next
        // reads. Only the latest turn's messages are scanned (recency); the
        // observer dedups and bounds its own window.
        if let Some(id) = session_id.as_ref() {
            for m in &messages {
                self.prefetch.observe(id, &m.content.text());
            }
        }

        // Cross-turn KV reuse: if a session id is present, pin to its slot and
        // prefill only the new delta this turn. A fresh daemon restores the
        // pinned slot's KV from the durable store first (see `session.rs`).
        let kv_reuse = session_id.as_ref().map(|id| {
            let lookup = self
                .sessions
                .get(id, &loaded.alias, loaded.pool_size(), &loaded.kv_compat);
            crate::daemon::server::generation::KvReuse {
                slot: lookup.slot,
                cached_tokens: lookup.cached_tokens,
                restore: lookup.restore,
            }
        });

        let grammar = crate::daemon::server::generation::resolve_chat_grammar(
            params.response_format.as_ref(),
            params.tools.as_deref(),
            params.tool_choice.as_ref(),
        );
        // Phase-C route: always use the batched scheduler when available.
        // Both session affinity (cached prefix) AND durable session
        // restore (load_state_seq from saved blob) are now handled
        // batched-side. Falls back to the legacy pool path only when no
        // batcher is configured.
        let batched_available = loaded.batcher.read().await.is_some();
        let result = if batched_available {
            self.generate_text_batched(
                &loaded,
                &prompt,
                params.max_tokens,
                sampler_params,
                &all_stops,
                grammar,
                session_id.clone(),
                kv_reuse,
            )
            .await
        } else {
            self.generate_text(
                &loaded,
                &prompt,
                params.max_tokens,
                sampler_params,
                &all_stops,
                grammar,
                kv_reuse,
            )
            .await
        };

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens, timings, new_cached, seq_state)) => {
                // Write back the updated cached-token sequence for the session,
                // and persist the seq-state blob for restart-tolerance.
                if let (Some(id), Some(cached)) = (session_id.as_ref(), new_cached) {
                    self.sessions.put(
                        id,
                        &loaded.alias,
                        &loaded.kv_compat,
                        cached,
                        seq_state,
                    );
                }
                self.store.update_model_stats(
                    &loaded.alias,
                    1,
                    completion_tokens as u64,
                    prompt_tokens as u64,
                    0,
                );
                super::build_chat_completion_response(
                    &loaded.alias,
                    text,
                    prompt_tokens,
                    completion_tokens,
                    Some(timings),
                )
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
                None,
            )
            .await;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok((text, prompt_tokens, completion_tokens, timings, _new_cached, _seq_state)) => {
                self.store.update_model_stats(
                    &loaded.alias,
                    1,
                    completion_tokens as u64,
                    prompt_tokens as u64,
                    0,
                );
                super::build_completion_response(
                    &loaded.alias,
                    text,
                    prompt_tokens,
                    completion_tokens,
                    Some(timings),
                )
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
            .generate_text_streaming(
                loaded,
                params.prompt,
                params.max_tokens,
                sampler_params,
                all_stops,
                None,
            )
            .await
        {
            Ok((rx, prompt_tokens, request_id)) => Ok((rx, prompt_tokens, request_id, model_alias)),
            Err(e) => Err(Response::error(ErrorCode::GenerationFailed, e.to_string())),
        }
    }
}
