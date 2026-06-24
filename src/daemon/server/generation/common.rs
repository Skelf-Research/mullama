use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use super::super::super::protocol::{ResponseFormat, StreamChunk};
use super::super::prompt::find_stop_in_recent_window;
use crate::{token::TokenId, Context, Model, MullamaError, SamplerChain};

/// Controls whether tokens are buffered or streamed.
pub(super) enum TokenSink<'a> {
    Buffer,
    Stream {
        tx: &'a mpsc::Sender<StreamChunk>,
        request_id: Arc<str>,
        cancel_flag: &'a AtomicBool,
    },
}

/// Result of the core token generation loop.
pub(super) struct GenerationResult {
    pub generated: String,
    pub completion_tokens: u32,
    pub eval_ns: u64,
    /// Token ids that were actually fed back into the KV cache (i.e. decoded
    /// via `decode_single`), in order. EOG/stop tokens that broke the loop
    /// *before* being decoded are excluded — they are not in the KV. Used by
    /// the cross-turn KV-reuse path to extend the session's cached-token
    /// sequence so the next turn's prefix match includes this turn's output.
    pub generated_tokens: Vec<TokenId>,
}

/// Core token generation loop shared by all generation paths.
pub(super) fn generate_tokens(
    context: &mut Context,
    model: &Model,
    sampler: &mut SamplerChain,
    max_tokens: u32,
    stop_sequences: &[String],
    max_stop_len: usize,
    sink: &TokenSink<'_>,
) -> Result<GenerationResult, MullamaError> {
    let mut generated = String::with_capacity(match sink {
        TokenSink::Buffer => (max_tokens as usize) * 6,
        TokenSink::Stream { .. } => 256,
    });
    let mut completion_tokens = 0u32;
    let mut eval_ns = 0u64;
    let mut generated_tokens: Vec<TokenId> = Vec::with_capacity(max_tokens as usize);

    let mut index = 0u32;
    let mut sent_len = 0usize;
    let hold_back = max_stop_len.saturating_sub(1);
    let mut last_token_id = 0i32;

    for _ in 0..max_tokens {
        if let TokenSink::Stream { cancel_flag, .. } = sink {
            if cancel_flag.load(Ordering::Relaxed) {
                break;
            }
        }

        let next_token = sampler.sample(context, -1);

        if model.vocab_is_eog(next_token) {
            // Match Ollama's eval_count convention: the sampled EOG token is
            // counted even though it is not fed back into the model. mullama's
            // loop breaks before the per-token increment below, so without
            // this it would report one fewer completion token than Ollama for
            // every EOG-terminated generation.
            completion_tokens += 1;
            break;
        }

        if let Ok(text) = model.token_to_str(next_token, 0, false) {
            let previous_len = generated.len();
            generated.push_str(&text);

            if let Some(pos) =
                find_stop_in_recent_window(&generated, previous_len, stop_sequences, max_stop_len)
            {
                if let TokenSink::Stream { tx, request_id, .. } = sink {
                    if pos > sent_len {
                        let chunk = StreamChunk {
                            request_id: request_id.clone(),
                            index,
                            delta: generated[sent_len..pos].to_string(),
                            token_id: next_token,
                            thinking: None,
                            tool_calls: None,
                        };
                        let _ = tx.blocking_send(chunk);
                    }
                }
                generated.truncate(pos);
                // Match Ollama's eval_count: the token that produced the stop
                // sequence was sampled/decoded, so it is counted even though
                // its text is discarded here.
                let completion_tokens = completion_tokens + 1;
                return Ok(GenerationResult {
                    generated,
                    completion_tokens,
                    eval_ns,
                    generated_tokens,
                });
            }

            if let TokenSink::Stream { tx, request_id, .. } = sink {
                let mut flush_end = generated.len().saturating_sub(hold_back);
                while flush_end > sent_len && !generated.is_char_boundary(flush_end) {
                    flush_end -= 1;
                }
                if flush_end > sent_len {
                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
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

            last_token_id = next_token;
        }

        sampler.accept(next_token);
        let decode_start = Instant::now();
        context.decode_single(next_token)?;
        eval_ns += decode_start.elapsed().as_nanos() as u64;
        generated_tokens.push(next_token);
        completion_tokens += 1;
    }

    if let TokenSink::Stream { tx, request_id, .. } = sink {
        if sent_len < generated.len() {
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                index,
                delta: generated[sent_len..].to_string(),
                token_id: last_token_id,
                thinking: None,
                tool_calls: None,
            };
            let _ = tx.blocking_send(chunk);
        }
    }

    Ok(GenerationResult {
        generated,
        completion_tokens,
        eval_ns,
        generated_tokens,
    })
}

/// Resolve a response format into an optional GBNF grammar string.
pub(super) fn resolve_grammar(response_format: Option<&ResponseFormat>) -> Option<String> {
    match response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            match crate::structured_output::JsonSchemaConverter::convert(&json_schema.schema) {
                Ok(grammar) => Some(grammar.to_gbnf()),
                Err(e) => {
                    tracing::warn!("Failed to convert JSON schema to grammar: {}", e);
                    None
                }
            }
        }
        Some(ResponseFormat::JsonObject) => match crate::grammar::presets::json() {
            Ok(grammar) => Some(grammar.to_gbnf()),
            Err(e) => {
                tracing::warn!("Failed to create JSON grammar: {}", e);
                None
            }
        },
        Some(ResponseFormat::Text) | None => None,
    }
}
