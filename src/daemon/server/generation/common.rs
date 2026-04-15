use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use super::super::super::protocol::{ResponseFormat, StreamChunk};
use super::super::prompt::find_stop_in_recent_window;
use crate::{Context, Model, MullamaError, SamplerChain};

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
                return Ok(GenerationResult {
                    generated,
                    completion_tokens,
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
        context.decode_single(next_token)?;
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
