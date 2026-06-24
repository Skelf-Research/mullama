use crate::SamplerParams;

use super::super::models::{LoadedModel, ModelConfig};
use super::super::ollama::{OllamaClient, OllamaModel};
use super::super::protocol::ChatMessage;
#[cfg(feature = "multimodal")]
use super::super::protocol::{ContentPart, MessageContent};
use super::Daemon;

/// Trim a message history to the last `keep_turns` user turns, preserving any
/// leading system block. A "turn" is one user message plus every non-user
/// message (assistant / tool) that follows it up to the next user message, so
/// trimming never splits an assistant reply from its prompt.
///
/// This is the server-side half of "pruning old turns": bounding the rendered
/// prompt also bounds the pinned session's KV cache (the cache-is-the-
/// conversation model from `session.rs`), so a long-running agent can't grow
/// toward `n_ctx` and overflow. Older turns are simply forgotten — the
/// standard context-eviction trade-off, not a free lunch.
pub(super) fn trim_to_last_n_user_turns(messages: &[ChatMessage], keep_turns: u32) -> Vec<ChatMessage> {
    if keep_turns == 0 {
        return messages.to_vec();
    }
    // Preserve any leading run of system messages verbatim.
    let sys_end = messages
        .iter()
        .position(|m| !m.role.eq_ignore_ascii_case("system"))
        .unwrap_or(messages.len());

    let rest = &messages[sys_end..];
    if rest.is_empty() {
        return messages.to_vec();
    }

    // Walk from the end, counting user messages until we've seen `keep_turns`
    // of them; the cut point is the index (within `rest`) of that user msg.
    let mut user_count = 0u32;
    let mut cut = 0usize;
    for (i, m) in rest.iter().enumerate().rev() {
        if m.role.eq_ignore_ascii_case("user") {
            user_count += 1;
            cut = i;
            if user_count >= keep_turns {
                break;
            }
        }
    }
    if user_count < keep_turns {
        // Fewer user turns than the bound: keep everything.
        return messages.to_vec();
    }

    let mut out = Vec::with_capacity(sys_end + (rest.len() - cut));
    out.extend_from_slice(&messages[..sys_end]);
    out.extend_from_slice(&rest[cut..]);
    out
}

#[inline]
pub(super) fn find_stop_in_recent_window(
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

/// Resolve stop sequences for chat endpoints.
///
/// Prefers stop sequences from the model config; falls back to model's built-in
/// chat stop tokens. Then merges in any user-supplied stop sequences.
pub(crate) fn resolve_chat_stop_sequences(
    loaded: &LoadedModel,
    user_stops: Vec<String>,
) -> Vec<String> {
    let default_stops = if !loaded.config.stop_sequences.is_empty() {
        loaded.config.stop_sequences.clone()
    } else {
        loaded.model.get_chat_stop_sequences()
    };
    merge_stop_sequences(default_stops, user_stops)
}

pub(crate) fn merge_stop_sequences(base: Vec<String>, additional: Vec<String>) -> Vec<String> {
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

fn model_config_from_ollama_model(model: &OllamaModel) -> ModelConfig {
    ModelConfig {
        stop_sequences: model.get_stop_sequences(),
        system_prompt: model.system_prompt.clone(),
        temperature: model.parameters.temperature,
        top_p: model.parameters.top_p,
        top_k: model.parameters.top_k,
        context_size: model.parameters.num_ctx,
    }
}

pub(super) fn infer_ollama_model_config(path: &str) -> Option<ModelConfig> {
    let target = std::fs::canonicalize(path).ok()?;
    let client = OllamaClient::new().ok()?;

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

impl Daemon {
    pub(super) fn apply_default_system_prompt(
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
    fn build_sampler_params_inner(
        &self,
        loaded: &LoadedModel,
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

    /// Build sampler params from a `ChatCompletionParams` struct.
    ///
    /// Uses the standard default temperature of 0.7.
    pub(super) fn build_chat_sampler(
        &self,
        loaded: &LoadedModel,
        params: &super::super::protocol::ChatCompletionParams,
    ) -> SamplerParams {
        self.build_sampler_params_inner(
            loaded,
            params.temperature,
            params.top_p,
            params.top_k,
            params.frequency_penalty,
            params.presence_penalty,
            0.7,
        )
    }

    /// Build sampler params from a `CompletionParams` struct.
    ///
    /// Uses the standard default temperature of 0.7.
    pub(super) fn build_completion_sampler(
        &self,
        loaded: &LoadedModel,
        params: &super::super::protocol::CompletionParams,
    ) -> SamplerParams {
        self.build_sampler_params_inner(
            loaded,
            params.temperature,
            params.top_p,
            params.top_k,
            params.frequency_penalty,
            params.presence_penalty,
            0.7,
        )
    }

    pub fn build_chat_prompt(&self, model: &crate::Model, messages: &[ChatMessage]) -> String {
        let text_contents: Vec<String> = messages.iter().map(|m| m.content.text()).collect();
        let msg_tuples: Vec<(&str, &str)> = messages
            .iter()
            .zip(text_contents.iter())
            .map(|(m, content)| (m.role.as_str(), content.as_str()))
            .collect();

        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(e) => {
                tracing::warn!(
                    "Chat template failed: {}. Using generic format. \
                    Model may produce suboptimal output.",
                    e
                );

                let mut prompt = String::new();

                for (msg, content) in messages.iter().zip(text_contents.iter()) {
                    match msg.role.as_str() {
                        "system" => prompt.push_str(&format!("System: {}\n\n", content)),
                        "user" => prompt.push_str(&format!("User: {}\n\n", content)),
                        "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", content)),
                        _ => prompt.push_str(&format!("{}: {}\n\n", msg.role, content)),
                    }
                }

                prompt.push_str("Assistant:");
                prompt
            }
        }
    }

    /// Build prompt for vision models with image markers
    #[cfg(feature = "multimodal")]
    pub(super) fn build_vision_prompt(
        &self,
        model: &crate::Model,
        messages: &[ChatMessage],
    ) -> String {
        let mut processed_messages: Vec<(String, String)> = Vec::new();

        for msg in messages {
            let mut content = String::new();
            match &msg.content {
                MessageContent::Text(s) => content = s.clone(),
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => content.push_str(text),
                            ContentPart::ImageUrl { .. } => content.push_str("<__media__>"),
                        }
                    }
                }
            }
            processed_messages.push((msg.role.clone(), content));
        }

        let msg_tuples: Vec<(&str, &str)> = processed_messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();

        match model.apply_chat_template(None, &msg_tuples, true) {
            Ok(formatted) => formatted,
            Err(_) => {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::protocol::MessageContent;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn roles(msgs: &[ChatMessage]) -> Vec<&str> {
        msgs.iter().map(|m| m.role.as_str()).collect()
    }

    // [sys, u1, a1, u2, a2, u3, a3, u4]: three complete turns + a final user.
    fn sample() -> Vec<ChatMessage> {
        vec![
            msg("system", "s"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
            msg("assistant", "a2"),
            msg("user", "u3"),
            msg("assistant", "a3"),
            msg("user", "u4"),
        ]
    }

    #[test]
    fn trim_keep_last_two_user_turns() {
        let m = sample();
        let out = trim_to_last_n_user_turns(&m, 2);
        // system + u3,a3 + u4
        assert_eq!(roles(&out), vec!["system", "user", "assistant", "user"]);
    }

    #[test]
    fn trim_keep_one_user_turn_drops_assistant_of_older_turns() {
        let m = sample();
        let out = trim_to_last_n_user_turns(&m, 1);
        assert_eq!(roles(&out), vec!["system", "user"]);
        assert_eq!(out[1].content.text(), "u4");
    }

    #[test]
    fn trim_keep_zero_is_noop() {
        let m = sample();
        let out = trim_to_last_n_user_turns(&m, 0);
        assert_eq!(out.len(), m.len());
    }

    #[test]
    fn trim_bound_larger_than_history_is_noop() {
        let m = sample();
        let out = trim_to_last_n_user_turns(&m, 99);
        assert_eq!(out.len(), m.len());
    }

    #[test]
    fn trim_preserves_all_leading_system_messages() {
        let m = vec![
            msg("system", "s1"),
            msg("system", "s2"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
        ];
        let out = trim_to_last_n_user_turns(&m, 1);
        assert_eq!(roles(&out), vec!["system", "system", "user"]);
    }
}
