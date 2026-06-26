mod common;
mod streaming;
mod text;
mod tools;

#[cfg(feature = "multimodal")]
mod vision;

pub(crate) use text::KvReuse;

use super::super::protocol::{ResponseFormat, Tool, ToolChoice};

/// Resolve the GBNF grammar that should constrain decoding for a chat request,
/// combining the structured-output `response_format` and the tool-calling
/// `tools`/`tool_choice`. A forced tool call takes precedence over a JSON
/// response format (you can't satisfy both a tool-call shape and an arbitrary
/// JSON schema at once); otherwise the response-format grammar (if any) is
/// used. Returns `None` when decoding is unconstrained.
pub(crate) fn resolve_chat_grammar(
    response_format: Option<&ResponseFormat>,
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> Option<String> {
    if let Some(g) = tools::resolve_tool_grammar(tools, tool_choice) {
        return Some(g);
    }
    common::resolve_grammar(response_format)
}
