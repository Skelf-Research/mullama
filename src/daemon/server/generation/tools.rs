//! Tool-call grammar synthesis for constrained function calling.
//!
//! When a chat request supplies `tools` and a `tool_choice` that requires a
//! call (`"required"`, or a specific function), we constrain decoding with a
//! GBNF grammar so the model can only emit a structurally valid tool call:
//!
//! ```json
//! {"name": "<one of the allowed tool names>", "arguments": { ... }}
//! ```
//!
//! The `name` field is constrained to an alternation of the allowed tool-name
//! string literals, so the model cannot invent a function that wasn't offered.
//! The `arguments` field is constrained to a generic JSON object value (any
//! valid JSON object) — per-tool argument *schema* enforcement is a stronger
//! future step, but constraining name + JSON-object shape already guarantees a
//! parseable, dispatchable call, which is the property the server needs.
//!
//! `tool_choice` semantics:
//! - `"none"` -> no grammar (free text; the model may answer without calling).
//! - `"auto"` -> no grammar (the model decides; we don't force a call).
//! - `"required"` -> grammar over *all* tool names.
//! - `{type:"function", function:{name}}` -> grammar fixed to that one name.
//!
//! The synthesized grammar is plain GBNF returned as a string, fed into the
//! same `Sampler::grammar` path the `response_format` grammar uses — so the
//! streaming and non-streaming generators constrain identically.

use super::super::super::protocol::{Tool, ToolChoice};

/// Decide whether `tool_choice` forces a call, and over which tool names, then
/// synthesize the constraining GBNF. Returns `None` when no constraint applies
/// (`none`/`auto`, no tools, or an unknown forced name).
pub(super) fn resolve_tool_grammar(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> Option<String> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }

    // Which names are callable under this choice?
    let names: Vec<&str> = match tool_choice {
        // Default when tools are present but no explicit choice: OpenAI treats
        // this as "auto" — don't force a call.
        None => return None,
        Some(ToolChoice::Mode(m)) => match m.as_str() {
            "required" => tools.iter().map(|t| t.function.name.as_str()).collect(),
            // "auto" / "none" / anything else -> no constraint.
            _ => return None,
        },
        Some(ToolChoice::Specific { function, .. }) => {
            // Force exactly the named function, if it's actually offered.
            let n = tools
                .iter()
                .find(|t| t.function.name == function.name)
                .map(|t| t.function.name.as_str())?;
            vec![n]
        }
    };

    if names.is_empty() {
        return None;
    }
    Some(build_tool_call_gbnf(&names))
}

/// Build a GBNF that accepts exactly one JSON tool call whose `name` is one of
/// `names` and whose `arguments` is any JSON object.
fn build_tool_call_gbnf(names: &[&str]) -> String {
    // name alternation: "\"get_weather\"" | "\"send_email\"" ...
    let name_alt = names
        .iter()
        .map(|n| format!("\"\\\"{}\\\"\"", escape_gbnf_literal(n)))
        .collect::<Vec<_>>()
        .join(" | ");

    // A compact but complete JSON value grammar for the arguments object. We
    // anchor `root` at an object so the top level is always a tool call.
    format!(
        r#"root   ::= "{{" ws "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws object ws "}}"
name   ::= {name_alt}
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{{" ws ( member ( ws "," ws member )* )? ws "}}"
member ::= string ws ":" ws value
array  ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex    ::= [0-9a-fA-F]
number ::= "-"? int frac? exp?
int    ::= "0" | [1-9] [0-9]*
frac   ::= "." [0-9]+
exp    ::= [eE] [-+]? [0-9]+
ws     ::= [ \t\n]*
"#
    )
}

/// Escape a tool name for inclusion inside a GBNF double-quoted literal that
/// itself contains an escaped JSON quote. Tool names are normally identifiers,
/// but guard against `"` and `\` defensively.
fn escape_gbnf_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::protocol::{FunctionDefinition, Tool, ToolChoice, ToolChoiceFunction};

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters: None,
                strict: false,
            },
        }
    }

    #[test]
    fn none_and_auto_and_missing_choice_yield_no_grammar() {
        let tools = vec![tool("get_weather")];
        assert!(resolve_tool_grammar(Some(&tools), None).is_none());
        assert!(resolve_tool_grammar(
            Some(&tools),
            Some(&ToolChoice::Mode("auto".into()))
        )
        .is_none());
        assert!(resolve_tool_grammar(
            Some(&tools),
            Some(&ToolChoice::Mode("none".into()))
        )
        .is_none());
        // No tools at all.
        assert!(resolve_tool_grammar(Some(&[]), Some(&ToolChoice::Mode("required".into()))).is_none());
        assert!(resolve_tool_grammar(None, Some(&ToolChoice::Mode("required".into()))).is_none());
    }

    #[test]
    fn required_includes_all_tool_names() {
        let tools = vec![tool("get_weather"), tool("send_email")];
        let g = resolve_tool_grammar(Some(&tools), Some(&ToolChoice::Mode("required".into())))
            .expect("required -> grammar");
        assert!(g.contains(r#"\"get_weather\""#));
        assert!(g.contains(r#"\"send_email\""#));
        assert!(g.contains("root"));
        assert!(g.contains(r#""\"name\"""#));
        assert!(g.contains(r#""\"arguments\"""#));
    }

    #[test]
    fn specific_choice_fixes_one_name() {
        let tools = vec![tool("get_weather"), tool("send_email")];
        let g = resolve_tool_grammar(
            Some(&tools),
            Some(&ToolChoice::Specific {
                choice_type: "function".into(),
                function: ToolChoiceFunction {
                    name: "send_email".into(),
                },
            }),
        )
        .expect("specific -> grammar");
        assert!(g.contains(r#"\"send_email\""#));
        assert!(!g.contains(r#"\"get_weather\""#), "must not allow the other tool");
    }

    #[test]
    fn specific_unknown_name_yields_no_grammar() {
        let tools = vec![tool("get_weather")];
        let g = resolve_tool_grammar(
            Some(&tools),
            Some(&ToolChoice::Specific {
                choice_type: "function".into(),
                function: ToolChoiceFunction {
                    name: "nonexistent".into(),
                },
            }),
        );
        assert!(g.is_none());
    }

    #[test]
    fn synthesized_grammar_compiles_as_gbnf() {
        // The grammar must be loadable by the grammar engine (well-formed GBNF).
        let tools = vec![tool("get_weather"), tool("do_thing")];
        let g = resolve_tool_grammar(Some(&tools), Some(&ToolChoice::Mode("required".into())))
            .unwrap();
        let parsed = crate::grammar::Grammar::from_gbnf(&g);
        assert!(parsed.is_ok(), "synthesized GBNF must parse: {:?}", parsed.err());
    }

    #[test]
    fn tool_name_with_special_chars_is_escaped() {
        let tools = vec![tool("weird\"name")];
        let g = resolve_tool_grammar(Some(&tools), Some(&ToolChoice::Mode("required".into())))
            .unwrap();
        // Should still parse despite the embedded quote.
        assert!(crate::grammar::Grammar::from_gbnf(&g).is_ok());
    }
}