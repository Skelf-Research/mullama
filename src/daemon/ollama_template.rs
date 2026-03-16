//! Ollama Go Template Converter
//!
//! Converts Ollama's Go template syntax to mullama's chat template format.
//!
//! ## Go Template Syntax
//!
//! Ollama uses Go's text/template syntax:
//! - `{{ .System }}` - System message
//! - `{{ .Prompt }}` - User prompt
//! - `{{ .Response }}` - Assistant response
//! - `{{- if .System }}...{{- end }}` - Conditional
//! - `{{ range .Messages }}...{{ end }}` - Loop over messages
//!
//! ## Mullama Template Format
//!
//! Mullama uses a simpler placeholder format:
//! - `{system}` - System message
//! - `{user}` or `{prompt}` - User message
//! - `{assistant}` - Assistant response

use regex::Regex;

/// Converted chat template
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The converted template string
    pub template: String,
    /// BOS (beginning of sequence) token if any
    pub bos_token: Option<String>,
    /// EOS (end of sequence) token if any
    pub eos_token: Option<String>,
    /// Stop sequences extracted from template
    pub stop_sequences: Vec<String>,
}

impl ChatTemplate {
    /// Create from an Ollama Go template string
    pub fn from_ollama_template(go_template: &str) -> Self {
        let converter = GoTemplateConverter::new();
        converter.convert(go_template)
    }

    /// Apply the template with given values
    pub fn apply(&self, system: Option<&str>, user: &str, assistant_prefix: bool) -> String {
        let mut result = self.template.clone();

        // Handle system message
        if let Some(sys) = system {
            result = result.replace("{system}", sys);
            // Remove conditional markers if system is present
            result = result.replace("{if_system}", "");
            result = result.replace("{end_if_system}", "");
        } else {
            // Remove system block if no system message
            result = remove_conditional_block(&result, "{if_system}", "{end_if_system}");
        }

        // Replace user/prompt placeholder
        result = result.replace("{user}", user);
        result = result.replace("{prompt}", user);

        // Handle assistant prefix
        if assistant_prefix {
            result = result.replace("{assistant}", "");
        } else {
            // Remove assistant placeholder for generation
            if let Some(pos) = result.find("{assistant}") {
                result = result[..pos].to_string();
            }
        }

        result
    }
}

/// Helper to remove a conditional block
fn remove_conditional_block(s: &str, start_marker: &str, end_marker: &str) -> String {
    if let Some(start) = s.find(start_marker) {
        if let Some(end) = s.find(end_marker) {
            let before = &s[..start];
            let after = &s[end + end_marker.len()..];
            return format!("{}{}", before, after);
        }
    }
    s.to_string()
}

/// Converts Go templates to mullama format
struct GoTemplateConverter;

impl GoTemplateConverter {
    fn new() -> Self {
        Self
    }

    fn convert(&self, go_template: &str) -> ChatTemplate {
        let mut template = go_template.to_string();
        let mut stop_sequences = Vec::new();

        // Convert Go template variables to mullama placeholders
        template = self.convert_variables(&template);

        // Convert conditionals
        template = self.convert_conditionals(&template);

        // Convert range loops (simplified - just mark the structure)
        template = self.convert_ranges(&template);

        // Extract potential stop sequences from the template
        stop_sequences.extend(self.extract_stop_sequences(&template));

        // Clean up whitespace control markers
        template = self.clean_whitespace_markers(&template);

        // Detect BOS/EOS tokens
        let bos_token = self.detect_bos_token(&template);
        let eos_token = self.detect_eos_token(&template);

        ChatTemplate {
            template,
            bos_token,
            eos_token,
            stop_sequences,
        }
    }

    /// Convert Go template variables to mullama placeholders
    fn convert_variables(&self, template: &str) -> String {
        let mut result = template.to_string();

        // Main variable mappings
        let mappings = [
            (r"\{\{\s*\.System\s*\}\}", "{system}"),
            (r"\{\{\s*\.Prompt\s*\}\}", "{user}"),
            (r"\{\{\s*\.Response\s*\}\}", "{assistant}"),
            (r"\{\{\s*\.First\s*\}\}", ""), // Loop index, usually not needed
            // Handle .Content in message loops
            (r"\{\{\s*\.Content\s*\}\}", "{content}"),
            (r"\{\{\s*\.Role\s*\}\}", "{role}"),
        ];

        for (pattern, replacement) in mappings {
            let re = Regex::new(pattern).unwrap();
            result = re.replace_all(&result, replacement).to_string();
        }

        // Handle whitespace-trimming variants
        let trimming_mappings = [
            (r"\{\{-\s*\.System\s*-?\}\}", "{system}"),
            (r"\{\{-?\s*\.System\s*-\}\}", "{system}"),
            (r"\{\{-\s*\.Prompt\s*-?\}\}", "{user}"),
            (r"\{\{-?\s*\.Prompt\s*-\}\}", "{user}"),
            (r"\{\{-\s*\.Response\s*-?\}\}", "{assistant}"),
            (r"\{\{-?\s*\.Response\s*-\}\}", "{assistant}"),
        ];

        for (pattern, replacement) in trimming_mappings {
            let re = Regex::new(pattern).unwrap();
            result = re.replace_all(&result, replacement).to_string();
        }

        result
    }

    /// Convert Go template conditionals
    fn convert_conditionals(&self, template: &str) -> String {
        let mut result = template.to_string();

        // Convert if .System conditionals
        // {{- if .System }}...{{- end }}
        let if_system_re = Regex::new(r"\{\{-?\s*if\s+\.System\s*-?\}\}").unwrap();
        result = if_system_re.replace_all(&result, "{if_system}").to_string();

        // Convert if .First conditionals (for message loops)
        let if_first_re = Regex::new(r"\{\{-?\s*if\s+\.First\s*-?\}\}").unwrap();
        result = if_first_re.replace_all(&result, "{if_first}").to_string();

        // Convert if not .First
        let if_not_first_re = Regex::new(r"\{\{-?\s*if\s+not\s+\.First\s*-?\}\}").unwrap();
        result = if_not_first_re
            .replace_all(&result, "{if_not_first}")
            .to_string();

        // Convert else
        let else_re = Regex::new(r"\{\{-?\s*else\s*-?\}\}").unwrap();
        result = else_re.replace_all(&result, "{else}").to_string();

        // Convert end
        let end_re = Regex::new(r"\{\{-?\s*end\s*-?\}\}").unwrap();
        // Try to match end markers to their corresponding if markers
        // For simplicity, just use a generic marker
        result = end_re.replace_all(&result, "{end_if_system}").to_string();

        result
    }

    /// Convert Go template ranges (loops)
    fn convert_ranges(&self, template: &str) -> String {
        let mut result = template.to_string();

        // {{ range .Messages }}...{{ end }}
        let range_re = Regex::new(r"\{\{-?\s*range\s+\.Messages\s*-?\}\}").unwrap();
        result = range_re
            .replace_all(&result, "{foreach_message}")
            .to_string();

        // The end marker for ranges
        // Note: This is simplified; in practice we'd need to properly match nested structures
        result = result.replace(
            "{end_if_system}{end_if_system}",
            "{end_foreach}{end_if_system}",
        );

        result
    }

    /// Clean up whitespace control markers
    fn clean_whitespace_markers(&self, template: &str) -> String {
        let mut result = template.to_string();

        // Remove any remaining Go template markers
        let remaining_re = Regex::new(r"\{\{-?[^}]*-?\}\}").unwrap();
        result = remaining_re.replace_all(&result, "").to_string();

        // Normalize multiple newlines
        let multi_newline_re = Regex::new(r"\n{3,}").unwrap();
        result = multi_newline_re.replace_all(&result, "\n\n").to_string();

        result.trim().to_string()
    }

    /// Extract potential stop sequences from the template
    fn extract_stop_sequences(&self, template: &str) -> Vec<String> {
        let mut sequences = Vec::new();

        // Look for common end-of-turn markers
        let markers = [
            "<|end|>",
            "<|eot_id|>",
            "<|im_end|>",
            "<|start_header_id|>", // llama3 - new message header start indicates end of turn
            "</s>",
            "[/INST]",
            "<</SYS>>",
            "\n\nHuman:",
            "\n\nAssistant:",
        ];

        for marker in markers {
            if template.contains(marker) {
                sequences.push(marker.to_string());
            }
        }

        sequences
    }

    /// Detect BOS token
    fn detect_bos_token(&self, template: &str) -> Option<String> {
        let bos_patterns = ["<s>", "<|begin_of_text|>", "<|startoftext|>"];
        for pattern in bos_patterns {
            if template.starts_with(pattern) || template.contains(pattern) {
                return Some(pattern.to_string());
            }
        }
        None
    }

    /// Detect EOS token
    fn detect_eos_token(&self, template: &str) -> Option<String> {
        let eos_patterns = ["</s>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"];
        for pattern in eos_patterns {
            if template.ends_with(pattern) || template.contains(pattern) {
                return Some(pattern.to_string());
            }
        }
        None
    }
}

/// Format a chat conversation using the template
pub fn format_chat(
    template: &ChatTemplate,
    messages: &[(String, String)], // (role, content) pairs
    add_generation_prompt: bool,
) -> String {
    let mut result = String::new();

    // Find system message if any
    let system_msg = messages
        .iter()
        .find(|(role, _)| role == "system")
        .map(|(_, content)| content.as_str());

    // Get user messages
    let user_messages: Vec<_> = messages
        .iter()
        .filter(|(role, _)| role == "user" || role == "assistant")
        .collect();

    if let Some((_, last_user_content)) = user_messages.last() {
        result = template.apply(system_msg, last_user_content, !add_generation_prompt);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_template_conversion() {
        let go_template = r#"{{ .System }}
{{ .Prompt }}
{{ .Response }}"#;

        let template = ChatTemplate::from_ollama_template(go_template);
        assert!(template.template.contains("{system}"));
        assert!(template.template.contains("{user}"));
        assert!(template.template.contains("{assistant}"));
    }

    #[test]
    fn test_conditional_template() {
        let go_template = r#"{{- if .System }}<|system|>{{ .System }}<|end|>{{- end }}
<|user|>{{ .Prompt }}<|end|>
<|assistant|>"#;

        let template = ChatTemplate::from_ollama_template(go_template);
        assert!(template.template.contains("{if_system}"));
        assert!(template.template.contains("{system}"));
        assert!(template.stop_sequences.contains(&"<|end|>".to_string()));
    }

    #[test]
    fn test_llama3_style_template() {
        let go_template = r#"<|begin_of_text|>{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"#;

        let template = ChatTemplate::from_ollama_template(go_template);
        assert!(template.bos_token.is_some());
        assert!(template.stop_sequences.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_apply_template() {
        let template = ChatTemplate {
            template: "{if_system}<|system|>{system}<|end|>{end_if_system}<|user|>{user}<|end|><|assistant|>{assistant}".to_string(),
            bos_token: None,
            eos_token: Some("<|end|>".to_string()),
            stop_sequences: vec!["<|end|>".to_string()],
        };

        let result = template.apply(Some("You are helpful."), "Hello!", true);
        assert!(result.contains("You are helpful."));
        assert!(result.contains("Hello!"));
    }
}
