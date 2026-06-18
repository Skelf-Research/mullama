//! Modelfile Parser for Mullama
//!
//! Supports both Ollama-compatible Modelfile syntax and extended Mullamafile features.
//!
//! ## Ollama-Compatible Keys (Modelfile)
//!
//! ```dockerfile
//! FROM llama3.2:1b
//! PARAMETER temperature 0.7
//! PARAMETER num_ctx 8192
//! SYSTEM """
//! You are a helpful assistant.
//! """
//! TEMPLATE """
//! {{ .System }}
//! {{ .Prompt }}
//! """
//! MESSAGE user Hello!
//! MESSAGE assistant Hi there!
//! ```
//!
//! ## Mullama Extensions (Mullamafile)
//!
//! ```dockerfile
//! FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
//!
//! # Standard Modelfile keys
//! PARAMETER temperature 0.7
//! PARAMETER num_ctx 8192
//!
//! SYSTEM """
//! You are a helpful assistant.
//! """
//!
//! # Mullama-specific extensions
//! ADAPTER ./my-lora-adapter.safetensors
//! GPU_LAYERS 32
//! FLASH_ATTENTION true
//! VISION_PROJECTOR ./mmproj.gguf
//!
//! # Metadata
//! LICENSE MIT
//! AUTHOR "Your Name"
//! ```

mod parser;
mod types;

pub use parser::{find_modelfile, ModelfileParser};
pub use types::{
    Capabilities, ExecutionRecord, Message, Modelfile, ModelfileError, ParameterValue,
    ThinkingConfig, ToolFormat,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_modelfile() {
        let content = r#"
FROM llama3.2:1b
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are a helpful assistant."
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.from, "llama3.2:1b");
        assert_eq!(modelfile.temperature(), Some(0.7));
        assert_eq!(modelfile.num_ctx(), Some(4096));
        assert_eq!(
            modelfile.system,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_parse_multiline_system() {
        let content = r#"
FROM llama3.2:1b
SYSTEM """
You are a helpful assistant.
You answer questions clearly.
"""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.system.is_some());
        assert!(modelfile
            .system
            .as_ref()
            .unwrap()
            .contains("You are a helpful assistant."));
        assert!(modelfile
            .system
            .as_ref()
            .unwrap()
            .contains("You answer questions clearly."));
    }

    #[test]
    fn test_parse_single_line_triple_quoted_system() {
        let content = r#"
FROM llama3.2:1b
SYSTEM """You are a helpful assistant."""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(
            modelfile.system,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_parse_multiline_template_with_inline_close() {
        let content = r#"
FROM llama3.2:1b
TEMPLATE """Line one
Line two"""
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.template, Some("Line one\nLine two".to_string()));
    }

    #[test]
    fn test_parse_mullama_extensions() {
        let content = r#"
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
GPU_LAYERS 32
FLASH_ATTENTION true
VISION_PROJECTOR ./mmproj.gguf
ADAPTER ./lora.safetensors
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.is_huggingface());
        assert_eq!(modelfile.gpu_layers, Some(32));
        assert_eq!(modelfile.flash_attention, Some(true));
        assert!(modelfile.vision_projector.is_some());
        assert!(modelfile.adapter.is_some());
    }

    #[test]
    fn test_ollama_compat_rejects_extensions() {
        let content = r#"
FROM llama3.2:1b
GPU_LAYERS 32
"#;

        let parser = ModelfileParser::ollama_compatible();
        let result = parser.parse_str(content);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_messages() {
        let content = r#"
FROM llama3.2:1b
MESSAGE user "Hello!"
MESSAGE assistant "Hi there! How can I help?"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.messages.len(), 2);
        assert_eq!(modelfile.messages[0].role, "user");
        assert_eq!(modelfile.messages[0].content, "Hello!");
        assert_eq!(modelfile.messages[1].role, "assistant");
    }

    #[test]
    fn test_missing_from() {
        let content = r#"
PARAMETER temperature 0.7
"#;

        let parser = ModelfileParser::new();
        let result = parser.parse_str(content);

        assert!(matches!(result, Err(ModelfileError::MissingFrom)));
    }

    #[test]
    fn test_serialize_modelfile() {
        let mut modelfile = Modelfile::from_model("llama3.2:1b");
        modelfile.set_parameter("temperature", ParameterValue::Float(0.7));
        modelfile.system = Some("You are helpful.".to_string());
        modelfile.gpu_layers = Some(32);

        let output = modelfile.to_string();

        assert!(output.contains("FROM llama3.2:1b"));
        assert!(output.contains("PARAMETER temperature 0.7"));
        assert!(output.contains("SYSTEM"));
        assert!(output.contains("GPU_LAYERS 32"));
    }

    #[test]
    fn test_is_alias() {
        let m1 = Modelfile::from_model("llama3.2:1b");
        assert!(m1.is_alias());

        let m2 = Modelfile::from_model("hf:TheBloke/Llama-2-7B-GGUF");
        assert!(m2.is_huggingface());
        assert!(!m2.is_alias());

        let m3 = Modelfile::from_model("./model.gguf");
        assert!(m3.is_local_path());
        assert!(!m3.is_alias());
    }

    #[test]
    fn test_parse_thinking_config() {
        let content = r#"
FROM qwq:32b
THINKING start "<think>"
THINKING end "</think>"
THINKING enabled true
CAPABILITY thinking true
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.thinking.is_some());
        let thinking = modelfile.thinking.as_ref().unwrap();
        assert_eq!(thinking.start_token, "<think>");
        assert_eq!(thinking.end_token, "</think>");
        assert!(thinking.enabled);
        assert!(modelfile.capabilities.thinking);
    }

    #[test]
    fn test_parse_multiple_stop_sequences() {
        let content = r#"
FROM qwen2.5:7b
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.stop_sequences.len(), 2);
        assert!(modelfile.stop_sequences.contains(&"<|im_end|>".to_string()));
        assert!(modelfile
            .stop_sequences
            .contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn test_parse_capabilities() {
        let content = r#"
FROM llama3.2:1b
CAPABILITY json true
CAPABILITY tools true
CAPABILITY thinking false
CAPABILITY vision true
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.capabilities.json);
        assert!(modelfile.capabilities.tools);
        assert!(!modelfile.capabilities.thinking);
        assert!(modelfile.capabilities.vision);
    }

    #[test]
    fn test_parse_toolformat() {
        let content = r#"
FROM qwen2.5:7b
TOOLFORMAT style "qwen"
TOOLFORMAT call_start "<tool_call>"
TOOLFORMAT call_end "</tool_call>"
TOOLFORMAT result_start "<tool_response>"
TOOLFORMAT result_end "</tool_response>"
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert!(modelfile.tool_format.is_some());
        let tf = modelfile.tool_format.as_ref().unwrap();
        assert_eq!(tf.style, "qwen");
        assert_eq!(tf.call_start, "<tool_call>");
        assert_eq!(tf.call_end, "</tool_call>");
        assert_eq!(tf.result_start, "<tool_response>");
        assert_eq!(tf.result_end, "</tool_response>");
    }

    #[test]
    fn test_serialize_thinking_and_capabilities() {
        let mut modelfile = Modelfile::from_model("qwq:32b");
        modelfile.thinking = Some(ThinkingConfig {
            start_token: "<think>".to_string(),
            end_token: "</think>".to_string(),
            enabled: true,
        });
        modelfile.capabilities.thinking = true;
        modelfile.stop_sequences.push("<|im_end|>".to_string());

        let output = modelfile.to_string();

        assert!(output.contains("THINKING start \"<think>\""));
        assert!(output.contains("THINKING end \"</think>\""));
        assert!(output.contains("THINKING enabled true"));
        assert!(output.contains("CAPABILITY thinking true"));
        assert!(output.contains("PARAMETER stop \"<|im_end|>\""));
    }

    #[test]
    fn test_parse_revision_from_hf_spec() {
        let content = r#"
FROM hf:meta-llama/Llama-3.2-1B-Instruct-GGUF@abc123def
PARAMETER temperature 0.7
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(modelfile.from, "hf:meta-llama/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(modelfile.revision, Some("abc123def".to_string()));
        assert_eq!(
            modelfile.full_model_ref(),
            "hf:meta-llama/Llama-3.2-1B-Instruct-GGUF@abc123def"
        );
    }

    #[test]
    fn test_parse_digest() {
        let content = r#"
FROM hf:org/model
DIGEST sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        assert_eq!(
            modelfile.digest,
            Some(
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_parse_digest_without_prefix() {
        let content = r#"
FROM hf:org/model
DIGEST e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        // Should auto-prefix with sha256:
        assert_eq!(
            modelfile.digest,
            Some(
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_serialize_with_revision_and_digest() {
        let mut modelfile = Modelfile::from_model("hf:org/repo");
        modelfile.revision = Some("main".to_string());
        modelfile.digest = Some("sha256:abc123".to_string());

        let output = modelfile.to_string();

        assert!(output.contains("FROM hf:org/repo@main"));
        assert!(output.contains("DIGEST sha256:abc123"));
    }

    #[test]
    fn test_execution_record_id_generation() {
        let id1 = ExecutionRecord::generate_id();
        let id2 = ExecutionRecord::generate_id();

        assert!(id1.starts_with("exec_"));
        assert!(id2.starts_with("exec_"));
        // IDs should be unique (technically could collide but extremely unlikely)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_execution_record_config_hash() {
        let hash1 = ExecutionRecord::hash_config(0.7, 0.9, 40, 4096, &["<|im_end|>".to_string()]);
        let hash2 = ExecutionRecord::hash_config(0.7, 0.9, 40, 4096, &["<|im_end|>".to_string()]);
        let hash3 = ExecutionRecord::hash_config(0.8, 0.9, 40, 4096, &["<|im_end|>".to_string()]);

        // Same config = same hash
        assert_eq!(hash1, hash2);
        // Different config = different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_execution_record_log_line() {
        let record = ExecutionRecord {
            id: "exec_test123".to_string(),
            timestamp: 1700000000000,
            model_digest: "sha256:abcdef123456".to_string(),
            model_ref: "hf:org/model".to_string(),
            revision: None,
            config_hash: "abc123".to_string(),
            backend_version: "1.0.0".to_string(),
            gpu_info: None,
            context_size: 4096,
            gpu_layers: 32,
            temperature: 0.7,
            prompt_tokens: 100,
            completion_tokens: 50,
            duration_ms: 1500,
            success: true,
            error: None,
        };

        let log_line = record.to_log_line();

        assert!(log_line.contains("\"id\":\"exec_test123\""));
        assert!(log_line.contains("\"success\":true"));
        assert!(log_line.contains("\"tokens\":150"));
    }

    #[test]
    fn test_local_path_no_revision_extraction() {
        // Local paths with @ should NOT have revision extracted
        let content = r#"
FROM ./models/my@special-model.gguf
"#;

        let parser = ModelfileParser::new();
        let modelfile = parser.parse_str(content).unwrap();

        // Should keep the full path, no revision extraction
        assert_eq!(modelfile.from, "./models/my@special-model.gguf");
        assert_eq!(modelfile.revision, None);
    }
}
