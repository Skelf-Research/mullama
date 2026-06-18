//! # Mullama - Unified CLI
//!
//! A multi-model LLM server with IPC and OpenAI-compatible HTTP API.
//!
//! ## Commands
//!
//! ```bash
//! mullama serve       # Start the daemon server
//! mullama chat        # Interactive TUI client
//! mullama run "..."   # One-shot text generation
//! mullama models      # List loaded models
//! mullama load        # Load a model
//! mullama unload      # Unload a model
//! mullama status      # Show daemon status
//! mullama cache       # Manage model cache
//! mullama pull        # Download a model from HuggingFace
//! ```
//!
//! ## HuggingFace Model Support
//!
//! ```bash
//! # Download and serve HuggingFace models
//! mullama serve --model hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf
//!
//! # Auto-detect best quantization
//! mullama serve --model hf:TheBloke/Llama-2-7B-GGUF
//!
//! # With custom alias
//! mullama serve --model llama:hf:TheBloke/Llama-2-7B-GGUF
//!
//! # Pre-download model
//! mullama pull hf:TheBloke/Llama-2-7B-GGUF
//! ```

use clap::Parser;

#[path = "mullama/args.rs"]
mod args;
#[path = "mullama/commands.rs"]
mod commands;
#[path = "mullama/daemon_cmds.rs"]
mod daemon_cmds;
#[path = "mullama/hf_cmds/mod.rs"]
mod hf_cmds;
#[path = "mullama/runtime_cmds.rs"]
mod runtime_cmds;
#[path = "mullama/server_cmds.rs"]
mod server_cmds;
#[path = "mullama/shared.rs"]
mod shared;

pub(crate) use args::{CacheAction, Cli, Commands, DaemonAction};
use daemon_cmds::handle_daemon_action;
use hf_cmds::{
    copy_model, create_model, handle_cache_action, list_all_models, pull_model, remove_model,
    search_models, show_model_details, show_repo_info, show_running_models,
};
use runtime_cmds::{
    cli_stop_daemon, embed_text, list_models, load_model, ping_daemon, run_chat,
    run_model_with_prompt, set_default, show_status, tokenize_text, unload_model,
};
use server_cmds::run_server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing/logging with env-filter support
    // Use MULLAMA_LOG or RUST_LOG env vars to control log levels
    // e.g., MULLAMA_LOG=info or RUST_LOG=mullama=debug,tower_http=info
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MULLAMA_LOG")
                .or_else(|_| tracing_subscriber::EnvFilter::try_from_env("RUST_LOG"))
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new("mullama=info,tower_http=info")
                }),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    commands::run(cli).await
}
