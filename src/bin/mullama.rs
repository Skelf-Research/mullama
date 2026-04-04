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

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use mullama::daemon::{DEFAULT_HTTP_PORT, DEFAULT_SOCKET};

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

#[derive(Parser)]
#[command(name = "mullama")]
#[command(author, version, about = "Multi-model LLM server and client")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all local models (cached and custom)
    #[command(alias = "ls")]
    List {
        /// Show detailed information (size, date, path)
        #[arg(short, long)]
        verbose: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove a model from disk
    #[command(alias = "delete")]
    Rm {
        /// Model name or path to remove
        name: String,

        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },

    /// Show running models (processes)
    Ps {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show model details
    Show {
        /// Model name to show
        name: String,

        /// Show the Modelfile/Mullamafile
        #[arg(long)]
        modelfile: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Start the daemon server
    #[command(alias = "start")]
    Serve {
        /// Models to load (format: alias:path or just path)
        /// Can be specified multiple times
        #[arg(short, long, value_name = "SPEC")]
        model: Vec<String>,

        /// Path to multimodal projector for vision models (applies to first model)
        #[arg(long)]
        mmproj: Option<PathBuf>,

        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// HTTP port for OpenAI-compatible API (0 to disable)
        #[arg(short = 'p', long, default_value_t = DEFAULT_HTTP_PORT)]
        http_port: u16,

        /// HTTP bind address
        #[arg(long, default_value = "127.0.0.1")]
        http_addr: String,

        /// API key for HTTP endpoints (Authorization: Bearer <key>)
        /// If omitted and auth is required, a secure key is generated at startup.
        #[arg(long)]
        api_key: Option<String>,

        /// Always require API key auth, even when bound to localhost.
        #[arg(long)]
        require_api_key: bool,

        /// Hard server limit for generation max_tokens
        #[arg(long, default_value = "4096")]
        max_tokens_limit: u32,

        /// Maximum HTTP request body size in MB
        #[arg(long, default_value = "2")]
        max_request_body_mb: u32,

        /// Maximum number of concurrent HTTP requests
        #[arg(long, default_value = "64")]
        max_concurrent_requests: usize,

        /// Maximum HTTP requests per second
        #[arg(long, default_value = "200")]
        max_requests_per_second: u64,

        /// Default GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Default context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Number of contexts in each loaded model pool
        #[arg(long, default_value_t = mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE)]
        context_pool_size: usize,

        /// Threads per model
        #[arg(short, long)]
        threads: Option<i32>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// TLS certificate file path (enables HTTPS)
        #[arg(long)]
        tls_cert: Option<String>,

        /// TLS private key file path
        #[arg(long)]
        tls_key: Option<String>,

        /// Enable flash attention
        #[arg(long)]
        flash_attn: bool,

        /// KV cache type for keys (f16, f32, q8_0, q4_0)
        #[arg(long)]
        cache_type_k: Option<String>,

        /// KV cache type for values (f16, f32, q8_0, q4_0)
        #[arg(long)]
        cache_type_v: Option<String>,

        /// Disable memory-mapped model loading
        #[arg(long)]
        no_mmap: bool,

        /// Lock model weights in physical memory
        #[arg(long)]
        mlock: bool,

        /// Batch size for prompt processing
        #[arg(long)]
        batch_size: Option<u32>,

        /// RoPE frequency base override
        #[arg(long)]
        rope_freq_base: Option<f32>,

        /// RoPE frequency scale override
        #[arg(long)]
        rope_freq_scale: Option<f32>,

        /// Tensor split mode (none, layer, row)
        #[arg(long)]
        split_mode: Option<String>,

        /// KV cache defragmentation threshold (0.0 to disable)
        #[arg(long)]
        defrag_thold: Option<f32>,
    },

    /// Interactive TUI chat client
    #[command(alias = "tui")]
    Chat {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Connection timeout in seconds
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },

    /// Run a model with a prompt (auto-starts daemon if needed)
    Run {
        /// Model to run (e.g., llama3.2:1b, phi3, hf:TheBloke/Llama-2-7B-GGUF)
        model: String,

        /// The prompt to send (optional - opens interactive mode if not provided)
        prompt: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "512")]
        max_tokens: u32,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Image file for vision models
        #[arg(short, long)]
        image: Option<PathBuf>,

        /// HTTP port for vision requests (uses HTTP API instead of IPC)
        #[arg(long, default_value = "8080")]
        http_port: u16,

        /// Number of GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Show generation stats
        #[arg(long)]
        stats: bool,

        /// Enable flash attention
        #[arg(long)]
        flash_attn: bool,

        /// KV cache type for keys
        #[arg(long)]
        cache_type_k: Option<String>,

        /// KV cache type for values
        #[arg(long)]
        cache_type_v: Option<String>,

        /// Disable memory-mapped model loading
        #[arg(long)]
        no_mmap: bool,

        /// Lock model weights in physical memory
        #[arg(long)]
        mlock: bool,

        /// Batch size for prompt processing
        #[arg(long)]
        batch_size: Option<u32>,
    },

    /// List loaded models
    Models {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Load a model into the daemon
    Load {
        /// Model specification (format: alias:path or just path)
        spec: String,

        /// Number of GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Path to multimodal projector for vision models (mmproj.gguf)
        #[arg(long)]
        mmproj: Option<PathBuf>,

        /// Enable flash attention
        #[arg(long)]
        flash_attn: bool,

        /// KV cache type for keys
        #[arg(long)]
        cache_type_k: Option<String>,

        /// KV cache type for values
        #[arg(long)]
        cache_type_v: Option<String>,

        /// Disable memory-mapped model loading
        #[arg(long)]
        no_mmap: bool,

        /// Lock model weights in physical memory
        #[arg(long)]
        mlock: bool,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Unload a model from the daemon
    Unload {
        /// Model alias to unload
        alias: String,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Set the default model
    Default {
        /// Model alias to set as default
        alias: String,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Show daemon status
    Status {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Ping the daemon
    Ping {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Shutdown the daemon
    Stop {
        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Force shutdown even with active requests
        #[arg(short, long)]
        force: bool,
    },

    /// Tokenize text using a model
    Tokenize {
        /// Text to tokenize
        text: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Generate embeddings for text
    Embed {
        /// Text(s) to embed
        text: Vec<String>,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// IPC socket to connect to
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Download a model from HuggingFace
    #[command(alias = "download")]
    Pull {
        /// Model specification (e.g., hf:TheBloke/Llama-2-7B-GGUF:model.Q4_K_M.gguf)
        spec: String,

        /// Quiet mode (no progress bar)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Manage the model cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Search for models on HuggingFace
    #[command(alias = "find")]
    Search {
        /// Search query (e.g., "llama 7b", "mistral gguf", "phi")
        query: String,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,

        /// Show all models (not just GGUF)
        #[arg(long)]
        all: bool,

        /// Show available GGUF files for each result
        #[arg(short, long)]
        files: bool,
    },

    /// Show details about a HuggingFace repository
    Info {
        /// Repository ID (e.g., TheBloke/Llama-2-7B-GGUF)
        repo: String,
    },

    /// Create a model from a Modelfile
    Create {
        /// Name for the new model
        name: String,

        /// Path to Modelfile (default: ./Modelfile or ./Mullamafile)
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Download base model if not cached
        #[arg(long, default_value = "true")]
        download: bool,

        /// Quiet mode (no progress bar)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Copy/rename a model
    #[command(alias = "copy")]
    Cp {
        /// Source model name
        source: String,

        /// Destination model name
        destination: String,
    },

    /// Manage the daemon process
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the daemon in background
    Start {
        /// HTTP port for OpenAI-compatible API
        #[arg(short = 'p', long, default_value = "8080")]
        http_port: u16,

        /// HTTP bind address
        #[arg(long, default_value = "127.0.0.1")]
        http_addr: String,

        /// API key for HTTP endpoints (Authorization: Bearer <key>)
        #[arg(long)]
        api_key: Option<String>,

        /// Always require API key auth, even when bound to localhost.
        #[arg(long)]
        require_api_key: bool,

        /// Default GPU layers to offload
        #[arg(short, long, default_value = "0")]
        gpu_layers: i32,

        /// Default context size
        #[arg(short, long, default_value = "4096")]
        context_size: u32,

        /// Number of contexts in each loaded model pool
        #[arg(long, default_value_t = mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE)]
        context_pool_size: usize,

        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Stop the daemon
    Stop {
        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Force stop (SIGKILL)
        #[arg(short, long)]
        force: bool,
    },

    /// Restart the daemon
    Restart {
        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },

    /// Show daemon status
    Status {
        /// IPC socket address
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output
        #[arg(short, long)]
        follow: bool,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// List cached models
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show cache directory path
    Path,

    /// Show cache size
    Size,

    /// Remove a cached model
    Remove {
        /// Repository ID (e.g., TheBloke/Llama-2-7B-GGUF)
        repo_id: String,

        /// Filename to remove (if not specified, removes all files from repo)
        #[arg(short, long)]
        filename: Option<String>,
    },

    /// Clear all cached models
    Clear {
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing/logging with env-filter support
    // Use MULLAMA_LOG or RUST_LOG env vars to control log levels
    // e.g., MULLAMA_LOG=info or RUST_LOG=mullama=debug,tower_http=info
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MULLAMA_LOG")
                .or_else(|_| tracing_subscriber::EnvFilter::try_from_env("RUST_LOG"))
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("mullama=info,tower_http=info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::List { verbose, json } => {
            list_all_models(verbose, json).await?;
        }

        Commands::Rm { name, force } => {
            remove_model(&name, force).await?;
        }

        Commands::Ps { socket, json } => {
            show_running_models(&socket, json).await?;
        }

        Commands::Show {
            name,
            modelfile,
            json,
        } => {
            show_model_details(&name, modelfile, json).await?;
        }

        Commands::Serve {
            model,
            mmproj,
            socket,
            http_port,
            http_addr,
            api_key,
            require_api_key,
            max_tokens_limit,
            max_request_body_mb,
            max_concurrent_requests,
            max_requests_per_second,
            gpu_layers,
            context_size,
            context_pool_size,
            threads,
            verbose,
            tls_cert,
            tls_key,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            batch_size,
            rope_freq_base,
            rope_freq_scale,
            split_mode,
            defrag_thold,
        } => {
            if tls_cert.is_some() != tls_key.is_some() {
                eprintln!("Error: --tls-cert and --tls-key must both be specified for HTTPS");
                std::process::exit(1);
            }
            run_server(
                model,
                mmproj,
                socket,
                http_port,
                http_addr,
                api_key,
                require_api_key,
                max_tokens_limit,
                max_request_body_mb,
                max_concurrent_requests,
                max_requests_per_second,
                gpu_layers,
                context_size,
                context_pool_size,
                threads,
                verbose,
                flash_attn,
                cache_type_k,
                cache_type_v,
                no_mmap,
                mlock,
                batch_size,
                rope_freq_base,
                rope_freq_scale,
                split_mode,
                defrag_thold,
            )
            .await?;
        }

        Commands::Chat { socket, timeout } => {
            run_chat(&socket, timeout)?;
        }

        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            socket,
            image,
            http_port,
            gpu_layers,
            context_size,
            stats,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            batch_size,
        } => {
            run_model_with_prompt(
                &model,
                prompt.as_deref(),
                max_tokens,
                temperature,
                &socket,
                image.as_ref(),
                http_port,
                gpu_layers,
                context_size,
                stats,
                flash_attn,
                cache_type_k,
                cache_type_v,
                no_mmap,
                mlock,
                batch_size,
            )
            .await?;
        }

        Commands::Models { socket, verbose } => {
            list_models(&socket, verbose)?;
        }

        Commands::Load {
            spec,
            gpu_layers,
            context_size,
            mmproj,
            flash_attn,
            cache_type_k,
            cache_type_v,
            no_mmap,
            mlock,
            socket,
        } => {
            load_model(
                &socket, &spec, gpu_layers, context_size, mmproj,
                flash_attn, cache_type_k, cache_type_v, no_mmap, mlock,
            )?;
        }

        Commands::Unload { alias, socket } => {
            unload_model(&socket, &alias)?;
        }

        Commands::Default { alias, socket } => {
            set_default(&socket, &alias)?;
        }

        Commands::Status { socket, json } => {
            show_status(&socket, json)?;
        }

        Commands::Ping { socket } => {
            ping_daemon(&socket)?;
        }

        Commands::Stop { socket, force: _ } => {
            cli_stop_daemon(&socket)?;
        }

        Commands::Tokenize {
            text,
            model,
            socket,
        } => {
            tokenize_text(&socket, &text, model.as_deref())?;
        }

        Commands::Embed {
            text,
            model,
            socket,
            json,
        } => {
            embed_text(&socket, &text, model.as_deref(), json)?;
        }

        Commands::Pull { spec, quiet } => {
            pull_model(&spec, !quiet).await?;
        }

        Commands::Cache { action } => {
            handle_cache_action(action).await?;
        }

        Commands::Search {
            query,
            limit,
            all,
            files,
        } => {
            search_models(&query, limit, !all, files).await?;
        }

        Commands::Info { repo } => {
            show_repo_info(&repo).await?;
        }

        Commands::Create {
            name,
            file,
            download,
            quiet,
        } => {
            create_model(&name, file, download, !quiet).await?;
        }

        Commands::Cp {
            source,
            destination,
        } => {
            copy_model(&source, &destination).await?;
        }

        Commands::Daemon { action } => {
            handle_daemon_action(action)?;
        }
    }

    Ok(())
}
