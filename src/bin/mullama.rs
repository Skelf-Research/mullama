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

// Use mimalloc for better allocation performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use mullama::daemon::spawn::default_log_path;
use mullama::daemon::{
    create_openai_router, daemon_status, ensure_daemon_running, is_daemon_running, registry,
    resolve_model_name, resolve_model_path, spawn_daemon, stop_daemon, Daemon, DaemonBuilder,
    DaemonClient, GgufFileInfo, HfDownloader, HfModelSpec, HfSearchResult, ModelConfig,
    ResolvedModel, SpawnConfig, SpawnResult, TuiApp, DEFAULT_HTTP_PORT, DEFAULT_SOCKET,
};
use mullama::modelfile::{find_modelfile, Modelfile, ModelfileParser};
use rand::{distributions::Alphanumeric, Rng};

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
    let cli = Cli::parse();

    match cli.command {
        Commands::List { verbose, json } => {
            list_all_models(verbose, json).await?;
        }

        Commands::Rm { name, force } => {
            remove_model(&name, force).await?;
        }

        Commands::Ps { socket, json } => {
            show_running_models(&socket, json)?;
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
        } => {
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
            socket,
        } => {
            load_model(&socket, &spec, gpu_layers, context_size, mmproj)?;
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

// ==================== Server ====================

fn is_loopback_http_addr(addr: &str) -> bool {
    matches!(addr, "127.0.0.1" | "localhost" | "::1")
}

fn generate_api_key() -> String {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(40)
        .map(char::from)
        .collect();
    format!("mullama_{}", suffix)
}

async fn run_server(
    models: Vec<String>,
    mmproj: Option<PathBuf>,
    socket: String,
    http_port: u16,
    http_addr: String,
    api_key: Option<String>,
    require_api_key: bool,
    max_tokens_limit: u32,
    max_request_body_mb: u32,
    max_concurrent_requests: usize,
    max_requests_per_second: u64,
    gpu_layers: i32,
    context_size: u32,
    context_pool_size: usize,
    threads: Option<i32>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut resolved_api_key = api_key.or_else(|| std::env::var("MULLAMA_API_KEY").ok());
    let is_loopback_bind = is_loopback_http_addr(&http_addr);
    let enforce_api_key =
        http_port > 0 && (require_api_key || !is_loopback_bind || resolved_api_key.is_some());
    let mut generated_api_key = false;

    if enforce_api_key && resolved_api_key.is_none() {
        resolved_api_key = Some(generate_api_key());
        generated_api_key = true;
    }

    // Initialize backend
    mullama::backend_init();

    println!("Starting Mullama Daemon...");
    println!("  IPC Socket: {}", socket);
    if http_port > 0 {
        println!("  HTTP API:   http://{}:{}", http_addr, http_port);
        if enforce_api_key {
            if generated_api_key {
                println!("  HTTP Auth:  enabled (generated API key)");
            } else {
                println!("  HTTP Auth:  enabled");
            }
            if let Some(ref key) = resolved_api_key {
                println!("  API Key:    {}", key);
            }
        } else {
            println!("  HTTP Auth:  disabled (localhost compatibility mode)");
        }
    }
    println!("  GPU Layers: {}", gpu_layers);
    println!("  Context:    {}", context_size);
    println!("  Ctx Pool:   {}", context_pool_size);
    println!("  Max Tokens: {}", max_tokens_limit);
    println!("  Body Limit: {} MB", max_request_body_mb);
    if let Some(ref mmp) = mmproj {
        println!("  MMProj:     {}", mmp.display());
    }
    println!();

    // Resolve model paths (download HF/Ollama models if needed)
    let mut resolved_models: Vec<(String, PathBuf, Option<ModelConfig>)> = Vec::new();
    for spec in &models {
        use mullama::daemon::registry::{resolve_model_name, ResolvedModel};

        match resolve_model_name(spec) {
            ResolvedModel::LocalPath(path) => {
                let alias = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                resolved_models.push((alias, path, None));
            }
            ResolvedModel::HuggingFace { spec: hf_spec, .. } => {
                println!("Resolving HuggingFace model: {}", hf_spec);
                match resolve_model_path(&hf_spec, true).await {
                    Ok((alias, path)) => {
                        println!("  -> {} at {}", alias, path.display());
                        resolved_models.push((alias, path, None));
                    }
                    Err(e) => {
                        eprintln!("Failed to resolve {}: {}", spec, e);
                        continue;
                    }
                }
            }
            ResolvedModel::Ollama { name, tag } => {
                use mullama::daemon::OllamaClient;
                let model_name = format!("{}:{}", name, tag);
                // Use name-tag format for alias to avoid colon conflicts with alias:path format
                let alias = format!("{}-{}", name, tag);
                println!("Resolving Ollama model: {}", model_name);
                let client = match OllamaClient::new() {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Failed to initialize Ollama client: {}", e);
                        continue;
                    }
                };
                // Check if cached, if not pull it
                if let Some(model) = client.get_cached(&model_name) {
                    println!("  -> {} at {}", alias, model.gguf_path.display());
                    let config = ModelConfig {
                        stop_sequences: model.get_stop_sequences(),
                        system_prompt: model.system_prompt.clone(),
                        temperature: model.parameters.temperature,
                        top_p: model.parameters.top_p,
                        top_k: model.parameters.top_k,
                        context_size: model.parameters.num_ctx,
                    };
                    resolved_models.push((alias.clone(), model.gguf_path.clone(), Some(config)));
                } else {
                    println!("  Pulling from Ollama registry...");
                    match client.pull(&model_name, true).await {
                        Ok(model) => {
                            println!("  -> {} at {}", alias, model.gguf_path.display());
                            let config = ModelConfig {
                                stop_sequences: model.get_stop_sequences(),
                                system_prompt: model.system_prompt.clone(),
                                temperature: model.parameters.temperature,
                                top_p: model.parameters.top_p,
                                top_k: model.parameters.top_k,
                                context_size: model.parameters.num_ctx,
                            };
                            resolved_models.push((
                                alias.clone(),
                                model.gguf_path.clone(),
                                Some(config),
                            ));
                        }
                        Err(e) => {
                            eprintln!("Failed to pull {}: {}", model_name, e);
                            continue;
                        }
                    }
                }
            }
            ResolvedModel::Unknown(name) => {
                // Try as a local path
                let path = PathBuf::from(&name);
                if path.exists() {
                    let alias = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "model".to_string());
                    resolved_models.push((alias, path, None));
                } else {
                    eprintln!(
                        "Unknown model: {} (not found locally or in registries)",
                        name
                    );
                    continue;
                }
            }
        }
    }
    println!();

    // Build daemon configuration
    let mut builder = DaemonBuilder::new()
        .ipc_socket(&socket)
        .default_gpu_layers(gpu_layers)
        .default_context_size(context_size)
        .default_context_pool_size(context_pool_size)
        .http_api_key(resolved_api_key)
        .enforce_http_api_key(enforce_api_key)
        .max_tokens_per_request(max_tokens_limit)
        .max_request_body_bytes((max_request_body_mb as usize) * 1024 * 1024)
        .max_concurrent_http_requests(max_concurrent_requests)
        .max_requests_per_second(max_requests_per_second);

    if http_port > 0 {
        builder = builder.http_port(http_port).http_addr(&http_addr);
    } else {
        builder = builder.disable_http();
    }

    if let Some(t) = threads {
        builder = builder.threads_per_model(t);
    }

    // Add resolved models
    for (alias, path, _) in &resolved_models {
        builder = builder.model(format!("{}:{}", alias, path.display()));
    }

    let (daemon, mut initial_models) = builder.build();
    let daemon = std::sync::Arc::new(daemon);

    let model_configs_by_alias: std::collections::HashMap<String, ModelConfig> = resolved_models
        .iter()
        .filter_map(|(alias, _path, config)| config.clone().map(|c| (alias.clone(), c)))
        .collect();
    for config in &mut initial_models {
        if let Some(model_config) = model_configs_by_alias.get(&config.alias) {
            config.model_config = Some(model_config.clone());
        }
    }

    // Apply mmproj to the first model if specified
    if let Some(ref mmp) = mmproj {
        if let Some(first) = initial_models.first_mut() {
            first.mmproj_path = Some(mmp.display().to_string());
        }
    }

    // Load initial models
    for config in initial_models {
        print!("Loading model '{}'... ", config.alias);
        io::stdout().flush()?;

        match daemon.models.load(config.clone()).await {
            Ok(info) => {
                println!("OK");
                if verbose {
                    println!("    Path: {}", info.path);
                    println!("    Parameters: {}M", info.parameters / 1_000_000);
                    println!("    Context: {}", info.context_size);
                }
            }
            Err(e) => {
                println!("FAILED");
                eprintln!("    Error: {}", e);
            }
        }
    }

    if resolved_models.is_empty() {
        println!("No models specified. Use --model to load models.");
        println!("You can also load models via the API or TUI.");
        println!();
        println!("Examples:");
        println!("  mullama serve --model ./model.gguf");
        println!("  mullama serve --model hf:TheBloke/Llama-2-7B-GGUF");
        println!(
            "  mullama serve --model llama:hf:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q4_K_M.gguf"
        );
    }

    println!();
    println!("Daemon ready. Press Ctrl+C to stop.");
    println!();

    // Start IPC server
    let ipc_daemon = daemon.clone();
    let ipc_socket = socket.clone();
    let ipc_handle = tokio::spawn(async move {
        if let Err(e) = run_ipc_server(ipc_daemon, &ipc_socket).await {
            eprintln!("IPC server error: {}", e);
        }
    });

    // Start HTTP server if enabled
    let http_handle = if http_port > 0 {
        let http_daemon = daemon.clone();
        let addr = format!("{}:{}", http_addr, http_port);
        Some(tokio::spawn(async move {
            let router = create_openai_router(http_daemon);
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => listener,
                Err(e) => {
                    eprintln!("Failed to bind HTTP listener at {}: {}", addr, e);
                    return;
                }
            };
            if let Err(e) = axum::serve(listener, router).await {
                eprintln!("HTTP server error: {}", e);
            }
        }))
    } else {
        None
    };

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    println!("\nShutting down...");

    // Signal shutdown
    daemon
        .shutdown
        .store(true, std::sync::atomic::Ordering::SeqCst);

    // Give servers time to cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cleanup
    mullama::backend_free();

    Ok(())
}

async fn run_ipc_server(
    daemon: std::sync::Arc<Daemon>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use mullama::daemon::{Request, Response};
    use nng::options::{Options, RecvTimeout};
    use nng::{Protocol, Socket};

    let socket = Socket::new(Protocol::Rep0)?;
    socket.listen(addr)?;
    socket.set_opt::<RecvTimeout>(Some(Duration::from_millis(250)))?;

    loop {
        if daemon.is_shutdown() {
            break;
        }

        // Non-blocking receive with timeout
        match socket.recv() {
            Ok(msg) => {
                let request = match Request::from_bytes(&msg) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Invalid request: {}", e);
                        continue;
                    }
                };

                let response = daemon.handle_request(request).await;

                let resp_bytes = match response.to_bytes() {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Serialization error: {}", e);
                        continue;
                    }
                };

                if let Err(e) = socket.send(nng::Message::from(resp_bytes.as_slice())) {
                    eprintln!("Send error: {:?}", e);
                }
            }
            Err(nng::Error::TimedOut) => {
                continue;
            }
            Err(e) => {
                if !daemon.is_shutdown() {
                    eprintln!("Receive error: {}", e);
                }
                break;
            }
        }
    }

    Ok(())
}

// ==================== Client Commands ====================

fn connect(socket: &str) -> Result<DaemonClient, Box<dyn std::error::Error>> {
    // Try to connect first
    match DaemonClient::connect_with_timeout(socket, Duration::from_millis(500)) {
        Ok(client) => return Ok(client),
        Err(_) => {
            // Daemon not running, try to auto-spawn it
            eprintln!("Daemon not running, starting it automatically...");

            let config = SpawnConfig {
                socket: socket.to_string(),
                log_file: Some(default_log_path()),
                ..Default::default()
            };

            if let Err(e) = ensure_daemon_running(&config) {
                return Err(format!(
                    "Failed to start daemon automatically: {}\n\
                    You can start the daemon manually with: mullama serve",
                    e
                )
                .into());
            }

            eprintln!("Daemon started successfully, connecting...");

            // Now try to connect again
            DaemonClient::connect_with_timeout(socket, Duration::from_secs(5))
                .map_err(|e| format!("Failed to connect to daemon after starting: {}", e).into())
        }
    }
}

fn run_chat(socket: &str, _timeout: u64) -> Result<(), Box<dyn std::error::Error>> {
    // Use auto-spawning connect
    let client = connect(socket)?;

    // Verify connection
    match client.ping() {
        Ok((uptime, version)) => {
            println!(
                "Connected to Mullama daemon v{} (uptime: {}s)",
                version, uptime
            );
        }
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            return Err(e.into());
        }
    }

    // Start TUI
    let mut app = TuiApp::new(client);
    app.run()?;

    Ok(())
}

/// Run a model with a prompt (Ollama-style: auto-starts daemon and loads model)
async fn run_model_with_prompt(
    model_spec: &str,
    prompt: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    socket: &str,
    image: Option<&PathBuf>,
    http_port: u16,
    gpu_layers: i32,
    context_size: u32,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::registry::{resolve_model_name, ResolvedModel};
    use mullama::daemon::OllamaClient;

    // Step 1: Resolve the model to get the path and alias
    let (model_alias, model_path) = match resolve_model_name(model_spec) {
        ResolvedModel::Ollama { name, tag } => {
            let model_name = format!("{}:{}", name, tag);
            let alias = format!("{}-{}", name, tag); // Use dash for daemon alias

            // Check if cached, if not pull it
            let client = OllamaClient::new()?;
            let model = if let Some(m) = client.get_cached(&model_name) {
                m
            } else {
                eprintln!("Pulling {}...", model_name);
                client.pull(&model_name, true).await?
            };
            (alias, model.gguf_path)
        }
        ResolvedModel::HuggingFace { spec, mmproj: _ } => {
            eprintln!("Resolving HuggingFace model: {}", spec);
            let (alias, path) = resolve_model_path(&spec, true).await?;
            (alias, path)
        }
        ResolvedModel::LocalPath(path) => {
            let alias = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string());
            (alias, path)
        }
        ResolvedModel::Unknown(name) => {
            // Try as local path
            let path = PathBuf::from(&name);
            if path.exists() {
                let alias = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                (alias, path)
            } else {
                return Err(format!(
                    "Unknown model: {} (not found locally or in registries)",
                    name
                )
                .into());
            }
        }
    };

    // Step 2: Check if daemon is running, auto-start if needed
    let client = match connect(socket) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Starting daemon...");

            // Start daemon in background
            let config = SpawnConfig {
                binary_path: None,
                socket: socket.to_string(),
                http_port,
                gpu_layers,
                context_size,
                startup_timeout: std::time::Duration::from_secs(60),
                background: true,
                log_file: Some(default_log_path()),
                ..Default::default()
            };

            match spawn_daemon(&config) {
                SpawnResult::AlreadyRunning => {}
                SpawnResult::Spawned { .. } => {}
                SpawnResult::Failed(e) => {
                    return Err(format!("Failed to start daemon: {}", e).into());
                }
            }

            // Wait for daemon to be ready
            let mut attempts = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if let Ok(c) = connect(socket) {
                    break c;
                }
                attempts += 1;
                if attempts > 150 {
                    // 30 seconds
                    return Err("Timed out waiting for daemon to start".into());
                }
            }
        }
    };

    // Step 3: Check if model is loaded, load if needed
    let loaded_models = client.list_models()?;
    let model_loaded = loaded_models.iter().any(|m| m.alias == model_alias);

    if !model_loaded {
        eprintln!("Loading {}...", model_alias);

        match client.load_model_with_options(
            &model_alias,
            &model_path.display().to_string(),
            gpu_layers,
            context_size,
        ) {
            Ok(_) => {}
            Err(e) => {
                return Err(format!("Failed to load model: {}", e).into());
            }
        }
    }

    // Step 4: Run the prompt or enter interactive mode
    if let Some(prompt_text) = prompt {
        if image.is_some() {
            run_vision_prompt(
                http_port,
                prompt_text,
                Some(&model_alias),
                max_tokens,
                temperature,
                image.unwrap(),
                stats,
            )
            .await?;
        } else {
            let result = client.chat(prompt_text, Some(&model_alias), max_tokens, temperature)?;
            println!("{}", result.text);

            if stats {
                eprintln!();
                eprintln!(
                    "--- {} tokens in {}ms ({:.1} tok/s) using {} ---",
                    result.completion_tokens,
                    result.duration_ms,
                    result.tokens_per_second(),
                    result.model
                );
            }
        }
    } else {
        // Interactive mode - simple REPL
        eprintln!(">>> Send a message (/? for help)");

        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();

        loop {
            print!(">>> ");
            stdout.flush()?;

            let mut input = String::new();
            stdin.read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            match input {
                "/bye" | "/exit" | "/quit" => break,
                "/?" | "/help" => {
                    eprintln!("Available commands:");
                    eprintln!("  /bye, /exit, /quit  - Exit interactive mode");
                    eprintln!("  /clear              - Clear conversation");
                    eprintln!("  /?                  - Show this help");
                    continue;
                }
                "/clear" => {
                    eprintln!("(Conversation cleared)");
                    continue;
                }
                _ => {}
            }

            match client.chat(input, Some(&model_alias), max_tokens, temperature) {
                Ok(result) => {
                    println!("{}", result.text);
                    if stats {
                        eprintln!(
                            "--- {} tokens in {}ms ({:.1} tok/s) ---",
                            result.completion_tokens,
                            result.duration_ms,
                            result.tokens_per_second()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
            println!();
        }
    }

    Ok(())
}

fn run_prompt(
    socket: &str,
    prompt: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let result = client.chat(prompt, model, max_tokens, temperature)?;

    println!("{}", result.text);

    if stats {
        eprintln!();
        eprintln!(
            "--- {} tokens in {}ms ({:.1} tok/s) using {} ---",
            result.completion_tokens,
            result.duration_ms,
            result.tokens_per_second(),
            result.model
        );
    }

    Ok(())
}

/// Run vision prompt using HTTP API
async fn run_vision_prompt(
    http_port: u16,
    prompt: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    image_path: &PathBuf,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine;

    let start = std::time::Instant::now();

    // Read and encode image
    let image_data = std::fs::read(image_path).map_err(|e| {
        format!(
            "Failed to read image file '{}': {}",
            image_path.display(),
            e
        )
    })?;

    // Detect image type from extension
    let mime_type = match image_path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        _ => "image/jpeg", // Default to JPEG
    };

    let base64_image = base64::engine::general_purpose::STANDARD.encode(&image_data);
    let image_url = format!("data:{};base64,{}", mime_type, base64_image);

    // Build OpenAI vision API request
    let request_body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": false
    });

    // Send request to HTTP API
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", http_port);

    let mut request = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body);

    if let Ok(api_key) = std::env::var("MULLAMA_API_KEY") {
        request = request.bearer_auth(api_key);
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("Failed to connect to daemon at {}: {}", url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("Vision request failed ({}): {}", status, error_text).into());
    }

    // Parse response
    let resp_json: serde_json::Value = response.json().await?;

    let text = resp_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("(no response)");

    println!("{}", text);

    if stats {
        let duration = start.elapsed();
        let completion_tokens = resp_json["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or(0);
        let model_used = resp_json["model"].as_str().unwrap_or("unknown");

        eprintln!();
        eprintln!(
            "--- {} tokens in {}ms ({:.1} tok/s) using {} ---",
            completion_tokens,
            duration.as_millis(),
            completion_tokens as f64 / duration.as_secs_f64(),
            model_used
        );
    }

    Ok(())
}

fn list_models(socket: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let models = client.list_models()?;

    if models.is_empty() {
        println!("No models loaded.");
        println!("Use 'mullama load <path>' to load a model.");
        return Ok(());
    }

    println!("Loaded models:\n");
    for model in models {
        let default_marker = if model.is_default { " (default)" } else { "" };
        println!("  {}{}", model.alias, default_marker);

        if verbose {
            println!("    Path:       {}", model.info.path);
            println!("    Parameters: {}M", model.info.parameters / 1_000_000);
            println!("    Context:    {}", model.info.context_size);
            println!("    GPU layers: {}", model.info.gpu_layers);
            if model.active_requests > 0 {
                println!("    Active:     {} requests", model.active_requests);
            }
            println!();
        }
    }

    Ok(())
}

fn load_model(
    socket: &str,
    spec: &str,
    gpu_layers: i32,
    context_size: u32,
    mmproj: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if mmproj.is_some() {
        eprintln!("Warning: --mmproj is not yet supported via IPC protocol.");
        eprintln!("         Vision models can be loaded directly via the server:");
        eprintln!("         mullama serve --model model.gguf (with mmproj support coming soon)");
        eprintln!();
    }

    let client = connect(socket)?;

    // Parse spec
    let (alias, path) = if let Some(pos) = spec.find(':') {
        (spec[..pos].to_string(), spec[pos + 1..].to_string())
    } else {
        let p = std::path::Path::new(spec);
        let alias = p
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        (alias, spec.to_string())
    };

    print!("Loading model '{}'... ", alias);
    io::stdout().flush()?;

    match client.load_model_with_options(&alias, &path, gpu_layers, context_size) {
        Ok((alias, info)) => {
            println!("OK");
            println!("  Parameters: {}M", info.parameters / 1_000_000);
            println!("  Context:    {}", info.context_size);
        }
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn unload_model(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    print!("Unloading model '{}'... ", alias);
    io::stdout().flush()?;

    match client.unload_model(alias) {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn set_default(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    match client.set_default_model(alias) {
        Ok(()) => println!("Default model set to '{}'", alias),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

fn show_status(socket: &str, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let status = client.status()?;

    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Mullama Daemon Status");
        println!("=====================");
        println!("Version:         {}", status.version);
        println!("Uptime:          {}s", status.uptime_secs);
        println!("Models loaded:   {}", status.models_loaded);
        if let Some(ref default) = status.default_model {
            println!("Default model:   {}", default);
        }
        if let Some(ref http) = status.http_endpoint {
            println!("HTTP endpoint:   {}", http);
        }
        println!("IPC endpoint:    {}", status.ipc_endpoint);
        println!();
        println!("Statistics:");
        println!("  Total requests:   {}", status.stats.requests_total);
        println!("  Tokens generated: {}", status.stats.tokens_generated);
        println!("  Active requests:  {}", status.stats.active_requests);
        println!("  GPU available:    {}", status.stats.gpu_available);
    }

    Ok(())
}

fn ping_daemon(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let client = connect(socket)?;
    let (uptime, version) = client.ping()?;
    let latency = start.elapsed();

    println!("Pong from mullama v{}", version);
    println!("  Daemon uptime: {}s", uptime);
    println!("  Round-trip:    {:?}", latency);

    Ok(())
}

fn cli_stop_daemon(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    print!("Shutting down daemon... ");
    io::stdout().flush()?;

    match client.shutdown() {
        Ok(()) => println!("OK"),
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

fn tokenize_text(
    socket: &str,
    text: &str,
    model: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let tokens = client.tokenize(text, model)?;

    println!("Tokens ({}): {:?}", tokens.len(), tokens);

    Ok(())
}

fn embed_text(
    socket: &str,
    texts: &[String],
    model: Option<&str>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    if texts.is_empty() {
        return Err("No text provided".into());
    }

    if texts.len() == 1 {
        // Single embedding
        let result = client.embed(&texts[0], model)?;

        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": result.model,
                    "dimension": result.dimension(),
                    "prompt_tokens": result.prompt_tokens,
                    "embedding": result.embedding,
                }))?
            );
        } else {
            println!("Model: {}", result.model);
            println!("Dimension: {}", result.dimension());
            println!("Tokens: {}", result.prompt_tokens);
            println!(
                "Embedding (first 10): {:?}...",
                &result.embedding[..result.embedding.len().min(10)]
            );
        }
    } else {
        // Batch embedding
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = client.embed_batch(&text_refs, model)?;

        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": result.model,
                    "count": result.count(),
                    "dimension": result.dimension(),
                    "prompt_tokens": result.prompt_tokens,
                    "embeddings": result.embeddings,
                }))?
            );
        } else {
            println!("Model: {}", result.model);
            println!("Count: {}", result.count());
            println!("Dimension: {}", result.dimension());
            println!("Tokens: {}", result.prompt_tokens);
            for (i, emb) in result.embeddings.iter().enumerate() {
                println!("  [{}]: {:?}...", i, &emb[..emb.len().min(5)]);
            }
        }
    }

    Ok(())
}

// ==================== HuggingFace / Cache Commands ====================

async fn pull_model(spec: &str, show_progress: bool) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::registry::{resolve_model_name, ResolvedModel};
    use mullama::daemon::OllamaClient;

    // First try to resolve the model name
    let resolved = resolve_model_name(spec);

    match resolved {
        ResolvedModel::Ollama { name, tag } => {
            // Pull from Ollama registry
            let client = OllamaClient::new()?;
            let model_name = format!("{}:{}", name, tag);

            let model = client.pull(&model_name, show_progress).await?;

            println!();
            println!("Model pulled successfully!");
            println!("  Name: {}:{}", model.name, model.tag);
            println!("  Path: {}", model.gguf_path.display());
            println!("  Size: {}", format_size(model.total_size));

            if model.template.is_some() {
                println!("  Template: included");
            }
            if model.system_prompt.is_some() {
                println!("  System prompt: included");
            }
            if model.projector_path.is_some() {
                println!("  Vision projector: included");
            }

            println!();
            println!("To use this model:");
            println!("  mullama run {}:{} \"Hello!\"", model.name, model.tag);
            println!("  mullama serve --model {}:{}", model.name, model.tag);

            Ok(())
        }

        ResolvedModel::HuggingFace { spec: hf_spec, .. } => {
            // Pull from HuggingFace
            pull_from_huggingface(&hf_spec, show_progress).await
        }

        ResolvedModel::LocalPath(path) => Err(format!(
            "'{}' is a local path, not a downloadable model",
            path.display()
        )
        .into()),

        ResolvedModel::Unknown(_) => {
            // Not a known alias, try as direct HF spec
            pull_from_huggingface(spec, show_progress).await
        }
    }
}

async fn pull_from_huggingface(
    spec: &str,
    show_progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let hf_spec = HfModelSpec::parse(spec).ok_or_else(|| {
        format!(
            "Unknown model '{}'\n\
             Use Ollama format (e.g., llama3:1b) or HF format: hf:owner/repo:filename.gguf",
            spec
        )
    })?;

    let downloader = HfDownloader::new()?;

    println!("Downloading from HuggingFace...");
    println!("  Repository: {}", hf_spec.repo_id);

    if let Some(ref filename) = hf_spec.filename {
        println!("  File: {}", filename);
    } else {
        println!("  File: (auto-detecting best GGUF)");
    }
    println!();

    let path = downloader.download_spec(&hf_spec, show_progress).await?;

    println!();
    println!("Model downloaded successfully!");
    println!("  Path: {}", path.display());
    println!();
    println!("To use this model:");
    println!(
        "  mullama serve --model {}:{}",
        hf_spec.get_alias(),
        path.display()
    );

    Ok(())
}

async fn handle_cache_action(action: CacheAction) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    match action {
        CacheAction::List { verbose } => {
            let models = downloader.list_cached();

            if models.is_empty() {
                println!("No cached models.");
                println!();
                println!("Download models with:");
                println!("  mullama pull hf:TheBloke/Llama-2-7B-GGUF");
                return Ok(());
            }

            println!("Cached models:\n");
            for model in models {
                println!("  {} / {}", model.repo_id, model.filename);
                if verbose {
                    println!("    Path: {}", model.local_path.display());
                    println!(
                        "    Size: {:.2} GB",
                        model.size_bytes as f64 / 1_073_741_824.0
                    );
                    println!("    Downloaded: {}", model.downloaded_at);
                    println!();
                }
            }

            if !verbose {
                println!();
                println!("Use --verbose for more details.");
            }
        }

        CacheAction::Path => {
            println!("{}", downloader.cache_dir().display());
            println!();
            println!("Override with MULLAMA_CACHE_DIR environment variable.");
        }

        CacheAction::Size => {
            let size = downloader.cache_size();
            let models = downloader.list_cached();

            println!("Cache size: {:.2} GB", size as f64 / 1_073_741_824.0);
            println!("Models cached: {}", models.len());
            println!("Cache directory: {}", downloader.cache_dir().display());
        }

        CacheAction::Remove { repo_id, filename } => {
            if let Some(filename) = filename {
                print!("Removing {} / {}... ", repo_id, filename);
                io::stdout().flush()?;
                downloader.remove_cached(&repo_id, &filename)?;
                println!("OK");
            } else {
                // Remove all files from repo
                let models = downloader.list_cached();
                let to_remove: Vec<_> = models.iter().filter(|m| m.repo_id == repo_id).collect();

                if to_remove.is_empty() {
                    println!("No cached files found for {}", repo_id);
                    return Ok(());
                }

                for model in to_remove {
                    print!("Removing {}... ", model.filename);
                    io::stdout().flush()?;
                    downloader.remove_cached(&model.repo_id, &model.filename)?;
                    println!("OK");
                }
            }
        }

        CacheAction::Clear { force } => {
            if !force {
                let models = downloader.list_cached();
                let size = downloader.cache_size();

                println!(
                    "This will remove {} models ({:.2} GB).",
                    models.len(),
                    size as f64 / 1_073_741_824.0
                );
                print!("Are you sure? [y/N] ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;

                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            print!("Clearing cache... ");
            io::stdout().flush()?;
            downloader.clear_cache()?;
            println!("OK");
        }
    }

    Ok(())
}

// ==================== Search Commands ====================

async fn search_models(
    query: &str,
    limit: usize,
    gguf_only: bool,
    show_files: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    println!("Searching HuggingFace for '{}'...\n", query);

    let results = downloader.search(query, gguf_only, limit).await?;

    if results.is_empty() {
        println!("No models found.");
        if gguf_only {
            println!("Try --all to search all models (not just GGUF).");
        }
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        // Header line
        print!("{}. ", i + 1);
        print!("{}", result.id);
        if result.is_gguf() {
            print!(" [GGUF]");
        }
        println!();

        // Metadata line
        print!("   ");
        print!("Downloads: {}", result.downloads_formatted());
        if let Some(likes) = result.likes {
            print!(" | Likes: {}", likes);
        }
        if let Some(ref pipeline) = result.pipeline_tag {
            print!(" | {}", pipeline);
        }
        println!();

        // Usage hint
        println!("   Use: mullama serve --model hf:{}", result.id);

        // Show files if requested
        if show_files && result.is_gguf() {
            match downloader.list_gguf_files(&result.id).await {
                Ok(files) => {
                    println!("   Files:");
                    for file in files.iter().take(5) {
                        print!("     - {}", file.filename);
                        print!(" ({})", file.size_formatted());
                        if let Some(ref q) = file.quantization {
                            print!(" [{}]", q);
                        }
                        println!();
                    }
                    if files.len() > 5 {
                        println!("     ... and {} more files", files.len() - 5);
                    }
                }
                Err(_) => {
                    println!("   (Could not fetch file list)");
                }
            }
        }

        println!();
    }

    println!("Found {} models.", results.len());
    if !show_files && gguf_only {
        println!("Use --files to show available GGUF files.");
    }

    Ok(())
}

async fn show_repo_info(repo_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    println!("Fetching info for {}...\n", repo_id);

    // Get GGUF files
    let files = downloader.list_gguf_files(repo_id).await?;

    println!("Repository: {}", repo_id);
    println!("URL: https://huggingface.co/{}", repo_id);
    println!();
    println!("Available GGUF files ({}):", files.len());
    println!();

    // Group by quantization type
    let mut by_quant: std::collections::HashMap<String, Vec<&GgufFileInfo>> =
        std::collections::HashMap::new();
    for file in &files {
        let key = file
            .quantization
            .clone()
            .unwrap_or_else(|| "Other".to_string());
        by_quant.entry(key).or_default().push(file);
    }

    // Sort quantization types by preference
    let quant_order = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0", "Q4_1", "Q8_0", "Q6_K", "Q3_K_M", "Q3_K_S",
        "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ2_M",
        "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S", "F16", "F32", "Other",
    ];

    for quant in quant_order {
        if let Some(files) = by_quant.get(quant) {
            for file in files {
                println!(
                    "  {:12} {:>10}  {}",
                    file.quantization.as_deref().unwrap_or("-"),
                    file.size_formatted(),
                    file.filename
                );
            }
        }
    }

    // Show any remaining that weren't in our order
    for (quant, files) in &by_quant {
        if !quant_order.contains(&quant.as_str()) {
            for file in files {
                println!(
                    "  {:12} {:>10}  {}",
                    file.quantization.as_deref().unwrap_or("-"),
                    file.size_formatted(),
                    file.filename
                );
            }
        }
    }

    println!();
    println!("Quick start:");
    println!("  mullama pull hf:{}", repo_id);
    println!("  mullama serve --model hf:{}", repo_id);

    // Check if any are cached
    let cached = downloader.list_cached();
    let cached_from_repo: Vec<_> = cached.iter().filter(|c| c.repo_id == repo_id).collect();
    if !cached_from_repo.is_empty() {
        println!();
        println!("Cached locally:");
        for c in cached_from_repo {
            println!(
                "  {} ({:.2} GB)",
                c.filename,
                c.size_bytes as f64 / 1_073_741_824.0
            );
        }
    }

    Ok(())
}

// ==================== New Ollama-like Commands ====================

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable time ago
fn format_time_ago(time: &chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(*time);

    if duration.num_days() > 30 {
        format!("{} months ago", duration.num_days() / 30)
    } else if duration.num_days() > 0 {
        format!("{} days ago", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{} hours ago", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{} minutes ago", duration.num_minutes())
    } else {
        "just now".to_string()
    }
}

/// List all local models (cached HuggingFace + Ollama + custom models)
async fn list_all_models(
    verbose: bool,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::OllamaClient;

    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    // Load Ollama models
    let ollama_client = OllamaClient::new().ok();
    let ollama_models: Vec<_> = ollama_client
        .as_ref()
        .map(|c| c.list_cached())
        .unwrap_or_default();

    // Also check ~/.mullama/models/ for custom models
    let mullama_dir = dirs::home_dir()
        .map(|h| h.join(".mullama").join("models"))
        .unwrap_or_else(|| PathBuf::from(".mullama/models"));

    let mut custom_models: Vec<(String, PathBuf, u64, chrono::DateTime<chrono::Utc>)> = Vec::new();

    if mullama_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&mullama_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        let size = metadata.len();
                        let modified = metadata
                            .modified()
                            .map(|t| chrono::DateTime::<chrono::Utc>::from(t))
                            .unwrap_or_else(|_| chrono::Utc::now());
                        custom_models.push((name, path, size, modified));
                    }
                }
            }
        }
    }

    if json_output {
        let mut models_json = Vec::new();

        // Ollama models first
        for model in &ollama_models {
            let model_name = format!("{}:{}", model.name, model.tag);
            models_json.push(serde_json::json!({
                "name": model_name,
                "source": "ollama",
                "size": model.total_size,
                "size_formatted": format_size(model.total_size),
                "modified": model.pulled_at,
                "path": model.gguf_path,
                "template": model.template.is_some(),
                "system_prompt": model.system_prompt.is_some(),
            }));
        }

        for model in &cached {
            models_json.push(serde_json::json!({
                "name": format!("{}:{}", model.repo_id.replace('/', "-"),
                    model.filename.trim_end_matches(".gguf")),
                "source": "huggingface",
                "repo_id": model.repo_id,
                "filename": model.filename,
                "size": model.size_bytes,
                "size_formatted": format_size(model.size_bytes),
                "modified": model.downloaded_at,
                "path": model.local_path,
            }));
        }

        for (name, path, size, modified) in &custom_models {
            models_json.push(serde_json::json!({
                "name": name,
                "source": "local",
                "size": size,
                "size_formatted": format_size(*size),
                "modified": modified.to_rfc3339(),
                "path": path,
            }));
        }

        println!("{}", serde_json::to_string_pretty(&models_json)?);
        return Ok(());
    }

    if cached.is_empty() && custom_models.is_empty() && ollama_models.is_empty() {
        println!("No models found.");
        println!();
        println!("Download models with:");
        println!("  mullama pull llama3.2:1b");
        println!("  mullama pull hf:TheBloke/Llama-2-7B-GGUF");
        return Ok(());
    }

    println!("NAME                                      SIZE       MODIFIED");

    // Print Ollama models first
    for model in &ollama_models {
        let name = format!("{}:{}", model.name, model.tag);
        let name_display = if name.len() > 40 {
            format!("{}...", &name[..37])
        } else {
            name.clone()
        };

        let modified = chrono::DateTime::parse_from_rfc3339(&model.pulled_at)
            .map(|dt| format_time_ago(&dt.with_timezone(&chrono::Utc)))
            .unwrap_or_else(|_| model.pulled_at.clone());

        println!(
            "{:<42} {:>10} {}",
            name_display,
            format_size(model.total_size),
            modified
        );

        if verbose {
            println!("    Source:   Ollama Registry");
            println!("    Path:     {}", model.gguf_path.display());
            if model.template.is_some() {
                println!("    Template: Yes");
            }
            if model.system_prompt.is_some() {
                println!("    System:   Yes");
            }
            println!();
        }
    }

    // Print cached HF models
    for model in &cached {
        let name = format!(
            "{}:{}",
            model.repo_id.split('/').last().unwrap_or(&model.repo_id),
            model.filename.trim_end_matches(".gguf")
        );
        let name_display = if name.len() > 40 {
            format!("{}...", &name[..37])
        } else {
            name.clone()
        };

        let modified = chrono::DateTime::parse_from_rfc3339(&model.downloaded_at)
            .map(|dt| format_time_ago(&dt.with_timezone(&chrono::Utc)))
            .unwrap_or_else(|_| model.downloaded_at.clone());

        println!(
            "{:<42} {:>10} {}",
            name_display,
            format_size(model.size_bytes),
            modified
        );

        if verbose {
            println!("    Source: HuggingFace");
            println!("    Path:   {}", model.local_path.display());
            println!("    Repo:   {}", model.repo_id);
            println!();
        }
    }

    // Print custom models
    for (name, path, size, modified) in &custom_models {
        let name_display = if name.len() > 40 {
            format!("{}...", &name[..37])
        } else {
            name.clone()
        };

        println!(
            "{:<42} {:>10} {}",
            name_display,
            format_size(*size),
            format_time_ago(modified)
        );

        if verbose {
            println!("    Source: Local");
            println!("    Path:   {}", path.display());
            println!();
        }
    }

    let total_count = cached.len() + custom_models.len() + ollama_models.len();
    let total_size: u64 = cached.iter().map(|m| m.size_bytes).sum::<u64>()
        + custom_models.iter().map(|(_, _, s, _)| s).sum::<u64>()
        + ollama_models.iter().map(|m| m.total_size).sum::<u64>();

    println!();
    println!(
        "{} model(s), {} total",
        total_count,
        format_size(total_size)
    );

    Ok(())
}

/// Remove a model from disk
async fn remove_model(name: &str, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    // Try to find the model by name
    let mut found = None;

    // Check if it's a direct path
    if PathBuf::from(name).exists() {
        found = Some(("path".to_string(), PathBuf::from(name)));
    } else {
        // Check cached models
        for model in &cached {
            let short_name = format!(
                "{}:{}",
                model.repo_id.split('/').last().unwrap_or(&model.repo_id),
                model.filename.trim_end_matches(".gguf")
            );

            if model.filename == name
                || model.repo_id == name
                || short_name == name
                || model.filename.trim_end_matches(".gguf") == name
            {
                found = Some((model.repo_id.clone(), model.local_path.clone()));
                break;
            }
        }

        // Check ~/.mullama/models/
        if found.is_none() {
            let mullama_dir = dirs::home_dir()
                .map(|h| h.join(".mullama").join("models"))
                .unwrap_or_else(|| PathBuf::from(".mullama/models"));

            let model_path = mullama_dir.join(format!("{}.gguf", name));
            if model_path.exists() {
                found = Some(("local".to_string(), model_path));
            }
        }
    }

    let (source, path) = match found {
        Some(f) => f,
        None => {
            eprintln!("Model '{}' not found.", name);
            eprintln!();
            eprintln!("Use 'mullama list' to see available models.");
            return Err("Model not found".into());
        }
    };

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    if !force {
        println!("Will remove: {}", path.display());
        println!("Size: {}", format_size(size));
        print!("Are you sure? [y/N] ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    print!("Removing {}... ", name);
    io::stdout().flush()?;

    std::fs::remove_file(&path)?;

    // If it's from HF cache, also remove metadata
    if source != "path" && source != "local" {
        let meta_path = path.with_extension("json");
        if meta_path.exists() {
            let _ = std::fs::remove_file(meta_path);
        }
    }

    println!("OK");
    println!("Freed {}", format_size(size));

    Ok(())
}

/// Show running models (similar to docker ps)
fn show_running_models(socket: &str, json_output: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let models = client.list_models()?;

    if json_output {
        let mut models_json = Vec::new();
        for model in &models {
            models_json.push(serde_json::json!({
                "name": model.alias,
                "is_default": model.is_default,
                "path": model.info.path,
                "parameters": model.info.parameters,
                "parameters_formatted": format!("{}M", model.info.parameters / 1_000_000),
                "context_size": model.info.context_size,
                "gpu_layers": model.info.gpu_layers,
                "active_requests": model.active_requests,
            }));
        }
        println!("{}", serde_json::to_string_pretty(&models_json)?);
        return Ok(());
    }

    if models.is_empty() {
        println!("No models currently running.");
        println!();
        println!("Load a model with:");
        println!("  mullama serve --model ./model.gguf");
        println!("  mullama load ./model.gguf");
        return Ok(());
    }

    println!("NAME                 SIZE       GPU      CONTEXT    ACTIVE");

    for model in &models {
        let default_marker = if model.is_default { "*" } else { " " };
        let name = format!("{}{}", default_marker, model.alias);
        let name_display = if name.len() > 20 {
            format!("{}...", &name[..17])
        } else {
            name
        };

        let size = format!("{}M", model.info.parameters / 1_000_000);
        let gpu = if model.info.gpu_layers > 0 {
            format!("{} layers", model.info.gpu_layers)
        } else {
            "CPU".to_string()
        };
        let active = if model.active_requests > 0 {
            format!("{} req", model.active_requests)
        } else {
            "-".to_string()
        };

        println!(
            "{:<20} {:>10} {:>12} {:>10} {:>8}",
            name_display, size, gpu, model.info.context_size, active
        );
    }

    println!();
    println!("* = default model");

    Ok(())
}

/// Show model details
async fn show_model_details(
    name: &str,
    show_modelfile: bool,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    // Try to find the model
    let mut found = None;

    for model in &cached {
        let short_name = format!(
            "{}:{}",
            model.repo_id.split('/').last().unwrap_or(&model.repo_id),
            model.filename.trim_end_matches(".gguf")
        );

        if model.filename == name
            || model.repo_id == name
            || short_name == name
            || model.filename.trim_end_matches(".gguf") == name
        {
            found = Some(model);
            break;
        }
    }

    let model = match found {
        Some(m) => m,
        None => {
            eprintln!("Model '{}' not found.", name);
            eprintln!();
            eprintln!("Use 'mullama list' to see available models.");
            return Err("Model not found".into());
        }
    };

    if json_output {
        let info = serde_json::json!({
            "name": format!("{}:{}", model.repo_id.replace('/', "-"),
                model.filename.trim_end_matches(".gguf")),
            "repo_id": model.repo_id,
            "filename": model.filename,
            "size": model.size_bytes,
            "size_formatted": format_size(model.size_bytes),
            "downloaded": model.downloaded_at,
            "path": model.local_path,
        });
        println!("{}", serde_json::to_string_pretty(&info)?);
        return Ok(());
    }

    if show_modelfile {
        // Check if there's a Modelfile next to the model
        let modelfile_path = model.local_path.with_extension("modelfile");
        let mullamafile_path = model.local_path.with_extension("mullamafile");

        if modelfile_path.exists() {
            let content = std::fs::read_to_string(&modelfile_path)?;
            println!("{}", content);
        } else if mullamafile_path.exists() {
            let content = std::fs::read_to_string(&mullamafile_path)?;
            println!("{}", content);
        } else {
            // Generate a default Modelfile
            println!("# Modelfile for {}", model.filename);
            println!("# Auto-generated - no custom Modelfile found");
            println!();
            println!("FROM {}", model.local_path.display());
            println!();
            println!("PARAMETER temperature 0.7");
            println!("PARAMETER top_p 0.9");
            println!("PARAMETER num_ctx 4096");
        }
        return Ok(());
    }

    // Show detailed info
    println!("Model: {}", model.filename.trim_end_matches(".gguf"));
    println!();
    println!("  Repository:  {}", model.repo_id);
    println!("  Filename:    {}", model.filename);
    println!("  Size:        {}", format_size(model.size_bytes));
    println!("  Downloaded:  {}", model.downloaded_at);
    println!("  Path:        {}", model.local_path.display());
    println!();
    println!("Quick start:");
    println!("  mullama serve --model {}", model.local_path.display());

    Ok(())
}

/// Create a model from a Modelfile/Mullamafile
async fn create_model(
    name: &str,
    file: Option<PathBuf>,
    download: bool,
    show_progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Find the Modelfile
    let modelfile_path = if let Some(path) = file {
        if !path.exists() {
            return Err(format!("Modelfile not found: {}", path.display()).into());
        }
        path
    } else {
        find_modelfile(".").ok_or("No Modelfile or Mullamafile found in current directory")?
    };

    println!("Reading {}...", modelfile_path.display());

    // Parse the Modelfile
    let parser = ModelfileParser::new();
    let modelfile = parser.parse_file(&modelfile_path)?;

    println!("  FROM: {}", modelfile.from);

    if let Some(ref system) = modelfile.system {
        let preview = if system.len() > 50 {
            format!("{}...", &system[..50])
        } else {
            system.clone()
        };
        println!("  SYSTEM: {}", preview.replace('\n', " "));
    }

    if !modelfile.parameters.is_empty() {
        println!("  Parameters: {}", modelfile.parameters.len());
    }

    if modelfile.gpu_layers.is_some() || modelfile.flash_attention.is_some() {
        println!("  Mullama extensions: enabled");
    }

    // Resolve the base model
    let base_model_path = resolve_base_model(&modelfile.from, download, show_progress).await?;

    println!();
    println!("Base model: {}", base_model_path.display());

    // Create the model directory
    let mullama_dir = dirs::home_dir()
        .map(|h| h.join(".mullama").join("models"))
        .unwrap_or_else(|| PathBuf::from(".mullama/models"));

    std::fs::create_dir_all(&mullama_dir)?;

    let model_dir = mullama_dir.join(name);
    std::fs::create_dir_all(&model_dir)?;

    // Save the Mullamafile
    let mullamafile_dest = model_dir.join("Mullamafile");
    let mut saved_modelfile = modelfile.clone();
    saved_modelfile.from = base_model_path.display().to_string();
    saved_modelfile.save(&mullamafile_dest)?;

    // Create a symlink or copy to the model
    let model_link = model_dir.join("model.gguf");
    if model_link.exists() {
        std::fs::remove_file(&model_link)?;
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&base_model_path, &model_link)?;
    }

    #[cfg(windows)]
    {
        std::fs::copy(&base_model_path, &model_link)?;
    }

    // Create metadata
    let metadata = serde_json::json!({
        "name": name,
        "created": chrono::Utc::now().to_rfc3339(),
        "base_model": base_model_path.display().to_string(),
        "system": modelfile.system,
        "parameters": modelfile.parameters.iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect::<std::collections::HashMap<_, _>>(),
        "gpu_layers": modelfile.gpu_layers,
        "flash_attention": modelfile.flash_attention,
    });

    let metadata_path = model_dir.join("metadata.json");
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!();
    println!("Created model '{}' successfully!", name);
    println!();
    println!("Model location: {}", model_dir.display());
    println!();
    println!("To use this model:");
    println!("  mullama serve --model {}", model_link.display());
    println!("  mullama run --model {} \"Hello!\"", name);

    Ok(())
}

/// Resolve a base model reference to a local path
async fn resolve_base_model(
    from: &str,
    download: bool,
    show_progress: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Check if it's a local path
    let path = PathBuf::from(from);
    if path.exists() {
        return Ok(path);
    }

    // Check if it's an HF spec
    if from.starts_with("hf:") {
        if !download {
            return Err("Base model is HuggingFace spec but --download=false".into());
        }

        println!();
        println!("Downloading base model from HuggingFace...");
        let (_, resolved_path) = resolve_model_path(from, show_progress).await?;
        return Ok(resolved_path);
    }

    // Try to resolve as an alias
    let resolved = resolve_model_name(from);
    match resolved {
        ResolvedModel::LocalPath(p) => {
            if p.exists() {
                Ok(p)
            } else {
                Err(format!("Local model not found: {}", p.display()).into())
            }
        }
        ResolvedModel::HuggingFace { spec, .. } => {
            if !download {
                return Err(format!(
                    "Model '{}' needs to be downloaded. Use --download=true or run 'mullama pull {}'",
                    from, from
                ).into());
            }

            println!();
            println!("Downloading '{}' from HuggingFace...", from);
            let (_, resolved_path) = resolve_model_path(&spec, show_progress).await?;
            Ok(resolved_path)
        }
        ResolvedModel::Ollama { name, tag } => {
            use mullama::daemon::OllamaClient;

            let model_name = format!("{}:{}", name, tag);

            // Check if Ollama model is cached
            let client = OllamaClient::new()?;
            if let Some(model) = client.get_cached(&model_name) {
                Ok(model.gguf_path)
            } else if download {
                println!();
                println!("Pulling '{}' from Ollama registry...", model_name);
                let model = client.pull(&model_name, show_progress).await?;
                Ok(model.gguf_path)
            } else {
                Err(format!(
                    "Ollama model '{}' not downloaded. Use --download=true or run 'mullama pull {}'",
                    model_name, model_name
                ).into())
            }
        }
        ResolvedModel::Unknown(name) => {
            // Check if it's cached
            let downloader = HfDownloader::new()?;
            let cached = downloader.list_cached();

            for model in cached {
                let short_name = format!(
                    "{}:{}",
                    model.repo_id.split('/').last().unwrap_or(&model.repo_id),
                    model.filename.trim_end_matches(".gguf")
                );

                if model.filename == name
                    || short_name == name
                    || model.filename.trim_end_matches(".gguf") == name
                {
                    return Ok(model.local_path);
                }
            }

            Err(format!(
                "Unknown model '{}'. Use a local path, HF spec (hf:owner/repo), or a known alias.",
                name
            )
            .into())
        }
    }
}

/// Copy/rename a model
async fn copy_model(source: &str, destination: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mullama_dir = dirs::home_dir()
        .map(|h| h.join(".mullama").join("models"))
        .unwrap_or_else(|| PathBuf::from(".mullama/models"));

    // Find source model
    let source_dir = mullama_dir.join(source);
    let source_mullamafile = source_dir.join("Mullamafile");

    if !source_dir.exists() {
        // Try to find in cache
        let downloader = HfDownloader::new()?;
        let cached = downloader.list_cached();

        let mut found = None;
        for model in &cached {
            let short_name = format!(
                "{}:{}",
                model.repo_id.split('/').last().unwrap_or(&model.repo_id),
                model.filename.trim_end_matches(".gguf")
            );

            if model.filename == source
                || short_name == source
                || model.filename.trim_end_matches(".gguf") == source
            {
                found = Some(model);
                break;
            }
        }

        if let Some(model) = found {
            // Create a new model from the cached one
            let dest_dir = mullama_dir.join(destination);
            std::fs::create_dir_all(&dest_dir)?;

            // Create Mullamafile
            let modelfile = Modelfile::from_model(model.local_path.display().to_string());
            modelfile.save(dest_dir.join("Mullamafile"))?;

            // Create symlink
            let model_link = dest_dir.join("model.gguf");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&model.local_path, &model_link)?;
            #[cfg(windows)]
            std::fs::copy(&model.local_path, &model_link)?;

            // Create metadata
            let metadata = serde_json::json!({
                "name": destination,
                "created": chrono::Utc::now().to_rfc3339(),
                "copied_from": source,
                "base_model": model.local_path.display().to_string(),
            });
            std::fs::write(
                dest_dir.join("metadata.json"),
                serde_json::to_string_pretty(&metadata)?,
            )?;

            println!("Copied '{}' to '{}'", source, destination);
            println!("Model location: {}", dest_dir.display());
            return Ok(());
        }

        return Err(format!("Model '{}' not found", source).into());
    }

    // Copy custom model
    let dest_dir = mullama_dir.join(destination);

    if dest_dir.exists() {
        return Err(format!("Destination '{}' already exists", destination).into());
    }

    // Copy directory contents
    std::fs::create_dir_all(&dest_dir)?;

    for entry in std::fs::read_dir(&source_dir)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest_dir.join(entry.file_name());

        if src_path.is_file() {
            std::fs::copy(&src_path, &dest_path)?;
        } else if src_path.is_symlink() {
            let target = std::fs::read_link(&src_path)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &dest_path)?;
            #[cfg(windows)]
            std::fs::copy(&src_path, &dest_path)?;
        }
    }

    // Update metadata
    let metadata_path = dest_dir.join("metadata.json");
    if metadata_path.exists() {
        let content = std::fs::read_to_string(&metadata_path)?;
        let mut metadata: serde_json::Value = serde_json::from_str(&content)?;
        metadata["name"] = serde_json::json!(destination);
        metadata["copied_from"] = serde_json::json!(source);
        metadata["copied_at"] = serde_json::json!(chrono::Utc::now().to_rfc3339());
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    }

    println!("Copied '{}' to '{}'", source, destination);
    println!("Model location: {}", dest_dir.display());

    Ok(())
}

// ==================== Daemon Management ====================

/// Handle daemon management actions
fn handle_daemon_action(action: DaemonAction) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        DaemonAction::Start {
            http_port,
            http_addr,
            api_key,
            require_api_key,
            gpu_layers,
            context_size,
            context_pool_size,
            socket,
        } => {
            daemon_start(
                &socket,
                http_port,
                &http_addr,
                api_key,
                require_api_key,
                gpu_layers,
                context_size,
                context_pool_size,
            )?;
        }

        DaemonAction::Stop { socket, force: _ } => {
            daemon_stop(&socket)?;
        }

        DaemonAction::Restart { socket } => {
            daemon_restart(&socket)?;
        }

        DaemonAction::Status { socket, json } => {
            daemon_show_status(&socket, json)?;
        }

        DaemonAction::Logs { lines, follow } => {
            daemon_logs(lines, follow)?;
        }
    }

    Ok(())
}

/// Start the daemon in background
fn daemon_start(
    socket: &str,
    http_port: u16,
    http_addr: &str,
    api_key: Option<String>,
    require_api_key: bool,
    gpu_layers: i32,
    context_size: u32,
    context_pool_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if already running
    if is_daemon_running(socket) {
        println!("Daemon is already running.");
        if let Ok(info) = daemon_status(socket) {
            println!("  Version: {}", info.version);
            println!("  Uptime:  {}s", info.uptime_secs);
            println!("  Models:  {}", info.models_loaded);
        }
        return Ok(());
    }

    println!("Starting Mullama daemon...");

    let config = SpawnConfig {
        socket: socket.to_string(),
        http_port,
        http_addr: http_addr.to_string(),
        api_key,
        require_api_key,
        gpu_layers,
        context_size,
        context_pool_size,
        background: true,
        log_file: Some(default_log_path()),
        ..Default::default()
    };

    match spawn_daemon(&config) {
        SpawnResult::AlreadyRunning => {
            println!("Daemon is already running.");
        }
        SpawnResult::Spawned { pid } => {
            // Wait for daemon to be ready
            print!("Waiting for daemon to start");
            io::stdout().flush()?;

            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(30);

            while start.elapsed() < timeout {
                if is_daemon_running(socket) {
                    println!(" OK");
                    println!();
                    println!("Daemon started successfully!");
                    if let Some(pid) = pid {
                        println!("  PID:     {}", pid);
                    }
                    println!("  Socket:  {}", socket);
                    println!("  HTTP:    http://{}:{}", http_addr, http_port);
                    println!("  Logs:    {}", default_log_path().display());
                    return Ok(());
                }
                print!(".");
                io::stdout().flush()?;
                std::thread::sleep(Duration::from_millis(500));
            }

            println!(" FAILED");
            eprintln!("Daemon did not start within {} seconds.", timeout.as_secs());
            eprintln!("Check logs at: {}", default_log_path().display());
        }
        SpawnResult::Failed(e) => {
            eprintln!("Failed to start daemon: {}", e);
        }
    }

    Ok(())
}

/// Stop the daemon
fn daemon_stop(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running(socket) {
        println!("Daemon is not running.");
        return Ok(());
    }

    print!("Stopping daemon... ");
    io::stdout().flush()?;

    match stop_daemon(socket) {
        Ok(()) => {
            // Wait for daemon to stop
            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(10);

            while start.elapsed() < timeout {
                if !is_daemon_running(socket) {
                    println!("OK");
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            println!("TIMEOUT");
            eprintln!("Daemon did not stop within {} seconds.", timeout.as_secs());
        }
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

/// Restart the daemon
fn daemon_restart(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Get current config from running daemon if possible
    let (http_port, gpu_layers, context_size) = if let Ok(info) = daemon_status(socket) {
        let port = info
            .http_endpoint
            .and_then(|e| e.split(':').last().and_then(|p| p.parse().ok()))
            .unwrap_or(8080);
        (port, 0, 4096) // We don't have these from status
    } else {
        (8080, 0, 4096)
    };

    // Stop
    if is_daemon_running(socket) {
        println!("Stopping daemon...");
        daemon_stop(socket)?;
        std::thread::sleep(Duration::from_millis(500));
    }

    // Start
    daemon_start(
        socket,
        http_port,
        "127.0.0.1",
        None,
        false,
        gpu_layers,
        context_size,
        mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE,
    )?;

    Ok(())
}

/// Show daemon status
fn daemon_show_status(socket: &str, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running(socket) {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "running": false,
                    "socket": socket,
                }))?
            );
        } else {
            println!("Daemon is not running.");
            println!();
            println!("Start with: mullama daemon start");
        }
        return Ok(());
    }

    match daemon_status(socket) {
        Ok(info) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "running": true,
                        "version": info.version,
                        "uptime_secs": info.uptime_secs,
                        "models_loaded": info.models_loaded,
                        "socket": info.socket,
                        "http_endpoint": info.http_endpoint,
                    }))?
                );
            } else {
                println!("Mullama Daemon Status");
                println!("=====================");
                println!("Running:     Yes");
                println!("Version:     {}", info.version);
                println!("Uptime:      {}s", info.uptime_secs);
                println!("Models:      {}", info.models_loaded);
                println!("Socket:      {}", info.socket);
                if let Some(ref http) = info.http_endpoint {
                    println!("HTTP:        {}", http);
                }
                println!("Logs:        {}", default_log_path().display());
            }
        }
        Err(e) => {
            eprintln!("Failed to get status: {}", e);
        }
    }

    Ok(())
}

/// Show daemon logs
fn daemon_logs(lines: usize, follow: bool) -> Result<(), Box<dyn std::error::Error>> {
    let log_path = default_log_path();

    if !log_path.exists() {
        println!("No log file found at: {}", log_path.display());
        return Ok(());
    }

    if follow {
        // Use tail -f
        let status = std::process::Command::new("tail")
            .arg("-f")
            .arg("-n")
            .arg(lines.to_string())
            .arg(&log_path)
            .status()?;

        if !status.success() {
            eprintln!("Failed to follow logs");
        }
    } else {
        // Read last N lines
        let file = std::fs::File::open(&log_path)?;
        let reader = std::io::BufReader::new(file);
        let all_lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

        let start = if all_lines.len() > lines {
            all_lines.len() - lines
        } else {
            0
        };

        for line in &all_lines[start..] {
            println!("{}", line);
        }
    }

    Ok(())
}
