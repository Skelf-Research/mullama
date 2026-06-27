use std::path::PathBuf;

use clap::{Parser, Subcommand};
use mullama::daemon::{DEFAULT_HTTP_PORT, DEFAULT_SOCKET};

#[derive(Parser)]
#[command(name = "mullama")]
#[command(author, version, about = "Multi-model LLM server and client")]
#[command(propagate_version = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    #[command(alias = "ls")]
    List {
        #[arg(short, long)]
        verbose: bool,
        #[arg(long)]
        json: bool,
    },
    #[command(alias = "delete")]
    Rm {
        name: String,
        #[arg(short, long)]
        force: bool,
    },
    Ps {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(long)]
        json: bool,
    },
    Show {
        name: String,
        #[arg(long)]
        modelfile: bool,
        #[arg(long)]
        json: bool,
    },
    #[command(alias = "start")]
    Serve {
        #[arg(short, long, value_name = "SPEC")]
        model: Vec<String>,
        #[arg(long)]
        mmproj: Option<PathBuf>,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short = 'p', long, default_value_t = DEFAULT_HTTP_PORT)]
        http_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        http_addr: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        require_api_key: bool,
        #[arg(long, default_value = "4096")]
        max_tokens_limit: u32,
        #[arg(long, default_value = "2")]
        max_request_body_mb: u32,
        #[arg(long, default_value = "64")]
        max_concurrent_requests: usize,
        #[arg(long, default_value = "200")]
        max_requests_per_second: u64,
        #[arg(short, long, default_value_t = mullama::default_gpu_layers())]
        gpu_layers: i32,
        #[arg(short, long, default_value = "4096")]
        context_size: u32,
        #[arg(long, default_value_t = mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE)]
        context_pool_size: usize,
        #[arg(short, long)]
        threads: Option<i32>,
        #[arg(short, long)]
        verbose: bool,
        #[arg(long)]
        tls_cert: Option<String>,
        #[arg(long)]
        tls_key: Option<String>,
        #[arg(long)]
        flash_attn: bool,
        #[arg(long)]
        cache_type_k: Option<String>,
        #[arg(long)]
        cache_type_v: Option<String>,
        #[arg(long)]
        no_mmap: bool,
        #[arg(long)]
        mlock: bool,
        #[arg(long)]
        batch_size: Option<u32>,
        /// Physical micro-batch size (`n_ubatch`). Kernel dispatch granularity
        /// on Metal/CUDA; defaults to llama.cpp's 512. Tune per-device with a
        /// bench sweep.
        #[arg(long)]
        ubatch_size: Option<u32>,
        /// Max concurrent sequences per context (`n_seq_max`). Phase-C
        /// scaffolding; today the daemon still serves one request per context,
        /// but a higher value enables cheap `kv_cache_seq_cp` for branching
        /// agentic patterns and is the substrate for batched concurrent decode.
        #[arg(long)]
        n_seq_max: Option<u32>,
        #[arg(long)]
        rope_freq_base: Option<f32>,
        #[arg(long)]
        rope_freq_scale: Option<f32>,
        #[arg(long)]
        split_mode: Option<String>,
        #[arg(long)]
        defrag_thold: Option<f32>,
        /// Session pre-warm mode: `off`, `idle` (only when no requests are in
        /// flight), or `active` (parallel-fill — pre-warm during live decodes;
        /// best on Apple Silicon). Defaults to `active` on macOS, `idle` else.
        #[arg(long)]
        hydration: Option<String>,
    },
    #[command(alias = "tui")]
    Chat {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },
    Run {
        model: String,
        prompt: Option<String>,
        #[arg(short = 'n', long, default_value = "512")]
        max_tokens: u32,
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short, long)]
        image: Option<PathBuf>,
        #[arg(long, default_value = "8080")]
        http_port: u16,
        #[arg(short, long, default_value_t = mullama::default_gpu_layers())]
        gpu_layers: i32,
        #[arg(short, long, default_value = "4096")]
        context_size: u32,
        #[arg(long)]
        stats: bool,
        #[arg(long)]
        flash_attn: bool,
        #[arg(long)]
        cache_type_k: Option<String>,
        #[arg(long)]
        cache_type_v: Option<String>,
        #[arg(long)]
        no_mmap: bool,
        #[arg(long)]
        mlock: bool,
        #[arg(long)]
        batch_size: Option<u32>,
    },
    Models {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short, long)]
        verbose: bool,
    },
    Load {
        spec: String,
        #[arg(short, long, default_value_t = mullama::default_gpu_layers())]
        gpu_layers: i32,
        #[arg(short, long, default_value = "4096")]
        context_size: u32,
        #[arg(long)]
        mmproj: Option<PathBuf>,
        #[arg(long)]
        flash_attn: bool,
        #[arg(long)]
        cache_type_k: Option<String>,
        #[arg(long)]
        cache_type_v: Option<String>,
        #[arg(long)]
        no_mmap: bool,
        #[arg(long)]
        mlock: bool,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Unload {
        alias: String,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Default {
        alias: String,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Status {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(long)]
        json: bool,
    },
    Ping {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Stop {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short, long)]
        force: bool,
    },
    Tokenize {
        text: String,
        #[arg(short, long)]
        model: Option<String>,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Embed {
        text: Vec<String>,
        #[arg(short, long)]
        model: Option<String>,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(long)]
        json: bool,
    },
    #[command(alias = "download")]
    Pull {
        spec: String,
        #[arg(short, long)]
        quiet: bool,
    },
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    #[command(alias = "find")]
    Search {
        query: String,
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
        #[arg(long)]
        all: bool,
        #[arg(short, long)]
        files: bool,
    },
    Info {
        repo: String,
    },
    Create {
        name: String,
        #[arg(short, long)]
        file: Option<PathBuf>,
        #[arg(long, default_value = "true")]
        download: bool,
        #[arg(short, long)]
        quiet: bool,
    },
    #[command(alias = "copy")]
    Cp {
        source: String,
        destination: String,
    },
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
}

#[derive(Subcommand)]
pub(crate) enum DaemonAction {
    Start {
        #[arg(short = 'p', long, default_value = "8080")]
        http_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        http_addr: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        require_api_key: bool,
        #[arg(short, long, default_value_t = mullama::default_gpu_layers())]
        gpu_layers: i32,
        #[arg(short, long, default_value = "4096")]
        context_size: u32,
        #[arg(long, default_value_t = mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE)]
        context_pool_size: usize,
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Stop {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(short, long)]
        force: bool,
    },
    Restart {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    Status {
        #[arg(short, long, default_value = DEFAULT_SOCKET)]
        socket: String,
        #[arg(long)]
        json: bool,
    },
    Logs {
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,
        #[arg(short, long)]
        follow: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum CacheAction {
    List {
        #[arg(short, long)]
        verbose: bool,
    },
    Path,
    Size,
    Remove {
        repo_id: String,
        #[arg(short, long)]
        filename: Option<String>,
    },
    Clear {
        #[arg(short, long)]
        force: bool,
    },
}
