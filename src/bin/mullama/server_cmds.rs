use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use mullama::daemon::{
    create_openai_router, resolve_model_path, Daemon, DaemonBuilder, ModelConfig,
};

use crate::shared::{derive_alias_from_path, generate_api_key, is_loopback_http_addr};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_server(
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
    flash_attn: bool,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    no_mmap: bool,
    mlock: bool,
    batch_size: Option<u32>,
    rope_freq_base: Option<f32>,
    rope_freq_scale: Option<f32>,
    split_mode: Option<String>,
    defrag_thold: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::registry::{resolve_model_name, ResolvedModel};
    use mullama::daemon::OllamaClient;

    let mut resolved_api_key = api_key.or_else(|| std::env::var("MULLAMA_API_KEY").ok());
    let is_loopback_bind = is_loopback_http_addr(&http_addr);
    let enforce_api_key =
        http_port > 0 && (require_api_key || !is_loopback_bind || resolved_api_key.is_some());
    let mut generated_api_key = false;

    if enforce_api_key && resolved_api_key.is_none() {
        resolved_api_key = Some(generate_api_key());
        generated_api_key = true;
    }

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

    let mut resolved_models: Vec<(String, PathBuf, Option<ModelConfig>)> = Vec::new();
    for spec in &models {
        match resolve_model_name(spec) {
            ResolvedModel::LocalPath(path) => {
                resolved_models.push((derive_alias_from_path(&path), path, None));
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
                let model_name = format!("{}:{}", name, tag);
                let alias = format!("{}-{}", name, tag);
                println!("Resolving Ollama model: {}", model_name);
                let client = match OllamaClient::new() {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Failed to initialize Ollama client: {}", e);
                        continue;
                    }
                };
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
                let path = PathBuf::from(&name);
                if path.exists() {
                    resolved_models.push((derive_alias_from_path(&path), path, None));
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

    for (alias, path, _) in &resolved_models {
        builder = builder.model(format!("{}:{}", alias, path.display()));
    }

    let (mut daemon, mut initial_models) = builder.build();

    daemon.config.model_defaults.flash_attn = flash_attn;
    daemon.config.model_defaults.use_mmap = if no_mmap { Some(false) } else { None };
    daemon.config.model_defaults.use_mlock = mlock;
    daemon.config.model_defaults.cache_type_k = cache_type_k.clone();
    daemon.config.model_defaults.cache_type_v = cache_type_v.clone();
    daemon.config.model_defaults.n_batch = batch_size;
    daemon.config.model_defaults.rope_freq_base = rope_freq_base;
    daemon.config.model_defaults.rope_freq_scale = rope_freq_scale;
    daemon.config.model_defaults.defrag_thold = defrag_thold;
    daemon.config.model_defaults.split_mode = split_mode.clone();

    for model_config in &mut initial_models {
        if flash_attn {
            model_config.flash_attn = true;
        }
        if no_mmap {
            model_config.use_mmap = Some(false);
        }
        if mlock {
            model_config.use_mlock = true;
        }
        if let Some(ref k) = cache_type_k {
            model_config.cache_type_k = Some(k.clone());
        }
        if let Some(ref v) = cache_type_v {
            model_config.cache_type_v = Some(v.clone());
        }
        if let Some(batch) = batch_size {
            model_config.n_batch = Some(batch);
        }
        if let Some(base) = rope_freq_base {
            model_config.rope_freq_base = Some(base);
        }
        if let Some(scale) = rope_freq_scale {
            model_config.rope_freq_scale = Some(scale);
        }
        if let Some(thold) = defrag_thold {
            model_config.defrag_thold = Some(thold);
        }
        if let Some(ref mode) = split_mode {
            model_config.split_mode = Some(mode.clone());
        }
    }

    let daemon = std::sync::Arc::new(daemon);

    let model_configs_by_alias: std::collections::HashMap<String, ModelConfig> = resolved_models
        .iter()
        .filter_map(|(alias, _path, config)| config.clone().map(|c| (alias.clone(), c)))
        .collect();
    for config in &mut initial_models {
        if let Some(model_config) = model_configs_by_alias.get(&config.alias) {
            config.model_config = Some(model_config.clone());
            if let Some(ctx) = model_config.context_size {
                if config.context_size == context_size {
                    config.context_size = ctx;
                }
            }
        }
    }

    if let Some(ref mmp) = mmproj {
        if let Some(first) = initial_models.first_mut() {
            first.mmproj_path = Some(mmp.display().to_string());
        }
    }

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

    let ipc_daemon = daemon.clone();
    let ipc_socket = socket.clone();
    let _ipc_handle = tokio::spawn(async move {
        if let Err(e) = run_ipc_server(ipc_daemon, &ipc_socket).await {
            eprintln!("IPC server error: {}", e);
        }
    });

    let _http_handle = if http_port > 0 {
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

    tokio::signal::ctrl_c().await?;
    println!("\nShutting down...");

    daemon
        .shutdown
        .store(true, std::sync::atomic::Ordering::SeqCst);

    tokio::time::sleep(Duration::from_millis(100)).await;

    mullama::backend_free();

    Ok(())
}

pub(crate) async fn run_ipc_server(
    daemon: std::sync::Arc<Daemon>,
    addr: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use mullama::daemon::Request;
    use nng::options::{Options, RecvTimeout};
    use nng::{Protocol, Socket};

    let socket = Socket::new(Protocol::Rep0)?;
    socket.listen(addr)?;
    socket.set_opt::<RecvTimeout>(Some(Duration::from_millis(250)))?;

    loop {
        if daemon.is_shutdown() {
            break;
        }

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
            Err(nng::Error::TimedOut) => continue,
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
