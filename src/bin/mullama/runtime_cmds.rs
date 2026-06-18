use std::io::{self, Write};
use std::path::PathBuf;

use mullama::daemon::spawn::default_log_path;
use mullama::daemon::{
    resolve_model_name, resolve_model_path, spawn_daemon, ChatMessage, MessageContent,
    ResolvedModel, SpawnConfig, SpawnResult, TuiApp,
};

use crate::shared::{connect, derive_alias_from_path};

pub(crate) fn run_chat(socket: &str, timeout: u64) -> Result<(), Box<dyn std::error::Error>> {
    let connect_timeout = if timeout > 0 {
        std::time::Duration::from_secs(timeout)
    } else {
        std::time::Duration::from_secs(5)
    };
    let client = mullama::daemon::DaemonClient::connect_with_timeout(socket, connect_timeout)?;

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

    let mut app = TuiApp::new(client);
    app.run()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_model_with_prompt(
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
    flash_attn: bool,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    no_mmap: bool,
    mlock: bool,
    batch_size: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::OllamaClient;

    let (model_alias, model_path) = match resolve_model_name(model_spec) {
        ResolvedModel::Ollama { name, tag } => {
            let model_name = format!("{}:{}", name, tag);
            let alias = format!("{}-{}", name, tag);

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
        ResolvedModel::LocalPath(path) => (derive_alias_from_path(&path), path),
        ResolvedModel::Unknown(name) => {
            let path = PathBuf::from(&name);
            if path.exists() {
                (derive_alias_from_path(&path), path)
            } else {
                return Err(format!(
                    "Unknown model: {} (not found locally or in registries)",
                    name
                )
                .into());
            }
        }
    };

    let client = match connect(socket) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Starting daemon...");

            let config = SpawnConfig {
                binary_path: None,
                socket: socket.to_string(),
                http_port,
                gpu_layers,
                context_size,
                startup_timeout: std::time::Duration::from_secs(60),
                background: true,
                log_file: Some(default_log_path()),
                flash_attn,
                cache_type_k: cache_type_k.clone(),
                cache_type_v: cache_type_v.clone(),
                ..Default::default()
            };

            match spawn_daemon(&config) {
                SpawnResult::AlreadyRunning => {}
                SpawnResult::Spawned { .. } => {}
                SpawnResult::Failed(e) => {
                    return Err(format!("Failed to start daemon: {}", e).into());
                }
            }

            let mut attempts = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if let Ok(c) = connect(socket) {
                    break c;
                }
                attempts += 1;
                if attempts > 150 {
                    return Err("Timed out waiting for daemon to start".into());
                }
            }
        }
    };

    let loaded_models = client.list_models()?;
    let model_loaded = loaded_models.iter().any(|m| m.alias == model_alias);

    if !model_loaded {
        eprintln!("Loading {}...", model_alias);

        let use_mmap = if no_mmap { Some(false) } else { None };
        match client.load_model_full(
            &model_alias,
            &model_path.display().to_string(),
            gpu_layers,
            context_size,
            flash_attn,
            cache_type_k.clone(),
            cache_type_v.clone(),
            use_mmap,
            mlock,
            batch_size,
        ) {
            Ok(_) => {}
            Err(e) => {
                return Err(format!("Failed to load model: {}", e).into());
            }
        }
    }

    if let Some(prompt_text) = prompt {
        if let Some(image_path) = image {
            let http_addr = std::env::var("MULLAMA_HTTP_ADDR")
                .unwrap_or_else(|_| format!("127.0.0.1:{}", http_port));
            run_vision_prompt(
                &http_addr,
                prompt_text,
                Some(&model_alias),
                max_tokens,
                temperature,
                image_path,
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
        eprintln!(">>> Send a message (/? for help)");

        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();

        let mut conversation: Vec<mullama::daemon::ChatMessage> = Vec::new();

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
                    eprintln!("  /clear              - Clear conversation history");
                    eprintln!("  /?                  - Show this help");
                    continue;
                }
                "/clear" => {
                    conversation.clear();
                    eprintln!("(Conversation cleared)");
                    continue;
                }
                _ => {}
            }

            conversation.push(ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(input.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });

            match client.chat_completion(conversation.clone(), Some(&model_alias), max_tokens, temperature) {
                Ok(result) => {
                    println!("{}", result.text);
                    conversation.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: MessageContent::Text(result.text.clone()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
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
                    conversation.pop();
                    eprintln!("Error: {}", e);
                }
            }
            println!();
        }
    }

    Ok(())
}

pub(crate) async fn run_vision_prompt(
    http_addr: &str,
    prompt: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: f32,
    image_path: &PathBuf,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine;

    let start = std::time::Instant::now();

    let image_data = std::fs::read(image_path).map_err(|e| {
        format!(
            "Failed to read image file '{}': {}",
            image_path.display(),
            e
        )
    })?;

    let mime_type = match image_path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        _ => "image/jpeg",
    };

    let base64_image = base64::engine::general_purpose::STANDARD.encode(&image_data);
    let image_url = format!("data:{};base64,{}", mime_type, base64_image);

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

    let client = reqwest::Client::new();
    let url = format!("http://{}/v1/chat/completions", http_addr);

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

pub(crate) fn list_models(socket: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_model(
    socket: &str,
    spec: &str,
    gpu_layers: i32,
    context_size: u32,
    mmproj: Option<PathBuf>,
    flash_attn: bool,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    no_mmap: bool,
    mlock: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if mmproj.is_some() {
        eprintln!("Warning: --mmproj is not yet supported via IPC protocol.");
        eprintln!("         Vision models can be loaded directly via the server:");
        eprintln!("         mullama serve --model model.gguf (with mmproj support coming soon)");
        eprintln!();
    }

    let client = connect(socket)?;

    let (alias, path) = if let Some(pos) = spec.find(':') {
        (spec[..pos].to_string(), spec[pos + 1..].to_string())
    } else {
        let p = std::path::Path::new(spec);
        (derive_alias_from_path(p), spec.to_string())
    };

    print!("Loading model '{}'... ", alias);
    io::stdout().flush()?;

    let use_mmap = if no_mmap { Some(false) } else { None };

    match client.load_model_full(
        &alias,
        &path,
        gpu_layers,
        context_size,
        flash_attn,
        cache_type_k,
        cache_type_v,
        use_mmap,
        mlock,
        None,
    ) {
        Ok((_alias, info)) => {
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

pub(crate) fn unload_model(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
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

pub(crate) fn set_default(socket: &str, alias: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    match client.set_default_model(alias) {
        Ok(()) => println!("Default model set to '{}'", alias),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

pub(crate) fn show_status(socket: &str, json: bool) -> Result<(), Box<dyn std::error::Error>> {
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

pub(crate) fn ping_daemon(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let client = connect(socket)?;
    let (uptime, version) = client.ping()?;
    let latency = start.elapsed();

    println!("Pong from mullama v{}", version);
    println!("  Daemon uptime: {}s", uptime);
    println!("  Round-trip:    {:?}", latency);

    Ok(())
}

pub(crate) fn cli_stop_daemon(socket: &str, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;

    if force {
        print!("Force shutting down daemon... ");
    } else {
        print!("Shutting down daemon... ");
    }
    io::stdout().flush()?;

    match client.shutdown() {
        Ok(()) => println!("OK"),
        Err(e) => {
            if force {
                match std::process::Command::new("pkill")
                    .arg("-f")
                    .arg("mullama serve")
                    .output()
                {
                    Ok(_) => println!("OK (force killed)"),
                    Err(kill_err) => println!("FAILED (kill error: {})", kill_err),
                }
            } else {
                println!("FAILED");
                eprintln!("Error: {}", e);
            }
        }
    }

    Ok(())
}

pub(crate) fn tokenize_text(
    socket: &str,
    text: &str,
    model: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let tokens = client.tokenize(text, model)?;

    println!("Tokens ({}): {:?}", tokens.len(), tokens);

    Ok(())
}

pub(crate) fn embed_text(
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
