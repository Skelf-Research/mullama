use std::io::{self, Write};
use std::path::PathBuf;

use mullama::daemon::HfDownloader;

use super::common::{
    cached_model_short_name, find_cached_model, local_model_path, print_default_modelfile,
    truncate_display,
};
use crate::shared::{connect, format_size, format_time_ago};

pub(crate) async fn list_all_models(
    verbose: bool,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::OllamaClient;

    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    let ollama_client = OllamaClient::new().ok();
    let ollama_models: Vec<_> = ollama_client
        .as_ref()
        .map(|c| c.list_cached())
        .unwrap_or_default();

    let mullama_dir = super::common::mullama_models_dir();
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
                            .map(chrono::DateTime::<chrono::Utc>::from)
                            .unwrap_or_else(|_| chrono::Utc::now());
                        custom_models.push((name, path, size, modified));
                    }
                }
            }
        }
    }

    if json_output {
        let mut models_json = Vec::new();

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
                "name": format!("{}:{}",
                    model.repo_id.replace('/', "-"),
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

    for model in &ollama_models {
        let name = format!("{}:{}", model.name, model.tag);
        let modified = chrono::DateTime::parse_from_rfc3339(&model.pulled_at)
            .map(|dt| format_time_ago(&dt.with_timezone(&chrono::Utc)))
            .unwrap_or_else(|_| model.pulled_at.clone());

        println!(
            "{:<42} {:>10} {}",
            truncate_display(&name, 40),
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

    for model in &cached {
        let name = cached_model_short_name(model);
        let modified = chrono::DateTime::parse_from_rfc3339(&model.downloaded_at)
            .map(|dt| format_time_ago(&dt.with_timezone(&chrono::Utc)))
            .unwrap_or_else(|_| model.downloaded_at.clone());

        println!(
            "{:<42} {:>10} {}",
            truncate_display(&name, 40),
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

    for (name, path, size, modified) in &custom_models {
        println!(
            "{:<42} {:>10} {}",
            truncate_display(name, 40),
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

pub(crate) async fn remove_model(
    name: &str,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    let found = if PathBuf::from(name).exists() {
        Some(("path".to_string(), PathBuf::from(name)))
    } else if let Some(model) = find_cached_model(&cached, name) {
        Some((model.repo_id.clone(), model.local_path.clone()))
    } else {
        let model_path = local_model_path(name);
        model_path
            .exists()
            .then_some(("local".to_string(), model_path))
    };

    let (source, path) = match found {
        Some(found) => found,
        None => {
            eprintln!("Model '{}' not found.", name);
            eprintln!();
            eprintln!("Use 'mullama list' to see available models.");
            return Err("Model not found".into());
        }
    };

    if !force {
        println!("This will permanently delete:");
        println!("  {}", path.display());
        print!("Are you sure? [y/N] ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    if source == "path" || source == "local" {
        std::fs::remove_file(&path)?;
    } else {
        let model = cached
            .iter()
            .find(|m| m.local_path == path)
            .ok_or("Cached model not found")?;
        downloader.remove_cached(&model.repo_id, &model.filename)?;
    }

    println!("Deleted '{}'", name);
    Ok(())
}

pub(crate) async fn show_running_models(
    socket: &str,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = connect(socket)?;
    let models = client.list_models()?;

    if json_output {
        let mut models_json = Vec::new();
        for model in &models {
            models_json.push(serde_json::json!({
                "alias": model.alias,
                "path": model.info.path,
                "parameters": model.info.parameters,
                "context_size": model.info.context_size,
                "gpu_layers": model.info.gpu_layers,
                "is_default": model.is_default,
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
            truncate_display(&name, 20),
            size,
            gpu,
            model.info.context_size,
            active
        );
    }

    println!();
    println!("* = default model");

    Ok(())
}

pub(crate) async fn show_model_details(
    name: &str,
    show_modelfile: bool,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;
    let cached = downloader.list_cached();

    let model = find_cached_model(&cached, name).ok_or_else(|| {
        eprintln!("Model '{}' not found.", name);
        eprintln!();
        eprintln!("Use 'mullama list' to see available models.");
        "Model not found"
    })?;

    if json_output {
        let info = serde_json::json!({
            "name": format!("{}:{}",
                model.repo_id.replace('/', "-"),
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
        let modelfile_path = model.local_path.with_extension("modelfile");
        let mullamafile_path = model.local_path.with_extension("mullamafile");

        if modelfile_path.exists() {
            println!("{}", std::fs::read_to_string(&modelfile_path)?);
        } else if mullamafile_path.exists() {
            println!("{}", std::fs::read_to_string(&mullamafile_path)?);
        } else {
            print_default_modelfile(&model.local_path, &model.filename);
        }
        return Ok(());
    }

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
