use std::path::PathBuf;

use mullama::daemon::{resolve_model_name, resolve_model_path, HfDownloader, ResolvedModel};
use mullama::modelfile::{find_modelfile, Modelfile, ModelfileParser};

use super::common::{find_cached_model, mullama_models_dir};

pub(crate) async fn create_model(
    name: &str,
    file: Option<PathBuf>,
    download: bool,
    show_progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let modelfile_path = if let Some(path) = file {
        if !path.exists() {
            return Err(format!("Modelfile not found: {}", path.display()).into());
        }
        path
    } else {
        find_modelfile(".").ok_or("No Modelfile or Mullamafile found in current directory")?
    };

    println!("Reading {}...", modelfile_path.display());

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

    let base_model_path = resolve_base_model(&modelfile.from, download, show_progress).await?;

    println!();
    println!("Base model: {}", base_model_path.display());

    let mullama_dir = mullama_models_dir();
    std::fs::create_dir_all(&mullama_dir)?;

    let model_dir = mullama_dir.join(name);
    std::fs::create_dir_all(&model_dir)?;

    let mullamafile_dest = model_dir.join("Mullamafile");
    let mut saved_modelfile = modelfile.clone();
    saved_modelfile.from = base_model_path.display().to_string();
    saved_modelfile.save(&mullamafile_dest)?;

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

    std::fs::write(
        model_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?,
    )?;

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

pub(crate) async fn resolve_base_model(
    from: &str,
    download: bool,
    show_progress: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = PathBuf::from(from);
    if path.exists() {
        return Ok(path);
    }

    if from.starts_with("hf:") {
        if !download {
            return Err("Base model is HuggingFace spec but --download=false".into());
        }

        println!();
        println!("Downloading base model from HuggingFace...");
        let (_, resolved_path) = resolve_model_path(from, show_progress).await?;
        return Ok(resolved_path);
    }

    let resolved = resolve_model_name(from);
    match resolved {
        ResolvedModel::LocalPath(path) => {
            if path.exists() {
                Ok(path)
            } else {
                Err(format!("Local model not found: {}", path.display()).into())
            }
        }
        ResolvedModel::HuggingFace { spec, .. } => {
            if !download {
                return Err(format!(
                    "Model '{}' needs to be downloaded. Use --download=true or run 'mullama pull {}'",
                    from, from
                )
                .into());
            }

            println!();
            println!("Downloading '{}' from HuggingFace...", from);
            let (_, resolved_path) = resolve_model_path(&spec, show_progress).await?;
            Ok(resolved_path)
        }
        ResolvedModel::Ollama { name, tag } => {
            use mullama::daemon::OllamaClient;

            let model_name = format!("{}:{}", name, tag);
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
                )
                .into())
            }
        }
        ResolvedModel::Unknown(name) => {
            let downloader = HfDownloader::new()?;
            let cached = downloader.list_cached();

            if let Some(model) = find_cached_model(&cached, &name) {
                Ok(model.local_path.clone())
            } else {
                Err(format!(
                    "Unknown model '{}'. Use a local path, HF spec (hf:owner/repo), or a known alias.",
                    name
                )
                .into())
            }
        }
    }
}

pub(crate) async fn copy_model(
    source: &str,
    destination: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mullama_dir = mullama_models_dir();
    let source_dir = mullama_dir.join(source);

    if !source_dir.exists() {
        let downloader = HfDownloader::new()?;
        let cached = downloader.list_cached();

        if let Some(model) = find_cached_model(&cached, source) {
            let dest_dir = mullama_dir.join(destination);
            std::fs::create_dir_all(&dest_dir)?;

            let modelfile = Modelfile::from_model(model.local_path.display().to_string());
            modelfile.save(dest_dir.join("Mullamafile"))?;

            let model_link = dest_dir.join("model.gguf");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&model.local_path, &model_link)?;
            #[cfg(windows)]
            std::fs::copy(&model.local_path, &model_link)?;

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

    let dest_dir = mullama_dir.join(destination);

    if dest_dir.exists() {
        return Err(format!("Destination '{}' already exists", destination).into());
    }

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
