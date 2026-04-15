use mullama::daemon::{resolve_model_name, GgufFileInfo, HfDownloader, HfModelSpec, ResolvedModel};

use crate::shared::format_size;

pub(crate) async fn pull_model(
    spec: &str,
    show_progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mullama::daemon::OllamaClient;

    let resolved = resolve_model_name(spec);

    match resolved {
        ResolvedModel::Ollama { name, tag } => {
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
            pull_from_huggingface(&hf_spec, show_progress).await
        }
        ResolvedModel::LocalPath(path) => Err(format!(
            "'{}' is a local path, not a downloadable model",
            path.display()
        )
        .into()),
        ResolvedModel::Unknown(_) => pull_from_huggingface(spec, show_progress).await,
    }
}

pub(crate) async fn pull_from_huggingface(
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

pub(crate) async fn search_models(
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
        print!("{}. ", i + 1);
        print!("{}", result.id);
        if result.is_gguf() {
            print!(" [GGUF]");
        }
        println!();

        print!("   ");
        print!("Downloads: {}", result.downloads_formatted());
        if let Some(likes) = result.likes {
            print!(" | Likes: {}", likes);
        }
        if let Some(ref pipeline) = result.pipeline_tag {
            print!(" | {}", pipeline);
        }
        println!();

        println!("   Use: mullama serve --model hf:{}", result.id);

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

pub(crate) async fn show_repo_info(repo_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let downloader = HfDownloader::new()?;

    println!("Fetching info for {}...\n", repo_id);

    let files = downloader.list_gguf_files(repo_id).await?;

    println!("Repository: {}", repo_id);
    println!("URL: https://huggingface.co/{}", repo_id);
    println!();
    println!("Available GGUF files ({}):", files.len());
    println!();

    let mut by_quant: std::collections::HashMap<String, Vec<&GgufFileInfo>> =
        std::collections::HashMap::new();
    for file in &files {
        let key = file
            .quantization
            .clone()
            .unwrap_or_else(|| "Other".to_string());
        by_quant.entry(key).or_default().push(file);
    }

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
