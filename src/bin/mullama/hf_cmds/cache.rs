use std::io::{self, Write};

use mullama::daemon::HfDownloader;

use crate::CacheAction;

pub(crate) async fn handle_cache_action(
    action: CacheAction,
) -> Result<(), Box<dyn std::error::Error>> {
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
