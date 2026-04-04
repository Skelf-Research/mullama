use std::path::{Path, PathBuf};

use mullama::hf::CachedModel;

pub(super) fn mullama_models_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".mullama").join("models"))
        .unwrap_or_else(|| PathBuf::from(".mullama/models"))
}

pub(super) fn local_model_path(name: &str) -> PathBuf {
    mullama_models_dir().join(format!("{}.gguf", name))
}

pub(super) fn cached_model_short_name(model: &CachedModel) -> String {
    format!(
        "{}:{}",
        model.repo_id.split('/').next_back().unwrap_or(&model.repo_id),
        model.filename.trim_end_matches(".gguf")
    )
}

pub(super) fn find_cached_model<'a>(
    cached: &'a [CachedModel],
    name: &str,
) -> Option<&'a CachedModel> {
    cached.iter().find(|model| {
        model.filename == name
            || model.repo_id == name
            || cached_model_short_name(model) == name
            || model.filename.trim_end_matches(".gguf") == name
    })
}

pub(super) fn truncate_display(name: &str, max_len: usize) -> String {
    if name.len() > max_len {
        format!("{}...", &name[..max_len - 3])
    } else {
        name.to_string()
    }
}

pub(super) fn print_default_modelfile(path: &Path, filename: &str) {
    println!("# Modelfile for {}", filename);
    println!("# Auto-generated - no custom Modelfile found");
    println!();
    println!("FROM {}", path.display());
    println!();
    println!("PARAMETER temperature 0.7");
    println!("PARAMETER top_p 0.9");
    println!("PARAMETER num_ctx 4096");
}
