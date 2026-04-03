pub(super) fn format_size(bytes: u64) -> String {
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

pub(super) fn model_config_from_modelfile(
    modelfile: &crate::modelfile::Modelfile,
) -> crate::daemon::models::ModelConfig {
    let mut stop_sequences = modelfile.stop_sequences.clone();
    if stop_sequences.is_empty() {
        if let Some(stop) = modelfile.stop() {
            stop_sequences = stop;
        }
    }

    crate::daemon::models::ModelConfig {
        stop_sequences,
        system_prompt: modelfile.system.clone(),
        temperature: modelfile.temperature().map(|v| v as f32),
        top_p: modelfile.top_p().map(|v| v as f32),
        top_k: modelfile.top_k().and_then(|v| i32::try_from(v).ok()),
        context_size: modelfile.num_ctx().and_then(|v| u32::try_from(v).ok()),
    }
}
