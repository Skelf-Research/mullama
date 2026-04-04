//! Hugging Face Hub integration for model discovery and downloading
//!
//! This module provides comprehensive integration with Hugging Face Hub for:
//! - Searching and discovering GGUF models
//! - Listing available quantizations for models
//! - Downloading models with progress tracking
//! - Basic model validation and testing after download

mod client;
mod types;

use crate::error::MullamaError;
use std::path::Path;

/// Hugging Face Hub API base URL
const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_MODELS_BASE: &str = "https://huggingface.co";

pub use client::HFClient;
pub use types::{
    DownloadProgress, GGUFFile, HFModelInfo, ModelSearchFilters, ModelTestResult,
    ProgressCallback, QuantizationType,
};

/// URL encoding helper
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut result = String::new();
        for c in s.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                    result.push(c);
                }
                ' ' => result.push_str("%20"),
                _ => {
                    for b in c.to_string().bytes() {
                        result.push_str(&format!("%{:02X}", b));
                    }
                }
            }
        }
        result
    }
}

/// Convenience functions for quick operations
pub mod quick {
    use super::*;

    /// Download the best quantization of a model for given VRAM
    pub fn download_best_for_vram(
        model_id: &str,
        vram_mb: u64,
        download_dir: Option<&Path>,
    ) -> Result<PathBuf, MullamaError> {
        let client = if let Some(dir) = download_dir {
            HFClient::with_download_dir(dir).with_token_from_env()
        } else {
            HFClient::new().with_token_from_env()
        };

        let gguf_files = client.list_gguf_files(model_id)?;

        let best_file = gguf_files
            .iter()
            .filter(|f| f.estimated_vram_mb() <= vram_mb)
            .max_by_key(|f| f.quantization.quality_rating())
            .ok_or_else(|| {
                MullamaError::HuggingFaceError(format!(
                    "No suitable quantization found for {} MB VRAM",
                    vram_mb
                ))
            })?;

        client.download_gguf(model_id, best_file, None)
    }

    /// Download the smallest quantization of a model
    pub fn download_smallest(
        model_id: &str,
        download_dir: Option<&Path>,
    ) -> Result<PathBuf, MullamaError> {
        let client = if let Some(dir) = download_dir {
            HFClient::with_download_dir(dir).with_token_from_env()
        } else {
            HFClient::new().with_token_from_env()
        };

        let gguf_files = client.list_gguf_files(model_id)?;

        let smallest = gguf_files
            .iter()
            .min_by_key(|f| f.size)
            .ok_or_else(|| MullamaError::HuggingFaceError("No GGUF files found".to_string()))?;

        client.download_gguf(model_id, smallest, None)
    }

    /// Search for GGUF models by name
    pub fn search_gguf(query: &str, limit: usize) -> Result<Vec<HFModelInfo>, MullamaError> {
        let client = HFClient::new();
        let filters = ModelSearchFilters::new()
            .with_query(query)
            .gguf_only()
            .sort_by_downloads()
            .with_limit(limit);

        client.search_models(&filters)
    }

    /// Get popular GGUF models
    pub fn popular_models(limit: usize) -> Result<Vec<HFModelInfo>, MullamaError> {
        let client = HFClient::new();
        client.get_popular_gguf_models(limit)
    }

    /// Download and test a model
    pub fn download_and_test(
        model_id: &str,
        quantization: Option<QuantizationType>,
    ) -> Result<(PathBuf, ModelTestResult), MullamaError> {
        let client = HFClient::new().with_token_from_env();

        let gguf_files = client.list_gguf_files(model_id)?;

        let file = if let Some(quant) = quantization {
            gguf_files
                .iter()
                .find(|f| f.quantization == quant)
                .or_else(|| gguf_files.first())
        } else {
            // Default to Q4_K_M or similar
            gguf_files
                .iter()
                .find(|f| matches!(f.quantization, QuantizationType::Q4_K_M))
                .or_else(|| {
                    gguf_files
                        .iter()
                        .find(|f| matches!(f.quantization, QuantizationType::Q4_0))
                })
                .or_else(|| gguf_files.first())
        };

        let file = file.ok_or_else(|| {
            MullamaError::HuggingFaceError("No suitable GGUF file found".to_string())
        })?;

        let path = client.download_gguf(model_id, file, None)?;
        let test_result = client.test_model(&path)?;

        Ok((path, test_result))
    }

    use std::path::PathBuf;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_from_filename() {
        assert_eq!(
            QuantizationType::from_filename("model-q4_k_m.gguf"),
            QuantizationType::Q4_K_M
        );
        assert_eq!(
            QuantizationType::from_filename("llama-7b-Q8_0.gguf"),
            QuantizationType::Q8_0
        );
        assert_eq!(
            QuantizationType::from_filename("model-f16.gguf"),
            QuantizationType::F16
        );
    }

    #[test]
    fn test_quantization_quality() {
        assert!(QuantizationType::F16.quality_rating() > QuantizationType::Q4_K_M.quality_rating());
        assert!(
            QuantizationType::Q4_K_M.quality_rating() > QuantizationType::Q2_K.quality_rating()
        );
    }

    #[test]
    fn test_gguf_file_size_human() {
        let file = GGUFFile {
            filename: "test.gguf".to_string(),
            size: 4 * 1024 * 1024 * 1024, // 4 GB
            quantization: QuantizationType::Q4_K_M,
            download_url: String::new(),
            sha256: None,
        };

        assert!(file.size_human().contains("GB"));
    }

    #[test]
    fn test_search_filters_builder() {
        let filters = ModelSearchFilters::new()
            .with_query("llama")
            .gguf_only()
            .sort_by_downloads()
            .with_limit(10);

        assert_eq!(filters.query, Some("llama".to_string()));
        assert!(filters.gguf_only);
        assert_eq!(filters.limit, Some(10));
    }

    #[test]
    fn test_progress_percentage() {
        let progress = DownloadProgress {
            downloaded: 50,
            total: 100,
            speed_bps: 1000,
            eta_seconds: 50,
            filename: "test.gguf".to_string(),
        };

        assert_eq!(progress.percentage(), 50.0);
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoding::encode("hello world"), "hello%20world");
        assert_eq!(urlencoding::encode("test-123"), "test-123");
    }
}

/// Integration tests that require network access
/// Run with: cargo test --features full -- --ignored --nocapture
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Downloads the smallest available SLM (SmolLM2-135M) and tests it
    ///
    /// This test:
    /// 1. Lists available GGUF files for SmolLM2-135M (one of the smallest LLMs)
    /// 2. Downloads the smallest quantization
    /// 3. Loads and tests the model
    /// 4. Generates a few tokens to verify it works
    ///
    /// Run with: cargo test test_download_smallest_slm -- --ignored --nocapture
    #[test]
    #[ignore] // Ignored by default since it downloads files
    fn test_download_smallest_slm() {
        println!("\n=== Testing Smallest SLM Download ===\n");

        // SmolLM2-135M is one of the smallest capable LLMs (~70-270MB depending on quant)
        // Alternative tiny models:
        // - "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" (~400MB-2GB)
        // - "Qwen/Qwen2.5-0.5B-Instruct-GGUF" (~300MB-1GB)
        let model_id = "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF";

        let client = HFClient::new().with_token_from_env();

        // Step 1: List available GGUF files
        println!("Step 1: Listing GGUF files for {}...", model_id);
        let gguf_files = match client.list_gguf_files(model_id) {
            Ok(files) => files,
            Err(_e) => {
                // Try alternative model if SmolLM2 not found
                println!("SmolLM2 not found, trying TinyLlama...");
                let alt_model = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
                client
                    .list_gguf_files(alt_model)
                    .expect("Failed to list GGUF files for alternative model")
            }
        };

        println!("\nFound {} GGUF files:", gguf_files.len());
        for file in &gguf_files {
            println!(
                "  - {} ({}) [{}]",
                file.filename,
                file.size_human(),
                file.quantization
            );
        }

        // Step 2: Find and download the smallest file
        let smallest = gguf_files
            .iter()
            .min_by_key(|f| f.size)
            .expect("No GGUF files found");

        println!(
            "\nStep 2: Downloading smallest file: {} ({})",
            smallest.filename,
            smallest.size_human()
        );

        let download_start = std::time::Instant::now();
        let model_path = client
            .download_gguf(
                model_id,
                smallest,
                Some(Box::new(|progress| {
                    print!(
                        "\r  Progress: {:.1}% ({}) - ETA: {}     ",
                        progress.percentage(),
                        progress.speed_human(),
                        progress.eta_human()
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                })),
            )
            .expect("Failed to download model");

        let download_time = download_start.elapsed();
        println!("\n  Downloaded to: {:?}", model_path);
        println!("  Download time: {:.2}s", download_time.as_secs_f64());

        // Verify the file exists and has content
        assert!(model_path.exists(), "Downloaded file should exist");
        let file_size = std::fs::metadata(&model_path)
            .expect("Failed to get file metadata")
            .len();
        assert!(file_size > 0, "Downloaded file should not be empty");
        println!("  File size: {} bytes", file_size);

        // Step 3: Test the model
        println!("\nStep 3: Testing the model...");
        let test_result = client
            .test_model(&model_path)
            .expect("Failed to test model");

        println!("\n=== Test Results ===");
        println!("  Load successful: {}", test_result.load_success);
        println!("  Load time: {}ms", test_result.load_time_ms);
        println!("  Parameters: {}", format_params(test_result.n_params));
        println!("  Context size: {}", test_result.n_ctx);
        println!("  Embedding dim: {}", test_result.n_embd);
        println!("  Layers: {}", test_result.n_layers);
        println!("  Vocab size: {}", test_result.vocab_size);
        println!("  Tokenization works: {}", test_result.tokenization_works);
        println!("  Generation works: {}", test_result.generation_works);

        if let Some(ref output) = test_result.sample_output {
            println!("\n  Sample output: \"{}\"", output);
        }

        if let Some(ref error) = test_result.error {
            println!("\n  Error: {}", error);
        }

        // Assertions
        assert!(test_result.load_success, "Model should load successfully");
        assert!(test_result.tokenization_works, "Tokenization should work");

        println!("\n=== Test Passed! ===\n");
    }

    /// Test searching for small language models
    #[test]
    #[ignore]
    fn test_search_small_models() {
        println!("\n=== Searching for Small Language Models ===\n");

        let client = HFClient::new();

        // Search for small/tiny models
        let filters = ModelSearchFilters::new()
            .with_query("tiny llama GGUF")
            .gguf_only()
            .sort_by_downloads()
            .with_limit(5);

        let models = client
            .search_models(&filters)
            .expect("Failed to search models");

        println!("Found {} models:\n", models.len());
        for model in &models {
            println!("  {} ({} downloads)", model.model_id, model.downloads);
            if let Some(ref desc) = model.description {
                let short_desc: String = desc.chars().take(80).collect();
                println!("    {}", short_desc);
            }
        }

        assert!(!models.is_empty(), "Should find at least one model");
    }

    /// Test listing popular GGUF models
    #[test]
    #[ignore]
    fn test_popular_gguf_models() {
        println!("\n=== Popular GGUF Models ===\n");

        let client = HFClient::new();
        let models = client
            .get_popular_gguf_models(10)
            .expect("Failed to get popular models");

        println!("Top {} GGUF models by downloads:\n", models.len());
        for (i, model) in models.iter().enumerate() {
            println!(
                "  {}. {} - {} downloads",
                i + 1,
                model.model_id,
                model.downloads
            );
        }

        assert!(!models.is_empty(), "Should find popular models");
    }

    /// Quick download and test helper
    #[test]
    #[ignore]
    fn test_quick_download_and_test() {
        println!("\n=== Quick Download and Test ===\n");

        // Use the quick API to download and test
        let result = quick::download_and_test(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            Some(QuantizationType::Q4_K_M),
        );

        match result {
            Ok((path, test_result)) => {
                println!("Downloaded to: {:?}", path);
                println!("Load time: {}ms", test_result.load_time_ms);
                println!("Generation works: {}", test_result.generation_works);

                if let Some(output) = test_result.sample_output {
                    println!("Sample: {}", output);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
                // Don't fail the test if network issues
            }
        }
    }

    /// Helper to format parameter count
    fn format_params(n: u64) -> String {
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", n as f64 / 1_000.0)
        } else {
            format!("{}", n)
        }
    }
}
