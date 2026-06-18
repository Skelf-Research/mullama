//! # Configuration Management Example
//!
//! This example demonstrates comprehensive configuration management in Mullama
//! using serde for serialization, environment variables, and validation.
//!
//! Run with: cargo run --example configuration_management

use mullama::config::presets;
use mullama::prelude::*;
use serde_json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("📋 Configuration Management Example");
    println!("===================================");

    // Example 1: Basic configuration creation
    demonstrate_basic_configuration().await?;

    // Example 2: Configuration serialization
    demonstrate_serialization().await?;

    // Example 3: Environment variable loading
    demonstrate_environment_config().await?;

    // Example 4: Configuration validation
    demonstrate_validation().await?;

    // Example 5: Configuration presets
    demonstrate_presets().await?;

    // Example 6: Configuration merging
    demonstrate_merging().await?;

    // Example 7: Advanced configuration patterns
    demonstrate_advanced_patterns().await?;

    println!("\n✨ All configuration patterns demonstrated successfully!");
    Ok(())
}

async fn demonstrate_basic_configuration() -> Result<(), MullamaError> {
    println!("\n🏗️ Basic Configuration");
    println!("======================");

    // Create a default configuration
    let default_config = MullamaConfig::default();
    println!("1️⃣ Default configuration created");
    println!("   Model path: {:?}", default_config.model.path);
    println!("   Context size: {}", default_config.context.n_ctx);
    println!("   Temperature: {}", default_config.sampling.temperature);

    // Create a custom configuration
    let custom_config = MullamaConfig {
        model: mullama::config::ModelConfig {
            path: "custom/model/path.gguf".to_string(),
            gpu_layers: 24,
            context_size: 4096,
            use_mmap: true,
            use_mlock: false,
            check_tensors: true,
            vocab_only: false,
            kv_overrides: HashMap::new(),
        },
        context: mullama::config::ContextConfig {
            n_ctx: 4096,
            n_batch: 1024,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads: 8,
            n_threads_batch: 4,
            embeddings: false,
            flash_attn: true,
            offload_kqv: true,
            type_k: mullama::KvCacheType::default(),
            type_v: mullama::KvCacheType::default(),
        },
        sampling: mullama::config::SamplingConfig {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            repeat_last_n: 64,
            seed: 0,
            token_penalties: HashMap::new(),
        },
        performance: mullama::config::PerformanceConfig {
            enable_monitoring: true,
            memory_optimization: 2,
            cpu_optimizations: mullama::config::CpuOptimizations {
                enable_simd: true,
                enable_threading: true,
                thread_affinity: None,
            },
            gpu_optimizations: mullama::config::GpuOptimizations {
                enable_gpu: true,
                device_id: 0,
                memory_pool_size: Some(2048),
                optimize_memory: true,
            },
        },
        logging: mullama::config::LoggingConfig {
            level: "debug".to_string(),
            performance: true,
            file: Some(std::path::PathBuf::from("mullama.log")),
            structured: true,
        },
        metadata: {
            let mut metadata = HashMap::new();
            metadata.insert(
                "version".to_string(),
                serde_json::Value::String("1.0.0".to_string()),
            );
            metadata.insert(
                "author".to_string(),
                serde_json::Value::String("user".to_string()),
            );
            metadata
        },
    };

    println!("\n2️⃣ Custom configuration created");
    println!("   Model path: {}", custom_config.model.path);
    println!("   GPU layers: {}", custom_config.model.gpu_layers);
    println!("   Context size: {}", custom_config.context.n_ctx);
    println!("   Flash attention: {}", custom_config.context.flash_attn);
    println!("   Temperature: {}", custom_config.sampling.temperature);
    println!(
        "   Performance monitoring: {}",
        custom_config.performance.enable_monitoring
    );

    Ok(())
}

async fn demonstrate_serialization() -> Result<(), MullamaError> {
    println!("\n💾 Configuration Serialization");
    println!("==============================");

    let config = MullamaConfig {
        model: mullama::config::ModelConfig {
            path: "example/model.gguf".to_string(),
            gpu_layers: 16,
            context_size: 2048,
            ..Default::default()
        },
        sampling: mullama::config::SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        },
        ..Default::default()
    };

    // Serialize to JSON
    println!("1️⃣ JSON serialization:");
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| MullamaError::ConfigError(format!("JSON serialization failed: {}", e)))?;

    println!("📄 Configuration as JSON:");
    println!("{}", json);

    // Deserialize from JSON
    println!("\n2️⃣ JSON deserialization:");
    let deserialized_config: MullamaConfig = serde_json::from_str(&json)
        .map_err(|e| MullamaError::ConfigError(format!("JSON deserialization failed: {}", e)))?;

    println!("✅ Successfully deserialized configuration");
    println!("   Model path: {}", deserialized_config.model.path);
    println!("   GPU layers: {}", deserialized_config.model.gpu_layers);

    // Example of saving/loading to/from file (simulated)
    println!("\n3️⃣ File operations (simulated):");
    // config.to_json_file("config.json")?;
    // let loaded_config = MullamaConfig::from_json_file("config.json")?;
    println!("   ✅ Configuration would be saved to config.json");
    println!("   ✅ Configuration would be loaded from config.json");

    Ok(())
}

async fn demonstrate_environment_config() -> Result<(), MullamaError> {
    println!("\n🌍 Environment Variable Configuration");
    println!("====================================");

    // Set some example environment variables
    std::env::set_var("MULLAMA_MODEL_PATH", "/path/to/model.gguf");
    std::env::set_var("MULLAMA_MODEL_GPU_LAYERS", "32");
    std::env::set_var("MULLAMA_CONTEXT_N_CTX", "4096");
    std::env::set_var("MULLAMA_SAMPLING_TEMPERATURE", "0.8");
    std::env::set_var("MULLAMA_SAMPLING_TOP_K", "50");

    println!("1️⃣ Environment variables set:");
    println!("   MULLAMA_MODEL_PATH=/path/to/model.gguf");
    println!("   MULLAMA_MODEL_GPU_LAYERS=32");
    println!("   MULLAMA_CONTEXT_N_CTX=4096");
    println!("   MULLAMA_SAMPLING_TEMPERATURE=0.8");
    println!("   MULLAMA_SAMPLING_TOP_K=50");

    // Load configuration from environment
    let env_config = MullamaConfig::from_env()?;
    println!("\n2️⃣ Configuration loaded from environment:");
    println!("   Model path: {}", env_config.model.path);
    println!("   GPU layers: {}", env_config.model.gpu_layers);
    println!("   Context size: {}", env_config.context.n_ctx);
    println!("   Temperature: {}", env_config.sampling.temperature);
    println!("   Top-k: {}", env_config.sampling.top_k);

    // Demonstrate environment variable precedence
    println!("\n3️⃣ Environment variable patterns:");
    println!("   📝 Naming convention: MULLAMA_<SECTION>_<FIELD>");
    println!("   📝 Examples:");
    println!("      - MULLAMA_MODEL_PATH");
    println!("      - MULLAMA_SAMPLING_TEMPERATURE");
    println!("      - MULLAMA_CONTEXT_N_THREADS");
    println!("      - MULLAMA_PERFORMANCE_ENABLE_MONITORING");

    // Clean up environment variables
    std::env::remove_var("MULLAMA_MODEL_PATH");
    std::env::remove_var("MULLAMA_MODEL_GPU_LAYERS");
    std::env::remove_var("MULLAMA_CONTEXT_N_CTX");
    std::env::remove_var("MULLAMA_SAMPLING_TEMPERATURE");
    std::env::remove_var("MULLAMA_SAMPLING_TOP_K");

    Ok(())
}

async fn demonstrate_validation() -> Result<(), MullamaError> {
    println!("\n✅ Configuration Validation");
    println!("===========================");

    // Valid configuration
    println!("1️⃣ Valid configuration:");
    let valid_config = MullamaConfig {
        model: mullama::config::ModelConfig {
            path: "valid/model.gguf".to_string(),
            gpu_layers: 16,
            context_size: 2048,
            ..Default::default()
        },
        sampling: mullama::config::SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            ..Default::default()
        },
        ..Default::default()
    };

    match valid_config.validate() {
        Ok(()) => println!("   ✅ Configuration is valid"),
        Err(e) => println!("   ❌ Validation failed: {}", e),
    }

    // Invalid configurations
    println!("\n2️⃣ Invalid configurations:");

    // Empty model path
    let mut invalid_config = valid_config.clone();
    invalid_config.model.path = String::new();
    match invalid_config.validate() {
        Ok(()) => println!("   ⚠️  Should have failed: empty model path"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }

    // Negative temperature
    let mut invalid_config = valid_config.clone();
    invalid_config.sampling.temperature = -0.5;
    match invalid_config.validate() {
        Ok(()) => println!("   ⚠️  Should have failed: negative temperature"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }

    // Invalid top-p
    let mut invalid_config = valid_config.clone();
    invalid_config.sampling.top_p = 1.5;
    match invalid_config.validate() {
        Ok(()) => println!("   ⚠️  Should have failed: top-p > 1.0"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }

    // Zero context size
    let mut invalid_config = valid_config.clone();
    invalid_config.context.n_ctx = 0;
    match invalid_config.validate() {
        Ok(()) => println!("   ⚠️  Should have failed: zero context size"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }

    println!("\n3️⃣ Validation rules:");
    println!("   📏 Model path cannot be empty");
    println!("   📏 Context size must be > 0");
    println!("   📏 Batch size must be > 0");
    println!("   📏 Temperature must be >= 0");
    println!("   📏 Top-p must be between 0 and 1");
    println!("   📏 Repeat penalty must be > 0");
    println!("   📏 GPU layers cannot be negative");

    Ok(())
}

async fn demonstrate_presets() -> Result<(), MullamaError> {
    println!("\n🎯 Configuration Presets");
    println!("========================");

    // Creative writing preset
    println!("1️⃣ Creative writing preset:");
    let creative_config = presets::creative_writing();
    println!(
        "   🎨 Temperature: {}",
        creative_config.sampling.temperature
    );
    println!("   🎨 Top-k: {}", creative_config.sampling.top_k);
    println!("   🎨 Top-p: {}", creative_config.sampling.top_p);
    println!(
        "   🎨 Repeat penalty: {}",
        creative_config.sampling.repeat_penalty
    );

    // Code generation preset
    println!("\n2️⃣ Code generation preset:");
    let code_config = presets::code_generation();
    println!("   💻 Temperature: {}", code_config.sampling.temperature);
    println!("   💻 Top-k: {}", code_config.sampling.top_k);
    println!("   💻 Top-p: {}", code_config.sampling.top_p);
    println!(
        "   💻 Repeat penalty: {}",
        code_config.sampling.repeat_penalty
    );

    // Question answering preset
    println!("\n3️⃣ Question answering preset:");
    let qa_config = presets::question_answering();
    println!("   ❓ Temperature: {}", qa_config.sampling.temperature);
    println!("   ❓ Top-k: {}", qa_config.sampling.top_k);
    println!("   ❓ Top-p: {}", qa_config.sampling.top_p);

    // Chatbot preset
    println!("\n4️⃣ Chatbot preset:");
    let chatbot_config = presets::chatbot();
    println!("   💬 Temperature: {}", chatbot_config.sampling.temperature);
    println!("   💬 Context size: {}", chatbot_config.context.n_ctx);

    // Performance optimized preset
    println!("\n5️⃣ Performance optimized preset:");
    let perf_config = presets::performance_optimized();
    println!("   ⚡ Batch size: {}", perf_config.context.n_batch);
    println!("   ⚡ Flash attention: {}", perf_config.context.flash_attn);
    println!(
        "   ⚡ Memory optimization: {}",
        perf_config.performance.memory_optimization
    );

    // Memory optimized preset
    println!("\n6️⃣ Memory optimized preset:");
    let mem_config = presets::memory_optimized();
    println!("   💾 Context size: {}", mem_config.context.n_ctx);
    println!("   💾 Batch size: {}", mem_config.context.n_batch);
    println!(
        "   💾 Memory optimization: {}",
        mem_config.performance.memory_optimization
    );

    Ok(())
}

async fn demonstrate_merging() -> Result<(), MullamaError> {
    println!("\n🔀 Configuration Merging");
    println!("========================");

    // Base configuration
    let mut base_config = presets::chatbot();
    println!("1️⃣ Base configuration (chatbot preset):");
    println!("   Model path: {:?}", base_config.model.path);
    println!("   Temperature: {}", base_config.sampling.temperature);
    println!("   Context size: {}", base_config.context.n_ctx);

    // Override configuration
    let override_config = MullamaConfig {
        model: mullama::config::ModelConfig {
            path: "specialized/model.gguf".to_string(),
            gpu_layers: 40,
            ..Default::default()
        },
        sampling: mullama::config::SamplingConfig {
            temperature: 0.9,
            top_k: 60,
            ..Default::default()
        },
        metadata: {
            let mut metadata = HashMap::new();
            metadata.insert(
                "environment".to_string(),
                serde_json::Value::String("production".to_string()),
            );
            metadata
        },
        ..Default::default()
    };

    println!("\n2️⃣ Override configuration:");
    println!("   Model path: {}", override_config.model.path);
    println!("   GPU layers: {}", override_config.model.gpu_layers);
    println!("   Temperature: {}", override_config.sampling.temperature);

    // Merge configurations
    base_config.merge(override_config);

    println!("\n3️⃣ Merged configuration:");
    println!("   Model path: {}", base_config.model.path);
    println!("   GPU layers: {}", base_config.model.gpu_layers);
    println!("   Temperature: {}", base_config.sampling.temperature);
    println!(
        "   Context size: {} (preserved from base)",
        base_config.context.n_ctx
    );
    println!("   Metadata entries: {}", base_config.metadata.len());

    Ok(())
}

async fn demonstrate_advanced_patterns() -> Result<(), MullamaError> {
    println!("\n🚀 Advanced Configuration Patterns");
    println!("==================================");

    // Pattern 1: Conditional configuration
    println!("1️⃣ Conditional configuration:");
    let is_production =
        std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()) == "production";

    let conditional_config = if is_production {
        println!("   🏭 Production environment detected");
        let mut config = presets::performance_optimized();
        config.logging.level = "warn".to_string();
        config.performance.enable_monitoring = true;
        config
    } else {
        println!("   🛠️  Development environment detected");
        let mut config = presets::chatbot();
        config.logging.level = "debug".to_string();
        config.performance.enable_monitoring = false;
        config
    };

    println!("   Log level: {}", conditional_config.logging.level);
    println!(
        "   Monitoring: {}",
        conditional_config.performance.enable_monitoring
    );

    // Pattern 2: Configuration hierarchy
    println!("\n2️⃣ Configuration hierarchy:");
    let mut hierarchical_config = MullamaConfig::default();

    // Load from file (simulated)
    println!("   📄 Loading base config from file...");

    // Override with environment variables
    if let Ok(env_config) = MullamaConfig::from_env() {
        println!("   🌍 Merging environment variables...");
        hierarchical_config.merge(env_config);
    }

    // Override with command line arguments (simulated)
    println!("   ⌨️  Applying command line overrides...");
    if let Ok(model_path) = std::env::var("CLI_MODEL_PATH") {
        hierarchical_config.model.path = model_path;
    }

    // Pattern 3: Dynamic configuration
    println!("\n3️⃣ Dynamic configuration:");
    let available_memory = 8192; // MB, would be detected at runtime
    let gpu_available = true; // Would be detected at runtime

    let mut dynamic_config = presets::chatbot();

    if available_memory < 4096 {
        println!("   💾 Low memory detected, applying memory optimizations");
        dynamic_config.merge(presets::memory_optimized());
    } else if available_memory > 16384 {
        println!("   🚀 High memory detected, applying performance optimizations");
        dynamic_config.merge(presets::performance_optimized());
    }

    if gpu_available {
        println!("   🎮 GPU detected, enabling GPU acceleration");
        dynamic_config.model.gpu_layers = 32;
        dynamic_config.performance.gpu_optimizations.enable_gpu = true;
    } else {
        println!("   💻 No GPU detected, using CPU-only configuration");
        dynamic_config.model.gpu_layers = 0;
        dynamic_config.performance.gpu_optimizations.enable_gpu = false;
    }

    // Pattern 4: Configuration conversion
    println!("\n4️⃣ Configuration conversion:");
    let _model_params = dynamic_config.to_model_params();
    let _context_params = dynamic_config.to_context_params();
    let _sampler_params = dynamic_config.to_sampler_params();

    println!("   ✅ Converted to ModelParams");
    println!("   ✅ Converted to ContextParams");
    println!("   ✅ Converted to SamplerParams");
    println!("   🔧 Ready for direct use with builders");

    Ok(())
}
