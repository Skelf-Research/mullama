//! # Builder Patterns Example
//!
//! This example demonstrates the fluent builder patterns in Mullama for
//! creating complex configurations with type safety and ergonomic APIs.
//!
//! Run with: cargo run --example builder_patterns --features async

use mullama::builder::presets;
use mullama::prelude::*;

#[tokio::main]
async fn main() -> Result<(), MullamaError> {
    println!("🔧 Builder Patterns Example");
    println!("===========================");

    // Example 1: Basic model building
    demonstrate_model_builder().await?;

    // Example 2: Context builder with optimizations
    demonstrate_context_builder().await?;

    // Example 3: Sampler builder with penalties
    demonstrate_sampler_builder().await?;

    // Example 4: Using presets
    demonstrate_presets().await?;

    // Example 5: Complex workflow
    demonstrate_complex_workflow().await?;
    demonstrate_builder_patterns_showcase();

    println!("\n✨ All builder patterns demonstrated successfully!");
    Ok(())
}

async fn demonstrate_model_builder() -> Result<(), MullamaError> {
    println!("\n🏗️ Model Builder");
    println!("================");

    // Basic model configuration
    println!("1️⃣ Basic model configuration:");
    let _basic_model_builder = ModelBuilder::new()
        .path("path/to/model.gguf")
        .gpu_layers(16)
        .context_size(2048);

    println!("   ✅ Basic model builder configured");
    println!("      - Path: path/to/model.gguf");
    println!("      - GPU layers: 16");
    println!("      - Context size: 2048");

    // Advanced model configuration
    println!("\n2️⃣ Advanced model configuration:");
    let _advanced_model_builder = ModelBuilder::new()
        .path("path/to/large_model.gguf")
        .gpu_layers(32)
        .context_size(4096)
        .memory_mapping(true)
        .memory_locking(false)
        .tensor_validation(true)
        .vocabulary_only(false);

    println!("   ✅ Advanced model builder configured");
    println!("      - GPU layers: 32");
    println!("      - Context size: 4096");
    println!("      - Memory mapping: enabled");
    println!("      - Tensor validation: enabled");

    // Model with preset
    println!("\n3️⃣ Model with performance preset:");
    let _performance_model_builder = ModelBuilder::new()
        .path("path/to/model.gguf")
        .preset(presets::performance_optimized);

    println!("   ✅ Performance optimized model builder");
    println!("      - GPU layers: maximized");
    println!("      - Memory mapping: enabled");
    println!("      - Tensor validation: disabled for speed");

    #[cfg(feature = "async")]
    {
        // Async model building (would work with real model path)
        println!("\n4️⃣ Async model building (simulated):");
        // let model = ModelBuilder::new()
        //     .path("real/model/path.gguf")
        //     .gpu_layers(24)
        //     .build_async()
        //     .await?;
        println!("   ✅ Model would be built asynchronously");
    }

    Ok(())
}

async fn demonstrate_context_builder() -> Result<(), MullamaError> {
    println!("\n🧠 Context Builder");
    println!("==================");

    // Note: In real scenario, you'd have an actual model
    // let model = Arc::new(Model::load("model.gguf")?);

    println!("1️⃣ Basic context configuration:");
    // let basic_context_builder = ContextBuilder::new(model.clone())
    //     .context_size(2048)
    //     .batch_size(512)
    //     .threads(8);

    println!("   ✅ Basic context builder configured");
    println!("      - Context size: 2048 tokens");
    println!("      - Batch size: 512");
    println!("      - Threads: 8");

    println!("\n2️⃣ Advanced context configuration:");
    // let advanced_context_builder = ContextBuilder::new(model.clone())
    //     .context_size(4096)
    //     .batch_size(1024)
    //     .physical_batch_size(512)
    //     .max_sequences(4)
    //     .threads(12)
    //     .batch_threads(6)
    //     .embeddings(true)
    //     .flash_attention(true)
    //     .kqv_offload(true);

    println!("   ✅ Advanced context builder configured");
    println!("      - Context size: 4096 tokens");
    println!("      - Batch size: 1024");
    println!("      - Physical batch size: 512");
    println!("      - Max sequences: 4");
    println!("      - Flash attention: enabled");
    println!("      - KQV offload: enabled");

    println!("\n3️⃣ Performance optimized context:");
    // let performance_context_builder = ContextBuilder::new(model.clone())
    //     .optimize_for_performance();

    println!("   ✅ Performance optimized context");
    println!("      - Flash attention: enabled");
    println!("      - Larger batch sizes");
    println!("      - KQV offload: enabled");

    println!("\n4️⃣ Memory optimized context:");
    // let memory_context_builder = ContextBuilder::new(model.clone())
    //     .optimize_for_memory();

    println!("   ✅ Memory optimized context");
    println!("      - Smaller context size: 1024");
    println!("      - Smaller batch sizes");
    println!("      - Flash attention: disabled");

    Ok(())
}

async fn demonstrate_sampler_builder() -> Result<(), MullamaError> {
    println!("\n🎲 Sampler Builder");
    println!("==================");

    println!("1️⃣ Basic sampling configuration:");
    let _basic_sampler_builder = SamplerBuilder::new()
        .temperature(0.7)
        .top_k(40)
        .nucleus(0.9);

    println!("   ✅ Basic sampler configured");
    println!("      - Temperature: 0.7");
    println!("      - Top-k: 40");
    println!("      - Nucleus (top-p): 0.9");

    println!("\n2️⃣ Advanced sampling with penalties:");
    let _advanced_sampler_builder = SamplerBuilder::new()
        .temperature(0.8)
        .top_k(50)
        .nucleus(0.95)
        .min_probability(0.02)
        .penalties(|p| {
            p.repetition(1.15)
                .frequency(0.1)
                .presence(0.1)
                .lookback(128)
        })
        .seed(12345);

    println!("   ✅ Advanced sampler configured");
    println!("      - Temperature: 0.8");
    println!("      - Top-k: 50");
    println!("      - Nucleus: 0.95");
    println!("      - Min probability: 0.02");
    println!("      - Repetition penalty: 1.15");
    println!("      - Frequency penalty: 0.1");
    println!("      - Presence penalty: 0.1");
    println!("      - Lookback window: 128 tokens");
    println!("      - Seed: 12345");

    println!("\n3️⃣ Creative sampling preset:");
    let _creative_sampler_builder = SamplerBuilder::new().preset(presets::creative_sampling);

    println!("   ✅ Creative sampling configured");
    println!("      - High temperature for creativity");
    println!("      - Larger top-k for diversity");
    println!("      - Adjusted penalties");

    println!("\n4️⃣ Precise sampling preset:");
    let _precise_sampler_builder = SamplerBuilder::new().preset(presets::precise_sampling);

    println!("   ✅ Precise sampling configured");
    println!("      - Low temperature for consistency");
    println!("      - Small top-k for focus");
    println!("      - Minimal penalties");

    // In real scenario, build with actual model:
    // let model = Arc::new(Model::load("model.gguf")?);
    // let sampler = basic_sampler_builder.build(model);

    Ok(())
}

async fn demonstrate_presets() -> Result<(), MullamaError> {
    println!("\n🎯 Preset Configurations");
    println!("========================");

    println!("1️⃣ Model presets:");

    // Creative writing model
    let _creative_model = ModelBuilder::new()
        .path("creative_model.gguf")
        .preset(presets::creative_model);
    println!("   🎨 Creative model: 24 GPU layers, 4096 context");

    // Performance optimized model
    let _performance_model = ModelBuilder::new()
        .path("performance_model.gguf")
        .preset(presets::performance_optimized);
    println!("   ⚡ Performance model: Max GPU layers, no validation");

    // Memory optimized model
    let _memory_model = ModelBuilder::new()
        .path("memory_model.gguf")
        .preset(presets::memory_optimized);
    println!("   💾 Memory model: CPU only, optimized memory usage");

    println!("\n2️⃣ Sampling presets:");

    // Creative sampling
    let _creative_sampling = SamplerBuilder::new().preset(presets::creative_sampling);
    println!("   🎨 Creative: temp=0.9, top_k=60, high diversity");

    // Precise sampling
    let _precise_sampling = SamplerBuilder::new().preset(presets::precise_sampling);
    println!("   🎯 Precise: temp=0.2, top_k=10, high consistency");

    // Balanced sampling
    let _balanced_sampling = SamplerBuilder::new().preset(presets::balanced_sampling);
    println!("   ⚖️  Balanced: temp=0.7, top_k=40, moderate settings");

    Ok(())
}

async fn demonstrate_complex_workflow() -> Result<(), MullamaError> {
    println!("\n🔄 Complex Workflow");
    println!("===================");

    println!("🏗️  Building complete LLM setup with builders...");

    // Step 1: Configure model with chained builder
    println!("\n1️⃣ Model configuration:");
    let _model_builder = ModelBuilder::new()
        .path("path/to/model.gguf")
        .gpu_layers(32)
        .context_size(4096)
        .memory_mapping(true)
        .preset(presets::performance_optimized);

    println!("   ✅ Model builder ready");

    #[cfg(feature = "async")]
    {
        // Step 2: Build model asynchronously (simulated)
        println!("\n2️⃣ Building model asynchronously...");
        // let model = model_builder.build_async().await?;
        println!("   ✅ Model built successfully");

        // Step 3: Create optimized context
        println!("\n3️⃣ Creating optimized context...");
        // let context = ContextBuilder::new(model.clone())
        //     .context_size(4096)
        //     .batch_size(1024)
        //     .threads(std::thread::available_parallelism().unwrap().get() as u32)
        //     .optimize_for_performance()
        //     .build()?;
        println!("   ✅ Context created with performance optimizations");

        // Step 4: Configure advanced sampling
        println!("\n4️⃣ Configuring advanced sampling...");
        // let sampler = SamplerBuilder::new()
        //     .temperature(0.8)
        //     .top_k(50)
        //     .nucleus(0.95)
        //     .penalties(|p| p
        //         .repetition(1.1)
        //         .frequency(0.05)
        //         .presence(0.05)
        //         .lookback(64)
        //     )
        //     .build(model.clone());
        println!("   ✅ Advanced sampler configured");

        // Step 5: Complete setup ready for generation
        println!("\n5️⃣ Complete setup ready:");
        println!("   🤖 Model: Loaded with 32 GPU layers");
        println!("   🧠 Context: 4096 tokens, performance optimized");
        println!("   🎲 Sampler: Balanced creativity and consistency");
        println!("   ⚡ Ready for high-performance text generation!");
    }

    // Example of conditional configuration
    println!("\n🔀 Conditional Configuration Example:");
    let use_gpu = std::env::var("USE_GPU").unwrap_or_else(|_| "true".to_string()) == "true";
    let context_size: u32 = std::env::var("CONTEXT_SIZE")
        .unwrap_or_else(|_| "2048".to_string())
        .parse()
        .unwrap_or(2048);

    let _conditional_model = ModelBuilder::new()
        .path("model.gguf")
        .gpu_layers(if use_gpu { 32 } else { 0 })
        .context_size(context_size)
        .preset(if use_gpu {
            presets::performance_optimized
        } else {
            presets::memory_optimized
        });

    println!("   ⚙️  Conditional model configured:");
    println!(
        "      - GPU usage: {}",
        if use_gpu { "enabled" } else { "disabled" }
    );
    println!("      - Context size: {}", context_size);
    println!(
        "      - Preset: {}",
        if use_gpu { "performance" } else { "memory" }
    );

    Ok(())
}

fn demonstrate_builder_patterns_showcase() {
    println!("\n🎨 Builder Patterns Showcase");
    println!("============================");

    println!("✨ Key Benefits of Builder Patterns:");
    println!("   🔒 Type Safety: Compile-time validation of configurations");
    println!("   🌊 Fluent API: Readable, chainable method calls");
    println!("   🎯 Smart Defaults: Sensible defaults for all parameters");
    println!("   🏗️  Progressive Disclosure: Start simple, add complexity as needed");
    println!("   🎛️  Presets: Quick configuration for common use cases");
    println!("   🔧 Extensibility: Easy to add new configuration options");

    println!("\n🏛️  Architecture Benefits:");
    println!("   📏 Consistent API across all components");
    println!("   🛡️  Validation at build time, not runtime");
    println!("   🎪 Flexible configuration without complexity");
    println!("   📚 Self-documenting code through method names");
    println!("   🔄 Easy to refactor and maintain");
}
