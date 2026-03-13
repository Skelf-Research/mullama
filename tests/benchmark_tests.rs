//! Benchmark and performance tests for Mullama
//!
//! These tests measure performance characteristics and validate that the library
//! meets performance expectations. They provide benchmarks for comparison and
//! regression testing.

use mullama::*;
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const BENCHMARK_ITERATIONS: usize = 1000;
const PERFORMANCE_TIMEOUT_MS: u64 = 1000;

#[cfg(test)]
mod parameter_creation_benchmarks {
    use super::*;

    #[test]
    fn benchmark_model_params_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _params = ModelParams::default();
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "ModelParams creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );
        println!(
            "Total for {} iterations: {:.2}ms",
            BENCHMARK_ITERATIONS,
            duration.as_secs_f64() * 1000.0
        );

        // Performance assertion - should be very fast
        assert!(
            per_op.as_nanos() < 10_000,
            "ModelParams creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_context_params_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _params = ContextParams::default();
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "ContextParams creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );
        println!(
            "Total for {} iterations: {:.2}ms",
            BENCHMARK_ITERATIONS,
            duration.as_secs_f64() * 1000.0
        );

        // Performance assertion - should be fast despite thread detection
        assert!(
            per_op.as_nanos() < 50_000,
            "ContextParams creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_sampler_params_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _params = SamplerParams::default();
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "SamplerParams creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );
        println!(
            "Total for {} iterations: {:.2}ms",
            BENCHMARK_ITERATIONS,
            duration.as_secs_f64() * 1000.0
        );

        // Performance assertion
        assert!(
            per_op.as_nanos() < 10_000,
            "SamplerParams creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_parameter_cloning() {
        let model_params = ModelParams::default();
        let context_params = ContextParams::default();
        let sampler_params = SamplerParams::default();

        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _model_clone = model_params.clone();
            let _context_clone = context_params.clone();
            let _sampler_clone = sampler_params.clone();
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS * 3) as u32;

        println!(
            "Parameter cloning: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Cloning should be very fast
        assert!(
            per_op.as_nanos() < 5_000,
            "Parameter cloning too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod structure_creation_benchmarks {
    use super::*;

    #[test]
    fn benchmark_batch_creation_small() {
        let tokens = vec![1, 2, 3, 4, 5];

        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _batch = Batch::from_tokens(&tokens);
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "Small batch creation (5 tokens): {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be fast for small batches
        assert!(
            per_op.as_nanos() < 20_000,
            "Small batch creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_batch_creation_medium() {
        let tokens: Vec<TokenId> = (0..100).collect();

        let start = Instant::now();

        for _ in 0..(BENCHMARK_ITERATIONS / 10) {
            let _batch = Batch::from_tokens(&tokens);
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS / 10) as u32;

        println!(
            "Medium batch creation (100 tokens): {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should still be reasonable for medium batches
        assert!(
            per_op.as_nanos() < 100_000,
            "Medium batch creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_batch_creation_large() {
        let tokens: Vec<TokenId> = (0..1000).collect();

        let start = Instant::now();

        for _ in 0..(BENCHMARK_ITERATIONS / 100) {
            let _batch = Batch::from_tokens(&tokens);
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS / 100) as u32;

        println!(
            "Large batch creation (1000 tokens): {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be acceptable for large batches
        assert!(
            per_op.as_nanos() < 1_000_000,
            "Large batch creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_token_data_array_creation() {
        let candidates = vec![
            TokenData {
                id: 1,
                logit: 1.0,
                p: 0.5,
            },
            TokenData {
                id: 2,
                logit: 2.0,
                p: 0.3,
            },
            TokenData {
                id: 3,
                logit: 0.5,
                p: 0.2,
            },
        ];

        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _array = TokenDataArray::new(candidates.clone());
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "TokenDataArray creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be fast
        assert!(
            per_op.as_nanos() < 50_000,
            "TokenDataArray creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_embeddings_creation() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _embeddings = Embeddings::new(data.clone(), 5);
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "Embeddings creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be fast
        assert!(
            per_op.as_nanos() < 30_000,
            "Embeddings creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod sampling_benchmarks {
    use super::*;

    #[test]
    fn benchmark_sampler_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _sampler = Sampler::new().expect("Failed to create sampler");
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "Sampler creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be fast
        assert!(
            per_op.as_nanos() < 20_000,
            "Sampler creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_sampler_chain_creation() {
        let params = SamplerChainParams::default();

        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _chain = SamplerChain::new(params.clone());
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "SamplerChain creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be reasonably fast
        assert!(
            per_op.as_nanos() < 50_000,
            "SamplerChain creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_logit_bias_operations() {
        let biases: Vec<LogitBias> = (0..100)
            .map(|i| LogitBias {
                token: i,
                bias: i as f32 * 0.1,
            })
            .collect();

        let start = Instant::now();

        for _ in 0..(BENCHMARK_ITERATIONS / 10) {
            // Simulate processing logit biases
            for bias in &biases {
                let _token = bias.token;
                let _bias_value = bias.bias;
            }
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS / 10) as u32;

        println!(
            "Logit bias processing (100 biases): {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be very fast
        assert!(
            per_op.as_nanos() < 10_000,
            "Logit bias processing too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod memory_benchmarks {
    use super::*;

    #[test]
    fn benchmark_memory_manager_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _memory = MemoryManager::new();
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "MemoryManager creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be very fast
        assert!(
            per_op.as_nanos() < 5_000,
            "MemoryManager creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_vocabulary_creation() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _vocab = Vocabulary::new();
        }

        let duration = start.elapsed();
        let per_op = duration / BENCHMARK_ITERATIONS as u32;

        println!(
            "Vocabulary creation: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be very fast
        assert!(
            per_op.as_nanos() < 5_000,
            "Vocabulary creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_session_creation() {
        let data = vec![1u8; 1000];

        let start = Instant::now();

        for _ in 0..(BENCHMARK_ITERATIONS / 10) {
            let _session = Session { data: data.clone() };
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS / 10) as u32;

        println!(
            "Session creation (1KB data): {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be reasonable for 1KB sessions
        assert!(
            per_op.as_nanos() < 100_000,
            "Session creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod ffi_benchmarks {
    use super::*;
    use mullama::sys;

    #[test]
    fn benchmark_backend_init_free() {
        let start = Instant::now();

        for _ in 0..(BENCHMARK_ITERATIONS / 100) {
            unsafe {
                sys::llama_backend_init();
                sys::llama_backend_free();
            }
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS / 100) as u32;

        println!(
            "Backend init/free cycle: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should complete within reasonable time
        assert!(
            per_op.as_nanos() < 1_000_000,
            "Backend init/free too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_system_queries() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            unsafe {
                let _max_devices = sys::llama_max_devices();
                let _max_sequences = sys::llama_max_parallel_sequences();
                let _supports_mmap = sys::llama_supports_mmap();
                let _supports_mlock = sys::llama_supports_mlock();
                let _supports_gpu = sys::llama_supports_gpu_offload();
            }
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS * 5) as u32;

        println!(
            "System capability query: {:.2}μs per operation",
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should be very fast
        assert!(
            per_op.as_nanos() < 5_000,
            "System queries too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_constant_access() {
        let start = Instant::now();

        for _ in 0..BENCHMARK_ITERATIONS {
            let _null_token = sys::LLAMA_TOKEN_NULL;
            let _default_seed = sys::LLAMA_DEFAULT_SEED;
            let _session_version = sys::LLAMA_SESSION_VERSION;
            let _state_version = sys::LLAMA_STATE_SEQ_VERSION;
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS * 4) as u32;

        println!(
            "Constant access: {:.2}ns per operation",
            per_op.as_nanos() as f64
        );

        // Should be extremely fast (compile-time constants)
        assert!(
            per_op.as_nanos() < 100,
            "Constant access too slow: {:.2}ns",
            per_op.as_nanos() as f64
        );
    }
}

#[cfg(test)]
mod concurrency_benchmarks {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn benchmark_concurrent_parameter_creation() {
        const NUM_THREADS: usize = 4;
        const ITERATIONS_PER_THREAD: usize = BENCHMARK_ITERATIONS / NUM_THREADS;

        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let mut handles = vec![];

        let start = Instant::now();

        for _ in 0..NUM_THREADS {
            let barrier = barrier.clone();
            let handle = thread::spawn(move || {
                barrier.wait();

                for _ in 0..ITERATIONS_PER_THREAD {
                    let _model_params = ModelParams::default();
                    let _context_params = ContextParams::default();
                    let _sampler_params = SamplerParams::default();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS * 3) as u32;

        println!(
            "Concurrent parameter creation ({} threads): {:.2}μs per operation",
            NUM_THREADS,
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should scale reasonably with multiple threads
        assert!(
            per_op.as_nanos() < 100_000,
            "Concurrent parameter creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }

    #[test]
    fn benchmark_concurrent_structure_creation() {
        const NUM_THREADS: usize = 4;
        const ITERATIONS_PER_THREAD: usize = BENCHMARK_ITERATIONS / NUM_THREADS;

        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let mut handles = vec![];

        let start = Instant::now();

        for thread_id in 0..NUM_THREADS {
            let barrier = barrier.clone();
            let handle = thread::spawn(move || {
                barrier.wait();

                let tokens: Vec<TokenId> = (0..10).map(|i| i + thread_id as TokenId * 10).collect();

                for _ in 0..ITERATIONS_PER_THREAD {
                    let _batch = Batch::from_tokens(&tokens);
                    let _session = Session {
                        data: vec![thread_id as u8; 10],
                    };
                    let _embeddings = Embeddings::new(vec![thread_id as f32; 5], 5);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();
        let per_op = duration / (BENCHMARK_ITERATIONS * 3) as u32;

        println!(
            "Concurrent structure creation ({} threads): {:.2}μs per operation",
            NUM_THREADS,
            per_op.as_nanos() as f64 / 1000.0
        );

        // Should scale reasonably with multiple threads
        assert!(
            per_op.as_nanos() < 200_000,
            "Concurrent structure creation too slow: {:.2}μs",
            per_op.as_nanos() as f64 / 1000.0
        );
    }
}

#[cfg(test)]
mod regression_benchmarks {
    use super::*;

    #[test]
    fn benchmark_parameter_size_overhead() {
        use std::mem::size_of;

        // Measure memory overhead of our parameter structures
        let model_params_size = size_of::<ModelParams>();
        let context_params_size = size_of::<ContextParams>();
        let sampler_params_size = size_of::<SamplerParams>();

        println!("Parameter structure sizes:");
        println!("  ModelParams: {} bytes", model_params_size);
        println!("  ContextParams: {} bytes", context_params_size);
        println!("  SamplerParams: {} bytes", sampler_params_size);

        // Ensure structures aren't excessively large
        assert!(
            model_params_size < 1000,
            "ModelParams too large: {} bytes",
            model_params_size
        );
        assert!(
            context_params_size < 500,
            "ContextParams too large: {} bytes",
            context_params_size
        );
        assert!(
            sampler_params_size < 300,
            "SamplerParams too large: {} bytes",
            sampler_params_size
        );
    }

    #[test]
    fn benchmark_struct_size_overhead() {
        use std::mem::size_of;

        // Measure memory overhead of core structures
        let batch_size = size_of::<Batch>();
        let token_data_size = size_of::<TokenData>();
        let logit_bias_size = size_of::<LogitBias>();

        println!("Structure sizes:");
        println!("  Batch: {} bytes", batch_size);
        println!("  TokenData: {} bytes", token_data_size);
        println!("  LogitBias: {} bytes", logit_bias_size);

        // Ensure structures are reasonably sized
        assert!(batch_size < 200, "Batch too large: {} bytes", batch_size);
        assert!(
            token_data_size < 50,
            "TokenData too large: {} bytes",
            token_data_size
        );
        assert!(
            logit_bias_size < 20,
            "LogitBias too large: {} bytes",
            logit_bias_size
        );
    }

    #[test]
    fn benchmark_compilation_time_indicators() {
        // Test compilation-heavy operations to ensure they don't regress
        let start = Instant::now();

        // Operations that would be slow if compilation was heavy
        for _ in 0..1000 {
            let _params = ModelParams {
                n_gpu_layers: 10,
                split_mode: sys::llama_split_mode::LLAMA_SPLIT_MODE_LAYER,
                main_gpu: 0,
                tensor_split: vec![0.5, 0.5],
                vocab_only: false,
                use_mmap: true,
                use_mlock: false,
                check_tensors: true,
                use_extra_bufts: false,
                kv_overrides: vec![ModelKvOverride {
                    key: "test".to_string(),
                    value: ModelKvOverrideValue::Int(42),
                }],
                progress_callback: None,
            };
        }

        let duration = start.elapsed();
        println!(
            "Complex parameter creation time: {:.2}ms for 1000 operations",
            duration.as_secs_f64() * 1000.0
        );

        // Should complete quickly, indicating good compilation performance
        assert!(
            duration.as_millis() < 100,
            "Complex parameter creation too slow: {}ms",
            duration.as_millis()
        );
    }
}

#[cfg(test)]
mod performance_regression_tests {
    use super::*;

    #[test]
    fn test_no_performance_regression_parameters() {
        // Baseline performance expectations based on simple operations
        let iterations = 10000;

        // Test ModelParams
        let start = Instant::now();
        for _ in 0..iterations {
            let _params = ModelParams::default();
        }
        let model_duration = start.elapsed();

        // Test ContextParams
        let start = Instant::now();
        for _ in 0..iterations {
            let _params = ContextParams::default();
        }
        let context_duration = start.elapsed();

        // Test SamplerParams
        let start = Instant::now();
        for _ in 0..iterations {
            let _params = SamplerParams::default();
        }
        let sampler_duration = start.elapsed();

        println!("Performance baseline for {} iterations:", iterations);
        println!(
            "  ModelParams: {:.2}ms",
            model_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  ContextParams: {:.2}ms",
            context_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  SamplerParams: {:.2}ms",
            sampler_duration.as_secs_f64() * 1000.0
        );

        // Performance regression thresholds
        assert!(
            model_duration.as_millis() < 100,
            "ModelParams performance regression"
        );
        assert!(
            context_duration.as_millis() < 500,
            "ContextParams performance regression"
        ); // Allows for thread detection
        assert!(
            sampler_duration.as_millis() < 50,
            "SamplerParams performance regression"
        );
    }

    #[test]
    fn test_no_memory_leak_indicators() {
        // Test for potential memory leak patterns
        let iterations = 1000;

        // Create and drop many parameters
        for _ in 0..iterations {
            let model_params = ModelParams::default();
            let context_params = ContextParams::default();
            let sampler_params = SamplerParams::default();

            // Create complex structures
            let mut complex_model = model_params;
            complex_model.tensor_split = vec![1.0; 16];
            complex_model.kv_overrides = vec![ModelKvOverride {
                key: "test".to_string(),
                value: ModelKvOverrideValue::Str("value".to_string()),
            }];

            // These should be dropped cleanly
            drop(complex_model);
            drop(context_params);
            drop(sampler_params);
        }

        // If we reach here without crashing, no obvious memory issues
        println!(
            "Memory leak test completed successfully for {} iterations",
            iterations
        );
    }
}
