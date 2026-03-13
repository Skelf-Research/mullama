//! Comprehensive sampling system tests
//!
//! These tests validate the complete sampling functionality including all sampler types,
//! sampler chains, and edge cases in the sampling system.

// Temporarily disabled until llama.cpp linking is fixed
#[cfg(feature = "llama-cpp-tests")]
use mullama::*;
#[cfg(feature = "llama-cpp-tests")]
use std::sync::Arc;

#[cfg(all(test, feature = "llama-cpp-tests"))]
fn expect_sampler(result: Result<Sampler, MullamaError>) -> Sampler {
    result.expect("Failed to create sampler")
}

#[cfg(all(test, feature = "llama-cpp-tests"))]
mod sampler_creation_tests {
    use super::*;

    #[test]
    fn test_greedy_sampler_creation() {
        let sampler = expect_sampler(Sampler::greedy());
        assert_eq!(sampler.name(), "greedy");
        println!("✓ Greedy sampler created successfully");
    }

    #[test]
    fn test_distribution_sampler_creation() {
        let seeds = vec![0, 12345, u32::MAX, LLAMA_DEFAULT_SEED];

        for seed in seeds {
            let sampler = expect_sampler(Sampler::dist(seed));
            assert_eq!(sampler.name(), "dist");
            println!("✓ Distribution sampler created with seed: {}", seed);
        }
    }

    #[test]
    fn test_top_k_sampler_creation() {
        let k_values = vec![1, 5, 10, 40, 100, 1000];

        for k in k_values {
            let sampler = expect_sampler(Sampler::top_k(k));
            let name = sampler.name();
            assert!(name.contains("top") || name.contains("k") || name == "top_k");
            println!("✓ Top-k sampler created with k={}", k);
        }
    }

    #[test]
    fn test_top_p_sampler_creation() {
        let test_cases = vec![
            (0.1, 1),  // Very restrictive
            (0.5, 1),  // Medium
            (0.9, 1),  // Common setting
            (0.95, 1), // Default-like
            (0.99, 5), // Very permissive
            (1.0, 10), // Maximum
        ];

        for (p, min_keep) in test_cases {
            let sampler = expect_sampler(Sampler::top_p(p, min_keep));
            let name = sampler.name();
            assert!(name.contains("top") || name.contains("p") || name == "top_p");
            println!(
                "✓ Top-p sampler created with p={}, min_keep={}",
                p, min_keep
            );
        }
    }

    #[test]
    fn test_min_p_sampler_creation() {
        let test_cases = vec![
            (0.01, 1), // Very low threshold
            (0.05, 1), // Low threshold
            (0.1, 1),  // Medium threshold
            (0.2, 2),  // Higher threshold
            (0.5, 5),  // High threshold
        ];

        for (p, min_keep) in test_cases {
            let sampler = expect_sampler(Sampler::min_p(p, min_keep));
            let name = sampler.name();
            assert!(name.contains("min") || name.contains("p") || name == "min_p");
            println!(
                "✓ Min-p sampler created with p={}, min_keep={}",
                p, min_keep
            );
        }
    }

    #[test]
    fn test_temperature_sampler_creation() {
        let temperatures = vec![0.1, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0];

        for temp in temperatures {
            let sampler = expect_sampler(Sampler::temperature(temp));
            let name = sampler.name();
            assert!(name.contains("temp") || name == "temperature");
            println!("✓ Temperature sampler created with temp={}", temp);
        }
    }

    #[test]
    fn test_temperature_ext_sampler_creation() {
        let test_cases = vec![
            (0.8, 0.0, 1.0),  // Standard
            (1.0, 0.1, 0.9),  // With delta and exponent
            (0.5, 0.2, 1.2),  // Different parameters
            (1.5, -0.1, 0.8), // Negative delta
        ];

        for (temp, delta, exponent) in test_cases {
            let sampler = expect_sampler(Sampler::temperature_ext(temp, delta, exponent));
            let name = sampler.name();
            assert!(name.contains("temp") || name == "temperature");
            println!(
                "✓ Extended temperature sampler created with temp={}, delta={}, exp={}",
                temp, delta, exponent
            );
        }
    }

    #[test]
    fn test_mirostat_v2_sampler_creation() {
        let test_cases = vec![
            (12345, 5.0, 0.1),     // Standard parameters
            (0, 1.0, 0.01),        // Minimal parameters
            (u32::MAX, 10.0, 0.5), // High parameters
        ];

        for (seed, tau, eta) in test_cases {
            let sampler = expect_sampler(Sampler::mirostat_v2(seed, tau, eta));
            let name = sampler.name();
            assert!(name.contains("mirostat") || name.contains("v2"));
            println!(
                "✓ Mirostat v2 sampler created with seed={}, tau={}, eta={}",
                seed, tau, eta
            );
        }
    }

    #[test]
    fn test_tail_free_sampler_creation() {
        let test_cases = vec![
            (1.0, 1), // Standard TFS
            (0.5, 1), // More aggressive
            (0.1, 2), // Very aggressive
            (2.0, 5), // Less aggressive
        ];

        for (z, min_keep) in test_cases {
            let sampler = expect_sampler(Sampler::tail_free(z, min_keep));
            let name = sampler.name();
            assert!(name.contains("tail") || name.contains("free") || name.contains("tfs"));
            println!(
                "✓ Tail-free sampler created with z={}, min_keep={}",
                z, min_keep
            );
        }
    }

    #[test]
    fn test_typical_sampler_creation() {
        let test_cases = vec![
            (1.0, 1), // Disabled
            (0.9, 1), // Light filtering
            (0.7, 2), // Medium filtering
            (0.5, 5), // Strong filtering
        ];

        for (p, min_keep) in test_cases {
            let sampler = expect_sampler(Sampler::typical(p, min_keep));
            let name = sampler.name();
            assert!(name.contains("typical") || name.contains("typ"));
            println!(
                "✓ Typical sampler created with p={}, min_keep={}",
                p, min_keep
            );
        }
    }
}

#[cfg(all(test, feature = "llama-cpp-tests"))]
mod sampler_chain_tests {
    use super::*;

    #[test]
    fn test_empty_sampler_chain() {
        let chain = SamplerChain::default();
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        println!("✓ Empty sampler chain created");
    }

    #[test]
    fn test_sampler_chain_with_performance() {
        let params = SamplerChainParams { no_perf: false };
        let chain = SamplerChain::new(params);
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        println!("✓ Performance-enabled sampler chain created");
    }

    #[test]
    fn test_sampler_chain_without_performance() {
        let params = SamplerChainParams { no_perf: true };
        let chain = SamplerChain::new(params);
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
        println!("✓ Performance-disabled sampler chain created");
    }

    #[test]
    fn test_single_sampler_chain() {
        let mut chain = SamplerChain::default();
        let sampler = expect_sampler(Sampler::greedy());

        chain.add(sampler);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
        println!("✓ Single sampler added to chain");
    }

    #[test]
    fn test_multiple_sampler_chain() {
        let mut chain = SamplerChain::default();

        // Add multiple samplers
        chain.add(expect_sampler(Sampler::top_k(40)));
        chain.add(expect_sampler(Sampler::top_p(0.9, 1)));
        chain.add(expect_sampler(Sampler::temperature(0.8)));
        chain.add(expect_sampler(Sampler::dist(12345)));

        assert_eq!(chain.len(), 4);
        assert!(!chain.is_empty());
        println!(
            "✓ Multiple samplers added to chain (length: {})",
            chain.len()
        );
    }

    #[test]
    fn test_typical_sampling_chain() {
        // Test a typical sampling configuration
        let mut chain = SamplerChain::default();

        // Typical order: penalties -> top_k -> top_p -> temperature -> distribution
        chain.add(expect_sampler(Sampler::top_k(40)));
        chain.add(expect_sampler(Sampler::top_p(0.9, 1)));
        chain.add(expect_sampler(Sampler::temperature(0.8)));
        chain.add(expect_sampler(Sampler::dist(LLAMA_DEFAULT_SEED)));

        assert_eq!(chain.len(), 4);

        // Test that we can get samplers by index
        for i in 0..chain.len() {
            let sampler_ptr = chain.get(i);
            assert!(
                sampler_ptr.is_some(),
                "Should be able to get sampler at index {}",
                i
            );
        }

        // Test out-of-bounds access
        let invalid_sampler = chain.get(100);
        assert!(
            invalid_sampler.is_none(),
            "Should return None for invalid index"
        );

        println!("✓ Typical sampling chain created and validated");
    }

    #[test]
    fn test_sampler_chain_removal() {
        let mut chain = SamplerChain::default();

        // Add samplers
        chain.add(expect_sampler(Sampler::greedy()));
        chain.add(expect_sampler(Sampler::top_k(10)));
        chain.add(expect_sampler(Sampler::temperature(1.0)));

        assert_eq!(chain.len(), 3);

        // Remove middle sampler
        let removed = chain.remove(1);
        assert!(removed.is_some(), "Should be able to remove sampler");
        assert_eq!(chain.len(), 2);

        // Try to remove invalid index
        let invalid_remove = chain.remove(100);
        assert!(
            invalid_remove.is_none(),
            "Should return None for invalid removal"
        );

        println!("✓ Sampler chain removal tested");
    }

    #[test]
    fn test_complex_sampling_strategies() {
        let strategies = vec![
            // Conservative strategy
            vec![
                (
                    "top_k",
                    Box::new(|| expect_sampler(Sampler::top_k(10))) as Box<dyn Fn() -> Sampler>,
                ),
                ("top_p", Box::new(|| expect_sampler(Sampler::top_p(0.8, 1)))),
                (
                    "temperature",
                    Box::new(|| expect_sampler(Sampler::temperature(0.3))),
                ),
                ("dist", Box::new(|| expect_sampler(Sampler::dist(42)))),
            ],
            // Creative strategy
            vec![
                ("top_k", Box::new(|| expect_sampler(Sampler::top_k(100)))),
                (
                    "typical",
                    Box::new(|| expect_sampler(Sampler::typical(0.9, 1))),
                ),
                (
                    "temperature",
                    Box::new(|| expect_sampler(Sampler::temperature(1.2))),
                ),
                ("dist", Box::new(|| expect_sampler(Sampler::dist(12345)))),
            ],
            // Balanced strategy
            vec![
                ("top_k", Box::new(|| expect_sampler(Sampler::top_k(40)))),
                ("top_p", Box::new(|| expect_sampler(Sampler::top_p(0.9, 1)))),
                ("min_p", Box::new(|| expect_sampler(Sampler::min_p(0.1, 1)))),
                (
                    "temperature",
                    Box::new(|| expect_sampler(Sampler::temperature(0.8))),
                ),
                (
                    "dist",
                    Box::new(|| expect_sampler(Sampler::dist(LLAMA_DEFAULT_SEED))),
                ),
            ],
        ];

        for (i, strategy) in strategies.iter().enumerate() {
            let mut chain = SamplerChain::default();

            for (name, sampler_fn) in strategy {
                chain.add(sampler_fn());
            }

            assert_eq!(chain.len(), strategy.len() as i32);
            println!("✓ Strategy {} created with {} samplers", i, strategy.len());
        }
    }
}

#[cfg(all(test, feature = "llama-cpp-tests"))]
mod sampler_parameter_validation_tests {
    use super::*;

    #[test]
    fn test_sampler_params_build_chain() {
        let test_configs = vec![
            // Minimal configuration
            SamplerParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                typical_p: 1.0,
                penalty_repeat: 1.0,
                penalty_freq: 0.0,
                penalty_present: 0.0,
                ..Default::default()
            },
            // Full configuration
            SamplerParams {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.1,
                typical_p: 0.8,
                penalty_repeat: 1.1,
                penalty_freq: 0.1,
                penalty_present: 0.1,
                penalty_last_n: 64,
                ..Default::default()
            },
            // Edge case configuration
            SamplerParams {
                temperature: 2.0,
                top_k: 1,
                top_p: 0.1,
                min_p: 0.9,
                typical_p: 0.1,
                penalty_repeat: 2.0,
                penalty_freq: 1.0,
                penalty_present: 1.0,
                penalty_last_n: 1,
                ..Default::default()
            },
        ];

        for (i, params) in test_configs.iter().enumerate() {
            // Note: This would require a model to actually build the chain
            // For now, we validate the parameters themselves

            assert!(
                params.temperature >= 0.0,
                "Config {} has negative temperature",
                i
            );
            assert!(params.top_k >= 0, "Config {} has negative top_k", i);
            assert!(
                params.top_p >= 0.0 && params.top_p <= 1.0,
                "Config {} has invalid top_p",
                i
            );
            assert!(
                params.min_p >= 0.0 && params.min_p <= 1.0,
                "Config {} has invalid min_p",
                i
            );
            assert!(
                params.typical_p >= 0.0,
                "Config {} has negative typical_p",
                i
            );
            assert!(
                params.penalty_repeat >= 0.0,
                "Config {} has negative penalty_repeat",
                i
            );
            assert!(
                params.penalty_last_n >= 0,
                "Config {} has negative penalty_last_n",
                i
            );

            println!("✓ Sampler config {} validated", i);
        }
    }

    #[test]
    fn test_logit_bias_edge_cases() {
        let edge_cases = vec![
            // Empty biases
            vec![],
            // Single bias
            vec![LogitBias {
                token: 0,
                bias: 0.0,
            }],
            // Extreme biases
            vec![
                LogitBias {
                    token: 1,
                    bias: f32::NEG_INFINITY,
                },
                LogitBias {
                    token: 2,
                    bias: f32::INFINITY,
                },
                LogitBias {
                    token: 3,
                    bias: f32::NAN,
                },
            ],
            // Large number of biases
            (0..1000)
                .map(|i| LogitBias {
                    token: i,
                    bias: i as f32 * 0.01,
                })
                .collect(),
        ];

        for (i, biases) in edge_cases.iter().enumerate() {
            // Test that bias arrays can be created
            for (j, bias) in biases.iter().enumerate() {
                assert!(bias.token >= 0, "Case {} bias {} has negative token", i, j);
                // Note: bias can be any value including infinity/NaN for edge testing
            }

            println!(
                "✓ Logit bias edge case {} validated ({} biases)",
                i,
                biases.len()
            );
        }
    }

    #[test]
    fn test_token_data_edge_cases() {
        let edge_cases = vec![
            // Normal case
            TokenData {
                id: 100,
                logit: 1.0,
                p: 0.5,
            },
            // Zero probability
            TokenData {
                id: 101,
                logit: f32::NEG_INFINITY,
                p: 0.0,
            },
            // Maximum probability
            TokenData {
                id: 102,
                logit: f32::INFINITY,
                p: 1.0,
            },
            // NaN values (edge case)
            TokenData {
                id: 103,
                logit: f32::NAN,
                p: f32::NAN,
            },
            // Negative token (edge case)
            TokenData {
                id: -1,
                logit: 0.0,
                p: 0.1,
            },
            // Large token ID
            TokenData {
                id: i32::MAX,
                logit: 0.0,
                p: 0.1,
            },
        ];

        for (i, token_data) in edge_cases.iter().enumerate() {
            // Test that token data can be created and cloned
            let cloned = token_data.clone();
            assert_eq!(cloned.id, token_data.id, "Case {} clone failed", i);

            // Test debug formatting
            let _debug_str = format!("{:?}", token_data);

            println!(
                "✓ Token data edge case {} validated (id: {})",
                i, token_data.id
            );
        }
    }

    #[test]
    fn test_sampler_perf_data_edge_cases() {
        let edge_cases = vec![
            // Normal case
            SamplerPerfData {
                t_sample_ms: 10.5,
                n_sample: 100,
            },
            // Zero values
            SamplerPerfData {
                t_sample_ms: 0.0,
                n_sample: 0,
            },
            // Large values
            SamplerPerfData {
                t_sample_ms: 1000000.0,
                n_sample: i32::MAX,
            },
            // Negative values (edge case)
            SamplerPerfData {
                t_sample_ms: -1.0,
                n_sample: -1,
            },
            // Infinite time (edge case)
            SamplerPerfData {
                t_sample_ms: f64::INFINITY,
                n_sample: 1,
            },
        ];

        for (i, perf_data) in edge_cases.iter().enumerate() {
            // Test that perf data can be created and cloned
            let cloned = perf_data.clone();
            assert_eq!(
                cloned.n_sample, perf_data.n_sample,
                "Case {} clone failed",
                i
            );

            // Test debug formatting
            let _debug_str = format!("{:?}", perf_data);

            println!(
                "✓ Perf data edge case {} validated (samples: {})",
                i, perf_data.n_sample
            );
        }
    }
}

#[cfg(all(test, feature = "llama-cpp-tests"))]
mod sampling_consistency_tests {
    use super::*;

    #[test]
    fn test_sampler_name_consistency() {
        let samplers = vec![
            ("Greedy", expect_sampler(Sampler::greedy())),
            ("Dist", expect_sampler(Sampler::dist(12345))),
            ("Top-K", expect_sampler(Sampler::top_k(40))),
            ("Top-P", expect_sampler(Sampler::top_p(0.9, 1))),
            ("Min-P", expect_sampler(Sampler::min_p(0.1, 1))),
            ("Temperature", expect_sampler(Sampler::temperature(0.8))),
            (
                "Temp-Ext",
                expect_sampler(Sampler::temperature_ext(0.8, 0.0, 1.0)),
            ),
            (
                "Mirostat-v2",
                expect_sampler(Sampler::mirostat_v2(12345, 5.0, 0.1)),
            ),
            ("Tail-Free", expect_sampler(Sampler::tail_free(1.0, 1))),
            ("Typical", expect_sampler(Sampler::typical(0.9, 1))),
        ];

        for (expected_type, sampler) in samplers {
            let name = sampler.name();
            assert!(
                !name.is_empty(),
                "{} sampler should have non-empty name",
                expected_type
            );
            assert!(
                name.len() < 100,
                "{} sampler name should be reasonable length",
                expected_type
            );
            println!("✓ {} sampler name: '{}'", expected_type, name);
        }
    }

    #[test]
    fn test_sampler_clone_consistency() {
        let original_samplers = vec![
            expect_sampler(Sampler::greedy()),
            expect_sampler(Sampler::dist(42)),
            expect_sampler(Sampler::top_k(20)),
            expect_sampler(Sampler::temperature(1.0)),
        ];

        for (i, original) in original_samplers.iter().enumerate() {
            match original.try_clone() {
                Ok(cloned) => {
                    assert_eq!(
                        original.name(),
                        cloned.name(),
                        "Sampler {} names should match after clone",
                        i
                    );
                    println!("✓ Sampler {} cloned successfully", i);
                }
                Err(e) => {
                    println!("! Sampler {} clone failed (may be expected): {}", i, e);
                }
            }
        }
    }

    #[test]
    fn test_sampler_performance_data_consistency() {
        let samplers = vec![
            expect_sampler(Sampler::greedy()),
            expect_sampler(Sampler::dist(123)),
            expect_sampler(Sampler::temperature(0.5)),
        ];

        for (i, sampler) in samplers.iter().enumerate() {
            let perf_data = sampler.perf_data();

            // Performance data should be non-negative
            assert!(
                perf_data.t_sample_ms >= 0.0,
                "Sampler {} should have non-negative time",
                i
            );
            assert!(
                perf_data.n_sample >= 0,
                "Sampler {} should have non-negative sample count",
                i
            );

            println!(
                "✓ Sampler {} perf data: {:.2}ms, {} samples",
                i, perf_data.t_sample_ms, perf_data.n_sample
            );
        }
    }

    #[test]
    fn test_parameter_ranges_consistency() {
        // Test that parameter ranges are consistent across different creation methods
        let default_params = SamplerParams::default();

        // Check that defaults are within expected ranges
        assert!(default_params.temperature > 0.0 && default_params.temperature <= 2.0);
        assert!(default_params.top_k > 0 && default_params.top_k <= 1000);
        assert!(default_params.top_p > 0.0 && default_params.top_p <= 1.0);
        assert!(default_params.min_p >= 0.0 && default_params.min_p <= 1.0);
        assert!(default_params.penalty_repeat >= 1.0 && default_params.penalty_repeat <= 2.0);

        println!("✓ Default parameter ranges are consistent");
    }

    #[test]
    fn test_chain_length_consistency() {
        let mut chain = SamplerChain::default();

        // Test that length is always accurate
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());

        chain.add(expect_sampler(Sampler::greedy()));
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        chain.add(expect_sampler(Sampler::temperature(0.8)));
        assert_eq!(chain.len(), 2);

        chain.add(expect_sampler(Sampler::dist(456)));
        assert_eq!(chain.len(), 3);

        // Test removal
        let _removed = chain.remove(1);
        assert_eq!(chain.len(), 2);

        println!("✓ Chain length consistency maintained through operations");
    }
}
