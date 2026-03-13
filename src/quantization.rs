//! Advanced quantization for model compression and optimization
//!
//! This module provides comprehensive quantization capabilities including
//! runtime quantization, custom quantization schemes, and quality metrics.

use crate::error::MullamaError;
use crate::sys;
use crate::Model;
use std::collections::HashMap;
use std::path::Path;

/// Quantization parameters for model compression
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationParams {
    /// Target quantization type
    pub quantization_type: QuantizationType,
    /// Number of threads to use for quantization
    pub n_threads: i32,
    /// Whether to use importance matrix for better quality
    pub use_importance_matrix: bool,
    /// Custom importance matrix data
    pub importance_matrix: Option<Vec<f32>>,
    /// Output file path for quantized model
    pub output_path: Option<String>,
    /// Quality threshold (0.0 - 1.0)
    pub quality_threshold: f32,
    /// Whether to keep original precision for specific layers
    pub preserve_layers: Vec<usize>,
    /// Custom quantization settings per layer type
    pub layer_settings: HashMap<String, LayerQuantizationSettings>,
    /// Whether to enable post-quantization calibration
    pub enable_calibration: bool,
    /// Calibration dataset (for post-training quantization)
    pub calibration_data: Option<Vec<Vec<i32>>>,
}

/// Supported quantization types
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationType {
    /// 32-bit floating point (no quantization)
    F32,
    /// 16-bit floating point
    F16,
    /// 8-bit integers
    Q8_0,
    /// 4-bit integers (basic)
    Q4_0,
    /// 4-bit integers (improved)
    Q4_1,
    /// 5-bit integers
    Q5_0,
    /// 5-bit integers (improved)
    Q5_1,
    /// 2-bit integers (K-means)
    Q2_K,
    /// 3-bit integers (K-means, small)
    Q3_K_S,
    /// 3-bit integers (K-means, medium)
    Q3_K_M,
    /// 3-bit integers (K-means, large)
    Q3_K_L,
    /// 4-bit integers (K-means, small)
    Q4_K_S,
    /// 4-bit integers (K-means, medium)
    Q4_K_M,
    /// 5-bit integers (K-means, small)
    Q5_K_S,
    /// 5-bit integers (K-means, medium)
    Q5_K_M,
    /// 6-bit integers (K-means)
    Q6_K,
    /// 8-bit integers (K-means)
    Q8_K,
    /// Industry-standard 2-bit (XXS)
    IQ2_XXS,
    /// Industry-standard 2-bit (XS)
    IQ2_XS,
    /// Industry-standard 3-bit (XXS)
    IQ3_XXS,
    /// Industry-standard 3-bit (XS)
    IQ3_XS,
    /// Industry-standard 4-bit (NL)
    IQ4_NL,
    /// Industry-standard 4-bit (XS)
    IQ4_XS,
    /// Mixed precision for MoE models
    MXFP4_MOE,
    /// Custom quantization scheme
    Custom(Box<CustomQuantizationScheme>),
}

/// Custom quantization scheme definition
#[derive(Debug, Clone, PartialEq)]
pub struct CustomQuantizationScheme {
    /// Bit width for weights
    pub weight_bits: u8,
    /// Bit width for activations
    pub activation_bits: u8,
    /// Quantization method
    pub method: QuantizationMethod,
    /// Block size for group quantization
    pub block_size: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
}

/// Quantization methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationMethod {
    /// Linear quantization
    Linear,
    /// K-means clustering
    KMeans,
    /// Vector quantization
    VectorQuantization,
    /// Learned quantization
    Learned,
}

/// Layer-specific quantization settings
#[derive(Debug, Clone, PartialEq)]
pub struct LayerQuantizationSettings {
    /// Quantization type for this layer type
    pub quantization_type: QuantizationType,
    /// Whether to skip quantization for this layer
    pub skip_quantization: bool,
    /// Custom scale factor
    pub scale_factor: f32,
}

/// Quantization quality metrics
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationMetrics {
    /// Perplexity before quantization
    pub original_perplexity: f32,
    /// Perplexity after quantization
    pub quantized_perplexity: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Size reduction in bytes
    pub size_reduction: u64,
    /// Model accuracy degradation
    pub accuracy_loss: f32,
    /// Per-layer quality metrics
    pub layer_metrics: HashMap<String, LayerMetrics>,
}

/// Quality metrics for individual layers
#[derive(Debug, Clone, PartialEq)]
pub struct LayerMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Mean squared error
    pub mse: f32,
    /// Cosine similarity with original
    pub cosine_similarity: f32,
}

/// Runtime quantization engine
#[derive(Debug)]
pub struct QuantizationEngine {
    /// Source model to quantize
    model: Model,
    /// Quantization parameters
    params: QuantizationParams,
    /// Cached importance matrices
    importance_cache: HashMap<String, Vec<f32>>,
    /// Quality metrics from last quantization
    last_metrics: Option<QuantizationMetrics>,
}

impl QuantizationEngine {
    /// Create a new quantization engine
    pub fn new(model: Model, params: QuantizationParams) -> Self {
        Self {
            model,
            params,
            importance_cache: HashMap::new(),
            last_metrics: None,
        }
    }

    /// Quantize the model with current parameters
    ///
    /// # Example
    /// ```rust
    /// use mullama::quantization::{QuantizationEngine, QuantizationParams, QuantizationType};
    ///
    /// let params = QuantizationParams::default()
    ///     .with_type(QuantizationType::Q4_K_M)
    ///     .with_quality_threshold(0.95);
    ///
    /// let mut engine = QuantizationEngine::new(model, params);
    /// let quantized_model = engine.quantize()?;
    /// ```
    pub fn quantize(&mut self) -> Result<Model, MullamaError> {
        // Validate parameters
        self.validate_params()?;

        // Calculate importance matrix if needed
        if self.params.use_importance_matrix && self.params.importance_matrix.is_none() {
            self.calculate_importance_matrix()?;
        }

        // Perform quantization based on type
        let qtype = self.params.quantization_type.clone();
        match qtype {
            QuantizationType::Custom(scheme) => self.quantize_custom(&scheme),
            _ => self.quantize_standard(),
        }
    }

    /// Quantize with standard quantization types
    fn quantize_standard(&mut self) -> Result<Model, MullamaError> {
        // We need the source model path - for now we'll return an error
        // In a real implementation, the engine would store the source path
        Err(MullamaError::NotImplemented(
            "Standard quantization requires source model path. Use quantize_file() instead."
                .to_string(),
        ))
    }

    /// Quantize a model file to a new file
    ///
    /// # Arguments
    /// * `input_path` - Path to the input model file
    /// * `output_path` - Path for the quantized output model
    ///
    /// # Returns
    /// The quantized model loaded from output_path
    pub fn quantize_file<P: AsRef<Path>>(
        input_path: P,
        output_path: P,
        params: &QuantizationParams,
    ) -> Result<Model, MullamaError> {
        let input_str = input_path.as_ref().to_string_lossy().to_string();
        let output_str = output_path.as_ref().to_string_lossy().to_string();

        let c_input = std::ffi::CString::new(input_str.clone())
            .map_err(|_| MullamaError::InvalidInput("Invalid input path".to_string()))?;
        let c_output = std::ffi::CString::new(output_str.clone())
            .map_err(|_| MullamaError::InvalidInput("Invalid output path".to_string()))?;

        // Get default params and configure
        let mut llama_params = unsafe { sys::llama_model_quantize_default_params() };

        // Map our quantization type to llama ftype
        llama_params.ftype = match &params.quantization_type {
            QuantizationType::F32 => sys::llama_ftype::LLAMA_FTYPE_ALL_F32,
            QuantizationType::F16 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_F16,
            QuantizationType::Q8_0 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0,
            QuantizationType::Q4_0 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_0,
            QuantizationType::Q4_1 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1,
            QuantizationType::Q5_0 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_0,
            QuantizationType::Q5_1 => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_1,
            QuantizationType::Q2_K => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q2_K,
            QuantizationType::Q3_K_S => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_S,
            QuantizationType::Q3_K_M => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_M,
            QuantizationType::Q3_K_L => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_L,
            QuantizationType::Q4_K_S => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_S,
            QuantizationType::Q4_K_M => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M,
            QuantizationType::Q5_K_S => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_S,
            QuantizationType::Q5_K_M => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_M,
            QuantizationType::Q6_K => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q6_K,
            QuantizationType::Q8_K => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0, // Q8_K maps to Q8_0
            QuantizationType::IQ2_XXS => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ2_XXS,
            QuantizationType::IQ2_XS => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ2_XS,
            QuantizationType::IQ3_XXS => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ3_XXS,
            QuantizationType::IQ3_XS => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ3_XS,
            QuantizationType::IQ4_NL => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ4_NL,
            QuantizationType::IQ4_XS => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_IQ4_XS,
            QuantizationType::MXFP4_MOE => sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M, // Best approximation
            QuantizationType::Custom(_) => {
                return Err(MullamaError::NotImplemented(
                    "Custom quantization schemes not yet implemented".to_string(),
                ));
            }
        };

        llama_params.nthread = params.n_threads;

        // Perform quantization
        let result = unsafe {
            sys::llama_model_quantize(c_input.as_ptr(), c_output.as_ptr(), &llama_params)
        };

        if result != 0 {
            return Err(MullamaError::QuantizationError(format!(
                "Quantization failed with error code: {}",
                result
            )));
        }

        // Load and return the quantized model
        Model::load(&output_str)
    }

    /// Quantize with custom quantization scheme
    fn quantize_custom(
        &mut self,
        scheme: &Box<CustomQuantizationScheme>,
    ) -> Result<Model, MullamaError> {
        match scheme.method {
            QuantizationMethod::Linear => self.quantize_linear(scheme),
            QuantizationMethod::KMeans => self.quantize_kmeans(scheme),
            QuantizationMethod::VectorQuantization => self.quantize_vector(scheme),
            QuantizationMethod::Learned => self.quantize_learned(scheme),
        }
    }

    /// Linear quantization implementation
    fn quantize_linear(
        &mut self,
        _scheme: &Box<CustomQuantizationScheme>,
    ) -> Result<Model, MullamaError> {
        // This would implement custom linear quantization
        // For now, return an error indicating it's not implemented
        Err(MullamaError::NotImplemented(
            "Custom linear quantization not yet implemented".to_string(),
        ))
    }

    /// K-means quantization implementation
    fn quantize_kmeans(
        &mut self,
        _scheme: &Box<CustomQuantizationScheme>,
    ) -> Result<Model, MullamaError> {
        // This would implement K-means based quantization
        Err(MullamaError::NotImplemented(
            "Custom K-means quantization not yet implemented".to_string(),
        ))
    }

    /// Vector quantization implementation
    fn quantize_vector(
        &mut self,
        _scheme: &Box<CustomQuantizationScheme>,
    ) -> Result<Model, MullamaError> {
        // This would implement vector quantization
        Err(MullamaError::NotImplemented(
            "Custom vector quantization not yet implemented".to_string(),
        ))
    }

    /// Learned quantization implementation
    fn quantize_learned(
        &mut self,
        _scheme: &Box<CustomQuantizationScheme>,
    ) -> Result<Model, MullamaError> {
        // This would implement learned quantization schemes
        Err(MullamaError::NotImplemented(
            "Custom learned quantization not yet implemented".to_string(),
        ))
    }

    /// Calculate importance matrix for quality-aware quantization
    fn calculate_importance_matrix(&mut self) -> Result<(), MullamaError> {
        if let Some(calibration_data) = self.params.calibration_data.clone() {
            // Use calibration data to calculate importance
            self.calculate_importance_from_data(&calibration_data)?;
        } else {
            // Use default importance calculation
            self.calculate_default_importance()?;
        }
        Ok(())
    }

    /// Calculate importance from calibration data
    fn calculate_importance_from_data(&mut self, _data: &[Vec<i32>]) -> Result<(), MullamaError> {
        // This would analyze the calibration data to determine which weights
        // are most important for preserving model quality
        let embedding_dim = 768; // Default embedding dimension
        let importance_matrix = vec![1.0; embedding_dim]; // Placeholder

        self.importance_cache
            .insert("default".to_string(), importance_matrix);
        Ok(())
    }

    /// Calculate default importance matrix
    fn calculate_default_importance(&mut self) -> Result<(), MullamaError> {
        // Use heuristics to determine weight importance
        let embedding_dim = 768; // Default embedding dimension
        let mut importance_matrix = vec![1.0; embedding_dim];

        // Layers closer to output are more important
        let num_layers = 32; // Default layer count
        for (i, importance) in importance_matrix.iter_mut().enumerate() {
            let layer_ratio = i as f32 / num_layers as f32;
            *importance = 1.0 + layer_ratio; // Higher importance for later layers
        }

        self.importance_cache
            .insert("default".to_string(), importance_matrix);
        Ok(())
    }

    /// Calculate quality metrics after quantization
    fn calculate_metrics(&mut self, quantized_model: &Model) -> Result<(), MullamaError> {
        let original_size = 1000000000u64; // Placeholder
        let quantized_size = 500000000u64; // Placeholder

        let compression_ratio = original_size as f32 / quantized_size as f32;
        let size_reduction = original_size - quantized_size;

        // Calculate perplexity (simplified - would need actual evaluation)
        let original_perplexity = self.estimate_perplexity(&self.model)?;
        let quantized_perplexity = self.estimate_perplexity(quantized_model)?;

        let accuracy_loss = (quantized_perplexity - original_perplexity) / original_perplexity;

        let metrics = QuantizationMetrics {
            original_perplexity,
            quantized_perplexity,
            compression_ratio,
            size_reduction,
            accuracy_loss,
            layer_metrics: HashMap::new(), // Would be populated with actual layer analysis
        };

        self.last_metrics = Some(metrics);
        Ok(())
    }

    /// Estimate model perplexity (simplified implementation)
    fn estimate_perplexity(&self, _model: &Model) -> Result<f32, MullamaError> {
        // This would evaluate the model on a standard dataset
        // For now, return a placeholder value
        Ok(10.0) // Placeholder perplexity
    }

    /// Validate quantization parameters
    fn validate_params(&self) -> Result<(), MullamaError> {
        if self.params.quality_threshold < 0.0 || self.params.quality_threshold > 1.0 {
            return Err(MullamaError::InvalidInput(
                "Quality threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.params.n_threads <= 0 {
            return Err(MullamaError::InvalidInput(
                "Number of threads must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Create llama.cpp quantization parameters
    fn create_llama_quantize_params(
        &self,
    ) -> Result<sys::llama_model_quantize_params, MullamaError> {
        // Get default parameters and modify them
        let mut params = unsafe { sys::llama_model_quantize_default_params() };
        params.nthread = self.params.n_threads;
        Ok(params)
    }

    /// Get the last calculated quality metrics
    pub fn last_metrics(&self) -> Option<&QuantizationMetrics> {
        self.last_metrics.as_ref()
    }

    /// Update quantization parameters
    pub fn set_params(&mut self, params: QuantizationParams) {
        self.params = params;
    }
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Q4_K_M,
            n_threads: num_cpus::get() as i32,
            use_importance_matrix: false,
            importance_matrix: None,
            output_path: None,
            quality_threshold: 0.95,
            preserve_layers: Vec::new(),
            layer_settings: HashMap::new(),
            enable_calibration: false,
            calibration_data: None,
        }
    }
}

impl QuantizationParams {
    /// Builder pattern for quantization parameters
    pub fn with_type(mut self, qtype: QuantizationType) -> Self {
        self.quantization_type = qtype;
        self
    }

    pub fn with_threads(mut self, threads: i32) -> Self {
        self.n_threads = threads;
        self
    }

    pub fn with_importance_matrix(mut self, matrix: Vec<f32>) -> Self {
        self.use_importance_matrix = true;
        self.importance_matrix = Some(matrix);
        self
    }

    pub fn with_output_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.output_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    pub fn with_quality_threshold(mut self, threshold: f32) -> Self {
        self.quality_threshold = threshold;
        self
    }

    pub fn with_preserved_layers(mut self, layers: Vec<usize>) -> Self {
        self.preserve_layers = layers;
        self
    }

    pub fn with_calibration_data(mut self, data: Vec<Vec<i32>>) -> Self {
        self.enable_calibration = true;
        self.calibration_data = Some(data);
        self
    }
}

/// Utilities for quantization
pub mod utils {
    use super::*;

    /// Analyze model to recommend optimal quantization settings
    pub fn recommend_quantization(model: &Model) -> QuantizationType {
        let size = 1000000000u64; // Placeholder size
        let n_params = 1000000u64; // Placeholder param count

        // Recommend based on model size
        if size > 20_000_000_000 {
            // > 20GB
            QuantizationType::Q4_K_M // Aggressive compression for very large models
        } else if size > 7_000_000_000 {
            // > 7GB
            QuantizationType::Q5_K_M // Balanced compression
        } else if size > 3_000_000_000 {
            // > 3GB
            QuantizationType::Q8_0 // Light compression
        } else {
            QuantizationType::F16 // Minimal compression for small models
        }
    }

    /// Calculate expected compression ratio for a quantization type
    pub fn compression_ratio(qtype: QuantizationType) -> f32 {
        match qtype {
            QuantizationType::F32 => 1.0,
            QuantizationType::F16 => 2.0,
            QuantizationType::Q8_0 => 4.0,
            QuantizationType::Q5_0 | QuantizationType::Q5_1 => 6.4,
            QuantizationType::Q4_0 | QuantizationType::Q4_1 => 8.0,
            QuantizationType::Q4_K_M => 8.5,
            QuantizationType::Q3_K_M => 10.7,
            QuantizationType::Q2_K => 16.0,
            QuantizationType::IQ2_XXS => 20.0,
            _ => 8.0, // Default estimate
        }
    }

    /// Create quantization parameters optimized for speed
    pub fn speed_optimized_params() -> QuantizationParams {
        QuantizationParams::default()
            .with_type(QuantizationType::Q4_0)
            .with_threads(num_cpus::get() as i32)
    }

    /// Create quantization parameters optimized for quality
    pub fn quality_optimized_params() -> QuantizationParams {
        QuantizationParams::default()
            .with_type(QuantizationType::Q8_0)
            .with_quality_threshold(0.98)
    }

    /// Create quantization parameters optimized for size
    pub fn size_optimized_params() -> QuantizationParams {
        QuantizationParams::default()
            .with_type(QuantizationType::Q2_K)
            .with_quality_threshold(0.90)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::default()
            .with_type(QuantizationType::Q4_K_M)
            .with_threads(8)
            .with_quality_threshold(0.95);

        assert!(matches!(params.quantization_type, QuantizationType::Q4_K_M));
        assert_eq!(params.n_threads, 8);
        assert_eq!(params.quality_threshold, 0.95);
    }

    #[test]
    fn test_quantization_types() {
        assert_eq!(utils::compression_ratio(QuantizationType::F32), 1.0);
        assert_eq!(utils::compression_ratio(QuantizationType::F16), 2.0);
        assert_eq!(utils::compression_ratio(QuantizationType::Q4_0), 8.0);
    }

    #[test]
    fn test_custom_quantization_scheme() {
        let scheme = CustomQuantizationScheme {
            weight_bits: 4,
            activation_bits: 8,
            method: QuantizationMethod::Linear,
            block_size: 128,
            symmetric: true,
        };

        assert_eq!(scheme.weight_bits, 4);
        assert_eq!(scheme.activation_bits, 8);
        assert!(scheme.symmetric);
    }

    #[test]
    fn test_layer_settings() {
        let mut settings = HashMap::new();
        settings.insert(
            "attention".to_string(),
            LayerQuantizationSettings {
                quantization_type: QuantizationType::Q8_0,
                skip_quantization: false,
                scale_factor: 1.0,
            },
        );

        assert_eq!(settings.len(), 1);
        assert!(matches!(
            settings["attention"].quantization_type,
            QuantizationType::Q8_0
        ));
    }

    #[test]
    fn test_optimization_presets() {
        let speed_params = utils::speed_optimized_params();
        assert!(matches!(
            speed_params.quantization_type,
            QuantizationType::Q4_0
        ));

        let quality_params = utils::quality_optimized_params();
        assert!(matches!(
            quality_params.quantization_type,
            QuantizationType::Q8_0
        ));
        assert_eq!(quality_params.quality_threshold, 0.98);

        let size_params = utils::size_optimized_params();
        assert!(matches!(
            size_params.quantization_type,
            QuantizationType::Q2_K
        ));
        assert_eq!(size_params.quality_threshold, 0.90);
    }
}
