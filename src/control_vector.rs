//! Control vectors for steering model behavior
//!
//! Control vectors provide a powerful mechanism for guiding model output
//! without retraining, allowing fine-grained control over generation style,
//! content, and behavior patterns.

use crate::error::MullamaError;
use crate::sys;
use crate::Context;
use crate::Model;
use std::collections::HashMap;
use std::path::Path;

/// A control vector that can influence model behavior
#[derive(Debug, Clone)]
pub struct ControlVector {
    /// The vector values for each layer
    layers: Vec<LayerVector>,
    /// Metadata about the control vector
    metadata: ControlVectorMetadata,
    /// Current strength/scale factor
    strength: f32,
}

/// Vector data for a specific model layer
#[derive(Debug, Clone)]
pub struct LayerVector {
    /// Layer index in the model
    layer_index: usize,
    /// Vector values (typically matches embedding dimension)
    values: Vec<f32>,
    /// Layer-specific scaling factor
    layer_scale: f32,
}

/// Metadata about a control vector
#[derive(Debug, Clone)]
pub struct ControlVectorMetadata {
    /// Human-readable name for the control vector
    pub name: String,
    /// Description of what this vector controls
    pub description: String,
    /// Recommended strength range
    pub recommended_strength: (f32, f32),
    /// Embedding dimension this vector was created for
    pub embedding_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Version of the control vector format
    pub version: String,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl ControlVector {
    /// Create a new control vector
    pub fn new(name: String, description: String, embedding_dim: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|i| LayerVector {
                layer_index: i,
                values: vec![0.0; embedding_dim],
                layer_scale: 1.0,
            })
            .collect();

        let metadata = ControlVectorMetadata {
            name,
            description,
            recommended_strength: (0.1, 2.0),
            embedding_dim,
            num_layers,
            version: "1.0".to_string(),
            custom: HashMap::new(),
        };

        Self {
            layers,
            metadata,
            strength: 1.0,
        }
    }

    /// Load a control vector from file
    ///
    /// Supports multiple formats:
    /// - `.json` - JSON format with metadata
    /// - `.npz` - NumPy compressed format
    /// - `.safetensors` - SafeTensors format
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        match extension.to_lowercase().as_str() {
            "json" => Self::load_json(path),
            "npz" => Self::load_npz(path),
            "safetensors" => Self::load_safetensors(path),
            _ => Err(MullamaError::InvalidInput(format!(
                "Unsupported control vector format: {}",
                extension
            ))),
        }
    }

    /// Save control vector to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), MullamaError> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json");

        match extension.to_lowercase().as_str() {
            "json" => self.save_json(path),
            "npz" => self.save_npz(path),
            "safetensors" => self.save_safetensors(path),
            _ => self.save_json(path), // Default to JSON
        }
    }

    /// Create a control vector from difference between positive and negative examples
    ///
    /// This is the primary method for creating control vectors from training data
    pub fn from_difference(
        positive_activations: &[Vec<Vec<f32>>], // [examples][layers][features]
        negative_activations: &[Vec<Vec<f32>>],
        name: String,
        description: String,
    ) -> Result<Self, MullamaError> {
        if positive_activations.is_empty() || negative_activations.is_empty() {
            return Err(MullamaError::InvalidInput(
                "Need at least one positive and negative example".to_string(),
            ));
        }

        let num_layers = positive_activations[0].len();
        let embedding_dim = positive_activations[0][0].len();

        // Validate dimensions
        for activations in positive_activations
            .iter()
            .chain(negative_activations.iter())
        {
            if activations.len() != num_layers {
                return Err(MullamaError::InvalidInput(
                    "Inconsistent number of layers in activations".to_string(),
                ));
            }
            for layer_activations in activations {
                if layer_activations.len() != embedding_dim {
                    return Err(MullamaError::InvalidInput(
                        "Inconsistent embedding dimensions in activations".to_string(),
                    ));
                }
            }
        }

        let mut control_vector = Self::new(name, description, embedding_dim, num_layers);

        // Calculate mean difference for each layer
        for layer_idx in 0..num_layers {
            // Calculate positive mean
            let mut positive_mean = vec![0.0; embedding_dim];
            for example in positive_activations {
                for (i, &val) in example[layer_idx].iter().enumerate() {
                    positive_mean[i] += val;
                }
            }
            for val in &mut positive_mean {
                *val /= positive_activations.len() as f32;
            }

            // Calculate negative mean
            let mut negative_mean = vec![0.0; embedding_dim];
            for example in negative_activations {
                for (i, &val) in example[layer_idx].iter().enumerate() {
                    negative_mean[i] += val;
                }
            }
            for val in &mut negative_mean {
                *val /= negative_activations.len() as f32;
            }

            // Calculate difference (positive - negative)
            let difference: Vec<f32> = positive_mean
                .iter()
                .zip(negative_mean.iter())
                .map(|(pos, neg)| pos - neg)
                .collect();

            control_vector.layers[layer_idx].values = difference;
        }

        // Normalize the control vector
        control_vector.normalize();

        Ok(control_vector)
    }

    /// Set the strength of the control vector
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }

    /// Get the current strength
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Get metadata
    pub fn metadata(&self) -> &ControlVectorMetadata {
        &self.metadata
    }

    /// Get layer vector for a specific layer
    pub fn get_layer(&self, layer_index: usize) -> Option<&LayerVector> {
        self.layers.get(layer_index)
    }

    /// Set layer-specific scaling
    pub fn set_layer_scale(&mut self, layer_index: usize, scale: f32) -> Result<(), MullamaError> {
        if layer_index >= self.layers.len() {
            return Err(MullamaError::InvalidInput(format!(
                "Layer index {} out of range",
                layer_index
            )));
        }
        self.layers[layer_index].layer_scale = scale;
        Ok(())
    }

    /// Normalize the control vector to unit length per layer
    pub fn normalize(&mut self) {
        for layer in &mut self.layers {
            let magnitude: f32 = layer.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for value in &mut layer.values {
                    *value /= magnitude;
                }
            }
        }
    }

    /// Scale the control vector by a factor
    pub fn scale(&mut self, factor: f32) {
        for layer in &mut self.layers {
            for value in &mut layer.values {
                *value *= factor;
            }
        }
    }

    /// Combine with another control vector
    pub fn combine_with(&mut self, other: &ControlVector, weight: f32) -> Result<(), MullamaError> {
        if self.layers.len() != other.layers.len() {
            return Err(MullamaError::InvalidInput(
                "Control vectors have different numbers of layers".to_string(),
            ));
        }

        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            if self_layer.values.len() != other_layer.values.len() {
                return Err(MullamaError::InvalidInput(
                    "Control vectors have different embedding dimensions".to_string(),
                ));
            }

            for (self_val, other_val) in self_layer.values.iter_mut().zip(other_layer.values.iter())
            {
                *self_val += other_val * weight;
            }
        }

        Ok(())
    }

    /// Get the effective vector values for a layer (applying strength and layer scale)
    pub fn get_effective_values(&self, layer_index: usize) -> Option<Vec<f32>> {
        self.layers.get(layer_index).map(|layer| {
            layer
                .values
                .iter()
                .map(|&val| val * self.strength * layer.layer_scale)
                .collect()
        })
    }

    /// Validate that this control vector is compatible with a model
    pub fn validate_compatibility(&self, model: &Model) -> Result<(), MullamaError> {
        let model_layers = model.n_layer() as usize;
        let model_embd = model.n_embd() as usize;

        if self.layers.len() != model_layers {
            return Err(MullamaError::InvalidInput(format!(
                "Control vector has {} layers, but model has {}",
                self.layers.len(),
                model_layers
            )));
        }

        if self.metadata.embedding_dim != model_embd {
            return Err(MullamaError::InvalidInput(format!(
                "Control vector has embedding dimension {}, but model has {}",
                self.metadata.embedding_dim, model_embd
            )));
        }

        Ok(())
    }

    /// Apply this control vector to a context
    ///
    /// # Arguments
    /// * `ctx` - The context to apply the control vector to
    /// * `il_start` - Start layer index (0 for all)
    /// * `il_end` - End layer index (-1 for all)
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn apply(&self, ctx: &mut Context, il_start: i32, il_end: i32) -> Result<(), MullamaError> {
        // Flatten all layer values into a single vector with strength applied
        let mut data: Vec<f32> = Vec::new();

        for layer in &self.layers {
            for &value in &layer.values {
                data.push(value * self.strength * layer.layer_scale);
            }
        }

        let result = unsafe {
            sys::llama_control_vector_apply(
                ctx.as_ptr(),
                data.as_ptr(),
                data.len(),
                self.metadata.embedding_dim as i32,
                il_start,
                il_end,
            )
        };

        if result != 0 {
            return Err(MullamaError::ControlVectorError(format!(
                "Failed to apply control vector: error code {}",
                result
            )));
        }

        Ok(())
    }

    /// Apply this control vector to all layers of a context
    pub fn apply_all(&self, ctx: &mut Context) -> Result<(), MullamaError> {
        self.apply(ctx, 0, -1)
    }

    /// Load from JSON format
    fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MullamaError::ControlVectorError(format!("Failed to read file: {}", e)))?;

        let json_data: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to parse JSON: {}", e))
        })?;

        // Parse metadata
        let metadata_json = json_data.get("metadata").ok_or_else(|| {
            MullamaError::InvalidInput("Missing metadata in control vector file".to_string())
        })?;

        let metadata = ControlVectorMetadata {
            name: metadata_json
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("Unnamed")
                .to_string(),
            description: metadata_json
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            recommended_strength: (
                metadata_json
                    .get("recommended_strength")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.1) as f32,
                metadata_json
                    .get("recommended_strength")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.get(1))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(2.0) as f32,
            ),
            embedding_dim: metadata_json
                .get("embedding_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            num_layers: metadata_json
                .get("num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            version: metadata_json
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("1.0")
                .to_string(),
            custom: HashMap::new(),
        };

        // Parse layers
        let layers_json = json_data
            .get("layers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                MullamaError::InvalidInput("Missing or invalid layers data".to_string())
            })?;

        let layers: Result<Vec<LayerVector>, MullamaError> = layers_json
            .iter()
            .enumerate()
            .map(|(i, layer_json)| {
                let values: Result<Vec<f32>, MullamaError> = layer_json
                    .get("values")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        MullamaError::InvalidInput(format!("Missing values for layer {}", i))
                    })?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| {
                                MullamaError::InvalidInput(format!("Invalid value in layer {}", i))
                            })
                            .map(|f| f as f32)
                    })
                    .collect();

                Ok(LayerVector {
                    layer_index: i,
                    values: values?,
                    layer_scale: layer_json
                        .get("layer_scale")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32,
                })
            })
            .collect();

        let strength = json_data
            .get("strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        Ok(Self {
            layers: layers?,
            metadata,
            strength,
        })
    }

    /// Save to JSON format
    fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), MullamaError> {
        let json_data = serde_json::json!({
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "recommended_strength": [self.metadata.recommended_strength.0, self.metadata.recommended_strength.1],
                "embedding_dim": self.metadata.embedding_dim,
                "num_layers": self.metadata.num_layers,
                "version": self.metadata.version,
                "custom": self.metadata.custom
            },
            "strength": self.strength,
            "layers": self.layers.iter().map(|layer| serde_json::json!({
                "layer_index": layer.layer_index,
                "values": layer.values,
                "layer_scale": layer.layer_scale
            })).collect::<Vec<_>>()
        });

        let content = serde_json::to_string_pretty(&json_data).map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to serialize JSON: {}", e))
        })?;

        std::fs::write(path, content).map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load from NPZ format (placeholder)
    fn load_npz<P: AsRef<Path>>(_path: P) -> Result<Self, MullamaError> {
        Err(MullamaError::NotImplemented(
            "NPZ format loading not yet implemented".to_string(),
        ))
    }

    /// Save to NPZ format (placeholder)
    fn save_npz<P: AsRef<Path>>(&self, _path: P) -> Result<(), MullamaError> {
        Err(MullamaError::NotImplemented(
            "NPZ format saving not yet implemented".to_string(),
        ))
    }

    /// Load from SafeTensors format (placeholder)
    fn load_safetensors<P: AsRef<Path>>(_path: P) -> Result<Self, MullamaError> {
        Err(MullamaError::NotImplemented(
            "SafeTensors format loading not yet implemented".to_string(),
        ))
    }

    /// Save to SafeTensors format (placeholder)
    fn save_safetensors<P: AsRef<Path>>(&self, _path: P) -> Result<(), MullamaError> {
        Err(MullamaError::NotImplemented(
            "SafeTensors format saving not yet implemented".to_string(),
        ))
    }
}

/// Manager for multiple control vectors
#[derive(Debug)]
pub struct ControlVectorManager {
    vectors: HashMap<String, ControlVector>,
    active_vectors: Vec<(String, f32)>, // (name, strength)
}

impl ControlVectorManager {
    /// Create a new control vector manager
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            active_vectors: Vec::new(),
        }
    }

    /// Add a control vector
    pub fn add_vector(&mut self, name: String, vector: ControlVector) {
        self.vectors.insert(name, vector);
    }

    /// Load and add a control vector from file
    pub fn load_vector<P: AsRef<Path>>(
        &mut self,
        name: String,
        path: P,
    ) -> Result<(), MullamaError> {
        let vector = ControlVector::load(path)?;
        self.vectors.insert(name, vector);
        Ok(())
    }

    /// Activate a control vector with specified strength
    pub fn activate(&mut self, name: String, strength: f32) -> Result<(), MullamaError> {
        if !self.vectors.contains_key(&name) {
            return Err(MullamaError::InvalidInput(format!(
                "Control vector '{}' not found",
                name
            )));
        }

        // Remove if already active
        self.active_vectors.retain(|(n, _)| n != &name);

        // Add with new strength
        self.active_vectors.push((name, strength));

        Ok(())
    }

    /// Deactivate a control vector
    pub fn deactivate(&mut self, name: &str) {
        self.active_vectors.retain(|(n, _)| n != name);
    }

    /// Get a control vector by name
    pub fn get_vector(&self, name: &str) -> Option<&ControlVector> {
        self.vectors.get(name)
    }

    /// Get a mutable reference to a control vector
    pub fn get_vector_mut(&mut self, name: &str) -> Option<&mut ControlVector> {
        self.vectors.get_mut(name)
    }

    /// Get all active vectors
    pub fn active_vectors(&self) -> &[(String, f32)] {
        &self.active_vectors
    }

    /// Calculate combined control vector for a specific layer
    pub fn get_combined_vector(&self, layer_index: usize) -> Option<Vec<f32>> {
        if self.active_vectors.is_empty() {
            return None;
        }

        let first_vector = self.vectors.get(&self.active_vectors[0].0)?;
        let embedding_dim = first_vector.metadata.embedding_dim;

        let mut combined = vec![0.0; embedding_dim];

        for (name, strength) in &self.active_vectors {
            if let Some(vector) = self.vectors.get(name) {
                if let Some(layer_values) = vector.get_effective_values(layer_index) {
                    for (i, &value) in layer_values.iter().enumerate() {
                        if i < combined.len() {
                            combined[i] += value * strength;
                        }
                    }
                }
            }
        }

        Some(combined)
    }

    /// Validate all vectors against a model
    pub fn validate_compatibility(&self, model: &Model) -> Result<(), MullamaError> {
        for (name, vector) in &self.vectors {
            vector
                .validate_compatibility(model)
                .map_err(|e| MullamaError::InvalidInput(format!("Vector '{}': {}", name, e)))?;
        }
        Ok(())
    }

    /// Clear all vectors
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.active_vectors.clear();
    }

    /// Get vector names
    pub fn vector_names(&self) -> Vec<&String> {
        self.vectors.keys().collect()
    }
}

impl Default for ControlVectorManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for creating control vectors
pub mod utils {
    use super::*;

    /// Create a control vector for encouraging specific behavior
    pub fn create_behavior_vector(
        behavior_description: &str,
        embedding_dim: usize,
        num_layers: usize,
        intensity: f32,
    ) -> ControlVector {
        let mut vector = ControlVector::new(
            format!("behavior_{}", behavior_description.replace(' ', "_")),
            format!("Encourages {}", behavior_description),
            embedding_dim,
            num_layers,
        );

        // In a real implementation, this would be based on training data
        // For now, we'll create a simple pattern
        for layer in &mut vector.layers {
            for (i, value) in layer.values.iter_mut().enumerate() {
                *value =
                    (intensity * ((i as f32).sin() + (layer.layer_index as f32).cos())) / 100.0;
            }
        }

        vector.normalize();
        vector
    }

    /// Create a control vector for discouraging specific behavior
    pub fn create_anti_behavior_vector(
        behavior_description: &str,
        embedding_dim: usize,
        num_layers: usize,
        intensity: f32,
    ) -> ControlVector {
        let mut vector =
            create_behavior_vector(behavior_description, embedding_dim, num_layers, intensity);
        vector.scale(-1.0); // Invert to discourage behavior
        vector.metadata.name = format!("anti_behavior_{}", behavior_description.replace(' ', "_"));
        vector.metadata.description = format!("Discourages {}", behavior_description);
        vector
    }

    /// Create a control vector for style transfer
    pub fn create_style_vector(
        style_name: &str,
        embedding_dim: usize,
        num_layers: usize,
    ) -> ControlVector {
        ControlVector::new(
            format!("style_{}", style_name.replace(' ', "_")),
            format!("Applies {} writing style", style_name),
            embedding_dim,
            num_layers,
        )
    }
}

/// Predefined control vectors for common use cases
pub mod presets {
    use super::*;

    /// Create a helpfulness control vector
    pub fn helpful_assistant(embedding_dim: usize, num_layers: usize) -> ControlVector {
        utils::create_behavior_vector(
            "helpful and informative responses",
            embedding_dim,
            num_layers,
            1.0,
        )
    }

    /// Create a creative writing control vector
    pub fn creative_writing(embedding_dim: usize, num_layers: usize) -> ControlVector {
        utils::create_style_vector("creative and imaginative", embedding_dim, num_layers)
    }

    /// Create a technical accuracy control vector
    pub fn technical_accuracy(embedding_dim: usize, num_layers: usize) -> ControlVector {
        utils::create_behavior_vector(
            "technical accuracy and precision",
            embedding_dim,
            num_layers,
            1.2,
        )
    }

    /// Create an anti-harmful content control vector
    pub fn safety_filter(embedding_dim: usize, num_layers: usize) -> ControlVector {
        utils::create_anti_behavior_vector(
            "harmful or inappropriate content",
            embedding_dim,
            num_layers,
            2.0,
        )
    }

    /// Create a conciseness control vector
    pub fn concise_responses(embedding_dim: usize, num_layers: usize) -> ControlVector {
        utils::create_behavior_vector(
            "concise and direct communication",
            embedding_dim,
            num_layers,
            0.8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_vector_creation() {
        let vector = ControlVector::new("test".to_string(), "Test vector".to_string(), 128, 24);

        assert_eq!(vector.metadata.name, "test");
        assert_eq!(vector.metadata.embedding_dim, 128);
        assert_eq!(vector.layers.len(), 24);
        assert_eq!(vector.strength, 1.0);
    }

    #[test]
    fn test_control_vector_manager() {
        let mut manager = ControlVectorManager::new();

        let vector = ControlVector::new("test".to_string(), "Test vector".to_string(), 128, 24);

        manager.add_vector("test".to_string(), vector);
        assert_eq!(manager.vector_names().len(), 1);

        manager.activate("test".to_string(), 0.5).unwrap();
        assert_eq!(manager.active_vectors().len(), 1);
        assert_eq!(manager.active_vectors()[0].1, 0.5);
    }

    #[test]
    fn test_vector_combination() {
        let mut vector1 =
            ControlVector::new("test1".to_string(), "Test vector 1".to_string(), 4, 2);

        let vector2 = ControlVector::new("test2".to_string(), "Test vector 2".to_string(), 4, 2);

        // Set some test values
        vector1.layers[0].values = vec![1.0, 0.0, 0.0, 0.0];
        vector1.layers[1].values = vec![0.0, 1.0, 0.0, 0.0];

        let mut vector2_modified = vector2.clone();
        vector2_modified.layers[0].values = vec![0.0, 0.0, 1.0, 0.0];
        vector2_modified.layers[1].values = vec![0.0, 0.0, 0.0, 1.0];

        vector1.combine_with(&vector2_modified, 0.5).unwrap();

        // Check that combination worked
        assert_eq!(vector1.layers[0].values[0], 1.0);
        assert_eq!(vector1.layers[0].values[2], 0.5);
    }

    #[test]
    fn test_preset_vectors() {
        let helpful = presets::helpful_assistant(128, 24);
        assert!(helpful.metadata.name.contains("helpful"));

        let creative = presets::creative_writing(128, 24);
        assert!(creative.metadata.name.contains("creative"));

        let safety = presets::safety_filter(128, 24);
        assert!(safety.metadata.name.contains("anti_behavior"));
    }
}
