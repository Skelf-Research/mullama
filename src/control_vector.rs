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
#[cfg(feature = "daemon")]
use std::io::{Read, Write};
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
                    .and_then(|arr| arr.first())
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

    /// Load from NPZ format
    fn load_npz<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        #[cfg(feature = "daemon")]
        {
            let file = std::fs::File::open(path).map_err(|e| {
                MullamaError::ControlVectorError(format!("Failed to open NPZ file: {}", e))
            })?;
            let mut archive = zip::ZipArchive::new(file).map_err(|e| {
                MullamaError::ControlVectorError(format!("Failed to read NPZ archive: {}", e))
            })?;

            let mut layers = Vec::new();
            let mut embedding_dim = 0usize;
            let mut num_layers = 0usize;

            for i in 0..archive.len() {
                let mut entry = archive.by_index(i).map_err(|e| {
                    MullamaError::ControlVectorError(format!("Failed to read NPZ entry: {}", e))
                })?;
                let name = entry.name().to_string();

                if !name.ends_with(".npy") {
                    continue;
                }

                let layer_idx: usize = name.trim_end_matches(".npy").parse().unwrap_or(i);

                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(|e| {
                    MullamaError::ControlVectorError(format!(
                        "Failed to read NPZ entry data: {}",
                        e
                    ))
                })?;

                let values = parse_npy_data(&data)?;
                if embedding_dim == 0 {
                    embedding_dim = values.len();
                }
                layers.push((layer_idx, values));
            }

            if layers.is_empty() {
                return Err(MullamaError::ControlVectorError(
                    "No layer data found in NPZ file".to_string(),
                ));
            }

            layers.sort_by_key(|(idx, _)| *idx);
            let max_idx = layers.iter().map(|(idx, _)| *idx).max().unwrap_or(0);
            num_layers = max_idx + 1;

            let mut vector = Self::new(
                "npz_control_vector".to_string(),
                "Loaded from NPZ format".to_string(),
                embedding_dim,
                num_layers,
            );

            for (idx, values) in layers {
                if idx < vector.layers.len() {
                    vector.layers[idx].values = values;
                }
            }

            Ok(vector)
        }

        #[cfg(not(feature = "daemon"))]
        {
            let _ = path;
            Err(MullamaError::NotImplemented(
                "NPZ format requires 'daemon' feature (zip crate)".to_string(),
            ))
        }
    }

    /// Save to NPZ format
    fn save_npz<P: AsRef<Path>>(&self, path: P) -> Result<(), MullamaError> {
        #[cfg(feature = "daemon")]
        {
            let file = std::fs::File::create(path).map_err(|e| {
                MullamaError::ControlVectorError(format!("Failed to create NPZ file: {}", e))
            })?;
            let mut zip_writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            for layer in &self.layers {
                let entry_name = format!("{}.npy", layer.layer_index);
                zip_writer.start_file(&entry_name, options).map_err(|e| {
                    MullamaError::ControlVectorError(format!("Failed to write NPZ entry: {}", e))
                })?;

                let npy_data = build_npy_data(&layer.values);
                zip_writer.write_all(&npy_data).map_err(|e| {
                    MullamaError::ControlVectorError(format!("Failed to write NPZ data: {}", e))
                })?;
            }

            zip_writer.finish().map_err(|e| {
                MullamaError::ControlVectorError(format!("Failed to finalize NPZ archive: {}", e))
            })?;

            Ok(())
        }

        #[cfg(not(feature = "daemon"))]
        {
            let _ = path;
            Err(MullamaError::NotImplemented(
                "NPZ format requires 'daemon' feature (zip crate)".to_string(),
            ))
        }
    }

    /// Load from SafeTensors format
    fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, MullamaError> {
        let data = std::fs::read(path).map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to read SafeTensors file: {}", e))
        })?;

        if data.len() < 8 {
            return Err(MullamaError::ControlVectorError(
                "SafeTensors file too small".to_string(),
            ));
        }

        let header_size = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| MullamaError::ControlVectorError("Invalid header size".to_string()))?,
        ) as usize;

        if data.len() < 8 + header_size {
            return Err(MullamaError::ControlVectorError(
                "SafeTensors file truncated".to_string(),
            ));
        }

        let header_str = std::str::from_utf8(&data[8..8 + header_size]).map_err(|e| {
            MullamaError::ControlVectorError(format!("Invalid SafeTensors header: {}", e))
        })?;

        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to parse SafeTensors header: {}", e))
        })?;

        let body_offset = 8 + header_size;
        let mut layers = Vec::new();
        let mut embedding_dim = 0usize;

        for (tensor_name, tensor_info) in &header {
            if tensor_name == "__metadata__" {
                continue;
            }

            let info = tensor_info.as_object().ok_or_else(|| {
                MullamaError::ControlVectorError(format!(
                    "Invalid tensor info for '{}'",
                    tensor_name
                ))
            })?;

            let offsets = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    MullamaError::ControlVectorError(format!(
                        "Missing data_offsets for '{}'",
                        tensor_name
                    ))
                })?;

            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            let dtype = info.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32");

            let tensor_data = &data[body_offset + start..body_offset + end];
            let values = match dtype {
                "F32" | "BF16" | "F16" => parse_safetensor_f32(tensor_data, dtype)?,
                "F64" => parse_safetensor_f64(tensor_data)?,
                _ => {
                    return Err(MullamaError::ControlVectorError(format!(
                        "Unsupported SafeTensors dtype: {}",
                        dtype
                    )));
                }
            };

            if embedding_dim == 0 {
                embedding_dim = values.len();
            }

            let layer_idx: usize = tensor_name
                .trim_start_matches("layer_")
                .trim_start_matches("layers.")
                .parse()
                .unwrap_or(layers.len());

            layers.push((layer_idx, values));
        }

        if layers.is_empty() {
            return Err(MullamaError::ControlVectorError(
                "No tensor data found in SafeTensors file".to_string(),
            ));
        }

        layers.sort_by_key(|(idx, _)| *idx);
        let max_idx = layers.iter().map(|(idx, _)| *idx).max().unwrap_or(0);
        let num_layers = max_idx + 1;

        let mut vector = Self::new(
            "safetensors_control_vector".to_string(),
            "Loaded from SafeTensors format".to_string(),
            embedding_dim,
            num_layers,
        );

        if let Some(meta) = header.get("__metadata__") {
            if let Some(obj) = meta.as_object() {
                if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
                    vector.metadata.name = name.to_string();
                }
                if let Some(desc) = obj.get("description").and_then(|v| v.as_str()) {
                    vector.metadata.description = desc.to_string();
                }
            }
        }

        for (idx, values) in layers {
            if idx < vector.layers.len() {
                vector.layers[idx].values = values;
            }
        }

        Ok(vector)
    }

    /// Save to SafeTensors format
    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), MullamaError> {
        let mut tensors = serde_json::Map::new();

        let mut current_offset: u64 = 0;
        let mut data_offsets = Vec::new();

        for layer in &self.layers {
            let tensor_name = format!("layer_{}", layer.layer_index);
            let byte_len = (layer.values.len() * std::mem::size_of::<f32>()) as u64;
            data_offsets.push((current_offset, current_offset + byte_len));

            let mut shape = serde_json::Map::new();
            shape.insert(
                "dtype".to_string(),
                serde_json::Value::String("F32".to_string()),
            );
            shape.insert(
                "shape".to_string(),
                serde_json::json!([1, layer.values.len()]),
            );
            shape.insert(
                "data_offsets".to_string(),
                serde_json::json!([current_offset, current_offset + byte_len]),
            );

            tensors.insert(tensor_name, serde_json::Value::Object(shape));
            current_offset += byte_len;
        }

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "name".to_string(),
            serde_json::Value::String(self.metadata.name.clone()),
        );
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(self.metadata.description.clone()),
        );
        metadata.insert(
            "recommended_strength".to_string(),
            serde_json::json!([
                self.metadata.recommended_strength.0,
                self.metadata.recommended_strength.1
            ]),
        );
        tensors.insert(
            "__metadata__".to_string(),
            serde_json::Value::Object(metadata),
        );

        let header_json =
            serde_json::to_string(&serde_json::Value::Object(tensors)).map_err(|e| {
                MullamaError::ControlVectorError(format!("Failed to serialize header: {}", e))
            })?;

        let header_bytes = header_json.as_bytes();
        let header_size = (header_bytes.len() as u64).to_le_bytes();

        let mut file = std::fs::File::create(path).map_err(|e| {
            MullamaError::ControlVectorError(format!("Failed to create SafeTensors file: {}", e))
        })?;

        use std::io::Write;
        file.write_all(&header_size)
            .map_err(|e| MullamaError::ControlVectorError(format!("Write error: {}", e)))?;
        file.write_all(header_bytes)
            .map_err(|e| MullamaError::ControlVectorError(format!("Write error: {}", e)))?;

        for layer in &self.layers {
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    layer.values.as_ptr() as *const u8,
                    layer.values.len() * std::mem::size_of::<f32>(),
                )
            };
            file.write_all(byte_slice)
                .map_err(|e| MullamaError::ControlVectorError(format!("Write error: {}", e)))?;
        }

        Ok(())
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

/// Parse NPY binary data (numpy .npy format) into f32 values
fn parse_npy_data(data: &[u8]) -> Result<Vec<f32>, MullamaError> {
    let magic = b"\x93NUMPY";
    if data.len() < magic.len() || &data[..magic.len()] != magic {
        return Err(MullamaError::ControlVectorError(
            "Invalid NPY format: missing magic number".to_string(),
        ));
    }

    let version = data.get(magic.len()).copied().unwrap_or(0);
    let header_len_size = if version <= 1 { 2usize } else { 4 };

    if data.len() < magic.len() + 1 + header_len_size {
        return Err(MullamaError::ControlVectorError(
            "Invalid NPY format: truncated header".to_string(),
        ));
    }

    let header_len_offset = magic.len() + 1;
    let header_len = if version <= 1 {
        u16::from_le_bytes([data[header_len_offset], data[header_len_offset + 1]]) as usize
    } else {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[header_len_offset..header_len_offset + 4]);
        u32::from_le_bytes(bytes) as usize
    };

    let data_offset = header_len_offset + header_len_size + header_len;
    if data_offset > data.len() {
        return Err(MullamaError::ControlVectorError(
            "Invalid NPY format: data offset beyond file".to_string(),
        ));
    }

    let tensor_data = &data[data_offset..];
    let float_count = tensor_data.len() / 4;
    let mut values = Vec::with_capacity(float_count);

    for chunk in tensor_data.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        values.push(f32::from_le_bytes(bytes));
    }

    Ok(values)
}

/// Build NPY binary data from f32 values
fn build_npy_data(values: &[f32]) -> Vec<u8> {
    let dict_str = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
        values.len()
    );
    let header_str = format!("{:60}", format!("{}\n", dict_str));
    let header_bytes = header_str.as_bytes();
    let header_len = header_bytes.len() as u16;

    let mut result = Vec::with_capacity(8 + header_bytes.len() + values.len() * 4);
    result.extend_from_slice(b"\x93NUMPY");
    result.push(0x01); // version
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(header_bytes);

    for val in values {
        result.extend_from_slice(&val.to_le_bytes());
    }

    result
}

/// Parse SafeTensors tensor data as f32 (supports F32, F16, BF16)
fn parse_safetensor_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, MullamaError> {
    match dtype {
        "F32" => {
            let count = data.len() / 4;
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(values)
        }
        "F16" => {
            let count = data.len() / 2;
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let half = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                values.push(half);
            }
            Ok(values)
        }
        "BF16" => {
            let count = data.len() / 2;
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_bits = (bits as u32) << 16;
                values.push(f32::from_bits(f32_bits));
            }
            Ok(values)
        }
        _ => Err(MullamaError::ControlVectorError(format!(
            "Unsupported dtype for f32 conversion: {}",
            dtype
        ))),
    }
}

/// Parse SafeTensors tensor data as f64 -> f32
fn parse_safetensor_f64(data: &[u8]) -> Result<Vec<f32>, MullamaError> {
    let count = data.len() / 8;
    let mut values = Vec::with_capacity(count);
    for chunk in data.chunks_exact(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(bytes) as f32);
    }
    Ok(values)
}

/// Convert IEEE 754 half-precision (f16) to f32
fn half_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            let result: u32 = sign << 31;
            f32::from_bits(result)
        } else {
            let mut m = mantissa;
            let mut e: u32 = 0;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            let result = sign << 31 | (127 - 15 - e) << 23 | (m & 0x3ff) << 13;
            f32::from_bits(result)
        }
    } else if exponent == 0x1f {
        let result = sign << 31 | 0xffu32 << 23 | mantissa << 13;
        f32::from_bits(result)
    } else {
        let result = sign << 31 | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        f32::from_bits(result)
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
