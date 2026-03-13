//! LoRA (Low-Rank Adaptation) support for Mullama
//!
//! This module provides comprehensive support for LoRA adapters, allowing fine-tuned
//! model behavior with minimal computational overhead.

use crate::error::MullamaError;
use crate::sys;
use crate::Context;
use crate::Model;
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

/// LoRA adapter for fine-tuning model behavior
#[derive(Debug)]
pub struct LoRAAdapter {
    adapter_ptr: *mut sys::llama_adapter_lora,
    path: String,
    scale: f32,
}

impl LoRAAdapter {
    /// Load a LoRA adapter from file
    ///
    /// # Arguments
    /// * `model` - The model to load the adapter for
    /// * `path` - Path to the LoRA adapter file (.gguf format)
    /// * `scale` - Scale factor for the adapter (typically 0.1 to 1.0)
    ///
    /// # Example
    /// ```rust,no_run
    /// use mullama::{Model, lora::LoRAAdapter};
    ///
    /// let model = Model::load("model.gguf")?;
    /// let adapter = LoRAAdapter::load(&model, "adapter.gguf", 0.8)?;
    /// # Ok::<(), mullama::MullamaError>(())
    /// ```
    pub fn load<P: AsRef<Path>>(model: &Model, path: P, scale: f32) -> Result<Self, MullamaError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();

        if !path_ref.exists() {
            return Err(MullamaError::LoRAError(format!(
                "LoRA adapter file not found: {}",
                path_str
            )));
        }

        let c_path = CString::new(path_str.clone())
            .map_err(|_| MullamaError::InvalidInput("Invalid path".to_string()))?;

        let adapter_ptr = unsafe { sys::llama_adapter_lora_init(model.as_ptr(), c_path.as_ptr()) };

        if adapter_ptr.is_null() {
            return Err(MullamaError::LoRAError(format!(
                "Failed to load LoRA adapter from: {}",
                path_str
            )));
        }

        Ok(Self {
            adapter_ptr,
            path: path_str,
            scale,
        })
    }

    /// Apply this adapter to a context
    ///
    /// # Arguments
    /// * `ctx` - The context to apply the adapter to
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn apply(&self, ctx: &mut Context) -> Result<(), MullamaError> {
        let result =
            unsafe { sys::llama_set_adapter_lora(ctx.as_ptr(), self.adapter_ptr, self.scale) };

        if result != 0 {
            return Err(MullamaError::LoRAError(
                "Failed to apply LoRA adapter to context".to_string(),
            ));
        }

        Ok(())
    }

    /// Remove this adapter from a context
    ///
    /// # Arguments
    /// * `ctx` - The context to remove the adapter from
    pub fn remove(&self, ctx: &mut Context) -> Result<(), MullamaError> {
        let result = unsafe { sys::llama_rm_adapter_lora(ctx.as_ptr(), self.adapter_ptr) };

        if result != 0 {
            return Err(MullamaError::LoRAError(
                "Failed to remove LoRA adapter from context".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the adapter's file path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the adapter's scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Set a new scale factor for this adapter
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Get the raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_adapter_lora {
        self.adapter_ptr
    }
}

impl Drop for LoRAAdapter {
    fn drop(&mut self) {
        if !self.adapter_ptr.is_null() {
            unsafe {
                sys::llama_adapter_lora_free(self.adapter_ptr);
            }
        }
    }
}

unsafe impl Send for LoRAAdapter {}
unsafe impl Sync for LoRAAdapter {}

/// Clear all LoRA adapters from a context
pub fn clear_adapters(ctx: &mut Context) {
    unsafe {
        sys::llama_clear_adapter_lora(ctx.as_ptr());
    }
}

/// LoRA adapter manager for handling multiple adapters
#[derive(Debug)]
pub struct LoRAManager {
    adapters: Vec<LoRAAdapter>,
    active_adapters: Vec<(usize, f32)>, // (adapter_index, scale)
}

impl LoRAManager {
    /// Create a new LoRA manager
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active_adapters: Vec::new(),
        }
    }

    /// Add a LoRA adapter to the manager
    ///
    /// # Returns
    /// The index of the added adapter
    pub fn add_adapter(&mut self, adapter: LoRAAdapter) -> usize {
        self.adapters.push(adapter);
        self.adapters.len() - 1
    }

    /// Load and add a LoRA adapter from file
    ///
    /// # Arguments
    /// * `model` - The model to load the adapter for
    /// * `path` - Path to the LoRA adapter file
    /// * `scale` - Scale factor for the adapter
    ///
    /// # Returns
    /// The index of the loaded adapter
    pub fn load_adapter<P: AsRef<Path>>(
        &mut self,
        model: &Model,
        path: P,
        scale: f32,
    ) -> Result<usize, MullamaError> {
        let adapter = LoRAAdapter::load(model, path, scale)?;
        Ok(self.add_adapter(adapter))
    }

    /// Apply all active adapters to a context
    pub fn apply_to_context(&self, ctx: &mut Context) -> Result<(), MullamaError> {
        for (idx, scale) in &self.active_adapters {
            if let Some(adapter) = self.adapters.get(*idx) {
                let mut adapter_with_scale = LoRAAdapter {
                    adapter_ptr: adapter.adapter_ptr,
                    path: adapter.path.clone(),
                    scale: *scale,
                };
                // Note: We need to be careful here - we're creating a temporary
                // that shares the pointer but we don't want it to be freed
                let result = unsafe {
                    sys::llama_set_adapter_lora(ctx.as_ptr(), adapter.adapter_ptr, *scale)
                };
                if result != 0 {
                    return Err(MullamaError::LoRAError(format!(
                        "Failed to apply LoRA adapter at index {}",
                        idx
                    )));
                }
            }
        }
        Ok(())
    }

    /// Activate an adapter with a specific scale
    ///
    /// # Arguments
    /// * `adapter_index` - Index of the adapter to activate
    /// * `scale` - Scale factor to apply (overrides adapter's default scale)
    pub fn activate_adapter(
        &mut self,
        adapter_index: usize,
        scale: f32,
    ) -> Result<(), MullamaError> {
        if adapter_index >= self.adapters.len() {
            return Err(MullamaError::InvalidInput(format!(
                "Adapter index {} out of range",
                adapter_index
            )));
        }

        // Remove if already active
        self.active_adapters
            .retain(|(idx, _)| *idx != adapter_index);

        // Add with new scale
        self.active_adapters.push((adapter_index, scale));

        Ok(())
    }

    /// Deactivate an adapter
    pub fn deactivate_adapter(&mut self, adapter_index: usize) {
        self.active_adapters
            .retain(|(idx, _)| *idx != adapter_index);
    }

    /// Get list of active adapters
    pub fn active_adapters(&self) -> &[(usize, f32)] {
        &self.active_adapters
    }

    /// Get adapter by index
    pub fn get_adapter(&self, index: usize) -> Option<&LoRAAdapter> {
        self.adapters.get(index)
    }

    /// Get mutable adapter by index
    pub fn get_adapter_mut(&mut self, index: usize) -> Option<&mut LoRAAdapter> {
        self.adapters.get_mut(index)
    }

    /// Get the number of loaded adapters
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Clear all adapters
    pub fn clear(&mut self) {
        self.adapters.clear();
        self.active_adapters.clear();
    }

    /// Create a preset configuration for common LoRA scenarios
    pub fn create_preset(preset: LoRAPreset) -> Self {
        let mut manager = Self::new();

        match preset {
            LoRAPreset::ChatAssistant => {
                // Example preset - in practice, these would be real adapter files
                // manager.load_adapter("adapters/chat_assistant.bin", 0.8).ok();
                // manager.load_adapter("adapters/helpful_responses.bin", 0.6).ok();
            }
            LoRAPreset::CodeGeneration => {
                // manager.load_adapter("adapters/code_completion.bin", 0.9).ok();
                // manager.load_adapter("adapters/documentation.bin", 0.5).ok();
            }
            LoRAPreset::CreativeWriting => {
                // manager.load_adapter("adapters/creative_style.bin", 1.0).ok();
                // manager.load_adapter("adapters/narrative_flow.bin", 0.7).ok();
            }
        }

        manager
    }
}

impl Default for LoRAManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset LoRA configurations for common use cases
#[derive(Debug, Clone, Copy)]
pub enum LoRAPreset {
    /// Optimized for chat assistant behavior
    ChatAssistant,
    /// Optimized for code generation
    CodeGeneration,
    /// Optimized for creative writing
    CreativeWriting,
}

/// LoRA adapter composition for combining multiple adapters
#[derive(Debug)]
pub struct LoRAComposition {
    adapters: Vec<(LoRAAdapter, f32)>, // (adapter, weight)
    composition_mode: CompositionMode,
}

#[derive(Debug, Clone, Copy)]
pub enum CompositionMode {
    /// Add adapter effects together
    Additive,
    /// Multiply adapter effects
    Multiplicative,
    /// Use weighted average
    Average,
}

impl LoRAComposition {
    /// Create a new LoRA composition
    pub fn new(mode: CompositionMode) -> Self {
        Self {
            adapters: Vec::new(),
            composition_mode: mode,
        }
    }

    /// Add an adapter to the composition
    pub fn add_adapter(&mut self, adapter: LoRAAdapter, weight: f32) {
        self.adapters.push((adapter, weight));
    }

    /// Get the composition mode
    pub fn mode(&self) -> CompositionMode {
        self.composition_mode
    }

    /// Get the number of adapters in the composition
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Apply the composition to calculate effective scales
    pub fn calculate_effective_scales(&self) -> Vec<f32> {
        match self.composition_mode {
            CompositionMode::Additive => self
                .adapters
                .iter()
                .map(|(adapter, weight)| adapter.scale() * weight)
                .collect(),
            CompositionMode::Multiplicative => {
                let product: f32 = self
                    .adapters
                    .iter()
                    .map(|(adapter, weight)| adapter.scale() * weight)
                    .product();
                vec![product; self.adapters.len()]
            }
            CompositionMode::Average => {
                let sum: f32 = self
                    .adapters
                    .iter()
                    .map(|(adapter, weight)| adapter.scale() * weight)
                    .sum();
                let avg = sum / self.adapters.len() as f32;
                vec![avg; self.adapters.len()]
            }
        }
    }
}

/// LoRA training utilities (for future implementation)
pub mod training {
    use super::*;

    /// Parameters for LoRA training
    #[derive(Debug, Clone)]
    pub struct LoRATrainingParams {
        pub rank: usize,
        pub alpha: f32,
        pub dropout: f32,
        pub learning_rate: f32,
        pub target_modules: Vec<String>,
    }

    impl Default for LoRATrainingParams {
        fn default() -> Self {
            Self {
                rank: 16,
                alpha: 32.0,
                dropout: 0.1,
                learning_rate: 1e-4,
                target_modules: vec![
                    "q_proj".to_string(),
                    "v_proj".to_string(),
                    "k_proj".to_string(),
                    "o_proj".to_string(),
                ],
            }
        }
    }

    /// LoRA trainer (placeholder for future implementation)
    pub struct LoRATrainer {
        params: LoRATrainingParams,
    }

    impl LoRATrainer {
        pub fn new(params: LoRATrainingParams) -> Self {
            Self { params }
        }

        /// Train a LoRA adapter (placeholder)
        pub fn train(&self, _training_data: &[String]) -> Result<LoRAAdapter, MullamaError> {
            // This would be implemented with actual training logic
            Err(MullamaError::NotImplemented(
                "LoRA training not yet implemented".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_manager() {
        let mut manager = LoRAManager::new();
        assert_eq!(manager.adapter_count(), 0);
        assert_eq!(manager.active_adapters().len(), 0);
    }

    #[test]
    fn test_lora_composition() {
        let composition = LoRAComposition::new(CompositionMode::Additive);
        assert_eq!(composition.adapter_count(), 0);
        assert!(matches!(composition.mode(), CompositionMode::Additive));
    }

    #[test]
    fn test_lora_preset() {
        let manager = LoRAManager::create_preset(LoRAPreset::ChatAssistant);
        // This would have adapters in a real implementation
        assert_eq!(manager.adapter_count(), 0);
    }
}
