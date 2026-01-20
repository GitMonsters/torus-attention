//! # Model Checkpointing
//!
//! Save and load Torus LLM model weights and configurations.
//!
//! ## File Format
//!
//! Checkpoints are stored as a directory containing:
//! - `config.json` - Model configuration
//! - `model.safetensors` - Model weights in safetensors format
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::checkpoint::{save_checkpoint, load_checkpoint};
//!
//! // Save a trained model
//! save_checkpoint(&model.config(), &varmap, "checkpoints/my-model", None, None)?;
//!
//! // Load it back
//! let (model, varmap, metadata) = load_checkpoint("checkpoints/my-model", &device)?;
//! ```

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::fs;
use std::path::Path;

use crate::error::TorusError;
use crate::llm::{TorusLLM, TorusLLMConfig};
use crate::TorusResult;

/// Metadata stored with checkpoints
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Model configuration
    pub config: TorusLLMConfig,
    /// Training step when saved
    pub step: Option<usize>,
    /// Training loss when saved
    pub loss: Option<f64>,
    /// Timestamp
    pub timestamp: String,
    /// Version info
    pub version: String,
}

impl CheckpointMetadata {
    pub fn new(config: TorusLLMConfig) -> Self {
        Self {
            config,
            step: None,
            loss: None,
            timestamp: chrono_timestamp(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn with_training_info(mut self, step: usize, loss: f64) -> Self {
        self.step = Some(step);
        self.loss = Some(loss);
        self
    }
}

/// Get current timestamp as string
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", secs)
}

/// Save model checkpoint to a directory
///
/// Creates the directory structure:
/// ```text
/// checkpoint_dir/
/// ├── config.json      # Model configuration and metadata
/// └── model.safetensors # Model weights
/// ```
pub fn save_checkpoint(
    config: &TorusLLMConfig,
    varmap: &VarMap,
    checkpoint_dir: impl AsRef<Path>,
    step: Option<usize>,
    loss: Option<f64>,
) -> TorusResult<()> {
    let dir = checkpoint_dir.as_ref();

    // Create directory
    fs::create_dir_all(dir)
        .map_err(|e| TorusError::Io(format!("Failed to create checkpoint dir: {}", e)))?;

    // Save config
    let mut metadata = CheckpointMetadata::new(config.clone());
    if let (Some(s), Some(l)) = (step, loss) {
        metadata = metadata.with_training_info(s, l);
    }

    let config_path = dir.join("config.json");
    let config_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| TorusError::SerializationError(e.to_string()))?;
    fs::write(&config_path, config_json)
        .map_err(|e| TorusError::Io(format!("Failed to write config: {}", e)))?;

    // Save model weights using VarMap's built-in save method
    let model_path = dir.join("model.safetensors");
    varmap
        .save(&model_path)
        .map_err(|e| TorusError::SerializationError(format!("Failed to save model: {}", e)))?;

    log::info!("Saved checkpoint to {:?}", dir);
    Ok(())
}

/// Load model checkpoint from a directory
pub fn load_checkpoint(
    checkpoint_dir: impl AsRef<Path>,
    device: &Device,
) -> TorusResult<(TorusLLM, VarMap, CheckpointMetadata)> {
    let dir = checkpoint_dir.as_ref();

    // Load config
    let config_path = dir.join("config.json");
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| TorusError::Io(format!("Failed to read config: {}", e)))?;
    let metadata: CheckpointMetadata = serde_json::from_str(&config_str)
        .map_err(|e| TorusError::SerializationError(e.to_string()))?;

    // Create VarMap and model structure first (this initializes all variables)
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TorusLLM::new(metadata.config.clone(), vb)?;

    // Load the saved weights into the varmap
    let model_path = dir.join("model.safetensors");
    let mut varmap_mut = varmap;
    varmap_mut
        .load(&model_path)
        .map_err(|e| TorusError::SerializationError(format!("Failed to load model: {}", e)))?;

    log::info!("Loaded checkpoint from {:?}", dir);
    Ok((model, varmap_mut, metadata))
}

/// Check if a checkpoint exists
pub fn checkpoint_exists(checkpoint_dir: impl AsRef<Path>) -> bool {
    let dir = checkpoint_dir.as_ref();
    dir.join("config.json").exists() && dir.join("model.safetensors").exists()
}

/// List available checkpoints in a directory
pub fn list_checkpoints(
    checkpoints_root: impl AsRef<Path>,
) -> TorusResult<Vec<(String, CheckpointMetadata)>> {
    let root = checkpoints_root.as_ref();
    let mut checkpoints = Vec::new();

    if !root.exists() {
        return Ok(checkpoints);
    }

    for entry in fs::read_dir(root).map_err(|e| TorusError::Io(e.to_string()))? {
        let entry = entry.map_err(|e| TorusError::Io(e.to_string()))?;
        let path = entry.path();

        if path.is_dir() && checkpoint_exists(&path) {
            let config_path = path.join("config.json");
            if let Ok(config_str) = fs::read_to_string(&config_path) {
                if let Ok(metadata) = serde_json::from_str::<CheckpointMetadata>(&config_str) {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    checkpoints.push((name, metadata));
                }
            }
        }
    }

    // Sort by step (most recent first)
    checkpoints.sort_by(|a, b| b.1.step.cmp(&a.1.step));

    Ok(checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_metadata() {
        let config = TorusLLMConfig::tiny();
        let metadata = CheckpointMetadata::new(config).with_training_info(1000, 0.5);

        assert_eq!(metadata.step, Some(1000));
        assert_eq!(metadata.loss, Some(0.5));
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let device = Device::Cpu;
        let config = TorusLLMConfig::tiny();

        // Create model
        let (_model, varmap) = TorusLLM::new_random(config.clone(), &device).unwrap();

        // Save checkpoint
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("test-checkpoint");

        save_checkpoint(&config, &varmap, &checkpoint_path, Some(100), Some(0.25)).unwrap();

        // Verify files exist
        assert!(checkpoint_exists(&checkpoint_path));

        // Load checkpoint
        let (loaded_model, _loaded_varmap, metadata) =
            load_checkpoint(&checkpoint_path, &device).unwrap();

        assert_eq!(metadata.step, Some(100));
        assert_eq!(metadata.loss, Some(0.25));
        assert_eq!(loaded_model.config().vocab_size, config.vocab_size);
    }
}
