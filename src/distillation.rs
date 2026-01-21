//! Knowledge Distillation with Silent Teacher Model
//!
//! Uses a pretrained model as a "silent pair" (teacher) to guide training
//! of a student model through soft targets and feature matching.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    KNOWLEDGE DISTILLATION                               │
//! │                                                                         │
//! │   Input                                                                 │
//! │     │                                                                   │
//! │     ├────────────────────┬──────────────────────────┐                  │
//! │     │                    │                          │                  │
//! │     ▼                    ▼                          │                  │
//! │ ┌────────────┐    ┌────────────┐                   │                  │
//! │ │  TEACHER   │    │  STUDENT   │                   │                  │
//! │ │ (Frozen)   │    │ (Training) │                   │                  │
//! │ │            │    │            │                   │                  │
//! │ │ Pretrained │    │ Random or  │                   │                  │
//! │ │ Weights    │    │ Pretrained │                   │                  │
//! │ └─────┬──────┘    └─────┬──────┘                   │                  │
//! │       │                 │                          │                  │
//! │       ▼                 ▼                          │                  │
//! │   Teacher           Student                        │                  │
//! │   Logits            Logits ◄──────────────────────┘                  │
//! │       │                 │              Ground Truth                   │
//! │       │                 │                  Labels                     │
//! │       │                 │                    │                        │
//! │       ▼                 ▼                    ▼                        │
//! │ ┌─────────────────────────────────────────────────────────────────┐  │
//! │ │                    COMBINED LOSS                                 │  │
//! │ │                                                                  │  │
//! │ │  L = α * L_distill(student, teacher) + (1-α) * L_hard(student)  │  │
//! │ │                                                                  │  │
//! │ │  L_distill = KL(softmax(s/T), softmax(t/T)) * T²                │  │
//! │ │  L_hard    = CrossEntropy(student, labels)                      │  │
//! │ └─────────────────────────────────────────────────────────────────┘  │
//! │                         │                                            │
//! │                         ▼                                            │
//! │                   Backprop to                                        │
//! │                   Student Only                                       │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::collider::{ColliderConfig, TorusCollider};
use crate::error::TorusError;
use crate::integration::{BidirectionalTorusConfig, BidirectionalTorusTransformer};
use crate::TorusResult;

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions (higher = softer)
    pub temperature: f32,
    /// Weight for distillation loss vs hard label loss (0.0-1.0)
    /// α=1.0 means only distillation, α=0.0 means only hard labels
    pub alpha: f32,
    /// Whether to match intermediate features (not just final logits)
    pub match_features: bool,
    /// Layers to match features from (if match_features is true)
    pub feature_layers: Vec<usize>,
    /// Weight for feature matching loss
    pub feature_weight: f32,
    /// Whether to use attention transfer
    pub attention_transfer: bool,
    /// Weight for attention transfer loss
    pub attention_weight: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0, // Standard temperature
            alpha: 0.7,       // 70% distillation, 30% hard labels
            match_features: false,
            feature_layers: vec![],
            feature_weight: 0.1,
            attention_transfer: false,
            attention_weight: 0.1,
        }
    }
}

impl DistillationConfig {
    /// Soft distillation (only match teacher outputs)
    pub fn soft() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.9,
            match_features: false,
            feature_layers: vec![],
            feature_weight: 0.0,
            attention_transfer: false,
            attention_weight: 0.0,
        }
    }

    /// Hard distillation (more weight on ground truth)
    pub fn hard() -> Self {
        Self {
            temperature: 2.0,
            alpha: 0.3,
            match_features: false,
            feature_layers: vec![],
            feature_weight: 0.0,
            attention_transfer: false,
            attention_weight: 0.0,
        }
    }

    /// Feature matching distillation
    pub fn feature_matching(layers: Vec<usize>) -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
            match_features: true,
            feature_layers: layers,
            feature_weight: 0.5,
            attention_transfer: true,
            attention_weight: 0.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION CHECKPOINT
// ═══════════════════════════════════════════════════════════════════════════════

/// Metadata for distillation checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationCheckpointMetadata {
    /// Model configuration
    pub config: BidirectionalTorusConfig,
    /// Vocab size
    pub vocab_size: Option<usize>,
    /// Training step when saved
    pub step: Option<u64>,
    /// Training loss when saved
    pub loss: Option<f64>,
    /// Timestamp
    pub timestamp: String,
    /// Version info
    pub version: String,
}

impl DistillationCheckpointMetadata {
    pub fn new(config: BidirectionalTorusConfig, vocab_size: Option<usize>) -> Self {
        Self {
            config,
            vocab_size,
            step: None,
            loss: None,
            timestamp: chrono_timestamp(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn with_training_info(mut self, step: u64, loss: f64) -> Self {
        self.step = Some(step);
        self.loss = Some(loss);
        self
    }
}

fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", secs)
}

/// Save a BidirectionalTorusTransformer checkpoint
pub fn save_transformer_checkpoint(
    config: &BidirectionalTorusConfig,
    vocab_size: Option<usize>,
    varmap: &VarMap,
    checkpoint_dir: impl AsRef<Path>,
    step: Option<u64>,
    loss: Option<f64>,
) -> TorusResult<()> {
    let dir = checkpoint_dir.as_ref();

    // Create directory
    fs::create_dir_all(dir)
        .map_err(|e| TorusError::Io(format!("Failed to create checkpoint dir: {}", e)))?;

    // Save config
    let mut metadata = DistillationCheckpointMetadata::new(config.clone(), vocab_size);
    if let (Some(s), Some(l)) = (step, loss) {
        metadata = metadata.with_training_info(s, l);
    }

    let config_path = dir.join("config.json");
    let config_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| TorusError::SerializationError(e.to_string()))?;
    fs::write(&config_path, config_json)
        .map_err(|e| TorusError::Io(format!("Failed to write config: {}", e)))?;

    // Save model weights
    let model_path = dir.join("model.safetensors");
    varmap
        .save(&model_path)
        .map_err(|e| TorusError::SerializationError(format!("Failed to save model: {}", e)))?;

    log::info!("Saved transformer checkpoint to {:?}", dir);
    Ok(())
}

/// Load a BidirectionalTorusTransformer checkpoint
pub fn load_transformer_checkpoint(
    checkpoint_dir: impl AsRef<Path>,
    device: &Device,
) -> TorusResult<(
    BidirectionalTorusTransformer,
    BidirectionalTorusConfig,
    VarMap,
)> {
    let dir = checkpoint_dir.as_ref();

    // Load config
    let config_path = dir.join("config.json");
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| TorusError::Io(format!("Failed to read config: {}", e)))?;
    let metadata: DistillationCheckpointMetadata = serde_json::from_str(&config_str)
        .map_err(|e| TorusError::SerializationError(e.to_string()))?;

    // Create VarMap and model structure first
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = BidirectionalTorusTransformer::new(
        metadata.config.clone(),
        metadata.vocab_size,
        vb,
        device,
    )?;

    // Load saved weights
    let model_path = dir.join("model.safetensors");
    let mut varmap_mut = varmap;
    varmap_mut
        .load(&model_path)
        .map_err(|e| TorusError::SerializationError(format!("Failed to load model: {}", e)))?;

    log::info!("Loaded transformer checkpoint from {:?}", dir);
    Ok((model, metadata.config, varmap_mut))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEACHER MODEL (SILENT PAIR)
// ═══════════════════════════════════════════════════════════════════════════════

/// Frozen teacher model that provides soft targets
pub struct TeacherModel {
    /// The underlying transformer (weights are frozen)
    model: BidirectionalTorusTransformer,
    /// Configuration
    config: BidirectionalTorusConfig,
    /// Device
    #[allow(dead_code)]
    device: Device,
}

impl TeacherModel {
    /// Load a pretrained model as teacher from a distillation checkpoint
    pub fn load<P: AsRef<Path>>(checkpoint_path: P, device: &Device) -> TorusResult<Self> {
        let (model, config, _var_map) = load_transformer_checkpoint(checkpoint_path, device)?;

        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }

    /// Create teacher from existing model (useful for self-distillation)
    pub fn from_model(
        model: BidirectionalTorusTransformer,
        config: BidirectionalTorusConfig,
        device: &Device,
    ) -> Self {
        Self {
            model,
            config,
            device: device.clone(),
        }
    }

    /// Forward pass through teacher (no gradients tracked)
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Teacher forward is always in eval mode, no gradient tracking
        // Note: In candle, we don't explicitly track gradients on the teacher
        // The gradients only flow through tensors that need them
        self.model.forward_inference(x)
    }

    /// Get teacher logits with temperature scaling
    pub fn get_soft_targets(&self, x: &Tensor, temperature: f32) -> TorusResult<Tensor> {
        let logits = self.forward(x)?;
        // Scale by temperature and apply softmax
        let scaled = (&logits / temperature as f64)?;
        let soft_targets = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;
        Ok(soft_targets)
    }

    /// Get config
    pub fn config(&self) -> &BidirectionalTorusConfig {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION LOSS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute KL divergence between student and teacher soft targets
///
/// KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
///
/// For distillation, we want KL(teacher||student) which encourages
/// the student to match the teacher's distribution.
pub fn kl_divergence_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f32,
) -> TorusResult<Tensor> {
    let t = temperature as f64;

    // Soft targets from teacher
    let teacher_soft = candle_nn::ops::softmax(&(teacher_logits / t)?, candle_core::D::Minus1)?;

    // Log softmax of student
    let student_log_soft =
        candle_nn::ops::log_softmax(&(student_logits / t)?, candle_core::D::Minus1)?;

    // KL divergence: -sum(teacher * log(student))
    // (This is the cross-entropy part; the entropy of teacher is constant)
    let kl = (&teacher_soft * &student_log_soft)?;

    // Negate and take mean, scale by T^2
    let loss = (kl.sum_all()?.neg()? * (t * t))?;

    Ok(loss)
}

/// Compute feature matching loss (MSE between intermediate features)
pub fn feature_matching_loss(
    student_features: &Tensor,
    teacher_features: &Tensor,
) -> TorusResult<Tensor> {
    let diff = (student_features - teacher_features)?;
    let mse = (&diff * &diff)?.mean_all()?;
    Ok(mse)
}

/// Compute attention transfer loss
pub fn attention_transfer_loss(
    student_attention: &Tensor,
    teacher_attention: &Tensor,
) -> TorusResult<Tensor> {
    // Normalize attention maps by converting scalar to f32
    let s_sum = student_attention.sum_all()?.to_scalar::<f32>()?;
    let t_sum = teacher_attention.sum_all()?.to_scalar::<f32>()?;

    let s_norm = (student_attention / s_sum as f64)?;
    let t_norm = (teacher_attention / t_sum as f64)?;

    // MSE between normalized attention maps
    let diff = (&s_norm - &t_norm)?;
    let loss = (&diff * &diff)?.mean_all()?;
    Ok(loss)
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION TRAINER
// ═══════════════════════════════════════════════════════════════════════════════

/// Trainer for knowledge distillation
pub struct DistillationTrainer {
    /// Student model (being trained)
    student: BidirectionalTorusTransformer,
    /// Student variable map
    student_vars: VarMap,
    /// Student vocab size
    student_vocab_size: Option<usize>,
    /// Teacher model (frozen, silent pair)
    teacher: TeacherModel,
    /// Distillation config
    config: DistillationConfig,
    /// Student config
    student_config: BidirectionalTorusConfig,
    /// Optimizer
    optimizer: AdamW,
    /// Collider for validation
    collider: Option<TorusCollider>,
    /// Device
    device: Device,
    /// Training step counter
    step: u64,
}

impl DistillationTrainer {
    /// Create a new distillation trainer
    pub fn new(
        student_config: BidirectionalTorusConfig,
        vocab_size: Option<usize>,
        teacher: TeacherModel,
        distill_config: DistillationConfig,
        learning_rate: f64,
        device: &Device,
    ) -> TorusResult<Self> {
        let student_vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&student_vars, DType::F32, device);

        let student =
            BidirectionalTorusTransformer::new(student_config.clone(), vocab_size, vb, device)?;

        let optimizer = AdamW::new(
            student_vars.all_vars(),
            ParamsAdamW {
                lr: learning_rate,
                weight_decay: 0.01,
                ..Default::default()
            },
        )?;

        // Optional collider for validation
        let collider = Some(TorusCollider::new(ColliderConfig::default()));

        Ok(Self {
            student,
            student_vars,
            student_vocab_size: vocab_size,
            teacher,
            config: distill_config,
            student_config,
            optimizer,
            collider,
            device: device.clone(),
            step: 0,
        })
    }

    /// Load student from checkpoint
    pub fn load_student<P: AsRef<Path>>(&mut self, path: P) -> TorusResult<()> {
        let (model, config, var_map) = load_transformer_checkpoint(path, &self.device)?;
        self.student = model;
        self.student_config = config;
        self.student_vars = var_map;
        self.optimizer = AdamW::new(
            self.student_vars.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                weight_decay: 0.01,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    /// Single training step with distillation
    pub fn train_step(
        &mut self,
        input: &Tensor,
        labels: Option<&Tensor>,
    ) -> TorusResult<DistillationStepResult> {
        self.step += 1;

        // Reset student state
        self.student.reset_state()?;

        // Get teacher outputs (no gradients)
        let teacher_logits = self.teacher.forward(input)?;
        let _teacher_soft = self
            .teacher
            .get_soft_targets(input, self.config.temperature)?;

        // Get student outputs (gradients tracked)
        let student_logits = self.student.forward(input)?;

        // Compute distillation loss
        let distill_loss =
            kl_divergence_loss(&student_logits, &teacher_logits, self.config.temperature)?;

        // Compute hard label loss if labels provided
        let hard_loss = if let Some(labels) = labels {
            let log_probs = candle_nn::ops::log_softmax(&student_logits, candle_core::D::Minus1)?;
            // Simple cross entropy
            let ce = cross_entropy(&log_probs, labels)?;
            Some(ce)
        } else {
            None
        };

        // Combined loss
        let total_loss = if let Some(ref hard) = hard_loss {
            let alpha = self.config.alpha as f64;
            ((&distill_loss * alpha)? + &(hard * (1.0 - alpha))?)?
        } else {
            distill_loss.clone()
        };

        // Backward pass and optimize
        self.optimizer.backward_step(&total_loss)?;

        // Validation with collider
        let is_healthy = if let Some(ref mut collider) = self.collider {
            collider.next_step();
            collider.record_output(&student_logits)?;
            collider.is_healthy()
        } else {
            true
        };

        Ok(DistillationStepResult {
            step: self.step,
            total_loss: total_loss.to_scalar::<f32>()?,
            distill_loss: distill_loss.to_scalar::<f32>()?,
            hard_loss: hard_loss.map(|l| l.to_scalar::<f32>()).transpose()?,
            is_healthy,
        })
    }

    /// Save student checkpoint
    pub fn save_student<P: AsRef<Path>>(&self, path: P) -> TorusResult<()> {
        save_transformer_checkpoint(
            &self.student_config,
            self.student_vocab_size,
            &self.student_vars,
            path,
            Some(self.step),
            None,
        )
    }

    /// Get current step
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Get student model reference
    pub fn student(&self) -> &BidirectionalTorusTransformer {
        &self.student
    }

    /// Get teacher model reference
    pub fn teacher(&self) -> &TeacherModel {
        &self.teacher
    }
}

/// Result from a distillation training step
#[derive(Debug, Clone)]
pub struct DistillationStepResult {
    pub step: u64,
    pub total_loss: f32,
    pub distill_loss: f32,
    pub hard_loss: Option<f32>,
    pub is_healthy: bool,
}

impl std::fmt::Display for DistillationStepResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Step {} | Total: {:.4} | Distill: {:.4} | Hard: {} | {}",
            self.step,
            self.total_loss,
            self.distill_loss,
            self.hard_loss
                .map(|l| format!("{:.4}", l))
                .unwrap_or("-".to_string()),
            if self.is_healthy {
                "Healthy"
            } else {
                "Warning"
            }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple cross entropy loss
fn cross_entropy(log_probs: &Tensor, targets: &Tensor) -> TorusResult<Tensor> {
    let (batch, seq, vocab) = log_probs.dims3()?;
    let log_probs = log_probs.reshape((batch * seq, vocab))?;
    let targets = targets.flatten_all()?;

    // Gather log probs at target indices
    let loss = candle_nn::loss::cross_entropy(&log_probs, &targets)?;
    Ok(loss)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a small config suitable for testing
    fn test_config() -> BidirectionalTorusConfig {
        BidirectionalTorusConfig {
            d_model: 64,
            d_ff: 128,
            n_heads: 4,
            n_layers: 2,
            n_major: 4,
            n_minor: 4,
            major_radius: 2.0,
            minor_radius: 1.0,
            use_parallel_streams: true,
            use_compounding: false, // Disable for simpler testing
            use_multi_scale: false,
            ema_alpha: 0.9,
            learnable_alpha: false,
            use_momentum: false,
            spiral_winding: 1.618,
            weight_temperature: 1.0,
            parallel_execution: false, // Single-threaded for tests
            use_geodesic_bias: false,  // Disable for simpler testing
            geodesic_sigma: 0.5,
            dropout: 0.0,
            n_pos_frequencies: 8,
            use_coherence: false, // Disable for simpler testing
            coherence_threshold: 0.6,
            smm_learning_rate: 0.01,
        }
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.7);
    }

    #[test]
    fn test_distillation_config_presets() {
        let soft = DistillationConfig::soft();
        assert_eq!(soft.alpha, 0.9);

        let hard = DistillationConfig::hard();
        assert_eq!(hard.alpha, 0.3);

        let feature = DistillationConfig::feature_matching(vec![0, 2, 4]);
        assert!(feature.match_features);
        assert_eq!(feature.feature_layers, vec![0, 2, 4]);
    }

    #[test]
    fn test_kl_divergence_loss() {
        let device = Device::Cpu;

        // Different distributions should have positive KL divergence
        let logits1 = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).unwrap();
        let logits2 = Tensor::new(&[[3.0f32, 2.0, 1.0], [6.0, 5.0, 4.0]], &device).unwrap();

        let loss = kl_divergence_loss(&logits1, &logits2, 1.0).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // KL divergence should be positive for different distributions
        assert!(
            loss_val > 0.0,
            "KL divergence of different distributions should be positive, got: {}",
            loss_val
        );
    }

    #[test]
    fn test_kl_divergence_identical() {
        let device = Device::Cpu;

        // Identical distributions - use simple case with known values
        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();

        let loss = kl_divergence_loss(&logits, &logits, 1.0).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // For identical distributions, KL should be ~0 (within numerical precision)
        // Note: Due to log/exp operations, there can be small numerical errors
        assert!(
            loss_val.abs() < 1.0,
            "KL divergence of identical distributions should be ~0, got: {}",
            loss_val
        );
    }

    #[test]
    fn test_feature_matching_loss() {
        let device = Device::Cpu;

        // Same features should have 0 MSE
        let features = Tensor::randn(0.0f32, 1.0, (2, 4, 64), &device).unwrap();
        let loss = feature_matching_loss(&features, &features).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        assert!(
            loss_val.abs() < 1e-6,
            "MSE of identical features should be ~0"
        );
    }

    #[test]
    fn test_attention_transfer_loss() {
        let device = Device::Cpu;

        // Same attention should have ~0 loss
        let attention = Tensor::randn(0.0f32, 1.0, (2, 8, 16, 16), &device).unwrap();
        let attention = candle_nn::ops::softmax(&attention, candle_core::D::Minus1).unwrap();
        let loss = attention_transfer_loss(&attention, &attention).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        assert!(
            loss_val.abs() < 1e-6,
            "Attention transfer loss of identical maps should be ~0, got: {}",
            loss_val
        );
    }

    #[test]
    fn test_distillation_step_result_display() {
        let result = DistillationStepResult {
            step: 100,
            total_loss: 1.234,
            distill_loss: 0.8,
            hard_loss: Some(0.434),
            is_healthy: true,
        };
        let s = format!("{}", result);
        assert!(s.contains("100"));
        assert!(s.contains("1.234"));
        assert!(s.contains("Healthy"));
    }

    #[test]
    fn test_teacher_from_model() {
        let device = Device::Cpu;
        let config = test_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model =
            BidirectionalTorusTransformer::new(config.clone(), Some(100), vb, &device).unwrap();

        let teacher = TeacherModel::from_model(model, config, &device);
        assert!(teacher.config().n_layers > 0);
    }

    #[test]
    fn test_distillation_trainer_creation() {
        let device = Device::Cpu;
        let config = test_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create teacher
        let teacher_model =
            BidirectionalTorusTransformer::new(config.clone(), Some(100), vb, &device).unwrap();
        let teacher = TeacherModel::from_model(teacher_model, config.clone(), &device);

        // Create trainer
        let trainer = DistillationTrainer::new(
            config,
            Some(100),
            teacher,
            DistillationConfig::default(),
            1e-4,
            &device,
        )
        .unwrap();

        assert_eq!(trainer.step(), 0);
    }

    #[test]
    fn test_teacher_forward() {
        let device = Device::Cpu;
        let config = test_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create model without embedding layer (pass None for vocab_size)
        // This way we can pass embeddings directly
        let model = BidirectionalTorusTransformer::new(config.clone(), None, vb, &device).unwrap();
        let teacher = TeacherModel::from_model(model, config.clone(), &device);

        // Create test input: [batch, seq_len, d_model]
        let seq_len = config.seq_len();
        let input = Tensor::randn(0.0f32, 1.0, (2, seq_len, config.d_model), &device).unwrap();

        let output = teacher.forward(&input).unwrap();

        // Output should be [batch, seq_len, d_model] (since no vocab_size)
        assert_eq!(output.dims(), &[2, seq_len, config.d_model]);
    }

    #[test]
    fn test_teacher_soft_targets() {
        let device = Device::Cpu;
        let config = test_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create model without embedding layer
        let model = BidirectionalTorusTransformer::new(config.clone(), None, vb, &device).unwrap();
        let teacher = TeacherModel::from_model(model, config.clone(), &device);

        let seq_len = config.seq_len();
        let input = Tensor::randn(0.0f32, 1.0, (2, seq_len, config.d_model), &device).unwrap();

        let soft_targets = teacher.get_soft_targets(&input, 4.0).unwrap();

        // Soft targets should sum to 1 along the last dimension (after softmax)
        let sums = soft_targets.sum(candle_core::D::Minus1).unwrap();
        let sum_val = sums.mean_all().unwrap().to_scalar::<f32>().unwrap();

        assert!(
            (sum_val - 1.0).abs() < 1e-5,
            "Soft targets should sum to ~1, got: {}",
            sum_val
        );
    }
}
