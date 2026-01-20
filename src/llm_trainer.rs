//! # LLM Training Pipeline
//!
//! Complete training pipeline for the Torus LLM including:
//! - Gradient descent with AdamW optimizer
//! - Learning rate scheduling with warmup
//! - Checkpoint saving and resumption
//! - Validation and early stopping
//! - Logging and metrics
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::llm_trainer::{LLMTrainer, LLMTrainingConfig};
//!
//! let config = LLMTrainingConfig::default();
//! let mut trainer = LLMTrainer::new(config, train_dataset, val_dataset)?;
//! trainer.train()?;
//! ```

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use std::path::Path;

use crate::checkpoint::{checkpoint_exists, load_checkpoint, save_checkpoint};
use crate::dataset::{DataLoader, TextDataset};
use crate::llm::{TorusLLM, TorusLLMConfig};
use crate::TorusResult;

/// Configuration for LLM training
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMTrainingConfig {
    /// Model configuration
    pub model: TorusLLMConfig,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Maximum number of steps (overrides epochs if set)
    pub max_steps: Option<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay for AdamW
    pub weight_decay: f64,
    /// Warmup steps for learning rate
    pub warmup_steps: usize,
    /// Steps between logging
    pub log_interval: usize,
    /// Steps between validation
    pub eval_interval: usize,
    /// Steps between checkpoint saves
    pub save_interval: usize,
    /// Directory for checkpoints
    pub checkpoint_dir: String,
    /// Whether to resume from latest checkpoint
    pub resume: bool,
    /// Early stopping patience (epochs without improvement)
    pub patience: Option<usize>,
    /// Gradient clipping max norm
    pub max_grad_norm: Option<f64>,
}

impl Default for LLMTrainingConfig {
    fn default() -> Self {
        Self {
            model: TorusLLMConfig::tiny(),
            batch_size: 32,
            num_epochs: 10,
            max_steps: None,
            learning_rate: 3e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            log_interval: 10,
            eval_interval: 100,
            save_interval: 500,
            checkpoint_dir: "checkpoints".to_string(),
            resume: true,
            patience: Some(3),
            max_grad_norm: Some(1.0),
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    pub step: usize,
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub tokens_per_sec: f64,
}

/// The LLM Trainer
pub struct LLMTrainer {
    config: LLMTrainingConfig,
    model: TorusLLM,
    varmap: VarMap,
    optimizer: AdamW,
    train_data: TextDataset,
    val_data: Option<TextDataset>,
    device: Device,
    step: usize,
    epoch: usize,
    best_val_loss: f64,
    patience_counter: usize,
}

impl LLMTrainer {
    /// Create a new trainer
    pub fn new(
        config: LLMTrainingConfig,
        train_data: TextDataset,
        val_data: Option<TextDataset>,
        device: Device,
    ) -> TorusResult<Self> {
        // Try to resume from checkpoint
        let (model, varmap, step, epoch) =
            if config.resume && checkpoint_exists(&config.checkpoint_dir) {
                log::info!("Resuming from checkpoint...");
                let (model, varmap, metadata) = load_checkpoint(&config.checkpoint_dir, &device)?;
                let step = metadata.step.unwrap_or(0);
                (model, varmap, step, 0)
            } else {
                log::info!("Creating new model...");
                let (model, varmap) = TorusLLM::new_random(config.model.clone(), &device)?;
                (model, varmap, 0, 0)
            };

        // Create optimizer
        let params = ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            config,
            model,
            varmap,
            optimizer,
            train_data,
            val_data,
            device,
            step,
            epoch,
            best_val_loss: f64::INFINITY,
            patience_counter: 0,
        })
    }

    /// Run training
    pub fn train(&mut self) -> TorusResult<()> {
        log::info!("Starting training...");
        log::info!(
            "  Model: {} layers, {} hidden dim",
            self.config.model.num_layers,
            self.config.model.hidden_dim
        );
        log::info!("  Train examples: {}", self.train_data.len());
        if let Some(ref val) = self.val_data {
            log::info!("  Val examples: {}", val.len());
        }
        log::info!("  Batch size: {}", self.config.batch_size);
        log::info!("  Learning rate: {}", self.config.learning_rate);

        let max_steps = self.config.max_steps.unwrap_or(usize::MAX);

        for epoch in self.epoch..self.config.num_epochs {
            self.epoch = epoch;
            log::info!("Epoch {}/{}", epoch + 1, self.config.num_epochs);

            // Create data loader
            let mut train_loader = DataLoader::new(
                self.train_data.clone(),
                self.config.batch_size,
                true,
                self.device.clone(),
            );

            let mut epoch_loss = 0.0;
            let mut epoch_batches = 0;

            while let Some(batch_result) = train_loader.next_batch() {
                let batch = batch_result?;

                // Forward pass
                let loss = self.train_step(&batch.input_ids, &batch.labels)?;
                epoch_loss += loss;
                epoch_batches += 1;

                self.step += 1;

                // Logging
                if self.step.is_multiple_of(self.config.log_interval) {
                    let avg_loss = epoch_loss / epoch_batches as f64;
                    let lr = self.get_learning_rate();
                    log::info!(
                        "Step {} | Loss: {:.4} | LR: {:.2e}",
                        self.step,
                        avg_loss,
                        lr
                    );
                }

                // Validation
                if self.step.is_multiple_of(self.config.eval_interval) {
                    if let Some(ref val_data) = self.val_data {
                        let val_loss = self.evaluate(val_data)?;
                        log::info!("Step {} | Val Loss: {:.4}", self.step, val_loss);

                        // Early stopping check
                        if val_loss < self.best_val_loss {
                            self.best_val_loss = val_loss;
                            self.patience_counter = 0;

                            // Save best model
                            self.save_checkpoint("best")?;
                        } else {
                            self.patience_counter += 1;
                            if let Some(patience) = self.config.patience {
                                if self.patience_counter >= patience {
                                    log::info!("Early stopping triggered after {} evaluations without improvement", patience);
                                    return Ok(());
                                }
                            }
                        }
                    }
                }

                // Checkpointing
                if self.step.is_multiple_of(self.config.save_interval) {
                    self.save_checkpoint("latest")?;
                }

                // Max steps check
                if self.step >= max_steps {
                    log::info!("Reached max steps ({})", max_steps);
                    self.save_checkpoint("final")?;
                    return Ok(());
                }
            }

            let avg_epoch_loss = epoch_loss / epoch_batches.max(1) as f64;
            log::info!(
                "Epoch {} complete | Avg Loss: {:.4}",
                epoch + 1,
                avg_epoch_loss
            );
        }

        // Save final checkpoint
        self.save_checkpoint("final")?;
        log::info!("Training complete!");

        Ok(())
    }

    /// Single training step
    fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor) -> TorusResult<f64> {
        // Update learning rate
        let lr = self.get_learning_rate();
        self.optimizer.set_learning_rate(lr);

        // Forward pass
        let logits = self.model.forward(input_ids)?;

        // Compute cross-entropy loss
        let loss = cross_entropy_loss(&logits, labels)?;

        // Backward pass
        self.optimizer.backward_step(&loss)?;

        // Get loss value
        let loss_val = loss.to_scalar::<f32>()? as f64;

        Ok(loss_val)
    }

    /// Evaluate on a dataset
    fn evaluate(&self, dataset: &TextDataset) -> TorusResult<f64> {
        let mut loader = DataLoader::new(
            dataset.clone(),
            self.config.batch_size,
            false,
            self.device.clone(),
        );

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some(batch_result) = loader.next_batch() {
            let batch = batch_result?;

            // Forward pass (no gradients needed)
            let logits = self.model.forward(&batch.input_ids)?;
            let loss = cross_entropy_loss(&logits, &batch.labels)?;

            total_loss += loss.to_scalar::<f32>()? as f64;
            num_batches += 1;
        }

        Ok(total_loss / num_batches.max(1) as f64)
    }

    /// Get current learning rate with warmup
    fn get_learning_rate(&self) -> f64 {
        let base_lr = self.config.learning_rate;
        let warmup = self.config.warmup_steps;

        if self.step < warmup {
            // Linear warmup
            base_lr * (self.step as f64 / warmup as f64)
        } else {
            // Could add decay here
            base_lr
        }
    }

    /// Save a checkpoint
    fn save_checkpoint(&self, name: &str) -> TorusResult<()> {
        let path = Path::new(&self.config.checkpoint_dir).join(name);
        save_checkpoint(
            self.model.config(),
            &self.varmap,
            &path,
            Some(self.step),
            Some(self.best_val_loss),
        )?;
        log::info!("Saved checkpoint to {:?}", path);
        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &TorusLLM {
        &self.model
    }

    /// Get the varmap
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Get current step
    pub fn step(&self) -> usize {
        self.step
    }
}

/// Compute cross-entropy loss for language modeling
fn cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> TorusResult<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;

    // Reshape logits to [batch * seq, vocab]
    let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;

    // Reshape labels to [batch * seq]
    let labels_flat = labels.flatten_all()?;

    // Compute log softmax
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;

    // Gather the log probs for the correct labels
    let labels_i64 = labels_flat.to_dtype(DType::I64)?;
    let loss = log_probs.gather(&labels_i64.unsqueeze(1)?, 1)?.squeeze(1)?;

    // Mean negative log likelihood
    let loss = loss.neg()?.mean_all()?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = LLMTrainingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 10);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let device = Device::Cpu;

        // Create dummy logits and labels
        let batch_size = 2;
        let seq_len = 4;
        let vocab_size = 10;

        let logits = Tensor::randn(0f32, 1.0, (batch_size, seq_len, vocab_size), &device).unwrap();
        let labels = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let loss = cross_entropy_loss(&logits, &labels).unwrap();

        // Loss should be a scalar
        assert_eq!(loss.dims(), &[] as &[usize]);

        // Loss should be positive
        let loss_val = loss.to_scalar::<f32>().unwrap();
        assert!(loss_val > 0.0);
    }

    #[test]
    fn test_trainer_creation() {
        let device = Device::Cpu;
        let config = LLMTrainingConfig {
            resume: false,
            ..Default::default()
        };

        let dataset = TextDataset::synthetic(
            100,
            config.model.to_torus_config().seq_len(),
            config.model.vocab_size,
        );

        let trainer = LLMTrainer::new(config, dataset, None, device);
        assert!(trainer.is_ok());
    }
}
