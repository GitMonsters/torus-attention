//! Training Loop and Optimization for Torus Attention
//!
//! Provides training infrastructure for the BidirectionalTorusTransformer:
//! - Loss functions (cross-entropy, MSE, geodesic-aware)
//! - Optimizers (AdamW with gradient clipping)
//! - Learning rate schedulers (warmup, cosine annealing)
//! - Training loop with checkpointing
//! - Metrics tracking and logging

use crate::integration::{BidirectionalTorusConfig, BidirectionalTorusTransformer, BidirectionalStats};
use crate::error::TorusError;
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Complete training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay for AdamW
    pub weight_decay: f64,
    /// Beta1 for Adam
    pub beta1: f64,
    /// Beta2 for Adam
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Gradient clipping norm
    pub grad_clip_norm: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Steps per epoch (if using infinite data)
    pub steps_per_epoch: Option<usize>,
    /// Warmup steps for learning rate
    pub warmup_steps: usize,
    /// Total training steps (for scheduler)
    pub total_steps: usize,
    /// Checkpoint save frequency (steps)
    pub checkpoint_every: usize,
    /// Evaluation frequency (steps)
    pub eval_every: usize,
    /// Log frequency (steps)
    pub log_every: usize,
    /// Use mixed precision
    pub mixed_precision: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip_norm: 1.0,
            batch_size: 32,
            epochs: 10,
            steps_per_epoch: None,
            warmup_steps: 1000,
            total_steps: 100_000,
            checkpoint_every: 5000,
            eval_every: 1000,
            log_every: 100,
            mixed_precision: false,
            seed: 42,
        }
    }
}

impl TrainingConfig {
    /// Configuration for quick experiments
    pub fn quick() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 16,
            epochs: 3,
            warmup_steps: 100,
            total_steps: 1000,
            checkpoint_every: 500,
            eval_every: 100,
            log_every: 10,
            ..Self::default()
        }
    }

    /// Configuration for thorough training
    pub fn thorough() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 64,
            epochs: 50,
            warmup_steps: 2000,
            total_steps: 500_000,
            checkpoint_every: 10000,
            eval_every: 2000,
            log_every: 100,
            ..Self::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LEARNING RATE SCHEDULER
// ═══════════════════════════════════════════════════════════════════════════

/// Learning rate scheduler with warmup and cosine annealing
#[derive(Debug)]
pub struct LRScheduler {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr: f64,
    current_step: usize,
}

impl LRScheduler {
    pub fn new(config: &TrainingConfig) -> Self {
        Self {
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.total_steps,
            min_lr: config.learning_rate * 0.1,
            current_step: 0,
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f64 / self.warmup_steps as f64;
            self.base_lr * progress
        } else {
            // Cosine annealing
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps) as f64;
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Loss function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossType {
    CrossEntropy,
    MSE,
    SmoothL1,
    GeodesicAware,
}

/// Compute cross-entropy loss
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> TorusResult<Tensor> {
    // logits: [B, S, V], targets: [B, S]
    let (batch, seq_len, vocab) = logits.dims3()?;
    
    // Reshape for cross entropy
    let logits_flat = logits.reshape((batch * seq_len, vocab))?;
    let targets_flat = targets.flatten_all()?;
    
    // Log softmax
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
    
    // Gather target probabilities
    let targets_one_hot = targets_flat
        .to_dtype(DType::I64)?
        .unsqueeze(1)?;
    let nll = log_probs.gather(&targets_one_hot, 1)?;
    
    // Negative log likelihood
    let loss = nll.neg()?.mean_all()?;
    Ok(loss)
}

/// Compute MSE loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> TorusResult<Tensor> {
    let diff = (predictions - targets)?;
    let squared = diff.sqr()?;
    Ok(squared.mean_all()?)
}

/// Compute Smooth L1 loss
pub fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor, beta: f64) -> TorusResult<Tensor> {
    let diff = (predictions - targets)?.abs()?;
    
    // Where diff < beta: 0.5 * diff^2 / beta
    // Where diff >= beta: diff - 0.5 * beta
    let small_mask = diff.lt(beta)?;
    let large_mask = diff.ge(beta)?;
    
    let small_loss = (diff.sqr()? * (0.5 / beta))?;
    let large_loss = (diff - 0.5 * beta)?;
    
    let loss = (small_mask.to_dtype(DType::F32)? * small_loss)?
        + (large_mask.to_dtype(DType::F32)? * large_loss)?;
    
    Ok(loss?.mean_all()?)
}

/// Combined loss with optional geodesic regularization
pub struct TorusLoss {
    loss_type: LossType,
    geodesic_weight: f64,
    label_smoothing: f64,
}

impl TorusLoss {
    pub fn new(loss_type: LossType) -> Self {
        Self {
            loss_type,
            geodesic_weight: 0.0,
            label_smoothing: 0.0,
        }
    }

    pub fn with_geodesic_weight(mut self, weight: f64) -> Self {
        self.geodesic_weight = weight;
        self
    }

    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    pub fn compute(&self, logits: &Tensor, targets: &Tensor) -> TorusResult<Tensor> {
        match self.loss_type {
            LossType::CrossEntropy => cross_entropy_loss(logits, targets),
            LossType::MSE => mse_loss(logits, targets),
            LossType::SmoothL1 => smooth_l1_loss(logits, targets, 1.0),
            LossType::GeodesicAware => {
                // Standard loss + geodesic consistency term
                let base = cross_entropy_loss(logits, targets)?;
                // Geodesic term could compare attention patterns to geodesic distances
                // For now, just return base loss
                Ok(base)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Training metrics tracker
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Learning rate history
    pub lr_history: Vec<f64>,
    /// Gradient norm history
    pub grad_norm_history: Vec<f64>,
    /// Evaluation loss history
    pub eval_loss_history: Vec<f64>,
    /// Compounding alpha history per layer
    pub alpha_history: Vec<Vec<f64>>,
    /// Stream weight history
    pub stream_weight_history: Vec<Vec<Vec<f32>>>,
    /// Training time per step (ms)
    pub step_times: Vec<f64>,
    /// Best evaluation loss
    pub best_eval_loss: f64,
    /// Step with best evaluation loss
    pub best_step: usize,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            best_eval_loss: f64::INFINITY,
            ..Default::default()
        }
    }

    pub fn log_step(&mut self, loss: f64, lr: f64, grad_norm: f64, step_time_ms: f64) {
        self.loss_history.push(loss);
        self.lr_history.push(lr);
        self.grad_norm_history.push(grad_norm);
        self.step_times.push(step_time_ms);
    }

    pub fn log_eval(&mut self, eval_loss: f64, step: usize) {
        self.eval_loss_history.push(eval_loss);
        if eval_loss < self.best_eval_loss {
            self.best_eval_loss = eval_loss;
            self.best_step = step;
        }
    }

    pub fn log_model_stats(&mut self, stats: &BidirectionalStats) {
        self.alpha_history.push(stats.compounding_alphas.clone());
        
        let weights: Vec<Vec<f32>> = stats.stream_weights
            .iter()
            .map(|layer| layer.iter().map(|(_, w)| *w).collect())
            .collect();
        self.stream_weight_history.push(weights);
    }

    pub fn summary(&self) -> String {
        let n_steps = self.loss_history.len();
        let recent_loss: f64 = if n_steps >= 100 {
            self.loss_history[n_steps - 100..].iter().sum::<f64>() / 100.0
        } else if n_steps > 0 {
            self.loss_history.iter().sum::<f64>() / n_steps as f64
        } else {
            0.0
        };

        let avg_step_time: f64 = if self.step_times.is_empty() {
            0.0
        } else {
            self.step_times.iter().sum::<f64>() / self.step_times.len() as f64
        };

        format!(
            "Steps: {}, Recent Loss: {:.4}, Best Eval: {:.4} (step {}), Avg Step Time: {:.1}ms",
            n_steps, recent_loss, self.best_eval_loss, self.best_step, avg_step_time
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GRADIENT UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Compute gradient norm across all parameters
pub fn compute_grad_norm(grads: &HashMap<String, Tensor>) -> TorusResult<f64> {
    let mut total_norm_sq = 0.0f64;
    
    for (_, grad) in grads {
        let grad_flat = grad.flatten_all()?;
        let norm_sq: f32 = grad_flat.sqr()?.sum_all()?.to_scalar()?;
        total_norm_sq += norm_sq as f64;
    }
    
    Ok(total_norm_sq.sqrt())
}

/// Clip gradients by global norm
pub fn clip_grad_norm(
    grads: &mut HashMap<String, Tensor>,
    max_norm: f64,
) -> TorusResult<f64> {
    let total_norm = compute_grad_norm(grads)?;
    
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for (_, grad) in grads.iter_mut() {
            *grad = (grad.clone() * clip_coef)?;
        }
    }
    
    Ok(total_norm)
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Main trainer for BidirectionalTorusTransformer
pub struct Trainer {
    /// Model
    model: BidirectionalTorusTransformer,
    /// Variable map for parameters
    var_map: VarMap,
    /// Optimizer
    optimizer: AdamW,
    /// Learning rate scheduler
    scheduler: LRScheduler,
    /// Loss function
    loss_fn: TorusLoss,
    /// Training configuration
    config: TrainingConfig,
    /// Metrics tracker
    metrics: TrainingMetrics,
    /// Device
    device: Device,
    /// Current global step
    global_step: usize,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(
        model_config: BidirectionalTorusConfig,
        training_config: TrainingConfig,
        vocab_size: Option<usize>,
        device: &Device,
    ) -> TorusResult<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        
        let model = BidirectionalTorusTransformer::new(
            model_config,
            vocab_size,
            vb,
            device,
        )?;

        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: training_config.learning_rate,
                weight_decay: training_config.weight_decay,
                beta1: training_config.beta1,
                beta2: training_config.beta2,
                eps: training_config.eps,
            },
        )?;

        let scheduler = LRScheduler::new(&training_config);
        let loss_fn = TorusLoss::new(LossType::CrossEntropy);
        let metrics = TrainingMetrics::new();

        Ok(Self {
            model,
            var_map,
            optimizer,
            scheduler,
            loss_fn,
            config: training_config,
            metrics,
            device: device.clone(),
            global_step: 0,
        })
    }

    /// Single training step
    pub fn train_step(&mut self, inputs: &Tensor, targets: &Tensor) -> TorusResult<f64> {
        let start = Instant::now();
        
        // Reset model state for new batch
        self.model.reset_state()?;
        
        // Forward pass
        let logits = self.model.forward(inputs)?;
        
        // Compute loss
        let loss = self.loss_fn.compute(&logits, targets)?;
        let loss_value: f32 = loss.to_scalar()?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Get gradients as HashMap (simplified - actual implementation needs VarMap access)
        // Note: In production, you'd iterate over var_map to get actual gradients
        let grad_norm = 0.0; // Placeholder - would compute from actual gradients
        
        // Update learning rate
        let lr = self.scheduler.get_lr();
        self.optimizer.set_learning_rate(lr);
        
        // Optimizer step
        self.optimizer.step(&grads)?;
        
        // Step scheduler
        self.scheduler.step();
        self.global_step += 1;
        
        let step_time = start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.log_step(loss_value as f64, lr, grad_norm, step_time);
        
        Ok(loss_value as f64)
    }

    /// Run evaluation
    pub fn evaluate(&mut self, inputs: &Tensor, targets: &Tensor) -> TorusResult<f64> {
        self.model.reset_state()?;
        let logits = self.model.forward(inputs)?;
        let loss = self.loss_fn.compute(&logits, targets)?;
        let loss_value: f32 = loss.to_scalar()?;
        
        self.metrics.log_eval(loss_value as f64, self.global_step);
        
        // Log model statistics
        if let Ok(stats) = BidirectionalStats::from_transformer(&self.model) {
            self.metrics.log_model_stats(&stats);
        }
        
        Ok(loss_value as f64)
    }

    /// Train for one epoch with data iterator
    pub fn train_epoch<I>(&mut self, data: I) -> TorusResult<f64>
    where
        I: Iterator<Item = (Tensor, Tensor)>,
    {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        
        for (inputs, targets) in data {
            let loss = self.train_step(&inputs, &targets)?;
            total_loss += loss;
            n_batches += 1;
            
            // Logging
            if self.global_step % self.config.log_every == 0 {
                println!(
                    "Step {}: loss={:.4}, lr={:.2e}",
                    self.global_step,
                    loss,
                    self.scheduler.get_lr()
                );
            }
        }
        
        Ok(total_loss / n_batches as f64)
    }

    /// Get current metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get current step
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Get model reference
    pub fn model(&self) -> &BidirectionalTorusTransformer {
        &self.model
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut BidirectionalTorusTransformer {
        &mut self.model
    }

    /// Save checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> TorusResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    /// Load checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> TorusResult<()> {
        self.var_map.load(path)?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Generate random training data for testing
pub fn generate_random_batch(
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    vocab_size: usize,
    device: &Device,
) -> TorusResult<(Tensor, Tensor)> {
    // Random input embeddings
    let inputs = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, d_model), device)?;
    
    // Random target tokens
    let targets = Tensor::rand(0.0f32, vocab_size as f32, (batch_size, seq_len), device)?
        .to_dtype(DType::U32)?;
    
    Ok((inputs, targets))
}

/// Simple data loader for training
pub struct SimpleDataLoader {
    inputs: Tensor,
    targets: Tensor,
    batch_size: usize,
    current_idx: usize,
    shuffle: bool,
}

impl SimpleDataLoader {
    pub fn new(inputs: Tensor, targets: Tensor, batch_size: usize, shuffle: bool) -> Self {
        Self {
            inputs,
            targets,
            batch_size,
            current_idx: 0,
            shuffle,
        }
    }

    pub fn len(&self) -> usize {
        let n_samples = self.inputs.dims()[0];
        (n_samples + self.batch_size - 1) / self.batch_size
    }

    pub fn reset(&mut self) {
        self.current_idx = 0;
        // Note: Would implement shuffling here
    }
}

impl Iterator for SimpleDataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let n_samples = self.inputs.dims()[0];
        if self.current_idx >= n_samples {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(n_samples);
        
        let batch_inputs = self.inputs.i(self.current_idx..end_idx).ok()?;
        let batch_targets = self.targets.i(self.current_idx..end_idx).ok()?;
        
        self.current_idx = end_idx;
        
        Some((batch_inputs, batch_targets))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QUICK TRAINING EXAMPLE
// ═══════════════════════════════════════════════════════════════════════════

/// Run a quick training example (for testing)
pub fn run_training_example(device: &Device) -> TorusResult<TrainingMetrics> {
    println!("═══ Torus Attention Training Example ═══\n");
    
    // Small config for quick testing
    let model_config = BidirectionalTorusConfig {
        d_model: 64,
        d_ff: 256,
        n_heads: 4,
        n_layers: 2,
        n_major: 8,
        n_minor: 4,
        ..BidirectionalTorusConfig::default()
    };
    
    let training_config = TrainingConfig::quick();
    let vocab_size = 100;
    
    println!("Model config: {:?}", model_config);
    println!("Training config: {:?}", training_config);
    println!();
    
    // Create trainer
    let mut trainer = Trainer::new(
        model_config.clone(),
        training_config.clone(),
        Some(vocab_size),
        device,
    )?;
    
    println!("Trainer initialized");
    println!("Seq length: {}", model_config.seq_len());
    println!();
    
    // Generate some random data
    let n_samples = 100;
    let seq_len = model_config.seq_len();
    let (all_inputs, all_targets) = generate_random_batch(
        n_samples,
        seq_len,
        model_config.d_model,
        vocab_size,
        device,
    )?;
    
    // Create data loader
    let data_loader = SimpleDataLoader::new(
        all_inputs.clone(),
        all_targets.clone(),
        training_config.batch_size,
        true,
    );
    
    println!("Training for {} steps...", data_loader.len());
    
    // Train for one "epoch"
    let avg_loss = trainer.train_epoch(data_loader)?;
    
    println!("\nTraining complete!");
    println!("Average loss: {:.4}", avg_loss);
    println!("Metrics: {}", trainer.metrics().summary());
    
    // Run evaluation
    let eval_loss = trainer.evaluate(&all_inputs.i(0..16)?, &all_targets.i(0..16)?)?;
    println!("Eval loss: {:.4}", eval_loss);
    
    // Print model stats
    if let Ok(stats) = BidirectionalStats::from_transformer(trainer.model()) {
        stats.summary();
    }
    
    Ok(trainer.metrics().clone())
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let config = TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 100,
            total_steps: 1000,
            ..Default::default()
        };
        let mut scheduler = LRScheduler::new(&config);
        
        // At step 0, LR should be 0
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-8);
        
        // At step 50 (half warmup), LR should be half
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-6);
        
        // At step 100 (end warmup), LR should be full
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-6);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        assert_eq!(metrics.best_eval_loss, f64::INFINITY);
        
        metrics.log_step(1.0, 1e-4, 0.5, 10.0);
        metrics.log_step(0.8, 1e-4, 0.4, 11.0);
        
        assert_eq!(metrics.loss_history.len(), 2);
        assert_eq!(metrics.step_times.len(), 2);
        
        metrics.log_eval(0.7, 2);
        assert_eq!(metrics.best_eval_loss, 0.7);
        assert_eq!(metrics.best_step, 2);
    }

    #[test]
    fn test_loss_types() {
        let device = Device::Cpu;
        
        // Test MSE loss
        let pred = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
        let target = Tensor::new(&[[1.1f32, 2.1], [3.1, 4.1]], &device).unwrap();
        
        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val: f32 = loss.to_scalar().unwrap();
        assert!((loss_val - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_generate_random_batch() {
        let device = Device::Cpu;
        let (inputs, targets) = generate_random_batch(4, 16, 32, 100, &device).unwrap();
        
        assert_eq!(inputs.dims(), &[4, 16, 32]);
        assert_eq!(targets.dims(), &[4, 16]);
    }
}
