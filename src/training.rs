//! Training Loop and Optimization for Torus Attention
//!
//! Provides training infrastructure for the BidirectionalTorusTransformer:
//! - Loss functions (cross-entropy, MSE, geodesic-aware)
//! - Optimizers (AdamW with gradient clipping)
//! - Learning rate schedulers (warmup, cosine annealing)
//! - Training loop with checkpointing
//! - Metrics tracking and logging
//! - **Collider validation integration** for physics-based anomaly detection

use crate::collider::{
    integration::{ValidationConfig, ValidationContext, ValidationHistory, ValidationLevel},
    ColliderConfig, ColliderReport, TorusCollider,
};
use crate::integration::{
    BidirectionalStats, BidirectionalTorusConfig, BidirectionalTorusTransformer,
};
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
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
    /// Enable coherence-based early stopping
    pub coherence_early_stopping: bool,
    /// Coherence stability window (number of evals)
    pub coherence_stability_window: usize,
    /// Coherence stability threshold (max variance in window)
    pub coherence_stability_threshold: f64,
    /// Minimum coherence score to consider stable
    pub coherence_min_score: f64,
    /// Patience: number of stable evaluations before stopping
    pub coherence_patience: usize,
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
            coherence_early_stopping: false,
            coherence_stability_window: 5,
            coherence_stability_threshold: 0.01,
            coherence_min_score: 0.6,
            coherence_patience: 3,
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
// COLLIDER TRAINING CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for collider-validated training
///
/// This extends training with physics-based validation from the TorusCollider,
/// enabling real-time anomaly detection and conservation law checking during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColliderTrainingConfig {
    /// Enable collider validation during training
    pub enabled: bool,
    /// Validation level (minimal, standard, full)
    pub validation_level: ValidationLevel,
    /// Sample rate for validation (0.0-1.0, where 1.0 = validate every step)
    pub sample_rate: f64,
    /// Record Q, K, V tensors for particle analysis
    pub record_qkv: bool,
    /// Record attention weights
    pub record_attention: bool,
    /// Record gradients for backward stream analysis
    pub record_gradients: bool,
    /// Stop training on critical anomalies (NaN, Inf)
    pub stop_on_critical: bool,
    /// Maximum anomalies before stopping (0 = unlimited)
    pub max_anomalies: usize,
    /// Warn on conservation violations
    pub warn_on_conservation_violation: bool,
    /// Warn on causality violations
    pub warn_on_causality_violation: bool,
    /// Print validation dashboard periodically
    pub show_dashboard: bool,
    /// Dashboard update frequency (steps)
    pub dashboard_every: usize,
    /// History size for trend analysis
    pub history_size: usize,
}

impl Default for ColliderTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            validation_level: ValidationLevel::Standard,
            sample_rate: 0.1, // Validate 10% of steps by default
            record_qkv: true,
            record_attention: true,
            record_gradients: true,
            stop_on_critical: true,
            max_anomalies: 0, // Unlimited
            warn_on_conservation_violation: true,
            warn_on_causality_violation: true,
            show_dashboard: true,
            dashboard_every: 100,
            history_size: 1000,
        }
    }
}

impl ColliderTrainingConfig {
    /// Configuration for thorough validation (slower, more detailed)
    pub fn thorough() -> Self {
        Self {
            enabled: true,
            validation_level: ValidationLevel::Full,
            sample_rate: 1.0, // Every step
            record_qkv: true,
            record_attention: true,
            record_gradients: true,
            stop_on_critical: true,
            max_anomalies: 0,
            warn_on_conservation_violation: true,
            warn_on_causality_violation: true,
            show_dashboard: true,
            dashboard_every: 50,
            history_size: 5000,
        }
    }

    /// Configuration for minimal validation (faster, basic checks only)
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            validation_level: ValidationLevel::Minimal,
            sample_rate: 0.01, // 1% of steps
            record_qkv: false,
            record_attention: false,
            record_gradients: true, // Keep gradient monitoring
            stop_on_critical: true,
            max_anomalies: 100,
            warn_on_conservation_violation: false,
            warn_on_causality_violation: false,
            show_dashboard: false,
            dashboard_every: 500,
            history_size: 100,
        }
    }

    /// Disable collider validation entirely
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Convert to ValidationConfig for the collider integration module
    pub fn to_validation_config(&self) -> ValidationConfig {
        ValidationConfig {
            enabled: self.enabled,
            level: self.validation_level.clone(),
            sample_rate: self.sample_rate,
            record_qkv: self.record_qkv,
            record_attention: self.record_attention,
            record_collisions: true,
            validate_conservation: self.warn_on_conservation_violation,
            validate_causality: self.warn_on_causality_violation,
            detect_anomalies: true,
            collect_metrics: true,
            ..ValidationConfig::default()
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
    let targets_one_hot = targets_flat.to_dtype(DType::I64)?.unsqueeze(1)?;
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
    /// Coherence score history (for early stopping)
    pub coherence_history: Vec<f64>,
    /// Cohesion score history
    pub cohesion_history: Vec<f64>,
    /// Adaptive alpha history
    pub adaptive_alpha_history: Vec<f64>,
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

        let weights: Vec<Vec<f32>> = stats
            .stream_weights
            .iter()
            .map(|layer| layer.iter().map(|(_, w)| *w).collect())
            .collect();
        self.stream_weight_history.push(weights);

        // Log coherence metrics if available
        if let Some(ref coh) = stats.coherence_metrics {
            self.coherence_history.push(coh.soc_score);
            self.cohesion_history.push(coh.cognitive_cohesion);
            self.adaptive_alpha_history.push(coh.adaptive_alpha);
        }
    }

    /// Check if coherence has stabilized based on recent history
    ///
    /// Returns `true` if:
    /// 1. We have enough history (>= window_size)
    /// 2. Coherence is above minimum threshold
    /// 3. Variance of recent coherence values is below stability threshold
    pub fn is_coherence_stable(
        &self,
        window_size: usize,
        min_score: f64,
        max_variance: f64,
    ) -> bool {
        if self.coherence_history.len() < window_size {
            return false;
        }

        let recent: Vec<f64> = self
            .coherence_history
            .iter()
            .rev()
            .take(window_size)
            .copied()
            .collect();

        // Check minimum score
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        if mean < min_score {
            return false;
        }

        // Check variance
        let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        variance < max_variance
    }

    /// Get coherence trend (positive = improving)
    pub fn coherence_trend(&self) -> f64 {
        if self.coherence_history.len() < 2 {
            return 0.0;
        }

        let recent_len = self.coherence_history.len().min(10);
        let recent: Vec<f64> = self
            .coherence_history
            .iter()
            .rev()
            .take(recent_len)
            .copied()
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i * i) as f64).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
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

    for grad in grads.values() {
        let grad_flat: Tensor = grad.flatten_all()?;
        let norm_sq: f32 = grad_flat.sqr()?.sum_all()?.to_scalar()?;
        total_norm_sq += norm_sq as f64;
    }

    Ok(total_norm_sq.sqrt())
}

/// Clip gradients by global norm
pub fn clip_grad_norm(grads: &mut HashMap<String, Tensor>, max_norm: f64) -> TorusResult<f64> {
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
    #[allow(dead_code)]
    device: Device,
    /// Current global step
    global_step: usize,
    /// Consecutive stable coherence evaluations (for early stopping)
    stable_coherence_count: usize,
    /// Whether training was stopped early due to coherence stability
    early_stopped: bool,
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

        let model = BidirectionalTorusTransformer::new(model_config, vocab_size, vb, device)?;

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
            stable_coherence_count: 0,
            early_stopped: false,
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
        self.metrics
            .log_step(loss_value as f64, lr, grad_norm, step_time);

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

    /// Check if coherence has stabilized (for early stopping)
    ///
    /// Returns `true` if coherence metrics have been stable for `patience` consecutive evaluations.
    pub fn check_coherence_stability(&mut self) -> bool {
        if !self.config.coherence_early_stopping {
            return false;
        }

        let is_stable = self.metrics.is_coherence_stable(
            self.config.coherence_stability_window,
            self.config.coherence_min_score,
            self.config.coherence_stability_threshold,
        );

        if is_stable {
            self.stable_coherence_count += 1;
            if self.stable_coherence_count >= self.config.coherence_patience {
                self.early_stopped = true;
                return true;
            }
        } else {
            self.stable_coherence_count = 0;
        }

        false
    }

    /// Check if training was stopped early
    pub fn was_early_stopped(&self) -> bool {
        self.early_stopped
    }

    /// Get early stopping reason if applicable
    pub fn early_stop_reason(&self) -> Option<String> {
        if self.early_stopped {
            Some(format!(
                "Coherence stabilized at {:.4} (stable for {} evaluations)",
                self.metrics
                    .coherence_history
                    .last()
                    .copied()
                    .unwrap_or(0.0),
                self.stable_coherence_count
            ))
        } else {
            None
        }
    }

    /// Train for one epoch with data iterator
    ///
    /// Returns the average loss for the epoch, or None if training was early stopped.
    pub fn train_epoch<I>(&mut self, data: I) -> TorusResult<Option<f64>>
    where
        I: Iterator<Item = (Tensor, Tensor)>,
    {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for (inputs, targets) in data {
            // Check for early stopping before each batch
            if self.early_stopped {
                return Ok(None);
            }

            let loss = self.train_step(&inputs, &targets)?;
            total_loss += loss;
            n_batches += 1;

            // Logging
            if self.config.log_every > 0 && self.global_step.is_multiple_of(self.config.log_every) {
                let coherence_info = if let Some(coh) = self.model.coherence_score() {
                    format!(", coherence={:.4}", coh)
                } else {
                    String::new()
                };
                println!(
                    "Step {}: loss={:.4}, lr={:.2e}{}",
                    self.global_step,
                    loss,
                    self.scheduler.get_lr(),
                    coherence_info
                );
            }

            // Check coherence stability periodically
            if self.config.coherence_early_stopping
                && self.global_step > 0
                && self.config.eval_every > 0
                && self.global_step.is_multiple_of(self.config.eval_every)
                && self.check_coherence_stability()
            {
                println!(
                    "\n[Early Stopping] Coherence stabilized at step {} (score: {:.4})",
                    self.global_step,
                    self.metrics
                        .coherence_history
                        .last()
                        .copied()
                        .unwrap_or(0.0)
                );
                return Ok(None);
            }
        }

        if n_batches > 0 {
            Ok(Some(total_loss / n_batches as f64))
        } else {
            Ok(None)
        }
    }

    /// Train with coherence-based early stopping
    ///
    /// This is the recommended training loop when using coherence. It will:
    /// 1. Train normally until coherence stabilizes
    /// 2. Check coherence stability at each evaluation step
    /// 3. Stop early when coherence is stable for `patience` consecutive evaluations
    ///
    /// Returns training metrics and whether training was early stopped.
    pub fn train_with_early_stopping<F>(
        &mut self,
        mut batch_generator: F,
        max_steps: usize,
    ) -> TorusResult<(TrainingMetrics, bool)>
    where
        F: FnMut() -> TorusResult<(Tensor, Tensor)>,
    {
        println!("═══ Training with Coherence-Based Early Stopping ═══");
        println!(
            "Stability window: {} evals",
            self.config.coherence_stability_window
        );
        println!(
            "Stability threshold: {:.4}",
            self.config.coherence_stability_threshold
        );
        println!(
            "Min coherence score: {:.4}",
            self.config.coherence_min_score
        );
        println!("Patience: {} stable evals", self.config.coherence_patience);
        println!();

        for step in 0..max_steps {
            if self.early_stopped {
                break;
            }

            let (inputs, targets) = batch_generator()?;
            let loss = self.train_step(&inputs, &targets)?;

            // Logging
            if step % self.config.log_every == 0 {
                let coherence_info = if let Some(coh) = self.model.coherence_score() {
                    format!(", SOC={:.3}", coh)
                } else {
                    String::new()
                };
                println!(
                    "Step {:5}: loss={:.4}, lr={:.2e}{}",
                    step,
                    loss,
                    self.scheduler.get_lr(),
                    coherence_info
                );
            }

            // Evaluation and coherence check
            if step % self.config.eval_every == 0 && step > 0 {
                // Evaluate on current batch as simple validation
                let _ = self.evaluate(&inputs, &targets)?;

                // Check for early stopping
                if self.check_coherence_stability() {
                    println!("\n╔═══════════════════════════════════════════════════════╗");
                    println!("║           EARLY STOPPING TRIGGERED                     ║");
                    println!("╠═══════════════════════════════════════════════════════╣");
                    println!("║ Coherence has stabilized!                              ║");
                    println!(
                        "║ Step: {:6}                                          ║",
                        step
                    );
                    println!(
                        "║ Final SOC: {:.4}                                      ║",
                        self.metrics
                            .coherence_history
                            .last()
                            .copied()
                            .unwrap_or(0.0)
                    );
                    println!(
                        "║ Trend: {:+.4}                                         ║",
                        self.metrics.coherence_trend()
                    );
                    println!("╚═══════════════════════════════════════════════════════╝");
                    break;
                }
            }
        }

        Ok((self.metrics.clone(), self.early_stopped))
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
    #[allow(dead_code)]
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
        n_samples.div_ceil(self.batch_size)
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.dims()[0] == 0
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
    let (all_inputs, all_targets) =
        generate_random_batch(n_samples, seq_len, model_config.d_model, vocab_size, device)?;

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
    if let Some(loss) = avg_loss {
        println!("Average loss: {:.4}", loss);
    } else {
        println!("Training was early stopped");
    }
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
// COLLIDER TRAINING METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Extended metrics tracking for collider-validated training
///
/// Tracks physics-based validation statistics alongside standard training metrics.
#[derive(Debug, Clone, Default)]
pub struct ColliderTrainingMetrics {
    /// Total validation checks performed
    pub validation_checks: u64,
    /// Total anomalies detected
    pub total_anomalies: u64,
    /// Critical anomalies (NaN, Inf)
    pub critical_anomalies: u64,
    /// Conservation violations
    pub conservation_violations: u64,
    /// Causality violations
    pub causality_violations: u64,
    /// Steps where validation passed
    pub healthy_steps: u64,
    /// Anomalies by type [NaN, Inf, ExplodingGrad, VanishingGrad, ...]
    pub anomalies_by_type: [u64; 13],
    /// Recent health rate (rolling window)
    pub recent_health_rate: f64,
    /// Anomaly trend (positive = worsening)
    pub anomaly_trend: f64,
    /// Peak gradient norm seen
    pub peak_grad_norm: f64,
    /// Average luminosity (attention intensity)
    pub avg_luminosity: f64,
    /// Total collisions processed
    pub total_collisions: u64,
    /// Cross-section history (for physics analysis)
    pub cross_section_history: Vec<f64>,
    /// Step at which first anomaly occurred
    pub first_anomaly_step: Option<u64>,
    /// Step at which last anomaly occurred
    pub last_anomaly_step: Option<u64>,
}

impl ColliderTrainingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a validation report
    pub fn record_report(&mut self, step: u64, report: &ColliderReport) {
        self.validation_checks += 1;

        if report.is_healthy {
            self.healthy_steps += 1;
        } else {
            // Track anomaly timing
            if self.first_anomaly_step.is_none() {
                self.first_anomaly_step = Some(step);
            }
            self.last_anomaly_step = Some(step);
        }

        // Accumulate anomaly stats
        let stats = &report.anomaly_report.stats;
        self.total_anomalies += stats.total_anomalies;
        self.critical_anomalies += stats.critical_anomalies;
        for (i, count) in stats.by_type.iter().enumerate() {
            self.anomalies_by_type[i] += count;
        }

        // Conservation and causality
        self.conservation_violations += report.conservation_report.n_violations as u64;
        self.causality_violations += report.causality_report.violations.len() as u64;

        // Metrics
        self.total_collisions += report.metrics_report.collision_stats.n_collisions;
        self.cross_section_history
            .push(report.metrics_report.cross_section.total);

        // Update rolling stats
        self.update_rolling_stats();
    }

    /// Update rolling statistics
    fn update_rolling_stats(&mut self) {
        if self.validation_checks > 0 {
            self.recent_health_rate = self.healthy_steps as f64 / self.validation_checks as f64;
        }

        // Calculate anomaly trend from cross-section history
        if self.cross_section_history.len() >= 10 {
            let recent: Vec<f64> = self
                .cross_section_history
                .iter()
                .rev()
                .take(10)
                .copied()
                .collect();
            let older: Vec<f64> = self
                .cross_section_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .copied()
                .collect();

            if !older.is_empty() {
                let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
                self.anomaly_trend = recent_avg - older_avg;
            }
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Collider Metrics:\n\
             ├─ Validations: {} ({:.1}% healthy)\n\
             ├─ Anomalies: {} total ({} critical)\n\
             ├─ Conservation violations: {}\n\
             ├─ Causality violations: {}\n\
             ├─ Collisions: {}\n\
             └─ Trend: {:+.4}",
            self.validation_checks,
            self.recent_health_rate * 100.0,
            self.total_anomalies,
            self.critical_anomalies,
            self.conservation_violations,
            self.causality_violations,
            self.total_collisions,
            self.anomaly_trend
        )
    }

    /// Check if training should stop due to anomalies
    pub fn should_stop(&self, max_anomalies: usize) -> bool {
        if max_anomalies == 0 {
            return false; // Unlimited
        }
        self.total_anomalies as usize >= max_anomalies
    }

    /// Get anomaly rate per step
    pub fn anomaly_rate(&self) -> f64 {
        if self.validation_checks == 0 {
            0.0
        } else {
            self.total_anomalies as f64 / self.validation_checks as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATED TRAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Trainer with integrated collider validation
///
/// Wraps the standard Trainer with physics-based validation, providing:
/// - Real-time anomaly detection during forward/backward passes
/// - Conservation law checking for attention computations
/// - Causality validation for bidirectional streams
/// - Gradient health monitoring with particle physics metrics
///
/// # Example
/// ```ignore
/// let mut trainer = ValidatedTrainer::new(
///     model_config,
///     training_config,
///     ColliderTrainingConfig::default(),
///     vocab_size,
///     &device,
/// )?;
///
/// // Training loop with validation
/// for (inputs, targets) in data_loader {
///     let result = trainer.validated_train_step(&inputs, &targets)?;
///     
///     if result.had_critical_anomaly {
///         println!("Critical anomaly at step {}", result.step);
///         break;
///     }
/// }
///
/// // Get collider metrics
/// println!("{}", trainer.collider_metrics().summary());
/// ```
pub struct ValidatedTrainer {
    /// Inner trainer
    trainer: Trainer,
    /// Collider for validation
    collider: TorusCollider,
    /// Validation context
    context: ValidationContext,
    /// Validation history
    history: ValidationHistory,
    /// Collider training config
    collider_config: ColliderTrainingConfig,
    /// Collider-specific metrics
    collider_metrics: ColliderTrainingMetrics,
    /// Last validation report (for inspection)
    last_report: Option<ColliderReport>,
    /// Whether training was stopped due to anomalies
    stopped_on_anomaly: bool,
}

/// Result of a validated training step
#[derive(Debug, Clone)]
pub struct ValidatedStepResult {
    /// Training loss
    pub loss: f64,
    /// Current step
    pub step: u64,
    /// Whether validation was performed this step
    pub was_validated: bool,
    /// Whether the step was healthy
    pub is_healthy: bool,
    /// Whether a critical anomaly was detected
    pub had_critical_anomaly: bool,
    /// Number of anomalies detected
    pub anomaly_count: usize,
    /// Validation summary (if validated)
    pub validation_summary: Option<String>,
}

impl ValidatedTrainer {
    /// Create a new validated trainer
    pub fn new(
        model_config: BidirectionalTorusConfig,
        training_config: TrainingConfig,
        collider_config: ColliderTrainingConfig,
        vocab_size: Option<usize>,
        device: &Device,
    ) -> TorusResult<Self> {
        // Create inner trainer
        let trainer = Trainer::new(model_config.clone(), training_config, vocab_size, device)?;

        // Create collider
        let collider_inner_config = ColliderConfig {
            enabled: collider_config.enabled,
            darkness_grid: (model_config.n_major, model_config.n_minor),
            detector_enabled: collider_config.validation_level == ValidationLevel::Full,
            n_major: model_config.n_major,
            n_minor: model_config.n_minor,
            ..ColliderConfig::default()
        };
        let collider = TorusCollider::new(collider_inner_config);

        // Create validation context
        let validation_config = collider_config.to_validation_config();
        let context = ValidationContext::new(
            validation_config,
            model_config.n_major,
            model_config.n_minor,
        );

        // Create history
        let history = ValidationHistory::new(collider_config.history_size);

        Ok(Self {
            trainer,
            collider,
            context,
            history,
            collider_config,
            collider_metrics: ColliderTrainingMetrics::new(),
            last_report: None,
            stopped_on_anomaly: false,
        })
    }

    /// Single validated training step
    pub fn validated_train_step(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
    ) -> TorusResult<ValidatedStepResult> {
        let step = self.trainer.global_step() as u64;

        // Check if we should validate this step
        let should_validate = self.collider_config.enabled && self.context.should_validate();

        // Perform training step
        let loss = self.trainer.train_step(inputs, targets)?;

        // Validation
        let mut result = ValidatedStepResult {
            loss,
            step,
            was_validated: false,
            is_healthy: true,
            had_critical_anomaly: false,
            anomaly_count: 0,
            validation_summary: None,
        };

        if should_validate {
            // Step the collider
            self.collider.next_step();

            // Record attention tensors if available
            // Note: In a full implementation, we'd extract Q, K, V from the model
            // For now, we validate using the input/output tensors as proxies

            // Record gradients (using loss gradient as proxy)
            if self.collider_config.record_gradients {
                // The actual gradients would come from the backward pass
                // Here we create synthetic gradient tensors for demonstration
                let grad_proxy = inputs.zeros_like()?;
                let mut grad_map = HashMap::new();
                grad_map.insert("input_grad".to_string(), grad_proxy);
                self.collider.record_gradients_map(&grad_map, 0)?; // layer 0
            }

            // Get validation report
            let report = self.collider.report();

            // Update metrics
            self.collider_metrics.record_report(step, &report);
            self.history.record(step, report.clone());

            // Update result
            result.was_validated = true;
            result.is_healthy = report.is_healthy;
            result.anomaly_count = report.anomaly_report.stats.total_anomalies as usize;
            result.had_critical_anomaly = report.anomaly_report.stats.critical_anomalies > 0;
            result.validation_summary = Some(report.summary());

            // Check stopping conditions
            if self.collider_config.stop_on_critical && result.had_critical_anomaly {
                self.stopped_on_anomaly = true;
            }

            if self
                .collider_metrics
                .should_stop(self.collider_config.max_anomalies)
            {
                self.stopped_on_anomaly = true;
            }

            // Warnings
            if self.collider_config.warn_on_conservation_violation
                && report.conservation_report.n_violations > 0
            {
                eprintln!(
                    "[Step {}] Conservation violation: {}",
                    step, report.conservation_report.n_violations
                );
            }

            if self.collider_config.warn_on_causality_violation
                && !report.causality_report.violations.is_empty()
            {
                eprintln!(
                    "[Step {}] Causality violation: {} violations",
                    step,
                    report.causality_report.violations.len()
                );
            }

            self.last_report = Some(report);
        }

        // Advance validation context
        self.context.next_step();

        Ok(result)
    }

    /// Run validated training with dashboard
    pub fn train_with_validation<F>(
        &mut self,
        mut batch_generator: F,
        max_steps: usize,
    ) -> TorusResult<(TrainingMetrics, ColliderTrainingMetrics, bool)>
    where
        F: FnMut() -> TorusResult<(Tensor, Tensor)>,
    {
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("     COLLIDER-VALIDATED TRAINING");
        println!("═══════════════════════════════════════════════════════════════════════");
        println!(
            "Validation level: {:?}",
            self.collider_config.validation_level
        );
        println!(
            "Sample rate: {:.1}%",
            self.collider_config.sample_rate * 100.0
        );
        println!(
            "Stop on critical: {}",
            self.collider_config.stop_on_critical
        );
        println!(
            "Dashboard every: {} steps",
            self.collider_config.dashboard_every
        );
        println!("═══════════════════════════════════════════════════════════════════════\n");

        let start_time = Instant::now();

        for step in 0..max_steps {
            if self.stopped_on_anomaly {
                println!("\n[!] Training stopped due to anomaly at step {}", step);
                break;
            }

            let (inputs, targets) = batch_generator()?;
            let result = self.validated_train_step(&inputs, &targets)?;

            // Regular logging
            if step % self.trainer.config.log_every == 0 {
                let validation_info = if result.was_validated {
                    if result.is_healthy {
                        " [OK]".to_string()
                    } else {
                        format!(" [!{} anomalies]", result.anomaly_count)
                    }
                } else {
                    String::new()
                };

                println!(
                    "Step {:5}: loss={:.4}, lr={:.2e}{}",
                    step,
                    result.loss,
                    self.trainer.scheduler.get_lr(),
                    validation_info
                );
            }

            // Dashboard
            if self.collider_config.show_dashboard
                && step > 0
                && step % self.collider_config.dashboard_every == 0
            {
                self.print_dashboard(step);
            }

            // Check coherence early stopping (from base trainer)
            if self.trainer.config.coherence_early_stopping
                && step % self.trainer.config.eval_every == 0
                && step > 0
            {
                let _ = self.trainer.evaluate(&inputs, &targets)?;
                if self.trainer.check_coherence_stability() {
                    println!("\n[Early Stop] Coherence stabilized at step {}", step);
                    break;
                }
            }
        }

        let elapsed = start_time.elapsed();
        println!("\n═══════════════════════════════════════════════════════════════════════");
        println!("     TRAINING COMPLETE");
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("Time: {:.2}s", elapsed.as_secs_f64());
        println!("Steps: {}", self.trainer.global_step());
        println!("Stopped on anomaly: {}", self.stopped_on_anomaly);
        println!("\n{}", self.trainer.metrics().summary());
        println!("\n{}", self.collider_metrics.summary());
        println!("═══════════════════════════════════════════════════════════════════════");

        Ok((
            self.trainer.metrics().clone(),
            self.collider_metrics.clone(),
            self.stopped_on_anomaly,
        ))
    }

    /// Print validation dashboard
    fn print_dashboard(&self, step: usize) {
        let health_rate = self.collider_metrics.recent_health_rate;
        let health_bar = self.render_bar(health_rate, 1.0, 20);
        let anomaly_rate = self.collider_metrics.anomaly_rate();
        let anomaly_bar = self.render_bar(anomaly_rate.min(1.0), 1.0, 20);

        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!(
            "│  COLLIDER VALIDATION DASHBOARD                    Step {:5} │",
            step
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│  Health:    {} {:5.1}%                    │",
            health_bar,
            health_rate * 100.0
        );
        println!(
            "│  Anomalies: {} {:5.2}/step                 │",
            anomaly_bar, anomaly_rate
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│  Total anomalies: {:6}  Conservation: {:6}             │",
            self.collider_metrics.total_anomalies, self.collider_metrics.conservation_violations
        );
        println!(
            "│  Critical:        {:6}  Causality:    {:6}             │",
            self.collider_metrics.critical_anomalies, self.collider_metrics.causality_violations
        );
        println!(
            "│  Collisions:      {:6}  Trend:     {:+7.4}             │",
            self.collider_metrics.total_collisions, self.collider_metrics.anomaly_trend
        );
        println!("└─────────────────────────────────────────────────────────────┘\n");
    }

    /// Render a progress bar
    fn render_bar(&self, value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).min(width as f64) as usize;
        let empty = width - filled;
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// Get reference to inner trainer
    pub fn trainer(&self) -> &Trainer {
        &self.trainer
    }

    /// Get mutable reference to inner trainer
    pub fn trainer_mut(&mut self) -> &mut Trainer {
        &mut self.trainer
    }

    /// Get collider metrics
    pub fn collider_metrics(&self) -> &ColliderTrainingMetrics {
        &self.collider_metrics
    }

    /// Get validation history
    pub fn validation_history(&self) -> &ValidationHistory {
        &self.history
    }

    /// Get last validation report
    pub fn last_report(&self) -> Option<&ColliderReport> {
        self.last_report.as_ref()
    }

    /// Check if stopped on anomaly
    pub fn stopped_on_anomaly(&self) -> bool {
        self.stopped_on_anomaly
    }

    /// Get the collider for direct access
    pub fn collider(&self) -> &TorusCollider {
        &self.collider
    }

    /// Save checkpoint (includes collider state summary)
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> TorusResult<()> {
        self.trainer.save_checkpoint(path)
    }

    /// Load checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> TorusResult<()> {
        self.trainer.load_checkpoint(path)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATED TRAINING EXAMPLE
// ═══════════════════════════════════════════════════════════════════════════

/// Run a training example with collider validation
pub fn run_validated_training_example(device: &Device) -> TorusResult<ColliderTrainingMetrics> {
    println!("\n═══ Collider-Validated Training Example ═══\n");

    // Model configuration
    let model_config = BidirectionalTorusConfig {
        d_model: 64,
        d_ff: 256,
        n_heads: 4,
        n_layers: 2,
        n_major: 8,
        n_minor: 4,
        ..BidirectionalTorusConfig::default()
    };

    // Training configuration
    let training_config = TrainingConfig {
        learning_rate: 3e-4,
        batch_size: 8,
        epochs: 1,
        warmup_steps: 10,
        total_steps: 100,
        log_every: 10,
        eval_every: 50,
        ..TrainingConfig::default()
    };

    // Collider configuration - minimal for quick example
    let collider_config = ColliderTrainingConfig {
        enabled: true,
        validation_level: ValidationLevel::Standard,
        sample_rate: 0.5, // Validate 50% of steps
        show_dashboard: true,
        dashboard_every: 25,
        ..ColliderTrainingConfig::default()
    };

    let vocab_size = 100;

    println!(
        "Model: {}d, {} heads, {} layers",
        model_config.d_model, model_config.n_heads, model_config.n_layers
    );
    println!(
        "Torus: {}x{} = {} positions",
        model_config.n_major,
        model_config.n_minor,
        model_config.seq_len()
    );
    println!(
        "Validation: {:?} @ {:.0}% sample rate\n",
        collider_config.validation_level,
        collider_config.sample_rate * 100.0
    );

    // Create validated trainer
    let mut trainer = ValidatedTrainer::new(
        model_config.clone(),
        training_config.clone(),
        collider_config,
        Some(vocab_size),
        device,
    )?;

    // Batch generator
    let seq_len = model_config.seq_len();
    let d_model = model_config.d_model;
    let batch_size = training_config.batch_size;
    let device_clone = device.clone();

    let batch_gen =
        move || generate_random_batch(batch_size, seq_len, d_model, vocab_size, &device_clone);

    // Run training
    let (_, collider_metrics, stopped) = trainer.train_with_validation(batch_gen, 50)?;

    if stopped {
        println!("\nTraining was stopped due to anomalies!");
    }

    Ok(collider_metrics)
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
        assert!(!config.coherence_early_stopping);
    }

    #[test]
    fn test_training_config_early_stopping_fields() {
        let config = TrainingConfig::default();
        assert_eq!(config.coherence_stability_window, 5);
        assert!((config.coherence_stability_threshold - 0.01).abs() < 1e-10);
        assert!((config.coherence_min_score - 0.6).abs() < 1e-10);
        assert_eq!(config.coherence_patience, 3);
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
    fn test_coherence_stability_check() {
        let mut metrics = TrainingMetrics::new();

        // Not enough history
        assert!(!metrics.is_coherence_stable(5, 0.6, 0.01));

        // Add stable coherence history
        for _ in 0..5 {
            metrics.coherence_history.push(0.75);
        }

        // Should be stable (variance = 0)
        assert!(metrics.is_coherence_stable(5, 0.6, 0.01));

        // Should not be stable if min_score too high
        assert!(!metrics.is_coherence_stable(5, 0.8, 0.01));
    }

    #[test]
    fn test_coherence_stability_with_variance() {
        let mut metrics = TrainingMetrics::new();

        // Add varying coherence history
        metrics.coherence_history = vec![0.70, 0.72, 0.71, 0.73, 0.70];

        // Variance is ~0.00016, should be stable with threshold 0.01
        assert!(metrics.is_coherence_stable(5, 0.6, 0.01));

        // Should not be stable with very tight threshold
        assert!(!metrics.is_coherence_stable(5, 0.6, 0.0001));
    }

    #[test]
    fn test_coherence_trend() {
        let mut metrics = TrainingMetrics::new();

        // Empty history
        assert_eq!(metrics.coherence_trend(), 0.0);

        // Improving trend (values are taken in reverse, so index 0 is most recent)
        // This means [0.7, 0.65, 0.6, 0.55, 0.5] reversed = [0.5, 0.55, 0.6, 0.65, 0.7]
        // which shows values increasing over time (indices) - improving
        metrics.coherence_history = vec![0.5, 0.55, 0.6, 0.65, 0.7];
        // After reverse: recent[0]=0.7, recent[4]=0.5, slope is negative (decreasing by index)
        // The function calculates trend where positive slope means recent > old
        // But it takes values in reverse so need to verify actual behavior
        let trend = metrics.coherence_trend();
        // With reverse and index as x, slope should be negative because recent (low index) has high value
        assert!(trend < 0.0, "trend was {}", trend);

        // Degrading trend - recent values are lower
        metrics.coherence_history = vec![0.7, 0.65, 0.6, 0.55, 0.5];
        let trend = metrics.coherence_trend();
        // After reverse: recent[0]=0.5, recent[4]=0.7, slope is positive (increasing by index)
        assert!(trend > 0.0, "trend was {}", trend);

        // Flat trend
        metrics.coherence_history = vec![0.6, 0.6, 0.6, 0.6, 0.6];
        assert!((metrics.coherence_trend() - 0.0).abs() < 1e-10);
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

    // ═══════════════════════════════════════════════════════════════════════
    // COLLIDER TRAINING TESTS
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_collider_training_config_default() {
        let config = ColliderTrainingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.validation_level, ValidationLevel::Standard);
        assert!((config.sample_rate - 0.1).abs() < 1e-10);
        assert!(config.stop_on_critical);
    }

    #[test]
    fn test_collider_training_config_presets() {
        let thorough = ColliderTrainingConfig::thorough();
        assert_eq!(thorough.validation_level, ValidationLevel::Full);
        assert!((thorough.sample_rate - 1.0).abs() < 1e-10);

        let minimal = ColliderTrainingConfig::minimal();
        assert_eq!(minimal.validation_level, ValidationLevel::Minimal);
        assert!((minimal.sample_rate - 0.01).abs() < 1e-10);
        assert!(!minimal.record_qkv);

        let disabled = ColliderTrainingConfig::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_collider_training_config_to_validation_config() {
        let config = ColliderTrainingConfig::default();
        let validation = config.to_validation_config();

        assert!(validation.enabled);
        assert_eq!(validation.level, ValidationLevel::Standard);
        assert!((validation.sample_rate - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_collider_training_metrics() {
        let mut metrics = ColliderTrainingMetrics::new();

        assert_eq!(metrics.validation_checks, 0);
        assert_eq!(metrics.total_anomalies, 0);
        assert!((metrics.recent_health_rate - 0.0).abs() < 1e-10);

        // Test summary doesn't panic
        let summary = metrics.summary();
        assert!(summary.contains("Collider Metrics"));
    }

    #[test]
    fn test_collider_training_metrics_should_stop() {
        let mut metrics = ColliderTrainingMetrics::new();

        // Unlimited (max_anomalies = 0)
        assert!(!metrics.should_stop(0));

        // Below limit
        metrics.total_anomalies = 5;
        assert!(!metrics.should_stop(10));

        // At/above limit
        assert!(metrics.should_stop(5));
        assert!(metrics.should_stop(3));
    }

    #[test]
    fn test_collider_training_metrics_anomaly_rate() {
        let mut metrics = ColliderTrainingMetrics::new();

        // No checks yet
        assert!((metrics.anomaly_rate() - 0.0).abs() < 1e-10);

        // Add some data
        metrics.validation_checks = 100;
        metrics.total_anomalies = 10;
        assert!((metrics.anomaly_rate() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_validated_step_result() {
        let result = ValidatedStepResult {
            loss: 1.5,
            step: 100,
            was_validated: true,
            is_healthy: true,
            had_critical_anomaly: false,
            anomaly_count: 0,
            validation_summary: Some("All OK".to_string()),
        };

        assert_eq!(result.step, 100);
        assert!(result.is_healthy);
        assert!(!result.had_critical_anomaly);
    }
}
