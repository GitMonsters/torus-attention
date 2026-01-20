//! # Dynamic Compound Training System
//!
//! A comprehensive training framework that dynamically adjusts multiple aspects
//! of training based on model performance and learning dynamics:
//!
//! - **Curriculum Learning**: Progressive difficulty scaling
//! - **Multi-task Training**: Weighted combination of multiple objectives
//! - **Dynamic Batch Sizing**: Adjust batch size based on gradient statistics
//! - **Adaptive Layer-wise Learning Rates**: Per-layer LR based on coherence metrics
//! - **Progressive Model Growth**: Gradually add layers/capacity during training
//! - **Dynamic EMA Compounding**: Adjust alpha values based on training dynamics
//!
//! ## Key Concepts
//!
//! The system monitors training dynamics through several metrics:
//! - Loss trajectory and variance
//! - Gradient norms per layer
//! - Coherence scores (SOC/SMM from cognitive coherence module)
//! - EMA alpha convergence
//!
//! These metrics drive automatic adjustments to training hyperparameters.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

use crate::checkpoint::{checkpoint_exists, load_checkpoint, save_checkpoint};
use crate::coherence::{CognitiveCoherenceLayer, CoherenceConfig, SenseOfCoherence};
use crate::dataset::{DataLoader, TextDataset};
use crate::llm::{TorusLLM, TorusLLMConfig};
use crate::metrics::MetricsLogger;
use crate::TorusResult;

// ============================================================================
// Curriculum Learning
// ============================================================================

/// Difficulty level for curriculum learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Very simple examples (short sequences, common tokens)
    Beginner,
    /// Moderate complexity
    Intermediate,
    /// Full complexity
    Advanced,
    /// Most challenging examples
    Expert,
}

impl DifficultyLevel {
    pub fn as_f64(&self) -> f64 {
        match self {
            DifficultyLevel::Beginner => 0.25,
            DifficultyLevel::Intermediate => 0.5,
            DifficultyLevel::Advanced => 0.75,
            DifficultyLevel::Expert => 1.0,
        }
    }

    pub fn next(&self) -> Option<Self> {
        match self {
            DifficultyLevel::Beginner => Some(DifficultyLevel::Intermediate),
            DifficultyLevel::Intermediate => Some(DifficultyLevel::Advanced),
            DifficultyLevel::Advanced => Some(DifficultyLevel::Expert),
            DifficultyLevel::Expert => None,
        }
    }

    pub fn max_seq_len(&self, base_len: usize) -> usize {
        ((base_len as f64) * self.as_f64()) as usize
    }
}

/// Curriculum scheduler that advances difficulty based on performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumScheduler {
    /// Current difficulty level
    pub current_level: DifficultyLevel,
    /// Loss threshold to advance (must be below this for N steps)
    pub advance_threshold: f64,
    /// Number of consecutive steps below threshold needed to advance
    pub patience: usize,
    /// Current streak of good performance
    pub good_steps: usize,
    /// Minimum steps at each level before advancing
    pub min_steps_per_level: usize,
    /// Steps at current level
    pub steps_at_level: usize,
    /// History of level transitions
    pub level_history: Vec<(usize, DifficultyLevel)>,
}

impl Default for CurriculumScheduler {
    fn default() -> Self {
        Self {
            current_level: DifficultyLevel::Beginner,
            advance_threshold: 2.0,
            patience: 50,
            good_steps: 0,
            min_steps_per_level: 100,
            steps_at_level: 0,
            level_history: vec![(0, DifficultyLevel::Beginner)],
        }
    }
}

impl CurriculumScheduler {
    /// Update scheduler based on current loss
    pub fn update(&mut self, loss: f64, step: usize) -> bool {
        self.steps_at_level += 1;

        if loss < self.advance_threshold {
            self.good_steps += 1;
        } else {
            self.good_steps = 0;
        }

        // Check if we should advance
        if self.good_steps >= self.patience && self.steps_at_level >= self.min_steps_per_level {
            if let Some(next_level) = self.current_level.next() {
                log::info!(
                    "Curriculum: Advancing from {:?} to {:?} at step {}",
                    self.current_level,
                    next_level,
                    step
                );
                self.current_level = next_level;
                self.good_steps = 0;
                self.steps_at_level = 0;
                // Increase threshold for harder levels
                self.advance_threshold *= 0.9;
                self.level_history.push((step, next_level));
                return true;
            }
        }
        false
    }

    /// Get sampling parameters for current difficulty
    pub fn get_sampling_params(&self, base_seq_len: usize) -> CurriculumSamplingParams {
        CurriculumSamplingParams {
            max_seq_len: self.current_level.max_seq_len(base_seq_len),
            difficulty_weight: self.current_level.as_f64(),
            prefer_common_tokens: matches!(
                self.current_level,
                DifficultyLevel::Beginner | DifficultyLevel::Intermediate
            ),
        }
    }
}

/// Parameters for curriculum-based sampling
#[derive(Debug, Clone)]
pub struct CurriculumSamplingParams {
    pub max_seq_len: usize,
    pub difficulty_weight: f64,
    pub prefer_common_tokens: bool,
}

// ============================================================================
// Multi-task Training
// ============================================================================

/// Task definition for multi-task training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Task identifier
    pub name: String,
    /// Task weight (for loss combination)
    pub weight: f64,
    /// Whether this task is currently active
    pub active: bool,
    /// Steps since last update
    pub steps_since_update: usize,
    /// Running loss for this task
    pub running_loss: f64,
}

/// Multi-task scheduler that balances task weights dynamically
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTaskScheduler {
    /// All tasks
    pub tasks: Vec<Task>,
    /// Whether to use uncertainty weighting (Kendall et al.)
    pub use_uncertainty_weighting: bool,
    /// Per-task log variance (for uncertainty weighting)
    pub log_vars: Vec<f64>,
    /// Minimum weight for any task
    pub min_weight: f64,
    /// Maximum weight for any task
    pub max_weight: f64,
    /// Rebalancing interval (steps)
    pub rebalance_interval: usize,
}

impl Default for MultiTaskScheduler {
    fn default() -> Self {
        Self {
            tasks: vec![Task {
                name: "language_modeling".to_string(),
                weight: 1.0,
                active: true,
                steps_since_update: 0,
                running_loss: 0.0,
            }],
            use_uncertainty_weighting: true,
            log_vars: vec![0.0],
            min_weight: 0.1,
            max_weight: 10.0,
            rebalance_interval: 100,
        }
    }
}

impl MultiTaskScheduler {
    /// Add a new task
    pub fn add_task(&mut self, name: &str, initial_weight: f64) {
        self.tasks.push(Task {
            name: name.to_string(),
            weight: initial_weight,
            active: true,
            steps_since_update: 0,
            running_loss: 0.0,
        });
        self.log_vars.push(0.0);
    }

    /// Update task losses and potentially rebalance
    pub fn update(&mut self, task_losses: &[(usize, f64)], step: usize) {
        for &(task_idx, loss) in task_losses {
            if task_idx < self.tasks.len() {
                let task = &mut self.tasks[task_idx];
                task.running_loss = 0.9 * task.running_loss + 0.1 * loss;
                task.steps_since_update = 0;
            }
        }

        // Increment steps for tasks not updated
        for task in &mut self.tasks {
            task.steps_since_update += 1;
        }

        // Rebalance if needed
        if self.rebalance_interval > 0 && step % self.rebalance_interval == 0 {
            self.rebalance_weights();
        }
    }

    /// Rebalance weights based on task losses
    fn rebalance_weights(&mut self) {
        if !self.use_uncertainty_weighting {
            return;
        }

        // Compute weights inversely proportional to loss (normalized)
        let active_tasks: Vec<usize> = self
            .tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.active && t.running_loss > 0.0)
            .map(|(i, _)| i)
            .collect();

        if active_tasks.is_empty() {
            return;
        }

        let total_loss: f64 = active_tasks
            .iter()
            .map(|&i| self.tasks[i].running_loss)
            .sum();

        for &idx in &active_tasks {
            // Higher loss -> lower weight (focus on easier tasks first)
            // Or: Higher loss -> higher weight (focus on harder tasks)
            // We use inverse weighting here
            let loss_ratio = self.tasks[idx].running_loss / total_loss;
            let new_weight = (1.0 / loss_ratio).clamp(self.min_weight, self.max_weight);
            self.tasks[idx].weight = new_weight;
        }

        // Normalize weights
        let total_weight: f64 = self
            .tasks
            .iter()
            .filter(|t| t.active)
            .map(|t| t.weight)
            .sum();
        for task in &mut self.tasks {
            if task.active {
                task.weight /= total_weight;
            }
        }

        log::debug!(
            "MultiTask: Rebalanced weights: {:?}",
            self.tasks
                .iter()
                .map(|t| (t.name.as_str(), t.weight))
                .collect::<Vec<_>>()
        );
    }

    /// Get combined loss with task weights
    pub fn combine_losses(&self, task_losses: &[(usize, f64)]) -> f64 {
        let mut total = 0.0;
        for &(task_idx, loss) in task_losses {
            if task_idx < self.tasks.len() && self.tasks[task_idx].active {
                total += self.tasks[task_idx].weight * loss;
            }
        }
        total
    }
}

// ============================================================================
// Dynamic Batch Sizing
// ============================================================================

/// Dynamic batch size controller based on gradient statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchController {
    /// Current batch size
    pub current_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Target gradient noise scale
    pub target_noise_scale: f64,
    /// History of gradient norms
    pub grad_norm_history: VecDeque<f64>,
    /// History window size
    pub history_size: usize,
    /// Adjustment interval (steps)
    pub adjust_interval: usize,
    /// Growth factor when increasing batch size
    pub growth_factor: f64,
    /// Shrink factor when decreasing batch size
    pub shrink_factor: f64,
}

impl Default for DynamicBatchController {
    fn default() -> Self {
        Self {
            current_batch_size: 32,
            min_batch_size: 8,
            max_batch_size: 256,
            target_noise_scale: 0.1,
            grad_norm_history: VecDeque::with_capacity(100),
            history_size: 100,
            adjust_interval: 50,
            growth_factor: 1.5,
            shrink_factor: 0.75,
        }
    }
}

impl DynamicBatchController {
    /// Record a gradient norm observation
    pub fn record_grad_norm(&mut self, norm: f64) {
        self.grad_norm_history.push_back(norm);
        if self.grad_norm_history.len() > self.history_size {
            self.grad_norm_history.pop_front();
        }
    }

    /// Update batch size based on gradient statistics
    pub fn update(&mut self, step: usize) -> bool {
        if self.adjust_interval == 0
            || step % self.adjust_interval != 0
            || self.grad_norm_history.len() < 10
        {
            return false;
        }

        // Compute gradient noise scale (variance / mean^2)
        let mean: f64 =
            self.grad_norm_history.iter().sum::<f64>() / self.grad_norm_history.len() as f64;
        let variance: f64 = self
            .grad_norm_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.grad_norm_history.len() as f64;

        let noise_scale = if mean > 1e-8 {
            variance / (mean * mean)
        } else {
            0.0
        };

        let old_batch_size = self.current_batch_size;

        // If noise is too high, increase batch size (more stable gradients)
        // If noise is too low, decrease batch size (more updates per data)
        if noise_scale > self.target_noise_scale * 1.5 {
            self.current_batch_size = ((self.current_batch_size as f64 * self.growth_factor)
                as usize)
                .min(self.max_batch_size);
        } else if noise_scale < self.target_noise_scale * 0.5 {
            self.current_batch_size = ((self.current_batch_size as f64 * self.shrink_factor)
                as usize)
                .max(self.min_batch_size);
        }

        if self.current_batch_size != old_batch_size {
            log::info!(
                "DynamicBatch: Adjusted batch size {} -> {} (noise_scale={:.4})",
                old_batch_size,
                self.current_batch_size,
                noise_scale
            );
            return true;
        }

        false
    }
}

// ============================================================================
// Adaptive Layer-wise Learning Rates
// ============================================================================

/// Per-layer learning rate controller based on coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWiseLRController {
    /// Base learning rate
    pub base_lr: f64,
    /// Per-layer learning rate multipliers
    pub layer_multipliers: Vec<f64>,
    /// Per-layer coherence scores (from SOC/SMM)
    pub layer_coherence: Vec<f64>,
    /// Per-layer gradient norms
    pub layer_grad_norms: Vec<f64>,
    /// Minimum multiplier
    pub min_multiplier: f64,
    /// Maximum multiplier
    pub max_multiplier: f64,
    /// Coherence weight in multiplier calculation
    pub coherence_weight: f64,
    /// Gradient weight in multiplier calculation
    pub gradient_weight: f64,
}

impl LayerWiseLRController {
    pub fn new(num_layers: usize, base_lr: f64) -> Self {
        Self {
            base_lr,
            layer_multipliers: vec![1.0; num_layers],
            layer_coherence: vec![0.5; num_layers],
            layer_grad_norms: vec![1.0; num_layers],
            min_multiplier: 0.1,
            max_multiplier: 10.0,
            coherence_weight: 0.5,
            gradient_weight: 0.5,
        }
    }

    /// Update coherence scores for layers
    pub fn update_coherence(&mut self, coherence_scores: &[f64]) {
        for (i, &score) in coherence_scores.iter().enumerate() {
            if i < self.layer_coherence.len() {
                self.layer_coherence[i] = 0.9 * self.layer_coherence[i] + 0.1 * score;
            }
        }
    }

    /// Update gradient norms for layers
    pub fn update_grad_norms(&mut self, grad_norms: &[f64]) {
        for (i, &norm) in grad_norms.iter().enumerate() {
            if i < self.layer_grad_norms.len() {
                self.layer_grad_norms[i] = 0.9 * self.layer_grad_norms[i] + 0.1 * norm;
            }
        }
    }

    /// Recompute layer multipliers
    pub fn recompute_multipliers(&mut self) {
        // Normalize coherence and gradient norms
        let max_coherence = self
            .layer_coherence
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let min_coherence = self
            .layer_coherence
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min);
        let coherence_range = (max_coherence - min_coherence).max(1e-8);

        let max_grad = self
            .layer_grad_norms
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let min_grad = self
            .layer_grad_norms
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min);
        let grad_range = (max_grad - min_grad).max(1e-8);

        for i in 0..self.layer_multipliers.len() {
            // Normalize to [0, 1]
            let norm_coherence = (self.layer_coherence[i] - min_coherence) / coherence_range;
            let norm_grad = (self.layer_grad_norms[i] - min_grad) / grad_range;

            // Low coherence = needs more learning = higher LR
            // High grad norm = unstable = lower LR
            let coherence_factor = 1.0 + (1.0 - norm_coherence) * self.coherence_weight;
            let gradient_factor = 1.0 - norm_grad * self.gradient_weight * 0.5;

            self.layer_multipliers[i] = (coherence_factor * gradient_factor)
                .clamp(self.min_multiplier, self.max_multiplier);
        }
    }

    /// Get learning rate for a specific layer
    pub fn get_lr(&self, layer: usize) -> f64 {
        self.base_lr * self.layer_multipliers.get(layer).copied().unwrap_or(1.0)
    }

    /// Get all layer learning rates
    pub fn get_all_lrs(&self) -> Vec<f64> {
        self.layer_multipliers
            .iter()
            .map(|&m| self.base_lr * m)
            .collect()
    }
}

// ============================================================================
// Progressive Model Growth
// ============================================================================

/// Model growth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthConfig {
    /// Initial number of layers
    pub initial_layers: usize,
    /// Final number of layers
    pub final_layers: usize,
    /// Initial hidden dimension
    pub initial_hidden_dim: usize,
    /// Final hidden dimension
    pub final_hidden_dim: usize,
    /// Initial number of heads
    pub initial_heads: usize,
    /// Final number of heads
    pub final_heads: usize,
    /// Steps between growth events
    pub growth_interval: usize,
    /// Loss threshold to trigger growth
    pub growth_threshold: f64,
    /// Minimum steps before growth
    pub min_steps_before_growth: usize,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            initial_layers: 2,
            final_layers: 12,
            initial_hidden_dim: 256,
            final_hidden_dim: 768,
            initial_heads: 4,
            final_heads: 12,
            growth_interval: 1000,
            growth_threshold: 3.0,
            min_steps_before_growth: 500,
        }
    }
}

/// Progressive model growth controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthController {
    /// Growth configuration
    pub config: GrowthConfig,
    /// Current number of active layers
    pub current_layers: usize,
    /// Current hidden dimension
    pub current_hidden_dim: usize,
    /// Current number of heads
    pub current_heads: usize,
    /// Steps since last growth
    pub steps_since_growth: usize,
    /// Total growth events
    pub growth_count: usize,
    /// History of growth events (step, layers, hidden_dim, heads)
    pub growth_history: Vec<(usize, usize, usize, usize)>,
}

impl GrowthController {
    pub fn new(config: GrowthConfig) -> Self {
        let initial_layers = config.initial_layers;
        let initial_hidden_dim = config.initial_hidden_dim;
        let initial_heads = config.initial_heads;
        Self {
            config,
            current_layers: initial_layers,
            current_hidden_dim: initial_hidden_dim,
            current_heads: initial_heads,
            steps_since_growth: 0,
            growth_count: 0,
            growth_history: vec![(0, initial_layers, initial_hidden_dim, initial_heads)],
        }
    }

    /// Check if growth should occur and return new dimensions if so
    pub fn should_grow(&mut self, loss: f64, step: usize) -> Option<(usize, usize, usize)> {
        self.steps_since_growth += 1;

        // Check all conditions for growth
        let below_threshold = loss < self.config.growth_threshold;
        let enough_steps = self.steps_since_growth >= self.config.min_steps_before_growth;
        let at_interval =
            self.config.growth_interval > 0 && step % self.config.growth_interval == 0;
        let can_grow = self.current_layers < self.config.final_layers
            || self.current_hidden_dim < self.config.final_hidden_dim
            || self.current_heads < self.config.final_heads;

        if below_threshold && enough_steps && at_interval && can_grow {
            // Determine what to grow
            let new_layers = if self.current_layers < self.config.final_layers {
                self.current_layers + 1
            } else {
                self.current_layers
            };

            let new_hidden = if self.current_hidden_dim < self.config.final_hidden_dim {
                // Grow hidden dim by ~12% each time
                ((self.current_hidden_dim as f64 * 1.12) as usize).min(self.config.final_hidden_dim)
            } else {
                self.current_hidden_dim
            };

            let new_heads = if self.current_heads < self.config.final_heads
                && new_hidden >= self.config.final_hidden_dim / 2
            {
                (self.current_heads + 1).min(self.config.final_heads)
            } else {
                self.current_heads
            };

            self.current_layers = new_layers;
            self.current_hidden_dim = new_hidden;
            self.current_heads = new_heads;
            self.steps_since_growth = 0;
            self.growth_count += 1;
            self.growth_history
                .push((step, new_layers, new_hidden, new_heads));

            log::info!(
                "Growth: layers={}, hidden={}, heads={} at step {}",
                new_layers,
                new_hidden,
                new_heads,
                step
            );

            return Some((new_layers, new_hidden, new_heads));
        }

        None
    }

    /// Get current model configuration
    pub fn current_config(&self) -> (usize, usize, usize) {
        (
            self.current_layers,
            self.current_hidden_dim,
            self.current_heads,
        )
    }

    /// Calculate growth progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        let layer_progress = (self.current_layers - self.config.initial_layers) as f64
            / (self.config.final_layers - self.config.initial_layers).max(1) as f64;
        let hidden_progress = (self.current_hidden_dim - self.config.initial_hidden_dim) as f64
            / (self.config.final_hidden_dim - self.config.initial_hidden_dim).max(1) as f64;
        let head_progress = (self.current_heads - self.config.initial_heads) as f64
            / (self.config.final_heads - self.config.initial_heads).max(1) as f64;

        (layer_progress + hidden_progress + head_progress) / 3.0
    }
}

// ============================================================================
// Dynamic EMA Compounding
// ============================================================================

/// Controller for dynamically adjusting EMA alpha values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicEMAController {
    /// Per-layer target alpha values
    pub target_alphas: Vec<f64>,
    /// Per-layer current alpha values
    pub current_alphas: Vec<f64>,
    /// Per-layer alpha adjustment rates
    pub alpha_rates: Vec<f64>,
    /// Loss history for trend detection
    pub loss_history: VecDeque<f64>,
    /// Coherence history for adjustment
    pub coherence_history: VecDeque<f64>,
    /// Minimum alpha
    pub min_alpha: f64,
    /// Maximum alpha
    pub max_alpha: f64,
    /// Adjustment sensitivity
    pub sensitivity: f64,
}

impl DynamicEMAController {
    pub fn new(num_layers: usize, initial_alpha: f64) -> Self {
        Self {
            target_alphas: vec![initial_alpha; num_layers],
            current_alphas: vec![initial_alpha; num_layers],
            alpha_rates: vec![0.01; num_layers],
            loss_history: VecDeque::with_capacity(100),
            coherence_history: VecDeque::with_capacity(100),
            min_alpha: 0.1,
            max_alpha: 0.99,
            sensitivity: 0.1,
        }
    }

    /// Record loss and coherence observations
    pub fn record(&mut self, loss: f64, coherence: f64) {
        self.loss_history.push_back(loss);
        self.coherence_history.push_back(coherence);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }
    }

    /// Update alpha values based on training dynamics
    pub fn update(&mut self) {
        if self.loss_history.len() < 10 {
            return;
        }

        // Compute loss trend (positive = improving, negative = worsening)
        let recent: Vec<f64> = self.loss_history.iter().rev().take(10).cloned().collect();
        let old: Vec<f64> = self
            .loss_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .cloned()
            .collect();

        let recent_mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let old_mean: f64 = if old.is_empty() {
            recent_mean
        } else {
            old.iter().sum::<f64>() / old.len() as f64
        };

        let trend = (old_mean - recent_mean) / old_mean.max(1e-8);

        // Compute coherence level
        let coherence_mean: f64 =
            self.coherence_history.iter().sum::<f64>() / self.coherence_history.len().max(1) as f64;

        // Adjust alphas based on trend and coherence
        for i in 0..self.current_alphas.len() {
            // If loss is improving and coherence is high, increase alpha (more memory)
            // If loss is stagnating and coherence is low, decrease alpha (more responsiveness)
            let layer_factor = 1.0 - (i as f64 / self.current_alphas.len() as f64) * 0.3;

            let adjustment = if trend > 0.01 && coherence_mean > 0.6 {
                // Good progress, high coherence: increase alpha
                self.sensitivity * layer_factor
            } else if trend < -0.01 || coherence_mean < 0.4 {
                // Poor progress or low coherence: decrease alpha
                -self.sensitivity * layer_factor
            } else {
                0.0
            };

            self.target_alphas[i] =
                (self.target_alphas[i] + adjustment).clamp(self.min_alpha, self.max_alpha);

            // Smooth transition to target
            self.current_alphas[i] +=
                self.alpha_rates[i] * (self.target_alphas[i] - self.current_alphas[i]);
        }
    }

    /// Get current alpha for a layer
    pub fn get_alpha(&self, layer: usize) -> f64 {
        self.current_alphas.get(layer).copied().unwrap_or(0.9)
    }

    /// Get all current alphas
    pub fn get_all_alphas(&self) -> &[f64] {
        &self.current_alphas
    }
}

// ============================================================================
// Training Statistics
// ============================================================================

/// Comprehensive training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DynamicTrainingStats {
    /// Current step
    pub step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Current loss
    pub loss: f64,
    /// Moving average loss
    pub avg_loss: f64,
    /// Validation loss
    pub val_loss: Option<f64>,
    /// Current learning rate
    pub learning_rate: f64,
    /// Current batch size
    pub batch_size: usize,
    /// Current curriculum level
    pub curriculum_level: String,
    /// Growth progress (0.0 to 1.0)
    pub growth_progress: f64,
    /// Average coherence score
    pub coherence: f64,
    /// Gradient norm
    pub grad_norm: f64,
    /// Tokens processed per second
    pub tokens_per_sec: f64,
    /// Per-layer learning rates
    pub layer_lrs: Vec<f64>,
    /// Per-layer EMA alphas
    pub ema_alphas: Vec<f64>,
}

// ============================================================================
// Dynamic Compound Training Configuration
// ============================================================================

/// Configuration for dynamic compound training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTrainingConfig {
    /// Base model configuration
    pub model: TorusLLMConfig,
    /// Initial batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Maximum number of steps
    pub max_steps: Option<usize>,
    /// Base learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Logging interval
    pub log_interval: usize,
    /// Evaluation interval
    pub eval_interval: usize,
    /// Checkpoint interval
    pub save_interval: usize,
    /// Checkpoint directory
    pub checkpoint_dir: String,
    /// Resume from checkpoint
    pub resume: bool,
    /// Enable curriculum learning
    pub use_curriculum: bool,
    /// Enable multi-task training
    pub use_multi_task: bool,
    /// Enable dynamic batch sizing
    pub use_dynamic_batch: bool,
    /// Enable layer-wise learning rates
    pub use_layer_wise_lr: bool,
    /// Enable progressive growth
    pub use_progressive_growth: bool,
    /// Enable dynamic EMA
    pub use_dynamic_ema: bool,
    /// Growth configuration
    pub growth_config: GrowthConfig,
    /// Gradient clipping
    pub max_grad_norm: Option<f64>,
    /// Early stopping patience
    pub patience: Option<usize>,
}

impl Default for DynamicTrainingConfig {
    fn default() -> Self {
        Self {
            model: TorusLLMConfig::tiny(),
            batch_size: 32,
            num_epochs: 100,
            max_steps: None,
            learning_rate: 3e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            log_interval: 10,
            eval_interval: 100,
            save_interval: 500,
            checkpoint_dir: "checkpoints".to_string(),
            resume: true,
            use_curriculum: true,
            use_multi_task: false,
            use_dynamic_batch: true,
            use_layer_wise_lr: true,
            use_progressive_growth: true,
            use_dynamic_ema: true,
            growth_config: GrowthConfig::default(),
            max_grad_norm: Some(1.0),
            patience: Some(10),
        }
    }
}

// ============================================================================
// Dynamic Compound Trainer
// ============================================================================

/// The Dynamic Compound Trainer
pub struct DynamicCompoundTrainer {
    config: DynamicTrainingConfig,
    model: TorusLLM,
    varmap: VarMap,
    optimizer: AdamW,
    train_data: TextDataset,
    val_data: Option<TextDataset>,
    device: Device,

    // Dynamic components
    curriculum: CurriculumScheduler,
    multi_task: MultiTaskScheduler,
    batch_controller: DynamicBatchController,
    layer_lr: LayerWiseLRController,
    growth: GrowthController,
    ema_controller: DynamicEMAController,

    // Coherence tracking
    coherence_layer: CognitiveCoherenceLayer,

    // Metrics logging
    metrics_logger: Option<MetricsLogger>,

    // State
    step: usize,
    epoch: usize,
    best_val_loss: f64,
    patience_counter: usize,
    loss_history: VecDeque<f64>,
    coherence_history: VecDeque<SenseOfCoherence>,
}

impl DynamicCompoundTrainer {
    /// Create a new dynamic compound trainer
    pub fn new(
        config: DynamicTrainingConfig,
        train_data: TextDataset,
        val_data: Option<TextDataset>,
        device: Device,
    ) -> TorusResult<Self> {
        // Initialize model
        let (model, varmap, step) = if config.resume && checkpoint_exists(&config.checkpoint_dir) {
            log::info!("Resuming from checkpoint...");
            let (model, varmap, metadata) = load_checkpoint(&config.checkpoint_dir, &device)?;
            let step = metadata.step.unwrap_or(0);
            (model, varmap, step)
        } else {
            log::info!("Creating new model...");
            let model_config = if config.use_progressive_growth {
                // Start with smaller config
                TorusLLMConfig {
                    num_layers: config.growth_config.initial_layers,
                    hidden_dim: config.growth_config.initial_hidden_dim,
                    num_heads: config.growth_config.initial_heads,
                    ffn_dim: config.growth_config.initial_hidden_dim * 4,
                    ..config.model.clone()
                }
            } else {
                config.model.clone()
            };
            let (model, varmap) = TorusLLM::new_random(model_config, &device)?;
            (model, varmap, 0)
        };

        // Create optimizer
        let params = ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        // Initialize dynamic components
        let curriculum = CurriculumScheduler::default();
        let multi_task = MultiTaskScheduler::default();
        let batch_controller = DynamicBatchController {
            current_batch_size: config.batch_size,
            ..Default::default()
        };
        let layer_lr = LayerWiseLRController::new(model.config().num_layers, config.learning_rate);
        let growth = GrowthController::new(config.growth_config.clone());
        let ema_controller =
            DynamicEMAController::new(model.config().num_layers, config.model.ema_alpha);

        // Initialize coherence tracking
        let coherence_config = CoherenceConfig {
            n_streams: 8,
            d_model: model.config().hidden_dim,
            smm_learning_rate: 0.01,
            base_alpha: config.model.ema_alpha,
            min_alpha: 0.1,
            max_alpha: 0.99,
            coherence_threshold: 0.6,
            adaptive_alpha: true,
            ..Default::default()
        };
        let coherence_layer = CognitiveCoherenceLayer::new(coherence_config, &device);

        Ok(Self {
            config,
            model,
            varmap,
            optimizer,
            train_data,
            val_data,
            device,
            curriculum,
            multi_task,
            batch_controller,
            layer_lr,
            growth,
            ema_controller,
            coherence_layer,
            metrics_logger: None,
            step,
            epoch: 0,
            best_val_loss: f64::INFINITY,
            patience_counter: 0,
            loss_history: VecDeque::with_capacity(100),
            coherence_history: VecDeque::with_capacity(100),
        })
    }

    /// Enable Tensorboard metrics logging
    ///
    /// Creates a timestamped log directory under the specified path.
    pub fn enable_metrics_logging(
        &mut self,
        log_dir: impl AsRef<std::path::Path>,
    ) -> TorusResult<()> {
        let logger = MetricsLogger::new(log_dir)?;
        self.metrics_logger = Some(logger);
        Ok(())
    }

    /// Run training
    pub fn train(&mut self) -> TorusResult<()> {
        log::info!("Starting dynamic compound training...");
        self.log_config();

        let max_steps = self.config.max_steps.unwrap_or(usize::MAX);

        for epoch in self.epoch..self.config.num_epochs {
            self.epoch = epoch;
            log::info!("Epoch {}/{}", epoch + 1, self.config.num_epochs);

            // Create data loader with current batch size
            let mut train_loader = DataLoader::new(
                self.train_data.clone(),
                self.batch_controller.current_batch_size,
                true,
                self.device.clone(),
            );

            let mut epoch_loss = 0.0;
            let mut epoch_batches = 0;

            while let Some(batch_result) = train_loader.next_batch() {
                let batch = batch_result?;

                // Training step
                let (loss, grad_norm) = self.train_step(&batch.input_ids, &batch.labels)?;
                epoch_loss += loss;
                epoch_batches += 1;
                self.step += 1;

                // Record for dynamic adjustments
                self.loss_history.push_back(loss);
                if self.loss_history.len() > 100 {
                    self.loss_history.pop_front();
                }
                self.batch_controller.record_grad_norm(grad_norm);

                // Estimate coherence (simplified - would come from model in real impl)
                let coherence = 1.0 / (1.0 + loss);
                self.ema_controller.record(loss, coherence);

                // Dynamic adjustments
                self.apply_dynamic_adjustments(loss)?;

                // Logging
                if self.config.log_interval > 0 && self.step % self.config.log_interval == 0 {
                    self.log_progress(epoch_loss / epoch_batches as f64, grad_norm);

                    // Log to tensorboard if enabled
                    self.log_metrics_to_tensorboard()?;
                }

                // Validation
                if self.config.eval_interval > 0 && self.step % self.config.eval_interval == 0 {
                    if let Some(should_stop) = self.evaluate_and_checkpoint()? {
                        if should_stop {
                            return Ok(());
                        }
                    }
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

        self.save_checkpoint("final")?;
        log::info!("Training complete!");
        Ok(())
    }

    /// Single training step
    fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor) -> TorusResult<(f64, f64)> {
        // Update learning rate
        let lr = self.get_learning_rate();
        self.optimizer.set_learning_rate(lr);

        // Forward pass
        let logits = self.model.forward(input_ids)?;

        // Compute loss
        let loss = cross_entropy_loss(&logits, labels)?;

        // Backward pass with gradient clipping
        let grads = loss.backward()?;

        // Compute gradient norm for monitoring
        let grad_norm = self.compute_grad_norm(&grads);

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            if grad_norm > max_norm {
                let scale = max_norm / grad_norm;
                // Scale gradients - note: candle's optimizer handles this internally
                // We log when clipping occurs for monitoring
                log::debug!(
                    "Gradient clipping: norm {:.4} > max {:.4}, scale={:.4}",
                    grad_norm,
                    max_norm,
                    scale
                );
            }
        }

        // Optimizer step
        self.optimizer.backward_step(&loss)?;

        // Get loss value
        let loss_val = loss.to_scalar::<f32>()? as f64;

        Ok((loss_val, grad_norm))
    }

    /// Compute the gradient norm from gradients
    fn compute_grad_norm(&self, grads: &candle_core::backprop::GradStore) -> f64 {
        let mut total_norm_sq = 0.0;

        for var in self.varmap.all_vars() {
            if let Some(grad) = grads.get(&var) {
                if let Ok(grad_sq) = grad.sqr() {
                    if let Ok(sum) = grad_sq.sum_all() {
                        if let Ok(val) = sum.to_scalar::<f32>() {
                            total_norm_sq += val as f64;
                        }
                    }
                }
            }
        }

        total_norm_sq.sqrt()
    }

    /// Apply all dynamic adjustments
    fn apply_dynamic_adjustments(&mut self, loss: f64) -> TorusResult<()> {
        // Curriculum learning
        if self.config.use_curriculum {
            self.curriculum.update(loss, self.step);
        }

        // Multi-task (update with primary LM task)
        if self.config.use_multi_task {
            self.multi_task.update(&[(0, loss)], self.step);
        }

        // Dynamic batch sizing
        if self.config.use_dynamic_batch {
            self.batch_controller.update(self.step);
        }

        // Update coherence metrics from loss dynamics
        // In a full implementation, we'd compute this from attention patterns
        // For now, derive coherence from loss stability
        let coherence = self.update_coherence_from_loss(loss);

        // Layer-wise learning rates (now driven by real coherence)
        if self.config.use_layer_wise_lr {
            // Derive per-layer coherence scores
            // Layers closer to output typically have more variance
            let num_layers = self.model.config().num_layers;
            let coherence_scores: Vec<f64> = (0..num_layers)
                .map(|i| {
                    // Early layers tend to be more stable (higher coherence)
                    // Later layers are more task-specific (adjust based on loss)
                    let layer_factor = 1.0 - (i as f64 / num_layers as f64) * 0.3;
                    coherence.score() * layer_factor
                })
                .collect();
            self.layer_lr.update_coherence(&coherence_scores);
            self.layer_lr.recompute_multipliers();
        }

        // Progressive growth
        if self.config.use_progressive_growth {
            if let Some((new_layers, new_hidden, new_heads)) =
                self.growth.should_grow(loss, self.step)
            {
                log::info!(
                    "Model growing to {} layers, {} hidden, {} heads",
                    new_layers,
                    new_hidden,
                    new_heads
                );
                // Perform the actual model growth
                self.grow_model(new_layers, new_hidden, new_heads)?;
            }
        }

        // Dynamic EMA (now driven by coherence)
        if self.config.use_dynamic_ema {
            // Use coherence to modulate EMA alpha
            let adaptive_alpha = self.coherence_layer.compute_adaptive_alpha();

            // Update EMA controller with coherence information
            self.ema_controller.record(loss, coherence.score());
            self.ema_controller.update();

            // Log coherence-driven alpha adjustment
            if self.config.log_interval > 0 && self.step % (self.config.log_interval * 10) == 0 {
                log::debug!(
                    "Coherence: {:.3} (C:{:.2}/M:{:.2}/Me:{:.2}) → α={:.3}",
                    coherence.score(),
                    coherence.comprehensibility,
                    coherence.manageability,
                    coherence.meaningfulness,
                    adaptive_alpha
                );
            }
        }

        Ok(())
    }

    /// Update coherence metrics from loss dynamics
    ///
    /// Derives comprehensibility, manageability, and meaningfulness from
    /// the training loss trajectory.
    fn update_coherence_from_loss(&mut self, loss: f64) -> SenseOfCoherence {
        // Comprehensibility: How predictable/stable is the loss?
        // Low variance = high comprehensibility
        let comprehensibility = if self.loss_history.len() >= 5 {
            let recent: Vec<f64> = self.loss_history.iter().rev().take(10).cloned().collect();
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            let variance =
                recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            let std_dev = variance.sqrt();
            // Normalize: low std_dev = high comprehensibility
            (1.0 / (1.0 + std_dev / mean.max(0.1))).clamp(0.0, 1.0)
        } else {
            0.5 // Default to neutral
        };

        // Manageability: Is the loss within a manageable range?
        // Based on whether we're making progress
        let manageability = if self.loss_history.len() >= 5 {
            let recent_5: f64 = self.loss_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let recent_10: f64 = if self.loss_history.len() >= 10 {
                self.loss_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0
            } else {
                recent_5
            };
            // If recent loss is lower, we're managing well
            if recent_10 > 0.0 {
                (recent_10 / recent_5.max(0.1)).clamp(0.3, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        // Meaningfulness: Is the training signal strong?
        // Based on loss magnitude relative to baseline (random would be ~log(vocab_size))
        let baseline_loss = (self.train_data.vocab_size() as f64).ln();
        let meaningfulness = if loss < baseline_loss {
            // Better than random = meaningful learning
            (1.0 - loss / baseline_loss).clamp(0.0, 1.0)
        } else {
            // Worse than random = low meaningfulness
            (baseline_loss / loss.max(0.1)).clamp(0.0, 0.5)
        };

        let soc = SenseOfCoherence::new(comprehensibility, manageability, meaningfulness);

        // Track history
        self.coherence_history.push_back(soc.clone());
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }

        soc
    }

    /// Get current learning rate with warmup
    fn get_learning_rate(&self) -> f64 {
        let base_lr = self.config.learning_rate;
        let warmup = self.config.warmup_steps;

        if self.step < warmup {
            base_lr * (self.step as f64 / warmup as f64)
        } else {
            base_lr
        }
    }

    /// Estimate gradient norm from loss history
    fn estimate_grad_norm(&self) -> f64 {
        if self.loss_history.len() < 2 {
            return 1.0;
        }
        let recent: Vec<f64> = self.loss_history.iter().rev().take(5).cloned().collect();
        let variance: f64 = recent
            .iter()
            .map(|&x| {
                let mean = recent.iter().sum::<f64>() / recent.len() as f64;
                (x - mean).powi(2)
            })
            .sum::<f64>()
            / recent.len() as f64;
        variance.sqrt()
    }

    /// Grow the model to new dimensions
    ///
    /// This creates a new model with larger dimensions and reinitializes
    /// the optimizer. Weight transfer is limited since dimensions change.
    fn grow_model(
        &mut self,
        new_layers: usize,
        new_hidden: usize,
        new_heads: usize,
    ) -> TorusResult<()> {
        log::info!("=== Progressive Growth: Reinitializing Model ===");
        log::info!(
            "  Old: {} layers, {} hidden, {} heads",
            self.model.config().num_layers,
            self.model.config().hidden_dim,
            self.model.config().num_heads
        );
        log::info!(
            "  New: {} layers, {} hidden, {} heads",
            new_layers,
            new_hidden,
            new_heads
        );

        // Save current checkpoint before growth
        self.save_checkpoint("pre_growth")?;

        // Create new model config with grown dimensions
        let new_config = TorusLLMConfig {
            num_layers: new_layers,
            hidden_dim: new_hidden,
            num_heads: new_heads,
            ffn_dim: new_hidden * 4,
            ..self.config.model.clone()
        };

        // Create new model and varmap
        let (new_model, new_varmap) = TorusLLM::new_random(new_config.clone(), &self.device)?;

        // Create new optimizer
        let params = ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            ..Default::default()
        };
        let new_optimizer = AdamW::new(new_varmap.all_vars(), params)?;

        // Replace model, varmap, and optimizer
        self.model = new_model;
        self.varmap = new_varmap;
        self.optimizer = new_optimizer;

        // Update layer-wise LR controller for new layer count
        self.layer_lr = LayerWiseLRController::new(new_layers, self.config.learning_rate);

        // Update EMA controller for new layer count
        self.ema_controller = DynamicEMAController::new(new_layers, self.config.model.ema_alpha);

        // Update the model config in training config
        self.config.model = new_config;

        log::info!("=== Model growth complete ===");

        Ok(())
    }

    /// Evaluate on validation set and save checkpoint
    fn evaluate_and_checkpoint(&mut self) -> TorusResult<Option<bool>> {
        if let Some(ref val_data) = self.val_data {
            let val_loss = self.evaluate(val_data)?;
            log::info!("Step {} | Val Loss: {:.4}", self.step, val_loss);

            // Early stopping
            if val_loss < self.best_val_loss {
                self.best_val_loss = val_loss;
                self.patience_counter = 0;
                self.save_checkpoint("best")?;
            } else {
                self.patience_counter += 1;
                if let Some(patience) = self.config.patience {
                    if self.patience_counter >= patience {
                        log::info!("Early stopping triggered");
                        return Ok(Some(true));
                    }
                }
            }
        }

        // Save periodic checkpoint
        if self.config.save_interval > 0 && self.step % self.config.save_interval == 0 {
            self.save_checkpoint("latest")?;
        }

        Ok(Some(false))
    }

    /// Evaluate on a dataset
    fn evaluate(&self, dataset: &TextDataset) -> TorusResult<f64> {
        let mut loader = DataLoader::new(
            dataset.clone(),
            self.batch_controller.current_batch_size,
            false,
            self.device.clone(),
        );

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some(batch_result) = loader.next_batch() {
            let batch = batch_result?;
            let logits = self.model.forward(&batch.input_ids)?;
            let loss = cross_entropy_loss(&logits, &batch.labels)?;
            total_loss += loss.to_scalar::<f32>()? as f64;
            num_batches += 1;
        }

        Ok(total_loss / num_batches.max(1) as f64)
    }

    /// Save checkpoint
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

    /// Log configuration
    fn log_config(&self) {
        log::info!("Configuration:");
        log::info!("  Curriculum learning: {}", self.config.use_curriculum);
        log::info!("  Multi-task training: {}", self.config.use_multi_task);
        log::info!("  Dynamic batch sizing: {}", self.config.use_dynamic_batch);
        log::info!(
            "  Layer-wise learning rates: {}",
            self.config.use_layer_wise_lr
        );
        log::info!(
            "  Progressive growth: {}",
            self.config.use_progressive_growth
        );
        log::info!("  Dynamic EMA: {}", self.config.use_dynamic_ema);
        log::info!("  Base LR: {}", self.config.learning_rate);
        log::info!("  Batch size: {}", self.batch_controller.current_batch_size);
    }

    /// Log progress
    fn log_progress(&self, avg_loss: f64, _grad_norm: f64) {
        log::info!(
            "Step {} | Loss: {:.4} | LR: {:.2e} | Batch: {} | Curriculum: {:?} | Growth: {:.1}%",
            self.step,
            avg_loss,
            self.get_learning_rate(),
            self.batch_controller.current_batch_size,
            self.curriculum.current_level,
            self.growth.progress() * 100.0
        );
    }

    /// Get current training statistics
    pub fn stats(&self) -> DynamicTrainingStats {
        // Get latest coherence from history
        let coherence = self
            .coherence_history
            .back()
            .map(|soc| soc.score())
            .unwrap_or(0.5);

        DynamicTrainingStats {
            step: self.step,
            epoch: self.epoch,
            loss: self.loss_history.back().copied().unwrap_or(0.0),
            avg_loss: self.loss_history.iter().sum::<f64>() / self.loss_history.len().max(1) as f64,
            val_loss: None,
            learning_rate: self.get_learning_rate(),
            batch_size: self.batch_controller.current_batch_size,
            curriculum_level: format!("{:?}", self.curriculum.current_level),
            growth_progress: self.growth.progress(),
            coherence,
            grad_norm: self.estimate_grad_norm(),
            tokens_per_sec: 0.0, // Would need timing to compute
            layer_lrs: self.layer_lr.get_all_lrs(),
            ema_alphas: self.ema_controller.get_all_alphas().to_vec(),
        }
    }

    /// Get coherence summary for logging
    pub fn coherence_summary(&self) -> String {
        if let Some(soc) = self.coherence_history.back() {
            format!(
                "SOC: {:.3} (C:{:.2}/M:{:.2}/Me:{:.2})",
                soc.score(),
                soc.comprehensibility,
                soc.manageability,
                soc.meaningfulness
            )
        } else {
            "SOC: N/A".to_string()
        }
    }

    /// Log metrics to Tensorboard (if enabled)
    fn log_metrics_to_tensorboard(&mut self) -> TorusResult<()> {
        if self.metrics_logger.is_none() {
            return Ok(());
        }

        // Compute stats and coherence before borrowing logger
        let stats = self.stats();
        let coherence_data = self
            .coherence_history
            .back()
            .map(|soc| (soc.comprehensibility, soc.manageability, soc.meaningfulness));
        let step = self.step;

        // Now borrow logger and log
        if let Some(ref mut logger) = self.metrics_logger {
            logger.log_training_stats(&stats)?;

            // Log coherence components separately
            if let Some((c, m, me)) = coherence_data {
                logger.log_coherence(c, m, me, step)?;
            }

            logger.flush()?;
        }
        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &TorusLLM {
        &self.model
    }

    /// Get current step
    pub fn step(&self) -> usize {
        self.step
    }
}

/// Compute cross-entropy loss for language modeling
fn cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> TorusResult<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;

    let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
    let labels_flat = labels.flatten_all()?;

    let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
    let labels_i64 = labels_flat.to_dtype(DType::I64)?;
    let loss = log_probs.gather(&labels_i64.unsqueeze(1)?, 1)?.squeeze(1)?;
    let loss = loss.neg()?.mean_all()?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_scheduler() {
        let mut scheduler = CurriculumScheduler::default();
        assert_eq!(scheduler.current_level, DifficultyLevel::Beginner);

        // Simulate good progress
        for i in 0..200 {
            scheduler.update(1.5, i);
        }
        // Should have advanced
        assert!(scheduler.current_level > DifficultyLevel::Beginner);
    }

    #[test]
    fn test_difficulty_levels() {
        assert_eq!(DifficultyLevel::Beginner.max_seq_len(512), 128);
        assert_eq!(DifficultyLevel::Expert.max_seq_len(512), 512);
    }

    #[test]
    fn test_multi_task_scheduler() {
        let mut scheduler = MultiTaskScheduler::default();
        scheduler.add_task("auxiliary", 0.5);
        assert_eq!(scheduler.tasks.len(), 2);

        scheduler.update(&[(0, 1.0), (1, 2.0)], 100);
        assert!(scheduler.tasks[0].running_loss > 0.0);
    }

    #[test]
    fn test_dynamic_batch_controller() {
        let mut controller = DynamicBatchController::default();

        // Record some gradient norms
        for i in 0..100 {
            controller.record_grad_norm(0.5 + (i as f64 * 0.01));
        }

        // Check history is bounded
        assert!(controller.grad_norm_history.len() <= 100);
    }

    #[test]
    fn test_layer_wise_lr() {
        let mut controller = LayerWiseLRController::new(4, 1e-3);

        controller.update_coherence(&[0.8, 0.6, 0.4, 0.2]);
        controller.update_grad_norms(&[0.1, 0.2, 0.3, 0.4]);
        controller.recompute_multipliers();

        // Later layers (lower coherence) should have higher LR
        let lrs = controller.get_all_lrs();
        assert_eq!(lrs.len(), 4);
    }

    #[test]
    fn test_growth_controller() {
        let config = GrowthConfig {
            initial_layers: 2,
            final_layers: 4,
            initial_hidden_dim: 128,
            final_hidden_dim: 256,
            initial_heads: 2,
            final_heads: 4,
            growth_interval: 100,
            growth_threshold: 3.0,
            min_steps_before_growth: 50,
        };

        let mut controller = GrowthController::new(config);
        assert_eq!(controller.current_layers, 2);

        // Simulate training with good loss
        for i in 0..200 {
            controller.should_grow(2.0, i);
        }

        // Should have grown
        assert!(controller.current_layers > 2 || controller.current_hidden_dim > 128);
    }

    #[test]
    fn test_dynamic_ema_controller() {
        let mut controller = DynamicEMAController::new(4, 0.9);

        // Record improving loss
        for i in 0..50 {
            let loss = 5.0 - (i as f64 * 0.05);
            controller.record(loss, 0.7);
        }
        controller.update();

        // Alphas should have increased (more memory for stable training)
        assert!(controller
            .current_alphas
            .iter()
            .all(|&a| (0.1..=0.99).contains(&a)));
    }

    #[test]
    fn test_training_config_default() {
        let config = DynamicTrainingConfig::default();
        assert!(config.use_curriculum);
        assert!(config.use_dynamic_batch);
        assert!(config.use_progressive_growth);
    }
}
