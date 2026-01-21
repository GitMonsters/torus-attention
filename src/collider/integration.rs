//! Compound Integration of TorusCollider into Attention Layers
//!
//! This module provides seamless integration of the CERN Hadron Collider-inspired
//! validation system into the actual attention computation layers for real-time
//! validation during both training and inference.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    VALIDATED ATTENTION PIPELINE                             │
//! │                                                                             │
//! │  Input Tensor                                                               │
//! │       │                                                                     │
//! │       ▼                                                                     │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │  ValidatedParallelProcessor                                         │   │
//! │  │  ┌─────────────────────────────────────────────────────────────┐   │   │
//! │  │  │  8× ValidatedStream                                         │   │   │
//! │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │   │   │
//! │  │  │  │ Q,K,V   │→ │ Q·K     │→ │ Attn@V  │                     │   │   │
//! │  │  │  │ Project │  │ Scores  │  │ Output  │                     │   │   │
//! │  │  │  └────┬────┘  └────┬────┘  └────┬────┘                     │   │   │
//! │  │  │       │            │            │                          │   │   │
//! │  │  │       ▼            ▼            ▼                          │   │   │
//! │  │  │  ┌─────────────────────────────────────────────────────┐  │   │   │
//! │  │  │  │           ColliderHooks (real-time)                 │  │   │   │
//! │  │  │  │  • record_qkv()      • record_attention()           │  │   │   │
//! │  │  │  │  • record_collision() • validate_causality()        │  │   │   │
//! │  │  │  └─────────────────────────────────────────────────────┘  │   │   │
//! │  │  └─────────────────────────────────────────────────────────────┘   │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                     │
//! │       ▼                                                                     │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │  TorusCollider (always-on validation)                               │   │
//! │  │  • Particle Generation    • Conservation Laws                       │   │
//! │  │  • Anomaly Detection      • Causality Validation                    │   │
//! │  │  • Metrics Collection     • Real-time Reporting                     │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │       │                                                                     │
//! │       ▼                                                                     │
//! │  Output Tensor + ValidationReport                                          │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use rustyworm::collider::integration::{
//!     ValidatedTransformer, ColliderIntegration, ValidationConfig,
//! };
//!
//! // Wrap existing transformer with validation
//! let config = ValidationConfig::default();
//! let mut validated = ValidatedTransformer::wrap(transformer, config);
//!
//! // Forward pass with real-time validation
//! let (output, report) = validated.forward_validated(&input)?;
//!
//! // Check health
//! if !report.is_healthy {
//!     println!("Anomalies detected: {}", report.summary());
//! }
//! ```

use crate::collider::{ColliderConfig, ColliderReport, TorusCollider};
use crate::parallel_streams::{ParallelStreamProcessor, StreamId};
use crate::TorusResult;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for real-time validation during attention computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable validation (can be disabled for pure inference speed)
    pub enabled: bool,

    /// Validation level (affects performance vs detail trade-off)
    pub level: ValidationLevel,

    /// Record Q, K, V tensors as particles
    pub record_qkv: bool,

    /// Record attention weights
    pub record_attention: bool,

    /// Record collision events at Q·K computation
    pub record_collisions: bool,

    /// Validate conservation laws
    pub validate_conservation: bool,

    /// Validate causality (speed of darkness for backward streams)
    pub validate_causality: bool,

    /// Detect numerical anomalies (NaN, Inf, gradient issues)
    pub detect_anomalies: bool,

    /// Collect detailed metrics
    pub collect_metrics: bool,

    /// Sample rate for validation (1.0 = every step, 0.1 = 10% of steps)
    pub sample_rate: f64,

    /// Alert threshold for anomalies before pausing
    pub alert_threshold: usize,

    /// Enable real-time reporting to stdout
    pub realtime_reporting: bool,

    /// Reporting interval (in steps)
    pub report_interval: usize,

    /// Store history for post-training analysis
    pub store_history: bool,

    /// Maximum history size
    pub max_history_size: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: ValidationLevel::Standard,
            record_qkv: true,
            record_attention: true,
            record_collisions: true,
            validate_conservation: true,
            validate_causality: true,
            detect_anomalies: true,
            collect_metrics: true,
            sample_rate: 1.0,
            alert_threshold: 100,
            realtime_reporting: false,
            report_interval: 100,
            store_history: true,
            max_history_size: 10000,
        }
    }
}

impl ValidationConfig {
    /// Full validation for debugging
    pub fn full() -> Self {
        Self {
            enabled: true,
            level: ValidationLevel::Full,
            record_qkv: true,
            record_attention: true,
            record_collisions: true,
            validate_conservation: true,
            validate_causality: true,
            detect_anomalies: true,
            collect_metrics: true,
            sample_rate: 1.0,
            alert_threshold: 10,
            realtime_reporting: true,
            report_interval: 10,
            store_history: true,
            max_history_size: 100000,
        }
    }

    /// Minimal validation for production
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            level: ValidationLevel::Minimal,
            record_qkv: false,
            record_attention: false,
            record_collisions: false,
            validate_conservation: false,
            validate_causality: false,
            detect_anomalies: true, // Always detect NaN/Inf
            collect_metrics: false,
            sample_rate: 0.01, // 1% of steps
            alert_threshold: 1000,
            realtime_reporting: false,
            report_interval: 1000,
            store_history: false,
            max_history_size: 1000,
        }
    }

    /// Disabled for maximum speed
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            level: ValidationLevel::Minimal,
            record_qkv: false,
            record_attention: false,
            record_collisions: false,
            validate_conservation: false,
            validate_causality: false,
            detect_anomalies: false,
            collect_metrics: false,
            sample_rate: 0.0,
            alert_threshold: usize::MAX,
            realtime_reporting: false,
            report_interval: usize::MAX,
            store_history: false,
            max_history_size: 0,
        }
    }

    /// Training mode with anomaly focus
    pub fn training() -> Self {
        Self {
            enabled: true,
            level: ValidationLevel::Standard,
            record_qkv: true,
            record_attention: true,
            record_collisions: true,
            validate_conservation: true,
            validate_causality: true,
            detect_anomalies: true,
            collect_metrics: true,
            sample_rate: 0.1, // 10% of steps for efficiency
            alert_threshold: 50,
            realtime_reporting: false,
            report_interval: 100,
            store_history: true,
            max_history_size: 50000,
        }
    }
}

/// Validation detail level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Minimal: Only critical anomaly detection
    Minimal,
    /// Standard: Anomalies + basic conservation
    Standard,
    /// Full: Complete particle physics simulation
    Full,
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER HOOKS TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for injecting collider validation into attention modules
pub trait ColliderHooks {
    /// Called after Q, K, V projection
    fn on_qkv_computed(
        &self,
        collider: &mut TorusCollider,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        stream_id: usize,
        layer: usize,
    ) -> TorusResult<()>;

    /// Called after attention scores computed (Q·K)
    fn on_scores_computed(
        &self,
        collider: &mut TorusCollider,
        scores: &Tensor,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<()>;

    /// Called after softmax attention weights
    fn on_attention_computed(
        &self,
        collider: &mut TorusCollider,
        attention: &Tensor,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<()>;

    /// Called after attention output (attention @ V)
    fn on_output_computed(
        &self,
        collider: &mut TorusCollider,
        output: &Tensor,
        stream_id: usize,
        layer: usize,
    ) -> TorusResult<()>;

    /// Called during backward pass with gradients
    fn on_gradients_computed(
        &self,
        collider: &mut TorusCollider,
        gradients: &HashMap<String, Tensor>,
        layer: usize,
    ) -> TorusResult<()>;
}

/// Default implementation of ColliderHooks
#[derive(Debug, Clone)]
pub struct DefaultColliderHooks {
    config: ValidationConfig,
    #[allow(dead_code)]
    n_major: usize,
    #[allow(dead_code)]
    n_minor: usize,
}

impl DefaultColliderHooks {
    pub fn new(config: ValidationConfig, n_major: usize, n_minor: usize) -> Self {
        Self {
            config,
            n_major,
            n_minor,
        }
    }
}

impl ColliderHooks for DefaultColliderHooks {
    fn on_qkv_computed(
        &self,
        collider: &mut TorusCollider,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        stream_id: usize,
        layer: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled || !self.config.record_qkv {
            return Ok(());
        }

        collider.record_qkv(q, k, v, stream_id)?;
        collider.set_layer(layer);

        // Check for NaN/Inf in QKV
        if self.config.detect_anomalies {
            collider
                .anomaly
                .check_tensor(q, "query", layer, Some(stream_id))?;
            collider
                .anomaly
                .check_tensor(k, "key", layer, Some(stream_id))?;
            collider
                .anomaly
                .check_tensor(v, "value", layer, Some(stream_id))?;
        }

        Ok(())
    }

    fn on_scores_computed(
        &self,
        collider: &mut TorusCollider,
        scores: &Tensor,
        stream_id: usize,
        layer: usize,
        _head: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check for numerical issues in attention scores
        if self.config.detect_anomalies {
            collider
                .anomaly
                .check_tensor(scores, "attention_scores", layer, Some(stream_id))?;
        }

        Ok(())
    }

    fn on_attention_computed(
        &self,
        collider: &mut TorusCollider,
        attention: &Tensor,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled || !self.config.record_attention {
            return Ok(());
        }

        collider.record_attention(attention, stream_id, head)?;

        // Check attention weight validity
        if self.config.detect_anomalies {
            collider.anomaly.check_tensor(
                attention,
                "attention_weights",
                layer,
                Some(stream_id),
            )?;

            // Check for attention collapse (all weights on one token)
            collider
                .anomaly
                .check_attention_collapse(attention, layer, head)?;
        }

        Ok(())
    }

    fn on_output_computed(
        &self,
        collider: &mut TorusCollider,
        output: &Tensor,
        stream_id: usize,
        layer: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check output tensor
        if self.config.detect_anomalies {
            collider
                .anomaly
                .check_tensor(output, "attention_output", layer, Some(stream_id))?;
        }

        // Validate stream causality for backward streams
        if self.config.validate_causality && stream_id % 2 == 1 {
            // Odd stream IDs are backward
            collider.validate_streams();
        }

        Ok(())
    }

    fn on_gradients_computed(
        &self,
        collider: &mut TorusCollider,
        gradients: &HashMap<String, Tensor>,
        layer: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check for gradient anomalies
        if self.config.detect_anomalies {
            for (name, grad) in gradients {
                collider
                    .anomaly
                    .check_tensor(grad, &format!("gradient_{}", name), layer, None)?;
            }

            // Check for exploding/vanishing gradients
            collider.anomaly.check_gradient_health(gradients, layer)?;
        }

        // Record gradients as backward-flowing particles (tachyons)
        collider.record_gradients_map(gradients, layer)?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Thread-safe validation context shared across layers
#[derive(Clone)]
pub struct ValidationContext {
    /// The collider instance (wrapped in RwLock for thread-safety)
    collider: Arc<RwLock<TorusCollider>>,
    /// Configuration
    config: ValidationConfig,
    /// Hooks implementation
    hooks: Arc<DefaultColliderHooks>,
    /// Current step counter
    step: Arc<RwLock<u64>>,
    /// Validation history
    history: Arc<RwLock<ValidationHistory>>,
    /// Start time for timing
    start_time: Instant,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new(config: ValidationConfig, n_major: usize, n_minor: usize) -> Self {
        let collider_config = ColliderConfig {
            enabled: config.enabled,
            detector_enabled: config.level == ValidationLevel::Full,
            darkness_grid: (64, 64),
            anomaly_thresholds: Default::default(),
            metrics_buffer_size: config.max_history_size,
            conservation_tolerance: 1e-5,
            causality_tolerance: 1e-6,
            n_major,
            n_minor,
        };

        Self {
            collider: Arc::new(RwLock::new(TorusCollider::new(collider_config))),
            config: config.clone(),
            hooks: Arc::new(DefaultColliderHooks::new(config.clone(), n_major, n_minor)),
            step: Arc::new(RwLock::new(0)),
            history: Arc::new(RwLock::new(ValidationHistory::new(config.max_history_size))),
            start_time: Instant::now(),
        }
    }

    /// Check if validation should run this step (based on sample rate)
    pub fn should_validate(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        let step = *self.step.read().unwrap();
        let sample_interval = (1.0 / self.config.sample_rate) as u64;
        step % sample_interval == 0
    }

    /// Get current step
    pub fn current_step(&self) -> u64 {
        *self.step.read().unwrap()
    }

    /// Advance to next step
    pub fn next_step(&self) {
        let mut step = self.step.write().unwrap();
        *step += 1;

        if self.config.enabled {
            let mut collider = self.collider.write().unwrap();
            collider.next_step();
        }
    }

    /// Get the collider for direct access
    pub fn collider(&self) -> &Arc<RwLock<TorusCollider>> {
        &self.collider
    }

    /// Execute with collider (thread-safe)
    pub fn with_collider<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut TorusCollider) -> R,
    {
        let mut collider = self.collider.write().unwrap();
        f(&mut collider)
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        if !self.config.enabled {
            return true;
        }
        let collider = self.collider.read().unwrap();
        collider.is_healthy()
    }

    /// Get current report
    pub fn report(&self) -> ColliderReport {
        let collider = self.collider.read().unwrap();
        collider.report()
    }

    /// Get validation summary
    pub fn summary(&self) -> String {
        let collider = self.collider.read().unwrap();
        let elapsed = self.start_time.elapsed();
        let step = *self.step.read().unwrap();

        format!(
            "═══════════════════════════════════════════════════════════════\n\
             VALIDATION CONTEXT - Step {} ({:.2}s elapsed)\n\
             ═══════════════════════════════════════════════════════════════\n\
             {}\n\
             ═══════════════════════════════════════════════════════════════",
            step,
            elapsed.as_secs_f64(),
            collider.summary()
        )
    }

    /// Record to history if enabled
    pub fn record_to_history(&self) {
        if !self.config.store_history {
            return;
        }

        let report = self.report();
        let step = self.current_step();

        let mut history = self.history.write().unwrap();
        history.record(step, report);
    }

    /// Get history
    pub fn history(&self) -> ValidationHistory {
        self.history.read().unwrap().clone()
    }

    /// Reset the context
    pub fn reset(&self) {
        let mut collider = self.collider.write().unwrap();
        collider.reset();

        let mut step = self.step.write().unwrap();
        *step = 0;

        let mut history = self.history.write().unwrap();
        history.clear();
    }
}

impl std::fmt::Debug for ValidationContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationContext")
            .field("config", &self.config)
            .field("step", &self.current_step())
            .field("is_healthy", &self.is_healthy())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

/// Historical record of validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationHistory {
    /// Maximum size
    max_size: usize,
    /// Step numbers
    steps: Vec<u64>,
    /// Health status per step
    health: Vec<bool>,
    /// Anomaly counts per step
    anomaly_counts: Vec<usize>,
    /// Conservation violation counts
    conservation_violations: Vec<usize>,
    /// Causality violation counts
    causality_violations: Vec<usize>,
    /// Collision counts
    collision_counts: Vec<u64>,
}

/// Lightweight summary of a history record
#[derive(Debug, Clone)]
pub struct HistorySummary {
    pub is_healthy: bool,
    pub anomaly_count: usize,
    pub conservation_violations: usize,
    pub causality_violations: usize,
    pub collision_count: u64,
}

impl HistorySummary {
    /// Get total anomalies as u64 (for compatibility)
    pub fn total_anomalies(&self) -> u64 {
        self.anomaly_count as u64
    }
}

impl ValidationHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            steps: Vec::with_capacity(max_size),
            health: Vec::with_capacity(max_size),
            anomaly_counts: Vec::with_capacity(max_size),
            conservation_violations: Vec::with_capacity(max_size),
            causality_violations: Vec::with_capacity(max_size),
            collision_counts: Vec::with_capacity(max_size),
        }
    }

    pub fn record(&mut self, step: u64, report: ColliderReport) {
        if self.steps.len() >= self.max_size {
            // Remove oldest entry
            self.steps.remove(0);
            self.health.remove(0);
            self.anomaly_counts.remove(0);
            self.conservation_violations.remove(0);
            self.causality_violations.remove(0);
            self.collision_counts.remove(0);
        }

        self.steps.push(step);
        self.health.push(report.is_healthy);
        self.anomaly_counts
            .push(report.anomaly_report.stats.total_anomalies as usize);
        self.conservation_violations
            .push(report.conservation_report.n_violations);
        self.causality_violations
            .push(report.causality_report.violations.len());
        self.collision_counts
            .push(report.metrics_report.collision_stats.n_collisions);
    }

    pub fn clear(&mut self) {
        self.steps.clear();
        self.health.clear();
        self.anomaly_counts.clear();
        self.conservation_violations.clear();
        self.causality_violations.clear();
        self.collision_counts.clear();
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get health rate over last N steps
    pub fn health_rate(&self, window: usize) -> f64 {
        let n = self.health.len().min(window);
        if n == 0 {
            return 1.0;
        }
        let healthy = self.health.iter().rev().take(n).filter(|&&h| h).count();
        healthy as f64 / n as f64
    }

    /// Get anomaly trend (positive = increasing)
    pub fn anomaly_trend(&self, window: usize) -> f64 {
        let n = self.anomaly_counts.len().min(window);
        if n < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.anomaly_counts.iter().rev().take(n).collect();
        let first_half: f64 =
            recent[n / 2..].iter().map(|&&x| x as f64).sum::<f64>() / (n / 2) as f64;
        let second_half: f64 =
            recent[..n / 2].iter().map(|&&x| x as f64).sum::<f64>() / (n / 2) as f64;
        second_half - first_half
    }

    /// Get recent history as (step, report_summary) pairs
    ///
    /// This is a lightweight method that returns basic info without full reports.
    pub fn recent(&self, n: usize) -> Vec<(u64, HistorySummary)> {
        let take = self.steps.len().min(n);
        let start = self.steps.len().saturating_sub(take);

        (start..self.steps.len())
            .map(|i| {
                (
                    self.steps[i],
                    HistorySummary {
                        is_healthy: self.health[i],
                        anomaly_count: self.anomaly_counts[i],
                        conservation_violations: self.conservation_violations[i],
                        causality_violations: self.causality_violations[i],
                        collision_count: self.collision_counts[i],
                    },
                )
            })
            .collect()
    }

    /// Summary statistics
    pub fn summary(&self) -> String {
        format!(
            "History: {} records, health rate: {:.1}%, anomaly trend: {:.2}",
            self.len(),
            self.health_rate(100) * 100.0,
            self.anomaly_trend(100)
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATED STREAM
// ═══════════════════════════════════════════════════════════════════════════════

/// A processing stream with integrated collision validation
pub struct ValidatedStream {
    /// Stream identifier
    stream_id: StreamId,
    /// Validation context
    context: ValidationContext,
    /// Layer index
    layer: usize,
    /// Number of heads
    #[allow(dead_code)]
    n_heads: usize,
    /// Head dimension
    #[allow(dead_code)]
    head_dim: usize,
    /// Scale factor
    scale: f64,
}

impl ValidatedStream {
    pub fn new(
        stream_id: StreamId,
        context: ValidationContext,
        layer: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            stream_id,
            context,
            layer,
            n_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Process Q, K, V with validation hooks
    pub fn process_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> TorusResult<(Tensor, Option<Tensor>)> {
        let should_validate = self.context.should_validate();

        // Hook: QKV computed
        if should_validate {
            self.context.with_collider(|collider| {
                self.context.hooks.on_qkv_computed(
                    collider,
                    q,
                    k,
                    v,
                    self.stream_id as usize,
                    self.layer,
                )
            })?;
        }

        // Compute attention scores: Q @ K^T
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores * self.scale)?;

        // Hook: Scores computed
        if should_validate {
            self.context.with_collider(|collider| {
                self.context.hooks.on_scores_computed(
                    collider,
                    &scores,
                    self.stream_id as usize,
                    self.layer,
                    0, // All heads
                )
            })?;
        }

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            let mask_broadcast = mask.broadcast_as(scores.dims())?;
            scores.where_cond(
                &mask_broadcast,
                &Tensor::new(f32::NEG_INFINITY, scores.device())?,
            )?
        } else {
            scores
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;

        // Hook: Attention weights computed
        if should_validate {
            self.context.with_collider(|collider| {
                self.context.hooks.on_attention_computed(
                    collider,
                    &attn_weights,
                    self.stream_id as usize,
                    self.layer,
                    0,
                )
            })?;
        }

        // Apply to values
        let output = attn_weights.matmul(v)?;

        // Hook: Output computed
        if should_validate {
            self.context.with_collider(|collider| {
                self.context.hooks.on_output_computed(
                    collider,
                    &output,
                    self.stream_id as usize,
                    self.layer,
                )
            })?;
        }

        // Return attention weights for inspection if needed
        let attn_for_return = if self.context.config.record_attention {
            Some(attn_weights)
        } else {
            None
        };

        Ok((output, attn_for_return))
    }

    /// Get stream ID
    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    /// Check if this is a backward (anti-causal) stream
    pub fn is_backward(&self) -> bool {
        matches!(
            self.stream_id,
            StreamId::MajorBackward
                | StreamId::MinorBackward
                | StreamId::SpiralCCW
                | StreamId::CrossVtoU
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATED LAYER OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

/// Output from a validated layer
#[derive(Debug)]
pub struct ValidatedLayerOutput {
    /// The output tensor
    pub output: Tensor,
    /// Attention weights (if recorded)
    pub attention: Option<Tensor>,
    /// Validation report for this layer
    pub validation: LayerValidationReport,
}

/// Validation report for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerValidationReport {
    /// Layer index
    pub layer: usize,
    /// Is this layer healthy?
    pub is_healthy: bool,
    /// Anomalies detected in this layer
    pub anomaly_count: usize,
    /// Conservation violations
    pub conservation_violations: usize,
    /// Causality violations (for backward streams)
    pub causality_violations: usize,
    /// Stream reports
    pub stream_reports: Vec<StreamValidationReport>,
}

impl LayerValidationReport {
    pub fn healthy(layer: usize) -> Self {
        Self {
            layer,
            is_healthy: true,
            anomaly_count: 0,
            conservation_violations: 0,
            causality_violations: 0,
            stream_reports: Vec::new(),
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Layer {}: {} (anomalies: {}, conservation: {}, causality: {})",
            self.layer,
            if self.is_healthy { "HEALTHY" } else { "ISSUES" },
            self.anomaly_count,
            self.conservation_violations,
            self.causality_violations
        )
    }
}

/// Validation report for a single stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamValidationReport {
    /// Stream ID
    pub stream_id: usize,
    /// Stream name
    pub stream_name: String,
    /// Is backward (anti-causal) stream
    pub is_backward: bool,
    /// Anomalies in this stream
    pub anomaly_count: usize,
    /// Causality status
    pub causality_ok: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATED PARALLEL PROCESSOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Parallel stream processor with integrated validation
pub struct ValidatedParallelProcessor {
    /// Inner processor
    inner: ParallelStreamProcessor,
    /// Validation context
    context: ValidationContext,
    /// Layer index
    layer: usize,
    /// Validated streams
    validated_streams: Vec<ValidatedStream>,
}

impl ValidatedParallelProcessor {
    /// Wrap an existing processor with validation
    pub fn wrap(inner: ParallelStreamProcessor, context: ValidationContext, layer: usize) -> Self {
        let n_heads = inner.config().n_heads;
        let head_dim = inner.config().d_model / n_heads;

        let validated_streams = vec![
            ValidatedStream::new(
                StreamId::MajorForward,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::MajorBackward,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::MinorForward,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::MinorBackward,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::SpiralCW,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::SpiralCCW,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::CrossUtoV,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
            ValidatedStream::new(
                StreamId::CrossVtoU,
                context.clone(),
                layer,
                n_heads,
                head_dim,
            ),
        ];

        Self {
            inner,
            context,
            layer,
            validated_streams,
        }
    }

    /// Forward pass with validation
    pub fn forward_validated(&self, x: &Tensor) -> TorusResult<ValidatedLayerOutput> {
        // Set layer in collider
        self.context.with_collider(|collider| {
            collider.set_layer(self.layer);
        });

        // Execute inner forward pass
        let output = self.inner.forward(x)?;

        // Build validation report
        let validation = if self.context.should_validate() {
            self.build_layer_report()
        } else {
            LayerValidationReport::healthy(self.layer)
        };

        Ok(ValidatedLayerOutput {
            output,
            attention: None, // Could be extracted from inner if needed
            validation,
        })
    }

    /// Build validation report for this layer
    fn build_layer_report(&self) -> LayerValidationReport {
        let collider = self.context.collider().read().unwrap();
        let report = collider.report();

        let stream_reports: Vec<_> = self
            .validated_streams
            .iter()
            .map(|s| StreamValidationReport {
                stream_id: s.stream_id as usize,
                stream_name: format!("{:?}", s.stream_id),
                is_backward: s.is_backward(),
                anomaly_count: 0, // Would need per-stream tracking
                causality_ok: report.causality_report.is_valid,
            })
            .collect();

        LayerValidationReport {
            layer: self.layer,
            is_healthy: report.is_healthy,
            anomaly_count: report.anomaly_report.stats.total_anomalies as usize,
            conservation_violations: report.conservation_report.n_violations,
            causality_violations: report.causality_report.violations.len(),
            stream_reports,
        }
    }

    /// Get the inner processor
    pub fn inner(&self) -> &ParallelStreamProcessor {
        &self.inner
    }

    /// Get validation context
    pub fn context(&self) -> &ValidationContext {
        &self.context
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER INTEGRATION TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for types that can be integrated with the collider
pub trait ColliderIntegration {
    /// The output type with validation report
    type ValidatedOutput;

    /// Forward pass with validation
    fn forward_with_validation(
        &mut self,
        input: &Tensor,
        context: &ValidationContext,
    ) -> TorusResult<Self::ValidatedOutput>;

    /// Backward pass with validation (for training)
    fn backward_with_validation(
        &mut self,
        gradients: &HashMap<String, Tensor>,
        context: &ValidationContext,
    ) -> TorusResult<()>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Training step result with validation
#[derive(Debug)]
pub struct ValidatedTrainingStep {
    /// Loss value
    pub loss: f32,
    /// Gradient norm
    pub grad_norm: f64,
    /// Validation report
    pub validation: ColliderReport,
    /// Is healthy (no critical anomalies)
    pub is_healthy: bool,
    /// Should pause training (too many anomalies)
    pub should_pause: bool,
    /// Warning messages
    pub warnings: Vec<String>,
}

impl ValidatedTrainingStep {
    pub fn summary(&self) -> String {
        let status = if self.is_healthy {
            "HEALTHY"
        } else if self.should_pause {
            "CRITICAL - PAUSE RECOMMENDED"
        } else {
            "ISSUES DETECTED"
        };

        let mut summary = format!(
            "Training Step: loss={:.4}, grad_norm={:.4}, status={}\n",
            self.loss, self.grad_norm, status
        );

        if !self.warnings.is_empty() {
            summary.push_str("Warnings:\n");
            for w in &self.warnings {
                summary.push_str(&format!("  - {}\n", w));
            }
        }

        summary
    }
}

/// Validated training loop integration
pub struct ValidatedTrainer {
    /// Validation context
    context: ValidationContext,
    /// Total anomalies since last reset
    total_anomalies: usize,
    /// Steps with issues
    steps_with_issues: usize,
    /// Consecutive healthy steps
    consecutive_healthy: usize,
}

impl ValidatedTrainer {
    pub fn new(context: ValidationContext) -> Self {
        Self {
            context,
            total_anomalies: 0,
            steps_with_issues: 0,
            consecutive_healthy: 0,
        }
    }

    /// Record a training step
    pub fn record_step(
        &mut self,
        loss: f32,
        grad_norm: f64,
        gradients: Option<&HashMap<String, Tensor>>,
        layer: usize,
    ) -> TorusResult<ValidatedTrainingStep> {
        let mut warnings = Vec::new();

        // Record gradients if provided
        if let Some(grads) = gradients {
            self.context.with_collider(|collider| {
                self.context
                    .hooks
                    .on_gradients_computed(collider, grads, layer)
            })?;
        }

        // Get validation report
        let report = self.context.report();

        // Update counters
        if report.is_healthy {
            self.consecutive_healthy += 1;
        } else {
            self.steps_with_issues += 1;
            self.consecutive_healthy = 0;
            self.total_anomalies += report.anomaly_report.stats.total_anomalies as usize;
        }

        // Check for issues
        if report.anomaly_report.stats.critical_anomalies > 0 {
            warnings.push(format!(
                "Critical anomalies detected: {}",
                report.anomaly_report.stats.critical_anomalies
            ));
        }

        if !report.conservation_report.all_conserved {
            warnings.push(format!(
                "Conservation violations: {}",
                report.conservation_report.n_violations
            ));
        }

        if !report.causality_report.is_valid {
            warnings.push(format!(
                "Causality violations: {}",
                report.causality_report.violations.len()
            ));
        }

        if grad_norm > 100.0 {
            warnings.push(format!("High gradient norm: {:.2}", grad_norm));
        }

        if loss.is_nan() || loss.is_infinite() {
            warnings.push("Loss is NaN or Infinite!".to_string());
        }

        // Determine if we should pause
        let should_pause = self.total_anomalies > self.context.config.alert_threshold
            || report.anomaly_report.stats.critical_anomalies > 10
            || loss.is_nan()
            || loss.is_infinite();

        // Advance step
        self.context.next_step();
        self.context.record_to_history();

        // Real-time reporting
        if self.context.config.realtime_reporting {
            let step = self.context.current_step();
            if step % self.context.config.report_interval as u64 == 0 {
                println!("{}", self.context.summary());
            }
        }

        Ok(ValidatedTrainingStep {
            loss,
            grad_norm,
            validation: report.clone(),
            is_healthy: report.is_healthy,
            should_pause,
            warnings,
        })
    }

    /// Get statistics
    pub fn stats(&self) -> TrainerValidationStats {
        TrainerValidationStats {
            total_steps: self.context.current_step(),
            total_anomalies: self.total_anomalies,
            steps_with_issues: self.steps_with_issues,
            consecutive_healthy: self.consecutive_healthy,
            health_rate: self.context.history().health_rate(100),
            anomaly_trend: self.context.history().anomaly_trend(100),
        }
    }

    /// Get context
    pub fn context(&self) -> &ValidationContext {
        &self.context
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_anomalies = 0;
        self.steps_with_issues = 0;
        self.consecutive_healthy = 0;
    }
}

/// Statistics from the validated trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerValidationStats {
    pub total_steps: u64,
    pub total_anomalies: usize,
    pub steps_with_issues: usize,
    pub consecutive_healthy: usize,
    pub health_rate: f64,
    pub anomaly_trend: f64,
}

impl TrainerValidationStats {
    pub fn summary(&self) -> String {
        format!(
            "Training Validation: {} steps, {} anomalies, {:.1}% healthy, trend: {:.2}",
            self.total_steps,
            self.total_anomalies,
            self.health_rate * 100.0,
            self.anomaly_trend
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL-TIME DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════

/// Real-time validation dashboard output
#[derive(Debug, Clone)]
pub struct ValidationDashboard {
    context: ValidationContext,
}

impl ValidationDashboard {
    pub fn new(context: ValidationContext) -> Self {
        Self { context }
    }

    /// Generate ASCII dashboard
    pub fn render(&self) -> String {
        let report = self.context.report();
        let history = self.context.history();
        let step = self.context.current_step();

        let health_indicator = if report.is_healthy { "[OK]" } else { "[!!]" };

        let anomaly_bar = self.render_bar(
            report.anomaly_report.stats.total_anomalies as f64,
            100.0,
            20,
        );

        let health_bar = self.render_bar(history.health_rate(100), 1.0, 20);

        format!(
            r#"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TORUS COLLIDER REAL-TIME DASHBOARD                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Step: {:>8} │ Status: {:>6} │ Health Rate: {:>5.1}%                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ANOMALIES      │ {} │ {:>5} total                          ║
║ HEALTH         │ {} │ {:.1}% over last 100                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Conservation   │ {} violations                                              ║
║ Causality      │ {} violations                                              ║
║ Collisions     │ {} recorded                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Anomaly Trend  │ {:>+.2} (negative = improving)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"#,
            step,
            health_indicator,
            history.health_rate(100) * 100.0,
            anomaly_bar,
            report.anomaly_report.stats.total_anomalies,
            health_bar,
            history.health_rate(100) * 100.0,
            report.conservation_report.n_violations,
            report.causality_report.violations.len(),
            report.metrics_report.collision_stats.n_collisions,
            history.anomaly_trend(100)
        )
    }

    fn render_bar(&self, value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).min(width as f64) as usize;
        let empty = width - filled;
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// Get context
    pub fn context(&self) -> &ValidationContext {
        &self.context
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATED TRANSFORMER (COMPOUND INTEGRATION)
// ═══════════════════════════════════════════════════════════════════════════════

use crate::integration::{BidirectionalTorusConfig, BidirectionalTorusTransformer};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

/// Complete validated transformer output with physics metrics
#[derive(Debug, Clone)]
pub struct ValidatedOutput {
    /// The output tensor from the transformer
    pub output: Tensor,
    /// Whether this step was validated
    pub was_validated: bool,
    /// The validation report (if validated)
    pub report: Option<ColliderReport>,
    /// Is the computation healthy?
    pub is_healthy: bool,
    /// Number of anomalies detected
    pub anomaly_count: usize,
    /// Number of conservation violations
    pub conservation_violations: usize,
    /// Number of causality violations
    pub causality_violations: usize,
    /// Forward pass duration in milliseconds
    pub forward_time_ms: f64,
    /// Validation overhead in milliseconds
    pub validation_time_ms: f64,
}

impl ValidatedOutput {
    /// Get just the output tensor
    pub fn tensor(&self) -> &Tensor {
        &self.output
    }

    /// Check if any critical anomalies occurred
    pub fn has_critical_anomaly(&self) -> bool {
        self.report
            .as_ref()
            .map(|r| r.anomaly_report.stats.critical_anomalies > 0)
            .unwrap_or(false)
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        if !self.was_validated {
            return format!(
                "ValidatedOutput [not validated, forward={:.2}ms]",
                self.forward_time_ms
            );
        }

        let status = if self.is_healthy { "OK" } else { "ISSUES" };
        format!(
            "ValidatedOutput [{}] anomalies={}, conservation={}, causality={}, forward={:.2}ms, validation={:.2}ms",
            status,
            self.anomaly_count,
            self.conservation_violations,
            self.causality_violations,
            self.forward_time_ms,
            self.validation_time_ms
        )
    }
}

/// Compound validated transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedTransformerConfig {
    /// Model configuration
    pub model: BidirectionalTorusConfig,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Vocabulary size (None for embedding-only mode)
    pub vocab_size: Option<usize>,
    /// Enable gradient recording for backward stream validation
    pub track_gradients: bool,
    /// Record intermediate attention patterns
    pub record_attention_patterns: bool,
    /// Alert threshold for consecutive anomalies
    pub anomaly_alert_threshold: usize,
    /// Auto-pause training on critical anomaly
    pub auto_pause_on_critical: bool,
}

impl Default for ValidatedTransformerConfig {
    fn default() -> Self {
        Self {
            model: BidirectionalTorusConfig::default(),
            validation: ValidationConfig::default(),
            vocab_size: None,
            track_gradients: true,
            record_attention_patterns: true,
            anomaly_alert_threshold: 5,
            auto_pause_on_critical: true,
        }
    }
}

impl ValidatedTransformerConfig {
    /// Create a minimal config for testing
    pub fn minimal_test() -> Self {
        Self {
            model: BidirectionalTorusConfig {
                d_model: 64,
                d_ff: 256,
                n_heads: 4,
                n_layers: 2,
                n_major: 8,
                n_minor: 4,
                ..BidirectionalTorusConfig::default()
            },
            validation: ValidationConfig::minimal(),
            vocab_size: Some(100),
            track_gradients: false,
            record_attention_patterns: false,
            anomaly_alert_threshold: 10,
            auto_pause_on_critical: false,
        }
    }

    /// Create a full validation config
    pub fn full_validation() -> Self {
        Self {
            validation: ValidationConfig::full(),
            track_gradients: true,
            record_attention_patterns: true,
            anomaly_alert_threshold: 3,
            auto_pause_on_critical: true,
            ..Self::default()
        }
    }
}

/// Validated Transformer - Compound integration of BidirectionalTorusTransformer with TorusCollider
///
/// This is the primary integration point that combines:
/// - **BidirectionalTorusTransformer**: 8-stream bidirectional attention on torus manifold
/// - **TorusCollider**: CERN-inspired physics validation
/// - **Coherence Tracking**: Self-organizing criticality for training dynamics
/// - **Real-time Monitoring**: Dashboard and alerting
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────┐
/// │                      VALIDATED TRANSFORMER                                  │
/// │                                                                             │
/// │  ┌───────────────────────────────────────────────────────────────────────┐ │
/// │  │  BidirectionalTorusTransformer                                        │ │
/// │  │  • 8 parallel streams (4 forward + 4 backward)                       │ │
/// │  │  • Torus manifold position encodings                                 │ │
/// │  │  • Compounding layer accumulation                                    │ │
/// │  │  • Coherence-based adaptive blending                                 │ │
/// │  └───────────────────────────────────────────────────────────────────────┘ │
/// │                              ▼                                             │
/// │  ┌───────────────────────────────────────────────────────────────────────┐ │
/// │  │  TorusCollider (Physics Validation)                                   │ │
/// │  │  • Particle representation of Q, K, V, Attention                     │ │
/// │  │  • Conservation law checking (energy, momentum, charge)              │ │
/// │  │  • Causality validation (speed of darkness for backward streams)     │ │
/// │  │  • Anomaly detection (NaN, Inf, gradient explosions)                 │ │
/// │  │  • Cross-section and luminosity metrics                              │ │
/// │  └───────────────────────────────────────────────────────────────────────┘ │
/// │                              ▼                                             │
/// │  ValidatedOutput { tensor, report, metrics }                               │
/// └─────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use rustyworm::collider::integration::{ValidatedTransformer, ValidatedTransformerConfig};
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let config = ValidatedTransformerConfig::default();
/// let mut transformer = ValidatedTransformer::new(config, &device)?;
///
/// // Forward pass with automatic validation
/// let output = transformer.forward_validated(&input)?;
///
/// if output.has_critical_anomaly() {
///     println!("Critical anomaly detected!");
///     println!("{}", output.report.unwrap().summary());
/// }
///
/// // Access physics metrics
/// println!("Health rate: {:.2}%", transformer.health_rate() * 100.0);
/// println!("Total anomalies: {}", transformer.total_anomalies());
/// ```
pub struct ValidatedTransformer {
    /// Inner transformer
    transformer: BidirectionalTorusTransformer,
    /// Physics collider
    collider: TorusCollider,
    /// Validation context
    context: ValidationContext,
    /// Validation history
    history: ValidationHistory,
    /// Configuration
    config: ValidatedTransformerConfig,
    /// Variable map (for training)
    var_map: VarMap,
    /// Device
    #[allow(dead_code)]
    device: Device,
    /// Current step
    step: u64,
    /// Consecutive anomaly counter
    consecutive_anomalies: usize,
    /// Whether auto-paused
    is_paused: bool,
    /// Last validation report
    last_report: Option<ColliderReport>,
    /// Total forward passes
    total_forwards: u64,
    /// Total validated forwards
    validated_forwards: u64,
}

impl ValidatedTransformer {
    /// Create a new validated transformer
    pub fn new(config: ValidatedTransformerConfig, device: &Device) -> TorusResult<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let transformer = BidirectionalTorusTransformer::new(
            config.model.clone(),
            config.vocab_size,
            vb,
            device,
        )?;

        let collider_config = ColliderConfig {
            enabled: config.validation.enabled,
            darkness_grid: (config.model.n_major, config.model.n_minor),
            detector_enabled: config.validation.level == ValidationLevel::Full,
            n_major: config.model.n_major,
            n_minor: config.model.n_minor,
            ..ColliderConfig::default()
        };
        let collider = TorusCollider::new(collider_config);

        let context = ValidationContext::new(
            config.validation.clone(),
            config.model.n_major,
            config.model.n_minor,
        );

        let history = ValidationHistory::new(config.validation.max_history_size);

        Ok(Self {
            transformer,
            collider,
            context,
            history,
            config,
            var_map,
            device: device.clone(),
            step: 0,
            consecutive_anomalies: 0,
            is_paused: false,
            last_report: None,
            total_forwards: 0,
            validated_forwards: 0,
        })
    }

    /// Wrap an existing transformer with validation
    pub fn wrap(
        transformer: BidirectionalTorusTransformer,
        var_map: VarMap,
        config: ValidatedTransformerConfig,
        device: &Device,
    ) -> TorusResult<Self> {
        let collider_config = ColliderConfig {
            enabled: config.validation.enabled,
            darkness_grid: (config.model.n_major, config.model.n_minor),
            detector_enabled: config.validation.level == ValidationLevel::Full,
            n_major: config.model.n_major,
            n_minor: config.model.n_minor,
            ..ColliderConfig::default()
        };
        let collider = TorusCollider::new(collider_config);

        let context = ValidationContext::new(
            config.validation.clone(),
            config.model.n_major,
            config.model.n_minor,
        );

        let history = ValidationHistory::new(config.validation.max_history_size);

        Ok(Self {
            transformer,
            collider,
            context,
            history,
            config,
            var_map,
            device: device.clone(),
            step: 0,
            consecutive_anomalies: 0,
            is_paused: false,
            last_report: None,
            total_forwards: 0,
            validated_forwards: 0,
        })
    }

    /// Forward pass with automatic validation
    ///
    /// This is the main entry point for validated inference/training.
    /// Validation is performed according to the sample rate in the config.
    pub fn forward_validated(&mut self, x: &Tensor) -> TorusResult<ValidatedOutput> {
        self.total_forwards += 1;
        self.step += 1;

        // Check if paused
        if self.is_paused {
            return Err(crate::TorusError::ComputationError(
                "Transformer is paused due to critical anomaly. Call resume() to continue."
                    .to_string(),
            ));
        }

        // Determine if we should validate this step
        let should_validate = self.config.validation.enabled && self.context.should_validate();

        // Time the forward pass
        let forward_start = Instant::now();

        // Reset state for new sequence
        self.transformer.reset_state()?;

        // Forward pass through transformer
        let output = self.transformer.forward(x)?;

        let forward_time = forward_start.elapsed().as_secs_f64() * 1000.0;

        // Validation
        let validation_start = Instant::now();

        let (report, is_healthy, anomaly_count, conservation_violations, causality_violations) =
            if should_validate {
                self.validated_forwards += 1;
                self.collider.next_step();

                // Record input tensor for anomaly detection
                self.collider.record_input(x)?;

                // Record output tensor
                self.collider.record_output(&output)?;

                // Get validation report
                let report = self.collider.report();

                // Update history
                self.history.record(self.step, report.clone());

                // Track consecutive anomalies
                if report.is_healthy {
                    self.consecutive_anomalies = 0;
                } else {
                    self.consecutive_anomalies += 1;
                }

                // Auto-pause on critical or threshold
                if self.config.auto_pause_on_critical {
                    if report.anomaly_report.stats.critical_anomalies > 0 {
                        self.is_paused = true;
                    }
                    if self.consecutive_anomalies >= self.config.anomaly_alert_threshold {
                        self.is_paused = true;
                    }
                }

                let is_healthy = report.is_healthy;
                let anomaly_count = report.anomaly_report.stats.total_anomalies as usize;
                let conservation_violations = report.conservation_report.n_violations;
                let causality_violations = report.causality_report.violations.len();

                self.last_report = Some(report.clone());

                (
                    Some(report),
                    is_healthy,
                    anomaly_count,
                    conservation_violations,
                    causality_violations,
                )
            } else {
                (None, true, 0, 0, 0)
            };

        let validation_time = validation_start.elapsed().as_secs_f64() * 1000.0;

        // Advance context
        self.context.next_step();

        Ok(ValidatedOutput {
            output,
            was_validated: should_validate,
            report,
            is_healthy,
            anomaly_count,
            conservation_violations,
            causality_violations,
            forward_time_ms: forward_time,
            validation_time_ms: validation_time,
        })
    }

    /// Forward pass without validation (for inference speed)
    pub fn forward(&mut self, x: &Tensor) -> TorusResult<Tensor> {
        self.total_forwards += 1;
        self.step += 1;
        self.transformer.reset_state()?;
        self.transformer.forward(x)
    }

    /// Resume after auto-pause
    pub fn resume(&mut self) {
        self.is_paused = false;
        self.consecutive_anomalies = 0;
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// Get health rate (fraction of healthy steps)
    pub fn health_rate(&self) -> f64 {
        self.history.health_rate(100)
    }

    /// Get total anomalies
    pub fn total_anomalies(&self) -> u64 {
        self.history
            .recent(1000)
            .iter()
            .map(|(_, r)| r.anomaly_count as u64)
            .sum()
    }

    /// Get anomaly trend (positive = worsening)
    pub fn anomaly_trend(&self) -> f64 {
        self.history.anomaly_trend(100)
    }

    /// Get last validation report
    pub fn last_report(&self) -> Option<&ColliderReport> {
        self.last_report.as_ref()
    }

    /// Get validation history
    pub fn history(&self) -> &ValidationHistory {
        &self.history
    }

    /// Get the collider for direct access
    pub fn collider(&self) -> &TorusCollider {
        &self.collider
    }

    /// Get mutable collider
    pub fn collider_mut(&mut self) -> &mut TorusCollider {
        &mut self.collider
    }

    /// Get inner transformer
    pub fn transformer(&self) -> &BidirectionalTorusTransformer {
        &self.transformer
    }

    /// Get mutable transformer
    pub fn transformer_mut(&mut self) -> &mut BidirectionalTorusTransformer {
        &mut self.transformer
    }

    /// Get variable map (for training)
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Get current step
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Get total forward passes
    pub fn total_forwards(&self) -> u64 {
        self.total_forwards
    }

    /// Get validated forward passes
    pub fn validated_forwards(&self) -> u64 {
        self.validated_forwards
    }

    /// Get validation rate (actual vs configured)
    pub fn actual_validation_rate(&self) -> f64 {
        if self.total_forwards == 0 {
            0.0
        } else {
            self.validated_forwards as f64 / self.total_forwards as f64
        }
    }

    /// Get config
    pub fn config(&self) -> &ValidatedTransformerConfig {
        &self.config
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.config.model.seq_len()
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.config.model.d_model
    }

    /// Print status summary
    pub fn print_status(&self) {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║              VALIDATED TRANSFORMER STATUS                      ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Step: {:8}    Paused: {:5}                              ║",
            self.step,
            if self.is_paused { "YES" } else { "NO" }
        );
        println!(
            "║  Total forwards: {:8}    Validated: {:8}              ║",
            self.total_forwards, self.validated_forwards
        );
        println!(
            "║  Validation rate: {:5.1}% (config: {:5.1}%)                     ║",
            self.actual_validation_rate() * 100.0,
            self.config.validation.sample_rate * 100.0
        );
        println!(
            "║  Health rate: {:5.1}%    Anomaly trend: {:+.4}                ║",
            self.health_rate() * 100.0,
            self.anomaly_trend()
        );
        println!(
            "║  Consecutive anomalies: {:3} / {:3} (threshold)                 ║",
            self.consecutive_anomalies, self.config.anomaly_alert_threshold
        );
        println!("╚════════════════════════════════════════════════════════════════╝");
    }

    /// Get comprehensive summary
    pub fn summary(&self) -> String {
        format!(
            "ValidatedTransformer [step={}, health={:.1}%, anomalies={}, paused={}]",
            self.step,
            self.health_rate() * 100.0,
            self.total_anomalies(),
            self.is_paused
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUND EXAMPLE
// ═══════════════════════════════════════════════════════════════════════════════

/// Run a compound integration example demonstrating all features
pub fn run_compound_example(device: &Device) -> TorusResult<()> {
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("           COMPOUND INTEGRATION EXAMPLE");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Create configuration
    let config = ValidatedTransformerConfig {
        model: BidirectionalTorusConfig {
            d_model: 64,
            d_ff: 256,
            n_heads: 4,
            n_layers: 2,
            n_major: 8,
            n_minor: 4,
            ..BidirectionalTorusConfig::default()
        },
        validation: ValidationConfig {
            enabled: true,
            level: ValidationLevel::Standard,
            sample_rate: 0.5, // 50% validation
            ..ValidationConfig::default()
        },
        vocab_size: Some(100),
        anomaly_alert_threshold: 5,
        auto_pause_on_critical: true,
        ..ValidatedTransformerConfig::default()
    };

    println!("Configuration:");
    println!(
        "  Model: {}d, {} heads, {} layers",
        config.model.d_model, config.model.n_heads, config.model.n_layers
    );
    println!(
        "  Torus: {}x{} = {} positions",
        config.model.n_major,
        config.model.n_minor,
        config.model.seq_len()
    );
    println!(
        "  Validation: {:?} @ {:.0}% sample rate",
        config.validation.level,
        config.validation.sample_rate * 100.0
    );
    println!();

    // Create validated transformer
    let mut transformer = ValidatedTransformer::new(config.clone(), device)?;

    println!("Created ValidatedTransformer");
    println!("  Sequence length: {}", transformer.seq_len());
    println!("  Model dimension: {}", transformer.d_model());
    println!();

    // Generate test input
    let batch_size = 4;
    let seq_len = transformer.seq_len();
    let d_model = transformer.d_model();

    let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, d_model), device)?;
    println!("Input shape: {:?}", input.dims());

    // Run multiple forward passes
    println!("\nRunning 20 forward passes...\n");

    for i in 0..20 {
        let output = transformer.forward_validated(&input)?;

        if i % 5 == 0 || !output.is_healthy {
            let status = if output.is_healthy { "OK" } else { "!!" };
            println!(
                "  Step {:2}: [{}] validated={}, anomalies={}, time={:.2}ms",
                i,
                status,
                output.was_validated,
                output.anomaly_count,
                output.forward_time_ms + output.validation_time_ms
            );
        }

        if transformer.is_paused() {
            println!("\n  ⚠ Transformer auto-paused due to anomalies!");
            transformer.resume();
            println!("  ↻ Resumed\n");
        }
    }

    // Print final status
    println!();
    transformer.print_status();

    // Print collider summary
    println!("\n{}", transformer.collider().summary());

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{AnomalyReport, CausalityReport, ConservationReport, MetricsReport};

    #[test]
    fn test_validation_config_presets() {
        let full = ValidationConfig::full();
        assert!(full.enabled);
        assert_eq!(full.level, ValidationLevel::Full);
        assert!(full.record_qkv);

        let minimal = ValidationConfig::minimal();
        assert!(minimal.enabled);
        assert_eq!(minimal.level, ValidationLevel::Minimal);
        assert!(!minimal.record_qkv);

        let disabled = ValidationConfig::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_validation_context_creation() {
        let config = ValidationConfig::default();
        let context = ValidationContext::new(config, 32, 16);

        assert!(context.is_healthy());
        assert_eq!(context.current_step(), 0);
    }

    #[test]
    fn test_validation_context_step() {
        let config = ValidationConfig::default();
        let context = ValidationContext::new(config, 32, 16);

        context.next_step();
        assert_eq!(context.current_step(), 1);

        for _ in 0..10 {
            context.next_step();
        }
        assert_eq!(context.current_step(), 11);
    }

    #[test]
    fn test_validation_history() {
        let mut history = ValidationHistory::new(100);

        assert!(history.is_empty());
        assert_eq!(history.health_rate(10), 1.0);

        // Add some records
        for i in 0..10 {
            let report = ColliderReport {
                step: i,
                is_healthy: i % 2 == 0,
                anomaly_report: AnomalyReport::default(),
                conservation_report: ConservationReport::new(vec![], None, 0),
                causality_report: CausalityReport::default(),
                metrics_report: MetricsReport::default(),
                detector_summary: None,
            };
            history.record(i, report);
        }

        assert_eq!(history.len(), 10);
        assert_eq!(history.health_rate(10), 0.5);
    }

    #[test]
    fn test_layer_validation_report() {
        let report = LayerValidationReport::healthy(3);
        assert!(report.is_healthy);
        assert_eq!(report.layer, 3);
        println!("{}", report.summary());
    }

    #[test]
    fn test_sample_rate() {
        let mut config = ValidationConfig::default();
        config.sample_rate = 0.5; // 50% of steps

        let context = ValidationContext::new(config, 32, 16);

        let mut validated_count = 0;
        for _ in 0..100 {
            if context.should_validate() {
                validated_count += 1;
            }
            context.next_step();
        }

        // Should be approximately 50
        assert!(validated_count >= 40 && validated_count <= 60);
    }

    #[test]
    fn test_validated_trainer() {
        let config = ValidationConfig::training();
        let context = ValidationContext::new(config, 32, 16);
        let mut trainer = ValidatedTrainer::new(context);

        // Simulate 10 training steps
        for i in 0..10 {
            let result = trainer
                .record_step(1.0 - i as f32 * 0.1, 5.0, None, 0)
                .unwrap();
            assert!(result.is_healthy);
            assert!(!result.should_pause);
        }

        let stats = trainer.stats();
        assert_eq!(stats.total_steps, 10);
        assert_eq!(stats.steps_with_issues, 0);
    }

    #[test]
    fn test_dashboard_render() {
        let config = ValidationConfig::default();
        let context = ValidationContext::new(config, 32, 16);
        let dashboard = ValidationDashboard::new(context);

        let output = dashboard.render();
        assert!(output.contains("TORUS COLLIDER"));
        assert!(output.contains("Step:"));
        println!("{}", output);
    }

    #[test]
    fn test_trainer_anomaly_detection() {
        let mut config = ValidationConfig::default();
        config.alert_threshold = 5;

        let context = ValidationContext::new(config, 32, 16);
        let mut trainer = ValidatedTrainer::new(context);

        // Normal step
        let result = trainer.record_step(1.0, 5.0, None, 0).unwrap();
        assert!(!result.should_pause);

        // Step with NaN loss should trigger pause
        let result = trainer.record_step(f32::NAN, 5.0, None, 0).unwrap();
        assert!(result.should_pause);
        assert!(!result.warnings.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // ValidatedTransformer Tests
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_validated_transformer_config_default() {
        let config = ValidatedTransformerConfig::default();
        assert!(config.track_gradients);
        assert!(config.record_attention_patterns);
        assert_eq!(config.anomaly_alert_threshold, 5);
        assert!(config.auto_pause_on_critical);
        assert!(config.vocab_size.is_none());
    }

    #[test]
    fn test_validated_transformer_creation() {
        use candle_core::Device;

        // Use small config matching test_full_pipeline_shapes
        let mut config = ValidatedTransformerConfig::default();
        config.vocab_size = Some(100);
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;

        let device = Device::Cpu;
        let transformer = ValidatedTransformer::new(config, &device);

        assert!(
            transformer.is_ok(),
            "Failed to create transformer: {:?}",
            transformer.err()
        );
        let transformer = transformer.unwrap();
        assert!(!transformer.is_paused());
        assert_eq!(transformer.health_rate(), 1.0);
        assert_eq!(transformer.total_anomalies(), 0);
    }

    #[test]
    fn test_validated_transformer_forward() {
        use candle_core::{Device, Tensor};

        let mut config = ValidatedTransformerConfig::default();
        // No vocab_size - expects 3D embedding input
        config.vocab_size = None;
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;

        let device = Device::Cpu;
        let mut transformer = ValidatedTransformer::new(config.clone(), &device).unwrap();

        // Create input tensor (batch_size=1, seq_len = n_major * n_minor = 16, d_model=32)
        let seq_len = config.model.n_major * config.model.n_minor;
        let input =
            Tensor::randn(0.0f32, 1.0, (1, seq_len, config.model.d_model), &device).unwrap();

        // Forward with validation
        let output = transformer.forward_validated(&input);
        assert!(output.is_ok(), "Forward failed: {:?}", output.err());

        let output = output.unwrap();
        assert!(output.was_validated);
        assert!(output.is_healthy);
        assert!(output.forward_time_ms > 0.0);
    }

    #[test]
    fn test_validated_transformer_no_validation() {
        use candle_core::{Device, Tensor};

        let mut config = ValidatedTransformerConfig::default();
        config.vocab_size = None; // No embedding - 3D input
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;

        let device = Device::Cpu;
        let mut transformer = ValidatedTransformer::new(config.clone(), &device).unwrap();

        // Forward without validation - 3D input
        let seq_len = config.model.n_major * config.model.n_minor;
        let input =
            Tensor::randn(0.0f32, 1.0, (1, seq_len, config.model.d_model), &device).unwrap();

        let output = transformer.forward(&input);
        assert!(output.is_ok(), "Forward failed: {:?}", output.err());

        // Shape should be [batch, seq, d_model]
        let output = output.unwrap();
        let dims = output.dims();
        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], seq_len); // seq
        assert_eq!(dims[2], config.model.d_model); // d_model
    }

    #[test]
    fn test_validated_transformer_pause_resume() {
        use candle_core::Device;

        let mut config = ValidatedTransformerConfig::default();
        config.vocab_size = Some(100);
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;
        config.auto_pause_on_critical = true;

        let device = Device::Cpu;
        let mut transformer = ValidatedTransformer::new(config, &device).unwrap();

        // Initially not paused
        assert!(!transformer.is_paused());

        // Test resume functionality
        transformer.resume();
        assert!(!transformer.is_paused());
    }

    #[test]
    fn test_validated_transformer_collider_access() {
        use candle_core::Device;

        let mut config = ValidatedTransformerConfig::default();
        config.vocab_size = Some(100);
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;

        let device = Device::Cpu;
        let transformer = ValidatedTransformer::new(config, &device).unwrap();

        // Should be able to access collider
        let collider = transformer.collider();
        assert!(collider.is_enabled());
    }

    #[test]
    fn test_validated_transformer_history_access() {
        use candle_core::{Device, Tensor};

        let mut config = ValidatedTransformerConfig::default();
        config.vocab_size = None; // No embedding
        config.model.d_model = 32;
        config.model.d_ff = 128;
        config.model.n_heads = 4;
        config.model.n_layers = 2;
        config.model.n_major = 4;
        config.model.n_minor = 4;

        let device = Device::Cpu;
        let mut transformer = ValidatedTransformer::new(config.clone(), &device).unwrap();

        // Forward a few times with 3D input
        let seq_len = config.model.n_major * config.model.n_minor;
        let input =
            Tensor::randn(0.0f32, 1.0, (1, seq_len, config.model.d_model), &device).unwrap();

        for _ in 0..3 {
            let _ = transformer.forward_validated(&input);
        }

        // Check history
        let history = transformer.history();
        assert!(!history.is_empty());
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_validated_output_methods() {
        let output = ValidatedOutput {
            output: Tensor::zeros(
                (1, 16, 64),
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap(),
            was_validated: true,
            report: None,
            is_healthy: true,
            anomaly_count: 0,
            conservation_violations: 0,
            causality_violations: 0,
            forward_time_ms: 10.0,
            validation_time_ms: 5.0,
        };

        assert!(output.is_healthy);
        assert!(!output.has_critical_anomaly());

        let summary = output.summary();
        assert!(summary.contains("OK"));
    }

    #[test]
    fn test_validated_output_with_issues() {
        let output = ValidatedOutput {
            output: Tensor::zeros(
                (1, 16, 64),
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap(),
            was_validated: true,
            report: None,
            is_healthy: false,
            anomaly_count: 3,
            conservation_violations: 1,
            causality_violations: 0,
            forward_time_ms: 10.0,
            validation_time_ms: 5.0,
        };

        assert!(!output.is_healthy);
        assert_eq!(output.anomaly_count, 3);

        let summary = output.summary();
        assert!(summary.contains("ISSUES"));
        assert!(summary.contains("anomalies=3"));
    }

    #[test]
    fn test_history_summary() {
        let summary = HistorySummary {
            is_healthy: true,
            anomaly_count: 2,
            conservation_violations: 1,
            causality_violations: 0,
            collision_count: 100,
        };

        assert!(summary.is_healthy);
        assert_eq!(summary.total_anomalies(), 2);
    }

    #[test]
    fn test_validation_history_recent() {
        let mut history = ValidationHistory::new(100);

        // Add some records
        for i in 0..10 {
            let report = ColliderReport {
                step: i,
                is_healthy: true,
                anomaly_report: AnomalyReport::default(),
                conservation_report: ConservationReport::new(vec![], None, 0),
                causality_report: CausalityReport::default(),
                metrics_report: MetricsReport::default(),
                detector_summary: None,
            };
            history.record(i, report);
        }

        // Get recent 5
        let recent = history.recent(5);
        assert_eq!(recent.len(), 5);

        // Should be the last 5 entries
        let steps: Vec<_> = recent.iter().map(|(step, _)| *step).collect();
        assert_eq!(steps, vec![5, 6, 7, 8, 9]);
    }
}
