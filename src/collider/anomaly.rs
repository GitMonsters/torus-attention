//! Anomaly Detection for Torus Collider
//!
//! Detects anomalies in the attention computation "collisions":
//! - NaN/Inf values (particle "explosions")
//! - Exploding gradients (runaway energy)
//! - Vanishing gradients (particle "absorption")
//! - Attention weight anomalies (unusual collision patterns)
//! - Numerical instabilities (detector malfunctions)
//!
//! # Physics Analogy
//!
//! In particle physics, anomaly detection is critical:
//! - Detector malfunctions → NaN/Inf in neural networks
//! - Energy spikes → Exploding gradients
//! - Missing energy → Vanishing gradients
//! - Unusual event topology → Attention pattern anomalies
//!
//! This module provides always-on monitoring during training and inference.

use crate::collider::particles::{Particle, ParticleBeam};
use crate::geometry::TorusCoordinate;
use crate::TorusResult;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// NaN value detected
    NaN,
    /// Infinite value detected
    Inf,
    /// Negative infinite value
    NegInf,
    /// Gradient magnitude too large (exploding)
    ExplodingGradient,
    /// Gradient magnitude too small (vanishing)
    VanishingGradient,
    /// Attention weight out of bounds
    AttentionOutOfBounds,
    /// Attention entropy too low (over-focused)
    AttentionCollapse,
    /// Attention entropy too high (uniform/useless)
    AttentionDiffuse,
    /// Energy conservation violation
    EnergyViolation,
    /// Momentum conservation violation
    MomentumViolation,
    /// Numerical instability (rapid oscillation)
    NumericalInstability,
    /// Dead neuron (always zero)
    DeadNeuron,
    /// Saturated neuron (always max/min)
    SaturatedNeuron,
}

impl AnomalyType {
    /// Get severity level (0-3: info, warning, error, critical)
    pub fn severity(&self) -> u8 {
        match self {
            Self::NaN | Self::Inf | Self::NegInf => 3, // Critical
            Self::ExplodingGradient => 3,              // Critical
            Self::VanishingGradient => 2,              // Error
            Self::AttentionOutOfBounds => 2,           // Error
            Self::AttentionCollapse => 1,              // Warning
            Self::AttentionDiffuse => 1,               // Warning
            Self::EnergyViolation => 2,                // Error
            Self::MomentumViolation => 2,              // Error
            Self::NumericalInstability => 2,           // Error
            Self::DeadNeuron => 1,                     // Warning
            Self::SaturatedNeuron => 1,                // Warning
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::NaN => "NaN Detected",
            Self::Inf => "Positive Infinity",
            Self::NegInf => "Negative Infinity",
            Self::ExplodingGradient => "Exploding Gradient",
            Self::VanishingGradient => "Vanishing Gradient",
            Self::AttentionOutOfBounds => "Attention Out of Bounds",
            Self::AttentionCollapse => "Attention Collapse",
            Self::AttentionDiffuse => "Attention Diffuse",
            Self::EnergyViolation => "Energy Violation",
            Self::MomentumViolation => "Momentum Violation",
            Self::NumericalInstability => "Numerical Instability",
            Self::DeadNeuron => "Dead Neuron",
            Self::SaturatedNeuron => "Saturated Neuron",
        }
    }

    /// Check if this is a critical anomaly that should halt training
    pub fn is_critical(&self) -> bool {
        self.severity() >= 3
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY EVENT
// ═══════════════════════════════════════════════════════════════════════════════

/// A detected anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// Unique event ID
    pub id: u64,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// When it occurred (step/iteration)
    pub step: u64,
    /// Layer where it occurred
    pub layer: usize,
    /// Head (for multi-head attention)
    pub head: Option<usize>,
    /// Stream ID
    pub stream_id: Option<usize>,
    /// Position on torus (if applicable)
    pub position: Option<TorusCoordinate>,
    /// The problematic value
    pub value: f64,
    /// Threshold that was violated
    pub threshold: f64,
    /// Additional context
    pub context: String,
    /// Particle ID if associated with a particle
    pub particle_id: Option<u64>,
}

impl AnomalyEvent {
    /// Create a new anomaly event
    pub fn new(anomaly_type: AnomalyType, step: u64, layer: usize, value: f64) -> Self {
        Self {
            id: 0,
            anomaly_type,
            step,
            layer,
            head: None,
            stream_id: None,
            position: None,
            value,
            threshold: 0.0,
            context: String::new(),
            particle_id: None,
        }
    }

    /// Builder: set head
    pub fn with_head(mut self, head: usize) -> Self {
        self.head = Some(head);
        self
    }

    /// Builder: set stream
    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    /// Builder: set position
    pub fn with_position(mut self, position: TorusCoordinate) -> Self {
        self.position = Some(position);
        self
    }

    /// Builder: set threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Builder: set context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }

    /// Builder: set particle
    pub fn with_particle(mut self, particle_id: u64) -> Self {
        self.particle_id = Some(particle_id);
        self
    }

    /// Get severity
    pub fn severity(&self) -> u8 {
        self.anomaly_type.severity()
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        let mut s = format!(
            "[{}] {} at step {}, layer {}",
            match self.severity() {
                3 => "CRITICAL",
                2 => "ERROR",
                1 => "WARNING",
                _ => "INFO",
            },
            self.anomaly_type.name(),
            self.step,
            self.layer
        );

        if let Some(h) = self.head {
            s.push_str(&format!(", head {}", h));
        }
        if let Some(sid) = self.stream_id {
            s.push_str(&format!(", stream {}", sid));
        }

        s.push_str(&format!(": value={:.6}", self.value));

        if self.threshold != 0.0 {
            s.push_str(&format!(" (threshold={:.6})", self.threshold));
        }

        if !self.context.is_empty() {
            s.push_str(&format!(" - {}", self.context));
        }

        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY THRESHOLDS
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration thresholds for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    /// Maximum gradient magnitude before "exploding"
    pub max_gradient: f64,
    /// Minimum gradient magnitude before "vanishing"
    pub min_gradient: f64,
    /// Maximum allowed activation value
    pub max_activation: f64,
    /// Minimum attention weight (should be ≥ 0)
    pub min_attention: f64,
    /// Maximum attention weight (should be ≤ 1)
    pub max_attention: f64,
    /// Minimum attention entropy (below = collapsed)
    pub min_entropy: f64,
    /// Maximum attention entropy (above = diffuse)
    pub max_entropy: f64,
    /// Energy conservation tolerance (fraction)
    pub energy_tolerance: f64,
    /// Momentum conservation tolerance (fraction)
    pub momentum_tolerance: f64,
    /// Minimum value to not be considered "dead"
    pub dead_threshold: f64,
    /// Number of steps to track for instability detection
    pub instability_window: usize,
    /// Instability oscillation threshold
    pub instability_threshold: f64,
    /// Attention collapse threshold (max weight before considered collapsed)
    pub attention_collapse_threshold: f64,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            max_gradient: 1000.0,
            min_gradient: 1e-10,
            max_activation: 1e6,
            min_attention: -0.01, // Small negative allowed for numerical reasons
            max_attention: 1.01,  // Small over-1 allowed for numerical reasons
            min_entropy: 0.1,
            max_entropy: 0.95, // Relative to max entropy
            energy_tolerance: 0.05,
            momentum_tolerance: 0.05,
            dead_threshold: 1e-7,
            instability_window: 10,
            instability_threshold: 5.0,
            attention_collapse_threshold: 0.95, // > 95% weight on one token
        }
    }
}

impl AnomalyThresholds {
    /// Strict thresholds for debugging
    pub fn strict() -> Self {
        Self {
            max_gradient: 100.0,
            min_gradient: 1e-8,
            max_activation: 1e4,
            min_attention: 0.0,
            max_attention: 1.0,
            min_entropy: 0.2,
            max_entropy: 0.9,
            energy_tolerance: 0.01,
            momentum_tolerance: 0.01,
            dead_threshold: 1e-6,
            instability_window: 5,
            instability_threshold: 3.0,
            attention_collapse_threshold: 0.9,
        }
    }

    /// Relaxed thresholds for production
    pub fn relaxed() -> Self {
        Self {
            max_gradient: 10000.0,
            min_gradient: 1e-12,
            max_activation: 1e8,
            min_attention: -0.1,
            max_attention: 1.1,
            min_entropy: 0.05,
            max_entropy: 0.99,
            energy_tolerance: 0.1,
            momentum_tolerance: 0.1,
            dead_threshold: 1e-9,
            instability_window: 20,
            instability_threshold: 10.0,
            attention_collapse_threshold: 0.99,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR ANALYZER
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyzes tensors for anomalies
pub struct TensorAnalyzer {
    /// Thresholds for detection
    pub thresholds: AnomalyThresholds,
}

impl TensorAnalyzer {
    /// Create a new analyzer
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        Self { thresholds }
    }

    /// Check a tensor for NaN/Inf values
    pub fn check_nan_inf(&self, tensor: &Tensor, _context: &str) -> TorusResult<Vec<AnomalyType>> {
        let mut anomalies = Vec::new();
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        for &val in &flat {
            if val.is_nan() {
                anomalies.push(AnomalyType::NaN);
                break;
            }
            if val.is_infinite() {
                if val > 0.0 {
                    anomalies.push(AnomalyType::Inf);
                } else {
                    anomalies.push(AnomalyType::NegInf);
                }
                break;
            }
        }

        Ok(anomalies)
    }

    /// Check gradient magnitude
    pub fn check_gradient(&self, grad: &Tensor) -> TorusResult<Option<AnomalyType>> {
        let flat: Vec<f32> = grad.flatten_all()?.to_vec1()?;
        let magnitude: f64 = flat.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();

        if magnitude > self.thresholds.max_gradient {
            return Ok(Some(AnomalyType::ExplodingGradient));
        }
        if magnitude < self.thresholds.min_gradient && magnitude > 0.0 {
            return Ok(Some(AnomalyType::VanishingGradient));
        }

        Ok(None)
    }

    /// Check attention weights
    pub fn check_attention(&self, weights: &Tensor) -> TorusResult<Vec<AnomalyType>> {
        let mut anomalies = Vec::new();
        let flat: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        // Check bounds
        for &val in &flat {
            if (val as f64) < self.thresholds.min_attention {
                anomalies.push(AnomalyType::AttentionOutOfBounds);
                break;
            }
            if (val as f64) > self.thresholds.max_attention {
                anomalies.push(AnomalyType::AttentionOutOfBounds);
                break;
            }
        }

        // Check entropy (assuming softmax output, last dimension is attention)
        let dims = weights.dims();
        if dims.len() >= 2 {
            let seq_len = dims[dims.len() - 1];
            let max_entropy = (seq_len as f64).ln();

            // Calculate average entropy
            let n_rows = flat.len() / seq_len;
            let mut total_entropy = 0.0;

            for row in 0..n_rows {
                let start = row * seq_len;
                let end = start + seq_len;
                let row_vals = &flat[start..end];

                let entropy: f64 = row_vals
                    .iter()
                    .map(|&p| {
                        let p = p as f64;
                        if p > 1e-10 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum();

                total_entropy += entropy;
            }

            let avg_entropy = total_entropy / n_rows as f64;
            let relative_entropy = avg_entropy / max_entropy;

            if relative_entropy < self.thresholds.min_entropy {
                anomalies.push(AnomalyType::AttentionCollapse);
            }
            if relative_entropy > self.thresholds.max_entropy {
                anomalies.push(AnomalyType::AttentionDiffuse);
            }
        }

        Ok(anomalies)
    }

    /// Check for dead neurons (all zeros)
    pub fn check_dead_neurons(&self, activations: &Tensor) -> TorusResult<Vec<usize>> {
        let flat: Vec<f32> = activations.flatten_all()?.to_vec1()?;
        let dims = activations.dims();

        // Assume last dimension is features
        let n_features = *dims.last().unwrap_or(&1);
        let n_samples = flat.len() / n_features;

        let mut dead_neurons = Vec::new();

        for feature in 0..n_features {
            let mut all_dead = true;
            for sample in 0..n_samples {
                let val = flat[sample * n_features + feature].abs() as f64;
                if val > self.thresholds.dead_threshold {
                    all_dead = false;
                    break;
                }
            }
            if all_dead {
                dead_neurons.push(feature);
            }
        }

        Ok(dead_neurons)
    }

    /// Get statistics about a tensor
    pub fn tensor_stats(&self, tensor: &Tensor) -> TorusResult<TensorStats> {
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let n = flat.len() as f64;

        let sum: f64 = flat.iter().map(|&v| v as f64).sum();
        let mean = sum / n;

        let sq_sum: f64 = flat.iter().map(|&v| (v as f64).powi(2)).sum();
        let variance = sq_sum / n - mean.powi(2);
        let std = variance.sqrt();

        let min = flat.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min);
        let max = flat
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);

        let has_nan = flat.iter().any(|v| v.is_nan());
        let has_inf = flat.iter().any(|v| v.is_infinite());
        let n_zeros = flat.iter().filter(|&&v| v.abs() < 1e-10).count();

        Ok(TensorStats {
            n_elements: flat.len(),
            mean,
            std,
            min,
            max,
            has_nan,
            has_inf,
            n_zeros,
            zero_fraction: n_zeros as f64 / n,
        })
    }
}

impl Default for TensorAnalyzer {
    fn default() -> Self {
        Self::new(AnomalyThresholds::default())
    }
}

/// Statistics about a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    /// Number of elements
    pub n_elements: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Contains NaN
    pub has_nan: bool,
    /// Contains Inf
    pub has_inf: bool,
    /// Number of zeros
    pub n_zeros: usize,
    /// Fraction of zeros
    pub zero_fraction: f64,
}

impl TensorStats {
    /// Check if tensor is healthy
    pub fn is_healthy(&self) -> bool {
        !self.has_nan
            && !self.has_inf
            && self.min > -1e10
            && self.max < 1e10
            && self.zero_fraction < 0.99
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "n={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}, zeros={:.1}%{}{}",
            self.n_elements,
            self.mean,
            self.std,
            self.min,
            self.max,
            self.zero_fraction * 100.0,
            if self.has_nan { " [NaN!]" } else { "" },
            if self.has_inf { " [Inf!]" } else { "" }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTICLE ANOMALY DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Detects anomalies in particles (tensor representations)
pub struct ParticleAnomalyDetector {
    /// Tensor analyzer
    pub analyzer: TensorAnalyzer,
    /// Thresholds
    pub thresholds: AnomalyThresholds,
}

impl ParticleAnomalyDetector {
    /// Create a new detector
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        Self {
            analyzer: TensorAnalyzer::new(thresholds.clone()),
            thresholds,
        }
    }

    /// Check a particle for anomalies
    pub fn check_particle(&self, particle: &Particle) -> Vec<AnomalyType> {
        let mut anomalies = Vec::new();

        // Check for infinite energy
        if particle.energy().is_nan() {
            anomalies.push(AnomalyType::NaN);
        }
        if particle.energy().is_infinite() {
            if particle.energy() > 0.0 {
                anomalies.push(AnomalyType::Inf);
            } else {
                anomalies.push(AnomalyType::NegInf);
            }
        }

        // Check momentum components
        let p = &particle.momentum;
        if p.px.is_nan() || p.py.is_nan() || p.pz.is_nan() {
            anomalies.push(AnomalyType::NaN);
        }

        // Check for extremely high energy (exploding)
        if particle.energy().abs() > self.thresholds.max_activation {
            anomalies.push(AnomalyType::ExplodingGradient);
        }

        // Check for near-zero energy (vanishing)
        if particle.energy().abs() < self.thresholds.min_gradient && particle.energy().abs() > 0.0 {
            anomalies.push(AnomalyType::VanishingGradient);
        }

        anomalies
    }

    /// Check a beam for anomalies
    pub fn check_beam(&self, beam: &ParticleBeam) -> Vec<(u64, Vec<AnomalyType>)> {
        beam.particles
            .iter()
            .filter_map(|p| {
                let anomalies = self.check_particle(p);
                if anomalies.is_empty() {
                    None
                } else {
                    Some((p.id, anomalies))
                }
            })
            .collect()
    }

    /// Check conservation laws between beams
    pub fn check_conservation(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> Vec<AnomalyType> {
        let mut anomalies = Vec::new();

        // Energy conservation
        let e_in = incoming.total_energy;
        let e_out = outgoing.total_energy;
        let e_diff = (e_in - e_out).abs();
        let e_avg = (e_in + e_out) / 2.0 + 1e-10;

        if e_diff / e_avg > self.thresholds.energy_tolerance {
            anomalies.push(AnomalyType::EnergyViolation);
        }

        // Momentum conservation
        let p_in = &incoming.total_momentum;
        let p_out = &outgoing.total_momentum;

        let px_diff = (p_in.px - p_out.px).abs();
        let py_diff = (p_in.py - p_out.py).abs();
        let pz_diff = (p_in.pz - p_out.pz).abs();

        let p_mag = p_in.three_momentum_magnitude() + 1e-10;
        let p_diff = (px_diff + py_diff + pz_diff) / p_mag;

        if p_diff > self.thresholds.momentum_tolerance {
            anomalies.push(AnomalyType::MomentumViolation);
        }

        anomalies
    }
}

impl Default for ParticleAnomalyDetector {
    fn default() -> Self {
        Self::new(AnomalyThresholds::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY MONITOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Continuous anomaly monitoring system
#[derive(Debug)]
pub struct AnomalyMonitor {
    /// Detected events
    pub events: Vec<AnomalyEvent>,
    /// Thresholds
    pub thresholds: AnomalyThresholds,
    /// Current step
    pub step: u64,
    /// History for instability detection (value, step)
    pub history: VecDeque<(f64, u64)>,
    /// Event counter
    next_event_id: u64,
    /// Statistics
    pub stats: AnomalyStats,
}

/// Anomaly statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyStats {
    /// Total checks performed
    pub total_checks: u64,
    /// Total anomalies detected
    pub total_anomalies: u64,
    /// Critical anomalies
    pub critical_anomalies: u64,
    /// Anomalies by type
    pub by_type: [u64; 13],
    /// Last anomaly step
    pub last_anomaly_step: Option<u64>,
}

impl AnomalyMonitor {
    /// Create a new monitor
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        Self {
            events: Vec::new(),
            thresholds,
            step: 0,
            history: VecDeque::with_capacity(100),
            next_event_id: 0,
            stats: AnomalyStats::default(),
        }
    }

    /// Advance to next step
    pub fn next_step(&mut self) {
        self.step += 1;
    }

    /// Record an anomaly
    pub fn record(&mut self, mut event: AnomalyEvent) {
        event.id = self.next_event_id;
        self.next_event_id += 1;
        event.step = self.step;

        self.stats.total_anomalies += 1;
        if event.anomaly_type.is_critical() {
            self.stats.critical_anomalies += 1;
        }

        let type_idx = match event.anomaly_type {
            AnomalyType::NaN => 0,
            AnomalyType::Inf => 1,
            AnomalyType::NegInf => 2,
            AnomalyType::ExplodingGradient => 3,
            AnomalyType::VanishingGradient => 4,
            AnomalyType::AttentionOutOfBounds => 5,
            AnomalyType::AttentionCollapse => 6,
            AnomalyType::AttentionDiffuse => 7,
            AnomalyType::EnergyViolation => 8,
            AnomalyType::MomentumViolation => 9,
            AnomalyType::NumericalInstability => 10,
            AnomalyType::DeadNeuron => 11,
            AnomalyType::SaturatedNeuron => 12,
        };
        self.stats.by_type[type_idx] += 1;
        self.stats.last_anomaly_step = Some(self.step);

        self.events.push(event);
    }

    /// Check a value and record if anomalous
    pub fn check_value(
        &mut self,
        value: f64,
        layer: usize,
        context: &str,
    ) -> Option<&AnomalyEvent> {
        self.stats.total_checks += 1;

        if value.is_nan() {
            self.record(
                AnomalyEvent::new(AnomalyType::NaN, self.step, layer, value).with_context(context),
            );
            return self.events.last();
        }

        if value.is_infinite() {
            let anomaly_type = if value > 0.0 {
                AnomalyType::Inf
            } else {
                AnomalyType::NegInf
            };
            self.record(
                AnomalyEvent::new(anomaly_type, self.step, layer, value).with_context(context),
            );
            return self.events.last();
        }

        // Track history for instability detection
        self.history.push_back((value, self.step));
        if self.history.len() > self.thresholds.instability_window {
            self.history.pop_front();
        }

        // Check for instability
        if self.history.len() >= 3 {
            let values: Vec<f64> = self.history.iter().map(|(v, _)| *v).collect();
            if self.detect_oscillation(&values) {
                self.record(
                    AnomalyEvent::new(AnomalyType::NumericalInstability, self.step, layer, value)
                        .with_context(format!("{} - rapid oscillation", context)),
                );
                return self.events.last();
            }
        }

        None
    }

    /// Check for oscillation pattern
    fn detect_oscillation(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }

        // Count sign changes in differences
        let mut sign_changes = 0;
        for i in 2..values.len() {
            let diff1 = values[i - 1] - values[i - 2];
            let diff2 = values[i] - values[i - 1];
            if diff1 * diff2 < 0.0 {
                sign_changes += 1;
            }
        }

        // Also check amplitude
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let amplitude = max - min;

        sign_changes >= values.len() - 2 && amplitude > self.thresholds.instability_threshold
    }

    /// Check a gradient value
    pub fn check_gradient(
        &mut self,
        magnitude: f64,
        layer: usize,
        context: &str,
    ) -> Option<&AnomalyEvent> {
        self.stats.total_checks += 1;

        if magnitude > self.thresholds.max_gradient {
            self.record(
                AnomalyEvent::new(AnomalyType::ExplodingGradient, self.step, layer, magnitude)
                    .with_threshold(self.thresholds.max_gradient)
                    .with_context(context),
            );
            return self.events.last();
        }

        if magnitude < self.thresholds.min_gradient && magnitude > 0.0 {
            self.record(
                AnomalyEvent::new(AnomalyType::VanishingGradient, self.step, layer, magnitude)
                    .with_threshold(self.thresholds.min_gradient)
                    .with_context(context),
            );
            return self.events.last();
        }

        None
    }

    /// Get recent anomalies (last N steps)
    pub fn recent_anomalies(&self, n_steps: u64) -> Vec<&AnomalyEvent> {
        let cutoff = self.step.saturating_sub(n_steps);
        self.events.iter().filter(|e| e.step >= cutoff).collect()
    }

    /// Get critical anomalies
    pub fn critical_anomalies(&self) -> Vec<&AnomalyEvent> {
        self.events.iter().filter(|e| e.severity() >= 3).collect()
    }

    /// Check if system is healthy (no recent critical anomalies)
    pub fn is_healthy(&self, lookback_steps: u64) -> bool {
        let cutoff = self.step.saturating_sub(lookback_steps);
        !self
            .events
            .iter()
            .any(|e| e.step >= cutoff && e.severity() >= 3)
    }

    /// Clear old events (keep last N)
    pub fn prune(&mut self, keep_last: usize) {
        if self.events.len() > keep_last {
            let drain_count = self.events.len() - keep_last;
            self.events.drain(0..drain_count);
        }
    }

    /// Get report
    pub fn report(&self) -> AnomalyReport {
        AnomalyReport {
            step: self.step,
            stats: self.stats.clone(),
            recent_events: self.recent_anomalies(100).into_iter().cloned().collect(),
            is_healthy: self.is_healthy(10),
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Anomaly Monitor @ step {}\n\
             ├─ Total checks: {}\n\
             ├─ Total anomalies: {} ({} critical)\n\
             ├─ NaN/Inf: {}/{}/{}\n\
             ├─ Gradient: explode={}, vanish={}\n\
             ├─ Attention: bounds={}, collapse={}, diffuse={}\n\
             ├─ Conservation: energy={}, momentum={}\n\
             ├─ Other: instability={}, dead={}, saturated={}\n\
             └─ Status: {}",
            self.step,
            self.stats.total_checks,
            self.stats.total_anomalies,
            self.stats.critical_anomalies,
            self.stats.by_type[0],
            self.stats.by_type[1],
            self.stats.by_type[2],
            self.stats.by_type[3],
            self.stats.by_type[4],
            self.stats.by_type[5],
            self.stats.by_type[6],
            self.stats.by_type[7],
            self.stats.by_type[8],
            self.stats.by_type[9],
            self.stats.by_type[10],
            self.stats.by_type[11],
            self.stats.by_type[12],
            if self.is_healthy(10) {
                "HEALTHY"
            } else {
                "ANOMALIES DETECTED"
            }
        )
    }

    /// Check a tensor for anomalies (NaN, Inf, etc.)
    pub fn check_tensor(
        &mut self,
        tensor: &Tensor,
        context: &str,
        layer: usize,
        stream_id: Option<usize>,
    ) -> TorusResult<Vec<AnomalyType>> {
        let mut anomalies = Vec::new();

        // Get tensor stats
        let flat = tensor.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;

        // Check for NaN/Inf
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut neginf_count = 0;

        for &v in &values {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                if v > 0.0 {
                    inf_count += 1;
                } else {
                    neginf_count += 1;
                }
            }
        }

        if nan_count > 0 {
            let mut event = AnomalyEvent::new(AnomalyType::NaN, self.step, layer, nan_count as f64)
                .with_context(format!("{} ({} NaN values)", context, nan_count));
            if let Some(sid) = stream_id {
                event = event.with_stream(sid);
            }
            self.record(event);
            anomalies.push(AnomalyType::NaN);
        }

        if inf_count > 0 {
            let mut event = AnomalyEvent::new(AnomalyType::Inf, self.step, layer, inf_count as f64)
                .with_context(format!("{} ({} Inf values)", context, inf_count));
            if let Some(sid) = stream_id {
                event = event.with_stream(sid);
            }
            self.record(event);
            anomalies.push(AnomalyType::Inf);
        }

        if neginf_count > 0 {
            let mut event =
                AnomalyEvent::new(AnomalyType::NegInf, self.step, layer, neginf_count as f64)
                    .with_context(format!("{} ({} -Inf values)", context, neginf_count));
            if let Some(sid) = stream_id {
                event = event.with_stream(sid);
            }
            self.record(event);
            anomalies.push(AnomalyType::NegInf);
        }

        self.stats.total_checks += 1;
        Ok(anomalies)
    }

    /// Check attention weights for collapse (all weight on one token)
    pub fn check_attention_collapse(
        &mut self,
        attention: &Tensor,
        layer: usize,
        head: usize,
    ) -> TorusResult<Option<AnomalyType>> {
        let flat = attention.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;

        if values.is_empty() {
            return Ok(None);
        }

        // Find max attention weight
        let max_weight = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Check for collapse (one token has > 95% attention)
        if max_weight > self.thresholds.attention_collapse_threshold as f32 {
            let event = AnomalyEvent::new(
                AnomalyType::AttentionCollapse,
                self.step,
                layer,
                max_weight as f64,
            )
            .with_head(head)
            .with_threshold(self.thresholds.attention_collapse_threshold)
            .with_context("Attention collapsed to single token");
            self.record(event);
            return Ok(Some(AnomalyType::AttentionCollapse));
        }

        // Check for diffuse attention (uniform distribution)
        let n = values.len() as f32;
        let uniform = 1.0 / n;
        let variance: f32 = values.iter().map(|&v| (v - uniform).powi(2)).sum::<f32>() / n;

        if variance < 1e-6 && n > 10.0 {
            let event = AnomalyEvent::new(
                AnomalyType::AttentionDiffuse,
                self.step,
                layer,
                variance as f64,
            )
            .with_head(head)
            .with_context("Attention is uniformly diffuse");
            self.record(event);
            return Ok(Some(AnomalyType::AttentionDiffuse));
        }

        self.stats.total_checks += 1;
        Ok(None)
    }

    /// Check gradient health across all parameters
    pub fn check_gradient_health(
        &mut self,
        gradients: &std::collections::HashMap<String, Tensor>,
        layer: usize,
    ) -> TorusResult<Vec<AnomalyType>> {
        let mut anomalies = Vec::new();

        for (name, grad) in gradients {
            let flat = grad.flatten_all()?;
            let values: Vec<f32> = flat.to_vec1()?;

            if values.is_empty() {
                continue;
            }

            // Compute gradient norm
            let norm: f64 = values
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt();

            // Check for exploding gradient
            if norm > self.thresholds.max_gradient {
                let event =
                    AnomalyEvent::new(AnomalyType::ExplodingGradient, self.step, layer, norm)
                        .with_threshold(self.thresholds.max_gradient)
                        .with_context(format!("Gradient {} exploding", name));
                self.record(event);
                anomalies.push(AnomalyType::ExplodingGradient);
            }

            // Check for vanishing gradient
            if norm < self.thresholds.min_gradient && norm > 0.0 {
                let event =
                    AnomalyEvent::new(AnomalyType::VanishingGradient, self.step, layer, norm)
                        .with_threshold(self.thresholds.min_gradient)
                        .with_context(format!("Gradient {} vanishing", name));
                self.record(event);
                anomalies.push(AnomalyType::VanishingGradient);
            }

            // Check for NaN in gradients
            let nan_count = values.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                let event = AnomalyEvent::new(AnomalyType::NaN, self.step, layer, nan_count as f64)
                    .with_context(format!("Gradient {} has {} NaN values", name, nan_count));
                self.record(event);
                anomalies.push(AnomalyType::NaN);
            }
        }

        self.stats.total_checks += 1;
        Ok(anomalies)
    }
}

impl Default for AnomalyMonitor {
    fn default() -> Self {
        Self::new(AnomalyThresholds::default())
    }
}

/// Anomaly report
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyReport {
    /// Current step
    pub step: u64,
    /// Statistics
    pub stats: AnomalyStats,
    /// Recent events
    pub recent_events: Vec<AnomalyEvent>,
    /// Overall health
    pub is_healthy: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::particles::{FourMomentum, ParticleFlavor};

    #[test]
    fn test_anomaly_type_severity() {
        assert_eq!(AnomalyType::NaN.severity(), 3);
        assert_eq!(AnomalyType::VanishingGradient.severity(), 2);
        assert_eq!(AnomalyType::AttentionCollapse.severity(), 1);
    }

    #[test]
    fn test_anomaly_event() {
        let event = AnomalyEvent::new(AnomalyType::ExplodingGradient, 100, 3, 9999.9)
            .with_head(2)
            .with_stream(4)
            .with_threshold(1000.0)
            .with_context("test gradient");

        assert_eq!(event.severity(), 3);
        println!("{}", event.summary());
    }

    #[test]
    fn test_tensor_stats() {
        let analyzer = TensorAnalyzer::default();

        // Create a test tensor
        let device = candle_core::Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5], &device).unwrap();

        let stats = analyzer.tensor_stats(&tensor).unwrap();
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert!(stats.is_healthy());
        println!("{}", stats.summary());
    }

    #[test]
    fn test_particle_anomaly_detector() {
        let detector = ParticleAnomalyDetector::default();

        // Normal particle
        let normal = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 3.0, 4.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        assert!(detector.check_particle(&normal).is_empty());

        // Particle with extreme energy
        let extreme = Particle::new(
            1,
            ParticleFlavor::Query,
            FourMomentum::new(1e10, 0.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let anomalies = detector.check_particle(&extreme);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_anomaly_monitor() {
        let mut monitor = AnomalyMonitor::default();

        // Check some normal values
        for _ in 0..10 {
            monitor.next_step();
            monitor.check_value(1.0, 0, "test");
        }
        assert!(monitor.is_healthy(10));

        // Check a NaN
        monitor.check_value(f64::NAN, 0, "nan test");
        assert!(!monitor.is_healthy(1));

        println!("{}", monitor.summary());
    }

    #[test]
    fn test_oscillation_detection() {
        let mut monitor = AnomalyMonitor::new(AnomalyThresholds {
            instability_window: 5,
            instability_threshold: 1.0,
            ..Default::default()
        });

        // Create oscillating pattern
        let values = [10.0, -10.0, 10.0, -10.0, 10.0];
        for (i, &v) in values.iter().enumerate() {
            monitor.step = i as u64;
            monitor.check_value(v, 0, "oscillation test");
        }

        // Should have detected instability
        assert!(monitor
            .events
            .iter()
            .any(|e| e.anomaly_type == AnomalyType::NumericalInstability));
    }

    #[test]
    fn test_gradient_check() {
        let mut monitor = AnomalyMonitor::new(AnomalyThresholds {
            max_gradient: 100.0,
            min_gradient: 1e-5,
            ..Default::default()
        });

        // Normal gradient
        monitor.check_gradient(10.0, 0, "normal");
        assert!(monitor.events.is_empty());

        // Exploding gradient
        monitor.check_gradient(1000.0, 0, "exploding");
        assert_eq!(monitor.events.len(), 1);
        assert_eq!(
            monitor.events[0].anomaly_type,
            AnomalyType::ExplodingGradient
        );

        // Vanishing gradient
        monitor.check_gradient(1e-10, 0, "vanishing");
        assert_eq!(monitor.events.len(), 2);
        assert_eq!(
            monitor.events[1].anomaly_type,
            AnomalyType::VanishingGradient
        );
    }
}
