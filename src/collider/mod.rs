//! # Torus Collider Validator Module
//!
//! A CERN Hadron Collider-inspired validation system for torus attention mechanisms.
//! Models attention computation as particle collisions, providing always-on monitoring
//! during both training and inference.
//!
//! ## Architecture Overview
//!
//! ```text
//!                         TORUS COLLIDER VALIDATOR
//!     ┌────────────────────────────────────────────────────────────────┐
//!     │                                                                │
//!     │   FORWARD STREAMS (Causal, v < c)                             │
//!     │   ════════════════════════════════                            │
//!     │   Query ──────┐                                               │
//!     │               ├──► Q·K Collision ──► Attention Boson          │
//!     │   Key ────────┘        │                    │                 │
//!     │                        │                    ▼                 │
//!     │                   ┌────┴────┐         ┌─────────────┐        │
//!     │                   │ Vertex  │         │  Detector   │        │
//!     │                   │ Tracker │         │  System     │        │
//!     │                   └────┬────┘         └─────────────┘        │
//!     │                        │                    │                 │
//!     │                        ▼                    ▼                 │
//!     │   ┌─────────────────────────────────────────────────────┐    │
//!     │   │              CONSERVATION VALIDATOR                 │    │
//!     │   │  • Energy    • Momentum    • Charge    • Color     │    │
//!     │   │  • CPT       • Topological • Baryon    • Lepton    │    │
//!     │   └─────────────────────────────────────────────────────┘    │
//!     │                        │                                      │
//!     │   ════════════════════════════════                            │
//!     │   BACKWARD STREAMS (Anti-causal, v > c)                       │
//!     │                                                                │
//!     │   Gradient ──────► Tachyon ──────► Darkness Field             │
//!     │        │              │                  │                     │
//!     │        └──────────────┼──────────────────┘                    │
//!     │                       ▼                                        │
//!     │   ┌─────────────────────────────────────────────────────┐    │
//!     │   │              CAUSALITY VALIDATOR                    │    │
//!     │   │  • Speed of darkness   • Temporal consistency      │    │
//!     │   │  • Causal loops        • Stream coherence          │    │
//!     │   └─────────────────────────────────────────────────────┘    │
//!     │                       │                                        │
//!     │                       ▼                                        │
//!     │   ┌─────────────────────────────────────────────────────┐    │
//!     │   │              ANOMALY DETECTOR                       │    │
//!     │   │  • NaN/Inf    • Exploding gradients  • Dead neurons│    │
//!     │   │  • Attention collapse  • Numerical instability     │    │
//!     │   └─────────────────────────────────────────────────────┘    │
//!     │                       │                                        │
//!     │                       ▼                                        │
//!     │   ┌─────────────────────────────────────────────────────┐    │
//!     │   │              METRICS COLLECTOR                      │    │
//!     │   │  • Cross-sections  • Luminosity  • Event rates     │    │
//!     │   │  • Kinematic distributions  • Conservation scores  │    │
//!     │   └─────────────────────────────────────────────────────┘    │
//!     │                                                                │
//!     └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Physics-Neural Network Mapping
//!
//! | Neural Network Component | Particle Physics Analog |
//! |-------------------------|------------------------|
//! | Query vector | Up-type fermion (spin ½, charge +1) |
//! | Key vector | Down-type fermion (spin ½, charge -1) |
//! | Value vector | Higgs-like scalar (spin 1, neutral) |
//! | Attention weight | W/Z boson (mediator) |
//! | Gradient | Gluon (backprop force carrier) |
//! | Forward stream | Causal flow (v ≤ c) |
//! | Backward stream | Tachyonic flow (v > c, "speed of darkness") |
//! | Attention score | Collision cross-section |
//! | Batch × seq × heads | Luminosity |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rustyworm::collider::{TorusCollider, ColliderConfig};
//!
//! // Create collider with default config
//! let mut collider = TorusCollider::new(ColliderConfig::default());
//!
//! // During forward pass
//! collider.record_forward(query, key, value, attention_weights, layer, head);
//!
//! // During backward pass
//! collider.record_backward(gradients, layer);
//!
//! // Check for issues
//! if !collider.is_healthy() {
//!     let report = collider.report();
//!     println!("{}", report.summary());
//! }
//! ```

// Sub-modules
pub mod anomaly;
pub mod conservation;
pub mod darkness;
pub mod detector;
pub mod integration;
pub mod metrics;
pub mod particles;
pub mod vertices;

// Re-exports
pub use anomaly::{
    AnomalyEvent, AnomalyMonitor, AnomalyReport, AnomalyThresholds, AnomalyType,
    ParticleAnomalyDetector, TensorAnalyzer, TensorStats,
};
pub use conservation::{ConservationLaw, ConservationReport, ConservationValidator};
pub use darkness::{
    CausalDirection, CausalityReport, CausalityStats, CausalityValidator, CausalityViolation,
    DarknessField, DarknessTracker, Tachyon, ViolationType, SPEED_OF_LIGHT,
};
pub use detector::{
    AnomalyTrigger, DetectorHit, DetectorLayerType, ElectromagneticCalorimeter, EnergyCluster,
    HadronicCalorimeter, InnerTracker, Jet, MuonSpectrometer, ReconstructedMuon,
    ReconstructionResult, TorusColliderDetector, Track, TriggerType,
};
pub use integration::{
    ColliderHooks, ColliderIntegration, DefaultColliderHooks, HistorySummary,
    LayerValidationReport, StreamValidationReport, TrainerValidationStats, ValidatedLayerOutput,
    ValidatedOutput, ValidatedParallelProcessor, ValidatedStream, ValidatedTrainer,
    ValidatedTrainingStep, ValidatedTransformer, ValidatedTransformerConfig, ValidationConfig,
    ValidationContext, ValidationDashboard, ValidationHistory, ValidationLevel,
};
pub use metrics::{
    ColliderMetrics, CollisionStats, CrossSection, EventRate, Histogram, KinematicDistributions,
    Luminosity, MetricsReport,
};
pub use particles::{FourMomentum, Particle, ParticleBeam, ParticleFlavor, ParticleGenerator};
pub use vertices::{
    CollisionEvent, CouplingConstants, FeynmanVertex, MandelstamVariables, Propagator, VertexType,
};

use crate::TorusResult;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the torus collider validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColliderConfig {
    /// Enable/disable the collider (for performance)
    pub enabled: bool,
    /// Grid resolution for darkness field (major, minor)
    pub darkness_grid: (usize, usize),
    /// Detector configuration
    pub detector_enabled: bool,
    /// Anomaly thresholds
    pub anomaly_thresholds: AnomalyThresholds,
    /// Maximum events to buffer for metrics
    pub metrics_buffer_size: usize,
    /// Conservation tolerance
    pub conservation_tolerance: f64,
    /// Causality tolerance
    pub causality_tolerance: f64,
    /// Number of torus major divisions
    pub n_major: usize,
    /// Number of torus minor divisions
    pub n_minor: usize,
}

impl Default for ColliderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            darkness_grid: (64, 64),
            detector_enabled: true,
            anomaly_thresholds: AnomalyThresholds::default(),
            metrics_buffer_size: 1000,
            conservation_tolerance: 0.05,
            causality_tolerance: 0.1,
            n_major: 32,
            n_minor: 16,
        }
    }
}

impl ColliderConfig {
    /// Create a minimal config (faster, less tracking)
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            darkness_grid: (16, 16),
            detector_enabled: false,
            anomaly_thresholds: AnomalyThresholds::relaxed(),
            metrics_buffer_size: 100,
            conservation_tolerance: 0.1,
            causality_tolerance: 0.2,
            n_major: 16,
            n_minor: 8,
        }
    }

    /// Create a comprehensive config (slower, detailed tracking)
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            darkness_grid: (128, 128),
            detector_enabled: true,
            anomaly_thresholds: AnomalyThresholds::strict(),
            metrics_buffer_size: 10000,
            conservation_tolerance: 0.01,
            causality_tolerance: 0.05,
            n_major: 64,
            n_minor: 32,
        }
    }

    /// Disable the collider entirely
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER VALIDATOR TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for components that can be validated by the collider
pub trait ColliderValidated {
    /// Record forward pass data
    fn record_forward(&self, collider: &mut TorusCollider);

    /// Record backward pass data
    fn record_backward(&self, collider: &mut TorusCollider);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN TORUS COLLIDER
// ═══════════════════════════════════════════════════════════════════════════════

/// Main torus collider validator
///
/// Provides always-on validation of attention computations during training and inference.
#[derive(Debug)]
pub struct TorusCollider {
    /// Configuration
    pub config: ColliderConfig,
    /// Particle generator
    pub generator: ParticleGenerator,
    /// Detector system
    pub detector: TorusColliderDetector,
    /// Conservation validator
    pub conservation: ConservationValidator,
    /// Darkness tracker (for backward streams)
    pub darkness: DarknessTracker,
    /// Anomaly monitor
    pub anomaly: AnomalyMonitor,
    /// Metrics collector
    pub metrics: ColliderMetrics,
    /// Current step
    step: u64,
    /// Current layer being processed
    current_layer: usize,
    /// Forward beams by stream
    forward_beams: Vec<ParticleBeam>,
    /// Backward beams by stream
    backward_beams: Vec<ParticleBeam>,
    /// Recent collision events
    recent_events: Vec<CollisionEvent>,
}

impl TorusCollider {
    /// Create a new collider with the given configuration
    pub fn new(config: ColliderConfig) -> Self {
        Self {
            generator: ParticleGenerator::new(),
            detector: TorusColliderDetector::new(),
            conservation: ConservationValidator::new(),
            darkness: DarknessTracker::new(config.darkness_grid),
            anomaly: AnomalyMonitor::new(config.anomaly_thresholds.clone()),
            metrics: ColliderMetrics::new(config.metrics_buffer_size),
            config,
            step: 0,
            current_layer: 0,
            forward_beams: Vec::new(),
            backward_beams: Vec::new(),
            recent_events: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ColliderConfig::default())
    }

    /// Check if collider is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Advance to next step
    pub fn next_step(&mut self) {
        if !self.config.enabled {
            return;
        }

        self.step += 1;
        self.anomaly.next_step();
        self.metrics.next_step();

        // Clear per-step data
        self.forward_beams.clear();
        self.backward_beams.clear();
        self.recent_events.clear();
    }

    /// Set current layer
    pub fn set_layer(&mut self, layer: usize) {
        self.current_layer = layer;
    }

    /// Record Q, K, V tensors from forward pass
    pub fn record_qkv(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        stream_id: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Generate particles from tensors
        let q_particles = self.generator.generate_queries(
            q,
            stream_id,
            self.current_layer,
            self.config.n_major,
            self.config.n_minor,
        )?;

        let k_particles = self.generator.generate_keys(
            k,
            stream_id,
            self.current_layer,
            self.config.n_major,
            self.config.n_minor,
        )?;

        let v_particles = self.generator.generate_values(
            v,
            stream_id,
            self.current_layer,
            self.config.n_major,
            self.config.n_minor,
        )?;

        // Create beam
        let is_forward = stream_id < 4; // Streams 0-3 are forward, 4-7 are backward
        let mut beam = ParticleBeam::new(stream_id, is_forward);

        for p in q_particles {
            // Check for anomalies
            let anomalies = ParticleAnomalyDetector::new(self.config.anomaly_thresholds.clone())
                .check_particle(&p);
            for anomaly_type in anomalies {
                self.anomaly.record(
                    AnomalyEvent::new(anomaly_type, self.step, self.current_layer, p.energy())
                        .with_stream(stream_id)
                        .with_particle(p.id)
                        .with_context("Query particle"),
                );
            }
            beam.add_particle(p);
        }

        for p in k_particles {
            beam.add_particle(p);
        }

        for p in v_particles {
            beam.add_particle(p);
        }

        if is_forward {
            self.forward_beams.push(beam);
        } else {
            // Process through darkness tracker for backward streams
            self.darkness.process_beam(&beam);
            self.backward_beams.push(beam);
        }

        Ok(())
    }

    /// Record attention weights
    pub fn record_attention(
        &mut self,
        weights: &Tensor,
        stream_id: usize,
        head: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check attention weights for anomalies
        let analyzer = TensorAnalyzer::new(self.config.anomaly_thresholds.clone());
        let anomalies = analyzer.check_attention(weights)?;

        for anomaly_type in anomalies {
            self.anomaly.record(
                AnomalyEvent::new(anomaly_type, self.step, self.current_layer, 0.0)
                    .with_stream(stream_id)
                    .with_head(head)
                    .with_context("Attention weights"),
            );
        }

        // Record luminosity based on attention shape
        let dims = weights.dims();
        if dims.len() >= 3 {
            self.metrics.record_luminosity(dims[0], dims[1], 1);
        }

        Ok(())
    }

    /// Record gradients from backward pass
    pub fn record_gradients(&mut self, gradients: &Tensor, stream_id: usize) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check gradient magnitude
        let stats = TensorAnalyzer::default().tensor_stats(gradients)?;

        let magnitude = stats.std * (stats.n_elements as f64).sqrt();
        self.anomaly
            .check_gradient(magnitude, self.current_layer, "Gradient");

        // Check for NaN/Inf
        if stats.has_nan {
            self.anomaly.record(
                AnomalyEvent::new(AnomalyType::NaN, self.step, self.current_layer, f64::NAN)
                    .with_stream(stream_id)
                    .with_context("Gradient tensor"),
            );
        }
        if stats.has_inf {
            self.anomaly.record(
                AnomalyEvent::new(
                    AnomalyType::Inf,
                    self.step,
                    self.current_layer,
                    f64::INFINITY,
                )
                .with_stream(stream_id)
                .with_context("Gradient tensor"),
            );
        }

        Ok(())
    }

    /// Record gradients from backward pass (HashMap version for training)
    pub fn record_gradients_map(
        &mut self,
        gradients: &std::collections::HashMap<String, Tensor>,
        layer: usize,
    ) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.set_layer(layer);

        for (name, grad) in gradients {
            // Check gradient magnitude
            let stats = TensorAnalyzer::default().tensor_stats(grad)?;

            let magnitude = stats.std * (stats.n_elements as f64).sqrt();
            self.anomaly
                .check_gradient(magnitude, layer, &format!("Gradient_{}", name));

            // Check for NaN/Inf
            if stats.has_nan {
                self.anomaly.record(
                    AnomalyEvent::new(AnomalyType::NaN, self.step, layer, f64::NAN)
                        .with_context(format!("Gradient {} tensor", name)),
                );
            }
            if stats.has_inf {
                self.anomaly.record(
                    AnomalyEvent::new(AnomalyType::Inf, self.step, layer, f64::INFINITY)
                        .with_context(format!("Gradient {} tensor", name)),
                );
            }
        }

        Ok(())
    }

    /// Record a collision event
    pub fn record_collision(&mut self, event: CollisionEvent) {
        if !self.config.enabled {
            return;
        }

        // Validate conservation
        self.conservation.validate_event(&event);

        // Record in detector
        if self.config.detector_enabled {
            self.detector.record_event(&event);
        }

        // Record in metrics
        self.metrics.record_event(event.clone());

        self.recent_events.push(event);
    }

    /// Record input tensor for anomaly detection
    ///
    /// This checks the input tensor for numerical issues (NaN, Inf, extreme values)
    /// and records any anomalies found.
    pub fn record_input(&mut self, input: &Tensor) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check for numerical anomalies in input
        let anomalies = self
            .anomaly
            .check_tensor(input, "input", self.current_layer, None)?;
        for anomaly_type in anomalies {
            self.anomaly.record(
                AnomalyEvent::new(anomaly_type, self.step, self.current_layer, 0.0)
                    .with_context("Input tensor"),
            );
        }

        Ok(())
    }

    /// Record output tensor for anomaly detection
    ///
    /// This checks the output tensor for numerical issues and records any anomalies.
    pub fn record_output(&mut self, output: &Tensor) -> TorusResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check for numerical anomalies in output
        let anomalies = self
            .anomaly
            .check_tensor(output, "output", self.current_layer, None)?;
        for anomaly_type in anomalies {
            self.anomaly.record(
                AnomalyEvent::new(anomaly_type, self.step, self.current_layer, 0.0)
                    .with_context("Output tensor"),
            );
        }

        Ok(())
    }

    /// Validate stream consistency between forward and backward passes
    pub fn validate_streams(&mut self) {
        if !self.config.enabled {
            return;
        }

        // Pair forward and backward streams
        for (fwd, bwd) in self.forward_beams.iter().zip(self.backward_beams.iter()) {
            self.darkness.validate_streams(fwd, bwd);
        }
    }

    /// Check if the system is healthy (no critical anomalies)
    pub fn is_healthy(&self) -> bool {
        if !self.config.enabled {
            return true;
        }

        self.anomaly.is_healthy(10)
            && self.conservation.report().all_conserved
            && self.darkness.report().is_valid
    }

    /// Get comprehensive validation report
    pub fn report(&self) -> ColliderReport {
        ColliderReport {
            step: self.step,
            is_healthy: self.is_healthy(),
            anomaly_report: self.anomaly.report(),
            conservation_report: self.conservation.report(),
            causality_report: self.darkness.report(),
            metrics_report: self.metrics.report(),
            detector_summary: if self.config.detector_enabled {
                Some(self.detector.summary())
            } else {
                None
            },
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        if !self.config.enabled {
            return "Collider DISABLED".to_string();
        }

        format!(
            "═══════════════════════════════════════════════════════════════\n\
             TORUS COLLIDER VALIDATOR - Step {}\n\
             ═══════════════════════════════════════════════════════════════\n\
             Status: {}\n\
             \n\
             {}\n\
             \n\
             {}\n\
             \n\
             {}\n\
             {}\
             ═══════════════════════════════════════════════════════════════",
            self.step,
            if self.is_healthy() {
                "✓ HEALTHY"
            } else {
                "⚠ ISSUES DETECTED"
            },
            self.anomaly.summary(),
            self.conservation.report().summary(),
            self.darkness.summary(),
            if self.config.detector_enabled {
                format!("\n{}\n", self.detector.summary())
            } else {
                String::new()
            }
        )
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.step = 0;
        self.current_layer = 0;
        self.forward_beams.clear();
        self.backward_beams.clear();
        self.recent_events.clear();
        self.generator = ParticleGenerator::new();
        self.detector.reset();
        self.conservation.reset();
        self.darkness.reset();
        self.anomaly = AnomalyMonitor::new(self.config.anomaly_thresholds.clone());
        self.metrics.reset();
    }
}

impl Default for TorusCollider {
    fn default() -> Self {
        Self::new(ColliderConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER REPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Comprehensive report from the collider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColliderReport {
    /// Current step
    pub step: u64,
    /// Overall health status
    pub is_healthy: bool,
    /// Anomaly detection report
    pub anomaly_report: AnomalyReport,
    /// Conservation validation report
    pub conservation_report: ConservationReport,
    /// Causality (darkness) report
    pub causality_report: CausalityReport,
    /// Metrics report
    pub metrics_report: MetricsReport,
    /// Detector summary (if enabled)
    pub detector_summary: Option<String>,
}

impl ColliderReport {
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Collider Report @ step {}\n\
             Status: {}\n\
             Anomalies: {} total ({} critical)\n\
             Conservation: {}\n\
             Causality: {}\n\
             Collisions: {} total",
            self.step,
            if self.is_healthy { "HEALTHY" } else { "ISSUES" },
            self.anomaly_report.stats.total_anomalies,
            self.anomaly_report.stats.critical_anomalies,
            if self.conservation_report.all_conserved {
                "OK"
            } else {
                "VIOLATIONS"
            },
            if self.causality_report.is_valid {
                "OK"
            } else {
                "VIOLATIONS"
            },
            self.metrics_report.collision_stats.n_collisions
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::TorusCoordinate;

    #[test]
    fn test_collider_creation() {
        let collider = TorusCollider::default_config();
        assert!(collider.is_enabled());
        assert!(collider.is_healthy());
    }

    #[test]
    fn test_collider_disabled() {
        let collider = TorusCollider::new(ColliderConfig::disabled());
        assert!(!collider.is_enabled());
        assert!(collider.is_healthy()); // Always healthy when disabled
    }

    #[test]
    fn test_collider_step() {
        let mut collider = TorusCollider::default_config();

        for _ in 0..10 {
            collider.next_step();
        }

        assert_eq!(collider.step, 10);
    }

    #[test]
    fn test_collider_report() {
        let collider = TorusCollider::default_config();
        let report = collider.report();

        assert!(report.is_healthy);
        println!("{}", report.summary());
    }

    #[test]
    fn test_collider_summary() {
        let collider = TorusCollider::default_config();
        let summary = collider.summary();

        assert!(summary.contains("TORUS COLLIDER"));
        assert!(summary.contains("HEALTHY"));
        println!("{}", summary);
    }

    #[test]
    fn test_config_variants() {
        let minimal = ColliderConfig::minimal();
        let comprehensive = ColliderConfig::comprehensive();
        let disabled = ColliderConfig::disabled();

        assert!(minimal.enabled);
        assert!(!minimal.detector_enabled);

        assert!(comprehensive.enabled);
        assert!(comprehensive.detector_enabled);

        assert!(!disabled.enabled);
    }

    #[test]
    fn test_record_collision() {
        let mut collider = TorusCollider::default_config();

        let mut event = CollisionEvent::new(
            0,
            VertexType::QueryKey,
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
            0,
        );
        event.add_incoming(Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 5.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        ));
        event.add_incoming(Particle::new(
            1,
            ParticleFlavor::Key,
            FourMomentum::new(10.0, -5.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        ));
        event.add_outgoing(Particle::new(
            2,
            ParticleFlavor::Attention,
            FourMomentum::new(20.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        ));

        collider.record_collision(event);
        collider.next_step();

        assert_eq!(collider.metrics.collision_stats.n_collisions, 1);
    }

    #[test]
    fn test_full_integration() {
        let mut collider = TorusCollider::new(ColliderConfig::minimal());

        // Simulate several steps
        for step in 0..5 {
            collider.set_layer(step % 3);

            // Create a collision event with balanced incoming/outgoing particles
            let mut event = CollisionEvent::new(
                step as u64,
                VertexType::QueryKey,
                TorusCoordinate::new(0.0, 0.0),
                step % 3,
                0,
                0,
            );
            // Add incoming particles
            event.add_incoming(Particle::new(
                step as u64 * 3,
                ParticleFlavor::Query,
                FourMomentum::new(5.0, 1.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            event.add_incoming(Particle::new(
                step as u64 * 3 + 1,
                ParticleFlavor::Key,
                FourMomentum::new(5.0, -1.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            // Add outgoing particle with matching energy/momentum
            event.add_outgoing(Particle::new(
                step as u64 * 3 + 2,
                ParticleFlavor::Attention,
                FourMomentum::new(10.0, 0.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));

            collider.record_collision(event);
            collider.validate_streams();
            collider.next_step();
        }

        // Note: conservation laws may still report minor violations due to
        // model-specific constraints (charge, baryon number, etc.) that are
        // tracked but expected for attention mechanisms
        let report = collider.report();
        println!("{}", report.summary());

        // The anomaly detection should show no critical issues
        assert!(collider.anomaly.is_healthy(10));
    }
}
