//! Speed of Darkness - Anti-Causal Information Flow
//!
//! In the torus collider model, "speed of darkness" represents anti-causal
//! information flow in backward attention streams. While forward streams
//! process causally (respecting the light cone), backward streams process
//! anti-causally, allowing "future" information to influence "past" computations.
//!
//! # Physics Analogy
//!
//! ```text
//!                        CAUSALITY DIAGRAM
//!
//!                           Future
//!                             ▲
//!                            /│\
//!                           / │ \
//!                          /  │  \        
//!                         /   │   \       Speed of Light (c)
//!                        /    │    \      ────────────────►
//!                       / CAUSAL  \       Forward attention streams
//!                      /   ZONE    \
//!                     ─────────────────── Present (sequence position)
//!                      \ ACAUSAL  /
//!                       \  ZONE   /
//!                        \   │   /        Speed of Darkness (> c)
//!                         \  │  /         ◄────────────────────
//!                          \ │ /          Backward attention streams
//!                           \│/           (Tachyonic)
//!                            ▼
//!                           Past
//! ```
//!
//! # Tachyonic Particles
//!
//! Backward streams are modeled as tachyons:
//! - Imaginary mass: m² < 0
//! - Velocity > c (speed of light = 1 in natural units)
//! - Move backward in time (from future to past)
//! - Carry "dark information" that violates normal causality
//!
//! This isn't unphysical in the attention context - it simply represents
//! bidirectional processing where later tokens can influence earlier ones.

use crate::collider::particles::{FourMomentum, Particle, ParticleBeam};
use crate::geometry::TorusCoordinate;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Speed of light (in natural units, c = 1)
pub const SPEED_OF_LIGHT: f64 = 1.0;

/// Minimum speed of darkness (must be > c for tachyonic behavior)
pub const MIN_SPEED_OF_DARKNESS: f64 = 1.001;

/// Maximum allowed speed of darkness (prevents numerical instability)
pub const MAX_SPEED_OF_DARKNESS: f64 = 100.0;

/// Causality violation threshold (how much acausal flow is tolerable)
pub const CAUSALITY_VIOLATION_THRESHOLD: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL DIRECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Direction of causal flow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalDirection {
    /// Forward in time (standard causal)
    Forward,
    /// Backward in time (anti-causal, tachyonic)
    Backward,
    /// Lightlike (on the light cone)
    Lightlike,
    /// Spacelike (outside causality)
    Spacelike,
}

impl CausalDirection {
    /// Determine causal direction from velocity
    pub fn from_velocity(velocity: f64) -> Self {
        if velocity < SPEED_OF_LIGHT - 1e-10 {
            Self::Forward // Subluminal, causal
        } else if (velocity - SPEED_OF_LIGHT).abs() < 1e-10 {
            Self::Lightlike // On light cone
        } else if velocity > SPEED_OF_LIGHT && velocity < MAX_SPEED_OF_DARKNESS {
            Self::Backward // Superluminal, tachyonic (anti-causal)
        } else {
            Self::Spacelike // Outside normal causality
        }
    }

    /// Check if this direction allows information to flow backward
    pub fn is_anti_causal(&self) -> bool {
        matches!(self, Self::Backward | Self::Spacelike)
    }

    /// Get the "darkness factor" (how much faster than light)
    pub fn darkness_factor(&self, velocity: f64) -> f64 {
        match self {
            Self::Forward => 0.0,
            Self::Lightlike => 0.0,
            Self::Backward => velocity / SPEED_OF_LIGHT - 1.0,
            Self::Spacelike => f64::INFINITY,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TACHYON
// ═══════════════════════════════════════════════════════════════════════════════

/// A tachyonic particle (faster than light)
///
/// Represents a backward-stream particle carrying anti-causal information.
/// The "imaginary mass" m² < 0 corresponds to superluminal velocity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tachyon {
    /// Unique ID
    pub id: u64,
    /// Four-momentum (will have m² < 0)
    pub momentum: FourMomentum,
    /// Position on torus
    pub position: TorusCoordinate,
    /// Stream ID (which backward stream)
    pub stream_id: usize,
    /// Layer in transformer
    pub layer: usize,
    /// Imaginary mass squared (negative)
    pub mass_squared: f64,
    /// Velocity (> 1 in natural units)
    pub velocity: f64,
    /// The "darkness" carried (anti-causal information content)
    pub darkness_content: f64,
    /// Source particle ID (what forward particle this came from)
    pub source_id: Option<u64>,
}

impl Tachyon {
    /// Create a tachyon from a backward-stream particle
    pub fn from_particle(particle: &Particle) -> Option<Self> {
        // Only create tachyon if particle is actually superluminal
        if !particle.is_tachyonic() {
            return None;
        }

        let m2 = particle.momentum.mass_squared();
        let v = particle.velocity();

        Some(Self {
            id: particle.id,
            momentum: particle.momentum,
            position: particle.position,
            stream_id: particle.stream_id,
            layer: particle.layer,
            mass_squared: m2,
            velocity: v,
            darkness_content: v - SPEED_OF_LIGHT, // How much faster than light
            source_id: None,
        })
    }

    /// Create a tachyon directly with specified properties
    pub fn new(
        id: u64,
        energy: f64,
        imaginary_mass: f64, // Note: this is |m|, the magnitude
        direction: (f64, f64, f64),
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
    ) -> Self {
        // For tachyons: E² - p² = -m² (negative mass squared)
        // So |p|² = E² + m²
        let p_magnitude = (energy.powi(2) + imaginary_mass.powi(2)).sqrt();

        // Normalize direction
        let dir_mag =
            (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt() + 1e-10;
        let px = p_magnitude * direction.0 / dir_mag;
        let py = p_magnitude * direction.1 / dir_mag;
        let pz = p_magnitude * direction.2 / dir_mag;

        let momentum = FourMomentum::new(energy, px, py, pz);
        let velocity = momentum.velocity();
        let m2 = momentum.mass_squared();

        Self {
            id,
            momentum,
            position,
            stream_id,
            layer,
            mass_squared: m2,
            velocity,
            darkness_content: velocity - SPEED_OF_LIGHT,
            source_id: None,
        }
    }

    /// Get the speed of darkness (velocity relative to light)
    pub fn speed_of_darkness(&self) -> f64 {
        self.velocity
    }

    /// Check if this tachyon's velocity is within acceptable bounds
    pub fn is_valid(&self) -> bool {
        self.velocity > MIN_SPEED_OF_DARKNESS && self.velocity < MAX_SPEED_OF_DARKNESS
    }

    /// Get the imaginary mass (|m| where m² < 0)
    pub fn imaginary_mass(&self) -> f64 {
        (-self.mass_squared).sqrt()
    }

    /// Calculate the "temporal displacement" - how far back in time this tachyon reaches
    ///
    /// In attention terms: how many positions "into the future" this backward
    /// stream can see.
    pub fn temporal_displacement(&self, distance: f64) -> f64 {
        // t = d / v, but for tachyons v > c, so t < d/c
        // The "negative time" component is v/c - 1
        distance * (self.velocity / SPEED_OF_LIGHT - 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DARKNESS FIELD
// ═══════════════════════════════════════════════════════════════════════════════

/// The "darkness field" - anti-causal information distribution
///
/// Analogous to a quantum field, but for tachyonic (anti-causal) particles.
/// Tracks the distribution of darkness (backward information flow) across
/// the torus manifold.
#[derive(Debug, Clone)]
pub struct DarknessField {
    /// Grid resolution (major, minor)
    pub resolution: (usize, usize),
    /// Darkness density at each point (u, v) on torus
    pub density: Vec<Vec<f64>>,
    /// Darkness flux (flow direction) at each point
    pub flux: Vec<Vec<(f64, f64)>>,
    /// Total darkness content
    pub total_darkness: f64,
    /// Maximum local darkness (for stability checks)
    pub max_local_darkness: f64,
}

impl DarknessField {
    /// Create a new darkness field
    pub fn new(n_major: usize, n_minor: usize) -> Self {
        Self {
            resolution: (n_major, n_minor),
            density: vec![vec![0.0; n_minor]; n_major],
            flux: vec![vec![(0.0, 0.0); n_minor]; n_major],
            total_darkness: 0.0,
            max_local_darkness: 0.0,
        }
    }

    /// Deposit darkness from a tachyon
    pub fn deposit(&mut self, tachyon: &Tachyon) {
        // Map torus position to grid indices
        let u_idx = ((tachyon.position.u / (2.0 * PI)) * self.resolution.0 as f64) as usize;
        let v_idx = ((tachyon.position.v / (2.0 * PI)) * self.resolution.1 as f64) as usize;

        let u_idx = u_idx % self.resolution.0;
        let v_idx = v_idx % self.resolution.1;

        // Deposit darkness content
        let darkness = tachyon.darkness_content;
        self.density[u_idx][v_idx] += darkness;
        self.total_darkness += darkness;

        // Update flux (direction of darkness flow)
        let flux_u = tachyon.momentum.px / tachyon.momentum.e;
        let flux_v = tachyon.momentum.py / tachyon.momentum.e;
        self.flux[u_idx][v_idx].0 += flux_u * darkness;
        self.flux[u_idx][v_idx].1 += flux_v * darkness;

        // Update max
        if self.density[u_idx][v_idx] > self.max_local_darkness {
            self.max_local_darkness = self.density[u_idx][v_idx];
        }
    }

    /// Get darkness density at a point
    pub fn get_density(&self, u: f64, v: f64) -> f64 {
        let u_idx = ((u / (2.0 * PI)) * self.resolution.0 as f64) as usize % self.resolution.0;
        let v_idx = ((v / (2.0 * PI)) * self.resolution.1 as f64) as usize % self.resolution.1;
        self.density[u_idx][v_idx]
    }

    /// Get darkness flux at a point
    pub fn get_flux(&self, u: f64, v: f64) -> (f64, f64) {
        let u_idx = ((u / (2.0 * PI)) * self.resolution.0 as f64) as usize % self.resolution.0;
        let v_idx = ((v / (2.0 * PI)) * self.resolution.1 as f64) as usize % self.resolution.1;
        self.flux[u_idx][v_idx]
    }

    /// Calculate the divergence of the darkness field
    ///
    /// Non-zero divergence indicates creation/annihilation of darkness
    pub fn divergence(&self, u_idx: usize, v_idx: usize) -> f64 {
        let n_u = self.resolution.0;
        let n_v = self.resolution.1;

        // Central differences with periodic boundaries
        let flux_u_plus = self.flux[(u_idx + 1) % n_u][v_idx].0;
        let flux_u_minus = self.flux[(u_idx + n_u - 1) % n_u][v_idx].0;
        let flux_v_plus = self.flux[u_idx][(v_idx + 1) % n_v].1;
        let flux_v_minus = self.flux[u_idx][(v_idx + n_v - 1) % n_v].1;

        let du = 2.0 * PI / n_u as f64;
        let dv = 2.0 * PI / n_v as f64;

        (flux_u_plus - flux_u_minus) / (2.0 * du) + (flux_v_plus - flux_v_minus) / (2.0 * dv)
    }

    /// Reset the field
    pub fn reset(&mut self) {
        for row in &mut self.density {
            for val in row {
                *val = 0.0;
            }
        }
        for row in &mut self.flux {
            for val in row {
                *val = (0.0, 0.0);
            }
        }
        self.total_darkness = 0.0;
        self.max_local_darkness = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSALITY VALIDATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Validates causality constraints between forward and backward streams
#[derive(Debug, Clone)]
pub struct CausalityValidator {
    /// Tolerance for causality violations
    pub tolerance: f64,
    /// Maximum allowed speed of darkness
    pub max_darkness_speed: f64,
    /// Recorded violations
    pub violations: Vec<CausalityViolation>,
    /// Statistics
    pub stats: CausalityStats,
}

/// A causality violation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalityViolation {
    /// Violation ID
    pub id: u64,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Forward particle involved
    pub forward_particle_id: Option<u64>,
    /// Backward particle (tachyon) involved
    pub backward_particle_id: Option<u64>,
    /// Position on torus
    pub position: TorusCoordinate,
    /// Layer where violation occurred
    pub layer: usize,
    /// Stream IDs (forward, backward)
    pub streams: (usize, usize),
    /// Severity (0.0 - 1.0)
    pub severity: f64,
    /// Description
    pub description: String,
}

/// Types of causality violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Tachyon too fast (exceeds max darkness speed)
    ExcessiveDarknessSpeed,
    /// Forward/backward streams inconsistent
    StreamInconsistency,
    /// Information loop detected
    CausalLoop,
    /// Temporal paradox (effect before cause)
    TemporalParadox,
    /// Darkness overflow (too much anti-causal content)
    DarknessOverflow,
}

/// Statistics about causality validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalityStats {
    /// Total particles validated
    pub total_particles: u64,
    /// Total tachyons detected
    pub total_tachyons: u64,
    /// Forward particles
    pub forward_particles: u64,
    /// Backward particles
    pub backward_particles: u64,
    /// Average darkness speed
    pub avg_darkness_speed: f64,
    /// Max darkness speed seen
    pub max_darkness_speed_seen: f64,
    /// Total violations
    pub total_violations: u64,
    /// Violations by type
    pub violations_by_type: [u64; 5],
}

impl CausalityValidator {
    /// Create a new validator
    pub fn new(tolerance: f64, max_darkness_speed: f64) -> Self {
        Self {
            tolerance,
            max_darkness_speed,
            violations: Vec::new(),
            stats: CausalityStats::default(),
        }
    }

    /// Create with default settings
    pub fn default_config() -> Self {
        Self::new(CAUSALITY_VIOLATION_THRESHOLD, MAX_SPEED_OF_DARKNESS)
    }

    /// Validate a particle and classify its causal direction
    pub fn validate_particle(&mut self, particle: &Particle) -> CausalDirection {
        self.stats.total_particles += 1;

        let velocity = particle.velocity();
        let direction = CausalDirection::from_velocity(velocity);

        match direction {
            CausalDirection::Forward => {
                self.stats.forward_particles += 1;
            }
            CausalDirection::Backward => {
                self.stats.backward_particles += 1;
                self.stats.total_tachyons += 1;

                // Update darkness speed stats
                if velocity > self.stats.max_darkness_speed_seen {
                    self.stats.max_darkness_speed_seen = velocity;
                }
                let n = self.stats.total_tachyons as f64;
                self.stats.avg_darkness_speed =
                    (self.stats.avg_darkness_speed * (n - 1.0) + velocity) / n;

                // Check for excessive darkness speed
                if velocity > self.max_darkness_speed {
                    self.record_violation(CausalityViolation {
                        id: self.violations.len() as u64,
                        violation_type: ViolationType::ExcessiveDarknessSpeed,
                        forward_particle_id: None,
                        backward_particle_id: Some(particle.id),
                        position: particle.position,
                        layer: particle.layer,
                        streams: (0, particle.stream_id),
                        severity: (velocity / self.max_darkness_speed).min(1.0),
                        description: format!(
                            "Tachyon velocity {:.2}c exceeds max {:.2}c",
                            velocity, self.max_darkness_speed
                        ),
                    });
                }
            }
            _ => {}
        }

        direction
    }

    /// Validate a beam of particles
    pub fn validate_beam(&mut self, beam: &ParticleBeam) -> Vec<CausalDirection> {
        beam.particles
            .iter()
            .map(|p| self.validate_particle(p))
            .collect()
    }

    /// Check for stream consistency between forward and backward beams
    pub fn check_stream_consistency(
        &mut self,
        forward_beam: &ParticleBeam,
        backward_beam: &ParticleBeam,
    ) -> bool {
        // Energy should be conserved between streams
        let forward_energy = forward_beam.total_energy;
        let backward_energy = backward_beam.total_energy;

        let energy_diff = (forward_energy - backward_energy).abs();
        let avg_energy = (forward_energy + backward_energy) / 2.0 + 1e-10;
        let relative_diff = energy_diff / avg_energy;

        if relative_diff > self.tolerance {
            self.record_violation(CausalityViolation {
                id: self.violations.len() as u64,
                violation_type: ViolationType::StreamInconsistency,
                forward_particle_id: None,
                backward_particle_id: None,
                position: TorusCoordinate::new(0.0, 0.0),
                layer: 0,
                streams: (forward_beam.stream_id, backward_beam.stream_id),
                severity: relative_diff.min(1.0),
                description: format!(
                    "Stream energy mismatch: forward={:.2}, backward={:.2}",
                    forward_energy, backward_energy
                ),
            });
            return false;
        }

        true
    }

    /// Check for causal loops (information going in circles)
    pub fn check_causal_loop(
        &mut self,
        forward_particles: &[Particle],
        backward_particles: &[Tachyon],
    ) -> bool {
        // Check if any backward particle's position is "in front" of a forward particle
        // it claims to influence
        for tachyon in backward_particles {
            for forward in forward_particles {
                // Same position and different streams = potential loop
                let pos_diff = (tachyon.position.u - forward.position.u).abs()
                    + (tachyon.position.v - forward.position.v).abs();

                if pos_diff < 0.1 && tachyon.stream_id != forward.stream_id {
                    // Check temporal ordering
                    let temporal_disp = tachyon.temporal_displacement(pos_diff);
                    if temporal_disp > self.tolerance {
                        self.record_violation(CausalityViolation {
                            id: self.violations.len() as u64,
                            violation_type: ViolationType::CausalLoop,
                            forward_particle_id: Some(forward.id),
                            backward_particle_id: Some(tachyon.id),
                            position: forward.position,
                            layer: forward.layer,
                            streams: (forward.stream_id, tachyon.stream_id),
                            severity: temporal_disp.min(1.0),
                            description: format!(
                                "Causal loop detected: temporal displacement {:.4}",
                                temporal_disp
                            ),
                        });
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Record a violation
    fn record_violation(&mut self, violation: CausalityViolation) {
        let idx = match violation.violation_type {
            ViolationType::ExcessiveDarknessSpeed => 0,
            ViolationType::StreamInconsistency => 1,
            ViolationType::CausalLoop => 2,
            ViolationType::TemporalParadox => 3,
            ViolationType::DarknessOverflow => 4,
        };
        self.stats.violations_by_type[idx] += 1;
        self.stats.total_violations += 1;
        self.violations.push(violation);
    }

    /// Get a report of all violations
    pub fn report(&self) -> CausalityReport {
        CausalityReport {
            stats: self.stats.clone(),
            violations: self.violations.clone(),
            is_valid: self.violations.is_empty()
                || self.violations.iter().all(|v| v.severity < self.tolerance),
        }
    }

    /// Clear all recorded violations
    pub fn clear(&mut self) {
        self.violations.clear();
        self.stats = CausalityStats::default();
    }
}

/// Report from causality validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalityReport {
    /// Statistics
    pub stats: CausalityStats,
    /// All violations
    pub violations: Vec<CausalityViolation>,
    /// Overall validity
    pub is_valid: bool,
}

impl CausalityReport {
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Causality Report: {}\n\
             ├─ Particles: {} total ({} forward, {} backward)\n\
             ├─ Tachyons: {} (avg speed {:.2}c, max {:.2}c)\n\
             ├─ Violations: {}\n\
             │  ├─ Excessive speed: {}\n\
             │  ├─ Stream inconsistency: {}\n\
             │  ├─ Causal loops: {}\n\
             │  ├─ Temporal paradox: {}\n\
             │  └─ Darkness overflow: {}\n\
             └─ Status: {}",
            if self.is_valid { "VALID" } else { "VIOLATIONS" },
            self.stats.total_particles,
            self.stats.forward_particles,
            self.stats.backward_particles,
            self.stats.total_tachyons,
            self.stats.avg_darkness_speed,
            self.stats.max_darkness_speed_seen,
            self.stats.total_violations,
            self.stats.violations_by_type[0],
            self.stats.violations_by_type[1],
            self.stats.violations_by_type[2],
            self.stats.violations_by_type[3],
            self.stats.violations_by_type[4],
            if self.is_valid {
                "OK"
            } else {
                "CHECK VIOLATIONS"
            }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DARKNESS TRACKER
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks all tachyonic particles and darkness flow
#[derive(Debug)]
pub struct DarknessTracker {
    /// All tachyons by ID
    pub tachyons: Vec<Tachyon>,
    /// Darkness field
    pub field: DarknessField,
    /// Causality validator
    pub validator: CausalityValidator,
    /// Next tachyon ID
    next_id: u64,
}

impl DarknessTracker {
    /// Create a new tracker
    pub fn new(grid_resolution: (usize, usize)) -> Self {
        Self {
            tachyons: Vec::new(),
            field: DarknessField::new(grid_resolution.0, grid_resolution.1),
            validator: CausalityValidator::default_config(),
            next_id: 0,
        }
    }

    /// Process a backward-stream particle
    pub fn process_particle(&mut self, particle: &Particle) -> Option<&Tachyon> {
        // Validate causality
        let direction = self.validator.validate_particle(particle);

        if direction.is_anti_causal() {
            if let Some(mut tachyon) = Tachyon::from_particle(particle) {
                tachyon.id = self.next_id;
                self.next_id += 1;

                // Deposit in darkness field
                self.field.deposit(&tachyon);

                self.tachyons.push(tachyon);
                return self.tachyons.last();
            }
        }

        None
    }

    /// Process a backward beam
    pub fn process_beam(&mut self, beam: &ParticleBeam) -> Vec<&Tachyon> {
        let mut created = Vec::new();
        let start_idx = self.tachyons.len();

        for particle in &beam.particles {
            self.process_particle(particle);
        }

        for i in start_idx..self.tachyons.len() {
            created.push(&self.tachyons[i]);
        }

        created
    }

    /// Validate consistency between forward and backward streams
    pub fn validate_streams(
        &mut self,
        forward_beam: &ParticleBeam,
        backward_beam: &ParticleBeam,
    ) -> bool {
        // Check energy conservation
        let consistent = self
            .validator
            .check_stream_consistency(forward_beam, backward_beam);

        // Check for causal loops
        let backward_tachyons: Vec<_> = backward_beam
            .particles
            .iter()
            .filter_map(|p| Tachyon::from_particle(p))
            .collect();

        let no_loops = !self
            .validator
            .check_causal_loop(&forward_beam.particles, &backward_tachyons);

        consistent && no_loops
    }

    /// Get total darkness content
    pub fn total_darkness(&self) -> f64 {
        self.field.total_darkness
    }

    /// Get the causality report
    pub fn report(&self) -> CausalityReport {
        self.validator.report()
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.tachyons.clear();
        self.field.reset();
        self.validator.clear();
        self.next_id = 0;
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Darkness Tracker:\n\
             ├─ Tachyons: {}\n\
             ├─ Total darkness: {:.4}\n\
             ├─ Max local darkness: {:.4}\n\
             └─ Validity: {}",
            self.tachyons.len(),
            self.field.total_darkness,
            self.field.max_local_darkness,
            if self.validator.violations.is_empty() {
                "OK"
            } else {
                "VIOLATIONS"
            }
        )
    }
}

impl Default for DarknessTracker {
    fn default() -> Self {
        Self::new((64, 64))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::particles::ParticleFlavor;

    #[test]
    fn test_causal_direction_from_velocity() {
        assert_eq!(
            CausalDirection::from_velocity(0.5),
            CausalDirection::Forward
        );
        assert_eq!(
            CausalDirection::from_velocity(1.0),
            CausalDirection::Lightlike
        );
        assert_eq!(
            CausalDirection::from_velocity(2.0),
            CausalDirection::Backward
        );
        assert_eq!(
            CausalDirection::from_velocity(1000.0),
            CausalDirection::Spacelike
        );
    }

    #[test]
    fn test_tachyon_creation() {
        let tachyon = Tachyon::new(
            0,
            10.0,            // energy
            5.0,             // imaginary mass
            (1.0, 0.0, 0.0), // direction
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
        );

        assert!(tachyon.mass_squared < 0.0);
        assert!(tachyon.velocity > SPEED_OF_LIGHT);
        assert!(tachyon.darkness_content > 0.0);
    }

    #[test]
    fn test_tachyon_from_particle() {
        // Create a tachyonic particle (|p| > E)
        let momentum = FourMomentum::new(3.0, 4.0, 0.0, 0.0);
        assert!(momentum.is_tachyonic());

        let particle = Particle::new(
            0,
            ParticleFlavor::Gradient,
            momentum,
            TorusCoordinate::new(0.0, 0.0),
        );

        let tachyon = Tachyon::from_particle(&particle);
        assert!(tachyon.is_some());

        let tachyon = tachyon.unwrap();
        assert!(tachyon.velocity > 1.0);
    }

    #[test]
    fn test_darkness_field() {
        let mut field = DarknessField::new(16, 16);

        let tachyon = Tachyon::new(
            0,
            10.0,
            5.0,
            (1.0, 0.0, 0.0),
            TorusCoordinate::new(PI / 2.0, PI / 2.0),
            0,
            0,
        );

        field.deposit(&tachyon);

        assert!(field.total_darkness > 0.0);
        assert!(field.max_local_darkness > 0.0);
    }

    #[test]
    fn test_causality_validator() {
        let mut validator = CausalityValidator::default_config();

        // Normal particle
        let normal = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 3.0, 4.0, 0.0), // |p| = 5 < E = 10
            TorusCoordinate::new(0.0, 0.0),
        );
        assert_eq!(
            validator.validate_particle(&normal),
            CausalDirection::Forward
        );

        // Tachyonic particle
        let tachyon = Particle::new(
            1,
            ParticleFlavor::Gradient,
            FourMomentum::new(3.0, 4.0, 0.0, 0.0), // |p| = 4 > E = 3
            TorusCoordinate::new(0.0, 0.0),
        );
        assert_eq!(
            validator.validate_particle(&tachyon),
            CausalDirection::Backward
        );

        let report = validator.report();
        assert_eq!(report.stats.forward_particles, 1);
        assert_eq!(report.stats.backward_particles, 1);
    }

    #[test]
    fn test_darkness_tracker() {
        let mut tracker = DarknessTracker::new((32, 32));

        // Process a tachyonic particle
        let particle = Particle::new(
            0,
            ParticleFlavor::Gradient,
            FourMomentum::new(3.0, 4.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );

        let result = tracker.process_particle(&particle);
        assert!(result.is_some());
        assert_eq!(tracker.tachyons.len(), 1);
        assert!(tracker.total_darkness() > 0.0);

        println!("{}", tracker.summary());
        println!("{}", tracker.report().summary());
    }

    #[test]
    fn test_temporal_displacement() {
        let tachyon = Tachyon::new(
            0,
            10.0,
            5.0,
            (1.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
        );

        let displacement = tachyon.temporal_displacement(1.0);
        assert!(displacement > 0.0); // Tachyons cause positive temporal displacement
    }
}
