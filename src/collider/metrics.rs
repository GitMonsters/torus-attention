//! Collision Metrics and Statistics
//!
//! Tracks statistics about attention "collisions" using particle physics metrics:
//! - Cross-sections (collision probabilities)
//! - Luminosity (throughput)
//! - Event rates
//! - Kinematic distributions
//!
//! # Physics Analogies
//!
//! | Particle Physics | Attention Mechanism |
//! |------------------|---------------------|
//! | Cross-section σ  | Attention probability |
//! | Luminosity L     | Batch size × seq length |
//! | Event rate R = σL| Effective attention |
//! | √s (CM energy)   | Total activation energy |
//! | Differential dσ/dΩ | Attention distribution |

use crate::collider::particles::{Particle, ParticleBeam};
use crate::collider::vertices::CollisionEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-SECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-section measurement (in "attention barns")
///
/// In particle physics, cross-section σ measures collision probability.
/// Here it measures the probability that Q·K → Attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSection {
    /// Total cross-section
    pub total: f64,
    /// Elastic cross-section (Q·K → Q'·K')
    pub elastic: f64,
    /// Inelastic cross-section (Q·K → Attention + X)
    pub inelastic: f64,
    /// Differential cross-section dσ/dΩ at various angles
    pub differential: Vec<(f64, f64)>, // (angle, dσ/dΩ)
    /// Cross-section by interaction type
    pub by_interaction: HashMap<String, f64>,
    /// Statistical uncertainty
    pub uncertainty: f64,
}

impl CrossSection {
    /// Create a new cross-section measurement
    pub fn new() -> Self {
        Self {
            total: 0.0,
            elastic: 0.0,
            inelastic: 0.0,
            differential: Vec::new(),
            by_interaction: HashMap::new(),
            uncertainty: 0.0,
        }
    }

    /// Compute from collision events
    pub fn from_events(events: &[CollisionEvent], luminosity: f64) -> Self {
        let n_total = events.len() as f64;
        let n_elastic = events.iter().filter(|e| e.is_elastic()).count() as f64;
        let n_inelastic = n_total - n_elastic;

        // σ = N / L
        let total = n_total / luminosity.max(1e-10);
        let elastic = n_elastic / luminosity.max(1e-10);
        let inelastic = n_inelastic / luminosity.max(1e-10);

        // Compute by interaction type
        let mut by_interaction = HashMap::new();
        for event in events {
            let key = format!("{:?}", event.vertex_type);
            *by_interaction.entry(key).or_insert(0.0) += 1.0 / luminosity.max(1e-10);
        }

        // Differential cross-section (binned by scattering angle)
        let n_bins = 10;
        let mut angle_counts = vec![0.0; n_bins];
        for event in events {
            if let Some(angle) = event.scattering_angle() {
                let bin = ((angle / PI) * n_bins as f64) as usize;
                let bin = bin.min(n_bins - 1);
                angle_counts[bin] += 1.0;
            }
        }

        let differential: Vec<(f64, f64)> = angle_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let angle = (i as f64 + 0.5) * PI / n_bins as f64;
                let solid_angle = 2.0 * PI * (angle.sin()); // Approximate solid angle element
                let dsigma = count / luminosity.max(1e-10) / solid_angle.max(1e-10);
                (angle, dsigma)
            })
            .collect();

        // Statistical uncertainty: √N / L
        let uncertainty = n_total.sqrt() / luminosity.max(1e-10);

        Self {
            total,
            elastic,
            inelastic,
            differential,
            by_interaction,
            uncertainty,
        }
    }

    /// Get inelasticity ratio
    pub fn inelasticity(&self) -> f64 {
        if self.total > 0.0 {
            self.inelastic / self.total
        } else {
            0.0
        }
    }

    /// Get forward-backward asymmetry from differential cross-section
    pub fn forward_backward_asymmetry(&self) -> f64 {
        if self.differential.is_empty() {
            return 0.0;
        }

        let mid = self.differential.len() / 2;
        let forward: f64 = self.differential[..mid].iter().map(|(_, ds)| ds).sum();
        let backward: f64 = self.differential[mid..].iter().map(|(_, ds)| ds).sum();

        (forward - backward) / (forward + backward + 1e-10)
    }
}

impl Default for CrossSection {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUMINOSITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Luminosity tracking
///
/// In particle physics, luminosity L = N₁N₂f / A
/// where N₁, N₂ are particle numbers, f is collision frequency, A is cross-sectional area.
///
/// For attention: L = batch_size × seq_len × n_heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Luminosity {
    /// Instantaneous luminosity
    pub instantaneous: f64,
    /// Integrated luminosity (total over time)
    pub integrated: f64,
    /// Peak luminosity seen
    pub peak: f64,
    /// Average luminosity
    pub average: f64,
    /// Number of measurements
    n_measurements: u64,
}

impl Luminosity {
    /// Create a new luminosity tracker
    pub fn new() -> Self {
        Self {
            instantaneous: 0.0,
            integrated: 0.0,
            peak: 0.0,
            average: 0.0,
            n_measurements: 0,
        }
    }

    /// Record a luminosity measurement
    pub fn record(&mut self, luminosity: f64) {
        self.instantaneous = luminosity;
        self.integrated += luminosity;

        if luminosity > self.peak {
            self.peak = luminosity;
        }

        self.n_measurements += 1;
        self.average = self.integrated / self.n_measurements as f64;
    }

    /// Compute luminosity from attention configuration
    pub fn from_attention(batch_size: usize, seq_len: usize, n_heads: usize) -> f64 {
        (batch_size * seq_len * seq_len * n_heads) as f64
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.instantaneous = 0.0;
        self.integrated = 0.0;
        self.peak = 0.0;
        self.average = 0.0;
        self.n_measurements = 0;
    }
}

impl Default for Luminosity {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT RATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Event rate tracking
///
/// R = σ × L (events per unit time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRate {
    /// Current event rate
    pub current: f64,
    /// Total events
    pub total_events: u64,
    /// Events by type
    pub by_type: HashMap<String, u64>,
    /// Time window for rate calculation
    pub window_size: u64,
    /// Events in current window
    events_in_window: u64,
    /// Current step
    step: u64,
}

impl EventRate {
    /// Create a new event rate tracker
    pub fn new(window_size: u64) -> Self {
        Self {
            current: 0.0,
            total_events: 0,
            by_type: HashMap::new(),
            window_size,
            events_in_window: 0,
            step: 0,
        }
    }

    /// Record an event
    pub fn record_event(&mut self, event_type: &str) {
        self.total_events += 1;
        self.events_in_window += 1;
        *self.by_type.entry(event_type.to_string()).or_insert(0) += 1;
    }

    /// Advance to next step and update rate
    pub fn next_step(&mut self) {
        self.step += 1;

        if self.step % self.window_size == 0 {
            self.current = self.events_in_window as f64 / self.window_size as f64;
            self.events_in_window = 0;
        }
    }

    /// Get average rate
    pub fn average_rate(&self) -> f64 {
        self.total_events as f64 / self.step.max(1) as f64
    }
}

impl Default for EventRate {
    fn default() -> Self {
        Self::new(100)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KINEMATIC DISTRIBUTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Histogram for tracking distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin edges
    pub edges: Vec<f64>,
    /// Bin counts
    pub counts: Vec<u64>,
    /// Total entries
    pub total: u64,
    /// Underflow count
    pub underflow: u64,
    /// Overflow count
    pub overflow: u64,
    /// Sum of values (for mean)
    sum: f64,
    /// Sum of squared values (for variance)
    sum_sq: f64,
}

impl Histogram {
    /// Create a histogram with uniform bins
    pub fn new(min: f64, max: f64, n_bins: usize) -> Self {
        let step = (max - min) / n_bins as f64;
        let edges: Vec<f64> = (0..=n_bins).map(|i| min + i as f64 * step).collect();
        Self {
            edges,
            counts: vec![0; n_bins],
            total: 0,
            underflow: 0,
            overflow: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Fill histogram with a value
    pub fn fill(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;
        self.sum_sq += value * value;

        if value < self.edges[0] {
            self.underflow += 1;
            return;
        }
        if value >= *self.edges.last().unwrap() {
            self.overflow += 1;
            return;
        }

        // Binary search for bin
        let bin = self.edges.partition_point(|&e| e <= value) - 1;
        if bin < self.counts.len() {
            self.counts[bin] += 1;
        }
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        if self.total > 0 {
            self.sum / self.total as f64
        } else {
            0.0
        }
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        if self.total > 1 {
            let mean = self.mean();
            self.sum_sq / self.total as f64 - mean * mean
        } else {
            0.0
        }
    }

    /// Get standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get bin center for a given index
    pub fn bin_center(&self, bin: usize) -> f64 {
        if bin < self.counts.len() {
            (self.edges[bin] + self.edges[bin + 1]) / 2.0
        } else {
            0.0
        }
    }

    /// Get normalized counts (density)
    pub fn density(&self) -> Vec<f64> {
        let bin_width = self.edges[1] - self.edges[0];
        let total = self.total as f64;
        self.counts
            .iter()
            .map(|&c| c as f64 / total / bin_width)
            .collect()
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.total = 0;
        self.underflow = 0;
        self.overflow = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

/// Kinematic distribution tracker
#[derive(Debug, Clone)]
pub struct KinematicDistributions {
    /// Energy distribution
    pub energy: Histogram,
    /// Transverse momentum distribution
    pub pt: Histogram,
    /// Pseudorapidity distribution
    pub eta: Histogram,
    /// Azimuthal angle distribution
    pub phi: Histogram,
    /// Invariant mass distribution
    pub mass: Histogram,
    /// Rapidity distribution
    pub rapidity: Histogram,
}

impl KinematicDistributions {
    /// Create new distribution tracker
    pub fn new() -> Self {
        Self {
            energy: Histogram::new(0.0, 100.0, 50),
            pt: Histogram::new(0.0, 50.0, 50),
            eta: Histogram::new(-5.0, 5.0, 50),
            phi: Histogram::new(-PI, PI, 36),
            mass: Histogram::new(0.0, 100.0, 50),
            rapidity: Histogram::new(-5.0, 5.0, 50),
        }
    }

    /// Fill from a particle
    pub fn fill_particle(&mut self, particle: &Particle) {
        self.energy.fill(particle.energy());
        self.pt.fill(particle.momentum.transverse_momentum());
        self.eta.fill(particle.momentum.pseudorapidity());
        self.phi.fill(particle.momentum.azimuthal_angle());
        self.mass.fill(particle.mass());
        self.rapidity.fill(particle.momentum.rapidity());
    }

    /// Fill from a beam
    pub fn fill_beam(&mut self, beam: &ParticleBeam) {
        for particle in &beam.particles {
            self.fill_particle(particle);
        }
    }

    /// Reset all distributions
    pub fn reset(&mut self) {
        self.energy.reset();
        self.pt.reset();
        self.eta.reset();
        self.phi.reset();
        self.mass.reset();
        self.rapidity.reset();
    }

    /// Get summary
    pub fn summary(&self) -> String {
        format!(
            "Kinematics (n={}):\n\
             ├─ E:   mean={:.2}, std={:.2}\n\
             ├─ pT:  mean={:.2}, std={:.2}\n\
             ├─ η:   mean={:.2}, std={:.2}\n\
             ├─ φ:   mean={:.2}, std={:.2}\n\
             ├─ m:   mean={:.2}, std={:.2}\n\
             └─ y:   mean={:.2}, std={:.2}",
            self.energy.total,
            self.energy.mean(),
            self.energy.std(),
            self.pt.mean(),
            self.pt.std(),
            self.eta.mean(),
            self.eta.std(),
            self.phi.mean(),
            self.phi.std(),
            self.mass.mean(),
            self.mass.std(),
            self.rapidity.mean(),
            self.rapidity.std()
        )
    }
}

impl Default for KinematicDistributions {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLISION STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Comprehensive collision statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionStats {
    /// Total collisions
    pub n_collisions: u64,
    /// Successful interactions
    pub n_interactions: u64,
    /// Failed/rejected collisions
    pub n_rejected: u64,
    /// Average center-of-mass energy
    pub avg_sqrt_s: f64,
    /// Average multiplicity (particles per collision)
    pub avg_multiplicity: f64,
    /// Average Q² (momentum transfer squared)
    pub avg_q_squared: f64,
    /// Collisions by interaction type
    pub by_type: HashMap<String, u64>,
    /// Running sums for averages
    sum_sqrt_s: f64,
    sum_multiplicity: f64,
    sum_q_squared: f64,
}

impl CollisionStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            n_collisions: 0,
            n_interactions: 0,
            n_rejected: 0,
            avg_sqrt_s: 0.0,
            avg_multiplicity: 0.0,
            avg_q_squared: 0.0,
            by_type: HashMap::new(),
            sum_sqrt_s: 0.0,
            sum_multiplicity: 0.0,
            sum_q_squared: 0.0,
        }
    }

    /// Record a collision event
    pub fn record(&mut self, event: &CollisionEvent) {
        self.n_collisions += 1;
        self.n_interactions += 1;

        // Update sums
        self.sum_sqrt_s += event.center_of_mass_energy();
        self.sum_multiplicity += event.outgoing.len() as f64;
        self.sum_q_squared += event.q_squared().unwrap_or(0.0);

        // Update averages
        let n = self.n_interactions as f64;
        self.avg_sqrt_s = self.sum_sqrt_s / n;
        self.avg_multiplicity = self.sum_multiplicity / n;
        self.avg_q_squared = self.sum_q_squared / n;

        // Count by type
        let type_key = format!("{:?}", event.vertex_type);
        *self.by_type.entry(type_key).or_insert(0) += 1;
    }

    /// Record a rejected collision
    pub fn record_rejection(&mut self) {
        self.n_collisions += 1;
        self.n_rejected += 1;
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_collisions > 0 {
            self.n_interactions as f64 / self.n_collisions as f64
        } else {
            0.0
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.n_collisions = 0;
        self.n_interactions = 0;
        self.n_rejected = 0;
        self.avg_sqrt_s = 0.0;
        self.avg_multiplicity = 0.0;
        self.avg_q_squared = 0.0;
        self.by_type.clear();
        self.sum_sqrt_s = 0.0;
        self.sum_multiplicity = 0.0;
        self.sum_q_squared = 0.0;
    }

    /// Get summary
    pub fn summary(&self) -> String {
        let mut type_summary = String::new();
        for (t, count) in &self.by_type {
            type_summary.push_str(&format!("\n│  ├─ {}: {}", t, count));
        }

        format!(
            "Collision Statistics:\n\
             ├─ Total: {} ({} accepted, {} rejected)\n\
             ├─ Acceptance: {:.1}%\n\
             ├─ <√s>: {:.2}\n\
             ├─ <multiplicity>: {:.2}\n\
             ├─ <Q²>: {:.2}\n\
             ├─ By type:{}",
            self.n_collisions,
            self.n_interactions,
            self.n_rejected,
            self.acceptance_rate() * 100.0,
            self.avg_sqrt_s,
            self.avg_multiplicity,
            self.avg_q_squared,
            if type_summary.is_empty() {
                " (none)".to_string()
            } else {
                type_summary
            }
        )
    }
}

impl Default for CollisionStats {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLIDER METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete metrics for the torus collider
#[derive(Debug)]
pub struct ColliderMetrics {
    /// Cross-section measurements
    pub cross_section: CrossSection,
    /// Luminosity tracking
    pub luminosity: Luminosity,
    /// Event rate
    pub event_rate: EventRate,
    /// Collision statistics
    pub collision_stats: CollisionStats,
    /// Kinematic distributions
    pub kinematics: KinematicDistributions,
    /// Event buffer for cross-section calculation
    event_buffer: Vec<CollisionEvent>,
    /// Buffer size
    buffer_size: usize,
    /// Current step
    step: u64,
}

impl ColliderMetrics {
    /// Create new metrics tracker
    pub fn new(buffer_size: usize) -> Self {
        Self {
            cross_section: CrossSection::new(),
            luminosity: Luminosity::new(),
            event_rate: EventRate::new(100),
            collision_stats: CollisionStats::new(),
            kinematics: KinematicDistributions::new(),
            event_buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            step: 0,
        }
    }

    /// Record a collision event
    pub fn record_event(&mut self, event: CollisionEvent) {
        // Update statistics
        self.collision_stats.record(&event);

        // Update kinematics
        for particle in &event.outgoing {
            self.kinematics.fill_particle(particle);
        }

        // Buffer for cross-section
        self.event_buffer.push(event);

        // Update event rate
        self.event_rate.record_event("collision");
    }

    /// Record a rejected collision
    pub fn record_rejection(&mut self) {
        self.collision_stats.record_rejection();
        self.event_rate.record_event("rejected");
    }

    /// Record luminosity
    pub fn record_luminosity(&mut self, batch_size: usize, seq_len: usize, n_heads: usize) {
        let lum = Luminosity::from_attention(batch_size, seq_len, n_heads);
        self.luminosity.record(lum);
    }

    /// Advance to next step
    pub fn next_step(&mut self) {
        self.step += 1;
        self.event_rate.next_step();

        // Update cross-section periodically
        if self.event_buffer.len() >= self.buffer_size {
            self.cross_section =
                CrossSection::from_events(&self.event_buffer, self.luminosity.integrated);
            self.event_buffer.clear();
        }
    }

    /// Get comprehensive report
    pub fn report(&self) -> MetricsReport {
        MetricsReport {
            step: self.step,
            cross_section: self.cross_section.clone(),
            luminosity: self.luminosity.clone(),
            collision_stats: self.collision_stats.clone(),
            event_rate: self.event_rate.current,
            kinematics_summary: self.kinematics.summary(),
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.cross_section = CrossSection::new();
        self.luminosity.reset();
        self.event_rate = EventRate::new(100);
        self.collision_stats.reset();
        self.kinematics.reset();
        self.event_buffer.clear();
        self.step = 0;
    }

    /// Get summary
    pub fn summary(&self) -> String {
        format!(
            "Collider Metrics @ step {}\n\
             ═══════════════════════════════════════\n\
             Cross-Section:\n\
             ├─ σ_total: {:.4} ± {:.4}\n\
             ├─ σ_elastic: {:.4}\n\
             ├─ σ_inelastic: {:.4}\n\
             ├─ Inelasticity: {:.1}%\n\
             └─ F/B asymmetry: {:.3}\n\
             \n\
             Luminosity:\n\
             ├─ Instantaneous: {:.2}\n\
             ├─ Integrated: {:.2}\n\
             ├─ Peak: {:.2}\n\
             └─ Average: {:.2}\n\
             \n\
             Event Rate: {:.2} events/step\n\
             \n\
             {}\n\
             \n\
             {}",
            self.step,
            self.cross_section.total,
            self.cross_section.uncertainty,
            self.cross_section.elastic,
            self.cross_section.inelastic,
            self.cross_section.inelasticity() * 100.0,
            self.cross_section.forward_backward_asymmetry(),
            self.luminosity.instantaneous,
            self.luminosity.integrated,
            self.luminosity.peak,
            self.luminosity.average,
            self.event_rate.current,
            self.collision_stats.summary(),
            self.kinematics.summary()
        )
    }
}

impl Default for ColliderMetrics {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Metrics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    /// Current step
    pub step: u64,
    /// Cross-section
    pub cross_section: CrossSection,
    /// Luminosity
    pub luminosity: Luminosity,
    /// Collision statistics
    pub collision_stats: CollisionStats,
    /// Event rate
    pub event_rate: f64,
    /// Kinematics summary
    pub kinematics_summary: String,
}

impl Default for MetricsReport {
    fn default() -> Self {
        Self {
            step: 0,
            cross_section: CrossSection::default(),
            luminosity: Luminosity::default(),
            collision_stats: CollisionStats::default(),
            event_rate: 0.0,
            kinematics_summary: String::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::particles::{FourMomentum, ParticleFlavor};
    use crate::collider::vertices::VertexType;
    use crate::geometry::TorusCoordinate;

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(0.0, 10.0, 10);

        // Fill with some values
        for i in 0..100 {
            hist.fill(i as f64 / 10.0);
        }

        assert_eq!(hist.total, 100);
        assert!((hist.mean() - 4.95).abs() < 0.1);
        println!("Mean: {}, Std: {}", hist.mean(), hist.std());
    }

    #[test]
    fn test_luminosity() {
        let mut lum = Luminosity::new();

        for i in 1..=10 {
            lum.record(i as f64 * 100.0);
        }

        assert_eq!(lum.instantaneous, 1000.0);
        assert_eq!(lum.peak, 1000.0);
        assert!((lum.average - 550.0).abs() < 1.0);
    }

    #[test]
    fn test_collision_stats() {
        let mut stats = CollisionStats::new();

        // Create some mock events
        for i in 0..10 {
            let mut event = CollisionEvent::new(
                i,
                VertexType::QueryKey,
                TorusCoordinate::new(0.0, 0.0),
                0,
                0,
                0,
            );
            // Add incoming particles for center_of_mass_energy calculation
            event.add_incoming(Particle::new(
                i * 2,
                ParticleFlavor::Query,
                FourMomentum::new(10.0, 5.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            event.add_incoming(Particle::new(
                i * 2 + 1,
                ParticleFlavor::Key,
                FourMomentum::new(10.0, -5.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            // Add outgoing particles
            event.add_outgoing(Particle::new(
                i * 2 + 100,
                ParticleFlavor::Output,
                FourMomentum::new(10.0, 1.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            event.add_outgoing(Particle::new(
                i * 2 + 101,
                ParticleFlavor::Output,
                FourMomentum::new(10.0, -1.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            stats.record(&event);
        }

        assert_eq!(stats.n_interactions, 10);
        assert!((stats.avg_multiplicity - 2.0).abs() < 0.1);
        println!("{}", stats.summary());
    }

    #[test]
    fn test_kinematics() {
        let mut kin = KinematicDistributions::new();

        // Fill with particles
        for i in 0..100 {
            let particle = Particle::new(
                i,
                ParticleFlavor::Output,
                FourMomentum::new(
                    10.0 + (i as f64 / 10.0),
                    (i as f64 / 50.0).cos() * 5.0,
                    (i as f64 / 50.0).sin() * 5.0,
                    i as f64 / 25.0 - 2.0,
                ),
                TorusCoordinate::new(0.0, 0.0),
            );
            kin.fill_particle(&particle);
        }

        println!("{}", kin.summary());
        assert_eq!(kin.energy.total, 100);
    }

    #[test]
    fn test_collider_metrics() {
        let mut metrics = ColliderMetrics::new(100);

        // Record some luminosity
        metrics.record_luminosity(32, 128, 8);

        // Record some events
        for i in 0..50 {
            let mut event = CollisionEvent::new(
                i,
                VertexType::QueryKey,
                TorusCoordinate::new(0.0, 0.0),
                0,
                0,
                0,
            );
            event.add_outgoing(Particle::new(
                i,
                ParticleFlavor::Output,
                FourMomentum::new(10.0, 1.0, 0.0, 0.0),
                TorusCoordinate::new(0.0, 0.0),
            ));
            metrics.record_event(event);
            metrics.next_step();
        }

        println!("{}", metrics.summary());
    }
}
