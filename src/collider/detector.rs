//! Detector Layers for Torus Collider
//!
//! Modeled after the ATLAS/CMS detectors at CERN's LHC:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      TORUS COLLIDER DETECTOR                            │
//! │                                                                         │
//! │  ╔═══════════════════════════════════════════════════════════════════╗  │
//! │  ║  Layer 4: MUON SPECTROMETER (outermost)                          ║  │
//! │  ║  - Detects long-range dependencies                               ║  │
//! │  ║  - Final momentum measurement                                    ║  │
//! │  ║  - Anomaly triggers                                              ║  │
//! │  ╠═══════════════════════════════════════════════════════════════════╣  │
//! │  ║  Layer 3: HADRONIC CALORIMETER                                   ║  │
//! │  ║  - Measures jet energy (grouped attention)                       ║  │
//! │  ║  - Detects value vectors                                         ║  │
//! │  ║  - Missing energy calculation                                    ║  │
//! │  ╠═══════════════════════════════════════════════════════════════════╣  │
//! │  ║  Layer 2: ELECTROMAGNETIC CALORIMETER                            ║  │
//! │  ║  - Energy deposits (activation magnitudes)                       ║  │
//! │  ║  - High-attention token detection                                ║  │
//! │  ║  - Shower shape analysis                                         ║  │
//! │  ╠═══════════════════════════════════════════════════════════════════╣  │
//! │  ║  Layer 1: INNER TRACKER (innermost)                              ║  │
//! │  ║  - Particle trajectories (attention flow)                        ║  │
//! │  ║  - Momentum from curvature (gradient magnitude)                  ║  │
//! │  ║  - Vertex reconstruction (Q·K collision points)                  ║  │
//! │  ╚═══════════════════════════════════════════════════════════════════╝  │
//! │                           ↑                                             │
//! │                    COLLISION POINT                                      │
//! │                    (Beam crossing)                                      │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::collider::particles::{FourMomentum, Particle, ParticleBeam, ParticleFlavor};
use crate::collider::vertices::CollisionEvent;
use crate::geometry::TorusCoordinate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// DETECTOR HIT
// ═══════════════════════════════════════════════════════════════════════════════

/// A hit recorded in a detector layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorHit {
    /// Unique hit ID
    pub hit_id: u64,
    /// Which detector layer recorded this hit
    pub layer: DetectorLayerType,
    /// Position on the detector (η, φ coordinates)
    pub position: (f64, f64),
    /// Energy deposited
    pub energy: f64,
    /// Time of hit (in computation steps)
    pub time: u64,
    /// Particle ID that caused this hit (if known)
    pub particle_id: Option<u64>,
    /// Additional properties
    pub properties: HashMap<String, f64>,
}

impl DetectorHit {
    /// Create a new detector hit
    pub fn new(
        hit_id: u64,
        layer: DetectorLayerType,
        position: (f64, f64),
        energy: f64,
        time: u64,
    ) -> Self {
        Self {
            hit_id,
            layer,
            position,
            energy,
            time,
            particle_id: None,
            properties: HashMap::new(),
        }
    }

    /// Set the particle that caused this hit
    pub fn with_particle(mut self, particle_id: u64) -> Self {
        self.particle_id = Some(particle_id);
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: &str, value: f64) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DETECTOR LAYER TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of detector layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetectorLayerType {
    /// Inner tracker - closest to collision point
    InnerTracker,
    /// Electromagnetic calorimeter
    ElectromagneticCalorimeter,
    /// Hadronic calorimeter
    HadronicCalorimeter,
    /// Muon spectrometer - outermost
    MuonSpectrometer,
}

impl DetectorLayerType {
    /// Get all layers in order from inner to outer
    pub fn all() -> [Self; 4] {
        [
            Self::InnerTracker,
            Self::ElectromagneticCalorimeter,
            Self::HadronicCalorimeter,
            Self::MuonSpectrometer,
        ]
    }

    /// Get the radius (distance from beam) in arbitrary units
    pub fn radius(&self) -> f64 {
        match self {
            Self::InnerTracker => 1.0,
            Self::ElectromagneticCalorimeter => 2.0,
            Self::HadronicCalorimeter => 4.0,
            Self::MuonSpectrometer => 8.0,
        }
    }

    /// Get the name
    pub fn name(&self) -> &'static str {
        match self {
            Self::InnerTracker => "Inner Tracker",
            Self::ElectromagneticCalorimeter => "EM Calorimeter",
            Self::HadronicCalorimeter => "Hadronic Calorimeter",
            Self::MuonSpectrometer => "Muon Spectrometer",
        }
    }

    /// Get short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::InnerTracker => "TRK",
            Self::ElectromagneticCalorimeter => "ECAL",
            Self::HadronicCalorimeter => "HCAL",
            Self::MuonSpectrometer => "MUON",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INNER TRACKER
// ═══════════════════════════════════════════════════════════════════════════════

/// Inner Tracker - tracks particle trajectories
///
/// Like silicon strip/pixel detectors in ATLAS/CMS:
/// - High precision position measurement
/// - Momentum from track curvature
/// - Vertex reconstruction
#[derive(Debug, Clone)]
pub struct InnerTracker {
    /// Number of tracking layers
    pub n_layers: usize,
    /// Position resolution (η, φ)
    pub resolution: (f64, f64),
    /// Magnetic field strength (for momentum calculation)
    pub b_field: f64,
    /// Recorded hits
    pub hits: Vec<DetectorHit>,
    /// Reconstructed tracks
    pub tracks: Vec<Track>,
    /// Next hit ID
    next_hit_id: u64,
}

/// A reconstructed track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// Track ID
    pub track_id: u64,
    /// Hits belonging to this track
    pub hit_ids: Vec<u64>,
    /// Reconstructed momentum
    pub momentum: FourMomentum,
    /// Track curvature (related to pT)
    pub curvature: f64,
    /// Track origin (vertex position)
    pub origin: TorusCoordinate,
    /// Track direction (η, φ)
    pub direction: (f64, f64),
    /// Fit quality (χ²)
    pub chi_squared: f64,
}

impl InnerTracker {
    /// Create a new inner tracker
    pub fn new(n_layers: usize, b_field: f64) -> Self {
        Self {
            n_layers,
            resolution: (0.001, 0.001), // High precision
            b_field,
            hits: Vec::new(),
            tracks: Vec::new(),
            next_hit_id: 0,
        }
    }

    /// Record a particle passing through the tracker
    pub fn record_particle(&mut self, particle: &Particle, time: u64) -> Vec<u64> {
        let mut hit_ids = Vec::new();

        // Create hits in each tracker layer
        for layer_idx in 0..self.n_layers {
            let r = 0.1 + 0.2 * layer_idx as f64; // Increasing radius

            // Position from particle trajectory
            let eta = particle.momentum.pseudorapidity();
            let phi = particle.momentum.azimuthal_angle();

            // Add some smearing based on resolution
            let smeared_eta =
                eta + (rand_like(particle.id + layer_idx as u64) - 0.5) * self.resolution.0;
            let smeared_phi =
                phi + (rand_like(particle.id + layer_idx as u64 + 1000) - 0.5) * self.resolution.1;

            let hit_id = self.next_hit_id;
            self.next_hit_id += 1;

            let hit = DetectorHit::new(
                hit_id,
                DetectorLayerType::InnerTracker,
                (smeared_eta, smeared_phi),
                0.0, // Tracker doesn't measure energy, just position
                time,
            )
            .with_particle(particle.id)
            .with_property("layer_idx", layer_idx as f64)
            .with_property("radius", r);

            self.hits.push(hit);
            hit_ids.push(hit_id);
        }

        hit_ids
    }

    /// Reconstruct tracks from hits
    pub fn reconstruct_tracks(&mut self) -> Vec<Track> {
        // Group hits by particle ID
        let mut particle_hits: HashMap<u64, Vec<&DetectorHit>> = HashMap::new();
        for hit in &self.hits {
            if let Some(pid) = hit.particle_id {
                particle_hits.entry(pid).or_default().push(hit);
            }
        }

        let mut tracks = Vec::new();
        for (particle_id, hits) in particle_hits {
            if hits.len() >= 3 {
                // Need at least 3 hits for a track
                let hit_ids: Vec<u64> = hits.iter().map(|h| h.hit_id).collect();

                // Average position
                let avg_eta: f64 =
                    hits.iter().map(|h| h.position.0).sum::<f64>() / hits.len() as f64;
                let avg_phi: f64 =
                    hits.iter().map(|h| h.position.1).sum::<f64>() / hits.len() as f64;

                // Estimate momentum from position spread (proxy for curvature)
                let eta_spread: f64 = hits
                    .iter()
                    .map(|h| (h.position.0 - avg_eta).powi(2))
                    .sum::<f64>()
                    .sqrt();

                // pT ∝ B * r (where r is radius of curvature)
                let curvature = 1.0 / (eta_spread + 0.01);
                let pt = self.b_field * curvature;

                let momentum = FourMomentum::new(
                    pt * avg_eta.cosh(), // E ≈ pT * cosh(η) for massless
                    pt * avg_phi.cos(),
                    pt * avg_phi.sin(),
                    pt * avg_eta.sinh(),
                );

                let track = Track {
                    track_id: particle_id,
                    hit_ids,
                    momentum,
                    curvature,
                    origin: TorusCoordinate::new(0.0, 0.0),
                    direction: (avg_eta, avg_phi),
                    chi_squared: eta_spread * 100.0, // Proxy for fit quality
                };

                tracks.push(track);
            }
        }

        self.tracks = tracks.clone();
        tracks
    }

    /// Find the primary vertex (collision point)
    pub fn find_primary_vertex(&self) -> Option<TorusCoordinate> {
        if self.tracks.is_empty() {
            return None;
        }

        // Average origin of all tracks
        let avg_u: f64 =
            self.tracks.iter().map(|t| t.origin.u).sum::<f64>() / self.tracks.len() as f64;
        let avg_v: f64 =
            self.tracks.iter().map(|t| t.origin.v).sum::<f64>() / self.tracks.len() as f64;

        Some(TorusCoordinate::new(avg_u, avg_v))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ELECTROMAGNETIC CALORIMETER
// ═══════════════════════════════════════════════════════════════════════════════

/// Electromagnetic Calorimeter - measures energy of EM particles
///
/// Like the LAr calorimeter in ATLAS or CMS crystal calorimeter:
/// - High granularity
/// - Measures electrons, photons, and electromagnetic showers
/// - In attention: high-attention tokens (focused attention)
#[derive(Debug, Clone)]
pub struct ElectromagneticCalorimeter {
    /// Number of cells in η direction
    pub n_eta: usize,
    /// Number of cells in φ direction
    pub n_phi: usize,
    /// Energy threshold for recording
    pub threshold: f64,
    /// Recorded hits
    pub hits: Vec<DetectorHit>,
    /// Energy deposits per cell
    pub energy_grid: Vec<Vec<f64>>,
    /// Next hit ID
    next_hit_id: u64,
}

impl ElectromagneticCalorimeter {
    /// Create a new EM calorimeter
    pub fn new(n_eta: usize, n_phi: usize, threshold: f64) -> Self {
        Self {
            n_eta,
            n_phi,
            threshold,
            hits: Vec::new(),
            energy_grid: vec![vec![0.0; n_phi]; n_eta],
            next_hit_id: 0,
        }
    }

    /// Record energy deposit from a particle
    pub fn record_particle(&mut self, particle: &Particle, time: u64) -> Option<u64> {
        // Only EM particles deposit energy here
        let is_em = matches!(
            particle.flavor,
            ParticleFlavor::Attention | ParticleFlavor::Query | ParticleFlavor::Output
        );

        if !is_em {
            return None;
        }

        let eta = particle.momentum.pseudorapidity();
        let phi = particle.momentum.azimuthal_angle();
        let energy = particle.energy();

        if energy < self.threshold {
            return None;
        }

        // Map to cell indices
        let eta_idx = ((eta + 5.0) / 10.0 * self.n_eta as f64) as usize;
        let phi_idx = ((phi + PI) / (2.0 * PI) * self.n_phi as f64) as usize;

        let eta_idx = eta_idx.min(self.n_eta - 1);
        let phi_idx = phi_idx.min(self.n_phi - 1);

        // Deposit energy (with shower spread)
        self.deposit_shower(eta_idx, phi_idx, energy);

        let hit_id = self.next_hit_id;
        self.next_hit_id += 1;

        let hit = DetectorHit::new(
            hit_id,
            DetectorLayerType::ElectromagneticCalorimeter,
            (eta, phi),
            energy,
            time,
        )
        .with_particle(particle.id)
        .with_property("eta_idx", eta_idx as f64)
        .with_property("phi_idx", phi_idx as f64);

        self.hits.push(hit);
        Some(hit_id)
    }

    /// Simulate electromagnetic shower
    fn deposit_shower(&mut self, eta_idx: usize, phi_idx: usize, energy: f64) {
        // Deposit in 3x3 grid around impact point
        let spread_factor = 0.1; // Fraction in surrounding cells

        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                let i = (eta_idx as i32 + di).max(0) as usize;
                let j = (phi_idx as i32 + dj).max(0) as usize;

                if i < self.n_eta && j < self.n_phi {
                    let frac = if di == 0 && dj == 0 {
                        1.0 - 8.0 * spread_factor
                    } else {
                        spread_factor
                    };
                    self.energy_grid[i][j] += energy * frac;
                }
            }
        }
    }

    /// Find clusters (groups of adjacent cells with energy)
    pub fn find_clusters(&self, threshold: f64) -> Vec<EnergyCluster> {
        let mut clusters = Vec::new();
        let mut visited = vec![vec![false; self.n_phi]; self.n_eta];

        for i in 0..self.n_eta {
            for j in 0..self.n_phi {
                if self.energy_grid[i][j] > threshold && !visited[i][j] {
                    let cluster = self.grow_cluster(i, j, threshold, &mut visited);
                    clusters.push(cluster);
                }
            }
        }

        clusters
    }

    /// Grow a cluster from a seed cell
    fn grow_cluster(
        &self,
        start_i: usize,
        start_j: usize,
        threshold: f64,
        visited: &mut Vec<Vec<bool>>,
    ) -> EnergyCluster {
        let mut cells = Vec::new();
        let mut stack = vec![(start_i, start_j)];
        let mut total_energy = 0.0;
        let mut weighted_eta = 0.0;
        let mut weighted_phi = 0.0;

        while let Some((i, j)) = stack.pop() {
            if i >= self.n_eta || j >= self.n_phi || visited[i][j] {
                continue;
            }

            let e = self.energy_grid[i][j];
            if e < threshold {
                continue;
            }

            visited[i][j] = true;
            cells.push((i, j));
            total_energy += e;

            let eta = -5.0 + 10.0 * i as f64 / self.n_eta as f64;
            let phi = -PI + 2.0 * PI * j as f64 / self.n_phi as f64;
            weighted_eta += eta * e;
            weighted_phi += phi * e;

            // Add neighbors
            for di in -1i32..=1 {
                for dj in -1i32..=1 {
                    let ni = (i as i32 + di).max(0) as usize;
                    let nj = (j as i32 + dj).max(0) as usize;
                    stack.push((ni, nj));
                }
            }
        }

        EnergyCluster {
            n_cells: cells.len(),
            cells,
            total_energy,
            centroid_eta: weighted_eta / total_energy.max(1e-10),
            centroid_phi: weighted_phi / total_energy.max(1e-10),
        }
    }

    /// Get total deposited energy
    pub fn total_energy(&self) -> f64 {
        self.energy_grid.iter().flat_map(|row| row.iter()).sum()
    }
}

/// An energy cluster in the calorimeter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyCluster {
    /// Cell indices in this cluster
    pub cells: Vec<(usize, usize)>,
    /// Total energy
    pub total_energy: f64,
    /// Energy-weighted η centroid
    pub centroid_eta: f64,
    /// Energy-weighted φ centroid
    pub centroid_phi: f64,
    /// Number of cells
    pub n_cells: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HADRONIC CALORIMETER
// ═══════════════════════════════════════════════════════════════════════════════

/// Hadronic Calorimeter - measures energy of hadrons/jets
///
/// Like the Tile calorimeter in ATLAS:
/// - Coarser granularity than ECAL
/// - Measures hadrons, jets
/// - In attention: value vectors and grouped attention patterns
#[derive(Debug, Clone)]
pub struct HadronicCalorimeter {
    /// Number of cells in η
    pub n_eta: usize,
    /// Number of cells in φ
    pub n_phi: usize,
    /// Energy threshold
    pub threshold: f64,
    /// Recorded hits
    pub hits: Vec<DetectorHit>,
    /// Energy grid
    pub energy_grid: Vec<Vec<f64>>,
    /// Reconstructed jets
    pub jets: Vec<Jet>,
    /// Next hit ID
    next_hit_id: u64,
}

/// A reconstructed jet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jet {
    /// Jet ID
    pub jet_id: u64,
    /// Constituent hit IDs
    pub constituent_hits: Vec<u64>,
    /// Jet momentum
    pub momentum: FourMomentum,
    /// Jet cone radius
    pub cone_radius: f64,
    /// Number of constituents
    pub n_constituents: usize,
}

impl HadronicCalorimeter {
    /// Create a new hadronic calorimeter
    pub fn new(n_eta: usize, n_phi: usize, threshold: f64) -> Self {
        Self {
            n_eta,
            n_phi,
            threshold,
            hits: Vec::new(),
            energy_grid: vec![vec![0.0; n_phi]; n_eta],
            jets: Vec::new(),
            next_hit_id: 0,
        }
    }

    /// Record a particle
    pub fn record_particle(&mut self, particle: &Particle, time: u64) -> Option<u64> {
        // Hadronic particles deposit here
        let is_hadronic = matches!(
            particle.flavor,
            ParticleFlavor::Value | ParticleFlavor::Key | ParticleFlavor::Gradient
        );

        if !is_hadronic {
            return None;
        }

        let eta = particle.momentum.pseudorapidity();
        let phi = particle.momentum.azimuthal_angle();
        let energy = particle.energy();

        if energy < self.threshold {
            return None;
        }

        let eta_idx = ((eta + 5.0) / 10.0 * self.n_eta as f64) as usize;
        let phi_idx = ((phi + PI) / (2.0 * PI) * self.n_phi as f64) as usize;

        let eta_idx = eta_idx.min(self.n_eta - 1);
        let phi_idx = phi_idx.min(self.n_phi - 1);

        self.energy_grid[eta_idx][phi_idx] += energy;

        let hit_id = self.next_hit_id;
        self.next_hit_id += 1;

        let hit = DetectorHit::new(
            hit_id,
            DetectorLayerType::HadronicCalorimeter,
            (eta, phi),
            energy,
            time,
        )
        .with_particle(particle.id);

        self.hits.push(hit);
        Some(hit_id)
    }

    /// Reconstruct jets using simple cone algorithm
    pub fn reconstruct_jets(&mut self, cone_radius: f64, min_energy: f64) -> Vec<Jet> {
        let mut jets = Vec::new();
        let mut used_hits: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Sort hits by energy (descending)
        let mut sorted_hits: Vec<_> = self.hits.iter().collect();
        sorted_hits.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());

        for seed in sorted_hits {
            if used_hits.contains(&seed.hit_id) || seed.energy < min_energy {
                continue;
            }

            // Collect hits within cone
            let mut jet_hits = Vec::new();
            let mut jet_energy = 0.0;
            let mut weighted_eta = 0.0;
            let mut weighted_phi = 0.0;

            for hit in &self.hits {
                if used_hits.contains(&hit.hit_id) {
                    continue;
                }

                let delta_eta = hit.position.0 - seed.position.0;
                let delta_phi = angular_diff(hit.position.1, seed.position.1);
                let delta_r = (delta_eta.powi(2) + delta_phi.powi(2)).sqrt();

                if delta_r < cone_radius {
                    jet_hits.push(hit.hit_id);
                    jet_energy += hit.energy;
                    weighted_eta += hit.position.0 * hit.energy;
                    weighted_phi += hit.position.1 * hit.energy;
                    used_hits.insert(hit.hit_id);
                }
            }

            if jet_hits.len() >= 2 {
                let avg_eta = weighted_eta / jet_energy;
                let avg_phi = weighted_phi / jet_energy;

                let jet = Jet {
                    jet_id: jets.len() as u64,
                    constituent_hits: jet_hits.clone(),
                    momentum: FourMomentum::new(
                        jet_energy * avg_eta.cosh(),
                        jet_energy * avg_phi.cos() * avg_eta.cosh().recip(),
                        jet_energy * avg_phi.sin() * avg_eta.cosh().recip(),
                        jet_energy * avg_eta.tanh(),
                    ),
                    cone_radius,
                    n_constituents: jet_hits.len(),
                };

                jets.push(jet);
            }
        }

        self.jets = jets.clone();
        jets
    }

    /// Calculate missing transverse energy (MET)
    pub fn missing_et(&self) -> f64 {
        let mut px_sum = 0.0;
        let mut py_sum = 0.0;

        for hit in &self.hits {
            let pt = hit.energy / hit.position.0.cosh();
            px_sum += pt * hit.position.1.cos();
            py_sum += pt * hit.position.1.sin();
        }

        (px_sum.powi(2) + py_sum.powi(2)).sqrt()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MUON SPECTROMETER
// ═══════════════════════════════════════════════════════════════════════════════

/// Muon Spectrometer - detects penetrating particles
///
/// Like the ATLAS muon system:
/// - Outermost detector
/// - Only penetrating particles reach here
/// - In attention: long-range dependencies, anomalies
#[derive(Debug, Clone)]
pub struct MuonSpectrometer {
    /// Number of stations
    pub n_stations: usize,
    /// Magnetic field for momentum measurement
    pub b_field: f64,
    /// Recorded hits
    pub hits: Vec<DetectorHit>,
    /// Reconstructed muons
    pub muons: Vec<ReconstructedMuon>,
    /// Anomaly triggers
    pub triggers: Vec<AnomalyTrigger>,
    /// Next hit ID
    next_hit_id: u64,
}

/// A reconstructed muon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructedMuon {
    /// Muon ID
    pub muon_id: u64,
    /// Momentum
    pub momentum: FourMomentum,
    /// Quality score
    pub quality: f64,
    /// Is this a cosmic ray muon (background)?
    pub is_cosmic: bool,
}

/// An anomaly trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyTrigger {
    /// Trigger ID
    pub trigger_id: u64,
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Time of trigger
    pub time: u64,
    /// Associated particle ID
    pub particle_id: Option<u64>,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Measured value
    pub measured: f64,
}

/// Types of anomaly triggers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerType {
    /// High pT muon
    HighPtMuon,
    /// Multiple muons
    MultiMuon,
    /// Out of time (non-collision)
    OutOfTime,
    /// Unusual trajectory
    UnusualTrajectory,
    /// High multiplicity
    HighMultiplicity,
}

impl MuonSpectrometer {
    /// Create a new muon spectrometer
    pub fn new(n_stations: usize, b_field: f64) -> Self {
        Self {
            n_stations,
            b_field,
            hits: Vec::new(),
            muons: Vec::new(),
            triggers: Vec::new(),
            next_hit_id: 0,
        }
    }

    /// Record a particle (only "muon-like" particles penetrate this far)
    pub fn record_particle(&mut self, particle: &Particle, time: u64) -> Option<u64> {
        // Only high-momentum or gradient particles reach the muon system
        let is_penetrating = particle.momentum.transverse_momentum() > 5.0
            || matches!(particle.flavor, ParticleFlavor::Gradient);

        if !is_penetrating {
            return None;
        }

        let eta = particle.momentum.pseudorapidity();
        let phi = particle.momentum.azimuthal_angle();

        let hit_id = self.next_hit_id;
        self.next_hit_id += 1;

        let hit = DetectorHit::new(
            hit_id,
            DetectorLayerType::MuonSpectrometer,
            (eta, phi),
            particle.energy(),
            time,
        )
        .with_particle(particle.id)
        .with_property("pt", particle.momentum.transverse_momentum());

        self.hits.push(hit.clone());

        // Check for triggers
        self.check_triggers(&hit, particle, time);

        Some(hit_id)
    }

    /// Check for anomaly triggers
    fn check_triggers(&mut self, _hit: &DetectorHit, particle: &Particle, time: u64) {
        let pt = particle.momentum.transverse_momentum();

        // High pT trigger
        if pt > 20.0 {
            self.triggers.push(AnomalyTrigger {
                trigger_id: self.triggers.len() as u64,
                trigger_type: TriggerType::HighPtMuon,
                time,
                particle_id: Some(particle.id),
                threshold: 20.0,
                measured: pt,
            });
        }

        // High multiplicity trigger
        if self.hits.len() > 10 {
            self.triggers.push(AnomalyTrigger {
                trigger_id: self.triggers.len() as u64,
                trigger_type: TriggerType::HighMultiplicity,
                time,
                particle_id: None,
                threshold: 10.0,
                measured: self.hits.len() as f64,
            });
        }
    }

    /// Reconstruct muons from hits
    pub fn reconstruct_muons(&mut self) -> Vec<ReconstructedMuon> {
        let mut muons = Vec::new();

        // Group hits by particle ID
        let mut particle_hits: HashMap<u64, Vec<&DetectorHit>> = HashMap::new();
        for hit in &self.hits {
            if let Some(pid) = hit.particle_id {
                particle_hits.entry(pid).or_default().push(hit);
            }
        }

        for (particle_id, hits) in particle_hits {
            if hits.len() >= 2 {
                // Average momentum from hits
                let avg_pt: f64 = hits
                    .iter()
                    .filter_map(|h| h.properties.get("pt"))
                    .sum::<f64>()
                    / hits.len() as f64;

                let avg_eta: f64 =
                    hits.iter().map(|h| h.position.0).sum::<f64>() / hits.len() as f64;
                let avg_phi: f64 =
                    hits.iter().map(|h| h.position.1).sum::<f64>() / hits.len() as f64;

                let momentum = FourMomentum::new(
                    avg_pt * avg_eta.cosh(),
                    avg_pt * avg_phi.cos(),
                    avg_pt * avg_phi.sin(),
                    avg_pt * avg_eta.sinh(),
                );

                let muon = ReconstructedMuon {
                    muon_id: particle_id,
                    momentum,
                    quality: 1.0 / (hits.len() as f64), // More hits = better quality
                    is_cosmic: false,
                };

                muons.push(muon);
            }
        }

        self.muons = muons.clone();
        muons
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete detector system with all layers
#[derive(Debug)]
pub struct TorusColliderDetector {
    /// Inner tracker
    pub tracker: InnerTracker,
    /// EM calorimeter
    pub ecal: ElectromagneticCalorimeter,
    /// Hadronic calorimeter
    pub hcal: HadronicCalorimeter,
    /// Muon spectrometer
    pub muon: MuonSpectrometer,
    /// Current event number
    pub event_number: u64,
    /// Global time counter
    pub time: u64,
}

impl TorusColliderDetector {
    /// Create a new detector with default configuration
    pub fn new() -> Self {
        Self {
            tracker: InnerTracker::new(4, 2.0), // 4 layers, 2T field
            ecal: ElectromagneticCalorimeter::new(100, 64, 0.1),
            hcal: HadronicCalorimeter::new(50, 32, 0.5),
            muon: MuonSpectrometer::new(3, 0.5),
            event_number: 0,
            time: 0,
        }
    }

    /// Record a collision event
    pub fn record_event(&mut self, event: &CollisionEvent) {
        self.event_number += 1;
        self.time += 1;

        // Record all outgoing particles
        for particle in &event.outgoing {
            self.record_particle(particle);
        }
    }

    /// Record a single particle through all detector layers
    pub fn record_particle(&mut self, particle: &Particle) {
        let time = self.time;

        // Tracker (all charged particles)
        if particle.charge() != 0 {
            self.tracker.record_particle(particle, time);
        }

        // EM Calorimeter
        self.ecal.record_particle(particle, time);

        // Hadronic Calorimeter
        self.hcal.record_particle(particle, time);

        // Muon Spectrometer (penetrating particles only)
        self.muon.record_particle(particle, time);
    }

    /// Record a particle beam
    pub fn record_beam(&mut self, beam: &ParticleBeam) {
        for particle in &beam.particles {
            self.record_particle(particle);
        }
    }

    /// Reconstruct all physics objects
    pub fn reconstruct(&mut self) -> ReconstructionResult {
        let tracks = self.tracker.reconstruct_tracks();
        let primary_vertex = self.tracker.find_primary_vertex();
        let em_clusters = self.ecal.find_clusters(0.5);
        let jets = self.hcal.reconstruct_jets(0.4, 10.0);
        let missing_et = self.hcal.missing_et();
        let muons = self.muon.reconstruct_muons();
        let triggers = self.muon.triggers.clone();

        ReconstructionResult {
            event_number: self.event_number,
            tracks,
            primary_vertex,
            em_clusters,
            jets,
            missing_et,
            muons,
            triggers,
            total_energy: self.ecal.total_energy()
                + self
                    .hcal
                    .energy_grid
                    .iter()
                    .flat_map(|r| r.iter())
                    .sum::<f64>(),
        }
    }

    /// Reset detector for new event
    pub fn reset(&mut self) {
        self.tracker.hits.clear();
        self.tracker.tracks.clear();
        self.ecal.hits.clear();
        self.ecal.energy_grid = vec![vec![0.0; self.ecal.n_phi]; self.ecal.n_eta];
        self.hcal.hits.clear();
        self.hcal.energy_grid = vec![vec![0.0; self.hcal.n_phi]; self.hcal.n_eta];
        self.hcal.jets.clear();
        self.muon.hits.clear();
        self.muon.muons.clear();
        self.muon.triggers.clear();
    }

    /// Get detector summary
    pub fn summary(&self) -> String {
        format!(
            "Torus Collider Detector - Event {}\n\
             ├─ Tracker: {} hits, {} tracks\n\
             ├─ ECAL: {} hits, {:.2} GeV total\n\
             ├─ HCAL: {} hits, {} jets\n\
             └─ Muon: {} hits, {} muons, {} triggers",
            self.event_number,
            self.tracker.hits.len(),
            self.tracker.tracks.len(),
            self.ecal.hits.len(),
            self.ecal.total_energy(),
            self.hcal.hits.len(),
            self.hcal.jets.len(),
            self.muon.hits.len(),
            self.muon.muons.len(),
            self.muon.triggers.len()
        )
    }
}

impl Default for TorusColliderDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of event reconstruction
#[derive(Debug, Clone)]
pub struct ReconstructionResult {
    /// Event number
    pub event_number: u64,
    /// Reconstructed tracks
    pub tracks: Vec<Track>,
    /// Primary vertex
    pub primary_vertex: Option<TorusCoordinate>,
    /// EM clusters
    pub em_clusters: Vec<EnergyCluster>,
    /// Jets
    pub jets: Vec<Jet>,
    /// Missing transverse energy
    pub missing_et: f64,
    /// Muons
    pub muons: Vec<ReconstructedMuon>,
    /// Triggers
    pub triggers: Vec<AnomalyTrigger>,
    /// Total reconstructed energy
    pub total_energy: f64,
}

impl ReconstructionResult {
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Event {} Reconstruction:\n\
             ├─ {} tracks\n\
             ├─ {} EM clusters\n\
             ├─ {} jets\n\
             ├─ {} muons\n\
             ├─ MET: {:.2} GeV\n\
             ├─ Total E: {:.2} GeV\n\
             └─ {} triggers",
            self.event_number,
            self.tracks.len(),
            self.em_clusters.len(),
            self.jets.len(),
            self.muons.len(),
            self.missing_et,
            self.total_energy,
            self.triggers.len()
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple pseudo-random number generator (deterministic for testing)
fn rand_like(seed: u64) -> f64 {
    let x = seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
    x as f64 / u64::MAX as f64
}

/// Angular difference accounting for wrap-around
fn angular_diff(phi1: f64, phi2: f64) -> f64 {
    let mut diff = phi1 - phi2;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }
    diff
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_layer_types() {
        let layers = DetectorLayerType::all();
        assert_eq!(layers.len(), 4);
        assert!(layers[0].radius() < layers[3].radius());
    }

    #[test]
    fn test_inner_tracker() {
        let mut tracker = InnerTracker::new(4, 2.0);

        let particle = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 5.0, 3.0, 1.0),
            TorusCoordinate::new(0.0, 0.0),
        );

        let hits = tracker.record_particle(&particle, 0);
        assert_eq!(hits.len(), 4); // One hit per layer
    }

    #[test]
    fn test_em_calorimeter() {
        let mut ecal = ElectromagneticCalorimeter::new(100, 64, 0.1);

        let particle = Particle::new(
            0,
            ParticleFlavor::Attention,
            FourMomentum::new(10.0, 5.0, 3.0, 1.0),
            TorusCoordinate::new(0.0, 0.0),
        );

        let hit_id = ecal.record_particle(&particle, 0);
        assert!(hit_id.is_some());
        assert!(ecal.total_energy() > 0.0);
    }

    #[test]
    fn test_full_detector() {
        let mut detector = TorusColliderDetector::new();

        let particle = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(50.0, 25.0, 15.0, 10.0),
            TorusCoordinate::new(0.0, 0.0),
        );

        detector.record_particle(&particle);
        let result = detector.reconstruct();

        println!("{}", result.summary());
        assert!(result.total_energy > 0.0);
    }

    #[test]
    fn test_jet_reconstruction() {
        let mut hcal = HadronicCalorimeter::new(50, 32, 0.1);

        // Add several hits close together
        for i in 0..5 {
            let particle = Particle::new(
                i,
                ParticleFlavor::Value,
                FourMomentum::new(10.0, 5.0 + i as f64 * 0.1, 3.0, 1.0),
                TorusCoordinate::new(0.0, 0.0),
            );
            hcal.record_particle(&particle, 0);
        }

        let jets = hcal.reconstruct_jets(0.4, 5.0);
        assert!(!jets.is_empty());
    }
}
