//! Feynman Vertices and Interaction Couplings
//!
//! Models attention computation as particle interactions at vertices:
//!
//! ```text
//!        Query (q)
//!           │
//!           │  coupling g_QK
//!           ▼
//!     ══════╋══════  QK Vertex (dot product)
//!           │
//!           │  emits Attention boson
//!           ▼
//!      Attention (A)
//!           │
//!           │  coupling g_AV
//!           ▼
//!     ══════╋══════  AV Vertex (weighted sum)
//!           │
//!           ▼
//!       Output (O)
//! ```
//!
//! # Feynman Rules for Attention
//!
//! 1. **QK Vertex**: Query meets Key → Attention weight
//!    - Coupling: g_QK = 1/√d_k (scaled dot product)
//!    - Matrix element: M = g_QK * (Q · K)
//!
//! 2. **AV Vertex**: Attention weight × Value → Output
//!    - Coupling: g_AV = 1.0 (direct multiplication)
//!    - Matrix element: M = A * V
//!
//! 3. **Propagators**: Internal lines carry momentum
//!    - Fermion propagator: i/(p̸ - m)
//!    - Boson propagator: -i*g_μν/(p² - m²)

use crate::collider::particles::{FourMomentum, Particle};
use crate::geometry::TorusCoordinate;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of interaction vertices in attention computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexType {
    /// Query-Key interaction producing attention weight
    /// Q + K → A (like e⁺e⁻ → γ)
    QueryKey,

    /// Attention-Value interaction producing output
    /// A + V → O (like γ + e → e)
    AttentionValue,

    /// Stream mixing vertex (8 streams combine)
    /// Σ streams → combined (like gluon self-interaction)
    StreamMixing,

    /// Layer transition (residual connection)
    /// X + F(X) → X' (like Compton scattering)
    LayerTransition,

    /// EMA compounding vertex
    /// α*new + (1-α)*old → compound
    Compounding,

    /// Gradient vertex (backpropagation)
    /// ∂L/∂x flows backward
    Gradient,
}

impl VertexType {
    /// Get the number of incoming particles
    pub fn n_incoming(&self) -> usize {
        match self {
            VertexType::QueryKey => 2,
            VertexType::AttentionValue => 2,
            VertexType::StreamMixing => 8,
            VertexType::LayerTransition => 2,
            VertexType::Compounding => 2,
            VertexType::Gradient => 1,
        }
    }

    /// Get the number of outgoing particles
    pub fn n_outgoing(&self) -> usize {
        match self {
            VertexType::QueryKey => 1,       // Attention boson
            VertexType::AttentionValue => 1, // Output
            VertexType::StreamMixing => 1,   // Combined output
            VertexType::LayerTransition => 1,
            VertexType::Compounding => 1,
            VertexType::Gradient => 2, // Gradient splits to multiple paths
        }
    }

    /// Get the base coupling constant
    pub fn base_coupling(&self) -> f64 {
        match self {
            VertexType::QueryKey => 1.0,        // Scaled by 1/√d_k later
            VertexType::AttentionValue => 1.0,  // Direct multiplication
            VertexType::StreamMixing => 0.125,  // 1/8 for 8 streams
            VertexType::LayerTransition => 1.0, // Residual connection
            VertexType::Compounding => 0.9,     // Default EMA alpha
            VertexType::Gradient => 1.0,        // Chain rule
        }
    }

    /// Get the name of this vertex type
    pub fn name(&self) -> &'static str {
        match self {
            VertexType::QueryKey => "QK",
            VertexType::AttentionValue => "AV",
            VertexType::StreamMixing => "MIX",
            VertexType::LayerTransition => "RES",
            VertexType::Compounding => "EMA",
            VertexType::Gradient => "GRAD",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COUPLING CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Coupling constants for vertex interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingConstants {
    /// QK coupling: 1/√d_k (attention scaling)
    pub g_qk: f64,

    /// AV coupling: direct multiplication weight
    pub g_av: f64,

    /// Stream mixing coupling (softmax temperature)
    pub g_mix: f64,

    /// Residual connection strength
    pub g_res: f64,

    /// EMA compounding alpha
    pub g_ema: f64,

    /// Gradient flow coupling
    pub g_grad: f64,

    /// Fine structure constant analog (overall interaction strength)
    pub alpha: f64,
}

impl CouplingConstants {
    /// Create coupling constants for a given model dimension
    pub fn for_model(d_model: usize, n_heads: usize) -> Self {
        let d_k = d_model / n_heads;
        Self {
            g_qk: 1.0 / (d_k as f64).sqrt(),
            g_av: 1.0,
            g_mix: 1.0,
            g_res: 1.0,
            g_ema: 0.9,
            g_grad: 1.0,
            alpha: 1.0 / 137.0, // Fine structure constant (for fun)
        }
    }

    /// Get the effective coupling for a vertex type
    pub fn coupling_for(&self, vertex_type: VertexType) -> f64 {
        match vertex_type {
            VertexType::QueryKey => self.g_qk,
            VertexType::AttentionValue => self.g_av,
            VertexType::StreamMixing => self.g_mix,
            VertexType::LayerTransition => self.g_res,
            VertexType::Compounding => self.g_ema,
            VertexType::Gradient => self.g_grad,
        }
    }

    /// Compute running coupling (like QCD running coupling)
    /// Coupling "runs" with energy scale (layer depth)
    pub fn running_coupling(&self, vertex_type: VertexType, layer: usize, n_layers: usize) -> f64 {
        let base = self.coupling_for(vertex_type);

        // Asymptotic freedom-like behavior: coupling decreases at high energy (deep layers)
        let scale = 1.0 - 0.1 * (layer as f64 / n_layers as f64);
        base * scale.max(0.5)
    }
}

impl Default for CouplingConstants {
    fn default() -> Self {
        Self::for_model(256, 8)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MANDELSTAM VARIABLES
// ═══════════════════════════════════════════════════════════════════════════════

/// Mandelstam variables for 2→2 scattering
///
/// For process: a + b → c + d
/// - s = (p_a + p_b)² = center-of-mass energy squared
/// - t = (p_a - p_c)² = momentum transfer squared
/// - u = (p_a - p_d)² = crossing variable
///
/// Satisfy: s + t + u = Σm²
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MandelstamVariables {
    /// s-channel: (p1 + p2)² - center of mass energy squared
    pub s: f64,
    /// t-channel: (p1 - p3)² - momentum transfer squared
    pub t: f64,
    /// u-channel: (p1 - p4)² - crossing variable
    pub u: f64,
}

impl MandelstamVariables {
    /// Compute Mandelstam variables for 2→2 scattering
    pub fn compute(
        p1: &FourMomentum,
        p2: &FourMomentum,
        p3: &FourMomentum,
        p4: &FourMomentum,
    ) -> Self {
        let s_sum = p1.add(p2);
        let t_diff = p1.subtract(p3);
        let u_diff = p1.subtract(p4);

        Self {
            s: s_sum.mass_squared(),
            t: t_diff.mass_squared(),
            u: u_diff.mass_squared(),
        }
    }

    /// Compute for 2→1 process (like Q + K → A)
    pub fn compute_2to1(p1: &FourMomentum, p2: &FourMomentum) -> Self {
        let s_sum = p1.add(p2);
        Self {
            s: s_sum.mass_squared(),
            t: 0.0,
            u: 0.0,
        }
    }

    /// Get center-of-mass energy √s
    pub fn sqrt_s(&self) -> f64 {
        if self.s >= 0.0 {
            self.s.sqrt()
        } else {
            0.0
        }
    }

    /// Check if this is a valid physical configuration
    /// (s > 0 for timelike, masses satisfy constraints)
    pub fn is_physical(&self) -> bool {
        self.s >= 0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLISION EVENT
// ═══════════════════════════════════════════════════════════════════════════════

/// A collision event at a vertex
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    /// Unique event ID
    pub event_id: u64,

    /// Type of vertex where collision occurred
    pub vertex_type: VertexType,

    /// Incoming particles
    pub incoming: Vec<Particle>,

    /// Outgoing particles
    pub outgoing: Vec<Particle>,

    /// Position on torus where collision occurred
    pub vertex_position: TorusCoordinate,

    /// Mandelstam variables (if applicable)
    pub mandelstam: Option<MandelstamVariables>,

    /// Matrix element squared |M|²
    pub matrix_element_squared: f64,

    /// Cross section (probability of this interaction)
    pub cross_section: f64,

    /// Layer in transformer where this occurred
    pub layer: usize,

    /// Head index (for multi-head attention)
    pub head: usize,

    /// Stream ID
    pub stream_id: usize,

    /// Timestamp (step in computation)
    pub timestamp: u64,
}

impl CollisionEvent {
    /// Create a new collision event
    pub fn new(
        event_id: u64,
        vertex_type: VertexType,
        vertex_position: TorusCoordinate,
        layer: usize,
        head: usize,
        stream_id: usize,
    ) -> Self {
        Self {
            event_id,
            vertex_type,
            incoming: Vec::new(),
            outgoing: Vec::new(),
            vertex_position,
            mandelstam: None,
            matrix_element_squared: 0.0,
            cross_section: 0.0,
            layer,
            head,
            stream_id,
            timestamp: 0,
        }
    }

    /// Add an incoming particle
    pub fn add_incoming(&mut self, particle: Particle) {
        self.incoming.push(particle);
    }

    /// Add an outgoing particle
    pub fn add_outgoing(&mut self, particle: Particle) {
        self.outgoing.push(particle);
    }

    /// Compute total incoming four-momentum
    pub fn total_incoming_momentum(&self) -> FourMomentum {
        self.incoming
            .iter()
            .fold(FourMomentum::new(0.0, 0.0, 0.0, 0.0), |acc, p| {
                acc.add(&p.momentum)
            })
    }

    /// Compute total outgoing four-momentum
    pub fn total_outgoing_momentum(&self) -> FourMomentum {
        self.outgoing
            .iter()
            .fold(FourMomentum::new(0.0, 0.0, 0.0, 0.0), |acc, p| {
                acc.add(&p.momentum)
            })
    }

    /// Check four-momentum conservation
    pub fn check_momentum_conservation(&self, tolerance: f64) -> bool {
        let p_in = self.total_incoming_momentum();
        let p_out = self.total_outgoing_momentum();

        let de = (p_in.e - p_out.e).abs();
        let dpx = (p_in.px - p_out.px).abs();
        let dpy = (p_in.py - p_out.py).abs();
        let dpz = (p_in.pz - p_out.pz).abs();

        de < tolerance && dpx < tolerance && dpy < tolerance && dpz < tolerance
    }

    /// Get total incoming charge
    pub fn total_incoming_charge(&self) -> i32 {
        self.incoming.iter().map(|p| p.charge()).sum()
    }

    /// Get total outgoing charge
    pub fn total_outgoing_charge(&self) -> i32 {
        self.outgoing.iter().map(|p| p.charge()).sum()
    }

    /// Check charge conservation
    pub fn check_charge_conservation(&self) -> bool {
        self.total_incoming_charge() == self.total_outgoing_charge()
    }

    /// Compute center-of-mass energy
    pub fn center_of_mass_energy(&self) -> f64 {
        let p_total = self.total_incoming_momentum();
        let s = p_total.mass_squared();
        if s >= 0.0 {
            s.sqrt()
        } else {
            0.0
        }
    }

    /// Compute invariant mass of outgoing particles
    pub fn invariant_mass(&self) -> f64 {
        self.total_outgoing_momentum().mass()
    }

    /// Check if this is an elastic collision (same particles in and out)
    pub fn is_elastic(&self) -> bool {
        if self.incoming.len() != self.outgoing.len() {
            return false;
        }
        // Simple check: same number of each flavor type
        let incoming_flavors: Vec<_> = self.incoming.iter().map(|p| p.flavor).collect();
        let outgoing_flavors: Vec<_> = self.outgoing.iter().map(|p| p.flavor).collect();
        incoming_flavors == outgoing_flavors
    }

    /// Compute scattering angle (angle between incoming and outgoing momenta)
    pub fn scattering_angle(&self) -> Option<f64> {
        if self.incoming.is_empty() || self.outgoing.is_empty() {
            return None;
        }

        let p_in = self.total_incoming_momentum();
        let p_out = self.total_outgoing_momentum();

        // Compute angle between 3-momenta
        let dot = p_in.px * p_out.px + p_in.py * p_out.py + p_in.pz * p_out.pz;
        let mag_in = p_in.three_momentum_magnitude();
        let mag_out = p_out.three_momentum_magnitude();

        if mag_in > 1e-10 && mag_out > 1e-10 {
            let cos_theta = (dot / (mag_in * mag_out)).clamp(-1.0, 1.0);
            Some(cos_theta.acos())
        } else {
            None
        }
    }

    /// Get momentum transfer squared (Q²)
    pub fn q_squared(&self) -> Option<f64> {
        if self.incoming.is_empty() || self.outgoing.is_empty() {
            return None;
        }
        let q = self
            .total_incoming_momentum()
            .subtract(&self.total_outgoing_momentum());
        Some(-q.mass_squared()) // Q² = -t (negative of Mandelstam t)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEYNMAN VERTEX
// ═══════════════════════════════════════════════════════════════════════════════

/// A Feynman vertex representing an interaction point
#[derive(Debug, Clone)]
pub struct FeynmanVertex {
    /// Vertex type
    pub vertex_type: VertexType,

    /// Coupling constant at this vertex
    pub coupling: f64,

    /// Position on the torus manifold
    pub position: TorusCoordinate,

    /// Layer index
    pub layer: usize,

    /// Whether this vertex has been evaluated
    pub evaluated: bool,

    /// Result of evaluation (matrix element)
    pub matrix_element: Option<f64>,
}

impl FeynmanVertex {
    /// Create a new vertex
    pub fn new(
        vertex_type: VertexType,
        coupling: f64,
        position: TorusCoordinate,
        layer: usize,
    ) -> Self {
        Self {
            vertex_type,
            coupling,
            position,
            layer,
            evaluated: false,
            matrix_element: None,
        }
    }

    /// Evaluate the vertex with given incoming momenta
    /// Returns the matrix element M
    pub fn evaluate(&mut self, incoming: &[FourMomentum]) -> f64 {
        let m = match self.vertex_type {
            VertexType::QueryKey => {
                // M = g * (Q · K) = g * Σ q_i * k_i
                if incoming.len() >= 2 {
                    let q = &incoming[0];
                    let k = &incoming[1];
                    self.coupling * q.dot(k)
                } else {
                    0.0
                }
            }

            VertexType::AttentionValue => {
                // M = A * V (direct multiplication)
                if incoming.len() >= 2 {
                    self.coupling * incoming[0].e * incoming[1].e
                } else {
                    0.0
                }
            }

            VertexType::StreamMixing => {
                // M = Σ w_i * stream_i (weighted sum)
                let total_energy: f64 = incoming.iter().map(|p| p.e).sum();
                self.coupling * total_energy / incoming.len() as f64
            }

            VertexType::LayerTransition => {
                // M = x + f(x) (residual)
                if incoming.len() >= 2 {
                    self.coupling * (incoming[0].e + incoming[1].e)
                } else if !incoming.is_empty() {
                    self.coupling * incoming[0].e
                } else {
                    0.0
                }
            }

            VertexType::Compounding => {
                // M = α * new + (1-α) * old
                if incoming.len() >= 2 {
                    let alpha = self.coupling;
                    alpha * incoming[0].e + (1.0 - alpha) * incoming[1].e
                } else {
                    0.0
                }
            }

            VertexType::Gradient => {
                // M = ∂L/∂x (chain rule)
                if !incoming.is_empty() {
                    self.coupling * incoming[0].e
                } else {
                    0.0
                }
            }
        };

        self.matrix_element = Some(m);
        self.evaluated = true;
        m
    }

    /// Compute the differential cross section
    /// dσ/dΩ ∝ |M|² / s
    pub fn differential_cross_section(&self, s: f64) -> f64 {
        match self.matrix_element {
            Some(m) => {
                if s > 1e-10 {
                    m.powi(2) / s
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    /// Get the vertex factor (Feynman rule)
    pub fn vertex_factor(&self) -> f64 {
        // In QFT, vertex factor is -i * g
        // We use real numbers, so just g
        self.coupling
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEYNMAN DIAGRAM
// ═══════════════════════════════════════════════════════════════════════════════

/// A complete Feynman diagram for an attention computation
#[derive(Debug, Clone)]
pub struct FeynmanDiagram {
    /// Diagram ID
    pub id: u64,

    /// All vertices in this diagram
    pub vertices: Vec<FeynmanVertex>,

    /// External incoming particles
    pub external_incoming: Vec<Particle>,

    /// External outgoing particles
    pub external_outgoing: Vec<Particle>,

    /// Internal propagators (virtual particles)
    pub propagators: Vec<Propagator>,

    /// Total amplitude (product of all matrix elements)
    pub amplitude: Option<f64>,

    /// Layer this diagram represents
    pub layer: usize,
}

impl FeynmanDiagram {
    /// Create a new diagram
    pub fn new(id: u64, layer: usize) -> Self {
        Self {
            id,
            vertices: Vec::new(),
            external_incoming: Vec::new(),
            external_outgoing: Vec::new(),
            propagators: Vec::new(),
            amplitude: None,
            layer,
        }
    }

    /// Create a QK→A diagram (Query-Key to Attention)
    pub fn qk_diagram(
        id: u64,
        query: Particle,
        key: Particle,
        coupling: f64,
        layer: usize,
    ) -> Self {
        let position = TorusCoordinate::new(
            (query.position.u + key.position.u) / 2.0,
            (query.position.v + key.position.v) / 2.0,
        );

        let vertex = FeynmanVertex::new(VertexType::QueryKey, coupling, position, layer);

        Self {
            id,
            vertices: vec![vertex],
            external_incoming: vec![query, key],
            external_outgoing: Vec::new(), // Filled after evaluation
            propagators: Vec::new(),
            amplitude: None,
            layer,
        }
    }

    /// Add a vertex to the diagram
    pub fn add_vertex(&mut self, vertex: FeynmanVertex) {
        self.vertices.push(vertex);
    }

    /// Add an external incoming particle
    pub fn add_incoming(&mut self, particle: Particle) {
        self.external_incoming.push(particle);
    }

    /// Add an external outgoing particle
    pub fn add_outgoing(&mut self, particle: Particle) {
        self.external_outgoing.push(particle);
    }

    /// Evaluate the diagram and compute amplitude
    pub fn evaluate(&mut self) -> f64 {
        let incoming_momenta: Vec<FourMomentum> =
            self.external_incoming.iter().map(|p| p.momentum).collect();

        let mut total_amplitude = 1.0;

        // Evaluate each vertex
        for vertex in &mut self.vertices {
            let m = vertex.evaluate(&incoming_momenta);
            total_amplitude *= m;
        }

        // Multiply by propagator factors
        for prop in &self.propagators {
            total_amplitude *= prop.propagator_factor();
        }

        self.amplitude = Some(total_amplitude);
        total_amplitude
    }

    /// Get the squared amplitude |M|²
    pub fn amplitude_squared(&self) -> f64 {
        self.amplitude.map(|a| a.powi(2)).unwrap_or(0.0)
    }

    /// Compute cross section
    pub fn cross_section(&self) -> f64 {
        let s = self
            .external_incoming
            .iter()
            .fold(FourMomentum::new(0.0, 0.0, 0.0, 0.0), |acc, p| {
                acc.add(&p.momentum)
            })
            .mass_squared();

        if s > 1e-10 {
            self.amplitude_squared() / s
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPAGATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// A propagator (internal line in Feynman diagram)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Propagator {
    /// Four-momentum flowing through propagator
    pub momentum: FourMomentum,

    /// Mass of the propagating particle
    pub mass: f64,

    /// Whether this is a fermion (affects sign)
    pub is_fermion: bool,

    /// Decay width (for unstable particles)
    pub width: f64,
}

impl Propagator {
    /// Create a new propagator
    pub fn new(momentum: FourMomentum, mass: f64, is_fermion: bool) -> Self {
        Self {
            momentum,
            mass,
            is_fermion,
            width: 0.0,
        }
    }

    /// Create a massless propagator (like photon/gluon)
    pub fn massless(momentum: FourMomentum) -> Self {
        Self::new(momentum, 0.0, false)
    }

    /// Create a fermion propagator
    pub fn fermion(momentum: FourMomentum, mass: f64) -> Self {
        Self::new(momentum, mass, true)
    }

    /// Compute the propagator factor
    ///
    /// Scalar: 1/(p² - m²)
    /// Fermion: 1/(p̸ - m) ≈ (p̸ + m)/(p² - m²)
    /// With width: 1/(p² - m² + imΓ)
    pub fn propagator_factor(&self) -> f64 {
        let p2 = self.momentum.mass_squared();
        let m2 = self.mass.powi(2);

        let denominator = if self.width > 0.0 {
            // Breit-Wigner form (magnitude only, ignoring phase)
            let real = p2 - m2;
            let imag = self.mass * self.width;
            (real.powi(2) + imag.powi(2)).sqrt()
        } else {
            (p2 - m2).abs().max(1e-10) // Avoid division by zero
        };

        1.0 / denominator
    }

    /// Check if this is on-shell (p² = m²)
    pub fn is_on_shell(&self, tolerance: f64) -> bool {
        let p2 = self.momentum.mass_squared();
        let m2 = self.mass.powi(2);
        (p2 - m2).abs() < tolerance
    }

    /// Check if this is a virtual particle (off-shell)
    pub fn is_virtual(&self, tolerance: f64) -> bool {
        !self.is_on_shell(tolerance)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS SECTION CALCULATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Computes cross sections for various processes
pub struct CrossSectionCalculator {
    /// Coupling constants
    pub couplings: CouplingConstants,
}

impl CrossSectionCalculator {
    /// Create a new calculator
    pub fn new(couplings: CouplingConstants) -> Self {
        Self { couplings }
    }

    /// Compute QK→A cross section (Query + Key → Attention)
    ///
    /// σ(QK→A) ∝ g²_QK * |Q·K|² / s
    pub fn qk_cross_section(&self, q: &FourMomentum, k: &FourMomentum) -> f64 {
        let s = q.add(k).mass_squared();
        if s <= 0.0 {
            return 0.0;
        }

        let qk_dot = q.dot(k);
        let g = self.couplings.g_qk;

        g.powi(2) * qk_dot.powi(2) / s
    }

    /// Compute AV→O cross section (Attention + Value → Output)
    pub fn av_cross_section(&self, a: &FourMomentum, v: &FourMomentum) -> f64 {
        let s = a.add(v).mass_squared();
        if s <= 0.0 {
            return 0.0;
        }

        let g = self.couplings.g_av;
        g.powi(2) * a.e * v.e / s
    }

    /// Compute total cross section for attention layer
    pub fn attention_layer_cross_section(
        &self,
        queries: &[FourMomentum],
        keys: &[FourMomentum],
        values: &[FourMomentum],
    ) -> f64 {
        let mut total = 0.0;

        // Sum over all Q-K pairs
        for q in queries {
            for k in keys {
                total += self.qk_cross_section(q, k);
            }
        }

        // Average over values
        if !values.is_empty() {
            total /= values.len() as f64;
        }

        total
    }

    /// Compute differential cross section dσ/dΩ
    pub fn differential_cross_section(
        &self,
        incoming: &MandelstamVariables,
        scattering_angle: f64,
    ) -> f64 {
        let s = incoming.s;
        if s <= 0.0 {
            return 0.0;
        }

        // Rutherford-like angular dependence
        let cos_theta = scattering_angle.cos();
        let angular_factor = 1.0 / (1.0 - cos_theta + 0.01).powi(2);

        self.couplings.alpha.powi(2) * angular_factor / s
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::particles::ParticleFlavor;
    use std::f64::consts::PI;

    #[test]
    fn test_vertex_type_properties() {
        assert_eq!(VertexType::QueryKey.n_incoming(), 2);
        assert_eq!(VertexType::QueryKey.n_outgoing(), 1);
        assert_eq!(VertexType::StreamMixing.n_incoming(), 8);
    }

    #[test]
    fn test_coupling_constants() {
        let c = CouplingConstants::for_model(256, 8);
        // d_k = 256/8 = 32, g_qk = 1/√32 ≈ 0.177
        assert!((c.g_qk - 1.0 / 32.0_f64.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_mandelstam_variables() {
        // Two massless particles colliding head-on
        let p1 = FourMomentum::new(10.0, 10.0, 0.0, 0.0);
        let p2 = FourMomentum::new(10.0, -10.0, 0.0, 0.0);

        let m = MandelstamVariables::compute_2to1(&p1, &p2);

        // s = (E1 + E2)² - (p1 + p2)² = 20² - 0² = 400
        assert!((m.s - 400.0).abs() < 1e-10);
        assert!((m.sqrt_s() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_collision_event_conservation() {
        let mut event = CollisionEvent::new(
            0,
            VertexType::QueryKey,
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
            0,
        );

        // Add incoming particles
        let q = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 5.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let k = Particle::new(
            1,
            ParticleFlavor::Key,
            FourMomentum::new(10.0, -5.0, 0.0, 0.0),
            TorusCoordinate::new(PI, 0.0),
        );

        event.add_incoming(q);
        event.add_incoming(k);

        // Add outgoing particle (attention boson with same total momentum)
        let a = Particle::new(
            2,
            ParticleFlavor::Attention,
            FourMomentum::new(20.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(PI / 2.0, 0.0),
        );
        event.add_outgoing(a);

        // Check conservation
        assert!(event.check_momentum_conservation(1e-10));
        assert!(event.check_charge_conservation()); // +1 + -1 = 0
    }

    #[test]
    fn test_feynman_vertex_evaluation() {
        let mut vertex = FeynmanVertex::new(
            VertexType::QueryKey,
            0.1, // coupling
            TorusCoordinate::new(0.0, 0.0),
            0,
        );

        let q = FourMomentum::new(10.0, 1.0, 0.0, 0.0);
        let k = FourMomentum::new(10.0, 1.0, 0.0, 0.0);

        let m = vertex.evaluate(&[q, k]);

        // M = g * (Q · K) = 0.1 * (10*10 - 1*1) = 0.1 * 99 = 9.9
        assert!((m - 9.9).abs() < 0.01);
        assert!(vertex.evaluated);
    }

    #[test]
    fn test_propagator() {
        // Massless propagator
        let p = FourMomentum::new(10.0, 6.0, 8.0, 0.0); // p² = 100 - 36 - 64 = 0
        let prop = Propagator::massless(p);

        // On-shell massless particle
        assert!(prop.is_on_shell(0.1));

        // Off-shell massive propagator
        let p2 = FourMomentum::new(10.0, 0.0, 0.0, 0.0); // p² = 100
        let prop2 = Propagator::new(p2, 5.0, false); // m² = 25

        // p² - m² = 100 - 25 = 75
        assert!((1.0 / prop2.propagator_factor() - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_cross_section_calculator() {
        let calc = CrossSectionCalculator::new(CouplingConstants::default());

        let q = FourMomentum::new(10.0, 10.0, 0.0, 0.0);
        let k = FourMomentum::new(10.0, -10.0, 0.0, 0.0);

        let sigma = calc.qk_cross_section(&q, &k);

        // Should be positive
        assert!(sigma > 0.0);
    }
}
