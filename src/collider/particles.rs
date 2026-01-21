//! Particle Physics Types for Torus Collider Validator
//!
//! Models attention computation as particle collisions where:
//! - Query particles (spin ½, charge +1) seek interactions
//! - Key particles (spin ½, charge -1) respond to queries
//! - Value particles (spin 1, charge 0) carry information
//! - Attention bosons mediate Q·K interactions
//! - Gradient fermions carry backpropagation force
//!
//! # Physics Mapping
//!
//! | Neural Network | Particle Physics |
//! |----------------|------------------|
//! | Query vector   | Up-type fermion  |
//! | Key vector     | Down-type fermion|
//! | Value vector   | Higgs-like scalar|
//! | Attention weight| W/Z boson       |
//! | Gradient       | Gluon            |

use crate::geometry::TorusCoordinate;
use crate::TorusResult;
use candle_core::{Device, IndexOp, Tensor};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// FOUR-MOMENTUM (Relativistic Energy-Momentum)
// ═══════════════════════════════════════════════════════════════════════════════

/// Four-momentum vector in natural units (c = 1)
///
/// p^μ = (E, p_x, p_y, p_z)
///
/// Satisfies the mass-shell condition: E² - |p|² = m²
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FourMomentum {
    /// Energy component (timelike)
    pub e: f64,
    /// Momentum x-component (maps to major loop u)
    pub px: f64,
    /// Momentum y-component (maps to minor loop v)
    pub py: f64,
    /// Momentum z-component (maps to model dimension)
    pub pz: f64,
}

impl FourMomentum {
    /// Create a new four-momentum
    pub fn new(e: f64, px: f64, py: f64, pz: f64) -> Self {
        Self { e, px, py, pz }
    }

    /// Create from torus position and energy
    ///
    /// Maps torus coordinates to momentum space:
    /// - u (major angle) → p_x
    /// - v (minor angle) → p_y
    /// - activation magnitude → E
    pub fn from_torus(coord: &TorusCoordinate, energy: f64, pz: f64) -> Self {
        // Map angular position to momentum (periodic → bounded)
        let px = energy * coord.u.cos();
        let py = energy * coord.v.cos();
        Self::new(energy, px, py, pz)
    }

    /// Create from tensor statistics
    ///
    /// Energy = L2 norm of tensor
    /// Momentum = directional components from tensor structure
    pub fn from_tensor(tensor: &Tensor) -> TorusResult<Self> {
        let flat = tensor.flatten_all()?;
        let n = flat.elem_count();

        // Energy = L2 norm (total magnitude)
        let energy = flat.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;

        // Momentum components from tensor statistics
        let _mean = flat.mean_all()?.to_scalar::<f32>()? as f64;
        let values: Vec<f32> = flat.to_vec1()?;

        // px = mean of first quarter (major direction)
        let quarter = n / 4;
        let px: f64 = values[..quarter].iter().map(|&v| v as f64).sum::<f64>() / quarter as f64;

        // py = mean of second quarter (minor direction)
        let py: f64 = values[quarter..2 * quarter]
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>()
            / quarter as f64;

        // pz = mean of remaining (depth direction)
        let pz: f64 =
            values[2 * quarter..].iter().map(|&v| v as f64).sum::<f64>() / (n - 2 * quarter) as f64;

        Ok(Self::new(energy, px * energy, py * energy, pz * energy))
    }

    /// Compute invariant mass squared: m² = E² - |p|²
    pub fn mass_squared(&self) -> f64 {
        self.e.powi(2) - self.px.powi(2) - self.py.powi(2) - self.pz.powi(2)
    }

    /// Compute invariant mass: m = √(E² - |p|²)
    /// Returns 0 for spacelike (tachyonic) momenta
    pub fn mass(&self) -> f64 {
        let m2 = self.mass_squared();
        if m2 >= 0.0 {
            m2.sqrt()
        } else {
            0.0 // Tachyonic - used for "speed of darkness"
        }
    }

    /// Check if this is a tachyon (imaginary mass, m² < 0)
    /// Tachyons travel faster than light - represents anti-causal flow
    pub fn is_tachyonic(&self) -> bool {
        self.mass_squared() < 0.0
    }

    /// Compute 3-momentum magnitude: |p| = √(p_x² + p_y² + p_z²)
    pub fn three_momentum_magnitude(&self) -> f64 {
        (self.px.powi(2) + self.py.powi(2) + self.pz.powi(2)).sqrt()
    }

    /// Compute velocity: v = |p| / E (in natural units)
    /// Returns > 1 for tachyons (faster than light)
    pub fn velocity(&self) -> f64 {
        if self.e.abs() > 1e-10 {
            self.three_momentum_magnitude() / self.e
        } else {
            0.0
        }
    }

    /// Compute rapidity: y = 0.5 * ln((E + p_z) / (E - p_z))
    pub fn rapidity(&self) -> f64 {
        let denom = self.e - self.pz;
        if denom.abs() > 1e-10 {
            0.5 * ((self.e + self.pz) / denom).ln()
        } else {
            f64::INFINITY
        }
    }

    /// Compute transverse momentum: p_T = √(p_x² + p_y²)
    pub fn transverse_momentum(&self) -> f64 {
        (self.px.powi(2) + self.py.powi(2)).sqrt()
    }

    /// Compute pseudorapidity: η = -ln(tan(θ/2)) where θ is polar angle
    pub fn pseudorapidity(&self) -> f64 {
        let p = self.three_momentum_magnitude();
        if p.abs() > 1e-10 {
            0.5 * ((p + self.pz) / (p - self.pz + 1e-10)).ln()
        } else {
            0.0
        }
    }

    /// Compute azimuthal angle: φ = atan2(p_y, p_x)
    pub fn azimuthal_angle(&self) -> f64 {
        self.py.atan2(self.px)
    }

    /// Lorentz boost along z-axis
    pub fn boost_z(&self, beta: f64) -> Self {
        let gamma = 1.0 / (1.0 - beta.powi(2)).sqrt();
        Self {
            e: gamma * (self.e - beta * self.pz),
            px: self.px,
            py: self.py,
            pz: gamma * (self.pz - beta * self.e),
        }
    }

    /// Add two four-momenta
    pub fn add(&self, other: &Self) -> Self {
        Self {
            e: self.e + other.e,
            px: self.px + other.px,
            py: self.py + other.py,
            pz: self.pz + other.pz,
        }
    }

    /// Subtract two four-momenta
    pub fn subtract(&self, other: &Self) -> Self {
        Self {
            e: self.e - other.e,
            px: self.px - other.px,
            py: self.py - other.py,
            pz: self.pz - other.pz,
        }
    }

    /// Minkowski inner product: p · q = E₁E₂ - p⃗₁·p⃗₂
    pub fn dot(&self, other: &Self) -> f64 {
        self.e * other.e - self.px * other.px - self.py * other.py - self.pz * other.pz
    }
}

impl std::ops::Add for FourMomentum {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        FourMomentum::add(&self, &other)
    }
}

impl std::ops::Sub for FourMomentum {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        FourMomentum::subtract(&self, &other)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTICLE FLAVORS (Types of particles in the collider)
// ═══════════════════════════════════════════════════════════════════════════════

/// Particle flavor - the type/role of the particle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleFlavor {
    /// Query particle - initiates attention interaction (up-type fermion)
    Query,
    /// Key particle - responds to query (down-type fermion)
    Key,
    /// Value particle - carries information (Higgs-like scalar)
    Value,
    /// Attention boson - mediates Q·K interaction (W/Z-like)
    Attention,
    /// Gradient fermion - carries backpropagation force (gluon-like)
    Gradient,
    /// Output particle - final state after all interactions
    Output,
    /// Ghost particle - virtual intermediate state
    Ghost,
}

impl ParticleFlavor {
    /// Get the spin of this particle type
    pub fn spin(&self) -> f64 {
        match self {
            ParticleFlavor::Query => 0.5,     // Fermion
            ParticleFlavor::Key => 0.5,       // Fermion
            ParticleFlavor::Value => 1.0,     // Vector boson
            ParticleFlavor::Attention => 1.0, // Vector boson
            ParticleFlavor::Gradient => 1.0,  // Gluon (vector)
            ParticleFlavor::Output => 0.5,    // Fermion
            ParticleFlavor::Ghost => 0.0,     // Scalar ghost
        }
    }

    /// Get the electric-like charge of this particle
    pub fn charge(&self) -> i32 {
        match self {
            ParticleFlavor::Query => 1,     // Positive (seeks)
            ParticleFlavor::Key => -1,      // Negative (responds)
            ParticleFlavor::Value => 0,     // Neutral
            ParticleFlavor::Attention => 0, // Neutral boson
            ParticleFlavor::Gradient => 0,  // Color-charged but electrically neutral
            ParticleFlavor::Output => 0,    // Combined, neutral
            ParticleFlavor::Ghost => 0,     // Ghost
        }
    }

    /// Check if this is a fermion (half-integer spin)
    pub fn is_fermion(&self) -> bool {
        (self.spin() * 2.0) as i32 % 2 == 1
    }

    /// Check if this is a boson (integer spin)
    pub fn is_boson(&self) -> bool {
        !self.is_fermion()
    }

    /// Get the color charge dimension (SU(3) representation)
    /// 3 for quarks, 8 for gluons, 1 for colorless
    pub fn color_dimension(&self) -> usize {
        match self {
            ParticleFlavor::Query => 3,    // Fundamental rep
            ParticleFlavor::Key => 3,      // Fundamental rep
            ParticleFlavor::Gradient => 8, // Adjoint rep (gluon)
            _ => 1,                        // Color singlet
        }
    }

    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ParticleFlavor::Query => "query",
            ParticleFlavor::Key => "key",
            ParticleFlavor::Value => "value",
            ParticleFlavor::Attention => "attention",
            ParticleFlavor::Gradient => "gradient",
            ParticleFlavor::Output => "output",
            ParticleFlavor::Ghost => "ghost",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTICLE (Main particle struct)
// ═══════════════════════════════════════════════════════════════════════════════

/// A particle in the torus collider
///
/// Represents a tensor (Query, Key, Value, etc.) as a particle with
/// quantum numbers, momentum, and position on the torus manifold.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Unique identifier
    pub id: u64,
    /// Particle type/flavor
    pub flavor: ParticleFlavor,
    /// Four-momentum
    pub momentum: FourMomentum,
    /// Position on torus manifold
    pub position: TorusCoordinate,
    /// Color charge (SU(3) - for stream mixing)
    /// Represented as (r, g, b) with r + g + b = 0 for colorless
    pub color: [f64; 3],
    /// Stream index this particle belongs to (0-7)
    pub stream_id: usize,
    /// Layer index in the transformer
    pub layer: usize,
    /// Head index for multi-head attention
    pub head: usize,
    /// Whether this is an antiparticle
    pub is_anti: bool,
    /// Reference to the underlying tensor (optional)
    tensor_data: Option<Vec<f32>>,
}

impl Particle {
    /// Create a new particle
    pub fn new(
        id: u64,
        flavor: ParticleFlavor,
        momentum: FourMomentum,
        position: TorusCoordinate,
    ) -> Self {
        Self {
            id,
            flavor,
            momentum,
            position,
            color: [0.0, 0.0, 0.0], // Colorless by default
            stream_id: 0,
            layer: 0,
            head: 0,
            is_anti: false,
            tensor_data: None,
        }
    }

    /// Create a query particle from a tensor
    pub fn query_from_tensor(
        id: u64,
        tensor: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<Self> {
        let momentum = FourMomentum::from_tensor(tensor)?;
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        Ok(Self {
            id,
            flavor: ParticleFlavor::Query,
            momentum,
            position,
            color: [1.0, 0.0, -1.0], // Red-antiblue (example color)
            stream_id,
            layer,
            head,
            is_anti: false,
            tensor_data: Some(data),
        })
    }

    /// Create a key particle from a tensor
    pub fn key_from_tensor(
        id: u64,
        tensor: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<Self> {
        let momentum = FourMomentum::from_tensor(tensor)?;
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        Ok(Self {
            id,
            flavor: ParticleFlavor::Key,
            momentum,
            position,
            color: [-1.0, 0.0, 1.0], // Antired-blue (conjugate color)
            stream_id,
            layer,
            head,
            is_anti: false,
            tensor_data: Some(data),
        })
    }

    /// Create a value particle from a tensor
    pub fn value_from_tensor(
        id: u64,
        tensor: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<Self> {
        let momentum = FourMomentum::from_tensor(tensor)?;
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        Ok(Self {
            id,
            flavor: ParticleFlavor::Value,
            momentum,
            position,
            color: [0.0, 0.0, 0.0], // Colorless
            stream_id,
            layer,
            head,
            is_anti: false,
            tensor_data: Some(data),
        })
    }

    /// Create an attention boson from attention weights
    pub fn attention_from_weights(
        id: u64,
        weights: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<Self> {
        let momentum = FourMomentum::from_tensor(weights)?;
        let data: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        Ok(Self {
            id,
            flavor: ParticleFlavor::Attention,
            momentum,
            position,
            color: [0.0, 0.0, 0.0], // Colorless (singlet)
            stream_id,
            layer,
            head,
            is_anti: false,
            tensor_data: Some(data),
        })
    }

    /// Create a gradient particle (for backpropagation)
    pub fn gradient_from_tensor(
        id: u64,
        grad: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
    ) -> TorusResult<Self> {
        let momentum = FourMomentum::from_tensor(grad)?;
        let data: Vec<f32> = grad.flatten_all()?.to_vec1()?;

        // Gradients are "backward-moving" - negate momentum
        let backward_momentum =
            FourMomentum::new(momentum.e, -momentum.px, -momentum.py, -momentum.pz);

        Ok(Self {
            id,
            flavor: ParticleFlavor::Gradient,
            momentum: backward_momentum,
            position,
            color: [0.0, 0.0, 0.0], // Gluons are color-charged but we track separately
            stream_id,
            layer,
            head: 0,
            is_anti: true, // Gradients flow "backward"
            tensor_data: Some(data),
        })
    }

    /// Get the spin of this particle
    pub fn spin(&self) -> f64 {
        self.flavor.spin()
    }

    /// Get the electric charge
    pub fn charge(&self) -> i32 {
        let base = self.flavor.charge();
        if self.is_anti {
            -base
        } else {
            base
        }
    }

    /// Get the invariant mass
    pub fn mass(&self) -> f64 {
        self.momentum.mass()
    }

    /// Check if this is a tachyon (faster than light)
    pub fn is_tachyonic(&self) -> bool {
        self.momentum.is_tachyonic()
    }

    /// Get the velocity (can be > 1 for tachyons)
    pub fn velocity(&self) -> f64 {
        self.momentum.velocity()
    }

    /// Get the energy
    pub fn energy(&self) -> f64 {
        self.momentum.e
    }

    /// Check if color is neutral (colorless)
    pub fn is_color_neutral(&self) -> bool {
        let sum: f64 = self.color.iter().sum();
        sum.abs() < 1e-10
    }

    /// Get the antiparticle
    pub fn antiparticle(&self) -> Self {
        let mut anti = self.clone();
        anti.is_anti = !self.is_anti;
        anti.color = [-self.color[0], -self.color[1], -self.color[2]];
        anti.momentum = FourMomentum::new(
            self.momentum.e,
            -self.momentum.px,
            -self.momentum.py,
            -self.momentum.pz,
        );
        anti
    }

    /// Get tensor data if available
    pub fn tensor_data(&self) -> Option<&[f32]> {
        self.tensor_data.as_deref()
    }

    /// Reconstruct tensor from stored data
    pub fn to_tensor(&self, shape: &[usize], device: &Device) -> TorusResult<Option<Tensor>> {
        match &self.tensor_data {
            Some(data) => {
                let tensor = Tensor::from_vec(data.clone(), shape, device)?;
                Ok(Some(tensor))
            }
            None => Ok(None),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTICLE BEAM (Collection of particles in a stream)
// ═══════════════════════════════════════════════════════════════════════════════

/// A beam of particles (like in the LHC)
///
/// Represents a collection of particles traveling together,
/// such as all queries in a single attention head.
#[derive(Debug, Clone)]
pub struct ParticleBeam {
    /// Particles in this beam
    pub particles: Vec<Particle>,
    /// Beam energy (sum of particle energies)
    pub total_energy: f64,
    /// Beam momentum (sum of particle momenta)
    pub total_momentum: FourMomentum,
    /// Stream ID this beam belongs to
    pub stream_id: usize,
    /// Direction: true = forward (causal), false = backward (anti-causal)
    pub is_forward: bool,
}

impl ParticleBeam {
    /// Create an empty beam
    pub fn new(stream_id: usize, is_forward: bool) -> Self {
        Self {
            particles: Vec::new(),
            total_energy: 0.0,
            total_momentum: FourMomentum::new(0.0, 0.0, 0.0, 0.0),
            stream_id,
            is_forward,
        }
    }

    /// Add a particle to the beam
    pub fn add_particle(&mut self, particle: Particle) {
        self.total_energy += particle.energy();
        self.total_momentum = self.total_momentum.add(&particle.momentum);
        self.particles.push(particle);
    }

    /// Get the number of particles
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Check if beam is empty
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Get the invariant mass of the beam
    pub fn invariant_mass(&self) -> f64 {
        self.total_momentum.mass()
    }

    /// Get the center-of-mass energy √s
    pub fn center_of_mass_energy(&self) -> f64 {
        self.total_momentum.mass_squared().abs().sqrt()
    }

    /// Get total charge of the beam
    pub fn total_charge(&self) -> i32 {
        self.particles.iter().map(|p| p.charge()).sum()
    }

    /// Get total color charge
    pub fn total_color(&self) -> [f64; 3] {
        let mut color = [0.0, 0.0, 0.0];
        for p in &self.particles {
            color[0] += p.color[0];
            color[1] += p.color[1];
            color[2] += p.color[2];
        }
        color
    }

    /// Check if beam is color neutral
    pub fn is_color_neutral(&self) -> bool {
        let color = self.total_color();
        color.iter().map(|c| c.abs()).sum::<f64>() < 1e-10
    }

    /// Filter particles by flavor
    pub fn filter_by_flavor(&self, flavor: ParticleFlavor) -> Vec<&Particle> {
        self.particles
            .iter()
            .filter(|p| p.flavor == flavor)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTICLE GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Generates particles from attention computation tensors
#[derive(Debug)]
pub struct ParticleGenerator {
    /// Counter for unique particle IDs
    next_id: u64,
}

impl ParticleGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Generate a unique particle ID
    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Generate query particles from Q tensor
    /// Shape: [batch, seq, heads, dim]
    pub fn generate_queries(
        &mut self,
        q: &Tensor,
        stream_id: usize,
        layer: usize,
        n_major: usize,
        n_minor: usize,
    ) -> TorusResult<Vec<Particle>> {
        let shape = q.dims();
        let seq_len = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let n_heads = if shape.len() >= 3 { shape[2] } else { 1 };

        let mut particles = Vec::new();

        for head in 0..n_heads {
            for pos in 0..seq_len {
                // Map sequence position to torus coordinate
                let u = 2.0 * PI * (pos % n_major) as f64 / n_major as f64;
                let v = 2.0 * PI * (pos / n_major % n_minor) as f64 / n_minor as f64;
                let coord = TorusCoordinate::new(u, v);

                // Extract this position's query vector
                let q_pos = if shape.len() >= 3 {
                    q.i((.., pos, head, ..))?
                } else {
                    q.i((.., pos))?
                };

                let particle = Particle::query_from_tensor(
                    self.next_id(),
                    &q_pos,
                    coord,
                    stream_id,
                    layer,
                    head,
                )?;
                particles.push(particle);
            }
        }

        Ok(particles)
    }

    /// Generate key particles from K tensor
    pub fn generate_keys(
        &mut self,
        k: &Tensor,
        stream_id: usize,
        layer: usize,
        n_major: usize,
        n_minor: usize,
    ) -> TorusResult<Vec<Particle>> {
        let shape = k.dims();
        let seq_len = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let n_heads = if shape.len() >= 3 { shape[2] } else { 1 };

        let mut particles = Vec::new();

        for head in 0..n_heads {
            for pos in 0..seq_len {
                let u = 2.0 * PI * (pos % n_major) as f64 / n_major as f64;
                let v = 2.0 * PI * (pos / n_major % n_minor) as f64 / n_minor as f64;
                let coord = TorusCoordinate::new(u, v);

                let k_pos = if shape.len() >= 3 {
                    k.i((.., pos, head, ..))?
                } else {
                    k.i((.., pos))?
                };

                let particle = Particle::key_from_tensor(
                    self.next_id(),
                    &k_pos,
                    coord,
                    stream_id,
                    layer,
                    head,
                )?;
                particles.push(particle);
            }
        }

        Ok(particles)
    }

    /// Generate value particles from V tensor
    pub fn generate_values(
        &mut self,
        v: &Tensor,
        stream_id: usize,
        layer: usize,
        n_major: usize,
        n_minor: usize,
    ) -> TorusResult<Vec<Particle>> {
        let shape = v.dims();
        let seq_len = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let n_heads = if shape.len() >= 3 { shape[2] } else { 1 };

        let mut particles = Vec::new();

        for head in 0..n_heads {
            for pos in 0..seq_len {
                let u = 2.0 * PI * (pos % n_major) as f64 / n_major as f64;
                let v_angle = 2.0 * PI * (pos / n_major % n_minor) as f64 / n_minor as f64;
                let coord = TorusCoordinate::new(u, v_angle);

                let v_pos = if shape.len() >= 3 {
                    v.i((.., pos, head, ..))?
                } else {
                    v.i((.., pos))?
                };

                let particle = Particle::value_from_tensor(
                    self.next_id(),
                    &v_pos,
                    coord,
                    stream_id,
                    layer,
                    head,
                )?;
                particles.push(particle);
            }
        }

        Ok(particles)
    }

    /// Generate attention boson from attention weights
    pub fn generate_attention_boson(
        &mut self,
        weights: &Tensor,
        position: TorusCoordinate,
        stream_id: usize,
        layer: usize,
        head: usize,
    ) -> TorusResult<Particle> {
        Particle::attention_from_weights(self.next_id(), weights, position, stream_id, layer, head)
    }
}

impl Default for ParticleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_momentum_creation() {
        let p = FourMomentum::new(10.0, 3.0, 4.0, 0.0);
        assert!((p.e - 10.0).abs() < 1e-10);
        assert!((p.three_momentum_magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_four_momentum_mass() {
        // Massless particle: E = |p|
        let massless = FourMomentum::new(5.0, 3.0, 4.0, 0.0);
        assert!(massless.mass().abs() < 1e-10);

        // Massive particle at rest: E = m, p = 0
        let massive = FourMomentum::new(1.0, 0.0, 0.0, 0.0);
        assert!((massive.mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_four_momentum_tachyon() {
        // Tachyon: |p| > E (m² < 0)
        let tachyon = FourMomentum::new(3.0, 4.0, 0.0, 0.0);
        assert!(tachyon.is_tachyonic());
        assert!(tachyon.velocity() > 1.0); // Faster than light!
    }

    #[test]
    fn test_four_momentum_addition() {
        let p1 = FourMomentum::new(10.0, 1.0, 0.0, 0.0);
        let p2 = FourMomentum::new(10.0, -1.0, 0.0, 0.0);
        let sum = p1 + p2;

        assert!((sum.e - 20.0).abs() < 1e-10);
        assert!(sum.px.abs() < 1e-10); // Momenta cancel
    }

    #[test]
    fn test_particle_flavor_properties() {
        assert!(ParticleFlavor::Query.is_fermion());
        assert!(ParticleFlavor::Key.is_fermion());
        assert!(ParticleFlavor::Attention.is_boson());
        assert_eq!(ParticleFlavor::Query.charge(), 1);
        assert_eq!(ParticleFlavor::Key.charge(), -1);
    }

    #[test]
    fn test_particle_creation() {
        let p = FourMomentum::new(10.0, 1.0, 2.0, 3.0);
        let pos = TorusCoordinate::new(0.0, 0.0);
        let particle = Particle::new(0, ParticleFlavor::Query, p, pos);

        assert_eq!(particle.flavor, ParticleFlavor::Query);
        assert_eq!(particle.charge(), 1);
        assert!((particle.energy() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_antiparticle() {
        let p = FourMomentum::new(10.0, 1.0, 2.0, 3.0);
        let pos = TorusCoordinate::new(0.0, 0.0);
        let particle = Particle::new(0, ParticleFlavor::Query, p, pos);
        let anti = particle.antiparticle();

        assert_eq!(anti.charge(), -particle.charge());
        assert!(anti.is_anti);
    }

    #[test]
    fn test_particle_beam() {
        let mut beam = ParticleBeam::new(0, true);

        let p1 = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 1.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let p2 = Particle::new(
            1,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, -1.0, 0.0, 0.0),
            TorusCoordinate::new(PI, 0.0),
        );

        beam.add_particle(p1);
        beam.add_particle(p2);

        assert_eq!(beam.len(), 2);
        assert!((beam.total_energy - 20.0).abs() < 1e-10);
        assert!(beam.total_momentum.px.abs() < 1e-10); // Momenta cancel
    }

    #[test]
    fn test_mandelstam_s() {
        // Two particles colliding head-on with equal energy
        let p1 = FourMomentum::new(10.0, 10.0, 0.0, 0.0); // Massless, moving +x
        let p2 = FourMomentum::new(10.0, -10.0, 0.0, 0.0); // Massless, moving -x

        let sum = p1 + p2;
        let s = sum.mass_squared();

        // s = (20)² - 0² = 400
        assert!((s - 400.0).abs() < 1e-10);
    }
}
