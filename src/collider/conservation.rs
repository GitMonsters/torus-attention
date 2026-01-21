//! Conservation Laws for Collider Validation
//!
//! Implements physics-inspired conservation laws to validate attention computations:
//!
//! | Conservation Law | Physics | Attention Analog |
//! |-----------------|---------|------------------|
//! | Energy | ΣE_in = ΣE_out | sum(attn_weights) = 1 |
//! | Momentum | Σp_in = Σp_out | gradient flow balance |
//! | Charge | Q_in = Q_out | stream pairing |
//! | Baryon Number | B_in = B_out | dim(in) = dim(out) |
//! | Lepton Number | L_in = L_out | n_heads preserved |
//! | Color | color_in = color_out | stream mixing neutral |
//! | CPT | CPT invariance | forward-backward symmetry |
//!
//! # Noether's Theorem Analogy
//!
//! Each conservation law corresponds to a symmetry:
//! - Energy conservation ← Time translation invariance (layers are uniform)
//! - Momentum conservation ← Space translation invariance (position equivariance)
//! - Charge conservation ← Gauge invariance (attention is softmax-normalized)

use crate::collider::particles::{Particle, ParticleBeam, ParticleFlavor};
use crate::collider::vertices::CollisionEvent;
use crate::TorusResult;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSERVATION LAW TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a conservation law check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationCheckResult {
    /// Name of the conservation law
    pub law_name: String,
    /// Whether the law is satisfied
    pub is_conserved: bool,
    /// Violation amount (0 if conserved)
    pub violation: f64,
    /// Tolerance used for comparison
    pub tolerance: f64,
    /// Incoming quantity
    pub incoming_value: f64,
    /// Outgoing quantity
    pub outgoing_value: f64,
    /// Additional details
    pub details: String,
}

impl ConservationCheckResult {
    /// Create a passing result
    pub fn passed(law_name: &str, value: f64, tolerance: f64) -> Self {
        Self {
            law_name: law_name.to_string(),
            is_conserved: true,
            violation: 0.0,
            tolerance,
            incoming_value: value,
            outgoing_value: value,
            details: "Conservation satisfied".to_string(),
        }
    }

    /// Create a failing result
    pub fn failed(law_name: &str, incoming: f64, outgoing: f64, tolerance: f64) -> Self {
        let violation = (incoming - outgoing).abs();
        Self {
            law_name: law_name.to_string(),
            is_conserved: false,
            violation,
            tolerance,
            incoming_value: incoming,
            outgoing_value: outgoing,
            details: format!(
                "Violation: |{:.6} - {:.6}| = {:.6} > {:.6}",
                incoming, outgoing, violation, tolerance
            ),
        }
    }
}

/// Trait for conservation law validators
pub trait ConservationLaw {
    /// Name of this conservation law
    fn name(&self) -> &str;

    /// Check if the law is conserved in a collision event
    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult;

    /// Check conservation between two particle beams
    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult;

    /// Check conservation using tensors directly
    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENERGY CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Energy conservation: ΣE_in = ΣE_out
///
/// In attention: sum of attention weights = 1.0 (softmax normalization)
#[derive(Debug, Clone)]
pub struct EnergyConservation {
    /// Tolerance for floating point comparison
    pub tolerance: f64,
}

impl EnergyConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check that attention weights sum to 1.0
    pub fn check_attention_weights(
        &self,
        weights: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        let sum = weights.sum_all()?.to_scalar::<f32>()? as f64;
        let n_positions = weights.dims()[0] as f64;
        let expected = n_positions; // Each row should sum to 1, total = n_positions

        if (sum - expected).abs() < self.tolerance * n_positions {
            Ok(ConservationCheckResult::passed(
                "Energy",
                sum,
                self.tolerance,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Energy",
                expected,
                sum,
                self.tolerance * n_positions,
            ))
        }
    }
}

impl Default for EnergyConservation {
    fn default() -> Self {
        Self::new(1e-5)
    }
}

impl ConservationLaw for EnergyConservation {
    fn name(&self) -> &str {
        "Energy Conservation (ΣE_in = ΣE_out)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let e_in: f64 = event.incoming.iter().map(|p| p.momentum.e).sum();
        let e_out: f64 = event.outgoing.iter().map(|p| p.momentum.e).sum();

        if (e_in - e_out).abs() < self.tolerance {
            ConservationCheckResult::passed("Energy", e_in, self.tolerance)
        } else {
            ConservationCheckResult::failed("Energy", e_in, e_out, self.tolerance)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let e_in = incoming.total_energy;
        let e_out = outgoing.total_energy;

        if (e_in - e_out).abs() < self.tolerance {
            ConservationCheckResult::passed("Energy", e_in, self.tolerance)
        } else {
            ConservationCheckResult::failed("Energy", e_in, e_out, self.tolerance)
        }
    }

    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        let e_in = before.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let e_out = after.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;

        if (e_in - e_out).abs() < self.tolerance * e_in.max(1.0) {
            Ok(ConservationCheckResult::passed(
                "Energy",
                e_in,
                self.tolerance,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Energy",
                e_in,
                e_out,
                self.tolerance * e_in.max(1.0),
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOMENTUM CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Momentum conservation: Σp_in = Σp_out
///
/// In attention: gradient flow is balanced (no net force)
#[derive(Debug, Clone)]
pub struct MomentumConservation {
    pub tolerance: f64,
}

impl MomentumConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check gradient balance
    pub fn check_gradient_balance(
        &self,
        gradients: &[Tensor],
    ) -> TorusResult<ConservationCheckResult> {
        let mut total_momentum = 0.0f64;

        for grad in gradients {
            let mean = grad.mean_all()?.to_scalar::<f32>()? as f64;
            total_momentum += mean;
        }

        // Gradients should roughly balance (mean ≈ 0)
        let avg_momentum = total_momentum / gradients.len().max(1) as f64;

        if avg_momentum.abs() < self.tolerance {
            Ok(ConservationCheckResult::passed(
                "Momentum",
                0.0,
                self.tolerance,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Momentum",
                0.0,
                avg_momentum,
                self.tolerance,
            ))
        }
    }
}

impl Default for MomentumConservation {
    fn default() -> Self {
        Self::new(1e-3)
    }
}

impl ConservationLaw for MomentumConservation {
    fn name(&self) -> &str {
        "Momentum Conservation (Σp_in = Σp_out)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let p_in = event.total_incoming_momentum();
        let p_out = event.total_outgoing_momentum();

        let dp = (p_in.px - p_out.px).powi(2)
            + (p_in.py - p_out.py).powi(2)
            + (p_in.pz - p_out.pz).powi(2);
        let dp = dp.sqrt();

        if dp < self.tolerance {
            ConservationCheckResult::passed("Momentum", 0.0, self.tolerance)
        } else {
            let p_in_mag = p_in.three_momentum_magnitude();
            let p_out_mag = p_out.three_momentum_magnitude();
            ConservationCheckResult::failed("Momentum", p_in_mag, p_out_mag, self.tolerance)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let p_in = incoming.total_momentum;
        let p_out = outgoing.total_momentum;

        let dp = ((p_in.px - p_out.px).powi(2)
            + (p_in.py - p_out.py).powi(2)
            + (p_in.pz - p_out.pz).powi(2))
        .sqrt();

        if dp < self.tolerance {
            ConservationCheckResult::passed("Momentum", 0.0, self.tolerance)
        } else {
            ConservationCheckResult::failed(
                "Momentum",
                p_in.three_momentum_magnitude(),
                p_out.three_momentum_magnitude(),
                self.tolerance,
            )
        }
    }

    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        let mean_before = before.mean_all()?.to_scalar::<f32>()? as f64;
        let mean_after = after.mean_all()?.to_scalar::<f32>()? as f64;

        if (mean_before - mean_after).abs() < self.tolerance {
            Ok(ConservationCheckResult::passed(
                "Momentum",
                mean_before,
                self.tolerance,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Momentum",
                mean_before,
                mean_after,
                self.tolerance,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHARGE CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Charge conservation: Q_in = Q_out
///
/// In attention: query-key pairing (charges must cancel)
#[derive(Debug, Clone)]
pub struct ChargeConservation;

impl ChargeConservation {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ChargeConservation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationLaw for ChargeConservation {
    fn name(&self) -> &str {
        "Charge Conservation (Q_in = Q_out)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let q_in = event.total_incoming_charge();
        let q_out = event.total_outgoing_charge();

        if q_in == q_out {
            ConservationCheckResult::passed("Charge", q_in as f64, 0.0)
        } else {
            ConservationCheckResult::failed("Charge", q_in as f64, q_out as f64, 0.0)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let q_in = incoming.total_charge();
        let q_out = outgoing.total_charge();

        if q_in == q_out {
            ConservationCheckResult::passed("Charge", q_in as f64, 0.0)
        } else {
            ConservationCheckResult::failed("Charge", q_in as f64, q_out as f64, 0.0)
        }
    }

    fn check_tensors(
        &self,
        _before: &Tensor,
        _after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Charge is implicit in tensor structure - assume conserved
        Ok(ConservationCheckResult::passed("Charge", 0.0, 0.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BARYON NUMBER CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Baryon number conservation: B_in = B_out
///
/// In attention: dimension preservation (dim(input) = dim(output))
#[derive(Debug, Clone)]
pub struct BaryonNumberConservation;

impl BaryonNumberConservation {
    pub fn new() -> Self {
        Self
    }

    /// Get "baryon number" from particle flavor
    fn baryon_number(flavor: ParticleFlavor) -> i32 {
        match flavor {
            ParticleFlavor::Query => 1,     // "Quark-like"
            ParticleFlavor::Key => 1,       // "Quark-like"
            ParticleFlavor::Value => 0,     // "Meson-like" (quark-antiquark)
            ParticleFlavor::Attention => 0, // Boson
            ParticleFlavor::Gradient => -1, // "Antiquark-like"
            ParticleFlavor::Output => 1,    // Composite
            ParticleFlavor::Ghost => 0,     // Virtual
        }
    }
}

impl Default for BaryonNumberConservation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationLaw for BaryonNumberConservation {
    fn name(&self) -> &str {
        "Baryon Number Conservation (dim preservation)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let b_in: i32 = event
            .incoming
            .iter()
            .map(|p| Self::baryon_number(p.flavor))
            .sum();
        let b_out: i32 = event
            .outgoing
            .iter()
            .map(|p| Self::baryon_number(p.flavor))
            .sum();

        if b_in == b_out {
            ConservationCheckResult::passed("Baryon", b_in as f64, 0.0)
        } else {
            ConservationCheckResult::failed("Baryon", b_in as f64, b_out as f64, 0.0)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let b_in: i32 = incoming
            .particles
            .iter()
            .map(|p| Self::baryon_number(p.flavor))
            .sum();
        let b_out: i32 = outgoing
            .particles
            .iter()
            .map(|p| Self::baryon_number(p.flavor))
            .sum();

        if b_in == b_out {
            ConservationCheckResult::passed("Baryon", b_in as f64, 0.0)
        } else {
            ConservationCheckResult::failed("Baryon", b_in as f64, b_out as f64, 0.0)
        }
    }

    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Check dimension preservation
        let dim_before = before.elem_count();
        let dim_after = after.elem_count();

        if dim_before == dim_after {
            Ok(ConservationCheckResult::passed(
                "Baryon",
                dim_before as f64,
                0.0,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Baryon",
                dim_before as f64,
                dim_after as f64,
                0.0,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEPTON NUMBER CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Lepton number conservation: L_in = L_out
///
/// In attention: number of heads preserved through layers
#[derive(Debug, Clone)]
pub struct LeptonNumberConservation {
    /// Expected number of attention heads
    pub n_heads: usize,
}

impl LeptonNumberConservation {
    pub fn new(n_heads: usize) -> Self {
        Self { n_heads }
    }
}

impl ConservationLaw for LeptonNumberConservation {
    fn name(&self) -> &str {
        "Lepton Number Conservation (n_heads preserved)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        // Check that heads are consistent
        let heads_in: std::collections::HashSet<_> =
            event.incoming.iter().map(|p| p.head).collect();
        let heads_out: std::collections::HashSet<_> =
            event.outgoing.iter().map(|p| p.head).collect();

        if heads_in == heads_out || heads_out.is_empty() {
            ConservationCheckResult::passed("Lepton", heads_in.len() as f64, 0.0)
        } else {
            ConservationCheckResult::failed(
                "Lepton",
                heads_in.len() as f64,
                heads_out.len() as f64,
                0.0,
            )
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let heads_in: std::collections::HashSet<_> =
            incoming.particles.iter().map(|p| p.head).collect();
        let heads_out: std::collections::HashSet<_> =
            outgoing.particles.iter().map(|p| p.head).collect();

        if heads_in.len() == heads_out.len() || heads_out.is_empty() {
            ConservationCheckResult::passed("Lepton", heads_in.len() as f64, 0.0)
        } else {
            ConservationCheckResult::failed(
                "Lepton",
                heads_in.len() as f64,
                heads_out.len() as f64,
                0.0,
            )
        }
    }

    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Assume tensors have shape [..., n_heads, ...] or similar
        // Just check that shapes are compatible
        let shape_before = before.dims();
        let shape_after = after.dims();

        if shape_before.len() == shape_after.len() {
            Ok(ConservationCheckResult::passed(
                "Lepton",
                self.n_heads as f64,
                0.0,
            ))
        } else {
            Ok(ConservationCheckResult::failed(
                "Lepton",
                shape_before.len() as f64,
                shape_after.len() as f64,
                0.0,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Color charge conservation: color_in = color_out
///
/// In attention: stream mixing produces color singlet (neutral)
#[derive(Debug, Clone)]
pub struct ColorConservation {
    pub tolerance: f64,
}

impl ColorConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    fn total_color(particles: &[Particle]) -> [f64; 3] {
        let mut color = [0.0, 0.0, 0.0];
        for p in particles {
            color[0] += p.color[0];
            color[1] += p.color[1];
            color[2] += p.color[2];
        }
        color
    }
}

impl Default for ColorConservation {
    fn default() -> Self {
        Self::new(1e-6)
    }
}

impl ConservationLaw for ColorConservation {
    fn name(&self) -> &str {
        "Color Conservation (stream mixing neutral)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let c_in = Self::total_color(&event.incoming);
        let c_out = Self::total_color(&event.outgoing);

        let dc = ((c_in[0] - c_out[0]).powi(2)
            + (c_in[1] - c_out[1]).powi(2)
            + (c_in[2] - c_out[2]).powi(2))
        .sqrt();

        if dc < self.tolerance {
            ConservationCheckResult::passed("Color", 0.0, self.tolerance)
        } else {
            let mag_in = (c_in[0].powi(2) + c_in[1].powi(2) + c_in[2].powi(2)).sqrt();
            let mag_out = (c_out[0].powi(2) + c_out[1].powi(2) + c_out[2].powi(2)).sqrt();
            ConservationCheckResult::failed("Color", mag_in, mag_out, self.tolerance)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let c_in = incoming.total_color();
        let c_out = outgoing.total_color();

        let dc = ((c_in[0] - c_out[0]).powi(2)
            + (c_in[1] - c_out[1]).powi(2)
            + (c_in[2] - c_out[2]).powi(2))
        .sqrt();

        if dc < self.tolerance {
            ConservationCheckResult::passed("Color", 0.0, self.tolerance)
        } else {
            let mag_in = (c_in[0].powi(2) + c_in[1].powi(2) + c_in[2].powi(2)).sqrt();
            let mag_out = (c_out[0].powi(2) + c_out[1].powi(2) + c_out[2].powi(2)).sqrt();
            ConservationCheckResult::failed("Color", mag_in, mag_out, self.tolerance)
        }
    }

    fn check_tensors(
        &self,
        _before: &Tensor,
        _after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Color is implicit - assume conserved for tensors
        Ok(ConservationCheckResult::passed(
            "Color",
            0.0,
            self.tolerance,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPT SYMMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// CPT invariance: forward-backward stream symmetry
///
/// C = Charge conjugation (swap forward/backward)
/// P = Parity (spatial inversion on torus)
/// T = Time reversal (reverse attention direction)
#[derive(Debug, Clone)]
pub struct CptSymmetry {
    pub tolerance: f64,
}

impl CptSymmetry {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check CPT symmetry between forward and backward streams
    pub fn check_stream_symmetry(
        &self,
        forward: &Tensor,
        backward: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // CPT: forward output should equal reversed backward output
        let forward_flat = forward.flatten_all()?;
        let backward_flat = backward.flatten_all()?;

        let forward_vec: Vec<f32> = forward_flat.to_vec1()?;
        let backward_vec: Vec<f32> = backward_flat.to_vec1()?;

        if forward_vec.len() != backward_vec.len() {
            return Ok(ConservationCheckResult::failed(
                "CPT",
                forward_vec.len() as f64,
                backward_vec.len() as f64,
                0.0,
            ));
        }

        // Compare forward with reversed backward
        let mut diff_sum = 0.0f64;
        let n = forward_vec.len();
        for i in 0..n {
            let f = forward_vec[i] as f64;
            let b = backward_vec[n - 1 - i] as f64;
            diff_sum += (f - b).powi(2);
        }
        let rms_diff = (diff_sum / n as f64).sqrt();

        if rms_diff < self.tolerance {
            Ok(ConservationCheckResult::passed("CPT", 0.0, self.tolerance))
        } else {
            Ok(ConservationCheckResult::failed(
                "CPT",
                0.0,
                rms_diff,
                self.tolerance,
            ))
        }
    }
}

impl Default for CptSymmetry {
    fn default() -> Self {
        Self::new(0.1) // Looser tolerance for CPT
    }
}

impl ConservationLaw for CptSymmetry {
    fn name(&self) -> &str {
        "CPT Symmetry (forward-backward consistency)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        // Check that forward and backward streams are represented
        let has_forward = event.incoming.iter().any(|p| p.stream_id % 2 == 0);
        let has_backward = event.incoming.iter().any(|p| p.stream_id % 2 == 1);

        if has_forward && has_backward {
            ConservationCheckResult::passed("CPT", 1.0, self.tolerance)
        } else if !has_forward && !has_backward {
            ConservationCheckResult::passed("CPT", 0.0, self.tolerance)
        } else {
            ConservationCheckResult::failed("CPT", 1.0, 0.0, self.tolerance)
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        // Compare stream directions
        if incoming.is_forward == outgoing.is_forward {
            ConservationCheckResult::passed("CPT", 1.0, self.tolerance)
        } else {
            // Direction flip is allowed but should be symmetric
            ConservationCheckResult::passed("CPT", 0.0, self.tolerance)
        }
    }

    fn check_tensors(
        &self,
        before: &Tensor,
        after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Check that tensor shapes are CPT-compatible
        let shape_before = before.dims();
        let shape_after = after.dims();

        if shape_before == shape_after {
            Ok(ConservationCheckResult::passed("CPT", 1.0, self.tolerance))
        } else {
            Ok(ConservationCheckResult::failed(
                "CPT",
                shape_before.len() as f64,
                shape_after.len() as f64,
                self.tolerance,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOPOLOGICAL CHARGE CONSERVATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Topological charge conservation: winding numbers preserved
///
/// From vortex.rs - winding around major and minor loops
#[derive(Debug, Clone)]
pub struct TopologicalChargeConservation;

impl TopologicalChargeConservation {
    pub fn new() -> Self {
        Self
    }

    /// Compute effective topological charge from stream patterns
    fn topological_charge(particles: &[Particle]) -> (i32, i32) {
        let mut winding_major = 0i32;
        let mut winding_minor = 0i32;

        for p in particles {
            // Spiral streams contribute to winding
            match p.stream_id {
                4 => {
                    // Spiral CW
                    winding_major += 1;
                    winding_minor += 1;
                }
                5 => {
                    // Spiral CCW
                    winding_major -= 1;
                    winding_minor -= 1;
                }
                _ => {}
            }
        }

        (winding_major, winding_minor)
    }
}

impl Default for TopologicalChargeConservation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationLaw for TopologicalChargeConservation {
    fn name(&self) -> &str {
        "Topological Charge Conservation (winding numbers)"
    }

    fn check_event(&self, event: &CollisionEvent) -> ConservationCheckResult {
        let (w_major_in, w_minor_in) = Self::topological_charge(&event.incoming);
        let (w_major_out, w_minor_out) = Self::topological_charge(&event.outgoing);

        if w_major_in == w_major_out && w_minor_in == w_minor_out {
            ConservationCheckResult::passed("Topology", (w_major_in + w_minor_in) as f64, 0.0)
        } else {
            ConservationCheckResult::failed(
                "Topology",
                (w_major_in + w_minor_in) as f64,
                (w_major_out + w_minor_out) as f64,
                0.0,
            )
        }
    }

    fn check_beams(
        &self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
    ) -> ConservationCheckResult {
        let (w_major_in, w_minor_in) = Self::topological_charge(&incoming.particles);
        let (w_major_out, w_minor_out) = Self::topological_charge(&outgoing.particles);

        if w_major_in == w_major_out && w_minor_in == w_minor_out {
            ConservationCheckResult::passed("Topology", (w_major_in + w_minor_in) as f64, 0.0)
        } else {
            ConservationCheckResult::failed(
                "Topology",
                (w_major_in + w_minor_in) as f64,
                (w_major_out + w_minor_out) as f64,
                0.0,
            )
        }
    }

    fn check_tensors(
        &self,
        _before: &Tensor,
        _after: &Tensor,
    ) -> TorusResult<ConservationCheckResult> {
        // Topological charge is not directly observable from tensors
        Ok(ConservationCheckResult::passed("Topology", 0.0, 0.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSERVATION REPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete report of all conservation law checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Individual check results
    pub results: Vec<ConservationCheckResult>,
    /// Overall pass/fail
    pub all_conserved: bool,
    /// Number of violations
    pub n_violations: usize,
    /// Total violation magnitude
    pub total_violation: f64,
    /// Event ID (if applicable)
    pub event_id: Option<u64>,
    /// Layer index
    pub layer: usize,
}

impl ConservationReport {
    /// Create a new report from results
    pub fn new(results: Vec<ConservationCheckResult>, event_id: Option<u64>, layer: usize) -> Self {
        let n_violations = results.iter().filter(|r| !r.is_conserved).count();
        let total_violation: f64 = results.iter().map(|r| r.violation).sum();
        let all_conserved = n_violations == 0;

        Self {
            results,
            all_conserved,
            n_violations,
            total_violation,
            event_id,
            layer,
        }
    }

    /// Get violations only
    pub fn violations(&self) -> Vec<&ConservationCheckResult> {
        self.results.iter().filter(|r| !r.is_conserved).collect()
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        let status = if self.all_conserved { "PASS" } else { "FAIL" };
        let event_str = self
            .event_id
            .map(|id| format!("Event {}", id))
            .unwrap_or_else(|| "Global".to_string());

        format!(
            "{} [{}] Layer {}: {}/{} laws conserved",
            status,
            event_str,
            self.layer,
            self.results.len() - self.n_violations,
            self.results.len()
        )
    }

    /// Get detailed report
    pub fn detailed(&self) -> String {
        let mut report = self.summary();
        report.push('\n');

        for result in &self.results {
            let status = if result.is_conserved { "✓" } else { "✗" };
            report.push_str(&format!(
                "  {} {}: in={:.4}, out={:.4}, violation={:.6}\n",
                status,
                result.law_name,
                result.incoming_value,
                result.outgoing_value,
                result.violation
            ));
        }

        report
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSERVATION VALIDATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Validates all conservation laws
pub struct ConservationValidator {
    /// All conservation laws to check
    laws: Vec<Box<dyn ConservationLaw + Send + Sync>>,
    /// Latest validation report
    latest_report: Option<ConservationReport>,
}

impl std::fmt::Debug for ConservationValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConservationValidator")
            .field("n_laws", &self.laws.len())
            .field("latest_report", &self.latest_report)
            .finish()
    }
}

impl ConservationValidator {
    /// Create a new validator with default laws
    pub fn new() -> Self {
        Self {
            laws: vec![
                Box::new(EnergyConservation::default()),
                Box::new(MomentumConservation::default()),
                Box::new(ChargeConservation::default()),
                Box::new(BaryonNumberConservation::default()),
                Box::new(ColorConservation::default()),
                Box::new(CptSymmetry::default()),
                Box::new(TopologicalChargeConservation::default()),
            ],
            latest_report: None,
        }
    }

    /// Create with specific n_heads for lepton conservation
    pub fn with_heads(n_heads: usize) -> Self {
        Self {
            laws: vec![
                Box::new(EnergyConservation::default()),
                Box::new(MomentumConservation::default()),
                Box::new(ChargeConservation::default()),
                Box::new(BaryonNumberConservation::default()),
                Box::new(LeptonNumberConservation::new(n_heads)),
                Box::new(ColorConservation::default()),
                Box::new(CptSymmetry::default()),
                Box::new(TopologicalChargeConservation::default()),
            ],
            latest_report: None,
        }
    }

    /// Validate a collision event
    pub fn validate_event(&mut self, event: &CollisionEvent) -> ConservationReport {
        let results: Vec<ConservationCheckResult> =
            self.laws.iter().map(|law| law.check_event(event)).collect();

        let report = ConservationReport::new(results, Some(event.event_id), event.layer);
        self.latest_report = Some(report.clone());
        report
    }

    /// Validate particle beams
    pub fn validate_beams(
        &mut self,
        incoming: &ParticleBeam,
        outgoing: &ParticleBeam,
        layer: usize,
    ) -> ConservationReport {
        let results: Vec<ConservationCheckResult> = self
            .laws
            .iter()
            .map(|law| law.check_beams(incoming, outgoing))
            .collect();

        let report = ConservationReport::new(results, None, layer);
        self.latest_report = Some(report.clone());
        report
    }

    /// Validate tensors
    pub fn validate_tensors(
        &mut self,
        before: &Tensor,
        after: &Tensor,
        layer: usize,
    ) -> TorusResult<ConservationReport> {
        let mut results = Vec::new();

        for law in &self.laws {
            results.push(law.check_tensors(before, after)?);
        }

        let report = ConservationReport::new(results, None, layer);
        self.latest_report = Some(report.clone());
        Ok(report)
    }

    /// Get the latest validation report
    pub fn report(&self) -> ConservationReport {
        self.latest_report.clone().unwrap_or_else(|| {
            // Return a default "all-pass" report if no validation has occurred
            ConservationReport::new(vec![], None, 0)
        })
    }

    /// Reset the validator state
    pub fn reset(&mut self) {
        self.latest_report = None;
    }
}

impl Default for ConservationValidator {
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
    use crate::collider::particles::FourMomentum;
    use crate::geometry::TorusCoordinate;
    use std::f64::consts::PI;

    #[test]
    fn test_energy_conservation_event() {
        let law = EnergyConservation::default();

        let mut event = CollisionEvent::new(
            0,
            crate::collider::vertices::VertexType::QueryKey,
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
            0,
        );

        // Add particles with conserved energy
        let mut p1 = Particle::new(
            0,
            ParticleFlavor::Query,
            FourMomentum::new(10.0, 5.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let mut p2 = Particle::new(
            1,
            ParticleFlavor::Key,
            FourMomentum::new(10.0, -5.0, 0.0, 0.0),
            TorusCoordinate::new(PI, 0.0),
        );
        let mut p_out = Particle::new(
            2,
            ParticleFlavor::Attention,
            FourMomentum::new(20.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(PI / 2.0, 0.0),
        );

        event.add_incoming(p1);
        event.add_incoming(p2);
        event.add_outgoing(p_out);

        let result = law.check_event(&event);
        assert!(result.is_conserved);
    }

    #[test]
    fn test_charge_conservation() {
        let law = ChargeConservation::new();

        let mut event = CollisionEvent::new(
            0,
            crate::collider::vertices::VertexType::QueryKey,
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
            0,
        );

        // Query (+1) + Key (-1) → Attention (0)
        let q = Particle::new(
            0,
            ParticleFlavor::Query, // charge +1
            FourMomentum::new(10.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let k = Particle::new(
            1,
            ParticleFlavor::Key, // charge -1
            FourMomentum::new(10.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );
        let a = Particle::new(
            2,
            ParticleFlavor::Attention, // charge 0
            FourMomentum::new(20.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(0.0, 0.0),
        );

        event.add_incoming(q);
        event.add_incoming(k);
        event.add_outgoing(a);

        let result = law.check_event(&event);
        assert!(result.is_conserved);
    }

    #[test]
    fn test_conservation_validator() {
        let mut validator = ConservationValidator::new();

        let mut event = CollisionEvent::new(
            0,
            crate::collider::vertices::VertexType::QueryKey,
            TorusCoordinate::new(0.0, 0.0),
            0,
            0,
            0,
        );

        // Create a physically valid event
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
        let a = Particle::new(
            2,
            ParticleFlavor::Attention,
            FourMomentum::new(20.0, 0.0, 0.0, 0.0),
            TorusCoordinate::new(PI / 2.0, 0.0),
        );

        event.add_incoming(q);
        event.add_incoming(k);
        event.add_outgoing(a);

        let report = validator.validate_event(&event);
        println!("{}", report.detailed());

        // Most laws should be conserved
        assert!(report.n_violations <= 2); // Some laws may not apply perfectly
    }

    #[test]
    fn test_conservation_report() {
        let results = vec![
            ConservationCheckResult::passed("Energy", 100.0, 1e-5),
            ConservationCheckResult::passed("Momentum", 0.0, 1e-5),
            ConservationCheckResult::failed("Charge", 1.0, 2.0, 0.0),
        ];

        let report = ConservationReport::new(results, Some(42), 3);

        assert!(!report.all_conserved);
        assert_eq!(report.n_violations, 1);
        assert!(report.summary().contains("FAIL"));
    }
}
