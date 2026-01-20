//! Vortex and Spiral Dynamics for Torus Attention
//!
//! Vortices on a torus follow spiral paths that wind around both
//! the major and minor loops. This module implements:
//! - Vortex position tracking
//! - Spiral attention patterns
//! - Helical information flow
//! - Topological winding numbers

use crate::geometry::{TorusCoordinate, TorusManifold};
use crate::periodic::PeriodicBoundary;
use crate::TorusResult;
use candle_core::{DType, Device, Tensor};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Vortex on the torus manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vortex {
    /// Position on torus
    pub position: TorusCoordinate,
    /// Circulation strength (positive = counterclockwise)
    pub circulation: f64,
    /// Winding number around major loop
    pub winding_major: i32,
    /// Winding number around minor loop
    pub winding_minor: i32,
    /// Core size (for regularization)
    pub core_size: f64,
}

impl Vortex {
    /// Create a new vortex
    pub fn new(
        u: f64,
        v: f64,
        circulation: f64,
        winding_major: i32,
        winding_minor: i32,
    ) -> Self {
        Self {
            position: TorusCoordinate::new(u, v),
            circulation,
            winding_major,
            winding_minor,
            core_size: 0.1,
        }
    }

    /// Compute the velocity field induced by this vortex at a point
    /// Uses the Biot-Savart law adapted for torus geometry
    pub fn velocity_at(&self, point: &TorusCoordinate) -> (f64, f64) {
        let du = TorusCoordinate::angular_distance(point.u, self.position.u);
        let dv = TorusCoordinate::angular_distance(point.v, self.position.v);
        
        // Distance with core regularization
        let r2 = du * du + dv * dv + self.core_size * self.core_size;
        let _r = r2.sqrt();

        // Velocity perpendicular to displacement (2D vortex on flat torus)
        let vel_scale = self.circulation / (2.0 * PI * r2);
        let vel_u = -dv * vel_scale; // perpendicular component
        let vel_v = du * vel_scale;

        (vel_u, vel_v)
    }

    /// Evolve vortex position based on background flow
    pub fn advect(&mut self, vel_u: f64, vel_v: f64, dt: f64) {
        self.position = TorusCoordinate::new(
            self.position.u + vel_u * dt,
            self.position.v + vel_v * dt,
        );
    }

    /// Get the spiral path traced by this vortex
    pub fn spiral_path(&self, n_points: usize, total_angle: f64) -> Vec<TorusCoordinate> {
        let winding = self.winding_major as f64 / self.winding_minor.max(1) as f64;
        let dt = total_angle / n_points as f64;

        (0..n_points)
            .map(|i| {
                let t = i as f64 * dt;
                TorusCoordinate::new(
                    self.position.u + t,
                    self.position.v + winding * t,
                )
            })
            .collect()
    }
}

/// Vortex dynamics system on torus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexDynamics {
    /// Collection of vortices
    pub vortices: Vec<Vortex>,
    /// Torus manifold
    pub manifold: TorusManifold,
    /// Time step for evolution
    pub dt: f64,
    /// Current simulation time
    pub time: f64,
}

impl VortexDynamics {
    /// Create a new vortex dynamics system
    pub fn new(manifold: TorusManifold, dt: f64) -> Self {
        Self {
            vortices: Vec::new(),
            manifold,
            dt,
            time: 0.0,
        }
    }

    /// Add a vortex to the system
    pub fn add_vortex(&mut self, vortex: Vortex) {
        self.vortices.push(vortex);
    }

    /// Create a vortex-antivortex pair (topologically neutral)
    pub fn add_vortex_pair(&mut self, u: f64, v: f64, separation: f64, circulation: f64) {
        self.vortices.push(Vortex::new(
            u - separation / 2.0,
            v,
            circulation,
            0,
            0,
        ));
        self.vortices.push(Vortex::new(
            u + separation / 2.0,
            v,
            -circulation,
            0,
            0,
        ));
    }

    /// Compute total velocity field at a point from all vortices
    pub fn total_velocity(&self, point: &TorusCoordinate) -> (f64, f64) {
        self.vortices
            .iter()
            .map(|v| v.velocity_at(point))
            .fold((0.0, 0.0), |(au, av), (vu, vv)| (au + vu, av + vv))
    }

    /// Evolve the system by one time step
    pub fn step(&mut self) {
        // Compute velocities at each vortex position (excluding self-interaction)
        let velocities: Vec<(f64, f64)> = (0..self.vortices.len())
            .into_par_iter()
            .map(|i| {
                let pos = &self.vortices[i].position;
                self.vortices
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, v)| v.velocity_at(pos))
                    .fold((0.0, 0.0), |(au, av), (vu, vv)| (au + vu, av + vv))
            })
            .collect();

        // Advect vortices
        for (vortex, (vel_u, vel_v)) in self.vortices.iter_mut().zip(velocities.iter()) {
            vortex.advect(*vel_u, *vel_v, self.dt);
        }

        self.time += self.dt;
    }

    /// Run simulation for n steps
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Compute velocity field on a grid
    pub fn velocity_field(&self, n_major: usize, n_minor: usize) -> (Array2<f64>, Array2<f64>) {
        let boundary = PeriodicBoundary::new(n_major, n_minor);
        let mut vel_u = Array2::zeros((n_major, n_minor));
        let mut vel_v = Array2::zeros((n_major, n_minor));

        for i in 0..n_major {
            for j in 0..n_minor {
                let coord = TorusCoordinate::new(
                    i as f64 * boundary.du,
                    j as f64 * boundary.dv,
                );
                let (vu, vv) = self.total_velocity(&coord);
                vel_u[[i, j]] = vu;
                vel_v[[i, j]] = vv;
            }
        }

        (vel_u, vel_v)
    }

    /// Compute vorticity field (curl of velocity)
    pub fn vorticity_field(&self, n_major: usize, n_minor: usize) -> Array2<f64> {
        let boundary = PeriodicBoundary::new(n_major, n_minor);
        let (vel_u, vel_v) = self.velocity_field(n_major, n_minor);
        
        let (_, grad_u_v) = boundary.gradient(&vel_u);
        let (grad_v_u, _) = boundary.gradient(&vel_v);

        // Vorticity = ∂v/∂u - ∂u/∂v
        &grad_v_u - &grad_u_v
    }

    /// Total circulation (should be conserved)
    pub fn total_circulation(&self) -> f64 {
        self.vortices.iter().map(|v| v.circulation).sum()
    }

    /// Compute energy of the vortex configuration
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;

        // Self-energy (logarithmic divergence regularized by core size)
        for v in &self.vortices {
            energy += v.circulation * v.circulation * (1.0 / v.core_size).ln();
        }

        // Interaction energy
        for i in 0..self.vortices.len() {
            for j in (i + 1)..self.vortices.len() {
                let d = self.vortices[i].position.geodesic_distance(&self.vortices[j].position);
                let d_reg = d.max(0.01);
                energy += self.vortices[i].circulation * self.vortices[j].circulation * d_reg.ln();
            }
        }

        energy / (4.0 * PI)
    }
}

/// Spiral attention pattern generator
#[derive(Debug, Clone)]
pub struct SpiralAttention {
    /// Winding number (ratio of major to minor loops)
    pub winding: f64,
    /// Bandwidth of attention along spiral
    pub bandwidth: f64,
    /// Number of spiral arms
    pub n_arms: usize,
}

impl SpiralAttention {
    /// Create a new spiral attention pattern
    pub fn new(winding: f64, bandwidth: f64, n_arms: usize) -> Self {
        Self {
            winding,
            bandwidth,
            n_arms,
        }
    }

    /// Golden spiral attention (based on golden ratio)
    pub fn golden() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(phi, PI / 8.0, 1)
    }

    /// Compute attention weights based on spiral distance
    pub fn attention_weights(
        &self,
        n_major: usize,
        n_minor: usize,
        device: &Device,
    ) -> TorusResult<Tensor> {
        let boundary = PeriodicBoundary::new(n_major, n_minor);
        let n = n_major * n_minor;
        let mut weights = vec![0.0f32; n * n];

        let neg_inv_2bw2 = -0.5 / (self.bandwidth * self.bandwidth);

        for i in 0..n_major {
            for j in 0..n_minor {
                let src_idx = i * n_minor + j;
                let u_src = i as f64 * boundary.du;
                let v_src = j as f64 * boundary.dv;

                for ti in 0..n_major {
                    for tj in 0..n_minor {
                        let tgt_idx = ti * n_minor + tj;
                        let u_tgt = ti as f64 * boundary.du;
                        let v_tgt = tj as f64 * boundary.dv;

                        // Compute minimum spiral distance across all arms
                        let mut min_dist = f64::MAX;
                        for arm in 0..self.n_arms {
                            let arm_offset = 2.0 * PI * arm as f64 / self.n_arms as f64;
                            
                            // Spiral phase for source and target
                            let spiral_src = u_src + self.winding * v_src + arm_offset;
                            let spiral_tgt = u_tgt + self.winding * v_tgt + arm_offset;
                            
                            let dist = TorusCoordinate::angular_distance(spiral_src, spiral_tgt);
                            min_dist = min_dist.min(dist);
                        }

                        // Gaussian attention weight
                        let weight = (min_dist * min_dist * neg_inv_2bw2).exp();
                        weights[src_idx * n + tgt_idx] = weight as f32;
                    }
                }
            }
        }

        // Normalize rows (softmax-like)
        for i in 0..n {
            let row_start = i * n;
            let sum: f32 = weights[row_start..row_start + n].iter().sum();
            if sum > 0.0 {
                for j in 0..n {
                    weights[row_start + j] /= sum;
                }
            }
        }

        let tensor = Tensor::from_vec(weights, (n, n), device)?;
        Ok(tensor)
    }

    /// Generate spiral sampling positions
    pub fn sample_positions(&self, n_points: usize, start_phase: f64) -> Vec<TorusCoordinate> {
        let dt = 2.0 * PI / n_points as f64;
        (0..n_points)
            .map(|i| {
                let t = i as f64 * dt * self.n_arms as f64 + start_phase;
                TorusCoordinate::new(t, t / self.winding)
            })
            .collect()
    }
}

/// Helical flow pattern for sequential attention
#[derive(Debug, Clone)]
pub struct HelicalFlow {
    /// Pitch of helix (advance per turn)
    pub pitch: f64,
    /// Radius of helix in the embedding
    pub radius: f64,
    /// Direction (1.0 = right-handed, -1.0 = left-handed)
    pub handedness: f64,
}

impl HelicalFlow {
    pub fn new(pitch: f64, radius: f64, right_handed: bool) -> Self {
        Self {
            pitch,
            radius,
            handedness: if right_handed { 1.0 } else { -1.0 },
        }
    }

    /// Map sequence position to torus coordinates via helix
    pub fn sequence_to_torus(&self, seq_idx: usize, seq_len: usize) -> TorusCoordinate {
        let t = 2.0 * PI * seq_idx as f64 / seq_len as f64;
        TorusCoordinate::new(
            t,
            self.handedness * t * self.pitch,
        )
    }

    /// Generate helical position encodings
    pub fn position_encodings(&self, seq_len: usize, d_model: usize, device: &Device) -> TorusResult<Tensor> {
        let mut encodings = vec![0.0f32; seq_len * d_model];

        for i in 0..seq_len {
            let coord = self.sequence_to_torus(i, seq_len);
            
            for d in 0..d_model {
                let freq = (d / 2 + 1) as f64;
                let val = if d % 2 == 0 {
                    // Sin encoding mixing u and v
                    (freq * coord.u + self.pitch * freq * coord.v).sin()
                } else {
                    // Cos encoding
                    (freq * coord.u + self.pitch * freq * coord.v).cos()
                };
                encodings[i * d_model + d] = (val * self.radius) as f32;
            }
        }

        let tensor = Tensor::from_vec(encodings, (seq_len, d_model), device)?;
        Ok(tensor)
    }
}

/// Direction of spiral rotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpiralDirection {
    /// Clockwise spiral (positive winding)
    Clockwise,
    /// Counter-clockwise spiral (negative winding)
    CounterClockwise,
}

impl SpiralDirection {
    /// Get sign multiplier
    pub fn sign(&self) -> f64 {
        match self {
            SpiralDirection::Clockwise => 1.0,
            SpiralDirection::CounterClockwise => -1.0,
        }
    }

    /// Get opposite direction
    pub fn reverse(&self) -> Self {
        match self {
            SpiralDirection::Clockwise => SpiralDirection::CounterClockwise,
            SpiralDirection::CounterClockwise => SpiralDirection::Clockwise,
        }
    }
}

/// Directional spiral flow for bidirectional processing
#[derive(Debug, Clone)]
pub struct DirectionalSpiral {
    /// Base spiral pattern
    pub spiral: SpiralAttention,
    /// Direction of flow
    pub direction: SpiralDirection,
}

impl DirectionalSpiral {
    /// Create a new directional spiral
    pub fn new(winding: f64, bandwidth: f64, n_arms: usize, direction: SpiralDirection) -> Self {
        // Adjust winding based on direction
        let effective_winding = winding * direction.sign();
        Self {
            spiral: SpiralAttention::new(effective_winding, bandwidth, n_arms),
            direction,
        }
    }

    /// Create clockwise golden spiral
    pub fn golden_cw() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(phi, PI / 8.0, 1, SpiralDirection::Clockwise)
    }

    /// Create counter-clockwise golden spiral
    pub fn golden_ccw() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(phi, PI / 8.0, 1, SpiralDirection::CounterClockwise)
    }

    /// Generate causal attention mask for directional spiral
    /// CW: can only attend to positions earlier on the spiral
    /// CCW: can only attend to positions later on the spiral
    pub fn causal_attention_weights(
        &self,
        n_major: usize,
        n_minor: usize,
        device: &Device,
    ) -> TorusResult<Tensor> {
        let boundary = PeriodicBoundary::new(n_major, n_minor);
        let n = n_major * n_minor;
        let mut weights = vec![0.0f32; n * n];

        let neg_inv_2bw2 = -0.5 / (self.spiral.bandwidth * self.spiral.bandwidth);
        let is_cw = self.direction == SpiralDirection::Clockwise;

        for i in 0..n_major {
            for j in 0..n_minor {
                let src_idx = i * n_minor + j;
                let u_src = i as f64 * boundary.du;
                let v_src = j as f64 * boundary.dv;
                let spiral_src = u_src + self.spiral.winding * v_src;

                for ti in 0..n_major {
                    for tj in 0..n_minor {
                        let tgt_idx = ti * n_minor + tj;
                        let u_tgt = ti as f64 * boundary.du;
                        let v_tgt = tj as f64 * boundary.dv;
                        let spiral_tgt = u_tgt + self.spiral.winding * v_tgt;

                        // Causal constraint based on direction
                        let can_attend = if is_cw {
                            // CW: attend to earlier spiral positions
                            spiral_tgt <= spiral_src + 0.001
                        } else {
                            // CCW: attend to later spiral positions
                            spiral_tgt >= spiral_src - 0.001
                        };

                        if can_attend {
                            // Distance along spiral
                            let spiral_dist = TorusCoordinate::angular_distance(spiral_src, spiral_tgt);
                            let weight = (spiral_dist * spiral_dist * neg_inv_2bw2).exp();
                            weights[src_idx * n + tgt_idx] = weight as f32;
                        }
                    }
                }
            }
        }

        // Normalize rows
        for i in 0..n {
            let row_start = i * n;
            let sum: f32 = weights[row_start..row_start + n].iter().sum();
            if sum > 0.0 {
                for j in 0..n {
                    weights[row_start + j] /= sum;
                }
            }
        }

        Ok(Tensor::from_vec(weights, (n, n), device)?)
    }

    /// Sample positions along the directional spiral
    pub fn sample_positions(&self, n_points: usize, start_phase: f64) -> Vec<TorusCoordinate> {
        let direction_mult = self.direction.sign();
        let dt = direction_mult * 2.0 * PI / n_points as f64;
        
        (0..n_points)
            .map(|i| {
                let t = i as f64 * dt + start_phase;
                TorusCoordinate::new(t, t / self.spiral.winding)
            })
            .collect()
    }
}

/// Bidirectional spiral pair for symmetric processing
#[derive(Debug, Clone)]
pub struct BidirectionalSpiral {
    /// Clockwise spiral
    pub cw: DirectionalSpiral,
    /// Counter-clockwise spiral
    pub ccw: DirectionalSpiral,
}

impl BidirectionalSpiral {
    /// Create a bidirectional spiral pair
    pub fn new(winding: f64, bandwidth: f64, n_arms: usize) -> Self {
        Self {
            cw: DirectionalSpiral::new(winding, bandwidth, n_arms, SpiralDirection::Clockwise),
            ccw: DirectionalSpiral::new(winding, bandwidth, n_arms, SpiralDirection::CounterClockwise),
        }
    }

    /// Create golden ratio bidirectional spiral
    pub fn golden() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(phi, PI / 8.0, 1)
    }

    /// Get both attention weight matrices
    pub fn attention_weights(
        &self,
        n_major: usize,
        n_minor: usize,
        device: &Device,
    ) -> TorusResult<(Tensor, Tensor)> {
        let cw_weights = self.cw.causal_attention_weights(n_major, n_minor, device)?;
        let ccw_weights = self.ccw.causal_attention_weights(n_major, n_minor, device)?;
        Ok((cw_weights, ccw_weights))
    }

    /// Combine CW and CCW flows with given weights
    pub fn combine_flows(
        &self,
        cw_output: &Tensor,
        ccw_output: &Tensor,
        cw_weight: f64,
    ) -> TorusResult<Tensor> {
        let ccw_weight = 1.0 - cw_weight;
        let combined = (cw_output * cw_weight)? + (ccw_output * ccw_weight)?;
        Ok(combined?)
    }
}

/// Multi-arm bidirectional spiral with configurable arms
#[derive(Debug, Clone)]
pub struct MultiArmSpiral {
    /// Number of spiral arms
    pub n_arms: usize,
    /// Winding number
    pub winding: f64,
    /// Bandwidth
    pub bandwidth: f64,
    /// Per-arm directional spirals (alternating CW/CCW)
    pub arms: Vec<DirectionalSpiral>,
}

impl MultiArmSpiral {
    /// Create a multi-arm spiral with alternating directions
    pub fn new(n_arms: usize, winding: f64, bandwidth: f64) -> Self {
        let arms = (0..n_arms)
            .map(|i| {
                let direction = if i % 2 == 0 {
                    SpiralDirection::Clockwise
                } else {
                    SpiralDirection::CounterClockwise
                };
                // Offset each arm
                let arm_winding = winding + (i as f64 * 0.1);
                DirectionalSpiral::new(arm_winding, bandwidth, 1, direction)
            })
            .collect();

        Self {
            n_arms,
            winding,
            bandwidth,
            arms,
        }
    }

    /// Create a 4-arm golden spiral (common in nature)
    pub fn golden_quad() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        Self::new(4, phi, PI / 6.0)
    }

    /// Generate combined attention from all arms
    pub fn combined_attention(
        &self,
        n_major: usize,
        n_minor: usize,
        device: &Device,
    ) -> TorusResult<Tensor> {
        let n = n_major * n_minor;
        let mut combined = Tensor::zeros((n, n), DType::F32, device)?;

        for arm in &self.arms {
            let arm_weights = arm.causal_attention_weights(n_major, n_minor, device)?;
            combined = (combined + arm_weights)?;
        }

        // Normalize
        let scale = 1.0 / self.n_arms as f64;
        combined = (combined * scale)?;

        Ok(combined)
    }
}

/// Topological charge calculator
pub struct TopologicalCharge;

impl TopologicalCharge {
    /// Compute winding number of a closed path on the torus
    pub fn winding_number(path: &[TorusCoordinate]) -> (i32, i32) {
        if path.len() < 2 {
            return (0, 0);
        }

        let mut total_du = 0.0;
        let mut total_dv = 0.0;

        for i in 0..path.len() {
            let curr = &path[i];
            let next = &path[(i + 1) % path.len()];

            // Signed angular difference (not absolute)
            let du = Self::signed_angular_diff(curr.u, next.u);
            let dv = Self::signed_angular_diff(curr.v, next.v);

            total_du += du;
            total_dv += dv;
        }

        // Winding numbers are total angle divided by 2π
        let winding_major = (total_du / (2.0 * PI)).round() as i32;
        let winding_minor = (total_dv / (2.0 * PI)).round() as i32;

        (winding_major, winding_minor)
    }

    fn signed_angular_diff(a: f64, b: f64) -> f64 {
        let diff = b - a;
        if diff > PI {
            diff - 2.0 * PI
        } else if diff < -PI {
            diff + 2.0 * PI
        } else {
            diff
        }
    }

    /// Compute total topological charge of a vortex system
    pub fn total_charge(vortices: &[Vortex]) -> (i32, i32) {
        vortices.iter().fold((0, 0), |(maj, min), v| {
            (maj + v.winding_major, min + v.winding_minor)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vortex_creation() {
        let vortex = Vortex::new(0.0, 0.0, 1.0, 1, 0);
        assert!((vortex.circulation - 1.0).abs() < 1e-10);
        assert_eq!(vortex.winding_major, 1);
    }

    #[test]
    fn test_vortex_velocity() {
        let vortex = Vortex::new(0.0, 0.0, 1.0, 0, 0);
        let point = TorusCoordinate::new(0.5, 0.0);
        let (vu, vv) = vortex.velocity_at(&point);
        
        // Velocity should be perpendicular to displacement
        // Since displacement is in u-direction, velocity should be in v-direction
        assert!(vu.abs() < 0.01, "u-component should be near zero, got {}", vu);
        assert!(vv.abs() > 0.01, "v-component should be non-zero, got {}", vv);
    }

    #[test]
    fn test_circulation_conservation() {
        let manifold = TorusManifold::unit();
        let mut dynamics = VortexDynamics::new(manifold, 0.01);
        dynamics.add_vortex_pair(PI, PI, 0.5, 1.0);
        
        let initial_circ = dynamics.total_circulation();
        dynamics.run(100);
        let final_circ = dynamics.total_circulation();
        
        // Total circulation should be conserved (zero for vortex pair)
        assert!((initial_circ - final_circ).abs() < 1e-10);
        assert!(initial_circ.abs() < 1e-10);
    }

    #[test]
    fn test_spiral_attention() {
        let spiral = SpiralAttention::golden();
        assert!(spiral.winding > 1.6 && spiral.winding < 1.62);
    }

    #[test]
    fn test_winding_number() {
        // A path that winds once around the major loop
        let path: Vec<TorusCoordinate> = (0..100)
            .map(|i| TorusCoordinate::new(2.0 * PI * i as f64 / 100.0, 0.0))
            .collect();
        
        let (winding_major, winding_minor) = TopologicalCharge::winding_number(&path);
        assert_eq!(winding_major, 1);
        assert_eq!(winding_minor, 0);
    }
}
