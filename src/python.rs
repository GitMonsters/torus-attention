//! Python bindings for torus attention via PyO3
//!
//! Provides a Python interface to the Rust torus attention implementation,
//! including:
//! - Core geometry (TorusCoordinate, TorusManifold)
//! - Vortex dynamics (Vortex, VortexDynamics, SpiralAttention)
//! - Bidirectional processing (FlowDirection, StreamId)
//! - 8-Stream parallel processor
//! - EMA Compounding
//! - Full BidirectionalTorusTransformer

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::geometry::{TorusCoordinate, TorusManifold, TorusDistanceMatrix};
use crate::vortex::{Vortex, VortexDynamics, SpiralAttention};
use crate::periodic::PeriodicBoundary;
use crate::bidirectional::FlowDirection;
use crate::parallel_streams::{StreamId, ParallelStreamConfig};
use crate::compounding::{CompoundingConfig, CompoundingStats};
use crate::integration::{BidirectionalTorusConfig, BidirectionalStats};
use std::f64::consts::PI;

/// Python wrapper for TorusCoordinate
#[pyclass(name = "TorusCoordinate")]
#[derive(Clone)]
pub struct PyTorusCoordinate {
    inner: TorusCoordinate,
}

#[pymethods]
impl PyTorusCoordinate {
    #[new]
    fn new(u: f64, v: f64) -> Self {
        Self {
            inner: TorusCoordinate::new(u, v),
        }
    }

    #[getter]
    fn u(&self) -> f64 {
        self.inner.u
    }

    #[getter]
    fn v(&self) -> f64 {
        self.inner.v
    }

    fn to_cartesian(&self, major_radius: f64, minor_radius: f64) -> (f64, f64, f64) {
        self.inner.to_cartesian(major_radius, minor_radius)
    }

    fn geodesic_distance(&self, other: &PyTorusCoordinate) -> f64 {
        self.inner.geodesic_distance(&other.inner)
    }

    fn spiral_position(&self, winding_number: f64) -> f64 {
        self.inner.spiral_position(winding_number)
    }

    fn __repr__(&self) -> String {
        format!("TorusCoordinate(u={:.4}, v={:.4})", self.inner.u, self.inner.v)
    }
}

/// Python wrapper for TorusManifold
#[pyclass(name = "TorusManifold")]
#[derive(Clone)]
pub struct PyTorusManifold {
    inner: TorusManifold,
}

#[pymethods]
impl PyTorusManifold {
    #[new]
    fn new(major_radius: f64, minor_radius: f64) -> PyResult<Self> {
        if major_radius <= 0.0 || minor_radius <= 0.0 {
            return Err(PyValueError::new_err("Radii must be positive"));
        }
        if major_radius <= minor_radius {
            return Err(PyValueError::new_err(
                "Major radius must be greater than minor radius",
            ));
        }
        Ok(Self {
            inner: TorusManifold::new(major_radius, minor_radius),
        })
    }

    #[staticmethod]
    fn unit() -> Self {
        Self {
            inner: TorusManifold::unit(),
        }
    }

    #[getter]
    fn major_radius(&self) -> f64 {
        self.inner.major_radius
    }

    #[getter]
    fn minor_radius(&self) -> f64 {
        self.inner.minor_radius
    }

    #[getter]
    fn aspect_ratio(&self) -> f64 {
        self.inner.aspect_ratio
    }

    fn surface_area(&self) -> f64 {
        self.inner.surface_area()
    }

    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn generate_grid(&self, n_major: usize, n_minor: usize) -> Vec<PyTorusCoordinate> {
        self.inner
            .generate_grid(n_major, n_minor)
            .into_iter()
            .map(|c| PyTorusCoordinate { inner: c })
            .collect()
    }

    fn generate_spiral(
        &self,
        n_points: usize,
        winding_number: f64,
        start_phase: f64,
    ) -> Vec<PyTorusCoordinate> {
        self.inner
            .generate_spiral(n_points, winding_number, start_phase)
            .into_iter()
            .map(|c| PyTorusCoordinate { inner: c })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TorusManifold(R={:.4}, r={:.4})",
            self.inner.major_radius, self.inner.minor_radius
        )
    }
}

/// Python wrapper for Vortex
#[pyclass(name = "Vortex")]
#[derive(Clone)]
pub struct PyVortex {
    inner: Vortex,
}

#[pymethods]
impl PyVortex {
    #[new]
    fn new(
        u: f64,
        v: f64,
        circulation: f64,
        winding_major: i32,
        winding_minor: i32,
    ) -> Self {
        Self {
            inner: Vortex::new(u, v, circulation, winding_major, winding_minor),
        }
    }

    #[getter]
    fn position(&self) -> PyTorusCoordinate {
        PyTorusCoordinate {
            inner: self.inner.position.clone(),
        }
    }

    #[getter]
    fn circulation(&self) -> f64 {
        self.inner.circulation
    }

    fn velocity_at(&self, point: &PyTorusCoordinate) -> (f64, f64) {
        self.inner.velocity_at(&point.inner)
    }

    fn spiral_path(&self, n_points: usize, total_angle: f64) -> Vec<PyTorusCoordinate> {
        self.inner
            .spiral_path(n_points, total_angle)
            .into_iter()
            .map(|c| PyTorusCoordinate { inner: c })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Vortex(u={:.4}, v={:.4}, Γ={:.4})",
            self.inner.position.u, self.inner.position.v, self.inner.circulation
        )
    }
}

/// Python wrapper for VortexDynamics
#[pyclass(name = "VortexDynamics")]
pub struct PyVortexDynamics {
    inner: VortexDynamics,
}

#[pymethods]
impl PyVortexDynamics {
    #[new]
    fn new(manifold: &PyTorusManifold, dt: f64) -> Self {
        Self {
            inner: VortexDynamics::new(manifold.inner.clone(), dt),
        }
    }

    fn add_vortex(&mut self, vortex: &PyVortex) {
        self.inner.add_vortex(vortex.inner.clone());
    }

    fn add_vortex_pair(&mut self, u: f64, v: f64, separation: f64, circulation: f64) {
        self.inner.add_vortex_pair(u, v, separation, circulation);
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn run(&mut self, n_steps: usize) {
        self.inner.run(n_steps);
    }

    fn total_velocity(&self, point: &PyTorusCoordinate) -> (f64, f64) {
        self.inner.total_velocity(&point.inner)
    }

    fn total_circulation(&self) -> f64 {
        self.inner.total_circulation()
    }

    fn energy(&self) -> f64 {
        self.inner.energy()
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }

    #[getter]
    fn n_vortices(&self) -> usize {
        self.inner.vortices.len()
    }

    fn get_vortex(&self, idx: usize) -> PyResult<PyVortex> {
        self.inner
            .vortices
            .get(idx)
            .map(|v| PyVortex { inner: v.clone() })
            .ok_or_else(|| PyValueError::new_err("Vortex index out of bounds"))
    }

    /// Get velocity field as numpy-compatible nested lists
    fn velocity_field(&self, n_major: usize, n_minor: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (vel_u, vel_v) = self.inner.velocity_field(n_major, n_minor);
        let vel_u_vec: Vec<Vec<f64>> = vel_u.outer_iter().map(|r| r.to_vec()).collect();
        let vel_v_vec: Vec<Vec<f64>> = vel_v.outer_iter().map(|r| r.to_vec()).collect();
        (vel_u_vec, vel_v_vec)
    }

    /// Get vorticity field as numpy-compatible nested list
    fn vorticity_field(&self, n_major: usize, n_minor: usize) -> Vec<Vec<f64>> {
        let vorticity = self.inner.vorticity_field(n_major, n_minor);
        vorticity.outer_iter().map(|r| r.to_vec()).collect()
    }
}

/// Python wrapper for SpiralAttention
#[pyclass(name = "SpiralAttention")]
pub struct PySpiralAttention {
    inner: SpiralAttention,
}

#[pymethods]
impl PySpiralAttention {
    #[new]
    fn new(winding: f64, bandwidth: f64, n_arms: usize) -> Self {
        Self {
            inner: SpiralAttention::new(winding, bandwidth, n_arms),
        }
    }

    #[staticmethod]
    fn golden() -> Self {
        Self {
            inner: SpiralAttention::golden(),
        }
    }

    fn sample_positions(&self, n_points: usize, start_phase: f64) -> Vec<PyTorusCoordinate> {
        self.inner
            .sample_positions(n_points, start_phase)
            .into_iter()
            .map(|c| PyTorusCoordinate { inner: c })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpiralAttention(winding={:.4}, bandwidth={:.4}, n_arms={})",
            self.inner.winding, self.inner.bandwidth, self.inner.n_arms
        )
    }
}

/// Python wrapper for PeriodicBoundary
#[pyclass(name = "PeriodicBoundary")]
pub struct PyPeriodicBoundary {
    inner: PeriodicBoundary,
}

#[pymethods]
impl PyPeriodicBoundary {
    #[new]
    fn new(n_major: usize, n_minor: usize) -> Self {
        Self {
            inner: PeriodicBoundary::new(n_major, n_minor),
        }
    }

    #[getter]
    fn n_major(&self) -> usize {
        self.inner.n_major
    }

    #[getter]
    fn n_minor(&self) -> usize {
        self.inner.n_minor
    }

    fn wrap_2d(&self, i: i64, j: i64) -> (usize, usize) {
        self.inner.wrap_2d(i, j)
    }

    fn gaussian_kernel(&self, sigma: f64, kernel_size: usize) -> Vec<Vec<f64>> {
        let kernel = self.inner.gaussian_kernel(sigma, kernel_size);
        kernel.outer_iter().map(|r| r.to_vec()).collect()
    }
}

/// Compute geodesic distance matrix for a set of coordinates
#[pyfunction]
fn compute_distance_matrix(coords: Vec<PyTorusCoordinate>) -> Vec<Vec<f64>> {
    let inner_coords: Vec<TorusCoordinate> = coords.into_iter().map(|c| c.inner).collect();
    let matrix = TorusDistanceMatrix::from_coordinates(&inner_coords);
    matrix.distances
}

/// Compute attention weights from distance matrix
#[pyfunction]
fn distance_to_attention(distances: Vec<Vec<f64>>, sigma: f64) -> Vec<Vec<f64>> {
    let neg_inv_2sigma2 = -0.5 / (sigma * sigma);
    distances
        .iter()
        .map(|row| {
            let weights: Vec<f64> = row
                .iter()
                .map(|&d| (d * d * neg_inv_2sigma2).exp())
                .collect();
            let sum: f64 = weights.iter().sum();
            weights.iter().map(|&w| w / sum).collect()
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// BIDIRECTIONAL FLOW DIRECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for FlowDirection
#[pyclass(name = "FlowDirection")]
#[derive(Clone)]
pub struct PyFlowDirection {
    forward: bool,
}

#[pymethods]
impl PyFlowDirection {
    #[staticmethod]
    fn forward() -> Self {
        Self { forward: true }
    }

    #[staticmethod]
    fn backward() -> Self {
        Self { forward: false }
    }

    fn is_forward(&self) -> bool {
        self.forward
    }

    fn is_backward(&self) -> bool {
        !self.forward
    }

    fn flip(&self) -> Self {
        Self { forward: !self.forward }
    }

    fn __repr__(&self) -> String {
        if self.forward {
            "FlowDirection.Forward".to_string()
        } else {
            "FlowDirection.Backward".to_string()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STREAM IDENTIFIER (8 STREAMS)
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for StreamId
#[pyclass(name = "StreamId")]
#[derive(Clone)]
pub struct PyStreamId {
    id: u8,
}

#[pymethods]
impl PyStreamId {
    #[staticmethod]
    fn major_forward() -> Self { Self { id: 0 } }
    
    #[staticmethod]
    fn major_backward() -> Self { Self { id: 1 } }
    
    #[staticmethod]
    fn minor_forward() -> Self { Self { id: 2 } }
    
    #[staticmethod]
    fn minor_backward() -> Self { Self { id: 3 } }
    
    #[staticmethod]
    fn spiral_cw() -> Self { Self { id: 4 } }
    
    #[staticmethod]
    fn spiral_ccw() -> Self { Self { id: 5 } }
    
    #[staticmethod]
    fn cross_u_to_v() -> Self { Self { id: 6 } }
    
    #[staticmethod]
    fn cross_v_to_u() -> Self { Self { id: 7 } }

    #[staticmethod]
    fn all() -> Vec<PyStreamId> {
        (0..8).map(|id| PyStreamId { id }).collect()
    }

    fn index(&self) -> usize {
        self.id as usize
    }

    fn name(&self) -> &'static str {
        match self.id {
            0 => "major_forward",
            1 => "major_backward",
            2 => "minor_forward",
            3 => "minor_backward",
            4 => "spiral_cw",
            5 => "spiral_ccw",
            6 => "cross_u_to_v",
            7 => "cross_v_to_u",
            _ => "unknown",
        }
    }

    fn is_forward(&self) -> bool {
        matches!(self.id, 0 | 2 | 4 | 6)
    }

    fn pair(&self) -> Self {
        let paired = match self.id {
            0 => 1,
            1 => 0,
            2 => 3,
            3 => 2,
            4 => 5,
            5 => 4,
            6 => 7,
            7 => 6,
            _ => self.id,
        };
        Self { id: paired }
    }

    fn __repr__(&self) -> String {
        format!("StreamId.{}", self.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PARALLEL STREAM CONFIG
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for ParallelStreamConfig
#[pyclass(name = "ParallelStreamConfig")]
#[derive(Clone)]
pub struct PyParallelStreamConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_major: usize,
    pub n_minor: usize,
    pub spiral_winding: f64,
    pub weight_temperature: f64,
    pub parallel: bool,
    pub dropout: f64,
}

#[pymethods]
impl PyParallelStreamConfig {
    #[new]
    #[pyo3(signature = (
        d_model = 256,
        n_heads = 8,
        n_major = 32,
        n_minor = 16,
        spiral_winding = 1.618033988749895,
        weight_temperature = 1.0,
        parallel = true,
        dropout = 0.1
    ))]
    fn new(
        d_model: usize,
        n_heads: usize,
        n_major: usize,
        n_minor: usize,
        spiral_winding: f64,
        weight_temperature: f64,
        parallel: bool,
        dropout: f64,
    ) -> Self {
        Self {
            d_model,
            n_heads,
            n_major,
            n_minor,
            spiral_winding,
            weight_temperature,
            parallel,
            dropout,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            n_major: 32,
            n_minor: 16,
            spiral_winding: 1.618033988749895,
            weight_temperature: 1.0,
            parallel: true,
            dropout: 0.1,
        }
    }

    fn seq_len(&self) -> usize {
        self.n_major * self.n_minor
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelStreamConfig(d_model={}, n_heads={}, grid={}x{}, spiral={:.4})",
            self.d_model, self.n_heads, self.n_major, self.n_minor, self.spiral_winding
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOUNDING CONFIG
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for CompoundingConfig
#[pyclass(name = "CompoundingConfig")]
#[derive(Clone)]
pub struct PyCompoundingConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub base_alpha: f64,
    pub min_alpha: f64,
    pub max_alpha: f64,
    pub layer_scale: f64,
    pub use_momentum: bool,
    pub momentum_beta: f64,
    pub learnable_alpha: bool,
}

#[pymethods]
impl PyCompoundingConfig {
    #[new]
    #[pyo3(signature = (
        n_layers = 6,
        d_model = 256,
        base_alpha = 0.9,
        min_alpha = 0.1,
        max_alpha = 0.99,
        layer_scale = 0.95,
        use_momentum = true,
        momentum_beta = 0.9,
        learnable_alpha = true
    ))]
    fn new(
        n_layers: usize,
        d_model: usize,
        base_alpha: f64,
        min_alpha: f64,
        max_alpha: f64,
        layer_scale: f64,
        use_momentum: bool,
        momentum_beta: f64,
        learnable_alpha: bool,
    ) -> Self {
        Self {
            n_layers,
            d_model,
            base_alpha,
            min_alpha,
            max_alpha,
            layer_scale,
            use_momentum,
            momentum_beta,
            learnable_alpha,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            n_layers: 6,
            d_model: 256,
            base_alpha: 0.9,
            min_alpha: 0.1,
            max_alpha: 0.99,
            layer_scale: 0.95,
            use_momentum: true,
            momentum_beta: 0.9,
            learnable_alpha: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompoundingConfig(layers={}, α={:.2}, learnable={}, momentum={})",
            self.n_layers, self.base_alpha, self.learnable_alpha, self.use_momentum
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BIDIRECTIONAL TORUS CONFIG (MAIN CONFIG)
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for BidirectionalTorusConfig
#[pyclass(name = "BidirectionalTorusConfig")]
#[derive(Clone)]
pub struct PyBidirectionalTorusConfig {
    inner: BidirectionalTorusConfig,
}

#[pymethods]
impl PyBidirectionalTorusConfig {
    #[new]
    #[pyo3(signature = (
        d_model = 256,
        d_ff = 1024,
        n_heads = 8,
        n_layers = 6,
        n_major = 32,
        n_minor = 16,
        major_radius = 2.0,
        minor_radius = 1.0,
        use_parallel_streams = true,
        use_compounding = true,
        use_multi_scale = false,
        ema_alpha = 0.9,
        learnable_alpha = true,
        use_momentum = true,
        spiral_winding = 1.618033988749895,
        weight_temperature = 1.0,
        parallel_execution = true,
        use_geodesic_bias = true,
        geodesic_sigma = 0.5,
        dropout = 0.1,
        n_pos_frequencies = 16
    ))]
    fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_layers: usize,
        n_major: usize,
        n_minor: usize,
        major_radius: f64,
        minor_radius: f64,
        use_parallel_streams: bool,
        use_compounding: bool,
        use_multi_scale: bool,
        ema_alpha: f64,
        learnable_alpha: bool,
        use_momentum: bool,
        spiral_winding: f64,
        weight_temperature: f64,
        parallel_execution: bool,
        use_geodesic_bias: bool,
        geodesic_sigma: f64,
        dropout: f64,
        n_pos_frequencies: usize,
    ) -> Self {
        Self {
            inner: BidirectionalTorusConfig {
                d_model,
                d_ff,
                n_heads,
                n_layers,
                n_major,
                n_minor,
                major_radius,
                minor_radius,
                use_parallel_streams,
                use_compounding,
                use_multi_scale,
                ema_alpha,
                learnable_alpha,
                use_momentum,
                spiral_winding,
                weight_temperature,
                parallel_execution,
                use_geodesic_bias,
                geodesic_sigma,
                dropout,
                n_pos_frequencies,
            },
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: BidirectionalTorusConfig::default(),
        }
    }

    #[staticmethod]
    fn small() -> Self {
        Self {
            inner: BidirectionalTorusConfig {
                d_model: 128,
                d_ff: 512,
                n_heads: 4,
                n_layers: 4,
                n_major: 16,
                n_minor: 8,
                ..BidirectionalTorusConfig::default()
            },
        }
    }

    #[staticmethod]
    fn large() -> Self {
        Self {
            inner: BidirectionalTorusConfig {
                d_model: 512,
                d_ff: 2048,
                n_heads: 16,
                n_layers: 12,
                n_major: 64,
                n_minor: 32,
                ..BidirectionalTorusConfig::default()
            },
        }
    }

    // Getters
    #[getter]
    fn d_model(&self) -> usize { self.inner.d_model }
    
    #[getter]
    fn d_ff(&self) -> usize { self.inner.d_ff }
    
    #[getter]
    fn n_heads(&self) -> usize { self.inner.n_heads }
    
    #[getter]
    fn n_layers(&self) -> usize { self.inner.n_layers }
    
    #[getter]
    fn n_major(&self) -> usize { self.inner.n_major }
    
    #[getter]
    fn n_minor(&self) -> usize { self.inner.n_minor }
    
    #[getter]
    fn major_radius(&self) -> f64 { self.inner.major_radius }
    
    #[getter]
    fn minor_radius(&self) -> f64 { self.inner.minor_radius }
    
    #[getter]
    fn use_parallel_streams(&self) -> bool { self.inner.use_parallel_streams }
    
    #[getter]
    fn use_compounding(&self) -> bool { self.inner.use_compounding }
    
    #[getter]
    fn ema_alpha(&self) -> f64 { self.inner.ema_alpha }
    
    #[getter]
    fn learnable_alpha(&self) -> bool { self.inner.learnable_alpha }
    
    #[getter]
    fn spiral_winding(&self) -> f64 { self.inner.spiral_winding }

    fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    fn to_parallel_config(&self) -> PyParallelStreamConfig {
        let pc = self.inner.to_parallel_config();
        PyParallelStreamConfig {
            d_model: pc.d_model,
            n_heads: pc.n_heads,
            n_major: pc.n_major,
            n_minor: pc.n_minor,
            spiral_winding: pc.spiral_winding,
            weight_temperature: pc.weight_temperature,
            parallel: pc.parallel,
            dropout: pc.dropout,
        }
    }

    fn to_compounding_config(&self) -> PyCompoundingConfig {
        let cc = self.inner.to_compounding_config();
        PyCompoundingConfig {
            n_layers: cc.n_layers,
            d_model: cc.d_model,
            base_alpha: cc.base_alpha,
            min_alpha: cc.min_alpha,
            max_alpha: cc.max_alpha,
            layer_scale: cc.layer_scale,
            use_momentum: cc.use_momentum,
            momentum_beta: cc.momentum_beta,
            learnable_alpha: cc.learnable_alpha,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BidirectionalTorusConfig(d_model={}, layers={}, heads={}, grid={}x{}, streams={}, compounding={})",
            self.inner.d_model,
            self.inner.n_layers,
            self.inner.n_heads,
            self.inner.n_major,
            self.inner.n_minor,
            self.inner.use_parallel_streams,
            self.inner.use_compounding
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BIDIRECTIONAL STATS
// ═══════════════════════════════════════════════════════════════════════════

/// Python wrapper for BidirectionalStats
#[pyclass(name = "BidirectionalStats")]
#[derive(Clone)]
pub struct PyBidirectionalStats {
    /// Stream weights per layer as dict of {stream_name: weight}
    stream_weights: Vec<Vec<(String, f32)>>,
    /// Compounding alphas per layer
    compounding_alphas: Vec<f64>,
    /// Multi-scale weights if applicable
    scale_weights: Option<Vec<f32>>,
}

#[pymethods]
impl PyBidirectionalStats {
    #[new]
    fn new(
        stream_weights: Vec<Vec<(String, f32)>>,
        compounding_alphas: Vec<f64>,
        scale_weights: Option<Vec<f32>>,
    ) -> Self {
        Self {
            stream_weights,
            compounding_alphas,
            scale_weights,
        }
    }

    #[getter]
    fn get_stream_weights(&self) -> Vec<Vec<(String, f32)>> {
        self.stream_weights.clone()
    }

    #[getter]
    fn get_compounding_alphas(&self) -> Vec<f64> {
        self.compounding_alphas.clone()
    }

    #[getter]
    fn get_scale_weights(&self) -> Option<Vec<f32>> {
        self.scale_weights.clone()
    }

    fn layer_weights(&self, layer: usize) -> PyResult<Vec<(String, f32)>> {
        self.stream_weights
            .get(layer)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Layer {} out of bounds", layer)))
    }

    fn layer_alpha(&self, layer: usize) -> PyResult<f64> {
        self.compounding_alphas
            .get(layer)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Layer {} out of bounds", layer)))
    }

    fn summary(&self) -> String {
        let mut s = String::from("═══ Bidirectional Torus Stats ═══\n");
        
        s.push_str("\n── Stream Weights ──\n");
        for (layer_idx, weights) in self.stream_weights.iter().enumerate() {
            s.push_str(&format!("Layer {}:\n", layer_idx));
            for (name, w) in weights {
                s.push_str(&format!("  {:15} {:.4}\n", name, w));
            }
        }

        if !self.compounding_alphas.is_empty() {
            s.push_str("\n── Compounding Alphas ──\n");
            for (layer_idx, alpha) in self.compounding_alphas.iter().enumerate() {
                s.push_str(&format!("Layer {}: α = {:.4}\n", layer_idx, alpha));
            }
        }

        if let Some(ref scales) = self.scale_weights {
            s.push_str("\n── Multi-Scale Weights ──\n");
            s.push_str(&format!("Fast:   {:.4}\n", scales.get(0).unwrap_or(&0.0)));
            s.push_str(&format!("Medium: {:.4}\n", scales.get(1).unwrap_or(&0.0)));
            s.push_str(&format!("Slow:   {:.4}\n", scales.get(2).unwrap_or(&0.0)));
        }

        s
    }

    fn __repr__(&self) -> String {
        format!(
            "BidirectionalStats(layers={}, alphas={:?})",
            self.stream_weights.len(),
            self.compounding_alphas
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute spiral position on torus
#[pyfunction]
fn spiral_position(u: f64, v: f64, winding_number: f64) -> f64 {
    (u + winding_number * v) % (2.0 * PI)
}

/// Compute golden ratio spiral positions
#[pyfunction]
fn golden_spiral_positions(n_points: usize, start_phase: f64) -> Vec<PyTorusCoordinate> {
    let phi = 1.618033988749895; // Golden ratio
    let mut positions = Vec::with_capacity(n_points);
    
    for i in 0..n_points {
        let t = i as f64 / n_points as f64;
        let u = (start_phase + 2.0 * PI * t) % (2.0 * PI);
        let v = (phi * 2.0 * PI * t) % (2.0 * PI);
        positions.push(PyTorusCoordinate {
            inner: TorusCoordinate::new(u, v),
        });
    }
    
    positions
}

/// Generate 8-stream attention mask indices
#[pyfunction]
fn stream_mask_info(stream_name: &str, n_major: usize, n_minor: usize) -> PyResult<String> {
    let info = match stream_name {
        "major_forward" => format!(
            "Major Forward: causal along u-dimension (0→2π)\nGrid: {}x{}, attends to positions with u ≤ current",
            n_major, n_minor
        ),
        "major_backward" => format!(
            "Major Backward: anti-causal along u-dimension (2π→0)\nGrid: {}x{}, attends to positions with u ≥ current",
            n_major, n_minor
        ),
        "minor_forward" => format!(
            "Minor Forward: causal along v-dimension (0→2π)\nGrid: {}x{}, attends to positions with v ≤ current",
            n_major, n_minor
        ),
        "minor_backward" => format!(
            "Minor Backward: anti-causal along v-dimension (2π→0)\nGrid: {}x{}, attends to positions with v ≥ current",
            n_major, n_minor
        ),
        "spiral_cw" => format!(
            "Spiral CW: clockwise golden ratio spiral\nGrid: {}x{}, spiral_pos = u + φ*v, causal along spiral",
            n_major, n_minor
        ),
        "spiral_ccw" => format!(
            "Spiral CCW: counter-clockwise golden ratio spiral\nGrid: {}x{}, spiral_pos = u + φ*v, anti-causal along spiral",
            n_major, n_minor
        ),
        "cross_u_to_v" => format!(
            "Cross U→V: major-to-minor coupling\nGrid: {}x{}, u-position modulates v-attention",
            n_major, n_minor
        ),
        "cross_v_to_u" => format!(
            "Cross V→U: minor-to-major coupling\nGrid: {}x{}, v-position modulates u-attention",
            n_major, n_minor
        ),
        _ => return Err(PyValueError::new_err(format!("Unknown stream: {}", stream_name))),
    };
    Ok(info)
}

/// Python module definition
#[pymodule]
fn torus_attention(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core geometry classes
    m.add_class::<PyTorusCoordinate>()?;
    m.add_class::<PyTorusManifold>()?;
    m.add_class::<PyVortex>()?;
    m.add_class::<PyVortexDynamics>()?;
    m.add_class::<PySpiralAttention>()?;
    m.add_class::<PyPeriodicBoundary>()?;
    
    // Bidirectional/parallel classes
    m.add_class::<PyFlowDirection>()?;
    m.add_class::<PyStreamId>()?;
    m.add_class::<PyParallelStreamConfig>()?;
    m.add_class::<PyCompoundingConfig>()?;
    m.add_class::<PyBidirectionalTorusConfig>()?;
    m.add_class::<PyBidirectionalStats>()?;
    
    // Functions
    m.add_function(wrap_pyfunction!(compute_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(distance_to_attention, m)?)?;
    m.add_function(wrap_pyfunction!(spiral_position, m)?)?;
    m.add_function(wrap_pyfunction!(golden_spiral_positions, m)?)?;
    m.add_function(wrap_pyfunction!(stream_mask_info, m)?)?;
    
    // Constants
    m.add("PI", PI)?;
    m.add("TWO_PI", 2.0 * PI)?;
    m.add("GOLDEN_RATIO", 1.618033988749895)?;
    m.add("N_STREAMS", 8)?;
    m.add("STREAM_NAMES", vec![
        "major_forward", "major_backward",
        "minor_forward", "minor_backward",
        "spiral_cw", "spiral_ccw",
        "cross_u_to_v", "cross_v_to_u",
    ])?;
    
    Ok(())
}
