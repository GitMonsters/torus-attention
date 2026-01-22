//! # Tetrahedral Mixture of Agents (MOA)
//!
//! Implements a 64/128-agent architecture based on tetrahedral geometry,
//! integrated with the torus attention mechanism for massive parallel processing.
//!
//! ## Geometric Foundation
//!
//! ### 64-Point Tetrahedron
//! ```text
//!                    V0 (Vertex 0 - 16 agents)
//!                   /|\
//!                  / | \
//!                 /  |  \
//!                /   |   \
//!               /    |    \
//!              /     |     \
//!             /      |      \
//!            /       |       \
//!           /________|________\
//!          V1        |        V2
//!     (16 agents)    |    (16 agents)
//!                    |
//!                   V3 (16 agents)
//!
//! 4 vertices × 16 agents = 64 total agents
//! ```
//!
//! ### 128-Point Star Tetrahedron (Merkaba)
//! ```text
//! Two interlocking tetrahedra:
//! - Inner tetrahedron: 64 agents (pointing up)
//! - Outer tetrahedron: 64 agents (pointing down)
//! - Total: 128 agents with dual orientation
//! ```
//!
//! ## Integration with Torus
//!
//! The tetrahedron is embedded in the torus:
//! - Vertices map to torus coordinates via projection
//! - Agents communicate through both tetrahedral edges AND torus geodesics
//! - 8 torus streams × 8 tetrahedral faces = 64 hybrid pathways
//!
//! ## Scaling to 1.7T Parameters
//!
//! Multi-torus configuration:
//! - T³ = T¹ × T¹ × T¹ (3-torus)
//! - Each torus ring: ~500-600B parameters
//! - 128 MOA agents as routing/coordination nodes
//! - Total: 1.7T+ parameters with structured sparsity

use crate::geometry::TorusCoordinate;
use crate::parallel_streams::StreamId;
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// TETRAHEDRAL GEOMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// 3D point in Cartesian coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Euclidean distance to another point
    pub fn distance(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 1e-10 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }

    /// Linear interpolation between two points
    pub fn lerp(&self, other: &Point3D, t: f64) -> Self {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z),
        )
    }

    /// Project onto torus surface (approximate mapping)
    pub fn to_torus_coordinate(&self, major_radius: f64, minor_radius: f64) -> TorusCoordinate {
        // Map 3D point to torus angular coordinates
        // u = angle around major circle (from x-y projection)
        // v = angle around minor circle (from height z)
        let u = self.y.atan2(self.x);
        let r_xy = (self.x * self.x + self.y * self.y).sqrt();
        let v = self.z.atan2(r_xy - major_radius);
        TorusCoordinate::new(u, v)
    }
}

/// Vertex identifier in tetrahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TetrahedronVertex {
    V0 = 0, // Top vertex
    V1 = 1, // Base vertex 1
    V2 = 2, // Base vertex 2
    V3 = 3, // Base vertex 3
}

impl TetrahedronVertex {
    pub fn all() -> [TetrahedronVertex; 4] {
        [
            TetrahedronVertex::V0,
            TetrahedronVertex::V1,
            TetrahedronVertex::V2,
            TetrahedronVertex::V3,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Get the three vertices that form the opposite face
    pub fn opposite_face(&self) -> [TetrahedronVertex; 3] {
        match self {
            TetrahedronVertex::V0 => [
                TetrahedronVertex::V1,
                TetrahedronVertex::V2,
                TetrahedronVertex::V3,
            ],
            TetrahedronVertex::V1 => [
                TetrahedronVertex::V0,
                TetrahedronVertex::V2,
                TetrahedronVertex::V3,
            ],
            TetrahedronVertex::V2 => [
                TetrahedronVertex::V0,
                TetrahedronVertex::V1,
                TetrahedronVertex::V3,
            ],
            TetrahedronVertex::V3 => [
                TetrahedronVertex::V0,
                TetrahedronVertex::V1,
                TetrahedronVertex::V2,
            ],
        }
    }
}

/// Face identifier in tetrahedron (4 faces)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TetrahedronFace {
    F0 = 0, // Opposite to V0 (base)
    F1 = 1, // Opposite to V1
    F2 = 2, // Opposite to V2
    F3 = 3, // Opposite to V3
}

impl TetrahedronFace {
    pub fn all() -> [TetrahedronFace; 4] {
        [
            TetrahedronFace::F0,
            TetrahedronFace::F1,
            TetrahedronFace::F2,
            TetrahedronFace::F3,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Get the vertex opposite to this face
    pub fn opposite_vertex(&self) -> TetrahedronVertex {
        match self {
            TetrahedronFace::F0 => TetrahedronVertex::V0,
            TetrahedronFace::F1 => TetrahedronVertex::V1,
            TetrahedronFace::F2 => TetrahedronVertex::V2,
            TetrahedronFace::F3 => TetrahedronVertex::V3,
        }
    }

    /// Get the three vertices forming this face
    pub fn vertices(&self) -> [TetrahedronVertex; 3] {
        self.opposite_vertex().opposite_face()
    }
}

/// Edge identifier in tetrahedron (6 edges)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TetrahedronEdge {
    E01 = 0, // V0-V1
    E02 = 1, // V0-V2
    E03 = 2, // V0-V3
    E12 = 3, // V1-V2
    E13 = 4, // V1-V3
    E23 = 5, // V2-V3
}

impl TetrahedronEdge {
    pub fn all() -> [TetrahedronEdge; 6] {
        [
            TetrahedronEdge::E01,
            TetrahedronEdge::E02,
            TetrahedronEdge::E03,
            TetrahedronEdge::E12,
            TetrahedronEdge::E13,
            TetrahedronEdge::E23,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Get the two vertices forming this edge
    pub fn vertices(&self) -> (TetrahedronVertex, TetrahedronVertex) {
        match self {
            TetrahedronEdge::E01 => (TetrahedronVertex::V0, TetrahedronVertex::V1),
            TetrahedronEdge::E02 => (TetrahedronVertex::V0, TetrahedronVertex::V2),
            TetrahedronEdge::E03 => (TetrahedronVertex::V0, TetrahedronVertex::V3),
            TetrahedronEdge::E12 => (TetrahedronVertex::V1, TetrahedronVertex::V2),
            TetrahedronEdge::E13 => (TetrahedronVertex::V1, TetrahedronVertex::V3),
            TetrahedronEdge::E23 => (TetrahedronVertex::V2, TetrahedronVertex::V3),
        }
    }
}

/// Regular tetrahedron geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tetrahedron {
    /// Vertex positions in 3D space
    pub vertices: [Point3D; 4],
    /// Edge length
    pub edge_length: f64,
    /// Center (centroid)
    pub center: Point3D,
    /// Circumradius (distance from center to vertices)
    pub circumradius: f64,
    /// Inradius (distance from center to face centers)
    pub inradius: f64,
}

impl Tetrahedron {
    /// Create a regular tetrahedron with given edge length centered at origin
    pub fn regular(edge_length: f64) -> Self {
        // Regular tetrahedron vertices (centered at origin)
        // Using coordinates that give edge length = edge_length
        let a = edge_length / (2.0 * 2.0_f64.sqrt());
        let h = edge_length * (2.0 / 3.0_f64).sqrt(); // Height

        let vertices = [
            Point3D::new(0.0, 0.0, h * 0.75),                 // V0: top
            Point3D::new(-a * 3.0_f64.sqrt(), -a, -h * 0.25), // V1: base
            Point3D::new(a * 3.0_f64.sqrt(), -a, -h * 0.25),  // V2: base
            Point3D::new(0.0, 2.0 * a, -h * 0.25),            // V3: base
        ];

        let center = Point3D::origin(); // Centered at origin

        // Circumradius: R = a * sqrt(6) / 4 for unit edge length
        let circumradius = edge_length * (6.0_f64.sqrt()) / 4.0;
        // Inradius: r = a * sqrt(6) / 12
        let inradius = edge_length * (6.0_f64.sqrt()) / 12.0;

        Self {
            vertices,
            edge_length,
            center,
            circumradius,
            inradius,
        }
    }

    /// Create a unit tetrahedron (edge length = 1)
    pub fn unit() -> Self {
        Self::regular(1.0)
    }

    /// Get vertex position
    pub fn vertex(&self, v: TetrahedronVertex) -> Point3D {
        self.vertices[v.index()]
    }

    /// Get edge midpoint
    pub fn edge_midpoint(&self, e: TetrahedronEdge) -> Point3D {
        let (v1, v2) = e.vertices();
        self.vertex(v1).lerp(&self.vertex(v2), 0.5)
    }

    /// Get face center (centroid of triangle)
    pub fn face_center(&self, f: TetrahedronFace) -> Point3D {
        let verts = f.vertices();
        let p0 = self.vertex(verts[0]);
        let p1 = self.vertex(verts[1]);
        let p2 = self.vertex(verts[2]);
        Point3D::new(
            (p0.x + p1.x + p2.x) / 3.0,
            (p0.y + p1.y + p2.y) / 3.0,
            (p0.z + p1.z + p2.z) / 3.0,
        )
    }

    /// Generate N points distributed on this tetrahedron
    /// Points are placed at vertices, along edges, and on faces
    pub fn distribute_points(&self, n_per_vertex: usize) -> Vec<TetrahedralPoint> {
        let mut points = Vec::new();

        // Distribute points around each vertex
        for v in TetrahedronVertex::all() {
            let vertex_pos = self.vertex(v);
            let face = v.opposite_face();

            for i in 0..n_per_vertex {
                // Create points radiating from vertex toward face center
                let t = (i as f64 + 1.0) / (n_per_vertex as f64 + 1.0);
                let face_center = Point3D::new(
                    (self.vertex(face[0]).x + self.vertex(face[1]).x + self.vertex(face[2]).x)
                        / 3.0,
                    (self.vertex(face[0]).y + self.vertex(face[1]).y + self.vertex(face[2]).y)
                        / 3.0,
                    (self.vertex(face[0]).z + self.vertex(face[1]).z + self.vertex(face[2]).z)
                        / 3.0,
                );

                // Rotate around vertex axis for distribution
                let angle = 2.0 * PI * (i as f64) / (n_per_vertex as f64);
                let r = t * self.inradius * 0.5;
                let offset = Point3D::new(r * angle.cos(), r * angle.sin(), 0.0);

                let pos = Point3D::new(
                    vertex_pos.x * (1.0 - t * 0.3) + face_center.x * (t * 0.3) + offset.x * 0.1,
                    vertex_pos.y * (1.0 - t * 0.3) + face_center.y * (t * 0.3) + offset.y * 0.1,
                    vertex_pos.z * (1.0 - t * 0.3) + face_center.z * (t * 0.3) + offset.z * 0.1,
                );

                points.push(TetrahedralPoint {
                    position: pos,
                    primary_vertex: v,
                    local_index: i,
                    global_index: v.index() * n_per_vertex + i,
                });
            }
        }

        points
    }
}

/// A point within the tetrahedral structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrahedralPoint {
    /// 3D position
    pub position: Point3D,
    /// Primary vertex this point is associated with
    pub primary_vertex: TetrahedronVertex,
    /// Local index within the vertex cluster
    pub local_index: usize,
    /// Global index across all agents
    pub global_index: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// STAR TETRAHEDRON (MERKABA) - 128 AGENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Orientation of tetrahedron in star tetrahedron
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TetrahedronOrientation {
    /// Pointing up (apex at top)
    Up,
    /// Pointing down (apex at bottom) - inverted
    Down,
}

/// Star Tetrahedron (Merkaba) - two interlocking tetrahedra
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarTetrahedron {
    /// Upward-pointing tetrahedron
    pub upper: Tetrahedron,
    /// Downward-pointing tetrahedron (inverted)
    pub lower: Tetrahedron,
    /// Shared center
    pub center: Point3D,
    /// Edge length (same for both)
    pub edge_length: f64,
}

impl StarTetrahedron {
    /// Create a star tetrahedron with given edge length
    pub fn new(edge_length: f64) -> Self {
        let upper = Tetrahedron::regular(edge_length);

        // Create inverted tetrahedron (rotate 180° around z-axis and flip)
        let lower = Tetrahedron {
            vertices: [
                Point3D::new(
                    -upper.vertices[0].x,
                    -upper.vertices[0].y,
                    -upper.vertices[0].z,
                ),
                Point3D::new(
                    -upper.vertices[1].x,
                    -upper.vertices[1].y,
                    -upper.vertices[1].z,
                ),
                Point3D::new(
                    -upper.vertices[2].x,
                    -upper.vertices[2].y,
                    -upper.vertices[2].z,
                ),
                Point3D::new(
                    -upper.vertices[3].x,
                    -upper.vertices[3].y,
                    -upper.vertices[3].z,
                ),
            ],
            edge_length,
            center: Point3D::origin(),
            circumradius: upper.circumradius,
            inradius: upper.inradius,
        };

        Self {
            upper,
            lower,
            center: Point3D::origin(),
            edge_length,
        }
    }

    /// Generate 128 points distributed across both tetrahedra
    pub fn distribute_points(&self, n_per_vertex: usize) -> Vec<StarTetrahedralPoint> {
        let mut points = Vec::new();

        // Upper tetrahedron (64 points with n_per_vertex=16)
        for point in self.upper.distribute_points(n_per_vertex) {
            points.push(StarTetrahedralPoint {
                position: point.position,
                orientation: TetrahedronOrientation::Up,
                primary_vertex: point.primary_vertex,
                local_index: point.local_index,
                global_index: point.global_index,
            });
        }

        // Lower tetrahedron (64 points with n_per_vertex=16)
        let offset = n_per_vertex * 4; // 64 for upper
        for point in self.lower.distribute_points(n_per_vertex) {
            points.push(StarTetrahedralPoint {
                position: point.position,
                orientation: TetrahedronOrientation::Down,
                primary_vertex: point.primary_vertex,
                local_index: point.local_index,
                global_index: point.global_index + offset,
            });
        }

        points
    }
}

/// A point within the star tetrahedron structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarTetrahedralPoint {
    /// 3D position
    pub position: Point3D,
    /// Which tetrahedron (up or down)
    pub orientation: TetrahedronOrientation,
    /// Primary vertex this point is associated with
    pub primary_vertex: TetrahedronVertex,
    /// Local index within the vertex cluster
    pub local_index: usize,
    /// Global index across all 128 agents
    pub global_index: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT SPECIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Agent role based on position in tetrahedral structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TetrahedralAgentRole {
    /// Vertex agent - high-level coordination (4 per tetrahedron)
    Vertex(TetrahedronVertex),
    /// Edge agent - inter-vertex communication (along 6 edges)
    Edge(TetrahedronEdge),
    /// Face agent - surface processing (4 faces)
    Face(TetrahedronFace),
    /// Interior agent - deep processing
    Interior,
}

/// Specialization domain for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentDomain {
    /// Language and symbolic processing
    Language,
    /// Visual and spatial processing
    Vision,
    /// Reasoning and logic
    Reasoning,
    /// Memory and retrieval
    Memory,
    /// Planning and goal management
    Planning,
    /// Perception and sensory integration
    Perception,
    /// Action and motor control
    Action,
    /// Meta-cognition and self-reflection
    MetaCognition,
}

impl AgentDomain {
    /// Map vertex to primary domain
    pub fn from_vertex(v: TetrahedronVertex) -> Self {
        match v {
            TetrahedronVertex::V0 => AgentDomain::Reasoning, // Apex - high-level
            TetrahedronVertex::V1 => AgentDomain::Language,  // Base 1
            TetrahedronVertex::V2 => AgentDomain::Vision,    // Base 2
            TetrahedronVertex::V3 => AgentDomain::Memory,    // Base 3
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TETRAHEDRAL AGENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a tetrahedral agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrahedralAgentConfig {
    /// Agent hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads per agent
    pub n_heads: usize,
    /// Feedforward intermediate dimension
    pub ff_dim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use layer norm
    pub use_layer_norm: bool,
}

impl Default for TetrahedralAgentConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            n_heads: 8,
            ff_dim: 2048,
            dropout: 0.1,
            use_layer_norm: true,
        }
    }
}

/// A single agent in the tetrahedral MOA
#[derive(Debug)]
pub struct TetrahedralAgent {
    /// Agent ID (0-63 for tetrahedron, 0-127 for star tetrahedron)
    pub id: usize,
    /// 3D position in tetrahedral space
    pub position: Point3D,
    /// Mapped position on torus
    pub torus_position: TorusCoordinate,
    /// Primary role
    pub role: TetrahedralAgentRole,
    /// Specialization domain
    pub domain: AgentDomain,
    /// Input projection
    input_proj: Linear,
    /// Output projection
    output_proj: Linear,
    /// Self-attention (simplified)
    query: Linear,
    key: Linear,
    value: Linear,
    /// Feedforward layers
    ff1: Linear,
    ff2: Linear,
    /// Configuration
    config: TetrahedralAgentConfig,
}

impl TetrahedralAgent {
    /// Create a new tetrahedral agent
    pub fn new(
        id: usize,
        position: Point3D,
        role: TetrahedralAgentRole,
        domain: AgentDomain,
        input_dim: usize,
        config: TetrahedralAgentConfig,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let hidden_dim = config.hidden_dim;
        let ff_dim = config.ff_dim;

        // Map 3D position to torus coordinates
        let torus_position = position.to_torus_coordinate(2.0, 1.0);

        Ok(Self {
            id,
            position,
            torus_position,
            role,
            domain,
            input_proj: candle_nn::linear(input_dim, hidden_dim, vb.pp("input_proj"))?,
            output_proj: candle_nn::linear(hidden_dim, input_dim, vb.pp("output_proj"))?,
            query: candle_nn::linear(hidden_dim, hidden_dim, vb.pp("query"))?,
            key: candle_nn::linear(hidden_dim, hidden_dim, vb.pp("key"))?,
            value: candle_nn::linear(hidden_dim, hidden_dim, vb.pp("value"))?,
            ff1: candle_nn::linear(hidden_dim, ff_dim, vb.pp("ff1"))?,
            ff2: candle_nn::linear(ff_dim, hidden_dim, vb.pp("ff2"))?,
            config,
        })
    }

    /// Process input through this agent
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Project to hidden dim
        let h = self.input_proj.forward(x)?;

        // Self-attention (simplified single-head for efficiency)
        let q = self.query.forward(&h)?;
        let k = self.key.forward(&h)?;
        let v = self.value.forward(&h)?;

        let scale = (self.config.hidden_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(0, 1)?)?;
        let scores = (scores / scale)?;
        let attn = candle_nn::ops::softmax(&scores, 1)?;
        let attended = attn.matmul(&v)?;

        // Residual + feedforward
        let h = (&h + attended)?;
        let ff = self.ff1.forward(&h)?;
        let ff = ff.relu()?;
        let ff = self.ff2.forward(&ff)?;
        let h = (&h + ff)?;

        // Project back to output dim
        Ok(self.output_proj.forward(&h)?)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TETRAHEDRAL MOA CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the Tetrahedral MOA system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrahedralMOAConfig {
    /// Number of agents (64 or 128)
    pub n_agents: usize,
    /// Use star tetrahedron (128 agents) vs single tetrahedron (64 agents)
    pub use_star_tetrahedron: bool,
    /// Input/output dimension
    pub d_model: usize,
    /// Per-agent configuration
    pub agent_config: TetrahedralAgentConfig,
    /// Edge length for tetrahedron geometry
    pub edge_length: f64,
    /// Major radius for torus mapping
    pub torus_major_radius: f64,
    /// Minor radius for torus mapping
    pub torus_minor_radius: f64,
    /// Number of torus streams to integrate with
    pub n_torus_streams: usize,
    /// Routing temperature (for soft routing)
    pub routing_temperature: f64,
    /// Top-k agents for sparse routing
    pub top_k_routing: usize,
}

impl Default for TetrahedralMOAConfig {
    fn default() -> Self {
        Self {
            n_agents: 64,
            use_star_tetrahedron: false,
            d_model: 512,
            agent_config: TetrahedralAgentConfig::default(),
            edge_length: 1.0,
            torus_major_radius: 2.0,
            torus_minor_radius: 1.0,
            n_torus_streams: 8,
            routing_temperature: 1.0,
            top_k_routing: 8,
        }
    }
}

impl TetrahedralMOAConfig {
    /// Create config for 64-agent tetrahedron
    pub fn tetrahedron_64() -> Self {
        Self {
            n_agents: 64,
            use_star_tetrahedron: false,
            ..Default::default()
        }
    }

    /// Create config for 128-agent star tetrahedron
    pub fn star_tetrahedron_128() -> Self {
        Self {
            n_agents: 128,
            use_star_tetrahedron: true,
            ..Default::default()
        }
    }

    /// Create config for large-scale (toward 1.7T params)
    pub fn large_scale(n_agents: usize, d_model: usize) -> Self {
        Self {
            n_agents,
            use_star_tetrahedron: n_agents > 64,
            d_model,
            agent_config: TetrahedralAgentConfig {
                hidden_dim: d_model,
                n_heads: 16,
                ff_dim: d_model * 4,
                dropout: 0.1,
                use_layer_norm: true,
            },
            top_k_routing: (n_agents / 8).max(4),
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TETRAHEDRAL ROUTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Routes inputs to appropriate agents based on geometric and learned criteria
#[derive(Debug)]
pub struct TetrahedralRouter {
    /// Routing projection
    router_proj: Linear,
    /// Agent positions for distance-based routing
    agent_positions: Vec<Point3D>,
    /// Configuration
    config: TetrahedralMOAConfig,
}

impl TetrahedralRouter {
    /// Create a new router
    pub fn new(
        agent_positions: Vec<Point3D>,
        config: TetrahedralMOAConfig,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let n_agents = agent_positions.len();
        Ok(Self {
            router_proj: candle_nn::linear(config.d_model, n_agents, vb.pp("router"))?,
            agent_positions,
            config,
        })
    }

    /// Compute routing weights for input
    /// Returns (routing_weights, selected_indices)
    pub fn route(&self, x: &Tensor) -> TorusResult<(Tensor, Vec<usize>)> {
        // Learned routing scores
        let scores = self.router_proj.forward(x)?;

        // Apply temperature and softmax
        let scores = (scores / self.config.routing_temperature)?;
        let weights = candle_nn::ops::softmax(&scores, scores.dims().len() - 1)?;

        // Get top-k indices (simplified - take highest weights)
        let weights_vec: Vec<f32> = weights.flatten_all()?.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = weights_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected: Vec<usize> = indexed
            .iter()
            .take(self.config.top_k_routing)
            .map(|(i, _)| *i)
            .collect();

        Ok((weights, selected))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN TETRAHEDRAL MOA SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Tetrahedral Mixture of Agents - the main MOA system
pub struct TetrahedralMOA {
    /// All agents
    agents: Vec<TetrahedralAgent>,
    /// Router for input-to-agent mapping
    router: TetrahedralRouter,
    /// Output combination layer
    output_combine: Linear,
    /// Geometric structure (single or star tetrahedron)
    geometry: TetrahedralGeometry,
    /// Adjacency matrix for agent communication
    adjacency: Vec<Vec<f64>>,
    /// Configuration
    config: TetrahedralMOAConfig,
    /// Device
    device: Device,
}

impl std::fmt::Debug for TetrahedralMOA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TetrahedralMOA")
            .field("n_agents", &self.agents.len())
            .field("geometry", &self.geometry)
            .field("config", &self.config)
            .finish()
    }
}

/// Wrapper for either single or star tetrahedron
#[derive(Debug, Clone)]
pub enum TetrahedralGeometry {
    Single(Tetrahedron),
    Star(StarTetrahedron),
}

impl TetrahedralMOA {
    /// Create a new Tetrahedral MOA system
    pub fn new(config: TetrahedralMOAConfig, vb: VarBuilder, device: &Device) -> TorusResult<Self> {
        let (geometry, points) = if config.use_star_tetrahedron {
            let star = StarTetrahedron::new(config.edge_length);
            let points = star.distribute_points(16); // 16 per vertex * 4 vertices * 2 = 128
            let positions: Vec<Point3D> = points.iter().map(|p| p.position).collect();
            (TetrahedralGeometry::Star(star), positions)
        } else {
            let tetra = Tetrahedron::regular(config.edge_length);
            let points = tetra.distribute_points(16); // 16 per vertex * 4 = 64
            let positions: Vec<Point3D> = points.iter().map(|p| p.position).collect();
            (TetrahedralGeometry::Single(tetra), positions)
        };

        // Create agents
        let mut agents = Vec::with_capacity(config.n_agents);
        for (i, pos) in points.iter().enumerate() {
            let vertex = TetrahedronVertex::all()[i / 16 % 4];
            let role = TetrahedralAgentRole::Vertex(vertex);
            let domain = AgentDomain::from_vertex(vertex);

            let agent = TetrahedralAgent::new(
                i,
                *pos,
                role,
                domain,
                config.d_model,
                config.agent_config.clone(),
                vb.pp(format!("agent_{}", i)),
            )?;
            agents.push(agent);
        }

        // Create router
        let router = TetrahedralRouter::new(points.clone(), config.clone(), vb.pp("router"))?;

        // Output combination
        let output_combine = candle_nn::linear(
            config.d_model * config.top_k_routing,
            config.d_model,
            vb.pp("output_combine"),
        )?;

        // Build adjacency matrix based on distances
        let adjacency = Self::build_adjacency(&points, config.edge_length);

        Ok(Self {
            agents,
            router,
            output_combine,
            geometry,
            adjacency,
            config,
            device: device.clone(),
        })
    }

    /// Build adjacency matrix from agent positions
    fn build_adjacency(positions: &[Point3D], edge_length: f64) -> Vec<Vec<f64>> {
        let n = positions.len();
        let mut adj = vec![vec![0.0; n]; n];
        let threshold = edge_length * 1.5; // Connect if within 1.5x edge length

        for i in 0..n {
            for j in i + 1..n {
                let dist = positions[i].distance(&positions[j]);
                if dist < threshold {
                    let weight = 1.0 - (dist / threshold);
                    adj[i][j] = weight;
                    adj[j][i] = weight;
                }
            }
        }
        adj
    }

    /// Forward pass through the MOA
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Route input to top-k agents
        let (weights, selected) = self.router.route(x)?;

        // Process through selected agents
        let mut outputs = Vec::with_capacity(selected.len());
        for &agent_idx in &selected {
            if agent_idx < self.agents.len() {
                let agent_out = self.agents[agent_idx].forward(x)?;
                outputs.push(agent_out);
            }
        }

        // Combine outputs
        if outputs.is_empty() {
            return Ok(x.clone());
        }

        let combined = Tensor::cat(&outputs, 1)?;
        Ok(self.output_combine.forward(&combined)?)
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: usize) -> Option<&TetrahedralAgent> {
        self.agents.get(id)
    }

    /// Get number of agents
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get configuration
    pub fn config(&self) -> &TetrahedralMOAConfig {
        &self.config
    }

    /// Get adjacency matrix
    pub fn adjacency(&self) -> &Vec<Vec<f64>> {
        &self.adjacency
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-TORUS SCALING (TOWARD 1.7T)
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for multi-torus architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTorusConfig {
    /// Number of tori (typically 3 for T³)
    pub n_tori: usize,
    /// Parameters per torus (in billions)
    pub params_per_torus_billions: f64,
    /// MOA agents for inter-torus routing
    pub moa_agents: usize,
    /// Hidden dimension
    pub d_model: usize,
    /// Layers per torus
    pub layers_per_torus: usize,
}

impl MultiTorusConfig {
    /// Create config targeting ~1.7T parameters
    pub fn for_1_7t() -> Self {
        Self {
            n_tori: 3,
            params_per_torus_billions: 566.0, // ~566B * 3 = 1.7T
            moa_agents: 128,
            d_model: 16384,
            layers_per_torus: 96,
        }
    }

    /// Estimate total parameters
    pub fn estimate_params(&self) -> f64 {
        self.n_tori as f64 * self.params_per_torus_billions * 1e9
    }
}

/// Summary of tetrahedral MOA state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrahedralMOASummary {
    pub n_agents: usize,
    pub geometry_type: String,
    pub d_model: usize,
    pub top_k_routing: usize,
    pub agent_domains: HashMap<String, usize>,
}

impl TetrahedralMOA {
    /// Get summary of MOA state
    pub fn summary(&self) -> TetrahedralMOASummary {
        let mut agent_domains = HashMap::new();
        for agent in &self.agents {
            let domain_name = format!("{:?}", agent.domain);
            *agent_domains.entry(domain_name).or_insert(0) += 1;
        }

        TetrahedralMOASummary {
            n_agents: self.agents.len(),
            geometry_type: match &self.geometry {
                TetrahedralGeometry::Single(_) => "Tetrahedron (64)".to_string(),
                TetrahedralGeometry::Star(_) => "Star Tetrahedron (128)".to_string(),
            },
            d_model: self.config.d_model,
            top_k_routing: self.config.top_k_routing,
            agent_domains,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OCTOPUS LIMB ARCHITECTURE (Aligned with octotetrahedral-agi)
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of processing limb (inspired by octopus distributed cognition)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LimbType {
    /// Input encoding and feature extraction
    Perception,
    /// Pattern processing and inference
    Reasoning,
    /// Output generation and response formation
    Action,
    /// Memory access and retrieval
    Memory,
    /// Planning and goal decomposition
    Planning,
    /// Language and symbolic processing
    Language,
    /// Spatial and visual processing
    Spatial,
    /// Meta-cognitive monitoring
    MetaCognition,
}

impl LimbType {
    /// Get all limb types
    pub fn all() -> [LimbType; 8] {
        [
            LimbType::Perception,
            LimbType::Reasoning,
            LimbType::Action,
            LimbType::Memory,
            LimbType::Planning,
            LimbType::Language,
            LimbType::Spatial,
            LimbType::MetaCognition,
        ]
    }

    /// Map from tetrahedral vertex to primary limb type
    pub fn from_vertex(vertex: TetrahedronVertex, orientation: TetrahedronOrientation) -> Self {
        match (vertex, orientation) {
            (TetrahedronVertex::V0, TetrahedronOrientation::Up) => LimbType::Reasoning,
            (TetrahedronVertex::V1, TetrahedronOrientation::Up) => LimbType::Language,
            (TetrahedronVertex::V2, TetrahedronOrientation::Up) => LimbType::Spatial,
            (TetrahedronVertex::V3, TetrahedronOrientation::Up) => LimbType::Memory,
            (TetrahedronVertex::V0, TetrahedronOrientation::Down) => LimbType::MetaCognition,
            (TetrahedronVertex::V1, TetrahedronOrientation::Down) => LimbType::Perception,
            (TetrahedronVertex::V2, TetrahedronOrientation::Down) => LimbType::Action,
            (TetrahedronVertex::V3, TetrahedronOrientation::Down) => LimbType::Planning,
        }
    }
}

/// Configuration for an octopus limb
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctopusLimbConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension for local processing
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Whether to use LoRA adaptation (for perception limb)
    pub use_lora: bool,
    /// LoRA rank (if enabled)
    pub lora_rank: usize,
    /// Local buffer size for autonomous processing
    pub buffer_size: usize,
}

impl Default for OctopusLimbConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            hidden_dim: 1024,
            output_dim: 512,
            use_lora: false,
            lora_rank: 16,
            buffer_size: 64,
        }
    }
}

/// Octopus-inspired processing limb with local autonomy
/// Each limb can process independently before synchronizing with the hub
#[derive(Debug)]
pub struct OctopusLimb {
    /// Limb type/specialization
    pub limb_type: LimbType,
    /// Encoder: input -> hidden
    encoder: Linear,
    /// Processor: hidden -> hidden (local processing)
    processor: Linear,
    /// Decoder: hidden -> output
    decoder: Linear,
    /// LoRA adaptation matrices (if enabled)
    lora_a: Option<Linear>,
    lora_b: Option<Linear>,
    /// Local processing buffer (for autonomous computation)
    buffer_capacity: usize,
    /// Configuration
    config: OctopusLimbConfig,
}

impl OctopusLimb {
    /// Create a new octopus limb
    pub fn new(
        limb_type: LimbType,
        config: OctopusLimbConfig,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let encoder = candle_nn::linear(config.input_dim, config.hidden_dim, vb.pp("encoder"))?;
        let processor =
            candle_nn::linear(config.hidden_dim, config.hidden_dim, vb.pp("processor"))?;
        let decoder = candle_nn::linear(config.hidden_dim, config.output_dim, vb.pp("decoder"))?;

        let (lora_a, lora_b) = if config.use_lora {
            (
                Some(candle_nn::linear(
                    config.input_dim,
                    config.lora_rank,
                    vb.pp("lora_a"),
                )?),
                Some(candle_nn::linear(
                    config.lora_rank,
                    config.hidden_dim,
                    vb.pp("lora_b"),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            limb_type,
            encoder,
            processor,
            decoder,
            lora_a,
            lora_b,
            buffer_capacity: config.buffer_size,
            config,
        })
    }

    /// Process input through limb (local autonomous processing)
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Encode
        let mut h = self.encoder.forward(x)?;

        // Apply LoRA if available (low-rank adaptation)
        if let (Some(lora_a), Some(lora_b)) = (&self.lora_a, &self.lora_b) {
            let lora_out = lora_b.forward(&lora_a.forward(x)?)?;
            h = (&h + lora_out)?;
        }

        // Local processing (can be done autonomously)
        let h = self.processor.forward(&h)?;
        let h = h.relu()?;

        // Decode to output
        Ok(self.decoder.forward(&h)?)
    }

    /// Get limb type
    pub fn limb_type(&self) -> LimbType {
        self.limb_type
    }
}

/// Central hub that synchronizes all octopus limbs
/// Inspired by octopus central brain coordinating autonomous arms
#[derive(Debug)]
pub struct OctopusHub {
    /// All limbs
    limbs: Vec<OctopusLimb>,
    /// Synchronization weights (learned, performance-weighted)
    sync_proj: Linear,
    /// Output projection
    output_proj: Linear,
    /// Limb importance weights (adaptive)
    limb_weights: Vec<f64>,
    /// Hidden dimension
    d_model: usize,
}

impl OctopusHub {
    /// Create a new octopus hub with 8 limbs
    pub fn new(d_model: usize, vb: VarBuilder) -> TorusResult<Self> {
        let limb_config = OctopusLimbConfig {
            input_dim: d_model,
            hidden_dim: d_model * 2,
            output_dim: d_model,
            ..Default::default()
        };

        let mut limbs = Vec::new();
        for limb_type in LimbType::all() {
            let limb = OctopusLimb::new(
                limb_type,
                limb_config.clone(),
                vb.pp(format!("limb_{:?}", limb_type).to_lowercase()),
            )?;
            limbs.push(limb);
        }

        let n_limbs = limbs.len();
        let sync_proj = candle_nn::linear(d_model * n_limbs, d_model, vb.pp("sync"))?;
        let output_proj = candle_nn::linear(d_model, d_model, vb.pp("output"))?;

        Ok(Self {
            limbs,
            sync_proj,
            output_proj,
            limb_weights: vec![1.0 / 8.0; 8],
            d_model,
        })
    }

    /// Process input through all limbs and synchronize
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Process through each limb independently
        let mut limb_outputs = Vec::new();
        for limb in &self.limbs {
            let out = limb.forward(x)?;
            limb_outputs.push(out);
        }

        // Concatenate all limb outputs
        let combined = Tensor::cat(&limb_outputs, limb_outputs[0].dims().len() - 1)?;

        // Synchronize through central projection
        let synced = self.sync_proj.forward(&combined)?;

        // Final output projection
        Ok(self.output_proj.forward(&synced)?)
    }

    /// Update limb weights based on performance feedback
    pub fn update_limb_weights(&mut self, performance_scores: &[f64]) {
        if performance_scores.len() == self.limb_weights.len() {
            let sum: f64 = performance_scores.iter().sum();
            if sum > 0.0 {
                for (i, score) in performance_scores.iter().enumerate() {
                    self.limb_weights[i] = score / sum;
                }
            }
        }
    }

    /// Get number of limbs
    pub fn n_limbs(&self) -> usize {
        self.limbs.len()
    }

    /// Get limb by type
    pub fn get_limb(&self, limb_type: LimbType) -> Option<&OctopusLimb> {
        self.limbs.iter().find(|l| l.limb_type == limb_type)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RNA EDITING LAYER (Dynamic weight modulation)
// ═══════════════════════════════════════════════════════════════════════════════

/// RNA editing-inspired dynamic weight modulation
/// Octopuses edit RNA to adapt proteins on-the-fly; this layer adapts weights dynamically
#[derive(Debug)]
pub struct RNAEditingLayer {
    /// Multiple processing pathways (like RNA variants)
    pathways: Vec<Linear>,
    /// Gating mechanism for selecting/blending pathways
    pathway_gate: Linear,
    /// Attention head gates (modulate individual attention heads)
    head_gate: Linear,
    /// Temperature for soft selection
    temperature: f64,
    /// Number of pathways
    n_pathways: usize,
    /// Number of attention heads to modulate
    n_heads: usize,
}

impl RNAEditingLayer {
    /// Create a new RNA editing layer
    pub fn new(
        d_model: usize,
        n_pathways: usize,
        n_heads: usize,
        temperature: f64,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let mut pathways = Vec::new();
        for i in 0..n_pathways {
            pathways.push(candle_nn::linear(
                d_model,
                d_model,
                vb.pp(format!("pathway_{}", i)),
            )?);
        }

        let pathway_gate = candle_nn::linear(d_model, n_pathways, vb.pp("pathway_gate"))?;
        let head_gate = candle_nn::linear(d_model, n_heads, vb.pp("head_gate"))?;

        Ok(Self {
            pathways,
            pathway_gate,
            head_gate,
            temperature,
            n_pathways,
            n_heads,
        })
    }

    /// Forward pass with dynamic pathway selection
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Compute pathway gates (which pathways to use)
        let gate_logits = self.pathway_gate.forward(x)?;
        let gate_logits = (gate_logits / self.temperature)?;
        let pathway_weights = candle_nn::ops::softmax(&gate_logits, gate_logits.dims().len() - 1)?;

        // Process through each pathway
        let mut pathway_outputs = Vec::new();
        for pathway in &self.pathways {
            pathway_outputs.push(pathway.forward(x)?);
        }

        // Weighted combination of pathways
        let pathway_weights_vec: Vec<f32> = pathway_weights.flatten_all()?.to_vec1()?;
        let mut combined = pathway_outputs[0].zeros_like()?;
        for (i, output) in pathway_outputs.iter().enumerate() {
            let weight = pathway_weights_vec
                .get(i % pathway_weights_vec.len())
                .unwrap_or(&0.0);
            combined = (&combined + (output * (*weight as f64))?)?;
        }

        Ok(combined)
    }

    /// Get head modulation gates for external attention layers
    pub fn get_head_gates(&self, x: &Tensor) -> TorusResult<Tensor> {
        let gate_logits = self.head_gate.forward(x)?;
        Ok(candle_nn::ops::sigmoid(&gate_logits)?)
    }

    /// Adapt temperature based on task difficulty
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1).min(10.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKING MEMORY (4-Slot Differentiable Memory)
// ═══════════════════════════════════════════════════════════════════════════════

/// Semantic slot types for working memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemorySlotType {
    /// Current goal or task objective
    Goal,
    /// Environmental/situational context
    Context,
    /// Intermediate computation results
    Intermediate,
    /// Prepared output/response
    Output,
}

impl MemorySlotType {
    pub fn all() -> [MemorySlotType; 4] {
        [
            MemorySlotType::Goal,
            MemorySlotType::Context,
            MemorySlotType::Intermediate,
            MemorySlotType::Output,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// NTM-style differentiable working memory with 4 semantic slots
/// Based on Neural Turing Machine memory operations
#[derive(Debug)]
pub struct TetrahedralWorkingMemory {
    /// Memory slots [Goal, Context, Intermediate, Output]
    slot_dim: usize,
    /// Read head: computes attention over slots
    read_key: Linear,
    read_strength: Linear,
    /// Write head: determines what to write
    write_key: Linear,
    write_strength: Linear,
    write_content: Linear,
    /// Erase head: determines what to erase before writing
    erase_head: Linear,
    /// Slot embeddings (learnable slot identities)
    slot_embeddings: Linear,
    /// Current memory state (initialized to zeros, updated during forward)
    /// Note: In production, this would be passed as state
    n_slots: usize,
}

impl TetrahedralWorkingMemory {
    /// Create new working memory with 4 slots
    pub fn new(d_model: usize, vb: VarBuilder) -> TorusResult<Self> {
        let n_slots = 4;

        Ok(Self {
            slot_dim: d_model,
            read_key: candle_nn::linear(d_model, d_model, vb.pp("read_key"))?,
            read_strength: candle_nn::linear(d_model, 1, vb.pp("read_strength"))?,
            write_key: candle_nn::linear(d_model, d_model, vb.pp("write_key"))?,
            write_strength: candle_nn::linear(d_model, 1, vb.pp("write_strength"))?,
            write_content: candle_nn::linear(d_model, d_model, vb.pp("write_content"))?,
            erase_head: candle_nn::linear(d_model, d_model, vb.pp("erase_head"))?,
            slot_embeddings: candle_nn::linear(n_slots, d_model, vb.pp("slot_embed"))?,
            n_slots,
        })
    }

    /// Initialize memory state
    pub fn init_memory(&self, device: &Device) -> TorusResult<Tensor> {
        Ok(Tensor::zeros(
            (self.n_slots, self.slot_dim),
            DType::F32,
            device,
        )?)
    }

    /// Read from memory using content-based addressing
    pub fn read(&self, query: &Tensor, memory: &Tensor) -> TorusResult<Tensor> {
        // Compute read key
        let key = self.read_key.forward(query)?;
        let strength = self.read_strength.forward(query)?;
        let strength = strength.squeeze(strength.dims().len() - 1)?;

        // Content-based addressing: similarity between key and memory slots
        // memory: [n_slots, d_model], key: [batch, d_model] or [d_model]
        let key_expanded = if key.dims().len() == 1 {
            key.unsqueeze(0)?
        } else {
            key.clone()
        };

        // Compute cosine similarity
        let key_norm = (&key_expanded * &key_expanded)?.sum(1)?.sqrt()?;
        let mem_norm = (memory * memory)?.sum(1)?.sqrt()?;

        let similarity = key_expanded.matmul(&memory.transpose(0, 1)?)?;
        let key_norm_expanded = key_norm.unsqueeze(1)?;
        let mem_norm_expanded = mem_norm.unsqueeze(0)?;
        let norm_product = key_norm_expanded.broadcast_mul(&mem_norm_expanded)?;
        let similarity = (similarity / (norm_product + 1e-8)?)?;

        // Apply strength and softmax for attention weights
        let strength_expanded = if strength.dims().len() == 0 {
            strength.unsqueeze(0)?.unsqueeze(1)?
        } else {
            strength.unsqueeze(strength.dims().len())?
        };
        let weighted_sim = (similarity * strength_expanded)?;
        let attention = candle_nn::ops::softmax(&weighted_sim, weighted_sim.dims().len() - 1)?;

        // Read from memory
        Ok(attention.matmul(memory)?)
    }

    /// Write to memory with erase-then-write operation
    pub fn write(
        &self,
        input: &Tensor,
        memory: &Tensor,
        slot_type: MemorySlotType,
    ) -> TorusResult<Tensor> {
        let slot_idx = slot_type.index();

        // Compute write content
        let content = self.write_content.forward(input)?;

        // Compute erase vector (what to forget)
        let erase = self.erase_head.forward(input)?;
        let erase = candle_nn::ops::sigmoid(&erase)?;

        // Get the slot to update
        let slot = memory.i(slot_idx)?;

        // Erase then write: new_slot = slot * (1 - erase) + content
        let erased = (&slot * (1.0 - &erase)?)?;
        let new_slot = (&erased + &content)?;

        // Update memory (create new tensor with updated slot)
        let mut slots = Vec::new();
        for i in 0..self.n_slots {
            if i == slot_idx {
                slots.push(new_slot.clone());
            } else {
                slots.push(memory.i(i)?);
            }
        }

        Ok(Tensor::stack(&slots, 0)?)
    }

    /// Read from specific slot type
    pub fn read_slot(&self, memory: &Tensor, slot_type: MemorySlotType) -> TorusResult<Tensor> {
        Ok(memory.i(slot_type.index())?)
    }

    /// Get number of slots
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// Get slot dimension
    pub fn slot_dim(&self) -> usize {
        self.slot_dim
    }
}

/// Working memory summary for debugging/logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemorySummary {
    pub n_slots: usize,
    pub slot_dim: usize,
    pub slot_norms: Vec<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATED OCTOTETRAHEDRAL SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrated system combining Tetrahedral MOA, Octopus Limbs, RNA Editing, and Working Memory
/// This is the full OctoTetrahedral AGI architecture aligned with Python repos
#[derive(Debug)]
pub struct OctoTetrahedralSystem {
    /// Tetrahedral MOA (64/128 agents)
    pub moa: TetrahedralMOA,
    /// Octopus hub with 8 limbs
    pub octopus_hub: OctopusHub,
    /// RNA editing layer for dynamic adaptation
    pub rna_editing: RNAEditingLayer,
    /// Working memory (4 semantic slots)
    pub working_memory: TetrahedralWorkingMemory,
    /// Integration projection
    integration_proj: Linear,
    /// Configuration
    pub config: OctoTetrahedralConfig,
}

/// Configuration for the integrated system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctoTetrahedralConfig {
    /// MOA configuration
    pub moa_config: TetrahedralMOAConfig,
    /// Model dimension
    pub d_model: usize,
    /// Number of RNA editing pathways
    pub n_rna_pathways: usize,
    /// Number of attention heads for RNA head gating
    pub n_heads: usize,
    /// RNA editing temperature
    pub rna_temperature: f64,
}

impl Default for OctoTetrahedralConfig {
    fn default() -> Self {
        Self {
            moa_config: TetrahedralMOAConfig::star_tetrahedron_128(),
            d_model: 512,
            n_rna_pathways: 4,
            n_heads: 8,
            rna_temperature: 1.0,
        }
    }
}

impl OctoTetrahedralSystem {
    /// Create a new OctoTetrahedral system
    pub fn new(
        config: OctoTetrahedralConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let moa = TetrahedralMOA::new(config.moa_config.clone(), vb.pp("moa"), device)?;
        let octopus_hub = OctopusHub::new(config.d_model, vb.pp("octopus"))?;
        let rna_editing = RNAEditingLayer::new(
            config.d_model,
            config.n_rna_pathways,
            config.n_heads,
            config.rna_temperature,
            vb.pp("rna"),
        )?;
        let working_memory = TetrahedralWorkingMemory::new(config.d_model, vb.pp("wm"))?;
        let integration_proj = candle_nn::linear(
            config.d_model * 3, // MOA + Octopus + RNA outputs
            config.d_model,
            vb.pp("integrate"),
        )?;

        Ok(Self {
            moa,
            octopus_hub,
            rna_editing,
            working_memory,
            integration_proj,
            config,
        })
    }

    /// Forward pass through the integrated system
    pub fn forward(&self, x: &Tensor, memory: &Tensor) -> TorusResult<(Tensor, Tensor)> {
        // 1. Read context from working memory
        let context = self.working_memory.read(x, memory)?;

        // 2. Combine input with context
        let contextualized = (x + &context)?;

        // 3. Process through MOA (tetrahedral routing)
        let moa_out = self.moa.forward(&contextualized)?;

        // 4. Process through octopus hub (parallel limbs)
        let octopus_out = self.octopus_hub.forward(&contextualized)?;

        // 5. Apply RNA editing (dynamic adaptation)
        let rna_out = self.rna_editing.forward(&contextualized)?;

        // 6. Integrate all pathways
        let combined = Tensor::cat(&[moa_out, octopus_out, rna_out], x.dims().len() - 1)?;
        let output = self.integration_proj.forward(&combined)?;

        // 7. Update working memory with intermediate results
        let new_memory =
            self.working_memory
                .write(&output, memory, MemorySlotType::Intermediate)?;

        Ok((output, new_memory))
    }

    /// Initialize memory for this system
    pub fn init_memory(&self, device: &Device) -> TorusResult<Tensor> {
        self.working_memory.init_memory(device)
    }

    /// Get summary of system state
    pub fn summary(&self) -> OctoTetrahedralSummary {
        OctoTetrahedralSummary {
            moa_summary: self.moa.summary(),
            n_limbs: self.octopus_hub.n_limbs(),
            n_rna_pathways: self.config.n_rna_pathways,
            n_memory_slots: self.working_memory.n_slots(),
            d_model: self.config.d_model,
        }
    }
}

/// Summary of the OctoTetrahedral system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctoTetrahedralSummary {
    pub moa_summary: TetrahedralMOASummary,
    pub n_limbs: usize,
    pub n_rna_pathways: usize,
    pub n_memory_slots: usize,
    pub d_model: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TORUS-TETRAHEDRON INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrates tetrahedral MOA with torus attention streams
#[derive(Debug)]
pub struct TorusTetrahedralBridge {
    /// Mapping from torus streams to tetrahedral agents
    stream_to_agents: HashMap<StreamId, Vec<usize>>,
    /// Mapping from agents to torus coordinates
    agent_torus_coords: Vec<TorusCoordinate>,
    /// Projection for stream-agent communication
    stream_proj: Linear,
}

impl TorusTetrahedralBridge {
    /// Create a bridge between torus streams and tetrahedral agents
    pub fn new(moa: &TetrahedralMOA, d_model: usize, vb: VarBuilder) -> TorusResult<Self> {
        // Map each stream to a subset of agents based on torus position
        let mut stream_to_agents: HashMap<StreamId, Vec<usize>> = HashMap::new();

        for stream in StreamId::all() {
            stream_to_agents.insert(stream, Vec::new());
        }

        // Distribute agents to streams based on their torus coordinates
        for (i, agent) in moa.agents.iter().enumerate() {
            let coord = &agent.torus_position;

            // Assign to streams based on coordinate ranges
            if coord.u < PI {
                stream_to_agents
                    .get_mut(&StreamId::MajorForward)
                    .unwrap()
                    .push(i);
            } else {
                stream_to_agents
                    .get_mut(&StreamId::MajorBackward)
                    .unwrap()
                    .push(i);
            }

            if coord.v < PI {
                stream_to_agents
                    .get_mut(&StreamId::MinorForward)
                    .unwrap()
                    .push(i);
            } else {
                stream_to_agents
                    .get_mut(&StreamId::MinorBackward)
                    .unwrap()
                    .push(i);
            }

            // Spiral assignment based on winding
            let spiral_phase = coord.u + coord.v;
            if (spiral_phase / PI) as i32 % 2 == 0 {
                stream_to_agents
                    .get_mut(&StreamId::SpiralCW)
                    .unwrap()
                    .push(i);
            } else {
                stream_to_agents
                    .get_mut(&StreamId::SpiralCCW)
                    .unwrap()
                    .push(i);
            }

            // Cross streams based on coordinate dominance
            if coord.u > coord.v {
                stream_to_agents
                    .get_mut(&StreamId::CrossUtoV)
                    .unwrap()
                    .push(i);
            } else {
                stream_to_agents
                    .get_mut(&StreamId::CrossVtoU)
                    .unwrap()
                    .push(i);
            }
        }

        let agent_torus_coords: Vec<TorusCoordinate> =
            moa.agents.iter().map(|a| a.torus_position).collect();

        let stream_proj = candle_nn::linear(d_model, d_model, vb.pp("stream_proj"))?;

        Ok(Self {
            stream_to_agents,
            agent_torus_coords,
            stream_proj,
        })
    }

    /// Get agents assigned to a stream
    pub fn agents_for_stream(&self, stream: StreamId) -> &[usize] {
        self.stream_to_agents
            .get(&stream)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get torus coordinate for an agent
    pub fn agent_torus_coord(&self, agent_id: usize) -> Option<&TorusCoordinate> {
        self.agent_torus_coords.get(agent_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedron_geometry() {
        let tetra = Tetrahedron::unit();

        // Check edge length between adjacent vertices
        let v0 = tetra.vertex(TetrahedronVertex::V0);
        let v1 = tetra.vertex(TetrahedronVertex::V1);
        let dist = v0.distance(&v1);
        assert!(
            (dist - 1.0).abs() < 0.1,
            "Edge length should be ~1.0, got {}",
            dist
        );

        // Check all vertices exist
        assert_eq!(tetra.vertices.len(), 4);
    }

    #[test]
    fn test_star_tetrahedron() {
        let star = StarTetrahedron::new(1.0);

        // Check both tetrahedra exist
        assert_eq!(star.upper.vertices.len(), 4);
        assert_eq!(star.lower.vertices.len(), 4);

        // Lower should be inverted from upper
        let upper_v0 = star.upper.vertex(TetrahedronVertex::V0);
        let lower_v0 = star.lower.vertex(TetrahedronVertex::V0);
        assert!(
            (upper_v0.z + lower_v0.z).abs() < 0.01,
            "Lower should be inverted"
        );
    }

    #[test]
    fn test_64_point_distribution() {
        let tetra = Tetrahedron::regular(1.0);
        let points = tetra.distribute_points(16);

        assert_eq!(points.len(), 64, "Should have 64 points");

        // Check each vertex has 16 points
        for v in TetrahedronVertex::all() {
            let count = points.iter().filter(|p| p.primary_vertex == v).count();
            assert_eq!(count, 16, "Each vertex should have 16 points");
        }
    }

    #[test]
    fn test_128_point_distribution() {
        let star = StarTetrahedron::new(1.0);
        let points = star.distribute_points(16);

        assert_eq!(points.len(), 128, "Should have 128 points");

        // Check orientations
        let up_count = points
            .iter()
            .filter(|p| p.orientation == TetrahedronOrientation::Up)
            .count();
        let down_count = points
            .iter()
            .filter(|p| p.orientation == TetrahedronOrientation::Down)
            .count();
        assert_eq!(up_count, 64);
        assert_eq!(down_count, 64);
    }

    #[test]
    fn test_point_to_torus_mapping() {
        let point = Point3D::new(2.0, 0.0, 0.0);
        let coord = point.to_torus_coordinate(2.0, 1.0);

        // Point at (2, 0, 0) should map to u=0, v=0
        assert!(coord.u.abs() < 0.1 || (coord.u - 2.0 * PI).abs() < 0.1);
    }

    #[test]
    fn test_adjacency_matrix() {
        let positions = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(0.5, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0), // Far away
        ];

        let adj = TetrahedralMOA::build_adjacency(&positions, 1.0);

        // Close points should be connected
        assert!(adj[0][1] > 0.0);
        assert!(adj[1][0] > 0.0);

        // Far point should not be connected
        assert_eq!(adj[0][2], 0.0);
        assert_eq!(adj[2][0], 0.0);
    }

    #[test]
    fn test_multi_torus_config() {
        let config = MultiTorusConfig::for_1_7t();

        assert_eq!(config.n_tori, 3);
        assert_eq!(config.moa_agents, 128);

        let total_params = config.estimate_params();
        assert!(total_params > 1.5e12); // > 1.5T
        assert!(total_params < 2.0e12); // < 2T
    }

    #[test]
    fn test_agent_domain_mapping() {
        assert_eq!(
            AgentDomain::from_vertex(TetrahedronVertex::V0),
            AgentDomain::Reasoning
        );
        assert_eq!(
            AgentDomain::from_vertex(TetrahedronVertex::V1),
            AgentDomain::Language
        );
        assert_eq!(
            AgentDomain::from_vertex(TetrahedronVertex::V2),
            AgentDomain::Vision
        );
        assert_eq!(
            AgentDomain::from_vertex(TetrahedronVertex::V3),
            AgentDomain::Memory
        );
    }

    #[test]
    fn test_limb_types() {
        let limbs = LimbType::all();
        assert_eq!(limbs.len(), 8);

        // Test vertex-to-limb mapping for star tetrahedron
        assert_eq!(
            LimbType::from_vertex(TetrahedronVertex::V0, TetrahedronOrientation::Up),
            LimbType::Reasoning
        );
        assert_eq!(
            LimbType::from_vertex(TetrahedronVertex::V0, TetrahedronOrientation::Down),
            LimbType::MetaCognition
        );
    }

    #[test]
    fn test_memory_slot_types() {
        let slots = MemorySlotType::all();
        assert_eq!(slots.len(), 4);
        assert_eq!(MemorySlotType::Goal.index(), 0);
        assert_eq!(MemorySlotType::Context.index(), 1);
        assert_eq!(MemorySlotType::Intermediate.index(), 2);
        assert_eq!(MemorySlotType::Output.index(), 3);
    }

    #[test]
    fn test_octotetrahedral_config() {
        let config = OctoTetrahedralConfig::default();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_rna_pathways, 4);
        assert_eq!(config.n_heads, 8);
        assert!(config.moa_config.use_star_tetrahedron); // 128 agents
    }
}
