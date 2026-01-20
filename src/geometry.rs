//! Torus geometry and coordinate system
//!
//! A torus is defined by two radii:
//! - R (major radius): distance from center of tube to center of torus
//! - r (minor radius): radius of the tube itself
//!
//! Parametric equations:
//! x = (R + r*cos(v)) * cos(u)
//! y = (R + r*cos(v)) * sin(u)
//! z = r * sin(v)
//!
//! where u, v ∈ [0, 2π)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Represents a point on a torus manifold using angular coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TorusCoordinate {
    /// Major angle (around the main ring) in [0, 2π)
    pub u: f64,
    /// Minor angle (around the tube) in [0, 2π)
    pub v: f64,
}

impl TorusCoordinate {
    /// Create a new torus coordinate with automatic wrapping
    pub fn new(u: f64, v: f64) -> Self {
        Self {
            u: Self::wrap_angle(u),
            v: Self::wrap_angle(v),
        }
    }

    /// Wrap angle to [0, 2π)
    #[inline]
    pub fn wrap_angle(angle: f64) -> f64 {
        let two_pi = 2.0 * PI;
        let wrapped = angle % two_pi;
        if wrapped < 0.0 {
            wrapped + two_pi
        } else {
            wrapped
        }
    }

    /// Convert angular coordinates to Cartesian (x, y, z)
    pub fn to_cartesian(&self, major_radius: f64, minor_radius: f64) -> (f64, f64, f64) {
        let x = (major_radius + minor_radius * self.v.cos()) * self.u.cos();
        let y = (major_radius + minor_radius * self.v.cos()) * self.u.sin();
        let z = minor_radius * self.v.sin();
        (x, y, z)
    }

    /// Calculate geodesic distance to another point on the torus
    /// Uses the flat torus metric (approximation for attention)
    pub fn geodesic_distance(&self, other: &TorusCoordinate) -> f64 {
        let du = Self::angular_distance(self.u, other.u);
        let dv = Self::angular_distance(self.v, other.v);
        (du * du + dv * dv).sqrt()
    }

    /// Calculate the shortest angular distance considering periodicity
    #[inline]
    pub fn angular_distance(a: f64, b: f64) -> f64 {
        let diff = (a - b).abs();
        let two_pi = 2.0 * PI;
        if diff > PI {
            two_pi - diff
        } else {
            diff
        }
    }

    /// Get the spiral position for vortex dynamics
    /// Combines u and v to create a spiral index
    pub fn spiral_position(&self, winding_number: f64) -> f64 {
        self.u + winding_number * self.v
    }
}

/// Torus manifold with specified radii
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorusManifold {
    /// Major radius (R)
    pub major_radius: f64,
    /// Minor radius (r)
    pub minor_radius: f64,
    /// Aspect ratio R/r
    pub aspect_ratio: f64,
}

impl TorusManifold {
    /// Create a new torus manifold
    pub fn new(major_radius: f64, minor_radius: f64) -> Self {
        assert!(major_radius > 0.0, "Major radius must be positive");
        assert!(minor_radius > 0.0, "Minor radius must be positive");
        assert!(
            major_radius > minor_radius,
            "Major radius must be greater than minor radius for a ring torus"
        );

        Self {
            major_radius,
            minor_radius,
            aspect_ratio: major_radius / minor_radius,
        }
    }

    /// Create a unit torus with R=2, r=1
    pub fn unit() -> Self {
        Self::new(2.0, 1.0)
    }

    /// Surface area of the torus: 4π²Rr
    pub fn surface_area(&self) -> f64 {
        4.0 * PI * PI * self.major_radius * self.minor_radius
    }

    /// Volume of the torus: 2π²Rr²
    pub fn volume(&self) -> f64 {
        2.0 * PI * PI * self.major_radius * self.minor_radius * self.minor_radius
    }

    /// Generate a grid of coordinates on the torus surface
    pub fn generate_grid(&self, n_major: usize, n_minor: usize) -> Vec<TorusCoordinate> {
        let mut coords = Vec::with_capacity(n_major * n_minor);
        let du = 2.0 * PI / n_major as f64;
        let dv = 2.0 * PI / n_minor as f64;

        for i in 0..n_major {
            for j in 0..n_minor {
                coords.push(TorusCoordinate::new(i as f64 * du, j as f64 * dv));
            }
        }
        coords
    }

    /// Generate spiral coordinates following a vortex pattern
    pub fn generate_spiral(
        &self,
        n_points: usize,
        winding_number: f64,
        start_phase: f64,
    ) -> Vec<TorusCoordinate> {
        let mut coords = Vec::with_capacity(n_points);
        let dt = 2.0 * PI / n_points as f64;

        for i in 0..n_points {
            let t = i as f64 * dt + start_phase;
            let u = t;
            let v = winding_number * t;
            coords.push(TorusCoordinate::new(u, v));
        }
        coords
    }
}

/// Distance matrix on torus using geodesic distances
#[derive(Debug, Clone)]
pub struct TorusDistanceMatrix {
    pub distances: Vec<Vec<f64>>,
    pub size: usize,
}

impl TorusDistanceMatrix {
    /// Compute pairwise geodesic distances for a set of coordinates
    pub fn from_coordinates(coords: &[TorusCoordinate]) -> Self {
        let n = coords.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i + 1..n {
                let d = coords[i].geodesic_distance(&coords[j]);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }

        Self { distances, size: n }
    }

    /// Convert distances to attention weights using Gaussian kernel
    pub fn to_attention_weights(&self, sigma: f64) -> Vec<Vec<f64>> {
        let neg_inv_2sigma2 = -0.5 / (sigma * sigma);
        self.distances
            .iter()
            .map(|row| {
                let weights: Vec<f64> = row
                    .iter()
                    .map(|&d| (d * d * neg_inv_2sigma2).exp())
                    .collect();
                // Normalize (softmax-like)
                let sum: f64 = weights.iter().sum();
                weights.iter().map(|&w| w / sum).collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinate_wrapping() {
        let coord = TorusCoordinate::new(3.0 * PI, -PI / 2.0);
        assert!((coord.u - PI).abs() < 1e-10);
        assert!((coord.v - 3.0 * PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_distance() {
        let a = TorusCoordinate::new(0.0, 0.0);
        let b = TorusCoordinate::new(PI, 0.0);
        let dist = a.geodesic_distance(&b);
        assert!((dist - PI).abs() < 1e-10);
    }

    #[test]
    fn test_torus_manifold() {
        let torus = TorusManifold::new(2.0, 1.0);
        assert!((torus.aspect_ratio - 2.0).abs() < 1e-10);
    }
}
