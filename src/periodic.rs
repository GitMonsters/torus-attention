//! Periodic boundary conditions for torus topology
//!
//! The torus has periodic boundary conditions in both angular directions.
//! This module handles wrapping, interpolation, and convolution on the torus.

use crate::geometry::TorusCoordinate;
use candle_core::{DType, Device, Tensor};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Periodic boundary handler for torus manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicBoundary {
    /// Number of grid points in major (u) direction
    pub n_major: usize,
    /// Number of grid points in minor (v) direction
    pub n_minor: usize,
    /// Grid spacing in major direction
    pub du: f64,
    /// Grid spacing in minor direction
    pub dv: f64,
}

impl PeriodicBoundary {
    /// Create a new periodic boundary handler
    pub fn new(n_major: usize, n_minor: usize) -> Self {
        Self {
            n_major,
            n_minor,
            du: 2.0 * PI / n_major as f64,
            dv: 2.0 * PI / n_minor as f64,
        }
    }

    /// Wrap index with periodic boundary
    #[inline]
    pub fn wrap_index(&self, idx: i64, size: usize) -> usize {
        let s = size as i64;
        ((idx % s + s) % s) as usize
    }

    /// Get the wrapped 2D index (i, j) -> (wrapped_i, wrapped_j)
    #[inline]
    pub fn wrap_2d(&self, i: i64, j: i64) -> (usize, usize) {
        (
            self.wrap_index(i, self.n_major),
            self.wrap_index(j, self.n_minor),
        )
    }

    /// Convert continuous coordinates to grid indices with interpolation weights
    pub fn coord_to_grid(&self, coord: &TorusCoordinate) -> ((usize, usize), (usize, usize), f64, f64) {
        // Normalize to [0, 2π)
        let u = TorusCoordinate::wrap_angle(coord.u);
        let v = TorusCoordinate::wrap_angle(coord.v);

        // Get fractional grid position
        let fi = u / self.du;
        let fj = v / self.dv;

        // Integer parts (floor)
        let i0 = fi.floor() as usize % self.n_major;
        let j0 = fj.floor() as usize % self.n_minor;

        // Wrapped next indices
        let i1 = (i0 + 1) % self.n_major;
        let j1 = (j0 + 1) % self.n_minor;

        // Fractional parts for interpolation
        let wi = fi.fract();
        let wj = fj.fract();

        ((i0, j0), (i1, j1), wi, wj)
    }

    /// Bilinear interpolation with periodic boundary
    pub fn interpolate(&self, grid: &Array2<f64>, coord: &TorusCoordinate) -> f64 {
        let ((i0, j0), (i1, j1), wi, wj) = self.coord_to_grid(coord);

        let v00 = grid[[i0, j0]];
        let v10 = grid[[i1, j0]];
        let v01 = grid[[i0, j1]];
        let v11 = grid[[i1, j1]];

        // Bilinear interpolation
        let v0 = v00 * (1.0 - wi) + v10 * wi;
        let v1 = v01 * (1.0 - wi) + v11 * wi;
        v0 * (1.0 - wj) + v1 * wj
    }

    /// Apply a periodic convolution kernel (parallelized)
    pub fn convolve_periodic(&self, grid: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
        let kh = kernel.nrows() as i64;
        let kw = kernel.ncols() as i64;
        let kh_half = kh / 2;
        let kw_half = kw / 2;

        let mut result = Array2::zeros((self.n_major, self.n_minor));

        // Parallel iteration over output positions
        let results: Vec<(usize, usize, f64)> = (0..self.n_major)
            .into_par_iter()
            .flat_map(|i| {
                (0..self.n_minor).into_par_iter().map(move |j| {
                    let mut sum = 0.0;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let gi = i as i64 + ki - kh_half;
                            let gj = j as i64 + kj - kw_half;
                            let (wi, wj) = self.wrap_2d(gi, gj);
                            sum += grid[[wi, wj]] * kernel[[ki as usize, kj as usize]];
                        }
                    }
                    (i, j, sum)
                })
            })
            .collect();

        for (i, j, val) in results {
            result[[i, j]] = val;
        }

        result
    }

    /// Generate periodic position encodings for the torus
    /// Uses both sin and cos for each frequency in both directions
    pub fn position_encodings(&self, n_frequencies: usize, device: &Device) -> crate::TorusResult<Tensor> {
        let total_positions = self.n_major * self.n_minor;
        let encoding_dim = 4 * n_frequencies; // sin/cos for both u and v directions

        let mut encodings = vec![0.0f32; total_positions * encoding_dim];

        for i in 0..self.n_major {
            for j in 0..self.n_minor {
                let pos_idx = i * self.n_minor + j;
                let u = i as f64 * self.du;
                let v = j as f64 * self.dv;

                for f in 0..n_frequencies {
                    let freq = (f + 1) as f64;
                    let base_idx = pos_idx * encoding_dim + f * 4;

                    encodings[base_idx] = (freq * u).sin() as f32;
                    encodings[base_idx + 1] = (freq * u).cos() as f32;
                    encodings[base_idx + 2] = (freq * v).sin() as f32;
                    encodings[base_idx + 3] = (freq * v).cos() as f32;
                }
            }
        }

        let tensor = Tensor::from_vec(encodings, (total_positions, encoding_dim), device)?;
        Ok(tensor)
    }

    /// Create a periodic distance kernel (Gaussian on torus)
    pub fn gaussian_kernel(&self, sigma: f64, kernel_size: usize) -> Array2<f64> {
        let mut kernel = Array2::zeros((kernel_size, kernel_size));
        let half = kernel_size as f64 / 2.0;
        let neg_inv_2sigma2 = -0.5 / (sigma * sigma);

        let mut sum = 0.0;
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                // Use periodic distance
                let di = (i as f64 - half).abs();
                let dj = (j as f64 - half).abs();
                // Consider wrapping for kernel distances
                let di = di.min(kernel_size as f64 - di);
                let dj = dj.min(kernel_size as f64 - dj);
                let d2 = di * di + dj * dj;
                let val = (d2 * neg_inv_2sigma2).exp();
                kernel[[i, j]] = val;
                sum += val;
            }
        }

        // Normalize
        kernel.mapv_inplace(|x| x / sum);
        kernel
    }

    /// Shift tensor values with periodic wrapping
    pub fn periodic_shift(&self, grid: &Array2<f64>, shift_u: i64, shift_v: i64) -> Array2<f64> {
        let mut result = Array2::zeros((self.n_major, self.n_minor));
        
        for i in 0..self.n_major {
            for j in 0..self.n_minor {
                let (si, sj) = self.wrap_2d(i as i64 - shift_u, j as i64 - shift_v);
                result[[i, j]] = grid[[si, sj]];
            }
        }
        
        result
    }

    /// Compute periodic gradient (finite differences with wrapping)
    pub fn gradient(&self, grid: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut grad_u = Array2::zeros((self.n_major, self.n_minor));
        let mut grad_v = Array2::zeros((self.n_major, self.n_minor));

        for i in 0..self.n_major {
            for j in 0..self.n_minor {
                // Central differences with periodic wrapping
                let i_prev = self.wrap_index(i as i64 - 1, self.n_major);
                let i_next = self.wrap_index(i as i64 + 1, self.n_major);
                let j_prev = self.wrap_index(j as i64 - 1, self.n_minor);
                let j_next = self.wrap_index(j as i64 + 1, self.n_minor);

                grad_u[[i, j]] = (grid[[i_next, j]] - grid[[i_prev, j]]) / (2.0 * self.du);
                grad_v[[i, j]] = (grid[[i, j_next]] - grid[[i, j_prev]]) / (2.0 * self.dv);
            }
        }

        (grad_u, grad_v)
    }
}

/// Periodic attention mask that respects torus topology
#[derive(Debug, Clone)]
pub struct PeriodicAttentionMask {
    /// The attention mask tensor
    pub mask: Tensor,
    /// Window size in major direction
    pub window_major: usize,
    /// Window size in minor direction
    pub window_minor: usize,
}

impl PeriodicAttentionMask {
    /// Create a local attention mask with periodic boundaries
    pub fn new(
        boundary: &PeriodicBoundary,
        window_major: usize,
        window_minor: usize,
        device: &Device,
    ) -> crate::TorusResult<Self> {
        let n = boundary.n_major * boundary.n_minor;
        let mut mask_data = vec![0.0f32; n * n];

        for i in 0..boundary.n_major {
            for j in 0..boundary.n_minor {
                let src_idx = i * boundary.n_minor + j;
                
                // Create window around (i, j) with periodic wrapping
                for di in -(window_major as i64 / 2)..=(window_major as i64 / 2) {
                    for dj in -(window_minor as i64 / 2)..=(window_minor as i64 / 2) {
                        let (ti, tj) = boundary.wrap_2d(i as i64 + di, j as i64 + dj);
                        let tgt_idx = ti * boundary.n_minor + tj;
                        mask_data[src_idx * n + tgt_idx] = 1.0;
                    }
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (n, n), device)?;

        Ok(Self {
            mask,
            window_major,
            window_minor,
        })
    }

    /// Create a spiral attention pattern
    pub fn spiral(
        boundary: &PeriodicBoundary,
        winding: f64,
        bandwidth: f64,
        device: &Device,
    ) -> crate::TorusResult<Self> {
        let n = boundary.n_major * boundary.n_minor;
        let mut mask_data = vec![0.0f32; n * n];

        for i in 0..boundary.n_major {
            for j in 0..boundary.n_minor {
                let src_idx = i * boundary.n_minor + j;
                let u_src = i as f64 * boundary.du;
                let v_src = j as f64 * boundary.dv;
                let spiral_src = u_src + winding * v_src;

                for ti in 0..boundary.n_major {
                    for tj in 0..boundary.n_minor {
                        let tgt_idx = ti * boundary.n_minor + tj;
                        let u_tgt = ti as f64 * boundary.du;
                        let v_tgt = tj as f64 * boundary.dv;
                        let spiral_tgt = u_tgt + winding * v_tgt;

                        // Periodic spiral distance
                        let spiral_dist = TorusCoordinate::angular_distance(spiral_src, spiral_tgt);
                        
                        if spiral_dist < bandwidth {
                            mask_data[src_idx * n + tgt_idx] = 1.0;
                        }
                    }
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (n, n), device)?;

        Ok(Self {
            mask,
            window_major: boundary.n_major,
            window_minor: boundary.n_minor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_index() {
        let boundary = PeriodicBoundary::new(8, 8);
        assert_eq!(boundary.wrap_index(-1, 8), 7);
        assert_eq!(boundary.wrap_index(8, 8), 0);
        assert_eq!(boundary.wrap_index(10, 8), 2);
    }

    #[test]
    fn test_periodic_shift() {
        let boundary = PeriodicBoundary::new(4, 4);
        let mut grid = Array2::zeros((4, 4));
        grid[[0, 0]] = 1.0;
        
        let shifted = boundary.periodic_shift(&grid, 1, 0);
        assert!((shifted[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_symmetry() {
        let boundary = PeriodicBoundary::new(8, 8);
        let mut grid = Array2::zeros((8, 8));
        // Create a simple pattern
        for i in 0..8 {
            for j in 0..8 {
                grid[[i, j]] = (i as f64 * boundary.du).sin();
            }
        }
        
        let (grad_u, _) = boundary.gradient(&grid);
        // Gradient should be approximately cos
        let expected = (0.0_f64).cos();
        assert!((grad_u[[0, 0]] - expected).abs() < 0.2);
    }
}
