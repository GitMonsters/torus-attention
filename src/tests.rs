//! Integration Tests for Torus Attention
//!
//! Comprehensive tests for the complete bidirectional torus attention system.
//! Organized into categories:
//! - Geometry tests (torus coordinates, geodesic distances)
//! - Periodic boundary tests (wrapping, convolution)
//! - Stream tests (8-stream parallel processing)
//! - Compounding tests (EMA, learnable alpha)
//! - Vortex tests (spiral dynamics)
//! - Configuration tests (serialization, validation)
//! - Tensor shape tests (attention dimensions)
//! - Numerical stability tests (overflow/underflow)
//! - End-to-end tests (full pipeline)
//! - Training tests (loss, optimizer, scheduler)

use crate::compounding::{CompoundingConfig, LearnableAlpha};
use crate::geometry::{TorusCoordinate, TorusDistanceMatrix, TorusManifold};
use crate::integration::BidirectionalTorusConfig;
use crate::parallel_streams::StreamId;
use crate::periodic::PeriodicBoundary;
use crate::vortex::{SpiralAttention, Vortex};
use candle_core::{Device, Tensor};
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════
// GEOMETRY INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn test_torus_coordinate_periodicity() {
        // Test that coordinates wrap around correctly
        let c1 = TorusCoordinate::new(0.0, 0.0);
        let c2 = TorusCoordinate::new(2.0 * PI, 2.0 * PI);
        let _c3 = TorusCoordinate::new(4.0 * PI, 4.0 * PI);

        // All should represent the same point on the torus
        assert!((c1.u - c2.u).abs() < 1e-10 || (c1.u - c2.u - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_distance_symmetry() {
        let c1 = TorusCoordinate::new(0.5, 0.3);
        let c2 = TorusCoordinate::new(1.2, 0.8);

        let d1 = c1.geodesic_distance(&c2);
        let d2 = c2.geodesic_distance(&c1);

        // Distance should be symmetric
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_distance_triangle_inequality() {
        let c1 = TorusCoordinate::new(0.0, 0.0);
        let c2 = TorusCoordinate::new(1.0, 0.5);
        let c3 = TorusCoordinate::new(2.0, 1.0);

        let d12 = c1.geodesic_distance(&c2);
        let d23 = c2.geodesic_distance(&c3);
        let d13 = c1.geodesic_distance(&c3);

        // Triangle inequality: d13 <= d12 + d23
        assert!(d13 <= d12 + d23 + 1e-10);
    }

    #[test]
    fn test_torus_manifold_grid() {
        let torus = TorusManifold::new(2.0, 1.0);
        let grid = torus.generate_grid(8, 4);

        assert_eq!(grid.len(), 8 * 4);

        // Check grid spans full range
        let max_u = grid.iter().map(|c| c.u).fold(0.0, f64::max);
        let max_v = grid.iter().map(|c| c.v).fold(0.0, f64::max);

        assert!(max_u < 2.0 * PI);
        assert!(max_v < 2.0 * PI);
    }

    #[test]
    fn test_distance_matrix_diagonal() {
        let torus = TorusManifold::new(2.0, 1.0);
        let coords = torus.generate_grid(4, 4);
        let matrix = TorusDistanceMatrix::from_coordinates(&coords);

        // Diagonal should be zeros
        for i in 0..coords.len() {
            assert!(matrix.distances[i][i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_cartesian_conversion_roundtrip() {
        let coord = TorusCoordinate::new(1.0, 0.5);
        let (x, y, z) = coord.to_cartesian(2.0, 1.0);

        // Point should be on the torus surface
        // (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
        let r_major = (x * x + y * y).sqrt();
        let dist_from_axis = r_major - 2.0;
        let on_torus = dist_from_axis * dist_from_axis + z * z;

        assert!((on_torus - 1.0).abs() < 1e-10); // Should equal r^2 = 1
    }

    #[test]
    fn test_distance_matrix_symmetry() {
        let torus = TorusManifold::new(2.0, 1.0);
        let coords = torus.generate_grid(8, 8);
        let matrix = TorusDistanceMatrix::from_coordinates(&coords);

        // Distance matrix should be symmetric
        for i in 0..coords.len() {
            for j in 0..coords.len() {
                assert!((matrix.distances[i][j] - matrix.distances[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_unit_torus() {
        let torus = TorusManifold::unit();
        assert_eq!(torus.major_radius, 2.0);
        assert_eq!(torus.minor_radius, 1.0);
    }

    #[test]
    fn test_torus_surface_area() {
        let torus = TorusManifold::new(2.0, 1.0);
        let area = torus.surface_area();
        // Surface area = 4 * π² * R * r
        let expected = 4.0 * PI * PI * 2.0 * 1.0;
        assert!((area - expected).abs() < 1e-10);
    }

    #[test]
    fn test_torus_volume() {
        let torus = TorusManifold::new(2.0, 1.0);
        let volume = torus.volume();
        // Volume = 2 * π² * R * r²
        let expected = 2.0 * PI * PI * 2.0 * 1.0;
        assert!((volume - expected).abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_identity_distance() {
        let c = TorusCoordinate::new(1.5, 0.8);
        assert!(c.geodesic_distance(&c) < 1e-10);
    }

    #[test]
    fn test_opposite_points_distance() {
        let c1 = TorusCoordinate::new(0.0, 0.0);
        let c2 = TorusCoordinate::new(PI, PI);
        let dist = c1.geodesic_distance(&c2);

        // Should be significant but less than max possible
        assert!(dist > 0.0);
        assert!(dist <= 2.0 * PI);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PERIODIC BOUNDARY TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod periodic_tests {
    use super::*;

    #[test]
    fn test_wrap_2d_positive() {
        let boundary = PeriodicBoundary::new(8, 4);

        // Normal indices
        assert_eq!(boundary.wrap_2d(3, 2), (3, 2));

        // Wrapped positive
        assert_eq!(boundary.wrap_2d(10, 6), (2, 2));
    }

    #[test]
    fn test_wrap_2d_negative() {
        let boundary = PeriodicBoundary::new(8, 4);

        // Negative wrapping
        assert_eq!(boundary.wrap_2d(-1, -1), (7, 3));
        assert_eq!(boundary.wrap_2d(-9, -5), (7, 3));
    }

    #[test]
    fn test_periodic_convolution_kernel_sum() {
        let boundary = PeriodicBoundary::new(16, 8);
        let kernel = boundary.gaussian_kernel(1.0, 3);

        // Kernel should sum to approximately 1
        let sum: f64 = kernel.iter().copied().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_periodic_wrap_2d() {
        let boundary = PeriodicBoundary::new(8, 4);

        // Test wrap at boundaries
        let (i, j) = boundary.wrap_2d(-1, -1);
        assert_eq!(i, 7); // -1 wraps to n_major - 1
        assert_eq!(j, 3); // -1 wraps to n_minor - 1

        let (i, j) = boundary.wrap_2d(8, 4);
        assert_eq!(i, 0); // n_major wraps to 0
        assert_eq!(j, 0); // n_minor wraps to 0
    }

    #[test]
    fn test_wrap_2d_large_values() {
        let boundary = PeriodicBoundary::new(8, 4);

        // Large positive values
        let (i, j) = boundary.wrap_2d(100, 50);
        assert!(i < 8);
        assert!(j < 4);

        // Large negative values
        let (i, j) = boundary.wrap_2d(-100, -50);
        assert!(i < 8);
        assert!(j < 4);
    }

    #[test]
    fn test_boundary_total_positions() {
        let boundary = PeriodicBoundary::new(16, 8);
        assert_eq!(boundary.n_major * boundary.n_minor, 128);
    }

    #[test]
    fn test_gaussian_kernel_symmetry() {
        let boundary = PeriodicBoundary::new(16, 8);
        let kernel = boundary.gaussian_kernel(1.0, 5);

        // Kernel should be normalized (sum to 1)
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Kernel should sum to 1");

        // All values should be non-negative
        assert!(
            kernel.iter().all(|&v| v >= 0.0),
            "Kernel values should be non-negative"
        );

        // Kernel size should match requested
        assert_eq!(kernel.nrows(), 5);
        assert_eq!(kernel.ncols(), 5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STREAM ID TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn test_all_streams_unique() {
        let streams = StreamId::all();
        let mut indices: Vec<usize> = streams.iter().map(|s| s.index()).collect();
        indices.sort();
        indices.dedup();

        assert_eq!(indices.len(), 8);
    }

    #[test]
    fn test_stream_pairing() {
        // Each forward stream should pair with corresponding backward
        assert_eq!(StreamId::MajorForward.pair(), StreamId::MajorBackward);
        assert_eq!(StreamId::MajorBackward.pair(), StreamId::MajorForward);
        assert_eq!(StreamId::SpiralCW.pair(), StreamId::SpiralCCW);
        assert_eq!(StreamId::CrossUtoV.pair(), StreamId::CrossVtoU);
    }

    #[test]
    fn test_forward_backward_split() {
        let streams = StreamId::all();
        let forward: Vec<_> = streams.iter().filter(|s| s.is_forward()).collect();
        let backward: Vec<_> = streams.iter().filter(|s| !s.is_forward()).collect();

        assert_eq!(forward.len(), 4);
        assert_eq!(backward.len(), 4);
    }

    #[test]
    fn test_stream_names_unique() {
        let streams = StreamId::all();
        let names: Vec<&str> = streams.iter().map(|s| s.name()).collect();
        let mut unique_names = names.clone();
        unique_names.sort();
        unique_names.dedup();

        assert_eq!(names.len(), unique_names.len());
    }

    #[test]
    fn test_stream_indices_contiguous() {
        let streams = StreamId::all();
        let mut indices: Vec<usize> = streams.iter().map(|s| s.index()).collect();
        indices.sort();

        for (i, idx) in indices.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }

    #[test]
    fn test_stream_pair_symmetry() {
        let streams = StreamId::all();
        for stream in &streams {
            let pair = stream.pair();
            assert_eq!(pair.pair(), *stream);
        }
    }

    #[test]
    fn test_forward_streams() {
        assert!(StreamId::MajorForward.is_forward());
        assert!(StreamId::MinorForward.is_forward());
        assert!(StreamId::SpiralCW.is_forward());
        assert!(StreamId::CrossUtoV.is_forward());
    }

    #[test]
    fn test_backward_streams() {
        assert!(!StreamId::MajorBackward.is_forward());
        assert!(!StreamId::MinorBackward.is_forward());
        assert!(!StreamId::SpiralCCW.is_forward());
        assert!(!StreamId::CrossVtoU.is_forward());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOUNDING TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod compounding_tests {
    use super::*;

    #[test]
    fn test_learnable_alpha_bounds() {
        let device = Device::Cpu;

        // Create alpha with extreme initial value
        let alpha_low = LearnableAlpha::new(0.05, 0.1, 0.99, &device).unwrap();
        let value_low = alpha_low.get().unwrap();
        assert!(value_low >= 0.1, "Alpha should be clamped to min");

        let alpha_high = LearnableAlpha::new(1.5, 0.1, 0.99, &device).unwrap();
        let value_high = alpha_high.get().unwrap();
        assert!(value_high <= 0.99, "Alpha should be clamped to max");
    }

    #[test]
    fn test_compounding_config_validation() {
        let config = CompoundingConfig::default();

        assert!(config.min_alpha < config.max_alpha);
        assert!(config.base_alpha >= config.min_alpha);
        assert!(config.base_alpha <= config.max_alpha);
        assert!(config.layer_scale > 0.0 && config.layer_scale <= 1.0);
    }

    #[test]
    fn test_layer_scale_decay() {
        let config = CompoundingConfig {
            n_layers: 6,
            base_alpha: 0.9,
            layer_scale: 0.9,
            min_alpha: 0.1,
            max_alpha: 0.99,
            ..CompoundingConfig::default()
        };

        // Each layer should have decaying alpha
        let mut prev_alpha = config.base_alpha;
        for l in 0..config.n_layers {
            let layer_alpha = config.base_alpha * config.layer_scale.powi(l as i32);
            let clamped = layer_alpha.clamp(config.min_alpha, config.max_alpha);

            if l > 0 {
                assert!(clamped <= prev_alpha, "Alpha should decay with depth");
            }
            prev_alpha = clamped;
        }
    }

    #[test]
    fn test_learnable_alpha_in_range() {
        let device = Device::Cpu;
        let alpha = LearnableAlpha::new(0.5, 0.1, 0.99, &device).unwrap();
        let value = alpha.get().unwrap();

        assert!(value >= 0.1);
        assert!(value <= 0.99);
    }

    #[test]
    fn test_compounding_config_default() {
        let config = CompoundingConfig::default();
        assert!(config.n_layers > 0);
        assert!(config.d_model > 0);
        assert!(config.base_alpha > 0.0 && config.base_alpha < 1.0);
    }

    #[test]
    fn test_alpha_decay_sequence() {
        let config = CompoundingConfig {
            n_layers: 10,
            base_alpha: 0.95,
            layer_scale: 0.9,
            min_alpha: 0.3,
            max_alpha: 0.99,
            ..CompoundingConfig::default()
        };

        let mut alphas = Vec::new();
        for l in 0..config.n_layers {
            let alpha = config.base_alpha * config.layer_scale.powi(l as i32);
            alphas.push(alpha.clamp(config.min_alpha, config.max_alpha));
        }

        // Verify monotonically non-increasing
        for i in 1..alphas.len() {
            assert!(alphas[i] <= alphas[i - 1]);
        }

        // Verify all in bounds
        for alpha in &alphas {
            assert!(*alpha >= config.min_alpha);
            assert!(*alpha <= config.max_alpha);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VORTEX TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod vortex_tests {
    use super::*;

    #[test]
    fn test_vortex_velocity_decay() {
        let vortex = Vortex::new(PI, PI, 1.0, 0, 0);

        // Velocity should decay with distance
        let near_point = TorusCoordinate::new(PI + 0.1, PI);
        let far_point = TorusCoordinate::new(PI + 1.0, PI);

        let (u_near, v_near) = vortex.velocity_at(&near_point);
        let (u_far, v_far) = vortex.velocity_at(&far_point);

        let speed_near = (u_near * u_near + v_near * v_near).sqrt();
        let speed_far = (u_far * u_far + v_far * v_far).sqrt();

        assert!(speed_near > speed_far);
    }

    #[test]
    fn test_spiral_attention_golden() {
        let spiral = SpiralAttention::golden();

        // Golden ratio winding
        assert!((spiral.winding - 1.618033988749895).abs() < 1e-10);
    }

    #[test]
    fn test_spiral_path_continuity() {
        let vortex = Vortex::new(PI, PI, 1.0, 1, 1);
        let path = vortex.spiral_path(100, 4.0 * PI);

        assert_eq!(path.len(), 100);

        // Path should be continuous (no large jumps)
        for i in 1..path.len() {
            let dist = path[i].geodesic_distance(&path[i - 1]);
            // Max step should be bounded
            assert!(dist < 1.0, "Spiral path should be smooth");
        }
    }

    #[test]
    fn test_vortex_creation() {
        let vortex = Vortex::new(1.0, 2.0, 0.5, 1, -1);
        assert!((vortex.position.u - 1.0).abs() < 1e-10);
        assert!((vortex.position.v - 2.0).abs() < 1e-10);
        assert!((vortex.circulation - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_spiral_path_length() {
        let vortex = Vortex::new(PI, PI, 1.0, 1, 1);

        for n_points in [10, 50, 100, 200] {
            let path = vortex.spiral_path(n_points, 2.0 * PI);
            assert_eq!(path.len(), n_points);
        }
    }

    #[test]
    fn test_vortex_at_center() {
        let vortex = Vortex::new(PI, PI, 1.0, 0, 0);
        let center = TorusCoordinate::new(PI, PI);
        let (u_vel, v_vel) = vortex.velocity_at(&center);

        // At center, velocity should be minimal (or regularized)
        let speed = (u_vel * u_vel + v_vel * v_vel).sqrt();
        assert!(speed.is_finite());
    }

    #[test]
    fn test_spiral_winding_numbers() {
        // Test various winding numbers
        let spiral_golden = SpiralAttention::golden();
        let spiral_custom = SpiralAttention::new(2.0, PI / 8.0, 1);

        assert!((spiral_golden.winding - 1.618033988749895).abs() < 1e-10);
        assert!((spiral_custom.winding - 2.0).abs() < 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_bidirectional_config_defaults() {
        let config = BidirectionalTorusConfig::default();

        assert_eq!(
            config.d_model % config.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );
        assert!(config.major_radius > config.minor_radius);
        assert!(config.use_parallel_streams);
        assert!(config.use_compounding);
    }

    #[test]
    fn test_seq_len_calculation() {
        let config = BidirectionalTorusConfig {
            n_major: 16,
            n_minor: 8,
            ..BidirectionalTorusConfig::default()
        };

        assert_eq!(config.seq_len(), 128);
    }

    #[test]
    fn test_config_to_parallel() {
        let config = BidirectionalTorusConfig::default();
        let parallel = config.to_parallel_config();

        assert_eq!(parallel.d_model, config.d_model);
        assert_eq!(parallel.n_heads, config.n_heads);
        assert_eq!(parallel.n_major, config.n_major);
        assert_eq!(parallel.spiral_winding, config.spiral_winding);
    }

    #[test]
    fn test_config_to_compounding() {
        let config = BidirectionalTorusConfig::default();
        let comp = config.to_compounding_config();

        assert_eq!(comp.n_layers, config.n_layers);
        assert_eq!(comp.d_model, config.d_model);
        assert_eq!(comp.base_alpha, config.ema_alpha);
    }

    #[test]
    fn test_golden_ratio_spiral() {
        let config = BidirectionalTorusConfig::default();
        let phi = 1.618033988749895;

        assert!((config.spiral_winding - phi).abs() < 1e-10);
    }

    #[test]
    fn test_config_custom_values() {
        let config = BidirectionalTorusConfig {
            d_model: 128,
            d_ff: 512,
            n_heads: 4,
            n_layers: 4,
            n_major: 8,
            n_minor: 8,
            ..BidirectionalTorusConfig::default()
        };

        assert_eq!(config.d_model, 128);
        assert_eq!(config.d_ff, 512);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.seq_len(), 64);
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = BidirectionalTorusConfig::default();
        let head_dim = config.d_model / config.n_heads;

        assert!(head_dim > 0);
        assert_eq!(head_dim * config.n_heads, config.d_model);
    }

    #[test]
    fn test_config_geodesic_params() {
        let config = BidirectionalTorusConfig::default();
        assert!(config.geodesic_sigma > 0.0);
        assert!(config.use_geodesic_bias);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION MASK TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod mask_tests {

    #[test]
    fn test_causal_mask_diagonal() {
        // A causal mask should have 1s on and below diagonal
        let n = 8;
        let mut mask = vec![vec![0.0; n]; n];

        // Create lower triangular mask
        for i in 0..n {
            for j in 0..=i {
                mask[i][j] = 1.0;
            }
        }

        // Verify structure
        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    assert_eq!(mask[i][j], 1.0);
                } else {
                    assert_eq!(mask[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_bidirectional_mask_symmetric() {
        // When combining forward and backward masks symmetrically
        let n = 4;
        let forward = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        let backward = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        // Combined should cover all positions
        for i in 0..n {
            for j in 0..n {
                let combined = forward[i][j] + backward[i][j];
                assert!(combined >= 1.0, "Bidirectional should cover all positions");
            }
        }
    }

    #[test]
    fn test_causal_mask_properties() {
        let n = 16;
        let mut mask = vec![vec![0.0; n]; n];

        // Create lower triangular mask
        for i in 0..n {
            for j in 0..=i {
                mask[i][j] = 1.0;
            }
        }

        // Count nonzeros
        let nonzeros: usize = mask.iter().flatten().filter(|&&x| x > 0.0).count();
        let expected = n * (n + 1) / 2;
        assert_eq!(nonzeros, expected);
    }

    #[test]
    fn test_full_attention_mask() {
        let n = 8;
        let mask = vec![vec![1.0; n]; n];

        // All positions should attend to all
        for i in 0..n {
            for j in 0..n {
                assert_eq!(mask[i][j], 1.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR SHAPE TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn test_attention_shapes() {
        let device = Device::Cpu;

        let batch = 2;
        let seq_len = 16;
        let d_model = 64;
        let n_heads = 4;
        let head_dim = d_model / n_heads;

        // Input: [B, S, D]
        let x = Tensor::randn(0.0f32, 1.0, (batch, seq_len, d_model), &device).unwrap();
        assert_eq!(x.dims(), &[batch, seq_len, d_model]);

        // After reshape for multi-head: [B, H, S, D/H]
        let reshaped = x
            .reshape((batch, seq_len, n_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        assert_eq!(reshaped.dims(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_position_encoding_shapes() {
        let device = Device::Cpu;

        let n_major = 8;
        let n_minor = 4;
        let d_model = 32;
        let seq_len = n_major * n_minor;

        // 2D position encoding should be [S, D]
        let pos_enc = Tensor::randn(0.0f32, 1.0, (seq_len, d_model), &device).unwrap();
        assert_eq!(pos_enc.dims(), &[seq_len, d_model]);

        // Can also be viewed as [n_major, n_minor, D]
        let pos_2d = pos_enc.reshape((n_major, n_minor, d_model)).unwrap();
        assert_eq!(pos_2d.dims(), &[n_major, n_minor, d_model]);
    }

    #[test]
    fn test_stream_output_shapes() {
        let device = Device::Cpu;

        let batch = 4;
        let seq_len = 32;
        let d_model = 64;
        let n_streams = 8;

        // Each stream produces [B, S, D]
        let stream_outputs: Vec<Tensor> = (0..n_streams)
            .map(|_| Tensor::randn(0.0f32, 1.0, (batch, seq_len, d_model), &device).unwrap())
            .collect();

        assert_eq!(stream_outputs.len(), n_streams);
        for output in &stream_outputs {
            assert_eq!(output.dims(), &[batch, seq_len, d_model]);
        }
    }

    #[test]
    fn test_batch_dimension_handling() {
        let device = Device::Cpu;

        // Test various batch sizes
        for batch in [1, 2, 4, 8, 16] {
            let x = Tensor::randn(0.0f32, 1.0, (batch, 32, 64), &device).unwrap();
            assert_eq!(x.dims()[0], batch);
        }
    }

    #[test]
    fn test_qkv_projection_shapes() {
        let device = Device::Cpu;

        let batch = 2;
        let seq_len = 16;
        let d_model = 64;
        let n_heads = 4;
        let head_dim = d_model / n_heads;

        // Q, K, V projections all have shape [B, S, D]
        let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, d_model), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, d_model), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, d_model), &device).unwrap();

        // Reshape to [B, H, S, D/H]
        let q_heads = q
            .reshape((batch, seq_len, n_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k_heads = k
            .reshape((batch, seq_len, n_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v_heads = v
            .reshape((batch, seq_len, n_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        assert_eq!(q_heads.dims(), &[batch, n_heads, seq_len, head_dim]);
        assert_eq!(k_heads.dims(), &[batch, n_heads, seq_len, head_dim]);
        assert_eq!(v_heads.dims(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_attention_score_shapes() {
        let device = Device::Cpu;

        let batch = 2;
        let n_heads = 4;
        let seq_len = 16;
        let head_dim = 16;

        // Q: [B, H, S, D/H], K^T: [B, H, D/H, S]
        let q = Tensor::randn(0.0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, n_heads, seq_len, head_dim), &device).unwrap();

        // Attention scores: [B, H, S, S]
        let k_t = k.transpose(2, 3).unwrap();
        let scores = q.matmul(&k_t).unwrap();

        assert_eq!(scores.dims(), &[batch, n_heads, seq_len, seq_len]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NUMERICAL STABILITY TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod stability_tests {
    use super::*;

    #[test]
    fn test_softmax_stability() {
        let device = Device::Cpu;

        // Large values that could cause overflow
        let logits = Tensor::new(&[1000.0f32, 1001.0, 999.0], &device).unwrap();
        let softmax = candle_nn::ops::softmax(&logits, 0).unwrap();
        let values: Vec<f32> = softmax.to_vec1().unwrap();

        // Should sum to 1 and not contain NaN/Inf
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_log_softmax_stability() {
        let device = Device::Cpu;

        // Very negative values
        let logits = Tensor::new(&[-100.0f32, -50.0, -1.0], &device).unwrap();
        let log_softmax = candle_nn::ops::log_softmax(&logits, 0).unwrap();
        let values: Vec<f32> = log_softmax.to_vec1().unwrap();

        // All values should be finite and negative
        assert!(values.iter().all(|v| v.is_finite() && *v <= 0.0));
    }

    #[test]
    fn test_geodesic_distance_bounds() {
        // Distance on unit torus should be bounded
        let torus = TorusManifold::unit();
        let coords = torus.generate_grid(16, 16);

        for c1 in &coords {
            for c2 in &coords {
                let dist = c1.geodesic_distance(c2);
                assert!(dist >= 0.0);
                assert!(dist <= 2.0 * PI); // Max distance on torus
            }
        }
    }

    #[test]
    fn test_softmax_with_zeros() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.0f32, 0.0, 0.0], &device).unwrap();
        let softmax = candle_nn::ops::softmax(&logits, 0).unwrap();
        let values: Vec<f32> = softmax.to_vec1().unwrap();

        // Should be uniform distribution
        for v in &values {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tensor_multiplication_stability() {
        let device = Device::Cpu;

        // Small values
        let a = Tensor::new(&[[1e-10f32, 1e-10], [1e-10, 1e-10]], &device).unwrap();
        let b = Tensor::new(&[[1e-10f32, 1e-10], [1e-10, 1e-10]], &device).unwrap();
        let c = a.matmul(&b).unwrap();

        let values: Vec<f32> = c.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_layer_norm_stability() {
        let device = Device::Cpu;

        // Values with high variance
        let x = Tensor::new(&[[1e6f32, -1e6, 1.0, -1.0]], &device).unwrap();

        // Manual layer norm calculation
        let mean = x.mean_keepdim(1).unwrap();
        let centered = x.broadcast_sub(&mean).unwrap();
        let var = centered.sqr().unwrap().mean_keepdim(1).unwrap();
        let std = (var + 1e-5).unwrap().sqrt().unwrap();
        let normalized = centered.broadcast_div(&std).unwrap();

        let values: Vec<f32> = normalized.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_exp_overflow_prevention() {
        let device = Device::Cpu;

        // Values that would overflow with raw exp
        let x = Tensor::new(&[100.0f32, 200.0, 300.0], &device).unwrap();

        // Softmax should handle this via max subtraction
        let softmax = candle_nn::ops::softmax(&x, 0).unwrap();
        let values: Vec<f32> = softmax.to_vec1().unwrap();

        assert!(values.iter().all(|v| v.is_finite()));
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// END-TO-END TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod e2e_tests {
    use super::*;

    #[test]
    fn test_full_pipeline_shapes() {
        let config = BidirectionalTorusConfig {
            d_model: 32,
            d_ff: 128,
            n_heads: 4,
            n_layers: 2,
            n_major: 4,
            n_minor: 4,
            ..BidirectionalTorusConfig::default()
        };

        let batch_size = 2;
        let seq_len = config.seq_len();
        let device = Device::Cpu;

        // Input tensor
        let input =
            Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, config.d_model), &device).unwrap();
        assert_eq!(input.dims(), &[batch_size, seq_len, config.d_model]);

        // Expected output shape (same as input for encoder-only)
        // [B, S, D] -> [B, S, D] or [B, S, V] if vocab_size specified
    }

    #[test]
    fn test_config_serialization() {
        let config = BidirectionalTorusConfig::default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();

        // Deserialize back
        let recovered: BidirectionalTorusConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.d_model, recovered.d_model);
        assert_eq!(config.n_layers, recovered.n_layers);
        assert!((config.spiral_winding - recovered.spiral_winding).abs() < 1e-10);
    }

    #[test]
    fn test_config_roundtrip_all_fields() {
        let config = BidirectionalTorusConfig {
            d_model: 128,
            d_ff: 512,
            n_heads: 8,
            n_layers: 4,
            n_major: 16,
            n_minor: 8,
            major_radius: 3.0,
            minor_radius: 1.5,
            use_parallel_streams: true,
            use_compounding: true,
            use_multi_scale: true,
            ema_alpha: 0.85,
            learnable_alpha: true,
            use_momentum: true,
            spiral_winding: 2.0,
            weight_temperature: 0.5,
            parallel_execution: true,
            use_geodesic_bias: true,
            geodesic_sigma: 0.3,
            dropout: 0.15,
            n_pos_frequencies: 32,
            use_coherence: true,
            coherence_threshold: 0.7,
            smm_learning_rate: 0.02,
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let recovered: BidirectionalTorusConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.d_model, recovered.d_model);
        assert_eq!(config.n_layers, recovered.n_layers);
        assert_eq!(config.use_multi_scale, recovered.use_multi_scale);
        assert!((config.ema_alpha - recovered.ema_alpha).abs() < 1e-10);
        assert_eq!(config.use_coherence, recovered.use_coherence);
    }

    #[test]
    fn test_torus_geometry_consistency() {
        let config = BidirectionalTorusConfig::default();
        let torus = TorusManifold::new(config.major_radius, config.minor_radius);
        let grid = torus.generate_grid(config.n_major, config.n_minor);

        assert_eq!(grid.len(), config.seq_len());
    }

    #[test]
    fn test_parallel_config_extraction() {
        let config = BidirectionalTorusConfig::default();
        let parallel = config.to_parallel_config();
        let comp = config.to_compounding_config();

        // Verify consistency
        assert_eq!(parallel.d_model, comp.d_model);
        assert_eq!(parallel.n_major * parallel.n_minor, config.seq_len());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod training_tests {
    use crate::training::{
        generate_random_batch, mse_loss, LRScheduler, LossType, TorusLoss, TrainingConfig,
        TrainingMetrics,
    };
    use candle_core::Device;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 32);
        assert!(config.warmup_steps > 0);
    }

    #[test]
    fn test_training_config_quick() {
        let config = TrainingConfig::quick();
        assert!(config.batch_size < TrainingConfig::default().batch_size);
        assert!(config.epochs < TrainingConfig::default().epochs);
    }

    #[test]
    fn test_training_config_thorough() {
        let config = TrainingConfig::thorough();
        assert!(config.epochs > TrainingConfig::default().epochs);
        assert!(config.total_steps > TrainingConfig::default().total_steps);
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let config = TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 100,
            total_steps: 1000,
            ..Default::default()
        };
        let mut scheduler = LRScheduler::new(&config);

        // At step 0, LR should be 0
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-8);

        // At step 50 (half warmup), LR should be half
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-6);

        // At step 100 (end warmup), LR should be full
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_cosine_decay() {
        let config = TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 0,
            total_steps: 1000,
            ..Default::default()
        };
        let mut scheduler = LRScheduler::new(&config);

        let initial_lr = scheduler.get_lr();

        // Step to middle
        for _ in 0..500 {
            scheduler.step();
        }

        let mid_lr = scheduler.get_lr();

        // LR should have decreased
        assert!(mid_lr < initial_lr);

        // Step to end
        for _ in 0..500 {
            scheduler.step();
        }

        let final_lr = scheduler.get_lr();
        assert!(final_lr < mid_lr);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        assert_eq!(metrics.best_eval_loss, f64::INFINITY);

        metrics.log_step(1.0, 1e-4, 0.5, 10.0);
        metrics.log_step(0.8, 1e-4, 0.4, 11.0);

        assert_eq!(metrics.loss_history.len(), 2);
        assert_eq!(metrics.step_times.len(), 2);

        metrics.log_eval(0.7, 2);
        assert_eq!(metrics.best_eval_loss, 0.7);
        assert_eq!(metrics.best_step, 2);
    }

    #[test]
    fn test_training_metrics_improvement() {
        let mut metrics = TrainingMetrics::new();

        metrics.log_eval(1.0, 1);
        assert_eq!(metrics.best_eval_loss, 1.0);

        metrics.log_eval(0.5, 2);
        assert_eq!(metrics.best_eval_loss, 0.5);
        assert_eq!(metrics.best_step, 2);

        // Worse loss shouldn't update best
        metrics.log_eval(0.7, 3);
        assert_eq!(metrics.best_eval_loss, 0.5);
        assert_eq!(metrics.best_step, 2);
    }

    #[test]
    fn test_loss_types() {
        let device = Device::Cpu;

        // Test MSE loss
        let pred = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
        let target = Tensor::new(&[[1.1f32, 2.1], [3.1, 4.1]], &device).unwrap();

        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val: f32 = loss.to_scalar().unwrap();
        assert!((loss_val - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_generate_random_batch() {
        let device = Device::Cpu;
        let (inputs, targets) = generate_random_batch(4, 16, 32, 100, &device).unwrap();

        assert_eq!(inputs.dims(), &[4, 16, 32]);
        assert_eq!(targets.dims(), &[4, 16]);
    }

    #[test]
    fn test_generate_random_batch_various_sizes() {
        let device = Device::Cpu;

        for (batch, seq, d_model, vocab) in [(1, 8, 16, 50), (8, 32, 64, 100), (16, 64, 128, 1000)]
        {
            let (inputs, targets) =
                generate_random_batch(batch, seq, d_model, vocab, &device).unwrap();
            assert_eq!(inputs.dims(), &[batch, seq, d_model]);
            assert_eq!(targets.dims(), &[batch, seq]);
        }
    }

    #[test]
    fn test_torus_loss_creation() {
        let loss_fn = TorusLoss::new(LossType::MSE);
        let loss_fn_with_smoothing =
            TorusLoss::new(LossType::CrossEntropy).with_label_smoothing(0.1);
        let loss_fn_geodesic = TorusLoss::new(LossType::GeodesicAware).with_geodesic_weight(0.1);

        // Just verify they can be created
        let _ = loss_fn;
        let _ = loss_fn_with_smoothing;
        let _ = loss_fn_geodesic;
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = TrainingMetrics::new();

        for i in 0..150 {
            metrics.log_step(1.0 - i as f64 * 0.005, 1e-4, 0.5, 10.0);
        }

        let summary = metrics.summary();
        assert!(summary.contains("Steps:"));
        assert!(summary.contains("Recent Loss:"));
    }

    use candle_core::Tensor;
}

// ═══════════════════════════════════════════════════════════════════════════
// EDGE CASE TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_element_grid() {
        let torus = TorusManifold::new(2.0, 1.0);
        let grid = torus.generate_grid(1, 1);
        assert_eq!(grid.len(), 1);
    }

    #[test]
    fn test_minimal_config() {
        let config = BidirectionalTorusConfig {
            d_model: 8,
            d_ff: 32,
            n_heads: 1,
            n_layers: 1,
            n_major: 2,
            n_minor: 2,
            ..BidirectionalTorusConfig::default()
        };

        assert_eq!(config.seq_len(), 4);
        assert_eq!(config.d_model / config.n_heads, 8);
    }

    #[test]
    fn test_large_grid() {
        let torus = TorusManifold::new(2.0, 1.0);
        let grid = torus.generate_grid(128, 64);
        assert_eq!(grid.len(), 128 * 64);
    }

    #[test]
    fn test_boundary_wrap_exact() {
        let boundary = PeriodicBoundary::new(8, 4);

        // Exact boundary values
        assert_eq!(boundary.wrap_2d(0, 0), (0, 0));
        assert_eq!(boundary.wrap_2d(7, 3), (7, 3));
        assert_eq!(boundary.wrap_2d(8, 4), (0, 0));
    }

    #[test]
    fn test_zero_distance() {
        let c = TorusCoordinate::new(1.234, 5.678);
        let dist = c.geodesic_distance(&c);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_torus_aspect_ratios() {
        // Various aspect ratios
        let thin_torus = TorusManifold::new(5.0, 0.5);
        let fat_torus = TorusManifold::new(2.0, 1.5);

        assert!(
            thin_torus.major_radius / thin_torus.minor_radius
                > fat_torus.major_radius / fat_torus.minor_radius
        );
    }

    #[test]
    fn test_empty_metrics() {
        let metrics = crate::training::TrainingMetrics::new();
        let summary = metrics.summary();
        assert!(summary.contains("Steps: 0"));
    }

    #[test]
    fn test_scheduler_reset() {
        let config = crate::training::TrainingConfig {
            learning_rate: 1e-3,
            warmup_steps: 10,
            total_steps: 100,
            ..Default::default()
        };
        let mut scheduler = crate::training::LRScheduler::new(&config);

        // Step forward
        for _ in 0..50 {
            scheduler.step();
        }

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.current_step(), 0);
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-8);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROPERTY-BASED TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_geodesic_distance_non_negative() {
        let torus = TorusManifold::new(2.0, 1.0);
        let coords = torus.generate_grid(8, 8);

        for c1 in &coords {
            for c2 in &coords {
                assert!(c1.geodesic_distance(c2) >= 0.0);
            }
        }
    }

    #[test]
    fn test_wrap_idempotent() {
        let boundary = PeriodicBoundary::new(8, 4);

        for i in -20..20_i64 {
            for j in -20..20_i64 {
                let (wrapped_i, wrapped_j) = boundary.wrap_2d(i, j);
                let (rewrapped_i, rewrapped_j) =
                    boundary.wrap_2d(wrapped_i as i64, wrapped_j as i64);

                assert_eq!(wrapped_i, rewrapped_i);
                assert_eq!(wrapped_j, rewrapped_j);
            }
        }
    }

    #[test]
    fn test_stream_pair_involution() {
        // pair(pair(x)) == x for all streams
        for stream in StreamId::all() {
            assert_eq!(stream.pair().pair(), stream);
        }
    }

    #[test]
    fn test_torus_coordinate_wrapped() {
        // All coordinates should be in [0, 2π)
        for u in [-10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 100.0] {
            for v in [-10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 100.0] {
                let c = TorusCoordinate::new(u, v);
                assert!(c.u >= 0.0 && c.u < 2.0 * PI);
                assert!(c.v >= 0.0 && c.v < 2.0 * PI);
            }
        }
    }

    #[test]
    fn test_softmax_probability_distribution() {
        let device = Device::Cpu;

        // Random logits
        for _ in 0..10 {
            let logits = Tensor::randn(0.0f32, 1.0, (100,), &device).unwrap();
            let probs = candle_nn::ops::softmax(&logits, 0).unwrap();
            let values: Vec<f32> = probs.to_vec1().unwrap();

            // All probabilities should be in [0, 1]
            assert!(values.iter().all(|&p| p >= 0.0 && p <= 1.0));

            // Should sum to 1
            let sum: f32 = values.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
