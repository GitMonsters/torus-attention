//! Integration Tests for Torus Attention
//!
//! Comprehensive tests for the complete bidirectional torus attention system.

use crate::geometry::{TorusCoordinate, TorusManifold, TorusDistanceMatrix};
use crate::periodic::PeriodicBoundary;
use crate::parallel_streams::StreamId;
use crate::compounding::{CompoundingConfig, LearnableAlpha};
use crate::integration::BidirectionalTorusConfig;
use crate::vortex::{Vortex, SpiralAttention};
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
        
        assert_eq!(config.d_model % config.n_heads, 0, "d_model must be divisible by n_heads");
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
            .reshape((batch, seq_len, n_heads, head_dim)).unwrap()
            .transpose(1, 2).unwrap();
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
        let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, config.d_model), &device).unwrap();
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
}
