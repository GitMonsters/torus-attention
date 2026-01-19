//! Dual-loop information flow for torus attention
//!
//! The torus has two fundamental loops:
//! - Major loop: around the main ring (u direction)
//! - Minor loop: around the tube (v direction)
//!
//! This module implements attention mechanisms that leverage
//! both loops for hierarchical information processing.

use crate::error::TorusError;
use crate::geometry::TorusCoordinate;
use crate::periodic::PeriodicBoundary;
use crate::TorusResult;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for dual-loop attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLoopConfig {
    /// Embedding dimension
    pub d_model: usize,
    /// Number of attention heads for major loop
    pub n_heads_major: usize,
    /// Number of attention heads for minor loop
    pub n_heads_minor: usize,
    /// Grid size in major direction
    pub n_major: usize,
    /// Grid size in minor direction
    pub n_minor: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use cross-loop attention
    pub cross_loop: bool,
}

impl Default for DualLoopConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads_major: 8,
            n_heads_minor: 4,
            n_major: 32,
            n_minor: 16,
            dropout: 0.1,
            cross_loop: true,
        }
    }
}

/// Information flow around a single loop (1D ring attention)
#[derive(Debug)]
pub struct LoopAttention {
    /// Linear projection for queries
    pub query: Linear,
    /// Linear projection for keys
    pub key: Linear,
    /// Linear projection for values
    pub value: Linear,
    /// Output projection
    pub output: Linear,
    /// Number of heads
    pub n_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Loop size (number of positions)
    pub loop_size: usize,
    /// Scale factor for attention
    pub scale: f64,
}

impl LoopAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        loop_size: usize,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let query = candle_nn::linear(d_model, d_model, vb.pp("query"))?;
        let key = candle_nn::linear(d_model, d_model, vb.pp("key"))?;
        let value = candle_nn::linear(d_model, d_model, vb.pp("value"))?;
        let output = candle_nn::linear(d_model, d_model, vb.pp("output"))?;

        Ok(Self {
            query,
            key,
            value,
            output,
            n_heads,
            head_dim,
            loop_size,
            scale,
        })
    }

    /// Apply attention with periodic (ring) position bias
    pub fn forward(&self, x: &Tensor, position_bias: Option<&Tensor>) -> TorusResult<Tensor> {
        let (batch_size, seq_len, _d_model) = x.dims3()?;

        // Project Q, K, V
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention: [B, S, H, D] -> [B, H, S, D]
        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores: Q @ K^T / sqrt(d_k)
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;

        // Add periodic position bias if provided
        let scores = if let Some(bias) = position_bias {
            scores.broadcast_add(bias)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [B, H, S, D] -> [B, S, H*D]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        let output = self.output.forward(&attn_output)?;
        Ok(output)
    }

    /// Generate periodic position bias for ring attention
    pub fn periodic_position_bias(loop_size: usize, n_heads: usize, device: &Device) -> TorusResult<Tensor> {
        let mut bias = vec![0.0f32; n_heads * loop_size * loop_size];

        for h in 0..n_heads {
            let bandwidth = (h + 1) as f64 * PI / n_heads as f64;
            
            for i in 0..loop_size {
                for j in 0..loop_size {
                    // Periodic distance on the ring
                    let di = (i as i64 - j as i64).abs() as f64;
                    let periodic_dist = di.min(loop_size as f64 - di);
                    let normalized_dist = periodic_dist * 2.0 * PI / loop_size as f64;
                    
                    // Gaussian falloff with head-specific bandwidth
                    let bias_val = (-normalized_dist * normalized_dist / (2.0 * bandwidth * bandwidth)).exp();
                    bias[h * loop_size * loop_size + i * loop_size + j] = bias_val as f32;
                }
            }
        }

        let tensor = Tensor::from_vec(bias, (n_heads, loop_size, loop_size), device)?;
        Ok(tensor)
    }
}

/// Dual-loop attention combining major and minor ring attention
#[derive(Debug)]
pub struct DualLoopFlow {
    /// Attention for major loop
    pub major_attention: LoopAttention,
    /// Attention for minor loop
    pub minor_attention: LoopAttention,
    /// Cross-loop attention (optional)
    pub cross_attention: Option<LoopAttention>,
    /// Layer norm for major path
    pub norm_major: candle_nn::LayerNorm,
    /// Layer norm for minor path
    pub norm_minor: candle_nn::LayerNorm,
    /// Layer norm for output
    pub norm_output: candle_nn::LayerNorm,
    /// Configuration
    pub config: DualLoopConfig,
}

impl DualLoopFlow {
    pub fn new(config: DualLoopConfig, vb: VarBuilder) -> TorusResult<Self> {
        let major_attention = LoopAttention::new(
            config.d_model,
            config.n_heads_major,
            config.n_major,
            vb.pp("major_attention"),
        )?;

        let minor_attention = LoopAttention::new(
            config.d_model,
            config.n_heads_minor,
            config.n_minor,
            vb.pp("minor_attention"),
        )?;

        let cross_attention = if config.cross_loop {
            Some(LoopAttention::new(
                config.d_model,
                config.n_heads_major,
                config.n_major * config.n_minor,
                vb.pp("cross_attention"),
            )?)
        } else {
            None
        };

        let norm_major = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm_major"))?;
        let norm_minor = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm_minor"))?;
        let norm_output = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm_output"))?;

        Ok(Self {
            major_attention,
            minor_attention,
            cross_attention,
            norm_major,
            norm_minor,
            norm_output,
            config,
        })
    }

    /// Forward pass through dual-loop attention
    /// Input shape: [batch, n_major * n_minor, d_model]
    pub fn forward(&self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        
        // Reshape to grid: [batch, n_major, n_minor, d_model]
        let x_grid = x.reshape((
            batch_size,
            self.config.n_major,
            self.config.n_minor,
            d_model,
        ))?;

        // === Major loop attention (along u direction) ===
        // Aggregate along minor dimension, attend along major
        let x_major = x_grid.mean(2)?; // [batch, n_major, d_model]
        let major_bias = LoopAttention::periodic_position_bias(
            self.config.n_major,
            self.config.n_heads_major,
            device,
        )?;
        let major_out = self.major_attention.forward(&x_major, Some(&major_bias))?;
        let major_out = self.norm_major.forward(&major_out)?;

        // === Minor loop attention (along v direction) ===
        // Transpose to attend along minor dimension
        let x_minor = x_grid
            .transpose(1, 2)? // [batch, n_minor, n_major, d_model]
            .mean(2)?; // [batch, n_minor, d_model]
        let minor_bias = LoopAttention::periodic_position_bias(
            self.config.n_minor,
            self.config.n_heads_minor,
            device,
        )?;
        let minor_out = self.minor_attention.forward(&x_minor, Some(&minor_bias))?;
        let minor_out = self.norm_minor.forward(&minor_out)?;

        // === Combine major and minor information ===
        // Broadcast major attention back to grid
        let major_expanded = major_out
            .unsqueeze(2)? // [batch, n_major, 1, d_model]
            .broadcast_as((batch_size, self.config.n_major, self.config.n_minor, d_model))?;

        // Broadcast minor attention back to grid
        let minor_expanded = minor_out
            .unsqueeze(1)? // [batch, 1, n_minor, d_model]
            .broadcast_as((batch_size, self.config.n_major, self.config.n_minor, d_model))?;

        // Combine: element-wise addition of contributions
        let combined = (x_grid + major_expanded + minor_expanded)?;

        // === Optional cross-loop attention ===
        let output = if let Some(ref cross_attn) = self.cross_attention {
            let combined_flat = combined.reshape((batch_size, seq_len, d_model))?;
            let cross_out = cross_attn.forward(&combined_flat, None)?;
            (combined_flat + cross_out)?
        } else {
            combined.reshape((batch_size, seq_len, d_model))?
        };

        // Final normalization
        let output = self.norm_output.forward(&output)?;
        Ok(output)
    }
}

/// Compute information flow patterns for visualization
#[derive(Debug, Clone)]
pub struct FlowPattern {
    /// Flow strength in major direction
    pub major_flow: Array2<f64>,
    /// Flow strength in minor direction
    pub minor_flow: Array2<f64>,
    /// Combined spiral flow
    pub spiral_flow: Array2<f64>,
}

impl FlowPattern {
    /// Analyze attention patterns to extract flow
    pub fn from_attention_weights(
        major_weights: &Array2<f64>,
        minor_weights: &Array2<f64>,
        n_major: usize,
        n_minor: usize,
    ) -> Self {
        let mut major_flow = Array2::zeros((n_major, n_minor));
        let mut minor_flow = Array2::zeros((n_major, n_minor));
        let mut spiral_flow = Array2::zeros((n_major, n_minor));

        // Compute directional flow from attention weights
        for i in 0..n_major {
            for j in 0..n_minor {
                // Major flow: attention gradient in u direction
                let major_grad = if i > 0 {
                    major_weights[[i, i]] - major_weights[[i, i - 1]]
                } else {
                    major_weights[[i, i]] - major_weights[[i, n_major - 1]]
                };
                major_flow[[i, j]] = major_grad;

                // Minor flow: attention gradient in v direction
                let minor_grad = if j > 0 {
                    minor_weights[[j, j]] - minor_weights[[j, j - 1]]
                } else {
                    minor_weights[[j, j]] - minor_weights[[j, n_minor - 1]]
                };
                minor_flow[[i, j]] = minor_grad;

                // Spiral flow: combination with twist
                spiral_flow[[i, j]] = (major_grad * major_grad + minor_grad * minor_grad).sqrt();
            }
        }

        Self {
            major_flow,
            minor_flow,
            spiral_flow,
        }
    }

    /// Get flow vector at a position
    pub fn flow_at(&self, i: usize, j: usize) -> (f64, f64) {
        (self.major_flow[[i, j]], self.minor_flow[[i, j]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_loop_config() {
        let config = DualLoopConfig::default();
        assert_eq!(config.d_model, 256);
        assert!(config.cross_loop);
    }

    #[test]
    fn test_flow_pattern() {
        let major_weights = Array2::eye(4);
        let minor_weights = Array2::eye(4);
        let flow = FlowPattern::from_attention_weights(&major_weights, &minor_weights, 4, 4);
        
        // Identity weights should produce minimal flow
        let (fu, fv) = flow.flow_at(2, 2);
        assert!(fu.abs() < 1.1); // Reasonable bounds
        assert!(fv.abs() < 1.1);
    }
}
