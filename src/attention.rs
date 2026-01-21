//! Torus Attention Mechanism
//!
//! Core transformer-style attention adapted for torus manifold topology.
//! Combines periodic position encodings, geodesic distance-based attention,
//! and dual-loop information flow.

use crate::dual_loop::{DualLoopConfig, DualLoopFlow};
use crate::geometry::{TorusDistanceMatrix, TorusManifold};
use crate::periodic::PeriodicBoundary;
use crate::rmsnorm::{rms_norm, RmsNorm};
use crate::TorusResult;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for torus attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorusAttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Grid size in major (u) direction
    pub n_major: usize,
    /// Grid size in minor (v) direction
    pub n_minor: usize,
    /// Major radius of torus
    pub major_radius: f64,
    /// Minor radius of torus
    pub minor_radius: f64,
    /// Attention temperature (sigma for geodesic kernel)
    pub attention_sigma: f64,
    /// Whether to use geodesic distance bias
    pub use_geodesic_bias: bool,
    /// Whether to use vortex dynamics
    pub use_vortex: bool,
    /// Vortex winding number
    pub vortex_winding: f64,
    /// Dropout probability
    pub dropout: f64,
    /// Number of frequencies for position encoding
    pub n_pos_frequencies: usize,
}

impl Default for TorusAttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            n_major: 32,
            n_minor: 16,
            major_radius: 2.0,
            minor_radius: 1.0,
            attention_sigma: 0.5,
            use_geodesic_bias: true,
            use_vortex: true,
            vortex_winding: 1.0,
            dropout: 0.1,
            n_pos_frequencies: 16,
        }
    }
}

/// Main Torus Attention layer
#[derive(Debug)]
pub struct TorusAttention {
    /// Query projection
    query: Linear,
    /// Key projection
    key: Linear,
    /// Value projection
    value: Linear,
    /// Output projection
    output: Linear,
    /// Layer normalization (RMSNorm for Metal compatibility)
    norm: RmsNorm,
    /// Position encodings
    position_encodings: Tensor,
    /// Geodesic attention bias
    geodesic_bias: Option<Tensor>,
    /// Dual-loop attention (optional)
    dual_loop: Option<DualLoopFlow>,
    /// Configuration
    config: TorusAttentionConfig,
    /// Scale factor
    scale: f64,
}

impl TorusAttention {
    /// Create a new torus attention layer
    pub fn new(config: TorusAttentionConfig, vb: VarBuilder, device: &Device) -> TorusResult<Self> {
        let head_dim = config.d_model / config.n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Linear projections
        let query = candle_nn::linear(config.d_model, config.d_model, vb.pp("query"))?;
        let key = candle_nn::linear(config.d_model, config.d_model, vb.pp("key"))?;
        let value = candle_nn::linear(config.d_model, config.d_model, vb.pp("value"))?;
        let output = candle_nn::linear(config.d_model, config.d_model, vb.pp("output"))?;

        // Layer norm (RMSNorm)
        let norm = rms_norm(config.d_model, 1e-5, vb.pp("norm"))?;

        // Generate torus position encodings
        let boundary = PeriodicBoundary::new(config.n_major, config.n_minor);
        let position_encodings = boundary.position_encodings(config.n_pos_frequencies, device)?;

        // Generate geodesic distance bias if enabled
        let geodesic_bias = if config.use_geodesic_bias {
            Some(Self::compute_geodesic_bias(&config, device)?)
        } else {
            None
        };

        // Optional dual-loop attention
        let dual_loop = if config.use_vortex {
            let dl_config = DualLoopConfig {
                d_model: config.d_model,
                n_heads_major: config.n_heads / 2,
                n_heads_minor: config.n_heads / 2,
                n_major: config.n_major,
                n_minor: config.n_minor,
                dropout: config.dropout,
                cross_loop: true,
            };
            Some(DualLoopFlow::new(dl_config, vb.pp("dual_loop"))?)
        } else {
            None
        };

        Ok(Self {
            query,
            key,
            value,
            output,
            norm,
            position_encodings,
            geodesic_bias,
            dual_loop,
            config,
            scale,
        })
    }

    /// Compute geodesic distance-based attention bias
    fn compute_geodesic_bias(
        config: &TorusAttentionConfig,
        device: &Device,
    ) -> TorusResult<Tensor> {
        let torus = TorusManifold::new(config.major_radius, config.minor_radius);
        let coords = torus.generate_grid(config.n_major, config.n_minor);
        let distance_matrix = TorusDistanceMatrix::from_coordinates(&coords);

        let n = config.n_major * config.n_minor;
        let neg_inv_2sigma2 = -0.5 / (config.attention_sigma * config.attention_sigma);

        let mut bias_data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = distance_matrix.distances[i][j];
                bias_data[i * n + j] = (d * d * neg_inv_2sigma2).exp() as f32;
            }
        }

        let bias = Tensor::from_vec(bias_data, (1, 1, n, n), device)?;
        Ok(bias)
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let n_heads = self.config.n_heads;
        let head_dim = d_model / n_heads;

        // Add position encodings
        let _pos_enc = self.position_encodings.broadcast_as((
            batch_size,
            seq_len,
            self.position_encodings.dims()[1],
        ))?;

        // If position encoding dimension differs from d_model, we need to project or slice
        // For simplicity, we'll add after projecting x to include spatial information

        // Project Q, K, V
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention: [B, S, D] -> [B, H, S, D/H]
        let q = q
            .reshape((batch_size, seq_len, n_heads, head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, n_heads, head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, n_heads, head_dim))?
            .transpose(1, 2)?;

        // Attention scores: Q @ K^T * scale
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;

        // Add geodesic bias if available
        let scores = if let Some(ref bias) = self.geodesic_bias {
            // Broadcast bias across batch and heads
            let bias_broadcast = bias.broadcast_as(scores.dims())?;
            // Log bias acts as additive attention bias
            let log_bias = bias_broadcast.log()?;
            (scores + log_bias)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Apply to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [B, H, S, D/H] -> [B, S, D]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, d_model))?;

        // Output projection
        let output = self.output.forward(&attn_output)?;

        // Residual connection + norm
        let output = self.norm.forward(&(x + output)?)?;

        // Optional dual-loop processing
        let output = if let Some(ref dual_loop) = self.dual_loop {
            let dl_out = dual_loop.forward(&output, device)?;
            (output + dl_out)?
        } else {
            output
        };

        Ok(output)
    }

    /// Get the position encodings
    pub fn get_position_encodings(&self) -> &Tensor {
        &self.position_encodings
    }

    /// Get the geodesic bias matrix
    pub fn get_geodesic_bias(&self) -> Option<&Tensor> {
        self.geodesic_bias.as_ref()
    }
}

/// Feed-forward network for transformer block
#[derive(Debug)]
pub struct TorusFeedForward {
    linear1: Linear,
    linear2: Linear,
    norm: RmsNorm,
    activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    GELU,
    SiLU,
}

impl TorusFeedForward {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;
        let norm = rms_norm(d_model, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            linear1,
            linear2,
            norm,
            activation,
        })
    }

    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        let h = self.linear1.forward(x)?;

        let h = match self.activation {
            Activation::ReLU => h.relu()?,
            Activation::GELU => h.gelu_erf()?,
            Activation::SiLU => candle_nn::ops::silu(&h)?,
        };

        let h = self.linear2.forward(&h)?;
        let output = self.norm.forward(&(x + h)?)?;
        Ok(output)
    }
}

/// Complete Torus Transformer block
#[derive(Debug)]
pub struct TorusTransformerBlock {
    attention: TorusAttention,
    feed_forward: TorusFeedForward,
}

impl TorusTransformerBlock {
    pub fn new(
        config: TorusAttentionConfig,
        d_ff: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let attention = TorusAttention::new(config.clone(), vb.pp("attention"), device)?;
        let feed_forward =
            TorusFeedForward::new(config.d_model, d_ff, Activation::GELU, vb.pp("ff"))?;

        Ok(Self {
            attention,
            feed_forward,
        })
    }

    pub fn forward(&self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
        let attn_out = self.attention.forward(x, device)?;
        let output = self.feed_forward.forward(&attn_out)?;
        Ok(output)
    }
}

/// Stack of Torus Transformer blocks
#[derive(Debug)]
pub struct TorusTransformer {
    blocks: Vec<TorusTransformerBlock>,
    embedding: Linear,
    output_proj: Linear,
    #[allow(dead_code)]
    config: TorusAttentionConfig,
}

impl TorusTransformer {
    pub fn new(
        config: TorusAttentionConfig,
        n_layers: usize,
        d_ff: usize,
        vocab_size: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let embedding = candle_nn::linear(vocab_size, config.d_model, vb.pp("embedding"))?;
        let output_proj = candle_nn::linear(config.d_model, vocab_size, vb.pp("output"))?;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let block = TorusTransformerBlock::new(
                config.clone(),
                d_ff,
                vb.pp(format!("block_{}", i)),
                device,
            )?;
            blocks.push(block);
        }

        Ok(Self {
            blocks,
            embedding,
            output_proj,
            config,
        })
    }

    pub fn forward(&self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
        // Embed input
        let mut h = self.embedding.forward(x)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            h = block.forward(&h, device)?;
        }

        // Output projection
        let output = self.output_proj.forward(&h)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TorusAttentionConfig::default();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_heads, 8);
        assert!(config.use_geodesic_bias);
    }
}
