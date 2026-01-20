//! Unified Bidirectional Torus Attention Integration
//!
//! Combines all components into a cohesive bidirectional parallel
//! compounding attention system:
//!
//! Input → 8-Stream Parallel → Symmetric Combine → EMA Compound → Output
//!
//! Features:
//! - 8 parallel processing streams (major/minor forward/backward, spiral CW/CCW, cross U↔V)
//! - Symmetric bidirectional combination with learned weights
//! - Multi-layer EMA compounding with learnable α per layer
//! - Torus-aware position encodings and geodesic biases
//! - Full integration with the original torus attention system

use crate::attention::{Activation, TorusFeedForward};
use crate::bidirectional::TorusBidirectionalEncoding;
use crate::compounding::{CompoundingConfig, EMACompounding, MultiScaleCompounding};
use crate::geometry::{TorusDistanceMatrix, TorusManifold};
use crate::parallel_streams::{ParallelStreamConfig, ParallelStreamProcessor, StreamId};
use crate::TorusResult;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for the unified bidirectional system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalTorusConfig {
    /// Model dimension
    pub d_model: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Number of attention heads per stream
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Grid size in major direction
    pub n_major: usize,
    /// Grid size in minor direction
    pub n_minor: usize,
    /// Major radius of torus
    pub major_radius: f64,
    /// Minor radius of torus
    pub minor_radius: f64,
    /// Whether to use 8-stream parallel processing
    pub use_parallel_streams: bool,
    /// Whether to use EMA compounding
    pub use_compounding: bool,
    /// Whether to use multi-scale compounding
    pub use_multi_scale: bool,
    /// Base EMA alpha
    pub ema_alpha: f64,
    /// Whether EMA alpha is learnable
    pub learnable_alpha: bool,
    /// Whether to use momentum in compounding
    pub use_momentum: bool,
    /// Spiral winding number for spiral streams
    pub spiral_winding: f64,
    /// Temperature for weight softmax
    pub weight_temperature: f64,
    /// Whether to run streams in parallel (rayon)
    pub parallel_execution: bool,
    /// Whether to use geodesic distance bias
    pub use_geodesic_bias: bool,
    /// Sigma for geodesic attention kernel
    pub geodesic_sigma: f64,
    /// Dropout probability
    pub dropout: f64,
    /// Number of position encoding frequencies
    pub n_pos_frequencies: usize,
}

impl Default for BidirectionalTorusConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            d_ff: 1024,
            n_heads: 8,
            n_layers: 6,
            n_major: 32,
            n_minor: 16,
            major_radius: 2.0,
            minor_radius: 1.0,
            use_parallel_streams: true,
            use_compounding: true,
            use_multi_scale: false,
            ema_alpha: 0.9,
            learnable_alpha: true,
            use_momentum: true,
            spiral_winding: 1.618033988749895, // Golden ratio
            weight_temperature: 1.0,
            parallel_execution: true,
            use_geodesic_bias: true,
            geodesic_sigma: 0.5,
            dropout: 0.1,
            n_pos_frequencies: 16,
        }
    }
}

impl BidirectionalTorusConfig {
    /// Get sequence length (n_major * n_minor)
    pub fn seq_len(&self) -> usize {
        self.n_major * self.n_minor
    }

    /// Convert to parallel stream config
    pub fn to_parallel_config(&self) -> ParallelStreamConfig {
        ParallelStreamConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_major: self.n_major,
            n_minor: self.n_minor,
            spiral_winding: self.spiral_winding,
            weight_temperature: self.weight_temperature,
            parallel: self.parallel_execution,
            dropout: self.dropout,
        }
    }

    /// Convert to compounding config
    pub fn to_compounding_config(&self) -> CompoundingConfig {
        CompoundingConfig {
            n_layers: self.n_layers,
            d_model: self.d_model,
            base_alpha: self.ema_alpha,
            min_alpha: 0.1,
            max_alpha: 0.99,
            layer_scale: 0.95,
            use_momentum: self.use_momentum,
            momentum_beta: 0.9,
            learnable_alpha: self.learnable_alpha,
        }
    }
}

/// Single layer of the bidirectional torus transformer
#[derive(Debug)]
pub struct BidirectionalTorusLayer {
    /// 8-stream parallel processor
    parallel_streams: ParallelStreamProcessor,
    /// Feed-forward network
    feed_forward: TorusFeedForward,
    /// Pre-attention layer norm
    pre_norm: candle_nn::LayerNorm,
    /// Post-attention layer norm
    post_attn_norm: candle_nn::LayerNorm,
    /// Layer index
    #[allow(dead_code)]
    layer_idx: usize,
    /// Configuration
    #[allow(dead_code)]
    config: BidirectionalTorusConfig,
}

impl BidirectionalTorusLayer {
    pub fn new(
        layer_idx: usize,
        config: BidirectionalTorusConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let parallel_streams = ParallelStreamProcessor::new(
            config.to_parallel_config(),
            vb.pp("parallel"),
            device,
        )?;

        let feed_forward = TorusFeedForward::new(
            config.d_model,
            config.d_ff,
            Activation::GELU,
            vb.pp("ff"),
        )?;

        let pre_norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("pre_norm"))?;
        let post_attn_norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("post_attn_norm"))?;

        Ok(Self {
            parallel_streams,
            feed_forward,
            pre_norm,
            post_attn_norm,
            layer_idx,
            config,
        })
    }

    /// Forward pass through the layer
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Pre-norm
        let x_norm = self.pre_norm.forward(x)?;

        // 8-stream parallel attention
        let attn_out = self.parallel_streams.forward(&x_norm)?;

        // Residual
        let x = (x + attn_out)?;

        // Post-attention norm
        let x_norm = self.post_attn_norm.forward(&x)?;

        // Feed-forward
        let ff_out = self.feed_forward.forward(&x_norm)?;

        // Final residual
        let output = (x + ff_out)?;

        Ok(output)
    }

    /// Get current stream weights
    pub fn get_stream_weights(&self) -> TorusResult<Vec<(StreamId, f32)>> {
        self.parallel_streams.get_weights_named()
    }
}

/// Complete bidirectional torus transformer with compounding
#[derive(Debug)]
pub struct BidirectionalTorusTransformer {
    /// Input embedding (if needed)
    embedding: Option<Linear>,
    /// Bidirectional position encodings
    position_encodings: TorusBidirectionalEncoding,
    /// Geodesic distance bias
    #[allow(dead_code)]
    geodesic_bias: Option<Tensor>,
    /// Transformer layers
    layers: Vec<BidirectionalTorusLayer>,
    /// EMA compounding (single or multi-scale)
    compounding: Option<EMACompounding>,
    /// Multi-scale compounding (alternative)
    multi_scale_compounding: Option<MultiScaleCompounding>,
    /// Output projection
    output_proj: Linear,
    /// Final layer norm
    final_norm: candle_nn::LayerNorm,
    /// Configuration
    config: BidirectionalTorusConfig,
    /// Device
    #[allow(dead_code)]
    device: Device,
}

impl BidirectionalTorusTransformer {
    /// Create a new bidirectional torus transformer
    pub fn new(
        config: BidirectionalTorusConfig,
        vocab_size: Option<usize>,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        // Optional embedding layer
        let embedding = vocab_size.map(|vs| {
            candle_nn::linear(vs, config.d_model, vb.pp("embedding"))
        }).transpose()?;

        // Bidirectional position encodings
        let position_encodings = TorusBidirectionalEncoding::new(
            config.n_major,
            config.n_minor,
            config.d_model,
            config.n_pos_frequencies,
            device,
        )?;

        // Geodesic bias
        let geodesic_bias = if config.use_geodesic_bias {
            Some(Self::compute_geodesic_bias(&config, device)?)
        } else {
            None
        };

        // Build layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            let layer = BidirectionalTorusLayer::new(
                l,
                config.clone(),
                vb.pp(format!("layer_{}", l)),
                device,
            )?;
            layers.push(layer);
        }

        // Compounding
        let (compounding, multi_scale_compounding) = if config.use_compounding {
            if config.use_multi_scale {
                (
                    None,
                    Some(MultiScaleCompounding::new(
                        config.to_compounding_config(),
                        vb.pp("multi_scale_compounding"),
                        device,
                    )?),
                )
            } else {
                (
                    Some(EMACompounding::new(
                        config.to_compounding_config(),
                        vb.pp("compounding"),
                        device,
                    )?),
                    None,
                )
            }
        } else {
            (None, None)
        };

        // Output projection
        let output_proj = candle_nn::linear(
            config.d_model,
            vocab_size.unwrap_or(config.d_model),
            vb.pp("output"),
        )?;

        let final_norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("final_norm"))?;

        Ok(Self {
            embedding,
            position_encodings,
            geodesic_bias,
            layers,
            compounding,
            multi_scale_compounding,
            output_proj,
            final_norm,
            config,
            device: device.clone(),
        })
    }

    /// Compute geodesic distance bias matrix
    fn compute_geodesic_bias(config: &BidirectionalTorusConfig, device: &Device) -> TorusResult<Tensor> {
        let torus = TorusManifold::new(config.major_radius, config.minor_radius);
        let coords = torus.generate_grid(config.n_major, config.n_minor);
        let dist_matrix = TorusDistanceMatrix::from_coordinates(&coords);

        let n = config.seq_len();
        let neg_inv_2sigma2 = -0.5 / (config.geodesic_sigma * config.geodesic_sigma);

        let mut bias = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = dist_matrix.distances[i][j];
                bias[i * n + j] = (d * d * neg_inv_2sigma2).exp() as f32;
            }
        }

        Ok(Tensor::from_vec(bias, (1, 1, n, n), device)?)
    }

    /// Forward pass through the entire transformer
    pub fn forward(&mut self, x: &Tensor) -> TorusResult<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Embed if needed
        let mut h = if let Some(ref emb) = self.embedding {
            emb.forward(x)?
        } else {
            x.clone()
        };

        // Add position encodings
        let pos_enc = self.position_encodings.get_2d();
        let pos_enc_broadcast = pos_enc.unsqueeze(0)?.broadcast_as((batch_size, seq_len, self.config.d_model))?;
        h = (h + pos_enc_broadcast)?;

        // Process through layers with compounding
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_out = layer.forward(&h)?;

            // Apply compounding
            h = if let Some(ref mut comp) = self.compounding {
                comp.compound(layer_idx, &layer_out)?
            } else if let Some(ref mut multi) = self.multi_scale_compounding {
                multi.compound(layer_idx, &layer_out)?
            } else {
                layer_out
            };
        }

        // Final norm
        h = self.final_norm.forward(&h)?;

        // Output projection
        let output = self.output_proj.forward(&h)?;

        Ok(output)
    }

    /// Reset compounding state (for new sequences)
    pub fn reset_state(&mut self) -> TorusResult<()> {
        if let Some(ref mut comp) = self.compounding {
            comp.reset()?;
        }
        if let Some(ref mut multi) = self.multi_scale_compounding {
            multi.reset()?;
        }
        Ok(())
    }

    /// Get stream weights for all layers
    pub fn get_all_stream_weights(&self) -> TorusResult<Vec<Vec<(StreamId, f32)>>> {
        self.layers
            .iter()
            .map(|l| l.get_stream_weights())
            .collect()
    }

    /// Get compounding alphas
    pub fn get_compounding_alphas(&self) -> TorusResult<Option<Vec<f64>>> {
        if let Some(ref comp) = self.compounding {
            Ok(Some(comp.get_all_alphas()?))
        } else {
            Ok(None)
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BidirectionalTorusConfig {
        &self.config
    }

    /// Get position encodings
    pub fn position_encodings(&self) -> &TorusBidirectionalEncoding {
        &self.position_encodings
    }
}

/// Inference-optimized wrapper for the bidirectional transformer
#[derive(Debug)]
pub struct BidirectionalTorusInference {
    /// The transformer
    transformer: BidirectionalTorusTransformer,
    /// Cached key-value pairs per layer (for autoregressive generation)
    kv_cache: Vec<Option<(Tensor, Tensor)>>,
}

impl BidirectionalTorusInference {
    pub fn new(transformer: BidirectionalTorusTransformer) -> Self {
        let n_layers = transformer.config.n_layers;
        Self {
            transformer,
            kv_cache: vec![None; n_layers],
        }
    }

    /// Forward pass with optional KV caching
    pub fn forward(&mut self, x: &Tensor, _use_cache: bool) -> TorusResult<Tensor> {
        // For now, delegate to transformer
        // Full KV caching would require modifications to the attention layers
        self.transformer.forward(x)
    }

    /// Clear KV cache
    pub fn clear_cache(&mut self) {
        for cache in &mut self.kv_cache {
            *cache = None;
        }
    }

    /// Reset all state
    pub fn reset(&mut self) -> TorusResult<()> {
        self.clear_cache();
        self.transformer.reset_state()
    }
}

/// Statistics and diagnostics for the bidirectional system
#[derive(Debug, Clone)]
pub struct BidirectionalStats {
    /// Stream weights per layer
    pub stream_weights: Vec<Vec<(StreamId, f32)>>,
    /// Compounding alphas per layer
    pub compounding_alphas: Vec<f64>,
    /// Multi-scale weights (if applicable)
    pub scale_weights: Option<Vec<f32>>,
}

impl BidirectionalStats {
    /// Collect stats from transformer
    pub fn from_transformer(transformer: &BidirectionalTorusTransformer) -> TorusResult<Self> {
        let stream_weights = transformer.get_all_stream_weights()?;
        let compounding_alphas = transformer.get_compounding_alphas()?.unwrap_or_default();
        
        let scale_weights = if let Some(ref multi) = transformer.multi_scale_compounding {
            Some(multi.get_scale_weights()?)
        } else {
            None
        };

        Ok(Self {
            stream_weights,
            compounding_alphas,
            scale_weights,
        })
    }

    /// Print summary
    pub fn summary(&self) {
        println!("═══ Bidirectional Torus Stats ═══");
        
        println!("\n── Stream Weights ──");
        for (layer_idx, weights) in self.stream_weights.iter().enumerate() {
            println!("Layer {}:", layer_idx);
            for (id, w) in weights {
                println!("  {:15} {:.4}", id.name(), w);
            }
        }

        if !self.compounding_alphas.is_empty() {
            println!("\n── Compounding Alphas ──");
            for (layer_idx, alpha) in self.compounding_alphas.iter().enumerate() {
                println!("Layer {}: α = {:.4}", layer_idx, alpha);
            }
        }

        if let Some(ref scales) = self.scale_weights {
            println!("\n── Multi-Scale Weights ──");
            println!("Fast:   {:.4}", scales.get(0).unwrap_or(&0.0));
            println!("Medium: {:.4}", scales.get(1).unwrap_or(&0.0));
            println!("Slow:   {:.4}", scales.get(2).unwrap_or(&0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BidirectionalTorusConfig::default();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 6);
        assert!(config.use_parallel_streams);
        assert!(config.use_compounding);
        assert!(config.learnable_alpha);
    }

    #[test]
    fn test_seq_len() {
        let config = BidirectionalTorusConfig::default();
        assert_eq!(config.seq_len(), 32 * 16);
    }

    #[test]
    fn test_to_parallel_config() {
        let config = BidirectionalTorusConfig::default();
        let parallel = config.to_parallel_config();
        assert_eq!(parallel.d_model, config.d_model);
        assert_eq!(parallel.n_major, config.n_major);
    }

    #[test]
    fn test_to_compounding_config() {
        let config = BidirectionalTorusConfig::default();
        let comp = config.to_compounding_config();
        assert_eq!(comp.n_layers, config.n_layers);
        assert_eq!(comp.base_alpha, config.ema_alpha);
    }
}
