//! # Compounding Cognitive Cohesion Transformer
//!
//! An enhanced bidirectional torus transformer with full compounding cognitive
//! cohesion for AGI-relevant learning. This extends the base transformer with:
//!
//! - Hierarchical coherence (local, regional, global)
//! - Graph-based episodic memory
//! - Predictive coherence with error-driven learning
//! - Cross-layer alignment tracking
//! - Goal state generation for sensorimotor closure
//! - Meta-learning of coherence weights
//!
//! ## Architecture
//!
//! ```text
//! Input
//!   │
//!   ▼
//! Position Encodings
//!   │
//!   ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ For each layer:                                             │
//! │   ├─ Pre-Norm                                               │
//! │   ├─ 8-Stream Parallel Attention                            │
//! │   ├─ Residual                                               │
//! │   ├─ Post-Norm                                              │
//! │   ├─ Feed-Forward (GELU)                                    │
//! │   ├─ Residual                                               │
//! │   └─ COMPOUNDING COHESION:                                  │
//! │       ├─ Update local coherence (SOC from attention)        │
//! │       ├─ Update predictive coherence                        │
//! │       ├─ Buffer features to graph memory                    │
//! │       └─ Apply hierarchical α to EMA compounding            │
//! └─────────────────────────────────────────────────────────────┘
//!   │
//!   ▼
//! Post-Layer Processing:
//!   ├─ Propagate hierarchy (local → regional → global)
//!   ├─ Update cross-layer SMM
//!   └─ Generate goal state
//!   │
//!   ▼
//! Final LayerNorm → Output Projection
//!   │
//!   ▼
//! Post-Episode:
//!   ├─ Memory consolidation (buffer → graph)
//!   └─ Meta-learning weight update
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::{
//!     CompoundingCohesionTransformer, CompoundingTransformerConfig,
//! };
//!
//! let config = CompoundingTransformerConfig::default();
//! let mut transformer = CompoundingCohesionTransformer::new(config, None, vb, &device)?;
//!
//! // Forward pass (within episode)
//! let output = transformer.forward(&input)?;
//!
//! // Check compounding state
//! println!("{}", transformer.cohesion_summary());
//!
//! // Generate goal state
//! let goal = transformer.propose_goal();
//!
//! // End of episode: consolidate memory
//! let consolidation = transformer.end_episode()?;
//!
//! // Update meta-learning from task performance
//! transformer.meta_update(task_loss);
//! ```

use crate::attention::{Activation, TorusFeedForward};
use crate::bidirectional::TorusBidirectionalEncoding;
use crate::compounding::{CompoundingConfig, EMACompounding, MultiScaleCompounding};
use crate::compounding_cohesion::{
    CompoundingCohesionConfig, CompoundingCohesionSystem, ConsolidationResult,
    GoalState,
};
use crate::geometry::{TorusDistanceMatrix, TorusManifold};
use crate::parallel_streams::{ParallelStreamConfig, ParallelStreamProcessor, StreamId};
use crate::rmsnorm::{rms_norm, RmsNorm};
use crate::TorusResult;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for the compounding cohesion transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundingTransformerConfig {
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
    /// Spiral winding number
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
    /// Compounding cohesion configuration
    pub cohesion_config: CompoundingCohesionConfig,
}

impl Default for CompoundingTransformerConfig {
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
            spiral_winding: 1.618033988749895,
            weight_temperature: 1.0,
            parallel_execution: true,
            use_geodesic_bias: true,
            geodesic_sigma: 0.5,
            dropout: 0.1,
            n_pos_frequencies: 16,
            cohesion_config: CompoundingCohesionConfig::default(),
        }
    }
}

impl CompoundingTransformerConfig {
    /// Get sequence length
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

    /// Ensure cohesion config matches transformer config
    pub fn sync_cohesion_config(&mut self) {
        self.cohesion_config.n_layers = self.n_layers;
        self.cohesion_config.n_streams = 8;
        self.cohesion_config.d_model = self.d_model;
        self.cohesion_config.base_coherence.base_alpha = self.ema_alpha;
    }
}

/// Single layer of the compounding cohesion transformer
#[derive(Debug)]
struct CompoundingTransformerLayer {
    /// 8-stream parallel processor
    parallel_streams: ParallelStreamProcessor,
    /// Feed-forward network
    feed_forward: TorusFeedForward,
    /// Pre-attention layer norm
    pre_norm: RmsNorm,
    /// Post-attention layer norm
    post_attn_norm: RmsNorm,
    /// Layer index
    layer_idx: usize,
}

/// Output from a layer forward pass
struct LayerOutput {
    /// The main output tensor
    pub output: Tensor,
    /// Attention pattern proxy (for coherence updates)
    pub attention: Tensor,
}

impl CompoundingTransformerLayer {
    fn new(
        layer_idx: usize,
        config: &CompoundingTransformerConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let parallel_streams =
            ParallelStreamProcessor::new(config.to_parallel_config(), vb.pp("parallel"), device)?;

        let feed_forward =
            TorusFeedForward::new(config.d_model, config.d_ff, Activation::GELU, vb.pp("ff"))?;

        let pre_norm = rms_norm(config.d_model, 1e-5, vb.pp("pre_norm"))?;
        let post_attn_norm = rms_norm(config.d_model, 1e-5, vb.pp("post_attn_norm"))?;

        Ok(Self {
            parallel_streams,
            feed_forward,
            pre_norm,
            post_attn_norm,
            layer_idx,
        })
    }

    fn forward_with_attention(&self, x: &Tensor) -> TorusResult<LayerOutput> {
        // Pre-norm
        let x_norm = self.pre_norm.forward(x)?;

        // 8-stream parallel attention
        let attn_out = self.parallel_streams.forward(&x_norm)?;

        // Create attention proxy
        let attention = {
            let x_flat = x_norm.flatten_all()?;
            let out_flat = attn_out.flatten_all()?;
            let scores = (&x_flat * &out_flat)?;
            candle_nn::ops::softmax(&scores, 0)?
        };

        // Residual
        let x = (x + attn_out)?;

        // Post-attention norm
        let x_norm = self.post_attn_norm.forward(&x)?;

        // Feed-forward
        let ff_out = self.feed_forward.forward(&x_norm)?;

        // Final residual
        let output = (x + ff_out)?;

        Ok(LayerOutput { output, attention })
    }

    fn get_stream_weights(&self) -> TorusResult<Vec<(StreamId, f32)>> {
        self.parallel_streams.get_weights_named()
    }
}

/// Complete compounding cohesion transformer
#[derive(Debug)]
pub struct CompoundingCohesionTransformer {
    /// Input embedding (if needed)
    embedding: Option<Linear>,
    /// Bidirectional position encodings
    position_encodings: TorusBidirectionalEncoding,
    /// Geodesic distance bias
    #[allow(dead_code)]
    geodesic_bias: Option<Tensor>,
    /// Transformer layers
    layers: Vec<CompoundingTransformerLayer>,
    /// EMA compounding (basic)
    compounding: Option<EMACompounding>,
    /// Multi-scale compounding (alternative)
    multi_scale_compounding: Option<MultiScaleCompounding>,
    /// Full compounding cohesion system
    cohesion: CompoundingCohesionSystem,
    /// Output projection
    output_proj: Linear,
    /// Final layer norm
    final_norm: RmsNorm,
    /// Configuration
    config: CompoundingTransformerConfig,
    /// Device
    #[allow(dead_code)]
    device: Device,
}

impl CompoundingCohesionTransformer {
    /// Create a new compounding cohesion transformer
    pub fn new(
        mut config: CompoundingTransformerConfig,
        vocab_size: Option<usize>,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        // Sync configurations
        config.sync_cohesion_config();

        // Optional embedding layer
        let embedding = vocab_size
            .map(|vs| candle_nn::linear(vs, config.d_model, vb.pp("embedding")))
            .transpose()?;

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
            let layer = CompoundingTransformerLayer::new(
                l,
                &config,
                vb.pp(format!("layer_{}", l)),
                device,
            )?;
            layers.push(layer);
        }

        // Compounding (basic EMA, will be modulated by cohesion system)
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

        // Full compounding cohesion system
        let cohesion = CompoundingCohesionSystem::new(config.cohesion_config.clone(), device);

        // Output projection
        let output_proj = candle_nn::linear(
            config.d_model,
            vocab_size.unwrap_or(config.d_model),
            vb.pp("output"),
        )?;

        let final_norm = rms_norm(config.d_model, 1e-5, vb.pp("final_norm"))?;

        Ok(Self {
            embedding,
            position_encodings,
            geodesic_bias,
            layers,
            compounding,
            multi_scale_compounding,
            cohesion,
            output_proj,
            final_norm,
            config,
            device: device.clone(),
        })
    }

    /// Compute geodesic distance bias matrix
    fn compute_geodesic_bias(
        config: &CompoundingTransformerConfig,
        device: &Device,
    ) -> TorusResult<Tensor> {
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

    /// Forward pass with full compounding cohesion
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
        let pos_enc_broadcast =
            pos_enc
                .unsqueeze(0)?
                .broadcast_as((batch_size, seq_len, self.config.d_model))?;
        h = (h + pos_enc_broadcast)?;

        // Process through layers with compounding cohesion
        for layer in self.layers.iter() {
            let layer_output = layer.forward_with_attention(&h)?;
            let mut layer_out = layer_output.output;

            // Update compounding cohesion and get adaptive alpha
            let compound_result = self.cohesion.compound_forward(
                layer.layer_idx,
                &layer_out,
                &layer_output.attention,
            )?;

            // Apply EMA compounding with cohesion-modulated alpha
            if let Some(ref mut comp) = self.compounding {
                layer_out = comp.compound_with_alpha(
                    layer.layer_idx,
                    &layer_out,
                    compound_result.adaptive_alpha,
                )?;
            } else if let Some(ref mut multi) = self.multi_scale_compounding {
                layer_out = multi.compound(layer.layer_idx, &layer_out)?;
            }

            h = layer_out;
        }

        // Post-layer: propagate hierarchy and update cross-layer SMM
        self.cohesion.propagate_hierarchy()?;

        // Final norm
        h = self.final_norm.forward(&h)?;

        // Output projection
        let output = self.output_proj.forward(&h)?;

        Ok(output)
    }

    /// Forward pass for inference (no state updates)
    pub fn forward_inference(&self, x: &Tensor) -> TorusResult<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let mut h = if let Some(ref emb) = self.embedding {
            emb.forward(x)?
        } else {
            x.clone()
        };

        let pos_enc = self.position_encodings.get_2d();
        let pos_enc_broadcast =
            pos_enc
                .unsqueeze(0)?
                .broadcast_as((batch_size, seq_len, self.config.d_model))?;
        h = (h + pos_enc_broadcast)?;

        for layer in self.layers.iter() {
            let layer_output = layer.forward_with_attention(&h)?;
            h = layer_output.output;
        }

        h = self.final_norm.forward(&h)?;
        Ok(self.output_proj.forward(&h)?)
    }

    /// Propose a goal state for sensorimotor action
    pub fn propose_goal(&mut self) -> GoalState {
        self.cohesion.propose_goal_state()
    }

    /// End episode and consolidate memory
    pub fn end_episode(&mut self) -> TorusResult<ConsolidationResult> {
        let result = self.cohesion.post_episode_consolidation();
        self.cohesion.reset_episode();

        // Reset EMA compounding state
        if let Some(ref mut comp) = self.compounding {
            comp.reset()?;
        }
        if let Some(ref mut multi) = self.multi_scale_compounding {
            multi.reset()?;
        }

        Ok(result)
    }

    /// Update meta-learning from task performance (e.g., loss)
    pub fn meta_update(&mut self, task_performance: f64) {
        self.cohesion.update_meta_learning(task_performance);
    }

    /// Reset compounding state (for new sequences within episode)
    pub fn reset_state(&mut self) -> TorusResult<()> {
        if let Some(ref mut comp) = self.compounding {
            comp.reset()?;
        }
        if let Some(ref mut multi) = self.multi_scale_compounding {
            multi.reset()?;
        }
        Ok(())
    }

    /// Get cohesion system summary
    pub fn cohesion_summary(&self) -> String {
        self.cohesion.summary()
    }

    /// Check if system is in coherent compounding state
    pub fn is_compounding_coherent(&self) -> bool {
        self.cohesion.is_compounding_coherent()
    }

    /// Get global coherence score
    pub fn global_coherence(&self) -> f64 {
        self.cohesion.hierarchy.global_coherence()
    }

    /// Get stream weights for all layers
    pub fn get_all_stream_weights(&self) -> TorusResult<Vec<Vec<(StreamId, f32)>>> {
        self.layers.iter().map(|l| l.get_stream_weights()).collect()
    }

    /// Get configuration
    pub fn config(&self) -> &CompoundingTransformerConfig {
        &self.config
    }

    /// Get mutable reference to cohesion system
    pub fn cohesion_mut(&mut self) -> &mut CompoundingCohesionSystem {
        &mut self.cohesion
    }

    /// Get reference to cohesion system
    pub fn cohesion(&self) -> &CompoundingCohesionSystem {
        &self.cohesion
    }
}

/// Statistics for the compounding cohesion transformer
#[derive(Debug, Clone)]
pub struct CompoundingTransformerStats {
    /// Stream weights per layer
    pub stream_weights: Vec<Vec<(StreamId, f32)>>,
    /// Global coherence score
    pub global_coherence: f64,
    /// Cross-layer coherence
    pub cross_layer_coherence: f64,
    /// Prediction error
    pub prediction_error: f64,
    /// Is compounding coherent
    pub is_coherent: bool,
    /// Total graph nodes
    pub graph_nodes: usize,
    /// Episode count
    pub episode: usize,
    /// Inference step
    pub step: usize,
}

impl CompoundingTransformerStats {
    /// Collect stats from transformer
    pub fn from_transformer(transformer: &CompoundingCohesionTransformer) -> TorusResult<Self> {
        let stream_weights = transformer.get_all_stream_weights()?;
        let cohesion = &transformer.cohesion;

        let graph_nodes: usize = cohesion.stream_graphs.iter().map(|g| g.nodes.len()).sum();

        Ok(Self {
            stream_weights,
            global_coherence: cohesion.hierarchy.global_coherence(),
            cross_layer_coherence: cohesion.cross_layer_smm.inter_layer_coherence(),
            prediction_error: cohesion.prediction.avg_error(),
            is_coherent: cohesion.is_compounding_coherent(),
            graph_nodes,
            episode: cohesion.episode,
            step: cohesion.inference_step,
        })
    }

    /// Print summary
    pub fn summary(&self) {
        println!("═══ Compounding Cohesion Stats ═══");
        println!("Global Coherence:     {:.4}", self.global_coherence);
        println!("Cross-Layer Coherence: {:.4}", self.cross_layer_coherence);
        println!("Prediction Error:     {:.4}", self.prediction_error);
        println!(
            "Status:               {}",
            if self.is_coherent {
                "COHERENT"
            } else {
                "UNCERTAIN"
            }
        );
        println!("Graph Nodes:          {}", self.graph_nodes);
        println!("Episode:              {}", self.episode);
        println!("Step:                 {}", self.step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn test_config_default() {
        let config = CompoundingTransformerConfig::default();
        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 6);
        assert!(config.use_compounding);
    }

    #[test]
    fn test_config_sync() {
        let mut config = CompoundingTransformerConfig::default();
        config.n_layers = 12;
        config.d_model = 512;
        config.sync_cohesion_config();

        assert_eq!(config.cohesion_config.n_layers, 12);
        assert_eq!(config.cohesion_config.d_model, 512);
    }

    #[test]
    fn test_transformer_creation() {
        let device = Device::Cpu;
        let config = CompoundingTransformerConfig {
            d_model: 64,
            d_ff: 128,
            n_heads: 4,
            n_layers: 2,
            n_major: 8,
            n_minor: 4,
            ..Default::default()
        };

        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let transformer = CompoundingCohesionTransformer::new(config.clone(), None, vb, &device);
        assert!(transformer.is_ok());
    }
}
