//! Unified Bidirectional Torus Attention Integration
//!
//! Combines all components into a cohesive bidirectional parallel
//! compounding attention system:
//!
//! Input → 8-Stream Parallel → Symmetric Combine → Coherence-Modulated EMA → Output
//!
//! # Features
//!
//! - **8 parallel processing streams**: Major/minor forward/backward, spiral CW/CCW, cross U↔V
//! - **Symmetric bidirectional combination**: Learned weights for stream mixing
//! - **Multi-layer EMA compounding**: Learnable α per layer with momentum
//! - **Cognitive coherence integration**: SOC + SMM for adaptive compounding
//! - **Torus-aware position encodings**: Geodesic biases and periodic boundaries
//!
//! # Cognitive Coherence Integration
//!
//! The coherence module bridges cognitive cohesion (inter-stream alignment)
//! with psychological coherence (state stability):
//!
//! ```text
//! 8 Streams → SMM (alignment) → Coherence-weighted fusion
//!                   ↓
//!           SOC (stability) → Adaptive α for EMA
//! ```
//!
//! When coherence is enabled:
//! - Attention patterns are analyzed for comprehensibility, manageability, meaningfulness
//! - Stream alignment is tracked via Shared Mental Models
//! - EMA compounding alpha is dynamically adjusted based on coherence state
//!
//! # Example
//!
//! ```rust,ignore
//! use torus_attention::{BidirectionalTorusConfig, BidirectionalTorusTransformer};
//! use candle_core::{Device, DType, Tensor};
//! use candle_nn::VarMap;
//!
//! let device = Device::Cpu;
//! let mut config = BidirectionalTorusConfig::default();
//! config.use_coherence = true;
//!
//! let varmap = VarMap::new();
//! let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
//!
//! let mut transformer = BidirectionalTorusTransformer::new(
//!     config.clone(), None, vb, &device
//! ).unwrap();
//!
//! let input = Tensor::randn(0.0f32, 1.0, (1, config.seq_len(), config.d_model), &device).unwrap();
//! let output = transformer.forward(&input).unwrap();
//!
//! // Check coherence state
//! if let Some(coherent) = transformer.is_coherent() {
//!     println!("System coherent: {}", coherent);
//! }
//! ```

use crate::attention::{Activation, TorusFeedForward};
use crate::bidirectional::TorusBidirectionalEncoding;
use crate::coherence::{CognitiveCoherenceLayer, CoherenceConfig};
use crate::compounding::{CompoundingConfig, EMACompounding, MultiScaleCompounding};
use crate::geometry::{TorusDistanceMatrix, TorusManifold};
use crate::parallel_streams::{ParallelStreamConfig, ParallelStreamProcessor, StreamId};
use crate::rmsnorm::{rms_norm, RmsNorm};
use crate::TorusResult;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for the unified bidirectional torus transformer.
///
/// This configuration controls all aspects of the transformer architecture including:
/// - Model dimensions and layer counts
/// - Torus geometry parameters (radii, grid sizes)
/// - Processing options (parallel streams, compounding, coherence)
/// - Numerical hyperparameters (dropout, temperatures, learning rates)
///
/// # Coherence Configuration
///
/// When `use_coherence` is enabled, the transformer tracks:
/// - **Sense of Coherence (SOC)**: Comprehensibility, manageability, meaningfulness
/// - **Shared Mental Models (SMM)**: Inter-stream alignment
/// - **Adaptive Alpha**: EMA compounding rate adjusted by coherence state
///
/// # Example
///
/// ```rust
/// use torus_attention::BidirectionalTorusConfig;
///
/// // Default configuration with coherence enabled
/// let config = BidirectionalTorusConfig::default();
/// assert!(config.use_coherence);
///
/// // Custom configuration
/// let mut config = BidirectionalTorusConfig::default();
/// config.d_model = 512;
/// config.n_layers = 12;
/// config.use_coherence = true;
/// config.coherence_threshold = 0.7;
/// ```
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
    /// Whether to use cognitive coherence for adaptive compounding
    pub use_coherence: bool,
    /// Coherence threshold for stable processing
    pub coherence_threshold: f64,
    /// Learning rate for shared mental model updates
    pub smm_learning_rate: f64,
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
            use_coherence: true,
            coherence_threshold: 0.6,
            smm_learning_rate: 0.01,
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

    /// Convert to coherence config
    pub fn to_coherence_config(&self) -> CoherenceConfig {
        CoherenceConfig {
            n_streams: 8,
            d_model: self.d_model,
            smm_learning_rate: self.smm_learning_rate,
            base_alpha: self.ema_alpha,
            min_alpha: 0.1,
            max_alpha: 0.99,
            coherence_threshold: self.coherence_threshold,
            adaptive_alpha: true,
            comprehensibility_weight: 0.25,
            manageability_weight: 0.25,
            meaningfulness_weight: 0.50,
        }
    }
}

/// Single layer of the bidirectional torus transformer.
///
/// Each layer performs:
/// 1. Pre-normalization (RMSNorm for Metal compatibility)
/// 2. 8-stream parallel attention (major/minor forward/backward, spiral CW/CCW, cross U↔V)
/// 3. Residual connection
/// 4. Post-attention normalization (RMSNorm)
/// 5. Feed-forward network with GELU activation
/// 6. Final residual connection
///
/// When coherence tracking is enabled at the transformer level, the layer
/// returns both the output tensor and an attention proxy for coherence updates.
#[derive(Debug)]
pub struct BidirectionalTorusLayer {
    /// 8-stream parallel processor
    parallel_streams: ParallelStreamProcessor,
    /// Feed-forward network
    feed_forward: TorusFeedForward,
    /// Pre-attention layer norm (RMSNorm)
    pre_norm: RmsNorm,
    /// Post-attention layer norm (RMSNorm)
    post_attn_norm: RmsNorm,
    /// Layer index
    #[allow(dead_code)]
    layer_idx: usize,
    /// Configuration
    #[allow(dead_code)]
    config: BidirectionalTorusConfig,
}

/// Output from a layer forward pass, including optional attention for coherence.
///
/// This struct enables coherence tracking by providing:
/// - The main output tensor for downstream processing
/// - An optional attention pattern proxy for SOC metric computation
///
/// # Attention Proxy
///
/// The attention proxy is computed as a softmax of element-wise products
/// between the normalized input and attention output. This serves as a
/// "how much does the output attend to the input" signal that the
/// coherence module uses to compute comprehensibility and meaningfulness.
pub struct LayerOutput {
    /// The main output tensor from the layer
    pub output: Tensor,
    /// Combined attention pattern (for coherence updates)
    /// This is a flattened, normalized attention proxy
    pub attention: Option<Tensor>,
}

impl BidirectionalTorusLayer {
    pub fn new(
        layer_idx: usize,
        config: BidirectionalTorusConfig,
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
            config,
        })
    }

    /// Forward pass through the layer
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        self.forward_with_attention(x).map(|out| out.output)
    }

    /// Forward pass that also returns attention patterns for coherence tracking
    pub fn forward_with_attention(&self, x: &Tensor) -> TorusResult<LayerOutput> {
        // Pre-norm
        let x_norm = self.pre_norm.forward(x)?;

        // 8-stream parallel attention
        let attn_out = self.parallel_streams.forward(&x_norm)?;

        // Create a simple attention proxy from the output (normalized dot-product with input)
        // This serves as a "how much the output attends to the input" signal
        let attention = {
            let x_flat = x_norm.flatten_all()?;
            let out_flat = attn_out.flatten_all()?;
            // Compute attention-like pattern: softmax of dot products
            let scores = (&x_flat * &out_flat)?;
            let attn = candle_nn::ops::softmax(&scores, 0)?;
            Some(attn)
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

    /// Get current stream weights
    pub fn get_stream_weights(&self) -> TorusResult<Vec<(StreamId, f32)>> {
        self.parallel_streams.get_weights_named()
    }
}

/// Complete bidirectional torus transformer with compounding and coherence.
///
/// This is the main entry point for the torus attention architecture. It combines:
///
/// - **Position Encoding**: Bidirectional torus-aware encodings for the 2D manifold
/// - **8-Stream Parallel Processing**: Multiple information flow patterns
/// - **EMA Compounding**: Accumulating information across layers
/// - **Cognitive Coherence**: Adaptive compounding based on attention stability
///
/// # Architecture
///
/// ```text
/// Input
///   │
///   ▼
/// Embedding (optional, for vocab input)
///   │
///   ▼
/// + Position Encodings (torus bidirectional)
///   │
///   ▼
/// ┌─────────────────────────────────────────┐
/// │ For each layer:                         │
/// │   ├─ Pre-Norm                           │
/// │   ├─ 8-Stream Parallel Attention        │
/// │   ├─ Residual                           │
/// │   ├─ Post-Norm                          │
/// │   ├─ Feed-Forward (GELU)                │
/// │   ├─ Residual                           │
/// │   └─ Coherence-Modulated EMA Compound   │
/// └─────────────────────────────────────────┘
///   │
///   ▼
/// Final LayerNorm
///   │
///   ▼
/// Output Projection
/// ```
///
/// # Coherence Integration
///
/// When `config.use_coherence` is true:
/// 1. Each layer's attention proxy is fed to the coherence module
/// 2. SOC metrics (comprehensibility, manageability, meaningfulness) are updated
/// 3. An adaptive alpha is computed based on coherence state
/// 4. EMA compounding uses this adaptive alpha instead of learned alpha
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::{BidirectionalTorusConfig, BidirectionalTorusTransformer};
///
/// let device = Device::Cpu;
/// let config = BidirectionalTorusConfig::default();
/// let varmap = VarMap::new();
/// let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
///
/// let mut transformer = BidirectionalTorusTransformer::new(
///     config.clone(), None, vb, &device
/// )?;
///
/// // Forward pass
/// let output = transformer.forward(&input)?;
///
/// // Check coherence metrics
/// println!("SOC: {:.3}", transformer.coherence_score().unwrap_or(0.0));
/// println!("Cohesion: {:.3}", transformer.cohesion_score().unwrap_or(0.0));
/// ```
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
    /// Cognitive coherence layer (optional)
    coherence: Option<CognitiveCoherenceLayer>,
    /// Output projection
    output_proj: Linear,
    /// Final layer norm (RMSNorm for Metal compatibility)
    final_norm: RmsNorm,
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

        // Cognitive coherence layer
        let coherence = if config.use_coherence {
            Some(CognitiveCoherenceLayer::new(
                config.to_coherence_config(),
                device,
            ))
        } else {
            None
        };

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
            coherence,
            output_proj,
            final_norm,
            config,
            device: device.clone(),
        })
    }

    /// Compute geodesic distance bias matrix
    fn compute_geodesic_bias(
        config: &BidirectionalTorusConfig,
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
        let pos_enc_broadcast =
            pos_enc
                .unsqueeze(0)?
                .broadcast_as((batch_size, seq_len, self.config.d_model))?;
        h = (h + pos_enc_broadcast)?;

        // Process through layers with compounding and coherence
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Forward with attention for coherence tracking
            let layer_output = layer.forward_with_attention(&h)?;
            let mut layer_out = layer_output.output;

            // Update coherence if enabled and attention is available
            if let (Some(ref mut coherence), Some(ref attention)) =
                (&mut self.coherence, &layer_output.attention)
            {
                // Update SOC from attention patterns
                coherence.update_soc(attention, &layer_out)?;

                // Get adaptive alpha from coherence
                let adaptive_alpha = coherence.compute_adaptive_alpha();

                // Apply coherence-modulated compounding
                if let Some(ref mut comp) = self.compounding {
                    // Use coherence-adjusted alpha for this layer's compounding
                    layer_out = comp.compound_with_alpha(layer_idx, &layer_out, adaptive_alpha)?;
                } else if let Some(ref mut multi) = self.multi_scale_compounding {
                    layer_out = multi.compound(layer_idx, &layer_out)?;
                }
            } else {
                // Standard compounding without coherence
                layer_out = if let Some(ref mut comp) = self.compounding {
                    comp.compound(layer_idx, &layer_out)?
                } else if let Some(ref mut multi) = self.multi_scale_compounding {
                    multi.compound(layer_idx, &layer_out)?
                } else {
                    layer_out
                };
            }

            h = layer_out;
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
        self.layers.iter().map(|l| l.get_stream_weights()).collect()
    }

    /// Get compounding alphas
    pub fn get_compounding_alphas(&self) -> TorusResult<Option<Vec<f64>>> {
        if let Some(ref comp) = self.compounding {
            Ok(Some(comp.get_all_alphas()?))
        } else {
            Ok(None)
        }
    }

    /// Get current coherence state
    pub fn get_coherence(&self) -> Option<&CognitiveCoherenceLayer> {
        self.coherence.as_ref()
    }

    /// Get current coherence state (mutable)
    pub fn get_coherence_mut(&mut self) -> Option<&mut CognitiveCoherenceLayer> {
        self.coherence.as_mut()
    }

    /// Get current SOC score
    pub fn coherence_score(&self) -> Option<f64> {
        self.coherence.as_ref().map(|c| c.psychological_coherence())
    }

    /// Get current cognitive cohesion score
    pub fn cohesion_score(&self) -> Option<f64> {
        self.coherence.as_ref().map(|c| c.cognitive_cohesion())
    }

    /// Check if the system is in a coherent state
    pub fn is_coherent(&self) -> Option<bool> {
        self.coherence.as_ref().map(|c| c.is_coherent())
    }

    /// Get coherence summary string
    pub fn coherence_summary(&self) -> Option<String> {
        self.coherence.as_ref().map(|c| c.summary())
    }

    /// Get configuration
    pub fn config(&self) -> &BidirectionalTorusConfig {
        &self.config
    }

    /// Get position encodings
    pub fn position_encodings(&self) -> &TorusBidirectionalEncoding {
        &self.position_encodings
    }

    /// Forward pass for inference (no state updates, suitable for teacher model)
    ///
    /// Unlike `forward()`, this method:
    /// - Does not update compounding state
    /// - Does not update coherence state
    /// - Is suitable for frozen teacher models in knowledge distillation
    pub fn forward_inference(&self, x: &Tensor) -> TorusResult<Tensor> {
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

        // Process through layers WITHOUT compounding/coherence updates
        for layer in self.layers.iter() {
            let layer_output = layer.forward_with_attention(&h)?;
            h = layer_output.output;
        }

        // Final norm
        h = self.final_norm.forward(&h)?;

        // Output projection
        let output = self.output_proj.forward(&h)?;

        Ok(output)
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

/// Statistics and diagnostics for the bidirectional transformer.
///
/// Collects comprehensive metrics from a transformer instance including:
/// - Stream weights for all layers
/// - Compounding alpha values
/// - Multi-scale weights (if enabled)
/// - Coherence metrics (if enabled)
///
/// # Example
///
/// ```rust,ignore
/// let stats = BidirectionalStats::from_transformer(&transformer)?;
/// stats.summary(); // Prints formatted statistics
///
/// if let Some(coh) = stats.coherence_metrics {
///     println!("SOC: {:.3}, Coherent: {}", coh.soc_score, coh.is_coherent);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct BidirectionalStats {
    /// Stream weights per layer (8 streams per layer)
    pub stream_weights: Vec<Vec<(StreamId, f32)>>,
    /// Compounding alphas per layer
    pub compounding_alphas: Vec<f64>,
    /// Multi-scale weights (fast/medium/slow) if multi-scale compounding is enabled
    pub scale_weights: Option<Vec<f32>>,
    /// Coherence metrics (if coherence is enabled)
    pub coherence_metrics: Option<CoherenceMetrics>,
}

/// Coherence-specific metrics extracted from the transformer.
///
/// Provides a snapshot of the cognitive coherence state including:
/// - SOC (Sense of Coherence) components and overall score
/// - Cognitive cohesion (inter-stream alignment)
/// - Current adaptive alpha being used for compounding
/// - Trend analysis for monitoring coherence over time
///
/// # Interpretation
///
/// - **soc_score > 0.6**: System is in a coherent state (threshold configurable)
/// - **cognitive_cohesion > 0.7**: Streams are well-aligned
/// - **coherence_trend > 0**: Coherence is improving over time
#[derive(Debug, Clone)]
pub struct CoherenceMetrics {
    /// Sense of coherence score (0-1), weighted combination of components
    pub soc_score: f64,
    /// Comprehensibility: perceived clarity of attention patterns (0-1)
    /// High = sharp, consistent attention; Low = high entropy, scattered
    pub comprehensibility: f64,
    /// Manageability: capacity vs demand balance (0-1)
    /// High = within capacity; Low = overwhelmed
    pub manageability: f64,
    /// Meaningfulness: signal concentration/importance (0-1)
    /// High = strong signal-to-noise; Low = noise-dominated
    pub meaningfulness: f64,
    /// Cognitive cohesion: average inter-stream alignment (0-1)
    pub cognitive_cohesion: f64,
    /// Current adaptive alpha being used for EMA compounding
    pub adaptive_alpha: f64,
    /// Coherence trend: positive = improving, negative = degrading
    pub coherence_trend: f64,
    /// Whether system is currently in coherent state
    pub is_coherent: bool,
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

        // Extract coherence metrics if available
        let coherence_metrics = transformer.coherence.as_ref().map(|c| {
            let soc = c.soc();
            CoherenceMetrics {
                soc_score: soc.score(),
                comprehensibility: soc.comprehensibility,
                manageability: soc.manageability,
                meaningfulness: soc.meaningfulness,
                cognitive_cohesion: c.cognitive_cohesion(),
                adaptive_alpha: c.compute_adaptive_alpha(),
                coherence_trend: c.coherence_trend(),
                is_coherent: c.is_coherent(),
            }
        });

        Ok(Self {
            stream_weights,
            compounding_alphas,
            scale_weights,
            coherence_metrics,
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
            println!("Fast:   {:.4}", scales.first().unwrap_or(&0.0));
            println!("Medium: {:.4}", scales.get(1).unwrap_or(&0.0));
            println!("Slow:   {:.4}", scales.get(2).unwrap_or(&0.0));
        }

        if let Some(ref coh) = self.coherence_metrics {
            println!("\n── Cognitive Coherence ──");
            println!("SOC Score:         {:.4}", coh.soc_score);
            println!("  Comprehensibility: {:.4}", coh.comprehensibility);
            println!("  Manageability:     {:.4}", coh.manageability);
            println!("  Meaningfulness:    {:.4}", coh.meaningfulness);
            println!("Cognitive Cohesion: {:.4}", coh.cognitive_cohesion);
            println!("Adaptive Alpha:     {:.4}", coh.adaptive_alpha);
            println!("Coherence Trend:    {:+.4}", coh.coherence_trend);
            println!(
                "Status:             {}",
                if coh.is_coherent {
                    "COHERENT"
                } else {
                    "UNCERTAIN"
                }
            );
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
        assert!(config.use_coherence);
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

    #[test]
    fn test_to_coherence_config() {
        let config = BidirectionalTorusConfig::default();
        let coh = config.to_coherence_config();
        assert_eq!(coh.n_streams, 8);
        assert_eq!(coh.d_model, config.d_model);
        assert_eq!(coh.coherence_threshold, config.coherence_threshold);
    }

    #[test]
    fn test_config_without_coherence() {
        let config = BidirectionalTorusConfig {
            use_coherence: false,
            ..Default::default()
        };
        assert!(!config.use_coherence);
    }
}
