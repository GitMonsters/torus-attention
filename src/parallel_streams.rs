//! 8-Stream Parallel Processor for Torus Attention
//!
//! Implements 8 parallel processing streams that operate concurrently:
//! 1. Major Forward  - u: 0 → 2π
//! 2. Major Backward - u: 2π → 0
//! 3. Minor Forward  - v: 0 → 2π
//! 4. Minor Backward - v: 2π → 0
//! 5. Spiral CW      - clockwise spiral flow
//! 6. Spiral CCW     - counter-clockwise spiral flow
//! 7. Cross U→V      - major to minor coupling
//! 8. Cross V→U      - minor to major coupling
//!
//! All streams execute in parallel using rayon, with learned mixing weights.

use crate::bidirectional::{BidirectionalAttention, FlowDirection, SymmetricCombiner};
use crate::error::TorusError;
use crate::geometry::TorusCoordinate;
use crate::periodic::PeriodicBoundary;
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Identifier for each of the 8 processing streams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StreamId {
    MajorForward = 0,
    MajorBackward = 1,
    MinorForward = 2,
    MinorBackward = 3,
    SpiralCW = 4,
    SpiralCCW = 5,
    CrossUtoV = 6,
    CrossVtoU = 7,
}

impl StreamId {
    /// Get all stream IDs in order
    pub fn all() -> [StreamId; 8] {
        [
            StreamId::MajorForward,
            StreamId::MajorBackward,
            StreamId::MinorForward,
            StreamId::MinorBackward,
            StreamId::SpiralCW,
            StreamId::SpiralCCW,
            StreamId::CrossUtoV,
            StreamId::CrossVtoU,
        ]
    }

    /// Get index (0-7)
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            StreamId::MajorForward => "major_forward",
            StreamId::MajorBackward => "major_backward",
            StreamId::MinorForward => "minor_forward",
            StreamId::MinorBackward => "minor_backward",
            StreamId::SpiralCW => "spiral_cw",
            StreamId::SpiralCCW => "spiral_ccw",
            StreamId::CrossUtoV => "cross_u_to_v",
            StreamId::CrossVtoU => "cross_v_to_u",
        }
    }

    /// Check if this is a bidirectional pair member
    pub fn is_forward(&self) -> bool {
        matches!(
            self,
            StreamId::MajorForward
                | StreamId::MinorForward
                | StreamId::SpiralCW
                | StreamId::CrossUtoV
        )
    }

    /// Get the paired stream (forward <-> backward)
    pub fn pair(&self) -> StreamId {
        match self {
            StreamId::MajorForward => StreamId::MajorBackward,
            StreamId::MajorBackward => StreamId::MajorForward,
            StreamId::MinorForward => StreamId::MinorBackward,
            StreamId::MinorBackward => StreamId::MinorForward,
            StreamId::SpiralCW => StreamId::SpiralCCW,
            StreamId::SpiralCCW => StreamId::SpiralCW,
            StreamId::CrossUtoV => StreamId::CrossVtoU,
            StreamId::CrossVtoU => StreamId::CrossUtoV,
        }
    }
}

/// Configuration for parallel stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelStreamConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads per stream
    pub n_heads: usize,
    /// Grid size in major direction
    pub n_major: usize,
    /// Grid size in minor direction
    pub n_minor: usize,
    /// Spiral winding number for spiral streams
    pub spiral_winding: f64,
    /// Temperature for stream weight softmax
    pub weight_temperature: f64,
    /// Whether to use rayon parallelism
    pub parallel: bool,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for ParallelStreamConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            n_major: 32,
            n_minor: 16,
            spiral_winding: 1.618033988749895, // Golden ratio
            weight_temperature: 1.0,
            parallel: true,
            dropout: 0.1,
        }
    }
}

/// Single processing stream with directional attention
#[derive(Debug)]
pub struct ProcessingStream {
    /// Stream identifier
    pub id: StreamId,
    /// Query projection
    query: Linear,
    /// Key projection
    key: Linear,
    /// Value projection
    value: Linear,
    /// Output projection
    output: Linear,
    /// Attention mask (stream-specific)
    attention_mask: Tensor,
    /// Number of heads
    n_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Scale factor
    scale: f64,
}

impl ProcessingStream {
    pub fn new(
        id: StreamId,
        config: &ParallelStreamConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let query = candle_nn::linear(d_model, d_model, vb.pp("query"))?;
        let key = candle_nn::linear(d_model, d_model, vb.pp("key"))?;
        let value = candle_nn::linear(d_model, d_model, vb.pp("value"))?;
        let output = candle_nn::linear(d_model, d_model, vb.pp("output"))?;

        // Generate stream-specific attention mask
        let attention_mask = Self::generate_mask(id, config, device)?;

        Ok(Self {
            id,
            query,
            key,
            value,
            output,
            attention_mask,
            n_heads,
            head_dim,
            scale,
        })
    }

    /// Generate attention mask based on stream type
    fn generate_mask(
        id: StreamId,
        config: &ParallelStreamConfig,
        device: &Device,
    ) -> TorusResult<Tensor> {
        let n = config.n_major * config.n_minor;
        let mut mask = vec![0.0f32; n * n];

        match id {
            StreamId::MajorForward | StreamId::MajorBackward => {
                // Attention along major direction (rows of the grid)
                let is_forward = id == StreamId::MajorForward;
                for i in 0..config.n_major {
                    for j in 0..config.n_minor {
                        let src_idx = i * config.n_minor + j;
                        // Attend to same minor position, along major
                        for ti in 0..config.n_major {
                            let tgt_idx = ti * config.n_minor + j;
                            let can_attend = if is_forward { ti <= i } else { ti >= i };
                            if can_attend {
                                mask[src_idx * n + tgt_idx] = 1.0;
                            }
                        }
                    }
                }
            }
            StreamId::MinorForward | StreamId::MinorBackward => {
                // Attention along minor direction (columns of the grid)
                let is_forward = id == StreamId::MinorForward;
                for i in 0..config.n_major {
                    for j in 0..config.n_minor {
                        let src_idx = i * config.n_minor + j;
                        // Attend to same major position, along minor
                        for tj in 0..config.n_minor {
                            let tgt_idx = i * config.n_minor + tj;
                            let can_attend = if is_forward { tj <= j } else { tj >= j };
                            if can_attend {
                                mask[src_idx * n + tgt_idx] = 1.0;
                            }
                        }
                    }
                }
            }
            StreamId::SpiralCW | StreamId::SpiralCCW => {
                // Spiral attention pattern
                let winding = config.spiral_winding;
                let is_cw = id == StreamId::SpiralCW;
                let boundary = PeriodicBoundary::new(config.n_major, config.n_minor);

                for i in 0..config.n_major {
                    for j in 0..config.n_minor {
                        let src_idx = i * config.n_minor + j;
                        let u_src = i as f64 * boundary.du;
                        let v_src = j as f64 * boundary.dv;
                        let spiral_src = u_src + winding * v_src;

                        for ti in 0..config.n_major {
                            for tj in 0..config.n_minor {
                                let tgt_idx = ti * config.n_minor + tj;
                                let u_tgt = ti as f64 * boundary.du;
                                let v_tgt = tj as f64 * boundary.dv;
                                let spiral_tgt = u_tgt + winding * v_tgt;

                                // Spiral causality
                                let can_attend = if is_cw {
                                    spiral_tgt <= spiral_src + 0.01
                                } else {
                                    spiral_tgt >= spiral_src - 0.01
                                };

                                if can_attend {
                                    // Distance-based falloff along spiral
                                    let dist = TorusCoordinate::angular_distance(spiral_src, spiral_tgt);
                                    let weight = (-dist * dist / 2.0).exp();
                                    mask[src_idx * n + tgt_idx] = weight as f32;
                                }
                            }
                        }
                    }
                }
            }
            StreamId::CrossUtoV | StreamId::CrossVtoU => {
                // Cross-loop coupling attention
                let is_u_to_v = id == StreamId::CrossUtoV;
                
                for i in 0..config.n_major {
                    for j in 0..config.n_minor {
                        let src_idx = i * config.n_minor + j;
                        
                        for ti in 0..config.n_major {
                            for tj in 0..config.n_minor {
                                let tgt_idx = ti * config.n_minor + tj;
                                
                                // Cross attention: position in one dimension affects the other
                                let di = (i as i64 - ti as i64).abs() as f64;
                                let dj = (j as i64 - tj as i64).abs() as f64;
                                
                                // Wrap-aware distance
                                let di = di.min(config.n_major as f64 - di);
                                let dj = dj.min(config.n_minor as f64 - dj);
                                
                                // Cross coupling: influence decays with distance in source dim,
                                // but is broader in target dim
                                let weight = if is_u_to_v {
                                    // U position affects V attention
                                    (-di * di / (config.n_major as f64)).exp()
                                        * (-dj * dj / (2.0 * config.n_minor as f64)).exp()
                                } else {
                                    // V position affects U attention
                                    (-dj * dj / (config.n_minor as f64)).exp()
                                        * (-di * di / (2.0 * config.n_major as f64)).exp()
                                };
                                
                                mask[src_idx * n + tgt_idx] = weight as f32;
                            }
                        }
                    }
                }
            }
        }

        // Normalize rows
        for i in 0..n {
            let row_start = i * n;
            let sum: f32 = mask[row_start..row_start + n].iter().sum();
            if sum > 0.0 {
                for j in 0..n {
                    mask[row_start + j] /= sum;
                }
            }
        }

        Ok(Tensor::from_vec(mask, (n, n), device)?)
    }

    /// Forward pass for this stream
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;

        // Project Q, K, V
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head: [B, S, D] -> [B, H, S, D/H]
        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;

        // Apply stream-specific mask
        let mask_broadcast = self.attention_mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(scores.dims())?;
        
        // Multiply scores by mask (soft masking)
        let scores = (scores * mask_broadcast)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Apply to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        Ok(self.output.forward(&attn_output)?)
    }
}

/// Learnable weights for combining 8 streams
#[derive(Debug)]
pub struct StreamWeights {
    /// Logits for stream weights [8]
    logits: Tensor,
    /// Temperature for softmax
    temperature: f64,
}

impl StreamWeights {
    /// Create uniform initial weights
    pub fn new(device: &Device, temperature: f64) -> TorusResult<Self> {
        let logits = Tensor::zeros((8,), DType::F32, device)?;
        Ok(Self { logits, temperature })
    }

    /// Create from VarBuilder
    pub fn from_vb(vb: VarBuilder, temperature: f64) -> TorusResult<Self> {
        let logits = vb.get((8,), "stream_weights")?;
        Ok(Self { logits, temperature })
    }

    /// Get normalized weights
    pub fn get_weights(&self) -> TorusResult<Vec<f32>> {
        let scaled = (&self.logits / self.temperature)?;
        let weights = candle_nn::ops::softmax(&scaled, 0)?;
        Ok(weights.to_vec1()?)
    }

    /// Combine stream outputs with learned weights
    pub fn combine(&self, outputs: &[Tensor]) -> TorusResult<Tensor> {
        assert_eq!(outputs.len(), 8);

        let scaled = (&self.logits / self.temperature)?;
        let weights = candle_nn::ops::softmax(&scaled, 0)?;

        // Weighted sum of all streams
        let mut combined = outputs[0].broadcast_mul(&weights.i(0)?)?;
        for (i, output) in outputs.iter().enumerate().skip(1) {
            let weighted = output.broadcast_mul(&weights.i(i)?)?;
            combined = (combined + weighted)?;
        }

        Ok(combined)
    }

    /// Get logits for optimization
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

/// 8-stream parallel processor
#[derive(Debug)]
pub struct ParallelStreamProcessor {
    /// The 8 processing streams
    streams: Vec<ProcessingStream>,
    /// Learned weights for combining streams
    weights: StreamWeights,
    /// Layer normalization
    norm: candle_nn::LayerNorm,
    /// Configuration
    config: ParallelStreamConfig,
}

impl ParallelStreamProcessor {
    pub fn new(config: ParallelStreamConfig, vb: VarBuilder, device: &Device) -> TorusResult<Self> {
        let mut streams = Vec::with_capacity(8);
        
        for id in StreamId::all() {
            let stream = ProcessingStream::new(
                id,
                &config,
                vb.pp(id.name()),
                device,
            )?;
            streams.push(stream);
        }

        let weights = StreamWeights::from_vb(vb.pp("weights"), config.weight_temperature)?;
        let norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            streams,
            weights,
            norm,
            config,
        })
    }

    /// Forward pass through all 8 streams in parallel
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        let outputs: Vec<Tensor> = if self.config.parallel {
            // Parallel execution using rayon
            // Note: For GPU tensors, this parallelism is at the Rust level,
            // not CUDA level. Actual benefit depends on CPU overhead.
            self.streams
                .par_iter()
                .map(|stream| stream.forward(x))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Sequential execution
            self.streams
                .iter()
                .map(|stream| stream.forward(x))
                .collect::<Result<Vec<_>, _>>()?
        };

        // Combine with learned weights
        let combined = self.weights.combine(&outputs)?;

        // Residual + norm
        let output = self.norm.forward(&(x + combined)?)?;
        Ok(output)
    }

    /// Get current stream weights
    pub fn get_weights(&self) -> TorusResult<Vec<f32>> {
        self.weights.get_weights()
    }

    /// Get stream weights by name
    pub fn get_weights_named(&self) -> TorusResult<Vec<(StreamId, f32)>> {
        let weights = self.get_weights()?;
        Ok(StreamId::all()
            .iter()
            .zip(weights.iter())
            .map(|(&id, &w)| (id, w))
            .collect())
    }

    /// Get the configuration
    pub fn config(&self) -> &ParallelStreamConfig {
        &self.config
    }
}

/// Symmetric stream pairing for bidirectional balance
#[derive(Debug)]
pub struct SymmetricStreamPairs {
    /// Pair weights [4] for (major, minor, spiral, cross) pairs
    pair_logits: Tensor,
    /// Within-pair balance [4] for forward vs backward within each pair
    balance_logits: Tensor,
    /// Temperature
    temperature: f64,
}

impl SymmetricStreamPairs {
    pub fn new(device: &Device, temperature: f64) -> TorusResult<Self> {
        // 4 pairs: (major_f/b), (minor_f/b), (spiral_cw/ccw), (cross_u2v/v2u)
        let pair_logits = Tensor::zeros((4,), DType::F32, device)?;
        let balance_logits = Tensor::zeros((4,), DType::F32, device)?;
        Ok(Self {
            pair_logits,
            balance_logits,
            temperature,
        })
    }

    /// Get pair weights (how much each pair contributes)
    pub fn get_pair_weights(&self) -> TorusResult<Vec<f32>> {
        let scaled = (&self.pair_logits / self.temperature)?;
        Ok(candle_nn::ops::softmax(&scaled, 0)?.to_vec1()?)
    }

    /// Get balance weights (forward vs backward within each pair)
    /// Returns values in [0, 1] where 0.5 = perfect balance
    pub fn get_balance_weights(&self) -> TorusResult<Vec<f32>> {
        // Sigmoid for independent forward/backward balance
        let balanced: Vec<f32> = self.balance_logits.to_vec1()?;
        Ok(balanced.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
    }

    /// Combine 8 stream outputs using symmetric pairing
    pub fn combine(&self, outputs: &[Tensor]) -> TorusResult<Tensor> {
        let pair_weights = self.get_pair_weights()?;
        let balance = self.get_balance_weights()?;

        // Pairs: [0,1], [2,3], [4,5], [6,7]
        let pairs = [
            (0, 1), // major forward/backward
            (2, 3), // minor forward/backward
            (4, 5), // spiral cw/ccw
            (6, 7), // cross u2v/v2u
        ];

        let mut combined: Option<Tensor> = None;

        for (pair_idx, &(fwd_idx, bwd_idx)) in pairs.iter().enumerate() {
            let w_pair = pair_weights[pair_idx];
            let w_balance = balance[pair_idx];

            // Within pair: balance * forward + (1-balance) * backward
            let fwd = &outputs[fwd_idx];
            let bwd = &outputs[bwd_idx];
            let pair_output = ((fwd * w_balance as f64)? + (bwd * (1.0 - w_balance) as f64)?)?;

            // Scale by pair weight
            let weighted = (&pair_output * w_pair as f64)?;

            combined = Some(match combined {
                Some(c) => (c + weighted)?,
                None => weighted,
            });
        }

        Ok(combined.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_id() {
        assert_eq!(StreamId::MajorForward.index(), 0);
        assert_eq!(StreamId::CrossVtoU.index(), 7);
        assert_eq!(StreamId::all().len(), 8);
    }

    #[test]
    fn test_stream_pairs() {
        assert_eq!(StreamId::MajorForward.pair(), StreamId::MajorBackward);
        assert_eq!(StreamId::SpiralCW.pair(), StreamId::SpiralCCW);
        assert!(StreamId::MajorForward.is_forward());
        assert!(!StreamId::MajorBackward.is_forward());
    }

    #[test]
    fn test_stream_weights() {
        let device = Device::Cpu;
        let weights = StreamWeights::new(&device, 1.0).unwrap();
        let w = weights.get_weights().unwrap();
        
        // Should be uniform (1/8 each)
        assert_eq!(w.len(), 8);
        for &wi in &w {
            assert!((wi - 0.125).abs() < 0.01);
        }
    }
}
