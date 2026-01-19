//! Bidirectional Flow Primitives for Torus Attention
//!
//! Implements symmetric bidirectional information flow on the torus manifold.
//! Information flows both forward (0 → 2π) and backward (2π → 0) along each
//! loop, with learned symmetric combination weights.
//!
//! Key concepts:
//! - Forward flow: causal attention from past to future
//! - Backward flow: anti-causal attention from future to past
//! - Symmetric combination: ensures equal treatment of both directions

use crate::error::TorusError;
use crate::geometry::TorusCoordinate;
use crate::periodic::PeriodicBoundary;
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Direction of information flow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlowDirection {
    /// Forward flow: index 0 → N (past to future)
    Forward,
    /// Backward flow: index N → 0 (future to past)
    Backward,
}

impl FlowDirection {
    /// Get the opposite direction
    pub fn reverse(&self) -> Self {
        match self {
            FlowDirection::Forward => FlowDirection::Backward,
            FlowDirection::Backward => FlowDirection::Forward,
        }
    }

    /// Get multiplier for index calculations (-1 for backward)
    pub fn sign(&self) -> i64 {
        match self {
            FlowDirection::Forward => 1,
            FlowDirection::Backward => -1,
        }
    }

    /// Check if this is the forward direction
    pub fn is_forward(&self) -> bool {
        matches!(self, FlowDirection::Forward)
    }
}

/// Causal mask generator for directional attention
#[derive(Debug, Clone)]
pub struct CausalMask {
    /// Size of the sequence
    pub size: usize,
    /// Direction of causality
    pub direction: FlowDirection,
    /// Cached mask tensor
    mask: Option<Tensor>,
}

impl CausalMask {
    /// Create a new causal mask
    pub fn new(size: usize, direction: FlowDirection) -> Self {
        Self {
            size,
            direction,
            mask: None,
        }
    }

    /// Generate the mask tensor (cached after first call)
    pub fn get_mask(&mut self, device: &Device) -> TorusResult<Tensor> {
        if let Some(ref mask) = self.mask {
            return Ok(mask.clone());
        }

        let n = self.size;
        let mut mask_data = vec![0.0f32; n * n];

        match self.direction {
            FlowDirection::Forward => {
                // Lower triangular: position i can attend to positions 0..=i
                for i in 0..n {
                    for j in 0..=i {
                        mask_data[i * n + j] = 1.0;
                    }
                }
            }
            FlowDirection::Backward => {
                // Upper triangular: position i can attend to positions i..n
                for i in 0..n {
                    for j in i..n {
                        mask_data[i * n + j] = 1.0;
                    }
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (n, n), device)?;
        self.mask = Some(mask.clone());
        Ok(mask)
    }

    /// Clear cached mask (for memory management)
    pub fn clear_cache(&mut self) {
        self.mask = None;
    }
}

/// Symmetric weight combiner using softmax to ensure weights sum to 1
#[derive(Debug)]
pub struct SymmetricCombiner {
    /// Learnable logits for forward/backward weighting
    /// Shape: [2] where softmax([a, b]) gives [w_forward, w_backward]
    logits: Tensor,
    /// Temperature for softmax (lower = sharper distribution)
    temperature: f64,
}

impl SymmetricCombiner {
    /// Create a new symmetric combiner with equal initial weights
    pub fn new(device: &Device, temperature: f64) -> TorusResult<Self> {
        // Initialize with equal logits (will give 0.5, 0.5 after softmax)
        let logits = Tensor::zeros((2,), DType::F32, device)?;
        Ok(Self { logits, temperature })
    }

    /// Create from VarBuilder for training
    pub fn from_vb(vb: VarBuilder, temperature: f64) -> TorusResult<Self> {
        let logits = vb.get((2,), "combiner_logits")?;
        Ok(Self { logits, temperature })
    }

    /// Get the current weights [w_forward, w_backward]
    pub fn get_weights(&self) -> TorusResult<(f32, f32)> {
        let scaled = (&self.logits / self.temperature)?;
        let weights = candle_nn::ops::softmax(&scaled, 0)?;
        let w_vec: Vec<f32> = weights.to_vec1()?;
        Ok((w_vec[0], w_vec[1]))
    }

    /// Combine forward and backward tensors symmetrically
    pub fn combine(&self, forward: &Tensor, backward: &Tensor) -> TorusResult<Tensor> {
        let scaled = (&self.logits / self.temperature)?;
        let weights = candle_nn::ops::softmax(&scaled, 0)?;
        
        let w_forward = weights.i(0)?;
        let w_backward = weights.i(1)?;

        // weighted_sum = w_forward * forward + w_backward * backward
        let weighted_forward = forward.broadcast_mul(&w_forward)?;
        let weighted_backward = backward.broadcast_mul(&w_backward)?;
        
        let combined = (weighted_forward + weighted_backward)?;
        Ok(combined)
    }

    /// Get the logits tensor for optimization
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

/// Single-direction attention with causal masking
#[derive(Debug)]
pub struct DirectionalAttention {
    /// Query projection
    query: Linear,
    /// Key projection
    key: Linear,
    /// Value projection
    value: Linear,
    /// Output projection
    output: Linear,
    /// Direction of attention
    direction: FlowDirection,
    /// Causal mask
    causal_mask: CausalMask,
    /// Number of attention heads
    n_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Scale factor
    scale: f64,
}

impl DirectionalAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        seq_len: usize,
        direction: FlowDirection,
        vb: VarBuilder,
    ) -> TorusResult<Self> {
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let query = candle_nn::linear(d_model, d_model, vb.pp("query"))?;
        let key = candle_nn::linear(d_model, d_model, vb.pp("key"))?;
        let value = candle_nn::linear(d_model, d_model, vb.pp("value"))?;
        let output = candle_nn::linear(d_model, d_model, vb.pp("output"))?;

        let causal_mask = CausalMask::new(seq_len, direction);

        Ok(Self {
            query,
            key,
            value,
            output,
            direction,
            causal_mask,
            n_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass with directional causal masking
    pub fn forward(&mut self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
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

        // Attention scores: Q @ K^T * scale
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;

        // Apply causal mask
        let mask = self.causal_mask.get_mask(device)?;
        // Convert mask to attention mask (0 -> -inf, 1 -> 0)
        let mask_broadcast = mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(scores.dims())?;
        
        // Where mask is 0, set to large negative value
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let zeros = Tensor::zeros(scores.dims(), DType::F32, device)?;
        let mask_bias = mask_broadcast.where_cond(&zeros, &neg_inf)?;
        let scores = (scores + mask_bias)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Apply to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [B, H, S, D/H] -> [B, S, D]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        let output = self.output.forward(&attn_output)?;
        Ok(output)
    }

    /// Get the flow direction
    pub fn direction(&self) -> FlowDirection {
        self.direction
    }
}

/// Bidirectional attention combining forward and backward flows
#[derive(Debug)]
pub struct BidirectionalAttention {
    /// Forward direction attention
    forward: DirectionalAttention,
    /// Backward direction attention
    backward: DirectionalAttention,
    /// Symmetric combiner for mixing
    combiner: SymmetricCombiner,
    /// Layer normalization
    norm: candle_nn::LayerNorm,
    /// Model dimension
    d_model: usize,
}

impl BidirectionalAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        seq_len: usize,
        combiner_temperature: f64,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let forward = DirectionalAttention::new(
            d_model,
            n_heads,
            seq_len,
            FlowDirection::Forward,
            vb.pp("forward"),
        )?;

        let backward = DirectionalAttention::new(
            d_model,
            n_heads,
            seq_len,
            FlowDirection::Backward,
            vb.pp("backward"),
        )?;

        let combiner = SymmetricCombiner::from_vb(vb.pp("combiner"), combiner_temperature)?;
        let norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            forward,
            backward,
            combiner,
            norm,
            d_model,
        })
    }

    /// Forward pass through bidirectional attention
    pub fn forward(&mut self, x: &Tensor, device: &Device) -> TorusResult<Tensor> {
        // Compute forward and backward attention in parallel conceptually
        // (actual parallelism handled at stream level)
        let forward_out = self.forward.forward(x, device)?;
        let backward_out = self.backward.forward(x, device)?;

        // Symmetrically combine
        let combined = self.combiner.combine(&forward_out, &backward_out)?;

        // Residual + norm
        let output = self.norm.forward(&(x + combined)?)?;
        Ok(output)
    }

    /// Get current forward/backward weights
    pub fn get_weights(&self) -> TorusResult<(f32, f32)> {
        self.combiner.get_weights()
    }

    /// Get all learnable parameters for optimization
    pub fn combiner_logits(&self) -> &Tensor {
        self.combiner.logits()
    }
}

/// Bidirectional position encoding that works in both directions
#[derive(Debug, Clone)]
pub struct BidirectionalPositionEncoding {
    /// Forward position encodings
    forward_encodings: Tensor,
    /// Backward position encodings (reversed)
    backward_encodings: Tensor,
    /// Combined bidirectional encodings
    combined_encodings: Tensor,
}

impl BidirectionalPositionEncoding {
    /// Create bidirectional position encodings
    pub fn new(
        seq_len: usize,
        d_model: usize,
        n_frequencies: usize,
        device: &Device,
    ) -> TorusResult<Self> {
        let encoding_dim = d_model.min(4 * n_frequencies);
        
        // Forward encodings (standard sinusoidal)
        let mut forward_data = vec![0.0f32; seq_len * encoding_dim];
        for pos in 0..seq_len {
            for i in 0..encoding_dim / 2 {
                let freq = 1.0 / (10000.0_f64.powf(2.0 * i as f64 / encoding_dim as f64));
                let angle = pos as f64 * freq;
                forward_data[pos * encoding_dim + 2 * i] = angle.sin() as f32;
                forward_data[pos * encoding_dim + 2 * i + 1] = angle.cos() as f32;
            }
        }

        // Backward encodings (reversed positions)
        let mut backward_data = vec![0.0f32; seq_len * encoding_dim];
        for pos in 0..seq_len {
            let rev_pos = seq_len - 1 - pos;
            for i in 0..encoding_dim / 2 {
                let freq = 1.0 / (10000.0_f64.powf(2.0 * i as f64 / encoding_dim as f64));
                let angle = rev_pos as f64 * freq;
                backward_data[pos * encoding_dim + 2 * i] = angle.sin() as f32;
                backward_data[pos * encoding_dim + 2 * i + 1] = angle.cos() as f32;
            }
        }

        let forward_encodings = Tensor::from_vec(forward_data.clone(), (seq_len, encoding_dim), device)?;
        let backward_encodings = Tensor::from_vec(backward_data.clone(), (seq_len, encoding_dim), device)?;

        // Combined: concatenate or add forward and backward
        // Using addition for dimension preservation
        let combined_encodings = (&forward_encodings + &backward_encodings)?;

        Ok(Self {
            forward_encodings,
            backward_encodings,
            combined_encodings,
        })
    }

    /// Get encodings for a specific direction
    pub fn get(&self, direction: FlowDirection) -> &Tensor {
        match direction {
            FlowDirection::Forward => &self.forward_encodings,
            FlowDirection::Backward => &self.backward_encodings,
        }
    }

    /// Get combined bidirectional encodings
    pub fn combined(&self) -> &Tensor {
        &self.combined_encodings
    }
}

/// Periodic bidirectional encoding for torus topology
#[derive(Debug, Clone)]
pub struct TorusBidirectionalEncoding {
    /// Encodings for major loop (u direction)
    major: BidirectionalPositionEncoding,
    /// Encodings for minor loop (v direction)
    minor: BidirectionalPositionEncoding,
    /// Combined 2D encoding
    combined_2d: Tensor,
}

impl TorusBidirectionalEncoding {
    pub fn new(
        n_major: usize,
        n_minor: usize,
        d_model: usize,
        n_frequencies: usize,
        device: &Device,
    ) -> TorusResult<Self> {
        let major = BidirectionalPositionEncoding::new(n_major, d_model / 2, n_frequencies, device)?;
        let minor = BidirectionalPositionEncoding::new(n_minor, d_model / 2, n_frequencies, device)?;

        // Create 2D combined encoding by outer product-style combination
        let total_positions = n_major * n_minor;
        let major_combined = major.combined();
        let minor_combined = minor.combined();

        // Broadcast and concatenate
        let major_broadcast = major_combined
            .unsqueeze(1)?
            .broadcast_as((n_major, n_minor, d_model / 2))?
            .reshape((total_positions, d_model / 2))?;

        let minor_broadcast = minor_combined
            .unsqueeze(0)?
            .broadcast_as((n_major, n_minor, d_model / 2))?
            .reshape((total_positions, d_model / 2))?;

        let combined_2d = Tensor::cat(&[&major_broadcast, &minor_broadcast], 1)?;

        Ok(Self {
            major,
            minor,
            combined_2d,
        })
    }

    /// Get the full 2D bidirectional encoding
    pub fn get_2d(&self) -> &Tensor {
        &self.combined_2d
    }

    /// Get major loop encoding
    pub fn major(&self) -> &BidirectionalPositionEncoding {
        &self.major
    }

    /// Get minor loop encoding
    pub fn minor(&self) -> &BidirectionalPositionEncoding {
        &self.minor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_direction() {
        assert_eq!(FlowDirection::Forward.reverse(), FlowDirection::Backward);
        assert_eq!(FlowDirection::Backward.reverse(), FlowDirection::Forward);
        assert_eq!(FlowDirection::Forward.sign(), 1);
        assert_eq!(FlowDirection::Backward.sign(), -1);
    }

    #[test]
    fn test_causal_mask_forward() {
        let device = Device::Cpu;
        let mut mask = CausalMask::new(4, FlowDirection::Forward);
        let m = mask.get_mask(&device).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        
        // Forward mask should be lower triangular
        // [1, 0, 0, 0]
        // [1, 1, 0, 0]
        // [1, 1, 1, 0]
        // [1, 1, 1, 1]
        assert_eq!(data[0], 1.0);  // [0,0]
        assert_eq!(data[1], 0.0);  // [0,1]
        assert_eq!(data[4], 1.0);  // [1,0]
        assert_eq!(data[5], 1.0);  // [1,1]
        assert_eq!(data[15], 1.0); // [3,3]
    }

    #[test]
    fn test_causal_mask_backward() {
        let device = Device::Cpu;
        let mut mask = CausalMask::new(4, FlowDirection::Backward);
        let m = mask.get_mask(&device).unwrap();
        let data: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        
        // Backward mask should be upper triangular
        // [1, 1, 1, 1]
        // [0, 1, 1, 1]
        // [0, 0, 1, 1]
        // [0, 0, 0, 1]
        assert_eq!(data[0], 1.0);  // [0,0]
        assert_eq!(data[3], 1.0);  // [0,3]
        assert_eq!(data[4], 0.0);  // [1,0]
        assert_eq!(data[5], 1.0);  // [1,1]
        assert_eq!(data[12], 0.0); // [3,0]
        assert_eq!(data[15], 1.0); // [3,3]
    }

    #[test]
    fn test_symmetric_combiner() {
        let device = Device::Cpu;
        let combiner = SymmetricCombiner::new(&device, 1.0).unwrap();
        let (w_f, w_b) = combiner.get_weights().unwrap();
        
        // Initial weights should be equal (0.5, 0.5)
        assert!((w_f - 0.5).abs() < 0.01);
        assert!((w_b - 0.5).abs() < 0.01);
        assert!((w_f + w_b - 1.0).abs() < 0.001);
    }
}
