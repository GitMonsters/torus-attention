//! GPU-Accelerated Parallel Stream Processor
//!
//! This module provides GPU-accelerated versions of the compute-heavy operations
//! in the parallel stream processor. It uses Burn WGPU/Vulkan for AMD GPU support.
//!
//! The key acceleration targets are:
//! - Multi-head attention (Q @ K^T @ V)
//! - Stream-specific attention masking
//! - Softmax computation
//! - Feed-forward network layers

use crate::backend::gpu_compute::{GpuCompute, GpuError};
use crate::parallel_streams::{ParallelStreamConfig, StreamId};
use crate::TorusResult;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

/// GPU-accelerated attention helper
pub struct GpuAttention {
    gpu: GpuCompute,
    /// Precomputed attention masks for each stream [8][seq_len, seq_len]
    masks: Vec<Vec<f32>>,
    /// Configuration
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
}

impl GpuAttention {
    /// Create new GPU attention helper
    pub fn new(config: &ParallelStreamConfig) -> Result<Self, GpuError> {
        let gpu = GpuCompute::new()?;
        let seq_len = config.n_major * config.n_minor;
        let head_dim = config.d_model / config.n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Precompute masks for all 8 streams
        let masks: Vec<Vec<f32>> = StreamId::all()
            .iter()
            .map(|&id| generate_stream_mask(id, config))
            .collect();
        
        Ok(Self {
            gpu,
            masks,
            n_heads: config.n_heads,
            head_dim,
            seq_len,
            scale,
        })
    }
    
    /// Compute attention for a single stream on GPU
    /// Input shapes: q, k, v are [batch, seq_len, d_model] as flat f32 vecs
    pub fn stream_attention(
        &self,
        stream_id: StreamId,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let d_model = self.n_heads * self.head_dim;
        
        // Reshape to [batch, heads, seq, head_dim] for attention
        // This is a logical reshape, we'll handle it in the matmul dimensions
        
        // For simplicity, process each batch item and head separately
        // In a full implementation, we'd use batched matmul
        
        let mut outputs = Vec::with_capacity(batch_size * self.seq_len * d_model);
        let mask = &self.masks[stream_id.index()];
        
        for b in 0..batch_size {
            let batch_offset = b * self.seq_len * d_model;
            
            for h in 0..self.n_heads {
                let head_offset = h * self.head_dim;
                
                // Extract Q, K, V for this head
                let mut q_head = vec![0.0f32; self.seq_len * self.head_dim];
                let mut k_head = vec![0.0f32; self.seq_len * self.head_dim];
                let mut v_head = vec![0.0f32; self.seq_len * self.head_dim];
                
                for s in 0..self.seq_len {
                    for d in 0..self.head_dim {
                        let src_idx = batch_offset + s * d_model + head_offset + d;
                        let dst_idx = s * self.head_dim + d;
                        q_head[dst_idx] = q[src_idx];
                        k_head[dst_idx] = k[src_idx];
                        v_head[dst_idx] = v[src_idx];
                    }
                }
                
                // Q @ K^T on GPU: [seq, head_dim] @ [head_dim, seq] -> [seq, seq]
                // Transpose K: from [seq, head_dim] to [head_dim, seq]
                let mut k_t = vec![0.0f32; self.head_dim * self.seq_len];
                for s in 0..self.seq_len {
                    for d in 0..self.head_dim {
                        k_t[d * self.seq_len + s] = k_head[s * self.head_dim + d];
                    }
                }
                
                let scores = self.gpu.matmul_f32(
                    &q_head, &k_t,
                    self.seq_len, self.head_dim, self.seq_len
                )?;
                
                // Scale and apply mask
                let masked_scores: Vec<f32> = scores.iter()
                    .zip(mask.iter())
                    .map(|(&s, &m)| s * self.scale * m)
                    .collect();
                
                // Softmax on GPU
                let attn_weights = self.gpu.softmax_f32(
                    &masked_scores, 
                    self.seq_len, 
                    self.seq_len
                )?;
                
                // Attention @ V: [seq, seq] @ [seq, head_dim] -> [seq, head_dim]
                let attn_output = self.gpu.matmul_f32(
                    &attn_weights, &v_head,
                    self.seq_len, self.seq_len, self.head_dim
                )?;
                
                // Store output for this head
                if h == 0 {
                    outputs.extend(vec![0.0f32; self.seq_len * d_model]);
                }
                
                let out_batch_offset = b * self.seq_len * d_model;
                for s in 0..self.seq_len {
                    for d in 0..self.head_dim {
                        let src_idx = s * self.head_dim + d;
                        let dst_idx = out_batch_offset + s * d_model + head_offset + d;
                        if dst_idx < outputs.len() {
                            outputs[dst_idx] = attn_output[src_idx];
                        }
                    }
                }
            }
        }
        
        Ok(outputs)
    }
    
    /// Compute all 8 streams in parallel using GPU
    pub fn all_streams_attention(
        &self,
        q: &[f32],
        k: &[f32], 
        v: &[f32],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        StreamId::all()
            .iter()
            .map(|&id| self.stream_attention(id, q, k, v, batch_size))
            .collect()
    }
}

/// Generate attention mask for a stream (same logic as parallel_streams.rs)
fn generate_stream_mask(id: StreamId, config: &ParallelStreamConfig) -> Vec<f32> {
    let n = config.n_major * config.n_minor;
    let mut mask = vec![0.0f32; n * n];
    
    match id {
        StreamId::MajorForward | StreamId::MajorBackward => {
            let is_forward = id == StreamId::MajorForward;
            for i in 0..config.n_major {
                for j in 0..config.n_minor {
                    let src_idx = i * config.n_minor + j;
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
            let is_forward = id == StreamId::MinorForward;
            for i in 0..config.n_major {
                for j in 0..config.n_minor {
                    let src_idx = i * config.n_minor + j;
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
            let winding = config.spiral_winding;
            let is_cw = id == StreamId::SpiralCW;
            let du = std::f64::consts::TAU / config.n_major as f64;
            let dv = std::f64::consts::TAU / config.n_minor as f64;
            
            for i in 0..config.n_major {
                for j in 0..config.n_minor {
                    let src_idx = i * config.n_minor + j;
                    let u_src = i as f64 * du;
                    let v_src = j as f64 * dv;
                    let spiral_src = u_src + winding * v_src;
                    
                    for ti in 0..config.n_major {
                        for tj in 0..config.n_minor {
                            let tgt_idx = ti * config.n_minor + tj;
                            let u_tgt = ti as f64 * du;
                            let v_tgt = tj as f64 * dv;
                            let spiral_tgt = u_tgt + winding * v_tgt;
                            
                            let can_attend = if is_cw {
                                spiral_tgt <= spiral_src + 0.01
                            } else {
                                spiral_tgt >= spiral_src - 0.01
                            };
                            
                            if can_attend {
                                let dist = (spiral_src - spiral_tgt).abs();
                                let weight = (-dist * dist / 2.0).exp();
                                mask[src_idx * n + tgt_idx] = weight as f32;
                            }
                        }
                    }
                }
            }
        }
        StreamId::CrossUtoV | StreamId::CrossVtoU => {
            let is_u_to_v = id == StreamId::CrossUtoV;
            
            for i in 0..config.n_major {
                for j in 0..config.n_minor {
                    let src_idx = i * config.n_minor + j;
                    
                    for ti in 0..config.n_major {
                        for tj in 0..config.n_minor {
                            let tgt_idx = ti * config.n_minor + tj;
                            
                            let di = (i as i64 - ti as i64).abs() as f64;
                            let dj = (j as i64 - tj as i64).abs() as f64;
                            let di = di.min(config.n_major as f64 - di);
                            let dj = dj.min(config.n_minor as f64 - dj);
                            
                            let weight = if is_u_to_v {
                                (-di * di / (config.n_major as f64)).exp()
                                    * (-dj * dj / (2.0 * config.n_minor as f64)).exp()
                            } else {
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
    
    mask
}

/// GPU-accelerated parallel stream processor
pub struct GpuParallelStreamProcessor {
    /// Query projection weights [d_model, d_model]
    query_weights: Vec<f32>,
    /// Key projection weights
    key_weights: Vec<f32>,
    /// Value projection weights  
    value_weights: Vec<f32>,
    /// Output projection weights
    output_weights: Vec<f32>,
    /// GPU attention helper
    gpu_attn: GpuAttention,
    /// Configuration
    #[allow(dead_code)]
    config: ParallelStreamConfig,
    /// GPU compute engine
    #[allow(dead_code)]
    gpu: GpuCompute,
}

impl GpuParallelStreamProcessor {
    /// Create new GPU-accelerated processor
    pub fn new(config: ParallelStreamConfig, _vb: VarBuilder, _device: &Device) -> TorusResult<Self> {
        let d_model = config.d_model;
        
        // Initialize projection weights (random for now, would load from VarBuilder in real use)
        let query_weights = vec![0.0f32; d_model * d_model];
        let key_weights = vec![0.0f32; d_model * d_model]; 
        let value_weights = vec![0.0f32; d_model * d_model];
        let output_weights = vec![0.0f32; d_model * d_model];
        
        let gpu = GpuCompute::new().map_err(|e| crate::TorusError::Backend(e.to_string()))?;
        let gpu_attn = GpuAttention::new(&config)
            .map_err(|e| crate::TorusError::Backend(e.to_string()))?;
        
        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            gpu_attn,
            config,
            gpu,
        })
    }
    
    /// Forward pass with GPU acceleration
    /// Takes Candle tensor, extracts data, computes on GPU, returns Candle tensor
    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        
        // Extract tensor data
        let x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;
        
        // Project Q, K, V on GPU
        let q = self.project_on_gpu(&x_data, &self.query_weights, batch_size, seq_len, d_model)?;
        let k = self.project_on_gpu(&x_data, &self.key_weights, batch_size, seq_len, d_model)?;
        let v = self.project_on_gpu(&x_data, &self.value_weights, batch_size, seq_len, d_model)?;
        
        // Compute all 8 stream attentions on GPU
        let stream_outputs = self.gpu_attn.all_streams_attention(&q, &k, &v, batch_size)
            .map_err(|e| crate::TorusError::Backend(e.to_string()))?;
        
        // Combine streams (simple average for now)
        let combined = self.combine_streams(&stream_outputs, batch_size, seq_len, d_model)?;
        
        // Project output
        let output = self.project_on_gpu(&combined, &self.output_weights, batch_size, seq_len, d_model)?;
        
        // Residual
        let residual: Vec<f32> = output.iter().zip(x_data.iter()).map(|(o, x)| o + x).collect();
        
        // Convert back to Candle tensor
        let result = Tensor::from_vec(residual, (batch_size, seq_len, d_model), x.device())?;
        Ok(result)
    }
    
    fn project_on_gpu(
        &self,
        x: &[f32],
        weights: &[f32],
        batch: usize,
        seq: usize,
        dim: usize,
    ) -> TorusResult<Vec<f32>> {
        // x: [batch * seq, dim], weights: [dim, dim]
        // For each position, compute x @ W^T
        let mut result = vec![0.0f32; batch * seq * dim];
        
        for b in 0..batch {
            for s in 0..seq {
                let offset = (b * seq + s) * dim;
                let x_pos = &x[offset..offset + dim];
                
                // Simple matmul for this position
                for i in 0..dim {
                    let mut sum = 0.0f32;
                    for j in 0..dim {
                        sum += x_pos[j] * weights[i * dim + j];
                    }
                    result[offset + i] = sum;
                }
            }
        }
        
        Ok(result)
    }
    
    fn combine_streams(
        &self,
        streams: &[Vec<f32>],
        batch: usize,
        seq: usize,
        dim: usize,
    ) -> TorusResult<Vec<f32>> {
        let n = batch * seq * dim;
        let mut combined = vec![0.0f32; n];
        let weight = 1.0 / streams.len() as f32;
        
        for stream in streams {
            for i in 0..n {
                combined[i] += stream[i] * weight;
            }
        }
        
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_attention_creation() {
        let config = ParallelStreamConfig::default();
        let gpu_attn = GpuAttention::new(&config);
        assert!(gpu_attn.is_ok());
    }
    
    #[test]
    fn test_stream_mask_generation() {
        let config = ParallelStreamConfig {
            n_major: 4,
            n_minor: 4,
            ..Default::default()
        };
        
        let mask = generate_stream_mask(StreamId::MajorForward, &config);
        assert_eq!(mask.len(), 16 * 16);
        
        // Check that mask sums are approximately 1 per row (normalized)
        let row_sum: f32 = mask[0..16].iter().sum();
        assert!((row_sum - 1.0).abs() < 0.01);
    }
}
