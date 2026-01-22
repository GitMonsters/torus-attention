//! GPU Compute Module for AMD GPU Acceleration
//!
//! This module provides GPU-accelerated implementations of compute-heavy operations
//! using Burn with WGPU/Vulkan backend. It bridges between Candle tensors (used by
//! the main codebase) and Burn tensors (used for GPU computation).
//!
//! # Usage
//!
//! ```rust,ignore
//! use torus_attention::backend::gpu_compute::GpuCompute;
//!
//! let gpu = GpuCompute::new()?;
//!
//! // Accelerate large matrix multiplication
//! let result = gpu.matmul_f32(&a_data, &b_data, m, k, n)?;
//!
//! // Batch attention computation
//! let attn = gpu.batched_attention(&q, &k, &v, scale)?;
//! ```

use burn::tensor::{Tensor, TensorData, Distribution};
use burn_wgpu::{Wgpu, WgpuDevice};
use std::fmt;

/// Error type for GPU compute operations
#[derive(Debug)]
pub struct GpuError(pub String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU error: {}", self.0)
    }
}

impl std::error::Error for GpuError {}

impl From<String> for GpuError {
    fn from(s: String) -> Self {
        GpuError(s)
    }
}

impl From<&str> for GpuError {
    fn from(s: &str) -> Self {
        GpuError(s.to_string())
    }
}

/// GPU Compute engine using Burn WGPU backend for AMD GPU acceleration
pub struct GpuCompute {
    device: WgpuDevice,
}

impl GpuCompute {
    /// Create a new GPU compute engine
    pub fn new() -> Result<Self, GpuError> {
        let device = WgpuDevice::default();
        Ok(Self { device })
    }
    
    /// Create with a specific device
    pub fn with_device(device: WgpuDevice) -> Self {
        Self { device }
    }
    
    /// Get the device
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
    
    /// Matrix multiplication on GPU: [m, k] @ [k, n] -> [m, n]
    /// Takes raw f32 data and returns raw f32 data (for Candle interop)
    pub fn matmul_f32(
        &self, 
        a: &[f32], 
        b: &[f32], 
        m: usize, 
        k: usize, 
        n: usize
    ) -> Result<Vec<f32>, GpuError> {
        let a_data = TensorData::new(a.to_vec(), [m, k]);
        let b_data = TensorData::new(b.to_vec(), [k, n]);
        
        let a_tensor: Tensor<Wgpu, 2> = Tensor::from_data(a_data, &self.device);
        let b_tensor: Tensor<Wgpu, 2> = Tensor::from_data(b_data, &self.device);
        
        let c_tensor = a_tensor.matmul(b_tensor);
        let c_data: TensorData = c_tensor.into_data();
        
        c_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Batched matrix multiplication: [b, m, k] @ [b, k, n] -> [b, m, n]
    pub fn batched_matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let a_data = TensorData::new(a.to_vec(), [batch, m, k]);
        let b_data = TensorData::new(b.to_vec(), [batch, k, n]);
        
        let a_tensor: Tensor<Wgpu, 3> = Tensor::from_data(a_data, &self.device);
        let b_tensor: Tensor<Wgpu, 3> = Tensor::from_data(b_data, &self.device);
        
        let c_tensor = a_tensor.matmul(b_tensor);
        let c_data: TensorData = c_tensor.into_data();
        
        c_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Softmax on GPU along last dimension
    pub fn softmax_f32(&self, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, GpuError> {
        let data = TensorData::new(x.to_vec(), [rows, cols]);
        let tensor: Tensor<Wgpu, 2> = Tensor::from_data(data, &self.device);
        
        let result = burn::tensor::activation::softmax(tensor, 1);
        let result_data: TensorData = result.into_data();
        
        result_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Batched softmax: [b, h, s, s] -> softmax along last dim
    pub fn batched_softmax_f32(
        &self, 
        x: &[f32], 
        batch: usize,
        heads: usize,
        seq_len: usize
    ) -> Result<Vec<f32>, GpuError> {
        let data = TensorData::new(x.to_vec(), [batch, heads, seq_len, seq_len]);
        let tensor: Tensor<Wgpu, 4> = Tensor::from_data(data, &self.device);
        
        let result = burn::tensor::activation::softmax(tensor, 3);
        let result_data: TensorData = result.into_data();
        
        result_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Q: [b, h, s, d], K: [b, h, s, d], V: [b, h, s, d] -> [b, h, s, d]
    pub fn attention_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>, GpuError> {
        let q_data = TensorData::new(q.to_vec(), [batch, heads, seq_len, head_dim]);
        let k_data = TensorData::new(k.to_vec(), [batch, heads, seq_len, head_dim]);
        let v_data = TensorData::new(v.to_vec(), [batch, heads, seq_len, head_dim]);
        
        let q_tensor: Tensor<Wgpu, 4> = Tensor::from_data(q_data, &self.device);
        let k_tensor: Tensor<Wgpu, 4> = Tensor::from_data(k_data, &self.device);
        let v_tensor: Tensor<Wgpu, 4> = Tensor::from_data(v_data, &self.device);
        
        // Q @ K^T: [b, h, s, d] @ [b, h, d, s] -> [b, h, s, s]
        let k_t = k_tensor.swap_dims(2, 3);
        let scores = q_tensor.matmul(k_t);
        
        // Scale
        let scores = scores * scale;
        
        // Softmax along last dimension
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        
        // Attention @ V: [b, h, s, s] @ [b, h, s, d] -> [b, h, s, d]
        let output = attn_weights.matmul(v_tensor);
        
        let output_data: TensorData = output.into_data();
        output_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Element-wise operations on GPU: a * b + c (fused multiply-add)
    pub fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, GpuError> {
        let len = a.len();
        let a_data = TensorData::new(a.to_vec(), [len]);
        let b_data = TensorData::new(b.to_vec(), [len]);
        let c_data = TensorData::new(c.to_vec(), [len]);
        
        let a_tensor: Tensor<Wgpu, 1> = Tensor::from_data(a_data, &self.device);
        let b_tensor: Tensor<Wgpu, 1> = Tensor::from_data(b_data, &self.device);
        let c_tensor: Tensor<Wgpu, 1> = Tensor::from_data(c_data, &self.device);
        
        let result = a_tensor * b_tensor + c_tensor;
        let result_data: TensorData = result.into_data();
        
        result_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Layer normalization on GPU
    pub fn layer_norm_f32(
        &self, 
        x: &[f32], 
        gamma: &[f32], 
        beta: &[f32],
        batch: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, GpuError> {
        let x_data = TensorData::new(x.to_vec(), [batch, dim]);
        let gamma_data = TensorData::new(gamma.to_vec(), [dim]);
        let beta_data = TensorData::new(beta.to_vec(), [dim]);
        
        let x_tensor: Tensor<Wgpu, 2> = Tensor::from_data(x_data, &self.device);
        let gamma_tensor: Tensor<Wgpu, 1> = Tensor::from_data(gamma_data, &self.device);
        let beta_tensor: Tensor<Wgpu, 1> = Tensor::from_data(beta_data, &self.device);
        
        // Compute mean and variance
        let mean = x_tensor.clone().mean_dim(1);
        let x_centered = x_tensor.clone() - mean.clone().unsqueeze_dim(1);
        let variance = x_centered.clone().powf_scalar(2.0).mean_dim(1);
        
        // Normalize
        let std = (variance + eps).sqrt().unsqueeze_dim(1);
        let x_norm = x_centered / std;
        
        // Scale and shift
        let gamma_2d = gamma_tensor.unsqueeze_dim(0);
        let beta_2d = beta_tensor.unsqueeze_dim(0);
        let result = x_norm * gamma_2d + beta_2d;
        
        let result_data: TensorData = result.into_data();
        result_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// GELU activation on GPU
    pub fn gelu_f32(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>, GpuError> {
        let data = TensorData::new(x.to_vec(), shape.to_vec());
        let tensor: Tensor<Wgpu, 2> = Tensor::from_data(data, &self.device);
        
        let result = burn::tensor::activation::gelu(tensor);
        let result_data: TensorData = result.into_data();
        
        result_data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Create random tensor for initialization
    pub fn randn_f32(&self, shape: &[usize]) -> Result<Vec<f32>, GpuError> {
        let total: usize = shape.iter().product();
        let flat_shape = [total];
        
        let tensor: Tensor<Wgpu, 1> = Tensor::random(
            flat_shape, 
            Distribution::Normal(0.0, 1.0), 
            &self.device
        );
        
        let data: TensorData = tensor.into_data();
        data.to_vec::<f32>().map_err(|e| GpuError(format!("{:?}", e)))
    }
    
    /// Check if GPU is available
    pub fn is_available() -> bool {
        // WGPU will always provide some backend (even if just CPU fallback)
        true
    }
    
    /// Get backend info
    pub fn backend_info(&self) -> String {
        "Burn WGPU/Vulkan (AMD GPU)".to_string()
    }
}

impl Default for GpuCompute {
    fn default() -> Self {
        Self::new().expect("Failed to create GPU compute engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matmul() {
        let gpu = GpuCompute::new().unwrap();
        
        // Simple 2x2 matmul
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let c = gpu.matmul_f32(&a, &b, 2, 2, 2).unwrap();
        
        // Expected: [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 0.01);
        assert!((c[1] - 22.0).abs() < 0.01);
        assert!((c[2] - 43.0).abs() < 0.01);
        assert!((c[3] - 50.0).abs() < 0.01);
    }
    
    #[test]
    fn test_softmax() {
        let gpu = GpuCompute::new().unwrap();
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0];
        let result = gpu.softmax_f32(&x, 2, 3).unwrap();
        
        // Each row should sum to ~1
        let row1_sum: f32 = result[0..3].iter().sum();
        let row2_sum: f32 = result[3..6].iter().sum();
        
        assert!((row1_sum - 1.0).abs() < 0.01);
        assert!((row2_sum - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_attention() {
        let gpu = GpuCompute::new().unwrap();
        
        // Tiny attention: batch=1, heads=1, seq=2, dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = gpu.attention_f32(&q, &k, &v, 1, 1, 2, 2, 1.0).unwrap();
        
        // Should have same shape as V
        assert_eq!(result.len(), 4);
    }
}
