//! GPU-accelerated tensor operations
//!
//! This module provides GPU-accelerated versions of common tensor operations.
//! When the `amd-gpu` feature is enabled, operations automatically use the GPU.
//!
//! # Usage
//!
//! ```rust,ignore
//! use torus_attention::gpu_ops;
//!
//! // Automatically uses GPU if available
//! let result = gpu_ops::matmul(&a, &b)?;
//! let softmax_result = gpu_ops::softmax(&tensor, dim)?;
//! ```

#[cfg(feature = "amd-gpu")]
use crate::backend::GpuCompute;
#[cfg(feature = "amd-gpu")]
use std::sync::OnceLock;

use crate::TorusResult;
use candle_core::Tensor;

/// Global GPU compute instance (lazy initialized)
#[cfg(feature = "amd-gpu")]
static GPU: OnceLock<Option<GpuCompute>> = OnceLock::new();

/// Get or initialize the GPU compute instance
#[cfg(feature = "amd-gpu")]
fn get_gpu() -> Option<&'static GpuCompute> {
    GPU.get_or_init(|| {
        match GpuCompute::new() {
            Ok(gpu) => {
                log::info!("GPU initialized: {}", gpu.backend_info());
                Some(gpu)
            }
            Err(e) => {
                log::warn!("GPU initialization failed, falling back to CPU: {}", e);
                None
            }
        }
    }).as_ref()
}

/// Check if GPU is available and initialized
#[cfg(feature = "amd-gpu")]
pub fn is_gpu_available() -> bool {
    get_gpu().is_some()
}

#[cfg(not(feature = "amd-gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

/// GPU-accelerated matrix multiplication
/// Falls back to CPU if GPU is unavailable
#[cfg(feature = "amd-gpu")]
pub fn matmul(a: &Tensor, b: &Tensor) -> TorusResult<Tensor> {
    // For small tensors, CPU is often faster due to transfer overhead
    let total_elements: usize = a.dims().iter().product();
    
    if total_elements < 4096 {
        // Small tensor - use CPU
        return Ok(a.matmul(b)?);
    }
    
    if let Some(gpu) = get_gpu() {
        // Try GPU acceleration for larger tensors
        let a_dims = a.dims();
        let b_dims = b.dims();
        
        // Handle 2D matmul: [M, K] @ [K, N] -> [M, N]
        if a_dims.len() == 2 && b_dims.len() == 2 {
            let m = a_dims[0];
            let k = a_dims[1];
            let n = b_dims[1];
            
            let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
            let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;
            
            match gpu.matmul_f32(&a_data, &b_data, m, k, n) {
                Ok(result) => {
                    return Ok(Tensor::from_vec(result, (m, n), a.device())?);
                }
                Err(_) => {
                    // Fall back to CPU on error
                }
            }
        }
        
        // Handle 4D batched matmul: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        if a_dims.len() == 4 && b_dims.len() == 4 {
            let batch = a_dims[0];
            let heads = a_dims[1];
            let seq = a_dims[2];
            let dim = a_dims[3];
            
            // For attention: Q @ K^T where K^T has shape [B, H, D, S]
            let b_seq = b_dims[3];
            
            let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
            let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;
            
            // Process batch by batch, head by head
            let mut result_data = Vec::with_capacity(batch * heads * seq * b_seq);
            
            for bi in 0..batch {
                for hi in 0..heads {
                    let a_offset = (bi * heads + hi) * seq * dim;
                    let b_offset = (bi * heads + hi) * dim * b_seq;
                    
                    let a_slice = &a_data[a_offset..a_offset + seq * dim];
                    let b_slice = &b_data[b_offset..b_offset + dim * b_seq];
                    
                    match gpu.matmul_f32(a_slice, b_slice, seq, dim, b_seq) {
                        Ok(chunk) => result_data.extend(chunk),
                        Err(_) => {
                            // Fall back to CPU
                            return Ok(a.matmul(b)?);
                        }
                    }
                }
            }
            
            return Ok(Tensor::from_vec(result_data, (batch, heads, seq, b_seq), a.device())?);
        }
    }
    
    // Fall back to CPU
    Ok(a.matmul(b)?)
}

#[cfg(not(feature = "amd-gpu"))]
pub fn matmul(a: &Tensor, b: &Tensor) -> TorusResult<Tensor> {
    Ok(a.matmul(b)?)
}

/// GPU-accelerated softmax
#[cfg(feature = "amd-gpu")]
pub fn softmax(tensor: &Tensor, dim: usize) -> TorusResult<Tensor> {
    let dims = tensor.dims();
    let total_elements: usize = dims.iter().product();
    
    // For small tensors, use CPU
    if total_elements < 4096 {
        return Ok(candle_nn::ops::softmax(tensor, dim)?);
    }
    
    if let Some(gpu) = get_gpu() {
        // Handle 2D softmax along last dimension
        if dims.len() == 2 && dim == 1 {
            let rows = dims[0];
            let cols = dims[1];
            
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            
            match gpu.softmax_f32(&data, rows, cols) {
                Ok(result) => {
                    return Ok(Tensor::from_vec(result, (rows, cols), tensor.device())?);
                }
                Err(_) => {}
            }
        }
        
        // Handle 4D softmax (attention weights): [B, H, S, S] along last dim
        if dims.len() == 4 && dim == 3 {
            let batch = dims[0];
            let heads = dims[1];
            let seq1 = dims[2];
            let seq2 = dims[3];
            
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            
            match gpu.batched_softmax_f32(&data, batch, heads, seq1) {
                Ok(result) => {
                    return Ok(Tensor::from_vec(result, (batch, heads, seq1, seq2), tensor.device())?);
                }
                Err(_) => {}
            }
        }
    }
    
    // Fall back to CPU
    Ok(candle_nn::ops::softmax(tensor, dim)?)
}

#[cfg(not(feature = "amd-gpu"))]
pub fn softmax(tensor: &Tensor, dim: usize) -> TorusResult<Tensor> {
    Ok(candle_nn::ops::softmax(tensor, dim)?)
}

/// Print GPU status
pub fn print_gpu_status() {
    #[cfg(feature = "amd-gpu")]
    {
        if let Some(gpu) = get_gpu() {
            println!("GPU Status: ENABLED");
            println!("Backend: {}", gpu.backend_info());
        } else {
            println!("GPU Status: DISABLED (initialization failed)");
        }
    }
    
    #[cfg(not(feature = "amd-gpu"))]
    {
        println!("GPU Status: DISABLED (amd-gpu feature not enabled)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_matmul_small() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, (4, 4), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (4, 4), &device).unwrap();
        
        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.dims(), &[4, 4]);
    }
    
    #[test]
    #[cfg(feature = "amd-gpu")]
    fn test_matmul_large_gpu() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, (256, 256), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (256, 256), &device).unwrap();
        
        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.dims(), &[256, 256]);
    }
}
