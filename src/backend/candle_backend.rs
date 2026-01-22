//! Candle backend implementation
//!
//! Provides TensorBackend trait implementation for Candle (Metal/CUDA).

use crate::backend::TensorBackend;
use candle_core::{DType, Device, Tensor};
use std::fmt;

/// Candle-based backend (supports Metal on macOS, CUDA on Linux/Windows)
#[derive(Clone, Debug)]
pub struct CandleBackend;

/// Error wrapper for Candle errors
#[derive(Debug)]
pub struct CandleError(pub candle_core::Error);

impl fmt::Display for CandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Candle error: {}", self.0)
    }
}

impl std::error::Error for CandleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl From<candle_core::Error> for CandleError {
    fn from(e: candle_core::Error) -> Self {
        CandleError(e)
    }
}

impl TensorBackend for CandleBackend {
    type Tensor = Tensor;
    type Device = Device;
    type Error = CandleError;
    
    fn default_device() -> Self::Device {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            Device::Cpu
        }
    }
    
    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor::zeros(shape, DType::F32, device)?)
    }
    
    fn ones(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor::ones(shape, DType::F32, device)?)
    }
    
    fn randn(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor::randn(0f32, 1f32, shape, device)?)
    }
    
    fn rand(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor::rand(0f32, 1f32, shape, device)?)
    }
    
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(a.matmul(b)?)
    }
    
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok((a + b)?)
    }
    
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok((a * b)?)
    }
    
    fn softmax(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 {
            candle_core::D::Minus1
        } else {
            candle_core::D::Minus(dim as usize)
        };
        Ok(candle_nn::ops::softmax(tensor, dim)?)
    }
    
    fn transpose(tensor: &Self::Tensor, dim1: usize, dim2: usize) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.transpose(dim1, dim2)?)
    }
    
    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.reshape(shape)?)
    }
    
    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.dims().to_vec()
    }
    
    fn to_vec(tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        Ok(tensor.flatten_all()?.to_vec1()?)
    }
    
    fn from_slice(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        Ok(Tensor::from_slice(data, shape, device)?)
    }
    
    fn scale(tensor: &Self::Tensor, scalar: f32) -> Result<Self::Tensor, Self::Error> {
        Ok((tensor * scalar as f64)?)
    }
    
    fn exp(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.exp()?)
    }
    
    fn log(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.log()?)
    }
    
    fn sqrt(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.sqrt()?)
    }
    
    fn mean(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 {
            (tensor.dims().len() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Ok(tensor.mean(dim)?)
    }
    
    fn sum(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 {
            (tensor.dims().len() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Ok(tensor.sum(dim)?)
    }
    
    fn is_gpu() -> bool {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            true
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            false
        }
    }
    
    fn backend_name() -> &'static str {
        #[cfg(feature = "metal")]
        {
            "Candle (Metal)"
        }
        #[cfg(feature = "cuda")]
        {
            "Candle (CUDA)"
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            "Candle (CPU)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_ops() {
        let device = CandleBackend::default_device();
        
        let a = CandleBackend::randn(&[4, 4], &device).unwrap();
        let b = CandleBackend::randn(&[4, 4], &device).unwrap();
        
        let c = CandleBackend::matmul(&a, &b).unwrap();
        assert_eq!(CandleBackend::shape(&c), vec![4, 4]);
        
        let d = CandleBackend::add(&a, &b).unwrap();
        assert_eq!(CandleBackend::shape(&d), vec![4, 4]);
    }
}
