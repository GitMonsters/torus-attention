//! Burn backend implementation for AMD GPU support
//!
//! Provides TensorBackend trait implementation for Burn with Vulkan/WGPU backend.
//! This enables GPU acceleration on AMD hardware via Vulkan.

use crate::backend::TensorBackend;
use std::fmt;

use burn_wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData, Distribution};
use burn_autodiff::Autodiff;

/// Error wrapper for Burn errors
#[derive(Debug)]
pub struct BurnError(pub String);

impl fmt::Display for BurnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Burn error: {}", self.0)
    }
}

impl std::error::Error for BurnError {}

impl From<String> for BurnError {
    fn from(s: String) -> Self {
        BurnError(s)
    }
}

impl From<&str> for BurnError {
    fn from(s: &str) -> Self {
        BurnError(s.to_string())
    }
}

/// Burn backend using Vulkan/WGPU (works on AMD GPUs)
#[derive(Clone, Debug)]
pub struct BurnVulkanBackend;

type AutodiffWgpu = Autodiff<Wgpu>;

impl TensorBackend for BurnVulkanBackend {
    type Tensor = Tensor<AutodiffWgpu, 2>;
    type Device = WgpuDevice;
    type Error = BurnError;
    
    fn default_device() -> Self::Device {
        WgpuDevice::default()
    }
    
    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        Ok(Tensor::zeros([dim0, dim1], device))
    }
    
    fn ones(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        Ok(Tensor::ones([dim0, dim1], device))
    }
    
    fn randn(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        Ok(Tensor::random([dim0, dim1], Distribution::Normal(0.0, 1.0), device))
    }
    
    fn rand(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        Ok(Tensor::random([dim0, dim1], Distribution::Uniform(0.0, 1.0), device))
    }
    
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(a.clone().matmul(b.clone()))
    }
    
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(a.clone() + b.clone())
    }
    
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(a.clone() * b.clone())
    }
    
    fn softmax(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 { 1 } else { dim as usize };
        Ok(burn::tensor::activation::softmax(tensor.clone(), dim))
    }
    
    fn transpose(tensor: &Self::Tensor, _dim1: usize, _dim2: usize) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.clone().transpose())
    }
    
    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        Ok(tensor.clone().reshape([dim0, dim1]))
    }
    
    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.dims().to_vec()
    }
    
    fn to_vec(tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        let data: TensorData = tensor.clone().into_data();
        data.to_vec::<f32>().map_err(|e| BurnError(format!("{:?}", e)))
    }
    
    fn from_slice(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error> {
        let dim0 = shape.get(0).copied().unwrap_or(1);
        let dim1 = shape.get(1).copied().unwrap_or(1);
        let tensor_data = TensorData::new(data.to_vec(), [dim0, dim1]);
        Ok(Tensor::from_data(tensor_data, device))
    }
    
    fn scale(tensor: &Self::Tensor, scalar: f32) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.clone() * scalar)
    }
    
    fn exp(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.clone().exp())
    }
    
    fn log(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.clone().log())
    }
    
    fn sqrt(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Ok(tensor.clone().sqrt())
    }
    
    fn mean(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 { 1 } else { dim as usize };
        Ok(tensor.clone().mean_dim(dim))
    }
    
    fn sum(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error> {
        let dim = if dim < 0 { 1 } else { dim as usize };
        Ok(tensor.clone().sum_dim(dim))
    }
    
    fn is_gpu() -> bool {
        true
    }
    
    fn backend_name() -> &'static str {
        "Burn (WGPU/Vulkan - AMD GPU)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_ops() {
        let device = BurnVulkanBackend::default_device();
        
        let a = BurnVulkanBackend::randn(&[4, 4], &device).unwrap();
        let b = BurnVulkanBackend::randn(&[4, 4], &device).unwrap();
        
        let c = BurnVulkanBackend::matmul(&a, &b).unwrap();
        assert_eq!(BurnVulkanBackend::shape(&c), vec![4, 4]);
        
        let d = BurnVulkanBackend::add(&a, &b).unwrap();
        assert_eq!(BurnVulkanBackend::shape(&d), vec![4, 4]);
    }
}
