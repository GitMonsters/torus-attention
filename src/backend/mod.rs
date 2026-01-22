//! Backend abstraction layer for multi-GPU support
//!
//! This module provides a unified interface for tensor operations that can be
//! backed by either Candle (Metal/CUDA) or Burn (Vulkan/WGPU for AMD GPUs).
//!
//! # Usage
//!
//! The default backend uses Candle. For AMD GPU acceleration, enable the
//! `amd-gpu` feature which adds Burn with Vulkan/WGPU support.
//!
//! ```bash
//! # Default (CPU with candle)
//! cargo build --release
//!
//! # AMD GPU via Vulkan
//! cargo build --release --features amd-gpu
//! ```

pub mod candle_backend;

#[cfg(feature = "amd-gpu")]
pub mod burn_backend;

#[cfg(feature = "amd-gpu")]
pub mod gpu_compute;

#[cfg(feature = "amd-gpu")]
pub mod gpu_parallel;

use std::fmt::Debug;

/// Unified tensor type that abstracts over different backends
pub trait TensorBackend: Clone + Debug + Send + Sync + 'static {
    /// The underlying tensor type
    type Tensor: Clone + Debug + Send + Sync;
    
    /// The device type (CPU, GPU, etc.)
    type Device: Clone + Debug + Send + Sync;
    
    /// Error type for operations
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Get the default device
    fn default_device() -> Self::Device;
    
    /// Create a tensor filled with zeros
    fn zeros(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error>;
    
    /// Create a tensor filled with ones
    fn ones(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error>;
    
    /// Create a tensor with random values from normal distribution
    fn randn(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error>;
    
    /// Create a tensor with random values from uniform distribution [0, 1)
    fn rand(shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error>;
    
    /// Matrix multiplication
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Element-wise addition
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Element-wise multiplication
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Softmax along a dimension
    fn softmax(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error>;
    
    /// Transpose
    fn transpose(tensor: &Self::Tensor, dim1: usize, dim2: usize) -> Result<Self::Tensor, Self::Error>;
    
    /// Reshape tensor
    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    
    /// Get tensor shape
    fn shape(tensor: &Self::Tensor) -> Vec<usize>;
    
    /// Convert tensor to f32 vec (for debugging/logging)
    fn to_vec(tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;
    
    /// Create tensor from f32 slice
    fn from_slice(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, Self::Error>;
    
    /// Scale tensor by scalar
    fn scale(tensor: &Self::Tensor, scalar: f32) -> Result<Self::Tensor, Self::Error>;
    
    /// Exponential
    fn exp(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Natural logarithm
    fn log(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Square root
    fn sqrt(tensor: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    
    /// Mean along dimension
    fn mean(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error>;
    
    /// Sum along dimension
    fn sum(tensor: &Self::Tensor, dim: i64) -> Result<Self::Tensor, Self::Error>;
    
    /// Check if using GPU
    fn is_gpu() -> bool;
    
    /// Backend name for logging
    fn backend_name() -> &'static str;
}

// Re-export the candle backend as the default
pub use candle_backend::CandleBackend as DefaultBackend;

// Also export Burn backend when amd-gpu feature is enabled
#[cfg(feature = "amd-gpu")]
pub use burn_backend::BurnVulkanBackend as AmdGpuBackend;

// Export GPU compute module
#[cfg(feature = "amd-gpu")]
pub use gpu_compute::{GpuCompute, GpuError};

// Export GPU parallel processor
#[cfg(feature = "amd-gpu")]
pub use gpu_parallel::{GpuAttention, GpuParallelStreamProcessor};
