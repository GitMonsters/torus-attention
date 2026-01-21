//! # RMSNorm - Root Mean Square Layer Normalization
//!
//! A Metal-compatible alternative to LayerNorm, used by LLaMA and other modern LLMs.
//! RMSNorm normalizes using only the root mean square, without the mean subtraction
//! that LayerNorm uses. This makes it more efficient and easier to implement on Metal.
//!
//! ## Formula
//! ```text
//! RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//! ```
//!
//! ## Metal Compatibility
//! Unlike `candle_nn::LayerNorm`, RMSNorm only uses basic tensor operations
//! (mul, mean, sqrt, div) which are all supported by candle's Metal backend.

use candle_core::{DType, Result as CandleResult, Tensor, D};
use candle_nn::{Init, VarBuilder, VarMap};

/// RMSNorm layer - Metal-compatible alternative to LayerNorm
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMSNorm layer
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }

    /// Create RMSNorm with a fresh VarMap (for testing)
    #[allow(dead_code)]
    pub fn new_random(hidden_size: usize, eps: f64, device: &candle_core::Device) -> CandleResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::new(hidden_size, eps, vb)
    }

    /// Forward pass
    ///
    /// Computes: x * weight / sqrt(mean(x^2) + eps)
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Compute x^2
        let x_sq = x.sqr()?;
        
        // Compute mean of x^2 along last dimension, keeping dims
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        
        // Compute sqrt(mean(x^2) + eps)
        let rms = (mean_sq + self.eps)?.sqrt()?;
        
        // Normalize: x / rms
        let x_norm = x.broadcast_div(&rms)?;
        
        // Scale by weight
        x_norm.broadcast_mul(&self.weight)
    }
}

/// Helper function to create RMSNorm (mimics candle_nn::layer_norm API)
pub fn rms_norm(hidden_size: usize, eps: f64, vb: VarBuilder) -> CandleResult<RmsNorm> {
    RmsNorm::new(hidden_size, eps, vb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rmsnorm_basic() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let norm = RmsNorm::new_random(hidden_size, 1e-5, &device).unwrap();

        // Create input tensor [batch=2, seq=4, hidden=64]
        let x = Tensor::randn(0f32, 1.0, (2, 4, hidden_size), &device).unwrap();
        
        let output = norm.forward(&x).unwrap();
        
        // Output shape should match input
        assert_eq!(x.dims(), output.dims());
    }

    #[test]
    fn test_rmsnorm_normalization() {
        let device = Device::Cpu;
        let hidden_size = 8;
        let norm = RmsNorm::new_random(hidden_size, 1e-5, &device).unwrap();

        // Create a known input
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &device).unwrap();
        
        let output = norm.forward(&x).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        
        // Output should have reasonable values (normalized)
        for v in &output_vec {
            assert!(v.abs() < 10.0, "Output value {} is too large", v);
        }
    }

    #[test]
    fn test_rmsnorm_vs_manual() {
        let device = Device::Cpu;
        let hidden_size = 4;
        
        // Create norm with weight=1.0
        let norm = RmsNorm::new_random(hidden_size, 1e-5, &device).unwrap();
        
        // Input: [1, 2, 3, 4]
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device).unwrap();
        
        // Manual calculation:
        // x^2 = [1, 4, 9, 16]
        // mean(x^2) = 7.5
        // rms = sqrt(7.5 + 1e-5) ≈ 2.7386
        // x_norm = x / rms ≈ [0.365, 0.730, 1.095, 1.461]
        
        let output = norm.forward(&x).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        
        let expected_rms = (7.5f32 + 1e-5).sqrt();
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0].iter().map(|v| v / expected_rms).collect();
        
        for (got, exp) in output_vec.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "Got {} expected {}", got, exp);
        }
    }
}
