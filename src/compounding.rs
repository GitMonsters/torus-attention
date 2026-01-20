//! Learnable EMA Compounding for Torus Attention
//!
//! Implements exponential moving average (EMA) compounding with learnable
//! decay rates per layer. This allows information to accumulate across
//! layers with adaptive smoothing.
//!
//! Formula: h_t = α * new + (1 - α) * h_{t-1}
//!
//! Key features:
//! - Learnable α per layer (via sigmoid of learned parameter)
//! - Layer-wise decay scaling
//! - Momentum variants (Nesterov, Adam-style)
//! - State management for sequential processing

use crate::error::TorusError;
use crate::TorusResult;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

/// Configuration for EMA compounding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundingConfig {
    /// Number of layers to track
    pub n_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Base alpha value (before learning)
    pub base_alpha: f64,
    /// Minimum alpha (prevents complete forgetting)
    pub min_alpha: f64,
    /// Maximum alpha (prevents no memory)
    pub max_alpha: f64,
    /// Layer-wise alpha scaling factor (α_l = base * scale^l)
    pub layer_scale: f64,
    /// Whether to use momentum
    pub use_momentum: bool,
    /// Momentum coefficient (β for velocity term)
    pub momentum_beta: f64,
    /// Whether to use layer-specific learned alphas
    pub learnable_alpha: bool,
}

impl Default for CompoundingConfig {
    fn default() -> Self {
        Self {
            n_layers: 6,
            d_model: 256,
            base_alpha: 0.9,
            min_alpha: 0.1,
            max_alpha: 0.99,
            layer_scale: 0.95,
            use_momentum: true,
            momentum_beta: 0.9,
            learnable_alpha: true,
        }
    }
}

/// Single layer's compounding state
#[derive(Debug, Clone)]
pub struct LayerState {
    /// Accumulated hidden state
    hidden: Tensor,
    /// Velocity term for momentum (if enabled)
    velocity: Option<Tensor>,
    /// Number of updates applied
    step_count: usize,
}

impl LayerState {
    /// Create a new zeroed state
    pub fn new(shape: &[usize], device: &Device, with_momentum: bool) -> TorusResult<Self> {
        let hidden = Tensor::zeros(shape, DType::F32, device)?;
        let velocity = if with_momentum {
            Some(Tensor::zeros(shape, DType::F32, device)?)
        } else {
            None
        };
        Ok(Self {
            hidden,
            velocity,
            step_count: 0,
        })
    }

    /// Reset state to zeros
    pub fn reset(&mut self, device: &Device) -> TorusResult<()> {
        let shape = self.hidden.dims().to_vec();
        self.hidden = Tensor::zeros(shape.as_slice(), DType::F32, device)?;
        if let Some(ref mut vel) = self.velocity {
            *vel = Tensor::zeros(shape.as_slice(), DType::F32, device)?;
        }
        self.step_count = 0;
        Ok(())
    }

    /// Get current hidden state
    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    /// Get step count
    pub fn steps(&self) -> usize {
        self.step_count
    }
}

/// Learnable alpha parameter with sigmoid constraint
#[derive(Debug)]
pub struct LearnableAlpha {
    /// Raw parameter (before sigmoid)
    raw: Tensor,
    /// Minimum alpha
    min_alpha: f64,
    /// Maximum alpha
    max_alpha: f64,
}

impl LearnableAlpha {
    /// Create with initial alpha value
    pub fn new(initial_alpha: f64, min_alpha: f64, max_alpha: f64, device: &Device) -> TorusResult<Self> {
        // Clamp initial_alpha to valid range
        let clamped_alpha = initial_alpha.clamp(min_alpha, max_alpha);
        // Inverse sigmoid to get raw value from desired alpha
        let normalized = (clamped_alpha - min_alpha) / (max_alpha - min_alpha);
        // Clamp normalized to avoid ln(0) or ln(inf)
        let normalized = normalized.clamp(0.001, 0.999);
        let raw_value = (normalized / (1.0 - normalized)).ln();
        let raw = Tensor::new(&[raw_value as f32], device)?;
        
        Ok(Self {
            raw,
            min_alpha,
            max_alpha,
        })
    }

    /// Create from VarBuilder
    pub fn from_vb(vb: VarBuilder, min_alpha: f64, max_alpha: f64) -> TorusResult<Self> {
        let raw = vb.get((1,), "alpha_raw")?;
        Ok(Self {
            raw,
            min_alpha,
            max_alpha,
        })
    }

    /// Get current alpha value (constrained)
    pub fn get(&self) -> TorusResult<f64> {
        let sigmoid: Vec<f32> = candle_nn::ops::sigmoid(&self.raw)?.to_vec1()?;
        let s = sigmoid[0] as f64;
        Ok(self.min_alpha + s * (self.max_alpha - self.min_alpha))
    }

    /// Get raw tensor for optimization
    pub fn raw(&self) -> &Tensor {
        &self.raw
    }

    /// Get alpha as tensor for computation
    pub fn as_tensor(&self) -> TorusResult<Tensor> {
        let sigmoid = candle_nn::ops::sigmoid(&self.raw)?;
        let range = (self.max_alpha - self.min_alpha) as f32;
        let min = self.min_alpha as f32;
        let range_tensor = Tensor::new(&[range], sigmoid.device())?;
        let min_tensor = Tensor::new(&[min], sigmoid.device())?;
        let alpha = (sigmoid.mul(&range_tensor)? + min_tensor)?;
        Ok(alpha)
    }
}

/// EMA compounding layer
#[derive(Debug)]
pub struct EMACompounding {
    /// Per-layer learned alpha values
    alphas: Vec<LearnableAlpha>,
    /// Per-layer states
    states: Vec<Option<LayerState>>,
    /// Momentum beta values (if momentum enabled)
    momentum_betas: Option<Vec<LearnableAlpha>>,
    /// Configuration
    config: CompoundingConfig,
    /// Device
    device: Device,
}

impl EMACompounding {
    /// Create new EMA compounding module
    pub fn new(config: CompoundingConfig, vb: VarBuilder, device: &Device) -> TorusResult<Self> {
        let mut alphas = Vec::with_capacity(config.n_layers);
        
        for l in 0..config.n_layers {
            // Initial alpha with layer-wise scaling
            let layer_alpha = config.base_alpha * config.layer_scale.powi(l as i32);
            let layer_alpha = layer_alpha.clamp(config.min_alpha, config.max_alpha);
            
            let alpha = if config.learnable_alpha {
                LearnableAlpha::from_vb(
                    vb.pp(format!("layer_{}_alpha", l)),
                    config.min_alpha,
                    config.max_alpha,
                )?
            } else {
                LearnableAlpha::new(layer_alpha, config.min_alpha, config.max_alpha, device)?
            };
            alphas.push(alpha);
        }

        // Initialize momentum betas if enabled
        let momentum_betas = if config.use_momentum && config.learnable_alpha {
            let mut betas = Vec::with_capacity(config.n_layers);
            for l in 0..config.n_layers {
                let beta = LearnableAlpha::from_vb(
                    vb.pp(format!("layer_{}_beta", l)),
                    0.5,
                    0.999,
                )?;
                betas.push(beta);
            }
            Some(betas)
        } else {
            None
        };

        // States initialized lazily on first forward pass
        let states = vec![None; config.n_layers];

        Ok(Self {
            alphas,
            states,
            momentum_betas,
            config,
            device: device.clone(),
        })
    }

    /// Initialize state for a layer
    fn init_state(&mut self, layer: usize, shape: &[usize]) -> TorusResult<()> {
        if self.states[layer].is_none() {
            self.states[layer] = Some(LayerState::new(
                shape,
                &self.device,
                self.config.use_momentum,
            )?);
        }
        Ok(())
    }

    /// Apply EMA compounding for a layer
    pub fn compound(&mut self, layer: usize, new_value: &Tensor) -> TorusResult<Tensor> {
        if layer >= self.config.n_layers {
            return Err(TorusError::InvalidParameter(format!(
                "Layer {} exceeds configured layers {}",
                layer, self.config.n_layers
            )));
        }

        // Initialize state if needed
        self.init_state(layer, new_value.dims())?;

        let alpha = self.alphas[layer].as_tensor()?;
        let one_minus_alpha = 1.0 - alpha.to_vec1::<f32>()?[0] as f64;

        let state = self.states[layer].as_mut().unwrap();
        
        let h_new = if self.config.use_momentum {
            // Momentum-enhanced EMA
            let beta = if let Some(ref betas) = self.momentum_betas {
                betas[layer].get()?
            } else {
                self.config.momentum_beta
            };

            // Update velocity: v_t = β * v_{t-1} + (1 - β) * (new - h_{t-1})
            let diff = (new_value - &state.hidden)?;
            let vel = state.velocity.as_ref().unwrap();
            let new_velocity = ((vel * beta)? + (&diff * (1.0 - beta))?)?;
            
            // Update hidden with momentum: h_t = h_{t-1} + α * v_t
            let alpha_val = alpha.to_vec1::<f32>()?[0] as f64;
            let h = (&state.hidden + (&new_velocity * alpha_val)?)?;
            
            state.velocity = Some(new_velocity);
            h
        } else {
            // Standard EMA: h_t = α * new + (1 - α) * h_{t-1}
            let weighted_new = (new_value * alpha.to_vec1::<f32>()?[0] as f64)?;
            let weighted_old = (&state.hidden * one_minus_alpha)?;
            (weighted_new + weighted_old)?
        };

        state.hidden = h_new.clone();
        state.step_count += 1;

        Ok(h_new)
    }

    /// Apply EMA compounding with an externally-provided alpha value
    /// 
    /// This is used for coherence-modulated compounding where the alpha
    /// is dynamically adjusted based on coherence metrics (SOC/SMM).
    pub fn compound_with_alpha(&mut self, layer: usize, new_value: &Tensor, external_alpha: f64) -> TorusResult<Tensor> {
        if layer >= self.config.n_layers {
            return Err(TorusError::InvalidParameter(format!(
                "Layer {} exceeds configured layers {}",
                layer, self.config.n_layers
            )));
        }

        // Initialize state if needed
        self.init_state(layer, new_value.dims())?;

        // Use the externally-provided alpha instead of the learned one
        let alpha = external_alpha.clamp(self.config.min_alpha, self.config.max_alpha);
        let one_minus_alpha = 1.0 - alpha;

        let state = self.states[layer].as_mut().unwrap();
        
        let h_new = if self.config.use_momentum {
            // Momentum-enhanced EMA with external alpha
            let beta = if let Some(ref betas) = self.momentum_betas {
                betas[layer].get()?
            } else {
                self.config.momentum_beta
            };

            // Update velocity: v_t = β * v_{t-1} + (1 - β) * (new - h_{t-1})
            let diff = (new_value - &state.hidden)?;
            let vel = state.velocity.as_ref().unwrap();
            let new_velocity = ((vel * beta)? + (&diff * (1.0 - beta))?)?;
            
            // Update hidden with momentum: h_t = h_{t-1} + α * v_t
            let h = (&state.hidden + (&new_velocity * alpha)?)?;
            
            state.velocity = Some(new_velocity);
            h
        } else {
            // Standard EMA with external alpha: h_t = α * new + (1 - α) * h_{t-1}
            let weighted_new = (new_value * alpha)?;
            let weighted_old = (&state.hidden * one_minus_alpha)?;
            (weighted_new + weighted_old)?
        };

        state.hidden = h_new.clone();
        state.step_count += 1;

        Ok(h_new)
    }

    /// Compound with bias correction (Adam-style)
    pub fn compound_corrected(&mut self, layer: usize, new_value: &Tensor) -> TorusResult<Tensor> {
        let h = self.compound(layer, new_value)?;
        
        let state = self.states[layer].as_ref().unwrap();
        let t = state.step_count as f64;
        
        if t > 0.0 {
            let alpha = self.alphas[layer].get()?;
            // Bias correction: h_corrected = h / (1 - α^t)
            let correction = 1.0 - alpha.powi(t as i32);
            if correction > 1e-8 {
                return Ok((h / correction)?);
            }
        }
        
        Ok(h)
    }

    /// Reset all states
    pub fn reset(&mut self) -> TorusResult<()> {
        for state in &mut self.states {
            if let Some(ref mut s) = state {
                s.reset(&self.device)?;
            }
        }
        Ok(())
    }

    /// Reset specific layer state
    pub fn reset_layer(&mut self, layer: usize) -> TorusResult<()> {
        if let Some(ref mut state) = self.states[layer] {
            state.reset(&self.device)?;
        }
        Ok(())
    }

    /// Get current alpha for a layer
    pub fn get_alpha(&self, layer: usize) -> TorusResult<f64> {
        self.alphas[layer].get()
    }

    /// Get all alpha values
    pub fn get_all_alphas(&self) -> TorusResult<Vec<f64>> {
        self.alphas.iter().map(|a| a.get()).collect()
    }

    /// Get state for a layer
    pub fn get_state(&self, layer: usize) -> Option<&LayerState> {
        self.states[layer].as_ref()
    }

    /// Get all learnable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params: Vec<&Tensor> = self.alphas.iter().map(|a| a.raw()).collect();
        if let Some(ref betas) = self.momentum_betas {
            params.extend(betas.iter().map(|b| b.raw()));
        }
        params
    }
}

/// Multi-scale compounding with different time constants
#[derive(Debug)]
pub struct MultiScaleCompounding {
    /// Fast compounding (high alpha, responsive)
    fast: EMACompounding,
    /// Medium compounding (moderate alpha)
    medium: EMACompounding,
    /// Slow compounding (low alpha, long memory)
    slow: EMACompounding,
    /// Weights for combining scales
    scale_weights: Tensor,
    /// Temperature for scale weight softmax
    temperature: f64,
}

impl MultiScaleCompounding {
    pub fn new(base_config: CompoundingConfig, vb: VarBuilder, device: &Device) -> TorusResult<Self> {
        // Fast scale: high alpha (0.95)
        let mut fast_config = base_config.clone();
        fast_config.base_alpha = 0.95;
        fast_config.min_alpha = 0.9;
        let fast = EMACompounding::new(fast_config, vb.pp("fast"), device)?;

        // Medium scale: moderate alpha (0.8)
        let mut medium_config = base_config.clone();
        medium_config.base_alpha = 0.8;
        medium_config.min_alpha = 0.6;
        medium_config.max_alpha = 0.9;
        let medium = EMACompounding::new(medium_config, vb.pp("medium"), device)?;

        // Slow scale: low alpha (0.5)
        let mut slow_config = base_config;
        slow_config.base_alpha = 0.5;
        slow_config.min_alpha = 0.1;
        slow_config.max_alpha = 0.7;
        let slow = EMACompounding::new(slow_config, vb.pp("slow"), device)?;

        // Learnable scale weights
        let scale_weights = vb.get((3,), "scale_weights")?;

        Ok(Self {
            fast,
            medium,
            slow,
            scale_weights,
            temperature: 1.0,
        })
    }

    /// Compound at all scales and combine
    pub fn compound(&mut self, layer: usize, new_value: &Tensor) -> TorusResult<Tensor> {
        let h_fast = self.fast.compound(layer, new_value)?;
        let h_medium = self.medium.compound(layer, new_value)?;
        let h_slow = self.slow.compound(layer, new_value)?;

        // Weighted combination
        let weights = candle_nn::ops::softmax(&(&self.scale_weights / self.temperature)?, 0)?;
        
        let combined = (h_fast.broadcast_mul(&weights.i(0)?)?
            + h_medium.broadcast_mul(&weights.i(1)?)?
            + h_slow.broadcast_mul(&weights.i(2)?)?)?;

        Ok(combined)
    }

    /// Reset all scales
    pub fn reset(&mut self) -> TorusResult<()> {
        self.fast.reset()?;
        self.medium.reset()?;
        self.slow.reset()?;
        Ok(())
    }

    /// Get current scale weights
    pub fn get_scale_weights(&self) -> TorusResult<Vec<f32>> {
        let weights = candle_nn::ops::softmax(&(&self.scale_weights / self.temperature)?, 0)?;
        Ok(weights.to_vec1()?)
    }
}

/// Compounding statistics tracker
#[derive(Debug, Clone, Default)]
pub struct CompoundingStats {
    /// Running mean of hidden states per layer
    pub mean_hidden: Vec<f64>,
    /// Running variance of hidden states per layer
    pub var_hidden: Vec<f64>,
    /// Alpha values over time
    pub alpha_history: Vec<Vec<f64>>,
    /// Steps per layer
    pub steps: Vec<usize>,
}

impl CompoundingStats {
    pub fn new(n_layers: usize) -> Self {
        Self {
            mean_hidden: vec![0.0; n_layers],
            var_hidden: vec![0.0; n_layers],
            alpha_history: vec![Vec::new(); n_layers],
            steps: vec![0; n_layers],
        }
    }

    /// Update statistics from compounding module
    pub fn update(&mut self, compounding: &EMACompounding, layer: usize) -> TorusResult<()> {
        if let Some(state) = compounding.get_state(layer) {
            let hidden_flat: Vec<f32> = state.hidden().flatten_all()?.to_vec1()?;
            let n = hidden_flat.len() as f64;
            
            // Update running mean
            let mean: f64 = hidden_flat.iter().map(|&x| x as f64).sum::<f64>() / n;
            self.mean_hidden[layer] = mean;
            
            // Update running variance
            let var: f64 = hidden_flat
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            self.var_hidden[layer] = var;
            
            self.steps[layer] = state.steps();
        }
        
        // Track alpha
        let alpha = compounding.get_alpha(layer)?;
        self.alpha_history[layer].push(alpha);
        
        Ok(())
    }

    /// Get summary for a layer
    pub fn summary(&self, layer: usize) -> String {
        format!(
            "Layer {}: steps={}, mean={:.4}, var={:.4}, alphas={:?}",
            layer,
            self.steps[layer],
            self.mean_hidden[layer],
            self.var_hidden[layer],
            self.alpha_history[layer].last().unwrap_or(&0.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learnable_alpha() {
        let device = Device::Cpu;
        let alpha = LearnableAlpha::new(0.9, 0.1, 0.99, &device).unwrap();
        let value = alpha.get().unwrap();
        assert!((value - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_alpha_constraints() {
        let device = Device::Cpu;
        
        // Test that alpha stays within bounds
        let alpha_low = LearnableAlpha::new(0.05, 0.1, 0.99, &device).unwrap();
        let value_low = alpha_low.get().unwrap();
        assert!(value_low >= 0.1);
        
        let alpha_high = LearnableAlpha::new(1.0, 0.1, 0.99, &device).unwrap();
        let value_high = alpha_high.get().unwrap();
        assert!(value_high <= 0.99);
    }

    #[test]
    fn test_config_default() {
        let config = CompoundingConfig::default();
        assert_eq!(config.n_layers, 6);
        assert!(config.learnable_alpha);
        assert!(config.use_momentum);
    }
}
