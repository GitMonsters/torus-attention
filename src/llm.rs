//! # Torus LLM - Full Language Model Architecture
//!
//! This module provides a complete language model built on the Torus Attention mechanism,
//! suitable for text generation tasks. It includes:
//!
//! - Token embeddings with learned positional encoding
//! - Feed-forward network (FFN) layers with GELU activation
//! - Transformer blocks combining Torus Attention + FFN + Layer Normalization
//! - Full model with vocabulary projection for next-token prediction
//! - Text generation with various sampling strategies
//!
//! ## Architecture
//!
//! ```text
//! Input Tokens
//!      │
//!      ▼
//! ┌─────────────┐
//! │  Embedding  │  (token + position)
//! └─────────────┘
//!      │
//!      ▼
//! ┌─────────────┐
//! │  Dropout    │
//! └─────────────┘
//!      │
//!      ▼
//! ┌─────────────────────────────┐
//! │     Transformer Block ×N    │
//! │  ┌───────────────────────┐  │
//! │  │ LayerNorm + TorusAttn │  │
//! │  │         + Residual    │  │
//! │  └───────────────────────┘  │
//! │  ┌───────────────────────┐  │
//! │  │ LayerNorm + FFN       │  │
//! │  │         + Residual    │  │
//! │  └───────────────────────┘  │
//! └─────────────────────────────┘
//!      │
//!      ▼
//! ┌─────────────┐
//! │ Final LNorm │
//! └─────────────┘
//!      │
//!      ▼
//! ┌─────────────┐
//! │   LM Head   │  → Vocab logits
//! └─────────────┘
//! ```

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{
    embedding, linear, Dropout, Embedding, Linear, Module, VarBuilder,
    VarMap,
};
use serde::{Deserialize, Serialize};

use crate::error::TorusError;
use crate::integration::{BidirectionalTorusConfig, BidirectionalTorusLayer};
use crate::rmsnorm::{rms_norm, RmsNorm};
use crate::TorusResult;

/// Configuration for the Torus LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorusLLMConfig {
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Hidden dimension (model width)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// FFN intermediate dimension (typically 4x hidden_dim)
    pub ffn_dim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Whether to tie input/output embeddings
    pub tie_embeddings: bool,
    /// Torus-specific: major radius
    pub torus_major_radius: f64,
    /// Torus-specific: minor radius
    pub torus_minor_radius: f64,
    /// EMA alpha for compounding
    pub ema_alpha: f64,
    /// Enable coherence module
    pub use_coherence: bool,
}

impl Default for TorusLLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            max_seq_len: 2048,
            hidden_dim: 768, // GPT-2 small
            num_layers: 12,
            num_heads: 12,
            ffn_dim: 3072, // 4x hidden_dim
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            tie_embeddings: true,
            torus_major_radius: 2.0,
            torus_minor_radius: 1.0,
            ema_alpha: 0.9,
            use_coherence: true,
        }
    }
}

impl TorusLLMConfig {
    /// Create a tiny config for testing
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            max_seq_len: 512, // 32 * 16 = 512 to match torus grid
            hidden_dim: 256,  // Match d_model in BidirectionalTorusConfig
            num_layers: 2,
            num_heads: 8, // Match n_heads in BidirectionalTorusConfig
            ffn_dim: 1024,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            tie_embeddings: true,
            torus_major_radius: 2.0,
            torus_minor_radius: 1.0,
            ema_alpha: 0.9,
            use_coherence: false,
        }
    }

    /// Create a small config (similar to GPT-2 small)
    pub fn small() -> Self {
        Self::default()
    }

    /// Create a medium config
    pub fn medium() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 2048,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            ffn_dim: 4096,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            tie_embeddings: true,
            torus_major_radius: 2.0,
            torus_minor_radius: 1.0,
            ema_alpha: 0.9,
            use_coherence: true,
        }
    }

    /// Convert to BidirectionalTorusConfig for attention layers
    ///
    /// The torus grid size is determined by the sequence length:
    /// seq_len = n_major * n_minor
    pub fn to_torus_config(&self) -> BidirectionalTorusConfig {
        // Calculate grid sizes to fit the max sequence length
        // We want n_major * n_minor >= max_seq_len
        // Use factors that work well with the hidden_dim
        let n_major = 32;
        let n_minor = 16;

        BidirectionalTorusConfig {
            d_model: self.hidden_dim,
            d_ff: self.ffn_dim,
            n_heads: self.num_heads,
            n_layers: self.num_layers,
            n_major,
            n_minor,
            major_radius: self.torus_major_radius,
            minor_radius: self.torus_minor_radius,
            dropout: self.dropout,
            use_coherence: self.use_coherence,
            ema_alpha: self.ema_alpha,
            ..Default::default()
        }
    }
}

/// Feed-Forward Network with GELU activation
#[derive(Debug)]
pub struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(
        hidden_dim: usize,
        ffn_dim: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let up_proj = linear(hidden_dim, ffn_dim, vb.pp("up_proj"))?;
        let down_proj = linear(ffn_dim, hidden_dim, vb.pp("down_proj"))?;
        let dropout = Dropout::new(dropout as f32);
        Ok(Self {
            up_proj,
            down_proj,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.up_proj.forward(x)?;
        let x = x.gelu_erf()?;
        let x = self.down_proj.forward(&x)?;
        self.dropout.forward(&x, false)
    }
}

/// Single Transformer Block: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
#[derive(Debug)]
pub struct TransformerBlock {
    ln1: RmsNorm,
    attention: BidirectionalTorusLayer,
    ln2: RmsNorm,
    ffn: FeedForward,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(
        config: &TorusLLMConfig,
        layer_idx: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> TorusResult<Self> {
        let ln1 = rms_norm(config.hidden_dim, config.layer_norm_eps, vb.pp("ln1"))?;
        let ln2 = rms_norm(config.hidden_dim, config.layer_norm_eps, vb.pp("ln2"))?;

        let torus_config = config.to_torus_config();
        let attention =
            BidirectionalTorusLayer::new(layer_idx, torus_config, vb.pp("attention"), device)?;

        let ffn = FeedForward::new(
            config.hidden_dim,
            config.ffn_dim,
            config.dropout,
            vb.pp("ffn"),
        )?;

        let dropout = Dropout::new(config.dropout as f32);

        Ok(Self {
            ln1,
            attention,
            ln2,
            ffn,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> TorusResult<Tensor> {
        // Pre-norm attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attention.forward(&normed)?;
        let attn_out = self.dropout.forward(&attn_out, false)?;
        let x = (x + attn_out)?;

        // Pre-norm FFN with residual
        let normed = self.ln2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        let ffn_out = self.dropout.forward(&ffn_out, false)?;
        let x = (x + ffn_out)?;

        Ok(x)
    }
}

/// The full Torus LLM model
#[derive(Debug)]
pub struct TorusLLM {
    /// Model configuration
    config: TorusLLMConfig,
    /// Token embeddings
    token_emb: Embedding,
    /// Positional embeddings
    pos_emb: Embedding,
    /// Embedding dropout
    emb_dropout: Dropout,
    /// Transformer blocks
    layers: Vec<TransformerBlock>,
    /// Final layer norm (RMSNorm for Metal compatibility)
    final_ln: RmsNorm,
    /// Language model head (projects to vocab)
    lm_head: Option<Linear>,
    /// Device
    device: Device,
}

impl TorusLLM {
    /// Create a new Torus LLM from configuration
    pub fn new(config: TorusLLMConfig, vb: VarBuilder) -> TorusResult<Self> {
        let device = vb.device().clone();

        // Get the torus sequence length (n_major * n_minor)
        // Position embeddings need to cover the full torus grid
        let torus_config = config.to_torus_config();
        let torus_seq_len = torus_config.seq_len();
        let pos_emb_size = torus_seq_len.max(config.max_seq_len);

        // Token and position embeddings
        let token_emb = embedding(config.vocab_size, config.hidden_dim, vb.pp("token_emb"))?;
        let pos_emb = embedding(pos_emb_size, config.hidden_dim, vb.pp("pos_emb"))?;
        let emb_dropout = Dropout::new(config.dropout as f32);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = TransformerBlock::new(&config, i, vb.pp(format!("layer_{}", i)), &device)?;
            layers.push(block);
        }

        // Final layer norm (RMSNorm)
        let final_ln = rms_norm(config.hidden_dim, config.layer_norm_eps, vb.pp("final_ln"))?;

        // LM head (only if not tying embeddings)
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(linear(
                config.hidden_dim,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        Ok(Self {
            config,
            token_emb,
            pos_emb,
            emb_dropout,
            layers,
            final_ln,
            lm_head,
            device,
        })
    }

    /// Create with a new VarMap (for training from scratch)
    pub fn new_random(config: TorusLLMConfig, device: &Device) -> TorusResult<(Self, VarMap)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = Self::new(config, vb)?;
        Ok((model, varmap))
    }

    /// Forward pass - returns logits over vocabulary
    ///
    /// Note: The torus attention mechanism expects sequences of length n_major * n_minor (512).
    /// Shorter sequences are automatically padded, and the output is truncated to the original length.
    pub fn forward(&self, input_ids: &Tensor) -> TorusResult<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Get expected sequence length from torus config
        let torus_config = self.config.to_torus_config();
        let torus_seq_len = torus_config.seq_len();

        // Check sequence length
        if seq_len > torus_seq_len {
            return Err(TorusError::Dimension(format!(
                "Sequence length {} exceeds torus grid size {}",
                seq_len, torus_seq_len
            )));
        }

        // Pad input if needed
        let (input_ids, padded) = if seq_len < torus_seq_len {
            // Pad with zeros (or pad token)
            let pad_len = torus_seq_len - seq_len;
            let padding = Tensor::zeros((batch_size, pad_len), DType::U32, &self.device)?;
            let padded_input = Tensor::cat(&[input_ids, &padding], 1)?;
            (padded_input, true)
        } else {
            (input_ids.clone(), false)
        };

        let actual_seq_len = if padded { torus_seq_len } else { seq_len };

        // Get embeddings
        let token_emb = self.token_emb.forward(&input_ids)?;

        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::arange(0u32, actual_seq_len as u32, &self.device)?;
        let pos_emb = self.pos_emb.forward(&positions)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as((
            batch_size,
            actual_seq_len,
            self.config.hidden_dim,
        ))?;

        // Combine embeddings
        let mut hidden = (token_emb + pos_emb)?;
        hidden = self.emb_dropout.forward(&hidden, false)?;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer norm
        hidden = self.final_ln.forward(&hidden)?;

        // Project to vocabulary
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&hidden)?
        } else {
            // Tied embeddings: multiply by token embedding weights transposed
            // hidden: [batch, seq, hidden_dim]
            // emb_weights: [vocab_size, hidden_dim]
            // We want: hidden @ emb_weights.T -> [batch, seq, vocab_size]
            let emb_weights = self.token_emb.embeddings();
            // Use broadcast_matmul for batched operation
            hidden.broadcast_matmul(&emb_weights.t()?)?
        };

        // Truncate back to original sequence length if we padded
        let logits = if padded {
            logits.i((.., ..seq_len, ..))?
        } else {
            logits
        };

        Ok(logits)
    }

    /// Get the configuration
    pub fn config(&self) -> &TorusLLMConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Sampling strategy for text generation
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Always pick the most likely token
    Greedy,
    /// Sample from top-k most likely tokens
    TopK(usize),
    /// Sample from tokens with cumulative probability <= p
    TopP(f64),
    /// Temperature-scaled sampling
    Temperature(f64),
    /// Combined top-k, top-p, and temperature
    Combined {
        top_k: usize,
        top_p: f64,
        temperature: f64,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Combined {
            top_k: 50,
            top_p: 0.95,
            temperature: 0.8,
        }
    }
}

/// Text generator using a Torus LLM
pub struct TextGenerator {
    model: TorusLLM,
    strategy: SamplingStrategy,
}

impl TextGenerator {
    pub fn new(model: TorusLLM, strategy: SamplingStrategy) -> Self {
        Self { model, strategy }
    }

    /// Generate tokens given input token IDs
    /// Returns the generated token IDs (excluding the prompt)
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        stop_token: Option<u32>,
    ) -> TorusResult<Vec<u32>> {
        let device = self.model.device();
        let mut tokens: Vec<u32> = prompt_ids.to_vec();
        let mut generated = Vec::new();

        for _ in 0..max_new_tokens {
            // Truncate to max sequence length if needed
            let start = if tokens.len() > self.model.config.max_seq_len {
                tokens.len() - self.model.config.max_seq_len
            } else {
                0
            };
            let context = &tokens[start..];

            // Create input tensor
            let input = Tensor::new(context, device)?.unsqueeze(0)?;

            // Forward pass
            let logits = self.model.forward(&input)?;

            // Get logits for the last position
            let last_logits = logits.i((0, context.len() - 1, ..))?;

            // Sample next token
            let next_token = self.sample(&last_logits)?;

            // Check for stop token
            if let Some(stop) = stop_token {
                if next_token == stop {
                    break;
                }
            }

            tokens.push(next_token);
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Sample a token from logits using the configured strategy
    fn sample(&self, logits: &Tensor) -> TorusResult<u32> {
        let logits = match self.strategy {
            SamplingStrategy::Temperature(t)
            | SamplingStrategy::Combined { temperature: t, .. }
                if t != 1.0 =>
            {
                (logits / t)?
            }
            _ => logits.clone(),
        };

        match self.strategy {
            SamplingStrategy::Greedy => {
                let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
                Ok(token)
            }
            SamplingStrategy::TopK(k) => self.sample_top_k(&logits, k),
            SamplingStrategy::TopP(p) => self.sample_top_p(&logits, p),
            SamplingStrategy::Temperature(_) => self.sample_categorical(&logits),
            SamplingStrategy::Combined { top_k, top_p, .. } => {
                // Apply top-k first, then top-p
                let filtered = self.apply_top_k(&logits, top_k)?;
                self.sample_top_p(&filtered, top_p)
            }
        }
    }

    fn sample_top_k(&self, logits: &Tensor, k: usize) -> TorusResult<u32> {
        let filtered = self.apply_top_k(logits, k)?;
        self.sample_categorical(&filtered)
    }

    fn apply_top_k(&self, logits: &Tensor, k: usize) -> TorusResult<Tensor> {
        let vocab_size = logits.dim(D::Minus1)?;
        let k = k.min(vocab_size);

        // Get top-k values and indices
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = indexed[k - 1].1;

        // Mask out tokens below threshold
        let mask: Vec<f32> = logits_vec
            .iter()
            .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
            .collect();

        Ok(Tensor::new(mask, logits.device())?)
    }

    fn sample_top_p(&self, logits: &Tensor, p: f64) -> TorusResult<u32> {
        // Softmax to get probabilities
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff for cumulative probability
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumsum += *prob as f64;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Create masked logits
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let valid_indices: std::collections::HashSet<usize> =
            indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();

        let masked: Vec<f32> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if valid_indices.contains(&i) {
                    v
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();

        let masked_tensor = Tensor::new(masked, logits.device())?;
        self.sample_categorical(&masked_tensor)
    }

    fn sample_categorical(&self, logits: &Tensor) -> TorusResult<u32> {
        // Softmax to get probabilities
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Sample using random number
        let r: f32 = rand_float();
        let mut cumsum = 0.0;
        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probs_vec.len() - 1) as u32)
    }
}

/// Simple random float generator (0.0 to 1.0)
fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32 / u32::MAX as f32).fract()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = TorusLLMConfig::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_tiny_model_creation() {
        let device = Device::Cpu;
        let config = TorusLLMConfig::tiny();
        let (model, _varmap) = TorusLLM::new_random(config.clone(), &device).unwrap();
        assert_eq!(model.config().vocab_size, 1000);
        assert_eq!(model.config().num_layers, 2);
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::Cpu;
        let config = TorusLLMConfig::tiny();
        let (model, _varmap) = TorusLLM::new_random(config.clone(), &device).unwrap();

        // Create dummy input
        let batch_size = 2;
        let seq_len = 16;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        // Forward pass
        let logits = model.forward(&input_ids).unwrap();

        // Check output shape
        let (b, s, v) = logits.dims3().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);
        assert_eq!(v, config.vocab_size);
    }

    #[test]
    fn test_text_generator() {
        let device = Device::Cpu;
        let config = TorusLLMConfig::tiny();
        let (model, _varmap) = TorusLLM::new_random(config, &device).unwrap();

        let generator = TextGenerator::new(model, SamplingStrategy::Greedy);

        // Generate from a simple prompt
        let prompt = vec![1u32, 2, 3, 4, 5];
        let generated = generator.generate(&prompt, 10, None).unwrap();

        assert!(!generated.is_empty());
        assert!(generated.len() <= 10);
    }
}
