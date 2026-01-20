//! Cognitive Coherence Module for Torus Attention
//!
//! Implements psychological coherence theory (Antonovsky's SOC) and cognitive
//! cohesion principles for adaptive compounding in transformer architectures.
//!
//! # Theoretical Foundation
//!
//! ## Cognitive Cohesion
//! Shared understanding among processing streams regarding goals (attention targets),
//! roles (stream specialization), and processes (information flow patterns).
//! Operationalized through Shared Mental Models (SMMs).
//!
//! ## Psychological Coherence (SOC)
//! Based on Antonovsky's Salutogenic model, SOC represents the internal consistency
//! that allows the system to perceive inputs as:
//! - **Comprehensible**: Predictable, structured, clear
//! - **Manageable**: Within capacity to process
//! - **Meaningful**: Worth attending to, significant
//!
//! # The Bridge: Shared Mental Models
//!
//! SMMs connect cognitive cohesion (inter-stream alignment) with psychological
//! coherence (intra-state stability):
//!
//! ```text
//!   Cognitive Cohesion          Shared Mental Models         Psychological Coherence
//!   (Stream Agreement)  ←────→  (Aligned Frameworks)  ←────→  (State Stability)
//!         │                            │                            │
//!         ↓                            ↓                            ↓
//!   - Reduce uncertainty         - Bridge between            - Comprehensibility
//!   - Strengthen belief           individual streams         - Manageability  
//!   - Reinforce purpose          - Enable anticipation       - Meaningfulness
//! ```
//!
//! # Usage
//!
//! The coherence module can be used standalone or integrated into the
//! `BidirectionalTorusTransformer`:
//!
//! ```rust
//! use torus_attention::{SenseOfCoherence, SharedMentalModel, CognitiveCoherenceLayer};
//! use candle_core::Device;
//!
//! // Standalone SOC
//! let soc = SenseOfCoherence::healthy();
//! println!("Score: {:.3}", soc.score());
//! println!("Adaptive alpha: {:.3}", soc.adaptive_alpha(0.9, 0.1, 0.99));
//!
//! // Standalone SMM
//! let smm = SharedMentalModel::for_torus_attention();
//! println!("Cohesion: {:.3}", smm.cognitive_cohesion());
//!
//! // Full coherence layer
//! let device = Device::Cpu;
//! let coherence = CognitiveCoherenceLayer::for_torus_attention(256, &device);
//! println!("{}", coherence.summary());
//! ```
//!
//! # Integration with Transformer
//!
//! When integrated into `BidirectionalTorusTransformer`:
//! 1. Each forward pass updates SOC based on attention entropy and concentration
//! 2. The SMM alignment matrix is updated from stream attention patterns
//! 3. An adaptive alpha is computed: `α = base * (0.5 + 0.5 * (0.6*SOC + 0.4*cohesion*SOC))`
//! 4. This alpha replaces the learned alpha in EMA compounding

use crate::TorusResult;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// SENSE OF COHERENCE (SOC) - Antonovsky's Model
// ═══════════════════════════════════════════════════════════════════════════════

/// Antonovsky's Sense of Coherence (SOC) components
/// 
/// SOC predicts resilience and well-being. In attention mechanisms:
/// - High SOC → stable, confident attention patterns
/// - Low SOC → scattered, uncertain attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseOfCoherence {
    /// Comprehensibility: perceived clarity and predictability
    /// High = attention distributions are sharp and consistent
    /// Low = high entropy, unpredictable patterns
    pub comprehensibility: f64,
    
    /// Manageability: belief in available resources to cope
    /// High = capacity exceeds demand (attention not saturated)
    /// Low = overwhelmed, capacity exceeded
    pub manageability: f64,
    
    /// Meaningfulness: emotional significance and engagement
    /// High = strong signal, worth attending to
    /// Low = noise, irrelevant input
    pub meaningfulness: f64,
}

impl SenseOfCoherence {
    /// Create a new SOC state
    pub fn new(comprehensibility: f64, manageability: f64, meaningfulness: f64) -> Self {
        Self {
            comprehensibility: comprehensibility.clamp(0.0, 1.0),
            manageability: manageability.clamp(0.0, 1.0),
            meaningfulness: meaningfulness.clamp(0.0, 1.0),
        }
    }
    
    /// Default healthy coherence state
    pub fn healthy() -> Self {
        Self::new(0.8, 0.8, 0.9)
    }
    
    /// Low coherence (stressed/uncertain state)
    pub fn stressed() -> Self {
        Self::new(0.3, 0.4, 0.5)
    }
    
    /// Compute overall SOC score
    /// Meaningfulness is often weighted highest in psychological literature
    pub fn score(&self) -> f64 {
        // Weighted combination (meaningfulness emphasized)
        0.25 * self.comprehensibility + 
        0.25 * self.manageability + 
        0.50 * self.meaningfulness
    }
    
    /// Compute adaptive alpha for EMA compounding
    /// 
    /// High coherence → higher alpha (trust accumulated state more)
    /// Low coherence → lower alpha (rely more on fresh input)
    pub fn adaptive_alpha(&self, base_alpha: f64, min_alpha: f64, max_alpha: f64) -> f64 {
        let soc = self.score();
        // Scale: SOC=0 → 50% of base, SOC=1 → 100% of base
        let scaled = base_alpha * (0.5 + 0.5 * soc);
        scaled.clamp(min_alpha, max_alpha)
    }
    
    /// Check if coherence is above threshold for stable processing
    pub fn is_coherent(&self, threshold: f64) -> bool {
        self.score() >= threshold
    }
    
    /// Blend two SOC states (useful for stream combination)
    pub fn blend(&self, other: &Self, weight: f64) -> Self {
        let w = weight.clamp(0.0, 1.0);
        Self::new(
            self.comprehensibility * (1.0 - w) + other.comprehensibility * w,
            self.manageability * (1.0 - w) + other.manageability * w,
            self.meaningfulness * (1.0 - w) + other.meaningfulness * w,
        )
    }
}

impl Default for SenseOfCoherence {
    fn default() -> Self {
        Self::healthy()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED MENTAL MODELS (SMM) - The Bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared Mental Model between processing streams
/// 
/// SMMs enable:
/// - Efficient communication (streams "understand" each other)
/// - Anticipation (predict what other streams will attend to)
/// - Adaptation under pressure (coordinated response to novelty)
#[derive(Debug, Clone)]
pub struct SharedMentalModel {
    /// Alignment scores between stream pairs (8x8 for 8 streams)
    pub alignment_matrix: Vec<Vec<f64>>,
    
    /// Number of streams
    pub n_streams: usize,
    
    /// Task model alignment (what we're trying to do)
    pub task_alignment: f64,
    
    /// Team model alignment (how we work together)  
    pub team_alignment: f64,
}

impl SharedMentalModel {
    /// Create a new SMM for n streams
    pub fn new(n_streams: usize) -> Self {
        // Initialize with moderate alignment
        let alignment_matrix = vec![vec![0.5; n_streams]; n_streams];
        Self {
            alignment_matrix,
            n_streams,
            task_alignment: 0.5,
            team_alignment: 0.5,
        }
    }
    
    /// Create SMM for the standard 8-stream torus attention
    pub fn for_torus_attention() -> Self {
        let mut smm = Self::new(8);
        
        // Initialize with known stream relationships
        // Forward-backward pairs should have complementary alignment
        let pairs = [(0, 1), (2, 3), (4, 5), (6, 7)];
        for (fwd, bwd) in pairs {
            smm.alignment_matrix[fwd][bwd] = 0.9; // High complementary alignment
            smm.alignment_matrix[bwd][fwd] = 0.9;
        }
        
        // Same-direction streams share more
        for i in (0..8).step_by(2) {
            for j in (0..8).step_by(2) {
                if i != j {
                    smm.alignment_matrix[i][j] = 0.7;
                }
            }
        }
        
        smm.task_alignment = 0.8;
        smm.team_alignment = 0.75;
        smm
    }
    
    /// Update alignment based on observed attention patterns
    pub fn update_from_attention(
        &mut self,
        stream_attentions: &[Tensor],
        learning_rate: f64,
    ) -> TorusResult<()> {
        if stream_attentions.len() != self.n_streams {
            return Ok(()); // Skip if mismatch
        }
        
        // Compute pairwise cosine similarity between flattened attention patterns
        for i in 0..self.n_streams {
            for j in (i + 1)..self.n_streams {
                let flat_i = stream_attentions[i].flatten_all()?;
                let flat_j = stream_attentions[j].flatten_all()?;
                
                // Cosine similarity
                let dot = (&flat_i * &flat_j)?.sum_all()?.to_scalar::<f32>()? as f64;
                let norm_i = flat_i.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
                let norm_j = flat_j.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
                
                let similarity = if norm_i > 1e-8 && norm_j > 1e-8 {
                    (dot / (norm_i * norm_j)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                
                // Convert to alignment (0 to 1)
                let new_alignment = (similarity + 1.0) / 2.0;
                
                // EMA update
                self.alignment_matrix[i][j] = 
                    (1.0 - learning_rate) * self.alignment_matrix[i][j] + 
                    learning_rate * new_alignment;
                self.alignment_matrix[j][i] = self.alignment_matrix[i][j];
            }
        }
        
        Ok(())
    }
    
    /// Compute overall cognitive cohesion from alignment matrix
    pub fn cognitive_cohesion(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..self.n_streams {
            for j in (i + 1)..self.n_streams {
                sum += self.alignment_matrix[i][j];
                count += 1;
            }
        }
        
        if count > 0 { sum / count as f64 } else { 0.0 }
    }
    
    /// Get alignment between two specific streams
    pub fn stream_alignment(&self, stream_a: usize, stream_b: usize) -> f64 {
        if stream_a < self.n_streams && stream_b < self.n_streams {
            self.alignment_matrix[stream_a][stream_b]
        } else {
            0.0
        }
    }
    
    /// Compute combination weights based on alignment
    /// Streams with higher alignment to the group get more weight
    pub fn alignment_weights(&self) -> Vec<f64> {
        let mut weights = Vec::with_capacity(self.n_streams);
        
        for i in 0..self.n_streams {
            let avg_alignment: f64 = self.alignment_matrix[i].iter().sum::<f64>() 
                / self.n_streams as f64;
            weights.push(avg_alignment);
        }
        
        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 1e-8 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        
        weights
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE COHERENCE LAYER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for cognitive coherence tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Number of processing streams
    pub n_streams: usize,
    
    /// Model dimension
    pub d_model: usize,
    
    /// Learning rate for SMM updates
    pub smm_learning_rate: f64,
    
    /// Base alpha for EMA compounding
    pub base_alpha: f64,
    
    /// Minimum alpha (floor)
    pub min_alpha: f64,
    
    /// Maximum alpha (ceiling)
    pub max_alpha: f64,
    
    /// Coherence threshold for "stable" state
    pub coherence_threshold: f64,
    
    /// Whether to use adaptive alpha based on coherence
    pub adaptive_alpha: bool,
    
    /// Weight for comprehensibility in SOC
    pub comprehensibility_weight: f64,
    
    /// Weight for manageability in SOC
    pub manageability_weight: f64,
    
    /// Weight for meaningfulness in SOC
    pub meaningfulness_weight: f64,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            n_streams: 8,
            d_model: 256,
            smm_learning_rate: 0.01,
            base_alpha: 0.9,
            min_alpha: 0.1,
            max_alpha: 0.99,
            coherence_threshold: 0.6,
            adaptive_alpha: true,
            comprehensibility_weight: 0.25,
            manageability_weight: 0.25,
            meaningfulness_weight: 0.50,
        }
    }
}

/// Cognitive Coherence Layer
/// 
/// Tracks and maintains coherence across processing streams using:
/// 1. Sense of Coherence (SOC) for state stability
/// 2. Shared Mental Models (SMM) for inter-stream alignment
/// 3. Adaptive compounding based on coherence metrics
/// 
/// # The Synergy
/// 
/// When cognitive cohesion is strong, it fosters psychological coherence by:
/// - Reducing uncertainty through clear expectations (→ comprehensibility)
/// - Strengthening belief in collective capabilities (→ manageability)
/// - Reinforcing shared purpose and motivation (→ meaningfulness)
#[derive(Debug)]
pub struct CognitiveCoherenceLayer {
    /// Configuration
    config: CoherenceConfig,
    
    /// Current sense of coherence state
    soc: SenseOfCoherence,
    
    /// Shared mental model between streams
    smm: SharedMentalModel,
    
    /// Compounded coherence state (EMA of past coherence)
    coherence_state: Option<Tensor>,
    
    /// History of SOC scores for trend analysis
    soc_history: Vec<f64>,
    
    /// Maximum history length
    max_history: usize,
    
    /// Device for tensor operations
    #[allow(dead_code)]
    device: Device,
}

impl CognitiveCoherenceLayer {
    /// Create a new cognitive coherence layer
    pub fn new(config: CoherenceConfig, device: &Device) -> Self {
        Self {
            smm: SharedMentalModel::new(config.n_streams),
            soc: SenseOfCoherence::default(),
            coherence_state: None,
            soc_history: Vec::new(),
            max_history: 100,
            device: device.clone(),
            config,
        }
    }
    
    /// Create with default config for torus attention
    pub fn for_torus_attention(d_model: usize, device: &Device) -> Self {
        let config = CoherenceConfig {
            n_streams: 8,
            d_model,
            ..Default::default()
        };
        let mut layer = Self::new(config, device);
        layer.smm = SharedMentalModel::for_torus_attention();
        layer
    }
    
    /// Compute comprehensibility from attention entropy
    /// 
    /// Low entropy = high comprehensibility (clear, focused attention)
    /// High entropy = low comprehensibility (scattered, uncertain)
    pub fn compute_comprehensibility(&self, attention: &Tensor) -> TorusResult<f64> {
        // Compute entropy: -sum(p * log(p))
        let eps = 1e-10;
        let log_attn = (attention + eps)?.log()?;
        let entropy = -(attention * &log_attn)?.sum_all()?.to_scalar::<f32>()? as f64;
        
        // Normalize by max entropy (uniform distribution)
        let n = attention.elem_count();
        let max_entropy = (n as f64).ln();
        
        // Convert to comprehensibility (invert and normalize)
        let normalized_entropy = entropy / max_entropy.max(1.0);
        let comprehensibility = 1.0 - normalized_entropy.clamp(0.0, 1.0);
        
        Ok(comprehensibility)
    }
    
    /// Compute manageability from capacity vs demand
    /// 
    /// Capacity: available attention budget (typically 1.0 per position)
    /// Demand: variance/spread of attention (how much is needed)
    pub fn compute_manageability(&self, attention: &Tensor, _hidden: &Tensor) -> TorusResult<f64> {
        // Compute attention variance as proxy for demand
        let mean = attention.mean_all()?.to_scalar::<f32>()? as f64;
        let variance = ((attention - mean)?.sqr()?.mean_all()?.to_scalar::<f32>()?) as f64;
        
        // Higher variance = more demand = lower manageability
        // Use sigmoid-like transformation
        let demand = variance.sqrt();
        let capacity = 1.0; // Normalized capacity
        
        let manageability = capacity / (capacity + demand);
        
        Ok(manageability.clamp(0.0, 1.0))
    }
    
    /// Compute meaningfulness from signal strength
    /// 
    /// High signal-to-noise = meaningful
    /// Uses attention concentration (Gini-like coefficient)
    pub fn compute_meaningfulness(&self, attention: &Tensor) -> TorusResult<f64> {
        // Compute concentration: how much attention goes to top-k positions
        let flat = attention.flatten_all()?;
        let n = flat.elem_count();
        
        // Get max attention value as proxy for signal strength
        let max_attn = flat.max(0)?.to_scalar::<f32>()? as f64;
        let mean_attn = flat.mean_all()?.to_scalar::<f32>()? as f64;
        
        // Ratio of max to mean indicates concentration/meaningfulness
        let concentration = if mean_attn > 1e-8 {
            (max_attn / mean_attn).min(n as f64)
        } else {
            1.0
        };
        
        // Normalize to 0-1 range
        let meaningfulness = (concentration - 1.0) / (n as f64 - 1.0).max(1.0);
        
        Ok(meaningfulness.clamp(0.0, 1.0))
    }
    
    /// Update SOC from current attention and hidden states
    pub fn update_soc(
        &mut self,
        attention: &Tensor,
        hidden: &Tensor,
    ) -> TorusResult<SenseOfCoherence> {
        let comprehensibility = self.compute_comprehensibility(attention)?;
        let manageability = self.compute_manageability(attention, hidden)?;
        let meaningfulness = self.compute_meaningfulness(attention)?;
        
        self.soc = SenseOfCoherence::new(comprehensibility, manageability, meaningfulness);
        
        // Track history
        self.soc_history.push(self.soc.score());
        if self.soc_history.len() > self.max_history {
            self.soc_history.remove(0);
        }
        
        Ok(self.soc.clone())
    }
    
    /// Update shared mental model from stream attention patterns
    pub fn update_smm(&mut self, stream_attentions: &[Tensor]) -> TorusResult<()> {
        self.smm.update_from_attention(stream_attentions, self.config.smm_learning_rate)
    }
    
    /// Get current cognitive cohesion (inter-stream alignment)
    pub fn cognitive_cohesion(&self) -> f64 {
        self.smm.cognitive_cohesion()
    }
    
    /// Get current psychological coherence (SOC score)
    pub fn psychological_coherence(&self) -> f64 {
        self.soc.score()
    }
    
    /// Compute adaptive alpha based on both cohesion and coherence
    /// 
    /// The synergy: strong cognitive cohesion fosters psychological coherence
    pub fn compute_adaptive_alpha(&self) -> f64 {
        if !self.config.adaptive_alpha {
            return self.config.base_alpha;
        }
        
        let cohesion = self.cognitive_cohesion();
        let coherence = self.psychological_coherence();
        
        // Synergy factor: cohesion amplifies coherence
        let synergy = cohesion * coherence;
        
        // Blend individual coherence with synergy
        let combined = 0.6 * coherence + 0.4 * synergy;
        
        // Scale alpha based on combined score
        let alpha = self.config.base_alpha * (0.5 + 0.5 * combined);
        
        alpha.clamp(self.config.min_alpha, self.config.max_alpha)
    }
    
    /// Get stream combination weights based on alignment
    pub fn stream_weights(&self) -> Vec<f64> {
        self.smm.alignment_weights()
    }
    
    /// Apply coherence-modulated compounding
    /// 
    /// new_state = α * old_state + (1-α) * input
    /// where α is adapted based on coherence metrics
    pub fn coherent_compound(
        &mut self,
        input: &Tensor,
        attention: &Tensor,
    ) -> TorusResult<Tensor> {
        // Update SOC from current attention
        self.update_soc(attention, input)?;
        
        // Compute adaptive alpha
        let alpha = self.compute_adaptive_alpha();
        
        // Initialize state if needed
        let old_state = match &self.coherence_state {
            Some(state) => state.clone(),
            None => Tensor::zeros_like(input)?,
        };
        
        // EMA update: new = α * old + (1-α) * input
        let new_state = ((&old_state * alpha)? + (input * (1.0 - alpha))?)?;
        
        self.coherence_state = Some(new_state.clone());
        
        Ok(new_state)
    }
    
    /// Combine multiple stream outputs with coherence-weighted fusion
    pub fn coherent_stream_fusion(
        &mut self,
        stream_outputs: &[Tensor],
        stream_attentions: &[Tensor],
    ) -> TorusResult<Tensor> {
        if stream_outputs.is_empty() {
            return Err(candle_core::Error::Msg("No stream outputs".to_string()).into());
        }
        
        // Update SMM based on attention patterns
        self.update_smm(stream_attentions)?;
        
        // Get alignment-based weights
        let weights = self.stream_weights();
        
        // Weighted sum of stream outputs
        let mut fused = Tensor::zeros_like(&stream_outputs[0])?;
        
        for (output, weight) in stream_outputs.iter().zip(weights.iter()) {
            fused = (fused + (output * *weight)?)?;
        }
        
        // Apply coherent compounding to the fused output
        if let Some(attn) = stream_attentions.first() {
            self.coherent_compound(&fused, attn)
        } else {
            Ok(fused)
        }
    }
    
    /// Get coherence trend (positive = improving, negative = degrading)
    pub fn coherence_trend(&self) -> f64 {
        if self.soc_history.len() < 2 {
            return 0.0;
        }
        
        let recent_len = self.soc_history.len().min(10);
        let recent: Vec<f64> = self.soc_history.iter()
            .rev()
            .take(recent_len)
            .copied()
            .collect();
        
        if recent.len() < 2 {
            return 0.0;
        }
        
        // Simple linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i * i) as f64).sum();
        
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Get current SOC state
    pub fn soc(&self) -> &SenseOfCoherence {
        &self.soc
    }
    
    /// Get current SMM
    pub fn smm(&self) -> &SharedMentalModel {
        &self.smm
    }
    
    /// Check if system is in coherent state
    pub fn is_coherent(&self) -> bool {
        self.soc.is_coherent(self.config.coherence_threshold)
    }
    
    /// Get diagnostic summary
    pub fn summary(&self) -> String {
        format!(
            "Cognitive Coherence:\n\
             ├─ SOC Score: {:.3}\n\
             │  ├─ Comprehensibility: {:.3}\n\
             │  ├─ Manageability: {:.3}\n\
             │  └─ Meaningfulness: {:.3}\n\
             ├─ Cognitive Cohesion: {:.3}\n\
             ├─ Adaptive Alpha: {:.3}\n\
             ├─ Coherence Trend: {:+.4}\n\
             └─ Status: {}",
            self.soc.score(),
            self.soc.comprehensibility,
            self.soc.manageability,
            self.soc.meaningfulness,
            self.cognitive_cohesion(),
            self.compute_adaptive_alpha(),
            self.coherence_trend(),
            if self.is_coherent() { "COHERENT" } else { "UNCERTAIN" }
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE-AWARE COMPOUNDING EXTENSION
// ═══════════════════════════════════════════════════════════════════════════════

/// Extension trait for coherence-aware operations
pub trait CoherenceAware {
    /// Apply coherence-modulated processing
    fn with_coherence(&self, coherence: &CognitiveCoherenceLayer) -> TorusResult<Tensor>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soc_creation() {
        let soc = SenseOfCoherence::new(0.8, 0.7, 0.9);
        assert!((soc.comprehensibility - 0.8).abs() < 1e-10);
        assert!((soc.manageability - 0.7).abs() < 1e-10);
        assert!((soc.meaningfulness - 0.9).abs() < 1e-10);
    }
    
    #[test]
    fn test_soc_score() {
        let soc = SenseOfCoherence::new(0.8, 0.8, 0.8);
        // 0.25*0.8 + 0.25*0.8 + 0.5*0.8 = 0.8
        assert!((soc.score() - 0.8).abs() < 1e-10);
    }
    
    #[test]
    fn test_soc_adaptive_alpha() {
        let high_soc = SenseOfCoherence::healthy();
        let low_soc = SenseOfCoherence::stressed();
        
        let high_alpha = high_soc.adaptive_alpha(0.9, 0.1, 0.99);
        let low_alpha = low_soc.adaptive_alpha(0.9, 0.1, 0.99);
        
        // Higher SOC should give higher alpha
        assert!(high_alpha > low_alpha);
    }
    
    #[test]
    fn test_smm_creation() {
        let smm = SharedMentalModel::for_torus_attention();
        assert_eq!(smm.n_streams, 8);
        
        // Check forward-backward pair alignment
        assert!(smm.stream_alignment(0, 1) > 0.8);
    }
    
    #[test]
    fn test_smm_cognitive_cohesion() {
        let smm = SharedMentalModel::for_torus_attention();
        let cohesion = smm.cognitive_cohesion();
        
        // Should be moderate to high with default initialization
        assert!(cohesion > 0.5);
        assert!(cohesion <= 1.0);
    }
    
    #[test]
    fn test_coherence_layer_creation() {
        let device = Device::Cpu;
        let layer = CognitiveCoherenceLayer::for_torus_attention(64, &device);
        
        assert_eq!(layer.config.n_streams, 8);
        assert!(layer.is_coherent()); // Should start coherent
    }
    
    #[test]
    fn test_coherence_layer_adaptive_alpha() {
        let device = Device::Cpu;
        let layer = CognitiveCoherenceLayer::for_torus_attention(64, &device);
        
        let alpha = layer.compute_adaptive_alpha();
        assert!(alpha >= layer.config.min_alpha);
        assert!(alpha <= layer.config.max_alpha);
    }
    
    #[test]
    fn test_stream_weights() {
        let device = Device::Cpu;
        let layer = CognitiveCoherenceLayer::for_torus_attention(64, &device);
        
        let weights = layer.stream_weights();
        assert_eq!(weights.len(), 8);
        
        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_soc_blend() {
        let healthy = SenseOfCoherence::healthy();
        let stressed = SenseOfCoherence::stressed();
        
        let blended = healthy.blend(&stressed, 0.5);
        
        // Blended should be between the two
        assert!(blended.score() < healthy.score());
        assert!(blended.score() > stressed.score());
    }
    
    #[test]
    fn test_coherence_summary() {
        let device = Device::Cpu;
        let layer = CognitiveCoherenceLayer::for_torus_attention(64, &device);
        
        let summary = layer.summary();
        assert!(summary.contains("SOC Score"));
        assert!(summary.contains("Comprehensibility"));
        assert!(summary.contains("Cognitive Cohesion"));
    }
}
