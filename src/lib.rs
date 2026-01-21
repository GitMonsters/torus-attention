//! # Torus Attention Mechanism
//!
//! A transformer-style attention mechanism built on a torus manifold,
//! leveraging periodic boundary conditions, dual-loop (major/minor)
//! information flow, and spiral (vortex) dynamics.
//!
//! ## Features
//! - Torus topology with periodic boundary conditions
//! - Dual-loop attention (major and minor radii)
//! - Vortex/spiral information flow patterns
//! - **8-stream bidirectional parallel processing**
//! - **Learnable EMA compounding across layers**
//! - GPU acceleration via candle-core
//! - Python bindings via PyO3
//!
//! ## Architecture
//!
//! ```text
//! Input → Position Encoding → 8-Stream Parallel → EMA Compound → Output
//!                                    │
//!         ┌──────────────────────────┼──────────────────────────┐
//!         │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
//!         │   │Major Fwd│  │Major Bwd│  │Minor Fwd│  │Minor Bwd││
//!         │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│
//!         │   ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐│
//!         │   │Spiral CW│  │SpiralCCW│  │Cross U→V│  │Cross V→U││
//!         │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│
//!         └────────┴───────────┴───────────┴───────────┴────────┘
//!                              │
//!                    Symmetric Combine (learned weights)
//!                              │
//!                    EMA Compound (learnable α per layer)
//! ```

// Core modules
pub mod attention;
pub mod dual_loop;
pub mod error;
pub mod geometry;
pub mod periodic;
pub mod vortex;

// CERN Hadron Collider-inspired validator module
pub mod collider;

// Bidirectional parallel processing modules
pub mod bidirectional;
pub mod coherence;
pub mod compounding;
pub mod integration;
pub mod parallel_streams;

// Training infrastructure
pub mod metrics;
pub mod rmsnorm;
pub mod training;

// LLM and API server
pub mod api_server;
pub mod checkpoint;
pub mod dataset;
pub mod dynamic_trainer;
pub mod llm;
pub mod llm_trainer;
pub mod tokenizer;

// Integration tests
#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub mod python;

// Re-exports from core modules
pub use attention::{TorusAttention, TorusAttentionConfig, TorusTransformer};
pub use dual_loop::{DualLoopConfig, DualLoopFlow, LoopAttention};
pub use error::TorusError;
pub use geometry::{TorusCoordinate, TorusDistanceMatrix, TorusManifold};
pub use periodic::{PeriodicAttentionMask, PeriodicBoundary};
pub use vortex::{HelicalFlow, SpiralAttention, Vortex, VortexDynamics};

// Re-exports from bidirectional modules
pub use bidirectional::{
    BidirectionalAttention, CausalMask, FlowDirection, SymmetricCombiner,
    TorusBidirectionalEncoding,
};
pub use coherence::{
    CognitiveCoherenceLayer, CoherenceAware, CoherenceConfig, SenseOfCoherence, SharedMentalModel,
};
pub use compounding::{
    CompoundingConfig, CompoundingStats, EMACompounding, LearnableAlpha, MultiScaleCompounding,
};
pub use integration::{
    BidirectionalStats, BidirectionalTorusConfig, BidirectionalTorusInference,
    BidirectionalTorusLayer, BidirectionalTorusTransformer, CoherenceMetrics, LayerOutput,
};
pub use parallel_streams::{
    ParallelStreamConfig, ParallelStreamProcessor, ProcessingStream, StreamId, StreamWeights,
};
pub use training::{
    run_training_example, LRScheduler, LossType, TorusLoss, Trainer, TrainingConfig,
    TrainingMetrics,
};

// LLM exports
pub use api_server::{ApiHandler, ServerConfig};
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointMetadata};
pub use dataset::{Batch, DataLoader, TextDataset};
pub use llm::{
    FeedForward, SamplingStrategy, TextGenerator, TorusLLM, TorusLLMConfig, TransformerBlock,
};
pub use llm_trainer::{LLMTrainer, LLMTrainingConfig};
pub use tokenizer::{
    format_chat_prompt, BpeTokenizer, ChatMessage, SimpleTokenizer, SpecialTokens, Tokenizer,
};

// Dynamic training exports
pub use dynamic_trainer::{
    CurriculumSamplingParams, CurriculumScheduler, DifficultyLevel, DynamicBatchController,
    DynamicCompoundTrainer, DynamicEMAController, DynamicTrainingConfig, DynamicTrainingStats,
    GrowthConfig, GrowthController, LayerWiseLRController, MultiTaskScheduler, Task,
};

// Metrics exports
pub use metrics::{MetricsCollector, MetricsLogger};

// RMSNorm exports (Metal-compatible normalization)
pub use rmsnorm::{rms_norm, RmsNorm};

// Collider exports (CERN Hadron Collider-inspired validator)
pub use collider::{
    AnomalyEvent, AnomalyMonitor, AnomalyThresholds, AnomalyType, ColliderConfig, ColliderMetrics,
    ColliderReport, ConservationValidator, DarknessTracker, FourMomentum, Particle, ParticleBeam,
    ParticleFlavor, TorusCollider, TorusColliderDetector,
};

/// Result type for torus operations
pub type TorusResult<T> = Result<T, TorusError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        BidirectionalTorusConfig,
        BidirectionalTorusTransformer,
        CognitiveCoherenceLayer,
        CoherenceConfig,

        CompoundingConfig,

        EMACompounding,
        // Bidirectional types
        FlowDirection,
        LRScheduler,
        PeriodicBoundary,

        // Cognitive coherence types
        SenseOfCoherence,
        SharedMentalModel,
        StreamId,
        TorusAttention,
        TorusAttentionConfig,
        // Core types
        TorusCoordinate,
        TorusError,
        TorusManifold,
        // Result type
        TorusResult,
        Trainer,
        // Training types
        TrainingConfig,
        TrainingMetrics,

        VortexDynamics,
        
        // Collider types
        TorusCollider,
        ColliderConfig,
        AnomalyMonitor,
    };
}
