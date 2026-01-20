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
pub mod geometry;
pub mod attention;
pub mod vortex;
pub mod periodic;
pub mod dual_loop;
pub mod error;

// Bidirectional parallel processing modules
pub mod bidirectional;
pub mod parallel_streams;
pub mod compounding;
pub mod coherence;
pub mod integration;

// Training infrastructure
pub mod training;

// Integration tests
#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub mod python;

// Re-exports from core modules
pub use geometry::{TorusCoordinate, TorusManifold, TorusDistanceMatrix};
pub use attention::{TorusAttention, TorusAttentionConfig, TorusTransformer};
pub use vortex::{Vortex, VortexDynamics, SpiralAttention, HelicalFlow};
pub use periodic::{PeriodicBoundary, PeriodicAttentionMask};
pub use dual_loop::{DualLoopFlow, DualLoopConfig, LoopAttention};
pub use error::TorusError;

// Re-exports from bidirectional modules
pub use bidirectional::{
    FlowDirection, 
    BidirectionalAttention, 
    SymmetricCombiner,
    CausalMask,
    TorusBidirectionalEncoding,
};
pub use parallel_streams::{
    StreamId,
    ParallelStreamConfig,
    ParallelStreamProcessor,
    ProcessingStream,
    StreamWeights,
};
pub use compounding::{
    CompoundingConfig,
    EMACompounding,
    MultiScaleCompounding,
    LearnableAlpha,
    CompoundingStats,
};
pub use coherence::{
    SenseOfCoherence,
    SharedMentalModel,
    CoherenceConfig,
    CognitiveCoherenceLayer,
    CoherenceAware,
};
pub use integration::{
    BidirectionalTorusConfig,
    BidirectionalTorusLayer,
    BidirectionalTorusTransformer,
    BidirectionalTorusInference,
    BidirectionalStats,
};
pub use training::{
    TrainingConfig,
    Trainer,
    LRScheduler,
    TrainingMetrics,
    TorusLoss,
    LossType,
    run_training_example,
};

/// Result type for torus operations
pub type TorusResult<T> = Result<T, TorusError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        // Core types
        TorusCoordinate,
        TorusManifold,
        TorusAttention,
        TorusAttentionConfig,
        VortexDynamics,
        PeriodicBoundary,
        
        // Bidirectional types
        FlowDirection,
        StreamId,
        BidirectionalTorusConfig,
        BidirectionalTorusTransformer,
        EMACompounding,
        CompoundingConfig,
        
        // Cognitive coherence types
        SenseOfCoherence,
        SharedMentalModel,
        CognitiveCoherenceLayer,
        CoherenceConfig,
        
        // Training types
        TrainingConfig,
        Trainer,
        LRScheduler,
        TrainingMetrics,
        
        // Result type
        TorusResult,
        TorusError,
    };
}
