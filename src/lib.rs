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
//! - **Multi-GPU support**: Metal (macOS), CUDA (NVIDIA), ROCm (AMD), Vulkan (cross-platform)
//! - Python bindings via PyO3
//!
//! ## GPU Backends
//!
//! Build with different GPU backends using feature flags:
//!
//! ```bash
//! # AMD GPU via ROCm (recommended for AMD hardware)
//! cargo build --release --no-default-features --features burn-rocm
//!
//! # Any GPU via Vulkan/WGPU
//! cargo build --release --no-default-features --features burn-vulkan
//!
//! # Legacy: NVIDIA GPU via CUDA (candle)
//! cargo build --release --no-default-features --features cuda
//!
//! # Legacy: macOS GPU via Metal (candle)
//! cargo build --release --no-default-features --features metal
//! ```
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
pub mod backend;  // Multi-GPU backend abstraction
pub mod dual_loop;
pub mod error;
pub mod geometry;
pub mod gpu_ops;  // GPU-accelerated operations (auto-enabled with amd-gpu feature)
pub mod periodic;
pub mod vortex;

// CERN Hadron Collider-inspired validator module
pub mod collider;

// Knowledge distillation (silent teacher)
pub mod distillation;

// Bidirectional parallel processing modules
pub mod bidirectional;
pub mod coherence;
pub mod compounding;
pub mod compounding_cohesion;
pub mod compounding_transformer;
pub mod integration;
pub mod parallel_streams;

// Sensorimotor integration for full compounding closure
pub mod sensorimotor;

// Consequential reasoning and AGI systems
pub mod consequential;

// Unified AGI Core - Compounding Cognitive Cohesion
pub mod agi_core;

// Artificial General Coherence (AGC) - Holistic reasoning framework
pub mod general_coherence;

// Illichian Learning Webs - Self-directed learning via deschooling principles
pub mod learning_webs;

// Long-term Memory System - Episodic, semantic, and procedural memory
pub mod memory_system;

// LLM Integration - Connect language models to AGI Core
pub mod llm_integration;

// Explicability - Natural language decision explanations
pub mod explicability;

// Multi-Agent System - Peer agents for collaboration and learning
pub mod multi_agent;

// Real-World Interface - Sensor and actuator traits for embodiment
pub mod real_world;

// Tetrahedral Mixture of Agents - 64/128 agent geometric architecture
pub mod tetrahedral_moa;

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
pub use compounding_cohesion::{
    AdaptiveCoherenceWeights, CompoundingCohesionConfig, CompoundingCohesionSystem,
    CompoundingResult, ConsolidationResult, CrossLayerSMM, GoalState, GoalStateGenerator,
    GoalType, GraphEdge, GraphNode, GraphStats, HierarchicalCoherence, HierarchicalSOC,
    HierarchyLevel, PredictiveCoherence, StreamGraphMemory,
};
pub use compounding_transformer::{
    CompoundingCohesionTransformer, CompoundingTransformerConfig, CompoundingTransformerStats,
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

// Distillation exports (knowledge distillation with silent teacher)
pub use distillation::{
    DistillationConfig, DistillationStepResult, DistillationTrainer, TeacherModel,
    DistillationCheckpointMetadata, load_transformer_checkpoint, save_transformer_checkpoint,
};

// Sensorimotor exports (full compounding closure)
pub use sensorimotor::{
    Action, ActionResult, ActionType, CoherenceGuidedPolicy, CognitiveDissonanceTracker,
    DissonanceStats, Environment, EpisodeResult, HypothesisTestingPolicy, Landmark,
    LearningEnvConfig, LearningEnvStats, LearningGridEnvironment, MemoryContext,
    MemoryGuidedPolicy, MotorPolicy, MotorStats, MotorSystem, Observation, ObservationMetadata,
    Pose3D, ReactivePolicy, SeededRng, SensorimotorAgent, SensorimotorConfig,
    SimpleGridEnvironment, StepStats,
};

// Consequential reasoning exports (AGI systems)
pub use consequential::{
    AGIDecision, AGIReasoningSystem, AGIReasoningSummary, CausalGraph, CausalGraphSummary,
    CausalMechanism, CausalVariable, CompoundingMetrics, CompoundingSummary, ConsequenceNode,
    ConsequentialThinking, CounterfactualOutcome, DecisionMethod, Intervention, StreamVote,
    StreamVotingSystem, TransitionModel, VotingResult, VotingStats,
};

// Unified AGI Core exports (Compounding Cognitive Cohesion)
pub use agi_core::{
    AGICore, AGICoreConfig, AGICoreSummary,
    // Causal Discovery
    CausalDiscovery, CausalDiscoverySummary, CausalObservation, DiscoveredVariable,
    // Abstraction Hierarchy
    AbstractionHierarchy, AbstractionSummary, Concept,
    // World Model
    WorldModel, WorldModelSummary, WorldState, WorldTransition, SimulatedTrajectory,
    // Goal Hierarchy
    GoalHierarchy, GoalHierarchySummary, Goal, GoalPriority, GoalStatus,
    // Meta-Learning
    MetaLearner, MetaLearnerSummary, LearningMetrics, LearningEpisode,
    // Symbol System
    SymbolSystem, SymbolSystemSummary, Symbol, SymbolicExpression, SymbolRelation,
    // Compounding Analytics
    CompoundingAnalytics,
};

// Artificial General Coherence (AGC) exports
pub use general_coherence::{
    // Core AGC system
    ArtificialGeneralCoherence, AGCSummary, SOCSummary,
    // Psychological Coherence
    PsychologicalCoherence, DissonanceSource, DissonanceType, 
    RegulationMode, CoherenceGates, ResolutionStrategy,
    // Requisite Variety
    RequisiteVariety, VarietyMeasure, ExpansionRecommendation, ExpansionArea,
    // Homeostatic Equilibrium
    HomeostaticEquilibrium, HomeostaticSetpoints, HomeostaticControls,
    PIDGains, EquilibriumStatus,
    // Epistemic Integrity
    EpistemicIntegrity,
    // Ethical Alignment
    EthicalAlignment, Value, ActionEthicalStatus,
    // Foundational Principles for AGI
    FoundationalPrinciples, PrinciplesSummary,
    EmbodimentPrinciple, SymbolGroundingPrinciple, CausalityPrinciple,
    MemoryIntegrationPrinciple, MetacognitionPrinciple, AutonomyPrinciple,
    ExplicabilityPrinciple,
};

// Illichian Learning Webs exports (Deschooling principles for AGI)
pub use learning_webs::{
    // Core system
    LearningWebs, LearningWebsSummary, LearningPathRecommendation,
    // Ethos (Promethean vs Epimethean)
    LearningEthos,
    // Network 1: Educational Objects
    ObjectNetwork, EducationalObject, ObjectType,
    // Network 2: Skill Exchanges
    SkillExchangeNetwork, Skill, SkillExchange,
    // Network 3: Peer Matching
    PeerMatchingNetwork, LearningPeer,
    // Network 4: Educators-at-Large
    EducatorNetwork, EducatorAtLarge, GuidanceStyle,
    // Hidden Curriculum Detection
    HiddenCurriculumDetector, DeinstitutionalizationAction,
};

// Long-term Memory System exports
pub use memory_system::{
    // Core system
    MemorySystem, MemorySystemConfig, MemorySystemSummary, MemoryRetrievalResult,
    // Episodic Memory
    EpisodicMemoryStore, EpisodicMemory, EmotionalValence,
    // Semantic Memory
    SemanticMemoryStore, SemanticFact, SemanticRelation, RelationType,
    // Procedural Memory
    ProceduralMemoryStore, ProceduralSkill,
    // Working Memory
    WorkingMemory, WorkingMemoryItem, WorkingMemoryItemType,
};

// LLM Integration exports
pub use llm_integration::{
    // Core system
    LLMIntegration, LLMIntegrationConfig, LLMIntegrationSummary, LanguageResult,
    // Language Understanding
    LanguageProcessor, ParsedSemantics, Intent, Entity, EntityType, Relation, Sentiment,
    // Semantic Grounding
    SemanticGrounding, GroundedSymbol,
    // Reasoning Engine
    ReasoningEngine, ReasoningChain, ReasoningStep, ReasoningType,
};

// Explicability exports
pub use explicability::{
    // Core system
    ExplicabilitySystem, ExplicabilityConfig, ExplicabilitySummary,
    // Decision Tracing
    DecisionTracer, TracedDecision, DecisionAction, DecisionFactor,
    Alternative, DecisionOutcome, InfluenceDirection, FactorSource,
    // Explanation Generation
    ExplanationGenerator, Explanation, ExplanationType, DetailLevel,
    // Interactive Interface
    ExplanationQuery, ExplanationResponse,
};

// Multi-Agent System exports
pub use multi_agent::{
    // Core system
    MultiAgentSystem, MultiAgentConfig, MultiAgentSummary,
    // Agent Identity
    AgentId, AgentRole, AgentState, AgentProfile,
    // Messaging
    MessageRouter, AgentMessage, MessageType, MessagePriority, MessageContent,
    // Trust and Collaboration
    TrustManager, TrustRelation, CollaborationManager, CollaborativeTask, TaskStatus,
};

// Real-World Interface exports
pub use real_world::{
    // Core system
    RealWorldInterface, RealWorldConfig, RealWorldSummary,
    // Sensor traits and types
    Sensor, SensorId, SensorType, SensorState, SensorReading, SensorData, SensorError,
    // Actuator traits and types
    Actuator, ActuatorId, ActuatorType, ActuatorState, 
    ActuatorCommand, ActuatorTarget, ActuatorResult, ActuatorError, CommandType,
    // Sensor Fusion
    SensorFusion, FusedPerception,
    // Mock implementations for testing
    MockCamera, MockIMU, MockServo,
};

// Tetrahedral MOA exports (64/128 agent geometric architecture)
pub use tetrahedral_moa::{
    // Geometry
    Point3D, Tetrahedron, StarTetrahedron, TetrahedralGeometry,
    TetrahedronVertex, TetrahedronEdge, TetrahedronFace,
    TetrahedronOrientation, TetrahedralPoint, StarTetrahedralPoint,
    // Agents
    TetrahedralAgent, TetrahedralAgentConfig, TetrahedralAgentRole, AgentDomain,
    // MOA System
    TetrahedralMOA, TetrahedralMOAConfig, TetrahedralMOASummary,
    TetrahedralRouter, TorusTetrahedralBridge,
    // Multi-Torus Scaling
    MultiTorusConfig,
    // Octopus Limb Architecture
    LimbType, OctopusLimb, OctopusLimbConfig, OctopusHub,
    // RNA Editing Layer
    RNAEditingLayer,
    // Working Memory
    TetrahedralWorkingMemory, MemorySlotType, WorkingMemorySummary,
    // Integrated System
    OctoTetrahedralSystem, OctoTetrahedralConfig, OctoTetrahedralSummary,
};

// GPU compute exports (AMD GPU acceleration)
#[cfg(feature = "amd-gpu")]
pub use backend::{GpuCompute, GpuError};

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

        CompoundingCohesionTransformer,
        CompoundingConfig,
        // Full compounding cohesion types
        CompoundingCohesionConfig,
        CompoundingCohesionSystem,
        CompoundingResult,
        CompoundingTransformerConfig,

        EMACompounding,
        // Bidirectional types
        FlowDirection,
        // Goal states for sensorimotor closure
        GoalState,
        GoalType,
        // Hierarchical coherence
        HierarchicalCoherence,
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
        
        // Sensorimotor types for full compounding loop
        SensorimotorAgent,
        SensorimotorConfig,
        MotorSystem,
        Environment,
        SimpleGridEnvironment,
        LearningGridEnvironment,
        LearningEnvConfig,
        CognitiveDissonanceTracker,
        Observation,
        Action,
        ActionType,
        Pose3D,
        
        // AGI Reasoning types (consequential reasoning)
        AGIReasoningSystem,
        AGIDecision,
        CausalGraph,
        ConsequentialThinking,
        StreamVotingSystem,
        CompoundingMetrics,
        DecisionMethod,
        
        // Unified AGI Core types (Compounding Cognitive Cohesion)
        AGICore,
        AGICoreConfig,
        CausalDiscovery,
        AbstractionHierarchy,
        WorldModel,
        GoalHierarchy,
        GoalPriority,
        MetaLearner,
        SymbolSystem,
        CompoundingAnalytics,
        
        // Artificial General Coherence (AGC) types
        ArtificialGeneralCoherence,
        PsychologicalCoherence,
        RequisiteVariety,
        HomeostaticEquilibrium,
        EpistemicIntegrity,
        EthicalAlignment,
        DissonanceType,
        RegulationMode,
        EquilibriumStatus,
        
        // Illichian Learning Webs types (Deschooling for AGI)
        LearningWebs,
        LearningEthos,
        ObjectNetwork,
        SkillExchangeNetwork,
        PeerMatchingNetwork,
        EducatorNetwork,
        HiddenCurriculumDetector,
        DeinstitutionalizationAction,
        
        // Long-term Memory System types
        MemorySystem,
        MemorySystemConfig,
        EpisodicMemoryStore,
        SemanticMemoryStore,
        ProceduralMemoryStore,
        WorkingMemory,
        EmotionalValence,
        RelationType,
        
        // LLM Integration types
        LLMIntegration,
        LLMIntegrationConfig,
        LanguageProcessor,
        SemanticGrounding,
        ReasoningEngine,
        Intent,
        ReasoningType,
        
        // Explicability types
        ExplicabilitySystem,
        ExplicabilityConfig,
        DecisionTracer,
        ExplanationGenerator,
        ExplanationType,
        DetailLevel,
        InfluenceDirection,
        FactorSource,
        
        // Multi-Agent System types
        MultiAgentSystem,
        MultiAgentConfig,
        AgentId,
        AgentRole,
        AgentState,
        MessageType,
        MessagePriority,
        TrustManager,
        CollaborationManager,
        TaskStatus,
        
        // Real-World Interface types
        RealWorldInterface,
        RealWorldConfig,
        SensorType,
        SensorState,
        ActuatorType,
        ActuatorState,
        SensorFusion,
        FusedPerception,
        
        // Tetrahedral MOA types (64/128 agent geometric architecture)
        TetrahedralMOA,
        TetrahedralMOAConfig,
        Tetrahedron,
        StarTetrahedron,
        TetrahedralGeometry,
        TetrahedronVertex,
        TetrahedralAgent,
        AgentDomain,
        TetrahedralRouter,
        MultiTorusConfig,
        // Octopus Limb Architecture
        LimbType,
        OctopusHub,
        OctopusLimb,
        // RNA Editing and Working Memory
        RNAEditingLayer,
        TetrahedralWorkingMemory,
        MemorySlotType,
        // Integrated OctoTetrahedral System
        OctoTetrahedralSystem,
        OctoTetrahedralConfig,
    };
}
