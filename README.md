# Compounding Cognitive Cohesion

**An AGI-Relevant Learning System in Rust**

A torus transformer with psychological coherence principles, intrinsic motivation, causal reasoning, and emergent goal-directed behavior. Built on Metal GPU acceleration.

```
    ╭──────────────────────────────────────────────────────────────────╮
    │               COMPOUNDING COGNITIVE COHESION                      │
    │                                                                   │
    │   Coherence → Cohesion → Compounding → Consequential Thinking    │
    ╰──────────────────────────────────────────────────────────────────╯
```

## Core Hypothesis

AGI could emerge from the combination of:

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Coherence** | Self-monitoring comprehensibility, manageability, meaningfulness | `SenseOfCoherence` |
| **Cohesion** | 8-stream parallel processing with graph memory | `StreamGraphMemory` |
| **Compounding** | Knowledge builds on knowledge | `SuccessorRepresentation`, replay buffer |
| **Consequential Thinking** | Multi-step tree search | `ConsequentialThinking` (MCTS) |
| **Causal Reasoning** | Pearl-style do(X=x) calculus | `CausalGraph` |
| **Intrinsic Motivation** | Self-generated drives | Competence, learning progress, empowerment |
| **Voting Consensus** | Thousand Brains-style inference | `StreamVotingSystem` |

## Architecture

```
Input Observation
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    8-STREAM TORUS TRANSFORMER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Major Fwd → │ │ Major Bwd ← │ │ Minor Fwd → │ │ Minor Bwd ← │        │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘        │
│  ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐        │
│  │ Spiral CW ↻ │ │ Spiral CCW ↺│ │ Cross U→V   │ │ Cross V→U   │        │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘        │
└─────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPOUNDING COHESION SYSTEM                          │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │ Sense of Coherence│  │ Stream Graph      │  │ Successor         │    │
│  │ ├─ Comprehensible │  │ Memory            │  │ Representations   │    │
│  │ ├─ Manageable     │  │ ├─ Spatial nodes  │  │ ├─ Multi-step SR  │    │
│  │ └─ Meaningful     │  │ └─ Semantic edges │  │ └─ TD(λ) updates  │    │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘    │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │ Intrinsic         │  │ Hierarchical      │  │ Adaptive          │    │
│  │ Motivation        │  │ Coherence         │  │ Temperature       │    │
│  │ ├─ Competence     │  │ ├─ Local → Global │  │ └─ Explore/Exploit│    │
│  │ ├─ Learning prog. │  │ └─ Cross-layer    │  │                   │    │
│  │ ├─ Empowerment    │  └───────────────────┘  └───────────────────┘    │
│  │ └─ Info gain      │                                                  │
│  └───────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGI REASONING SYSTEM                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │ Causal Graph      │  │ Consequential     │  │ Stream Voting     │    │
│  │ (Pearl-style)     │  │ Thinking (MCTS)   │  │ (Thousand Brains) │    │
│  │ ├─ Variables      │  │ ├─ UCB selection  │  │ ├─ Flash inference│    │
│  │ ├─ Mechanisms     │  │ ├─ Tree expansion │  │ ├─ Evidence decay │    │
│  │ ├─ do(X=x)        │  │ ├─ Value backup   │  │ └─ Consensus      │    │
│  │ └─ Counterfactual │  │ └─ Learned model  │  │                   │    │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SENSORIMOTOR INTEGRATION                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │ Goal Generator    │  │ Memory-Guided     │  │ Cognitive         │    │
│  │ ├─ Explore        │  │ Policy            │  │ Dissonance        │    │
│  │ ├─ Refine         │  │ ├─ SR lookup      │  │ Tracker           │    │
│  │ ├─ Disambiguate   │  │ └─ Graph query    │  │ └─ CD-AI impl     │    │
│  │ └─ None           │  └───────────────────┘  └───────────────────┘    │
│  └───────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                            Action Output
```

## Quick Start

```bash
# Clone and build with Metal GPU (macOS)
cd rustyworm_simplecomplexity
cargo build --release

# Run learning experiment (200 episodes)
cargo run --example learning_experiment --release

# For CUDA (Linux/Windows):
cargo build --release --no-default-features --features cuda
```

## Project Structure

```
src/
├── compounding_cohesion.rs      # Core coherence system
│   ├── SenseOfCoherence         # Comprehensibility, manageability, meaningfulness
│   ├── HierarchicalCoherence    # Local → Regional → Global propagation
│   ├── StreamGraphMemory        # Per-stream graph with nodes/edges
│   ├── SuccessorRepresentation  # Multi-step planning (TD-style)
│   ├── IntrinsicMotivation      # 4 intrinsic drives
│   ├── GoalStateGenerator       # Emergent goal selection
│   └── CompoundingCohesionSystem# Full integration
│
├── consequential.rs             # AGI reasoning systems
│   ├── CausalGraph              # Pearl-style causal reasoning
│   ├── ConsequentialThinking    # MCTS tree search
│   ├── StreamVotingSystem       # Thousand Brains voting
│   ├── CompoundingMetrics       # Transfer learning measurement
│   └── AGIReasoningSystem       # Integrated reasoning
│
├── compounding_transformer.rs   # 8-stream torus transformer
│   └── CompoundingCohesionTransformer
│
├── sensorimotor.rs              # Sensorimotor integration
│   ├── LearningGridEnvironment  # 10x10 with landmarks, goals, obstacles
│   ├── MemoryGuidedPolicy       # Uses coherence for decisions
│   ├── CognitiveDissonanceTracker
│   └── SensorimotorAgent        # Full agent loop
│
└── examples/
    └── learning_experiment.rs   # 200-episode test
```

## Key Systems

### 1. Sense of Coherence (SOC)
Based on Antonovsky's salutogenic theory:

```rust
pub struct SenseOfCoherence {
    pub comprehensibility: f64,  // Can I understand what's happening?
    pub manageability: f64,      // Do I have resources to cope?
    pub meaningfulness: f64,     // Is this worth engaging with?
}
```

### 2. Intrinsic Motivation
Four self-generated drives:

| Signal | Description | Computation |
|--------|-------------|-------------|
| Competence Progress | Am I getting better? | Δ prediction error |
| Learning Progress | Is there something to learn? | Δ model complexity |
| Empowerment | Can I influence outcomes? | Reachable states |
| Information Gain | How much did I learn? | KL divergence |

### 3. Causal Graph (Pearl-style)

```rust
// Intervene: do(action = MoveNorth)
let outcome = causal_graph.intervene(Intervention {
    variable_id: action_var,
    value: 0.0,  // MoveNorth
});

// Counterfactual: What if I had moved South instead?
let counterfactual = causal_graph.counterfactual(
    action_var,
    1.0,  // MoveSouth
    &current_evidence
);
```

### 4. Stream Voting (Thousand Brains)

```rust
// Each stream votes independently
let votes: Vec<StreamVote> = streams.iter()
    .map(|s| s.vote_on_hypothesis())
    .collect();

// Aggregate with confidence
let result = voting_system.aggregate_votes(&votes);
// result.consensus, result.confidence, result.entropy
```

## Experiment Results

Latest 200-episode run with Metal GPU:

```
Learning Summary (200 episodes):
├─ Reward:     31.665 → 31.815 (+0.5%)
├─ Coherence:  0.308 → 0.306 (-0.7%)
├─ Graph Memory: 73 nodes, 508 edges
├─ Speed: 164.84ms per episode (Metal GPU)
└─ Learning: MARGINAL

Intrinsic Motivation:
├─ Competence Progress: -0.0000
├─ Learning Progress: +0.7300
├─ Empowerment: 1.0000
└─ Information Gain: 0.4310

Memory Systems:
├─ Successor Representations: 67
├─ Replay Buffer Size: 50
└─ Adaptive Temperature: 0.811
```

## Theoretical Foundation

### The Compounding Loop

```
                    ┌─────────────────────┐
                    │                     │
                    ▼                     │
              Observation                 │
                    │                     │
                    ▼                     │
            ┌───────────────┐             │
            │ Coherence     │             │
            │ Assessment    │             │
            └───────┬───────┘             │
                    │                     │
         ┌──────────┴──────────┐          │
         ▼                     ▼          │
   Comprehensible?        Meaningful?     │
         │                     │          │
         ▼                     ▼          │
   Update Graph           Generate Goal   │
         │                     │          │
         └──────────┬──────────┘          │
                    │                     │
                    ▼                     │
            ┌───────────────┐             │
            │ Causal        │             │
            │ Reasoning     │             │
            └───────┬───────┘             │
                    │                     │
                    ▼                     │
            ┌───────────────┐             │
            │ MCTS Tree     │             │
            │ Search        │             │
            └───────┬───────┘             │
                    │                     │
                    ▼                     │
            ┌───────────────┐             │
            │ Stream        │             │
            │ Voting        │             │
            └───────┬───────┘             │
                    │                     │
                    ▼                     │
               Action ───────────────────►│
```

### Why This Might Work

1. **Coherence as Meta-Learning Signal**: The system doesn't just learn from reward—it learns from *understanding itself*. Low coherence = high learning priority.

2. **Graph Memory as World Model**: Unlike flat replay buffers, graph memory preserves relational structure. Edges encode "what leads to what."

3. **Causal Reasoning Breaks Correlation**: Standard RL confuses correlation with causation. Pearl-style intervention enables "What if I had done X?" reasoning.

4. **Voting Enables Robustness**: No single stream has the answer. Consensus from multiple perspectives is more reliable.

5. **Intrinsic Motivation Drives Exploration**: Without external reward, the system still has reasons to act—curiosity, competence, empowerment.

## Dependencies

```toml
[dependencies]
candle-core = "0.9"           # Tensor operations (Metal/CUDA)
candle-nn = "0.9"             # Neural network layers
ndarray = "0.16"              # Multi-dimensional arrays
rayon = "1.10"                # Parallel iteration
rand = "0.8"                  # Random number generation
serde = "1.0"                 # Serialization
```

## Configuration

```rust
let config = CompoundingCohesionConfig {
    num_streams: 8,
    d_model: 64,
    coherence_threshold: 0.5,
    meta_learning_rate: 0.01,
    replay_buffer_size: 50,
    sr_gamma: 0.95,           // Successor representation discount
    sr_alpha: 0.1,            // SR learning rate
    exploration_bonus: 0.1,
    novelty_weight: 0.3,
    temperature_decay: 0.995,
};
```

## Future Directions

- [ ] Integrate consequential reasoning into action selection
- [ ] Add meta-cognitive monitoring (thinking about thinking)
- [ ] Implement hierarchical goal decomposition
- [ ] Add language grounding for semantic graphs
- [ ] Scale to larger environments
- [ ] Benchmark against standard RL baselines

## References

- Antonovsky, A. (1987). *Unraveling the Mystery of Health*
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Hawkins, J. (2021). *A Thousand Brains*
- Schmidhuber, J. (2010). *Formal Theory of Creativity, Fun, and Intrinsic Motivation*
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*

## License

MIT License
