//! # Compounding Cognitive Cohesion for AGI-Relevant Learning
//!
//! This module extends the basic cognitive coherence system with full compounding
//! capabilities inspired by Hierarchical Temporal Memory (HTM) and Monty's
//! sensorimotor learning framework.
//!
//! ## Key Additions Over Basic Coherence
//!
//! 1. **Hierarchical Coherence**: Multi-level SOC tracking (local, regional, global)
//! 2. **Graph-Based Memory**: Persistent feature graphs for episodic memory
//! 3. **Predictive Coherence**: Future state prediction with error-driven learning
//! 4. **Cross-Layer SMM**: Alignment tracking between layers, not just streams
//! 5. **Goal State Generation**: Proposals for sensorimotor closure
//! 6. **Meta-Learning**: Adaptive coherence weights based on task performance
//! 7. **Memory Consolidation**: Post-episode buffer-to-graph merging
//!
//! ## Theoretical Foundation
//!
//! ```text
//! COMPOUNDING COGNITIVE COHESION (Full AGI Framework)
//! ═══════════════════════════════════════════════════
//!
//! Layer N coherence ──────┐
//!          ↓              │
//! Layer N+1 coherence ←───┤ (hierarchical feedback)
//!          ↓              │
//! Layer N+2 coherence ←───┘
//!          ↓
//! Meta-coherence (coherence OF coherence)
//!          ↓
//! Self-modification of coherence params
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::compounding_cohesion::{
//!     CompoundingCohesionSystem, HierarchicalCoherence, GraphMemory,
//! };
//!
//! let mut system = CompoundingCohesionSystem::new(config, device);
//!
//! // Per-inference compounding
//! for layer_idx in 0..n_layers {
//!     let output = system.compound_forward(layer_idx, &hidden, &attention)?;
//! }
//!
//! // Cross-inference compounding (episodic)
//! system.post_episode_consolidation()?;
//!
//! // Goal state for sensorimotor closure
//! let goal = system.propose_goal_state()?;
//! ```

use crate::coherence::{
    CognitiveCoherenceLayer, CoherenceConfig, SenseOfCoherence, SharedMentalModel,
};
use crate::TorusResult;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the full compounding cognitive cohesion system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundingCohesionConfig {
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of parallel streams (8 for torus)
    pub n_streams: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of hierarchical levels (local, regional, global)
    pub n_hierarchy_levels: usize,
    /// Layers per regional group
    pub layers_per_region: usize,
    /// Maximum graph nodes per stream
    pub max_graph_nodes: usize,
    /// Feature dimension for graph nodes
    pub graph_feature_dim: usize,
    /// History length for predictive coherence
    pub prediction_history: usize,
    /// Learning rate for meta-learning
    pub meta_learning_rate: f64,
    /// Learning rate for graph updates
    pub graph_learning_rate: f64,
    /// Similarity threshold for graph node merging
    pub merge_threshold: f64,
    /// Enable predictive coherence
    pub use_prediction: bool,
    /// Enable goal state generation
    pub use_goal_states: bool,
    /// Enable memory consolidation
    pub use_consolidation: bool,
    /// Base coherence config
    pub base_coherence: CoherenceConfig,
}

impl Default for CompoundingCohesionConfig {
    fn default() -> Self {
        Self {
            n_layers: 6,
            n_streams: 8,
            d_model: 256,
            n_hierarchy_levels: 3, // local, regional, global
            layers_per_region: 2,
            max_graph_nodes: 1024,
            graph_feature_dim: 64,
            prediction_history: 10,
            meta_learning_rate: 0.001,
            graph_learning_rate: 0.01,
            merge_threshold: 0.9,
            use_prediction: true,
            use_goal_states: true,
            use_consolidation: true,
            base_coherence: CoherenceConfig::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRAPH-BASED EPISODIC MEMORY (Monty-inspired)
// ═══════════════════════════════════════════════════════════════════════════════

/// A node in the feature graph representing a learned pattern
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier
    pub id: usize,
    /// Feature vector at this node
    pub features: Vec<f32>,
    /// Activation count (for importance weighting)
    pub activation_count: usize,
    /// Last activation step
    pub last_activation: usize,
    /// Confidence/evidence score
    pub confidence: f64,
}

impl GraphNode {
    pub fn new(id: usize, features: Vec<f32>) -> Self {
        Self {
            id,
            features,
            activation_count: 1,
            last_activation: 0,
            confidence: 0.5,
        }
    }

    /// Compute cosine similarity with another feature vector
    pub fn similarity(&self, other: &[f32]) -> f64 {
        if self.features.len() != other.len() {
            return 0.0;
        }

        let dot: f32 = self
            .features
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_self: f32 = self.features.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self > 1e-8 && norm_other > 1e-8 {
            (dot / (norm_self * norm_other)) as f64
        } else {
            0.0
        }
    }

    /// Update features with exponential moving average
    pub fn update_features(&mut self, new_features: &[f32], learning_rate: f64) {
        for (old, new) in self.features.iter_mut().zip(new_features.iter()) {
            *old = (1.0 - learning_rate as f32) * *old + learning_rate as f32 * new;
        }
        self.activation_count += 1;
    }
}

/// Edge connecting two graph nodes
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID
    pub from: usize,
    /// Target node ID
    pub to: usize,
    /// Displacement vector (for pose-like relationships)
    pub displacement: Vec<f32>,
    /// Edge weight (learned)
    pub weight: f64,
    /// Traversal count
    pub traversal_count: usize,
    /// Cumulative reward observed when traversing this edge
    pub cumulative_reward: f64,
    /// Average reward per traversal
    pub avg_reward: f64,
}

/// Temporal transition record (for sequence learning)
#[derive(Debug, Clone)]
pub struct TemporalTransition {
    /// Features at time t
    pub features: Vec<f32>,
    /// Reward received after this state
    pub reward: f64,
    /// Timestamp (step within episode)
    pub step: usize,
}

/// Eligibility trace for an edge (for TD-learning style credit assignment)
#[derive(Debug, Clone)]
pub struct EligibilityTrace {
    /// Edge key (from, to)
    pub edge_key: (usize, usize),
    /// Current trace value (decays over time)
    pub trace: f64,
    /// Step when last updated
    pub last_step: usize,
}

/// A complete trajectory for experience replay
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Sequence of (features, reward) pairs
    pub transitions: Vec<TemporalTransition>,
    /// Total reward of this trajectory
    pub total_reward: f64,
    /// Episode this trajectory came from
    pub episode: usize,
}

/// Successor representation entry: expected future state features
#[derive(Debug, Clone)]
pub struct SuccessorRepresentation {
    /// Node ID this SR is for
    pub node_id: usize,
    /// Expected discounted future features (weighted average of reachable states)
    pub expected_features: Vec<f32>,
    /// Discount factor used
    pub gamma: f64,
    /// Number of updates
    pub update_count: usize,
}

/// Intrinsic motivation signals beyond curiosity
#[derive(Debug, Clone, Default)]
pub struct IntrinsicMotivation {
    /// Competence progress: improvement in prediction accuracy
    pub competence_progress: f64,
    /// Learning progress: rate of model change
    pub learning_progress: f64,
    /// Empowerment: perceived control over environment
    pub empowerment: f64,
    /// Information gain: reduction in uncertainty
    pub information_gain: f64,
}

/// Graph type for separate spatial vs semantic memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    /// Spatial memory (position-based)
    Spatial,
    /// Semantic memory (neural state-based)
    Semantic,
}

/// Graph-based memory for a single stream
#[derive(Debug, Clone)]
pub struct StreamGraphMemory {
    /// Stream identifier
    pub stream_id: usize,
    /// Graph type (spatial or semantic)
    pub graph_type: GraphType,
    /// Nodes in the graph
    pub nodes: Vec<GraphNode>,
    /// Edges between nodes (adjacency)
    pub edges: HashMap<(usize, usize), GraphEdge>,
    /// Feature buffer (short-term, pre-consolidation)
    pub buffer: VecDeque<Vec<f32>>,
    /// Temporal transition buffer (features + rewards for sequence learning)
    pub transition_buffer: VecDeque<TemporalTransition>,
    /// Eligibility traces for edges (for backward credit assignment)
    pub eligibility_traces: HashMap<(usize, usize), f64>,
    /// High-reward trajectory replay buffer
    pub replay_buffer: VecDeque<Trajectory>,
    /// Successor representations for multi-step planning
    pub successor_reps: HashMap<usize, SuccessorRepresentation>,
    /// Intrinsic motivation state
    pub intrinsic_motivation: IntrinsicMotivation,
    /// Previous prediction error (for learning progress)
    prev_prediction_error: f64,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Replay buffer capacity
    pub replay_capacity: usize,
    /// Next node ID
    next_node_id: usize,
    /// Configuration
    max_nodes: usize,
    feature_dim: usize,
    merge_threshold: f64,
    /// Current step within episode (for temporal transitions)
    current_step: usize,
    /// Eligibility trace decay rate (lambda in TD(λ))
    trace_decay: f64,
    /// Successor representation discount factor
    sr_gamma: f64,
}

impl StreamGraphMemory {
    pub fn new(
        stream_id: usize,
        max_nodes: usize,
        feature_dim: usize,
        merge_threshold: f64,
    ) -> Self {
        Self::new_with_type(
            stream_id,
            max_nodes,
            feature_dim,
            merge_threshold,
            GraphType::Spatial,
        )
    }

    pub fn new_with_type(
        stream_id: usize,
        max_nodes: usize,
        feature_dim: usize,
        merge_threshold: f64,
        graph_type: GraphType,
    ) -> Self {
        Self {
            stream_id,
            graph_type,
            nodes: Vec::new(),
            edges: HashMap::new(),
            buffer: VecDeque::with_capacity(256),
            transition_buffer: VecDeque::with_capacity(256),
            eligibility_traces: HashMap::new(),
            replay_buffer: VecDeque::with_capacity(50), // Store top 50 trajectories
            successor_reps: HashMap::new(),
            intrinsic_motivation: IntrinsicMotivation::default(),
            prev_prediction_error: 0.5,
            buffer_capacity: 256,
            replay_capacity: 50,
            next_node_id: 0,
            max_nodes,
            feature_dim,
            merge_threshold,
            current_step: 0,
            trace_decay: 0.9, // Lambda = 0.9 for TD(λ)
            sr_gamma: 0.95,   // Discount for successor representations
        }
    }

    /// Add features to the short-term buffer
    pub fn buffer_features(&mut self, features: Vec<f32>) {
        self.buffer.push_back(features);
        if self.buffer.len() > self.buffer_capacity {
            self.buffer.pop_front();
        }
    }

    /// Add features with reward to temporal transition buffer (for sequence learning)
    pub fn buffer_transition(&mut self, features: Vec<f32>, reward: f64) {
        self.transition_buffer.push_back(TemporalTransition {
            features,
            reward,
            step: self.current_step,
        });
        self.current_step += 1;
        if self.transition_buffer.len() > self.buffer_capacity {
            self.transition_buffer.pop_front();
        }
    }

    /// Reset step counter (call at start of episode)
    pub fn reset_episode(&mut self) {
        self.current_step = 0;
        // Clear eligibility traces at episode start
        self.eligibility_traces.clear();
    }

    /// Mark an edge as recently traversed (set eligibility trace to 1.0)
    pub fn mark_edge_traversed(&mut self, from: usize, to: usize) {
        self.eligibility_traces.insert((from, to), 1.0);
    }

    /// Decay all eligibility traces (call each step)
    pub fn decay_traces(&mut self) {
        for trace in self.eligibility_traces.values_mut() {
            *trace *= self.trace_decay;
        }
        // Remove traces that are too small
        self.eligibility_traces.retain(|_, v| *v > 0.01);
    }

    /// Update all edges with eligibility traces using received reward
    /// This propagates reward backward through recently visited edges
    pub fn update_traces_with_reward(&mut self, reward: f64, learning_rate: f64) {
        for ((from, to), trace) in &self.eligibility_traces {
            if let Some(edge) = self.edges.get_mut(&(*from, *to)) {
                // TD-style update: edge_reward += learning_rate * trace * reward
                let delta = learning_rate * trace * reward;
                edge.cumulative_reward += delta;
                edge.avg_reward = edge.cumulative_reward / edge.traversal_count.max(1) as f64;
                // Also update weight
                edge.weight = (edge.weight + delta * 0.1).clamp(0.01, 2.0);
            }
        }
    }

    /// Compute curiosity bonus for features (lower similarity = higher curiosity)
    pub fn curiosity_bonus(&self, features: &[f32]) -> f64 {
        if let Some((_, similarity)) = self.find_nearest(features) {
            // Curiosity is inversely related to similarity
            // Low similarity (novel state) = high curiosity
            // High similarity (familiar state) = low curiosity
            let novelty = 1.0 - similarity;
            // Scale to reasonable bonus range [0, 1]
            novelty.max(0.0).min(1.0)
        } else {
            // No nodes yet = maximum curiosity
            1.0
        }
    }

    /// Find the most similar node to given features
    pub fn find_nearest(&self, features: &[f32]) -> Option<(usize, f64)> {
        self.nodes
            .iter()
            .map(|node| (node.id, node.similarity(features)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Add a new node or update existing if similar
    pub fn add_or_update(&mut self, features: Vec<f32>, learning_rate: f64) -> usize {
        if let Some((id, similarity)) = self.find_nearest(&features) {
            if similarity >= self.merge_threshold {
                // Update existing node
                if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
                    node.update_features(&features, learning_rate);
                    return id;
                }
            }
        }

        // Add new node
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(GraphNode::new(id, features));

        // Prune if over capacity (remove least activated)
        if self.nodes.len() > self.max_nodes {
            self.nodes
                .sort_by(|a, b| b.activation_count.cmp(&a.activation_count));
            self.nodes.truncate(self.max_nodes);
        }

        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize, displacement: Vec<f32>) {
        let key = (from, to);
        if let Some(edge) = self.edges.get_mut(&key) {
            edge.traversal_count += 1;
            edge.weight = (edge.weight * 0.9) + 0.1; // Strengthen
        } else {
            self.edges.insert(
                key,
                GraphEdge {
                    from,
                    to,
                    displacement,
                    weight: 0.5,
                    traversal_count: 1,
                    cumulative_reward: 0.0,
                    avg_reward: 0.0,
                },
            );
        }
    }

    /// Add an edge with reward information
    pub fn add_edge_with_reward(
        &mut self,
        from: usize,
        to: usize,
        displacement: Vec<f32>,
        reward: f64,
    ) {
        let key = (from, to);
        if let Some(edge) = self.edges.get_mut(&key) {
            edge.traversal_count += 1;
            edge.cumulative_reward += reward;
            edge.avg_reward = edge.cumulative_reward / edge.traversal_count as f64;
            // Weight increases more for positive rewards
            edge.weight = (edge.weight * 0.9) + 0.1 * (1.0 + reward.max(-1.0).min(1.0));
        } else {
            self.edges.insert(
                key,
                GraphEdge {
                    from,
                    to,
                    displacement,
                    weight: 0.5 + reward.max(-0.5).min(0.5),
                    traversal_count: 1,
                    cumulative_reward: reward,
                    avg_reward: reward,
                },
            );
        }
    }

    /// Update edge reward (after observing result of traversal)
    pub fn update_edge_reward(&mut self, from: usize, to: usize, reward: f64) {
        let key = (from, to);
        if let Some(edge) = self.edges.get_mut(&key) {
            edge.cumulative_reward += reward;
            edge.avg_reward = edge.cumulative_reward / edge.traversal_count as f64;
            // Adjust weight based on reward
            let reward_factor = 1.0 + reward.max(-1.0).min(1.0) * 0.1;
            edge.weight = (edge.weight * reward_factor).clamp(0.01, 2.0);
        }
    }

    /// Suggest a direction based on graph memory using temperature-weighted selection
    /// Returns (dx, dy) displacement toward rewarding edges, plus similarity to nearest node
    pub fn suggest_direction(&self, features: &[f32]) -> Option<(f64, f64, f64)> {
        self.suggest_direction_with_temperature(features, 1.0)
    }

    /// Suggest direction with configurable temperature (lower = more greedy)
    pub fn suggest_direction_with_temperature(
        &self,
        features: &[f32],
        temperature: f64,
    ) -> Option<(f64, f64, f64)> {
        // 1. Find nearest node
        let (nearest_id, similarity) = self.find_nearest(features)?;

        if similarity < 0.1 {
            // Very different from anything we've seen - can't suggest
            return None;
        }

        // 2. Find outgoing edges from nearest node
        let outgoing_edges: Vec<&GraphEdge> = self
            .edges
            .values()
            .filter(|e| e.from == nearest_id && e.traversal_count > 0)
            .collect();

        if outgoing_edges.is_empty() {
            // No outgoing edges - try incoming edges (reversed direction)
            let incoming_edges: Vec<&GraphEdge> = self
                .edges
                .values()
                .filter(|e| e.to == nearest_id && e.traversal_count > 0)
                .collect();

            if incoming_edges.is_empty() {
                return None;
            }

            // Temperature-weighted selection from incoming edges (reversed)
            return self.softmax_select_incoming(&incoming_edges, temperature, similarity);
        }

        // 3. Temperature-weighted selection using softmax on rewards
        self.softmax_select_outgoing(&outgoing_edges, temperature, similarity)
    }

    /// Softmax-weighted selection from outgoing edges
    fn softmax_select_outgoing(
        &self,
        edges: &[&GraphEdge],
        temperature: f64,
        similarity: f64,
    ) -> Option<(f64, f64, f64)> {
        if edges.is_empty() {
            return None;
        }

        // Compute softmax weights from rewards
        let temp = temperature.max(0.1); // Prevent division by zero
        let max_reward = edges
            .iter()
            .map(|e| e.avg_reward)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_rewards: Vec<f64> = edges
            .iter()
            .map(|e| ((e.avg_reward - max_reward) / temp).exp())
            .collect();
        let sum_exp: f64 = exp_rewards.iter().sum();

        // Weighted average of displacement directions
        let mut dx_weighted = 0.0;
        let mut dy_weighted = 0.0;

        for (edge, &exp_r) in edges.iter().zip(exp_rewards.iter()) {
            let weight = exp_r / sum_exp;
            let dx = edge.displacement.get(0).copied().unwrap_or(0.0) as f64;
            let dy = edge.displacement.get(1).copied().unwrap_or(0.0) as f64;
            dx_weighted += dx * weight;
            dy_weighted += dy * weight;
        }

        // Normalize direction
        let mag = (dx_weighted * dx_weighted + dy_weighted * dy_weighted)
            .sqrt()
            .max(1e-8);
        Some((dx_weighted / mag, dy_weighted / mag, similarity))
    }

    /// Softmax-weighted selection from incoming edges (reversed)
    fn softmax_select_incoming(
        &self,
        edges: &[&GraphEdge],
        temperature: f64,
        similarity: f64,
    ) -> Option<(f64, f64, f64)> {
        if edges.is_empty() {
            return None;
        }

        // Compute softmax weights from rewards
        let temp = temperature.max(0.1);
        let max_reward = edges
            .iter()
            .map(|e| e.avg_reward)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_rewards: Vec<f64> = edges
            .iter()
            .map(|e| ((e.avg_reward - max_reward) / temp).exp())
            .collect();
        let sum_exp: f64 = exp_rewards.iter().sum();

        // Weighted average of REVERSED displacement directions
        let mut dx_weighted = 0.0;
        let mut dy_weighted = 0.0;

        for (edge, &exp_r) in edges.iter().zip(exp_rewards.iter()) {
            let weight = exp_r / sum_exp;
            // Reverse the displacement (go back toward rewarding state)
            let dx = -edge.displacement.get(0).copied().unwrap_or(0.0) as f64;
            let dy = -edge.displacement.get(1).copied().unwrap_or(0.0) as f64;
            dx_weighted += dx * weight;
            dy_weighted += dy * weight;
        }

        let mag = (dx_weighted * dx_weighted + dy_weighted * dy_weighted)
            .sqrt()
            .max(1e-8);
        Some((dx_weighted / mag, dy_weighted / mag, similarity))
    }

    /// Get best outgoing edge from a node
    pub fn best_outgoing_edge(&self, node_id: usize) -> Option<&GraphEdge> {
        self.edges
            .values()
            .filter(|e| e.from == node_id && e.traversal_count > 0)
            .max_by(|a, b| {
                a.avg_reward
                    .partial_cmp(&b.avg_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Consolidate buffer into graph (post-episode)
    /// Now uses transition buffer for reward-aware edge creation
    pub fn consolidate(&mut self, learning_rate: f64) -> usize {
        // First, consolidate from transition buffer (with rewards)
        let transitions: Vec<TemporalTransition> = self.transition_buffer.drain(..).collect();
        let mut prev_node_id: Option<usize> = None;
        let mut prev_reward: f64 = 0.0;
        let mut nodes_added = 0;

        for transition in transitions {
            let node_id = self.add_or_update(transition.features.clone(), learning_rate);
            nodes_added += 1;

            // Create edge from previous node WITH REWARD
            if let Some(prev_id) = prev_node_id {
                // Displacement is difference between feature vectors
                let displacement =
                    if let Some(prev_node) = self.nodes.iter().find(|n| n.id == prev_id) {
                        transition
                            .features
                            .iter()
                            .zip(prev_node.features.iter())
                            .map(|(a, b)| a - b)
                            .collect()
                    } else {
                        vec![0.0; self.feature_dim]
                    };
                // Use the reward received AFTER the previous state (temporal credit assignment)
                // The reward at transition t is assigned to edge (t-1) -> t
                self.add_edge_with_reward(prev_id, node_id, displacement, prev_reward);
            }
            prev_node_id = Some(node_id);
            prev_reward = transition.reward;
        }

        // Also consolidate from regular buffer (legacy support)
        let buffer_contents: Vec<Vec<f32>> = self.buffer.drain(..).collect();
        for features in buffer_contents {
            let node_id = self.add_or_update(features.clone(), learning_rate);
            nodes_added += 1;

            if let Some(prev_id) = prev_node_id {
                let displacement =
                    if let Some(prev_node) = self.nodes.iter().find(|n| n.id == prev_id) {
                        features
                            .iter()
                            .zip(prev_node.features.iter())
                            .map(|(a, b)| a - b)
                            .collect()
                    } else {
                        vec![0.0; self.feature_dim]
                    };
                self.add_edge(prev_id, node_id, displacement);
            }
            prev_node_id = Some(node_id);
        }

        // Reset step counter for next episode
        self.current_step = 0;

        nodes_added
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            buffer_size: self.buffer.len(),
            avg_activation: if self.nodes.is_empty() {
                0.0
            } else {
                self.nodes
                    .iter()
                    .map(|n| n.activation_count as f64)
                    .sum::<f64>()
                    / self.nodes.len() as f64
            },
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // EXPERIENCE REPLAY
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Store a trajectory in the replay buffer (prioritized by reward)
    pub fn store_trajectory(&mut self, episode: usize) {
        if self.transition_buffer.is_empty() {
            return;
        }

        let transitions: Vec<TemporalTransition> = self.transition_buffer.iter().cloned().collect();
        let total_reward: f64 = transitions.iter().map(|t| t.reward).sum();

        let trajectory = Trajectory {
            transitions,
            total_reward,
            episode,
        };

        // Insert sorted by reward (highest first)
        let insert_pos = self
            .replay_buffer
            .iter()
            .position(|t| t.total_reward < total_reward)
            .unwrap_or(self.replay_buffer.len());

        self.replay_buffer.insert(insert_pos, trajectory);

        // Keep only top trajectories
        while self.replay_buffer.len() > self.replay_capacity {
            self.replay_buffer.pop_back();
        }
    }

    /// Replay high-reward trajectories to strengthen good paths
    pub fn replay_trajectories(&mut self, n_replays: usize, learning_rate: f64) {
        if self.replay_buffer.is_empty() {
            return;
        }

        // Replay top trajectories
        let n = n_replays.min(self.replay_buffer.len());
        for i in 0..n {
            if let Some(trajectory) = self.replay_buffer.get(i).cloned() {
                // Strengthen edges along this trajectory
                let mut prev_node_id: Option<usize> = None;
                for transition in &trajectory.transitions {
                    if let Some((node_id, _)) = self.find_nearest(&transition.features) {
                        if let Some(prev_id) = prev_node_id {
                            // Boost this edge's reward
                            let key = (prev_id, node_id);
                            if let Some(edge) = self.edges.get_mut(&key) {
                                let boost = learning_rate * transition.reward * 0.5; // Decay for replay
                                edge.cumulative_reward += boost;
                                edge.avg_reward =
                                    edge.cumulative_reward / edge.traversal_count.max(1) as f64;
                                edge.weight = (edge.weight + boost * 0.05).clamp(0.01, 3.0);
                            }
                        }
                        prev_node_id = Some(node_id);
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // SUCCESSOR REPRESENTATIONS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Update successor representation for a node
    /// SR(s) = expected discounted future features starting from s
    pub fn update_successor_rep(
        &mut self,
        node_id: usize,
        next_features: &[f32],
        learning_rate: f64,
    ) {
        let gamma = self.sr_gamma;

        if let Some(node) = self.nodes.iter().find(|n| n.id == node_id) {
            let current_features = node.features.clone();

            let sr =
                self.successor_reps
                    .entry(node_id)
                    .or_insert_with(|| SuccessorRepresentation {
                        node_id,
                        expected_features: current_features.clone(),
                        gamma,
                        update_count: 0,
                    });

            // TD update: SR(s) = features(s) + gamma * SR(s')
            // We approximate SR(s') with next_features for now
            for (i, (sr_f, (curr_f, next_f))) in sr
                .expected_features
                .iter_mut()
                .zip(current_features.iter().zip(next_features.iter()))
                .enumerate()
            {
                let target = *curr_f + (gamma as f32) * next_f;
                *sr_f = *sr_f * (1.0 - learning_rate as f32) + target * learning_rate as f32;
            }
            sr.update_count += 1;
        }
    }

    /// Get expected future value from successor representation
    /// Computes dot product of SR with reward weights (features)
    pub fn successor_value(&self, node_id: usize, reward_features: &[f32]) -> f64 {
        if let Some(sr) = self.successor_reps.get(&node_id) {
            // Dot product of SR features with reward weights
            sr.expected_features
                .iter()
                .zip(reward_features.iter())
                .map(|(s, r)| (*s as f64) * (*r as f64))
                .sum()
        } else {
            0.0
        }
    }

    /// Plan multi-step using successor representations
    /// Returns expected cumulative reward for taking edge from current node
    pub fn plan_with_sr(
        &self,
        current_features: &[f32],
        reward_features: &[f32],
    ) -> Option<(f64, f64)> {
        let (nearest_id, similarity) = self.find_nearest(current_features)?;

        if similarity < 0.3 {
            return None; // Too unfamiliar
        }

        // Find best outgoing edge considering successor values
        let mut best_value = f64::NEG_INFINITY;
        let mut best_direction = (0.0, 0.0);

        for edge in self.edges.values().filter(|e| e.from == nearest_id) {
            // Immediate reward + discounted future value
            let immediate = edge.avg_reward;
            let future = self.successor_value(edge.to, reward_features) * self.sr_gamma;
            let total_value = immediate + future;

            if total_value > best_value {
                best_value = total_value;
                let dx = edge.displacement.get(0).copied().unwrap_or(0.0) as f64;
                let dy = edge.displacement.get(1).copied().unwrap_or(0.0) as f64;
                best_direction = (dx, dy);
            }
        }

        if best_value > f64::NEG_INFINITY {
            Some(best_direction)
        } else {
            None
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // INTRINSIC MOTIVATION
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Update intrinsic motivation signals
    pub fn update_intrinsic_motivation(&mut self, prediction_error: f64, model_change: f64) {
        let im = &mut self.intrinsic_motivation;

        // Competence progress: improvement in prediction
        let error_improvement = self.prev_prediction_error - prediction_error;
        im.competence_progress = im.competence_progress * 0.9 + error_improvement * 0.1;

        // Learning progress: rate of model change
        im.learning_progress = im.learning_progress * 0.9 + model_change * 0.1;

        // Information gain: based on graph growth
        let node_density = self.nodes.len() as f64 / self.max_nodes as f64;
        im.information_gain = (1.0 - node_density) * 0.5; // Higher when graph is sparse

        // Empowerment: number of reachable states from current position
        let avg_outgoing = if self.nodes.is_empty() {
            0.0
        } else {
            self.edges.len() as f64 / self.nodes.len() as f64
        };
        im.empowerment = (avg_outgoing / 5.0).min(1.0); // Normalize by expected connectivity

        self.prev_prediction_error = prediction_error;
    }

    /// Get total intrinsic motivation bonus
    pub fn intrinsic_bonus(&self) -> f64 {
        let im = &self.intrinsic_motivation;
        // Weighted combination of intrinsic signals
        let competence_weight = 0.3;
        let learning_weight = 0.3;
        let info_weight = 0.2;
        let empower_weight = 0.2;

        (im.competence_progress.max(0.0) * competence_weight
            + im.learning_progress.max(0.0) * learning_weight
            + im.information_gain * info_weight
            + im.empowerment * empower_weight)
            .clamp(0.0, 1.0)
    }

    /// Get detailed intrinsic motivation breakdown
    pub fn intrinsic_motivation(&self) -> &IntrinsicMotivation {
        &self.intrinsic_motivation
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub buffer_size: usize,
    pub avg_activation: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIERARCHICAL COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Hierarchical level for coherence tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyLevel {
    /// Per-layer coherence
    Local,
    /// Regional coherence (groups of layers)
    Regional,
    /// Global coherence (whole model)
    Global,
}

/// SOC state at a specific hierarchy level
#[derive(Debug, Clone)]
pub struct HierarchicalSOC {
    /// Level in hierarchy
    pub level: HierarchyLevel,
    /// SOC state
    pub soc: SenseOfCoherence,
    /// History for trend analysis
    pub history: VecDeque<f64>,
    /// Maximum history length
    pub max_history: usize,
}

impl HierarchicalSOC {
    pub fn new(level: HierarchyLevel, max_history: usize) -> Self {
        Self {
            level,
            soc: SenseOfCoherence::healthy(),
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Update from child SOC states
    pub fn update_from_children(&mut self, children: &[SenseOfCoherence]) {
        if children.is_empty() {
            return;
        }

        let n = children.len() as f64;
        let comp = children.iter().map(|c| c.comprehensibility).sum::<f64>() / n;
        let man = children.iter().map(|c| c.manageability).sum::<f64>() / n;
        let mean = children.iter().map(|c| c.meaningfulness).sum::<f64>() / n;

        self.soc = SenseOfCoherence::new(comp, man, mean);
        self.history.push_back(self.soc.score());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Compute trend
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self.history.iter().rev().take(10).copied().collect();
        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i * i) as f64).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }
}

/// Multi-level hierarchical coherence tracking
#[derive(Debug)]
pub struct HierarchicalCoherence {
    /// Per-layer (local) coherence
    pub local: Vec<CognitiveCoherenceLayer>,
    /// Regional coherence (groups of layers)
    pub regional: Vec<HierarchicalSOC>,
    /// Global coherence (whole model)
    pub global: HierarchicalSOC,
    /// Configuration
    pub config: CompoundingCohesionConfig,
    /// Device
    device: Device,
}

impl HierarchicalCoherence {
    pub fn new(config: CompoundingCohesionConfig, device: &Device) -> Self {
        // Create per-layer coherence
        let local: Vec<CognitiveCoherenceLayer> = (0..config.n_layers)
            .map(|_| CognitiveCoherenceLayer::new(config.base_coherence.clone(), device))
            .collect();

        // Create regional coherence
        let n_regions = (config.n_layers + config.layers_per_region - 1) / config.layers_per_region;
        let regional: Vec<HierarchicalSOC> = (0..n_regions)
            .map(|_| HierarchicalSOC::new(HierarchyLevel::Regional, config.prediction_history))
            .collect();

        // Create global coherence
        let global = HierarchicalSOC::new(HierarchyLevel::Global, config.prediction_history);

        Self {
            local,
            regional,
            global,
            config,
            device: device.clone(),
        }
    }

    /// Update local coherence for a specific layer
    pub fn update_local(
        &mut self,
        layer_idx: usize,
        attention: &Tensor,
        hidden: &Tensor,
    ) -> TorusResult<SenseOfCoherence> {
        if layer_idx >= self.local.len() {
            return Ok(SenseOfCoherence::default());
        }
        self.local[layer_idx].update_soc(attention, hidden)
    }

    /// Propagate coherence upward through hierarchy
    pub fn propagate_upward(&mut self) {
        // Update regional from local
        for (region_idx, regional_soc) in self.regional.iter_mut().enumerate() {
            let start_layer = region_idx * self.config.layers_per_region;
            let end_layer = (start_layer + self.config.layers_per_region).min(self.local.len());

            let local_socs: Vec<SenseOfCoherence> = (start_layer..end_layer)
                .map(|i| self.local[i].soc().clone())
                .collect();

            regional_soc.update_from_children(&local_socs);
        }

        // Update global from regional
        let regional_socs: Vec<SenseOfCoherence> =
            self.regional.iter().map(|r| r.soc.clone()).collect();
        self.global.update_from_children(&regional_socs);
    }

    /// Propagate coherence downward (top-down modulation)
    pub fn propagate_downward(&self) -> Vec<f64> {
        // Global coherence modulates regional, which modulates local
        let global_mod = self.global.soc.score();

        self.local
            .iter()
            .enumerate()
            .map(|(layer_idx, _local)| {
                let region_idx = layer_idx / self.config.layers_per_region;
                let regional_mod = if region_idx < self.regional.len() {
                    self.regional[region_idx].soc.score()
                } else {
                    0.5
                };

                // Blend global and regional modulation
                0.3 * global_mod + 0.7 * regional_mod
            })
            .collect()
    }

    /// Get adaptive alpha for a layer considering hierarchy
    pub fn hierarchical_alpha(&self, layer_idx: usize, base_alpha: f64) -> f64 {
        if layer_idx >= self.local.len() {
            return base_alpha;
        }

        let local_alpha = self.local[layer_idx].compute_adaptive_alpha();

        // Get hierarchical modulation
        let modulation = self.propagate_downward();
        let layer_mod = modulation.get(layer_idx).copied().unwrap_or(0.5);

        // Blend local alpha with hierarchical modulation
        let blended = 0.6 * local_alpha + 0.4 * (base_alpha * (0.5 + 0.5 * layer_mod));

        blended.clamp(0.1, 0.99)
    }

    /// Get global coherence score
    pub fn global_coherence(&self) -> f64 {
        self.global.soc.score()
    }

    /// Get coherence summary
    pub fn summary(&self) -> String {
        let local_avg: f64 = self
            .local
            .iter()
            .map(|l| l.psychological_coherence())
            .sum::<f64>()
            / self.local.len() as f64;
        let regional_avg: f64 = self.regional.iter().map(|r| r.soc.score()).sum::<f64>()
            / self.regional.len().max(1) as f64;

        format!(
            "Hierarchical Coherence:\n\
             ├─ Local (avg):    {:.3}\n\
             ├─ Regional (avg): {:.3}\n\
             ├─ Global:         {:.3}\n\
             └─ Global Trend:   {:+.4}",
            local_avg,
            regional_avg,
            self.global.soc.score(),
            self.global.trend()
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-LAYER SMM (Alignment between layers, not just streams)
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared mental model tracking alignment between layers
#[derive(Debug, Clone)]
pub struct CrossLayerSMM {
    /// Alignment matrix: [n_layers, n_layers]
    pub alignment: Vec<Vec<f64>>,
    /// Number of layers
    pub n_layers: usize,
    /// Learning rate for updates
    pub learning_rate: f64,
}

impl CrossLayerSMM {
    pub fn new(n_layers: usize, learning_rate: f64) -> Self {
        // Initialize with adjacent layers having higher alignment
        let mut alignment = vec![vec![0.5; n_layers]; n_layers];
        for i in 0..n_layers {
            alignment[i][i] = 1.0; // Self-alignment is perfect
            if i > 0 {
                alignment[i][i - 1] = 0.8;
                alignment[i - 1][i] = 0.8;
            }
        }

        Self {
            alignment,
            n_layers,
            learning_rate,
        }
    }

    /// Update alignment based on layer outputs
    pub fn update_from_outputs(&mut self, layer_outputs: &[Tensor]) -> TorusResult<()> {
        if layer_outputs.len() != self.n_layers {
            return Ok(());
        }

        // Compute pairwise cosine similarity
        for i in 0..self.n_layers {
            for j in (i + 1)..self.n_layers {
                let flat_i = layer_outputs[i].flatten_all()?;
                let flat_j = layer_outputs[j].flatten_all()?;

                let dot = (&flat_i * &flat_j)?.sum_all()?.to_scalar::<f32>()? as f64;
                let norm_i = flat_i.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
                let norm_j = flat_j.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;

                let similarity = if norm_i > 1e-8 && norm_j > 1e-8 {
                    (dot / (norm_i * norm_j)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                let new_alignment = (similarity + 1.0) / 2.0;

                // EMA update
                self.alignment[i][j] = (1.0 - self.learning_rate) * self.alignment[i][j]
                    + self.learning_rate * new_alignment;
                self.alignment[j][i] = self.alignment[i][j];
            }
        }

        Ok(())
    }

    /// Get inter-layer coherence (average off-diagonal alignment)
    pub fn inter_layer_coherence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.n_layers {
            for j in (i + 1)..self.n_layers {
                sum += self.alignment[i][j];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Get alignment between specific layers
    pub fn layer_alignment(&self, layer_a: usize, layer_b: usize) -> f64 {
        if layer_a < self.n_layers && layer_b < self.n_layers {
            self.alignment[layer_a][layer_b]
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTIVE COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Predictive coherence that forecasts future SOC states
#[derive(Debug, Clone)]
pub struct PredictiveCoherence {
    /// History of SOC states
    pub soc_history: VecDeque<SenseOfCoherence>,
    /// Predicted next SOC
    pub predicted_soc: SenseOfCoherence,
    /// Prediction error (drives learning)
    pub prediction_error: f64,
    /// Error history for tracking
    pub error_history: VecDeque<f64>,
    /// Maximum history length
    pub max_history: usize,
    /// Prediction weights (learned)
    pub weights: Vec<f64>,
}

impl PredictiveCoherence {
    pub fn new(max_history: usize) -> Self {
        // Simple linear prediction weights
        let weights: Vec<f64> = (0..max_history).map(|i| 1.0 / (i + 1) as f64).collect();
        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        Self {
            soc_history: VecDeque::with_capacity(max_history),
            predicted_soc: SenseOfCoherence::healthy(),
            prediction_error: 0.0,
            error_history: VecDeque::with_capacity(max_history),
            max_history,
            weights,
        }
    }

    /// Record actual SOC and compute prediction error
    pub fn record(&mut self, actual: SenseOfCoherence) {
        // Compute prediction error
        self.prediction_error = (self.predicted_soc.score() - actual.score()).abs();
        self.error_history.push_back(self.prediction_error);
        if self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        // Add to history
        self.soc_history.push_back(actual);
        if self.soc_history.len() > self.max_history {
            self.soc_history.pop_front();
        }

        // Update prediction for next step
        self.predict_next();
    }

    /// Predict the next SOC state
    pub fn predict_next(&mut self) {
        if self.soc_history.is_empty() {
            return;
        }

        let history: Vec<&SenseOfCoherence> = self.soc_history.iter().rev().collect();
        let n = history.len().min(self.weights.len());

        let mut pred_comp = 0.0;
        let mut pred_man = 0.0;
        let mut pred_mean = 0.0;
        let mut weight_sum = 0.0;

        for (i, soc) in history.iter().take(n).enumerate() {
            let w = self.weights[i];
            pred_comp += w * soc.comprehensibility;
            pred_man += w * soc.manageability;
            pred_mean += w * soc.meaningfulness;
            weight_sum += w;
        }

        if weight_sum > 0.0 {
            // Add trend extrapolation
            let trend = self.trend();
            pred_comp = (pred_comp / weight_sum + trend * 0.1).clamp(0.0, 1.0);
            pred_man = (pred_man / weight_sum + trend * 0.1).clamp(0.0, 1.0);
            pred_mean = (pred_mean / weight_sum + trend * 0.1).clamp(0.0, 1.0);
        }

        self.predicted_soc = SenseOfCoherence::new(pred_comp, pred_man, pred_mean);
    }

    /// Compute trend in SOC scores
    pub fn trend(&self) -> f64 {
        if self.soc_history.len() < 2 {
            return 0.0;
        }

        let scores: Vec<f64> = self.soc_history.iter().map(|s| s.score()).collect();
        let n = scores.len();
        if n < 2 {
            return 0.0;
        }

        // Simple difference
        let recent = scores.iter().rev().take(5).copied().collect::<Vec<_>>();
        if recent.len() < 2 {
            return 0.0;
        }

        (recent[0] - recent[recent.len() - 1]) / recent.len() as f64
    }

    /// Get average prediction error
    pub fn avg_error(&self) -> f64 {
        if self.error_history.is_empty() {
            return 0.0;
        }
        self.error_history.iter().sum::<f64>() / self.error_history.len() as f64
    }

    /// Update weights based on prediction error (meta-learning)
    pub fn update_weights(&mut self, learning_rate: f64) {
        if self.soc_history.len() < 2 {
            return;
        }

        // Gradient descent on prediction error
        // Increase weights for more recent states if prediction is undershooting
        let error_sign = if self.prediction_error > 0.0 {
            1.0
        } else {
            -1.0
        };

        for (i, w) in self.weights.iter_mut().enumerate() {
            let gradient = error_sign * (1.0 / (i + 1) as f64);
            *w = (*w - learning_rate * gradient).clamp(0.0, 1.0);
        }

        // Renormalize
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOAL STATE GENERATION (Sensorimotor Closure)
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of goal state to propose
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalType {
    /// Refine current hypothesis (exploitation)
    Refine,
    /// Explore for new information (exploration)
    Explore,
    /// Disambiguate between competing hypotheses
    Disambiguate,
    /// No action needed (confident state)
    None,
}

/// A proposed goal state for sensorimotor action
#[derive(Debug, Clone)]
pub struct GoalState {
    /// Type of goal
    pub goal_type: GoalType,
    /// Confidence in goal proposal (0-1)
    pub confidence: f64,
    /// Target layer for attention (if applicable)
    pub target_layer: Option<usize>,
    /// Target stream for attention (if applicable)
    pub target_stream: Option<usize>,
    /// Uncertainty that triggered this goal
    pub uncertainty_source: f64,
}

impl GoalState {
    pub fn none() -> Self {
        Self {
            goal_type: GoalType::None,
            confidence: 1.0,
            target_layer: None,
            target_stream: None,
            uncertainty_source: 0.0,
        }
    }

    pub fn explore(target_layer: usize, uncertainty: f64) -> Self {
        Self {
            goal_type: GoalType::Explore,
            confidence: 1.0 - uncertainty,
            target_layer: Some(target_layer),
            target_stream: None,
            uncertainty_source: uncertainty,
        }
    }

    pub fn refine(target_stream: usize) -> Self {
        Self {
            goal_type: GoalType::Refine,
            confidence: 0.8,
            target_layer: None,
            target_stream: Some(target_stream),
            uncertainty_source: 0.2,
        }
    }
}

/// Goal state generator based on coherence metrics
#[derive(Debug)]
pub struct GoalStateGenerator {
    /// Coherence threshold for confident state
    pub coherence_threshold: f64,
    /// Uncertainty threshold for exploration
    pub exploration_threshold: f64,
    /// Last generated goal
    pub last_goal: GoalState,
}

impl GoalStateGenerator {
    pub fn new(coherence_threshold: f64) -> Self {
        Self {
            coherence_threshold,
            exploration_threshold: 0.4,
            last_goal: GoalState::none(),
        }
    }

    /// Generate goal state from hierarchical coherence
    pub fn generate(&mut self, hierarchy: &HierarchicalCoherence) -> GoalState {
        let global = hierarchy.global.soc.score();
        let trend = hierarchy.global.trend();

        // If globally coherent and improving, no action needed
        if global >= self.coherence_threshold && trend >= 0.0 {
            self.last_goal = GoalState::none();
            return self.last_goal.clone();
        }

        // Find layer with lowest coherence (exploration target)
        let local_scores: Vec<f64> = hierarchy
            .local
            .iter()
            .map(|l| l.psychological_coherence())
            .collect();

        if let Some((min_layer, &min_score)) = local_scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if min_score < self.exploration_threshold {
                // Need to explore this layer
                self.last_goal = GoalState::explore(min_layer, 1.0 - min_score);
                return self.last_goal.clone();
            }
        }

        // Find stream with highest alignment (refinement target)
        if let Some(local) = hierarchy.local.first() {
            let weights = local.stream_weights();
            if let Some((max_stream, _)) = weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.last_goal = GoalState::refine(max_stream);
                return self.last_goal.clone();
            }
        }

        self.last_goal = GoalState::none();
        self.last_goal.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// META-LEARNING OF COHERENCE WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive weights for SOC components
#[derive(Debug, Clone)]
pub struct AdaptiveCoherenceWeights {
    /// Weight for comprehensibility
    pub comp_weight: f64,
    /// Weight for manageability
    pub man_weight: f64,
    /// Weight for meaningfulness
    pub mean_weight: f64,
    /// Performance history for meta-learning
    pub performance_history: VecDeque<f64>,
    /// Weight update history
    pub weight_history: VecDeque<(f64, f64, f64)>,
    /// Learning rate
    pub learning_rate: f64,
}

impl AdaptiveCoherenceWeights {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            comp_weight: 0.25,
            man_weight: 0.25,
            mean_weight: 0.50,
            performance_history: VecDeque::with_capacity(100),
            weight_history: VecDeque::with_capacity(100),
            learning_rate,
        }
    }

    /// Compute weighted SOC score
    pub fn weighted_score(&self, soc: &SenseOfCoherence) -> f64 {
        self.comp_weight * soc.comprehensibility
            + self.man_weight * soc.manageability
            + self.mean_weight * soc.meaningfulness
    }

    /// Update weights based on task performance
    pub fn update(&mut self, soc: &SenseOfCoherence, task_performance: f64) {
        self.performance_history.push_back(task_performance);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        self.weight_history
            .push_back((self.comp_weight, self.man_weight, self.mean_weight));
        if self.weight_history.len() > 100 {
            self.weight_history.pop_front();
        }

        if self.performance_history.len() < 10 {
            return;
        }

        // Compute performance gradient (are we improving?)
        let recent: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .take(10)
            .copied()
            .collect();
        let performance_trend = if recent.len() >= 2 {
            recent[0] - recent[recent.len() - 1]
        } else {
            0.0
        };

        // Adjust weights toward components that correlate with improvement
        if performance_trend > 0.0 {
            // Performance improving, reinforce current weighting
            // Increase weight of most active component
            if soc.comprehensibility > soc.manageability
                && soc.comprehensibility > soc.meaningfulness
            {
                self.comp_weight += self.learning_rate;
            } else if soc.manageability > soc.meaningfulness {
                self.man_weight += self.learning_rate;
            } else {
                self.mean_weight += self.learning_rate;
            }
        } else if performance_trend < 0.0 {
            // Performance degrading, shift weights
            if soc.comprehensibility < soc.manageability
                && soc.comprehensibility < soc.meaningfulness
            {
                self.comp_weight += self.learning_rate;
            } else if soc.manageability < soc.meaningfulness {
                self.man_weight += self.learning_rate;
            } else {
                self.mean_weight += self.learning_rate;
            }
        }

        // Normalize weights to sum to 1
        let sum = self.comp_weight + self.man_weight + self.mean_weight;
        if sum > 0.0 {
            self.comp_weight /= sum;
            self.man_weight /= sum;
            self.mean_weight /= sum;
        }

        // Clamp individual weights
        self.comp_weight = self.comp_weight.clamp(0.1, 0.6);
        self.man_weight = self.man_weight.clamp(0.1, 0.6);
        self.mean_weight = self.mean_weight.clamp(0.1, 0.6);

        // Re-normalize after clamping
        let sum = self.comp_weight + self.man_weight + self.mean_weight;
        self.comp_weight /= sum;
        self.man_weight /= sum;
        self.mean_weight /= sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUNDING COHESION SYSTEM (Main Integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete compounding cognitive cohesion system
#[derive(Debug)]
pub struct CompoundingCohesionSystem {
    /// Configuration
    pub config: CompoundingCohesionConfig,
    /// Hierarchical coherence tracking
    pub hierarchy: HierarchicalCoherence,
    /// Cross-layer SMM
    pub cross_layer_smm: CrossLayerSMM,
    /// Per-stream graph memory
    pub stream_graphs: Vec<StreamGraphMemory>,
    /// Predictive coherence
    pub prediction: PredictiveCoherence,
    /// Goal state generator
    pub goal_generator: GoalStateGenerator,
    /// Adaptive weights
    pub adaptive_weights: AdaptiveCoherenceWeights,
    /// Layer outputs buffer (for cross-layer SMM updates)
    pub layer_outputs: Vec<Option<Tensor>>,
    /// Current inference step
    pub inference_step: usize,
    /// Episode counter
    pub episode: usize,
    /// Device
    device: Device,
    /// Adaptive temperature for exploration vs exploitation
    /// Starts high (exploration) and decreases as memory builds up
    pub adaptive_temperature: f64,
}

impl CompoundingCohesionSystem {
    pub fn new(config: CompoundingCohesionConfig, device: &Device) -> Self {
        let hierarchy = HierarchicalCoherence::new(config.clone(), device);
        let cross_layer_smm =
            CrossLayerSMM::new(config.n_layers, config.base_coherence.smm_learning_rate);

        // Stream 0 is Spatial (position-based), others are Semantic (neural state-based)
        let stream_graphs: Vec<StreamGraphMemory> = (0..config.n_streams)
            .map(|i| {
                let graph_type = if i == 0 {
                    GraphType::Spatial
                } else {
                    GraphType::Semantic
                };
                StreamGraphMemory::new_with_type(
                    i,
                    config.max_graph_nodes,
                    config.graph_feature_dim,
                    config.merge_threshold,
                    graph_type,
                )
            })
            .collect();

        let prediction = PredictiveCoherence::new(config.prediction_history);
        let goal_generator = GoalStateGenerator::new(config.base_coherence.coherence_threshold);
        let adaptive_weights = AdaptiveCoherenceWeights::new(config.meta_learning_rate);
        let layer_outputs = vec![None; config.n_layers];

        Self {
            config,
            hierarchy,
            cross_layer_smm,
            stream_graphs,
            prediction,
            goal_generator,
            adaptive_weights,
            layer_outputs,
            inference_step: 0,
            episode: 0,
            device: device.clone(),
            adaptive_temperature: 2.0, // Start with high exploration
        }
    }

    /// Update adaptive temperature based on learning progress
    /// High temp = more exploration, low temp = more exploitation
    pub fn update_temperature(&mut self) {
        // Temperature decreases as:
        // 1. More nodes are accumulated (more knowledge)
        // 2. More episodes have passed (more experience)
        // 3. Coherence is higher (more confidence)

        let total_nodes: usize = self.stream_graphs.iter().map(|g| g.nodes.len()).sum();
        let node_factor = 1.0 / (1.0 + total_nodes as f64 / 50.0); // Decreases with nodes

        let episode_factor = 1.0 / (1.0 + self.episode as f64 / 100.0); // Decreases with episodes

        let coherence = self.hierarchy.global_coherence();
        let coherence_factor = 1.0 - coherence.min(1.0); // Lower coherence = higher temp

        // Combine factors: temperature ranges from 0.2 (exploit) to 2.0 (explore)
        let base_temp = 0.2;
        let temp_range = 1.8;

        self.adaptive_temperature = base_temp
            + temp_range * (node_factor * 0.4 + episode_factor * 0.4 + coherence_factor * 0.2);

        // Clamp to reasonable range
        self.adaptive_temperature = self.adaptive_temperature.clamp(0.2, 2.0);
    }

    /// Get current adaptive temperature
    pub fn temperature(&self) -> f64 {
        self.adaptive_temperature
    }

    /// Main forward compounding step for a layer
    pub fn compound_forward(
        &mut self,
        layer_idx: usize,
        hidden: &Tensor,
        attention: &Tensor,
    ) -> TorusResult<CompoundingResult> {
        // 1. Update local coherence
        let soc = self.hierarchy.update_local(layer_idx, attention, hidden)?;

        // 2. Update predictive coherence
        if self.config.use_prediction {
            self.prediction.record(soc.clone());
        }

        // 3. Store layer output for cross-layer SMM
        if layer_idx < self.layer_outputs.len() {
            self.layer_outputs[layer_idx] = Some(hidden.clone());
        }

        // 4. Compute hierarchical alpha
        let adaptive_alpha = self
            .hierarchy
            .hierarchical_alpha(layer_idx, self.config.base_coherence.base_alpha);

        // 5. Buffer features to stream graphs (distribute across streams)
        if !self.stream_graphs.is_empty() {
            // Extract mean features for graph
            let mean_hidden = hidden.mean(1)?; // [batch, d_model]
            let features: Vec<f32> = mean_hidden.flatten_all()?.to_vec1()?;
            let truncated: Vec<f32> = features
                .iter()
                .take(self.config.graph_feature_dim)
                .copied()
                .collect();

            // Distribute to streams based on layer and coherence
            // Stream 0: All layers (general memory)
            // Stream 1+: Layer-specific (if we have enough streams)
            self.stream_graphs[0].buffer_features(truncated.clone());

            // Also buffer to layer-specific stream if available
            let layer_stream_idx = (layer_idx % self.stream_graphs.len().saturating_sub(1)) + 1;
            if layer_stream_idx < self.stream_graphs.len() {
                // Create layer-modulated features (adds layer identity)
                let layer_features: Vec<f32> = truncated
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| {
                        let layer_mod =
                            ((layer_idx as f32 + 1.0) * (i as f32 + 1.0) * 0.1).sin() * 0.1;
                        f + layer_mod
                    })
                    .collect();
                self.stream_graphs[layer_stream_idx].buffer_features(layer_features);
            }
        }

        self.inference_step += 1;

        Ok(CompoundingResult {
            adaptive_alpha,
            soc_score: soc.score(),
            predicted_soc: self.prediction.predicted_soc.score(),
            prediction_error: self.prediction.prediction_error,
        })
    }

    /// Propagate coherence through hierarchy (call after all layers)
    pub fn propagate_hierarchy(&mut self) -> TorusResult<()> {
        self.hierarchy.propagate_upward();

        // Update cross-layer SMM
        let outputs: Vec<Tensor> = self
            .layer_outputs
            .iter()
            .filter_map(|o| o.clone())
            .collect();

        if outputs.len() == self.config.n_layers {
            self.cross_layer_smm.update_from_outputs(&outputs)?;
        }

        Ok(())
    }

    /// Generate goal state for sensorimotor closure
    pub fn propose_goal_state(&mut self) -> GoalState {
        if !self.config.use_goal_states {
            return GoalState::none();
        }
        self.goal_generator.generate(&self.hierarchy)
    }

    /// Post-episode memory consolidation
    pub fn post_episode_consolidation(&mut self) -> ConsolidationResult {
        if !self.config.use_consolidation {
            return ConsolidationResult::default();
        }

        let mut nodes_added = 0;
        for graph in &mut self.stream_graphs {
            // Store trajectory before consolidation (for experience replay)
            graph.store_trajectory(self.episode);

            // Consolidate buffer into graph
            nodes_added += graph.consolidate(self.config.graph_learning_rate);

            // Experience replay: strengthen high-reward paths
            // Replay top 3 trajectories with reduced learning rate
            graph.replay_trajectories(3, self.config.graph_learning_rate * 0.5);
        }

        // Update prediction weights (meta-learning)
        self.prediction
            .update_weights(self.config.meta_learning_rate);

        self.episode += 1;

        // Clear layer outputs
        for output in &mut self.layer_outputs {
            *output = None;
        }

        ConsolidationResult {
            nodes_added,
            episode: self.episode,
            graph_stats: self.stream_graphs.iter().map(|g| g.stats()).collect(),
        }
    }

    /// Update adaptive weights based on task performance
    pub fn update_meta_learning(&mut self, task_performance: f64) {
        let global_soc = self.hierarchy.global.soc.clone();
        self.adaptive_weights.update(&global_soc, task_performance);
    }

    /// Reset for new episode
    pub fn reset_episode(&mut self) {
        self.inference_step = 0;
        for output in &mut self.layer_outputs {
            *output = None;
        }
    }

    /// Get comprehensive summary
    pub fn summary(&self) -> String {
        let graph_nodes: usize = self.stream_graphs.iter().map(|g| g.nodes.len()).sum();
        let graph_edges: usize = self.stream_graphs.iter().map(|g| g.edges.len()).sum();

        format!(
            "{}\n\
             Cross-Layer Coherence: {:.3}\n\
             Prediction Error (avg): {:.4}\n\
             Graph Memory: {} nodes, {} edges\n\
             Episode: {}, Step: {}\n\
             Adaptive Weights: comp={:.3}, man={:.3}, mean={:.3}\n\
             Goal: {:?}",
            self.hierarchy.summary(),
            self.cross_layer_smm.inter_layer_coherence(),
            self.prediction.avg_error(),
            graph_nodes,
            graph_edges,
            self.episode,
            self.inference_step,
            self.adaptive_weights.comp_weight,
            self.adaptive_weights.man_weight,
            self.adaptive_weights.mean_weight,
            self.goal_generator.last_goal.goal_type,
        )
    }

    /// Check if system is in a coherent compounding state
    pub fn is_compounding_coherent(&self) -> bool {
        let global = self.hierarchy.global_coherence();
        let cross_layer = self.cross_layer_smm.inter_layer_coherence();
        let pred_error = self.prediction.avg_error();

        global >= self.config.base_coherence.coherence_threshold
            && cross_layer >= 0.5
            && pred_error < 0.2
    }

    /// Suggest a direction based on graph memory across all streams
    /// Returns (dx, dy, similarity) if a suggestion can be made
    pub fn suggest_direction(&self, features: &[f32]) -> Option<(f64, f64, f64)> {
        // Query all stream graphs and aggregate suggestions
        let mut suggestions: Vec<(f64, f64, f64, f64)> = Vec::new(); // (dx, dy, similarity, weight)

        for graph in &self.stream_graphs {
            if let Some((dx, dy, similarity)) = graph.suggest_direction(features) {
                // Weight by number of edges (more experienced graphs get more weight)
                let weight = (graph.edges.len() as f64).sqrt().max(1.0);
                suggestions.push((dx, dy, similarity, weight));
            }
        }

        if suggestions.is_empty() {
            return None;
        }

        // Weighted average of suggestions
        let total_weight: f64 = suggestions.iter().map(|(_, _, _, w)| w).sum();
        let dx_avg: f64 =
            suggestions.iter().map(|(dx, _, _, w)| dx * w).sum::<f64>() / total_weight;
        let dy_avg: f64 =
            suggestions.iter().map(|(_, dy, _, w)| dy * w).sum::<f64>() / total_weight;
        let sim_avg: f64 = suggestions.iter().map(|(_, _, s, w)| s * w).sum::<f64>() / total_weight;

        Some((dx_avg, dy_avg, sim_avg))
    }

    /// Find nearest memory node and return similarity
    pub fn nearest_memory_similarity(&self, features: &[f32]) -> f64 {
        self.stream_graphs
            .iter()
            .filter_map(|g| g.find_nearest(features))
            .map(|(_, sim)| sim)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Update edge rewards in all stream graphs
    pub fn update_edge_rewards(&mut self, features: &[f32], reward: f64) {
        for graph in &mut self.stream_graphs {
            if let Some((nearest_id, similarity)) = graph.find_nearest(features) {
                if similarity > 0.5 {
                    // Find all edges leading to this node and update their rewards
                    let edges_to_update: Vec<(usize, usize)> = graph
                        .edges
                        .keys()
                        .filter(|(_, to)| *to == nearest_id)
                        .copied()
                        .collect();

                    for (from, to) in edges_to_update {
                        graph.update_edge_reward(from, to, reward);
                    }
                }
            }
        }
    }

    /// Get the last buffered features (for memory context)
    pub fn last_features(&self) -> Option<Vec<f32>> {
        self.stream_graphs
            .first()
            .and_then(|g| g.buffer.back().cloned())
    }

    /// Buffer position-augmented features for spatial memory
    /// Uses grid cell encoding for better position-based differentiation
    /// Now uses transition buffer for reward-aware sequence learning
    pub fn buffer_position_features(&mut self, position: (f64, f64), reward: f64) {
        if self.stream_graphs.is_empty() {
            return;
        }

        // Create position-based features with grid cell encoding
        let feature_dim = self.config.graph_feature_dim;
        let pos_features = Self::encode_position_with_grid_cells(position, reward, feature_dim);

        // Buffer to stream 0 using transition buffer (with reward for sequence learning)
        self.stream_graphs[0].buffer_transition(pos_features, reward);
    }

    /// Compute curiosity bonus for current position
    /// Returns a value in [0, 1] where higher = more novel/unexplored
    pub fn curiosity_bonus(&self, position: (f64, f64)) -> f64 {
        if self.stream_graphs.is_empty() {
            return 1.0;
        }

        // Encode position to features
        let feature_dim = self.config.graph_feature_dim;
        let pos_features = Self::encode_position_with_grid_cells(position, 0.0, feature_dim);

        // Get curiosity from primary spatial memory (stream 0)
        self.stream_graphs[0].curiosity_bonus(&pos_features)
    }

    /// Get curiosity bonus from neural state features
    pub fn curiosity_bonus_features(&self, features: &[f32]) -> f64 {
        if self.stream_graphs.is_empty() {
            return 1.0;
        }

        // Average curiosity across all streams
        let mut total_curiosity = 0.0;
        let mut count = 0;
        for graph in &self.stream_graphs {
            total_curiosity += graph.curiosity_bonus(features);
            count += 1;
        }

        if count > 0 {
            total_curiosity / count as f64
        } else {
            1.0
        }
    }

    /// Reset all stream graphs for new episode
    pub fn reset_episode_graphs(&mut self) {
        for graph in &mut self.stream_graphs {
            graph.reset_episode();
        }
    }

    /// Mark a position transition in eligibility traces
    /// Call this when agent moves from one position to another
    pub fn mark_transition(&mut self, from_pos: (f64, f64), to_pos: (f64, f64)) {
        if self.stream_graphs.is_empty() {
            return;
        }

        let feature_dim = self.config.graph_feature_dim;
        let from_features = Self::encode_position_with_grid_cells(from_pos, 0.0, feature_dim);
        let to_features = Self::encode_position_with_grid_cells(to_pos, 0.0, feature_dim);

        // Find nearest nodes for both positions
        let graph = &mut self.stream_graphs[0];
        if let (Some((from_id, _)), Some((to_id, _))) = (
            graph.find_nearest(&from_features),
            graph.find_nearest(&to_features),
        ) {
            // Mark this edge as recently traversed
            graph.mark_edge_traversed(from_id, to_id);
        }
    }

    /// Decay eligibility traces (call each step)
    pub fn decay_traces(&mut self) {
        for graph in &mut self.stream_graphs {
            graph.decay_traces();
        }
    }

    /// Update all edges using eligibility traces with received reward
    /// This propagates reward backward through recently visited edges
    pub fn update_traces_with_reward(&mut self, reward: f64) {
        let lr = self.config.graph_learning_rate;
        for graph in &mut self.stream_graphs {
            graph.update_traces_with_reward(reward, lr);
        }
    }

    /// Encode position using grid cell-like representation for better spatial memory
    /// This creates distinct features for different positions using multiple frequencies
    /// NOTE: Reward is NOT included in position encoding - it's used for edge weights only
    fn encode_position_with_grid_cells(
        position: (f64, f64),
        _reward: f64, // Kept for API compatibility but not used in features
        feature_dim: usize,
    ) -> Vec<f32> {
        let mut features = vec![0.0f32; feature_dim];
        let (x, y) = position;

        // Raw position (normalized to ~0-1 range for 10x10 grid)
        features[0] = (x / 10.0) as f32;
        features[1] = (y / 10.0) as f32;

        // Slot 2: Reserved for position magnitude (not reward)
        features[2] = ((x * x + y * y).sqrt() / 14.14) as f32; // Distance from origin normalized

        // Grid cell encoding: multiple frequencies for different spatial scales
        // This is inspired by neuroscience - grid cells fire at regular intervals
        let grid_scales = [0.5, 1.0, 2.0, 4.0, 8.0]; // Different "wavelengths"
        let mut idx = 3;

        for &scale in &grid_scales {
            if idx + 4 >= feature_dim {
                break;
            }
            // Sin/cos encoding at different scales
            features[idx] = (x * scale).sin() as f32;
            features[idx + 1] = (x * scale).cos() as f32;
            features[idx + 2] = (y * scale).sin() as f32;
            features[idx + 3] = (y * scale).cos() as f32;
            idx += 4;
        }

        // Discrete grid cell (which "cell" is the agent in)
        let grid_x = (x * 2.0).floor() as i32; // 5 cells across
        let grid_y = (y * 2.0).floor() as i32;
        if idx < feature_dim {
            features[idx] = (grid_x as f32) / 20.0; // Normalize
        }
        if idx + 1 < feature_dim {
            features[idx + 1] = (grid_y as f32) / 20.0;
        }
        idx += 2;

        // Quadrant encoding (one-hot style)
        let quadrant = match (x > 5.0, y > 5.0) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        };
        for q in 0..4 {
            if idx < feature_dim {
                features[idx] = if q == quadrant { 1.0 } else { 0.0 };
                idx += 1;
            }
        }

        // Distance from center and corners
        let center = (5.0, 5.0);
        let dist_center = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
        if idx < feature_dim {
            features[idx] = (dist_center / 7.07) as f32; // Normalize by max possible
            idx += 1;
        }

        // Angle from center
        let angle_center = (y - center.1).atan2(x - center.0);
        if idx < feature_dim {
            features[idx] = angle_center.sin() as f32;
        }
        if idx + 1 < feature_dim {
            features[idx + 1] = angle_center.cos() as f32;
        }

        features
    }

    /// Suggest direction from position-based memory using grid cell encoding
    /// Uses adaptive temperature for exploration vs exploitation balance
    pub fn suggest_direction_from_position(&self, position: (f64, f64)) -> Option<(f64, f64, f64)> {
        // Create features for this position using grid cell encoding
        let feature_dim = self.config.graph_feature_dim;
        let pos_features = Self::encode_position_with_grid_cells(position, 0.0, feature_dim);

        // Query stream 0 for direction suggestion with adaptive temperature
        self.stream_graphs
            .first()?
            .suggest_direction_with_temperature(&pos_features, self.adaptive_temperature)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // SUCCESSOR REPRESENTATION INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Update successor representations for a position transition
    /// Call this when agent moves from one position to another
    pub fn update_successor_representation(&mut self, from_pos: (f64, f64), to_pos: (f64, f64)) {
        if self.stream_graphs.is_empty() {
            return;
        }

        let feature_dim = self.config.graph_feature_dim;
        let from_features = Self::encode_position_with_grid_cells(from_pos, 0.0, feature_dim);
        let to_features = Self::encode_position_with_grid_cells(to_pos, 0.0, feature_dim);
        let lr = self.config.graph_learning_rate;

        // Update SR in spatial graph (stream 0)
        let graph = &mut self.stream_graphs[0];
        if let Some((from_id, _)) = graph.find_nearest(&from_features) {
            graph.update_successor_rep(from_id, &to_features, lr);
        }
    }

    /// Plan direction using successor representations for multi-step value
    /// Returns (dx, dy) direction if planning succeeds
    pub fn plan_with_successor(&self, position: (f64, f64)) -> Option<(f64, f64)> {
        if self.stream_graphs.is_empty() {
            return None;
        }

        let feature_dim = self.config.graph_feature_dim;
        let pos_features = Self::encode_position_with_grid_cells(position, 0.0, feature_dim);

        // Create reward features (higher values near goal regions)
        // For now, use a simple heuristic: reward features emphasize positions
        // with high historical reward
        let mut reward_features = vec![0.1f32; feature_dim];
        // Add some structure: reward features prefer center area
        reward_features[0] = 0.5; // x component
        reward_features[1] = 0.5; // y component

        self.stream_graphs[0].plan_with_sr(&pos_features, &reward_features)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // INTRINSIC MOTIVATION INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Update intrinsic motivation across all graphs
    pub fn update_intrinsic_motivation(&mut self, prediction_error: f64) {
        // Compute model change as the rate of new node addition
        let total_nodes: usize = self.stream_graphs.iter().map(|g| g.nodes.len()).sum();
        let model_change = (total_nodes as f64 / 100.0).min(1.0); // Normalize

        for graph in &mut self.stream_graphs {
            graph.update_intrinsic_motivation(prediction_error, model_change);
        }
    }

    /// Get total intrinsic motivation bonus across all graphs
    pub fn total_intrinsic_bonus(&self) -> f64 {
        if self.stream_graphs.is_empty() {
            return 0.0;
        }

        // Average intrinsic bonus across all graphs
        let total: f64 = self.stream_graphs.iter().map(|g| g.intrinsic_bonus()).sum();
        total / self.stream_graphs.len() as f64
    }

    /// Get detailed intrinsic motivation from spatial graph
    pub fn spatial_intrinsic_motivation(&self) -> Option<&IntrinsicMotivation> {
        self.stream_graphs.first().map(|g| g.intrinsic_motivation())
    }
}

/// Result from a compounding forward step
#[derive(Debug, Clone)]
pub struct CompoundingResult {
    /// Adaptive alpha for this layer
    pub adaptive_alpha: f64,
    /// Current SOC score
    pub soc_score: f64,
    /// Predicted SOC score
    pub predicted_soc: f64,
    /// Prediction error
    pub prediction_error: f64,
}

/// Result from post-episode consolidation
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Nodes added to graphs
    pub nodes_added: usize,
    /// Current episode number
    pub episode: usize,
    /// Stats for each stream graph
    pub graph_stats: Vec<GraphStats>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_similarity() {
        let node = GraphNode::new(0, vec![1.0, 0.0, 0.0]);
        assert!((node.similarity(&[1.0, 0.0, 0.0]) - 1.0).abs() < 0.01);
        assert!((node.similarity(&[0.0, 1.0, 0.0])).abs() < 0.01);
    }

    #[test]
    fn test_stream_graph_memory() {
        let mut graph = StreamGraphMemory::new(0, 100, 64, 0.9);
        let features = vec![0.5f32; 64];
        graph.buffer_features(features.clone());
        assert_eq!(graph.buffer.len(), 1);

        let id = graph.add_or_update(features.clone(), 0.1);
        assert_eq!(id, 0);
        assert_eq!(graph.nodes.len(), 1);

        // Similar features should update, not add
        let id2 = graph.add_or_update(features.clone(), 0.1);
        assert_eq!(id2, 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_hierarchical_soc() {
        let mut hsoc = HierarchicalSOC::new(HierarchyLevel::Regional, 10);
        let children = vec![
            SenseOfCoherence::new(0.8, 0.7, 0.9),
            SenseOfCoherence::new(0.6, 0.8, 0.7),
        ];
        hsoc.update_from_children(&children);
        assert!((hsoc.soc.comprehensibility - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_cross_layer_smm() {
        let smm = CrossLayerSMM::new(4, 0.1);
        assert_eq!(smm.n_layers, 4);
        assert!((smm.layer_alignment(0, 1) - 0.8).abs() < 0.01); // Adjacent layers
        assert!((smm.layer_alignment(0, 0) - 1.0).abs() < 0.01); // Self
    }

    #[test]
    fn test_predictive_coherence() {
        let mut pred = PredictiveCoherence::new(10);
        for i in 0..5 {
            pred.record(SenseOfCoherence::new(0.5 + i as f64 * 0.1, 0.5, 0.5));
        }
        assert!(pred.soc_history.len() == 5);
        assert!(pred.trend() > 0.0); // Should be improving
    }

    #[test]
    fn test_adaptive_weights() {
        let mut weights = AdaptiveCoherenceWeights::new(0.01);
        let soc = SenseOfCoherence::new(0.8, 0.6, 0.7);

        for i in 0..20 {
            weights.update(&soc, 0.5 + i as f64 * 0.01); // Improving performance
        }

        // Weights should have shifted
        let sum = weights.comp_weight + weights.man_weight + weights.mean_weight;
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_goal_state_generator() {
        let device = Device::Cpu;
        let config = CompoundingCohesionConfig::default();
        let hierarchy = HierarchicalCoherence::new(config, &device);

        let mut generator = GoalStateGenerator::new(0.6);
        let goal = generator.generate(&hierarchy);

        // With healthy initial coherence, should be None or Refine
        assert!(matches!(goal.goal_type, GoalType::None | GoalType::Refine));
    }

    #[test]
    fn test_compounding_system_creation() {
        let device = Device::Cpu;
        let config = CompoundingCohesionConfig::default();
        let system = CompoundingCohesionSystem::new(config.clone(), &device);

        assert_eq!(system.stream_graphs.len(), config.n_streams);
        assert_eq!(system.hierarchy.local.len(), config.n_layers);
    }
}
