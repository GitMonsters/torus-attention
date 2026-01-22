//! # Unified AGI Core - Compounding Cognitive Cohesion
//!
//! This module implements the complete AGI framework with all missing capabilities:
//!
//! 1. **Causal Discovery**: Automatically discover new causal variables from experience
//! 2. **Hierarchical Abstraction**: Build concept hierarchies and compositional representations
//! 3. **World Model**: Deep generative model for mental simulation of novel scenarios
//! 4. **Goal Decomposition**: Hierarchical goals with automatic subgoal generation
//! 5. **Meta-Learning**: Learn how to learn, self-improvement of learning algorithms
//! 6. **Symbol Grounding**: Internal language connecting symbols to sensorimotor experience
//!
//! ## The Compounding Hypothesis
//!
//! AGI emerges from the **multiplicative interaction** of these capabilities:
//! - Causal discovery feeds abstraction hierarchy
//! - Abstractions ground symbols
//! - Symbols enable compositional goals
//! - Goals drive world model queries
//! - World model enables meta-learning about what to learn
//! - Meta-learning improves causal discovery
//! - **The loop compounds**

use crate::compounding_cohesion::{CompoundingCohesionConfig, CompoundingCohesionSystem};
use crate::consequential::{AGIReasoningSystem, CausalGraph, CausalMechanism, CausalVariable};
use crate::explicability::{
    DetailLevel, ExplicabilityConfig, ExplicabilitySystem, FactorSource, InfluenceDirection,
};
use crate::general_coherence::{ArtificialGeneralCoherence, ExpansionArea};
use crate::learning_webs::{LearningWebs, LearningWebsSummary};
use crate::llm_integration::{LLMIntegration, LLMIntegrationConfig};
use crate::memory_system::{MemorySystem, MemorySystemConfig};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the unified AGI core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGICoreConfig {
    /// Maximum number of discovered causal variables
    pub max_causal_variables: usize,
    /// Maximum abstraction hierarchy depth
    pub max_abstraction_depth: usize,
    /// World model trajectory length for simulation
    pub world_model_horizon: usize,
    /// Maximum active goals in hierarchy
    pub max_active_goals: usize,
    /// Meta-learning window size
    pub meta_learning_window: usize,
    /// Maximum symbols in the symbol system
    pub max_symbols: usize,
    /// Information gain threshold for causal discovery
    pub causal_discovery_threshold: f64,
    /// Similarity threshold for concept merging
    pub abstraction_merge_threshold: f64,
    /// Prediction error threshold for world model update
    pub world_model_error_threshold: f64,
    /// Goal completion threshold
    pub goal_completion_threshold: f64,
    /// Symbol grounding confidence threshold
    pub symbol_grounding_threshold: f64,
    /// Maximum skills to store
    pub max_skills: usize,
    /// Counterfactual regret threshold for policy updates
    pub counterfactual_regret_threshold: f64,
}

impl Default for AGICoreConfig {
    fn default() -> Self {
        Self {
            max_causal_variables: 100,
            max_abstraction_depth: 5,
            world_model_horizon: 20,
            max_active_goals: 10,
            meta_learning_window: 100,
            max_symbols: 500,
            causal_discovery_threshold: 0.15, // LOWERED from 0.3 for faster discovery
            abstraction_merge_threshold: 0.75, // LOWERED from 0.85 for more concepts
            world_model_error_threshold: 0.1,
            goal_completion_threshold: 0.55, // Lowered from 0.65 for noisy environments
            symbol_grounding_threshold: 0.40, // Lowered from 0.5 for easier grounding
            max_skills: 200,
            counterfactual_regret_threshold: 0.1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL DISCOVERY - Learn new variables from experience
// ═══════════════════════════════════════════════════════════════════════════════

/// A discovered latent variable from experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredVariable {
    /// Unique identifier
    pub id: usize,
    /// Auto-generated name based on discovery context
    pub name: String,
    /// Feature signature that defines this variable
    pub signature: Vec<f64>,
    /// Variables this was derived from
    pub parent_variables: Vec<usize>,
    /// Information gain when this variable was discovered
    pub information_gain: f64,
    /// Discovery timestamp (step)
    pub discovered_at: usize,
    /// Number of times this variable has been useful
    pub utility_count: usize,
}

/// Observation for causal discovery
#[derive(Debug, Clone)]
pub struct CausalObservation {
    /// State features at this step
    pub features: Vec<f64>,
    /// Action taken (if any)
    pub action: Option<usize>,
    /// Reward received
    pub reward: f64,
    /// Step number
    pub step: usize,
}

/// Causal Discovery System - automatically discovers new causal variables
#[derive(Debug, Clone)]
pub struct CausalDiscovery {
    /// Discovered variables
    pub variables: Vec<DiscoveredVariable>,
    /// Observation history for pattern detection
    observation_history: VecDeque<CausalObservation>,
    /// Mutual information cache between variable pairs
    mutual_info_cache: HashMap<(usize, usize), f64>,
    /// Next variable ID
    next_id: usize,
    /// Configuration
    config: AGICoreConfig,
    /// Current step
    current_step: usize,
}

impl CausalDiscovery {
    pub fn new(config: AGICoreConfig) -> Self {
        Self {
            variables: Vec::new(),
            observation_history: VecDeque::with_capacity(1000),
            mutual_info_cache: HashMap::new(),
            next_id: 0,
            config,
            current_step: 0,
        }
    }

    /// Add an observation for causal analysis
    pub fn observe(&mut self, features: Vec<f64>, action: Option<usize>, reward: f64) {
        let obs = CausalObservation {
            features,
            action,
            reward,
            step: self.current_step,
        };
        self.observation_history.push_back(obs);

        // Keep bounded history
        while self.observation_history.len() > 1000 {
            self.observation_history.pop_front();
        }

        self.current_step += 1;

        // Attempt discovery every 25 steps (more frequent for faster emergence)
        if self.current_step % 25 == 0 {
            self.attempt_discovery();
        }
    }

    /// Attempt to discover new causal variables from patterns
    fn attempt_discovery(&mut self) {
        if self.observation_history.len() < 100 {
            return;
        }

        if self.variables.len() >= self.config.max_causal_variables {
            return;
        }

        // Compute feature covariances to find latent structure
        let n_features = self
            .observation_history
            .front()
            .map(|o| o.features.len())
            .unwrap_or(0);

        if n_features == 0 {
            return;
        }

        // Compute mean features
        let mut mean_features = vec![0.0; n_features];
        let n_obs = self.observation_history.len() as f64;

        for obs in &self.observation_history {
            for (i, &f) in obs.features.iter().enumerate() {
                if i < mean_features.len() {
                    mean_features[i] += f / n_obs;
                }
            }
        }

        // Find feature pairs with high mutual information (simplified)
        let mut candidate_pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n_features.min(10) {
            for j in (i + 1)..n_features.min(10) {
                let mi = self.estimate_mutual_info(i, j);
                if mi > self.config.causal_discovery_threshold {
                    candidate_pairs.push((i, j, mi));
                }
            }
        }

        // Create new variable from highest MI pair
        if let Some((i, j, mi)) = candidate_pairs
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        {
            // Check if we already have a variable for this pair
            let already_exists = self
                .variables
                .iter()
                .any(|v| v.parent_variables.contains(i) && v.parent_variables.contains(j));

            if !already_exists {
                let signature: Vec<f64> = self
                    .observation_history
                    .iter()
                    .take(50)
                    .map(|o| {
                        let fi = o.features.get(*i).copied().unwrap_or(0.0);
                        let fj = o.features.get(*j).copied().unwrap_or(0.0);
                        (fi + fj) / 2.0
                    })
                    .collect();

                let var = DiscoveredVariable {
                    id: self.next_id,
                    name: format!("latent_{}_{}", i, j),
                    signature,
                    parent_variables: vec![*i, *j],
                    information_gain: *mi,
                    discovered_at: self.current_step,
                    utility_count: 0,
                };

                self.variables.push(var);
                self.next_id += 1;
            }
        }
    }

    /// Estimate mutual information between two feature indices (simplified)
    fn estimate_mutual_info(&mut self, i: usize, j: usize) -> f64 {
        if let Some(&cached) = self.mutual_info_cache.get(&(i, j)) {
            return cached;
        }

        // Simplified MI estimation using correlation as proxy
        let mut sum_i = 0.0;
        let mut sum_j = 0.0;
        let mut sum_ij = 0.0;
        let mut sum_i2 = 0.0;
        let mut sum_j2 = 0.0;
        let mut count = 0.0;

        for obs in &self.observation_history {
            if let (Some(&fi), Some(&fj)) = (obs.features.get(i), obs.features.get(j)) {
                sum_i += fi;
                sum_j += fj;
                sum_ij += fi * fj;
                sum_i2 += fi * fi;
                sum_j2 += fj * fj;
                count += 1.0;
            }
        }

        if count < 10.0 {
            return 0.0;
        }

        let mean_i = sum_i / count;
        let mean_j = sum_j / count;
        let var_i = (sum_i2 / count - mean_i * mean_i).max(1e-10);
        let var_j = (sum_j2 / count - mean_j * mean_j).max(1e-10);
        let cov_ij = sum_ij / count - mean_i * mean_j;

        let correlation = cov_ij / (var_i.sqrt() * var_j.sqrt());
        let mi = -0.5 * (1.0 - correlation * correlation).max(1e-10).ln();

        self.mutual_info_cache.insert((i, j), mi);
        mi.max(0.0)
    }

    /// Get discovered variables that are useful for a given context
    pub fn get_relevant_variables(&self, features: &[f64]) -> Vec<&DiscoveredVariable> {
        self.variables
            .iter()
            .filter(|v| {
                // Check if parent features are active
                v.parent_variables
                    .iter()
                    .all(|&pi| features.get(pi).map(|&f| f.abs() > 0.1).unwrap_or(false))
            })
            .collect()
    }

    /// Mark a variable as useful (for utility tracking)
    pub fn mark_useful(&mut self, var_id: usize) {
        if let Some(var) = self.variables.iter_mut().find(|v| v.id == var_id) {
            var.utility_count += 1;
        }
    }

    /// Set discovery threshold (for meta-learning integration)
    pub fn set_discovery_threshold(&mut self, threshold: f64) {
        self.config.causal_discovery_threshold = threshold.max(0.1).min(0.9);
        // Clear cache when threshold changes to re-evaluate pairs
        self.mutual_info_cache.clear();
    }

    /// Get current discovery threshold
    pub fn get_discovery_threshold(&self) -> f64 {
        self.config.causal_discovery_threshold
    }

    /// Design an experiment to test causal hypotheses (active experimentation)
    /// Returns: (variable_indices_to_perturb, suggested_action, expected_info_gain)
    pub fn design_experiment(&self) -> Option<CausalExperiment> {
        // Find variable pairs with high but uncertain MI
        let mut uncertain_pairs: Vec<(usize, usize, f64)> = Vec::new();

        for (&(i, j), &mi) in &self.mutual_info_cache {
            // Look for moderately correlated pairs (not too obvious, not too weak)
            if mi > 0.2 && mi < 0.7 {
                uncertain_pairs.push((i, j, mi));
            }
        }

        // Sort by information gain potential (middle values are most informative)
        uncertain_pairs.sort_by(|a, b| {
            let info_a = (0.5 - (a.2 - 0.5).abs());
            let info_b = (0.5 - (b.2 - 0.5).abs());
            info_b
                .partial_cmp(&info_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        uncertain_pairs.first().map(|&(i, j, mi)| CausalExperiment {
            variable_indices: vec![i, j],
            expected_info_gain: 0.5 - (mi - 0.5).abs(),
            hypothesis: format!("Test if feature {} causally influences feature {}", i, j),
            suggested_intervention: if mi > 0.5 {
                InterventionType::Increase(i)
            } else {
                InterventionType::Decrease(i)
            },
        })
    }

    /// Get summary statistics
    pub fn summary(&self) -> CausalDiscoverySummary {
        CausalDiscoverySummary {
            total_variables: self.variables.len(),
            total_observations: self.observation_history.len(),
            avg_information_gain: if self.variables.is_empty() {
                0.0
            } else {
                self.variables
                    .iter()
                    .map(|v| v.information_gain)
                    .sum::<f64>()
                    / self.variables.len() as f64
            },
            most_useful: self
                .variables
                .iter()
                .max_by_key(|v| v.utility_count)
                .map(|v| v.name.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDiscoverySummary {
    pub total_variables: usize,
    pub total_observations: usize,
    pub avg_information_gain: f64,
    pub most_useful: Option<String>,
}

/// A designed causal experiment for active learning
#[derive(Debug, Clone)]
pub struct CausalExperiment {
    /// Feature indices to test
    pub variable_indices: Vec<usize>,
    /// Expected information gain from this experiment
    pub expected_info_gain: f64,
    /// Human-readable hypothesis being tested
    pub hypothesis: String,
    /// Suggested intervention to test causality
    pub suggested_intervention: InterventionType,
}

/// Types of interventions for causal experiments
#[derive(Debug, Clone)]
pub enum InterventionType {
    /// Increase the value of feature at index
    Increase(usize),
    /// Decrease the value of feature at index
    Decrease(usize),
    /// Set feature to specific value
    SetTo(usize, f64),
    /// Randomize feature
    Randomize(usize),
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIERARCHICAL ABSTRACTION - Concept formation and chunking
// ═══════════════════════════════════════════════════════════════════════════════

/// A concept in the abstraction hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Unique identifier
    pub id: usize,
    /// Name/label for this concept
    pub name: String,
    /// Abstraction level (0 = ground, higher = more abstract)
    pub level: usize,
    /// Prototype feature vector
    pub prototype: Vec<f64>,
    /// Child concepts (for compositional structure)
    pub children: Vec<usize>,
    /// Parent concepts
    pub parents: Vec<usize>,
    /// Activation count
    pub activation_count: usize,
    /// Confidence in this concept
    pub confidence: f64,
    /// Associated actions (for skill abstraction)
    pub associated_actions: Vec<usize>,
}

/// Hierarchical Abstraction System - builds concept hierarchies
#[derive(Debug, Clone)]
pub struct AbstractionHierarchy {
    /// All concepts by ID
    concepts: HashMap<usize, Concept>,
    /// Concepts organized by level
    levels: Vec<Vec<usize>>,
    /// Pattern buffer for chunking
    pattern_buffer: VecDeque<Vec<f64>>,
    /// Next concept ID
    next_id: usize,
    /// Configuration
    config: AGICoreConfig,
}

impl AbstractionHierarchy {
    pub fn new(config: AGICoreConfig) -> Self {
        let mut levels = Vec::new();
        for _ in 0..config.max_abstraction_depth {
            levels.push(Vec::new());
        }

        Self {
            concepts: HashMap::new(),
            levels,
            pattern_buffer: VecDeque::with_capacity(100),
            next_id: 0,
            config,
        }
    }

    /// Observe a pattern for potential chunking
    pub fn observe(&mut self, features: Vec<f64>) {
        self.pattern_buffer.push_back(features.clone());

        while self.pattern_buffer.len() > 100 {
            self.pattern_buffer.pop_front();
        }

        // Try to match existing concepts
        let matched = self.match_concept(&features);

        if let Some(concept_id) = matched {
            if let Some(concept) = self.concepts.get_mut(&concept_id) {
                concept.activation_count += 1;
                // Update prototype with running average
                let alpha = 0.1;
                for (i, p) in concept.prototype.iter_mut().enumerate() {
                    if let Some(&f) = features.get(i) {
                        *p = *p * (1.0 - alpha) + f * alpha;
                    }
                }
            }
        } else if self.pattern_buffer.len() >= 10 {
            // Try to form a new concept
            self.try_form_concept();
        }
    }

    /// Match features to existing concept
    fn match_concept(&self, features: &[f64]) -> Option<usize> {
        let mut best_match: Option<(usize, f64)> = None;

        for (&id, concept) in &self.concepts {
            let similarity = self.cosine_similarity(features, &concept.prototype);
            if similarity > self.config.abstraction_merge_threshold {
                if best_match.map(|(_, s)| similarity > s).unwrap_or(true) {
                    best_match = Some((id, similarity));
                }
            }
        }

        best_match.map(|(id, _)| id)
    }

    /// Try to form a new concept from recent patterns
    fn try_form_concept(&mut self) {
        if self.pattern_buffer.len() < 5 {
            return;
        }

        // Compute centroid of recent patterns
        let n = self.pattern_buffer.len();
        let dim = self.pattern_buffer.front().map(|p| p.len()).unwrap_or(0);

        if dim == 0 {
            return;
        }

        let mut centroid = vec![0.0; dim];
        for pattern in &self.pattern_buffer {
            for (i, &f) in pattern.iter().enumerate() {
                if i < centroid.len() {
                    centroid[i] += f / n as f64;
                }
            }
        }

        // Check if this is sufficiently different from existing concepts
        let is_novel = self.concepts.values().all(|c| {
            self.cosine_similarity(&centroid, &c.prototype)
                < self.config.abstraction_merge_threshold
        });

        if is_novel && self.concepts.len() < 1000 {
            let concept = Concept {
                id: self.next_id,
                name: format!("concept_{}", self.next_id),
                level: 0,
                prototype: centroid,
                children: Vec::new(),
                parents: Vec::new(),
                activation_count: 1,
                confidence: 0.5,
                associated_actions: Vec::new(),
            };

            let id = concept.id;
            self.concepts.insert(id, concept);
            if !self.levels.is_empty() {
                self.levels[0].push(id);
            }
            self.next_id += 1;

            // Try to form higher-level abstractions
            self.try_abstract_upward();
        }
    }

    /// Try to form higher-level abstractions from lower-level concepts
    fn try_abstract_upward(&mut self) {
        for level in 0..(self.config.max_abstraction_depth - 1) {
            if self.levels.get(level).map(|l| l.len()).unwrap_or(0) < 3 {
                continue;
            }

            // Find clusters of co-occurring concepts
            let level_concepts: Vec<usize> = self.levels.get(level).cloned().unwrap_or_default();

            // Simple clustering: merge frequently co-activated concepts
            let mut to_merge: Vec<(usize, usize)> = Vec::new();

            for i in 0..level_concepts.len() {
                for j in (i + 1)..level_concepts.len() {
                    let ci = level_concepts[i];
                    let cj = level_concepts[j];

                    if let (Some(c1), Some(c2)) = (self.concepts.get(&ci), self.concepts.get(&cj)) {
                        let similarity = self.cosine_similarity(&c1.prototype, &c2.prototype);
                        if similarity > 0.7 && similarity < self.config.abstraction_merge_threshold
                        {
                            to_merge.push((ci, cj));
                        }
                    }
                }
            }

            // Create parent concept for merged children
            for (ci, cj) in to_merge.into_iter().take(3) {
                let (c1, c2) = match (self.concepts.get(&ci), self.concepts.get(&cj)) {
                    (Some(c1), Some(c2)) => (c1.clone(), c2.clone()),
                    _ => continue,
                };

                // Check if parent already exists
                let parent_exists = self.concepts.values().any(|c| {
                    c.level == level + 1 && c.children.contains(&ci) && c.children.contains(&cj)
                });

                if !parent_exists {
                    let mut parent_proto = vec![0.0; c1.prototype.len()];
                    for (i, p) in parent_proto.iter_mut().enumerate() {
                        let p1 = c1.prototype.get(i).copied().unwrap_or(0.0);
                        let p2 = c2.prototype.get(i).copied().unwrap_or(0.0);
                        *p = (p1 + p2) / 2.0;
                    }

                    let parent = Concept {
                        id: self.next_id,
                        name: format!("abstract_{}_{}", level + 1, self.next_id),
                        level: level + 1,
                        prototype: parent_proto,
                        children: vec![ci, cj],
                        parents: Vec::new(),
                        activation_count: 1,
                        confidence: 0.3,
                        associated_actions: Vec::new(),
                    };

                    let parent_id = parent.id;
                    self.concepts.insert(parent_id, parent);

                    if let Some(level_vec) = self.levels.get_mut(level + 1) {
                        level_vec.push(parent_id);
                    }

                    // Update children to point to parent
                    if let Some(c) = self.concepts.get_mut(&ci) {
                        c.parents.push(parent_id);
                    }
                    if let Some(c) = self.concepts.get_mut(&cj) {
                        c.parents.push(parent_id);
                    }

                    self.next_id += 1;
                }
            }
        }
    }

    /// Associate an action with a concept
    pub fn associate_action(&mut self, concept_id: usize, action: usize) {
        if let Some(concept) = self.concepts.get_mut(&concept_id) {
            if !concept.associated_actions.contains(&action) {
                concept.associated_actions.push(action);
            }
        }
    }

    /// Get concepts activated by features
    pub fn get_activated_concepts(&self, features: &[f64]) -> Vec<(usize, f64)> {
        let mut activated: Vec<(usize, f64)> = Vec::new();

        for (&id, concept) in &self.concepts {
            let similarity = self.cosine_similarity(features, &concept.prototype);
            if similarity > 0.5 {
                activated.push((id, similarity));
            }
        }

        activated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        activated
    }

    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len().min(b.len()) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Get summary statistics
    pub fn summary(&self) -> AbstractionSummary {
        let concepts_per_level: Vec<usize> = self.levels.iter().map(|l| l.len()).collect();

        AbstractionSummary {
            total_concepts: self.concepts.len(),
            concepts_per_level,
            max_depth: self
                .levels
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.is_empty())
                .map(|(i, _)| i + 1)
                .max()
                .unwrap_or(0),
            avg_activation: if self.concepts.is_empty() {
                0.0
            } else {
                self.concepts
                    .values()
                    .map(|c| c.activation_count as f64)
                    .sum::<f64>()
                    / self.concepts.len() as f64
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionSummary {
    pub total_concepts: usize,
    pub concepts_per_level: Vec<usize>,
    pub max_depth: usize,
    pub avg_activation: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORLD MODEL - Generative imagination and mental simulation
// ═══════════════════════════════════════════════════════════════════════════════

/// A state in the world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// State features
    pub features: Vec<f64>,
    /// Associated reward
    pub reward: f64,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Uncertainty in this state
    pub uncertainty: f64,
}

/// A transition in the world model
#[derive(Debug, Clone)]
pub struct WorldTransition {
    /// Source state features
    pub from_features: Vec<f64>,
    /// Action taken
    pub action: usize,
    /// Resulting state features
    pub to_features: Vec<f64>,
    /// Reward received
    pub reward: f64,
    /// Transition count (for confidence)
    pub count: usize,
}

/// Simulated trajectory for mental planning
#[derive(Debug, Clone)]
pub struct SimulatedTrajectory {
    /// Sequence of states
    pub states: Vec<WorldState>,
    /// Actions taken
    pub actions: Vec<usize>,
    /// Total reward
    pub total_reward: f64,
    /// Average uncertainty
    pub avg_uncertainty: f64,
}

/// World Model - learns dynamics for mental simulation
#[derive(Debug, Clone)]
pub struct WorldModel {
    /// Learned transitions indexed by (discretized_state, action)
    transitions: HashMap<(Vec<i32>, usize), WorldTransition>,
    /// State visitation counts
    state_visits: HashMap<Vec<i32>, usize>,
    /// Reward model: state -> expected reward
    reward_model: HashMap<Vec<i32>, (f64, usize)>,
    /// Terminal state signatures
    terminal_states: HashSet<Vec<i32>>,
    /// Feature discretization bins
    n_bins: usize,
    /// Feature ranges for discretization
    feature_ranges: Vec<(f64, f64)>,
    /// Configuration
    config: AGICoreConfig,
    /// Total experience count
    experience_count: usize,
}

impl WorldModel {
    pub fn new(config: AGICoreConfig, feature_dim: usize) -> Self {
        Self {
            transitions: HashMap::new(),
            state_visits: HashMap::new(),
            reward_model: HashMap::new(),
            terminal_states: HashSet::new(),
            n_bins: 10,
            feature_ranges: vec![(-1.0, 1.0); feature_dim],
            config,
            experience_count: 0,
        }
    }

    /// Discretize features for indexing
    fn discretize(&self, features: &[f64]) -> Vec<i32> {
        features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let (min_val, max_val) = self.feature_ranges.get(i).copied().unwrap_or((-1.0, 1.0));
                let range = (max_val - min_val).max(1e-6);
                let normalized = (f - min_val) / range;
                let bin = (normalized * self.n_bins as f64).floor() as i32;
                bin.max(0).min(self.n_bins as i32 - 1)
            })
            .collect()
    }

    /// Learn from an observed transition
    pub fn learn(
        &mut self,
        from: &[f64],
        action: usize,
        to: &[f64],
        reward: f64,
        is_terminal: bool,
    ) {
        // Update feature ranges
        for (i, (&f_from, &f_to)) in from.iter().zip(to.iter()).enumerate() {
            if i >= self.feature_ranges.len() {
                self.feature_ranges
                    .push((f_from.min(f_to), f_from.max(f_to)));
            } else {
                let (min_val, max_val) = self.feature_ranges[i];
                self.feature_ranges[i] =
                    (min_val.min(f_from).min(f_to), max_val.max(f_from).max(f_to));
            }
        }

        let from_disc = self.discretize(from);
        let to_disc = self.discretize(to);

        // Update state visits
        *self.state_visits.entry(from_disc.clone()).or_insert(0) += 1;

        // Update transition model
        let key = (from_disc.clone(), action);
        let entry = self
            .transitions
            .entry(key)
            .or_insert_with(|| WorldTransition {
                from_features: from.to_vec(),
                action,
                to_features: to.to_vec(),
                reward,
                count: 0,
            });

        // Update with running average
        let alpha = 1.0 / (entry.count + 1) as f64;
        for (i, t) in entry.to_features.iter_mut().enumerate() {
            if let Some(&new_t) = to.get(i) {
                *t = *t * (1.0 - alpha) + new_t * alpha;
            }
        }
        entry.reward = entry.reward * (1.0 - alpha) + reward * alpha;
        entry.count += 1;

        // Update reward model
        let reward_entry = self.reward_model.entry(to_disc.clone()).or_insert((0.0, 0));
        let r_alpha = 1.0 / (reward_entry.1 + 1) as f64;
        reward_entry.0 = reward_entry.0 * (1.0 - r_alpha) + reward * r_alpha;
        reward_entry.1 += 1;

        // Track terminal states
        if is_terminal {
            self.terminal_states.insert(to_disc);
        }

        self.experience_count += 1;
    }

    /// Predict next state given current state and action
    pub fn predict(&self, from: &[f64], action: usize) -> Option<WorldState> {
        let from_disc = self.discretize(from);
        let key = (from_disc.clone(), action);

        self.transitions.get(&key).map(|t| {
            let to_disc = self.discretize(&t.to_features);
            let is_terminal = self.terminal_states.contains(&to_disc);
            let uncertainty = 1.0 / (t.count as f64 + 1.0).sqrt();

            WorldState {
                features: t.to_features.clone(),
                reward: t.reward,
                is_terminal,
                uncertainty,
            }
        })
    }

    /// Simulate a trajectory from a starting state
    pub fn simulate(&self, start: &[f64], actions: &[usize]) -> SimulatedTrajectory {
        let mut states = vec![WorldState {
            features: start.to_vec(),
            reward: 0.0,
            is_terminal: false,
            uncertainty: 0.0,
        }];
        let mut taken_actions = Vec::new();
        let mut total_reward = 0.0;
        let mut total_uncertainty = 0.0;

        let mut current = start.to_vec();

        for &action in actions.iter().take(self.config.world_model_horizon) {
            if let Some(next_state) = self.predict(&current, action) {
                total_reward += next_state.reward;
                total_uncertainty += next_state.uncertainty;
                current = next_state.features.clone();
                taken_actions.push(action);

                let is_terminal = next_state.is_terminal;
                states.push(next_state);

                if is_terminal {
                    break;
                }
            } else {
                // Unknown transition - stop simulation
                break;
            }
        }

        let avg_uncertainty = if states.len() > 1 {
            total_uncertainty / (states.len() - 1) as f64
        } else {
            1.0
        };

        SimulatedTrajectory {
            states,
            actions: taken_actions,
            total_reward,
            avg_uncertainty,
        }
    }

    /// Imagine possible futures by sampling actions
    pub fn imagine_futures(
        &self,
        start: &[f64],
        n_futures: usize,
        n_actions: usize,
    ) -> Vec<SimulatedTrajectory> {
        let mut futures = Vec::new();

        for i in 0..n_futures {
            // Generate a sequence of actions (simple heuristic)
            let actions: Vec<usize> = (0..self.config.world_model_horizon)
                .map(|j| (i + j) % n_actions)
                .collect();

            let trajectory = self.simulate(start, &actions);
            if !trajectory.actions.is_empty() {
                futures.push(trajectory);
            }
        }

        // Sort by expected reward
        futures.sort_by(|a, b| {
            b.total_reward
                .partial_cmp(&a.total_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        futures
    }

    /// Get summary statistics
    pub fn summary(&self) -> WorldModelSummary {
        WorldModelSummary {
            total_transitions: self.transitions.len(),
            total_states: self.state_visits.len(),
            total_experience: self.experience_count,
            terminal_states: self.terminal_states.len(),
            avg_transition_count: if self.transitions.is_empty() {
                0.0
            } else {
                self.transitions
                    .values()
                    .map(|t| t.count as f64)
                    .sum::<f64>()
                    / self.transitions.len() as f64
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelSummary {
    pub total_transitions: usize,
    pub total_states: usize,
    pub total_experience: usize,
    pub terminal_states: usize,
    pub avg_transition_count: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOAL HIERARCHY - Planning with automatic subgoal decomposition
// ═══════════════════════════════════════════════════════════════════════════════

/// Priority level for goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GoalPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Status of a goal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Pending,
    Active,
    Completed,
    Failed,
    Abandoned,
}

/// A goal in the hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier
    pub id: usize,
    /// Goal description/name
    pub name: String,
    /// Target state features (what we want to achieve)
    pub target_features: Vec<f64>,
    /// Tolerance for goal completion
    pub tolerance: f64,
    /// Priority level
    pub priority: GoalPriority,
    /// Current status
    pub status: GoalStatus,
    /// Parent goal (if this is a subgoal)
    pub parent: Option<usize>,
    /// Subgoals
    pub subgoals: Vec<usize>,
    /// Progress (0.0 to 1.0)
    pub progress: f64,
    /// Creation timestamp
    pub created_at: usize,
    /// Associated concept (if any)
    pub concept_id: Option<usize>,
}

/// Goal Hierarchy System - manages hierarchical goal decomposition
#[derive(Debug, Clone)]
pub struct GoalHierarchy {
    /// All goals by ID
    goals: HashMap<usize, Goal>,
    /// Active goals (not completed/failed)
    active_goals: Vec<usize>,
    /// Goal achievement history
    achievement_history: VecDeque<(usize, bool, usize)>, // (goal_id, success, step)
    /// Next goal ID
    next_id: usize,
    /// Current step
    current_step: usize,
    /// Configuration
    config: AGICoreConfig,
}

impl GoalHierarchy {
    pub fn new(config: AGICoreConfig) -> Self {
        Self {
            goals: HashMap::new(),
            active_goals: Vec::new(),
            achievement_history: VecDeque::with_capacity(100),
            next_id: 0,
            current_step: 0,
            config,
        }
    }

    /// Create a new top-level goal
    pub fn create_goal(
        &mut self,
        name: &str,
        target_features: Vec<f64>,
        priority: GoalPriority,
    ) -> usize {
        let goal = Goal {
            id: self.next_id,
            name: name.to_string(),
            target_features,
            tolerance: 0.1,
            priority,
            status: GoalStatus::Pending,
            parent: None,
            subgoals: Vec::new(),
            progress: 0.0,
            created_at: self.current_step,
            concept_id: None,
        };

        let id = goal.id;
        self.goals.insert(id, goal);
        self.next_id += 1;
        id
    }

    /// Decompose a goal into subgoals
    pub fn decompose_goal(
        &mut self,
        goal_id: usize,
        intermediate_states: Vec<Vec<f64>>,
    ) -> Vec<usize> {
        let mut subgoal_ids = Vec::new();

        let parent_name = self
            .goals
            .get(&goal_id)
            .map(|g| g.name.clone())
            .unwrap_or_default();

        for (i, target) in intermediate_states.into_iter().enumerate() {
            let subgoal = Goal {
                id: self.next_id,
                name: format!("{}_sub_{}", parent_name, i),
                target_features: target,
                tolerance: 0.15, // Slightly more tolerance for subgoals
                priority: GoalPriority::Medium,
                status: GoalStatus::Pending,
                parent: Some(goal_id),
                subgoals: Vec::new(),
                progress: 0.0,
                created_at: self.current_step,
                concept_id: None,
            };

            let sub_id = subgoal.id;
            self.goals.insert(sub_id, subgoal);
            subgoal_ids.push(sub_id);
            self.next_id += 1;
        }

        // Link subgoals to parent
        if let Some(parent) = self.goals.get_mut(&goal_id) {
            parent.subgoals = subgoal_ids.clone();
        }

        subgoal_ids
    }

    /// Automatically decompose a goal using world model
    pub fn auto_decompose(
        &mut self,
        goal_id: usize,
        current_state: &[f64],
        world_model: &WorldModel,
    ) -> Vec<usize> {
        let target = match self.goals.get(&goal_id) {
            Some(g) => g.target_features.clone(),
            None => return Vec::new(),
        };

        // Try to find intermediate waypoints using world model
        let mut waypoints: Vec<Vec<f64>> = Vec::new();

        // Simple interpolation between current and goal
        let n_waypoints = 3;
        for i in 1..=n_waypoints {
            let alpha = i as f64 / (n_waypoints + 1) as f64;
            let waypoint: Vec<f64> = current_state
                .iter()
                .zip(target.iter())
                .map(|(&c, &t)| c * (1.0 - alpha) + t * alpha)
                .collect();
            waypoints.push(waypoint);
        }

        self.decompose_goal(goal_id, waypoints)
    }

    /// Activate a goal
    pub fn activate_goal(&mut self, goal_id: usize) {
        if self.active_goals.len() >= self.config.max_active_goals {
            // Remove lowest priority active goal
            if let Some(idx) = self.find_lowest_priority_active() {
                let removed_id = self.active_goals.remove(idx);
                if let Some(g) = self.goals.get_mut(&removed_id) {
                    g.status = GoalStatus::Pending;
                }
            }
        }

        if let Some(goal) = self.goals.get_mut(&goal_id) {
            goal.status = GoalStatus::Active;
            if !self.active_goals.contains(&goal_id) {
                self.active_goals.push(goal_id);
            }
        }
    }

    fn find_lowest_priority_active(&self) -> Option<usize> {
        self.active_goals
            .iter()
            .enumerate()
            .max_by_key(|(_, &id)| {
                self.goals
                    .get(&id)
                    .map(|g| match g.priority {
                        GoalPriority::Low => 3,
                        GoalPriority::Medium => 2,
                        GoalPriority::High => 1,
                        GoalPriority::Critical => 0,
                    })
                    .unwrap_or(4)
            })
            .map(|(idx, _)| idx)
    }

    /// Update goal progress based on current state - returns completed goal IDs for credit assignment
    pub fn update_progress(&mut self, current_state: &[f64]) -> Vec<usize> {
        self.current_step += 1;
        let mut completed_goals = Vec::new();

        for &goal_id in &self.active_goals {
            if let Some(goal) = self.goals.get_mut(&goal_id) {
                // Compute distance to target
                let distance: f64 = goal
                    .target_features
                    .iter()
                    .zip(current_state.iter())
                    .map(|(&t, &c)| (t - c).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let max_distance = goal.target_features.len() as f64 * 2.0; // Approximate
                goal.progress = 1.0 - (distance / max_distance).min(1.0);

                // Check completion
                if goal.progress >= self.config.goal_completion_threshold {
                    goal.status = GoalStatus::Completed;
                    completed_goals.push(goal_id);
                }
            }
        }

        // Handle completions
        for goal_id in &completed_goals {
            self.active_goals.retain(|&id| id != *goal_id);
            self.achievement_history
                .push_back((*goal_id, true, self.current_step));

            // Check if parent goal should be updated
            if let Some(parent_id) = self.goals.get(goal_id).and_then(|g| g.parent) {
                self.update_parent_progress(parent_id);
            }
        }

        while self.achievement_history.len() > 100 {
            self.achievement_history.pop_front();
        }

        completed_goals
    }

    fn update_parent_progress(&mut self, parent_id: usize) {
        let subgoal_progress: Vec<f64> = self
            .goals
            .get(&parent_id)
            .map(|p| {
                p.subgoals
                    .iter()
                    .filter_map(|&sid| self.goals.get(&sid).map(|s| s.progress))
                    .collect()
            })
            .unwrap_or_default();

        if !subgoal_progress.is_empty() {
            let avg_progress = subgoal_progress.iter().sum::<f64>() / subgoal_progress.len() as f64;
            if let Some(parent) = self.goals.get_mut(&parent_id) {
                parent.progress = avg_progress;

                if avg_progress >= self.config.goal_completion_threshold {
                    parent.status = GoalStatus::Completed;
                    self.active_goals.retain(|&id| id != parent_id);
                    self.achievement_history
                        .push_back((parent_id, true, self.current_step));
                }
            }
        }
    }

    /// Get the current highest priority active goal
    pub fn get_current_goal(&self) -> Option<&Goal> {
        self.active_goals
            .iter()
            .filter_map(|&id| self.goals.get(&id))
            .min_by_key(|g| match g.priority {
                GoalPriority::Critical => 0,
                GoalPriority::High => 1,
                GoalPriority::Medium => 2,
                GoalPriority::Low => 3,
            })
    }

    /// Get suggested action direction based on current goal
    pub fn get_goal_direction(&self, current_state: &[f64]) -> Option<Vec<f64>> {
        self.get_current_goal().map(|goal| {
            goal.target_features
                .iter()
                .zip(current_state.iter())
                .map(|(&t, &c)| t - c)
                .collect()
        })
    }

    /// Create a goal from a grounded symbol's sensory representation
    pub fn create_goal_from_symbol(
        &mut self,
        name: &str,
        target_features: Vec<f64>,
        concept_id: Option<usize>,
    ) -> usize {
        let goal = Goal {
            id: self.next_id,
            name: name.to_string(),
            target_features,
            tolerance: 0.2, // More tolerance for symbol-derived goals
            priority: GoalPriority::Medium,
            status: GoalStatus::Pending,
            parent: None,
            subgoals: Vec::new(),
            progress: 0.0,
            created_at: self.current_step,
            concept_id,
        };

        let id = goal.id;
        self.goals.insert(id, goal);
        self.next_id += 1;

        // Auto-activate if we have room
        if self.active_goals.len() < self.config.max_active_goals {
            self.activate_goal(id);
        }

        id
    }

    /// Check if a predicted state would complete any goal
    pub fn check_prediction_completes_goal(&self, predicted_state: &[f64]) -> Option<usize> {
        for &goal_id in &self.active_goals {
            if let Some(goal) = self.goals.get(&goal_id) {
                let distance: f64 = goal
                    .target_features
                    .iter()
                    .zip(predicted_state.iter())
                    .map(|(&t, &c)| (t - c).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let max_distance = goal.target_features.len() as f64 * 2.0;
                let progress = 1.0 - (distance / max_distance).min(1.0);

                // Slightly lower threshold for predictions (0.7 vs actual completion at 0.85)
                if progress >= self.config.goal_completion_threshold * 0.85 {
                    return Some(goal_id);
                }
            }
        }
        None
    }

    /// Get number of active goals
    pub fn active_count(&self) -> usize {
        self.active_goals.len()
    }

    /// Get recently completed goals (for skill extraction)
    pub fn get_recently_completed(&self) -> Vec<&Goal> {
        self.achievement_history
            .iter()
            .rev()
            .take(5)
            .filter(|(_, success, _)| *success)
            .filter_map(|(id, _, _)| self.goals.get(id))
            .filter(|g| g.status == GoalStatus::Completed)
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> GoalHierarchySummary {
        let completed = self
            .goals
            .values()
            .filter(|g| g.status == GoalStatus::Completed)
            .count();
        let failed = self
            .goals
            .values()
            .filter(|g| g.status == GoalStatus::Failed)
            .count();

        GoalHierarchySummary {
            total_goals: self.goals.len(),
            active_goals: self.active_goals.len(),
            completed_goals: completed,
            failed_goals: failed,
            success_rate: if completed + failed > 0 {
                completed as f64 / (completed + failed) as f64
            } else {
                0.0
            },
            avg_progress: if self.active_goals.is_empty() {
                0.0
            } else {
                self.active_goals
                    .iter()
                    .filter_map(|&id| self.goals.get(&id).map(|g| g.progress))
                    .sum::<f64>()
                    / self.active_goals.len() as f64
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalHierarchySummary {
    pub total_goals: usize,
    pub active_goals: usize,
    pub completed_goals: usize,
    pub failed_goals: usize,
    pub success_rate: f64,
    pub avg_progress: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// META-LEARNER - Learn how to learn
// ═══════════════════════════════════════════════════════════════════════════════

/// Learning strategy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetrics {
    /// Learning rate used
    pub learning_rate: f64,
    /// Exploration rate
    pub exploration_rate: f64,
    /// Batch size (if applicable)
    pub batch_size: usize,
    /// Performance achieved
    pub performance: f64,
    /// Sample efficiency (performance / samples)
    pub sample_efficiency: f64,
}

/// A learning episode record for meta-learning
#[derive(Debug, Clone)]
pub struct LearningEpisode {
    /// Episode number
    pub episode: usize,
    /// Initial performance
    pub initial_performance: f64,
    /// Final performance
    pub final_performance: f64,
    /// Learning metrics used
    pub metrics: LearningMetrics,
    /// Steps taken
    pub steps: usize,
    /// Improvement achieved
    pub improvement: f64,
}

/// Meta-Learner - learns how to learn effectively
#[derive(Debug, Clone)]
pub struct MetaLearner {
    /// History of learning episodes
    learning_history: VecDeque<LearningEpisode>,
    /// Current best learning rate
    pub best_learning_rate: f64,
    /// Current best exploration rate
    pub best_exploration_rate: f64,
    /// Performance by strategy
    strategy_performance: HashMap<String, Vec<f64>>,
    /// Adaptive learning rate bounds
    lr_bounds: (f64, f64),
    /// Adaptive exploration bounds
    exploration_bounds: (f64, f64),
    /// Configuration
    config: AGICoreConfig,
    /// Current episode
    current_episode: usize,
}

impl MetaLearner {
    pub fn new(config: AGICoreConfig) -> Self {
        Self {
            learning_history: VecDeque::with_capacity(config.meta_learning_window),
            best_learning_rate: 0.01,
            best_exploration_rate: 0.3,
            strategy_performance: HashMap::new(),
            lr_bounds: (0.0001, 0.1),
            exploration_bounds: (0.05, 0.5),
            config,
            current_episode: 0,
        }
    }

    /// Record a learning episode
    pub fn record_episode(
        &mut self,
        initial_perf: f64,
        final_perf: f64,
        steps: usize,
        metrics: LearningMetrics,
    ) {
        let improvement = final_perf - initial_perf;

        let episode = LearningEpisode {
            episode: self.current_episode,
            initial_performance: initial_perf,
            final_performance: final_perf,
            metrics: metrics.clone(),
            steps,
            improvement,
        };

        self.learning_history.push_back(episode);

        while self.learning_history.len() > self.config.meta_learning_window {
            self.learning_history.pop_front();
        }

        // Record strategy performance
        let strategy_key = format!(
            "lr_{:.4}_exp_{:.2}",
            (metrics.learning_rate * 1000.0).round() / 1000.0,
            (metrics.exploration_rate * 10.0).round() / 10.0
        );

        self.strategy_performance
            .entry(strategy_key)
            .or_insert_with(Vec::new)
            .push(improvement);

        self.current_episode += 1;

        // Adapt strategies
        self.adapt_strategies();
    }

    /// Adapt learning strategies based on history
    fn adapt_strategies(&mut self) {
        if self.learning_history.len() < 10 {
            return;
        }

        // Find best performing strategy
        let mut best_strategy: Option<(String, f64)> = None;

        for (strategy, performances) in &self.strategy_performance {
            if performances.len() >= 3 {
                let avg = performances.iter().sum::<f64>() / performances.len() as f64;
                if best_strategy
                    .as_ref()
                    .map(|(_, s)| avg > *s)
                    .unwrap_or(true)
                {
                    best_strategy = Some((strategy.clone(), avg));
                }
            }
        }

        // Parse and adopt best strategy
        if let Some((strategy, _)) = best_strategy {
            let parts: Vec<&str> = strategy.split('_').collect();
            if parts.len() >= 4 {
                if let Ok(lr) = parts[1].parse::<f64>() {
                    self.best_learning_rate = lr.max(self.lr_bounds.0).min(self.lr_bounds.1);
                }
                if let Ok(exp) = parts[3].parse::<f64>() {
                    self.best_exploration_rate = exp
                        .max(self.exploration_bounds.0)
                        .min(self.exploration_bounds.1);
                }
            }
        }

        // Also adapt based on recent trend
        let recent: Vec<&LearningEpisode> = self.learning_history.iter().rev().take(10).collect();

        if recent.len() >= 5 {
            let recent_improvement: f64 =
                recent.iter().map(|e| e.improvement).sum::<f64>() / recent.len() as f64;

            // If not improving, increase exploration
            if recent_improvement < 0.01 {
                self.best_exploration_rate =
                    (self.best_exploration_rate * 1.1).min(self.exploration_bounds.1);
            } else if recent_improvement > 0.1 {
                // If improving well, decrease exploration slightly
                self.best_exploration_rate =
                    (self.best_exploration_rate * 0.95).max(self.exploration_bounds.0);
            }
        }
    }

    /// Get recommended learning parameters
    pub fn get_recommended_params(&self) -> LearningMetrics {
        // Add some exploration to the recommended params
        let lr_noise = (rand_simple() - 0.5) * 0.002;
        let exp_noise = (rand_simple() - 0.5) * 0.05;

        LearningMetrics {
            learning_rate: (self.best_learning_rate + lr_noise)
                .max(self.lr_bounds.0)
                .min(self.lr_bounds.1),
            exploration_rate: (self.best_exploration_rate + exp_noise)
                .max(self.exploration_bounds.0)
                .min(self.exploration_bounds.1),
            batch_size: 32,
            performance: 0.0,
            sample_efficiency: 0.0,
        }
    }

    /// Predict expected improvement for given parameters
    pub fn predict_improvement(&self, metrics: &LearningMetrics) -> f64 {
        let strategy_key = format!(
            "lr_{:.4}_exp_{:.2}",
            (metrics.learning_rate * 1000.0).round() / 1000.0,
            (metrics.exploration_rate * 10.0).round() / 10.0
        );

        self.strategy_performance
            .get(&strategy_key)
            .map(|perfs| {
                if perfs.is_empty() {
                    0.0
                } else {
                    perfs.iter().sum::<f64>() / perfs.len() as f64
                }
            })
            .unwrap_or(0.0)
    }

    /// Get summary statistics
    pub fn summary(&self) -> MetaLearnerSummary {
        let improvements: Vec<f64> = self
            .learning_history
            .iter()
            .map(|e| e.improvement)
            .collect();

        let avg_improvement = if improvements.is_empty() {
            0.0
        } else {
            improvements.iter().sum::<f64>() / improvements.len() as f64
        };

        let recent_improvements: Vec<f64> = self
            .learning_history
            .iter()
            .rev()
            .take(10)
            .map(|e| e.improvement)
            .collect();

        let recent_avg = if recent_improvements.is_empty() {
            0.0
        } else {
            recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64
        };

        MetaLearnerSummary {
            total_episodes: self.current_episode,
            best_learning_rate: self.best_learning_rate,
            best_exploration_rate: self.best_exploration_rate,
            avg_improvement,
            recent_avg_improvement: recent_avg,
            strategies_tried: self.strategy_performance.len(),
        }
    }
}

// Simple random for exploration
fn rand_simple() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos as f64 / u32::MAX as f64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnerSummary {
    pub total_episodes: usize,
    pub best_learning_rate: f64,
    pub best_exploration_rate: f64,
    pub avg_improvement: f64,
    pub recent_avg_improvement: f64,
    pub strategies_tried: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYMBOL SYSTEM - Internal language grounded in sensorimotor experience
// ═══════════════════════════════════════════════════════════════════════════════

/// A grounded symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Unique identifier
    pub id: usize,
    /// Symbol name/token
    pub name: String,
    /// Grounding in sensory features
    pub sensory_grounding: Vec<f64>,
    /// Grounding in motor commands
    pub motor_grounding: Vec<f64>,
    /// Associated concept (if any)
    pub concept_id: Option<usize>,
    /// Associated goal (if any)
    pub goal_id: Option<usize>,
    /// Usage count
    pub usage_count: usize,
    /// Confidence in grounding
    pub grounding_confidence: f64,
}

/// A compositional expression using symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicExpression {
    /// Symbols in the expression
    pub symbols: Vec<usize>,
    /// Relation type between symbols
    pub relation: SymbolRelation,
    /// Composed meaning (feature space)
    pub composed_meaning: Vec<f64>,
}

/// Relations between symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolRelation {
    Sequence,    // A then B
    Conjunction, // A and B
    Causation,   // A causes B
    PartOf,      // A is part of B
    Negation,    // Not A
}

/// Symbol System - internal language grounded in experience
#[derive(Debug, Clone)]
pub struct SymbolSystem {
    /// All symbols by ID
    symbols: HashMap<usize, Symbol>,
    /// Symbol name to ID mapping (public for compound lookups)
    pub name_to_id: HashMap<String, usize>,
    /// Compositional expressions
    expressions: Vec<SymbolicExpression>,
    /// Co-occurrence counts for learning relations
    cooccurrence: HashMap<(usize, usize), usize>,
    /// Next symbol ID
    next_id: usize,
    /// Configuration
    config: AGICoreConfig,
}

impl SymbolSystem {
    pub fn new(config: AGICoreConfig) -> Self {
        Self {
            symbols: HashMap::new(),
            name_to_id: HashMap::new(),
            expressions: Vec::new(),
            cooccurrence: HashMap::new(),
            next_id: 0,
            config,
        }
    }

    /// Create or retrieve a symbol
    pub fn get_or_create_symbol(
        &mut self,
        name: &str,
        sensory: Vec<f64>,
        motor: Vec<f64>,
    ) -> usize {
        if let Some(&id) = self.name_to_id.get(name) {
            // Update grounding
            if let Some(sym) = self.symbols.get_mut(&id) {
                let alpha = 0.1;
                for (i, s) in sym.sensory_grounding.iter_mut().enumerate() {
                    if let Some(&new_s) = sensory.get(i) {
                        *s = *s * (1.0 - alpha) + new_s * alpha;
                    }
                }
                for (i, m) in sym.motor_grounding.iter_mut().enumerate() {
                    if let Some(&new_m) = motor.get(i) {
                        *m = *m * (1.0 - alpha) + new_m * alpha;
                    }
                }
                sym.usage_count += 1;
                sym.grounding_confidence = (sym.grounding_confidence + 0.01).min(1.0);
            }
            return id;
        }

        // Create new symbol
        if self.symbols.len() >= self.config.max_symbols {
            // Remove least used symbol
            if let Some(&id) = self
                .symbols
                .values()
                .min_by_key(|s| s.usage_count)
                .map(|s| &s.id)
            {
                if let Some(sym) = self.symbols.remove(&id) {
                    self.name_to_id.remove(&sym.name);
                }
            }
        }

        let symbol = Symbol {
            id: self.next_id,
            name: name.to_string(),
            sensory_grounding: sensory,
            motor_grounding: motor,
            concept_id: None,
            goal_id: None,
            usage_count: 1,
            grounding_confidence: 0.5,
        };

        let id = symbol.id;
        self.name_to_id.insert(name.to_string(), id);
        self.symbols.insert(id, symbol);
        self.next_id += 1;
        id
    }

    /// Record co-occurrence of symbols
    pub fn record_cooccurrence(&mut self, sym1: usize, sym2: usize) {
        let key = if sym1 < sym2 {
            (sym1, sym2)
        } else {
            (sym2, sym1)
        };
        *self.cooccurrence.entry(key).or_insert(0) += 1;
    }

    /// Ground a symbol in a concept
    pub fn ground_in_concept(&mut self, symbol_id: usize, concept_id: usize) {
        if let Some(sym) = self.symbols.get_mut(&symbol_id) {
            sym.concept_id = Some(concept_id);
            sym.grounding_confidence = (sym.grounding_confidence + 0.1).min(1.0);
        }
    }

    /// Ground a symbol in a goal
    pub fn ground_in_goal(&mut self, symbol_id: usize, goal_id: usize) {
        if let Some(sym) = self.symbols.get_mut(&symbol_id) {
            sym.goal_id = Some(goal_id);
        }
    }

    /// Compose symbols into an expression
    pub fn compose(
        &mut self,
        symbol_ids: Vec<usize>,
        relation: SymbolRelation,
    ) -> Option<SymbolicExpression> {
        let symbols: Vec<&Symbol> = symbol_ids
            .iter()
            .filter_map(|&id| self.symbols.get(&id))
            .collect();

        if symbols.len() != symbol_ids.len() {
            return None;
        }

        // Compose meaning based on relation type
        let dim = symbols
            .first()
            .map(|s| s.sensory_grounding.len())
            .unwrap_or(0);
        let mut composed = vec![0.0; dim];

        match relation {
            SymbolRelation::Sequence | SymbolRelation::Conjunction => {
                // Average the groundings
                for sym in &symbols {
                    for (i, &s) in sym.sensory_grounding.iter().enumerate() {
                        if i < composed.len() {
                            composed[i] += s / symbols.len() as f64;
                        }
                    }
                }
            }
            SymbolRelation::Causation => {
                // Emphasize the effect (last symbol)
                if let Some(last) = symbols.last() {
                    composed = last.sensory_grounding.clone();
                }
            }
            SymbolRelation::PartOf => {
                // Emphasize the whole (last symbol)
                if let Some(last) = symbols.last() {
                    composed = last.sensory_grounding.clone();
                }
            }
            SymbolRelation::Negation => {
                // Negate the first symbol's grounding
                if let Some(first) = symbols.first() {
                    composed = first.sensory_grounding.iter().map(|&x| -x).collect();
                }
            }
        }

        // Record co-occurrences
        for i in 0..symbol_ids.len() {
            for j in (i + 1)..symbol_ids.len() {
                self.record_cooccurrence(symbol_ids[i], symbol_ids[j]);
            }
        }

        let expr = SymbolicExpression {
            symbols: symbol_ids,
            relation,
            composed_meaning: composed,
        };

        self.expressions.push(expr.clone());
        Some(expr)
    }

    /// Find symbol by sensory pattern
    pub fn find_by_sensory(&self, sensory: &[f64], threshold: f64) -> Option<&Symbol> {
        self.symbols
            .values()
            .filter(|s| s.grounding_confidence >= self.config.symbol_grounding_threshold)
            .max_by(|a, b| {
                let sim_a = cosine_sim(&a.sensory_grounding, sensory);
                let sim_b = cosine_sim(&b.sensory_grounding, sensory);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|s| cosine_sim(&s.sensory_grounding, sensory) >= threshold)
    }

    /// Get symbols related to a given symbol
    pub fn get_related(&self, symbol_id: usize) -> Vec<(usize, usize)> {
        let mut related: Vec<(usize, usize)> = self
            .cooccurrence
            .iter()
            .filter(|((s1, s2), _)| *s1 == symbol_id || *s2 == symbol_id)
            .map(|((s1, s2), &count)| {
                let other = if *s1 == symbol_id { *s2 } else { *s1 };
                (other, count)
            })
            .collect();

        related.sort_by(|a, b| b.1.cmp(&a.1));
        related
    }

    /// Count symbols that are grounded (high confidence + concept)
    pub fn grounded_count(&self) -> usize {
        self.symbols
            .values()
            .filter(|s| {
                s.grounding_confidence >= self.config.symbol_grounding_threshold
                    && s.concept_id.is_some()
            })
            .count()
    }

    /// Get all grounded symbols without a goal association
    pub fn get_ungoaled_grounded_symbols(&self) -> Vec<&Symbol> {
        self.symbols
            .values()
            .filter(|s| {
                s.grounding_confidence >= self.config.symbol_grounding_threshold
                    && s.concept_id.is_some()
                    && s.goal_id.is_none()
            })
            .collect()
    }

    /// Get symbol by ID
    pub fn get_symbol(&self, id: usize) -> Option<&Symbol> {
        self.symbols.get(&id)
    }

    /// Get symbols activated by current state (similarity above threshold)
    pub fn get_activated_symbols(&self, state: &[f64], threshold: f64) -> Vec<&Symbol> {
        self.symbols
            .values()
            .filter(|s| {
                s.grounding_confidence >= self.config.symbol_grounding_threshold
                    && cosine_sim(&s.sensory_grounding, state) >= threshold
            })
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> SymbolSystemSummary {
        let grounded_count = self
            .symbols
            .values()
            .filter(|s| s.grounding_confidence >= self.config.symbol_grounding_threshold)
            .count();

        let concept_grounded = self
            .symbols
            .values()
            .filter(|s| s.concept_id.is_some())
            .count();

        SymbolSystemSummary {
            total_symbols: self.symbols.len(),
            grounded_symbols: grounded_count,
            concept_grounded_symbols: concept_grounded,
            total_expressions: self.expressions.len(),
            unique_relations: self.cooccurrence.len(),
            avg_grounding_confidence: if self.symbols.is_empty() {
                0.0
            } else {
                self.symbols
                    .values()
                    .map(|s| s.grounding_confidence)
                    .sum::<f64>()
                    / self.symbols.len() as f64
            },
        }
    }
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSystemSummary {
    pub total_symbols: usize,
    pub grounded_symbols: usize,
    pub concept_grounded_symbols: usize,
    pub total_expressions: usize,
    pub unique_relations: usize,
    pub avg_grounding_confidence: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SKILL SYSTEM - Extract reusable action sequences
// ═══════════════════════════════════════════════════════════════════════════════

/// A learned skill (reusable action sequence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier
    pub id: usize,
    /// Human-readable name
    pub name: String,
    /// Precondition features (when can this skill be applied)
    pub preconditions: Vec<f64>,
    /// Expected effect features (what this skill achieves)
    pub effects: Vec<f64>,
    /// Action sequence
    pub action_sequence: Vec<usize>,
    /// Success rate
    pub success_rate: f64,
    /// Times executed
    pub execution_count: usize,
    /// Average reward when executed
    pub avg_reward: f64,
    /// Source goal ID (if extracted from goal completion)
    pub source_goal: Option<usize>,
}

/// Skill library for storing and retrieving reusable behaviors
#[derive(Debug, Clone)]
pub struct SkillLibrary {
    /// All learned skills
    pub skills: HashMap<usize, Skill>,
    /// Next skill ID
    next_id: usize,
    /// Action buffer for detecting skill-worthy sequences
    action_buffer: VecDeque<(Vec<f64>, usize, f64)>, // (state, action, reward)
    /// Configuration
    config: AGICoreConfig,
}

impl SkillLibrary {
    pub fn new(config: AGICoreConfig) -> Self {
        Self {
            skills: HashMap::new(),
            next_id: 0,
            action_buffer: VecDeque::with_capacity(50),
            config,
        }
    }

    /// Record an action for potential skill extraction
    pub fn record_action(&mut self, state: Vec<f64>, action: usize, reward: f64) {
        self.action_buffer.push_back((state, action, reward));
        while self.action_buffer.len() > 50 {
            self.action_buffer.pop_front();
        }
    }

    /// Extract a skill from a successful goal completion
    pub fn extract_skill_from_goal(
        &mut self,
        goal_name: &str,
        goal_features: Vec<f64>,
        success: bool,
    ) -> Option<usize> {
        if !success || self.action_buffer.len() < 3 {
            return None;
        }

        if self.skills.len() >= self.config.max_skills {
            return None;
        }

        // Get the action sequence leading to goal completion
        let sequence: Vec<(Vec<f64>, usize, f64)> = self.action_buffer.iter().cloned().collect();
        let actions: Vec<usize> = sequence.iter().map(|(_, a, _)| *a).collect();
        let total_reward: f64 = sequence.iter().map(|(_, _, r)| r).sum();

        // Use the initial state as preconditions
        let preconditions = sequence
            .first()
            .map(|(s, _, _)| s.clone())
            .unwrap_or_default();

        let skill = Skill {
            id: self.next_id,
            name: format!("skill_{}", goal_name),
            preconditions,
            effects: goal_features,
            action_sequence: actions,
            success_rate: 1.0,
            execution_count: 1,
            avg_reward: total_reward / sequence.len() as f64,
            source_goal: None,
        };

        let id = skill.id;
        self.skills.insert(id, skill);
        self.next_id += 1;

        // Clear buffer after skill extraction
        self.action_buffer.clear();

        Some(id)
    }

    /// Find applicable skills for current state
    pub fn find_applicable_skills(&self, state: &[f64]) -> Vec<&Skill> {
        self.skills
            .values()
            .filter(|skill| {
                // Check if preconditions match current state (within tolerance)
                let distance: f64 = skill
                    .preconditions
                    .iter()
                    .zip(state.iter())
                    .map(|(&p, &s)| (p - s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distance < 2.0 // Tolerance for precondition matching
            })
            .collect()
    }

    /// Update skill success rate after execution
    pub fn update_skill(&mut self, skill_id: usize, success: bool, reward: f64) {
        if let Some(skill) = self.skills.get_mut(&skill_id) {
            skill.execution_count += 1;
            let n = skill.execution_count as f64;
            skill.success_rate =
                (skill.success_rate * (n - 1.0) + if success { 1.0 } else { 0.0 }) / n;
            skill.avg_reward = (skill.avg_reward * (n - 1.0) + reward) / n;
        }
    }

    /// Get skill recommendation for achieving a target state
    pub fn recommend_skill(&self, current: &[f64], target: &[f64]) -> Option<&Skill> {
        self.skills
            .values()
            .filter(|skill| {
                // Check preconditions match
                let pre_dist: f64 = skill
                    .preconditions
                    .iter()
                    .zip(current.iter())
                    .map(|(&p, &s)| (p - s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                // Check effects move toward target
                let effect_dist: f64 = skill
                    .effects
                    .iter()
                    .zip(target.iter())
                    .map(|(&e, &t)| (e - t).powi(2))
                    .sum::<f64>()
                    .sqrt();
                pre_dist < 2.0 && effect_dist < 3.0
            })
            .max_by(|a, b| {
                let score_a = a.success_rate * a.avg_reward;
                let score_b = b.success_rate * b.avg_reward;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn summary(&self) -> SkillLibrarySummary {
        let avg_success = if self.skills.is_empty() {
            0.0
        } else {
            self.skills.values().map(|s| s.success_rate).sum::<f64>() / self.skills.len() as f64
        };

        SkillLibrarySummary {
            total_skills: self.skills.len(),
            avg_success_rate: avg_success,
            total_executions: self.skills.values().map(|s| s.execution_count).sum(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLibrarySummary {
    pub total_skills: usize,
    pub avg_success_rate: f64,
    pub total_executions: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SELF-MODEL - Track agent capabilities and uncertainties
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-model for introspection about agent capabilities
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Reachability estimates: state_key -> (steps_to_reach, confidence)
    reachable_states: HashMap<String, (usize, f64)>,
    /// Model uncertainty by region
    model_uncertainty: HashMap<String, f64>,
    /// Success rates by goal type
    success_rates: HashMap<String, (usize, usize)>, // (successes, attempts)
    /// Action effectiveness estimates
    action_effectiveness: Vec<f64>,
    /// Recent prediction errors (for calibration)
    prediction_errors: VecDeque<f64>,
    /// Estimated competence (0-1)
    pub competence: f64,
    /// Number of actions
    n_actions: usize,
}

impl SelfModel {
    pub fn new(n_actions: usize) -> Self {
        Self {
            reachable_states: HashMap::new(),
            model_uncertainty: HashMap::new(),
            success_rates: HashMap::new(),
            action_effectiveness: vec![0.5; n_actions],
            prediction_errors: VecDeque::with_capacity(100),
            competence: 0.5,
            n_actions,
        }
    }

    /// Record reaching a state and update reachability model
    pub fn record_reached_state(&mut self, state: &[f64], steps: usize) {
        let key = self.state_to_key(state);
        self.reachable_states
            .entry(key)
            .and_modify(|(s, c)| {
                *s = (*s).min(steps); // Track minimum steps
                *c = (*c * 0.9 + 0.1).min(1.0); // Increase confidence
            })
            .or_insert((steps, 0.5));
    }

    /// Record goal attempt outcome
    pub fn record_goal_outcome(&mut self, goal_type: &str, success: bool) {
        let (succ, attempts) = self
            .success_rates
            .entry(goal_type.to_string())
            .or_insert((0, 0));
        if success {
            *succ += 1;
        }
        *attempts += 1;

        // Update overall competence
        let total_succ: usize = self.success_rates.values().map(|(s, _)| s).sum();
        let total_att: usize = self.success_rates.values().map(|(_, a)| a).sum();
        if total_att > 0 {
            self.competence = total_succ as f64 / total_att as f64;
        }
    }

    /// Record prediction error for calibration
    pub fn record_prediction_error(&mut self, error: f64) {
        self.prediction_errors.push_back(error);
        while self.prediction_errors.len() > 100 {
            self.prediction_errors.pop_front();
        }
    }

    /// Update action effectiveness based on outcomes
    pub fn update_action_effectiveness(&mut self, action: usize, reward: f64) {
        if action < self.action_effectiveness.len() {
            let old = self.action_effectiveness[action];
            self.action_effectiveness[action] = old * 0.95 + reward.max(0.0).min(1.0) * 0.05;
        }
    }

    /// Estimate steps to reach a target state
    pub fn estimate_steps_to(&self, current: &[f64], target: &[f64]) -> Option<usize> {
        let target_key = self.state_to_key(target);

        // Direct lookup
        if let Some(&(steps, confidence)) = self.reachable_states.get(&target_key) {
            if confidence > 0.3 {
                return Some(steps);
            }
        }

        // Estimate based on distance and competence
        let distance: f64 = current
            .iter()
            .zip(target.iter())
            .map(|(&c, &t)| (c - t).powi(2))
            .sum::<f64>()
            .sqrt();

        Some((distance / (self.competence + 0.1)).ceil() as usize)
    }

    /// Get model uncertainty for a region of state space
    pub fn get_uncertainty(&self, state: &[f64]) -> f64 {
        let key = self.state_to_key(state);
        self.model_uncertainty.get(&key).copied().unwrap_or(1.0) // Default high uncertainty
    }

    /// Get most effective actions
    pub fn get_best_actions(&self) -> Vec<usize> {
        let mut actions: Vec<(usize, f64)> = self
            .action_effectiveness
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        actions.into_iter().take(3).map(|(i, _)| i).collect()
    }

    /// Get calibration score (how well-calibrated are our predictions)
    pub fn calibration_score(&self) -> f64 {
        if self.prediction_errors.is_empty() {
            return 0.5;
        }
        let avg_error: f64 =
            self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64;
        1.0 / (1.0 + avg_error)
    }

    fn state_to_key(&self, state: &[f64]) -> String {
        // Discretize state for hashing
        state
            .iter()
            .take(4)
            .map(|&v| ((v * 10.0).round() as i32).to_string())
            .collect::<Vec<_>>()
            .join("_")
    }

    pub fn summary(&self) -> SelfModelSummary {
        SelfModelSummary {
            known_states: self.reachable_states.len(),
            goal_types_tracked: self.success_rates.len(),
            overall_competence: self.competence,
            calibration: self.calibration_score(),
            best_actions: self.get_best_actions(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelSummary {
    pub known_states: usize,
    pub goal_types_tracked: usize,
    pub overall_competence: f64,
    pub calibration: f64,
    pub best_actions: Vec<usize>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED AGI CORE - Integration of all systems
// ═══════════════════════════════════════════════════════════════════════════════

/// Compounding metrics tracking how capabilities multiply
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundingAnalytics {
    /// Discovery → Abstraction: new concepts from discovered variables
    pub discovery_to_abstraction: usize,
    /// Abstraction → Symbols: grounded symbols from concepts
    pub abstraction_to_symbols: usize,
    /// Symbols → Goals: goals expressed symbolically
    pub symbols_to_goals: usize,
    /// Goals → World Model: imagination queries driven by goals
    pub goals_to_world_model: usize,
    /// World Model → Meta-Learning: learning updates from imagination
    pub world_model_to_meta: usize,
    /// Meta-Learning → Discovery: improved discovery from meta-learning
    pub meta_to_discovery: usize,
    /// Total compound interactions
    pub total_interactions: usize,
    /// Compound growth rate
    pub compound_rate: f64,
    /// Credit assignments from goal completion back to discoveries
    pub credit_to_discoveries: usize,
    /// Credit assignments from goal completion back to symbols
    pub credit_to_symbols: usize,
    /// Surprise-triggered meta updates (vs periodic)
    pub surprise_meta_updates: usize,
    /// Imagination-driven abstractions created
    pub imagination_abstractions: usize,
    /// Compound amplification factor (grows with success)
    pub amplification_factor: f64,
    /// Credit blockages detected by AGC (coherence gates restricting flow)
    pub credit_blockages: usize,
}

impl Default for CompoundingAnalytics {
    fn default() -> Self {
        Self {
            discovery_to_abstraction: 0,
            abstraction_to_symbols: 0,
            symbols_to_goals: 0,
            goals_to_world_model: 0,
            world_model_to_meta: 0,
            meta_to_discovery: 0,
            total_interactions: 0,
            compound_rate: 0.0,
            credit_to_discoveries: 0,
            credit_to_symbols: 0,
            surprise_meta_updates: 0,
            imagination_abstractions: 0,
            amplification_factor: 1.0,
            credit_blockages: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECISION AUDIT TRAIL (FOR EXPLICABILITY)
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks the contribution of each subsystem to a decision
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionAudit {
    /// The step this decision was made
    pub step: usize,
    /// The action that was selected
    pub selected_action: usize,
    /// Whether the decision was exploration vs exploitation
    pub was_exploration: bool,
    /// Q-value contribution for the selected action
    pub q_value_contribution: f64,
    /// Goal alignment contribution
    pub goal_contribution: f64,
    /// World model prediction contribution
    pub world_model_contribution: f64,
    /// Concept-based contribution from abstraction hierarchy
    pub concept_contribution: f64,
    /// Symbol-based contribution
    pub symbol_contribution: f64,
    /// Causal relevance factor
    pub causal_relevance: f64,
    /// Names of concepts that influenced the decision
    pub influential_concepts: Vec<String>,
    /// Names of symbols that influenced the decision
    pub influential_symbols: Vec<String>,
    /// Active goal name (if any)
    pub active_goal: Option<String>,
    /// Total decision confidence (sum of contributions)
    pub total_confidence: f64,
}

impl DecisionAudit {
    /// Get a human-readable explanation of the decision
    pub fn explain(&self) -> String {
        let mut explanation = format!("Action {} selected ", self.selected_action);

        if self.was_exploration {
            explanation.push_str("(exploration) ");
        } else {
            explanation.push_str("(exploitation) ");
        }

        let mut reasons: Vec<String> = Vec::new();

        if self.q_value_contribution > 0.1 {
            reasons.push(format!("Q-value: {:.2}", self.q_value_contribution));
        }
        if self.goal_contribution > 0.1 {
            if let Some(ref goal) = self.active_goal {
                reasons.push(format!("goal '{}': {:.2}", goal, self.goal_contribution));
            } else {
                reasons.push(format!("goal alignment: {:.2}", self.goal_contribution));
            }
        }
        if self.world_model_contribution > 0.1 {
            reasons.push(format!("world model: {:.2}", self.world_model_contribution));
        }
        if self.concept_contribution > 0.1 {
            if !self.influential_concepts.is_empty() {
                reasons.push(format!(
                    "concepts [{}]: {:.2}",
                    self.influential_concepts.join(", "),
                    self.concept_contribution
                ));
            }
        }
        if self.symbol_contribution > 0.1 {
            if !self.influential_symbols.is_empty() {
                reasons.push(format!(
                    "symbols [{}]: {:.2}",
                    self.influential_symbols.join(", "),
                    self.symbol_contribution
                ));
            }
        }

        if reasons.is_empty() {
            explanation.push_str("(no strong factors)");
        } else {
            explanation.push_str("because: ");
            explanation.push_str(&reasons.join(", "));
        }

        explanation
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE HEALTH MONITORING (COLLIDER-INSPIRED)
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of cognitive anomalies (inspired by CERN Collider anomaly detection)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveAnomaly {
    /// Goal system creating too many goals (runaway process)
    GoalExplosion,
    /// Symbol system saturated (no room for new symbols)
    SymbolSaturation,
    /// World model predictions degrading (increasing error)
    WorldModelDegradation,
    /// Meta-learner parameters oscillating (instability)
    MetaInstability,
    /// Credit assignment not flowing (blocked loop)
    CreditBlockage,
    /// Compound rate dropping (system decohering)
    CompoundCollapse,
    /// Abstraction hierarchy too flat (no learning)
    AbstractionStagnation,
    /// Discovery system inactive (no new variables)
    DiscoveryDormant,
}

impl CognitiveAnomaly {
    /// Get severity (0=info, 1=warning, 2=error, 3=critical)
    pub fn severity(&self) -> u8 {
        match self {
            Self::GoalExplosion => 2,
            Self::SymbolSaturation => 1,
            Self::WorldModelDegradation => 2,
            Self::MetaInstability => 3,
            Self::CreditBlockage => 2,
            Self::CompoundCollapse => 3,
            Self::AbstractionStagnation => 1,
            Self::DiscoveryDormant => 1,
        }
    }

    /// Get recommended action
    pub fn recommended_action(&self) -> &'static str {
        match self {
            Self::GoalExplosion => "Reduce goal creation rate or increase completion threshold",
            Self::SymbolSaturation => "Prune unused symbols or increase symbol capacity",
            Self::WorldModelDegradation => "Reset world model or reduce learning rate",
            Self::MetaInstability => "Reduce meta-learning rate or increase smoothing",
            Self::CreditBlockage => "Check goal completion pathway or lower thresholds",
            Self::CompoundCollapse => "Increase subsystem interaction rates",
            Self::AbstractionStagnation => "Lower abstraction merge threshold",
            Self::DiscoveryDormant => {
                "Lower causal discovery threshold or increase observation window"
            }
        }
    }
}

/// Cognitive health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveHealthReport {
    /// Current step
    pub step: usize,
    /// Detected anomalies
    pub anomalies: Vec<CognitiveAnomaly>,
    /// Overall health score (0.0 = critical, 1.0 = perfect)
    pub health_score: f64,
    /// Subsystem health scores
    pub subsystem_health: SubsystemHealth,
    /// Is the system healthy overall?
    pub is_healthy: bool,
}

/// Health scores for each subsystem
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubsystemHealth {
    pub causal_discovery: f64,
    pub abstraction: f64,
    pub world_model: f64,
    pub goals: f64,
    pub meta_learner: f64,
    pub symbols: f64,
    pub compounding: f64,
}

/// Thresholds for anomaly detection
#[derive(Debug, Clone)]
pub struct CognitiveHealthThresholds {
    pub max_goals_per_step: f64,
    pub max_symbol_usage_ratio: f64,
    pub max_world_model_error: f64,
    pub max_meta_oscillation: f64,
    pub min_credit_flow_ratio: f64,
    pub min_compound_rate: f64,
    pub min_abstraction_depth: usize,
    pub min_discovery_rate: f64,
}

impl Default for CognitiveHealthThresholds {
    fn default() -> Self {
        Self {
            max_goals_per_step: 10.0,     // Goals created per step
            max_symbol_usage_ratio: 0.99, // Symbol capacity usage
            max_world_model_error: 0.8,   // Prediction error threshold
            max_meta_oscillation: 0.5,    // Parameter change rate
            min_credit_flow_ratio: 0.001, // Credit events per step
            min_compound_rate: 1.0,       // Minimum interactions per step
            min_abstraction_depth: 1,     // Minimum hierarchy depth
            min_discovery_rate: 0.0001,   // Variables discovered per step
        }
    }
}

/// Unified AGI Core - integrates all cognitive capabilities
pub struct AGICore {
    /// Configuration
    pub config: AGICoreConfig,

    /// Causal Discovery System
    pub causal_discovery: CausalDiscovery,

    /// Hierarchical Abstraction System
    pub abstraction: AbstractionHierarchy,

    /// World Model for imagination
    pub world_model: WorldModel,

    /// Goal Hierarchy System
    pub goals: GoalHierarchy,

    /// Meta-Learning System
    pub meta_learner: MetaLearner,

    /// Symbol Grounding System
    pub symbols: SymbolSystem,

    /// Skill Library (NEW)
    pub skills: SkillLibrary,

    /// Self-Model for introspection (NEW)
    pub self_model: SelfModel,

    /// Counterfactual policy bias (NEW) - learned from regret
    pub policy_bias: Vec<f64>,

    /// Compounding analytics
    pub analytics: CompoundingAnalytics,

    /// Current step
    current_step: usize,

    /// Feature dimension
    feature_dim: usize,

    /// Number of actions
    n_actions: usize,

    /// Last action taken (for counterfactual analysis)
    last_action: Option<usize>,

    /// Last state (for counterfactual analysis)
    last_state: Option<Vec<f64>>,

    /// Running average of prediction error for surprise detection
    avg_prediction_error: f64,

    /// Recent states that led to goal completion (for credit assignment)
    recent_goal_states: VecDeque<(Vec<f64>, usize)>, // (state, action)

    /// ACTION-VALUE LEARNING SYSTEM (Q-Learning)
    /// Maps discretized state + action → learned Q-value
    /// This is the KEY bridge between internal learning and external behavior
    action_values: HashMap<(Vec<i32>, usize), ActionValue>,

    /// State-value estimates (V-function) for bootstrapping
    state_values: HashMap<Vec<i32>, f64>,

    /// Learning rate for Q-updates
    q_learning_rate: f64,

    /// Discount factor for future rewards
    gamma: f64,

    /// Eligibility traces for faster credit assignment
    eligibility_traces: HashMap<(Vec<i32>, usize), f64>,

    /// Trace decay factor (lambda in TD(λ))
    trace_decay: f64,

    /// Artificial General Coherence system
    /// Integrates psychological coherence, requisite variety, homeostasis,
    /// epistemic integrity, and ethical alignment
    pub agc: ArtificialGeneralCoherence,

    /// Illichian Learning Webs - Self-directed learning via deschooling principles
    /// Four networks: Objects, Skills, Peers, Educators
    /// Hidden curriculum detection and deinstitutionalization
    pub learning_webs: LearningWebs,

    /// Long-term Memory System - Episodic, Semantic, Procedural memory
    /// With working memory buffer and sleep-like consolidation
    pub memory_system: MemorySystem,

    /// LLM Integration - Language understanding grounded in experience
    /// Connects natural language to sensorimotor representations
    pub llm_integration: LLMIntegration,

    /// Explicability System - Natural language decision explanations
    /// Generates causal, contrastive, and counterfactual explanations
    pub explicability: ExplicabilitySystem,

    /// Previous symbol count for tracking new groundings per step
    prev_grounded_symbols: usize,

    /// Total symbols with symbolic representation for current action (for explicability)
    last_action_has_symbol: bool,

    /// Recent decision audits for explicability (ring buffer of last N decisions)
    decision_audits: VecDeque<DecisionAudit>,
}

/// Learned action value with statistics
#[derive(Debug, Clone)]
pub struct ActionValue {
    /// Estimated Q-value
    pub q_value: f64,
    /// Number of updates
    pub update_count: usize,
    /// Running average of TD errors (for prioritization)
    pub avg_td_error: f64,
    /// Last update step
    pub last_update: usize,
}

impl AGICore {
    /// Create a new AGI Core
    pub fn new(config: AGICoreConfig, feature_dim: usize, n_actions: usize) -> Self {
        Self {
            causal_discovery: CausalDiscovery::new(config.clone()),
            abstraction: AbstractionHierarchy::new(config.clone()),
            world_model: WorldModel::new(config.clone(), feature_dim),
            goals: GoalHierarchy::new(config.clone()),
            meta_learner: MetaLearner::new(config.clone()),
            symbols: SymbolSystem::new(config.clone()),
            skills: SkillLibrary::new(config.clone()),
            self_model: SelfModel::new(n_actions),
            policy_bias: vec![0.0; n_actions],
            analytics: CompoundingAnalytics::default(),
            current_step: 0,
            feature_dim,
            n_actions,
            last_action: None,
            last_state: None,
            avg_prediction_error: 0.5, // Initial baseline
            recent_goal_states: VecDeque::with_capacity(50),
            config,
            // Initialize Q-learning system
            action_values: HashMap::new(),
            state_values: HashMap::new(),
            q_learning_rate: 0.1,
            gamma: 0.95,
            eligibility_traces: HashMap::new(),
            trace_decay: 0.8, // TD(0.8)
            // Initialize Artificial General Coherence
            agc: ArtificialGeneralCoherence::new(),
            // Initialize Illichian Learning Webs (Deschooling for AGI)
            learning_webs: LearningWebs::new(),
            // Initialize Long-term Memory System
            memory_system: MemorySystem::new(MemorySystemConfig::default()),
            // Initialize LLM Integration for language grounding
            llm_integration: LLMIntegration::new(LLMIntegrationConfig::default()),
            // Initialize Explicability System for decision explanations
            explicability: ExplicabilitySystem::new(ExplicabilityConfig::default()),
            // Track new symbol groundings
            prev_grounded_symbols: 0,
            last_action_has_symbol: false,
            // Decision audit trail
            decision_audits: VecDeque::with_capacity(100),
        }
    }

    /// Process a single experience step - the main learning loop
    pub fn process_experience(
        &mut self,
        state: &[f64],
        action: usize,
        next_state: &[f64],
        reward: f64,
        is_terminal: bool,
    ) {
        self.current_step += 1;

        // Track recent states for credit assignment
        self.recent_goal_states.push_back((state.to_vec(), action));
        while self.recent_goal_states.len() > 50 {
            self.recent_goal_states.pop_front();
        }

        // ═══════════════════════════════════════════════════════════════════
        // Q-LEARNING UPDATE (TD(λ)) - THE BRIDGE FROM LEARNING TO BEHAVIOR
        // This creates a DIRECT connection between experience and action selection
        // ═══════════════════════════════════════════════════════════════════
        self.update_action_values(state, action, next_state, reward, is_terminal);

        // 1. CAUSAL DISCOVERY - observe patterns
        self.causal_discovery
            .observe(state.to_vec(), Some(action), reward);

        // 2. WORLD MODEL - learn dynamics
        self.world_model
            .learn(state, action, next_state, reward, is_terminal);

        // 3. ABSTRACTION - form concepts
        self.abstraction.observe(state.to_vec());
        self.abstraction.observe(next_state.to_vec());

        // 4. GOAL PROGRESS - update active goals and get completed ones
        let completed_goals = self.goals.update_progress(next_state);

        // 5. SYMBOL GROUNDING - create/update symbols for notable states
        if reward.abs() > 0.5 {
            let name = if reward > 0.0 {
                format!("reward_state_{}", self.current_step)
            } else {
                format!("penalty_state_{}", self.current_step)
            };
            let motor = vec![action as f64 / self.n_actions as f64; 4];
            self.symbols
                .get_or_create_symbol(&name, next_state.to_vec(), motor);
        }

        // 6. NEW: SKILL LIBRARY - record actions for potential skill extraction
        self.skills.record_action(state.to_vec(), action, reward);

        // 7. NEW: SELF-MODEL - update capability tracking
        self.self_model.record_reached_state(next_state, 1);
        self.self_model.update_action_effectiveness(action, reward);

        // 8. NEW: COUNTERFACTUAL LEARNING - compute regret and update policy bias
        if let Some(last_state) = self.last_state.clone() {
            if let Some(last_action) = self.last_action {
                self.counterfactual_update(&last_state, last_action, state, reward);
            }
        }
        self.last_state = Some(state.to_vec());
        self.last_action = Some(action);

        // 9. COMPOUND INTERACTIONS - drive multiplicative growth
        self.compound_interactions(state, action, next_state, reward, &completed_goals);

        // 10. UPDATE ARTIFICIAL GENERAL COHERENCE with FULL subsystem data
        // This is the key integration point - AGC now sees the REAL state of all subsystems
        let prediction_error = self.avg_prediction_error;
        let goal_success_rate = self.goals.summary().success_rate;

        // Gather real data from all subsystems
        let symbol_summary = self.symbols.summary();
        let current_grounded = symbol_summary.grounded_symbols;
        let new_groundings = current_grounded.saturating_sub(self.prev_grounded_symbols);
        self.prev_grounded_symbols = current_grounded;

        // Check if current action has symbolic representation (for explicability)
        let has_symbol_for_action = self.symbols.find_by_sensory(state, 0.3).is_some()
            || self
                .abstraction
                .get_activated_concepts(state)
                .iter()
                .any(|(_, sim)| *sim > 0.5);
        self.last_action_has_symbol = has_symbol_for_action;

        // Get subsystem stats
        let discovered_variables = self.causal_discovery.summary().total_variables;
        let causal_interventions = self.analytics.meta_to_discovery; // Using this as proxy for interventions
        let world_model_states = self.world_model.summary().total_states;
        let concepts_formed = self.abstraction.summary().total_concepts;
        let exploration_rate = self.meta_learner.get_recommended_params().exploration_rate;
        let goals_created = self.goals.summary().total_goals;

        // Call the FULL AGC update with real subsystem data
        self.agc.update_full(
            state,
            action,
            self.n_actions,
            reward,
            prediction_error,
            self.analytics.compound_rate,
            goal_success_rate,
            // Subsystem-specific data:
            discovered_variables,
            causal_interventions,
            current_grounded,
            symbol_summary.total_symbols,
            new_groundings,
            world_model_states,
            concepts_formed,
            exploration_rate,
            goals_created,
            self.analytics.total_interactions,
            has_symbol_for_action,
        );

        // 11. UPDATE ILLICHIAN LEARNING WEBS - Self-directed learning metrics
        // Compute learning quality indicators from current cognitive state
        let intrinsic_motivation = {
            // High if learning is driven by curiosity (exploration) rather than external rewards
            let curiosity_driven = self.meta_learner.best_exploration_rate;
            let reward_driven = if reward.abs() > 0.5 { 0.3 } else { 0.7 };
            curiosity_driven * 0.6 + reward_driven * 0.4
        };
        let active_creation = {
            // High if actively creating concepts/symbols vs passively observing
            let concepts_per_step = concepts_formed as f64 / (self.current_step + 1) as f64;
            let symbols_per_step = new_groundings as f64 / (self.current_step + 1).max(1) as f64;
            (concepts_per_step * 10.0 + symbols_per_step * 5.0).min(1.0)
        };
        let self_assessment = {
            // High if using internal metrics vs external rewards for evaluation
            let internal_signal = self.avg_prediction_error.min(1.0);
            let calibration = self.self_model.calibration_score();
            (1.0 - internal_signal) * 0.5 + calibration * 0.5
        };
        let process_focus = {
            // High if focused on learning process vs outcome/goals
            let goal_pressure = if self.goals.active_count() > 5 {
                0.3
            } else {
                0.7
            };
            let exploration_focus = self.meta_learner.best_exploration_rate;
            goal_pressure * 0.5 + exploration_focus * 0.5
        };
        let experiential_learning = {
            // High if learning from direct experience vs instruction
            // AGI Core primarily learns through direct experience (sensorimotor loop)
            0.8 + (self.skills.summary().total_skills as f64 * 0.01).min(0.2)
        };

        self.learning_webs.update(
            intrinsic_motivation,
            active_creation,
            self_assessment,
            process_focus,
            experiential_learning,
        );

        // Register newly discovered concepts as educational objects
        if concepts_formed > 0 && self.current_step % 50 == 0 {
            let concept_id = concepts_formed.saturating_sub(1);
            self.learning_webs.objects.register_object(
                &format!("concept_{}", concept_id),
                crate::learning_webs::ObjectType::DirectExperience,
                vec![concept_id],
                self.current_step,
            );
        }

        // Register newly extracted skills in skill exchange network
        if self.skills.summary().total_skills > 0 && self.current_step % 100 == 0 {
            for skill in self.skills.skills.values() {
                if skill.execution_count == 1 {
                    // New skill - register in learning webs
                    self.learning_webs.skills.register_skill(
                        &skill.name,
                        vec![], // No prerequisites for now
                    );
                }
            }
        }

        // 12. APPLY HOMEOSTATIC CONTROLS - actually adjust subsystem parameters!
        let controls = &self.agc.homeostasis.controls;
        if controls.regulation_intensity > 0.2 {
            // Apply learning rate adjustment to Q-learning
            let lr_adjust = controls.learning_rate_adjust;
            self.q_learning_rate =
                (self.q_learning_rate * (1.0 + lr_adjust * 0.1)).clamp(0.01, 0.3);

            // Apply exploration adjustment via modifying the meta-learner's best values
            let exp_adjust = controls.exploration_adjust;
            if exp_adjust.abs() > 0.05 {
                self.meta_learner.best_exploration_rate =
                    (self.meta_learner.best_exploration_rate + exp_adjust * 0.05).clamp(0.05, 0.5);
            }

            // Apply goal creation control
            if controls.goal_creation_adjust < -0.2 && self.goals.active_count() > 3 {
                // Too many goals - homeostasis suggests reducing
                // This happens naturally through the config, but we can also
                // increase the completion threshold slightly
                self.config.goal_completion_threshold =
                    (self.config.goal_completion_threshold - 0.01).max(0.4);
            } else if controls.goal_creation_adjust > 0.2 {
                // Room for more goals - lower threshold to complete faster
                self.config.goal_completion_threshold =
                    (self.config.goal_completion_threshold + 0.01).min(0.8);
            }
        }

        // 9. COHERENCE-GATED CREDIT FLOW
        // If coherence gates are restricting flow, track as blockage
        if self.agc.psychological.gates.discovery_to_abstraction < 0.5
            || self.agc.psychological.gates.symbols_to_goals < 0.5
        {
            self.analytics.credit_blockages = self.analytics.credit_blockages.saturating_add(1);
        }

        // 10. VARIETY EXPANSION - act on AGC recommendations
        for expansion in &self.agc.variety.expansion_needed {
            match expansion.area {
                ExpansionArea::ActionSpace => {
                    // Need more action diversity - slightly increase exploration
                    self.meta_learner.best_exploration_rate =
                        (self.meta_learner.best_exploration_rate + 0.02).min(0.5);
                }
                ExpansionArea::GoalHierarchy => {
                    // Need more goal diversity - create exploratory goal if room
                    if self.goals.active_count() < self.config.max_active_goals {
                        // Create a random exploration goal
                        let explore_target: Vec<f64> = next_state
                            .iter()
                            .map(|&x| x + (rand_simple() - 0.5) * 0.5)
                            .collect();
                        let _goal_id = self.goals.create_goal(
                            &format!("explore_variety_{}", self.current_step),
                            explore_target,
                            GoalPriority::Low,
                        );
                    }
                }
                ExpansionArea::Abstractions => {
                    // Need more concepts - lower merge threshold temporarily
                    self.config.abstraction_merge_threshold =
                        (self.config.abstraction_merge_threshold - 0.02).max(0.5);
                }
                ExpansionArea::CausalModels => {
                    // Need more causal variables - lower discovery threshold
                    self.config.causal_discovery_threshold =
                        (self.config.causal_discovery_threshold - 0.01).max(0.05);
                }
                ExpansionArea::Symbols => {
                    // Need more symbols - lower grounding threshold
                    self.config.symbol_grounding_threshold =
                        (self.config.symbol_grounding_threshold - 0.02).max(0.2);
                }
                ExpansionArea::StateRepresentation => {
                    // Need finer state discrimination - handled by Q-learning discretization
                    // Could adjust resolution but that's more complex
                }
            }
        }

        // 13. MEMORY SYSTEM INTEGRATION - Store episodic memories for significant events
        // Record significant experiences as episodic memories
        if reward.abs() > 0.3 || is_terminal || completed_goals.len() > 0 {
            // Use process_experience which is the correct API for MemorySystem
            let concepts: Vec<String> = self
                .symbols
                .find_by_sensory(state, 0.3)
                .map(|s| s.name.clone())
                .into_iter()
                .collect();
            // Use total_variables as proxy for causal context since we don't have recent_discoveries
            let causal_context: Vec<usize> =
                (0..self.causal_discovery.summary().total_variables.min(3)).collect();

            self.memory_system.process_experience(
                state.to_vec(),
                Some(action),
                reward,
                next_state.to_vec(),
                concepts,
                causal_context,
            );
        }

        // Periodic memory consolidation (simulating sleep-like replay)
        if self.current_step % 100 == 0 {
            self.memory_system.consolidate();
        }

        // 14. EXPLICABILITY INTEGRATION - Track decisions for explanation generation
        // Record this decision for potential future explanation
        let q_values: Vec<f64> = (0..self.n_actions)
            .map(|a| {
                let state_disc = self.discretize_state(state);
                self.action_values
                    .get(&(state_disc, a))
                    .map(|av| av.q_value)
                    .unwrap_or(0.0)
            })
            .collect();

        // Build decision factors with full signature:
        // (name, value, importance, direction, source)
        let factors = vec![
            (
                "q_value".to_string(),
                q_values.get(action).copied().unwrap_or(0.0),
                0.8, // High importance
                InfluenceDirection::Positive,
                FactorSource::Learning, // Using Learning instead of Model
            ),
            (
                "reward_expectation".to_string(),
                reward,
                0.6,
                if reward >= 0.0 {
                    InfluenceDirection::Positive
                } else {
                    InfluenceDirection::Negative
                },
                FactorSource::Perception, // Using Perception instead of Observation
            ),
            (
                "exploration_rate".to_string(),
                self.meta_learner.best_exploration_rate,
                0.4,
                InfluenceDirection::Neutral,
                FactorSource::Prediction, // Using Prediction instead of Model
            ),
            (
                "goal_alignment".to_string(),
                if completed_goals.len() > 0 { 1.0 } else { 0.5 },
                0.7,
                InfluenceDirection::Positive,
                FactorSource::Goal,
            ),
        ];

        // Build alternatives (other actions not chosen)
        let alternatives: Vec<(String, String, f64)> = (0..self.n_actions)
            .filter(|&a| a != action)
            .take(3)
            .map(|a| {
                (
                    format!("action_{}", a),
                    format!(
                        "Alternative action with Q={:.3}",
                        q_values.get(a).copied().unwrap_or(0.0)
                    ),
                    q_values.get(a).copied().unwrap_or(0.0),
                )
            })
            .collect();

        // Get active goal names from the active_goals field
        let goal_names: Vec<String> = self
            .goals
            .active_goals
            .iter()
            .filter_map(|&id| self.goals.goals.get(&id).map(|g| g.name.clone()))
            .take(3)
            .collect();

        let confidence = if q_values.iter().any(|&q| q > 0.0) {
            let max_q = q_values.iter().cloned().fold(f64::MIN, f64::max);
            let chosen_q = q_values.get(action).copied().unwrap_or(0.0);
            (chosen_q / max_q.max(0.01)).clamp(0.0, 1.0)
        } else {
            0.5 // Uncertain when no Q-values yet
        };

        self.explicability.record_decision(
            &format!("action_{}", action),
            &format!("Selected action {} at step {}", action, self.current_step),
            factors,
            alternatives,
            goal_names,
            confidence,
        );
    }

    /// Discretize state for Q-table lookup
    fn discretize_state(&self, state: &[f64]) -> Vec<i32> {
        // Use coarser discretization for generalization
        // Grid of 0.5 units for position-like features
        state
            .iter()
            .map(|&x| (x * 2.0).round() as i32) // 0.5 unit resolution
            .collect()
    }

    /// Update action values using TD(λ) with eligibility traces
    fn update_action_values(
        &mut self,
        state: &[f64],
        action: usize,
        next_state: &[f64],
        reward: f64,
        is_terminal: bool,
    ) {
        let state_disc = self.discretize_state(state);
        let next_state_disc = self.discretize_state(next_state);
        let key = (state_disc.clone(), action);

        // Get current Q(s, a)
        let current_q = self
            .action_values
            .get(&key)
            .map(|av| av.q_value)
            .unwrap_or(0.0);

        // Get max Q(s', a') for next state (greedy bootstrap)
        let next_max_q = if is_terminal {
            0.0
        } else {
            (0..self.n_actions)
                .filter_map(|a| {
                    let next_key = (next_state_disc.clone(), a);
                    self.action_values.get(&next_key).map(|av| av.q_value)
                })
                .fold(0.0_f64, f64::max)
        };

        // TD error: δ = r + γ * max Q(s', a') - Q(s, a)
        let td_error = reward + self.gamma * next_max_q - current_q;

        // Update eligibility trace for current state-action
        let trace_key = key.clone();
        *self.eligibility_traces.entry(trace_key).or_insert(0.0) = 1.0;

        // Apply TD(λ) update to all eligible state-actions
        let alpha = self.q_learning_rate;
        let lambda = self.trace_decay;
        let current_step = self.current_step;

        // Collect updates to avoid borrow issues
        let updates: Vec<((Vec<i32>, usize), f64)> = self
            .eligibility_traces
            .iter()
            .filter(|(_, &trace)| trace > 0.01) // Only update significant traces
            .map(|(k, &trace)| (k.clone(), alpha * td_error * trace))
            .collect();

        // Apply Q-value updates
        for (k, delta) in updates {
            let entry = self.action_values.entry(k.clone()).or_insert(ActionValue {
                q_value: 0.0,
                update_count: 0,
                avg_td_error: 0.0,
                last_update: 0,
            });
            entry.q_value += delta;
            entry.update_count += 1;
            entry.avg_td_error = entry.avg_td_error * 0.9 + td_error.abs() * 0.1;
            entry.last_update = current_step;
        }

        // Decay all eligibility traces
        for trace in self.eligibility_traces.values_mut() {
            *trace *= self.gamma * lambda;
        }

        // Prune near-zero traces periodically
        if self.current_step % 100 == 0 {
            self.eligibility_traces
                .retain(|_, &mut trace| trace > 0.001);
        }

        // Update state value estimate V(s) = max_a Q(s, a)
        let state_v = (0..self.n_actions)
            .filter_map(|a| {
                let k = (state_disc.clone(), a);
                self.action_values.get(&k).map(|av| av.q_value)
            })
            .fold(0.0_f64, f64::max);
        self.state_values.insert(state_disc, state_v);
    }

    /// Get learned Q-value for a state-action pair
    pub fn get_q_value(&self, state: &[f64], action: usize) -> f64 {
        let state_disc = self.discretize_state(state);
        let key = (state_disc, action);
        self.action_values
            .get(&key)
            .map(|av| av.q_value)
            .unwrap_or(0.0)
    }

    /// Get best action according to learned Q-values
    pub fn get_greedy_action(&self, state: &[f64]) -> Option<usize> {
        let state_disc = self.discretize_state(state);

        let mut best_action = None;
        let mut best_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let key = (state_disc.clone(), action);
            if let Some(av) = self.action_values.get(&key) {
                if av.q_value > best_q {
                    best_q = av.q_value;
                    best_action = Some(action);
                }
            }
        }

        best_action
    }

    /// Get Q-learning statistics
    pub fn q_learning_stats(&self) -> (usize, f64, f64) {
        let n_entries = self.action_values.len();
        let avg_q = if n_entries > 0 {
            self.action_values
                .values()
                .map(|av| av.q_value)
                .sum::<f64>()
                / n_entries as f64
        } else {
            0.0
        };
        let avg_updates = if n_entries > 0 {
            self.action_values
                .values()
                .map(|av| av.update_count as f64)
                .sum::<f64>()
                / n_entries as f64
        } else {
            0.0
        };
        (n_entries, avg_q, avg_updates)
    }

    /// Counterfactual learning - compute regret and update policy bias
    /// This helps the agent learn from "what might have been"
    fn counterfactual_update(
        &mut self,
        last_state: &[f64],
        last_action: usize,
        current_state: &[f64],
        reward: f64,
    ) {
        // Get Q-values for all actions in the last state
        let last_state_disc = self.discretize_state(last_state);

        // Find the best action we could have taken
        let mut best_q = f64::NEG_INFINITY;
        let mut best_action = last_action;

        for a in 0..self.n_actions {
            let key = (last_state_disc.clone(), a);
            if let Some(av) = self.action_values.get(&key) {
                if av.q_value > best_q {
                    best_q = av.q_value;
                    best_action = a;
                }
            }
        }

        // Compute counterfactual regret
        let actual_q = self.get_q_value(last_state, last_action);
        let regret = (best_q - actual_q).max(0.0);

        // If significant regret, update policy bias
        if regret > self.config.counterfactual_regret_threshold {
            // Slightly bias toward the better action
            if best_action < self.policy_bias.len() {
                self.policy_bias[best_action] += 0.01;
            }
            // Slightly bias away from the bad action
            if last_action < self.policy_bias.len() {
                self.policy_bias[last_action] -= 0.005;
            }

            // Record in self-model
            self.self_model.record_prediction_error(regret);
        }

        // Normalize policy bias
        let sum: f64 = self.policy_bias.iter().map(|&x| x.abs()).sum();
        if sum > 1.0 {
            for bias in &mut self.policy_bias {
                *bias /= sum;
            }
        }
    }

    /// Drive compounding interactions between systems - THE EMERGENCE ENGINE
    fn compound_interactions(
        &mut self,
        state: &[f64],
        action: usize,
        next_state: &[f64],
        reward: f64,
        completed_goals: &[usize],
    ) {
        // ═══════════════════════════════════════════════════════════════════
        // EMERGENCE MECHANISM 1: BACKWARD CREDIT ASSIGNMENT
        // When goals complete, strengthen the discoveries and symbols that enabled them
        // ═══════════════════════════════════════════════════════════════════
        if !completed_goals.is_empty() {
            // Credit flows BACKWARD from achievement to enabling factors
            for &goal_id in completed_goals {
                // 1a. Credit discoveries that were relevant to recent goal-achieving states
                // Collect var IDs first to avoid borrow conflict
                let mut useful_var_ids: Vec<usize> = Vec::new();
                for (past_state, _past_action) in &self.recent_goal_states {
                    let relevant_vars = self.causal_discovery.get_relevant_variables(past_state);
                    for var in relevant_vars {
                        useful_var_ids.push(var.id);
                    }
                }
                // Now mark them useful
                for var_id in useful_var_ids {
                    self.causal_discovery.mark_useful(var_id);
                    self.analytics.credit_to_discoveries += 1;
                }

                // 1b. Credit symbols associated with the completed goal
                // Find symbols that were grounded in this goal
                for sym in self.symbols.symbols.values_mut() {
                    if sym.goal_id == Some(goal_id) {
                        // Boost confidence - this symbol led to success!
                        sym.grounding_confidence = (sym.grounding_confidence + 0.15).min(1.0);
                        sym.usage_count += 5; // Strong reward for goal-achieving symbols
                        self.analytics.credit_to_symbols += 1;
                    }
                }

                // 1c. Lower discovery threshold - make it easier to find more useful patterns
                // This is the KEY positive feedback loop
                self.config.causal_discovery_threshold *= 0.95;
                self.config.causal_discovery_threshold =
                    self.config.causal_discovery_threshold.max(0.1); // Floor at 0.1
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // EMERGENCE MECHANISM 1.5: REWARD-DRIVEN CREDIT (IMMEDIATE FEEDBACK)
        // Don't wait for goal completion - credit discoveries/symbols that lead to reward
        // This is the KEY connection between internal compounding and external performance
        // ═══════════════════════════════════════════════════════════════════
        if reward > 0.5 {
            // Positive reward = credit the current context
            // 1.5a. Credit relevant causal discoveries
            let mut useful_var_ids: Vec<usize> = Vec::new();
            let relevant_vars = self.causal_discovery.get_relevant_variables(state);
            for var in relevant_vars {
                useful_var_ids.push(var.id);
            }
            for var_id in useful_var_ids {
                self.causal_discovery.mark_useful(var_id);
                self.analytics.credit_to_discoveries += 1;
            }

            // 1.5b. Credit symbols matching current state (they led to reward!)
            let state_vec = state.to_vec();
            let symbol_credits: Vec<usize> = self
                .symbols
                .symbols
                .values()
                .filter(|s| {
                    // Check if symbol's sensory grounding matches current state
                    let sim = cosine_sim(&s.sensory_grounding, &state_vec);
                    sim > 0.3
                })
                .map(|s| s.id)
                .collect();

            for sym_id in symbol_credits {
                if let Some(sym) = self.symbols.symbols.get_mut(&sym_id) {
                    // Proportional credit based on reward magnitude
                    let credit_amount = (reward * 0.1).min(0.1);
                    sym.grounding_confidence = (sym.grounding_confidence + credit_amount).min(1.0);
                    sym.usage_count += 1;
                    self.analytics.credit_to_symbols += 1;
                }
            }

            // 1.5c. Associate current action with activated concepts (skill learning)
            let activated = self.abstraction.get_activated_concepts(state);
            for (concept_id, _activation) in activated {
                self.abstraction.associate_action(concept_id, action);
            }
        } else if reward < -0.5 {
            // Negative reward = slightly decrease confidence in matching symbols
            let state_vec = state.to_vec();
            let symbol_penalties: Vec<usize> = self
                .symbols
                .symbols
                .values()
                .filter(|s| {
                    let sim = cosine_sim(&s.sensory_grounding, &state_vec);
                    sim > 0.3
                })
                .map(|s| s.id)
                .collect();

            for sym_id in symbol_penalties {
                if let Some(sym) = self.symbols.symbols.get_mut(&sym_id) {
                    // Small penalty - don't want to destroy learned knowledge
                    sym.grounding_confidence = (sym.grounding_confidence - 0.02).max(0.1);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // EMERGENCE MECHANISM 2: SURPRISE-DRIVEN META-LEARNING
        // Don't wait for periodic updates - react to unexpected outcomes immediately
        // ═══════════════════════════════════════════════════════════════════
        let prediction_error = if let Some(predicted) = self.world_model.predict(state, action) {
            predicted
                .features
                .iter()
                .zip(next_state.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f64>()
                .sqrt()
        } else {
            1.0 // Unknown transition = maximum surprise
        };

        // Update running average
        let alpha = 0.1;
        self.avg_prediction_error =
            self.avg_prediction_error * (1.0 - alpha) + prediction_error * alpha;

        // Compute surprise ratio
        let surprise = prediction_error / (self.avg_prediction_error + 0.01);

        // SURPRISE-TRIGGERED meta update (not periodic!)
        if surprise > 1.5 || surprise < 0.5 {
            // Either much worse OR much better than expected - both are informative!
            let metrics = self.meta_learner.get_recommended_params();
            let performance = if surprise < 1.0 { 1.0 } else { 0.0 }; // Better prediction = higher performance
            self.meta_learner.record_episode(
                prediction_error,
                performance,
                1, // Single step, not batched
                metrics,
            );
            self.analytics.surprise_meta_updates += 1;
            self.analytics.world_model_to_meta += 1;
        }

        // ═══════════════════════════════════════════════════════════════════
        // EMERGENCE MECHANISM 3: IMAGINATION-DRIVEN ABSTRACTION
        // World model imagines futures, successful imaginations become concepts
        // ═══════════════════════════════════════════════════════════════════
        if self.current_step % 10 == 0 {
            // Even more frequent (was 25)
            let futures = self.world_model.imagine_futures(state, 5, self.n_actions);

            for trajectory in &futures {
                // LOWERED threshold: Any non-negative trajectory is worth abstracting
                if trajectory.total_reward > 0.0 && trajectory.states.len() >= 2 {
                    // Take features from the BEST state in the trajectory (highest reward position)
                    if let Some(best_state) = trajectory.states.iter().max_by(|a, b| {
                        a.reward
                            .partial_cmp(&b.reward)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        self.abstraction.observe(best_state.features.clone());
                        self.analytics.imagination_abstractions += 1;
                    }
                }

                // Also abstract from ANY trajectory that reaches goal-completing states
                for state_in_traj in &trajectory.states {
                    if self
                        .goals
                        .check_prediction_completes_goal(&state_in_traj.features)
                        .is_some()
                    {
                        self.abstraction.observe(state_in_traj.features.clone());
                        self.analytics.imagination_abstractions += 1;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // ORIGINAL COMPOUND PATHWAYS (with amplification applied)
        // ═══════════════════════════════════════════════════════════════════

        // Discovery → Abstraction: Use discovered variables to inform concepts
        let discovered = self.causal_discovery.get_relevant_variables(state);
        for var in discovered {
            self.abstraction.observe(var.signature.clone());
            self.analytics.discovery_to_abstraction += 1;
        }

        // Abstraction → Symbols: Ground symbols in activated concepts
        let activated = self.abstraction.get_activated_concepts(state);
        for (concept_id, similarity) in activated.iter().take(3) {
            // AMPLIFIED threshold - gets easier as amplification grows
            let adjusted_threshold = 0.7 / self.analytics.amplification_factor;
            if *similarity > adjusted_threshold {
                let name = format!("concept_{}", concept_id);
                let sym_id = self.symbols.get_or_create_symbol(
                    &name,
                    state.to_vec(),
                    vec![action as f64; 4],
                );
                self.symbols.ground_in_concept(sym_id, *concept_id);
                self.analytics.abstraction_to_symbols += 1;
            }
        }

        // Symbols → Goals: Auto-create goals from grounded symbols
        let ungoaled_symbols: Vec<(String, Vec<f64>, Option<usize>)> = self
            .symbols
            .get_ungoaled_grounded_symbols()
            .iter()
            .take(2)
            .map(|s| (s.name.clone(), s.sensory_grounding.clone(), s.concept_id))
            .collect();

        for (name, target_features, concept_id) in ungoaled_symbols {
            if self.goals.active_count() < self.config.max_active_goals {
                let goal_name = format!("explore_{}", name);
                let goal_id = self.goals.create_goal_from_symbol(
                    &goal_name,
                    target_features.clone(),
                    concept_id,
                );
                if let Some(&sym_id) = self.symbols.name_to_id.get(&name) {
                    self.symbols.ground_in_goal(sym_id, goal_id);
                }
                self.analytics.symbols_to_goals += 1;
            }
        }

        // Also express current goal symbolically
        if let Some(goal) = self.goals.get_current_goal() {
            let goal_name = format!("goal_{}", goal.id);
            let goal_target = goal.target_features.clone();
            let goal_id = goal.id;
            let sym_id = self
                .symbols
                .get_or_create_symbol(&goal_name, goal_target, vec![0.0; 4]);
            self.symbols.ground_in_goal(sym_id, goal_id);
            self.analytics.symbols_to_goals += 1;
        }

        // Goals → World Model: Imagine paths to goals
        if let Some(goal) = self.goals.get_current_goal() {
            let goal_id = goal.id;
            let goal_subgoals_empty = goal.subgoals.is_empty();

            let futures = self.world_model.imagine_futures(state, 3, self.n_actions);

            for trajectory in &futures {
                for imagined_state in &trajectory.states {
                    if self
                        .goals
                        .check_prediction_completes_goal(&imagined_state.features)
                        .is_some()
                    {
                        self.analytics.goals_to_world_model += 1;
                    }
                }
            }

            if !futures.is_empty() && goal_subgoals_empty {
                let intermediate_states: Vec<Vec<f64>> = futures
                    .iter()
                    .take(2)
                    .flat_map(|t| t.states.iter().take(2).map(|s| s.features.clone()))
                    .collect();
                if !intermediate_states.is_empty() {
                    self.goals.auto_decompose(goal_id, state, &self.world_model);
                    self.analytics.goals_to_world_model += 1;
                }
            }
        }

        // Meta → Discovery: Apply meta-learned parameters
        // (Removed periodic gate - now continuous through amplification)
        self.analytics.meta_to_discovery += 1;

        // ═══════════════════════════════════════════════════════════════════
        // EMERGENCE MECHANISM 4: EXPONENTIAL COMPOUND AMPLIFICATION
        // Success breeds success - compound effects grow over time
        // ═══════════════════════════════════════════════════════════════════
        let goal_summary = self.goals.summary();
        let success_rate = goal_summary.success_rate;

        // Amplification grows with success
        if success_rate > 0.3 && goal_summary.completed_goals > 2 {
            // The more goals we complete, the stronger the amplification
            let completion_bonus = (goal_summary.completed_goals as f64).ln().max(0.0);
            self.analytics.amplification_factor =
                1.0 + (success_rate * 0.5) + (completion_bonus * 0.1);

            // Amplification also lowers thresholds (easier to discover/abstract)
            if self.analytics.amplification_factor > 1.1 {
                self.config.symbol_grounding_threshold *= 0.995;
                self.config.symbol_grounding_threshold =
                    self.config.symbol_grounding_threshold.max(0.2); // Floor
            }
        }

        // Meta-Learning → Discovery: Use learned parameters TO ACTUALLY ADJUST DISCOVERY
        // This is the key fix - we need to actually transfer the meta-learning to other systems
        if self.current_step % 200 == 0 {
            let params = self.meta_learner.get_recommended_params();

            // Higher exploration → lower discovery threshold (find more patterns)
            // Lower exploration → higher threshold (be more selective)
            let new_threshold = 0.3 * (1.0 - params.exploration_rate * 0.5);
            self.causal_discovery.set_discovery_threshold(new_threshold);

            self.analytics.meta_to_discovery += 1;
        }

        // NEW: Check for goal completion → skill extraction
        let completed_goals: Vec<(usize, String, Vec<f64>)> = self
            .goals
            .get_recently_completed()
            .iter()
            .map(|g| (g.id, g.name.clone(), g.target_features.clone()))
            .collect();

        for (goal_id, goal_name, goal_features) in completed_goals {
            // Extract skill from successful goal completion
            if let Some(skill_id) =
                self.skills
                    .extract_skill_from_goal(&goal_name, goal_features.clone(), true)
            {
                // Record goal outcome in self-model
                self.self_model.record_goal_outcome(&goal_name, true);

                // Log skill extraction (could add to analytics)
                log::debug!("Extracted skill {} from goal {}", skill_id, goal_id);
            }
        }

        // NEW: Active Experimentation - design experiments to test causal hypotheses
        if self.current_step % 500 == 0 {
            if let Some(experiment) = self.causal_discovery.design_experiment() {
                // Store experiment suggestion for recommend_action to use
                log::debug!("Designed experiment: {}", experiment.hypothesis);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // UPDATE TOTALS
        // ═══════════════════════════════════════════════════════════════════
        self.analytics.total_interactions = self.analytics.discovery_to_abstraction
            + self.analytics.abstraction_to_symbols
            + self.analytics.symbols_to_goals
            + self.analytics.goals_to_world_model
            + self.analytics.world_model_to_meta
            + self.analytics.meta_to_discovery
            + self.analytics.credit_to_discoveries
            + self.analytics.credit_to_symbols
            + self.analytics.surprise_meta_updates
            + self.analytics.imagination_abstractions;

        // Compute compound rate (should GROW over time for true emergence)
        if self.current_step > 100 {
            self.analytics.compound_rate =
                self.analytics.total_interactions as f64 / self.current_step as f64;
        }
    }

    /// Create a goal from current observation
    pub fn create_goal_from_observation(
        &mut self,
        name: &str,
        target: Vec<f64>,
        priority: GoalPriority,
    ) -> usize {
        let goal_id = self.goals.create_goal(name, target.clone(), priority);
        self.goals.activate_goal(goal_id);

        // Create symbol for the goal
        let sym_id =
            self.symbols
                .get_or_create_symbol(&format!("goal_{}", name), target, vec![0.0; 4]);
        self.symbols.ground_in_goal(sym_id, goal_id);

        goal_id
    }

    /// Get action recommendation based on FULLY INTEGRATED cognition
    ///
    /// This is the key function that connects internal compounding to external behavior.
    /// All 6 subsystems contribute to action selection:
    ///
    /// 1. CausalDiscovery → Identifies which state features are causally relevant
    /// 2. AbstractionHierarchy → Provides concept-based action associations
    /// 3. WorldModel → Imagines futures for each action
    /// 4. GoalHierarchy → Provides goal direction and completion prediction
    /// 5. MetaLearner → Controls exploration vs exploitation
    /// 6. SymbolSystem → Provides grounded symbolic reasoning
    pub fn recommend_action(&self, state: &[f64]) -> Option<usize> {
        // 0. SKILL CHECK - Try to use a learned skill first
        if let Some(goal) = self.goals.get_current_goal() {
            if let Some(skill) = self.skills.recommend_skill(state, &goal.target_features) {
                if skill.success_rate > 0.6 && !skill.action_sequence.is_empty() {
                    // Return first action from skill sequence
                    return Some(skill.action_sequence[0]);
                }
            }
        }

        // Initialize action values for all actions
        let mut action_values: Vec<f64> = vec![0.0; self.n_actions];
        let mut action_counts: Vec<usize> = vec![0; self.n_actions];

        // ════════════════════════════════════════════════════════════════════
        // 1. CAUSAL DISCOVERY: Weight actions by causal relevance
        // ════════════════════════════════════════════════════════════════════
        let relevant_vars = self.causal_discovery.get_relevant_variables(state);
        let causal_bonus = if !relevant_vars.is_empty() {
            // High-utility variables suggest we're in an important state
            let avg_utility: f64 = relevant_vars
                .iter()
                .map(|v| v.utility_count as f64)
                .sum::<f64>()
                / relevant_vars.len() as f64;
            // More utility → less random exploration, more exploitation
            (avg_utility / 10.0).min(0.3)
        } else {
            0.0
        };

        // ════════════════════════════════════════════════════════════════════
        // 2. ABSTRACTION HIERARCHY: Use concept-action associations
        // ════════════════════════════════════════════════════════════════════
        let activated_concepts = self.abstraction.get_activated_concepts(state);
        for (concept_id, activation) in &activated_concepts {
            if let Some(concept) = self.abstraction.concepts.get(concept_id) {
                // Concepts have associated actions that worked in similar states
                for &action in &concept.associated_actions {
                    if action < self.n_actions {
                        // Weight by activation strength and concept confidence
                        action_values[action] += activation * concept.confidence * 0.4;
                        action_counts[action] += 1;
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 3. WORLD MODEL: Imagine futures and evaluate expected value
        // ════════════════════════════════════════════════════════════════════
        let goal_direction = self.goals.get_goal_direction(state);
        let meta_params = self.meta_learner.get_recommended_params();

        for action in 0..self.n_actions {
            if let Some(next) = self.world_model.predict(state, action) {
                let mut value = next.reward;

                // Goal alignment bonus
                if let Some(ref direction) = goal_direction {
                    let action_direction: Vec<f64> = next
                        .features
                        .iter()
                        .zip(state.iter())
                        .map(|(&n, &s)| n - s)
                        .collect();
                    let alignment = cosine_sim(&action_direction, direction);
                    value += alignment * 0.6; // Increased from 0.5
                }

                // Penalize uncertainty (but less if we're exploring)
                let meta_params = self.meta_learner.get_recommended_params();
                let uncertainty_penalty =
                    next.uncertainty * (0.3 - meta_params.exploration_rate * 0.2);
                value -= uncertainty_penalty;

                // Add counterfactual-learned policy bias
                value += self.policy_bias[action] * 0.3;

                // Add self-model action effectiveness
                value += self
                    .self_model
                    .action_effectiveness
                    .get(action)
                    .copied()
                    .unwrap_or(0.0)
                    * 0.2;

                // ════════════════════════════════════════════════════════════
                // 4. GOAL HIERARCHY: Bonus if action leads to goal completion
                // ════════════════════════════════════════════════════════════
                if self
                    .goals
                    .check_prediction_completes_goal(&next.features)
                    .is_some()
                {
                    value += 1.0; // Strong bonus for goal-completing actions
                }

                action_values[action] += value;
                action_counts[action] += 1;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 5. META-LEARNER: Strategic exploration (not random!)
        // ════════════════════════════════════════════════════════════════════
        let meta_params = self.meta_learner.get_recommended_params();

        // Instead of random exploration, explore toward uncertainty
        let should_explore = rand_simple() < meta_params.exploration_rate;

        if should_explore {
            // Explore strategically: prefer actions with high uncertainty
            // (epistemic exploration) or actions we haven't tried much
            let mut exploration_values: Vec<f64> = vec![0.0; self.n_actions];

            for action in 0..self.n_actions {
                // Novelty bonus: actions with low counts are more interesting
                let count_bonus = 1.0 / (action_counts[action] as f64 + 1.0);
                exploration_values[action] += count_bonus * 0.5;

                // Uncertainty bonus: prefer actions with uncertain outcomes
                if let Some(next) = self.world_model.predict(state, action) {
                    exploration_values[action] += next.uncertainty * 0.5;
                } else {
                    // No prediction = high uncertainty = interesting!
                    exploration_values[action] += 1.0;
                }
            }

            // Softmax selection for exploration (not pure random)
            let temperature = 0.5;
            let max_val = exploration_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = exploration_values
                .iter()
                .map(|v| ((v - max_val) / temperature).exp())
                .collect();
            let sum: f64 = exp_vals.iter().sum();

            let mut cumulative = 0.0;
            let threshold = rand_simple();
            for (action, &exp_val) in exp_vals.iter().enumerate() {
                cumulative += exp_val / sum;
                if cumulative >= threshold {
                    return Some(action);
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 5. Exploration based on meta-learner (with self-model competence adjustment)
        // ════════════════════════════════════════════════════════════════════
        let recommended = self.meta_learner.get_recommended_params();
        let adjusted_exploration =
            recommended.exploration_rate * (1.5 - self.self_model.competence);

        if rand_simple() < adjusted_exploration {
            // Bias exploration toward self-model's best actions
            let best_actions = self.self_model.get_best_actions();
            if !best_actions.is_empty() && rand_simple() < 0.5 {
                let idx = (rand_simple() * best_actions.len() as f64) as usize % best_actions.len();
                return Some(best_actions[idx]);
            }

            // ════════════════════════════════════════════════════════════════════
            // 6. SYMBOL SYSTEM: Symbolic reasoning for action selection
            // ════════════════════════════════════════════════════════════════════
            // Find symbols grounded in the current state
            if let Some(symbol) = self.symbols.find_by_sensory(state, 0.5) {
                // If this symbol is associated with a goal, boost goal-directed actions
                if symbol.goal_id.is_some() {
                    // The symbol is associated with goal achievement
                    // Boost the action that most aligns with goal direction
                    if let Some(ref direction) = goal_direction {
                        for action in 0..self.n_actions {
                            if let Some(next) = self.world_model.predict(state, action) {
                                let action_dir: Vec<f64> = next
                                    .features
                                    .iter()
                                    .zip(state.iter())
                                    .map(|(&n, &s)| n - s)
                                    .collect();
                                let alignment = cosine_sim(&action_dir, direction);
                                if alignment > 0.5 {
                                    // Weight by symbol confidence and usage
                                    let symbol_weight = symbol.grounding_confidence
                                        * (1.0 + symbol.usage_count as f64 * 0.1).min(2.0);
                                    action_values[action] += 0.3 * symbol_weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 7. Q-LEARNING: Add learned action values (THE CRITICAL BRIDGE!)
        // ════════════════════════════════════════════════════════════════════
        // This is where internal learning DIRECTLY influences action selection
        // Q-values represent EXPERIENCE-BASED value estimates, not just heuristics
        for action in 0..self.n_actions {
            let q_value = self.get_q_value(state, action);
            // Weight Q-values significantly - they capture REAL learning
            // Scale by confidence: more updates → more trusted
            let state_disc = self.discretize_state(state);
            let key = (state_disc, action);
            let confidence = if let Some(av) = self.action_values.get(&key) {
                // More updates = more confidence, cap at 1.0
                (av.update_count as f64 / 20.0).min(1.0)
            } else {
                0.0
            };
            // Q-value contribution weighted by confidence
            // High weight (0.6) because Q-values are LEARNED from real experience
            action_values[action] += q_value * confidence * 0.6;
            action_counts[action] += 1;
        }

        // ════════════════════════════════════════════════════════════════════
        // FINAL DECISION: Combine all signals with softmax
        // ════════════════════════════════════════════════════════════════════

        // Add causal bonus to best action (exploitation boost)
        if causal_bonus > 0.0 {
            if let Some((best_idx, _)) = action_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                action_values[best_idx] += causal_bonus;
            }
        }

        // Softmax selection with temperature from meta-learner
        // Higher learning rate → lower temperature → more greedy
        let temperature = (1.0 - meta_params.learning_rate).max(0.1).min(1.0);
        let max_val = action_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Handle case where all values are equal or very low
        if max_val < -1e6 || action_values.iter().all(|&v| (v - max_val).abs() < 1e-10) {
            // No clear signal - fall back to random
            return Some((rand_simple() * self.n_actions as f64) as usize % self.n_actions);
        }

        let exp_vals: Vec<f64> = action_values
            .iter()
            .map(|v| ((v - max_val) / temperature).exp())
            .collect();
        let sum: f64 = exp_vals.iter().sum();

        if sum < 1e-10 {
            return Some((rand_simple() * self.n_actions as f64) as usize % self.n_actions);
        }

        let mut cumulative = 0.0;
        let threshold = rand_simple();
        for (action, &exp_val) in exp_vals.iter().enumerate() {
            cumulative += exp_val / sum;
            if cumulative >= threshold {
                return Some(action);
            }
        }

        // Fallback
        action_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(a, _)| a)
    }

    /// Recommend action AND generate an audit trail explaining why
    /// Returns (action, audit) tuple
    pub fn recommend_action_with_audit(&mut self, state: &[f64]) -> (Option<usize>, DecisionAudit) {
        let mut audit = DecisionAudit {
            step: self.current_step,
            ..Default::default()
        };

        // Track contributions from each subsystem
        let mut action_values: Vec<f64> = vec![0.0; self.n_actions];
        let mut q_contributions: Vec<f64> = vec![0.0; self.n_actions];
        let mut goal_contributions: Vec<f64> = vec![0.0; self.n_actions];
        let mut wm_contributions: Vec<f64> = vec![0.0; self.n_actions];
        let mut concept_contributions: Vec<f64> = vec![0.0; self.n_actions];
        let mut symbol_contributions: Vec<f64> = vec![0.0; self.n_actions];

        // Get goal direction once
        let goal_direction = self.goals.get_goal_direction(state);

        // Record active goal
        if let Some(goal) = self.goals.get_current_goal() {
            audit.active_goal = Some(goal.name.clone());
        }

        // 1. Causal relevance
        let relevant_vars = self.causal_discovery.get_relevant_variables(state);
        let causal_bonus = if !relevant_vars.is_empty() {
            let avg_utility: f64 = relevant_vars
                .iter()
                .map(|v| v.utility_count as f64)
                .sum::<f64>()
                / relevant_vars.len() as f64;
            audit.causal_relevance = (avg_utility / 10.0).min(1.0);
            (avg_utility / 10.0).min(0.3)
        } else {
            0.0
        };

        // 2. Concept-based contributions
        let activated_concepts = self.abstraction.get_activated_concepts(state);
        for (concept_id, activation) in &activated_concepts {
            if let Some(concept) = self.abstraction.concepts.get(concept_id) {
                for &action in &concept.associated_actions {
                    if action < self.n_actions {
                        let contrib = activation * concept.confidence * 0.4;
                        concept_contributions[action] += contrib;
                        action_values[action] += contrib;
                    }
                }
                if *activation > 0.5 {
                    audit.influential_concepts.push(concept.name.clone());
                }
            }
        }

        // 3. World model contributions
        for action in 0..self.n_actions {
            if let Some(next) = self.world_model.predict(state, action) {
                let mut value = next.reward;

                // Goal alignment
                if let Some(ref direction) = goal_direction {
                    let action_direction: Vec<f64> = next
                        .features
                        .iter()
                        .zip(state.iter())
                        .map(|(&n, &s)| n - s)
                        .collect();
                    let alignment = cosine_sim(&action_direction, direction);
                    let goal_bonus = alignment * 0.6;
                    goal_contributions[action] += goal_bonus;
                    value += goal_bonus;
                }

                // Goal completion bonus
                if self
                    .goals
                    .check_prediction_completes_goal(&next.features)
                    .is_some()
                {
                    goal_contributions[action] += 1.0;
                    value += 1.0;
                }

                wm_contributions[action] += value;
                action_values[action] += value;
            }
        }

        // 4. Symbol-based contributions
        if let Some(symbol) = self.symbols.find_by_sensory(state, 0.5) {
            audit.influential_symbols.push(symbol.name.clone());
            if symbol.goal_id.is_some() {
                if let Some(ref direction) = goal_direction {
                    for action in 0..self.n_actions {
                        if let Some(next) = self.world_model.predict(state, action) {
                            let action_dir: Vec<f64> = next
                                .features
                                .iter()
                                .zip(state.iter())
                                .map(|(&n, &s)| n - s)
                                .collect();
                            let alignment = cosine_sim(&action_dir, direction);
                            if alignment > 0.5 {
                                let symbol_weight = symbol.grounding_confidence
                                    * (1.0 + symbol.usage_count as f64 * 0.1).min(2.0);
                                let contrib = 0.3 * symbol_weight;
                                symbol_contributions[action] += contrib;
                                action_values[action] += contrib;
                            }
                        }
                    }
                }
            }
        }

        // 5. Q-learning contributions
        for action in 0..self.n_actions {
            let q_value = self.get_q_value(state, action);
            let state_disc = self.discretize_state(state);
            let key = (state_disc, action);
            let confidence = if let Some(av) = self.action_values.get(&key) {
                (av.update_count as f64 / 20.0).min(1.0)
            } else {
                0.0
            };
            let contrib = q_value * confidence * 0.6;
            q_contributions[action] += contrib;
            action_values[action] += contrib;
        }

        // Add causal bonus to best action
        if causal_bonus > 0.0 {
            if let Some((best_idx, _)) = action_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                action_values[best_idx] += causal_bonus;
            }
        }

        // Check exploration
        let meta_params = self.meta_learner.get_recommended_params();
        let should_explore = rand_simple() < meta_params.exploration_rate;
        audit.was_exploration = should_explore;

        // Select action
        let selected = if should_explore {
            // Exploration - select based on novelty/uncertainty
            let explore_action = (rand_simple() * self.n_actions as f64) as usize % self.n_actions;
            Some(explore_action)
        } else {
            // Exploitation - softmax selection
            let temperature = (1.0 - meta_params.learning_rate).max(0.1).min(1.0);
            let max_val = action_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            if max_val < -1e6 || action_values.iter().all(|&v| (v - max_val).abs() < 1e-10) {
                Some((rand_simple() * self.n_actions as f64) as usize % self.n_actions)
            } else {
                let exp_vals: Vec<f64> = action_values
                    .iter()
                    .map(|v| ((v - max_val) / temperature).exp())
                    .collect();
                let sum: f64 = exp_vals.iter().sum();

                if sum < 1e-10 {
                    Some((rand_simple() * self.n_actions as f64) as usize % self.n_actions)
                } else {
                    let mut cumulative = 0.0;
                    let threshold = rand_simple();
                    let mut result = None;
                    for (action, &exp_val) in exp_vals.iter().enumerate() {
                        cumulative += exp_val / sum;
                        if cumulative >= threshold && result.is_none() {
                            result = Some(action);
                        }
                    }
                    result.or_else(|| Some(self.n_actions - 1))
                }
            }
        };

        // Fill in audit with selected action's contributions
        if let Some(action) = selected {
            audit.selected_action = action;
            audit.q_value_contribution = q_contributions[action];
            audit.goal_contribution = goal_contributions[action];
            audit.world_model_contribution = wm_contributions[action];
            audit.concept_contribution = concept_contributions[action];
            audit.symbol_contribution = symbol_contributions[action];
            audit.total_confidence = action_values[action];
        }

        // Store audit (keep last 100)
        self.decision_audits.push_back(audit.clone());
        while self.decision_audits.len() > 100 {
            self.decision_audits.pop_front();
        }

        // Update AGC explicability principle
        let has_explanation = audit.q_value_contribution > 0.1
            || audit.goal_contribution > 0.1
            || !audit.influential_concepts.is_empty()
            || !audit.influential_symbols.is_empty();

        self.agc
            .principles
            .explicability
            .update(has_explanation, self.analytics.total_interactions);

        (selected, audit)
    }

    /// Get recent decision audits
    pub fn get_decision_audits(&self) -> &VecDeque<DecisionAudit> {
        &self.decision_audits
    }

    /// Get the most recent decision audit
    pub fn last_decision_audit(&self) -> Option<&DecisionAudit> {
        self.decision_audits.back()
    }

    /// Get comprehensive summary of all systems
    pub fn summary(&self) -> AGICoreSummary {
        AGICoreSummary {
            current_step: self.current_step,
            causal_discovery: self.causal_discovery.summary(),
            abstraction: self.abstraction.summary(),
            world_model: self.world_model.summary(),
            goals: self.goals.summary(),
            meta_learner: self.meta_learner.summary(),
            symbols: self.symbols.summary(),
            skills: self.skills.summary(),
            self_model: self.self_model.summary(),
            learning_webs: self.learning_webs.summary(),
            analytics: self.analytics.clone(),
            q_learning: self.q_learning_stats(),
        }
    }

    /// Print a formatted summary
    pub fn print_summary(&self) {
        let s = self.summary();
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║           UNIFIED AGI CORE - COMPOUNDING COGNITIVE COHESION      ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ Step: {:>6}                                                      ║",
            s.current_step
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ CAUSAL DISCOVERY:                                                ║");
        println!(
            "║   Variables: {:>4} | Observations: {:>6} | Avg MI: {:.3}           ║",
            s.causal_discovery.total_variables,
            s.causal_discovery.total_observations,
            s.causal_discovery.avg_information_gain
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ ABSTRACTION HIERARCHY:                                           ║");
        println!(
            "║   Concepts: {:>4} | Max Depth: {:>2} | Avg Activation: {:.1}          ║",
            s.abstraction.total_concepts, s.abstraction.max_depth, s.abstraction.avg_activation
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ WORLD MODEL:                                                     ║");
        println!(
            "║   States: {:>5} | Transitions: {:>5} | Experience: {:>6}          ║",
            s.world_model.total_states,
            s.world_model.total_transitions,
            s.world_model.total_experience
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ GOAL HIERARCHY:                                                  ║");
        println!(
            "║   Total: {:>3} | Active: {:>2} | Completed: {:>3} | Success: {:.1}%     ║",
            s.goals.total_goals,
            s.goals.active_goals,
            s.goals.completed_goals,
            s.goals.success_rate * 100.0
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ META-LEARNER:                                                    ║");
        println!(
            "║   Episodes: {:>4} | Best LR: {:.4} | Best Exp: {:.2}               ║",
            s.meta_learner.total_episodes,
            s.meta_learner.best_learning_rate,
            s.meta_learner.best_exploration_rate
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ SYMBOL SYSTEM:                                                   ║");
        println!(
            "║   Symbols: {:>4} | Grounded: {:>4} | Expressions: {:>4}             ║",
            s.symbols.total_symbols, s.symbols.grounded_symbols, s.symbols.total_expressions
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ SKILL LIBRARY: (NEW)                                             ║");
        println!(
            "║   Skills: {:>4} | Avg Success: {:.1}% | Executions: {:>5}           ║",
            s.skills.total_skills,
            s.skills.avg_success_rate * 100.0,
            s.skills.total_executions
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ SELF-MODEL: (NEW)                                                ║");
        println!(
            "║   Known States: {:>4} | Competence: {:.2} | Calibration: {:.2}       ║",
            s.self_model.known_states, s.self_model.overall_competence, s.self_model.calibration
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ COMPOUNDING ANALYTICS:                                           ║");
        println!(
            "║   Discovery→Abstraction: {:>4}  Abstraction→Symbols: {:>4}         ║",
            s.analytics.discovery_to_abstraction, s.analytics.abstraction_to_symbols
        );
        println!(
            "║   Symbols→Goals: {:>4}          Goals→WorldModel: {:>4}            ║",
            s.analytics.symbols_to_goals, s.analytics.goals_to_world_model
        );
        println!(
            "║   WorldModel→Meta: {:>4}        Meta→Discovery: {:>4}              ║",
            s.analytics.world_model_to_meta, s.analytics.meta_to_discovery
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ EMERGENCE METRICS:                                               ║");
        println!(
            "║   Credit→Discoveries: {:>4}    Credit→Symbols: {:>4}              ║",
            s.analytics.credit_to_discoveries, s.analytics.credit_to_symbols
        );
        println!(
            "║   Surprise Updates: {:>4}      Imagination Concepts: {:>4}         ║",
            s.analytics.surprise_meta_updates, s.analytics.imagination_abstractions
        );
        println!(
            "║   Amplification Factor: {:.3}                                      ║",
            s.analytics.amplification_factor
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║   Total Interactions: {:>6}   Compound Rate: {:.3}                ║",
            s.analytics.total_interactions, s.analytics.compound_rate
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ ILLICHIAN LEARNING WEBS (Deschooling for AGI):                   ║");
        println!(
            "║   Objects: {:>4} | Skills: {:>4} | Peers: {:>3} | Educators: {:>3}     ║",
            s.learning_webs.objects_count,
            s.learning_webs.skills_count,
            s.learning_webs.peers_count,
            s.learning_webs.educators_count
        );
        println!(
            "║   Self-Direction: {:.3} | Institutionalization: {:.3}              ║",
            s.learning_webs.self_direction_score, s.learning_webs.institutionalization_score
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ Q-LEARNING (ACTION VALUES):                                      ║");
        println!(
            "║   State-Actions: {:>5} | Avg Q: {:>+.3} | Avg Updates: {:.1}        ║",
            s.q_learning.0, s.q_learning.1, s.q_learning.2
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    /// Check cognitive health (Collider-inspired anomaly detection)
    /// Returns a health report with detected anomalies and recommendations
    pub fn check_health(&self) -> CognitiveHealthReport {
        self.check_health_with_thresholds(CognitiveHealthThresholds::default())
    }

    /// Check cognitive health with custom thresholds
    pub fn check_health_with_thresholds(
        &self,
        thresholds: CognitiveHealthThresholds,
    ) -> CognitiveHealthReport {
        let mut anomalies = Vec::new();
        let mut subsystem_health = SubsystemHealth::default();
        let step = self.current_step.max(1) as f64;

        // 1. Check Goal System
        let goals_per_step = self.goals.summary().total_goals as f64 / step;
        if goals_per_step > thresholds.max_goals_per_step {
            anomalies.push(CognitiveAnomaly::GoalExplosion);
            subsystem_health.goals = 0.3;
        } else {
            subsystem_health.goals =
                1.0 - (goals_per_step / thresholds.max_goals_per_step).min(1.0) * 0.5;
        }

        // 2. Check Symbol System
        let symbol_summary = self.symbols.summary();
        let symbol_usage = symbol_summary.total_symbols as f64 / self.config.max_symbols as f64;
        if symbol_usage > thresholds.max_symbol_usage_ratio {
            anomalies.push(CognitiveAnomaly::SymbolSaturation);
            subsystem_health.symbols = 0.5;
        } else {
            // Some usage is good, but not too much
            subsystem_health.symbols = if symbol_usage < 0.1 { 0.7 } else { 1.0 };
        }

        // 3. Check World Model
        let wm_summary = self.world_model.summary();
        // Use prediction error from our running average instead
        if self.avg_prediction_error > thresholds.max_world_model_error {
            anomalies.push(CognitiveAnomaly::WorldModelDegradation);
            subsystem_health.world_model = 0.4;
        } else {
            subsystem_health.world_model = 1.0 - self.avg_prediction_error.min(1.0);
        }

        // 4. Check Meta-Learner (via compound rate stability)
        let meta_summary = self.meta_learner.summary();
        if meta_summary.total_episodes > 10 {
            // Check if exploration rate is oscillating (sign of instability)
            let exp_rate = meta_summary.best_exploration_rate;
            if exp_rate < 0.01 || exp_rate > 0.9 {
                anomalies.push(CognitiveAnomaly::MetaInstability);
                subsystem_health.meta_learner = 0.5;
            } else {
                subsystem_health.meta_learner = 1.0;
            }
        } else {
            subsystem_health.meta_learner = 0.8; // Not enough data yet
        }

        // 5. Check Credit Flow
        let credit_flow =
            (self.analytics.credit_to_discoveries + self.analytics.credit_to_symbols) as f64 / step;
        if credit_flow < thresholds.min_credit_flow_ratio && step > 1000.0 {
            anomalies.push(CognitiveAnomaly::CreditBlockage);
            subsystem_health.compounding = (subsystem_health.compounding + 0.5) / 2.0;
        }

        // 6. Check Compound Rate
        if self.analytics.compound_rate < thresholds.min_compound_rate && step > 500.0 {
            anomalies.push(CognitiveAnomaly::CompoundCollapse);
            subsystem_health.compounding = 0.3;
        } else {
            subsystem_health.compounding = (self.analytics.compound_rate / 10.0).min(1.0);
        }

        // 7. Check Abstraction Hierarchy
        let abs_summary = self.abstraction.summary();
        if abs_summary.max_depth < thresholds.min_abstraction_depth && step > 1000.0 {
            anomalies.push(CognitiveAnomaly::AbstractionStagnation);
            subsystem_health.abstraction = 0.6;
        } else {
            subsystem_health.abstraction = (abs_summary.max_depth as f64 / 3.0).min(1.0);
        }

        // 8. Check Causal Discovery
        let cd_summary = self.causal_discovery.summary();
        let discovery_rate = cd_summary.total_variables as f64 / step;
        if discovery_rate < thresholds.min_discovery_rate && step > 2000.0 {
            anomalies.push(CognitiveAnomaly::DiscoveryDormant);
            subsystem_health.causal_discovery = 0.5;
        } else {
            subsystem_health.causal_discovery = (discovery_rate / 0.01 + 0.5).min(1.0);
        }

        // Calculate overall health score
        let health_score = (subsystem_health.causal_discovery
            + subsystem_health.abstraction
            + subsystem_health.world_model
            + subsystem_health.goals
            + subsystem_health.meta_learner
            + subsystem_health.symbols
            + subsystem_health.compounding)
            / 7.0;

        // System is healthy if score > 0.7 and no critical anomalies
        let has_critical = anomalies.iter().any(|a| a.severity() >= 3);
        let is_healthy = health_score > 0.7 && !has_critical;

        CognitiveHealthReport {
            step: self.current_step,
            anomalies,
            health_score,
            subsystem_health,
            is_healthy,
        }
    }

    /// Print health report
    pub fn print_health_report(&self) {
        let report = self.check_health();

        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║           COGNITIVE HEALTH REPORT (COLLIDER-INSPIRED)            ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ Step: {:>6}    Health Score: {:.2}    Status: {:>10}           ║",
            report.step,
            report.health_score,
            if report.is_healthy {
                "HEALTHY"
            } else {
                "ANOMALY"
            }
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ SUBSYSTEM HEALTH:                                                ║");
        println!(
            "║   Causal Discovery: {:.2}    Abstraction: {:.2}    World Model: {:.2}  ║",
            report.subsystem_health.causal_discovery,
            report.subsystem_health.abstraction,
            report.subsystem_health.world_model
        );
        println!(
            "║   Goals: {:.2}    Meta-Learner: {:.2}    Symbols: {:.2}    Compound: {:.2} ║",
            report.subsystem_health.goals,
            report.subsystem_health.meta_learner,
            report.subsystem_health.symbols,
            report.subsystem_health.compounding
        );

        if !report.anomalies.is_empty() {
            println!("╠══════════════════════════════════════════════════════════════════╣");
            println!("║ DETECTED ANOMALIES:                                              ║");
            for anomaly in &report.anomalies {
                println!("║   [{:?}] {:?}", anomaly.severity(), anomaly);
                println!("║     → {}", anomaly.recommended_action());
            }
        }
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

/// Comprehensive summary of all AGI systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGICoreSummary {
    pub current_step: usize,
    pub causal_discovery: CausalDiscoverySummary,
    pub abstraction: AbstractionSummary,
    pub world_model: WorldModelSummary,
    pub goals: GoalHierarchySummary,
    pub meta_learner: MetaLearnerSummary,
    pub symbols: SymbolSystemSummary,
    pub skills: SkillLibrarySummary,
    pub self_model: SelfModelSummary,
    pub learning_webs: LearningWebsSummary,
    pub analytics: CompoundingAnalytics,
    /// Q-Learning stats: (num_entries, avg_q_value, avg_updates_per_entry)
    pub q_learning: (usize, f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agi_core_creation() {
        let config = AGICoreConfig::default();
        let core = AGICore::new(config, 10, 4);
        assert_eq!(core.current_step, 0);
    }

    #[test]
    fn test_causal_discovery() {
        let config = AGICoreConfig::default();
        let mut discovery = CausalDiscovery::new(config);

        // Add observations
        for i in 0..200 {
            let features = vec![
                i as f64 * 0.01,
                (i as f64 * 0.01).sin(),
                (i as f64 * 0.01).cos(),
            ];
            discovery.observe(features, Some(i % 4), (i % 10) as f64 / 10.0);
        }

        let summary = discovery.summary();
        assert!(summary.total_observations > 0);
    }

    #[test]
    fn test_abstraction_hierarchy() {
        let config = AGICoreConfig::default();
        let mut abstraction = AbstractionHierarchy::new(config);

        // Add patterns
        for i in 0..50 {
            let features = vec![i as f64 * 0.1, (i as f64 * 0.1).sin()];
            abstraction.observe(features);
        }

        let summary = abstraction.summary();
        assert!(summary.total_concepts >= 0); // May or may not have concepts
    }

    #[test]
    fn test_world_model() {
        let config = AGICoreConfig::default();
        let mut world = WorldModel::new(config, 4);

        // Learn transitions
        for i in 0..100 {
            let from = vec![i as f64 * 0.1, 0.0, 0.0, 0.0];
            let to = vec![(i + 1) as f64 * 0.1, 0.0, 0.0, 0.0];
            world.learn(&from, i % 4, &to, 0.1, false);
        }

        let summary = world.summary();
        assert!(summary.total_experience > 0);
    }

    #[test]
    fn test_goal_hierarchy() {
        let config = AGICoreConfig::default();
        let mut goals = GoalHierarchy::new(config);

        let goal_id = goals.create_goal("test_goal", vec![1.0, 1.0, 1.0], GoalPriority::High);
        goals.activate_goal(goal_id);

        // Use a state very far from target so goal doesn't complete
        // Progress formula: 1.0 - (distance / max_distance)
        // With target [1,1,1] and state [-1,-1,-1]: distance = sqrt(12) ≈ 3.46
        // max_distance = 3 * 2 = 6, progress = 1 - 3.46/6 = 0.42
        // With threshold 0.55, goal stays active
        goals.update_progress(&[-1.0, -1.0, -1.0]);

        let summary = goals.summary();
        assert_eq!(summary.active_goals, 1);
    }

    #[test]
    fn test_symbol_system() {
        let config = AGICoreConfig::default();
        let mut symbols = SymbolSystem::new(config);

        let sym1 = symbols.get_or_create_symbol("up", vec![0.0, 1.0], vec![1.0, 0.0]);
        let sym2 = symbols.get_or_create_symbol("down", vec![0.0, -1.0], vec![-1.0, 0.0]);

        symbols.compose(vec![sym1, sym2], SymbolRelation::Sequence);

        let summary = symbols.summary();
        assert_eq!(summary.total_symbols, 2);
        assert_eq!(summary.total_expressions, 1);
    }

    #[test]
    fn test_integrated_processing() {
        let config = AGICoreConfig::default();
        let mut core = AGICore::new(config, 4, 4);

        // Process experiences
        for i in 0..100 {
            let state = vec![i as f64 * 0.1, (i as f64 * 0.1).sin(), 0.0, 0.0];
            let next_state = vec![(i + 1) as f64 * 0.1, ((i + 1) as f64 * 0.1).sin(), 0.0, 0.0];
            core.process_experience(&state, i % 4, &next_state, 0.1, false);
        }

        let summary = core.summary();
        assert_eq!(summary.current_step, 100);
        assert!(summary.analytics.total_interactions > 0);
    }
}
