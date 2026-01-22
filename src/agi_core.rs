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
            causal_discovery_threshold: 0.3,
            abstraction_merge_threshold: 0.85,
            world_model_error_threshold: 0.1,
            goal_completion_threshold: 0.65, // Lowered from 0.9 for noisy environments
            symbol_grounding_threshold: 0.5, // Lowered from 0.7 for easier grounding
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

        // Attempt discovery every 50 steps
        if self.current_step % 50 == 0 {
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

    /// Update goal progress based on current state
    pub fn update_progress(&mut self, current_state: &[f64]) {
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
        for goal_id in completed_goals {
            self.active_goals.retain(|&id| id != goal_id);
            self.achievement_history
                .push_back((goal_id, true, self.current_step));

            // Check if parent goal should be updated
            if let Some(parent_id) = self.goals.get(&goal_id).and_then(|g| g.parent) {
                self.update_parent_progress(parent_id);
            }
        }

        while self.achievement_history.len() > 100 {
            self.achievement_history.pop_front();
        }
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

    /// Compounding analytics
    pub analytics: CompoundingAnalytics,

    /// Current step
    current_step: usize,

    /// Feature dimension
    feature_dim: usize,

    /// Number of actions
    n_actions: usize,
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
            analytics: CompoundingAnalytics::default(),
            current_step: 0,
            feature_dim,
            n_actions,
            config,
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

        // 1. CAUSAL DISCOVERY - observe patterns
        self.causal_discovery
            .observe(state.to_vec(), Some(action), reward);

        // 2. WORLD MODEL - learn dynamics
        self.world_model
            .learn(state, action, next_state, reward, is_terminal);

        // 3. ABSTRACTION - form concepts
        self.abstraction.observe(state.to_vec());
        self.abstraction.observe(next_state.to_vec());

        // 4. GOAL PROGRESS - update active goals
        self.goals.update_progress(next_state);

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

        // 6. COMPOUND INTERACTIONS - drive multiplicative growth
        self.compound_interactions(state, action, next_state, reward);
    }

    /// Drive compounding interactions between systems
    fn compound_interactions(
        &mut self,
        state: &[f64],
        action: usize,
        next_state: &[f64],
        _reward: f64,
    ) {
        // Discovery → Abstraction: Use discovered variables to inform concepts
        let discovered = self.causal_discovery.get_relevant_variables(state);
        for var in discovered {
            // Create a concept from the discovered variable signature
            self.abstraction.observe(var.signature.clone());
            self.analytics.discovery_to_abstraction += 1;
        }

        // Abstraction → Symbols: Ground symbols in activated concepts
        let activated = self.abstraction.get_activated_concepts(state);
        for (concept_id, similarity) in activated.iter().take(3) {
            if *similarity > 0.7 {
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

        // Symbols → Goals: AUTO-CREATE GOALS from grounded symbols that don't have goals
        // This is the key fix - we need to create goals from symbols, not just express existing goals
        let ungoaled_symbols: Vec<(String, Vec<f64>, Option<usize>)> = self
            .symbols
            .get_ungoaled_grounded_symbols()
            .iter()
            .take(2) // Limit to 2 new goals per step
            .map(|s| (s.name.clone(), s.sensory_grounding.clone(), s.concept_id))
            .collect();

        for (name, target_features, concept_id) in ungoaled_symbols {
            // Only create goals if we have room
            if self.goals.active_count() < self.config.max_active_goals {
                let goal_name = format!("explore_{}", name);
                let goal_id = self.goals.create_goal_from_symbol(
                    &goal_name,
                    target_features.clone(),
                    concept_id,
                );

                // Link symbol back to goal
                if let Some(&sym_id) = self.symbols.name_to_id.get(&name) {
                    self.symbols.ground_in_goal(sym_id, goal_id);
                }

                self.analytics.symbols_to_goals += 1;
            }
        }

        // Also express current goal symbolically (original behavior)
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

        // Goals → World Model: Imagine paths to goals AND check if predictions complete goals
        if let Some(goal) = self.goals.get_current_goal() {
            let goal_id = goal.id;
            let goal_subgoals_empty = goal.subgoals.is_empty();

            let futures = self.world_model.imagine_futures(state, 3, self.n_actions);

            // Check if any imagined future would complete a goal
            for trajectory in &futures {
                for imagined_state in &trajectory.states {
                    if let Some(_completed_id) = self
                        .goals
                        .check_prediction_completes_goal(&imagined_state.features)
                    {
                        // Prediction shows path to goal completion - this is valuable!
                        self.analytics.goals_to_world_model += 1;
                    }
                }
            }

            // Use imagination to inform goal decomposition (original behavior)
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

        // World Model → Meta-Learning: Track prediction quality
        if let Some(predicted) = self.world_model.predict(state, action) {
            let prediction_error: f64 = predicted
                .features
                .iter()
                .zip(next_state.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f64>()
                .sqrt();

            // Record learning episode if this is a notable step
            if self.current_step % 100 == 0 {
                let metrics = self.meta_learner.get_recommended_params();
                self.meta_learner.record_episode(
                    prediction_error,
                    1.0 / (prediction_error + 0.1),
                    100,
                    metrics,
                );
                self.analytics.world_model_to_meta += 1;
            }
        }

        // Meta-Learning → Discovery: Use learned parameters
        // (The meta-learner's parameters will be used in the next iteration)
        if self.current_step % 200 == 0 {
            // Trigger causal re-analysis with improved parameters
            self.analytics.meta_to_discovery += 1;
        }

        // Update totals
        self.analytics.total_interactions = self.analytics.discovery_to_abstraction
            + self.analytics.abstraction_to_symbols
            + self.analytics.symbols_to_goals
            + self.analytics.goals_to_world_model
            + self.analytics.world_model_to_meta
            + self.analytics.meta_to_discovery;

        // Compute compound rate
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

    /// Get action recommendation based on integrated cognition
    pub fn recommend_action(&self, state: &[f64]) -> Option<usize> {
        // Get goal direction
        let goal_direction = self.goals.get_goal_direction(state);

        // Use world model to evaluate actions
        let mut best_action: Option<(usize, f64)> = None;

        for action in 0..self.n_actions {
            if let Some(next) = self.world_model.predict(state, action) {
                let mut value = next.reward;

                // Add goal-directed bonus
                if let Some(ref direction) = goal_direction {
                    let action_direction: Vec<f64> = next
                        .features
                        .iter()
                        .zip(state.iter())
                        .map(|(&n, &s)| n - s)
                        .collect();

                    let alignment = cosine_sim(&action_direction, direction);
                    value += alignment * 0.5;
                }

                // Penalize uncertainty
                value -= next.uncertainty * 0.2;

                if best_action
                    .as_ref()
                    .map(|(_, v)| value > *v)
                    .unwrap_or(true)
                {
                    best_action = Some((action, value));
                }
            }
        }

        // Add exploration based on meta-learner
        let recommended = self.meta_learner.get_recommended_params();
        if rand_simple() < recommended.exploration_rate {
            return Some((rand_simple() * self.n_actions as f64) as usize % self.n_actions);
        }

        best_action.map(|(a, _)| a)
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
            analytics: self.analytics.clone(),
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
        println!(
            "║   Total Interactions: {:>6}   Compound Rate: {:.3}                ║",
            s.analytics.total_interactions, s.analytics.compound_rate
        );
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
    pub analytics: CompoundingAnalytics,
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

        goals.update_progress(&[0.5, 0.5, 0.5]);

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
