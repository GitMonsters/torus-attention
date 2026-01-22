//! # Consequential Reasoning and Causal Graphs for AGI
//!
//! This module implements Pearl-style causal reasoning, multi-step consequential
//! thinking, and voting mechanisms for emergent AGI capabilities.
//!
//! ## Key Components
//!
//! 1. **Causal Graphs**: Variables and directed edges representing causal mechanisms
//! 2. **Consequential Thinking**: Multi-step tree search with uncertainty
//! 3. **Stream Voting**: Thousand Brains-style consensus among parallel streams
//! 4. **Compounding Metrics**: Transfer learning and knowledge accumulation measures
//!
//! ## Theoretical Foundation
//!
//! Pearl's Causal Ladder:
//! - Level 1 (Association): P(Y|X) - What co-occurs?
//! - Level 2 (Intervention): P(Y|do(X)) - What if I do X?
//! - Level 3 (Counterfactual): P(Y_x|X',Y') - What if X had been different?

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL GRAPH SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// A variable in the causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalVariable {
    /// Unique identifier
    pub id: usize,
    /// Name/label for the variable
    pub name: String,
    /// Current observed value
    pub value: f64,
    /// Whether this is an action variable (can be intervened on)
    pub is_action: bool,
    /// Whether this is a reward/outcome variable
    pub is_outcome: bool,
    /// Uncertainty in the current value
    pub uncertainty: f64,
}

impl CausalVariable {
    pub fn new(id: usize, name: &str, is_action: bool, is_outcome: bool) -> Self {
        Self {
            id,
            name: name.to_string(),
            value: 0.0,
            is_action,
            is_outcome,
            uncertainty: 1.0, // Start with high uncertainty
        }
    }

    pub fn observe(&mut self, value: f64) {
        // Bayesian update of value with uncertainty reduction
        let alpha = 0.3; // Learning rate
        self.value = self.value * (1.0 - alpha) + value * alpha;
        self.uncertainty = (self.uncertainty * 0.95).max(0.1); // Reduce uncertainty
    }
}

/// A causal mechanism (edge) between variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalMechanism {
    /// Source variable ID
    pub from: usize,
    /// Target variable ID
    pub to: usize,
    /// Causal strength (can be negative for inhibition)
    pub strength: f64,
    /// Confidence in this causal relationship
    pub confidence: f64,
    /// Number of observations supporting this edge
    pub observation_count: usize,
    /// Time delay (in steps) for this causal effect
    pub delay: usize,
}

impl CausalMechanism {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            strength: 0.0,
            confidence: 0.1, // Start with low confidence
            observation_count: 0,
            delay: 1,
        }
    }

    /// Update mechanism based on observed intervention
    pub fn update(&mut self, cause_change: f64, effect_change: f64, learning_rate: f64) {
        // Estimate causal strength from intervention
        let observed_strength = if cause_change.abs() > 1e-6 {
            effect_change / cause_change
        } else {
            0.0
        };

        // Bayesian update of strength
        self.strength = self.strength * (1.0 - learning_rate) + observed_strength * learning_rate;
        self.observation_count += 1;

        // Increase confidence with observations
        self.confidence = (self.confidence + 0.05).min(0.95);
    }
}

/// An intervention on a variable
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Variable being intervened on
    pub variable_id: usize,
    /// Value being set
    pub value: f64,
    /// Prior value before intervention
    pub prior_value: f64,
    /// Timestamp of intervention
    pub step: usize,
}

/// Outcome of a counterfactual query
#[derive(Debug, Clone)]
pub struct CounterfactualOutcome {
    /// Variable queried
    pub variable_id: usize,
    /// Actual observed value
    pub actual_value: f64,
    /// Counterfactual value (what would have happened)
    pub counterfactual_value: f64,
    /// Confidence in the counterfactual
    pub confidence: f64,
    /// Causal attribution (how much each cause contributed)
    pub attributions: HashMap<usize, f64>,
}

/// Pearl-style causal graph for consequential reasoning
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Variables in the graph
    pub variables: Vec<CausalVariable>,
    /// Causal mechanisms (edges)
    pub mechanisms: HashMap<(usize, usize), CausalMechanism>,
    /// Intervention history for learning
    pub intervention_history: VecDeque<(Intervention, HashMap<usize, f64>)>,
    /// Maximum history size
    max_history: usize,
    /// Learning rate for causal discovery
    learning_rate: f64,
}

impl CausalGraph {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            variables: Vec::new(),
            mechanisms: HashMap::new(),
            intervention_history: VecDeque::with_capacity(100),
            max_history: 100,
            learning_rate,
        }
    }

    /// Create a causal graph for grid navigation
    pub fn for_navigation() -> Self {
        let mut graph = Self::new(0.1);

        // Add core variables
        graph.add_variable("position_x", false, false); // 0
        graph.add_variable("position_y", false, false); // 1
        graph.add_variable("action", true, false); // 2 (action variable)
        graph.add_variable("reward", false, true); // 3 (outcome)
        graph.add_variable("novelty", false, false); // 4
        graph.add_variable("coherence", false, false); // 5
        graph.add_variable("goal_distance", false, false); // 6

        // Add initial causal edges (hypothesized structure)
        // Action causes position change
        graph.add_mechanism(2, 0); // action -> position_x
        graph.add_mechanism(2, 1); // action -> position_y

        // Position affects reward and novelty
        graph.add_mechanism(0, 3); // position_x -> reward
        graph.add_mechanism(1, 3); // position_y -> reward
        graph.add_mechanism(0, 4); // position_x -> novelty
        graph.add_mechanism(1, 4); // position_y -> novelty

        // Goal distance affects reward
        graph.add_mechanism(6, 3); // goal_distance -> reward

        // Coherence affects action selection (feedback loop)
        graph.add_mechanism(5, 2); // coherence -> action

        // Novelty affects coherence
        graph.add_mechanism(4, 5); // novelty -> coherence

        graph
    }

    /// Add a new variable
    pub fn add_variable(&mut self, name: &str, is_action: bool, is_outcome: bool) -> usize {
        let id = self.variables.len();
        self.variables
            .push(CausalVariable::new(id, name, is_action, is_outcome));
        id
    }

    /// Add a causal mechanism between variables
    pub fn add_mechanism(&mut self, from: usize, to: usize) {
        self.mechanisms
            .insert((from, to), CausalMechanism::new(from, to));
    }

    /// Observe variable values (without intervention)
    pub fn observe(&mut self, observations: &HashMap<usize, f64>) {
        for (&var_id, &value) in observations {
            if let Some(var) = self.variables.get_mut(var_id) {
                var.observe(value);
            }
        }
    }

    /// Perform an intervention: do(X = x)
    /// Returns predicted outcomes for all variables
    pub fn intervene(
        &mut self,
        variable_id: usize,
        value: f64,
        step: usize,
    ) -> HashMap<usize, f64> {
        let prior_value = self
            .variables
            .get(variable_id)
            .map(|v| v.value)
            .unwrap_or(0.0);

        // Record intervention
        let intervention = Intervention {
            variable_id,
            value,
            prior_value,
            step,
        };

        // Set the intervened variable (cut incoming edges conceptually)
        if let Some(var) = self.variables.get_mut(variable_id) {
            var.value = value;
            var.uncertainty = 0.0; // Intervention removes uncertainty
        }

        // Propagate effects through the graph (topological order)
        let outcomes = self.propagate_effects(variable_id);

        // Store for learning
        self.intervention_history
            .push_back((intervention, outcomes.clone()));
        if self.intervention_history.len() > self.max_history {
            self.intervention_history.pop_front();
        }

        outcomes
    }

    /// Propagate causal effects from an intervened variable
    fn propagate_effects(&mut self, source: usize) -> HashMap<usize, f64> {
        let mut outcomes = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(source);
        visited.insert(source);

        while let Some(current) = queue.pop_front() {
            let current_value = self.variables.get(current).map(|v| v.value).unwrap_or(0.0);
            outcomes.insert(current, current_value);

            // Find all children (effects)
            let children: Vec<(usize, f64)> = self
                .mechanisms
                .iter()
                .filter(|((from, _), _)| *from == current)
                .map(|((_, to), mech)| (*to, mech.strength))
                .collect();

            for (child, strength) in children {
                if !visited.contains(&child) {
                    visited.insert(child);

                    // Compute effect on child
                    if let Some(child_var) = self.variables.get_mut(child) {
                        // Causal effect: child += strength * parent_value
                        let effect = strength * current_value;
                        child_var.value += effect * 0.1; // Scaled effect
                    }

                    queue.push_back(child);
                }
            }
        }

        outcomes
    }

    /// Counterfactual query: "What would Y be if X had been x?"
    /// Given current observations, compute what would have happened
    pub fn counterfactual(
        &self,
        query_variable: usize,
        intervention_variable: usize,
        counterfactual_value: f64,
    ) -> CounterfactualOutcome {
        // 1. Abduction: Use current values as evidence
        let actual_value = self
            .variables
            .get(query_variable)
            .map(|v| v.value)
            .unwrap_or(0.0);

        // 2. Compute counterfactual value by tracing paths
        let mut cf_value = actual_value;
        let mut attributions = HashMap::new();

        // Find paths from intervention to query
        let paths = self.find_causal_paths(intervention_variable, query_variable);

        for path in &paths {
            // Compute effect along this path
            let mut path_effect = 1.0;
            for i in 0..path.len() - 1 {
                if let Some(mech) = self.mechanisms.get(&(path[i], path[i + 1])) {
                    path_effect *= mech.strength;
                }
            }

            // Delta from counterfactual
            let current_intervention_value = self
                .variables
                .get(intervention_variable)
                .map(|v| v.value)
                .unwrap_or(0.0);
            let delta = counterfactual_value - current_intervention_value;

            cf_value += delta * path_effect;
            attributions.insert(intervention_variable, delta * path_effect);
        }

        // Compute confidence based on mechanism confidences along paths
        let confidence = if paths.is_empty() {
            0.0
        } else {
            paths
                .iter()
                .map(|path| {
                    path.windows(2)
                        .filter_map(|w| self.mechanisms.get(&(w[0], w[1])))
                        .map(|m| m.confidence)
                        .product::<f64>()
                })
                .sum::<f64>()
                / paths.len() as f64
        };

        CounterfactualOutcome {
            variable_id: query_variable,
            actual_value,
            counterfactual_value: cf_value,
            confidence,
            attributions,
        }
    }

    /// Find all causal paths between two variables (BFS)
    fn find_causal_paths(&self, from: usize, to: usize) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut queue: VecDeque<Vec<usize>> = VecDeque::new();

        queue.push_back(vec![from]);

        while let Some(path) = queue.pop_front() {
            let current = *path.last().unwrap();

            if current == to {
                paths.push(path);
                continue;
            }

            // Limit path length to prevent cycles
            if path.len() > 10 {
                continue;
            }

            // Find children
            for ((f, t), _) in &self.mechanisms {
                if *f == current && !path.contains(t) {
                    let mut new_path = path.clone();
                    new_path.push(*t);
                    queue.push_back(new_path);
                }
            }
        }

        paths
    }

    /// Learn causal structure from intervention outcome
    pub fn learn_from_outcome(&mut self, outcome_values: &HashMap<usize, f64>) {
        if let Some((intervention, predicted)) = self.intervention_history.back() {
            // Compare predicted vs actual for each variable
            for (&var_id, &actual_value) in outcome_values {
                let predicted_value = predicted.get(&var_id).copied().unwrap_or(0.0);
                let error = actual_value - predicted_value;

                // Find mechanisms that could explain this error
                for ((from, to), mechanism) in &mut self.mechanisms {
                    if *to == var_id {
                        // This mechanism affects the variable
                        let cause_value = self.variables.get(*from).map(|v| v.value).unwrap_or(0.0);

                        // Update mechanism based on error
                        mechanism.update(cause_value, error, self.learning_rate);
                    }
                }
            }
        }

        // Also observe the values
        self.observe(outcome_values);
    }

    /// Update confidence in a specific mechanism based on prediction accuracy
    pub fn update_mechanism_confidence(&mut self, from: usize, to: usize, accuracy_factor: f64) {
        if let Some(mechanism) = self.mechanisms.get_mut(&(from, to)) {
            // Exponential moving average update of confidence
            let alpha = 0.1; // Learning rate for confidence updates
            mechanism.confidence = (1.0 - alpha) * mechanism.confidence + alpha * accuracy_factor;
            // Clamp to valid range
            mechanism.confidence = mechanism.confidence.clamp(0.01, 1.0);
        }
    }

    /// Get summary of causal structure
    pub fn summary(&self) -> CausalGraphSummary {
        let total_confidence: f64 = self.mechanisms.values().map(|m| m.confidence).sum();
        let avg_confidence = if self.mechanisms.is_empty() {
            0.0
        } else {
            total_confidence / self.mechanisms.len() as f64
        };

        CausalGraphSummary {
            n_variables: self.variables.len(),
            n_mechanisms: self.mechanisms.len(),
            avg_confidence,
            n_interventions: self.intervention_history.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CausalGraphSummary {
    pub n_variables: usize,
    pub n_mechanisms: usize,
    pub avg_confidence: f64,
    pub n_interventions: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-STEP CONSEQUENTIAL THINKING (Tree Search)
// ═══════════════════════════════════════════════════════════════════════════════

/// A node in the consequence tree
#[derive(Debug, Clone)]
pub struct ConsequenceNode {
    /// State representation
    pub state: Vec<f32>,
    /// Action that led to this state (None for root)
    pub action: Option<usize>,
    /// Predicted value of this state
    pub value: f64,
    /// Uncertainty in the prediction
    pub uncertainty: f64,
    /// Depth in the tree
    pub depth: usize,
    /// Children nodes
    pub children: Vec<ConsequenceNode>,
    /// Visit count (for UCB)
    pub visits: usize,
}

impl ConsequenceNode {
    pub fn new(state: Vec<f32>, action: Option<usize>, depth: usize) -> Self {
        Self {
            state,
            action,
            value: 0.0,
            uncertainty: 1.0,
            depth,
            children: Vec::new(),
            visits: 0,
        }
    }

    /// UCB1 score for tree search
    pub fn ucb_score(&self, parent_visits: usize, exploration_weight: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }

        let exploitation = self.value;
        let exploration =
            exploration_weight * ((parent_visits as f64).ln() / self.visits as f64).sqrt();

        exploitation + exploration
    }
}

/// Multi-step consequential thinking engine
#[derive(Debug)]
pub struct ConsequentialThinking {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Number of actions to consider
    pub n_actions: usize,
    /// Discount factor for future rewards
    pub gamma: f64,
    /// Exploration weight for UCB
    pub exploration_weight: f64,
    /// World model for state prediction
    pub transition_model: TransitionModel,
    /// Value function approximation
    pub value_estimates: HashMap<Vec<i32>, f64>,
}

impl ConsequentialThinking {
    pub fn new(max_depth: usize, n_actions: usize, gamma: f64) -> Self {
        Self {
            max_depth,
            n_actions,
            gamma,
            exploration_weight: 1.414, // sqrt(2) for UCB1
            transition_model: TransitionModel::new(),
            value_estimates: HashMap::new(),
        }
    }

    /// Build consequence tree from current state
    pub fn build_tree(&mut self, root_state: Vec<f32>, simulations: usize) -> ConsequenceNode {
        let mut root = ConsequenceNode::new(root_state, None, 0);

        for _ in 0..simulations {
            self.simulate(&mut root);
        }

        root
    }

    /// Run a single simulation (selection, expansion, evaluation, backprop)
    fn simulate(&mut self, root: &mut ConsequenceNode) {
        // Selection: traverse to leaf using UCB
        let path = self.select(root);

        // Expansion: add children if not at max depth
        if let Some(leaf) = self.get_node_at_path(root, &path) {
            if leaf.depth < self.max_depth && leaf.children.is_empty() {
                self.expand(leaf);
            }

            // Evaluation: estimate leaf value
            let value = self.evaluate(&leaf.state, leaf.depth);

            // Backpropagation: update values along path
            self.backpropagate(root, &path, value);
        }
    }

    /// Select path through tree using UCB
    fn select(&self, node: &ConsequenceNode) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = node;

        while !current.children.is_empty() {
            let parent_visits = current.visits;
            let best_child_idx = current
                .children
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let score_a = a.ucb_score(parent_visits, self.exploration_weight);
                    let score_b = b.ucb_score(parent_visits, self.exploration_weight);
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            path.push(best_child_idx);
            current = &current.children[best_child_idx];
        }

        path
    }

    /// Expand a leaf node with children for all actions
    fn expand(&mut self, node: &mut ConsequenceNode) {
        for action in 0..self.n_actions {
            let next_state = self.transition_model.predict(&node.state, action);
            let child = ConsequenceNode::new(next_state, Some(action), node.depth + 1);
            node.children.push(child);
        }
    }

    /// Evaluate a state (heuristic value estimate)
    fn evaluate(&self, state: &[f32], depth: usize) -> f64 {
        // Discretize state for lookup
        let key: Vec<i32> = state.iter().map(|&x| (x * 10.0) as i32).collect();

        // Use stored estimate or compute heuristic
        let base_value = self.value_estimates.get(&key).copied().unwrap_or_else(|| {
            // Simple heuristic: prefer states with higher coherence-like properties
            let magnitude: f32 = state.iter().map(|x| x.abs()).sum();
            (1.0 / (1.0 + magnitude as f64)) * 0.5
        });

        // Discount by depth
        base_value * self.gamma.powi(depth as i32)
    }

    /// Backpropagate value through path
    fn backpropagate(&mut self, root: &mut ConsequenceNode, path: &[usize], value: f64) {
        let mut current = root;
        let mut discounted_value = value;

        current.visits += 1;
        current.value = (current.value * (current.visits - 1) as f64 + discounted_value)
            / current.visits as f64;

        for &idx in path {
            discounted_value *= self.gamma;
            if idx < current.children.len() {
                current = &mut current.children[idx];
                current.visits += 1;
                current.value = (current.value * (current.visits - 1) as f64 + discounted_value)
                    / current.visits as f64;
            }
        }
    }

    /// Get node at path
    fn get_node_at_path<'a>(
        &self,
        root: &'a mut ConsequenceNode,
        path: &[usize],
    ) -> Option<&'a mut ConsequenceNode> {
        let mut current = root;
        for &idx in path {
            if idx < current.children.len() {
                current = &mut current.children[idx];
            } else {
                return None;
            }
        }
        Some(current)
    }

    /// Get best action from root
    pub fn best_action(&self, root: &ConsequenceNode) -> Option<usize> {
        root.children
            .iter()
            .max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|node| node.action)
    }

    /// Update transition model from observation
    pub fn update_model(&mut self, state: &[f32], action: usize, next_state: &[f32], reward: f64) {
        self.transition_model.update(state, action, next_state);

        // Update value estimate
        let key: Vec<i32> = state.iter().map(|&x| (x * 10.0) as i32).collect();
        let current = self.value_estimates.entry(key).or_insert(0.0);
        *current = *current * 0.9 + reward * 0.1;
    }
}

/// Simple transition model for state prediction
#[derive(Debug)]
pub struct TransitionModel {
    /// Learned transitions: (state_key, action) -> delta
    transitions: HashMap<(Vec<i32>, usize), Vec<f32>>,
    /// Default delta per action
    default_deltas: Vec<Vec<f32>>,
}

impl TransitionModel {
    pub fn new() -> Self {
        // Default deltas for 4 directions + stay
        let default_deltas = vec![
            vec![0.0, 0.1, 0.0],  // Up
            vec![0.0, -0.1, 0.0], // Down
            vec![-0.1, 0.0, 0.0], // Left
            vec![0.1, 0.0, 0.0],  // Right
            vec![0.0, 0.0, 0.0],  // Stay
        ];

        Self {
            transitions: HashMap::new(),
            default_deltas,
        }
    }

    /// Predict next state given current state and action
    pub fn predict(&self, state: &[f32], action: usize) -> Vec<f32> {
        let key: Vec<i32> = state.iter().map(|&x| (x * 10.0) as i32).collect();

        let delta = self
            .transitions
            .get(&(key, action))
            .or_else(|| self.default_deltas.get(action))
            .cloned()
            .unwrap_or_else(|| vec![0.0; state.len()]);

        state
            .iter()
            .zip(delta.iter().cycle())
            .map(|(&s, &d)| s + d)
            .collect()
    }

    /// Update model from observed transition
    pub fn update(&mut self, state: &[f32], action: usize, next_state: &[f32]) {
        let key: Vec<i32> = state.iter().map(|&x| (x * 10.0) as i32).collect();
        let delta: Vec<f32> = state
            .iter()
            .zip(next_state.iter())
            .map(|(&s, &ns)| ns - s)
            .collect();

        // Running average update
        let entry = self
            .transitions
            .entry((key, action))
            .or_insert_with(|| delta.clone());
        for (e, d) in entry.iter_mut().zip(delta.iter()) {
            *e = *e * 0.9 + d * 0.1;
        }
    }
}

impl Default for TransitionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THOUSAND BRAINS VOTING MECHANISM
// ═══════════════════════════════════════════════════════════════════════════════

/// A vote from a single stream/column
#[derive(Debug, Clone)]
pub struct StreamVote {
    /// Stream ID
    pub stream_id: usize,
    /// Hypothesis being voted for (e.g., best action, object ID)
    pub hypothesis: usize,
    /// Confidence/evidence for this hypothesis
    pub evidence: f64,
    /// Features supporting this vote
    pub supporting_features: Vec<f32>,
}

/// Voting result from aggregating stream votes
#[derive(Debug, Clone)]
pub struct VotingResult {
    /// Winning hypothesis
    pub winner: usize,
    /// Confidence in winner
    pub confidence: f64,
    /// Evidence distribution across hypotheses
    pub evidence_distribution: HashMap<usize, f64>,
    /// Number of streams that agreed with winner
    pub consensus_count: usize,
    /// Total streams that voted
    pub total_voters: usize,
    /// Whether consensus was reached
    pub consensus_reached: bool,
}

/// Thousand Brains-style voting mechanism for consensus
#[derive(Debug)]
pub struct StreamVotingSystem {
    /// Number of streams (columns)
    pub n_streams: usize,
    /// Number of possible hypotheses
    pub n_hypotheses: usize,
    /// Consensus threshold (fraction of streams needed)
    pub consensus_threshold: f64,
    /// Evidence decay rate
    pub evidence_decay: f64,
    /// Current evidence state per stream per hypothesis
    pub evidence_state: Vec<Vec<f64>>,
    /// Vote history for learning
    pub vote_history: VecDeque<VotingResult>,
}

impl StreamVotingSystem {
    pub fn new(n_streams: usize, n_hypotheses: usize, consensus_threshold: f64) -> Self {
        Self {
            n_streams,
            n_hypotheses,
            consensus_threshold,
            evidence_decay: 0.95,
            evidence_state: vec![vec![0.0; n_hypotheses]; n_streams],
            vote_history: VecDeque::with_capacity(100),
        }
    }

    /// Receive a vote from a stream
    pub fn receive_vote(&mut self, vote: StreamVote) {
        if vote.stream_id < self.n_streams && vote.hypothesis < self.n_hypotheses {
            // Update evidence for this hypothesis
            let evidence = &mut self.evidence_state[vote.stream_id][vote.hypothesis];
            *evidence = (*evidence + vote.evidence).min(1.0);
        }
    }

    /// Decay evidence (call each step)
    pub fn decay_evidence(&mut self) {
        for stream in &mut self.evidence_state {
            for evidence in stream {
                *evidence *= self.evidence_decay;
            }
        }
    }

    /// Aggregate votes and determine consensus
    pub fn aggregate_votes(&mut self) -> VotingResult {
        let mut total_evidence: HashMap<usize, f64> = HashMap::new();
        let mut stream_winners: Vec<usize> = Vec::new();

        // Collect each stream's top hypothesis
        for (stream_id, evidence) in self.evidence_state.iter().enumerate() {
            let (best_hyp, best_evidence) = evidence
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(h, &e)| (h, e))
                .unwrap_or((0, 0.0));

            if best_evidence > 0.1 {
                stream_winners.push(best_hyp);
                *total_evidence.entry(best_hyp).or_insert(0.0) += best_evidence;
            }
        }

        // Find overall winner
        let (winner, total_winner_evidence) = total_evidence
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&h, &e)| (h, e))
            .unwrap_or((0, 0.0));

        // Count consensus
        let consensus_count = stream_winners.iter().filter(|&&h| h == winner).count();
        let consensus_fraction = consensus_count as f64 / self.n_streams as f64;
        let consensus_reached = consensus_fraction >= self.consensus_threshold;

        // Normalize evidence distribution
        let total: f64 = total_evidence.values().sum();
        let evidence_distribution: HashMap<usize, f64> = if total > 0.0 {
            total_evidence
                .into_iter()
                .map(|(k, v)| (k, v / total))
                .collect()
        } else {
            HashMap::new()
        };

        let result = VotingResult {
            winner,
            confidence: total_winner_evidence / self.n_streams as f64,
            evidence_distribution,
            consensus_count,
            total_voters: stream_winners.len(),
            consensus_reached,
        };

        self.vote_history.push_back(result.clone());
        if self.vote_history.len() > 100 {
            self.vote_history.pop_front();
        }

        result
    }

    /// Flash inference: Quick consensus check
    pub fn flash_inference(&self) -> Option<usize> {
        let mut hypothesis_counts: HashMap<usize, usize> = HashMap::new();

        for evidence in &self.evidence_state {
            if let Some((best_hyp, _)) = evidence
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                *hypothesis_counts.entry(best_hyp).or_insert(0) += 1;
            }
        }

        // Return if strong consensus
        for (hyp, count) in hypothesis_counts {
            if count as f64 / self.n_streams as f64 >= self.consensus_threshold {
                return Some(hyp);
            }
        }

        None
    }

    /// Reset evidence state
    pub fn reset(&mut self) {
        for stream in &mut self.evidence_state {
            for evidence in stream {
                *evidence = 0.0;
            }
        }
    }

    /// Get voting statistics
    pub fn stats(&self) -> VotingStats {
        let recent: Vec<_> = self.vote_history.iter().rev().take(10).collect();
        let count = recent.len().max(1) as f64;

        let avg_consensus = recent
            .iter()
            .map(|r| r.consensus_count as f64 / r.total_voters.max(1) as f64)
            .sum::<f64>()
            / count;
        let avg_confidence = recent.iter().map(|r| r.confidence).sum::<f64>() / count;

        VotingStats {
            avg_consensus_fraction: avg_consensus,
            avg_confidence,
            total_votes: self.vote_history.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VotingStats {
    pub avg_consensus_fraction: f64,
    pub avg_confidence: f64,
    pub total_votes: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUNDING / TRANSFER LEARNING METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Metrics for measuring knowledge compounding and transfer
#[derive(Debug, Clone)]
pub struct CompoundingMetrics {
    /// Learning curves per task/episode
    pub learning_curves: Vec<Vec<f64>>,
    /// Transfer scores: how much prior learning helped
    pub transfer_scores: Vec<f64>,
    /// Knowledge reuse rate
    pub reuse_rates: Vec<f64>,
    /// Abstraction level indicators
    pub abstraction_levels: Vec<f64>,
    /// Forward transfer: did episode N help episode N+K?
    pub forward_transfer: Vec<f64>,
    /// Backward transfer: did episode N+K improve retrospective episode N performance?
    pub backward_transfer: Vec<f64>,
}

impl CompoundingMetrics {
    pub fn new() -> Self {
        Self {
            learning_curves: Vec::new(),
            transfer_scores: Vec::new(),
            reuse_rates: Vec::new(),
            abstraction_levels: Vec::new(),
            forward_transfer: Vec::new(),
            backward_transfer: Vec::new(),
        }
    }

    /// Record a learning curve for an episode
    pub fn record_episode(&mut self, rewards: Vec<f64>) {
        if !rewards.is_empty() {
            self.learning_curves.push(rewards);
            self.compute_transfer_metrics();
        }
    }

    /// Compute transfer learning metrics
    fn compute_transfer_metrics(&mut self) {
        let n = self.learning_curves.len();
        if n < 2 {
            return;
        }

        // Forward transfer: compare initial performance across episodes
        // Higher initial performance in later episodes = positive forward transfer
        let initial_performances: Vec<f64> = self
            .learning_curves
            .iter()
            .map(|curve| curve.first().copied().unwrap_or(0.0))
            .collect();

        if n >= 2 {
            let early_avg = initial_performances[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
            let late_avg =
                initial_performances[n / 2..].iter().sum::<f64>() / ((n - n / 2) as f64).max(1.0);
            let forward = (late_avg - early_avg) / early_avg.abs().max(0.01);
            self.forward_transfer.push(forward);
        }

        // Reuse rate: how quickly performance improves (steeper = more reuse)
        let latest = self.learning_curves.last().unwrap();
        if latest.len() >= 2 {
            let improvement_rate = if latest.len() >= 10 {
                (latest[9] - latest[0]) / 10.0
            } else {
                (latest.last().unwrap() - latest[0]) / latest.len() as f64
            };
            self.reuse_rates.push(improvement_rate);
        }

        // Abstraction level: variance reduction in strategies (lower variance = more abstraction)
        if n >= 5 {
            let recent_finals: Vec<f64> = self.learning_curves[n - 5..]
                .iter()
                .filter_map(|c| c.last().copied())
                .collect();

            if !recent_finals.is_empty() {
                let mean = recent_finals.iter().sum::<f64>() / recent_finals.len() as f64;
                let variance = recent_finals
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / recent_finals.len() as f64;
                let stability = 1.0 / (1.0 + variance);
                self.abstraction_levels.push(stability);
            }
        }

        // Overall transfer score
        let transfer = if !self.forward_transfer.is_empty() && !self.reuse_rates.is_empty() {
            let ft = self.forward_transfer.last().unwrap_or(&0.0);
            let rr = self.reuse_rates.last().unwrap_or(&0.0);
            (ft + rr) / 2.0
        } else {
            0.0
        };
        self.transfer_scores.push(transfer);
    }

    /// Get summary of compounding effectiveness
    pub fn summary(&self) -> CompoundingSummary {
        let avg_transfer = if self.transfer_scores.is_empty() {
            0.0
        } else {
            self.transfer_scores.iter().sum::<f64>() / self.transfer_scores.len() as f64
        };

        let avg_forward = if self.forward_transfer.is_empty() {
            0.0
        } else {
            self.forward_transfer.iter().sum::<f64>() / self.forward_transfer.len() as f64
        };

        let avg_reuse = if self.reuse_rates.is_empty() {
            0.0
        } else {
            self.reuse_rates.iter().sum::<f64>() / self.reuse_rates.len() as f64
        };

        let avg_abstraction = if self.abstraction_levels.is_empty() {
            0.0
        } else {
            self.abstraction_levels.iter().sum::<f64>() / self.abstraction_levels.len() as f64
        };

        // Detect if compounding is occurring
        let compounding_detected = avg_forward > 0.05 || avg_reuse > 0.01;

        CompoundingSummary {
            n_episodes: self.learning_curves.len(),
            avg_transfer_score: avg_transfer,
            avg_forward_transfer: avg_forward,
            avg_reuse_rate: avg_reuse,
            avg_abstraction_level: avg_abstraction,
            compounding_detected,
        }
    }
}

impl Default for CompoundingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CompoundingSummary {
    pub n_episodes: usize,
    pub avg_transfer_score: f64,
    pub avg_forward_transfer: f64,
    pub avg_reuse_rate: f64,
    pub avg_abstraction_level: f64,
    pub compounding_detected: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATED AGI REASONING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrated system combining all AGI-relevant reasoning capabilities
#[derive(Debug)]
pub struct AGIReasoningSystem {
    /// Causal graph for consequential reasoning
    pub causal_graph: CausalGraph,
    /// Multi-step consequence planning
    pub consequence_engine: ConsequentialThinking,
    /// Stream voting for consensus
    pub voting_system: StreamVotingSystem,
    /// Compounding/transfer metrics
    pub compounding_metrics: CompoundingMetrics,
    /// Current episode for tracking
    pub current_episode: usize,
    /// Step counter within episode
    pub current_step: usize,
    /// Episode rewards for metrics
    episode_rewards: Vec<f64>,
}

impl AGIReasoningSystem {
    pub fn new(n_streams: usize, n_actions: usize) -> Self {
        Self {
            causal_graph: CausalGraph::for_navigation(),
            consequence_engine: ConsequentialThinking::new(5, n_actions, 0.95),
            voting_system: StreamVotingSystem::new(n_streams, n_actions, 0.6),
            compounding_metrics: CompoundingMetrics::new(),
            current_episode: 0,
            current_step: 0,
            episode_rewards: Vec::new(),
        }
    }

    /// Process a step: receive observations, reason, vote, decide
    pub fn process_step(
        &mut self,
        state: Vec<f32>,
        stream_evidences: Vec<(usize, Vec<f64>)>, // (stream_id, evidence per action)
        reward: f64,
    ) -> AGIDecision {
        self.current_step += 1;
        self.episode_rewards.push(reward);

        // 1. Update causal graph with observations
        let mut observations = HashMap::new();
        if state.len() >= 2 {
            observations.insert(0, state[0] as f64); // position_x
            observations.insert(1, state[1] as f64); // position_y
        }
        observations.insert(3, reward); // reward
        self.causal_graph.observe(&observations);

        // 2. Receive votes from streams
        for (stream_id, evidences) in stream_evidences {
            for (action, &evidence) in evidences.iter().enumerate() {
                if evidence > 0.1 {
                    self.voting_system.receive_vote(StreamVote {
                        stream_id,
                        hypothesis: action,
                        evidence,
                        supporting_features: state.clone(),
                    });
                }
            }
        }

        // 3. Check for flash inference (quick consensus)
        let flash_action = self.voting_system.flash_inference();

        // 4. Always aggregate votes to record voting activity
        let vote_result = self.voting_system.aggregate_votes();

        // 5. Decide: use flash, voting consensus, or tree search
        let (chosen_action, confidence, method) = if let Some(action) = flash_action {
            (action, 0.9, DecisionMethod::FlashConsensus)
        } else {
            // Build consequence tree for deeper reasoning
            let tree = self.consequence_engine.build_tree(state.clone(), 50);
            let tree_action = self.consequence_engine.best_action(&tree);

            // Combine: prefer voting if consensus, else use tree
            if vote_result.consensus_reached {
                (
                    vote_result.winner,
                    vote_result.confidence,
                    DecisionMethod::VotingConsensus,
                )
            } else if let Some(ta) = tree_action {
                (
                    ta,
                    tree.children.get(ta).map(|n| n.value).unwrap_or(0.5),
                    DecisionMethod::TreeSearch,
                )
            } else {
                (
                    vote_result.winner,
                    vote_result.confidence.max(0.2), // Ensure minimum confidence
                    DecisionMethod::LowConfidence,
                )
            }
        };

        // 5. Decay evidence for next step
        self.voting_system.decay_evidence();

        // 6. Perform counterfactual: "What if I had chosen differently?"
        let counterfactual = if chosen_action > 0 {
            Some(
                self.causal_graph
                    .counterfactual(3, 2, (chosen_action - 1) as f64),
            )
        } else {
            None
        };

        AGIDecision {
            action: chosen_action,
            confidence,
            method,
            counterfactual_insight: counterfactual
                .map(|cf| cf.counterfactual_value - cf.actual_value),
            step: self.current_step,
        }
    }

    /// End of episode: consolidate learning
    pub fn end_episode(&mut self) {
        // Record episode for compounding metrics
        self.compounding_metrics
            .record_episode(self.episode_rewards.clone());

        // Reset for next episode
        self.episode_rewards.clear();
        self.current_step = 0;
        self.current_episode += 1;
        self.voting_system.reset();
    }

    /// Learn from observed outcome
    pub fn learn(&mut self, state: &[f32], action: usize, next_state: &[f32], reward: f64) {
        // Update causal graph with observations
        let mut outcomes = HashMap::new();
        if next_state.len() >= 2 {
            outcomes.insert(0, next_state[0] as f64); // position_x
            outcomes.insert(1, next_state[1] as f64); // position_y
        }
        outcomes.insert(3, reward); // reward
        self.causal_graph.learn_from_outcome(&outcomes);

        // Perform causal intervention: record the action we took as do(action=X)
        // This builds the causal model of how actions affect outcomes
        let predicted = self
            .causal_graph
            .intervene(2, action as f64, self.current_step);

        // After observing the actual outcome, update mechanism strengths
        // This is where the causal model learns from experience
        if let Some(&predicted_reward) = predicted.get(&3) {
            let prediction_error = (reward - predicted_reward).abs();
            // Strengthen/weaken mechanisms based on prediction accuracy
            self.causal_graph
                .update_mechanism_confidence(2, 3, 1.0 / (1.0 + prediction_error));
        }

        // Update transition model for MCTS
        self.consequence_engine
            .update_model(state, action, next_state, reward);
    }

    /// Get comprehensive summary
    pub fn summary(&self) -> AGIReasoningSummary {
        AGIReasoningSummary {
            causal: self.causal_graph.summary(),
            voting: self.voting_system.stats(),
            compounding: self.compounding_metrics.summary(),
            total_steps: self.current_step,
            total_episodes: self.current_episode,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecisionMethod {
    FlashConsensus,
    VotingConsensus,
    TreeSearch,
    LowConfidence,
}

#[derive(Debug, Clone)]
pub struct AGIDecision {
    pub action: usize,
    pub confidence: f64,
    pub method: DecisionMethod,
    pub counterfactual_insight: Option<f64>,
    pub step: usize,
}

#[derive(Debug)]
pub struct AGIReasoningSummary {
    pub causal: CausalGraphSummary,
    pub voting: VotingStats,
    pub compounding: CompoundingSummary,
    pub total_steps: usize,
    pub total_episodes: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_graph_creation() {
        let graph = CausalGraph::for_navigation();
        assert_eq!(graph.variables.len(), 7);
        assert!(graph.mechanisms.len() > 5);
    }

    #[test]
    fn test_causal_intervention() {
        let mut graph = CausalGraph::for_navigation();
        let outcomes = graph.intervene(2, 1.0, 0); // do(action = 1)
        assert!(!outcomes.is_empty());
    }

    #[test]
    fn test_voting_system() {
        let mut voting = StreamVotingSystem::new(8, 4, 0.6);

        // All streams vote for action 2
        for stream_id in 0..8 {
            voting.receive_vote(StreamVote {
                stream_id,
                hypothesis: 2,
                evidence: 0.8,
                supporting_features: vec![0.5; 32],
            });
        }

        let result = voting.aggregate_votes();
        assert_eq!(result.winner, 2);
        assert!(result.consensus_reached);
    }

    #[test]
    fn test_compounding_metrics() {
        let mut metrics = CompoundingMetrics::new();

        // Simulate improving episodes
        metrics.record_episode(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        metrics.record_episode(vec![0.2, 0.3, 0.4, 0.5, 0.6]);
        metrics.record_episode(vec![0.3, 0.4, 0.5, 0.6, 0.7]);

        let summary = metrics.summary();
        assert_eq!(summary.n_episodes, 3);
        // Forward transfer should be positive (later episodes start better)
        assert!(summary.avg_forward_transfer >= 0.0);
    }

    #[test]
    fn test_consequence_tree() {
        let mut engine = ConsequentialThinking::new(3, 4, 0.95);
        let state = vec![0.5, 0.5, 0.0];
        let tree = engine.build_tree(state, 20);

        assert!(tree.visits > 0);
        // Should have children for each action
        assert!(!tree.children.is_empty());
    }

    #[test]
    fn test_agi_reasoning_system() {
        let mut system = AGIReasoningSystem::new(8, 4);

        // Simulate a step
        let state = vec![5.0, 5.0, 0.0];
        let evidences: Vec<(usize, Vec<f64>)> =
            (0..8).map(|s| (s, vec![0.2, 0.3, 0.8, 0.1])).collect();

        let decision = system.process_step(state, evidences, 0.5);

        assert!(decision.confidence > 0.0);
        assert!(decision.action < 4);
    }
}
