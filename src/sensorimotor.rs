//! # Sensorimotor Integration for Full Compounding Closure
//!
//! This module completes the compounding cognitive cohesion system by adding
//! sensorimotor closure - the ability to execute actions based on goal states
//! and receive new observations that feed back into the transformer.
//!
//! ## The Full Loop
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    SENSORIMOTOR COMPOUNDING LOOP                        │
//! │                                                                         │
//! │  Observation                                                            │
//! │      │                                                                  │
//! │      ▼                                                                  │
//! │  ┌─────────────────────────────────────────────┐                       │
//! │  │     CompoundingCohesionTransformer          │                       │
//! │  │  ┌─────────────────────────────────────┐    │                       │
//! │  │  │ Hierarchical Coherence              │    │                       │
//! │  │  │ Cross-Layer SMM                     │    │                       │
//! │  │  │ Predictive Coherence                │    │                       │
//! │  │  │ Graph Memory                        │    │                       │
//! │  │  └─────────────────────────────────────┘    │                       │
//! │  └──────────────────┬──────────────────────────┘                       │
//! │                     │                                                   │
//! │                     ▼                                                   │
//! │              GoalState Proposal                                         │
//! │                     │                                                   │
//! │                     ▼                                                   │
//! │  ┌─────────────────────────────────────────────┐                       │
//! │  │          MotorSystem                        │                       │
//! │  │  ┌─────────────────────────────────────┐    │                       │
//! │  │  │ Policy Selection                    │    │                       │
//! │  │  │ Action Generation                   │    │                       │
//! │  │  │ Actuator Interface                  │    │                       │
//! │  │  └─────────────────────────────────────┘    │                       │
//! │  └──────────────────┬──────────────────────────┘                       │
//! │                     │                                                   │
//! │                     ▼                                                   │
//! │                  Action                                                 │
//! │                     │                                                   │
//! │                     ▼                                                   │
//! │  ┌─────────────────────────────────────────────┐                       │
//! │  │          Environment                        │                       │
//! │  │  ┌─────────────────────────────────────┐    │                       │
//! │  │  │ State Transition                    │    │                       │
//! │  │  │ Observation Generation              │    │                       │
//! │  │  │ Reward/Feedback                     │    │                       │
//! │  │  └─────────────────────────────────────┘    │                       │
//! │  └──────────────────┬──────────────────────────┘                       │
//! │                     │                                                   │
//! │                     ▼                                                   │
//! │            New Observation ──────────────────────────────► (loop back) │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Monty-Inspired Design
//!
//! Following the Monty framework:
//! - **SensorModule**: Processes raw observations into features
//! - **MotorPolicy**: Converts goal states to actions
//! - **Environment**: Provides state transitions and observations
//! - **Agent**: Coordinates the full sensorimotor loop
//!
//! ## Usage
//!
//! ```rust,ignore
//! use torus_attention::sensorimotor::{
//!     SensorimotorAgent, SimpleEnvironment, MotorSystem,
//! };
//!
//! // Create agent with transformer and environment
//! let mut agent = SensorimotorAgent::new(transformer, environment, config);
//!
//! // Run episode
//! let episode_result = agent.run_episode(max_steps)?;
//!
//! // Access compounding statistics
//! println!("{}", agent.summary());
//! ```

use crate::compounding_cohesion::{GoalState, GoalType};
use crate::compounding_transformer::CompoundingCohesionTransformer;
use crate::consequential::{AGIDecision, AGIReasoningSystem, CompoundingMetrics, StreamVote};
use crate::TorusResult;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION AND ACTION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A sensory observation from the environment
#[derive(Debug, Clone)]
pub struct Observation {
    /// Raw feature tensor [batch, seq_len, d_model]
    pub features: Tensor,
    /// Optional pose/location information
    pub pose: Option<Pose3D>,
    /// Timestamp or step number
    pub timestep: usize,
    /// Whether this is a terminal observation
    pub terminal: bool,
    /// Additional metadata
    pub metadata: ObservationMetadata,
}

impl Observation {
    pub fn new(features: Tensor, timestep: usize) -> Self {
        Self {
            features,
            pose: None,
            timestep,
            terminal: false,
            metadata: ObservationMetadata::default(),
        }
    }

    pub fn with_pose(mut self, pose: Pose3D) -> Self {
        self.pose = Some(pose);
        self
    }

    pub fn terminal(mut self) -> Self {
        self.terminal = true;
        self
    }
}

/// Metadata for observations
#[derive(Debug, Clone, Default)]
pub struct ObservationMetadata {
    /// Source sensor ID
    pub sensor_id: Option<usize>,
    /// Confidence in observation
    pub confidence: f64,
    /// Whether observation is noisy
    pub noisy: bool,
}

/// 3D pose (position + orientation)
#[derive(Debug, Clone, Copy, Default)]
pub struct Pose3D {
    /// X position
    pub x: f64,
    /// Y position
    pub y: f64,
    /// Z position
    pub z: f64,
    /// Roll angle (rotation around X)
    pub roll: f64,
    /// Pitch angle (rotation around Y)
    pub pitch: f64,
    /// Yaw angle (rotation around Z)
    pub yaw: f64,
}

impl Pose3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    pub fn with_orientation(mut self, roll: f64, pitch: f64, yaw: f64) -> Self {
        self.roll = roll;
        self.pitch = pitch;
        self.yaw = yaw;
        self
    }

    /// Compute displacement to another pose
    pub fn displacement_to(&self, other: &Pose3D) -> Pose3D {
        Pose3D {
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
            roll: other.roll - self.roll,
            pitch: other.pitch - self.pitch,
            yaw: other.yaw - self.yaw,
        }
    }

    /// Apply displacement to get new pose
    pub fn apply_displacement(&self, displacement: &Pose3D) -> Pose3D {
        Pose3D {
            x: self.x + displacement.x,
            y: self.y + displacement.y,
            z: self.z + displacement.z,
            roll: self.roll + displacement.roll,
            pitch: self.pitch + displacement.pitch,
            yaw: self.yaw + displacement.yaw,
        }
    }

    /// Euclidean distance to another pose (position only)
    pub fn distance_to(&self, other: &Pose3D) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// An action to be executed in the environment
#[derive(Debug, Clone)]
pub struct Action {
    /// Action type
    pub action_type: ActionType,
    /// Target pose (for movement actions)
    pub target_pose: Option<Pose3D>,
    /// Action parameters (generic)
    pub parameters: Vec<f64>,
    /// Confidence in action
    pub confidence: f64,
    /// Source goal state
    pub source_goal: GoalType,
}

impl Action {
    pub fn noop() -> Self {
        Self {
            action_type: ActionType::NoOp,
            target_pose: None,
            parameters: vec![],
            confidence: 1.0,
            source_goal: GoalType::None,
        }
    }

    pub fn move_to(target: Pose3D, confidence: f64) -> Self {
        Self {
            action_type: ActionType::MoveTo,
            target_pose: Some(target),
            parameters: vec![],
            confidence,
            source_goal: GoalType::Explore,
        }
    }

    pub fn look_at(target: Pose3D, confidence: f64) -> Self {
        Self {
            action_type: ActionType::LookAt,
            target_pose: Some(target),
            parameters: vec![],
            confidence,
            source_goal: GoalType::Refine,
        }
    }

    pub fn sample(direction: Vec<f64>, confidence: f64) -> Self {
        Self {
            action_type: ActionType::Sample,
            target_pose: None,
            parameters: direction,
            confidence,
            source_goal: GoalType::Disambiguate,
        }
    }
}

/// Types of actions the motor system can produce
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// No action
    NoOp,
    /// Move to a target pose
    MoveTo,
    /// Look at / orient toward a target
    LookAt,
    /// Sample/probe at current location
    Sample,
    /// Move tangentially along surface
    MoveTangential,
    /// Move along principal curvature
    MoveCurvature,
    /// Random exploration
    RandomExplore,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOTOR POLICIES (Monty-inspired)
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for motor policies that convert goal states to actions
pub trait MotorPolicy: std::fmt::Debug {
    /// Generate action from goal state and current observation
    fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action;

    /// Update policy based on feedback
    fn update(&mut self, action: &Action, result: &ActionResult);

    /// Reset policy state
    fn reset(&mut self);

    /// Get policy name
    fn name(&self) -> &str;
}

/// Result of executing an action
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// Whether action succeeded
    pub success: bool,
    /// New observation after action
    pub observation: Option<Observation>,
    /// Reward signal (if applicable)
    pub reward: f64,
    /// Whether episode terminated
    pub done: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ActionResult {
    pub fn success(observation: Observation, reward: f64) -> Self {
        let done = observation.terminal;
        Self {
            success: true,
            observation: Some(observation),
            reward,
            done,
            error: None,
        }
    }

    pub fn failure(error: &str) -> Self {
        Self {
            success: false,
            observation: None,
            reward: 0.0,
            done: false,
            error: Some(error.to_string()),
        }
    }

    pub fn terminal(observation: Observation, reward: f64) -> Self {
        Self {
            success: true,
            observation: Some(observation),
            reward,
            done: true,
            error: None,
        }
    }
}

/// Simple reactive policy (model-free)
#[derive(Debug)]
pub struct ReactivePolicy {
    /// Step size for movements
    pub step_size: f64,
    /// Exploration noise
    pub exploration_noise: f64,
    /// History of recent actions
    pub action_history: VecDeque<Action>,
    /// Maximum history length
    pub max_history: usize,
}

impl Default for ReactivePolicy {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            exploration_noise: 0.05,
            action_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }
}

impl MotorPolicy for ReactivePolicy {
    fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action {
        let action = match goal.goal_type {
            GoalType::None => Action::noop(),

            GoalType::Explore => {
                // Random exploration movement
                let current_pose = observation.pose.unwrap_or_default();
                let noise = self.exploration_noise;
                let target = Pose3D::new(
                    current_pose.x + self.step_size * (1.0 + rand_f64() * noise),
                    current_pose.y + self.step_size * (rand_f64() - 0.5) * noise,
                    current_pose.z + self.step_size * (rand_f64() - 0.5) * noise,
                );
                Action::move_to(target, goal.confidence)
            }

            GoalType::Refine => {
                // Stay in place but reorient
                let current_pose = observation.pose.unwrap_or_default();
                let target = current_pose.with_orientation(
                    current_pose.roll + self.exploration_noise * (rand_f64() - 0.5),
                    current_pose.pitch + self.exploration_noise * (rand_f64() - 0.5),
                    current_pose.yaw,
                );
                Action::look_at(target, goal.confidence)
            }

            GoalType::Disambiguate => {
                // Sample in multiple directions
                let directions = vec![rand_f64() - 0.5, rand_f64() - 0.5, rand_f64() - 0.5];
                Action::sample(directions, goal.confidence)
            }
        };

        // Store in history
        self.action_history.push_back(action.clone());
        if self.action_history.len() > self.max_history {
            self.action_history.pop_front();
        }

        action
    }

    fn update(&mut self, _action: &Action, _result: &ActionResult) {
        // Reactive policy doesn't learn
    }

    fn reset(&mut self) {
        self.action_history.clear();
    }

    fn name(&self) -> &str {
        "ReactivePolicy"
    }
}

/// Goal-directed policy that uses coherence to guide actions
#[derive(Debug)]
pub struct CoherenceGuidedPolicy {
    /// Base reactive policy
    pub base_policy: ReactivePolicy,
    /// Coherence threshold for confident actions
    pub coherence_threshold: f64,
    /// Learning rate for policy updates
    pub learning_rate: f64,
    /// Success history for adaptation
    pub success_history: VecDeque<bool>,
}

impl CoherenceGuidedPolicy {
    pub fn new(coherence_threshold: f64) -> Self {
        Self {
            base_policy: ReactivePolicy::default(),
            coherence_threshold,
            learning_rate: 0.01,
            success_history: VecDeque::with_capacity(100),
        }
    }
}

impl MotorPolicy for CoherenceGuidedPolicy {
    fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action {
        // Modulate action confidence based on uncertainty source
        let confidence_mod = 1.0 - goal.uncertainty_source;

        let mut action = self.base_policy.generate_action(goal, observation);

        // If coherence is low, increase exploration
        if goal.uncertainty_source > (1.0 - self.coherence_threshold) {
            self.base_policy.exploration_noise *= 1.5;
            action = self.base_policy.generate_action(goal, observation);
            self.base_policy.exploration_noise /= 1.5;
        }

        action.confidence *= confidence_mod;
        action
    }

    fn update(&mut self, action: &Action, result: &ActionResult) {
        self.success_history.push_back(result.success);
        if self.success_history.len() > 100 {
            self.success_history.pop_front();
        }

        // Adapt step size based on success rate
        let success_rate = self.success_history.iter().filter(|&&s| s).count() as f64
            / self.success_history.len().max(1) as f64;

        if success_rate < 0.3 {
            // Too many failures, reduce step size
            self.base_policy.step_size *= 0.9;
        } else if success_rate > 0.8 {
            // Very successful, can be more aggressive
            self.base_policy.step_size *= 1.1;
        }

        self.base_policy.step_size = self.base_policy.step_size.clamp(0.01, 1.0);
    }

    fn reset(&mut self) {
        self.base_policy.reset();
        self.success_history.clear();
    }

    fn name(&self) -> &str {
        "CoherenceGuidedPolicy"
    }
}

/// Hypothesis-testing policy (model-based, Monty-style)
#[derive(Debug)]
pub struct HypothesisTestingPolicy {
    /// Base policy
    pub base_policy: ReactivePolicy,
    /// Hypotheses being tested (stream_id, confidence)
    pub hypotheses: Vec<(usize, f64)>,
    /// Diagnostic locations to visit
    pub diagnostic_queue: VecDeque<Pose3D>,
    /// Current hypothesis index
    pub current_hypothesis: usize,
}

impl HypothesisTestingPolicy {
    pub fn new() -> Self {
        Self {
            base_policy: ReactivePolicy::default(),
            hypotheses: vec![],
            diagnostic_queue: VecDeque::new(),
            current_hypothesis: 0,
        }
    }

    /// Add a hypothesis to test
    pub fn add_hypothesis(&mut self, stream_id: usize, confidence: f64) {
        self.hypotheses.push((stream_id, confidence));
    }

    /// Add a diagnostic location to visit
    pub fn add_diagnostic_location(&mut self, pose: Pose3D) {
        self.diagnostic_queue.push_back(pose);
    }
}

impl Default for HypothesisTestingPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl MotorPolicy for HypothesisTestingPolicy {
    fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action {
        // If we have diagnostic locations queued, visit them
        if let Some(target) = self.diagnostic_queue.pop_front() {
            return Action::move_to(target, goal.confidence);
        }

        // If goal is to disambiguate and we have hypotheses, test them
        if goal.goal_type == GoalType::Disambiguate && !self.hypotheses.is_empty() {
            // Find the two most confident hypotheses
            let mut sorted_hyps = self.hypotheses.clone();
            sorted_hyps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if sorted_hyps.len() >= 2 {
                // Generate diagnostic action to distinguish between top hypotheses
                let directions = vec![
                    sorted_hyps[0].0 as f64 / 8.0, // Normalized stream ID
                    sorted_hyps[1].0 as f64 / 8.0,
                    (sorted_hyps[0].1 - sorted_hyps[1].1), // Confidence difference
                ];
                return Action::sample(directions, goal.confidence);
            }
        }

        // Fall back to base policy
        self.base_policy.generate_action(goal, observation)
    }

    fn update(&mut self, _action: &Action, result: &ActionResult) {
        // Update hypothesis confidences based on result
        if result.success {
            // Increase confidence in current hypothesis
            if self.current_hypothesis < self.hypotheses.len() {
                self.hypotheses[self.current_hypothesis].1 *= 1.1;
                self.hypotheses[self.current_hypothesis].1 =
                    self.hypotheses[self.current_hypothesis].1.min(1.0);
            }
        } else {
            // Decrease confidence
            if self.current_hypothesis < self.hypotheses.len() {
                self.hypotheses[self.current_hypothesis].1 *= 0.9;
            }
        }

        // Move to next hypothesis if confidence is low
        if self.current_hypothesis < self.hypotheses.len()
            && self.hypotheses[self.current_hypothesis].1 < 0.3
        {
            self.current_hypothesis += 1;
        }
    }

    fn reset(&mut self) {
        self.base_policy.reset();
        self.hypotheses.clear();
        self.diagnostic_queue.clear();
        self.current_hypothesis = 0;
    }

    fn name(&self) -> &str {
        "HypothesisTestingPolicy"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY CONTEXT (Passed from cohesion system to motor)
// ═══════════════════════════════════════════════════════════════════════════════

/// Memory context from the cohesion system to guide actions
#[derive(Debug, Clone)]
pub struct MemoryContext {
    /// Current global coherence
    pub global_coherence: f64,
    /// Prediction error (expectation violation)
    pub prediction_error: f64,
    /// Number of graph nodes (memory richness)
    pub graph_nodes: usize,
    /// Most similar remembered state (if any)
    pub nearest_memory_similarity: f64,
    /// Suggested direction from memory (displacement to rewarding state)
    pub memory_suggested_direction: Option<(f64, f64)>,
    /// Coherence trend (positive = improving, negative = degrading)
    pub coherence_trend: f64,
    /// Current goal from cohesion system
    pub current_goal: GoalType,
}

impl Default for MemoryContext {
    fn default() -> Self {
        Self {
            global_coherence: 0.5,
            prediction_error: 0.0,
            graph_nodes: 0,
            nearest_memory_similarity: 0.0,
            memory_suggested_direction: None,
            coherence_trend: 0.0,
            current_goal: GoalType::Explore,
        }
    }
}

/// Memory-guided policy that uses graph memory and coherence for decisions
#[derive(Debug)]
pub struct MemoryGuidedPolicy {
    /// Base policy for fallback
    pub base_policy: ReactivePolicy,
    /// Current memory context
    pub memory_context: MemoryContext,
    /// Exploration bonus for low-coherence states
    pub exploration_bonus: f64,
    /// Exploitation bonus for high-coherence states  
    pub exploitation_bonus: f64,
    /// History of rewards per action type
    pub action_rewards: std::collections::HashMap<ActionType, (f64, usize)>,
}

impl MemoryGuidedPolicy {
    pub fn new() -> Self {
        Self {
            base_policy: ReactivePolicy::default(),
            memory_context: MemoryContext::default(),
            exploration_bonus: 0.3,
            exploitation_bonus: 0.5,
            action_rewards: std::collections::HashMap::new(),
        }
    }

    /// Update memory context from cohesion system
    pub fn update_memory_context(&mut self, context: MemoryContext) {
        self.memory_context = context;
    }

    /// Get best action type based on learned rewards
    fn best_action_type(&self) -> Option<ActionType> {
        self.action_rewards
            .iter()
            .filter(|(_, (_, count))| *count > 5) // Need enough samples
            .max_by(|a, b| {
                let avg_a = a.1 .0 / a.1 .1.max(1) as f64;
                let avg_b = b.1 .0 / b.1 .1.max(1) as f64;
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(action_type, _)| *action_type)
    }
}

impl Default for MemoryGuidedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl MotorPolicy for MemoryGuidedPolicy {
    fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action {
        let current_pose = observation.pose.unwrap_or(Pose3D::new(0.0, 0.0, 0.0));
        let ctx = &self.memory_context;

        // Decision 1: If we have memory-suggested direction and decent coherence, follow it
        if let Some((dx, dy)) = ctx.memory_suggested_direction {
            if ctx.global_coherence > 0.20 && ctx.nearest_memory_similarity > 0.2 {
                // Trust memory - move toward suggested direction
                let target = Pose3D::new(
                    current_pose.x + dx * self.base_policy.step_size,
                    current_pose.y + dy * self.base_policy.step_size,
                    current_pose.z,
                );
                return Action::move_to(target, goal.confidence * ctx.global_coherence);
            }
        }

        // Decision 2: Explore/exploit based on coherence and goal
        let should_explore = ctx.global_coherence < 0.20 ||  // Lower threshold for more exploitation
            ctx.prediction_error > 0.15 ||    // Higher threshold - more tolerant of prediction errors
            ctx.coherence_trend < -0.02 ||    // Stronger degradation threshold
            goal.goal_type == GoalType::Explore;

        if should_explore {
            // Exploration mode: random movements with larger steps
            let noise_x = (rand_f64() - 0.5) * 2.0 * self.exploration_bonus;
            let noise_y = (rand_f64() - 0.5) * 2.0 * self.exploration_bonus;
            let target = Pose3D::new(
                current_pose.x + noise_x * self.base_policy.step_size * 1.5,
                current_pose.y + noise_y * self.base_policy.step_size * 1.5,
                current_pose.z,
            );
            return Action::move_to(target, goal.confidence * 0.5);
        }

        // Decision 3: Exploit based on learned action rewards
        if let Some(best_action) = self.best_action_type() {
            match best_action {
                ActionType::MoveTo => {
                    // Move in a direction that has worked before
                    let dx = (rand_f64() - 0.5) * self.base_policy.step_size;
                    let dy = (rand_f64() - 0.5) * self.base_policy.step_size;
                    let target =
                        Pose3D::new(current_pose.x + dx, current_pose.y + dy, current_pose.z);
                    return Action::move_to(target, goal.confidence);
                }
                ActionType::LookAt => {
                    let target = Pose3D::new(
                        current_pose.x + rand_f64() * 2.0 - 1.0,
                        current_pose.y + rand_f64() * 2.0 - 1.0,
                        current_pose.z,
                    );
                    return Action::look_at(target, goal.confidence);
                }
                _ => {}
            }
        }

        // Fallback to base policy
        self.base_policy.generate_action(goal, observation)
    }

    fn update(&mut self, action: &Action, result: &ActionResult) {
        // Track reward per action type
        let entry = self
            .action_rewards
            .entry(action.action_type)
            .or_insert((0.0, 0));
        entry.0 += result.reward;
        entry.1 += 1;

        // Update base policy too
        self.base_policy.step_size = if result.reward > 0.0 {
            (self.base_policy.step_size * 1.05).min(1.0)
        } else {
            (self.base_policy.step_size * 0.95).max(0.1)
        };
    }

    fn reset(&mut self) {
        self.base_policy.reset();
        // Don't reset action_rewards - we want to keep learning across episodes
    }

    fn name(&self) -> &str {
        "MemoryGuidedPolicy"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOTOR SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Motor system that coordinates policy selection and action execution
#[derive(Debug)]
pub struct MotorSystem {
    /// Available policies
    policies: Vec<Box<dyn MotorPolicy>>,
    /// Current active policy index
    active_policy: usize,
    /// Action history
    action_history: VecDeque<Action>,
    /// Maximum history size
    max_history: usize,
    /// Total actions executed
    total_actions: usize,
    /// Memory context for memory-guided policy
    memory_context: Option<MemoryContext>,
}

impl MotorSystem {
    pub fn new() -> Self {
        let policies: Vec<Box<dyn MotorPolicy>> = vec![
            Box::new(ReactivePolicy::default()),
            Box::new(CoherenceGuidedPolicy::new(0.6)),
            Box::new(HypothesisTestingPolicy::new()),
            Box::new(MemoryGuidedPolicy::new()),
        ];

        Self {
            policies,
            active_policy: 3, // Default to memory-guided
            action_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            total_actions: 0,
            memory_context: None,
        }
    }

    /// Select policy based on goal type
    pub fn select_policy(&mut self, goal: &GoalState) {
        self.active_policy = match goal.goal_type {
            GoalType::None => 0,         // Reactive
            GoalType::Explore => 3,      // Memory-guided
            GoalType::Refine => 3,       // Memory-guided
            GoalType::Disambiguate => 2, // Hypothesis-testing
        };
    }

    /// Update memory context for memory-guided policy
    pub fn update_memory_context(&mut self, context: MemoryContext) {
        self.memory_context = Some(context);
    }

    /// Generate action using active policy
    pub fn generate_action(&mut self, goal: &GoalState, observation: &Observation) -> Action {
        self.select_policy(goal);

        // If we have memory context and using memory-guided policy, use enhanced generation
        if self.active_policy == 3 {
            if let Some(ctx) = &self.memory_context {
                return self.generate_memory_guided_action(goal, observation, ctx.clone());
            }
        }

        let action = self.policies[self.active_policy].generate_action(goal, observation);

        self.action_history.push_back(action.clone());
        if self.action_history.len() > self.max_history {
            self.action_history.pop_front();
        }
        self.total_actions += 1;

        action
    }

    /// Generate memory-guided action using context from cohesion system
    fn generate_memory_guided_action(
        &mut self,
        goal: &GoalState,
        observation: &Observation,
        ctx: MemoryContext,
    ) -> Action {
        let current_pose = observation.pose.unwrap_or(Pose3D::new(0.0, 0.0, 0.0));

        // Decision 1: If we have memory-suggested direction and decent coherence, follow it
        if let Some((dx, dy)) = ctx.memory_suggested_direction {
            if ctx.global_coherence > 0.20 && ctx.nearest_memory_similarity > 0.2 {
                let target = Pose3D::new(
                    current_pose.x + dx * 0.5,
                    current_pose.y + dy * 0.5,
                    current_pose.z,
                );
                let action = Action::move_to(target, goal.confidence * ctx.global_coherence);
                self.record_action(action.clone());
                return action;
            }
        }

        // Decision 2: Explore/exploit based on coherence
        let should_explore = ctx.global_coherence < 0.20
            || ctx.prediction_error > 0.15
            || ctx.coherence_trend < -0.02
            || goal.goal_type == GoalType::Explore;

        if should_explore {
            // Exploration: random movements with larger steps
            let noise_x = (rand_f64() - 0.5) * 2.0;
            let noise_y = (rand_f64() - 0.5) * 2.0;
            let target = Pose3D::new(
                current_pose.x + noise_x * 0.8,
                current_pose.y + noise_y * 0.8,
                current_pose.z,
            );
            let action = Action::move_to(target, goal.confidence * 0.5);
            self.record_action(action.clone());
            return action;
        }

        // Decision 3: Exploit - use base policy but with coherence-modulated confidence
        let mut action = self.policies[self.active_policy].generate_action(goal, observation);
        action.confidence *= ctx.global_coherence;
        self.record_action(action.clone());
        action
    }

    /// Record action in history
    fn record_action(&mut self, action: Action) {
        self.action_history.push_back(action);
        if self.action_history.len() > self.max_history {
            self.action_history.pop_front();
        }
        self.total_actions += 1;
    }

    /// Update active policy based on result
    pub fn update(&mut self, action: &Action, result: &ActionResult) {
        self.policies[self.active_policy].update(action, result);
    }

    /// Reset all policies
    pub fn reset(&mut self) {
        for policy in &mut self.policies {
            policy.reset();
        }
        self.action_history.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> MotorStats {
        let success_count = self
            .action_history
            .iter()
            .filter(|a| a.confidence > 0.5)
            .count();

        MotorStats {
            total_actions: self.total_actions,
            recent_actions: self.action_history.len(),
            high_confidence_ratio: success_count as f64 / self.action_history.len().max(1) as f64,
            active_policy: self.policies[self.active_policy].name().to_string(),
        }
    }
}

impl Default for MotorSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Motor system statistics
#[derive(Debug, Clone)]
pub struct MotorStats {
    pub total_actions: usize,
    pub recent_actions: usize,
    pub high_confidence_ratio: f64,
    pub active_policy: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENVIRONMENT INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for environments that provide observations and accept actions
pub trait Environment: std::fmt::Debug {
    /// Reset environment to initial state
    fn reset(&mut self) -> TorusResult<Observation>;

    /// Execute action and return result
    fn step(&mut self, action: &Action) -> TorusResult<ActionResult>;

    /// Get current observation without taking action
    fn observe(&self) -> TorusResult<Observation>;

    /// Check if episode is done
    fn is_done(&self) -> bool;

    /// Get environment name
    fn name(&self) -> &str;

    /// Get coverage (fraction of cells visited)
    fn coverage(&self) -> f64 {
        0.0 // Default implementation
    }

    /// Get generic stats as a tuple (coverage, novelty_reward, goal_reward, exploration_reward)
    fn get_stats(&self) -> (f64, f64, f64, f64) {
        (0.0, 0.0, 0.0, 0.0)
    }
}

/// Simple grid-based environment for testing
#[derive(Debug)]
pub struct SimpleGridEnvironment {
    /// Grid size
    pub size: usize,
    /// Current agent position
    pub agent_pos: Pose3D,
    /// Goal position
    pub goal_pos: Pose3D,
    /// Obstacles
    pub obstacles: Vec<Pose3D>,
    /// Current step
    pub step_count: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Feature dimension
    pub feature_dim: usize,
    /// Sequence length (for tensor shape)
    pub seq_len: usize,
    /// Device for tensors
    pub device: Device,
    /// Episode done flag
    done: bool,
}

impl SimpleGridEnvironment {
    pub fn new(size: usize, feature_dim: usize, seq_len: usize, device: &Device) -> Self {
        Self {
            size,
            agent_pos: Pose3D::new(0.0, 0.0, 0.0),
            goal_pos: Pose3D::new(size as f64 - 1.0, size as f64 - 1.0, 0.0),
            obstacles: vec![],
            step_count: 0,
            max_steps: size * size * 2,
            feature_dim,
            seq_len,
            device: device.clone(),
            done: false,
        }
    }

    /// Add obstacle at position
    pub fn add_obstacle(&mut self, pos: Pose3D) {
        self.obstacles.push(pos);
    }

    /// Check if position is valid
    fn is_valid_pos(&self, pos: &Pose3D) -> bool {
        pos.x >= 0.0
            && pos.x < self.size as f64
            && pos.y >= 0.0
            && pos.y < self.size as f64
            && !self.obstacles.iter().any(|o| o.distance_to(pos) < 0.5)
    }

    /// Generate observation tensor from current state
    fn generate_features(&self) -> TorusResult<Tensor> {
        // Create feature tensor encoding agent position, goal, and local surroundings
        let mut features = vec![0.0f32; self.seq_len * self.feature_dim];

        // Encode agent position in first few features
        let pos_idx = 0;
        if pos_idx < self.seq_len * self.feature_dim {
            features[pos_idx] = self.agent_pos.x as f32 / self.size as f32;
            features[pos_idx + 1] = self.agent_pos.y as f32 / self.size as f32;
        }

        // Encode goal direction
        let goal_idx = self.feature_dim;
        if goal_idx + 1 < self.seq_len * self.feature_dim {
            let dx = self.goal_pos.x - self.agent_pos.x;
            let dy = self.goal_pos.y - self.agent_pos.y;
            let dist = (dx * dx + dy * dy).sqrt().max(1e-8);
            features[goal_idx] = (dx / dist) as f32;
            features[goal_idx + 1] = (dy / dist) as f32;
        }

        // Encode distance to goal
        let dist_idx = self.feature_dim * 2;
        if dist_idx < self.seq_len * self.feature_dim {
            let dist = self.agent_pos.distance_to(&self.goal_pos);
            features[dist_idx] = (dist / (self.size as f64 * 1.414)) as f32;
        }

        // Add some spatial encoding across sequence
        for i in 0..self.seq_len {
            let base = i * self.feature_dim;
            if base + 3 < features.len() {
                features[base + 3] = (i as f32 / self.seq_len as f32).sin();
                features[base + 4] = (i as f32 / self.seq_len as f32).cos();
            }
        }

        let tensor = Tensor::from_vec(features, (1, self.seq_len, self.feature_dim), &self.device)?;
        Ok(tensor)
    }

    /// Compute reward for current state
    fn compute_reward(&self) -> f64 {
        let dist = self.agent_pos.distance_to(&self.goal_pos);
        let max_dist = self.size as f64 * 1.414;

        // Reward for being close to goal
        let proximity_reward = 1.0 - (dist / max_dist);

        // Bonus for reaching goal
        let goal_bonus = if dist < 0.5 { 10.0 } else { 0.0 };

        // Small step penalty
        let step_penalty = -0.01;

        proximity_reward + goal_bonus + step_penalty
    }
}

impl Environment for SimpleGridEnvironment {
    fn reset(&mut self) -> TorusResult<Observation> {
        self.agent_pos = Pose3D::new(0.0, 0.0, 0.0);
        self.step_count = 0;
        self.done = false;
        self.observe()
    }

    fn step(&mut self, action: &Action) -> TorusResult<ActionResult> {
        if self.done {
            return Ok(ActionResult::failure("Episode already done"));
        }

        self.step_count += 1;

        // Execute action
        let new_pos = match action.action_type {
            ActionType::MoveTo => action.target_pose.unwrap_or(self.agent_pos),
            ActionType::NoOp => self.agent_pos,
            _ => {
                // Small random movement for other action types
                Pose3D::new(
                    self.agent_pos.x + (rand_f64() - 0.5) * 0.5,
                    self.agent_pos.y + (rand_f64() - 0.5) * 0.5,
                    self.agent_pos.z,
                )
            }
        };

        // Check validity and update position
        if self.is_valid_pos(&new_pos) {
            self.agent_pos = new_pos;
        }

        // Check termination conditions
        let at_goal = self.agent_pos.distance_to(&self.goal_pos) < 0.5;
        let out_of_steps = self.step_count >= self.max_steps;
        self.done = at_goal || out_of_steps;

        let reward = self.compute_reward();
        let observation = self.observe()?;

        if self.done {
            Ok(ActionResult::terminal(observation, reward))
        } else {
            Ok(ActionResult::success(observation, reward))
        }
    }

    fn observe(&self) -> TorusResult<Observation> {
        let features = self.generate_features()?;
        let obs = Observation::new(features, self.step_count).with_pose(self.agent_pos);

        Ok(if self.done { obs.terminal() } else { obs })
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn name(&self) -> &str {
        "SimpleGridEnvironment"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENSORIMOTOR AGENT (Full Integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the sensorimotor agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorimotorConfig {
    /// Maximum steps per episode
    pub max_steps_per_episode: usize,
    /// Whether to update transformer during episode
    pub online_learning: bool,
    /// Reward discount factor
    pub gamma: f64,
    /// How often to consolidate memory (steps)
    pub consolidation_interval: usize,
    /// Whether to log detailed statistics
    pub verbose: bool,
}

impl Default for SensorimotorConfig {
    fn default() -> Self {
        Self {
            max_steps_per_episode: 1000,
            online_learning: true,
            gamma: 0.99,
            consolidation_interval: 100,
            verbose: false,
        }
    }
}

/// Complete sensorimotor agent with transformer, motor system, and environment
pub struct SensorimotorAgent {
    /// The compounding cohesion transformer
    pub transformer: CompoundingCohesionTransformer,
    /// Motor system for action generation
    pub motor_system: MotorSystem,
    /// Environment
    pub environment: Box<dyn Environment>,
    /// Configuration
    pub config: SensorimotorConfig,
    /// Episode count
    pub episode_count: usize,
    /// Total steps across all episodes
    pub total_steps: usize,
    /// Cumulative reward
    pub cumulative_reward: f64,
    /// Episode rewards history
    pub episode_rewards: Vec<f64>,
    /// Step-level statistics
    pub step_stats: VecDeque<StepStats>,
    /// AGI Reasoning System (causal graphs, MCTS, voting)
    pub agi_reasoning: Option<AGIReasoningSystem>,
}

impl std::fmt::Debug for SensorimotorAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SensorimotorAgent")
            .field("episode_count", &self.episode_count)
            .field("total_steps", &self.total_steps)
            .field("cumulative_reward", &self.cumulative_reward)
            .finish()
    }
}

/// Statistics for a single step
#[derive(Debug, Clone)]
pub struct StepStats {
    pub step: usize,
    pub action_type: ActionType,
    pub reward: f64,
    pub coherence: f64,
    pub goal_type: GoalType,
}

/// Result of running an episode
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    /// Total steps in episode
    pub steps: usize,
    /// Total reward
    pub total_reward: f64,
    /// Whether goal was reached
    pub success: bool,
    /// Final coherence score
    pub final_coherence: f64,
    /// Graph nodes after consolidation
    pub graph_nodes: usize,
    /// Episode number
    pub episode: usize,
    /// Number of steps where memory provided direction suggestion
    pub memory_guided_steps: usize,
    /// Average memory similarity during episode
    pub avg_memory_similarity: f64,
}

impl SensorimotorAgent {
    /// Create a new sensorimotor agent
    pub fn new(
        transformer: CompoundingCohesionTransformer,
        environment: Box<dyn Environment>,
        config: SensorimotorConfig,
    ) -> Self {
        Self {
            transformer,
            motor_system: MotorSystem::new(),
            environment,
            config,
            episode_count: 0,
            total_steps: 0,
            cumulative_reward: 0.0,
            episode_rewards: Vec::new(),
            step_stats: VecDeque::with_capacity(10000),
            agi_reasoning: None,
        }
    }

    /// Create a new sensorimotor agent with AGI reasoning enabled
    pub fn with_agi_reasoning(
        transformer: CompoundingCohesionTransformer,
        environment: Box<dyn Environment>,
        config: SensorimotorConfig,
        n_streams: usize,
        n_actions: usize,
    ) -> Self {
        Self {
            transformer,
            motor_system: MotorSystem::new(),
            environment,
            config,
            episode_count: 0,
            total_steps: 0,
            cumulative_reward: 0.0,
            episode_rewards: Vec::new(),
            step_stats: VecDeque::with_capacity(10000),
            agi_reasoning: Some(AGIReasoningSystem::new(n_streams, n_actions)),
        }
    }

    /// Enable AGI reasoning on an existing agent
    pub fn enable_agi_reasoning(&mut self, n_streams: usize, n_actions: usize) {
        self.agi_reasoning = Some(AGIReasoningSystem::new(n_streams, n_actions));
    }

    /// Run a single episode
    pub fn run_episode(&mut self) -> TorusResult<EpisodeResult> {
        // Reset environment and motor system
        let mut observation = self.environment.reset()?;
        self.motor_system.reset();
        self.transformer.reset_state()?;

        // Reset episode-level state in cohesion system
        self.transformer.cohesion_mut().reset_episode_graphs();

        // Update adaptive temperature for exploration/exploitation balance
        self.transformer.cohesion_mut().update_temperature();

        let mut episode_reward = 0.0;
        let mut steps = 0;
        let mut last_features: Option<Vec<f32>> = None;
        let mut memory_guided_steps = 0;
        let mut total_memory_similarity = 0.0;

        while !self.environment.is_done() && steps < self.config.max_steps_per_episode {
            // 1. Forward pass through transformer
            let _output = self.transformer.forward(&observation.features)?;

            // 2. Get goal state from coherence system
            let goal = self.transformer.propose_goal();

            // 2.5. Build memory context from cohesion system and pass to motor
            let cohesion = self.transformer.cohesion();

            // Get current position from observation
            let current_pos = observation.pose.unwrap_or_default();
            let position = (current_pos.x, current_pos.y);

            // Query memory for direction suggestion using POSITION-BASED memory
            let (memory_suggested_direction, nearest_memory_similarity) = {
                if let Some((dx, dy, sim)) = cohesion.suggest_direction_from_position(position) {
                    (Some((dx, dy)), sim)
                } else {
                    // Fall back to neural-state-based memory
                    let current_features = cohesion.last_features();
                    if let Some(ref features) = current_features {
                        let similarity = cohesion.nearest_memory_similarity(features);
                        let direction = cohesion
                            .suggest_direction(features)
                            .map(|(dx, dy, _)| (dx, dy));
                        (direction, similarity)
                    } else {
                        (None, 0.0)
                    }
                }
            };

            // Track memory-guided steps
            if memory_suggested_direction.is_some() {
                memory_guided_steps += 1;
            }
            total_memory_similarity += nearest_memory_similarity;

            let memory_context = MemoryContext {
                global_coherence: cohesion.hierarchy.global_coherence(),
                prediction_error: cohesion.prediction.prediction_error,
                graph_nodes: cohesion.stream_graphs.iter().map(|g| g.nodes.len()).sum(),
                nearest_memory_similarity,
                memory_suggested_direction,
                coherence_trend: cohesion.hierarchy.global.trend(),
                current_goal: goal.goal_type,
            };
            self.motor_system.update_memory_context(memory_context);

            // 3. Generate action from motor system
            let mut action = self.motor_system.generate_action(&goal, &observation);

            // 3.5. AGI Reasoning: Use causal graphs, MCTS, and voting to potentially override action
            let _agi_decision = if let Some(ref mut agi) = self.agi_reasoning {
                // Build state vector for AGI reasoning
                let state_vec = vec![
                    current_pos.x as f32,
                    current_pos.y as f32,
                    cohesion.hierarchy.global_coherence() as f32,
                ];

                // Build stream evidences from coherence signals
                // Each stream votes based on its overall coherence and goal alignment
                let stream_evidences: Vec<(usize, Vec<f64>)> = (0..8)
                    .map(|stream_id| {
                        // Use global coherence as base evidence (all streams share global state)
                        let base_evidence = cohesion.hierarchy.global_coherence();

                        // Create evidence for each action (4 directions + stay = 5 actions)
                        let evidences: Vec<f64> = (0..5)
                            .map(|action_idx| {
                                // Bias toward memory-suggested direction if available
                                let memory_bonus =
                                    if let Some((dx, dy)) = memory_suggested_direction {
                                        let action_matches = match action_idx {
                                            0 => dy > 0.0, // Up
                                            1 => dy < 0.0, // Down
                                            2 => dx < 0.0, // Left
                                            3 => dx > 0.0, // Right
                                            _ => false,    // Stay
                                        };
                                        if action_matches {
                                            0.3
                                        } else {
                                            0.0
                                        }
                                    } else {
                                        0.0
                                    };

                                // Add goal-based bias
                                let goal_bonus = match goal.goal_type {
                                    GoalType::Explore => {
                                        if action_idx < 4 {
                                            0.2
                                        } else {
                                            0.0
                                        }
                                    }
                                    GoalType::None => {
                                        if action_idx == 4 {
                                            0.1
                                        } else {
                                            0.0
                                        }
                                    }
                                    _ => 0.1,
                                };

                                (base_evidence * 0.5 + memory_bonus + goal_bonus).min(1.0)
                            })
                            .collect();

                        (stream_id, evidences)
                    })
                    .collect();

                // Process step through AGI reasoning (use 0.0 as placeholder reward, actual learning happens after)
                let decision = agi.process_step(state_vec, stream_evidences, 0.0);

                // If AGI has moderate+ confidence, modify action target (lowered from 0.85)
                // This ensures the AGI reasoning system actually influences behavior
                if decision.confidence > 0.3 && decision.action < 5 {
                    // Map AGI decision to movement direction
                    let (dx, dy) = match decision.action {
                        0 => (0.0, 1.0),  // Up
                        1 => (0.0, -1.0), // Down
                        2 => (-1.0, 0.0), // Left
                        3 => (1.0, 0.0),  // Right
                        _ => (0.0, 0.0),  // Stay
                    };

                    // Override with MoveTo action pointing in AGI-chosen direction
                    action = Action {
                        action_type: ActionType::MoveTo,
                        target_pose: Some(Pose3D::new(
                            current_pos.x + dx * 1.5,
                            current_pos.y + dy * 1.5,
                            0.0,
                        )),
                        parameters: vec![],
                        confidence: decision.confidence,
                        source_goal: goal.goal_type,
                    };
                }

                Some(decision)
            } else {
                None
            };

            // 4. Execute action in environment
            let result = self.environment.step(&action)?;

            // 5. Update motor system with feedback
            self.motor_system.update(&action, &result);

            // 5.5. Update edge rewards in graph memory based on action result
            // Also buffer position-based features for spatial memory
            let new_pos = result
                .observation
                .as_ref()
                .and_then(|o| o.pose)
                .unwrap_or(current_pos);

            // Compute curiosity bonus for the NEW position (encourages exploration)
            let curiosity = self
                .transformer
                .cohesion()
                .curiosity_bonus((new_pos.x, new_pos.y));

            // Get intrinsic motivation bonus (competence, learning progress, empowerment)
            let intrinsic_bonus = self.transformer.cohesion().total_intrinsic_bonus();

            // Augment reward with curiosity bonus + intrinsic motivation (scaled appropriately)
            let curiosity_bonus = curiosity * 0.4; // Scale curiosity contribution
            let intrinsic_contribution = intrinsic_bonus * 0.3; // Scale intrinsic motivation
            let augmented_reward = result.reward + curiosity_bonus + intrinsic_contribution;

            // Buffer with augmented reward (current position -> reward for reaching it)
            self.transformer
                .cohesion_mut()
                .buffer_position_features((new_pos.x, new_pos.y), augmented_reward);

            // Mark transition for eligibility traces (current -> new position)
            self.transformer
                .cohesion_mut()
                .mark_transition((current_pos.x, current_pos.y), (new_pos.x, new_pos.y));

            // Update successor representations for multi-step planning
            self.transformer
                .cohesion_mut()
                .update_successor_representation(
                    (current_pos.x, current_pos.y),
                    (new_pos.x, new_pos.y),
                );

            // Update eligibility traces with reward (TD(λ) style backward propagation)
            self.transformer
                .cohesion_mut()
                .update_traces_with_reward(augmented_reward);

            // Decay traces for next step
            self.transformer.cohesion_mut().decay_traces();

            // Update intrinsic motivation with prediction error
            let pred_error = self.transformer.cohesion().prediction.prediction_error;
            self.transformer
                .cohesion_mut()
                .update_intrinsic_motivation(pred_error);

            // Update edge rewards based on old position (direct update)
            if let Some(ref features) = last_features {
                self.transformer
                    .cohesion_mut()
                    .update_edge_rewards(features, augmented_reward);
            }
            last_features = self.transformer.cohesion().last_features();

            // 5.9. AGI Reasoning: Learn from observed transition
            if let Some(ref mut agi) = self.agi_reasoning {
                let prev_state = vec![
                    current_pos.x as f32,
                    current_pos.y as f32,
                    self.transformer.cohesion().hierarchy.global_coherence() as f32,
                ];
                let next_state = vec![
                    new_pos.x as f32,
                    new_pos.y as f32,
                    self.transformer.cohesion().hierarchy.global_coherence() as f32,
                ];
                // Map action to index based on movement direction
                let action_idx = if let Some(target) = action.target_pose {
                    let dx = target.x - current_pos.x;
                    let dy = target.y - current_pos.y;
                    if dy.abs() > dx.abs() {
                        if dy > 0.0 {
                            0
                        } else {
                            1
                        } // Up or Down
                    } else if dx.abs() > 0.1 {
                        if dx < 0.0 {
                            2
                        } else {
                            3
                        } // Left or Right
                    } else {
                        4 // Stay
                    }
                } else {
                    4 // No target = Stay
                };
                agi.learn(&prev_state, action_idx, &next_state, augmented_reward);
            }

            // 6. Update transformer meta-learning with augmented reward
            if self.config.online_learning {
                self.transformer.meta_update(augmented_reward);
            }

            // 7. Record statistics (use original reward for tracking)
            let coherence = self.transformer.global_coherence();
            self.step_stats.push_back(StepStats {
                step: self.total_steps + steps,
                action_type: action.action_type,
                reward: result.reward,
                coherence,
                goal_type: goal.goal_type,
            });

            episode_reward += result.reward; // Track original reward
            steps += 1;

            // 8. Periodic memory consolidation
            if steps % self.config.consolidation_interval == 0 {
                let _ = self.transformer.cohesion_mut().post_episode_consolidation();
            }

            // 9. Update observation for next iteration
            if let Some(obs) = result.observation {
                observation = obs;
            }

            if self.config.verbose {
                println!(
                    "Step {}: action={:?}, reward={:.3}, coherence={:.3}",
                    steps, action.action_type, result.reward, coherence
                );
            }
        }

        // End of episode: full consolidation
        let consolidation = self.transformer.end_episode()?;

        // AGI Reasoning: End episode consolidation
        if let Some(ref mut agi) = self.agi_reasoning {
            agi.end_episode();
        }

        // Update counters
        self.episode_count += 1;
        self.total_steps += steps;
        self.cumulative_reward += episode_reward;
        self.episode_rewards.push(episode_reward);

        let success = episode_reward > 0.0; // Simple success criterion

        Ok(EpisodeResult {
            steps,
            total_reward: episode_reward,
            success,
            final_coherence: self.transformer.global_coherence(),
            graph_nodes: consolidation.nodes_added,
            episode: self.episode_count,
            memory_guided_steps,
            avg_memory_similarity: if steps > 0 {
                total_memory_similarity / steps as f64
            } else {
                0.0
            },
        })
    }

    /// Run multiple episodes
    pub fn run_episodes(&mut self, n_episodes: usize) -> TorusResult<Vec<EpisodeResult>> {
        let mut results = Vec::with_capacity(n_episodes);
        for _ in 0..n_episodes {
            results.push(self.run_episode()?);
        }
        Ok(results)
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let avg_reward = if self.episode_rewards.is_empty() {
            0.0
        } else {
            self.episode_rewards.iter().sum::<f64>() / self.episode_rewards.len() as f64
        };

        let recent_coherence = self
            .step_stats
            .iter()
            .rev()
            .take(100)
            .map(|s| s.coherence)
            .sum::<f64>()
            / 100.0_f64.min(self.step_stats.len() as f64).max(1.0);

        let agi_summary = if let Some(ref agi) = self.agi_reasoning {
            let summary = agi.summary();
            format!(
                "\n├─ AGI Reasoning:\n\
                 │  ├─ Causal Graph: {} vars, {} mechanisms (conf: {:.3})\n\
                 │  ├─ Voting: avg consensus {:.2}%, confidence {:.3}\n\
                 │  ├─ Compounding: {} episodes, transfer={:.3}\n\
                 │  └─ Detected: {}",
                summary.causal.n_variables,
                summary.causal.n_mechanisms,
                summary.causal.avg_confidence,
                summary.voting.avg_consensus_fraction * 100.0,
                summary.voting.avg_confidence,
                summary.compounding.n_episodes,
                summary.compounding.avg_transfer_score,
                if summary.compounding.compounding_detected {
                    "YES"
                } else {
                    "no"
                }
            )
        } else {
            String::from("\n├─ AGI Reasoning: disabled")
        };

        format!(
            "SensorimotorAgent Summary:\n\
             ├─ Episodes:        {}\n\
             ├─ Total Steps:     {}\n\
             ├─ Avg Reward:      {:.4}\n\
             ├─ Cumulative:      {:.4}\n\
             ├─ Recent Coherence: {:.4}\n\
             ├─ Motor Stats:     {:?}{}\n\
             └─ Transformer:\n{}",
            self.episode_count,
            self.total_steps,
            avg_reward,
            self.cumulative_reward,
            recent_coherence,
            self.motor_system.stats(),
            agi_summary,
            self.transformer.cohesion_summary(),
        )
    }

    /// Get AGI reasoning summary (if enabled)
    pub fn agi_summary(&self) -> Option<String> {
        self.agi_reasoning.as_ref().map(|agi| {
            let s = agi.summary();
            format!(
                "AGI Reasoning Summary:\n\
                 ├─ Episodes: {}, Steps: {}\n\
                 ├─ Causal Graph:\n\
                 │  ├─ Variables: {}\n\
                 │  ├─ Mechanisms: {}\n\
                 │  ├─ Avg Confidence: {:.4}\n\
                 │  └─ Interventions: {}\n\
                 ├─ Voting System:\n\
                 │  ├─ Avg Consensus: {:.2}%\n\
                 │  ├─ Avg Confidence: {:.4}\n\
                 │  └─ Total Votes: {}\n\
                 └─ Compounding Metrics:\n\
                    ├─ Episodes Tracked: {}\n\
                    ├─ Avg Transfer Score: {:.4}\n\
                    ├─ Avg Forward Transfer: {:.4}\n\
                    ├─ Avg Reuse Rate: {:.4}\n\
                    ├─ Avg Abstraction: {:.4}\n\
                    └─ Compounding Detected: {}",
                s.total_episodes,
                s.total_steps,
                s.causal.n_variables,
                s.causal.n_mechanisms,
                s.causal.avg_confidence,
                s.causal.n_interventions,
                s.voting.avg_consensus_fraction * 100.0,
                s.voting.avg_confidence,
                s.voting.total_votes,
                s.compounding.n_episodes,
                s.compounding.avg_transfer_score,
                s.compounding.avg_forward_transfer,
                s.compounding.avg_reuse_rate,
                s.compounding.avg_abstraction_level,
                if s.compounding.compounding_detected {
                    "YES"
                } else {
                    "no"
                }
            )
        })
    }

    /// Check if compounding is working (coherence improving over time)
    pub fn is_compounding_effective(&self) -> bool {
        if self.step_stats.len() < 100 {
            return false;
        }

        // Compare recent coherence to earlier coherence
        let recent: Vec<f64> = self
            .step_stats
            .iter()
            .rev()
            .take(50)
            .map(|s| s.coherence)
            .collect();
        let earlier: Vec<f64> = self
            .step_stats
            .iter()
            .rev()
            .skip(50)
            .take(50)
            .map(|s| s.coherence)
            .collect();

        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let earlier_avg = earlier.iter().sum::<f64>() / earlier.len().max(1) as f64;

        // Compounding is effective if coherence is stable or improving
        recent_avg >= earlier_avg * 0.95
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple pseudo-random number generator (no external dependency)
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    nanos as f64 / u32::MAX as f64
}

/// Seeded pseudo-random number generator for reproducibility
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        // Simple xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Generate random f64 in [min, max)
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// Generate random usize in [0, max)
    pub fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING GRID ENVIRONMENT (Rich Learning Challenges)
// ═══════════════════════════════════════════════════════════════════════════════

/// A landmark in the environment that provides unique features
#[derive(Debug, Clone)]
pub struct Landmark {
    pub id: usize,
    pub pos: Pose3D,
    pub features: Vec<f32>,
    pub reward_bonus: f64,
    pub visited_count: usize,
}

impl Landmark {
    pub fn new(id: usize, pos: Pose3D, feature_dim: usize, rng: &mut SeededRng) -> Self {
        // Generate unique feature signature for this landmark
        let features: Vec<f32> = (0..feature_dim)
            .map(|i| {
                // Combine position-based and random components for unique signatures
                let pos_component = ((pos.x + pos.y) * (i as f64 + 1.0)).sin() as f32;
                let random_component = (rng.next_f64() - 0.5) as f32;
                pos_component * 0.7 + random_component * 0.3
            })
            .collect();

        Self {
            id,
            pos,
            features,
            reward_bonus: 0.1 + rng.next_f64() * 0.2,
            visited_count: 0,
        }
    }
}

/// Configuration for LearningGridEnvironment
#[derive(Debug, Clone)]
pub struct LearningEnvConfig {
    pub size: usize,
    pub feature_dim: usize,
    pub seq_len: usize,
    pub n_landmarks: usize,
    pub n_obstacles: usize,
    pub max_steps: usize,
    pub goal_move_probability: f64,
    pub obstacle_change_probability: f64,
    pub novelty_bonus: f64,
    pub exploration_reward: f64,
    pub cognitive_dissonance_rate: f64,
}

impl Default for LearningEnvConfig {
    fn default() -> Self {
        Self {
            size: 10,
            feature_dim: 64,
            seq_len: 16,
            n_landmarks: 5,
            n_obstacles: 3,
            max_steps: 200,
            goal_move_probability: 0.1,
            obstacle_change_probability: 0.05,
            novelty_bonus: 0.5,
            exploration_reward: 0.02,
            cognitive_dissonance_rate: 0.1,
        }
    }
}

/// Rich learning environment with landmarks, moving goals, and cognitive dissonance
#[derive(Debug)]
pub struct LearningGridEnvironment {
    config: LearningEnvConfig,
    agent_pos: Pose3D,
    goal_pos: Pose3D,
    landmarks: Vec<Landmark>,
    obstacles: Vec<Pose3D>,
    visited_cells: Vec<Vec<bool>>,
    step_count: usize,
    device: Device,
    done: bool,
    rng: SeededRng,
    /// Cognitive dissonance: expected vs actual state mismatch
    dissonance_active: bool,
    dissonance_magnitude: f64,
    /// Track exploration coverage
    cells_visited: usize,
    total_cells: usize,
    /// Previous goal for detecting goal changes
    prev_goal: Pose3D,
    /// Episode statistics
    pub episode_novelty_reward: f64,
    pub episode_goal_reward: f64,
    pub episode_exploration_reward: f64,
}

impl LearningGridEnvironment {
    pub fn new(config: LearningEnvConfig, device: &Device, seed: u64) -> Self {
        let mut rng = SeededRng::new(seed);
        let size = config.size;

        // Generate random starting position
        let agent_pos = Pose3D::new(
            rng.range(0.0, size as f64 / 3.0),
            rng.range(0.0, size as f64 / 3.0),
            0.0,
        );

        // Goal in opposite quadrant
        let goal_pos = Pose3D::new(
            rng.range(size as f64 * 0.6, size as f64 - 1.0),
            rng.range(size as f64 * 0.6, size as f64 - 1.0),
            0.0,
        );

        // Generate landmarks
        let landmarks: Vec<Landmark> = (0..config.n_landmarks)
            .map(|id| {
                let pos = Pose3D::new(
                    rng.range(1.0, size as f64 - 1.0),
                    rng.range(1.0, size as f64 - 1.0),
                    0.0,
                );
                Landmark::new(id, pos, config.feature_dim, &mut rng)
            })
            .collect();

        // Generate obstacles
        let obstacles: Vec<Pose3D> = (0..config.n_obstacles)
            .map(|_| {
                Pose3D::new(
                    rng.range(2.0, size as f64 - 2.0),
                    rng.range(2.0, size as f64 - 2.0),
                    0.0,
                )
            })
            .collect();

        let visited_cells = vec![vec![false; size]; size];
        let total_cells = size * size;

        Self {
            config,
            agent_pos,
            goal_pos,
            landmarks,
            obstacles,
            visited_cells,
            step_count: 0,
            device: device.clone(),
            done: false,
            rng,
            dissonance_active: false,
            dissonance_magnitude: 0.0,
            cells_visited: 0,
            total_cells,
            prev_goal: goal_pos,
            episode_novelty_reward: 0.0,
            episode_goal_reward: 0.0,
            episode_exploration_reward: 0.0,
        }
    }

    /// Check if a position is valid (in bounds and not obstacle)
    fn is_valid_pos(&self, pos: &Pose3D) -> bool {
        pos.x >= 0.0
            && pos.x < self.config.size as f64
            && pos.y >= 0.0
            && pos.y < self.config.size as f64
            && !self.obstacles.iter().any(|o| o.distance_to(pos) < 1.0)
    }

    /// Mark cell as visited and return novelty bonus
    fn visit_cell(&mut self, pos: &Pose3D) -> f64 {
        let x = pos.x.floor() as usize;
        let y = pos.y.floor() as usize;

        if x < self.config.size && y < self.config.size {
            if !self.visited_cells[x][y] {
                self.visited_cells[x][y] = true;
                self.cells_visited += 1;
                return self.config.novelty_bonus;
            }
        }
        0.0
    }

    /// Find nearest landmark and its features
    fn nearest_landmark(&self, pos: &Pose3D) -> Option<&Landmark> {
        self.landmarks
            .iter()
            .filter(|l| l.pos.distance_to(pos) < 3.0)
            .min_by(|a, b| {
                a.pos
                    .distance_to(pos)
                    .partial_cmp(&b.pos.distance_to(pos))
                    .unwrap()
            })
    }

    /// Inject cognitive dissonance (unexpected state change)
    fn maybe_inject_dissonance(&mut self) {
        if self.rng.next_f64() < self.config.cognitive_dissonance_rate {
            self.dissonance_active = true;
            self.dissonance_magnitude = self.rng.range(0.2, 0.8);
        } else {
            self.dissonance_active = false;
            self.dissonance_magnitude = 0.0;
        }
    }

    /// Maybe move the goal (creates need for continuous adaptation)
    fn maybe_move_goal(&mut self) {
        if self.rng.next_f64() < self.config.goal_move_probability {
            self.prev_goal = self.goal_pos;
            // Move goal by a small random amount
            let dx = self.rng.range(-2.0, 2.0);
            let dy = self.rng.range(-2.0, 2.0);
            self.goal_pos = Pose3D::new(
                (self.goal_pos.x + dx).clamp(1.0, self.config.size as f64 - 1.0),
                (self.goal_pos.y + dy).clamp(1.0, self.config.size as f64 - 1.0),
                0.0,
            );
        }
    }

    /// Generate rich observation features
    fn generate_features(&self) -> TorusResult<Tensor> {
        let seq_len = self.config.seq_len;
        let feature_dim = self.config.feature_dim;
        let mut features = vec![0.0f32; seq_len * feature_dim];

        // === Sequence position 0: Agent state ===
        let idx = 0;
        features[idx] = self.agent_pos.x as f32 / self.config.size as f32;
        features[idx + 1] = self.agent_pos.y as f32 / self.config.size as f32;
        features[idx + 2] = self.step_count as f32 / self.config.max_steps as f32;
        features[idx + 3] = self.cells_visited as f32 / self.total_cells as f32;

        // === Sequence position 1: Goal direction ===
        let idx = feature_dim;
        let dx = self.goal_pos.x - self.agent_pos.x;
        let dy = self.goal_pos.y - self.agent_pos.y;
        let dist = (dx * dx + dy * dy).sqrt().max(1e-8);
        features[idx] = (dx / dist) as f32;
        features[idx + 1] = (dy / dist) as f32;
        features[idx + 2] = (dist / (self.config.size as f64 * 1.414)) as f32;

        // Goal change indicator (cognitive dissonance signal)
        if self.prev_goal.distance_to(&self.goal_pos) > 0.5 {
            features[idx + 3] = 1.0; // Goal has moved!
        }

        // === Sequence positions 2-5: Nearest landmarks ===
        let mut landmark_dists: Vec<(usize, f64)> = self
            .landmarks
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.pos.distance_to(&self.agent_pos)))
            .collect();
        landmark_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (seq_pos, (landmark_idx, _dist)) in landmark_dists.iter().take(4).enumerate() {
            let idx = (seq_pos + 2) * feature_dim;
            if idx + feature_dim <= features.len() {
                let landmark = &self.landmarks[*landmark_idx];
                // Copy landmark features (limited to half the feature dim)
                let copy_len = (feature_dim / 2).min(landmark.features.len());
                features[idx..idx + copy_len].copy_from_slice(&landmark.features[..copy_len]);

                // Add relative position
                let rel_dx = landmark.pos.x - self.agent_pos.x;
                let rel_dy = landmark.pos.y - self.agent_pos.y;
                features[idx + copy_len] = rel_dx as f32 / self.config.size as f32;
                features[idx + copy_len + 1] = rel_dy as f32 / self.config.size as f32;
            }
        }

        // === Sequence positions 6-9: Obstacle proximity ===
        for (seq_pos, obstacle) in self.obstacles.iter().take(4).enumerate() {
            let idx = (seq_pos + 6) * feature_dim;
            if idx + 4 <= features.len() {
                let dist = obstacle.distance_to(&self.agent_pos);
                features[idx] = (1.0 - dist / self.config.size as f64).max(0.0) as f32;
                features[idx + 1] =
                    (obstacle.x - self.agent_pos.x) as f32 / self.config.size as f32;
                features[idx + 2] =
                    (obstacle.y - self.agent_pos.y) as f32 / self.config.size as f32;
            }
        }

        // === Cognitive dissonance injection ===
        if self.dissonance_active {
            // Add deterministic noise based on position and step to simulate expectation violation
            for i in 0..features.len() {
                // Deterministic pseudo-random based on index, step, and magnitude
                let noise_seed = (i as f64 * 0.1 + self.step_count as f64 * 0.01).sin();
                if noise_seed.abs() < self.dissonance_magnitude * 0.3 {
                    features[i] += (noise_seed * 0.5) as f32;
                }
            }
        }

        // === Remaining positions: spatial encoding ===
        for i in 10..seq_len {
            let idx = i * feature_dim;
            if idx + 5 <= features.len() {
                features[idx] = (i as f32 / seq_len as f32).sin();
                features[idx + 1] = (i as f32 / seq_len as f32).cos();
                features[idx + 2] = ((i * 2) as f32 / seq_len as f32).sin();
            }
        }

        let tensor = Tensor::from_vec(features, (1, seq_len, feature_dim), &self.device)?;
        Ok(tensor)
    }

    /// Compute comprehensive reward
    fn compute_reward(&mut self) -> f64 {
        let mut reward = 0.0;

        // 1. Goal proximity reward
        let dist = self.agent_pos.distance_to(&self.goal_pos);
        let max_dist = self.config.size as f64 * 1.414;
        let proximity = 1.0 - (dist / max_dist);
        reward += proximity * 0.5;
        self.episode_goal_reward += proximity * 0.5;

        // 2. Goal reached bonus
        if dist < 0.5 {
            reward += 5.0;
            self.episode_goal_reward += 5.0;
        }

        // 3. Novelty reward (visiting new cells)
        let pos = self.agent_pos; // Copy to avoid borrow conflict
        let novelty = self.visit_cell(&pos);
        reward += novelty;
        self.episode_novelty_reward += novelty;

        // 4. Landmark discovery bonus
        let landmark_bonus = self
            .landmarks
            .iter()
            .filter(|l| l.pos.distance_to(&pos) < 1.0 && l.visited_count == 0)
            .map(|l| l.reward_bonus)
            .sum::<f64>();
        if landmark_bonus > 0.0 {
            reward += landmark_bonus;
            self.episode_exploration_reward += landmark_bonus;
        }

        // 5. Exploration reward (coverage bonus)
        let coverage = self.cells_visited as f64 / self.total_cells as f64;
        reward += coverage * self.config.exploration_reward;
        self.episode_exploration_reward += coverage * self.config.exploration_reward;

        // 6. Small step penalty to encourage efficiency
        reward -= 0.01;

        // 7. Cognitive dissonance penalty (creates pressure to resolve)
        if self.dissonance_active {
            reward -= self.dissonance_magnitude * 0.1;
        }

        reward
    }

    /// Get exploration coverage ratio
    pub fn coverage(&self) -> f64 {
        self.cells_visited as f64 / self.total_cells as f64
    }

    /// Get environment statistics
    pub fn stats(&self) -> LearningEnvStats {
        LearningEnvStats {
            steps: self.step_count,
            coverage: self.coverage(),
            landmarks_visited: self
                .landmarks
                .iter()
                .filter(|l| l.visited_count > 0)
                .count(),
            total_landmarks: self.landmarks.len(),
            dissonance_active: self.dissonance_active,
            goal_distance: self.agent_pos.distance_to(&self.goal_pos),
            novelty_reward: self.episode_novelty_reward,
            goal_reward: self.episode_goal_reward,
            exploration_reward: self.episode_exploration_reward,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LearningEnvStats {
    pub steps: usize,
    pub coverage: f64,
    pub landmarks_visited: usize,
    pub total_landmarks: usize,
    pub dissonance_active: bool,
    pub goal_distance: f64,
    pub novelty_reward: f64,
    pub goal_reward: f64,
    pub exploration_reward: f64,
}

impl Environment for LearningGridEnvironment {
    fn reset(&mut self) -> TorusResult<Observation> {
        // Randomize starting position
        self.agent_pos = Pose3D::new(
            self.rng.range(0.0, self.config.size as f64 / 3.0),
            self.rng.range(0.0, self.config.size as f64 / 3.0),
            0.0,
        );

        // Reset visited cells
        for row in &mut self.visited_cells {
            for cell in row {
                *cell = false;
            }
        }
        self.cells_visited = 0;

        // Reset landmark visit counts
        for landmark in &mut self.landmarks {
            landmark.visited_count = 0;
        }

        // Reset episode stats
        self.step_count = 0;
        self.done = false;
        self.dissonance_active = false;
        self.episode_novelty_reward = 0.0;
        self.episode_goal_reward = 0.0;
        self.episode_exploration_reward = 0.0;

        self.observe()
    }

    fn step(&mut self, action: &Action) -> TorusResult<ActionResult> {
        if self.done {
            return Ok(ActionResult::failure("Episode already done"));
        }

        self.step_count += 1;

        // Maybe inject cognitive dissonance
        self.maybe_inject_dissonance();

        // Maybe move goal (creates need for adaptation)
        self.maybe_move_goal();

        // Execute action
        let new_pos = match action.action_type {
            ActionType::MoveTo => {
                // Move toward target or in target direction
                if let Some(target) = action.target_pose {
                    let dx = (target.x - self.agent_pos.x).clamp(-1.5, 1.5);
                    let dy = (target.y - self.agent_pos.y).clamp(-1.5, 1.5);
                    Pose3D::new(self.agent_pos.x + dx, self.agent_pos.y + dy, 0.0)
                } else {
                    // Random movement
                    Pose3D::new(
                        self.agent_pos.x + self.rng.range(-1.0, 1.0),
                        self.agent_pos.y + self.rng.range(-1.0, 1.0),
                        0.0,
                    )
                }
            }
            ActionType::LookAt => {
                // Look doesn't move, but small random drift
                Pose3D::new(
                    self.agent_pos.x + self.rng.range(-0.1, 0.1),
                    self.agent_pos.y + self.rng.range(-0.1, 0.1),
                    self.agent_pos.z,
                )
            }
            ActionType::Sample => {
                // Sample: move a tiny bit in parameter direction
                if action.parameters.len() >= 2 {
                    Pose3D::new(
                        self.agent_pos.x + action.parameters[0] as f64 * 0.3,
                        self.agent_pos.y + action.parameters[1] as f64 * 0.3,
                        self.agent_pos.z,
                    )
                } else {
                    self.agent_pos
                }
            }
            ActionType::RandomExplore => {
                // Large random movement for exploration
                Pose3D::new(
                    self.agent_pos.x + self.rng.range(-2.0, 2.0),
                    self.agent_pos.y + self.rng.range(-2.0, 2.0),
                    0.0,
                )
            }
            ActionType::NoOp => self.agent_pos,
            _ => {
                // Other actions: small random movement
                Pose3D::new(
                    self.agent_pos.x + self.rng.range(-0.5, 0.5),
                    self.agent_pos.y + self.rng.range(-0.5, 0.5),
                    self.agent_pos.z,
                )
            }
        };

        // Update position if valid
        if self.is_valid_pos(&new_pos) {
            self.agent_pos = Pose3D::new(
                new_pos.x.clamp(0.0, self.config.size as f64 - 0.01),
                new_pos.y.clamp(0.0, self.config.size as f64 - 0.01),
                0.0,
            );
        }

        // Update landmark visit counts
        for landmark in &mut self.landmarks {
            if landmark.pos.distance_to(&self.agent_pos) < 1.0 {
                landmark.visited_count += 1;
            }
        }

        // Check termination
        let at_goal = self.agent_pos.distance_to(&self.goal_pos) < 0.5;
        let out_of_steps = self.step_count >= self.config.max_steps;
        self.done = at_goal || out_of_steps;

        let reward = self.compute_reward();
        let observation = self.observe()?;

        if self.done {
            Ok(ActionResult::terminal(observation, reward))
        } else {
            Ok(ActionResult::success(observation, reward))
        }
    }

    fn observe(&self) -> TorusResult<Observation> {
        let features = self.generate_features()?;
        let obs = Observation::new(features, self.step_count).with_pose(self.agent_pos);
        Ok(if self.done { obs.terminal() } else { obs })
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn name(&self) -> &str {
        "LearningGridEnvironment"
    }

    fn coverage(&self) -> f64 {
        self.cells_visited as f64 / self.total_cells as f64
    }

    fn get_stats(&self) -> (f64, f64, f64, f64) {
        (
            self.coverage(),
            self.episode_novelty_reward,
            self.episode_goal_reward,
            self.episode_exploration_reward,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE DISSONANCE SYSTEM (CD-AI)
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks cognitive dissonance state for the agent
#[derive(Debug, Clone)]
pub struct CognitiveDissonanceTracker {
    /// History of prediction errors (expectation vs reality)
    prediction_errors: VecDeque<f64>,
    /// History of coherence drops
    coherence_drops: VecDeque<f64>,
    /// Current dissonance level
    pub dissonance_level: f64,
    /// Threshold for triggering adaptation
    pub adaptation_threshold: f64,
    /// Counter for resolution attempts
    pub resolution_attempts: usize,
    /// Whether currently in dissonance state
    pub in_dissonance: bool,
    max_history: usize,
}

impl CognitiveDissonanceTracker {
    pub fn new(adaptation_threshold: f64) -> Self {
        Self {
            prediction_errors: VecDeque::with_capacity(100),
            coherence_drops: VecDeque::with_capacity(100),
            dissonance_level: 0.0,
            adaptation_threshold,
            resolution_attempts: 0,
            in_dissonance: false,
            max_history: 100,
        }
    }

    /// Record a prediction error (expected vs actual observation)
    pub fn record_prediction_error(&mut self, error: f64) {
        self.prediction_errors.push_back(error);
        if self.prediction_errors.len() > self.max_history {
            self.prediction_errors.pop_front();
        }
        self.update_dissonance();
    }

    /// Record a coherence drop
    pub fn record_coherence_drop(&mut self, drop: f64) {
        self.coherence_drops.push_back(drop);
        if self.coherence_drops.len() > self.max_history {
            self.coherence_drops.pop_front();
        }
        self.update_dissonance();
    }

    /// Update overall dissonance level
    fn update_dissonance(&mut self) {
        let avg_error = if self.prediction_errors.is_empty() {
            0.0
        } else {
            self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
        };

        let avg_drop = if self.coherence_drops.is_empty() {
            0.0
        } else {
            self.coherence_drops.iter().sum::<f64>() / self.coherence_drops.len() as f64
        };

        // Dissonance is combination of prediction error and coherence drops
        self.dissonance_level = (avg_error * 0.6 + avg_drop * 0.4).clamp(0.0, 1.0);

        // Enter dissonance state if above threshold
        let prev_in_dissonance = self.in_dissonance;
        self.in_dissonance = self.dissonance_level > self.adaptation_threshold;

        // Reset resolution counter on new dissonance episode
        if self.in_dissonance && !prev_in_dissonance {
            self.resolution_attempts = 0;
        }
    }

    /// Record a resolution attempt
    pub fn record_resolution_attempt(&mut self) {
        self.resolution_attempts += 1;
    }

    /// Check if dissonance was resolved (level dropped below threshold)
    pub fn is_resolved(&self) -> bool {
        !self.in_dissonance && self.resolution_attempts > 0
    }

    /// Get summary statistics
    pub fn stats(&self) -> DissonanceStats {
        DissonanceStats {
            current_level: self.dissonance_level,
            in_dissonance: self.in_dissonance,
            resolution_attempts: self.resolution_attempts,
            avg_prediction_error: if self.prediction_errors.is_empty() {
                0.0
            } else {
                self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
            },
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.prediction_errors.clear();
        self.coherence_drops.clear();
        self.dissonance_level = 0.0;
        self.in_dissonance = false;
        self.resolution_attempts = 0;
    }
}

#[derive(Debug, Clone)]
pub struct DissonanceStats {
    pub current_level: f64,
    pub in_dissonance: bool,
    pub resolution_attempts: usize,
    pub avg_prediction_error: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose3d() {
        let p1 = Pose3D::new(0.0, 0.0, 0.0);
        let p2 = Pose3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.01);

        let disp = p1.displacement_to(&p2);
        assert!((disp.x - 3.0).abs() < 0.01);
        assert!((disp.y - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_action_creation() {
        let action = Action::noop();
        assert_eq!(action.action_type, ActionType::NoOp);

        let target = Pose3D::new(1.0, 2.0, 3.0);
        let move_action = Action::move_to(target, 0.8);
        assert_eq!(move_action.action_type, ActionType::MoveTo);
        assert!(move_action.target_pose.is_some());
    }

    #[test]
    fn test_reactive_policy() {
        let mut policy = ReactivePolicy::default();
        let goal = GoalState::explore(0, 0.5);
        let features = Tensor::zeros((1, 32, 64), DType::F32, &Device::Cpu).unwrap();
        let obs = Observation::new(features, 0).with_pose(Pose3D::new(0.0, 0.0, 0.0));

        let action = policy.generate_action(&goal, &obs);
        assert_eq!(action.action_type, ActionType::MoveTo);
    }

    #[test]
    fn test_motor_system() {
        let mut motor = MotorSystem::new();
        let goal = GoalState::explore(0, 0.5);
        let features = Tensor::zeros((1, 32, 64), DType::F32, &Device::Cpu).unwrap();
        let obs = Observation::new(features, 0);

        let action = motor.generate_action(&goal, &obs);
        assert!(motor.total_actions > 0);

        let stats = motor.stats();
        assert_eq!(stats.total_actions, 1);
    }

    #[test]
    fn test_simple_grid_environment() {
        let device = Device::Cpu;
        let mut env = SimpleGridEnvironment::new(5, 64, 32, &device);

        let obs = env.reset().unwrap();
        assert!(!obs.terminal);
        assert_eq!(obs.timestep, 0);

        let action = Action::move_to(Pose3D::new(1.0, 0.0, 0.0), 1.0);
        let result = env.step(&action).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_observation() {
        let features = Tensor::zeros((1, 32, 64), DType::F32, &Device::Cpu).unwrap();
        let obs = Observation::new(features, 5)
            .with_pose(Pose3D::new(1.0, 2.0, 3.0))
            .terminal();

        assert!(obs.terminal);
        assert_eq!(obs.timestep, 5);
        assert!(obs.pose.is_some());
    }
}
