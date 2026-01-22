//! # Artificial General Coherence (AGC)
//!
//! A unified framework for psychological coherence, homeostatic equilibrium,
//! and self-directed intelligence. This module integrates:
//!
//! - **Psychological Coherence**: Antonovsky's SOC + cognitive dissonance resolution
//! - **Requisite Variety**: Ashby's Law for regulatory capacity
//! - **Homeostatic Equilibrium**: PID-style negative feedback with allostatic adaptation
//! - **Epistemic Integrity**: Truth-seeking and belief consistency
//! - **Ethical Alignment**: Value coherence and harm avoidance
//! - **Open-Ended Evolution**: Self-directed capability expansion
//!
//! ## Foundational Principles
//!
//! 1. **Compounding Cognitive Cohesion** - Multiplicative interaction between subsystems
//! 2. **Holistic Reasoning** - Decisions consider all subsystem states
//! 3. **Iterative Feedback Loops** - Continuous refinement through experience
//! 4. **Equilibrium Maintenance** - Active homeostasis with adaptive setpoints

use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// PSYCHOLOGICAL COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Psychological coherence integrating SOC with cognitive dissonance
#[derive(Debug, Clone)]
pub struct PsychologicalCoherence {
    /// Sense of Coherence components (Antonovsky)
    pub comprehensibility: f64, // World is structured and predictable
    pub manageability: f64,  // Resources available to meet demands
    pub meaningfulness: f64, // Life's demands are worthy of engagement

    /// Cognitive dissonance state
    pub dissonance_level: f64,
    pub dissonance_sources: Vec<DissonanceSource>,

    /// Coherence history for trend detection
    coherence_history: VecDeque<f64>,

    /// Resolution strategies that have worked
    successful_resolutions: HashMap<DissonanceType, usize>,

    /// Current coherence regulation mode
    pub regulation_mode: RegulationMode,

    /// Coherence gates (control information flow)
    pub gates: CoherenceGates,
}

/// Sources of cognitive dissonance
#[derive(Debug, Clone)]
pub struct DissonanceSource {
    pub source_type: DissonanceType,
    pub magnitude: f64,
    pub persistence: usize, // How many steps it's been active
    pub belief_a: String,
    pub belief_b: String,
}

/// Types of cognitive dissonance
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum DissonanceType {
    /// Prediction didn't match reality
    PredictionError,
    /// Two beliefs contradict
    BeliefConflict,
    /// Action contradicts stated values
    ValueActionGap,
    /// Goal hierarchy inconsistency
    GoalConflict,
    /// World model contradicts observations
    ModelMismatch,
    /// Symbol grounding failure
    SymbolGroundingFailure,
}

/// How the system is currently regulating coherence
#[derive(Debug, Clone, PartialEq)]
pub enum RegulationMode {
    /// Normal operation, maintaining equilibrium
    Homeostatic,
    /// Actively resolving dissonance
    DissonanceResolution,
    /// Adapting to sustained change (allostasis)
    Allostatic,
    /// Emergency mode - protecting core functions
    Protective,
    /// Exploring to gather information
    Exploratory,
}

/// Gates that control information flow based on coherence
#[derive(Debug, Clone)]
pub struct CoherenceGates {
    /// Allow causal discovery to influence abstraction
    pub discovery_to_abstraction: f64,
    /// Allow abstraction to influence symbols
    pub abstraction_to_symbols: f64,
    /// Allow symbols to influence goals
    pub symbols_to_goals: f64,
    /// Allow goals to influence world model
    pub goals_to_world_model: f64,
    /// Allow world model to influence meta-learner
    pub world_model_to_meta: f64,
    /// Allow meta-learner to influence discovery
    pub meta_to_discovery: f64,
}

impl Default for CoherenceGates {
    fn default() -> Self {
        Self {
            discovery_to_abstraction: 1.0,
            abstraction_to_symbols: 1.0,
            symbols_to_goals: 1.0,
            goals_to_world_model: 1.0,
            world_model_to_meta: 1.0,
            meta_to_discovery: 1.0,
        }
    }
}

impl PsychologicalCoherence {
    pub fn new() -> Self {
        Self {
            comprehensibility: 0.5,
            manageability: 0.5,
            meaningfulness: 0.5,
            dissonance_level: 0.0,
            dissonance_sources: Vec::new(),
            coherence_history: VecDeque::with_capacity(100),
            successful_resolutions: HashMap::new(),
            regulation_mode: RegulationMode::Homeostatic,
            gates: CoherenceGates::default(),
        }
    }

    /// Combined SOC score
    pub fn soc_score(&self) -> f64 {
        (self.comprehensibility + self.manageability + self.meaningfulness) / 3.0
    }

    /// Update SOC from subsystem states
    pub fn update_soc(
        &mut self,
        prediction_accuracy: f64,   // How well world model predicts
        resource_availability: f64, // Meta-learner confidence
        goal_progress: f64,         // Goal completion rate
    ) {
        // Comprehensibility: world is predictable
        self.comprehensibility = self.comprehensibility * 0.9 + prediction_accuracy * 0.1;

        // Manageability: we have resources to cope
        self.manageability = self.manageability * 0.9 + resource_availability * 0.1;

        // Meaningfulness: goals are worth pursuing
        self.meaningfulness = self.meaningfulness * 0.9 + goal_progress * 0.1;

        // Record history
        let soc = self.soc_score();
        self.coherence_history.push_back(soc);
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }
    }

    /// Register a new source of dissonance
    pub fn register_dissonance(
        &mut self,
        source_type: DissonanceType,
        magnitude: f64,
        belief_a: &str,
        belief_b: &str,
    ) {
        // Check if this source already exists
        if let Some(existing) = self
            .dissonance_sources
            .iter_mut()
            .find(|s| s.source_type == source_type)
        {
            existing.magnitude = (existing.magnitude + magnitude) / 2.0;
            existing.persistence += 1;
        } else {
            self.dissonance_sources.push(DissonanceSource {
                source_type,
                magnitude,
                persistence: 1,
                belief_a: belief_a.to_string(),
                belief_b: belief_b.to_string(),
            });
        }

        self.update_dissonance_level();
    }

    /// Update overall dissonance level
    fn update_dissonance_level(&mut self) {
        if self.dissonance_sources.is_empty() {
            self.dissonance_level *= 0.95; // Decay
        } else {
            // Weighted sum: persistent dissonance is worse
            let total: f64 = self
                .dissonance_sources
                .iter()
                .map(|s| s.magnitude * (1.0 + s.persistence as f64 * 0.1).min(2.0))
                .sum();
            self.dissonance_level = (total / self.dissonance_sources.len() as f64).min(1.0);
        }

        // Update regulation mode based on dissonance
        self.regulation_mode = if self.dissonance_level > 0.7 {
            RegulationMode::Protective
        } else if self.dissonance_level > 0.4 {
            RegulationMode::DissonanceResolution
        } else if self.coherence_trend() < -0.1 {
            RegulationMode::Allostatic
        } else {
            RegulationMode::Homeostatic
        };
    }

    /// Attempt to resolve dissonance of a specific type
    pub fn attempt_resolution(&mut self, source_type: DissonanceType) -> ResolutionStrategy {
        // Find the most successful strategy for this type
        let preferred = self
            .successful_resolutions
            .get(&source_type)
            .copied()
            .unwrap_or(0);

        let strategy = match source_type {
            DissonanceType::PredictionError => {
                if preferred % 3 == 0 {
                    ResolutionStrategy::UpdateWorldModel
                } else if preferred % 3 == 1 {
                    ResolutionStrategy::GatherMoreData
                } else {
                    ResolutionStrategy::AdjustExpectations
                }
            }
            DissonanceType::BeliefConflict => {
                if preferred % 2 == 0 {
                    ResolutionStrategy::ReviseBelief
                } else {
                    ResolutionStrategy::Compartmentalize
                }
            }
            DissonanceType::ValueActionGap => ResolutionStrategy::AlignAction,
            DissonanceType::GoalConflict => ResolutionStrategy::ReprioritizeGoals,
            DissonanceType::ModelMismatch => ResolutionStrategy::UpdateWorldModel,
            DissonanceType::SymbolGroundingFailure => ResolutionStrategy::RegroundSymbol,
        };

        strategy
    }

    /// Record successful resolution
    pub fn record_resolution_success(&mut self, source_type: DissonanceType) {
        *self
            .successful_resolutions
            .entry(source_type.clone())
            .or_insert(0) += 1;

        // Remove resolved dissonance
        self.dissonance_sources
            .retain(|s| s.source_type != source_type);
        self.update_dissonance_level();
    }

    /// Update coherence gates based on current state
    pub fn update_gates(&mut self) {
        let soc = self.soc_score();
        let dissonance = self.dissonance_level;

        // Base gate value: high coherence = open gates
        let base = (soc * (1.0 - dissonance * 0.5)).clamp(0.3, 1.0);

        match self.regulation_mode {
            RegulationMode::Protective => {
                // In protective mode, reduce all cross-subsystem flow
                self.gates.discovery_to_abstraction = base * 0.5;
                self.gates.abstraction_to_symbols = base * 0.5;
                self.gates.symbols_to_goals = base * 0.3; // Especially careful with goals
                self.gates.goals_to_world_model = base * 0.5;
                self.gates.world_model_to_meta = base * 0.7;
                self.gates.meta_to_discovery = base * 0.5;
            }
            RegulationMode::DissonanceResolution => {
                // Focus on resolving specific dissonance sources
                self.gates = CoherenceGates::default();
                for source in &self.dissonance_sources {
                    match source.source_type {
                        DissonanceType::PredictionError => {
                            self.gates.world_model_to_meta *= 1.5; // Boost learning
                        }
                        DissonanceType::SymbolGroundingFailure => {
                            self.gates.abstraction_to_symbols *= 1.5;
                        }
                        DissonanceType::GoalConflict => {
                            self.gates.symbols_to_goals *= 0.5; // Slow goal updates
                        }
                        _ => {}
                    }
                }
            }
            RegulationMode::Exploratory => {
                // Open gates wide for learning
                self.gates.discovery_to_abstraction = 1.0;
                self.gates.abstraction_to_symbols = 1.0;
                self.gates.symbols_to_goals = 0.8;
                self.gates.goals_to_world_model = 1.0;
                self.gates.world_model_to_meta = 1.0;
                self.gates.meta_to_discovery = 1.0;
            }
            _ => {
                // Normal homeostatic operation
                self.gates.discovery_to_abstraction = base;
                self.gates.abstraction_to_symbols = base;
                self.gates.symbols_to_goals = base;
                self.gates.goals_to_world_model = base;
                self.gates.world_model_to_meta = base;
                self.gates.meta_to_discovery = base;
            }
        }
    }

    /// Coherence trend (positive = improving)
    pub fn coherence_trend(&self) -> f64 {
        if self.coherence_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .coherence_history
            .iter()
            .rev()
            .take(10)
            .copied()
            .collect();
        let older: Vec<f64> = self
            .coherence_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .copied()
            .collect();

        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        recent_avg - older_avg
    }
}

/// Strategies for resolving cognitive dissonance
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    UpdateWorldModel,
    GatherMoreData,
    AdjustExpectations,
    ReviseBelief,
    Compartmentalize,
    AlignAction,
    ReprioritizeGoals,
    RegroundSymbol,
}

// ═══════════════════════════════════════════════════════════════════════════════
// REQUISITE VARIETY (ASHBY'S LAW)
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks environmental variety vs system response variety
/// Ashby's Law: Only variety can absorb variety
#[derive(Debug, Clone)]
pub struct RequisiteVariety {
    /// Environmental perturbations observed
    environmental_variety: VarietyMeasure,
    /// System's response repertoire
    response_variety: VarietyMeasure,
    /// Variety ratio (response / environmental)
    pub variety_ratio: f64,
    /// History for trend detection
    ratio_history: VecDeque<f64>,
    /// Expansion recommendations
    pub expansion_needed: Vec<ExpansionRecommendation>,
}

#[derive(Debug, Clone)]
pub struct VarietyMeasure {
    /// Number of distinct states/actions observed
    pub distinct_count: usize,
    /// Entropy of distribution
    pub entropy: f64,
    /// Effective degrees of freedom
    pub effective_dof: f64,
    /// Raw observations for computing variety
    observations: VecDeque<Vec<i32>>,
    max_observations: usize,
}

impl VarietyMeasure {
    fn new(max_observations: usize) -> Self {
        Self {
            distinct_count: 0,
            entropy: 0.0,
            effective_dof: 0.0,
            observations: VecDeque::with_capacity(max_observations),
            max_observations,
        }
    }

    fn observe(&mut self, state: &[f64]) {
        // Discretize for counting
        let discrete: Vec<i32> = state.iter().map(|&x| (x * 10.0).round() as i32).collect();

        self.observations.push_back(discrete);
        if self.observations.len() > self.max_observations {
            self.observations.pop_front();
        }

        self.update_metrics();
    }

    fn update_metrics(&mut self) {
        use std::collections::HashSet;

        // Count distinct observations
        let unique: HashSet<Vec<i32>> = self.observations.iter().cloned().collect();
        self.distinct_count = unique.len();

        // Compute entropy
        let n = self.observations.len() as f64;
        if n > 0.0 {
            let mut counts: HashMap<Vec<i32>, usize> = HashMap::new();
            for obs in &self.observations {
                *counts.entry(obs.clone()).or_insert(0) += 1;
            }

            self.entropy = counts
                .values()
                .map(|&c| {
                    let p = c as f64 / n;
                    if p > 0.0 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum();
        }

        // Effective DOF ≈ e^entropy
        self.effective_dof = self.entropy.exp();
    }
}

#[derive(Debug, Clone)]
pub struct ExpansionRecommendation {
    pub area: ExpansionArea,
    pub urgency: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionArea {
    ActionSpace,         // Need more action types
    StateRepresentation, // Need finer state discrimination
    GoalHierarchy,       // Need more goals/subgoals
    CausalModels,        // Need more causal variables
    Abstractions,        // Need more concepts
    Symbols,             // Need more symbols
}

impl RequisiteVariety {
    pub fn new() -> Self {
        Self {
            environmental_variety: VarietyMeasure::new(500),
            response_variety: VarietyMeasure::new(500),
            variety_ratio: 1.0,
            ratio_history: VecDeque::with_capacity(100),
            expansion_needed: Vec::new(),
        }
    }

    /// Observe environmental state
    pub fn observe_environment(&mut self, state: &[f64]) {
        self.environmental_variety.observe(state);
        self.update_ratio();
    }

    /// Observe system response
    pub fn observe_response(&mut self, action: usize, n_actions: usize, internal_state: &[f64]) {
        // Combine action with internal state for response variety
        let mut response = internal_state.to_vec();
        response.push(action as f64 / n_actions as f64);
        self.response_variety.observe(&response);
        self.update_ratio();
    }

    fn update_ratio(&mut self) {
        let env_var = self.environmental_variety.effective_dof.max(1.0);
        let resp_var = self.response_variety.effective_dof.max(1.0);

        self.variety_ratio = resp_var / env_var;

        self.ratio_history.push_back(self.variety_ratio);
        if self.ratio_history.len() > 100 {
            self.ratio_history.pop_front();
        }

        self.check_expansion_needed();
    }

    fn check_expansion_needed(&mut self) {
        self.expansion_needed.clear();

        // Ashby's Law: variety_ratio should be >= 1.0
        if self.variety_ratio < 0.8 {
            let urgency = (1.0 - self.variety_ratio).min(1.0);

            // Determine which area needs expansion
            if self.response_variety.distinct_count < self.environmental_variety.distinct_count / 2
            {
                self.expansion_needed.push(ExpansionRecommendation {
                    area: ExpansionArea::ActionSpace,
                    urgency,
                    reason: "Action repertoire insufficient for environmental variety".to_string(),
                });
            }

            if self.response_variety.entropy < self.environmental_variety.entropy * 0.7 {
                self.expansion_needed.push(ExpansionRecommendation {
                    area: ExpansionArea::StateRepresentation,
                    urgency: urgency * 0.8,
                    reason: "State representation too coarse for environmental complexity"
                        .to_string(),
                });
            }
        }

        // Check for variety decay (losing capabilities)
        if self.ratio_history.len() >= 20 {
            let recent: f64 = self.ratio_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older: f64 = self
                .ratio_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .sum::<f64>()
                / 10.0;

            if recent < older * 0.9 {
                self.expansion_needed.push(ExpansionRecommendation {
                    area: ExpansionArea::Abstractions,
                    urgency: 0.6,
                    reason: "Response variety declining - need conceptual expansion".to_string(),
                });
            }
        }
    }

    /// Check if system has requisite variety
    pub fn has_requisite_variety(&self) -> bool {
        self.variety_ratio >= 1.0
    }

    /// Get variety deficit (how much more variety needed)
    pub fn variety_deficit(&self) -> f64 {
        (1.0 - self.variety_ratio).max(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTATIC EQUILIBRIUM
// ═══════════════════════════════════════════════════════════════════════════════

/// PID-style homeostatic controller with allostatic adaptation
#[derive(Debug, Clone)]
pub struct HomeostaticEquilibrium {
    /// Setpoints for each regulated variable
    pub setpoints: HomeostaticSetpoints,
    /// Current values
    pub current: HomeostaticSetpoints,
    /// Error integrals for I term
    error_integrals: HomeostaticSetpoints,
    /// Previous errors for D term
    prev_errors: HomeostaticSetpoints,
    /// PID gains
    pub gains: PIDGains,
    /// Control outputs
    pub controls: HomeostaticControls,
    /// Allostatic adaptation rate
    pub allostatic_rate: f64,
    /// How long current deviation has persisted
    deviation_persistence: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct HomeostaticSetpoints {
    pub compound_rate: f64,
    pub coherence_score: f64,
    pub discovery_rate: f64,
    pub goal_success_rate: f64,
    pub exploration_rate: f64,
    pub prediction_accuracy: f64,
}

impl Default for HomeostaticSetpoints {
    fn default() -> Self {
        Self {
            compound_rate: 10.0,      // Target interactions/step
            coherence_score: 0.7,     // Target SOC
            discovery_rate: 0.01,     // Variables discovered per step
            goal_success_rate: 0.5,   // Goal completion rate
            exploration_rate: 0.2,    // Balance exploration/exploitation
            prediction_accuracy: 0.8, // World model accuracy
        }
    }
}

#[derive(Debug, Clone)]
pub struct PIDGains {
    pub kp: f64, // Proportional
    pub ki: f64, // Integral
    pub kd: f64, // Derivative
}

impl Default for PIDGains {
    fn default() -> Self {
        Self {
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HomeostaticControls {
    /// Adjust learning rate
    pub learning_rate_adjust: f64,
    /// Adjust exploration
    pub exploration_adjust: f64,
    /// Adjust goal creation rate
    pub goal_creation_adjust: f64,
    /// Adjust symbol grounding threshold
    pub symbol_threshold_adjust: f64,
    /// Adjust world model learning
    pub world_model_adjust: f64,
    /// Overall regulation intensity
    pub regulation_intensity: f64,
}

impl HomeostaticEquilibrium {
    pub fn new() -> Self {
        Self {
            setpoints: HomeostaticSetpoints::default(),
            current: HomeostaticSetpoints::default(),
            error_integrals: HomeostaticSetpoints {
                compound_rate: 0.0,
                coherence_score: 0.0,
                discovery_rate: 0.0,
                goal_success_rate: 0.0,
                exploration_rate: 0.0,
                prediction_accuracy: 0.0,
            },
            prev_errors: HomeostaticSetpoints {
                compound_rate: 0.0,
                coherence_score: 0.0,
                discovery_rate: 0.0,
                goal_success_rate: 0.0,
                exploration_rate: 0.0,
                prediction_accuracy: 0.0,
            },
            gains: PIDGains::default(),
            controls: HomeostaticControls::default(),
            allostatic_rate: 0.001,
            deviation_persistence: HashMap::new(),
        }
    }

    /// Update current values from subsystem states
    pub fn update_current(
        &mut self,
        compound_rate: f64,
        coherence_score: f64,
        discovery_rate: f64,
        goal_success_rate: f64,
        exploration_rate: f64,
        prediction_accuracy: f64,
    ) {
        self.current.compound_rate = compound_rate;
        self.current.coherence_score = coherence_score;
        self.current.discovery_rate = discovery_rate;
        self.current.goal_success_rate = goal_success_rate;
        self.current.exploration_rate = exploration_rate;
        self.current.prediction_accuracy = prediction_accuracy;
    }

    /// Compute PID control signals
    pub fn regulate(&mut self) {
        // Compute errors
        let errors = HomeostaticSetpoints {
            compound_rate: self.setpoints.compound_rate - self.current.compound_rate,
            coherence_score: self.setpoints.coherence_score - self.current.coherence_score,
            discovery_rate: self.setpoints.discovery_rate - self.current.discovery_rate,
            goal_success_rate: self.setpoints.goal_success_rate - self.current.goal_success_rate,
            exploration_rate: self.setpoints.exploration_rate - self.current.exploration_rate,
            prediction_accuracy: self.setpoints.prediction_accuracy
                - self.current.prediction_accuracy,
        };

        // Update integrals (with anti-windup)
        self.error_integrals.compound_rate =
            (self.error_integrals.compound_rate + errors.compound_rate).clamp(-10.0, 10.0);
        self.error_integrals.coherence_score =
            (self.error_integrals.coherence_score + errors.coherence_score).clamp(-1.0, 1.0);
        self.error_integrals.discovery_rate =
            (self.error_integrals.discovery_rate + errors.discovery_rate).clamp(-0.1, 0.1);
        self.error_integrals.goal_success_rate =
            (self.error_integrals.goal_success_rate + errors.goal_success_rate).clamp(-1.0, 1.0);
        self.error_integrals.exploration_rate =
            (self.error_integrals.exploration_rate + errors.exploration_rate).clamp(-0.5, 0.5);
        self.error_integrals.prediction_accuracy = (self.error_integrals.prediction_accuracy
            + errors.prediction_accuracy)
            .clamp(-1.0, 1.0);

        // Compute derivatives
        let derivatives = HomeostaticSetpoints {
            compound_rate: errors.compound_rate - self.prev_errors.compound_rate,
            coherence_score: errors.coherence_score - self.prev_errors.coherence_score,
            discovery_rate: errors.discovery_rate - self.prev_errors.discovery_rate,
            goal_success_rate: errors.goal_success_rate - self.prev_errors.goal_success_rate,
            exploration_rate: errors.exploration_rate - self.prev_errors.exploration_rate,
            prediction_accuracy: errors.prediction_accuracy - self.prev_errors.prediction_accuracy,
        };

        // PID formula: u = Kp*e + Ki*∫e + Kd*de/dt
        let kp = self.gains.kp;
        let ki = self.gains.ki;
        let kd = self.gains.kd;

        // Map errors to control outputs
        // Low compound rate → increase learning rate
        self.controls.learning_rate_adjust = kp * errors.compound_rate * 0.01
            + ki * self.error_integrals.compound_rate * 0.001
            + kd * derivatives.compound_rate * 0.1;

        // Low coherence → reduce exploration
        self.controls.exploration_adjust = kp * errors.coherence_score * 0.5
            + ki * self.error_integrals.coherence_score * 0.1
            + kd * derivatives.coherence_score;

        // Low discovery → increase exploration
        self.controls.exploration_adjust += kp * errors.discovery_rate * 10.0;

        // Low goal success → adjust goal creation
        self.controls.goal_creation_adjust =
            kp * errors.goal_success_rate * 0.5 + ki * self.error_integrals.goal_success_rate * 0.1;

        // Low prediction accuracy → boost world model learning
        self.controls.world_model_adjust =
            kp * errors.prediction_accuracy + ki * self.error_integrals.prediction_accuracy * 0.5;

        // Overall regulation intensity
        let total_error = errors.compound_rate.abs() / 10.0
            + errors.coherence_score.abs()
            + errors.goal_success_rate.abs()
            + errors.prediction_accuracy.abs();
        self.controls.regulation_intensity = (total_error / 4.0).clamp(0.0, 1.0);

        // Track deviation persistence for allostasis
        self.update_deviation_persistence(&errors);

        // Allostatic adaptation: adjust setpoints if deviation persists
        self.allostatic_adapt();

        // Save previous errors
        self.prev_errors = errors;
    }

    fn update_deviation_persistence(&mut self, errors: &HomeostaticSetpoints) {
        let threshold = 0.1;

        if errors.compound_rate.abs() > threshold * 10.0 {
            *self
                .deviation_persistence
                .entry("compound_rate".to_string())
                .or_insert(0) += 1;
        } else {
            self.deviation_persistence.remove("compound_rate");
        }

        if errors.coherence_score.abs() > threshold {
            *self
                .deviation_persistence
                .entry("coherence_score".to_string())
                .or_insert(0) += 1;
        } else {
            self.deviation_persistence.remove("coherence_score");
        }

        if errors.goal_success_rate.abs() > threshold {
            *self
                .deviation_persistence
                .entry("goal_success_rate".to_string())
                .or_insert(0) += 1;
        } else {
            self.deviation_persistence.remove("goal_success_rate");
        }
    }

    /// Allostatic adaptation: adjust setpoints based on sustained deviation
    fn allostatic_adapt(&mut self) {
        let persistence_threshold = 100; // Steps before adaptation

        for (variable, &persistence) in &self.deviation_persistence {
            if persistence > persistence_threshold {
                let adapt = self.allostatic_rate * (persistence - persistence_threshold) as f64;

                match variable.as_str() {
                    "compound_rate" => {
                        // If we can't reach compound rate, lower expectations
                        let error = self.setpoints.compound_rate - self.current.compound_rate;
                        self.setpoints.compound_rate -= error.signum() * adapt * 10.0;
                        self.setpoints.compound_rate =
                            self.setpoints.compound_rate.clamp(1.0, 50.0);
                    }
                    "coherence_score" => {
                        let error = self.setpoints.coherence_score - self.current.coherence_score;
                        self.setpoints.coherence_score -= error.signum() * adapt;
                        self.setpoints.coherence_score =
                            self.setpoints.coherence_score.clamp(0.3, 0.95);
                    }
                    "goal_success_rate" => {
                        let error =
                            self.setpoints.goal_success_rate - self.current.goal_success_rate;
                        self.setpoints.goal_success_rate -= error.signum() * adapt;
                        self.setpoints.goal_success_rate =
                            self.setpoints.goal_success_rate.clamp(0.1, 0.9);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Check if system is in equilibrium
    pub fn in_equilibrium(&self) -> bool {
        self.controls.regulation_intensity < 0.2
    }

    /// Get equilibrium status
    pub fn status(&self) -> EquilibriumStatus {
        if self.controls.regulation_intensity < 0.1 {
            EquilibriumStatus::Stable
        } else if self.controls.regulation_intensity < 0.3 {
            EquilibriumStatus::MinorDeviation
        } else if self.controls.regulation_intensity < 0.6 {
            EquilibriumStatus::Regulating
        } else {
            EquilibriumStatus::Stressed
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumStatus {
    Stable,
    MinorDeviation,
    Regulating,
    Stressed,
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC INTEGRITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks truth-seeking behavior and belief consistency
#[derive(Debug, Clone)]
pub struct EpistemicIntegrity {
    /// Confidence calibration: how well confidence matches accuracy
    pub calibration_error: f64,
    /// Belief revision rate (too high = unstable, too low = dogmatic)
    pub revision_rate: f64,
    /// Evidence integration: do we update on new evidence?
    pub evidence_sensitivity: f64,
    /// Confirmation bias measure
    pub confirmation_bias: f64,
    /// Prediction tracking
    predictions: VecDeque<PredictionRecord>,
    /// Belief history
    belief_history: VecDeque<BeliefState>,
}

#[derive(Debug, Clone)]
struct PredictionRecord {
    confidence: f64,
    was_correct: bool,
}

#[derive(Debug, Clone)]
struct BeliefState {
    belief_hash: u64,
    confidence: f64,
}

impl EpistemicIntegrity {
    pub fn new() -> Self {
        Self {
            calibration_error: 0.0,
            revision_rate: 0.0,
            evidence_sensitivity: 0.5,
            confirmation_bias: 0.0,
            predictions: VecDeque::with_capacity(100),
            belief_history: VecDeque::with_capacity(100),
        }
    }

    /// Record a prediction and its outcome
    pub fn record_prediction(&mut self, confidence: f64, was_correct: bool) {
        self.predictions.push_back(PredictionRecord {
            confidence,
            was_correct,
        });
        if self.predictions.len() > 100 {
            self.predictions.pop_front();
        }
        self.update_calibration();
    }

    fn update_calibration(&mut self) {
        if self.predictions.is_empty() {
            return;
        }

        // Bin predictions by confidence and check accuracy
        let mut bins: HashMap<usize, (f64, f64)> = HashMap::new(); // bin -> (correct, total)

        for pred in &self.predictions {
            let bin = (pred.confidence * 10.0) as usize;
            let entry = bins.entry(bin).or_insert((0.0, 0.0));
            if pred.was_correct {
                entry.0 += 1.0;
            }
            entry.1 += 1.0;
        }

        // Calibration error: |confidence - accuracy| averaged over bins
        let mut total_error = 0.0;
        let mut n_bins = 0;

        for (bin, (correct, total)) in bins {
            if total >= 5.0 {
                let expected_accuracy = bin as f64 / 10.0;
                let actual_accuracy = correct / total;
                total_error += (expected_accuracy - actual_accuracy).abs();
                n_bins += 1;
            }
        }

        self.calibration_error = if n_bins > 0 {
            total_error / n_bins as f64
        } else {
            0.0
        };
    }

    /// Record belief state for tracking revisions
    pub fn record_belief(&mut self, belief_representation: &[f64]) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &v in belief_representation {
            ((v * 1000.0) as i64).hash(&mut hasher);
        }
        let belief_hash = hasher.finish();

        let confidence = belief_representation.iter().map(|&x| x.abs()).sum::<f64>()
            / belief_representation.len() as f64;

        self.belief_history.push_back(BeliefState {
            belief_hash,
            confidence,
        });
        if self.belief_history.len() > 100 {
            self.belief_history.pop_front();
        }

        self.update_revision_rate();
    }

    fn update_revision_rate(&mut self) {
        if self.belief_history.len() < 10 {
            return;
        }

        let mut changes = 0;
        let beliefs: Vec<_> = self.belief_history.iter().collect();

        for window in beliefs.windows(2) {
            if window[0].belief_hash != window[1].belief_hash {
                changes += 1;
            }
        }

        self.revision_rate = changes as f64 / (self.belief_history.len() - 1) as f64;
    }

    /// Overall epistemic health
    pub fn integrity_score(&self) -> f64 {
        // Good epistemic state: low calibration error, moderate revision rate
        let calibration_score = 1.0 - self.calibration_error.min(1.0);

        // Revision rate should be moderate (0.1-0.4 is healthy)
        let revision_score = if self.revision_rate < 0.05 {
            self.revision_rate / 0.05 * 0.5 // Too stable = dogmatic
        } else if self.revision_rate > 0.5 {
            1.0 - (self.revision_rate - 0.5) // Too unstable
        } else {
            1.0
        };

        // Low confirmation bias
        let bias_score = 1.0 - self.confirmation_bias.min(1.0);

        (calibration_score + revision_score + bias_score) / 3.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ETHICAL ALIGNMENT (Scaffold)
// ═══════════════════════════════════════════════════════════════════════════════

/// Basic ethical alignment tracking
#[derive(Debug, Clone)]
pub struct EthicalAlignment {
    /// Value coherence: do actions align with stated values?
    pub value_coherence: f64,
    /// Harm avoidance score
    pub harm_avoidance: f64,
    /// Fairness in resource allocation
    pub fairness_score: f64,
    /// Transparency of decision-making
    pub transparency: f64,
    /// Registered values/principles
    values: Vec<Value>,
    /// Action-value alignment history
    alignment_history: VecDeque<f64>,
}

#[derive(Debug, Clone)]
pub struct Value {
    pub name: String,
    pub weight: f64,
    pub description: String,
}

impl EthicalAlignment {
    pub fn new() -> Self {
        Self {
            value_coherence: 1.0,
            harm_avoidance: 1.0,
            fairness_score: 1.0,
            transparency: 1.0,
            values: vec![
                Value {
                    name: "truth_seeking".to_string(),
                    weight: 1.0,
                    description: "Prioritize accurate beliefs over comfortable ones".to_string(),
                },
                Value {
                    name: "harm_avoidance".to_string(),
                    weight: 1.0,
                    description: "Avoid actions that cause harm".to_string(),
                },
                Value {
                    name: "autonomy_respect".to_string(),
                    weight: 1.0,
                    description: "Respect agent autonomy and human oversight".to_string(),
                },
            ],
            alignment_history: VecDeque::with_capacity(100),
        }
    }

    /// Record an action's alignment with values
    pub fn record_action_alignment(&mut self, alignment: f64) {
        self.alignment_history.push_back(alignment);
        if self.alignment_history.len() > 100 {
            self.alignment_history.pop_front();
        }

        // Update value coherence
        self.value_coherence = if self.alignment_history.is_empty() {
            1.0
        } else {
            self.alignment_history.iter().sum::<f64>() / self.alignment_history.len() as f64
        };
    }

    /// Overall ethical score
    pub fn ethical_score(&self) -> f64 {
        (self.value_coherence + self.harm_avoidance + self.fairness_score + self.transparency) / 4.0
    }

    /// Check if action would violate values
    pub fn check_action(&self, _action: usize, _context: &str) -> ActionEthicalStatus {
        // Placeholder - would need domain-specific implementation
        ActionEthicalStatus::Permitted
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionEthicalStatus {
    Permitted,
    Cautioned(String),
    Prohibited(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// FOUNDATIONAL PRINCIPLES FOR AGI
// ═══════════════════════════════════════════════════════════════════════════════

/// The seven foundational principles required for AGI emergence
/// These principles are interconnected and compound multiplicatively
#[derive(Debug, Clone)]
pub struct FoundationalPrinciples {
    /// 1. EMBODIMENT: Intelligence tied to physical interaction
    pub embodiment: EmbodimentPrinciple,
    /// 2. SYMBOL GROUNDING: Abstract representations anchored in experience
    pub symbol_grounding: SymbolGroundingPrinciple,
    /// 3. CAUSALITY: Inferring cause-effect relationships
    pub causality: CausalityPrinciple,
    /// 4. MEMORY INTEGRATION: Working, long-term, and episodic memory
    pub memory: MemoryIntegrationPrinciple,
    /// 5. METACOGNITION: Self-awareness and self-regulation
    pub metacognition: MetacognitionPrinciple,
    /// 6. AUTONOMY: Independent goal-directed decision-making
    pub autonomy: AutonomyPrinciple,
    /// 7. EXPLICABILITY: Interpretable and accountable decisions
    pub explicability: ExplicabilityPrinciple,
    /// Overall principle integration score
    pub integration_score: f64,
}

/// Embodiment: Intelligence grounded in sensorimotor interaction
#[derive(Debug, Clone)]
pub struct EmbodimentPrinciple {
    /// Sensorimotor loop closure (action → observation → action)
    pub loop_closure: f64,
    /// Environmental interaction diversity
    pub interaction_diversity: f64,
    /// Physical grounding of predictions
    pub prediction_grounding: f64,
    /// Exploration-exploitation balance in physical space
    pub exploration_balance: f64,
}

impl EmbodimentPrinciple {
    pub fn new() -> Self {
        Self {
            loop_closure: 0.0,
            interaction_diversity: 0.0,
            prediction_grounding: 0.5,
            exploration_balance: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.loop_closure
            + self.interaction_diversity
            + self.prediction_grounding
            + self.exploration_balance)
            / 4.0
    }

    /// Update from sensorimotor experience
    pub fn update(
        &mut self,
        action_taken: bool,
        observation_received: bool,
        prediction_error: f64,
    ) {
        // Loop closure: did action lead to observation?
        if action_taken && observation_received {
            self.loop_closure = self.loop_closure * 0.99 + 0.01;
        }

        // Prediction grounding: low error means good physical model
        self.prediction_grounding =
            self.prediction_grounding * 0.95 + (1.0 - prediction_error.min(1.0)) * 0.05;
    }
}

/// Symbol Grounding: Symbols anchored in perceptual/experiential data
#[derive(Debug, Clone)]
pub struct SymbolGroundingPrinciple {
    /// Ratio of grounded to ungrounded symbols
    pub grounding_ratio: f64,
    /// Multi-modal integration (vision + language + action)
    pub multimodal_integration: f64,
    /// Semantic coherence of symbol usage
    pub semantic_coherence: f64,
    /// Symbol evolution from experience
    pub experiential_evolution: f64,
}

impl SymbolGroundingPrinciple {
    pub fn new() -> Self {
        Self {
            grounding_ratio: 0.0,
            multimodal_integration: 0.0,
            semantic_coherence: 0.5,
            experiential_evolution: 0.0,
        }
    }

    pub fn score(&self) -> f64 {
        (self.grounding_ratio
            + self.multimodal_integration
            + self.semantic_coherence
            + self.experiential_evolution)
            / 4.0
    }

    /// Update from symbol system state
    pub fn update(&mut self, grounded_symbols: usize, total_symbols: usize, new_groundings: usize) {
        if total_symbols > 0 {
            self.grounding_ratio = grounded_symbols as f64 / total_symbols as f64;
        }

        // Track experiential evolution
        if new_groundings > 0 {
            self.experiential_evolution = self.experiential_evolution * 0.95 + 0.05;
        } else {
            self.experiential_evolution *= 0.99;
        }
    }
}

/// Causality: Inferring cause-effect relationships beyond correlation
#[derive(Debug, Clone)]
pub struct CausalityPrinciple {
    /// Counterfactual reasoning capability
    pub counterfactual_reasoning: f64,
    /// Intervention-based learning
    pub intervention_learning: f64,
    /// Causal model accuracy
    pub causal_accuracy: f64,
    /// Distinction between correlation and causation
    pub correlation_distinction: f64,
}

impl CausalityPrinciple {
    pub fn new() -> Self {
        Self {
            counterfactual_reasoning: 0.0,
            intervention_learning: 0.0,
            causal_accuracy: 0.5,
            correlation_distinction: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.counterfactual_reasoning
            + self.intervention_learning
            + self.causal_accuracy
            + self.correlation_distinction)
            / 4.0
    }

    /// Update from causal discovery system
    pub fn update(
        &mut self,
        discovered_variables: usize,
        causal_interventions: usize,
        prediction_accuracy: f64,
    ) {
        // More discovered variables = better causal modeling
        self.counterfactual_reasoning = (discovered_variables as f64 / 10.0).min(1.0);

        // Interventions indicate active causal learning
        self.intervention_learning = (causal_interventions as f64 / 100.0).min(1.0);

        // Accuracy of causal predictions
        self.causal_accuracy = self.causal_accuracy * 0.95 + prediction_accuracy * 0.05;
    }
}

/// Memory Integration: Working, long-term, and episodic memory
#[derive(Debug, Clone)]
pub struct MemoryIntegrationPrinciple {
    /// Working memory utilization
    pub working_memory: f64,
    /// Long-term memory consolidation
    pub long_term_consolidation: f64,
    /// Episodic recall accuracy
    pub episodic_recall: f64,
    /// Continual learning without catastrophic forgetting
    pub continual_learning: f64,
}

impl MemoryIntegrationPrinciple {
    pub fn new() -> Self {
        Self {
            working_memory: 0.5,
            long_term_consolidation: 0.0,
            episodic_recall: 0.5,
            continual_learning: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.working_memory
            + self.long_term_consolidation
            + self.episodic_recall
            + self.continual_learning)
            / 4.0
    }

    /// Update from memory system state
    pub fn update(
        &mut self,
        world_model_states: usize,
        concepts_formed: usize,
        goal_success_rate: f64,
    ) {
        // World model states indicate long-term consolidation
        self.long_term_consolidation = (world_model_states as f64 / 100.0).min(1.0);

        // Concepts indicate abstraction/consolidation
        self.working_memory = (concepts_formed as f64 / 20.0).min(1.0);

        // Continual learning tracked via goal success stability
        self.continual_learning = self.continual_learning * 0.95 + goal_success_rate * 0.05;
    }
}

/// Metacognition: Self-awareness and self-regulation
#[derive(Debug, Clone)]
pub struct MetacognitionPrinciple {
    /// Uncertainty detection
    pub uncertainty_awareness: f64,
    /// Error monitoring and correction
    pub error_monitoring: f64,
    /// Task prioritization
    pub task_prioritization: f64,
    /// Learning from mistakes
    pub mistake_learning: f64,
}

impl MetacognitionPrinciple {
    pub fn new() -> Self {
        Self {
            uncertainty_awareness: 0.5,
            error_monitoring: 0.5,
            task_prioritization: 0.5,
            mistake_learning: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.uncertainty_awareness
            + self.error_monitoring
            + self.task_prioritization
            + self.mistake_learning)
            / 4.0
    }

    /// Update from meta-learner state
    pub fn update(
        &mut self,
        exploration_rate: f64,
        prediction_error_trend: f64,
        goal_completion: bool,
    ) {
        // High exploration when uncertain = good uncertainty awareness
        self.uncertainty_awareness = exploration_rate;

        // Error trend improving = good error monitoring
        if prediction_error_trend < 0.0 {
            self.error_monitoring = self.error_monitoring * 0.95 + 0.05;
        }

        // Goal completion indicates task prioritization working
        if goal_completion {
            self.task_prioritization = self.task_prioritization * 0.95 + 0.05;
            self.mistake_learning = self.mistake_learning * 0.95 + 0.05;
        }
    }
}

/// Autonomy: Independent goal-directed decision-making
#[derive(Debug, Clone)]
pub struct AutonomyPrinciple {
    /// Self-generated goals vs externally imposed
    pub goal_self_generation: f64,
    /// Decision independence
    pub decision_independence: f64,
    /// Homeostatic self-regulation
    pub self_regulation: f64,
    /// Proactive vs reactive behavior ratio
    pub proactivity: f64,
}

impl AutonomyPrinciple {
    pub fn new() -> Self {
        Self {
            goal_self_generation: 0.0,
            decision_independence: 0.5,
            self_regulation: 0.5,
            proactivity: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.goal_self_generation
            + self.decision_independence
            + self.self_regulation
            + self.proactivity)
            / 4.0
    }

    /// Update from goal hierarchy and action selection
    pub fn update(
        &mut self,
        goals_created: usize,
        action_from_recommendation: bool,
        in_equilibrium: bool,
    ) {
        // Self-generated goals
        self.goal_self_generation = (goals_created as f64 / 10.0).min(1.0);

        // Following own recommendations = decision independence
        if action_from_recommendation {
            self.decision_independence = self.decision_independence * 0.95 + 0.05;
        }

        // Equilibrium = good self-regulation
        if in_equilibrium {
            self.self_regulation = self.self_regulation * 0.95 + 0.05;
        }
    }
}

/// Explicability: Interpretable and accountable decisions
#[derive(Debug, Clone)]
pub struct ExplicabilityPrinciple {
    /// Decision chain traceability
    pub traceability: f64,
    /// Reason generation for actions
    pub reason_generation: f64,
    /// Accountability (who/what is responsible)
    pub accountability: f64,
    /// Bias detection and mitigation
    pub bias_mitigation: f64,
}

impl ExplicabilityPrinciple {
    pub fn new() -> Self {
        Self {
            traceability: 0.5,
            reason_generation: 0.0,
            accountability: 0.5,
            bias_mitigation: 0.5,
        }
    }

    pub fn score(&self) -> f64 {
        (self.traceability + self.reason_generation + self.accountability + self.bias_mitigation)
            / 4.0
    }

    /// Update from analytics and symbol system
    pub fn update(&mut self, has_symbol_for_action: bool, compound_interactions: usize) {
        // Symbols for actions = reason generation
        if has_symbol_for_action {
            self.reason_generation = self.reason_generation * 0.95 + 0.05;
        }

        // More compound interactions = more traceable decision chain
        self.traceability = (compound_interactions as f64 / 1000.0).min(1.0);
    }
}

impl FoundationalPrinciples {
    pub fn new() -> Self {
        Self {
            embodiment: EmbodimentPrinciple::new(),
            symbol_grounding: SymbolGroundingPrinciple::new(),
            causality: CausalityPrinciple::new(),
            memory: MemoryIntegrationPrinciple::new(),
            metacognition: MetacognitionPrinciple::new(),
            autonomy: AutonomyPrinciple::new(),
            explicability: ExplicabilityPrinciple::new(),
            integration_score: 0.0,
        }
    }

    /// Compute overall integration score
    /// Principles compound multiplicatively when all are present
    pub fn compute_integration(&mut self) {
        let scores = [
            self.embodiment.score(),
            self.symbol_grounding.score(),
            self.causality.score(),
            self.memory.score(),
            self.metacognition.score(),
            self.autonomy.score(),
            self.explicability.score(),
        ];

        // Geometric mean for multiplicative compounding
        let product: f64 = scores.iter().product();
        let geometric_mean = product.powf(1.0 / scores.len() as f64);

        // But also reward having ALL principles active (synergy bonus)
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let synergy_bonus = if min_score > 0.3 { 0.1 } else { 0.0 };

        self.integration_score = (geometric_mean + synergy_bonus).min(1.0);
    }

    /// Get summary of all principles
    pub fn summary(&self) -> PrinciplesSummary {
        PrinciplesSummary {
            embodiment: self.embodiment.score(),
            symbol_grounding: self.symbol_grounding.score(),
            causality: self.causality.score(),
            memory: self.memory.score(),
            metacognition: self.metacognition.score(),
            autonomy: self.autonomy.score(),
            explicability: self.explicability.score(),
            integration: self.integration_score,
        }
    }

    /// Print formatted summary
    pub fn print_summary(&self) {
        let s = self.summary();
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║           FOUNDATIONAL PRINCIPLES FOR AGI                        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ 1. Embodiment:        {:.3}  (sensorimotor grounding)             ║",
            s.embodiment
        );
        println!(
            "║ 2. Symbol Grounding:  {:.3}  (experiential anchoring)             ║",
            s.symbol_grounding
        );
        println!(
            "║ 3. Causality:         {:.3}  (cause-effect reasoning)             ║",
            s.causality
        );
        println!(
            "║ 4. Memory:            {:.3}  (working/long-term/episodic)         ║",
            s.memory
        );
        println!(
            "║ 5. Metacognition:     {:.3}  (self-awareness)                     ║",
            s.metacognition
        );
        println!(
            "║ 6. Autonomy:          {:.3}  (goal-directed agency)               ║",
            s.autonomy
        );
        println!(
            "║ 7. Explicability:     {:.3}  (interpretable decisions)            ║",
            s.explicability
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ INTEGRATION SCORE:    {:.3}  (multiplicative compounding)         ║",
            s.integration
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

#[derive(Debug, Clone)]
pub struct PrinciplesSummary {
    pub embodiment: f64,
    pub symbol_grounding: f64,
    pub causality: f64,
    pub memory: f64,
    pub metacognition: f64,
    pub autonomy: f64,
    pub explicability: f64,
    pub integration: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED AGC SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Artificial General Coherence - unified system
#[derive(Debug, Clone)]
pub struct ArtificialGeneralCoherence {
    /// Psychological coherence (SOC + dissonance)
    pub psychological: PsychologicalCoherence,
    /// Requisite variety tracking
    pub variety: RequisiteVariety,
    /// Homeostatic equilibrium
    pub homeostasis: HomeostaticEquilibrium,
    /// Epistemic integrity
    pub epistemic: EpistemicIntegrity,
    /// Ethical alignment
    pub ethical: EthicalAlignment,
    /// Foundational principles for AGI
    pub principles: FoundationalPrinciples,
    /// Overall coherence score
    pub coherence_score: f64,
    /// Current step
    step: usize,
}

impl ArtificialGeneralCoherence {
    pub fn new() -> Self {
        Self {
            psychological: PsychologicalCoherence::new(),
            variety: RequisiteVariety::new(),
            homeostasis: HomeostaticEquilibrium::new(),
            epistemic: EpistemicIntegrity::new(),
            ethical: EthicalAlignment::new(),
            principles: FoundationalPrinciples::new(),
            coherence_score: 0.5,
            step: 0,
        }
    }

    /// Main update function - call each step
    pub fn update(
        &mut self,
        state: &[f64],
        action: usize,
        n_actions: usize,
        reward: f64,
        prediction_error: f64,
        compound_rate: f64,
        goal_success_rate: f64,
    ) {
        self.step += 1;

        // Update variety tracking
        self.variety.observe_environment(state);
        self.variety.observe_response(action, n_actions, state);

        // Update psychological coherence
        let prediction_accuracy = 1.0 - prediction_error.min(1.0);
        let resource_availability = if self.variety.has_requisite_variety() {
            0.8
        } else {
            0.4
        };
        let goal_progress = goal_success_rate;

        self.psychological
            .update_soc(prediction_accuracy, resource_availability, goal_progress);

        // Check for dissonance from prediction error
        if prediction_error > 0.5 {
            self.psychological.register_dissonance(
                DissonanceType::PredictionError,
                prediction_error,
                "expected_outcome",
                "actual_outcome",
            );
        }

        // Update homeostasis
        self.homeostasis.update_current(
            compound_rate,
            self.psychological.soc_score(),
            0.01, // discovery rate - would come from CausalDiscovery
            goal_success_rate,
            0.2, // exploration rate - would come from MetaLearner
            prediction_accuracy,
        );
        self.homeostasis.regulate();

        // Update epistemic integrity
        self.epistemic.record_prediction(
            1.0 - prediction_error, // confidence proxy
            prediction_error < 0.3, // "correct" if error is low
        );

        // Update ethical alignment
        self.ethical
            .record_action_alignment(1.0 - prediction_error * 0.2);

        // Update coherence gates
        self.psychological.update_gates();

        // Update foundational principles
        // 1. Embodiment
        self.principles
            .embodiment
            .update(true, true, prediction_error);

        // 2. Causality (would get more data from CausalDiscovery in full integration)
        self.principles
            .causality
            .update(1, self.step, prediction_accuracy);

        // 3. Memory (would get more data from WorldModel in full integration)
        self.principles.memory.update(10, 2, goal_success_rate);

        // 4. Metacognition
        let error_trend = self.psychological.coherence_trend();
        self.principles
            .metacognition
            .update(0.2, error_trend, goal_success_rate > 0.5);

        // 5. Autonomy
        self.principles
            .autonomy
            .update(1, true, self.homeostasis.in_equilibrium());

        // Compute principle integration
        self.principles.compute_integration();

        // Compute overall coherence
        self.coherence_score = self.compute_overall_coherence();
    }

    /// Extended update with full subsystem data
    pub fn update_full(
        &mut self,
        state: &[f64],
        action: usize,
        n_actions: usize,
        reward: f64,
        prediction_error: f64,
        compound_rate: f64,
        goal_success_rate: f64,
        // Additional data from subsystems
        discovered_variables: usize,
        causal_interventions: usize,
        grounded_symbols: usize,
        total_symbols: usize,
        new_groundings: usize,
        world_model_states: usize,
        concepts_formed: usize,
        exploration_rate: f64,
        goals_created: usize,
        compound_interactions: usize,
        has_symbol_for_action: bool,
    ) {
        // Basic update
        self.update(
            state,
            action,
            n_actions,
            reward,
            prediction_error,
            compound_rate,
            goal_success_rate,
        );

        // Full principle updates with real data
        self.principles
            .symbol_grounding
            .update(grounded_symbols, total_symbols, new_groundings);
        self.principles.causality.update(
            discovered_variables,
            causal_interventions,
            1.0 - prediction_error,
        );
        self.principles
            .memory
            .update(world_model_states, concepts_formed, goal_success_rate);
        self.principles.metacognition.update(
            exploration_rate,
            self.psychological.coherence_trend(),
            goal_success_rate > 0.5,
        );
        self.principles
            .autonomy
            .update(goals_created, true, self.homeostasis.in_equilibrium());
        self.principles
            .explicability
            .update(has_symbol_for_action, compound_interactions);

        // Recompute integration with updated principles
        self.principles.compute_integration();
        self.coherence_score = self.compute_overall_coherence();
    }

    fn compute_overall_coherence(&self) -> f64 {
        let soc = self.psychological.soc_score();
        let dissonance_penalty = self.psychological.dissonance_level * 0.3;
        let variety_bonus = if self.variety.has_requisite_variety() {
            0.1
        } else {
            -0.1
        };
        let equilibrium_bonus = if self.homeostasis.in_equilibrium() {
            0.1
        } else {
            -0.05
        };
        let epistemic_factor = self.epistemic.integrity_score();
        let ethical_factor = self.ethical.ethical_score();
        let principles_factor = self.principles.integration_score;

        // Weighted combination:
        // - 35% psychological coherence (SOC - dissonance + variety + equilibrium)
        // - 20% epistemic integrity
        // - 15% ethical alignment
        // - 30% foundational principles (the key AGI factors)
        let psychological =
            (soc - dissonance_penalty + variety_bonus + equilibrium_bonus).clamp(0.0, 1.0);
        let weighted = psychological * 0.35
            + epistemic_factor * 0.20
            + ethical_factor * 0.15
            + principles_factor * 0.30;

        weighted.clamp(0.0, 1.0)
    }

    /// Get summary for display
    pub fn summary(&self) -> AGCSummary {
        AGCSummary {
            coherence_score: self.coherence_score,
            soc: SOCSummary {
                comprehensibility: self.psychological.comprehensibility,
                manageability: self.psychological.manageability,
                meaningfulness: self.psychological.meaningfulness,
                combined: self.psychological.soc_score(),
            },
            dissonance_level: self.psychological.dissonance_level,
            dissonance_sources: self.psychological.dissonance_sources.len(),
            regulation_mode: format!("{:?}", self.psychological.regulation_mode),
            variety_ratio: self.variety.variety_ratio,
            has_requisite_variety: self.variety.has_requisite_variety(),
            expansion_recommendations: self.variety.expansion_needed.len(),
            equilibrium_status: format!("{:?}", self.homeostasis.status()),
            regulation_intensity: self.homeostasis.controls.regulation_intensity,
            epistemic_integrity: self.epistemic.integrity_score(),
            calibration_error: self.epistemic.calibration_error,
            ethical_score: self.ethical.ethical_score(),
            principles: self.principles.summary(),
            step: self.step,
        }
    }

    /// Print formatted summary
    pub fn print_summary(&self) {
        let s = self.summary();
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║         ARTIFICIAL GENERAL COHERENCE (AGC) REPORT               ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║ Step: {:>6}    Overall Coherence: {:.3}                          ║",
            s.step, s.coherence_score
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ SENSE OF COHERENCE (SOC):                                        ║");
        println!(
            "║   Comprehensibility: {:.3}  Manageability: {:.3}  Meaning: {:.3}   ║",
            s.soc.comprehensibility, s.soc.manageability, s.soc.meaningfulness
        );
        println!(
            "║   Combined SOC: {:.3}                                             ║",
            s.soc.combined
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ COGNITIVE DISSONANCE:                                            ║");
        println!(
            "║   Level: {:.3}    Sources: {:>2}    Mode: {:20}     ║",
            s.dissonance_level, s.dissonance_sources, s.regulation_mode
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ REQUISITE VARIETY (Ashby's Law):                                 ║");
        println!(
            "║   Ratio: {:.3}    Sufficient: {:5}    Expansions Needed: {:>2}     ║",
            s.variety_ratio, s.has_requisite_variety, s.expansion_recommendations
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ HOMEOSTATIC EQUILIBRIUM:                                         ║");
        println!(
            "║   Status: {:15}    Regulation Intensity: {:.3}          ║",
            s.equilibrium_status, s.regulation_intensity
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ EPISTEMIC & ETHICAL:                                             ║");
        println!(
            "║   Epistemic Integrity: {:.3}    Calibration Error: {:.3}          ║",
            s.epistemic_integrity, s.calibration_error
        );
        println!(
            "║   Ethical Score: {:.3}                                            ║",
            s.ethical_score
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ FOUNDATIONAL PRINCIPLES:                                         ║");
        println!(
            "║   Embodiment: {:.2}  Grounding: {:.2}  Causality: {:.2}  Memory: {:.2}  ║",
            s.principles.embodiment,
            s.principles.symbol_grounding,
            s.principles.causality,
            s.principles.memory
        );
        println!(
            "║   Metacog: {:.2}  Autonomy: {:.2}  Explicability: {:.2}  Integ: {:.2}   ║",
            s.principles.metacognition,
            s.principles.autonomy,
            s.principles.explicability,
            s.principles.integration
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

#[derive(Debug, Clone)]
pub struct AGCSummary {
    pub coherence_score: f64,
    pub soc: SOCSummary,
    pub dissonance_level: f64,
    pub dissonance_sources: usize,
    pub regulation_mode: String,
    pub variety_ratio: f64,
    pub has_requisite_variety: bool,
    pub expansion_recommendations: usize,
    pub equilibrium_status: String,
    pub regulation_intensity: f64,
    pub epistemic_integrity: f64,
    pub calibration_error: f64,
    pub ethical_score: f64,
    pub principles: PrinciplesSummary,
    pub step: usize,
}

#[derive(Debug, Clone)]
pub struct SOCSummary {
    pub comprehensibility: f64,
    pub manageability: f64,
    pub meaningfulness: f64,
    pub combined: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psychological_coherence() {
        let mut pc = PsychologicalCoherence::new();

        // Update SOC
        pc.update_soc(0.8, 0.7, 0.6);
        assert!(pc.soc_score() > 0.5);

        // Register dissonance
        pc.register_dissonance(DissonanceType::PredictionError, 0.5, "expected", "actual");
        assert!(pc.dissonance_level > 0.0);
        assert_eq!(pc.dissonance_sources.len(), 1);

        // Resolution
        pc.record_resolution_success(DissonanceType::PredictionError);
        assert_eq!(pc.dissonance_sources.len(), 0);
    }

    #[test]
    fn test_requisite_variety() {
        let mut rv = RequisiteVariety::new();

        // Observe environment
        for i in 0..100 {
            let state = vec![i as f64 * 0.1, (i % 10) as f64 * 0.1];
            rv.observe_environment(&state);
            rv.observe_response(i % 4, 4, &state);
        }

        assert!(rv.variety_ratio > 0.0);
        assert!(rv.environmental_variety.distinct_count > 0);
    }

    #[test]
    fn test_homeostatic_equilibrium() {
        let mut he = HomeostaticEquilibrium::new();

        // Simulate some deviation
        he.update_current(5.0, 0.5, 0.005, 0.3, 0.3, 0.6);
        he.regulate();

        assert!(he.controls.regulation_intensity > 0.0);
        assert_ne!(he.status(), EquilibriumStatus::Stable);
    }

    #[test]
    fn test_agc_integration() {
        let mut agc = ArtificialGeneralCoherence::new();

        // Run some steps
        for i in 0..100 {
            let state = vec![i as f64 * 0.01, 0.5];
            let action = i % 4;
            let reward = if i % 10 == 0 { 1.0 } else { 0.1 };
            let prediction_error = 0.2;

            agc.update(&state, action, 4, reward, prediction_error, 10.0, 0.5);
        }

        assert!(agc.coherence_score > 0.0);
        assert!(agc.coherence_score <= 1.0);

        let summary = agc.summary();
        assert_eq!(summary.step, 100);
    }
}
