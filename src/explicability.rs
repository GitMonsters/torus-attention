//! # Explicability Module for AGI
//!
//! This module provides natural language explanations for AGI decisions,
//! making the system's reasoning transparent and interpretable.
//!
//! ## Key Features
//!
//! 1. **Decision Tracing**: Track the causal chain leading to decisions
//! 2. **Natural Language Generation**: Convert internal states to human-readable text
//! 3. **Counterfactual Explanations**: "Why not X?" explanations
//! 4. **Multi-Level Explanations**: From technical to layperson-friendly
//! 5. **Interactive Queries**: Answer follow-up questions about decisions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Explicability System                            │
//! │                                                                     │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
//! │  │  Decision   │    │ Explanation │    │    Natural Language     │ │
//! │  │   Tracer    │───▶│  Generator  │───▶│       Renderer          │ │
//! │  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
//! │         │                  │                       │               │
//! │         ▼                  ▼                       ▼               │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                  Explanation Types                           │   │
//! │  │  • Causal: "Because X caused Y"                             │   │
//! │  │  • Contrastive: "X instead of Y because..."                 │   │
//! │  │  • Counterfactual: "If X had been different..."             │   │
//! │  │  • Goal-based: "To achieve goal Z..."                       │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the explicability system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicabilityConfig {
    /// Maximum trace depth for decision tracking
    pub max_trace_depth: usize,
    /// Maximum number of explanations to cache
    pub max_cached_explanations: usize,
    /// Default explanation detail level
    pub default_detail_level: DetailLevel,
    /// Include confidence scores in explanations
    pub include_confidence: bool,
    /// Include timestamps in explanations
    pub include_timestamps: bool,
    /// Maximum factors to include in explanation
    pub max_factors: usize,
    /// Enable counterfactual generation
    pub enable_counterfactuals: bool,
    /// Minimum importance to include a factor
    pub importance_threshold: f64,
}

impl Default for ExplicabilityConfig {
    fn default() -> Self {
        Self {
            max_trace_depth: 10,
            max_cached_explanations: 100,
            default_detail_level: DetailLevel::Standard,
            include_confidence: true,
            include_timestamps: false,
            max_factors: 5,
            enable_counterfactuals: true,
            importance_threshold: 0.1,
        }
    }
}

/// Level of detail for explanations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Brief, one-sentence explanation
    Brief,
    /// Standard explanation with key factors
    Standard,
    /// Detailed explanation with full reasoning chain
    Detailed,
    /// Technical explanation with internal state details
    Technical,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECISION TRACING
// ═══════════════════════════════════════════════════════════════════════════════

/// A traced decision with its causal history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedDecision {
    /// Unique decision ID
    pub id: u64,
    /// Timestamp (step number)
    pub timestamp: usize,
    /// The action/decision taken
    pub action: DecisionAction,
    /// Factors that contributed to this decision
    pub factors: Vec<DecisionFactor>,
    /// Alternatives that were considered
    pub alternatives: Vec<Alternative>,
    /// Goals this decision serves
    pub serving_goals: Vec<String>,
    /// Confidence in the decision
    pub confidence: f64,
    /// Outcome (if known)
    pub outcome: Option<DecisionOutcome>,
}

/// The action or decision that was made
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAction {
    /// Action type
    pub action_type: String,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Human-readable description
    pub description: String,
}

/// A factor contributing to a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionFactor {
    /// Factor name/type
    pub name: String,
    /// Factor value
    pub value: f64,
    /// How much this factor contributed (importance)
    pub importance: f64,
    /// Direction of influence (positive/negative)
    pub direction: InfluenceDirection,
    /// Source of this factor
    pub source: FactorSource,
    /// Human-readable description
    pub description: String,
}

/// Direction of influence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfluenceDirection {
    Positive, // Supports the decision
    Negative, // Argues against (but was overridden)
    Neutral,  // Context without direction
}

/// Source of a decision factor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorSource {
    Perception, // Sensory input
    Memory,     // Retrieved from memory
    Goal,       // Goal-related
    Prediction, // World model prediction
    Emotion,    // Emotional/valence
    Rule,       // Explicit rule
    Learning,   // Learned association
    Default,    // Default behavior
}

/// An alternative that was considered but not chosen
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    /// Alternative action
    pub action: DecisionAction,
    /// Why it was not chosen
    pub rejection_reason: String,
    /// Score compared to chosen action
    pub relative_score: f64,
}

/// Outcome of a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Whether the decision succeeded
    pub success: bool,
    /// Reward received
    pub reward: f64,
    /// Description of what happened
    pub description: String,
    /// Timestamp of outcome
    pub timestamp: usize,
}

/// Decision tracer - tracks decisions and their causes
#[derive(Debug, Clone)]
pub struct DecisionTracer {
    /// Traced decisions
    decisions: VecDeque<TracedDecision>,
    /// Next decision ID
    next_id: u64,
    /// Configuration
    config: ExplicabilityConfig,
}

impl DecisionTracer {
    pub fn new(config: ExplicabilityConfig) -> Self {
        Self {
            decisions: VecDeque::with_capacity(config.max_cached_explanations),
            next_id: 0,
            config,
        }
    }

    /// Record a new decision
    pub fn record_decision(
        &mut self,
        action: DecisionAction,
        factors: Vec<DecisionFactor>,
        alternatives: Vec<Alternative>,
        serving_goals: Vec<String>,
        confidence: f64,
        timestamp: usize,
    ) -> u64 {
        let decision = TracedDecision {
            id: self.next_id,
            timestamp,
            action,
            factors,
            alternatives,
            serving_goals,
            confidence,
            outcome: None,
        };

        let id = decision.id;
        self.decisions.push_back(decision);
        self.next_id += 1;

        // Prune if over capacity
        while self.decisions.len() > self.config.max_cached_explanations {
            self.decisions.pop_front();
        }

        id
    }

    /// Record the outcome of a decision
    pub fn record_outcome(&mut self, decision_id: u64, outcome: DecisionOutcome) {
        if let Some(decision) = self.decisions.iter_mut().find(|d| d.id == decision_id) {
            decision.outcome = Some(outcome);
        }
    }

    /// Get a decision by ID
    pub fn get_decision(&self, decision_id: u64) -> Option<&TracedDecision> {
        self.decisions.iter().find(|d| d.id == decision_id)
    }

    /// Get recent decisions
    pub fn recent_decisions(&self, n: usize) -> Vec<&TracedDecision> {
        self.decisions.iter().rev().take(n).collect()
    }

    /// Get decisions in a time range
    pub fn decisions_in_range(&self, start: usize, end: usize) -> Vec<&TracedDecision> {
        self.decisions
            .iter()
            .filter(|d| d.timestamp >= start && d.timestamp <= end)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLANATION GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of explanation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationType {
    /// Causal explanation: "Because X caused Y"
    Causal,
    /// Contrastive: "X instead of Y because..."
    Contrastive,
    /// Counterfactual: "If X had been different..."
    Counterfactual,
    /// Goal-based: "To achieve goal Z..."
    GoalBased,
    /// Temporal: "After X happened, then Y..."
    Temporal,
    /// Analogical: "Similar to when..."
    Analogical,
}

/// A generated explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    /// Explanation ID
    pub id: u64,
    /// Decision being explained
    pub decision_id: u64,
    /// Type of explanation
    pub explanation_type: ExplanationType,
    /// Detail level
    pub detail_level: DetailLevel,
    /// The explanation text
    pub text: String,
    /// Key factors highlighted
    pub key_factors: Vec<String>,
    /// Confidence in the explanation
    pub confidence: f64,
    /// Follow-up questions that could be asked
    pub follow_up_questions: Vec<String>,
}

/// Explanation generator
#[derive(Debug, Clone)]
pub struct ExplanationGenerator {
    /// Generated explanations cache
    explanations: HashMap<u64, Vec<Explanation>>,
    /// Next explanation ID
    next_id: u64,
    /// Configuration
    config: ExplicabilityConfig,
    /// Templates for natural language generation
    templates: ExplanationTemplates,
}

/// Templates for generating natural language explanations
#[derive(Debug, Clone)]
struct ExplanationTemplates {
    causal: Vec<String>,
    contrastive: Vec<String>,
    counterfactual: Vec<String>,
    goal_based: Vec<String>,
}

impl Default for ExplanationTemplates {
    fn default() -> Self {
        Self {
            causal: vec![
                "I decided to {action} because {reason}.".to_string(),
                "The decision to {action} was based on {reason}.".to_string(),
                "{action} was chosen due to {reason}.".to_string(),
            ],
            contrastive: vec![
                "I chose {action} instead of {alternative} because {reason}.".to_string(),
                "{action} was preferred over {alternative} since {reason}.".to_string(),
                "Rather than {alternative}, I opted for {action} because {reason}.".to_string(),
            ],
            counterfactual: vec![
                "If {condition} had been different, I would have {alternative_action}.".to_string(),
                "Had {condition} been the case, the outcome would have been {alternative_outcome}."
                    .to_string(),
            ],
            goal_based: vec![
                "To achieve {goal}, I decided to {action}.".to_string(),
                "The action {action} was taken in pursuit of {goal}.".to_string(),
                "{action} serves the goal of {goal}.".to_string(),
            ],
        }
    }
}

impl ExplanationGenerator {
    pub fn new(config: ExplicabilityConfig) -> Self {
        Self {
            explanations: HashMap::new(),
            next_id: 0,
            config,
            templates: ExplanationTemplates::default(),
        }
    }

    /// Generate an explanation for a decision
    pub fn explain(
        &mut self,
        decision: &TracedDecision,
        explanation_type: ExplanationType,
        detail_level: Option<DetailLevel>,
    ) -> Explanation {
        let detail = detail_level.unwrap_or(self.config.default_detail_level);

        let text = match explanation_type {
            ExplanationType::Causal => self.generate_causal(decision, detail),
            ExplanationType::Contrastive => self.generate_contrastive(decision, detail),
            ExplanationType::Counterfactual => self.generate_counterfactual(decision, detail),
            ExplanationType::GoalBased => self.generate_goal_based(decision, detail),
            ExplanationType::Temporal => self.generate_temporal(decision, detail),
            ExplanationType::Analogical => self.generate_analogical(decision, detail),
        };

        let key_factors: Vec<String> = decision
            .factors
            .iter()
            .filter(|f| f.importance >= self.config.importance_threshold)
            .take(self.config.max_factors)
            .map(|f| f.name.clone())
            .collect();

        let follow_up_questions = self.generate_follow_ups(decision, explanation_type);

        let explanation = Explanation {
            id: self.next_id,
            decision_id: decision.id,
            explanation_type,
            detail_level: detail,
            text,
            key_factors,
            confidence: decision.confidence,
            follow_up_questions,
        };

        // Cache the explanation
        self.explanations
            .entry(decision.id)
            .or_insert_with(Vec::new)
            .push(explanation.clone());

        self.next_id += 1;

        explanation
    }

    fn generate_causal(&self, decision: &TracedDecision, detail: DetailLevel) -> String {
        let action = &decision.action.description;

        // Get top factors by importance
        let mut factors: Vec<&DecisionFactor> = decision
            .factors
            .iter()
            .filter(|f| f.direction == InfluenceDirection::Positive)
            .collect();
        factors.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let reason = match detail {
            DetailLevel::Brief => factors
                .first()
                .map(|f| f.description.clone())
                .unwrap_or_else(|| "various factors".to_string()),
            DetailLevel::Standard => {
                let descriptions: Vec<&str> = factors
                    .iter()
                    .take(3)
                    .map(|f| f.description.as_str())
                    .collect();
                descriptions.join(", and ")
            }
            DetailLevel::Detailed | DetailLevel::Technical => {
                let mut parts = Vec::new();
                for factor in factors.iter().take(5) {
                    let confidence_str = if self.config.include_confidence {
                        format!(" (importance: {:.0}%)", factor.importance * 100.0)
                    } else {
                        String::new()
                    };
                    parts.push(format!("{}{}", factor.description, confidence_str));
                }
                parts.join("; ")
            }
        };

        let template = &self.templates.causal[0];
        template
            .replace("{action}", action)
            .replace("{reason}", &reason)
    }

    fn generate_contrastive(&self, decision: &TracedDecision, detail: DetailLevel) -> String {
        let action = &decision.action.description;

        if decision.alternatives.is_empty() {
            return format!("I chose to {} (no alternatives were considered).", action);
        }

        let best_alt = &decision.alternatives[0];
        let alternative = &best_alt.action.description;
        let reason = &best_alt.rejection_reason;

        let template = &self.templates.contrastive[0];
        let base = template
            .replace("{action}", action)
            .replace("{alternative}", alternative)
            .replace("{reason}", reason);

        match detail {
            DetailLevel::Brief => base,
            DetailLevel::Standard => {
                format!(
                    "{} The alternative scored {:.0}% compared to the chosen action.",
                    base,
                    best_alt.relative_score * 100.0
                )
            }
            DetailLevel::Detailed | DetailLevel::Technical => {
                let alt_list: Vec<String> = decision
                    .alternatives
                    .iter()
                    .take(3)
                    .map(|a| {
                        format!(
                            "'{}' (score: {:.0}%, reason: {})",
                            a.action.description,
                            a.relative_score * 100.0,
                            a.rejection_reason
                        )
                    })
                    .collect();
                format!(
                    "{}\n\nOther alternatives considered:\n- {}",
                    base,
                    alt_list.join("\n- ")
                )
            }
        }
    }

    fn generate_counterfactual(&self, decision: &TracedDecision, detail: DetailLevel) -> String {
        if !self.config.enable_counterfactuals {
            return "Counterfactual explanations are disabled.".to_string();
        }

        // Find the most important positive factor
        let key_factor = decision
            .factors
            .iter()
            .filter(|f| f.direction == InfluenceDirection::Positive)
            .max_by(|a, b| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(factor) = key_factor {
            let condition = format!("{} had been different", factor.name);
            let alternative = if !decision.alternatives.is_empty() {
                decision.alternatives[0].action.description.clone()
            } else {
                "chosen differently".to_string()
            };

            let base = format!(
                "If {} had been different, I might have {}.",
                factor.description, alternative
            );

            match detail {
                DetailLevel::Brief | DetailLevel::Standard => base,
                DetailLevel::Detailed | DetailLevel::Technical => {
                    format!(
                        "{}\n\nThe factor '{}' contributed {:.0}% to the decision. \
                        Changing this would have shifted the outcome significantly.",
                        base,
                        factor.name,
                        factor.importance * 100.0
                    )
                }
            }
        } else {
            "No clear counterfactual can be identified for this decision.".to_string()
        }
    }

    fn generate_goal_based(&self, decision: &TracedDecision, detail: DetailLevel) -> String {
        let action = &decision.action.description;

        if decision.serving_goals.is_empty() {
            return format!(
                "I chose to {}. No explicit goals were being pursued.",
                action
            );
        }

        let goal = &decision.serving_goals[0];
        let template = &self.templates.goal_based[0];
        let base = template.replace("{action}", action).replace("{goal}", goal);

        match detail {
            DetailLevel::Brief => base,
            DetailLevel::Standard => {
                if decision.serving_goals.len() > 1 {
                    format!(
                        "{} This also contributes to: {}.",
                        base,
                        decision.serving_goals[1..].join(", ")
                    )
                } else {
                    base
                }
            }
            DetailLevel::Detailed | DetailLevel::Technical => {
                let goal_list: Vec<String> = decision
                    .serving_goals
                    .iter()
                    .enumerate()
                    .map(|(i, g)| format!("{}. {}", i + 1, g))
                    .collect();
                format!(
                    "{}\n\nAll goals being served:\n{}",
                    base,
                    goal_list.join("\n")
                )
            }
        }
    }

    fn generate_temporal(&self, decision: &TracedDecision, _detail: DetailLevel) -> String {
        let action = &decision.action.description;
        format!(
            "At step {}, I decided to {}. This followed from the sequence of prior observations and actions.",
            decision.timestamp, action
        )
    }

    fn generate_analogical(&self, decision: &TracedDecision, _detail: DetailLevel) -> String {
        let action = &decision.action.description;
        format!(
            "The decision to {} was based on patterns similar to previous successful decisions.",
            action
        )
    }

    fn generate_follow_ups(
        &self,
        decision: &TracedDecision,
        exp_type: ExplanationType,
    ) -> Vec<String> {
        let mut questions = Vec::new();

        // Common follow-ups
        questions.push("Why was this the best option?".to_string());

        if !decision.alternatives.is_empty() {
            questions.push(format!(
                "Why not {}?",
                decision.alternatives[0].action.description
            ));
        }

        // Type-specific follow-ups
        match exp_type {
            ExplanationType::Causal => {
                questions.push("What would have happened otherwise?".to_string());
            }
            ExplanationType::Contrastive => {
                questions.push("What made the chosen action better?".to_string());
            }
            ExplanationType::Counterfactual => {
                questions.push("What conditions would change your decision?".to_string());
            }
            ExplanationType::GoalBased => {
                questions.push("How does this help achieve the goal?".to_string());
            }
            _ => {}
        }

        if decision.outcome.is_none() {
            questions.push("What do you expect to happen?".to_string());
        } else {
            questions.push("Did it work as expected?".to_string());
        }

        questions
    }

    /// Get cached explanations for a decision
    pub fn get_explanations(&self, decision_id: u64) -> Option<&Vec<Explanation>> {
        self.explanations.get(&decision_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERACTIVE EXPLANATION INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

/// Query for an explanation
#[derive(Debug, Clone)]
pub struct ExplanationQuery {
    /// The question being asked
    pub question: String,
    /// Context (e.g., decision ID)
    pub context: Option<u64>,
    /// Requested detail level
    pub detail_level: Option<DetailLevel>,
    /// Requested explanation type
    pub explanation_type: Option<ExplanationType>,
}

/// Response to an explanation query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationResponse {
    /// The original question
    pub question: String,
    /// The answer
    pub answer: String,
    /// Related explanations
    pub related_explanations: Vec<Explanation>,
    /// Suggested follow-up questions
    pub follow_ups: Vec<String>,
    /// Confidence in the response
    pub confidence: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED EXPLICABILITY SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified explicability system
#[derive(Debug, Clone)]
pub struct ExplicabilitySystem {
    /// Decision tracer
    pub tracer: DecisionTracer,
    /// Explanation generator
    pub generator: ExplanationGenerator,
    /// Configuration
    config: ExplicabilityConfig,
    /// Current step
    current_step: usize,
}

impl ExplicabilitySystem {
    pub fn new(config: ExplicabilityConfig) -> Self {
        Self {
            tracer: DecisionTracer::new(config.clone()),
            generator: ExplanationGenerator::new(config.clone()),
            config,
            current_step: 0,
        }
    }

    /// Record a decision with factors
    pub fn record_decision(
        &mut self,
        action_type: &str,
        action_description: &str,
        factors: Vec<(String, f64, f64, InfluenceDirection, FactorSource)>,
        alternatives: Vec<(String, String, f64)>,
        goals: Vec<String>,
        confidence: f64,
    ) -> u64 {
        let action = DecisionAction {
            action_type: action_type.to_string(),
            parameters: HashMap::new(),
            description: action_description.to_string(),
        };

        let decision_factors: Vec<DecisionFactor> = factors
            .into_iter()
            .map(
                |(name, value, importance, direction, source)| DecisionFactor {
                    name: name.clone(),
                    value,
                    importance,
                    direction,
                    source,
                    description: name,
                },
            )
            .collect();

        let decision_alternatives: Vec<Alternative> = alternatives
            .into_iter()
            .map(|(desc, reason, score)| Alternative {
                action: DecisionAction {
                    action_type: "alternative".to_string(),
                    parameters: HashMap::new(),
                    description: desc,
                },
                rejection_reason: reason,
                relative_score: score,
            })
            .collect();

        let id = self.tracer.record_decision(
            action,
            decision_factors,
            decision_alternatives,
            goals,
            confidence,
            self.current_step,
        );

        self.current_step += 1;
        id
    }

    /// Record the outcome of a decision
    pub fn record_outcome(
        &mut self,
        decision_id: u64,
        success: bool,
        reward: f64,
        description: &str,
    ) {
        let outcome = DecisionOutcome {
            success,
            reward,
            description: description.to_string(),
            timestamp: self.current_step,
        };
        self.tracer.record_outcome(decision_id, outcome);
    }

    /// Get an explanation for a decision
    pub fn explain(
        &mut self,
        decision_id: u64,
        explanation_type: ExplanationType,
    ) -> Option<Explanation> {
        let decision = self.tracer.get_decision(decision_id)?;
        Some(self.generator.explain(decision, explanation_type, None))
    }

    /// Get all types of explanations for a decision
    pub fn explain_all(&mut self, decision_id: u64) -> Vec<Explanation> {
        let decision = match self.tracer.get_decision(decision_id) {
            Some(d) => d.clone(),
            None => return vec![],
        };

        let types = [
            ExplanationType::Causal,
            ExplanationType::Contrastive,
            ExplanationType::GoalBased,
        ];

        types
            .iter()
            .map(|&t| self.generator.explain(&decision, t, None))
            .collect()
    }

    /// Answer a natural language question about a decision
    pub fn answer_query(&mut self, query: ExplanationQuery) -> ExplanationResponse {
        let question = query.question.to_lowercase();

        // Determine what kind of question this is
        let exp_type = query.explanation_type.unwrap_or_else(|| {
            if question.contains("why not") || question.contains("instead") {
                ExplanationType::Contrastive
            } else if question.contains("if") || question.contains("would have") {
                ExplanationType::Counterfactual
            } else if question.contains("goal") || question.contains("purpose") {
                ExplanationType::GoalBased
            } else {
                ExplanationType::Causal
            }
        });

        // Get the relevant decision
        let decision = if let Some(id) = query.context {
            self.tracer.get_decision(id).cloned()
        } else {
            self.tracer.recent_decisions(1).first().cloned().cloned()
        };

        match decision {
            Some(d) => {
                let explanation = self.generator.explain(&d, exp_type, query.detail_level);

                ExplanationResponse {
                    question: query.question,
                    answer: explanation.text.clone(),
                    related_explanations: vec![explanation.clone()],
                    follow_ups: explanation.follow_up_questions,
                    confidence: explanation.confidence,
                }
            }
            None => ExplanationResponse {
                question: query.question,
                answer: "I don't have a record of that decision.".to_string(),
                related_explanations: vec![],
                follow_ups: vec!["What decision are you asking about?".to_string()],
                confidence: 0.0,
            },
        }
    }

    /// Get a summary of recent decisions with brief explanations
    pub fn summarize_recent(&mut self, n: usize) -> Vec<(u64, String)> {
        let decisions: Vec<TracedDecision> = self
            .tracer
            .recent_decisions(n)
            .into_iter()
            .cloned()
            .collect();

        decisions
            .iter()
            .map(|d| {
                let exp =
                    self.generator
                        .explain(d, ExplanationType::Causal, Some(DetailLevel::Brief));
                (d.id, exp.text)
            })
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> ExplicabilitySummary {
        ExplicabilitySummary {
            total_decisions: self.tracer.len(),
            decisions_with_outcomes: self
                .tracer
                .decisions
                .iter()
                .filter(|d| d.outcome.is_some())
                .count(),
            successful_decisions: self
                .tracer
                .decisions
                .iter()
                .filter(|d| d.outcome.as_ref().map(|o| o.success).unwrap_or(false))
                .count(),
            average_confidence: if self.tracer.is_empty() {
                0.0
            } else {
                self.tracer
                    .decisions
                    .iter()
                    .map(|d| d.confidence)
                    .sum::<f64>()
                    / self.tracer.len() as f64
            },
            total_explanations: self.generator.explanations.values().map(|v| v.len()).sum(),
            current_step: self.current_step,
        }
    }
}

/// Summary of explicability system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicabilitySummary {
    pub total_decisions: usize,
    pub decisions_with_outcomes: usize,
    pub successful_decisions: usize,
    pub average_confidence: f64,
    pub total_explanations: usize,
    pub current_step: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicability_system_creation() {
        let config = ExplicabilityConfig::default();
        let system = ExplicabilitySystem::new(config);

        assert!(system.tracer.is_empty());
    }

    #[test]
    fn test_record_decision() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![
                (
                    "goal_proximity".to_string(),
                    0.8,
                    0.7,
                    InfluenceDirection::Positive,
                    FactorSource::Goal,
                ),
                (
                    "obstacle_free".to_string(),
                    1.0,
                    0.3,
                    InfluenceDirection::Positive,
                    FactorSource::Perception,
                ),
            ],
            vec![(
                "move south".to_string(),
                "moves away from goal".to_string(),
                0.3,
            )],
            vec!["reach the target".to_string()],
            0.85,
        );

        assert_eq!(id, 0);
        assert_eq!(system.tracer.len(), 1);
    }

    #[test]
    fn test_record_outcome() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision("move", "move north", vec![], vec![], vec![], 0.8);

        system.record_outcome(id, true, 1.0, "Reached the goal!");

        let decision = system.tracer.get_decision(id).unwrap();
        assert!(decision.outcome.is_some());
        assert!(decision.outcome.as_ref().unwrap().success);
    }

    #[test]
    fn test_generate_causal_explanation() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![(
                "goal is north".to_string(),
                0.9,
                0.8,
                InfluenceDirection::Positive,
                FactorSource::Goal,
            )],
            vec![],
            vec!["find the goal".to_string()],
            0.9,
        );

        let explanation = system.explain(id, ExplanationType::Causal).unwrap();

        assert!(explanation.text.contains("move north"));
        assert!(!explanation.key_factors.is_empty());
    }

    #[test]
    fn test_generate_contrastive_explanation() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![],
            vec![("move south".to_string(), "wrong direction".to_string(), 0.3)],
            vec![],
            0.8,
        );

        let explanation = system.explain(id, ExplanationType::Contrastive).unwrap();

        assert!(explanation.text.contains("instead of") || explanation.text.contains("move south"));
    }

    #[test]
    fn test_generate_goal_based_explanation() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![],
            vec![],
            vec!["reach the treasure".to_string()],
            0.85,
        );

        let explanation = system.explain(id, ExplanationType::GoalBased).unwrap();

        assert!(
            explanation.text.contains("reach the treasure") || explanation.text.contains("goal")
        );
    }

    #[test]
    fn test_answer_query() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        system.record_decision(
            "move",
            "move north",
            vec![(
                "goal proximity".to_string(),
                0.8,
                0.7,
                InfluenceDirection::Positive,
                FactorSource::Goal,
            )],
            vec![],
            vec!["find the goal".to_string()],
            0.85,
        );

        let query = ExplanationQuery {
            question: "Why did you move north?".to_string(),
            context: Some(0),
            detail_level: None,
            explanation_type: None,
        };

        let response = system.answer_query(query);

        assert!(!response.answer.is_empty());
        assert!(response.confidence > 0.0);
    }

    #[test]
    fn test_explain_all() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![(
                "goal proximity".to_string(),
                0.8,
                0.7,
                InfluenceDirection::Positive,
                FactorSource::Goal,
            )],
            vec![("move south".to_string(), "wrong direction".to_string(), 0.3)],
            vec!["find the goal".to_string()],
            0.85,
        );

        let explanations = system.explain_all(id);

        // Should have causal, contrastive, and goal-based
        assert_eq!(explanations.len(), 3);
    }

    #[test]
    fn test_summarize_recent() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        for i in 0..5 {
            system.record_decision(
                "move",
                &format!("move direction {}", i),
                vec![],
                vec![],
                vec![],
                0.8,
            );
        }

        let summaries = system.summarize_recent(3);
        assert_eq!(summaries.len(), 3);
    }

    #[test]
    fn test_counterfactual_explanation() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![(
                "path clear".to_string(),
                1.0,
                0.9,
                InfluenceDirection::Positive,
                FactorSource::Perception,
            )],
            vec![("stay".to_string(), "no progress".to_string(), 0.2)],
            vec![],
            0.8,
        );

        let explanation = system.explain(id, ExplanationType::Counterfactual).unwrap();

        assert!(explanation.text.contains("If") || explanation.text.contains("different"));
    }

    #[test]
    fn test_follow_up_questions() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id = system.record_decision(
            "move",
            "move north",
            vec![],
            vec![("move south".to_string(), "blocked".to_string(), 0.1)],
            vec![],
            0.8,
        );

        let explanation = system.explain(id, ExplanationType::Causal).unwrap();

        assert!(!explanation.follow_up_questions.is_empty());
    }

    #[test]
    fn test_summary() {
        let config = ExplicabilityConfig::default();
        let mut system = ExplicabilitySystem::new(config);

        let id1 = system.record_decision("move", "move north", vec![], vec![], vec![], 0.9);
        let id2 = system.record_decision("move", "move south", vec![], vec![], vec![], 0.7);

        system.record_outcome(id1, true, 1.0, "success");
        system.record_outcome(id2, false, -0.5, "failed");

        let summary = system.summary();

        assert_eq!(summary.total_decisions, 2);
        assert_eq!(summary.decisions_with_outcomes, 2);
        assert_eq!(summary.successful_decisions, 1);
        assert!((summary.average_confidence - 0.8).abs() < 0.01);
    }
}
