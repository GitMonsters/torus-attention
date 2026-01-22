//! # Learning Experiment: Testing Compounding Cognitive Cohesion
//!
//! This experiment tests whether the compounding cohesion system actually learns
//! and adapts over multiple episodes.
//!
//! Key metrics tracked:
//! - Episode rewards (should improve over time)
//! - Exploration coverage (should increase)
//! - Coherence stability (should stabilize)
//! - Graph memory growth (should accumulate knowledge)
//! - Cognitive dissonance resolution (should adapt to changes)
//! - AGI Reasoning: Causal inference, MCTS planning, stream voting
//! - **NEW** Unified AGI Core: Causal discovery, abstraction, world model, goals, meta-learning, symbols
//!
//! Run with:
//! ```bash
//! cargo run --example learning_experiment --release
//! ```

use candle_core::Device;
use candle_nn::VarMap;
use std::time::Instant;
use torus_attention::{
    agi_core::{AGICore, AGICoreConfig, GoalPriority},
    compounding_cohesion::{CompoundingCohesionConfig, GoalType},
    compounding_transformer::{CompoundingCohesionTransformer, CompoundingTransformerConfig},
    sensorimotor::{
        CognitiveDissonanceTracker, Environment, LearningEnvConfig, LearningGridEnvironment,
        MotorSystem, Observation, SensorimotorAgent, SensorimotorConfig,
    },
    CoherenceConfig, TorusResult,
};

/// Learning metrics for a single episode
#[derive(Debug, Clone)]
struct EpisodeMetrics {
    episode: usize,
    total_reward: f64,
    coverage: f64,
    final_coherence: f64,
    prediction_error: f64,
    graph_nodes: usize,
    graph_edges: usize,
    steps: usize,
    novelty_reward: f64,
    goal_reward: f64,
    exploration_reward: f64,
    dissonance_events: usize,
    // AGI Core metrics
    agi_compound_rate: f64,
    agi_discovered_variables: usize,
    agi_concepts: usize,
    agi_symbols: usize,
    agi_active_goals: usize,
}

/// Aggregate metrics over multiple episodes
#[derive(Debug)]
struct LearningMetrics {
    episodes: Vec<EpisodeMetrics>,
    running_avg_reward: Vec<f64>,
    running_avg_coherence: Vec<f64>,
    running_avg_coverage: Vec<f64>,
}

impl LearningMetrics {
    fn new() -> Self {
        Self {
            episodes: Vec::new(),
            running_avg_reward: Vec::new(),
            running_avg_coherence: Vec::new(),
            running_avg_coverage: Vec::new(),
        }
    }

    fn record(&mut self, metrics: EpisodeMetrics) {
        let window = 5; // Running average window

        self.episodes.push(metrics);

        // Compute running averages
        let n = self.episodes.len();
        let start = n.saturating_sub(window);

        let avg_reward: f64 = self.episodes[start..n]
            .iter()
            .map(|e| e.total_reward)
            .sum::<f64>()
            / (n - start) as f64;
        self.running_avg_reward.push(avg_reward);

        let avg_coherence: f64 = self.episodes[start..n]
            .iter()
            .map(|e| e.final_coherence)
            .sum::<f64>()
            / (n - start) as f64;
        self.running_avg_coherence.push(avg_coherence);

        let avg_coverage: f64 = self.episodes[start..n]
            .iter()
            .map(|e| e.coverage)
            .sum::<f64>()
            / (n - start) as f64;
        self.running_avg_coverage.push(avg_coverage);
    }

    fn summary(&self) -> String {
        if self.episodes.is_empty() {
            return "No episodes recorded".to_string();
        }

        let n = self.episodes.len();
        let first_half = &self.episodes[..n / 2];
        let second_half = &self.episodes[n / 2..];

        let first_avg_reward: f64 =
            first_half.iter().map(|e| e.total_reward).sum::<f64>() / first_half.len() as f64;
        let second_avg_reward: f64 =
            second_half.iter().map(|e| e.total_reward).sum::<f64>() / second_half.len() as f64;

        let first_avg_coherence: f64 =
            first_half.iter().map(|e| e.final_coherence).sum::<f64>() / first_half.len() as f64;
        let second_avg_coherence: f64 =
            second_half.iter().map(|e| e.final_coherence).sum::<f64>() / second_half.len() as f64;

        let first_avg_coverage: f64 =
            first_half.iter().map(|e| e.coverage).sum::<f64>() / first_half.len() as f64;
        let second_avg_coverage: f64 =
            second_half.iter().map(|e| e.coverage).sum::<f64>() / second_half.len() as f64;

        let total_graph_nodes = self.episodes.last().map(|e| e.graph_nodes).unwrap_or(0);
        let total_graph_edges = self.episodes.last().map(|e| e.graph_edges).unwrap_or(0);

        let reward_improvement =
            (second_avg_reward - first_avg_reward) / first_avg_reward.abs().max(0.01) * 100.0;
        let coherence_change = (second_avg_coherence - first_avg_coherence)
            / first_avg_coherence.abs().max(0.01)
            * 100.0;
        let coverage_improvement =
            (second_avg_coverage - first_avg_coverage) / first_avg_coverage.abs().max(0.01) * 100.0;

        format!(
            "Learning Summary ({} episodes):\n\
             ├─ Reward:     {:.3} → {:.3} ({:+.1}%)\n\
             ├─ Coherence:  {:.3} → {:.3} ({:+.1}%)\n\
             ├─ Coverage:   {:.1}% → {:.1}% ({:+.1}%)\n\
             ├─ Graph Memory: {} nodes, {} edges\n\
             └─ Learning: {}",
            n,
            first_avg_reward,
            second_avg_reward,
            reward_improvement,
            first_avg_coherence,
            second_avg_coherence,
            coherence_change,
            first_avg_coverage * 100.0,
            second_avg_coverage * 100.0,
            coverage_improvement,
            total_graph_nodes,
            total_graph_edges,
            if reward_improvement > 5.0 || coverage_improvement > 10.0 {
                "DETECTED"
            } else {
                "NOT DETECTED"
            }
        )
    }
}

fn main() -> TorusResult<()> {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("       COMPOUNDING COGNITIVE COHESION - LEARNING EXPERIMENT");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Testing whether coherence-driven learning produces emergent behavior...");
    println!();

    let device = Device::Cpu;
    let start = Instant::now();

    // Configuration
    let n_episodes = 200; // More episodes to see learning trends
    let d_model = 64;
    let n_layers = 4;
    let n_heads = 4;
    let n_major = 4;
    let n_minor = 4;
    let seq_len = n_major * n_minor; // 16

    println!("Configuration:");
    println!("├─ Episodes: {}", n_episodes);
    println!(
        "├─ Model: d={}, layers={}, heads={}",
        d_model, n_layers, n_heads
    );
    println!("├─ Torus: {}x{} = {} positions", n_major, n_minor, seq_len);
    println!("└─ Environment: LearningGridEnvironment (10x10 with landmarks)");
    println!();

    // Create cohesion config with lower merge threshold for more unique nodes
    let cohesion_config = CompoundingCohesionConfig {
        n_layers,
        n_streams: 8,
        d_model,
        n_hierarchy_levels: 3,
        layers_per_region: 2,
        max_graph_nodes: 500,
        graph_feature_dim: 32,
        prediction_history: 100,
        meta_learning_rate: 0.015, // Slightly increased for faster adaptation
        graph_learning_rate: 0.25, // Increased for stronger edge learning
        merge_threshold: 0.60,     // Lower threshold = more unique nodes
        use_prediction: true,
        use_goal_states: true,
        use_consolidation: true,
        base_coherence: CoherenceConfig {
            n_streams: 8,
            d_model,
            smm_learning_rate: 0.1,
            base_alpha: 0.9,
            min_alpha: 0.1,
            max_alpha: 0.99,
            coherence_threshold: 0.5,
            adaptive_alpha: true,
            comprehensibility_weight: 0.33,
            manageability_weight: 0.33,
            meaningfulness_weight: 0.34,
        },
    };

    // Create transformer config (use default and override specific fields)
    let mut transformer_config = CompoundingTransformerConfig::default();
    transformer_config.d_model = d_model;
    transformer_config.d_ff = d_model * 4;
    transformer_config.n_heads = n_heads;
    transformer_config.n_layers = n_layers;
    transformer_config.n_major = n_major;
    transformer_config.n_minor = n_minor;
    transformer_config.dropout = 0.0;
    transformer_config.cohesion_config = cohesion_config;

    // Create varmap and transformer
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    println!("Creating CompoundingCohesionTransformer...");
    let transformer = CompoundingCohesionTransformer::new(
        transformer_config.clone(),
        None, // No vocab size for direct feature input
        vb,
        &device,
    )?;
    println!("Transformer created successfully.");
    println!();

    // Create environment config
    let env_config = LearningEnvConfig {
        size: 10,
        feature_dim: d_model,
        seq_len,
        n_landmarks: 8,
        n_obstacles: 4,
        max_steps: 100,
        goal_move_probability: 0.05,       // Goal moves occasionally
        obstacle_change_probability: 0.02, // Obstacles rarely change
        novelty_bonus: 0.3,                // Bonus for visiting new cells
        exploration_reward: 0.1,           // Reward for coverage
        cognitive_dissonance_rate: 0.1,    // 10% chance of dissonance per step
    };

    // Create agent with AGI reasoning enabled
    let env = LearningGridEnvironment::new(env_config.clone(), &device, 42);
    let agent_config = SensorimotorConfig {
        max_steps_per_episode: 100,
        online_learning: true,
        gamma: 0.99,
        consolidation_interval: 25,
        verbose: false,
    };

    // Create agent with AGI reasoning (8 streams, 5 actions: 4 directions + stay)
    let mut agent = SensorimotorAgent::with_agi_reasoning(
        transformer,
        Box::new(env),
        agent_config,
        8, // n_streams
        5, // n_actions (up, down, left, right, stay)
    );

    // Cognitive dissonance tracker
    let mut dissonance_tracker = CognitiveDissonanceTracker::new(0.3);
    let mut learning_metrics = LearningMetrics::new();

    // Create the unified AGI Core
    println!("Creating Unified AGI Core...");
    let agi_config = AGICoreConfig {
        max_causal_variables: 50,
        max_abstraction_depth: 4,
        world_model_horizon: 15,
        max_active_goals: 5,
        meta_learning_window: 50,
        max_symbols: 200,
        causal_discovery_threshold: 0.25,
        abstraction_merge_threshold: 0.80,
        world_model_error_threshold: 0.15,
        goal_completion_threshold: 0.85,
        symbol_grounding_threshold: 0.6,
    };
    let mut agi_core = AGICore::new(agi_config, d_model, 5); // 5 actions

    // Create initial exploration goal
    let exploration_goal = agi_core.create_goal_from_observation(
        "explore_environment",
        vec![0.5; d_model], // Generic target - will be updated
        GoalPriority::High,
    );
    println!("AGI Core initialized with exploration goal.");
    println!();

    println!("Running {} episodes...", n_episodes);
    println!();
    println!("Ep  | Reward  | Coverage | Coherence | MemGuide | MemSim | Nodes | Steps | Learning");
    println!("────┼─────────┼──────────┼───────────┼──────────┼────────┼───────┼───────┼─────────");

    for episode in 0..n_episodes {
        // Run episode
        let result = agent.run_episode()?;

        // Get coherence system stats
        let cohesion = agent.transformer.cohesion();
        let pred_error = cohesion.prediction.avg_error();

        // Count graph nodes/edges across all streams
        let graph_nodes: usize = cohesion.stream_graphs.iter().map(|g| g.nodes.len()).sum();
        let graph_edges: usize = cohesion.stream_graphs.iter().map(|g| g.edges.len()).sum();

        // Get environment stats from the actual running environment
        let (coverage, novelty_reward, goal_reward, exploration_reward) =
            agent.environment.get_stats();

        // Track dissonance
        dissonance_tracker.record_prediction_error(pred_error);
        if result.final_coherence < 0.4 {
            dissonance_tracker.record_coherence_drop(0.4 - result.final_coherence);
        }

        // Process experience through AGI Core
        // Create state representation from episode result
        let state_features: Vec<f64> = (0..d_model)
            .map(|i| {
                let base = (i as f64 / d_model as f64) * coverage;
                base + result.final_coherence * 0.1
            })
            .collect();

        // Simulate action taken (use episode modulo as proxy)
        let action = episode % 5;

        // Next state with slight perturbation
        let next_state: Vec<f64> = state_features
            .iter()
            .map(|&s| s + (result.total_reward * 0.01))
            .collect();

        // Feed to AGI Core
        agi_core.process_experience(
            &state_features,
            action,
            &next_state,
            result.total_reward as f64,
            false, // not terminal
        );

        // Get AGI Core summary for metrics
        let agi_summary = agi_core.summary();

        // Record metrics
        let metrics = EpisodeMetrics {
            episode,
            total_reward: result.total_reward,
            coverage,
            final_coherence: result.final_coherence,
            prediction_error: pred_error,
            graph_nodes,
            graph_edges,
            steps: result.steps,
            novelty_reward,
            goal_reward,
            exploration_reward,
            dissonance_events: if dissonance_tracker.in_dissonance {
                1
            } else {
                0
            },
            // AGI Core metrics
            agi_compound_rate: agi_summary.analytics.compound_rate,
            agi_discovered_variables: agi_summary.causal_discovery.total_variables,
            agi_concepts: agi_summary.abstraction.total_concepts,
            agi_symbols: agi_summary.symbols.total_symbols,
            agi_active_goals: agi_summary.goals.active_goals,
        };
        learning_metrics.record(metrics);

        // Print progress
        let learning_indicator = if episode > 5 {
            let recent = &learning_metrics.running_avg_reward;
            if recent.len() >= 2 && recent[recent.len() - 1] > recent[recent.len() - 2] {
                "+"
            } else {
                "-"
            }
        } else {
            "."
        };

        if episode % 5 == 0 || episode == n_episodes - 1 {
            println!(
                "{:3} | {:7.3} | {:7.1}% | {:9.3} | {:8} | {:6.3} | {:5} | {:5} | {}",
                episode,
                result.total_reward,
                coverage * 100.0,
                result.final_coherence,
                result.memory_guided_steps,
                result.avg_memory_similarity,
                graph_nodes,
                result.steps,
                learning_indicator
            );
        }

        // Replace environment for next episode (with new random seed)
        // Preserve AGI reasoning state across episodes
        if episode < n_episodes - 1 {
            let new_env = LearningGridEnvironment::new(
                env_config.clone(),
                &device,
                42 + (episode as u64 + 1) * 17, // Different seed each episode
            );
            // Extract AGI reasoning to preserve it
            let agi_reasoning = agent.agi_reasoning.take();
            agent = SensorimotorAgent::new(
                agent.transformer,
                Box::new(new_env),
                SensorimotorConfig {
                    max_steps_per_episode: 100,
                    online_learning: true,
                    gamma: 0.99,
                    consolidation_interval: 25,
                    verbose: false,
                },
            );
            // Restore AGI reasoning
            agent.agi_reasoning = agi_reasoning;
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                           RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("{}", learning_metrics.summary());
    println!();

    // Detailed coherence analysis
    println!("Coherence System State:");
    println!("{}", agent.transformer.cohesion().summary());
    println!();

    // Dissonance analysis
    let diss_stats = dissonance_tracker.stats();
    println!("Cognitive Dissonance Analysis:");
    println!("├─ Current Level: {:.3}", diss_stats.current_level);
    println!("├─ In Dissonance: {}", diss_stats.in_dissonance);
    println!("├─ Resolution Attempts: {}", diss_stats.resolution_attempts);
    println!(
        "└─ Avg Prediction Error: {:.4}",
        diss_stats.avg_prediction_error
    );
    println!();

    // Intrinsic motivation analysis
    if let Some(im) = agent.transformer.cohesion().spatial_intrinsic_motivation() {
        println!("Intrinsic Motivation (Spatial Graph):");
        println!("├─ Competence Progress: {:+.4}", im.competence_progress);
        println!("├─ Learning Progress: {:+.4}", im.learning_progress);
        println!("├─ Empowerment: {:.4}", im.empowerment);
        println!("└─ Information Gain: {:.4}", im.information_gain);
        println!();
    }

    // Successor representation stats
    let sr_count: usize = agent
        .transformer
        .cohesion()
        .stream_graphs
        .iter()
        .map(|g| g.successor_reps.len())
        .sum();
    let replay_count: usize = agent
        .transformer
        .cohesion()
        .stream_graphs
        .iter()
        .map(|g| g.replay_buffer.len())
        .sum();
    println!("Memory Systems:");
    println!("├─ Successor Representations: {}", sr_count);
    println!("├─ Replay Buffer Size: {}", replay_count);
    println!(
        "└─ Adaptive Temperature: {:.3}",
        agent.transformer.cohesion().temperature()
    );
    println!();

    // AGI Reasoning System Analysis
    if let Some(agi_summary) = agent.agi_summary() {
        println!("{}", agi_summary);
        println!();
    }

    // Unified AGI Core Analysis
    println!();
    agi_core.print_summary();

    let elapsed = start.elapsed();
    println!(
        "Total time: {:.2}s ({:.2}ms per episode)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / n_episodes as f64
    );
    println!();

    // Final assessment
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                         ASSESSMENT");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let first_rewards: f64 = learning_metrics.episodes[..10]
        .iter()
        .map(|e| e.total_reward)
        .sum::<f64>()
        / 10.0;
    let last_rewards: f64 = learning_metrics.episodes[learning_metrics.episodes.len() - 10..]
        .iter()
        .map(|e| e.total_reward)
        .sum::<f64>()
        / 10.0;

    let improvement = (last_rewards - first_rewards) / first_rewards.abs().max(0.01) * 100.0;

    println!("Key Findings:");
    println!();

    if improvement > 10.0 {
        println!("LEARNING DETECTED: Reward improved by {:.1}%", improvement);
        println!();
        println!("The compounding cohesion system shows signs of emergent learning:");
        println!("- Rewards increased over episodes");
        println!(
            "- Graph memory accumulated {} nodes",
            learning_metrics.episodes.last().unwrap().graph_nodes
        );
    } else if improvement > 0.0 {
        println!(
            "MARGINAL IMPROVEMENT: Reward improved by {:.1}%",
            improvement
        );
        println!();
        println!("The system shows some adaptation but not strong learning.");
        println!("Consider:");
        println!("- Increasing episode count");
        println!("- Adjusting meta-learning rate");
        println!("- Tuning coherence thresholds");
    } else {
        println!(
            "NO LEARNING DETECTED: Reward changed by {:.1}%",
            improvement
        );
        println!();
        println!("The compounding system may need adjustments:");
        println!("- Check if coherence signals are driving behavior");
        println!("- Verify graph memory is being utilized for decisions");
        println!("- Consider adding explicit gradient-based learning");
    }

    // AGI Core compounding assessment
    println!();
    let final_agi = agi_core.summary();
    println!("AGI Core Compounding Assessment:");
    println!(
        "├─ Compound Rate: {:.4} interactions/step",
        final_agi.analytics.compound_rate
    );
    println!(
        "├─ Total Interactions: {}",
        final_agi.analytics.total_interactions
    );
    println!(
        "├─ Discovery→Abstraction: {}",
        final_agi.analytics.discovery_to_abstraction
    );
    println!(
        "├─ Abstraction→Symbols: {}",
        final_agi.analytics.abstraction_to_symbols
    );
    println!("├─ Symbols→Goals: {}", final_agi.analytics.symbols_to_goals);
    println!(
        "├─ Goals→WorldModel: {}",
        final_agi.analytics.goals_to_world_model
    );
    println!(
        "├─ WorldModel→Meta: {}",
        final_agi.analytics.world_model_to_meta
    );
    println!(
        "└─ Meta→Discovery: {}",
        final_agi.analytics.meta_to_discovery
    );

    if final_agi.analytics.compound_rate > 0.5 {
        println!();
        println!("COMPOUNDING DETECTED: The AGI capabilities are multiplicatively interacting!");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
