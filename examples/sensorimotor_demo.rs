//! # Full Compounding Sensorimotor Loop Demo
//!
//! This example demonstrates the complete sensorimotor compounding loop:
//!
//! ```text
//! Observation → Transformer → GoalState → MotorSystem → Action → Environment → New Observation
//!     ↑                                                                              │
//!     └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example sensorimotor_demo
//! ```

use candle_core::Device;
use candle_nn::VarMap;
use torus_attention::{
    compounding_cohesion::GoalType,
    compounding_transformer::{CompoundingCohesionTransformer, CompoundingTransformerConfig},
    sensorimotor::{
        Action, ActionResult, ActionType, CoherenceGuidedPolicy, Environment,
        HypothesisTestingPolicy, MotorPolicy, MotorSystem, Observation, Pose3D, ReactivePolicy,
        SensorimotorAgent, SensorimotorConfig, SimpleGridEnvironment,
    },
    TorusResult,
};

fn main() -> TorusResult<()> {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("       FULL COMPOUNDING SENSORIMOTOR LOOP DEMONSTRATION");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    let device = Device::Cpu;

    // Part 1: Demonstrate the SimpleGridEnvironment
    demo_grid_environment(&device)?;

    // Part 2: Demonstrate Motor Policies
    demo_motor_policies(&device)?;

    // Part 3: Demonstrate the MotorSystem with policy selection
    demo_motor_system(&device)?;

    // Part 4: Create full CompoundingCohesionTransformer and run episodes
    demo_full_agent(&device)?;

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE!");
    println!("═══════════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Demonstrate the SimpleGridEnvironment
fn demo_grid_environment(device: &Device) -> TorusResult<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PART 1: SimpleGridEnvironment                                       │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create a 5x5 grid environment
    let grid_size = 5;
    let d_model = 64;
    let seq_len = 10;
    let mut env = SimpleGridEnvironment::new(grid_size, d_model, seq_len, device);

    // Reset and get initial observation
    let obs = env.reset()?;
    println!("Grid size: {}x{}", grid_size, grid_size);
    println!("Feature dimension: {}", d_model);
    println!("Initial observation shape: {:?}", obs.features.shape());

    if let Some(pose) = &obs.pose {
        println!(
            "Initial pose: ({:.1}, {:.1}, {:.1})",
            pose.x, pose.y, pose.z
        );
    }

    // Take some actions using the Action helper methods
    let actions = vec![
        Action::move_to(Pose3D::new(1.0, 0.0, 0.0), 0.9),
        Action::look_at(Pose3D::new(2.0, 1.0, 0.0), 0.8),
        Action::move_to(Pose3D::new(2.0, 1.0, 0.0), 0.85),
        Action::sample(vec![0.5, 0.5, 0.0], 0.7),
        Action::noop(),
    ];

    println!();
    println!("Taking actions through environment:");

    for (i, action) in actions.iter().enumerate() {
        let result = env.step(action)?;

        let pose_str = result
            .observation
            .as_ref()
            .and_then(|o| o.pose.as_ref())
            .map(|p| format!("({:.1}, {:.1})", p.x, p.y))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "  Step {}: {:?} → Pose: {}, Reward: {:.3}, Done: {}",
            i + 1,
            action.action_type,
            pose_str,
            result.reward,
            result.done
        );
    }

    println!();
    Ok(())
}

/// Demonstrate different motor policies
fn demo_motor_policies(device: &Device) -> TorusResult<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PART 2: Motor Policies                                              │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let d_model = 64;
    let seq_len = 10;

    // Create a dummy observation
    let features = candle_core::Tensor::randn(0f32, 1f32, (1, seq_len, d_model), device)?;
    let obs = Observation::new(features, 0).with_pose(Pose3D::new(2.0, 2.0, 0.0));

    // Test each policy type
    println!("Testing policies with different goal types:");
    println!();

    // 1. ReactivePolicy
    let reactive = ReactivePolicy::default();
    test_policy(&reactive, "ReactivePolicy", &obs)?;

    // 2. CoherenceGuidedPolicy
    let coherence_guided = CoherenceGuidedPolicy::new(0.7);
    test_policy(&coherence_guided, "CoherenceGuidedPolicy", &obs)?;

    // 3. HypothesisTestingPolicy
    let hypothesis = HypothesisTestingPolicy::new();
    test_policy(&hypothesis, "HypothesisTestingPolicy", &obs)?;

    println!();
    Ok(())
}

fn test_policy(policy: &dyn MotorPolicy, name: &str, obs: &Observation) -> TorusResult<()> {
    println!("  {}:", name);

    let goal_types = vec![
        GoalType::Explore,
        GoalType::Refine,
        GoalType::Disambiguate,
        GoalType::None,
    ];

    // Need mutable policy for generate_action
    let mut policy_clone: Box<dyn MotorPolicy> = match name {
        "ReactivePolicy" => Box::new(ReactivePolicy::default()),
        "CoherenceGuidedPolicy" => Box::new(CoherenceGuidedPolicy::new(0.7)),
        "HypothesisTestingPolicy" => Box::new(HypothesisTestingPolicy::new()),
        _ => Box::new(ReactivePolicy::default()),
    };

    for goal_type in goal_types {
        let goal = match goal_type {
            GoalType::None => torus_attention::compounding_cohesion::GoalState::none(),
            GoalType::Explore => torus_attention::compounding_cohesion::GoalState::explore(0, 0.5),
            GoalType::Refine => torus_attention::compounding_cohesion::GoalState::refine(0),
            GoalType::Disambiguate => torus_attention::compounding_cohesion::GoalState {
                goal_type: GoalType::Disambiguate,
                confidence: 0.6,
                target_layer: None,
                target_stream: Some(0),
                uncertainty_source: 0.4,
            },
        };

        let action = policy_clone.generate_action(&goal, obs);
        println!(
            "    {:?} → {:?} (conf: {:.2})",
            goal_type, action.action_type, action.confidence
        );
    }

    println!();
    Ok(())
}

/// Demonstrate the MotorSystem with automatic policy selection
fn demo_motor_system(device: &Device) -> TorusResult<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PART 3: MotorSystem with Policy Selection                           │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let d_model = 64;
    let seq_len = 10;

    // Create motor system (automatically creates default policies)
    let mut motor = MotorSystem::new();

    println!("MotorSystem created with default policies:");
    println!("  - ReactivePolicy (index 0)");
    println!("  - CoherenceGuidedPolicy (index 1)");
    println!("  - HypothesisTestingPolicy (index 2)");
    println!();

    // Create observation
    let features = candle_core::Tensor::randn(0f32, 1f32, (1, seq_len, d_model), device)?;
    let obs = Observation::new(features, 0).with_pose(Pose3D::new(2.0, 2.0, 0.0));

    // Test automatic policy selection for different goals
    println!("Automatic policy selection based on goal type:");
    println!();

    let test_cases = vec![
        (GoalType::Explore, "CoherenceGuidedPolicy"),
        (GoalType::Refine, "CoherenceGuidedPolicy"),
        (GoalType::Disambiguate, "HypothesisTestingPolicy"),
        (GoalType::None, "ReactivePolicy"),
    ];

    for (goal_type, expected_policy) in test_cases {
        let goal = match goal_type {
            GoalType::None => torus_attention::compounding_cohesion::GoalState::none(),
            GoalType::Explore => torus_attention::compounding_cohesion::GoalState::explore(0, 0.5),
            GoalType::Refine => torus_attention::compounding_cohesion::GoalState::refine(0),
            GoalType::Disambiguate => torus_attention::compounding_cohesion::GoalState {
                goal_type: GoalType::Disambiguate,
                confidence: 0.6,
                target_layer: None,
                target_stream: Some(0),
                uncertainty_source: 0.4,
            },
        };

        let action = motor.generate_action(&goal, &obs);

        println!(
            "  {:?} → Expected: {}, Action: {:?} (confidence: {:.2})",
            goal_type, expected_policy, action.action_type, action.confidence
        );
    }

    // Show motor stats
    let stats = motor.stats();
    println!();
    println!(
        "Motor stats: total_actions={}, high_conf_ratio={:.2}, active={}",
        stats.total_actions, stats.high_confidence_ratio, stats.active_policy
    );

    println!();
    Ok(())
}

/// Demonstrate the full SensorimotorAgent with CompoundingCohesionTransformer
fn demo_full_agent(device: &Device) -> TorusResult<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PART 4: Full SensorimotorAgent with CompoundingCohesionTransformer  │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // Configuration for a small model
    let d_model = 64;
    let n_layers = 4;
    let n_heads = 4;
    let grid_size = 5;
    let n_major = 4;
    let n_minor = 4;
    let seq_len = n_major * n_minor; // Must match transformer sequence length

    println!("Creating CompoundingCohesionTransformer:");
    println!("  d_model: {}", d_model);
    println!("  n_layers: {}", n_layers);
    println!("  n_heads: {}", n_heads);
    println!("  seq_len: {} ({}x{})", seq_len, n_major, n_minor);
    println!();

    // Create transformer config with defaults and override key params
    let mut config = CompoundingTransformerConfig::default();
    config.d_model = d_model;
    config.n_layers = n_layers;
    config.n_heads = n_heads;
    config.d_ff = d_model * 4;
    config.n_major = n_major;
    config.n_minor = n_minor;
    config.sync_cohesion_config();

    // Create VarBuilder for weights
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    // Create transformer
    let transformer = CompoundingCohesionTransformer::new(config, None, vb, device)?;

    println!("Transformer created successfully!");
    println!("  Hierarchical levels: Local → Regional → Global");
    println!("  8-Stream parallel processing enabled");
    println!();

    // Create environment
    let env = SimpleGridEnvironment::new(grid_size, d_model, seq_len, device);

    // Create sensorimotor config
    let sm_config = SensorimotorConfig {
        max_steps_per_episode: 20,
        online_learning: true,
        gamma: 0.99,
        consolidation_interval: 10,
        verbose: false,
    };

    // Create agent
    let mut agent = SensorimotorAgent::new(transformer, Box::new(env), sm_config);

    println!("Running sensorimotor episode...");
    println!();

    // Run episode
    let result = agent.run_episode()?;

    println!("Episode completed!");
    println!("  Total steps: {}", result.steps);
    println!("  Total reward: {:.3}", result.total_reward);
    println!("  Final coherence: {:.3}", result.final_coherence);
    println!("  Success: {}", result.success);
    println!("  Graph nodes: {}", result.graph_nodes);

    // Run a few more episodes to see learning
    println!();
    println!("Running additional episodes to demonstrate learning...");
    println!();

    let mut total_rewards = vec![result.total_reward];
    let mut final_coherences = vec![result.final_coherence];

    for i in 0..4 {
        // Meta-update between episodes
        agent.transformer.meta_update(0.0); // No external reward signal

        // Run next episode
        let result = agent.run_episode()?;
        total_rewards.push(result.total_reward);
        final_coherences.push(result.final_coherence);

        println!(
            "  Episode {}: steps={}, reward={:.3}, coherence={:.3}",
            i + 2,
            result.steps,
            result.total_reward,
            result.final_coherence
        );
    }

    // Show learning progress
    println!();
    println!("Learning progress:");
    println!(
        "  Avg reward (first 2): {:.3}",
        total_rewards.iter().take(2).sum::<f64>() / 2.0
    );
    println!(
        "  Avg reward (last 2): {:.3}",
        total_rewards.iter().skip(3).sum::<f64>() / 2.0
    );
    println!(
        "  Avg coherence (first 2): {:.3}",
        final_coherences.iter().take(2).sum::<f64>() / 2.0
    );
    println!(
        "  Avg coherence (last 2): {:.3}",
        final_coherences.iter().skip(3).sum::<f64>() / 2.0
    );

    // Check if compounding is effective
    println!();
    println!(
        "Compounding effective: {}",
        agent.is_compounding_effective()
    );

    // Get agent summary
    println!();
    println!("Agent summary:");
    println!("{}", agent.summary());

    Ok(())
}
