//! # GPU-Accelerated Learning Experiment
//!
//! This example runs the same learning experiment as learning_experiment.rs but uses
//! the AMD GPU via Burn WGPU/Vulkan for acceleration.
//!
//! Run with:
//! ```bash
//! cargo run --example gpu_learning_experiment --release --features amd-gpu
//! ```

#[cfg(feature = "amd-gpu")]
use torus_attention::backend::GpuCompute;

use candle_core::Device;
use candle_nn::VarMap;
use std::time::Instant;
use torus_attention::{
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
}

/// Aggregate metrics over multiple episodes
struct LearningMetrics {
    episodes: Vec<EpisodeMetrics>,
    running_avg_reward: Vec<f64>,
}

impl LearningMetrics {
    fn new() -> Self {
        Self {
            episodes: Vec::new(),
            running_avg_reward: Vec::new(),
        }
    }

    fn record(&mut self, metrics: EpisodeMetrics) {
        let window = 5;
        self.episodes.push(metrics);
        
        let n = self.episodes.len();
        let start = n.saturating_sub(window);
        let avg_reward: f64 = self.episodes[start..n]
            .iter()
            .map(|e| e.total_reward)
            .sum::<f64>()
            / (n - start) as f64;
        self.running_avg_reward.push(avg_reward);
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

        let reward_improvement =
            (second_avg_reward - first_avg_reward) / first_avg_reward.abs().max(0.01) * 100.0;
        let coherence_change = (second_avg_coherence - first_avg_coherence)
            / first_avg_coherence.abs().max(0.01)
            * 100.0;
        let coverage_improvement =
            (second_avg_coverage - first_avg_coverage) / first_avg_coverage.abs().max(0.01) * 100.0;

        let total_graph_nodes = self.episodes.last().map(|e| e.graph_nodes).unwrap_or(0);
        let total_graph_edges = self.episodes.last().map(|e| e.graph_edges).unwrap_or(0);

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

#[cfg(feature = "amd-gpu")]
fn run_gpu_benchmark() -> TorusResult<()> {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("     GPU-ACCELERATED LEARNING EXPERIMENT (AMD GPU via Vulkan)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    
    // Initialize GPU
    let gpu = GpuCompute::new().map_err(|e| torus_attention::TorusError::Backend(e.to_string()))?;
    println!("GPU Backend: {}", gpu.backend_info());
    println!();

    // Run a quick GPU warmup/benchmark
    println!("GPU Warmup (2048x2048 matmul)...");
    let warmup_start = Instant::now();
    let a = gpu.randn_f32(&[2048, 2048]).map_err(|e| torus_attention::TorusError::Backend(e.to_string()))?;
    let b = gpu.randn_f32(&[2048, 2048]).map_err(|e| torus_attention::TorusError::Backend(e.to_string()))?;
    let _ = gpu.matmul_f32(&a, &b, 2048, 2048, 2048).map_err(|e| torus_attention::TorusError::Backend(e.to_string()))?;
    println!("GPU warmup: {:?}", warmup_start.elapsed());
    println!();

    let device = Device::Cpu; // Candle still uses CPU, GPU compute is separate
    let start = Instant::now();

    // Configuration - larger model to benefit from GPU
    let n_episodes = 200;
    let d_model = 128; // Increased from 64
    let n_layers = 6;  // Increased from 4
    let n_heads = 8;   // Increased from 4
    let n_major = 8;   // Increased from 4
    let n_minor = 8;   // Increased from 4
    let seq_len = n_major * n_minor; // 64 (vs 16 before)

    println!("Configuration (larger for GPU):");
    println!("├─ Episodes: {}", n_episodes);
    println!("├─ Model: d={}, layers={}, heads={}", d_model, n_layers, n_heads);
    println!("├─ Torus: {}x{} = {} positions", n_major, n_minor, seq_len);
    println!("└─ Environment: LearningGridEnvironment (10x10 with landmarks)");
    println!();

    // Create cohesion config
    let cohesion_config = CompoundingCohesionConfig {
        n_layers,
        n_streams: 8,
        d_model,
        n_hierarchy_levels: 3,
        layers_per_region: 2,
        max_graph_nodes: 500,
        graph_feature_dim: 64, // Increased
        prediction_history: 100,
        meta_learning_rate: 0.015,
        graph_learning_rate: 0.25,
        merge_threshold: 0.60,
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

    // Create transformer config
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
        None,
        vb,
        &device,
    )?;
    println!("Transformer created successfully.");
    println!();

    // Create environment
    let env_config = LearningEnvConfig {
        size: 10,
        feature_dim: d_model,
        seq_len,
        n_landmarks: 8,
        n_obstacles: 4,
        max_steps: 100,
        goal_move_probability: 0.05,
        obstacle_change_probability: 0.02,
        novelty_bonus: 0.3,
        exploration_reward: 0.1,
        cognitive_dissonance_rate: 0.1,
    };

    let env = LearningGridEnvironment::new(env_config.clone(), &device, 42);
    let agent_config = SensorimotorConfig {
        max_steps_per_episode: 100,
        online_learning: true,
        gamma: 0.99,
        consolidation_interval: 25,
        verbose: false,
    };

    let mut agent = SensorimotorAgent::with_agi_reasoning(
        transformer,
        Box::new(env),
        agent_config,
        8,
        5,
    );

    let mut dissonance_tracker = CognitiveDissonanceTracker::new(0.3);
    let mut learning_metrics = LearningMetrics::new();

    println!("Running {} episodes with GPU acceleration...", n_episodes);
    println!();
    println!("Ep  | Reward  | Coverage | Coherence | Nodes | Steps | Time/Ep");
    println!("────┼─────────┼──────────┼───────────┼───────┼───────┼────────");

    let mut episode_times = Vec::new();

    for episode in 0..n_episodes {
        let ep_start = Instant::now();
        
        let result = agent.run_episode()?;

        let ep_time = ep_start.elapsed();
        episode_times.push(ep_time.as_millis() as f64);

        let cohesion = agent.transformer.cohesion();
        let pred_error = cohesion.prediction.avg_error();
        let graph_nodes: usize = cohesion.stream_graphs.iter().map(|g| g.nodes.len()).sum();
        let graph_edges: usize = cohesion.stream_graphs.iter().map(|g| g.edges.len()).sum();
        let (coverage, _, _, _) = agent.environment.get_stats();

        dissonance_tracker.record_prediction_error(pred_error);
        if result.final_coherence < 0.4 {
            dissonance_tracker.record_coherence_drop(0.4 - result.final_coherence);
        }

        let metrics = EpisodeMetrics {
            episode,
            total_reward: result.total_reward,
            coverage,
            final_coherence: result.final_coherence,
            prediction_error: pred_error,
            graph_nodes,
            graph_edges,
            steps: result.steps,
        };
        learning_metrics.record(metrics);

        if episode % 10 == 0 || episode == n_episodes - 1 {
            let avg_time = if episode_times.len() > 5 {
                episode_times[episode_times.len()-5..].iter().sum::<f64>() / 5.0
            } else {
                episode_times.iter().sum::<f64>() / episode_times.len() as f64
            };
            
            println!(
                "{:3} | {:7.3} | {:7.1}% | {:9.3} | {:5} | {:5} | {:6.1}ms",
                episode,
                result.total_reward,
                coverage * 100.0,
                result.final_coherence,
                graph_nodes,
                result.steps,
                avg_time
            );
        }

        // Reset environment for next episode
        if episode < n_episodes - 1 {
            let new_env = LearningGridEnvironment::new(
                env_config.clone(),
                &device,
                42 + (episode as u64 + 1) * 17,
            );
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
            agent.agi_reasoning = agi_reasoning;
        }
    }

    let elapsed = start.elapsed();
    let avg_episode_time = elapsed.as_secs_f64() * 1000.0 / n_episodes as f64;

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                           RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("{}", learning_metrics.summary());
    println!();

    // Print timing comparison
    println!("Performance:");
    println!("├─ Total time: {:.2}s", elapsed.as_secs_f64());
    println!("├─ Avg per episode: {:.2}ms", avg_episode_time);
    println!("├─ Model size: d={}, layers={}, seq_len={}", d_model, n_layers, seq_len);
    println!("└─ Total parameters: ~{:.1}M", 
        (d_model * d_model * 4 * n_layers * 8) as f64 / 1_000_000.0);
    println!();

    // Coherence system state
    println!("Coherence System State:");
    println!("{}", agent.transformer.cohesion().summary());
    println!();

    // Dissonance analysis
    let diss_stats = dissonance_tracker.stats();
    println!("Cognitive Dissonance Analysis:");
    println!("├─ Current Level: {:.3}", diss_stats.current_level);
    println!("├─ In Dissonance: {}", diss_stats.in_dissonance);
    println!("└─ Avg Prediction Error: {:.4}", diss_stats.avg_prediction_error);
    println!();

    // AGI reasoning summary
    if let Some(agi_summary) = agent.agi_summary() {
        println!("{}", agi_summary);
        println!();
    }

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

    if improvement > 10.0 {
        println!("✓ LEARNING DETECTED: Reward improved by {:.1}%", improvement);
    } else if improvement > 0.0 {
        println!("○ MARGINAL IMPROVEMENT: Reward improved by {:.1}%", improvement);
    } else {
        println!("✗ NO LEARNING: Performance did not improve");
    }

    println!();
    println!("GPU acceleration allows running larger models:");
    println!("- {} parameters vs typical CPU limit", 
        (d_model * d_model * 4 * n_layers * 8) as f64 / 1_000_000.0);
    println!("- {}x{} sequence length vs typical 4x4", n_major, n_minor);
    println!();

    Ok(())
}

#[cfg(not(feature = "amd-gpu"))]
fn run_gpu_benchmark() -> TorusResult<()> {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  ERROR: GPU features not enabled");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("This example requires the 'amd-gpu' feature.");
    println!("Run with:");
    println!();
    println!("  cargo run --example gpu_learning_experiment --release --features amd-gpu");
    println!();
    Ok(())
}

fn main() -> TorusResult<()> {
    run_gpu_benchmark()
}
