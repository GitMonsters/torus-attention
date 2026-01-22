//! Quick test to verify AGC integration improvements

use torus_attention::agi_core::{AGICore, AGICoreConfig, GoalPriority};

fn main() {
    println!("=== Quick AGC Integration Test ===\n");
    
    // Create AGI Core
    let config = AGICoreConfig::default();
    let feature_dim = 8;
    let n_actions = 4;
    let mut agi = AGICore::new(config, feature_dim, n_actions);
    
    // Run 500 simulated steps
    println!("Running 500 learning steps...\n");
    for step in 0..500 {
        // Simulate state (varies with step)
        let state: Vec<f64> = (0..feature_dim)
            .map(|i| (step as f64 * 0.01 + i as f64 * 0.1).sin())
            .collect();
        
        // Get action with audit
        let (action, audit) = agi.recommend_action_with_audit(&state);
        let action = action.unwrap_or(0);
        
        // Simulate next state and reward
        let next_state: Vec<f64> = state.iter().map(|x| x + 0.1).collect();
        let reward = if step % 20 == 0 { 1.0 } else { 0.1 };
        
        // Process experience
        agi.process_experience(&state, action, &next_state, reward, false);
        
        // Print progress every 100 steps
        if step % 100 == 0 || step == 499 {
            println!("Step {}:", step);
            
            // AGI Core summary
            let summary = agi.summary();
            println!("  Compound Rate: {:.3}", summary.analytics.compound_rate);
            println!("  Q-Learning: {} states, avg Q={:.3}", summary.q_learning.0, summary.q_learning.1);
            
            // AGC summary
            let agc = agi.agc.summary();
            println!("  AGC Coherence: {:.3}", agc.coherence_score);
            println!("  AGC Variety Ratio: {:.3} (sufficient: {})", agc.variety_ratio, agc.has_requisite_variety);
            println!("  AGC Equilibrium: {}", agc.equilibrium_status);
            
            // Principles
            let p = &agc.principles;
            println!("  Principles:");
            println!("    Embodiment: {:.3}", p.embodiment);
            println!("    Symbol Grounding: {:.3}", p.symbol_grounding);
            println!("    Causality: {:.3}", p.causality);
            println!("    Memory: {:.3}", p.memory);
            println!("    Metacognition: {:.3}", p.metacognition);
            println!("    Autonomy: {:.3}", p.autonomy);
            println!("    Explicability: {:.3}", p.explicability);
            println!("    Integration: {:.3}", p.integration);
            
            // Last audit
            if let Some(last_audit) = agi.last_decision_audit() {
                println!("  Last Decision: {}", last_audit.explain());
            }
            
            println!();
        }
    }
    
    // Final summary
    println!("=== Final State ===\n");
    agi.print_summary();
    agi.agc.print_summary();
    
    // Print decision audit stats
    let audits = agi.get_decision_audits();
    let exploration_count = audits.iter().filter(|a| a.was_exploration).count();
    println!("\nDecision Audit Summary:");
    println!("  Total audits: {}", audits.len());
    println!("  Exploration decisions: {} ({:.1}%)", 
        exploration_count, 
        100.0 * exploration_count as f64 / audits.len() as f64);
    
    // Average contributions
    let avg_q = audits.iter().map(|a| a.q_value_contribution).sum::<f64>() / audits.len() as f64;
    let avg_goal = audits.iter().map(|a| a.goal_contribution).sum::<f64>() / audits.len() as f64;
    let avg_wm = audits.iter().map(|a| a.world_model_contribution).sum::<f64>() / audits.len() as f64;
    println!("  Avg Q-value contribution: {:.3}", avg_q);
    println!("  Avg Goal contribution: {:.3}", avg_goal);
    println!("  Avg World Model contribution: {:.3}", avg_wm);
    
    println!("\n✓ AGC Integration Test Complete!");
}
