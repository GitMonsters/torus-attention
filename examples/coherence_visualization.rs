//! Coherence Visualization Example
//!
//! Demonstrates the cognitive coherence module in action, showing:
//! - Sense of Coherence (SOC) metrics during inference
//! - Shared Mental Model (SMM) alignment between streams
//! - Adaptive alpha dynamics based on coherence state
//!
//! Run with: cargo run --example coherence_visualization

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use torus_attention::{
    BidirectionalStats,
    BidirectionalTorusConfig,
    BidirectionalTorusTransformer,
    CognitiveCoherenceLayer,
    SenseOfCoherence,
    SharedMentalModel,
    TorusResult,
};

fn main() -> TorusResult<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           Cognitive Coherence Visualization Example               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let device = Device::Cpu;

    // ═══════════════════════════════════════════════════════════════════════════
    // Part 1: Standalone Coherence Components
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 1: Sense of Coherence (SOC) - Antonovsky's Model         │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Create different SOC states
    let healthy_soc = SenseOfCoherence::healthy();
    let stressed_soc = SenseOfCoherence::stressed();
    let custom_soc = SenseOfCoherence::new(0.7, 0.8, 0.95);

    println!("Healthy SOC State:");
    println!("  Comprehensibility: {:.3}", healthy_soc.comprehensibility);
    println!("  Manageability:     {:.3}", healthy_soc.manageability);
    println!("  Meaningfulness:    {:.3}", healthy_soc.meaningfulness);
    println!("  Overall Score:     {:.3}\n", healthy_soc.score());

    println!("Stressed SOC State:");
    println!("  Comprehensibility: {:.3}", stressed_soc.comprehensibility);
    println!("  Manageability:     {:.3}", stressed_soc.manageability);
    println!("  Meaningfulness:    {:.3}", stressed_soc.meaningfulness);
    println!("  Overall Score:     {:.3}\n", stressed_soc.score());

    println!("Custom SOC State (high meaningfulness):");
    println!("  Comprehensibility: {:.3}", custom_soc.comprehensibility);
    println!("  Manageability:     {:.3}", custom_soc.manageability);
    println!("  Meaningfulness:    {:.3}", custom_soc.meaningfulness);
    println!("  Overall Score:     {:.3}\n", custom_soc.score());

    // Demonstrate adaptive alpha
    println!("Adaptive Alpha (base=0.9, min=0.1, max=0.99):");
    println!("  Healthy SOC → α = {:.4}", healthy_soc.adaptive_alpha(0.9, 0.1, 0.99));
    println!("  Stressed SOC → α = {:.4}", stressed_soc.adaptive_alpha(0.9, 0.1, 0.99));
    println!("  Custom SOC → α = {:.4}\n", custom_soc.adaptive_alpha(0.9, 0.1, 0.99));

    // ═══════════════════════════════════════════════════════════════════════════
    // Part 2: Shared Mental Models (SMM)
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 2: Shared Mental Models (SMM) - Stream Alignment         │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let smm = SharedMentalModel::for_torus_attention();
    
    println!("8-Stream Torus Attention SMM:");
    println!("  Number of streams: {}", smm.n_streams);
    println!("  Task alignment:    {:.3}", smm.task_alignment);
    println!("  Team alignment:    {:.3}", smm.team_alignment);
    println!("  Cognitive cohesion: {:.3}\n", smm.cognitive_cohesion());

    // Print alignment matrix (abbreviated)
    println!("Stream Alignment Matrix (key pairs):");
    let stream_names = [
        "Major Fwd", "Major Bwd", "Minor Fwd", "Minor Bwd",
        "Spiral CW", "Spiral CCW", "Cross U→V", "Cross V→U"
    ];
    
    println!("  Forward-Backward pairs (high complementary alignment):");
    for (i, j) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
        println!("    {} ↔ {}: {:.3}", 
            stream_names[i], stream_names[j], 
            smm.alignment_matrix[i][j]);
    }

    println!("\n  Stream combination weights:");
    let weights = smm.alignment_weights();
    for (i, w) in weights.iter().enumerate() {
        println!("    {:12}: {:.4}", stream_names[i], w);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Part 3: Full Coherence Layer
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 3: Cognitive Coherence Layer - Full Integration          │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let mut coherence = CognitiveCoherenceLayer::for_torus_attention(64, &device);
    
    println!("Initial coherence state:");
    println!("{}\n", coherence.summary());

    // Simulate attention patterns and update coherence
    println!("Simulating attention updates...\n");
    
    for i in 0..5 {
        // Create synthetic attention patterns (increasingly focused)
        let focus_factor = 0.3 + (i as f64 * 0.1);
        let mut attn_data = vec![0.1f32; 64];
        // Make attention more concentrated over time
        for j in 0..16 {
            attn_data[j] = focus_factor as f32;
        }
        // Normalize
        let sum: f32 = attn_data.iter().sum();
        for a in &mut attn_data {
            *a /= sum;
        }
        
        let attention = Tensor::from_vec(attn_data, (64,), &device)?;
        let hidden = Tensor::randn(0.0f32, 1.0, (1, 64, 64), &device)?;
        
        coherence.update_soc(&attention, &hidden)?;
        
        println!("Update {}: SOC={:.3}, Cohesion={:.3}, α={:.4}", 
            i + 1,
            coherence.psychological_coherence(),
            coherence.cognitive_cohesion(),
            coherence.compute_adaptive_alpha()
        );
    }

    println!("\nFinal coherence state:");
    println!("{}\n", coherence.summary());

    // ═══════════════════════════════════════════════════════════════════════════
    // Part 4: Full Transformer with Coherence
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 4: Full Transformer with Coherence Integration           │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Create a small transformer for demonstration
    let mut config = BidirectionalTorusConfig::default();
    config.d_model = 64;
    config.d_ff = 128;
    config.n_layers = 2;
    config.n_major = 8;
    config.n_minor = 4;
    config.use_coherence = true;
    
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let mut transformer = BidirectionalTorusTransformer::new(
        config.clone(),
        None,
        vb,
        &device,
    )?;

    println!("Created transformer:");
    println!("  Model dimension: {}", config.d_model);
    println!("  Layers: {}", config.n_layers);
    println!("  Sequence length: {}", config.seq_len());
    println!("  Coherence enabled: {}\n", config.use_coherence);

    // Run inference
    let input = Tensor::randn(0.0f32, 1.0, (1, config.seq_len(), config.d_model), &device)?;
    
    println!("Running inference...");
    let _output = transformer.forward(&input)?;
    
    // Get coherence metrics
    if let Some(coh) = transformer.coherence_score() {
        println!("  Coherence score: {:.4}", coh);
    }
    if let Some(cohesion) = transformer.cohesion_score() {
        println!("  Cohesion score: {:.4}", cohesion);
    }
    if let Some(is_coherent) = transformer.is_coherent() {
        println!("  System coherent: {}", if is_coherent { "YES" } else { "NO" });
    }

    // Get full statistics
    let stats = BidirectionalStats::from_transformer(&transformer)?;
    println!("\nFull statistics:");
    stats.summary();

    // ═══════════════════════════════════════════════════════════════════════════
    // Part 5: Coherence Over Multiple Inferences
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Part 5: Coherence Dynamics Over Multiple Inferences           │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!("Inference | SOC Score | Cohesion | Adaptive α | Status");
    println!("----------|-----------|----------|------------|--------");

    for i in 0..10 {
        // Reset state for independent inferences
        transformer.reset_state()?;
        
        // Vary input noise to create different patterns
        let noise_scale = 0.5 + (i as f64 * 0.1);
        let input = Tensor::randn(0.0f32, noise_scale as f32, (1, config.seq_len(), config.d_model), &device)?;
        
        let _output = transformer.forward(&input)?;
        
        let soc = transformer.coherence_score().unwrap_or(0.0);
        let cohesion = transformer.cohesion_score().unwrap_or(0.0);
        let alpha = transformer.get_coherence()
            .map(|c| c.compute_adaptive_alpha())
            .unwrap_or(0.0);
        let status = if transformer.is_coherent().unwrap_or(false) { "COHERENT" } else { "UNCERTAIN" };
        
        println!("    {:2}    |   {:.4}  |  {:.4}  |   {:.4}   | {}", 
            i + 1, soc, cohesion, alpha, status);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Conclusion
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                           Summary                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("The Cognitive Coherence module provides:");
    println!();
    println!("  1. Sense of Coherence (SOC)");
    println!("     - Comprehensibility: clarity of attention patterns");
    println!("     - Manageability: capacity vs demand balance");
    println!("     - Meaningfulness: signal concentration/importance");
    println!();
    println!("  2. Shared Mental Models (SMM)");
    println!("     - Inter-stream alignment tracking");
    println!("     - Adaptive combination weights");
    println!("     - Cognitive cohesion measurement");
    println!();
    println!("  3. Adaptive EMA Compounding");
    println!("     - α adjusted based on coherence state");
    println!("     - High coherence → trust accumulated state");
    println!("     - Low coherence → rely more on fresh input");
    println!();
    println!("This creates a synergy where strong cognitive cohesion");
    println!("fosters psychological coherence, improving overall");
    println!("transformer performance and stability.");

    Ok(())
}
