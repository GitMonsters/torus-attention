//! Basic Usage Example for Torus Attention
//!
//! This example demonstrates the fundamental usage of the Torus Attention library,
//! including configuration, model creation, and inference.
//!
//! Run with: cargo run --example basic_usage

use torus_attention::{
    BidirectionalTorusConfig,
    BidirectionalTorusTransformer,
    TorusResult,
};
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn main() -> TorusResult<()> {
    println!("═══ Torus Attention Basic Usage Example ═══\n");
    
    // 1. Configure the model
    let config = BidirectionalTorusConfig {
        d_model: 128,       // Model dimension
        d_ff: 512,          // Feed-forward dimension  
        n_heads: 4,         // Number of attention heads
        n_layers: 3,        // Number of transformer layers
        n_major: 8,         // Grid size (major radius)
        n_minor: 8,         // Grid size (minor radius)
        use_parallel_streams: true,  // Enable 8-stream processing
        use_compounding: true,       // Enable EMA compounding
        learnable_alpha: true,       // Make EMA alpha learnable
        ..BidirectionalTorusConfig::default()
    };
    
    println!("Model Configuration:");
    println!("  d_model:    {}", config.d_model);
    println!("  n_heads:    {}", config.n_heads);
    println!("  n_layers:   {}", config.n_layers);
    println!("  seq_len:    {} ({}x{})", config.seq_len(), config.n_major, config.n_minor);
    println!("  spiral_winding: {:.6} (golden ratio)", config.spiral_winding);
    println!();
    
    // 2. Create the device (CPU or CUDA)
    let device = Device::Cpu;
    println!("Using device: {:?}\n", device);
    
    // 3. Initialize model parameters
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    
    // 4. Create the model
    let vocab_size = None; // No vocab for embedding-based input
    let mut model = BidirectionalTorusTransformer::new(
        config.clone(),
        vocab_size,
        vb,
        &device,
    )?;
    
    println!("Model created successfully!");
    
    // 5. Create sample input
    let batch_size = 2;
    let seq_len = config.seq_len();
    let input = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, config.d_model), &device)?;
    
    println!("Input shape:  {:?}", input.dims());
    
    // 6. Run forward pass
    let output = model.forward(&input)?;
    
    println!("Output shape: {:?}", output.dims());
    
    // 7. Get model statistics
    println!("\n── Model Statistics ──");
    
    // Stream weights
    let stream_weights = model.get_all_stream_weights()?;
    println!("\nStream weights (layer 0):");
    for (stream_id, weight) in &stream_weights[0] {
        println!("  {:15} {:.4}", stream_id.name(), weight);
    }
    
    // Compounding alphas
    if let Some(alphas) = model.get_compounding_alphas()? {
        println!("\nCompounding alphas:");
        for (i, alpha) in alphas.iter().enumerate() {
            println!("  Layer {}: α = {:.4}", i, alpha);
        }
    }
    
    // 8. Reset state for new sequence
    model.reset_state()?;
    println!("\nModel state reset for new sequence.");
    
    println!("\n═══ Example Complete ═══");
    
    Ok(())
}
