//! Training Quickstart Example
//!
//! This example shows how to set up and run training for a Torus Attention model,
//! including configuration, data loading, and the training loop.
//!
//! Run with: cargo run --example training_quickstart

use candle_core::{Device, IndexOp};
use torus_attention::{
    training::generate_random_batch, BidirectionalTorusConfig, LRScheduler, TorusResult, Trainer,
    TrainingConfig,
};

fn main() -> TorusResult<()> {
    println!("═══ Torus Attention Training Quickstart ═══\n");

    // ─────────────────────────────────────────────────────────────────────
    // 1. Model Configuration
    // ─────────────────────────────────────────────────────────────────────
    println!("── Model Configuration ──\n");

    let model_config = BidirectionalTorusConfig {
        d_model: 64, // Small for quick demo
        d_ff: 256,
        n_heads: 4,
        n_layers: 2, // Fewer layers for speed
        n_major: 8,
        n_minor: 4, // 8x4 = 32 sequence length
        use_parallel_streams: true,
        use_compounding: true,
        learnable_alpha: true,
        ..BidirectionalTorusConfig::default()
    };

    println!("Model:");
    println!("  d_model:  {}", model_config.d_model);
    println!("  d_ff:     {}", model_config.d_ff);
    println!("  n_heads:  {}", model_config.n_heads);
    println!("  n_layers: {}", model_config.n_layers);
    println!("  seq_len:  {}", model_config.seq_len());
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 2. Training Configuration
    // ─────────────────────────────────────────────────────────────────────
    println!("── Training Configuration ──\n");

    let training_config = TrainingConfig {
        learning_rate: 3e-4,
        weight_decay: 0.01,
        batch_size: 8,
        epochs: 1,
        warmup_steps: 50,
        total_steps: 200,
        log_every: 10,
        eval_every: 50,
        checkpoint_every: 100,
        ..TrainingConfig::default()
    };

    println!("Training:");
    println!("  learning_rate:  {:.0e}", training_config.learning_rate);
    println!("  weight_decay:   {}", training_config.weight_decay);
    println!("  batch_size:     {}", training_config.batch_size);
    println!("  warmup_steps:   {}", training_config.warmup_steps);
    println!("  total_steps:    {}", training_config.total_steps);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Learning Rate Schedule Visualization
    // ─────────────────────────────────────────────────────────────────────
    println!("── Learning Rate Schedule ──\n");

    let mut scheduler = LRScheduler::new(&training_config);

    println!("Step →  LR");
    for checkpoint in [0, 25, 50, 100, 150, 200] {
        while scheduler.current_step() < checkpoint {
            scheduler.step();
        }
        println!("{:>4} → {:.2e}", checkpoint, scheduler.get_lr());
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Create Trainer
    // ─────────────────────────────────────────────────────────────────────
    println!("── Initializing Trainer ──\n");

    let device = Device::Cpu;
    let vocab_size = 100; // Small vocabulary for demo

    let mut trainer = Trainer::new(
        model_config.clone(),
        training_config.clone(),
        Some(vocab_size),
        &device,
    )?;

    println!("Trainer initialized successfully!");
    println!("  Device: {:?}", device);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Generate Sample Data
    // ─────────────────────────────────────────────────────────────────────
    println!("── Generating Sample Data ──\n");

    let n_samples = 64;
    let seq_len = model_config.seq_len();

    let (all_inputs, all_targets) = generate_random_batch(
        n_samples,
        seq_len,
        model_config.d_model,
        vocab_size,
        &device,
    )?;

    println!("Generated {} samples", n_samples);
    println!("  Input shape:  {:?}", all_inputs.dims());
    println!("  Target shape: {:?}", all_targets.dims());
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Training Loop
    // ─────────────────────────────────────────────────────────────────────
    println!("── Training ──\n");

    let n_batches = n_samples / training_config.batch_size;

    for epoch in 0..training_config.epochs {
        println!("Epoch {}/{}:", epoch + 1, training_config.epochs);

        let mut epoch_loss = 0.0;

        for batch_idx in 0..n_batches {
            let start = batch_idx * training_config.batch_size;
            let end = start + training_config.batch_size;

            // Get batch
            let batch_inputs = all_inputs.i(start..end)?;
            let batch_targets = all_targets.i(start..end)?;

            // Training step
            let loss = trainer.train_step(&batch_inputs, &batch_targets)?;
            epoch_loss += loss;

            let step = trainer.global_step();

            // Log progress
            if step % training_config.log_every == 0 {
                println!("  Step {:>4}: loss = {:.4}", step, loss);
            }

            // Evaluation
            if step % training_config.eval_every == 0 && step > 0 {
                let eval_loss = trainer.evaluate(&batch_inputs, &batch_targets)?;
                println!("  [Eval]    loss = {:.4}", eval_loss);
            }
        }

        let avg_loss = epoch_loss / n_batches as f64;
        println!("  Epoch avg loss: {:.4}\n", avg_loss);
    }

    // ─────────────────────────────────────────────────────────────────────
    // 7. Training Summary
    // ─────────────────────────────────────────────────────────────────────
    println!("── Training Summary ──\n");

    let metrics = trainer.metrics();
    println!("{}", metrics.summary());

    // Show loss progression
    if !metrics.loss_history.is_empty() {
        println!("\nLoss progression:");
        let n = metrics.loss_history.len();
        let step = n.max(5) / 5;
        for i in (0..n).step_by(step) {
            println!("  Step {:>3}: {:.4}", i, metrics.loss_history[i]);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 8. Model Inspection
    // ─────────────────────────────────────────────────────────────────────
    println!("\n── Model State ──\n");

    // Get stream weights from final state
    let model = trainer.model();
    let stream_weights = model.get_all_stream_weights()?;

    println!("Final stream weights (layer 0):");
    for (stream_id, weight) in &stream_weights[0] {
        let bar = "█".repeat((weight * 20.0) as usize);
        println!("  {:15} {:.4} {}", stream_id.name(), weight, bar);
    }

    // Compounding alphas
    if let Some(alphas) = model.get_compounding_alphas()? {
        println!("\nFinal compounding alphas:");
        for (i, alpha) in alphas.iter().enumerate() {
            let bar = "█".repeat((alpha * 20.0) as usize);
            println!("  Layer {}: {:.4} {}", i, alpha, bar);
        }
    }

    println!("\n═══ Training Complete ═══");

    Ok(())
}
