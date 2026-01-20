//! # Dynamic Compound Training CLI
//!
//! Command-line interface for the dynamic compound training system.
//!
//! ## Usage
//!
//! ```bash
//! # Basic training with all dynamic features enabled
//! torus_dynamic_train --data corpus.txt --epochs 100 --all-dynamic
//!
//! # Training with specific features
//! torus_dynamic_train --data corpus.txt \
//!     --curriculum \
//!     --dynamic-batch \
//!     --progressive-growth \
//!     --dynamic-ema
//!
//! # Resume from checkpoint
//! torus_dynamic_train --data corpus.txt --resume --checkpoint-dir ./checkpoints
//! ```

use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;

use torus_attention::dataset::TextDataset;
use torus_attention::dynamic_trainer::{
    DynamicCompoundTrainer, DynamicTrainingConfig, GrowthConfig,
};
use torus_attention::llm::TorusLLMConfig;
use torus_attention::tokenizer::BpeTokenizer;

/// Dynamic Compound Training for Torus LLM
#[derive(Parser, Debug)]
#[command(name = "torus_dynamic_train")]
#[command(about = "Train a Torus LLM with dynamic compound training")]
#[command(version)]
struct Args {
    /// Training data file or directory
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Validation data file (optional)
    #[arg(long)]
    val_data: Option<PathBuf>,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Maximum number of training steps
    #[arg(long)]
    max_steps: Option<usize>,

    /// Initial batch size
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "3e-4")]
    learning_rate: f64,

    /// Warmup steps
    #[arg(long, default_value = "100")]
    warmup_steps: usize,

    /// Maximum sequence length
    #[arg(long, default_value = "512")]
    max_seq_len: usize,

    /// Vocabulary size
    #[arg(long, default_value = "50257")]
    vocab_size: usize,

    /// Hidden dimension
    #[arg(long, default_value = "256")]
    hidden_dim: usize,

    /// Number of layers
    #[arg(long, default_value = "6")]
    num_layers: usize,

    /// Number of attention heads
    #[arg(long, default_value = "8")]
    num_heads: usize,

    /// Dropout probability
    #[arg(long, default_value = "0.1")]
    dropout: f64,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Resume from checkpoint
    #[arg(long)]
    resume: bool,

    /// Log interval (steps)
    #[arg(long, default_value = "10")]
    log_interval: usize,

    /// Evaluation interval (steps)
    #[arg(long, default_value = "100")]
    eval_interval: usize,

    /// Checkpoint save interval (steps)
    #[arg(long, default_value = "500")]
    save_interval: usize,

    /// Early stopping patience
    #[arg(long, default_value = "10")]
    patience: usize,

    /// Gradient clipping max norm
    #[arg(long, default_value = "1.0")]
    max_grad_norm: f64,

    // Dynamic training features
    /// Enable curriculum learning
    #[arg(long)]
    curriculum: bool,

    /// Enable multi-task training
    #[arg(long)]
    multi_task: bool,

    /// Enable dynamic batch sizing
    #[arg(long)]
    dynamic_batch: bool,

    /// Enable layer-wise learning rates
    #[arg(long)]
    layer_wise_lr: bool,

    /// Enable progressive model growth
    #[arg(long)]
    progressive_growth: bool,

    /// Enable dynamic EMA compounding
    #[arg(long)]
    dynamic_ema: bool,

    /// Enable all dynamic features
    #[arg(long)]
    all_dynamic: bool,

    // Progressive growth options
    /// Initial layers for progressive growth
    #[arg(long, default_value = "2")]
    growth_initial_layers: usize,

    /// Final layers for progressive growth
    #[arg(long, default_value = "12")]
    growth_final_layers: usize,

    /// Initial hidden dim for progressive growth
    #[arg(long, default_value = "128")]
    growth_initial_hidden: usize,

    /// Final hidden dim for progressive growth
    #[arg(long, default_value = "768")]
    growth_final_hidden: usize,

    /// Growth interval (steps)
    #[arg(long, default_value = "1000")]
    growth_interval: usize,

    /// Use CUDA if available
    #[arg(long)]
    cuda: bool,

    /// Use Metal (Apple Silicon) if available
    #[arg(long)]
    metal: bool,

    /// Enable Tensorboard logging
    #[arg(long)]
    tensorboard: bool,

    /// Tensorboard log directory
    #[arg(long, default_value = "logs")]
    tensorboard_dir: PathBuf,

    /// Validation split ratio (if no separate val data provided)
    #[arg(long, default_value = "0.1")]
    val_split: f64,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Use synthetic data for testing (ignores --data)
    #[arg(long)]
    synthetic: bool,

    /// Number of synthetic examples
    #[arg(long, default_value = "1000")]
    synthetic_examples: usize,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    log::info!("Torus Dynamic Compound Training");
    log::info!("================================");

    // Select device
    let device = if args.cuda {
        Device::cuda_if_available(0)?
    } else if args.metal {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    log::info!("Using device: {:?}", device);

    // Create tokenizer (use fast BPE tokenizer)
    log::info!("Initializing BPE tokenizer...");
    let tokenizer = BpeTokenizer::new(args.vocab_size);

    // Load or generate training data
    let train_dataset = if args.synthetic {
        log::info!(
            "Generating {} synthetic examples...",
            args.synthetic_examples
        );
        TextDataset::synthetic(args.synthetic_examples, args.max_seq_len, args.vocab_size)
    } else if let Some(ref data_path) = args.data {
        log::info!("Loading training data from {:?}...", data_path);
        if data_path.is_dir() {
            TextDataset::from_directory(data_path, &tokenizer, args.max_seq_len)?
        } else {
            TextDataset::from_file(data_path, &tokenizer, args.max_seq_len)?
        }
    } else {
        log::info!("No data provided, using synthetic data...");
        TextDataset::synthetic(args.synthetic_examples, args.max_seq_len, args.vocab_size)
    };
    log::info!("Loaded {} training examples", train_dataset.len());

    // Load or split validation data
    let (train_dataset, val_dataset) = if let Some(val_path) = args.val_data {
        log::info!("Loading validation data from {:?}...", val_path);
        let val = if val_path.is_dir() {
            TextDataset::from_directory(&val_path, &tokenizer, args.max_seq_len)?
        } else {
            TextDataset::from_file(&val_path, &tokenizer, args.max_seq_len)?
        };
        log::info!("Loaded {} validation examples", val.len());
        (train_dataset, Some(val))
    } else if args.val_split > 0.0 && !args.synthetic && args.data.is_some() {
        log::info!(
            "Splitting {}% for validation...",
            (args.val_split * 100.0) as usize
        );
        let (train, val) = train_dataset.train_val_split(args.val_split);
        log::info!("Train: {}, Val: {}", train.len(), val.len());
        (train, Some(val))
    } else {
        (train_dataset, None)
    };

    // Determine which dynamic features to enable
    let use_all = args.all_dynamic;
    let use_curriculum = use_all || args.curriculum;
    let use_multi_task = use_all || args.multi_task;
    let use_dynamic_batch = use_all || args.dynamic_batch;
    let use_layer_wise_lr = use_all || args.layer_wise_lr;
    let use_progressive_growth = use_all || args.progressive_growth;
    let use_dynamic_ema = use_all || args.dynamic_ema;

    // Build model config
    let model_config = TorusLLMConfig {
        vocab_size: args.vocab_size,
        max_seq_len: args.max_seq_len,
        hidden_dim: args.hidden_dim,
        num_layers: args.num_layers,
        num_heads: args.num_heads,
        ffn_dim: args.hidden_dim * 4,
        dropout: args.dropout,
        layer_norm_eps: 1e-5,
        tie_embeddings: true,
        torus_major_radius: 2.0,
        torus_minor_radius: 1.0,
        ema_alpha: 0.9,
        use_coherence: true,
    };

    // Build growth config
    let growth_config = GrowthConfig {
        initial_layers: args.growth_initial_layers,
        final_layers: args.growth_final_layers,
        initial_hidden_dim: args.growth_initial_hidden,
        final_hidden_dim: args.growth_final_hidden,
        initial_heads: 4,
        final_heads: args.num_heads,
        growth_interval: args.growth_interval,
        growth_threshold: 3.0,
        min_steps_before_growth: 500,
    };

    // Build training config
    let training_config = DynamicTrainingConfig {
        model: model_config,
        batch_size: args.batch_size,
        num_epochs: args.epochs,
        max_steps: args.max_steps,
        learning_rate: args.learning_rate,
        weight_decay: 0.01,
        warmup_steps: args.warmup_steps,
        log_interval: args.log_interval,
        eval_interval: args.eval_interval,
        save_interval: args.save_interval,
        checkpoint_dir: args.checkpoint_dir.to_string_lossy().to_string(),
        resume: args.resume,
        use_curriculum,
        use_multi_task,
        use_dynamic_batch,
        use_layer_wise_lr,
        use_progressive_growth,
        use_dynamic_ema,
        growth_config,
        max_grad_norm: Some(args.max_grad_norm),
        patience: Some(args.patience),
    };

    // Log dynamic features
    log::info!("Dynamic features enabled:");
    log::info!("  Curriculum learning: {}", use_curriculum);
    log::info!("  Multi-task training: {}", use_multi_task);
    log::info!("  Dynamic batch sizing: {}", use_dynamic_batch);
    log::info!("  Layer-wise learning rates: {}", use_layer_wise_lr);
    log::info!("  Progressive model growth: {}", use_progressive_growth);
    log::info!("  Dynamic EMA compounding: {}", use_dynamic_ema);

    // Create trainer
    log::info!("Initializing trainer...");
    let mut trainer =
        DynamicCompoundTrainer::new(training_config, train_dataset, val_dataset, device)?;

    // Enable tensorboard logging if requested
    if args.tensorboard {
        log::info!("Enabling Tensorboard logging to {:?}", args.tensorboard_dir);
        trainer.enable_metrics_logging(&args.tensorboard_dir)?;
    }

    // Run training
    log::info!("Starting training...");
    trainer.train()?;

    // Print final stats
    let stats = trainer.stats();
    log::info!("Training complete!");
    log::info!("Final statistics:");
    log::info!("  Steps: {}", stats.step);
    log::info!("  Final loss: {:.4}", stats.loss);
    log::info!("  Avg loss: {:.4}", stats.avg_loss);
    log::info!("  Growth progress: {:.1}%", stats.growth_progress * 100.0);

    Ok(())
}
