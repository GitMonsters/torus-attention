//! # Torus LLM Training CLI
//!
//! Train a Torus LLM on text data.
//!
//! ## Usage
//!
//! ```bash
//! # Train on a text file
//! cargo run --bin torus_train -- --data data/corpus.txt --epochs 10
//!
//! # Train on a directory
//! cargo run --bin torus_train -- --data data/ --epochs 10 --batch-size 64
//!
//! # Resume training
//! cargo run --bin torus_train -- --data data/corpus.txt --resume
//!
//! # Use a larger model
//! cargo run --bin torus_train -- --data data/corpus.txt --model small
//! ```

use candle_core::Device;
use std::path::Path;
use torus_attention::{
    dataset::TextDataset,
    llm::TorusLLMConfig,
    llm_trainer::{LLMTrainer, LLMTrainingConfig},
    tokenizer::SimpleTokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              🌀 Torus LLM Training Script 🌀                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args)?;

    // Setup device
    let device = if cfg!(feature = "cuda") {
        match Device::cuda_if_available(0) {
            Ok(d) => {
                println!("🚀 Using CUDA device");
                d
            }
            Err(_) => {
                println!("💻 CUDA not available, using CPU");
                Device::Cpu
            }
        }
    } else {
        println!("💻 Using CPU device");
        Device::Cpu
    };

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new_basic(config.model.vocab_size);
    println!("📝 Tokenizer: {} vocab size", tokenizer.vocab_size());

    // Load dataset
    let seq_len = config.model.to_torus_config().seq_len();
    let data_path = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/train.txt");

    println!("📂 Loading data from: {}", data_path);

    let dataset = if Path::new(data_path).is_dir() {
        TextDataset::from_directory(data_path, &tokenizer, seq_len)?
    } else if data_path.ends_with(".jsonl") {
        TextDataset::from_jsonl(data_path, &tokenizer, seq_len)?
    } else {
        TextDataset::from_file(data_path, &tokenizer, seq_len)?
    };

    if dataset.is_empty() {
        // Create synthetic data for testing
        println!("⚠️  No data found, using synthetic data for testing");
        let synthetic = TextDataset::synthetic(1000, seq_len, config.model.vocab_size);
        run_training(config, synthetic, device)?;
    } else {
        println!("✅ Loaded {} examples", dataset.len());

        // Split into train/val
        let (train_data, val_data) = dataset.train_val_split(0.1);
        println!("   Train: {} examples", train_data.len());
        println!("   Val: {} examples", val_data.len());

        run_training_with_val(config, train_data, val_data, device)?;
    }

    Ok(())
}

fn run_training(
    config: LLMTrainingConfig,
    train_data: TextDataset,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = LLMTrainer::new(config, train_data, None, device)?;
    trainer.train()?;
    Ok(())
}

fn run_training_with_val(
    config: LLMTrainingConfig,
    train_data: TextDataset,
    val_data: TextDataset,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = LLMTrainer::new(config, train_data, Some(val_data), device)?;
    trainer.train()?;
    Ok(())
}

fn parse_args(args: &[String]) -> Result<LLMTrainingConfig, Box<dyn std::error::Error>> {
    let mut config = LLMTrainingConfig::default();

    // Parse model size
    if let Some(pos) = args.iter().position(|a| a == "--model") {
        if let Some(size) = args.get(pos + 1) {
            config.model = match size.as_str() {
                "tiny" => TorusLLMConfig::tiny(),
                "small" => TorusLLMConfig::small(),
                "medium" => TorusLLMConfig::medium(),
                _ => {
                    println!("Unknown model size '{}', using tiny", size);
                    TorusLLMConfig::tiny()
                }
            };
            println!("🔧 Model: {}", size);
        }
    }

    // Parse batch size
    if let Some(pos) = args.iter().position(|a| a == "--batch-size" || a == "-b") {
        if let Some(bs) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
            config.batch_size = bs;
        }
    }
    println!("📦 Batch size: {}", config.batch_size);

    // Parse epochs
    if let Some(pos) = args.iter().position(|a| a == "--epochs" || a == "-e") {
        if let Some(epochs) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
            config.num_epochs = epochs;
        }
    }
    println!("🔄 Epochs: {}", config.num_epochs);

    // Parse learning rate
    if let Some(pos) = args.iter().position(|a| a == "--lr") {
        if let Some(lr) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
            config.learning_rate = lr;
        }
    }
    println!("📈 Learning rate: {}", config.learning_rate);

    // Parse checkpoint dir
    if let Some(pos) = args
        .iter()
        .position(|a| a == "--checkpoint-dir" || a == "-c")
    {
        if let Some(dir) = args.get(pos + 1) {
            config.checkpoint_dir = dir.clone();
        }
    }
    println!("💾 Checkpoint dir: {}", config.checkpoint_dir);

    // Parse resume flag
    if args.iter().any(|a| a == "--resume" || a == "-r") {
        config.resume = true;
        println!("↩️  Resume: enabled");
    } else if args.iter().any(|a| a == "--no-resume") {
        config.resume = false;
    }

    // Parse max steps
    if let Some(pos) = args.iter().position(|a| a == "--max-steps") {
        if let Some(steps) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
            config.max_steps = Some(steps);
            println!("🛑 Max steps: {}", steps);
        }
    }

    // Parse log interval
    if let Some(pos) = args.iter().position(|a| a == "--log-interval") {
        if let Some(interval) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
            config.log_interval = interval;
        }
    }

    // Print help
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    println!();
    Ok(config)
}

fn print_help() {
    println!(
        "
Torus LLM Training Script

USAGE:
    torus_train [OPTIONS]

OPTIONS:
    --data <PATH>           Path to training data (file or directory)
    --model <SIZE>          Model size: tiny, small, medium (default: tiny)
    --batch-size, -b <N>    Batch size (default: 32)
    --epochs, -e <N>        Number of epochs (default: 10)
    --lr <RATE>             Learning rate (default: 3e-4)
    --max-steps <N>         Maximum training steps
    --checkpoint-dir, -c    Checkpoint directory (default: checkpoints)
    --resume, -r            Resume from latest checkpoint
    --no-resume             Start fresh, ignore existing checkpoints
    --log-interval <N>      Steps between logs (default: 10)
    --help, -h              Show this help message

EXAMPLES:
    # Train on a text file
    torus_train --data corpus.txt --epochs 10

    # Train with larger model
    torus_train --data corpus.txt --model small --batch-size 16

    # Resume interrupted training
    torus_train --data corpus.txt --resume
"
    );
}
