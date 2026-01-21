//! # Torus LLM API Server Binary
//!
//! Run this to start an OpenAI-compatible API server.
//!
//! ## Usage
//!
//! ```bash
//! # Start with a random model (for testing)
//! cargo run --bin torus_server
//!
//! # Load a trained checkpoint
//! cargo run --bin torus_server -- --checkpoint checkpoints/best
//!
//! # Specify port and host
//! cargo run --bin torus_server -- --checkpoint checkpoints/best --port 8080 --host 0.0.0.0
//! ```
//!
//! Then configure OpenCode to use it:
//!
//! ```toml
//! # opencode.toml
//! [provider.torus]
//! type = "openai"  
//! api_key = "not-needed"
//! base_url = "http://localhost:8080/v1"
//!
//! [model.torus]
//! provider = "torus"
//! model = "torus-small"
//! ```

use candle_core::Device;
use clap::Parser;
use std::sync::Arc;
use torus_attention::{
    api_server::{server::run_server, ApiHandler, ServerConfig},
    checkpoint::load_checkpoint,
    llm::{TorusLLM, TorusLLMConfig},
    tokenizer::SimpleTokenizer,
};

/// Torus LLM API Server
#[derive(Parser, Debug)]
#[command(name = "torus_server")]
#[command(about = "OpenAI-compatible API server for Torus Attention LLM")]
struct Args {
    /// Path to checkpoint directory to load trained model
    #[arg(long, short = 'c')]
    checkpoint: Option<String>,

    /// Model size when not loading checkpoint: tiny, small, medium
    #[arg(long, short = 's', default_value = "tiny")]
    size: String,

    /// Port to listen on
    #[arg(long, short = 'p', default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(long, short = 'H', default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Torus Attention LLM API Server                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args = Args::parse();

    // Check for CUDA
    let device = if cfg!(feature = "cuda") {
        match Device::cuda_if_available(0) {
            Ok(d) => {
                println!("[CUDA] Using CUDA device");
                d
            }
            Err(_) => {
                println!("[CPU] CUDA not available, using CPU");
                Device::Cpu
            }
        }
    } else {
        println!("[CPU] Using CPU device");
        Device::Cpu
    };

    // Load or create model
    let (model, model_config) = if let Some(ref checkpoint_path) = args.checkpoint {
        println!("[LOAD] Loading checkpoint from: {}", checkpoint_path);
        let (model, _varmap, metadata) = load_checkpoint(checkpoint_path, &device)?;
        println!(
            "[INFO] Loaded model: {} dim, {} layers, {} heads",
            metadata.config.hidden_dim, metadata.config.num_layers, metadata.config.num_heads
        );
        if let Some(loss) = metadata.loss {
            println!("[INFO] Training loss: {:.4}", loss);
        }
        if let Some(step) = metadata.step {
            println!("[INFO] Training step: {}", step);
        }
        (model, metadata.config)
    } else {
        // Select model configuration by size
        let model_config = match args.size.as_str() {
            "tiny" => {
                println!("[NEW] Creating tiny model (128 dim, 2 layers)...");
                TorusLLMConfig::tiny()
            }
            "small" => {
                println!("[NEW] Creating small model (768 dim, 12 layers)...");
                TorusLLMConfig::small()
            }
            "medium" => {
                println!("[NEW] Creating medium model (1024 dim, 24 layers)...");
                TorusLLMConfig::medium()
            }
            _ => {
                eprintln!("Unknown model size: {}. Using tiny.", args.size);
                TorusLLMConfig::tiny()
            }
        };

        println!("[INIT] Initializing random model...");
        let (model, _varmap) = TorusLLM::new_random(model_config.clone(), &device)?;
        (model, model_config)
    };

    let vocab_size = model_config.vocab_size;

    // Create tokenizer
    println!("[INIT] Initializing tokenizer...");
    let tokenizer = SimpleTokenizer::new_basic(vocab_size);

    // Server configuration
    let server_config = ServerConfig {
        host: args.host.clone(),
        port: args.port,
        model_config,
        max_tokens: 2048,
        default_temperature: 0.7,
    };

    // Create API handler
    let handler = Arc::new(ApiHandler::new(model, tokenizer, server_config.clone()));

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  Server starting on http://{}:{}", args.host, args.port);
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("OpenCode configuration:");
    println!();
    println!("   # Add to your opencode.toml:");
    println!("   [provider.torus]");
    println!("   type = \"openai\"");
    println!("   api_key = \"not-needed\"");
    println!("   base_url = \"http://{}:{}/v1\"", args.host, args.port);
    println!();
    println!("   [model.torus]");
    println!("   provider = \"torus\"");
    println!("   model = \"torus-small\"");
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!();

    // Run server
    run_server(handler, &server_config).await?;

    Ok(())
}
