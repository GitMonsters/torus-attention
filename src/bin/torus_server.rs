//! # Torus LLM API Server Binary
//!
//! Run this to start an OpenAI-compatible API server.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin torus_server
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
use std::sync::Arc;
use torus_attention::{
    api_server::{server::run_server, ApiHandler, ServerConfig},
    llm::{TorusLLM, TorusLLMConfig},
    tokenizer::SimpleTokenizer,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           🌀 Torus Attention LLM API Server 🌀              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_size = args.get(1).map(|s| s.as_str()).unwrap_or("tiny");
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8080);
    let host = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "127.0.0.1".to_string());

    // Select model configuration
    let model_config = match model_size {
        "tiny" => {
            println!("📦 Loading tiny model (128 dim, 2 layers)...");
            TorusLLMConfig::tiny()
        }
        "small" => {
            println!("📦 Loading small model (768 dim, 12 layers)...");
            TorusLLMConfig::small()
        }
        "medium" => {
            println!("📦 Loading medium model (1024 dim, 24 layers)...");
            TorusLLMConfig::medium()
        }
        _ => {
            eprintln!("Unknown model size: {}. Using tiny.", model_size);
            TorusLLMConfig::tiny()
        }
    };

    // Check for CUDA
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

    // Create model
    println!("🔧 Initializing model...");
    let vocab_size = model_config.vocab_size;
    let (model, _varmap) = TorusLLM::new_random(model_config.clone(), &device)?;

    // Create tokenizer
    println!("📝 Initializing tokenizer...");
    let tokenizer = SimpleTokenizer::new_basic(vocab_size);

    // Server configuration
    let server_config = ServerConfig {
        host: host.clone(),
        port,
        model_config,
        max_tokens: 2048,
        default_temperature: 0.7,
    };

    // Create API handler
    let handler = Arc::new(ApiHandler::new(model, tokenizer, server_config.clone()));

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  Server starting on http://{}:{}", host, port);
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("📋 OpenCode configuration:");
    println!();
    println!("   # Add to your opencode.toml:");
    println!("   [provider.torus]");
    println!("   type = \"openai\"");
    println!("   api_key = \"not-needed\"");
    println!("   base_url = \"http://{}:{}/v1\"", host, port);
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
