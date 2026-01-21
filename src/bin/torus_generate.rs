//! # Torus LLM Text Generation Binary
//!
//! Quick text generation from a trained checkpoint.
//!
//! ## Usage
//!
//! ```bash
//! # Generate from a checkpoint
//! cargo run --release --bin torus_generate -- -c checkpoints/best -p "The torus"
//!
//! # With custom parameters
//! cargo run --release --bin torus_generate -- \
//!     -c checkpoints/best \
//!     -p "Attention mechanisms" \
//!     --max-tokens 100 \
//!     --temperature 0.8
//! ```

use candle_core::Device;
use clap::Parser;
use torus_attention::{
    checkpoint::load_checkpoint,
    llm::{SamplingStrategy, TextGenerator, TorusLLM, TorusLLMConfig},
    tokenizer::SimpleTokenizer,
};

/// Torus LLM Text Generator
#[derive(Parser, Debug)]
#[command(name = "torus_generate")]
#[command(about = "Generate text from a Torus Attention LLM")]
struct Args {
    /// Path to checkpoint directory
    #[arg(long, short = 'c')]
    checkpoint: Option<String>,

    /// Prompt text to continue
    #[arg(long, short = 'p', default_value = "The")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, short = 'm', default_value = "50")]
    max_tokens: usize,

    /// Sampling temperature (higher = more random)
    #[arg(long, short = 't', default_value = "0.8")]
    temperature: f64,

    /// Top-k sampling (0 = disabled)
    #[arg(long, short = 'k', default_value = "50")]
    top_k: usize,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.9")]
    top_p: f64,

    /// Use greedy decoding (deterministic)
    #[arg(long)]
    greedy: bool,

    /// Number of samples to generate
    #[arg(long, short = 'n', default_value = "1")]
    num_samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Torus Attention LLM - Text Generator");
    println!("=====================================\n");

    let device = Device::Cpu;

    // Load or create model
    let (model, config) = if let Some(ref checkpoint_path) = args.checkpoint {
        println!("[LOAD] Loading checkpoint: {}", checkpoint_path);
        let (model, _varmap, metadata) = load_checkpoint(checkpoint_path, &device)?;
        println!(
            "[INFO] Model: {} dim, {} layers, {} heads, vocab {}",
            metadata.config.hidden_dim,
            metadata.config.num_layers,
            metadata.config.num_heads,
            metadata.config.vocab_size
        );
        if let Some(loss) = metadata.loss {
            println!("[INFO] Training loss: {:.4}", loss);
        }
        (model, metadata.config)
    } else {
        println!("[NEW] Creating random tiny model (for testing)");
        let config = TorusLLMConfig::tiny();
        let (model, _varmap) = TorusLLM::new_random(config.clone(), &device)?;
        (model, config)
    };

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new_basic(config.vocab_size);

    // Set up sampling strategy
    let strategy = if args.greedy {
        println!("[SAMPLE] Using greedy decoding");
        SamplingStrategy::Greedy
    } else {
        println!(
            "[SAMPLE] Temperature={}, Top-k={}, Top-p={}",
            args.temperature, args.top_k, args.top_p
        );
        SamplingStrategy::Combined {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
        }
    };

    let generator = TextGenerator::new(model, strategy);

    // Tokenize prompt
    let prompt_tokens = tokenizer.encode(&args.prompt);
    println!("\n[PROMPT] \"{}\"", args.prompt);
    println!("[TOKENS] {:?} ({} tokens)\n", prompt_tokens, prompt_tokens.len());

    // Generate samples
    for i in 0..args.num_samples {
        if args.num_samples > 1 {
            println!("--- Sample {} ---", i + 1);
        }

        match generator.generate(&prompt_tokens, args.max_tokens, None) {
            Ok(generated_tokens) => {
                let generated_text = tokenizer.decode(&generated_tokens);
                let full_text = format!("{}{}", args.prompt, generated_text);

                println!("[OUTPUT] {}", full_text);
                println!(
                    "[STATS] Generated {} tokens: {:?}\n",
                    generated_tokens.len(),
                    &generated_tokens[..generated_tokens.len().min(20)]
                );
            }
            Err(e) => {
                eprintln!("[ERROR] Generation failed: {}", e);
            }
        }
    }

    Ok(())
}
