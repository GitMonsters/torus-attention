# rustyWorm simpleCOMPLEXITY

**Torus Topology and Vortex Dynamics Torus Attention Mechanism in Rust**

A transformer-style attention mechanism built on a torus manifold, leveraging periodic boundary conditions, dual-loop (major/minor) information flow, spiral (vortex) dynamics, and **8-stream bidirectional parallel processing with learnable EMA compounding**.

## Features

### Core
- **Torus Geometry**: Full parametric torus with geodesic distance calculations
- **Periodic Boundaries**: Proper wrapping and convolution on closed manifolds
- **Dual-Loop Attention**: Separate attention for major (u) and minor (v) directions
- **Vortex Dynamics**: Point vortex simulation with conservation laws
- **Spiral Attention**: Golden ratio and multi-arm spiral attention patterns

### Bidirectional Processing (NEW)
- **8-Stream Parallel Processing**: Concurrent attention streams with rayon
- **Symmetric Bidirectional Flow**: Forward/backward with learned mixing weights
- **CW/CCW Spiral Attention**: Clockwise and counter-clockwise spiral flows
- **Cross-Loop Coupling**: U→V and V→U information transfer

### Compounding Integration (NEW)
- **Learnable EMA**: Per-layer exponential moving average with learnable α
- **Multi-Scale Compounding**: Fast/medium/slow time constants
- **Momentum Enhancement**: Adam-style momentum for compounding
- **Bias Correction**: Warmup compensation for early steps

### Infrastructure
- **GPU Acceleration**: Via candle-core (Hugging Face's Rust ML framework)
- **CPU Parallelism**: Via rayon for 8-stream concurrent execution
- **Python Bindings**: PyO3 integration for hybrid workflows

## Architecture

### 8-Stream Bidirectional Flow

```
Input
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    8-STREAM PARALLEL ATTENTION                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Major Fwd → │ │ Major Bwd ← │ │ Minor Fwd → │ │ Minor Bwd ← │        │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘        │
│  ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐        │
│  │ Spiral CW ↻ │ │ Spiral CCW ↺│ │ Cross U→V   │ │ Cross V→U   │        │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘        │
│         └────────┬──────┴────────┬──────┴────────┬──────┘               │
│                  ▼               ▼               ▼                      │
│              Symmetric Weighted Combination (softmax)                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EMA COMPOUNDING                                  │
│                  h_t = α·new + (1-α)·h_{t-1}                             │
│                    (learnable α per layer)                               │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
                              [× N layers]
                                   │
                                   ▼
                                Output
```

### Stream Types

| Stream | Direction | Description |
|--------|-----------|-------------|
| Major Forward | u: 0→2π | Causal attention along major loop |
| Major Backward | u: 2π→0 | Anti-causal attention along major loop |
| Minor Forward | v: 0→2π | Causal attention along minor loop |
| Minor Backward | v: 2π→0 | Anti-causal attention along minor loop |
| Spiral CW | ↻ | Clockwise spiral with golden winding |
| Spiral CCW | ↺ | Counter-clockwise spiral with golden winding |
| Cross U→V | u→v | Major-to-minor cross-loop coupling |
| Cross V→U | v→u | Minor-to-major cross-loop coupling |

## Dependencies

```toml
[dependencies]
# Core ML
candle-core = "0.4"
candle-nn = "0.4"
ndarray = "0.15"
burn = "0.11"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Parallelism
rayon = "1.7"
crossbeam = "0.8"
crossbeam-channel = "0.5"
parking_lot = "0.12"

# Async
tokio = { version = "1", features = ["full"] }

# Python
pyo3 = { version = "0.20", features = ["extension-module"] }

# Utilities
thiserror = "1.0"
log = "0.4"
env_logger = "0.10"
```

## Usage

### Basic Torus Geometry

```rust
use torus_attention::{TorusCoordinate, TorusManifold};
use std::f64::consts::PI;

// Create torus manifold
let torus = TorusManifold::new(2.0, 1.0);
println!("Surface area: {}", torus.surface_area());

// Generate grid
let coords = torus.generate_grid(32, 16);

// Geodesic distance
let p1 = TorusCoordinate::new(0.0, 0.0);
let p2 = TorusCoordinate::new(PI, PI/2.0);
let dist = p1.geodesic_distance(&p2);
```

### Bidirectional Attention

```rust
use torus_attention::{
    BidirectionalTorusConfig,
    BidirectionalTorusTransformer,
    StreamId,
};

// Configure
let config = BidirectionalTorusConfig {
    d_model: 256,
    n_heads: 8,
    n_layers: 6,
    n_major: 32,
    n_minor: 16,
    use_parallel_streams: true,
    use_compounding: true,
    learnable_alpha: true,
    ..Default::default()
};

// Stream weights are learned
for id in StreamId::all() {
    println!("{}: forward={}", id.name(), id.is_forward());
}
```

### EMA Compounding

```rust
use torus_attention::{CompoundingConfig, EMACompounding};

let config = CompoundingConfig {
    n_layers: 6,
    base_alpha: 0.9,
    learnable_alpha: true,
    use_momentum: true,
    ..Default::default()
};

// Alpha is constrained to [min_alpha, max_alpha] via sigmoid
// h_t = α·new + (1-α)·h_{t-1}
```

### Vortex Dynamics

```rust
use torus_attention::{
    VortexDynamics, TorusManifold,
    BidirectionalSpiral, SpiralDirection,
};

// Create vortex pair
let mut dynamics = VortexDynamics::new(TorusManifold::unit(), 0.01);
dynamics.add_vortex_pair(PI, PI, 0.5, 1.0);
dynamics.run(100);

// Bidirectional spiral (CW + CCW)
let bispiral = BidirectionalSpiral::golden();
let cw_positions = bispiral.cw.sample_positions(64, 0.0);
let ccw_positions = bispiral.ccw.sample_positions(64, 0.0);
```

## Building

```bash
# Build library
cargo build --release

# Build with Python bindings
cargo build --release --features python

# Build with CUDA support
cargo build --release --features cuda

# Run demo
cargo run --bin torus_demo

# Run tests
cargo test
```

## Training

### Quick Start

```rust
use torus_attention::{
    BidirectionalTorusConfig,
    TrainingConfig,
    Trainer,
    run_training_example,
};
use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu; // or Device::cuda_if_available(0)?
    
    // Run quick training example
    let metrics = run_training_example(&device)?;
    println!("{}", metrics.summary());
    
    Ok(())
}
```

### Custom Training Loop

```rust
use torus_attention::{
    BidirectionalTorusConfig,
    TrainingConfig,
    Trainer,
    training::generate_random_batch,
};
use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Model configuration
    let model_config = BidirectionalTorusConfig {
        d_model: 256,
        n_heads: 8,
        n_layers: 6,
        n_major: 32,
        n_minor: 16,
        use_parallel_streams: true,
        use_compounding: true,
        learnable_alpha: true,
        ..Default::default()
    };
    
    // Training configuration
    let training_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 32,
        epochs: 10,
        warmup_steps: 1000,
        ..Default::default()
    };
    
    // Create trainer
    let vocab_size = 32000;
    let mut trainer = Trainer::new(
        model_config.clone(),
        training_config.clone(),
        Some(vocab_size),
        &device,
    )?;
    
    // Training loop
    for epoch in 0..training_config.epochs {
        // Generate batch (replace with your data loader)
        let (inputs, targets) = generate_random_batch(
            training_config.batch_size,
            model_config.seq_len(),
            model_config.d_model,
            vocab_size,
            &device,
        )?;
        
        // Train step
        let loss = trainer.train_step(&inputs, &targets)?;
        
        if trainer.global_step() % 100 == 0 {
            println!("Step {}: loss={:.4}", trainer.global_step(), loss);
        }
    }
    
    // Save checkpoint
    trainer.save_checkpoint("checkpoint.safetensors")?;
    
    Ok(())
}
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 1e-4 | Base learning rate |
| weight_decay | 0.01 | AdamW weight decay |
| beta1 | 0.9 | Adam beta1 |
| beta2 | 0.999 | Adam beta2 |
| grad_clip_norm | 1.0 | Gradient clipping threshold |
| batch_size | 32 | Training batch size |
| epochs | 10 | Number of epochs |
| warmup_steps | 1000 | LR warmup steps |
| total_steps | 100000 | Total training steps |

### Learning Rate Schedule

The trainer uses warmup + cosine annealing:

```
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

## Mathematical Background

### EMA Compounding

Exponential Moving Average with learnable decay:

```
h_t = α · x_t + (1 - α) · h_{t-1}
```

Where α ∈ [α_min, α_max] is parameterized as:

```
α = α_min + σ(θ) · (α_max - α_min)
```

With momentum enhancement:

```
v_t = β · v_{t-1} + (1 - β) · (x_t - h_{t-1})
h_t = h_{t-1} + α · v_t
```

### Symmetric Bidirectional Combination

Forward and backward flows combined with learned softmax weights:

```
output = softmax([w_f, w_b]) · [forward, backward]
       = (e^{w_f} · forward + e^{w_b} · backward) / (e^{w_f} + e^{w_b})
```

Ensures w_forward + w_backward = 1 always.

### Geodesic Distance on Torus

```
d(p₁, p₂) = √(min(|Δu|, 2π-|Δu|)² + min(|Δv|, 2π-|Δv|)²)
```

### Spiral Winding

Golden ratio spiral winding:

```
φ = (1 + √5) / 2 ≈ 1.618034
spiral_position = u + φ · v
```

## Configuration Reference

### BidirectionalTorusConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 256 | Model dimension |
| d_ff | 1024 | Feed-forward dimension |
| n_heads | 8 | Attention heads per stream |
| n_layers | 6 | Number of transformer layers |
| n_major | 32 | Grid size in u direction |
| n_minor | 16 | Grid size in v direction |
| use_parallel_streams | true | Enable 8-stream processing |
| use_compounding | true | Enable EMA compounding |
| learnable_alpha | true | Learn α per layer |
| ema_alpha | 0.9 | Base EMA decay rate |
| spiral_winding | φ | Golden ratio winding |
| parallel_execution | true | Use rayon parallelism |

## License

MIT License - See LICENSE file

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Aref, "Point vortex dynamics: A classical mathematics playground"
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al., "Train Short, Test Long: Attention with Linear Biases"
