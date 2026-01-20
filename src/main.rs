//! Torus Attention Demo
//!
//! Demonstrates the torus attention mechanism with:
//! - Basic torus geometry
//! - Vortex dynamics
//! - 8-stream bidirectional parallel processing
//! - Learnable EMA compounding

use std::f64::consts::PI;
use torus_attention::{
    // Bidirectional processing
    bidirectional::{CausalMask, FlowDirection, SymmetricCombiner},
    compounding::{CompoundingConfig, LearnableAlpha},
    // Core geometry
    geometry::{TorusCoordinate, TorusManifold},
    integration::BidirectionalTorusConfig,
    parallel_streams::{ParallelStreamConfig, StreamId},
    periodic::PeriodicBoundary,
    // Vortex dynamics
    vortex::{BidirectionalSpiral, VortexDynamics},
};

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  rustyWorm simpleCOMPLEXITY - Bidirectional Torus Attention Demo     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // 1. Core torus geometry
    demo_torus_geometry();

    // 2. Periodic boundary conditions
    demo_periodic_boundary();

    // 3. Vortex dynamics with CW/CCW spirals
    demo_vortex_dynamics();

    // 4. Bidirectional flow primitives
    demo_bidirectional_flow();

    // 5. 8-stream parallel configuration
    demo_parallel_streams();

    // 6. EMA compounding configuration
    demo_compounding();

    // 7. Full system configuration
    demo_full_system();

    println!("\n✓ All demonstrations completed successfully!");
    println!("\n═══ System Ready for Training ═══");
}

fn demo_torus_geometry() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. TORUS GEOMETRY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let torus = TorusManifold::new(2.0, 1.0);
    println!("Torus Manifold:");
    println!("  Major radius (R): {}", torus.major_radius);
    println!("  Minor radius (r): {}", torus.minor_radius);
    println!("  Aspect ratio:     {:.2}", torus.aspect_ratio);
    println!("  Surface area:     {:.4}", torus.surface_area());
    println!("  Volume:           {:.4}", torus.volume());

    // Coordinate examples
    let coord = TorusCoordinate::new(PI / 4.0, PI / 3.0);
    let (x, y, z) = coord.to_cartesian(torus.major_radius, torus.minor_radius);
    println!("\nSample Coordinate:");
    println!("  Angular:   (u={:.4}, v={:.4})", coord.u, coord.v);
    println!("  Cartesian: ({:.4}, {:.4}, {:.4})", x, y, z);

    // Geodesic distance
    let coord2 = TorusCoordinate::new(PI, PI / 2.0);
    let dist = coord.geodesic_distance(&coord2);
    println!("  Distance to (π, π/2): {:.4}\n", dist);
}

fn demo_periodic_boundary() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. PERIODIC BOUNDARY CONDITIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let boundary = PeriodicBoundary::new(32, 16);
    println!("Grid Configuration:");
    println!("  Major points: {}", boundary.n_major);
    println!("  Minor points: {}", boundary.n_minor);
    println!("  Total positions: {}", boundary.n_major * boundary.n_minor);
    println!(
        "  Grid spacing: du={:.4}, dv={:.4}",
        boundary.du, boundary.dv
    );

    // Wrapping examples
    println!("\nIndex Wrapping (periodic):");
    let test_cases = [(-1, 0), (32, 0), (35, 20), (-5, -3)];
    for (i, j) in test_cases {
        let (wi, wj) = boundary.wrap_2d(i, j);
        println!("  ({:3}, {:3}) → ({:2}, {:2})", i, j, wi, wj);
    }
    println!();
}

fn demo_vortex_dynamics() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. VORTEX DYNAMICS & SPIRAL FLOWS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Vortex pair dynamics
    let manifold = TorusManifold::unit();
    let mut dynamics = VortexDynamics::new(manifold, 0.01);
    dynamics.add_vortex_pair(PI, PI, 0.5, 1.0);

    println!("Vortex-Antivortex Pair:");
    println!("  Initial circulation: {:.4}", dynamics.total_circulation());
    println!("  Initial energy:      {:.4}", dynamics.energy());

    dynamics.run(100);
    println!("\nAfter 100 steps (t={:.2}):", dynamics.time);
    println!(
        "  Final circulation:   {:.4} (conserved)",
        dynamics.total_circulation()
    );
    println!("  Final energy:        {:.4}", dynamics.energy());

    // Bidirectional spiral
    println!("\nBidirectional Spiral (Golden Ratio):");
    let bispiral = BidirectionalSpiral::golden();
    println!("  CW winding:  {:.6}", bispiral.cw.spiral.winding);
    println!("  CCW winding: {:.6}", bispiral.ccw.spiral.winding);

    // Sample spiral positions
    println!("\nCW Spiral Positions (first 4):");
    let cw_positions = bispiral.cw.sample_positions(8, 0.0);
    for (i, pos) in cw_positions.iter().take(4).enumerate() {
        println!("  {}: (u={:.4}, v={:.4})", i, pos.u, pos.v);
    }

    println!("\nCCW Spiral Positions (first 4):");
    let ccw_positions = bispiral.ccw.sample_positions(8, 0.0);
    for (i, pos) in ccw_positions.iter().take(4).enumerate() {
        println!("  {}: (u={:.4}, v={:.4})", i, pos.u, pos.v);
    }
    println!();
}

fn demo_bidirectional_flow() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. BIDIRECTIONAL FLOW PRIMITIVES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Flow Directions:");
    println!(
        "  Forward:  {} (sign={})",
        if FlowDirection::Forward.is_forward() {
            "→"
        } else {
            "←"
        },
        FlowDirection::Forward.sign()
    );
    println!(
        "  Backward: {} (sign={})",
        if FlowDirection::Backward.is_forward() {
            "→"
        } else {
            "←"
        },
        FlowDirection::Backward.sign()
    );

    // Causal masks
    println!("\nCausal Masks (4x4 example):");
    let device = candle_core::Device::Cpu;

    let mut forward_mask = CausalMask::new(4, FlowDirection::Forward);
    let mut backward_mask = CausalMask::new(4, FlowDirection::Backward);

    let fwd = forward_mask.get_mask(&device).unwrap();
    let bwd = backward_mask.get_mask(&device).unwrap();

    println!("\n  Forward (lower triangular):");
    let fwd_data: Vec<f32> = fwd.flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..4 {
        print!("    ");
        for j in 0..4 {
            print!("{:.0} ", fwd_data[i * 4 + j]);
        }
        println!();
    }

    println!("\n  Backward (upper triangular):");
    let bwd_data: Vec<f32> = bwd.flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..4 {
        print!("    ");
        for j in 0..4 {
            print!("{:.0} ", bwd_data[i * 4 + j]);
        }
        println!();
    }

    // Symmetric combiner
    println!("\nSymmetric Combiner:");
    let combiner = SymmetricCombiner::new(&device, 1.0).unwrap();
    let (w_f, w_b) = combiner.get_weights().unwrap();
    println!("  Initial weights: forward={:.4}, backward={:.4}", w_f, w_b);
    println!("  Sum = {:.4} (should be 1.0)", w_f + w_b);
    println!();
}

fn demo_parallel_streams() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. 8-STREAM PARALLEL PROCESSING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Stream Configuration:");
    println!("┌──────────────────┬─────────────┬────────────┐");
    println!("│ Stream           │ Direction   │ Pair       │");
    println!("├──────────────────┼─────────────┼────────────┤");

    for id in StreamId::all() {
        let dir = if id.is_forward() {
            "Forward "
        } else {
            "Backward"
        };
        let pair = id.pair();
        println!("│ {:16} │ {:11} │ {:10} │", id.name(), dir, pair.name());
    }
    println!("└──────────────────┴─────────────┴────────────┘");

    // Configuration
    let config = ParallelStreamConfig::default();
    println!("\nDefault Configuration:");
    println!("  Model dimension: {}", config.d_model);
    println!("  Attention heads: {}", config.n_heads);
    println!("  Grid size:       {}x{}", config.n_major, config.n_minor);
    println!(
        "  Spiral winding:  {:.6} (golden ratio)",
        config.spiral_winding
    );
    println!("  Parallel exec:   {}", config.parallel);
    println!();
}

fn demo_compounding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. LEARNABLE EMA COMPOUNDING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let config = CompoundingConfig::default();

    println!("EMA Compounding Configuration:");
    println!("  Number of layers: {}", config.n_layers);
    println!("  Base α:           {}", config.base_alpha);
    println!(
        "  α range:          [{}, {}]",
        config.min_alpha, config.max_alpha
    );
    println!("  Layer scale:      {}", config.layer_scale);
    println!("  Learnable α:      {}", config.learnable_alpha);
    println!("  Use momentum:     {}", config.use_momentum);
    println!("  Momentum β:       {}", config.momentum_beta);

    // Show layer-wise alpha values
    println!("\nLayer-wise α (with scaling):");
    for l in 0..config.n_layers {
        let alpha = config.base_alpha * config.layer_scale.powi(l as i32);
        let alpha = alpha.clamp(config.min_alpha, config.max_alpha);
        let bar_len = (alpha * 20.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  Layer {}: α={:.4} {}", l, alpha, bar);
    }

    // Learnable alpha example
    println!("\nLearnable Alpha:");
    let device = candle_core::Device::Cpu;
    let alpha = LearnableAlpha::new(0.9, 0.1, 0.99, &device).unwrap();
    println!("  Initial value: {:.4}", alpha.get().unwrap());
    println!("  Constrained to [{:.2}, {:.2}] via sigmoid", 0.1, 0.99);
    println!();
}

fn demo_full_system() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  7. FULL BIDIRECTIONAL TORUS SYSTEM");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let config = BidirectionalTorusConfig::default();

    println!("Bidirectional Torus Transformer Configuration:");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ MODEL ARCHITECTURE                                                  │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Model dimension:     {:4}                                          │",
        config.d_model
    );
    println!(
        "│  Feed-forward dim:    {:4}                                          │",
        config.d_ff
    );
    println!(
        "│  Attention heads:     {:4}                                          │",
        config.n_heads
    );
    println!(
        "│  Number of layers:    {:4}                                          │",
        config.n_layers
    );
    println!(
        "│  Sequence length:     {:4} ({}×{})                               │",
        config.seq_len(),
        config.n_major,
        config.n_minor
    );
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│ TORUS GEOMETRY                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Major radius:        {:4.1}                                          │",
        config.major_radius
    );
    println!(
        "│  Minor radius:        {:4.1}                                          │",
        config.minor_radius
    );
    println!(
        "│  Geodesic σ:          {:4.1}                                          │",
        config.geodesic_sigma
    );
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│ BIDIRECTIONAL PROCESSING                                            │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  8-stream parallel:   {:5}                                         │",
        config.use_parallel_streams
    );
    println!(
        "│  Spiral winding (φ):  {:6.4}                                       │",
        config.spiral_winding
    );
    println!(
        "│  Parallel execution:  {:5}                                         │",
        config.parallel_execution
    );
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│ COMPOUNDING                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  EMA compounding:     {:5}                                         │",
        config.use_compounding
    );
    println!(
        "│  Base α:              {:5.2}                                         │",
        config.ema_alpha
    );
    println!(
        "│  Learnable α:         {:5}                                         │",
        config.learnable_alpha
    );
    println!(
        "│  Use momentum:        {:5}                                         │",
        config.use_momentum
    );
    println!("└─────────────────────────────────────────────────────────────────────┘");

    // ASCII diagram
    println!("\nData Flow:");
    println!();
    println!("  Input");
    println!("    │");
    println!("    ▼");
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │              Bidirectional Position Encoding                │");
    println!("  │         (forward + backward torus coordinates)              │");
    println!("  └────────────────────────┬────────────────────────────────────┘");
    println!("                           │");
    println!("                           ▼");
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │                 8-STREAM PARALLEL ATTENTION                 │");
    println!("  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                    │");
    println!("  │  │Major→ │ │Major← │ │Minor→ │ │Minor← │                    │");
    println!("  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘                    │");
    println!("  │  ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐                    │");
    println!("  │  │Spiral │ │Spiral │ │Cross  │ │Cross  │                    │");
    println!("  │  │  CW   │ │  CCW  │ │ U→V   │ │ V→U   │                    │");
    println!("  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘                    │");
    println!("  │      └────┬────┴────┬────┴────┬────┘                        │");
    println!("  │           ▼         ▼         ▼                             │");
    println!("  │        Symmetric Weighted Combination                       │");
    println!("  │           (learnable softmax weights)                       │");
    println!("  └────────────────────────┬────────────────────────────────────┘");
    println!("                           │");
    println!("                           ▼");
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │                    EMA COMPOUNDING                          │");
    println!("  │           h_t = α·new + (1-α)·h_{{t-1}}                       │");
    println!("  │              (learnable α per layer)                        │");
    println!("  └────────────────────────┬────────────────────────────────────┘");
    println!("                           │");
    println!("                           ▼");
    println!("                       [× {} layers]", config.n_layers);
    println!("                           │");
    println!("                           ▼");
    println!("                        Output");
    println!();

    // Parameter count estimate
    let params_per_stream = config.d_model * config.d_model * 4; // Q,K,V,O
    let params_streams = params_per_stream * 8;
    let params_ff = config.d_model * config.d_ff * 2;
    let params_layer = params_streams + params_ff;
    let total_params = params_layer * config.n_layers;

    println!("Estimated Parameters:");
    println!("  Per stream:  {} (Q,K,V,O projections)", params_per_stream);
    println!("  8 streams:   {}", params_streams);
    println!("  Feed-forward:{}", params_ff);
    println!("  Per layer:   {}", params_layer);
    println!(
        "  Total:       {} ({:.1}M)",
        total_params,
        total_params as f64 / 1_000_000.0
    );
}
