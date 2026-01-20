//! Stream Analysis Example
//!
//! This example explores the 8-stream bidirectional parallel processing system,
//! showing how different streams capture different patterns in the data.
//!
//! Run with: cargo run --example stream_analysis

use torus_attention::{FlowDirection, PeriodicBoundary, StreamId};

fn main() {
    println!("═══ 8-Stream Bidirectional Analysis ═══\n");

    // ─────────────────────────────────────────────────────────────────────
    // 1. Stream Overview
    // ─────────────────────────────────────────────────────────────────────
    println!("── Stream Overview ──\n");

    let streams = StreamId::all();

    println!("The Torus Attention system uses 8 parallel processing streams:");
    println!(
        "┌──────────────────┬─────────────┬─────────────────┬────────────────────────────────────┐"
    );
    println!(
        "│ Stream           │ Direction   │ Pair            │ Purpose                            │"
    );
    println!(
        "├──────────────────┼─────────────┼─────────────────┼────────────────────────────────────┤"
    );

    for stream in &streams {
        let dir = if stream.is_forward() {
            "Forward"
        } else {
            "Backward"
        };
        let purpose = match stream {
            StreamId::MajorForward => "Causal along major radius",
            StreamId::MajorBackward => "Anti-causal along major radius",
            StreamId::MinorForward => "Causal along minor radius",
            StreamId::MinorBackward => "Anti-causal along minor radius",
            StreamId::SpiralCW => "Clockwise golden spiral",
            StreamId::SpiralCCW => "Counter-clockwise spiral",
            StreamId::CrossUtoV => "Cross-attention U→V",
            StreamId::CrossVtoU => "Cross-attention V→U",
        };
        println!(
            "│ {:16} │ {:11} │ {:15} │ {:34} │",
            stream.name(),
            dir,
            stream.pair().name(),
            purpose
        );
    }
    println!(
        "└──────────────────┴─────────────┴─────────────────┴────────────────────────────────────┘"
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 2. Direction Properties
    // ─────────────────────────────────────────────────────────────────────
    println!("── Flow Directions ──\n");

    let forward = FlowDirection::Forward;
    let backward = FlowDirection::Backward;

    println!("Forward direction:");
    println!("  Sign:    {}", forward.sign());
    println!("  Is Forward: {}", forward.is_forward());
    println!();

    println!("Backward direction:");
    println!("  Sign:    {}", backward.sign());
    println!("  Is Forward: {}", backward.is_forward());
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Stream Pairing
    // ─────────────────────────────────────────────────────────────────────
    println!("── Stream Pairing ──\n");

    println!("Each forward stream is paired with a backward stream:");
    println!();

    let forward_streams: Vec<_> = streams.iter().filter(|s| s.is_forward()).collect();

    for &stream in &forward_streams {
        let pair = stream.pair();
        println!("  {} ←→ {}", stream.name(), pair.name());

        // Verify involution: pair(pair(x)) == x
        assert_eq!(pair.pair(), *stream, "Pairing should be symmetric");
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Index Mapping
    // ─────────────────────────────────────────────────────────────────────
    println!("── Stream Index Mapping ──\n");

    println!("Stream indices (for tensor indexing):");
    for stream in &streams {
        println!("  {} → index {}", stream.name(), stream.index());
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Attention Pattern Visualization
    // ─────────────────────────────────────────────────────────────────────
    println!("── Attention Pattern Visualization ──\n");

    let n = 8; // 8x8 grid for visualization

    // Causal mask (forward)
    println!("Forward (Causal) Mask:");
    println!("  Query can attend to positions ≤ its position");
    println!();
    print!("     ");
    for j in 0..n {
        print!("{:>2} ", j);
    }
    println!("  (Key position)");
    for i in 0..n {
        print!("  {:>2} ", i);
        for j in 0..n {
            if j <= i {
                print!(" █ ");
            } else {
                print!(" · ");
            }
        }
        if i == n / 2 {
            println!(" ← Query position");
        } else {
            println!();
        }
    }
    println!();

    // Anti-causal mask (backward)
    println!("Backward (Anti-Causal) Mask:");
    println!("  Query can attend to positions ≥ its position");
    println!();
    print!("     ");
    for j in 0..n {
        print!("{:>2} ", j);
    }
    println!("  (Key position)");
    for i in 0..n {
        print!("  {:>2} ", i);
        for j in 0..n {
            if j >= i {
                print!(" █ ");
            } else {
                print!(" · ");
            }
        }
        if i == n / 2 {
            println!(" ← Query position");
        } else {
            println!();
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Spiral Pattern
    // ─────────────────────────────────────────────────────────────────────
    println!("── Golden Spiral Pattern ──\n");

    let phi = 1.618033988749895; // Golden ratio

    println!("Golden ratio (φ): {:.10}", phi);
    println!("The spiral winding number determines how quickly the spiral");
    println!("moves from the major to minor direction.");
    println!();

    // Visualize spiral traversal on a grid
    let grid_size = 8;
    let boundary = PeriodicBoundary::new(grid_size, grid_size);

    println!("Spiral traversal order (CW, first 16 positions):");
    print!("  ");
    for step in 0..16 {
        let u = step as f64 * 0.5; // Simplified spiral
        let v = step as f64 * 0.5 * phi;
        let (i, j) = boundary.wrap_2d(
            (u * grid_size as f64 / (2.0 * std::f64::consts::PI)) as i64,
            (v * grid_size as f64 / (2.0 * std::f64::consts::PI)) as i64,
        );
        print!("({},{}) ", i, j);
        if (step + 1) % 8 == 0 {
            print!("\n  ");
        }
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 7. Cross-Attention Pattern
    // ─────────────────────────────────────────────────────────────────────
    println!("── Cross-Attention (U↔V) ──\n");

    println!("Cross-attention streams allow information to flow between");
    println!("the major (U) and minor (V) directions of the torus.");
    println!();

    println!("U→V: Information flows from major to minor circle");
    println!("V→U: Information flows from minor to major circle");
    println!();

    // Visualize 2D grid structure
    let n_major = 4;
    let n_minor = 4;

    println!("2D grid structure ({}x{}):", n_major, n_minor);
    println!();
    println!("           Minor axis (V) →");
    println!("         ┌───┬───┬───┬───┐");
    for i in 0..n_major {
        if i == n_major / 2 {
            print!("  Major  │");
        } else if i == n_major / 2 + 1 {
            print!("  (U) ↓  │");
        } else {
            print!("         │");
        }
        for j in 0..n_minor {
            let idx = i * n_minor + j;
            print!("{:>2} │", idx);
        }
        println!();
        if i < n_major - 1 {
            println!("         ├───┼───┼───┼───┤");
        }
    }
    println!("         └───┴───┴───┴───┘");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 8. Combined Coverage
    // ─────────────────────────────────────────────────────────────────────
    println!("── Combined Bidirectional Coverage ──\n");

    println!("When forward and backward streams are combined, every position");
    println!("can attend to every other position through at least one stream:");
    println!();

    print!("     ");
    for j in 0..n {
        print!("{:>2} ", j);
    }
    println!("  (Key position)");

    for i in 0..n {
        print!("  {:>2} ", i);
        for _ in 0..n {
            // Combined forward + backward covers everything
            print!(" █ ");
        }
        if i == n / 2 {
            println!(" ← Full coverage");
        } else {
            println!();
        }
    }

    println!("\n═══ Stream Analysis Complete ═══");
}
