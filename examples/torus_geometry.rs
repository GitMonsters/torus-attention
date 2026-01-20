//! Torus Geometry Exploration Example
//!
//! This example demonstrates the geometric foundations of Torus Attention,
//! including coordinate systems, geodesic distances, and periodic boundaries.
//!
//! Run with: cargo run --example torus_geometry

use std::f64::consts::PI;
use torus_attention::{PeriodicBoundary, TorusCoordinate, TorusDistanceMatrix, TorusManifold};

fn main() {
    println!("═══ Torus Geometry Exploration ═══\n");

    // ─────────────────────────────────────────────────────────────────────
    // 1. Torus Coordinates
    // ─────────────────────────────────────────────────────────────────────
    println!("── Torus Coordinates ──\n");

    // Create coordinates on the torus (u, v in [0, 2π))
    let origin = TorusCoordinate::new(0.0, 0.0);
    let quarter = TorusCoordinate::new(PI / 2.0, PI / 2.0);
    let halfway = TorusCoordinate::new(PI, PI);

    println!("Origin:    u={:.4}, v={:.4}", origin.u, origin.v);
    println!("Quarter:   u={:.4}, v={:.4}", quarter.u, quarter.v);
    println!("Halfway:   u={:.4}, v={:.4}", halfway.u, halfway.v);

    // Coordinates wrap automatically
    let wrapped = TorusCoordinate::new(3.0 * PI, -PI);
    println!("Wrapped (3π, -π): u={:.4}, v={:.4}", wrapped.u, wrapped.v);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 2. Geodesic Distances
    // ─────────────────────────────────────────────────────────────────────
    println!("── Geodesic Distances ──\n");

    // Distance from origin to other points
    let d_to_quarter = origin.geodesic_distance(&quarter);
    let d_to_halfway = origin.geodesic_distance(&halfway);

    println!("Distance origin → quarter:  {:.4}", d_to_quarter);
    println!("Distance origin → halfway:  {:.4}", d_to_halfway);

    // Distance is symmetric
    let d_symmetric = halfway.geodesic_distance(&origin);
    println!("Distance halfway → origin:  {:.4} (symmetric)", d_symmetric);

    // Distance to self is zero
    let d_self = origin.geodesic_distance(&origin);
    println!("Distance origin → origin:   {:.4}", d_self);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Cartesian Embedding
    // ─────────────────────────────────────────────────────────────────────
    println!("── Cartesian Embedding ──\n");

    let major_radius = 2.0; // R: distance from torus center to tube center
    let minor_radius = 1.0; // r: radius of the tube

    let (x1, y1, z1) = origin.to_cartesian(major_radius, minor_radius);
    let (x2, y2, z2) = quarter.to_cartesian(major_radius, minor_radius);
    let (x3, y3, z3) = halfway.to_cartesian(major_radius, minor_radius);

    println!("Origin  → ({:.4}, {:.4}, {:.4})", x1, y1, z1);
    println!("Quarter → ({:.4}, {:.4}, {:.4})", x2, y2, z2);
    println!("Halfway → ({:.4}, {:.4}, {:.4})", x3, y3, z3);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Torus Manifold
    // ─────────────────────────────────────────────────────────────────────
    println!("── Torus Manifold ──\n");

    let torus = TorusManifold::new(major_radius, minor_radius);

    println!("Major radius (R): {}", torus.major_radius);
    println!("Minor radius (r): {}", torus.minor_radius);
    println!(
        "Aspect ratio:     {:.4}",
        torus.major_radius / torus.minor_radius
    );
    println!("Surface area:     {:.4}", torus.surface_area());
    println!("Volume:           {:.4}", torus.volume());
    println!();

    // Generate a grid of points
    let n_major = 8;
    let n_minor = 4;
    let grid = torus.generate_grid(n_major, n_minor);

    println!("Grid: {}x{} = {} points", n_major, n_minor, grid.len());
    println!("First 4 points:");
    for (i, coord) in grid.iter().take(4).enumerate() {
        println!("  {}: u={:.4}, v={:.4}", i, coord.u, coord.v);
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Distance Matrix
    // ─────────────────────────────────────────────────────────────────────
    println!("── Distance Matrix ──\n");

    let small_grid = torus.generate_grid(4, 4);
    let dist_matrix = TorusDistanceMatrix::from_coordinates(&small_grid);

    println!("Distance matrix (4x4 grid, showing first 4x4 block):");
    print!("       ");
    for j in 0..4 {
        print!("{:>6} ", j);
    }
    println!();
    for i in 0..4 {
        print!("{:>4}:  ", i);
        for j in 0..4 {
            print!("{:>6.3} ", dist_matrix.distances[i][j]);
        }
        println!();
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Periodic Boundaries
    // ─────────────────────────────────────────────────────────────────────
    println!("── Periodic Boundaries ──\n");

    let boundary = PeriodicBoundary::new(8, 4);

    println!("Grid: {}x{}", boundary.n_major, boundary.n_minor);
    println!("Total positions: {}", boundary.n_major * boundary.n_minor);
    println!();

    // Demonstrate wrapping
    println!("Index wrapping examples:");
    let test_indices = [
        (3, 2),   // Normal
        (-1, 0),  // Negative major
        (0, -1),  // Negative minor
        (10, 5),  // Over major
        (-9, -5), // Both negative
    ];

    for (i, j) in test_indices.iter() {
        let (wi, wj) = boundary.wrap_2d(*i as i64, *j as i64);
        println!("  ({:3}, {:2}) → ({}, {})", i, j, wi, wj);
    }
    println!();

    // Gaussian kernel for periodic smoothing
    let kernel = boundary.gaussian_kernel(1.0, 3);
    println!("Gaussian kernel (σ=1.0, size=3x3):");
    for i in 0..kernel.nrows() {
        print!("  ");
        for j in 0..kernel.ncols() {
            print!("{:.4} ", kernel[[i, j]]);
        }
        println!();
    }
    println!("  Sum: {:.6}", kernel.iter().sum::<f64>());

    println!("\n═══ Geometry Exploration Complete ═══");
}
