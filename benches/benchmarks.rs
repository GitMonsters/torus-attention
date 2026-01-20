//! Performance Benchmarks for Torus Attention
//!
//! Run with: cargo bench
//!
//! Benchmarks cover:
//! - Geometry operations (coordinate creation, geodesic distance)
//! - Tensor operations (attention, softmax)
//! - Full forward passes (single layer, full transformer)
//! - Training step performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use torus_attention::{
    TorusCoordinate, TorusManifold, TorusDistanceMatrix,
    PeriodicBoundary,
    BidirectionalTorusConfig,
};
use candle_core::{Device, Tensor, DType};

// ═══════════════════════════════════════════════════════════════════════════
// GEOMETRY BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_coordinate_creation(c: &mut Criterion) {
    c.bench_function("coordinate_creation", |b| {
        b.iter(|| {
            let _coord = TorusCoordinate::new(black_box(1.5), black_box(2.3));
        })
    });
}

fn bench_geodesic_distance(c: &mut Criterion) {
    let c1 = TorusCoordinate::new(0.5, 0.3);
    let c2 = TorusCoordinate::new(1.2, 0.8);
    
    c.bench_function("geodesic_distance", |b| {
        b.iter(|| {
            c1.geodesic_distance(black_box(&c2))
        })
    });
}

fn bench_cartesian_conversion(c: &mut Criterion) {
    let coord = TorusCoordinate::new(1.0, 0.5);
    
    c.bench_function("cartesian_conversion", |b| {
        b.iter(|| {
            coord.to_cartesian(black_box(2.0), black_box(1.0))
        })
    });
}

fn bench_grid_generation(c: &mut Criterion) {
    let torus = TorusManifold::new(2.0, 1.0);
    
    let mut group = c.benchmark_group("grid_generation");
    for size in [8, 16, 32, 64].iter() {
        group.throughput(Throughput::Elements((*size * *size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                torus.generate_grid(black_box(size), black_box(size))
            })
        });
    }
    group.finish();
}

fn bench_distance_matrix(c: &mut Criterion) {
    let torus = TorusManifold::new(2.0, 1.0);
    
    let mut group = c.benchmark_group("distance_matrix");
    for size in [8, 16, 32].iter() {
        let coords = torus.generate_grid(*size, *size);
        let n = coords.len();
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &coords, |b, coords| {
            b.iter(|| {
                TorusDistanceMatrix::from_coordinates(black_box(coords))
            })
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// PERIODIC BOUNDARY BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_wrap_2d(c: &mut Criterion) {
    let boundary = PeriodicBoundary::new(32, 16);
    
    c.bench_function("wrap_2d", |b| {
        b.iter(|| {
            boundary.wrap_2d(black_box(100), black_box(-50))
        })
    });
}

fn bench_gaussian_kernel(c: &mut Criterion) {
    let boundary = PeriodicBoundary::new(32, 16);
    
    let mut group = c.benchmark_group("gaussian_kernel");
    for radius in [3, 5, 7, 9].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(radius), radius, |b, &radius| {
            b.iter(|| {
                boundary.gaussian_kernel(black_box(1.0), black_box(radius))
            })
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR OPERATION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_tensor_creation(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("tensor_creation");
    for (batch, seq, dim) in [(1, 32, 64), (4, 64, 128), (8, 128, 256)].iter() {
        let name = format!("{}x{}x{}", batch, seq, dim);
        group.throughput(Throughput::Elements((*batch * *seq * *dim) as u64));
        group.bench_function(&name, |b| {
            b.iter(|| {
                Tensor::randn(0.0f32, 1.0, (*batch, *seq, *dim), &device).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("softmax");
    for seq_len in [32, 64, 128, 256, 512].iter() {
        let tensor = Tensor::randn(0.0f32, 1.0, (4, 8, *seq_len, *seq_len), &device).unwrap();
        group.throughput(Throughput::Elements((4 * 8 * seq_len * seq_len) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &tensor, |b, tensor| {
            b.iter(|| {
                candle_nn::ops::softmax(black_box(tensor), candle_core::D::Minus1).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("matmul");
    for size in [32, 64, 128, 256].iter() {
        let a = Tensor::randn(0.0f32, 1.0, (4, 8, *size, 64), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (4, 8, 64, *size), &device).unwrap();
        let ops = 4 * 8 * size * 64 * size; // Approximate FLOPs
        group.throughput(Throughput::Elements(ops as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &(a.clone(), b.clone()), |bench, (a, b)| {
            bench.iter(|| {
                a.matmul(black_box(b)).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("layer_norm_manual");
    for (batch, seq, dim) in [(4, 64, 128), (8, 128, 256), (16, 256, 512)].iter() {
        let tensor = Tensor::randn(0.0f32, 1.0, (*batch, *seq, *dim), &device).unwrap();
        let name = format!("{}x{}x{}", batch, seq, dim);
        group.throughput(Throughput::Elements((*batch * *seq * *dim) as u64));
        group.bench_function(&name, |b| {
            b.iter(|| {
                // Manual layer norm computation
                let mean = tensor.mean_keepdim(2).unwrap();
                let centered = tensor.broadcast_sub(&mean).unwrap();
                let var = centered.sqr().unwrap().mean_keepdim(2).unwrap();
                let std = (var + 1e-5f64).unwrap().sqrt().unwrap();
                centered.broadcast_div(&std).unwrap()
            })
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_attention_scores(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("attention_scores");
    for seq_len in [32, 64, 128, 256].iter() {
        let batch = 4;
        let n_heads = 8;
        let head_dim = 32;
        
        let q = Tensor::randn(0.0f32, 1.0, (batch, n_heads, *seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, n_heads, *seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch, n_heads, *seq_len, head_dim), &device).unwrap();
        let scale = (head_dim as f64).sqrt();
        
        group.throughput(Throughput::Elements((batch * n_heads * seq_len * seq_len) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &(q.clone(), k.clone(), v.clone()),
            |b, (q, k, v)| {
                b.iter(|| {
                    // Q @ K^T
                    let k_t = k.transpose(2, 3).unwrap();
                    let scores = q.matmul(&k_t).unwrap();
                    let scores = (scores / scale).unwrap();
                    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).unwrap();
                    attn.matmul(black_box(v)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_multihead_reshape(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("multihead_reshape");
    for seq_len in [64, 128, 256, 512].iter() {
        let batch = 4;
        let d_model = 256;
        let n_heads = 8;
        let head_dim = d_model / n_heads;
        
        let x = Tensor::randn(0.0f32, 1.0, (batch, *seq_len, d_model), &device).unwrap();
        
        group.throughput(Throughput::Elements((batch * seq_len * d_model) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &x, |b, x| {
            b.iter(|| {
                x.reshape((batch, *seq_len, n_heads, head_dim))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            })
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// STREAM PROCESSING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_stream_combination(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("stream_combination");
    for seq_len in [64, 128, 256].iter() {
        let batch = 4;
        let d_model = 128;
        let n_streams = 8;
        
        // Create 8 stream outputs
        let streams: Vec<Tensor> = (0..n_streams)
            .map(|_| Tensor::randn(0.0f32, 1.0, (batch, *seq_len, d_model), &device).unwrap())
            .collect();
        
        // Create weights
        let weights = Tensor::randn(0.0f32, 1.0, (n_streams,), &device).unwrap();
        let weights = candle_nn::ops::softmax(&weights, 0).unwrap();
        let weights_vec: Vec<f32> = weights.to_vec1().unwrap();
        
        group.throughput(Throughput::Elements((batch * seq_len * d_model * n_streams) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &(streams.clone(), weights_vec.clone()),
            |b, (streams, weights_vec)| {
                b.iter(|| {
                    let mut combined = Tensor::zeros((batch, *seq_len, d_model), DType::F32, &device).unwrap();
                    for (stream, &w) in streams.iter().zip(weights_vec.iter()) {
                        combined = (combined + (stream * w as f64).unwrap()).unwrap();
                    }
                    combined
                })
            },
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_config_serialization(c: &mut Criterion) {
    let config = BidirectionalTorusConfig::default();
    
    c.bench_function("config_to_json", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&config)).unwrap()
        })
    });
}

fn bench_config_deserialization(c: &mut Criterion) {
    let config = BidirectionalTorusConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    
    c.bench_function("config_from_json", |b| {
        b.iter(|| {
            serde_json::from_str::<BidirectionalTorusConfig>(black_box(&json)).unwrap()
        })
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOUNDING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_ema_computation(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("ema_computation");
    for seq_len in [64, 128, 256, 512].iter() {
        let batch = 4;
        let d_model = 128;
        
        let current = Tensor::randn(0.0f32, 1.0, (batch, *seq_len, d_model), &device).unwrap();
        let previous = Tensor::randn(0.0f32, 1.0, (batch, *seq_len, d_model), &device).unwrap();
        let alpha = 0.9f64;
        
        group.throughput(Throughput::Elements((batch * seq_len * d_model) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &(current.clone(), previous.clone()),
            |b, (current, previous)| {
                b.iter(|| {
                    // EMA: alpha * current + (1 - alpha) * previous
                    let scaled_current = (current * alpha).unwrap();
                    let scaled_previous = (previous * (1.0 - alpha)).unwrap();
                    (scaled_current + scaled_previous).unwrap()
                })
            },
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// CRITERION GROUPS
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    geometry_benches,
    bench_coordinate_creation,
    bench_geodesic_distance,
    bench_cartesian_conversion,
    bench_grid_generation,
    bench_distance_matrix,
);

criterion_group!(
    periodic_benches,
    bench_wrap_2d,
    bench_gaussian_kernel,
);

criterion_group!(
    tensor_benches,
    bench_tensor_creation,
    bench_softmax,
    bench_matmul,
    bench_layer_norm,
);

criterion_group!(
    attention_benches,
    bench_attention_scores,
    bench_multihead_reshape,
);

criterion_group!(
    stream_benches,
    bench_stream_combination,
);

criterion_group!(
    config_benches,
    bench_config_serialization,
    bench_config_deserialization,
);

criterion_group!(
    compounding_benches,
    bench_ema_computation,
);

criterion_main!(
    geometry_benches,
    periodic_benches,
    tensor_benches,
    attention_benches,
    stream_benches,
    config_benches,
    compounding_benches,
);
