//! Quick GPU test for AMD Radeon via Vulkan/WGPU
//!
//! Run with: cargo run --example gpu_test --release --features amd-gpu

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("           AMD GPU TEST - Burn WGPU/Vulkan Backend");
    println!("═══════════════════════════════════════════════════════════════\n");

    #[cfg(feature = "amd-gpu")]
    {
        use torus_attention::backend::{AmdGpuBackend, TensorBackend};
        
        println!("Backend: {}", AmdGpuBackend::backend_name());
        println!("GPU Mode: {}\n", if AmdGpuBackend::is_gpu() { "YES" } else { "NO" });
        
        let device = AmdGpuBackend::default_device();
        println!("Device: {:?}\n", device);
        
        println!("Running tensor operations on GPU...\n");
        
        // Test 1: Create tensors
        let start = std::time::Instant::now();
        let a = AmdGpuBackend::randn(&[1024, 1024], &device).expect("Failed to create tensor A");
        let b = AmdGpuBackend::randn(&[1024, 1024], &device).expect("Failed to create tensor B");
        println!("✓ Created 1024x1024 tensors: {:?}", start.elapsed());
        
        // Test 2: Matrix multiplication
        let start = std::time::Instant::now();
        let c = AmdGpuBackend::matmul(&a, &b).expect("Failed matmul");
        println!("✓ Matrix multiply (1024x1024): {:?}", start.elapsed());
        
        // Test 3: Element-wise ops
        let start = std::time::Instant::now();
        let d = AmdGpuBackend::add(&a, &b).expect("Failed add");
        let e = AmdGpuBackend::mul(&d, &c).expect("Failed mul");
        let f = AmdGpuBackend::exp(&e).expect("Failed exp");
        println!("✓ Element-wise ops (add, mul, exp): {:?}", start.elapsed());
        
        // Test 4: Softmax
        let start = std::time::Instant::now();
        let g = AmdGpuBackend::softmax(&f, -1).expect("Failed softmax");
        println!("✓ Softmax: {:?}", start.elapsed());
        
        // Test 5: Larger matmul benchmark
        println!("\nBenchmarking larger matrices...");
        let a_big = AmdGpuBackend::randn(&[2048, 2048], &device).expect("Failed big A");
        let b_big = AmdGpuBackend::randn(&[2048, 2048], &device).expect("Failed big B");
        
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = AmdGpuBackend::matmul(&a_big, &b_big).expect("Failed big matmul");
        }
        let elapsed = start.elapsed();
        println!("✓ 10x matmul (2048x2048): {:?} ({:.2?} per op)", elapsed, elapsed / 10);
        
        // Verify output shape
        println!("\nOutput shape: {:?}", AmdGpuBackend::shape(&g));
        
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("                    GPU TEST PASSED!");
        println!("═══════════════════════════════════════════════════════════════");
    }
    
    #[cfg(not(feature = "amd-gpu"))]
    {
        println!("ERROR: amd-gpu feature not enabled!");
        println!("Run with: cargo run --example gpu_test --release --features amd-gpu");
    }
}
