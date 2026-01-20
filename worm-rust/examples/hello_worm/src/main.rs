//! Simple hello world example using worm_std.

use worm_std::prelude::*;
use worm_std::io::Write;

fn main() {
    println!("Hello from Worm Rust!");
    println!();
    println!("{}", worm_std::security_info());
    
    // Demonstrate that basic I/O works
    let mut stdout = worm_std::io::stdout();
    writeln!(stdout, "Standard I/O: Working!").unwrap();
    
    // Demonstrate that filesystem works
    let current_dir = worm_std::env::current_dir().unwrap();
    println!("Current directory: {}", current_dir.display());
    
    // Demonstrate that safe commands work
    let output = Command::new("echo")
        .arg("Command execution: Working!")
        .output()
        .expect("failed to execute echo");
    
    print!("{}", String::from_utf8_lossy(&output.stdout));
    
    println!();
    println!("Worm Rust is ready for secure ML workloads!");
}
