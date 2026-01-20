//! Demonstrates that network tools are blocked in worm_std.

use worm_std::process::{Command, ProcessError};

fn main() {
    println!("Worm Rust Network Blocking Demo");
    println!("================================");
    println!();
    
    // List of commands to test
    let test_commands = [
        ("curl", vec!["--version"]),
        ("wget", vec!["--version"]),
        ("nc", vec!["-h"]),
        ("ssh", vec!["-V"]),
        ("ping", vec!["-c", "1", "localhost"]),
        ("nmap", vec!["--version"]),
        ("dig", vec!["localhost"]),
    ];
    
    println!("Testing blocked commands:");
    println!();
    
    for (cmd, args) in &test_commands {
        let result = Command::new(cmd)
            .args(args.iter())
            .output();
        
        match result {
            Ok(_) => {
                println!("  {} - ALLOWED (unexpected!)", cmd);
            }
            Err(ProcessError::BlockedExecutable(name)) => {
                println!("  {} - BLOCKED (expected)", name);
            }
            Err(e) => {
                println!("  {} - ERROR: {}", cmd, e);
            }
        }
    }
    
    println!();
    println!("Testing allowed commands:");
    println!();
    
    let allowed_commands = [
        ("ls", vec!["--version"]),
        ("cat", vec!["--version"]),
        ("echo", vec!["hello"]),
    ];
    
    for (cmd, args) in &allowed_commands {
        let result = Command::new(cmd)
            .args(args.iter())
            .output();
        
        match result {
            Ok(output) => {
                let status = if output.status.success() { "success" } else { "failed" };
                println!("  {} - ALLOWED ({})", cmd, status);
            }
            Err(ProcessError::BlockedExecutable(name)) => {
                println!("  {} - BLOCKED (unexpected!)", name);
            }
            Err(e) => {
                println!("  {} - ERROR: {} (may not be installed)", cmd, e);
            }
        }
    }
    
    println!();
    println!("Network isolation demo complete!");
    println!();
    println!("Security note: Even if curl/wget exist on this system,");
    println!("Worm Rust prevents their execution at runtime.");
}
