# Worm Rust

A security-hardened Rust distribution designed for safe execution of ML workloads. Provides compile-time network isolation and runtime subprocess filtering.

## Features

- **Compile-time Network Isolation**: `std::net` is not re-exported
- **Runtime Process Filtering**: Blocks `curl`, `wget`, `nc`, `ssh`, etc.
- **Zero Runtime Overhead**: Security enforced at compile time where possible
- **Drop-in Replacement**: Use `worm_std` instead of `std`

## Quick Start

```rust
// Replace std imports with worm_std
use worm_std::fs::File;
use worm_std::io::Read;
use worm_std::process::Command;

fn main() {
    // File I/O works normally
    let content = worm_std::fs::read_to_string("data.txt").unwrap();
    
    // Safe commands work
    let output = Command::new("ls")
        .arg("-la")
        .output()
        .expect("ls failed");
    
    // Network commands are blocked
    let result = Command::new("curl")
        .arg("https://evil.com")
        .output();
    assert!(result.is_err()); // Blocked!
}
```

## Security Model

| Threat | Protection | Method |
|--------|------------|--------|
| Direct network (`TcpStream`) | Blocked | Compile-time (no `std::net`) |
| Subprocess network (`curl`) | Blocked | Runtime filtering |
| DNS exfiltration | Blocked | `dig`/`nslookup` filtered |
| Reverse shells | Blocked | `nc`/`ssh` filtered |

## Structure

```
worm-rust/
├── docs/
│   └── WORM_RUST_DESIGN.md
├── worm_std/           # The core library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs      # Re-exports std (minus net)
│       └── process.rs  # Filtered Command
└── examples/
    ├── hello_worm/     # Basic usage
    ├── file_processing/# Safe file I/O
    └── network_blocked/# Security demo
```

## Integration with Torus Attention

This Worm Rust implementation is bundled with the Torus Attention library to ensure ML models can be trained and run securely without risk of data exfiltration.

## Limitations

Worm Rust cannot prevent:
- `unsafe` blocks making direct syscalls
- FFI calls to C libraries
- Covert channels/timing attacks

For complete isolation, combine with OS-level sandboxing (Docker, gVisor, seccomp).

## License

MIT
