//! # Worm Std
//!
//! A security-hardened Rust standard library that provides compile-time
//! network isolation. This crate re-exports all of `std` EXCEPT the `net`
//! module, preventing any direct network access.
//!
//! ## Usage
//!
//! Replace your `use std::*` imports with `use worm_std::*`:
//!
//! ```rust
//! // Instead of:
//! // use std::fs::File;
//! // use std::io::Read;
//!
//! // Use:
//! use worm_std::fs::File;
//! use worm_std::io::Read;
//! ```
//!
//! ## Security Guarantees
//!
//! - **No `std::net`**: The network module is not re-exported. Any attempt
//!   to use `TcpStream`, `UdpSocket`, etc. will fail at compile time.
//! - **Filtered `process::Command`**: Blocks dangerous executables like
//!   `curl`, `wget`, `ssh`, `nc`, etc.
//!
//! ## Limitations
//!
//! This crate cannot prevent:
//! - `unsafe` code from making syscalls directly
//! - FFI calls to C libraries with network access
//! - Covert channels or timing attacks
//!
//! For complete isolation, use in combination with OS-level sandboxing.

#![warn(missing_docs)]

// ============================================================================
// RE-EXPORTS FROM STD (excluding net)
// ============================================================================

// Core modules
pub use std::alloc;
pub use std::any;
pub use std::array;
pub use std::ascii;
pub use std::borrow;
pub use std::boxed;
pub use std::cell;
pub use std::char;
pub use std::clone;
pub use std::cmp;
pub use std::collections;
pub use std::convert;
pub use std::default;
pub use std::env;
pub use std::error;
pub use std::f32;
pub use std::f64;
pub use std::ffi;
pub use std::fmt;
pub use std::fs;
pub use std::future;
pub use std::hash;
pub use std::hint;
pub use std::i8;
pub use std::i16;
pub use std::i32;
pub use std::i64;
pub use std::i128;
pub use std::io;
pub use std::isize;
pub use std::iter;
pub use std::marker;
pub use std::mem;
pub use std::num;
pub use std::ops;
pub use std::option;
pub use std::os;
pub use std::panic;
pub use std::path;
pub use std::pin;
pub use std::ptr;
pub use std::rc;
pub use std::result;
pub use std::slice;
pub use std::str;
pub use std::string;
pub use std::sync;
pub use std::task;
pub use std::thread;
pub use std::time;
pub use std::u8;
pub use std::u16;
pub use std::u32;
pub use std::u64;
pub use std::u128;
pub use std::usize;
pub use std::vec;

// NOTE: std::net is INTENTIONALLY NOT RE-EXPORTED
// This provides compile-time network isolation

// ============================================================================
// CUSTOM MODULES
// ============================================================================

/// Filtered process module that blocks dangerous executables.
pub mod process;

// ============================================================================
// PRELUDE
// ============================================================================

/// A prelude module for convenient imports.
///
/// ```rust
/// use worm_std::prelude::*;
/// ```
pub mod prelude {
    pub use std::prelude::rust_2021::*;
    
    // Re-export common items
    pub use super::process::Command;
}

// ============================================================================
// VERSION INFO
// ============================================================================

/// Returns the worm_std version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns information about the security features.
pub fn security_info() -> &'static str {
    concat!(
        "Worm Std v", env!("CARGO_PKG_VERSION"), "\n",
        "Security features:\n",
        "  - std::net: BLOCKED (compile-time)\n",
        "  - Dangerous executables: BLOCKED (runtime)\n",
        "  - Blocked commands: curl, wget, nc, ssh, nmap, etc.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
    
    #[test]
    fn test_security_info() {
        let info = security_info();
        assert!(info.contains("BLOCKED"));
        assert!(info.contains("curl"));
    }
    
    #[test]
    fn test_fs_works() {
        // Verify filesystem access still works
        use super::fs;
        let _ = fs::metadata(".");
    }
    
    #[test]
    fn test_collections_work() {
        // Verify collections still work
        use super::collections::HashMap;
        let mut map: HashMap<&str, i32> = HashMap::new();
        map.insert("test", 42);
        assert_eq!(map.get("test"), Some(&42));
    }
}
