//! Filtered process module that blocks dangerous executables.
//!
//! This module provides a drop-in replacement for `std::process` that
//! blocks execution of network-related and data exfiltration tools.
//!
//! # Blocked Executables
//!
//! The following categories of executables are blocked:
//!
//! - **Network transfer**: `curl`, `wget`, `aria2c`
//! - **Netcat variants**: `nc`, `netcat`, `ncat`, `socat`
//! - **Remote access**: `ssh`, `scp`, `sftp`, `rsync`, `ftp`, `telnet`
//! - **Network diagnostics**: `ping`, `traceroute`, `tracepath`, `mtr`
//! - **DNS tools**: `dig`, `nslookup`, `host`, `whois`
//! - **Network scanners**: `nmap`, `masscan`, `zmap`
//!
//! # Example
//!
//! ```rust
//! use worm_std::process::Command;
//!
//! // This works fine:
//! let output = Command::new("ls")
//!     .arg("-la")
//!     .output()
//!     .expect("failed to list directory");
//!
//! // This returns an error:
//! let result = Command::new("curl")
//!     .arg("https://example.com")
//!     .output();
//! assert!(result.is_err());
//! ```

use std::ffi::{OsStr, OsString};
use std::io;
use std::path::Path;
use thiserror::Error;

// Re-export types that don't need filtering
pub use std::process::{
    Child, ChildStderr, ChildStdin, ChildStdout, ExitCode, ExitStatus, Output, Stdio, Termination,
};

/// Error returned when attempting to execute a blocked command.
#[derive(Debug, Error)]
pub enum ProcessError {
    /// The executable is blocked for security reasons.
    #[error("Blocked executable: '{0}' is not allowed in Worm Rust")]
    BlockedExecutable(String),
    
    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

/// List of executables that are blocked for security reasons.
const BLOCKED_EXECUTABLES: &[&str] = &[
    // Network transfer tools
    "curl",
    "wget",
    "aria2c",
    "axel",
    "httpie",
    "http",  // httpie alias
    
    // Netcat and variants
    "nc",
    "netcat",
    "ncat",
    "socat",
    "cryptcat",
    
    // Remote access
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "ftp",
    "lftp",
    "telnet",
    "rsh",
    "rlogin",
    
    // Network diagnostics
    "ping",
    "ping6",
    "traceroute",
    "traceroute6",
    "tracepath",
    "mtr",
    
    // DNS tools
    "dig",
    "nslookup",
    "host",
    "whois",
    "drill",
    
    // Network scanners
    "nmap",
    "masscan",
    "zmap",
    "unicornscan",
    
    // Packet tools
    "tcpdump",
    "wireshark",
    "tshark",
    "ettercap",
    
    // Reverse shell helpers
    "bash",  // Can be used for reverse shells - CONTROVERSIAL
    "sh",    // Can be used for reverse shells - CONTROVERSIAL
    "python",
    "python3",
    "perl",
    "ruby",
    "php",
    "node",
    "lua",
];

/// Additional executables blocked only when `allow_package_managers` is disabled.
#[cfg(not(feature = "allow_package_managers"))]
const BLOCKED_PACKAGE_MANAGERS: &[&str] = &[
    "cargo",
    "pip",
    "pip3",
    "npm",
    "yarn",
    "pnpm",
    "gem",
    "composer",
    "go",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "pacman",
    "brew",
];

#[cfg(feature = "allow_package_managers")]
const BLOCKED_PACKAGE_MANAGERS: &[&str] = &[];

/// Check if an executable name is blocked.
fn is_blocked(program: &OsStr) -> bool {
    let program_str = program.to_string_lossy();
    
    // Extract just the filename if it's a path
    let exe_name = Path::new(program_str.as_ref())
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or(program_str);
    
    let exe_lower = exe_name.to_lowercase();
    
    BLOCKED_EXECUTABLES.iter().any(|&blocked| exe_lower == blocked)
        || BLOCKED_PACKAGE_MANAGERS.iter().any(|&blocked| exe_lower == blocked)
}

/// A filtered process builder, providing fine-grained control over how
/// a new process should be spawned.
///
/// This is a security-filtered wrapper around `std::process::Command`
/// that blocks dangerous executables.
#[derive(Debug)]
pub struct Command {
    inner: std::process::Command,
    program: OsString,
}

impl Command {
    /// Constructs a new `Command` for launching the program at path `program`.
    ///
    /// # Security
    ///
    /// The program name is checked against a blocklist. If blocked, all
    /// execution methods (`spawn`, `output`, `status`) will return an error.
    pub fn new<S: AsRef<OsStr>>(program: S) -> Command {
        Command {
            inner: std::process::Command::new(&program),
            program: program.as_ref().to_owned(),
        }
    }
    
    /// Adds an argument to pass to the program.
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Command {
        self.inner.arg(arg);
        self
    }
    
    /// Adds multiple arguments to pass to the program.
    pub fn args<I, S>(&mut self, args: I) -> &mut Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.inner.args(args);
        self
    }
    
    /// Inserts or updates an environment variable mapping.
    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Command
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.inner.env(key, val);
        self
    }
    
    /// Inserts or updates multiple environment variable mappings.
    pub fn envs<I, K, V>(&mut self, vars: I) -> &mut Command
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.inner.envs(vars);
        self
    }
    
    /// Removes an environment variable mapping.
    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Command {
        self.inner.env_remove(key);
        self
    }
    
    /// Clears the entire environment map for the child process.
    pub fn env_clear(&mut self) -> &mut Command {
        self.inner.env_clear();
        self
    }
    
    /// Sets the working directory for the child process.
    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Command {
        self.inner.current_dir(dir);
        self
    }
    
    /// Configuration for the child process's standard input (stdin) handle.
    pub fn stdin<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stdin(cfg);
        self
    }
    
    /// Configuration for the child process's standard output (stdout) handle.
    pub fn stdout<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stdout(cfg);
        self
    }
    
    /// Configuration for the child process's standard error (stderr) handle.
    pub fn stderr<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stderr(cfg);
        self
    }
    
    /// Check if the command is blocked and return an appropriate error.
    fn check_blocked(&self) -> Result<(), ProcessError> {
        if is_blocked(&self.program) {
            Err(ProcessError::BlockedExecutable(
                self.program.to_string_lossy().into_owned(),
            ))
        } else {
            Ok(())
        }
    }
    
    /// Executes the command as a child process, returning a handle to it.
    ///
    /// # Errors
    ///
    /// Returns `ProcessError::BlockedExecutable` if the program is blocked.
    pub fn spawn(&mut self) -> Result<Child, ProcessError> {
        self.check_blocked()?;
        Ok(self.inner.spawn()?)
    }
    
    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// # Errors
    ///
    /// Returns `ProcessError::BlockedExecutable` if the program is blocked.
    pub fn output(&mut self) -> Result<Output, ProcessError> {
        self.check_blocked()?;
        Ok(self.inner.output()?)
    }
    
    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its status.
    ///
    /// # Errors
    ///
    /// Returns `ProcessError::BlockedExecutable` if the program is blocked.
    pub fn status(&mut self) -> Result<ExitStatus, ProcessError> {
        self.check_blocked()?;
        Ok(self.inner.status()?)
    }
    
    /// Returns the path to the program that was given to `Command::new`.
    pub fn get_program(&self) -> &OsStr {
        self.inner.get_program()
    }
    
    /// Returns an iterator of the arguments that will be passed to the program.
    pub fn get_args(&self) -> std::process::CommandArgs<'_> {
        self.inner.get_args()
    }
    
    /// Returns an iterator of the environment variables that will be set for
    /// the child process.
    pub fn get_envs(&self) -> std::process::CommandEnvs<'_> {
        self.inner.get_envs()
    }
    
    /// Returns the working directory for the child process.
    pub fn get_current_dir(&self) -> Option<&Path> {
        self.inner.get_current_dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allowed_command() {
        let result = Command::new("echo")
            .arg("hello")
            .output();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_blocked_curl() {
        let result = Command::new("curl")
            .arg("https://example.com")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_blocked_wget() {
        let result = Command::new("wget")
            .arg("https://example.com")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_blocked_nc() {
        let result = Command::new("nc")
            .arg("-l")
            .arg("1234")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_blocked_ssh() {
        let result = Command::new("ssh")
            .arg("user@host")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_blocked_path() {
        // Should still block even with full path
        let result = Command::new("/usr/bin/curl")
            .arg("https://example.com")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_blocked_case_insensitive() {
        let result = Command::new("CURL")
            .arg("https://example.com")
            .output();
        assert!(matches!(result, Err(ProcessError::BlockedExecutable(_))));
    }
    
    #[test]
    fn test_is_blocked() {
        assert!(is_blocked(OsStr::new("curl")));
        assert!(is_blocked(OsStr::new("wget")));
        assert!(is_blocked(OsStr::new("nc")));
        assert!(is_blocked(OsStr::new("/usr/bin/curl")));
        assert!(!is_blocked(OsStr::new("ls")));
        assert!(!is_blocked(OsStr::new("cat")));
        assert!(!is_blocked(OsStr::new("echo")));
    }
}
