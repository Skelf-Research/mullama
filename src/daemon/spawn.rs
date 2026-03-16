//! Daemon Auto-Spawn Functionality
//!
//! Provides utilities to automatically spawn the mullama daemon when needed.
//! This enables seamless usage where users can run `mullama run llama3.2:1b`
//! without first starting the daemon manually.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use super::client::DaemonClient;
use super::DEFAULT_SOCKET;

/// Configuration for auto-spawning the daemon
#[derive(Debug, Clone)]
pub struct SpawnConfig {
    /// Path to the mullama binary
    pub binary_path: Option<PathBuf>,

    /// Socket address
    pub socket: String,

    /// HTTP port (0 to disable)
    pub http_port: u16,

    /// HTTP bind address
    pub http_addr: String,

    /// API key for HTTP endpoints
    pub api_key: Option<String>,

    /// Always require API key auth, even when bound to localhost
    pub require_api_key: bool,

    /// Default GPU layers
    pub gpu_layers: i32,

    /// Default context size
    pub context_size: u32,

    /// Number of contexts in each loaded model pool
    pub context_pool_size: usize,

    /// Timeout for waiting for daemon to be ready
    pub startup_timeout: Duration,

    /// Whether to run in background (daemonize)
    pub background: bool,

    /// Log file path (for background mode)
    pub log_file: Option<PathBuf>,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            binary_path: None,
            socket: DEFAULT_SOCKET.to_string(),
            http_port: 8080,
            http_addr: "127.0.0.1".to_string(),
            api_key: None,
            require_api_key: false,
            gpu_layers: 0,
            context_size: 4096,
            context_pool_size: super::models::DEFAULT_CONTEXT_POOL_SIZE,
            startup_timeout: Duration::from_secs(30),
            background: true,
            log_file: None,
        }
    }
}

/// Result of a spawn operation
#[derive(Debug)]
pub enum SpawnResult {
    /// Daemon was already running
    AlreadyRunning,

    /// Daemon was spawned successfully
    Spawned {
        /// Process ID (if spawned in foreground)
        pid: Option<u32>,
    },

    /// Failed to spawn
    Failed(String),
}

/// Check if the daemon is running
pub fn is_daemon_running(socket: &str) -> bool {
    match DaemonClient::connect_with_timeout(socket, Duration::from_millis(500)) {
        Ok(client) => client.ping().is_ok(),
        Err(_) => false,
    }
}

/// Ensure the daemon is running, spawning it if necessary
pub fn ensure_daemon_running(config: &SpawnConfig) -> Result<(), String> {
    if is_daemon_running(&config.socket) {
        return Ok(());
    }

    match spawn_daemon(config) {
        SpawnResult::AlreadyRunning => Ok(()),
        SpawnResult::Spawned { .. } => {
            // Wait for daemon to be ready
            wait_for_daemon(&config.socket, config.startup_timeout)
        }
        SpawnResult::Failed(e) => Err(e),
    }
}

/// Spawn the daemon
pub fn spawn_daemon(config: &SpawnConfig) -> SpawnResult {
    // Check if already running
    if is_daemon_running(&config.socket) {
        return SpawnResult::AlreadyRunning;
    }

    // Find the mullama binary
    let binary = find_mullama_binary(config.binary_path.as_ref());
    let binary = match binary {
        Some(b) => b,
        None => return SpawnResult::Failed("Could not find mullama binary".to_string()),
    };

    // Build command
    let mut cmd = Command::new(&binary);
    cmd.arg("serve");
    cmd.arg("--socket").arg(&config.socket);
    cmd.arg("--http-port").arg(config.http_port.to_string());
    cmd.arg("--http-addr").arg(&config.http_addr);
    cmd.arg("--gpu-layers").arg(config.gpu_layers.to_string());
    cmd.arg("--context-size")
        .arg(config.context_size.to_string());
    cmd.arg("--context-pool-size")
        .arg(config.context_pool_size.to_string());
    if let Some(api_key) = &config.api_key {
        cmd.arg("--api-key").arg(api_key);
    }
    if config.require_api_key {
        cmd.arg("--require-api-key");
    }

    if config.background {
        // Spawn in background
        cmd.stdin(Stdio::null());

        if let Some(ref log_file) = config.log_file {
            // Create parent directory if needed
            if let Some(parent) = log_file.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            match std::fs::File::create(log_file) {
                Ok(file) => {
                    match file.try_clone() {
                        Ok(stdout_file) => {
                            cmd.stdout(Stdio::from(stdout_file));
                        }
                        Err(_) => {
                            cmd.stdout(Stdio::null());
                        }
                    }
                    cmd.stderr(Stdio::from(file));
                }
                Err(_) => {
                    cmd.stdout(Stdio::null());
                    cmd.stderr(Stdio::null());
                }
            }
        } else {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }

        // Platform-specific daemonization
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Create new process group
            cmd.process_group(0);
        }

        match cmd.spawn() {
            Ok(child) => SpawnResult::Spawned {
                pid: Some(child.id()),
            },
            Err(e) => SpawnResult::Failed(format!("Failed to spawn daemon: {}", e)),
        }
    } else {
        // Spawn in foreground (for debugging)
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());

        match cmd.spawn() {
            Ok(child) => SpawnResult::Spawned {
                pid: Some(child.id()),
            },
            Err(e) => SpawnResult::Failed(format!("Failed to spawn daemon: {}", e)),
        }
    }
}

/// Find the mullama binary
fn find_mullama_binary(override_path: Option<&PathBuf>) -> Option<PathBuf> {
    // Use override if provided
    if let Some(path) = override_path {
        if path.exists() {
            return Some(path.clone());
        }
    }

    // Check MULLAMA_BIN environment variable
    if let Ok(path) = std::env::var("MULLAMA_BIN") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check current executable's directory
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("mullama");
            if candidate.exists() {
                return Some(candidate);
            }
            #[cfg(windows)]
            {
                let candidate = dir.join("mullama.exe");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    // Check PATH
    if let Ok(output) = Command::new("which").arg("mullama").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Check common locations
    let common_paths = [
        "/usr/local/bin/mullama",
        "/usr/bin/mullama",
        "~/.local/bin/mullama",
        "~/.cargo/bin/mullama",
    ];

    for path in common_paths {
        let expanded = if path.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                home.join(&path[2..])
            } else {
                PathBuf::from(path)
            }
        } else {
            PathBuf::from(path)
        };

        if expanded.exists() {
            return Some(expanded);
        }
    }

    None
}

/// Wait for the daemon to be ready
fn wait_for_daemon(socket: &str, timeout: Duration) -> Result<(), String> {
    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(100);

    while start.elapsed() < timeout {
        if is_daemon_running(socket) {
            return Ok(());
        }
        std::thread::sleep(poll_interval);
    }

    Err(format!(
        "Daemon did not start within {} seconds",
        timeout.as_secs()
    ))
}

/// Stop the daemon
pub fn stop_daemon(socket: &str) -> Result<(), String> {
    match DaemonClient::connect_with_timeout(socket, Duration::from_secs(5)) {
        Ok(client) => client
            .shutdown()
            .map_err(|e| format!("Failed to shutdown daemon: {}", e)),
        Err(_) => Ok(()), // Already not running
    }
}

/// Get daemon status
pub fn daemon_status(socket: &str) -> Result<DaemonInfo, String> {
    let client = DaemonClient::connect_with_timeout(socket, Duration::from_secs(5))
        .map_err(|e| format!("Failed to connect: {}", e))?;

    let (uptime, version) = client
        .ping()
        .map_err(|e| format!("Failed to ping: {}", e))?;

    let status = client
        .status()
        .map_err(|e| format!("Failed to get status: {}", e))?;

    Ok(DaemonInfo {
        running: true,
        version,
        uptime_secs: uptime,
        models_loaded: status.models_loaded as u32,
        socket: socket.to_string(),
        http_endpoint: status.http_endpoint,
    })
}

/// Daemon information
#[derive(Debug, Clone)]
pub struct DaemonInfo {
    pub running: bool,
    pub version: String,
    pub uptime_secs: u64,
    pub models_loaded: u32,
    pub socket: String,
    pub http_endpoint: Option<String>,
}

/// Get the default log file path
pub fn default_log_path() -> PathBuf {
    if cfg!(target_os = "macos") {
        PathBuf::from("/tmp/mullamad.log")
    } else if cfg!(windows) {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mullama")
            .join("mullamad.log")
    } else {
        // Linux and other Unix
        PathBuf::from("/tmp/mullamad.log")
    }
}

/// Get the default PID file path
pub fn default_pid_path() -> PathBuf {
    if cfg!(target_os = "macos") {
        PathBuf::from("/tmp/mullamad.pid")
    } else if cfg!(windows) {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mullama")
            .join("mullamad.pid")
    } else {
        // Linux and other Unix - try XDG_RUNTIME_DIR first
        if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
            PathBuf::from(runtime_dir).join("mullamad.pid")
        } else {
            PathBuf::from("/tmp/mullamad.pid")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_binary() {
        // Just test that it doesn't panic
        let _ = find_mullama_binary(None);
    }

    #[test]
    fn test_spawn_config_default() {
        let config = SpawnConfig::default();
        assert_eq!(config.http_port, 8080);
        assert!(config.background);
    }
}
