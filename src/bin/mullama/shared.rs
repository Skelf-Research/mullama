use std::path::Path;
use std::time::Duration;

use mullama::daemon::spawn::default_log_path;
use mullama::daemon::{ensure_daemon_running, DaemonClient, SpawnConfig};
use rand::{distributions::Alphanumeric, Rng};

pub(crate) fn is_loopback_http_addr(addr: &str) -> bool {
    matches!(addr, "127.0.0.1" | "localhost" | "::1")
}

pub(crate) fn generate_api_key() -> String {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(40)
        .map(char::from)
        .collect();
    format!("mullama_{}", suffix)
}

pub(crate) fn connect(socket: &str) -> Result<DaemonClient, Box<dyn std::error::Error>> {
    match DaemonClient::connect_with_timeout(socket, Duration::from_millis(500)) {
        Ok(client) => Ok(client),
        Err(_) => {
            eprintln!("Daemon not running, starting it automatically...");

            let config = SpawnConfig {
                socket: socket.to_string(),
                log_file: Some(default_log_path()),
                ..Default::default()
            };

            if let Err(e) = ensure_daemon_running(&config) {
                return Err(format!(
                    "Failed to start daemon automatically: {}\n\
                    You can start the daemon manually with: mullama serve",
                    e
                )
                .into());
            }

            eprintln!("Daemon started successfully, connecting...");

            DaemonClient::connect_with_timeout(socket, Duration::from_secs(5))
                .map_err(|e| format!("Failed to connect to daemon after starting: {}", e).into())
        }
    }
}

pub(crate) use mullama::daemon::format_size;

pub(crate) fn format_time_ago(time: &chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(*time);

    if duration.num_days() > 30 {
        format!("{} months ago", duration.num_days() / 30)
    } else if duration.num_days() > 0 {
        format!("{} days ago", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{} hours ago", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{} minutes ago", duration.num_minutes())
    } else {
        "just now".to_string()
    }
}

pub(crate) fn derive_alias_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string())
}
