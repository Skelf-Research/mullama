use std::io::{self, BufRead, Write};
use std::time::Duration;

use crate::DaemonAction;
use mullama::daemon::spawn::default_log_path;
use mullama::daemon::{
    daemon_status, is_daemon_running, spawn_daemon, stop_daemon, SpawnConfig, SpawnResult,
};

pub(crate) fn handle_daemon_action(action: DaemonAction) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        DaemonAction::Start {
            http_port,
            http_addr,
            api_key,
            require_api_key,
            gpu_layers,
            context_size,
            context_pool_size,
            socket,
        } => {
            daemon_start(
                &socket,
                http_port,
                &http_addr,
                api_key,
                require_api_key,
                gpu_layers,
                context_size,
                context_pool_size,
            )?;
        }
        DaemonAction::Stop { socket, force: _ } => {
            daemon_stop(&socket)?;
        }
        DaemonAction::Restart { socket } => {
            daemon_restart(&socket)?;
        }
        DaemonAction::Status { socket, json } => {
            daemon_show_status(&socket, json)?;
        }
        DaemonAction::Logs { lines, follow } => {
            daemon_logs(lines, follow)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn daemon_start(
    socket: &str,
    http_port: u16,
    http_addr: &str,
    api_key: Option<String>,
    require_api_key: bool,
    gpu_layers: i32,
    context_size: u32,
    context_pool_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if is_daemon_running(socket) {
        println!("Daemon is already running.");
        if let Ok(info) = daemon_status(socket) {
            println!("  Version: {}", info.version);
            println!("  Uptime:  {}s", info.uptime_secs);
            println!("  Models:  {}", info.models_loaded);
        }
        return Ok(());
    }

    println!("Starting Mullama daemon...");

    let config = SpawnConfig {
        socket: socket.to_string(),
        http_port,
        http_addr: http_addr.to_string(),
        api_key,
        require_api_key,
        gpu_layers,
        context_size,
        context_pool_size,
        background: true,
        log_file: Some(default_log_path()),
        ..Default::default()
    };

    config.save().map_err(|e| {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>
    })?;

    match spawn_daemon(&config) {
        SpawnResult::AlreadyRunning => {
            println!("Daemon is already running.");
        }
        SpawnResult::Spawned { pid } => {
            print!("Waiting for daemon to start");
            io::stdout().flush()?;

            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(30);

            while start.elapsed() < timeout {
                if is_daemon_running(socket) {
                    println!(" OK");
                    println!();
                    println!("Daemon started successfully!");
                    if let Some(pid) = pid {
                        println!("  PID:     {}", pid);
                    }
                    println!("  Socket:  {}", socket);
                    println!("  HTTP:    http://{}:{}", http_addr, http_port);
                    println!("  Logs:    {}", default_log_path().display());
                    return Ok(());
                }
                print!(".");
                io::stdout().flush()?;
                std::thread::sleep(Duration::from_millis(500));
            }

            println!(" FAILED");
            eprintln!("Daemon did not start within {} seconds.", timeout.as_secs());
            eprintln!("Check logs at: {}", default_log_path().display());
        }
        SpawnResult::Failed(e) => {
            eprintln!("Failed to start daemon: {}", e);
        }
    }

    Ok(())
}

pub(crate) fn daemon_stop(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running(socket) {
        println!("Daemon is not running.");
        return Ok(());
    }

    print!("Stopping daemon... ");
    io::stdout().flush()?;

    match stop_daemon(socket) {
        Ok(()) => {
            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(10);

            while start.elapsed() < timeout {
                if !is_daemon_running(socket) {
                    println!("OK");
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            println!("TIMEOUT");
            eprintln!("Daemon did not stop within {} seconds.", timeout.as_secs());
        }
        Err(e) => {
            println!("FAILED");
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

pub(crate) fn daemon_restart(socket: &str) -> Result<(), Box<dyn std::error::Error>> {
    let saved_config = mullama::daemon::SpawnConfig::load();

    let (
        http_port,
        http_addr,
        api_key,
        require_api_key,
        gpu_layers,
        context_size,
        context_pool_size,
        flash_attn,
        cache_type_k,
        cache_type_v,
    ) = if let Some(ref cfg) = saved_config {
        (
            cfg.http_port,
            cfg.http_addr.clone(),
            cfg.api_key.clone(),
            cfg.require_api_key,
            cfg.gpu_layers,
            cfg.context_size,
            cfg.context_pool_size,
            cfg.flash_attn,
            cfg.cache_type_k.clone(),
            cfg.cache_type_v.clone(),
        )
    } else if let Ok(info) = daemon_status(socket) {
        let port = info
            .http_endpoint
            .and_then(|e| e.split(':').next_back().and_then(|p| p.parse().ok()))
            .unwrap_or(8080);
        (
            port,
            "127.0.0.1".to_string(),
            None,
            false,
            0,
            4096,
            mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE,
            false,
            None,
            None,
        )
    } else {
        (
            8080,
            "127.0.0.1".to_string(),
            None,
            false,
            0,
            4096,
            mullama::daemon::DEFAULT_CONTEXT_POOL_SIZE,
            false,
            None,
            None,
        )
    };

    if is_daemon_running(socket) {
        println!("Stopping daemon...");
        daemon_stop(socket)?;
        std::thread::sleep(Duration::from_millis(500));
    }

    let config = mullama::daemon::SpawnConfig {
        socket: socket.to_string(),
        http_port,
        http_addr,
        api_key,
        require_api_key,
        gpu_layers,
        context_size,
        context_pool_size,
        background: true,
        log_file: Some(default_log_path()),
        flash_attn,
        cache_type_k,
        cache_type_v,
        ..Default::default()
    };

    config.save()?;

    match spawn_daemon(&config) {
        mullama::daemon::SpawnResult::AlreadyRunning => {
            println!("Daemon is already running.");
        }
        mullama::daemon::SpawnResult::Spawned { pid } => {
            print!("Waiting for daemon to start");
            io::stdout().flush()?;

            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(30);

            while start.elapsed() < timeout {
                if is_daemon_running(socket) {
                    println!(" OK");
                    println!();
                    println!("Daemon restarted successfully!");
                    if let Some(pid) = pid {
                        println!("  PID:     {}", pid);
                    }
                    println!("  Socket:  {}", socket);
                    return Ok(());
                }
                print!(".");
                io::stdout().flush()?;
                std::thread::sleep(Duration::from_millis(500));
            }

            println!(" FAILED");
            eprintln!("Daemon did not start within {} seconds.", timeout.as_secs());
        }
        mullama::daemon::SpawnResult::Failed(e) => {
            eprintln!("Failed to restart daemon: {}", e);
        }
    }

    Ok(())
}

pub(crate) fn daemon_show_status(
    socket: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !is_daemon_running(socket) {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "running": false,
                    "socket": socket,
                }))?
            );
        } else {
            println!("Daemon is not running.");
            println!();
            println!("Start with: mullama daemon start");
        }
        return Ok(());
    }

    match daemon_status(socket) {
        Ok(info) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "running": true,
                        "version": info.version,
                        "uptime_secs": info.uptime_secs,
                        "models_loaded": info.models_loaded,
                        "socket": info.socket,
                        "http_endpoint": info.http_endpoint,
                    }))?
                );
            } else {
                println!("Mullama Daemon Status");
                println!("=====================");
                println!("Running:     Yes");
                println!("Version:     {}", info.version);
                println!("Uptime:      {}s", info.uptime_secs);
                println!("Models:      {}", info.models_loaded);
                println!("Socket:      {}", info.socket);
                if let Some(ref http) = info.http_endpoint {
                    println!("HTTP:        {}", http);
                }
                println!("Logs:        {}", default_log_path().display());
            }
        }
        Err(e) => {
            eprintln!("Failed to get status: {}", e);
        }
    }

    Ok(())
}

pub(crate) fn daemon_logs(lines: usize, follow: bool) -> Result<(), Box<dyn std::error::Error>> {
    let log_path = default_log_path();

    if !log_path.exists() {
        println!("No log file found at: {}", log_path.display());
        return Ok(());
    }

    if follow {
        let status = std::process::Command::new("tail")
            .arg("-f")
            .arg("-n")
            .arg(lines.to_string())
            .arg(&log_path)
            .status()?;

        if !status.success() {
            eprintln!("Failed to follow logs");
        }
    } else {
        let file = std::fs::File::open(&log_path)?;
        let reader = std::io::BufReader::new(file);
        let all_lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

        let start = if all_lines.len() > lines {
            all_lines.len() - lines
        } else {
            0
        };

        for line in &all_lines[start..] {
            println!("{}", line);
        }
    }

    Ok(())
}
