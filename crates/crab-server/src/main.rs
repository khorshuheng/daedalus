//! The `crab-server` binary (CRAB-124): parse flags, load config, and serve the
//! WebSocket RPC protocol.
//!
//! Binds a loopback address by default. Remote/mobile use is intended to sit
//! behind `tailscale serve` (HTTPS + tailnet identity); binding a
//! non-loopback address prints a warning. No public exposure is provided.

use std::path::PathBuf;

use clap::Parser;
use crab_core::config::{Config, PartialConfig, PROVIDERS};
use crab_core::session;
use crab_core::workspace::Workspace;
use crab_server::{build_router, AppState};

/// Crab headless server — RPC over WebSocket for remote clients.
#[derive(Parser, Debug)]
#[command(name = "crab-server", version, about)]
struct Cli {
    /// Address to bind. Defaults to loopback; put `tailscale serve` in front
    /// for a tailnet-only HTTPS endpoint.
    #[arg(long, default_value = "127.0.0.1:8787", value_name = "ADDR")]
    bind: String,

    /// Default workspace for sessions (default: current directory).
    #[arg(long, value_name = "PATH")]
    dir: Option<PathBuf>,

    /// Provider name from the registry (openai, anthropic, ollama, fake, …).
    #[arg(long, value_name = "NAME", value_parser = parse_provider)]
    provider: Option<&'static crab_core::config::ProviderInfo>,

    /// Model identifier (required unless the config file sets one).
    #[arg(long, value_name = "NAME")]
    model: Option<String>,

    /// Iteration cap per turn (default: 30).
    #[arg(long, value_name = "N", value_parser = parse_max_iterations)]
    max_iterations: Option<usize>,

    /// Config file (default: ~/.config/crab/config.toml).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

fn parse_provider(s: &str) -> Result<&'static crab_core::config::ProviderInfo, String> {
    crab_core::config::provider_by_name(s)
}

fn parse_max_iterations(s: &str) -> Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if n == 0 {
        return Err("max_iterations must be >= 1".into());
    }
    Ok(n)
}

/// Print the supported providers for `--help` clarity when none match.
fn provider_names() -> String {
    PROVIDERS
        .iter()
        .map(|p| p.name)
        .collect::<Vec<_>>()
        .join(", ")
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let cwd = std::env::current_dir().map_err(|e| format!("cannot determine cwd: {e}"))?;
    let default_workspace = cli.dir.clone().unwrap_or(cwd);

    let flags = PartialConfig {
        provider: cli.provider.map(|p| p.name.to_string()),
        model: cli.model.clone(),
        max_iterations: cli.max_iterations,
        ..Default::default()
    };
    let config = Config::load(default_workspace, cli.config.as_deref(), flags, None)
        .map_err(|e| format!("{e}\n(supported providers: {})", provider_names()))?;
    let workspace = Workspace::new(config.workspace.clone())?;

    let listener = tokio::net::TcpListener::bind(&cli.bind)
        .await
        .map_err(|e| format!("cannot bind {}: {e}", cli.bind))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("cannot read bound address: {e}"))?;
    eprintln!(
        "crab-server listening on {addr} (workspace {})",
        workspace.root().display()
    );
    if !addr.ip().is_loopback() {
        eprintln!(
            "crab-server: warning: bound to a non-loopback address ({addr}); \
             exposure should be tailnet-only via `tailscale serve` (no funnel)"
        );
    }

    let state = AppState::new(config, workspace, session::default_root());
    axum::serve(listener, build_router(state))
        .await
        .map_err(|e| format!("server error: {e}"))
}
