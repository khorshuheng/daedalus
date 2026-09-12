//! Crab CLI binary (CRAB-101, in the `crab` crate of the CRAB-117 workspace).
//!
//! Takes a positional `<prompt>` and a workspace directory (`--dir`, defaulting
//! to cwd) plus `--model`, `--provider`, `--max-iterations`, and `--config`.
//! Builds the app from the `crab_core` library (config CRAB-105, runtime
//! CRAB-116, providers CRAB-103, tools CRAB-102). Interactive use is the TUI
//! (the default on a terminal, CRAB-121) — the line-based REPL was removed in
//! CRAB-135. Headless/scripted use requires an explicit mode: `--mode json`
//! (one prompt -> every Event as JSONL) or `--mode rpc` (JSONL command/event
//! loop, no prompt needed).
//!
//! Exit codes:
//!   0 — clean exit
//!   1 — hard error (bad config/flags, provider unreachable)
//!   2 — iteration cap exceeded

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use crab_core::config::{Config, PartialConfig};
use crab_core::provider;
use crab_core::runtime::AgentRuntime;
use crab_core::session;
use crab_core::tools::resolver::ToolSet;
use crab_core::workspace::Workspace;

mod markdown;
mod modes;
mod tui;

use modes::Mode;

use clap::Parser;

/// Crab — a minimal coding agent.
#[derive(Parser, Debug)]
#[command(name = "crab", version, about, disable_help_flag = false)]
struct Cli {
    /// The instruction to give the model (not needed in --mode rpc, which
    /// reads commands from stdin).
    #[arg(value_name = "PROMPT", num_args = 0.., trailing_var_arg = false)]
    prompt_parts: Vec<String>,

    /// Workspace directory (default: current directory).
    #[arg(long, value_name = "PATH")]
    dir: Option<PathBuf>,

    /// Provider name from the registry (openai, anthropic, gemini, ollama,
    /// …). Default: openai.
    #[arg(long, value_name = "NAME", value_parser = parse_provider)]
    provider: Option<&'static crab_core::config::ProviderInfo>,

    /// Model identifier (provider-specific default).
    #[arg(long, value_name = "NAME")]
    model: Option<String>,

    /// Iteration cap for the agent loop (default: 30).
    #[arg(long, value_name = "N", value_parser = parse_max_iterations)]
    max_iterations: Option<usize>,

    /// Config file (default: ~/.config/crab/config.toml).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Provider API key (default: provider-native env var, then the keyring).
    #[arg(long, value_name = "KEY", hide = true)]
    api_key: Option<String>,

    /// json | rpc | tui. Interactive use (the TUI) is the default on a
    /// terminal; json and rpc are headless modes (rpc reads JSON commands
    /// from stdin and needs no <prompt>); tui is the full-screen interface.
    #[arg(long, value_name = "MODE", value_parser = parse_mode)]
    mode: Option<Mode>,

    /// TUI theme preset: dark | light (default: detected from the terminal).
    #[arg(long, value_name = "NAME", value_parser = parse_theme)]
    theme: Option<String>,
}

fn parse_provider(s: &str) -> Result<&'static crab_core::config::ProviderInfo, String> {
    crab_core::config::provider_by_name(s)
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    Mode::parse(s)
}

fn parse_max_iterations(s: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("invalid --max-iterations '{s}'"))
}

fn parse_theme(s: &str) -> Result<String, String> {
    if crab_core::theme::Theme::builtin(s).is_some() {
        Ok(s.to_string())
    } else {
        Err(format!("invalid --theme '{s}' (supported: dark, light)"))
    }
}

impl Cli {
    /// Reassemble positional prompt words and the config-file precedence the
    /// way the hand-rolled parser did (flags here are merged over env and
    /// file in `Config::load`).
    fn prompt(&self) -> String {
        self.prompt_parts.join(" ")
    }

    fn flags(&self) -> PartialConfig {
        PartialConfig {
            provider: self.provider.map(|p| p.name.to_string()),
            model: self.model.clone(),
            max_iterations: self.max_iterations,
            theme: self
                .theme
                .clone()
                .map(|name| crab_core::theme::ThemePartial {
                    name: Some(name),
                    ..Default::default()
                }),
            ..Default::default()
        }
    }
}

/// If `--config` was not given, use `~/.config/crab/config.toml` when present.
fn resolve_config_path(explicit: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p);
    }
    let candidate = crab_core::paths::config_dir().join("config.toml");
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

fn run(cli: Cli) -> Result<i32, String> {
    // Interactive use (the TUI) needs no initial prompt: it waits for the
    // user to type. The headless json mode requires one; rpc reads commands
    // from stdin instead.
    let needs_prompt = match cli.mode {
        None => false, // TUI: may start empty
        Some(Mode::Json) => true,
        Some(Mode::Rpc) | Some(Mode::Tui) => false,
    };
    if needs_prompt && cli.prompt_parts.is_empty() {
        return Err("no prompt given".into());
    }
    let cwd = std::env::current_dir().map_err(|e| format!("cannot determine cwd: {e}"))?;
    let default_workspace = cli.dir.clone().unwrap_or(cwd);
    let config_path = resolve_config_path(cli.config.clone());

    let config = Config::load(
        default_workspace,
        config_path.as_deref(),
        cli.flags(),
        cli.api_key.clone(),
    )?;

    let workspace = Workspace::new(config.workspace.clone())?;
    let provider = provider::from_config(&config);
    // CRAB-133: built-ins plus configured MCP servers. Server startup is
    // non-fatal — a failure is a warning and the agent keeps the rest.
    let (tools, mcp_warnings) = ToolSet::from_config(&config, config.max_output_bytes);
    for warning in &mcp_warnings {
        eprintln!("crab: warning: {warning}");
    }

    // No --mode: an interactive terminal gets the TUI (CRAB-135: the REPL
    // was removed). Piped/non-tty stdin without an explicit mode is an
    // error: crab is interactive, so a scripted session must say --mode
    // json or rpc. An explicit --mode tui runs the TUI even when stdin is
    // piped (raw-mode setup fails there with a clear error).
    let stdin_is_terminal = std::io::stdin().is_terminal();
    let mode = match cli.mode {
        Some(mode) => Some(mode),
        None if stdin_is_terminal => Some(Mode::Tui),
        None => {
            return Err("stdin is not a terminal: crab is interactive (TUI). \
                 For a scripted session pass --mode json or --mode rpc."
                .into())
        }
    };
    match mode {
        Some(Mode::Json) => {
            let (rt, mut rx) = AgentRuntime::new(config, provider, tools, workspace);
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let mut stdout = std::io::stdout();
            modes::run_json(&rt, &mut rx, &cli.prompt(), &mut stdout)
        }
        Some(Mode::Rpc) => {
            let (rt, mut rx) = AgentRuntime::new(config, provider, tools, workspace);
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let root = session::default_root();
            let reader: Box<dyn std::io::BufRead + Send> =
                Box::new(std::io::BufReader::new(std::io::stdin()));
            let mut stdout = std::io::stdout();
            modes::run_rpc(&rt, &mut rx, &root, reader, &mut stdout)
        }
        Some(Mode::Tui) | None => {
            let theme = config.theme.clone();
            let (rt, mut rx) = AgentRuntime::new(config, provider, tools, workspace);
            rt.set_interactive(true); // human present: no iteration cap
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let root = session::default_root();
            tui::run_tui(&rt, &mut rx, &cli.prompt(), root.as_path(), &theme)
        }
    }
}

/// Persist the current session history to disk (auto-save on exit, CRAB-109).
/// Failures are warnings only — quitting must never be blocked by persistence.
/// (CRAB-137 removed the auto-reflection that used to follow the save.)
fn auto_save(rt: &AgentRuntime, root: &Path) {
    match session::save_session(root, &rt.workspace_root(), &rt.history()) {
        Ok(path) => {
            eprintln!("session saved: {}", path.display());
        }
        Err(e) => {
            eprintln!("crab: warning: could not save session: {e}");
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let code = match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("crab: {e}");
            1
        }
    };
    std::process::exit(code);
}

/// True when the history contains at least one assistant message — i.e. the
/// conversation actually produced output.
#[cfg(test)]
fn has_conversation(history: &[crab_core::provider::Message]) -> bool {
    history
        .iter()
        .any(|m| matches!(m, crab_core::provider::Message::Assistant { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_core::config::Config;
    use crab_core::provider::Message;
    use crab_core::provider::{Provider, ProviderError, Response, StreamDelta};
    use crab_core::runtime::AgentRuntime;
    use crab_core::tools::resolver::ToolSet;
    use crab_core::workspace::Workspace;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// A provider that records every history it is given and answers "done".
    struct RecordingProvider {
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for RecordingProvider {
        fn complete<'a>(
            &'a self,
            history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<crab_core::provider::Completion, ProviderError>>
        {
            Box::pin(async move {
                self.histories.lock().unwrap().push(history.to_vec());
                Ok(crab_core::provider::Completion {
                    response: Response::Text("done".into()),
                    prompt_tokens: None,
                    aborted: false,
                })
            })
        }
    }

    /// Build a temp workspace + sessions root + a recording provider.
    fn setup(
        _name: &str,
        provider: Box<dyn Provider>,
    ) -> (tempfile::TempDir, PathBuf, AgentRuntime, Workspace) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("sessions");
        let tools = ToolSet::new(1000);
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(cfg, provider, tools, ws.clone());
        (dir, root, rt, ws)
    }

    #[test]
    fn clear_resets_history_for_the_next_message() {
        let (_dir, _root, rt, _ws) = setup(
            "clear",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        // Establish a conversation.
        assert_eq!(rt.run_once("initial prompt").unwrap(), "done");
        assert!(has_conversation(&rt.history()));

        rt.reset_sync();
        // After /clear the history is just the seed system prompt.
        assert_eq!(rt.history().len(), 1);

        // The next prompt starts a fresh conversation.
        assert_eq!(rt.run_once("follow-up after clear").unwrap(), "done");
        let h = rt.history();
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "follow-up after clear")));
        assert!(!h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "initial prompt")));
    }

    #[test]
    fn runtime_emits_events_on_a_run() {
        let (_dir, _root, rt, _ws) = setup(
            "events",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        // run_once drives synchronously and returns the final text.
        assert_eq!(rt.run_once("hello").unwrap(), "done");
        assert!(has_conversation(&rt.history()));
    }
}
