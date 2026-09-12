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

use crab_core::config::{Config, PartialConfig, ProviderKind};
use crab_core::memory;
use crab_core::provider::{self, Message};
use crab_core::reflect;
use crab_core::runtime::AgentRuntime;
use crab_core::session;
use crab_core::tools::resolver::ToolSet;
use crab_core::workspace::Workspace;

mod modes;
mod term;
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

    /// openai | anthropic | deepseek | fake (default: openai).
    #[arg(long, value_name = "NAME", value_parser = parse_provider)]
    provider: Option<ProviderKind>,

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
}

fn parse_provider(s: &str) -> Result<ProviderKind, String> {
    ProviderKind::parse(s)
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    Mode::parse(s)
}

fn parse_max_iterations(s: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("invalid --max-iterations '{s}'"))
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
            provider: self.provider,
            model: self.model.clone(),
            max_iterations: self.max_iterations,
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
    let tools = ToolSet::new(config.max_output_bytes);
    // Memory injection (CRAB-114): real sessions rank lessons for the
    // workspace into the system prompt.
    let memory_root = Some(memory::default_root());

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
            let (rt, rx) = AgentRuntime::new(config, provider, tools, workspace, memory_root);
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let mut stdout = std::io::stdout();
            modes::run_json(&rt, &rx, &cli.prompt(), &mut stdout)
        }
        Some(Mode::Rpc) => {
            let (rt, rx) = AgentRuntime::new(config, provider, tools, workspace, memory_root);
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let root = session::default_root();
            let reader: Box<dyn std::io::BufRead + Send> =
                Box::new(std::io::BufReader::new(std::io::stdin()));
            let mut stdout = std::io::stdout();
            modes::run_rpc(&rt, &rx, &root, reader, &mut stdout)
        }
        Some(Mode::Tui) | None => {
            let mem = memory_root.clone().unwrap_or_else(memory::default_root);
            let (rt, rx) = AgentRuntime::new(config, provider, tools, workspace, memory_root);
            rt.set_interactive(true); // human present: no iteration cap
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let root = session::default_root();
            tui::run_tui(&rt, &rx, &cli.prompt(), root.as_path(), mem.as_path())
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Persist the current session history to disk (auto-save on exit, CRAB-109)
/// and, when the conversation produced output, reflect it into memory
/// (auto-reflect at session end, CRAB-112). Failures are warnings only —
/// quitting must never be blocked by persistence or reflection.
fn auto_save(rt: &AgentRuntime, root: &Path, memory_root: &Path) {
    let saved = match session::save_session(root, &rt.workspace_root(), &rt.history()) {
        Ok(path) => {
            eprintln!("session saved: {}", path.display());
            Some(path)
        }
        Err(e) => {
            eprintln!("crab: warning: could not save session: {e}");
            None
        }
    };
    if !has_conversation(&rt.history()) {
        return;
    }
    let history = rt.history();
    // Skip a redundant auto-reflect when this exact conversation was already
    // reflected (e.g. the user ran /reflect, then quit) — CRAB-123 #1. The
    // marker stores a content fingerprint, so re-saving the same conversation
    // (a new session id each time) is recognized as already reflected.
    let fingerprint = crab_core::reflect::history_fingerprint(&history);
    if crab_core::reflect::last_reflected(memory_root, &rt.workspace_root()).as_deref()
        == Some(fingerprint.as_str())
    {
        return;
    }
    // The session was just saved, so lessons can carry its real id.
    let source = saved.as_deref().and_then(session::file_id);
    let ws = rt.workspace_root();
    match reflect::reflect_and_store(
        memory_root,
        &ws,
        rt.provider().as_ref(),
        term::cancel_flag(),
        &history,
        source,
        now_millis(),
    ) {
        Ok(0) => {}
        Ok(n) => {
            crab_core::reflect::mark_reflected(memory_root, &ws, &fingerprint);
            eprintln!("memory: reflected {n} new lesson(s) from this session");
        }
        Err(e) => eprintln!("crab: warning: could not reflect lessons: {e}"),
    }
}

/// True when the history contains at least one assistant message — i.e. the
/// conversation actually produced output. A session holding only the system
/// prompt (or an unanswered user message) is not worth resuming.
fn has_conversation(history: &[Message]) -> bool {
    history
        .iter()
        .any(|m| matches!(m, Message::Assistant { .. }))
}

/// `/reflect` (CRAB-112): reflect on the running conversation (auto-saving it
/// first so lessons carry a real session id as provenance) and append new
/// lessons to memory. Shared by the TUI command and its tests (CRAB-135: the
/// TUI is the only interactive frontend).
fn handle_reflect(rt: &AgentRuntime, root: &Path, memory_root: &Path) -> String {
    if !has_conversation(&rt.history()) {
        return "no conversation to reflect on".to_string();
    }
    let path = match session::save_session(root, &rt.workspace_root(), &rt.history()) {
        Ok(p) => p,
        Err(e) => return format!("could not save session: {e}"),
    };
    let history = rt.history();
    let ws = rt.workspace_root();
    let fingerprint = crab_core::reflect::history_fingerprint(&history);
    match reflect::reflect_and_store(
        memory_root,
        &ws,
        rt.provider().as_ref(),
        term::cancel_flag(),
        &history,
        session::file_id(&path),
        now_millis(),
    ) {
        Ok(0) => "reflected: no new lessons".to_string(),
        Ok(n) => {
            crab_core::reflect::mark_reflected(memory_root, &ws, &fingerprint);
            format!("reflected: added {n} new lesson(s)")
        }
        Err(e) => format!("reflection failed: {e}"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crab_core::config::Config;
    use crab_core::provider::{Provider, ProviderError, Response};
    use crab_core::runtime::AgentRuntime;
    use crab_core::tools::resolver::ToolSet;
    use crab_core::workspace::Workspace;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    /// A provider that records every history it is given and answers "done".
    struct RecordingProvider {
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for RecordingProvider {
        fn complete(
            &self,
            history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<crab_core::provider::Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            Ok(crab_core::provider::Completion {
                response: Response::Text("done".into()),
                prompt_tokens: None,
                aborted: false,
            })
        }
    }

    /// A provider that plays a scripted `Response` per call.
    struct ScriptedProvider {
        responses: Mutex<std::collections::VecDeque<crab_core::provider::Response>>,
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<crab_core::provider::Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(crab_core::provider::Response::Text("done".into()));
            Ok(crab_core::provider::Completion {
                response,
                prompt_tokens: None,
                aborted: false,
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
        let (rt, _rx) = AgentRuntime::new(cfg, provider, tools, ws.clone(), None);
        (dir, root, rt, ws)
    }

    fn mem_root(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("memory")
    }

    #[test]
    fn clear_resets_history_for_the_next_message() {
        let (dir, _root, rt, _ws) = setup(
            "clear",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
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
        let _ = memory_root;
    }

    #[test]
    fn reflect_dispatching_adds_lessons_with_session_provenance() {
        let (dir, root, rt, ws) = setup(
            "reflect",
            Box::new(ScriptedProvider {
                responses: Mutex::new(std::collections::VecDeque::from([
                    crab_core::provider::Response::Text("done with the task".into()),
                    crab_core::provider::Response::Text(
                        r#"[{"text":"always run make first","kind":"rule"}]"#.into(),
                    ),
                ])),
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);

        // Run a conversation turn first (consumes response 1).
        assert_eq!(rt.run_once("fix the build").unwrap(), "done with the task");

        let msg = handle_reflect(&rt, &root, &memory_root);
        assert!(msg.contains("added 1 new lesson"), "{msg}");

        // The lesson landed in memory with provenance back to the session.
        let lessons = crab_core::memory::list_lessons(&memory_root, ws.root()).unwrap();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, "always run make first");
        assert_eq!(lessons[0].cwd, ws.root().to_string_lossy());
        assert!(lessons[0].source_session_id.is_some());
        assert!(session::load_previous(&root, ws.root()).unwrap().is_some());
    }

    #[test]
    fn auto_reflect_is_skipped_when_conversation_was_already_reflected() {
        // Provider: turn answer, then one reflect response. After /reflect
        // marks the conversation, a later auto_save with the *same* history
        // must not fire a second reflect LLM call.
        let (dir, root, rt, ws) = setup(
            "reflect-skip",
            Box::new(ScriptedProvider {
                responses: Mutex::new(std::collections::VecDeque::from([
                    crab_core::provider::Response::Text("done".into()),
                    crab_core::provider::Response::Text(
                        r#"[{"text":"a rule","kind":"rule"}]"#.into(),
                    ),
                ])),
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        assert_eq!(rt.run_once("task").unwrap(), "done"); // call 1
        let msg = handle_reflect(&rt, &root, &memory_root);
        assert!(msg.contains("added 1 new lesson"), "{msg}"); // call 2 (reflect)

        // auto_save with the unchanged conversation: the fingerprint matches
        // the marker, so no third provider call and no new lesson.
        auto_save(&rt, &root, &memory_root);
        let lessons = crab_core::memory::list_lessons(&memory_root, ws.root()).unwrap();
        assert_eq!(lessons.len(), 1);
    }

    #[test]
    fn reflect_with_no_conversation_reports_clearly() {
        let (dir, root, rt, _ws) = setup(
            "reflect-empty",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        let msg = handle_reflect(&rt, &root, &memory_root);
        assert!(msg.contains("no conversation to reflect on"), "{msg}");
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
