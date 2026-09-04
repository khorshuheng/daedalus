//! Crab CLI binary (CRAB-101, in the `crab` crate of the CRAB-117 workspace).
//!
//! Takes a positional `<prompt>` and a workspace directory (`--dir`, defaulting
//! to cwd) plus `--model`, `--provider`, `--max-iterations`, and `--config`.
//! Builds the app from the `crab_core` library (config CRAB-105, runtime
//! CRAB-116, providers CRAB-103, tools CRAB-102). Interactive use is the REPL
//! (default on a terminal, `term` CRAB-110) or the TUI (`--mode tui`,
//! CRAB-121). Headless/scripted use requires an explicit mode: `--mode json`
//! (one prompt -> every Event as JSONL) or `--mode rpc` (JSONL command/event
//! loop, no prompt needed).
//!
//! Exit codes:
//!   0 — clean exit
//!   1 — hard error (bad config/flags, provider unreachable)
//!   2 — iteration cap exceeded

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use crab_core::config::{Config, PartialConfig, ProviderKind};
use crab_core::memory;
use crab_core::provider::{self, Message};
use crab_core::reflect;
use crab_core::runtime::{AgentRuntime, Event};
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

    /// json | rpc | tui. Interactive (REPL) is the default on a terminal;
    /// json and rpc are headless modes (rpc reads JSON commands from stdin
    /// and needs no <prompt>); tui is the full-screen interface.
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
    // Interactive modes (REPL on a terminal, TUI) need no initial prompt:
    // they wait for the user to type. The headless json mode requires one;
    // rpc reads commands from stdin instead.
    let needs_prompt = match cli.mode {
        None => false, // REPL: may start empty
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

    // No --mode: an interactive terminal gets the REPL (or the TUI with
    // --mode tui). Piped/non-tty stdin without an explicit mode is an error:
    // crab is interactive, so a scripted session must say --mode json or rpc.
    let stdin_is_terminal = std::io::stdin().is_terminal();
    match cli.mode {
        None if stdin_is_terminal => run_repl(
            config,
            provider,
            tools,
            workspace,
            memory_root,
            &cli.prompt(),
        ),
        None => Err("stdin is not a terminal: crab is interactive (REPL/TUI). \
                 For a scripted session pass --mode json or --mode rpc."
            .into()),
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
        Some(Mode::Tui) => {
            let (rt, rx) = AgentRuntime::new(config, provider, tools, workspace, memory_root);
            rt.set_interactive(true); // human present: no iteration cap
            let worker = rt.clone();
            let _worker_handle = std::thread::spawn(move || worker.run_forever());
            let root = session::default_root();
            tui::run_tui(&rt, &rx, &cli.prompt(), root.as_path())
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

/// How the REPL reacts to a slash command.
enum CommandOutcome {
    /// `/exit`/`/quit` — leave the REPL (after auto-saving).
    Exit,
    /// Show this message to the user.
    Message(String),
    /// Show this error to the user.
    Error(String),
}

/// The built-in command list shown by `/help` and after an unknown command.
const COMMANDS: &str = "\
Commands:
  /resume   Continue the previous session for this workspace
  /clear    Reset the conversation to a fresh context
  /reflect  Extract steering lessons from this session into memory
  /help     Show this help
  /exit     Quit and save the session (also /quit, /q)";

/// True when the history contains at least one assistant message — i.e. the
/// conversation actually produced output. A session holding only the system
/// prompt (or an unanswered user message) is not worth resuming.
fn has_conversation(history: &[Message]) -> bool {
    history
        .iter()
        .any(|m| matches!(m, Message::Assistant { .. }))
}

/// Dispatch a `/`-prefixed line to its handler (CRAB-110). Adding a command
/// is one match arm plus a handler; there is no plugin mechanism.
fn dispatch_command(
    rt: &AgentRuntime,
    root: &Path,
    memory_root: &Path,
    cmd: &str,
) -> CommandOutcome {
    match cmd {
        "/exit" | "/quit" | "/q" => CommandOutcome::Exit,
        "/help" => CommandOutcome::Message(COMMANDS.to_string()),
        "/resume" => CommandOutcome::Message(handle_resume(rt, root)),
        "/clear" => CommandOutcome::Message(handle_clear(rt)),
        "/reflect" => CommandOutcome::Message(handle_reflect(rt, root, memory_root)),
        other => CommandOutcome::Error(format!("unknown command '{other}'\n{COMMANDS}")),
    }
}

/// `/resume`: replace the running history with the previous session's, so the
/// next turn continues where that session left off. Reports clearly when
/// there is no previous session (or only an empty one) for this workspace.
fn handle_resume(rt: &AgentRuntime, root: &Path) -> String {
    match session::load_previous(root, &rt.workspace_root()) {
        Ok(Some(history)) if has_conversation(&history) => {
            let discarding = has_conversation(&rt.history());
            let n = history.len();
            rt.replace_history(history);
            if discarding {
                format!("resumed previous session ({n} messages); current conversation discarded")
            } else {
                format!("resumed previous session ({n} messages)")
            }
        }
        Ok(Some(_)) => "previous session has no conversation to resume".to_string(),
        Ok(None) => "no previous session for this workspace".to_string(),
        Err(e) => format!("could not load previous session: {e}"),
    }
}

/// `/clear`: reset the conversation to a fresh context — just the system
/// prompt, so the next message becomes the first user turn. Saved sessions
/// are left untouched (clear is a context reset, not a deletion). Runs
/// synchronously because the REPL only issues it while the worker is idle.
fn handle_clear(rt: &AgentRuntime) -> String {
    rt.reset_sync();
    "conversation cleared".to_string()
}

/// `/reflect`: reflect on the running conversation (auto-saving it first so
/// lessons carry a real session id as provenance) and append new lessons to
/// memory. On-demand trigger (CRAB-112).
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

/// Consume runtime events until the agent settles, printing text deltas and
/// tool markers. Returns the settled text (empty when interrupted).
fn consume_until_settled(rx: &Receiver<Event>, busy: &AtomicBool) -> String {
    busy.store(true, Ordering::SeqCst);
    let mut text = String::new();
    loop {
        match rx.recv() {
            Ok(Event::TextDelta { text: t }) => {
                print!("{t}");
                text.push_str(&t);
                let _ = std::io::stdout().flush();
            }
            Ok(Event::ToolStart { name, .. }) => {
                print!("\n⚙ {name}");
                let _ = std::io::stdout().flush();
            }
            Ok(Event::ToolEnd { ok, .. }) => {
                if !ok {
                    print!(" ✗");
                    let _ = std::io::stdout().flush();
                }
            }
            Ok(Event::AgentSettled { text: t, .. }) => {
                busy.store(false, Ordering::SeqCst);
                return if t.is_empty() { text } else { t };
            }
            Ok(Event::Error { message }) => {
                eprintln!("\ncrab: {message}");
                busy.store(false, Ordering::SeqCst);
                return String::new();
            }
            Ok(_) => {}
            Err(_) => {
                busy.store(false, Ordering::SeqCst);
                return String::new();
            }
        }
    }
}

/// The interactive REPL (CRAB-110) on top of the runtime (CRAB-116). Raw
/// input mode is on for the whole session; a single input thread line-edits
/// and reports Line/Cancel/Eof. While a turn runs, Esc/Ctrl-C abort it; at
/// the prompt, they terminate. A line beginning with `/` is a slash command;
/// anything else is a follow-up. The session is auto-saved on exit.
fn run_repl(
    config: Config,
    provider: Box<dyn crab_core::provider::Provider>,
    tools: ToolSet,
    workspace: Workspace,
    memory_root: Option<PathBuf>,
    initial: &str,
) -> Result<i32, String> {
    let _raw = term::RawMode::enable().map_err(|e| format!("cannot enable raw mode: {e}"))?;
    let busy = Arc::new(AtomicBool::new(false));
    let (rx_input, _input) = term::spawn_input(Arc::clone(&busy));
    let root = session::default_root();
    let mem = memory_root.clone().unwrap_or_else(memory::default_root);

    let (rt, rx_events) = AgentRuntime::new(config, provider, tools, workspace, memory_root);
    rt.set_interactive(true); // human present: no iteration cap
    let worker = rt.clone();
    let _worker_handle = std::thread::spawn(move || worker.run_forever());

    // First message: the initial prompt, when one was given. With no prompt
    // the REPL starts empty and waits for the first line at the prompt.
    if !initial.trim().is_empty() {
        rt.prompt(initial);
        let answer = consume_until_settled(&rx_events, &busy);
        if !answer.is_empty() {
            println!("{answer}");
        }
    }

    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        let event = rx_input.recv().map_err(|_| "input closed".to_string())?;
        match event {
            term::InputEvent::Line(line) => {
                let trimmed = line.trim();
                if trimmed.starts_with('/') {
                    match dispatch_command(&rt, &root, &mem, trimmed) {
                        CommandOutcome::Exit => {
                            auto_save(&rt, &root, &mem);
                            rt.shutdown();
                            return Ok(0);
                        }
                        CommandOutcome::Message(m) => println!("{m}"),
                        CommandOutcome::Error(e) => eprintln!("{e}"),
                    }
                } else if trimmed.is_empty() {
                    continue;
                } else if busy.load(Ordering::Relaxed) {
                    // Typing while the agent runs steers it (CRAB-116).
                    rt.steer(trimmed);
                } else {
                    rt.prompt(trimmed);
                    let answer = consume_until_settled(&rx_events, &busy);
                    if !answer.is_empty() {
                        println!("{answer}");
                    }
                }
            }
            term::InputEvent::Cancel | term::InputEvent::Eof => {
                if busy.load(Ordering::Relaxed) {
                    rt.abort();
                    let _ = consume_until_settled(&rx_events, &busy);
                }
                auto_save(&rt, &root, &mem);
                rt.shutdown();
                return Ok(0);
            }
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

        let msg = handle_clear(&rt);
        assert!(msg.contains("cleared"));
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
    fn resume_loads_the_previous_session() {
        let (dir, root, rt, ws) = setup(
            "resume",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        let prior = vec![
            Message::System("sys".into()),
            Message::User("q1".into()),
            Message::Assistant {
                text: Some("a1".into()),
                tool_calls: vec![],
            },
            Message::User("q2".into()),
            Message::Assistant {
                text: Some("a2".into()),
                tool_calls: vec![],
            },
        ];
        session::save_session(&root, ws.root(), &prior).unwrap();

        // A fresh runtime (no conversation yet).
        let msg = handle_resume(&rt, &root);
        assert!(msg.contains("resumed previous session (5 messages)"));
        assert_eq!(rt.history(), prior);
        let _ = memory_root;
    }

    #[test]
    fn resume_notes_when_the_current_conversation_is_discarded() {
        let (dir, root, rt, ws) = setup(
            "resume-discard",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        session::save_session(
            &root,
            ws.root(),
            &[
                Message::System("sys".into()),
                Message::User("q1".into()),
                Message::Assistant {
                    text: Some("a1".into()),
                    tool_calls: vec![],
                },
            ],
        )
        .unwrap();

        // Current session already produced a conversation.
        assert_eq!(rt.run_once("initial").unwrap(), "done");
        let msg = handle_resume(&rt, &root);
        assert!(msg.contains("resumed previous session (3 messages)"));
        assert!(msg.contains("current conversation discarded"));
        assert_eq!(rt.history().len(), 3); // the loaded session
        let _ = memory_root;
    }

    #[test]
    fn resume_with_no_previous_session_reports_clearly() {
        let (dir, root, rt, _ws) = setup(
            "resume-none",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        let msg = handle_resume(&rt, &root);
        assert!(msg.contains("no previous session for this workspace"));
        let _ = memory_root;
    }

    #[test]
    fn resume_skips_a_session_with_no_conversation() {
        let (dir, root, rt, ws) = setup(
            "resume-empty",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        session::save_session(&root, ws.root(), &[Message::System("sys".into())]).unwrap();

        let msg = handle_resume(&rt, &root);
        assert!(msg.contains("no conversation to resume"));
        let _ = memory_root;
    }

    #[test]
    fn unknown_command_is_reported_with_the_command_list() {
        let (dir, root, rt, _ws) = setup(
            "unknown",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        match dispatch_command(&rt, &root, &memory_root, "/nope") {
            CommandOutcome::Error(e) => {
                assert!(e.contains("unknown command '/nope'"));
                assert!(e.contains("/resume"));
                assert!(e.contains("/help"));
            }
            _ => panic!("expected an error outcome"),
        }
    }

    #[test]
    fn help_lists_the_commands_and_exit_returns_exit() {
        for cmd in ["/resume", "/clear", "/reflect", "/help", "/exit", "/quit"] {
            assert!(COMMANDS.contains(cmd), "missing {cmd} in help");
        }
        let (dir, root, rt, _ws) = setup(
            "help",
            Box::new(RecordingProvider {
                histories: Mutex::new(Vec::new()),
            }),
        );
        let memory_root = mem_root(&dir);
        match dispatch_command(&rt, &root, &memory_root, "/exit") {
            CommandOutcome::Exit => {}
            _ => panic!("expected exit outcome"),
        }
        match dispatch_command(&rt, &root, &memory_root, "/help") {
            CommandOutcome::Message(m) => assert_eq!(m, COMMANDS),
            _ => panic!("expected message outcome"),
        }
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

        let msg = match dispatch_command(&rt, &root, &memory_root, "/reflect") {
            CommandOutcome::Message(m) => m,
            _ => panic!("expected a message outcome"),
        };
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
        let msg = match dispatch_command(&rt, &root, &memory_root, "/reflect") {
            CommandOutcome::Message(m) => m,
            _ => panic!("expected message"),
        };
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
        let msg = match dispatch_command(&rt, &root, &memory_root, "/reflect") {
            CommandOutcome::Message(m) => m,
            _ => panic!("expected a message outcome"),
        };
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
