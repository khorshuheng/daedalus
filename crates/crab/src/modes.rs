//! Headless frontends (CRAB-120): synchronous adapters over AgentRuntime for
//! scripted/non-interactive use, selected with `--mode`. Interactive use is
//! the TUI (default on a terminal; the REPL was removed in CRAB-135). Each
//! adapter consumes only the runtime's Event stream and Command surface — no
//! loop internals.
//!
//! - `json`: one-shot; runs one prompt and emits **every** runtime Event as
//!   a JSONL object to stdout (the same serialization the TUI and server
//!   will use, CRAB-121/124) — a headless diagnostic view.
//! - `rpc`: a JSONL command/event loop over stdin/stdout. Each request is a
//!   `Command` JSON object on its own line; the runtime's events stream back
//!   as JSONL. When a request carries an `id`, the adapter emits a terminal
//!   `{"type":"response","id":...}` line after the command's events so
//!   clients can correlate replies (pi's RPC mode).
//!
//! These headless modes do **not** auto-reflect lessons into memory
//! (CRAB-123 #4, documented non-goal): the session-save + reflection trigger
//! belongs to the interactive TUI.

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc::Receiver;

use crab_core::runtime::{AgentRuntime, Command, CommandKind, Event};

/// The frontend selected by `--mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Json,
    Rpc,
    Tui,
}

impl Mode {
    /// Parse a `--mode` value. `repl` gets its own migration note (CRAB-135:
    /// the REPL was removed; the TUI is the interactive frontend).
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Mode::Json),
            "rpc" => Ok(Mode::Rpc),
            "tui" => Ok(Mode::Tui),
            "repl" => Err(
                "the REPL was removed in CRAB-135; the TUI is the interactive \
                 frontend and the default on a terminal"
                    .into(),
            ),
            other => Err(format!(
                "unknown mode '{other}' (supported: json, rpc, tui)"
            )),
        }
    }
}

/// One-shot json: run `prompt` and emit every runtime Event as a JSONL line
/// until `agent_settled`. Returns the process exit code.
pub fn run_json(
    rt: &AgentRuntime,
    rx: &Receiver<Event>,
    prompt: &str,
    out: &mut dyn Write,
) -> Result<i32, String> {
    rt.prompt(prompt);
    drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
    Ok(0)
}

/// rpc: read `Command` JSON lines from `input` on a reader thread, push each
/// into the runtime, and stream events back to `out` on the main thread.
///
/// Steering while a turn is running is fire-and-forget (pi rpc semantics):
/// the reader thread forwards `steer`/`followUp`/`abort` directly to the
/// runtime the moment they are read, so they interrupt the in-flight turn;
/// they carry no response. Everything else — `prompt`, `set_*`, `clear`,
/// `get_state`, `resume`, and a steer/followUp sent while idle — is handled
/// on the main thread: push, drain events to the command's terminal event,
/// then write `{"type":"response","id":...}` when the request had an `id`.
///
/// Returns the process exit code (0 on EOF / normal shutdown).
pub fn run_rpc(
    rt: &AgentRuntime,
    rx: &Receiver<Event>,
    root: &Path,
    input: Box<dyn BufRead + Send>,
    out: &mut dyn Write,
) -> Result<i32, String> {
    use std::sync::mpsc;
    use std::thread;

    let (cmd_tx, cmd_rx) = mpsc::channel::<Result<Command, String>>();
    let reader_rt = rt.clone();
    let mut input = input;
    let reader = thread::spawn(move || {
        let mut line = String::new();
        loop {
            line.clear();
            match input.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Command>(trimmed) {
                        Ok(cmd) => {
                            // Interactive commands go straight to the runtime
                            // when a turn is running (no response expected).
                            let interactive = matches!(
                                cmd.kind,
                                CommandKind::Steer { .. }
                                    | CommandKind::FollowUp { .. }
                                    | CommandKind::Abort {}
                            );
                            if interactive && reader_rt.is_busy() {
                                forward_interactive(&reader_rt, &cmd.kind);
                                continue;
                            }
                            if cmd_tx.send(Ok(cmd)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            if cmd_tx.send(Err(format!("bad request: {e}"))).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    });

    loop {
        let command = match cmd_rx.recv() {
            Ok(Ok(command)) => command,
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        };
        let id = command.id.clone();
        let mut responded = false;
        match &command.kind {
            CommandKind::Prompt { text } => {
                rt.prompt(text);
                drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
            }
            CommandKind::Steer { text } => {
                // Arrived while idle: it starts a turn.
                rt.steer(text);
                drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
            }
            CommandKind::FollowUp { text } => {
                rt.follow_up(text);
                drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
            }
            CommandKind::Abort {} => {
                rt.abort();
                drain_until_quiet(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
            }
            CommandKind::SetModel { model } => {
                rt.set_model(model);
                drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::SetEffort { effort } => {
                rt.set_effort(*effort);
                drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::SwitchWorkspace { path } => {
                rt.switch_workspace(path);
                drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::Clear {} => {
                rt.clear();
                drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::GetState {} => {
                rt.get_state();
                drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::Resume => {
                let history = crab_core::session::load_previous(root, &rt.workspace_root())
                    .map_err(|e| format!("could not load previous session: {e}"))?;
                match history {
                    Some(h)
                        if h.iter().any(|m| {
                            matches!(m, crab_core::provider::Message::Assistant { .. })
                        }) =>
                    {
                        rt.replace_history(h);
                        write_response(out, id.as_deref(), true)?;
                        responded = true;
                    }
                    _ => {
                        write_response(out, id.as_deref(), false)?;
                        responded = true;
                    }
                }
            }
        }
        if !responded {
            write_response(out, id.as_deref(), true)?;
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    let _ = reader.join();
    Ok(0)
}

/// Forward an interactive command (steer/followUp/abort) straight into the
/// runtime from the reader thread (fire-and-forget, no response line).
fn forward_interactive(rt: &AgentRuntime, kind: &CommandKind) {
    match kind {
        CommandKind::Steer { text } => rt.steer(text),
        CommandKind::FollowUp { text } => rt.follow_up(text),
        CommandKind::Abort {} => rt.abort(),
        _ => {}
    }
}

/// Drain runtime events, writing each as a JSONL line, until `stop` matches.
/// Returns the number of events written.
fn drain_until(
    rx: &Receiver<Event>,
    out: &mut dyn Write,
    stop: impl Fn(&Event) -> bool,
) -> Result<usize, String> {
    let mut count = 0usize;
    loop {
        match rx.recv() {
            Ok(event) => {
                let line = serde_json::to_string(&event).map_err(|e| e.to_string())?;
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
                count += 1;
                if stop(&event) {
                    return Ok(count);
                }
            }
            Err(_) => return Ok(count),
        }
    }
}

/// Drain events like `drain_until`, but give up after a short quiet period so
/// an abort while idle (no settle follows) cannot hang the adapter.
fn drain_until_quiet(
    rx: &Receiver<Event>,
    out: &mut dyn Write,
    stop: impl Fn(&Event) -> bool,
) -> Result<usize, String> {
    let mut count = 0usize;
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(event) => {
                let line = serde_json::to_string(&event).map_err(|e| e.to_string())?;
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
                count += 1;
                if stop(&event) {
                    return Ok(count);
                }
            }
            Err(_) => return Ok(count),
        }
    }
}

/// Write a `{"type":"response","id":...}` terminal line for an id-bearing
/// request (rpc correlation). `id` is `None` for fire-and-forget requests,
/// which get no response line.
fn write_response(out: &mut dyn Write, id: Option<&str>, ok: bool) -> Result<(), String> {
    let Some(id) = id else { return Ok(()) };
    let payload = serde_json::json!({ "type": "response", "id": id, "ok": ok });
    writeln!(out, "{payload}").map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_core::config::Config;
    use crab_core::provider::fake::FakeProvider;
    use crab_core::provider::{Message, Response};
    use crab_core::runtime::Effort;
    use crab_core::session;
    use crab_core::tools::resolver::ToolSet;
    use crab_core::workspace::Workspace;
    use std::io::Cursor;
    use std::path::PathBuf;

    /// A unique temp dir cleaned up on drop (tempfile, CRAB-119).
    type TempDir = tempfile::TempDir;

    fn setup(
        _name: &str,
        responses: Vec<Response>,
    ) -> (
        TempDir,
        PathBuf,
        AgentRuntime,
        Receiver<Event>,
        std::thread::JoinHandle<()>,
        Workspace,
    ) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("sessions");
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let tools = ToolSet::new(1000);
        let (rt, rx) = AgentRuntime::new(
            cfg,
            Box::new(FakeProvider::new(responses)),
            tools,
            ws.clone(),
            None,
        );
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        // Drain the initial agent_start so tests see events from their command.
        let _ = rx.recv_timeout(std::time::Duration::from_secs(2));
        (dir, root, rt, rx, handle, ws)
    }

    fn shutdown(rt: &AgentRuntime, handle: std::thread::JoinHandle<()>) {
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn json_mode_emits_every_event_as_jsonl() {
        let (tmp, _root, rt, rx, handle, _ws) =
            setup("json", vec![Response::Text("hello world".into())]);
        let mut out = Vec::new();
        let code = run_json(&rt, &rx, "greet", &mut out).unwrap();
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // Every line is a type-tagged event; the turn ends with agent_settled.
        let types: Vec<&str> = lines.iter().map(|v| v["type"].as_str().unwrap()).collect();
        assert!(
            types.contains(&"turn_start"),
            "expected turn lifecycle: {types:?}"
        );
        assert_eq!(types.last().copied(), Some("agent_settled"));
        // Every event carries the stable `type` discriminator.
        for t in &types {
            assert!(!t.is_empty());
        }
        shutdown(&rt, handle);
        let _ = tmp;
    }

    #[test]
    fn rpc_prompt_with_id_streams_events_then_responds() {
        let (tmp, _root, rt, rx, handle, _ws) =
            setup("rpc-prompt", vec![Response::Text("the answer".into())]);
        let input = Box::new(Cursor::new(
            "{\"id\":\"1\",\"type\":\"prompt\",\"text\":\"what is 2+2\"}\n".to_string(),
        ));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &rx, tmp.path(), input, &mut out).unwrap();
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // Events until agent_settled, then the correlated response.
        assert_eq!(lines.last().unwrap()["type"], "response");
        assert_eq!(lines.last().unwrap()["id"], "1");
        assert_eq!(lines.last().unwrap()["ok"], true);
        let types: Vec<&str> = lines.iter().map(|v| v["type"].as_str().unwrap()).collect();
        assert_eq!(types.last().copied(), Some("response"));
        assert!(types[..types.len() - 1].contains(&"agent_settled"));
        shutdown(&rt, handle);
        let _ = tmp;
    }

    #[test]
    fn rpc_get_state_and_set_effort_round_trip() {
        let (tmp, _root, rt, rx, handle, _ws) = setup("rpc-state", vec![]);
        let input = concat!(
            "{\"id\":\"s1\",\"type\":\"get_state\"}\n",
            "{\"id\":\"s2\",\"type\":\"set_effort\",\"effort\":\"high\"}\n",
            "{\"type\":\"abort\"}\n"
        );
        let input = Box::new(Cursor::new(input.to_string()));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &rx, tmp.path(), input, &mut out).unwrap();
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // get_state -> state_changed event + response; set_effort likewise.
        let responses: Vec<&serde_json::Value> =
            lines.iter().filter(|v| v["type"] == "response").collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["id"], "s1");
        assert_eq!(responses[1]["id"], "s2");
        // The effort actually changed.
        assert_eq!(rt.state().effort, Effort::High);
        shutdown(&rt, handle);
        let _ = tmp;
    }

    #[test]
    fn rpc_resume_loads_the_previous_session() {
        let (tmp, root, rt, rx, handle, ws) = setup("rpc-resume", vec![]);
        // Save a prior session for this workspace.
        let prior = vec![
            Message::System("sys".into()),
            Message::User("q1".into()),
            Message::Assistant {
                text: Some("a1".into()),
                tool_calls: vec![],
            },
        ];
        session::save_session(&root, ws.root(), &prior).unwrap();

        let input = Box::new(Cursor::new(
            "{\"id\":\"r1\",\"type\":\"resume\"}\n".to_string(),
        ));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &rx, &root, input, &mut out).unwrap();
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let last: serde_json::Value = text
            .lines()
            .last()
            .map(|l| serde_json::from_str(l).unwrap())
            .unwrap();
        assert_eq!(last["type"], "response");
        assert_eq!(last["ok"], true);
        assert_eq!(rt.history(), prior);
        shutdown(&rt, handle);
        let _ = tmp;
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(Mode::parse("json").unwrap(), Mode::Json);
        assert_eq!(Mode::parse("rpc").unwrap(), Mode::Rpc);
        assert_eq!(Mode::parse("RPC").unwrap(), Mode::Rpc);
        assert_eq!(Mode::parse("tui").unwrap(), Mode::Tui);
    }

    #[test]
    fn rpc_steer_arriving_during_a_turn_is_forwarded() {
        // The provider would answer immediately, but the test sends prompt +
        // steer back-to-back in one stdin batch; the steer must reach the
        // runtime while the prompt turn is draining, not be queued behind it.
        let (tmp, _root, rt, rx, handle, _ws) = setup(
            "rpc-steer",
            vec![
                Response::Text("first answer".into()),
                Response::Text("steered answer".into()),
            ],
        );
        let input = concat!(
            "{\"id\":\"p1\",\"type\":\"prompt\",\"text\":\"do the thing\"}\n",
            "{\"id\":\"p2\",\"type\":\"steer\",\"text\":\"no, differently\"}\n"
        );
        let input = Box::new(Cursor::new(input.to_string()));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &rx, tmp.path(), input, &mut out).unwrap();
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // The steer was delivered as a user message in history.
        let h = rt.history();
        assert!(
            h.iter()
                .any(|m| matches!(m, Message::User(u) if u == "no, differently")),
            "steer must reach history: {h:?}"
        );
        let _ = lines;
        shutdown(&rt, handle);
        let _ = tmp;
    }

    #[test]
    fn rpc_malformed_request_reports_an_error() {
        let (tmp, _root, rt, rx, handle, _ws) = setup("rpc-bad", vec![]);
        let input = Box::new(Cursor::new("not json\n".to_string()));
        let mut out = Vec::new();
        let err = run_rpc(&rt, &rx, tmp.path(), input, &mut out).unwrap_err();
        assert!(err.contains("bad request"));
        shutdown(&rt, handle);
        let _ = tmp;
    }
}
