//! Headless frontends: synchronous adapters over AgentRuntime for
//! scripted/non-interactive use, selected with `--mode`. Interactive use is
//! the TUI (default on a terminal; the REPL was removed). Each
//! adapter consumes only the runtime's Event stream and Command surface — no
//! loop internals.
//!
//! - `json`: one-shot; runs one prompt and emits **every** runtime Event as
//!   a JSONL object to stdout (the same serialization the TUI and server
//!   will use) — a headless diagnostic view.
//! - `rpc`: a JSONL command/event loop over stdin/stdout. Each request is a
//!   `Command` JSON object on its own line; the runtime's events stream back
//!   as JSONL. When a request carries an `id`, the adapter emits a terminal
//!   `{"type":"response","id":...}` line after the command's events so
//!   clients can correlate replies (pi's RPC mode).
//!
//! ## Exit codes
//!
//! Both headless modes exit `0` only when the command completed. A failed turn
//! exits `1` (provider error, rejected model, authentication, or an aborted
//! turn) and a turn that ran out of iteration budget exits `2`. The emitted
//! events remain the source of truth — a failing turn still streams its
//! `Event::Error` — but a script that only looks at `$?` is no longer told a
//! dead turn succeeded.
//!

use std::io::{BufRead, Write};
use std::path::Path;

use daedalus_core::runtime::{AgentRuntime, Command, CommandKind, ErrorKind, Event};
use daedalus_core::session::{self, SessionSaver};

/// Persist the runtime's current history through `saver` when there is a
/// conversation worth keeping. Best-effort: a failure must never break a turn.
fn persist(rt: &AgentRuntime, root: &Path, saver: &mut SessionSaver) {
    let history = rt.history();
    if session::has_conversation(&history) {
        let _ = saver.save(root, &rt.workspace_root(), &history);
    }
}

/// The runtime's event receiver type (tokio unbounded channel).
type EventRx = tokio::sync::mpsc::UnboundedReceiver<Event>;

/// What one drain saw: whether the command failed, and how. The runtime reports
/// a failed turn as `Event::Error` on the same stream as everything else, so the
/// adapters would otherwise have to re-read the JSONL they just wrote to notice;
/// this carries that verdict out of the drain instead.
#[derive(Debug, Default, Clone)]
struct DrainOutcome {
    /// True once any `Event::Error` was seen. Such a turn ends without an
    /// answer, and the adapters must not report it as completed.
    errored: bool,
    /// Classification of the error that ended the turn. A later error does not
    /// overwrite it: the first one explains why the turn stopped, and a tool
    /// that merely failed while the turn continued is not a terminal error.
    kind: Option<ErrorKind>,
}

impl DrainOutcome {
    /// Whether the command completed without a runtime error.
    fn ok(&self) -> bool {
        !self.errored
    }

    /// Fold one event into the outcome.
    fn observe(&mut self, event: &Event) {
        if let Event::Error { kind, .. } = event {
            self.errored = true;
            if self.kind.is_none() {
                self.kind = kind.clone();
            }
        }
    }
}

/// The process exit code for a failed drain: `2` for the iteration cap (the
/// turn was cut short by a budget the user can raise) and `1` for every other
/// failure.
fn exit_code_for(outcome: &DrainOutcome) -> i32 {
    match outcome.kind {
        Some(ErrorKind::IterationCap) => 2,
        _ => 1,
    }
}

/// Blocking receive with a timeout (tokio receivers have no
/// `blocking_recv_timeout`; a poll loop keeps the semantics).
fn recv_timeout<T>(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<T>,
    timeout: std::time::Duration,
) -> Option<T> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match rx.try_recv() {
            Ok(v) => return Some(v),
            Err(_) if std::time::Instant::now() > deadline => return None,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    }
}

/// The frontend selected by `--mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Json,
    Rpc,
    Tui,
}

impl Mode {
    /// Parse a `--mode` value. `repl` gets its own migration note:
    /// the REPL was removed and the TUI is the interactive frontend.
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Mode::Json),
            "rpc" => Ok(Mode::Rpc),
            "tui" => Ok(Mode::Tui),
            "repl" => Err("the REPL was removed; the TUI is the interactive \
                 frontend and the default on a terminal"
                .into()),
            other => Err(format!(
                "unknown mode '{other}' (supported: json, rpc, tui)"
            )),
        }
    }
}

/// One-shot json: run `prompt` and emit every runtime Event as a JSONL line
/// until `agent_settled`. Returns the process exit code (`0` completed, `1`
/// failed, `2` iteration cap).
pub fn run_json(
    rt: &AgentRuntime,
    rx: &mut EventRx,
    root: &Path,
    prompt: &str,
    out: &mut dyn Write,
) -> Result<i32, String> {
    rt.prompt(prompt);
    let outcome = drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
    // Persist the completed turn so `--mode json` sessions are resumable too.
    let mut saver = SessionSaver::new();
    persist(rt, root, &mut saver);
    Ok(if outcome.errored {
        exit_code_for(&outcome)
    } else {
        0
    })
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
/// Returns the process exit code (`0` on EOF / normal shutdown; the first
/// failed request decides otherwise).
pub fn run_rpc(
    rt: &AgentRuntime,
    rx: &mut EventRx,
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

    let mut exit = 0i32;
    let mut saver = SessionSaver::new();
    loop {
        let command = match cmd_rx.recv() {
            Ok(Ok(command)) => command,
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        };
        let id = command.id.clone();
        let mut responded = false;
        let mut outcome = DrainOutcome::default();
        match &command.kind {
            CommandKind::Prompt { text } => {
                rt.prompt(text);
                outcome = drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
                persist(rt, root, &mut saver);
            }
            CommandKind::Steer { text } => {
                // Arrived while idle: it starts a turn.
                rt.steer(text);
                outcome = drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
                persist(rt, root, &mut saver);
            }
            CommandKind::FollowUp { text } => {
                rt.follow_up(text);
                outcome = drain_until(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
                persist(rt, root, &mut saver);
            }
            CommandKind::Abort {} => {
                rt.abort();
                outcome = drain_until_quiet(rx, out, |e| matches!(e, Event::AgentSettled { .. }))?;
                // Keep whatever partial progress the aborted turn made.
                persist(rt, root, &mut saver);
            }
            CommandKind::SetModel { model } => {
                rt.set_model(model);
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::SetProvider { provider } => {
                rt.set_provider(provider);
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::SetEffort { effort } => {
                rt.set_effort(*effort);
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::SwitchWorkspace { path } => {
                rt.switch_workspace(path);
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::Clear {} => {
                rt.clear();
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::GetState {} => {
                rt.get_state();
                outcome = drain_until(rx, out, |e| matches!(e, Event::StateChanged { .. }))?;
            }
            CommandKind::ListModels {} => {
                rt.refresh_models();
                outcome = drain_until(rx, out, |e| matches!(e, Event::ModelsListed { .. }))?;
            }
            CommandKind::Resume => {
                let history = daedalus_core::session::load_previous(root, &rt.workspace_root())
                    .map_err(|e| format!("could not load previous session: {e}"))?;
                match history {
                    Some(h)
                        if h.iter().any(|m| {
                            matches!(m, daedalus_core::provider::Message::Assistant { .. })
                        }) =>
                    {
                        rt.replace_history(h);
                        write_response(out, id.as_deref(), true)?;
                        responded = true;
                    }
                    _ => {
                        write_response(out, id.as_deref(), false)?;
                        responded = true;
                        // Nothing to resume is a failed request, not a quiet
                        // success: the client asked for a session that is not
                        // there, so the process reports it too.
                        outcome.errored = true;
                    }
                }
            }
        }
        if !responded {
            // The correlated response mirrors whether the command worked, so
            // a client driving via `id` sees the failure the same way the exit
            // code reports it.
            write_response(out, id.as_deref(), outcome.ok())?;
        }
        if outcome.errored && exit == 0 {
            // Keep serving the remaining requests (a client may be probing),
            // but remember the first failure: the final exit code must not
            // claim the whole session succeeded.
            exit = exit_code_for(&outcome);
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    let _ = reader.join();
    Ok(exit)
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
fn drain_until(
    rx: &mut EventRx,
    out: &mut dyn Write,
    stop: impl Fn(&Event) -> bool,
) -> Result<DrainOutcome, String> {
    let mut outcome = DrainOutcome::default();
    loop {
        match rx.blocking_recv() {
            Some(event) => {
                let line = serde_json::to_string(&event).map_err(|e| e.to_string())?;
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
                outcome.observe(&event);
                if stop(&event) {
                    return Ok(outcome);
                }
            }
            None => return Ok(outcome),
        }
    }
}

/// Drain events like `drain_until`, but give up after a short quiet period so
/// an abort while idle (no settle follows) cannot hang the adapter.
fn drain_until_quiet(
    rx: &mut EventRx,
    out: &mut dyn Write,
    stop: impl Fn(&Event) -> bool,
) -> Result<DrainOutcome, String> {
    let mut outcome = DrainOutcome::default();
    loop {
        match recv_timeout(rx, std::time::Duration::from_millis(100)) {
            Some(event) => {
                let line = serde_json::to_string(&event).map_err(|e| e.to_string())?;
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
                outcome.observe(&event);
                if stop(&event) {
                    return Ok(outcome);
                }
            }
            None => return Ok(outcome),
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
    use daedalus_core::config::Config;
    use daedalus_core::provider::fake::FakeProvider;
    use daedalus_core::provider::{Message, Response, ToolCall};
    use daedalus_core::runtime::Effort;
    use daedalus_core::session;
    use daedalus_core::tools::resolver::ToolSet;
    use daedalus_core::workspace::Workspace;
    use std::io::Cursor;
    use std::path::PathBuf;

    /// A unique temp dir cleaned up on drop (tempfile).
    type TempDir = tempfile::TempDir;

    fn setup(
        _name: &str,
        responses: Vec<Response>,
    ) -> (
        TempDir,
        PathBuf,
        AgentRuntime,
        EventRx,
        std::thread::JoinHandle<()>,
        Workspace,
    ) {
        setup_with(_name, responses, 10)
    }

    /// Like `setup`, with an explicit iteration budget (to drive a turn into
    /// the cap).
    fn setup_with(
        _name: &str,
        responses: Vec<Response>,
        max_iterations: usize,
    ) -> (
        TempDir,
        PathBuf,
        AgentRuntime,
        EventRx,
        std::thread::JoinHandle<()>,
        Workspace,
    ) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().join("sessions");
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let cfg = Config {
            max_iterations,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let tools = ToolSet::new(1000);
        let (rt, mut rx) = AgentRuntime::new(
            cfg,
            Box::new(FakeProvider::new(responses)),
            tools,
            ws.clone(),
        );
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        // Drain the initial agent_start so tests see events from their command.
        let _ = recv_timeout(&mut rx, std::time::Duration::from_secs(2));
        (dir, root, rt, rx, handle, ws)
    }

    fn shutdown(rt: &AgentRuntime, handle: std::thread::JoinHandle<()>) {
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn json_mode_emits_every_event_as_jsonl() {
        let (tmp, _root, rt, mut rx, handle, _ws) =
            setup("json", vec![Response::Text("hello world".into())]);
        let mut out = Vec::new();
        let code = run_json(&rt, &mut rx, &_root, "greet", &mut out).unwrap();
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
        let (tmp, _root, rt, mut rx, handle, _ws) =
            setup("rpc-prompt", vec![Response::Text("the answer".into())]);
        let input = Box::new(Cursor::new(
            "{\"id\":\"1\",\"type\":\"prompt\",\"text\":\"what is 2+2\"}\n".to_string(),
        ));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &mut rx, tmp.path(), input, &mut out).unwrap();
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
        let (tmp, _root, rt, mut rx, handle, _ws) = setup("rpc-state", vec![]);
        let input = concat!(
            "{\"id\":\"s1\",\"type\":\"get_state\"}\n",
            "{\"id\":\"s2\",\"type\":\"set_effort\",\"effort\":\"high\"}\n",
            "{\"type\":\"abort\"}\n"
        );
        let input = Box::new(Cursor::new(input.to_string()));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &mut rx, tmp.path(), input, &mut out).unwrap();
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
        let (tmp, root, rt, mut rx, handle, ws) = setup("rpc-resume", vec![]);
        // Save a prior session for this workspace.
        let prior = vec![
            Message::System("sys".into()),
            Message::User("q1".into()),
            Message::Assistant {
                text: Some("a1".into()),
                tool_calls: vec![],
                reasoning: vec![],
            },
        ];
        session::save_session(&root, ws.root(), &prior).unwrap();

        let input = Box::new(Cursor::new(
            "{\"id\":\"r1\",\"type\":\"resume\"}\n".to_string(),
        ));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &mut rx, &root, input, &mut out).unwrap();
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
        let (tmp, _root, rt, mut rx, handle, _ws) = setup(
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
        let code = run_rpc(&rt, &mut rx, tmp.path(), input, &mut out).unwrap();
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
        let (tmp, _root, rt, mut rx, handle, _ws) = setup("rpc-bad", vec![]);
        let input = Box::new(Cursor::new("not json\n".to_string()));
        let mut out = Vec::new();
        let err = run_rpc(&rt, &mut rx, tmp.path(), input, &mut out).unwrap_err();
        assert!(err.contains("bad request"));
        shutdown(&rt, handle);
        let _ = tmp;
    }

    /// The iteration cap is the one failure a caller can act on (raise the
    /// budget), so it gets a distinct code; every other failure is a plain `1`.
    #[test]
    fn failure_exit_codes_distinguish_the_cap() {
        let cap = DrainOutcome {
            errored: true,
            kind: Some(ErrorKind::IterationCap),
        };
        assert_eq!(exit_code_for(&cap), 2);
        let auth = DrainOutcome {
            errored: true,
            kind: Some(ErrorKind::Auth),
        };
        assert_eq!(exit_code_for(&auth), 1);
        // An unclassified failure (a tool error the runtime only renders) is
        // still a failure.
        let unclassified = DrainOutcome {
            errored: true,
            kind: None,
        };
        assert_eq!(exit_code_for(&unclassified), 1);
        assert!(!unclassified.ok());
        assert!(DrainOutcome::default().ok());
    }

    /// A turn killed by the iteration cap used to exit 0: scripts could not
    /// tell a dead run from a finished one. The events keep the detail, the
    /// exit code answers "did this work".
    #[test]
    fn json_mode_exits_2_on_the_iteration_cap() {
        // A model that only ever asks for tools never answers.
        let tool_call = Response::ToolCalls(vec![ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            args: serde_json::json!({"command": "true"}),
        }]);
        let (tmp, _root, rt, mut rx, handle, _ws) = setup_with("json-cap", vec![tool_call; 3], 2);
        let mut out = Vec::new();
        let code = run_json(&rt, &mut rx, &_root, "loop forever", &mut out).unwrap();
        assert_eq!(code, 2, "the cap must not look like success");
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let error = lines
            .iter()
            .find(|v| v["type"] == "error")
            .unwrap_or_else(|| panic!("the failure is still streamed: {text}"));
        assert_eq!(error["kind"]["reason"], "iteration_cap");
        // The stream still ends in a settle: the exit code is the verdict, not
        // a replacement for the event vocabulary.
        assert_eq!(lines.last().unwrap()["type"], "agent_settled");
        shutdown(&rt, handle);
        let _ = tmp;
    }

    /// Likewise in rpc: a request that cannot be satisfied answers `ok: false`
    /// (so a client can react per-id) *and* moves the exit code off 0.
    #[test]
    fn rpc_failed_request_answers_not_ok_and_exits_nonzero() {
        let (tmp, root, rt, mut rx, handle, _ws) = setup("rpc-resume-none", vec![]);
        // An existing session directory with no saved session in it: resume has
        // nothing to restore.
        std::fs::create_dir_all(&root).unwrap();
        let input = Box::new(Cursor::new(
            "{\"id\":\"r1\",\"type\":\"resume\"}\n".to_string(),
        ));
        let mut out = Vec::new();
        let code = run_rpc(&rt, &mut rx, &root, input, &mut out).unwrap();
        assert_eq!(code, 1, "resuming nothing is a failure, not a no-op");
        let text = String::from_utf8(out).unwrap();
        let last: serde_json::Value = text
            .lines()
            .last()
            .map(|l| serde_json::from_str(l).unwrap())
            .unwrap();
        assert_eq!(last["type"], "response");
        assert_eq!(last["id"], "r1");
        assert_eq!(last["ok"], false);
        shutdown(&rt, handle);
        let _ = tmp;
    }
}
