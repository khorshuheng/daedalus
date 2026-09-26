//! End-to-end testing with the fake provider.
//!
//! No network required: drives the real agent loop with an in-memory scripted
//! provider against a temp workspace, plus a CLI smoke test.

use std::path::Path;
use std::process::Command;

use daedalus_core::config::{provider_by_name, Config};
use daedalus_core::provider::fake::FakeProvider;
use daedalus_core::provider::{Message, Response, ToolCall};
use daedalus_core::runtime::{AgentRuntime, RuntimeError};
use daedalus_core::tools::resolver::ToolSet;
use daedalus_core::workspace::Workspace;

/// A unique temp dir cleaned up on drop (tempfile).
type TempDir = tempfile::TempDir;
fn tempdir(_name: &str) -> TempDir {
    tempfile::tempdir().expect("create temp dir")
}

fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args,
    }
}

fn config_for(dir: &Path, max_iterations: usize) -> Config {
    Config {
        provider: provider_by_name("fake").unwrap(),
        max_iterations,
        workspace: dir.to_path_buf(),
        ..Config::defaults(dir.to_path_buf())
    }
}

/// A full multi-step session: write, read, edit, bash, then answer. Verifies
/// the tools ran against the temp workspace and the loop fed results back.
#[test]
fn integration_multi_step_session() {
    let tmp = tempdir("multi");
    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let cfg = config_for(tmp.path(), 10);

    let fake = FakeProvider::new(vec![
        Response::ToolCalls(vec![call(
            "w1",
            "write",
            serde_json::json!({"path": "a.txt", "content": "hello world"}),
        )]),
        Response::ToolCalls(vec![call(
            "r1",
            "read",
            serde_json::json!({"path": "a.txt"}),
        )]),
        Response::ToolCalls(vec![call(
            "e1",
            "edit",
            serde_json::json!({"path": "a.txt", "oldText": "world", "newText": "there"}),
        )]),
        Response::ToolCalls(vec![call(
            "b1",
            "bash",
            serde_json::json!({"command": "grep -c there a.txt"}),
        )]),
        Response::Text("all done".into()),
    ]);

    let tools = ToolSet::new(1000);
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws);

    let answer = rt.run_once("create a file and edit it").unwrap();
    assert_eq!(answer, "all done");

    // The tools actually mutated the workspace.
    assert_eq!(
        std::fs::read_to_string(tmp.path().join("a.txt")).unwrap(),
        "hello there"
    );

    // Every tool result was fed back (the final history includes the bash
    // result of the last tool call).
    let h = rt.history();
    assert!(h.iter().any(|m| {
        matches!(m, Message::ToolResult { result, .. } if result.contains("exit code: 0"))
    }));
}

/// Bounded-loop failure path via the lib API: an all-tool-calls provider must
/// terminate with an iteration-cap error, not run forever.
#[test]
fn integration_iteration_cap() {
    let tmp = tempdir("cap");
    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let cfg = config_for(tmp.path(), 3);

    let fake = FakeProvider::new(vec![
        Response::ToolCalls(vec![call(
            "a",
            "bash",
            serde_json::json!({"command": "true"}),
        )]),
        Response::ToolCalls(vec![call(
            "b",
            "bash",
            serde_json::json!({"command": "true"}),
        )]),
        Response::ToolCalls(vec![call(
            "c",
            "bash",
            serde_json::json!({"command": "true"}),
        )]),
        Response::ToolCalls(vec![call(
            "d",
            "bash",
            serde_json::json!({"command": "true"}),
        )]),
    ]);
    let tools = ToolSet::new(1000);
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws);

    let err = rt.run_once("keep going").unwrap_err();
    assert!(matches!(err, RuntimeError::IterationCap(3)));
}

/// CLI smoke: non-interactive stdin without an explicit --mode is rejected
/// with a pointer to the headless modes (print mode was removed; daedalus is
/// interactive by default).
#[test]
fn cli_smoke_piped_without_mode_is_rejected() {
    let tmp = tempdir("cli-no-mode");
    let out = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args(["hello", "--provider", "fake", "--model", "test", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run daedalus binary");
    assert!(
        !out.status.success(),
        "piped stdin without --mode must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not a terminal"), "{stderr}");
    assert!(stderr.contains("--mode json"), "{stderr}");
    assert!(stderr.contains("--mode rpc"), "{stderr}");
}

/// CLI smoke: `--mode repl` must fail at load with the migration note
/// (the REPL was removed).
#[test]
fn cli_smoke_repl_mode_is_removed() {
    let tmp = tempdir("cli-repl-removed");
    let out = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args(["hello", "--mode", "repl", "--provider", "fake", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run daedalus binary");
    assert!(!out.status.success(), "--mode repl must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("REPL was removed"), "{stderr}");
    assert!(stderr.contains("TUI"), "{stderr}");
}

/// CLI smoke: an unknown provider must fail fast with a non-zero exit.
#[test]
fn cli_smoke_unknown_provider_fails() {
    let tmp = tempdir("cli-bad");
    let out = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args(["hi", "--provider", "nope", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run daedalus binary");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown provider"));
    assert!(stderr.contains("fake"));
}

/// Paths outside the workspace are allowed (the workspace guard was removed in
/// favour of the bash timeout + prompt guidance).
#[test]
fn integration_path_outside_workspace_is_allowed() {
    let tmp = tempdir("outside");
    let outside = std::env::temp_dir().join(format!("daedalus-outside-it-{}", std::process::id()));
    let _ = std::fs::remove_file(&outside);

    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let cfg = config_for(tmp.path(), 5);
    let fake = FakeProvider::new(vec![
        Response::ToolCalls(vec![call(
            "e1",
            "write",
            serde_json::json!({"path": outside.to_string_lossy(), "content": "ok"}),
        )]),
        Response::Text("done".into()),
    ]);
    let tools = ToolSet::new(1000);
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws);
    let _ = rt.run_once("write outside").unwrap();

    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "ok");
    std::fs::remove_file(&outside).ok();
}

/// CLI smoke: `--mode rpc` drives a full session over stdin/stdout with no
/// terminal: send a prompt request with an id, expect events plus
/// a correlated response line.
#[test]
fn cli_rpc_mode_drives_a_session_over_stdio() {
    let tmp = tempdir("rpc");
    let script = r#"{"id":"1","type":"prompt","text":"say hi"}
"#;
    let mut child = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args([
            "--mode",
            "rpc",
            "--provider",
            "fake",
            "--model",
            "test",
            "--dir",
        ])
        .arg(tmp.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn daedalus rpc");
    use std::io::Write as _;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "rpc exited {}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!lines.is_empty());
    // Events stream, ending with the correlated response carrying our id.
    let last = lines.last().unwrap();
    assert_eq!(last["type"], "response");
    assert_eq!(last["id"], "1");
    assert!(lines.iter().any(|l| l["type"] == "agent_start"));
    assert!(lines.iter().any(|l| l["type"] == "agent_settled"));
}

/// CLI smoke: `--mode json` emits every Event as a JSONL line.
#[test]
fn cli_json_mode_emits_events_as_jsonl() {
    let tmp = tempdir("json");
    let out = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args([
            "hello",
            "--mode",
            "json",
            "--provider",
            "fake",
            "--model",
            "test",
            "--dir",
        ])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run daedalus json");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(lines.iter().any(|l| l["type"] == "agent_start"));
    // JSON output is the event stream, not a bare answer line.
    assert!(lines.iter().any(|l| l["type"] == "agent_settled"));
}

/// CLI smoke: unknown --mode fails fast.
#[test]
fn cli_unknown_mode_fails() {
    let tmp = tempdir("mode-bad");
    let out = Command::new(env!("CARGO_BIN_EXE_dd"))
        .args(["hi", "--mode", "nope", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run daedalus");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown mode"));
    assert!(stderr.contains("json, rpc, tui"));
}

/// The provider-reported prompt token count must flow through the
/// adapter into the runtime's `usage` event (context budgeting's anchor).
#[test]
fn usage_event_carries_provider_reported_prompt_tokens() {
    use daedalus_core::provider::{Completion, Provider, ProviderError, StreamDelta};
    use std::sync::Mutex;

    struct TokenReporter;
    impl Provider for TokenReporter {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async {
                Ok(Completion {
                    response: Response::Text("done".into()),
                    prompt_tokens: Some(4321),
                    aborted: false,
                })
            })
        }
    }
    let _ = Mutex::new(()); // no shared state; type anchor only

    let tmp = tempdir("usage-flow");
    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let cfg = config_for(tmp.path(), 5);
    let (rt, mut rx) = AgentRuntime::new(cfg, Box::new(TokenReporter), ToolSet::new(1000), ws);
    let worker = rt.clone();
    let handle = std::thread::spawn(move || worker.run_forever());
    rt.prompt("hi");
    let mut saw_usage = None;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while saw_usage.is_none() && std::time::Instant::now() < deadline {
        match rx.try_recv() {
            Ok(daedalus_core::runtime::Event::Usage { prompt_tokens }) => {
                saw_usage = prompt_tokens;
            }
            Ok(_) => {}
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
        }
    }
    rt.shutdown();
    handle.join().unwrap_or(());
    assert_eq!(
        saw_usage,
        Some(4321),
        "prompt_tokens must reach the usage event"
    );
}
