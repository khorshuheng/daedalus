//! End-to-end testing with the fake provider (CRAB-106).
//!
//! No network required: drives the real agent loop with an in-memory scripted
//! provider against a temp workspace, plus a CLI smoke test.

use std::path::{Path, PathBuf};
use std::process::Command;

use crab::config::{Config, ProviderKind};
use crab::provider::fake::FakeProvider;
use crab::provider::{Message, Response, ToolCall};
use crab::runtime::{AgentRuntime, RuntimeError};
use crab::tools::resolver::ToolSet;
use crab::workspace::Workspace;

/// RAII guard removing the temp dir on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!("crab-it-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        TempDir(base)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
        provider: ProviderKind::Fake,
        max_iterations,
        workspace: dir.to_path_buf(),
        ..Config::defaults(dir.to_path_buf())
    }
}

/// A full multi-step session: write, read, edit, bash, then answer. Verifies
/// the tools ran against the temp workspace and the loop fed results back.
#[test]
fn integration_multi_step_session() {
    let tmp = TempDir::new("multi");
    let ws = Workspace::new(tmp.0.clone()).unwrap();
    let cfg = config_for(&tmp.0, 10);

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
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws, None);

    let answer = rt.run_once("create a file and edit it").unwrap();
    assert_eq!(answer, "all done");

    // The tools actually mutated the workspace.
    assert_eq!(
        std::fs::read_to_string(tmp.0.join("a.txt")).unwrap(),
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
    let tmp = TempDir::new("cap");
    let ws = Workspace::new(tmp.0.clone()).unwrap();
    let cfg = config_for(&tmp.0, 3);

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
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws, None);

    let err = rt.run_once("keep going").unwrap_err();
    assert!(matches!(err, RuntimeError::IterationCap(3)));
}

/// CLI smoke test: `crab "hello" --provider fake --dir <tmp>` exits 0 and
/// prints a final answer.
#[test]
fn cli_smoke_fake_provider() {
    let tmp = TempDir::new("cli");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hello", "--provider", "fake", "--dir"])
        .arg(&tmp.0)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab binary");
    assert!(
        out.status.success(),
        "crab exited {}; stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("done"));
}

/// CLI smoke: an unknown provider must fail fast with a non-zero exit.
#[test]
fn cli_smoke_unknown_provider_fails() {
    let tmp = TempDir::new("cli-bad");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hi", "--provider", "nope", "--dir"])
        .arg(&tmp.0)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab binary");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown provider"));
    assert!(stderr.contains("fake"));
}

/// Path-escape through a tool call is rejected and handed back as an error.
#[test]
fn integration_path_escape_is_rejected() {
    let tmp = TempDir::new("escape");
    // Create an outside file to prove the tool never touches it.
    let outside = std::env::temp_dir().join(format!("crab-outside-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::write(&outside, "secret").unwrap();

    let ws = Workspace::new(tmp.0.clone()).unwrap();
    let cfg = config_for(&tmp.0, 5);
    let fake = FakeProvider::new(vec![
        Response::ToolCalls(vec![call(
            "e1",
            "write",
            serde_json::json!({"path": format!("../../{}", outside.file_name().unwrap().to_str().unwrap()), "content": "pwned"}),
        )]),
        Response::Text("ok".into()),
    ]);
    let tools = ToolSet::new(1000);
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws, None);
    let _ = rt.run_once("try to escape").unwrap();

    // The escape error was fed back, and the outside file is untouched.
    let h = rt.history();
    assert!(h.iter().any(|m| {
        matches!(m, Message::ToolResult { result, .. } if result.contains("escapes the workspace"))
    }));
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "secret");
    std::fs::remove_dir_all(&outside).ok();
}
