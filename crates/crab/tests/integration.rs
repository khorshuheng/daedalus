//! End-to-end testing with the fake provider (CRAB-106).
//!
//! No network required: drives the real agent loop with an in-memory scripted
//! provider against a temp workspace, plus a CLI smoke test.

use std::path::Path;
use std::process::Command;

use crab_core::config::{Config, ProviderKind};
use crab_core::provider::fake::FakeProvider;
use crab_core::provider::{Message, Response, ToolCall};
use crab_core::runtime::{AgentRuntime, RuntimeError};
use crab_core::tools::resolver::ToolSet;
use crab_core::workspace::Workspace;

/// A unique temp dir cleaned up on drop (tempfile, CRAB-119).
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
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws, None);

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
    let (rt, _rx) = AgentRuntime::new(cfg, Box::new(fake), tools, ws, None);

    let err = rt.run_once("keep going").unwrap_err();
    assert!(matches!(err, RuntimeError::IterationCap(3)));
}

/// CLI smoke: non-interactive stdin without an explicit --mode is rejected
/// with a pointer to the headless modes (print mode was removed; crab is
/// interactive by default).
#[test]
fn cli_smoke_piped_without_mode_is_rejected() {
    let tmp = tempdir("cli-no-mode");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hello", "--provider", "fake", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab binary");
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
/// (CRAB-135: the REPL was removed).
#[test]
fn cli_smoke_repl_mode_is_removed() {
    let tmp = tempdir("cli-repl-removed");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hello", "--mode", "repl", "--provider", "fake", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab binary");
    assert!(!out.status.success(), "--mode repl must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("REPL was removed"), "{stderr}");
    assert!(stderr.contains("TUI"), "{stderr}");
}

/// CLI smoke: an unknown provider must fail fast with a non-zero exit.
#[test]
fn cli_smoke_unknown_provider_fails() {
    let tmp = tempdir("cli-bad");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hi", "--provider", "nope", "--dir"])
        .arg(tmp.path())
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
    let tmp = tempdir("escape");
    // Create an outside file to prove the tool never touches it.
    let outside = std::env::temp_dir().join(format!("crab-outside-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::write(&outside, "secret").unwrap();

    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let cfg = config_for(tmp.path(), 5);
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

/// CLI smoke: `--mode rpc` drives a full session over stdin/stdout with no
/// terminal (CRAB-120): send a prompt request with an id, expect events plus
/// a correlated response line.
#[test]
fn cli_rpc_mode_drives_a_session_over_stdio() {
    let tmp = tempdir("rpc");
    let script = r#"{"id":"1","type":"prompt","text":"say hi"}
"#;
    let mut child = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["--mode", "rpc", "--provider", "fake", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn crab rpc");
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

/// CLI smoke: `--mode json` emits every Event as a JSONL line (CRAB-120).
#[test]
fn cli_json_mode_emits_events_as_jsonl() {
    let tmp = tempdir("json");
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hello", "--mode", "json", "--provider", "fake", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab json");
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
    let out = Command::new(env!("CARGO_BIN_EXE_crab"))
        .args(["hi", "--mode", "nope", "--dir"])
        .arg(tmp.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run crab");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown mode"));
    assert!(stderr.contains("json, rpc, tui"));
}

/// End-to-end memory pipeline (CRAB-111 epic acceptance): a session is
/// reflected into lessons, lessons persist in the JSONL log, the SQLite
/// index is rebuilt from them, and a later, similar task injects the relevant
/// lesson into its system prompt.
#[test]
fn memory_pipeline_reflect_store_index_inject() {
    use crab_core::reflect;
    use crab_core::runtime::AgentRuntime;

    let tmp = tempdir("memory-pipeline");
    let root = tmp.path().join("memory");
    let ws = Workspace::new(tmp.path().to_path_buf()).unwrap();
    let tools = ToolSet::new(1000);

    // 1. A session where the agent learned to build with make.
    let transcript = vec![
        Message::System("you are crab".into()),
        Message::User("how do i build this project".into()),
        Message::Assistant {
            text: Some("run make, not cargo".into()),
            tool_calls: vec![],
        },
    ];

    // 2. Reflect over it: the provider answers with a JSON lesson list.
    let fake = FakeProvider::new(vec![Response::Text(
        r#"[{"text":"build with make, never cargo","kind":"rule","tags":["build"]}]"#.into(),
    )]);
    let added = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(reflect::reflect_and_store(
            &root,
            ws.root(),
            &fake,
            tokio_util::sync::CancellationToken::new(),
            &transcript,
            Some("sess-mem".into()),
            100,
        ))
        .expect("reflect should succeed");
    assert_eq!(added, 1);

    // The JSONL log is the source of truth.
    let stored = crab_core::memory::list_lessons(&root, ws.root()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].text, "build with make, never cargo");
    assert_eq!(stored[0].source_session_id.as_deref(), Some("sess-mem"));

    // 3. Build the SQLite index from the log and confirm retrieval.
    let synced = crab_core::index::sync(&root, ws.root()).unwrap();
    assert_eq!(synced, 1);
    let hits = crab_core::index::search(&root, ws.root(), "how do i build", 5).unwrap();
    assert!(
        hits.iter().any(|l| l.text.contains("make")),
        "index should return the lesson for a build query: {hits:?}"
    );

    // 4. A later, similar session (new runtime, same memory root) injects the
    // lesson into its system prompt.
    let cfg = Config {
        provider: ProviderKind::Fake,
        max_iterations: 5,
        workspace: tmp.path().to_path_buf(),
        ..Config::defaults(tmp.path().to_path_buf())
    };
    let later_provider = FakeProvider::new(vec![Response::Text("ok".into())]);
    let (rt, _rx) = AgentRuntime::new(
        cfg,
        Box::new(later_provider),
        tools,
        ws.clone(),
        Some(root.clone()),
    );
    let answer = rt.run_once("how do i build").unwrap();
    assert_eq!(answer, "ok");
    let h = rt.history();
    let Some(Message::System(system)) = h.first() else {
        panic!("history must start with a system prompt");
    };
    assert!(
        system.contains("build with make, never cargo"),
        "later session should inject the lesson: {system}"
    );
}

/// CRAB-130: the provider-reported prompt token count must flow through the
/// adapter into the runtime's `usage` event (context budgeting's anchor).
#[test]
fn usage_event_carries_provider_reported_prompt_tokens() {
    use crab_core::provider::{Completion, Provider, ProviderError};
    use std::sync::Mutex;

    struct TokenReporter;
    impl Provider for TokenReporter {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_text: &'a mut (dyn FnMut(&str) + Send),
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
    let (rt, mut rx) =
        AgentRuntime::new(cfg, Box::new(TokenReporter), ToolSet::new(1000), ws, None);
    let worker = rt.clone();
    let handle = std::thread::spawn(move || worker.run_forever());
    rt.prompt("hi");
    let mut saw_usage = None;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while saw_usage.is_none() && std::time::Instant::now() < deadline {
        match rx.try_recv() {
            Ok(crab_core::runtime::Event::Usage { prompt_tokens }) => {
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
