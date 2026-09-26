//! Offline integration tests for the WebSocket server: a real axum
//! server on an ephemeral loopback port, a real WebSocket client, and the
//! scripted fake provider — no network, no external services.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use daedalus_core::config::{provider_by_name, Config};
use daedalus_core::workspace::Workspace;
use daedalus_server::{build_router, AppState};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type Client = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

fn fake_config(workspace: &Path) -> Config {
    fake_config_with(workspace, &[])
}

/// Fake-provider config with an optional identity -> workspace map.
fn fake_config_with(workspace: &Path, identities: &[(&str, &Path)]) -> Config {
    let mut config = Config {
        provider: provider_by_name("fake").unwrap(),
        model: "fake-model".into(),
        workspace: workspace.to_path_buf(),
        ..Config::defaults(workspace.to_path_buf())
    };
    for (identity, path) in identities {
        config
            .identity_workspaces
            .insert((*identity).to_string(), path.to_path_buf());
    }
    config
}

async fn start_state(state: AppState) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, build_router(state)).await;
    });
    addr
}

/// Start the router on an ephemeral port; return its address.
async fn start(dir: &Path) -> SocketAddr {
    let workspace = Workspace::new(dir.to_path_buf()).unwrap();
    let state = AppState::new(fake_config(dir), workspace, dir.join("sessions"));
    start_state(state).await
}

async fn connect(addr: SocketAddr) -> Client {
    connect_raw(addr, "/ws", &[]).await
}

async fn connect_workspace(addr: SocketAddr, workspace: &Path) -> Client {
    let path = workspace.to_string_lossy().replace(' ', "%20");
    connect_raw(addr, &format!("/ws?workspace={path}"), &[]).await
}

/// Connect with extra handshake headers (e.g. the tailnet identity header).
async fn connect_raw(addr: SocketAddr, path: &str, headers: &[(&str, &str)]) -> Client {
    let mut request = format!("ws://{addr}{path}").into_client_request().unwrap();
    for (name, value) in headers {
        request.headers_mut().insert(
            HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
    }
    let (ws, _) = connect_async(request).await.unwrap();
    ws
}

async fn send(ws: &mut Client, value: Value) {
    ws.send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

async fn next_event(ws: &mut Client) -> Value {
    loop {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for an event")
            .expect("socket closed")
            .expect("websocket error");
        if let Message::Text(text) = msg {
            return serde_json::from_str(&text).expect("event is JSON");
        }
    }
}

/// Collect events until one of `kind` is seen (inclusive).
async fn until(ws: &mut Client, kind: &str) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = next_event(ws).await;
        let done = event["type"] == kind;
        events.push(event);
        if done {
            return events;
        }
    }
}

async fn http_get(addr: SocketAddr, path: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    String::from_utf8_lossy(&buf).into_owned()
}

#[tokio::test]
async fn health_reports_ok() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let response = http_get(addr, "/health").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"status\":\"ok\""), "{response}");
}

#[tokio::test]
async fn server_starts_with_an_mcp_server_configured() {
    // Regression: the tool set (and MCP connect) used to be built inline on the
    // axum worker, panicking with "Cannot start a runtime from within a runtime"
    // for every connection once any `[[mcp_servers]]` entry was enabled.
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
    let mut config = fake_config(dir.path());
    config
        .mcp_servers
        .push(daedalus_core::mcp::McpServerConfig {
            name: "missing".into(),
            enabled: true,
            command: Some("daedalus-no-such-mcp-binary".into()),
            ..Default::default()
        });
    let state = AppState::new(config, workspace, dir.path().join("sessions"));
    let addr = start_state(state).await;

    let mut ws = connect(addr).await;
    send(&mut ws, json!({"type": "prompt", "text": "hello"})).await;
    let events = until(&mut ws, "agent_settled").await;
    assert!(events.iter().any(|e| e["type"] == "turn_start"));
}

#[tokio::test]
async fn settled_session_is_persisted_for_resume() {
    // Regression DAE-108: the server used to never save, so `resume` could only
    // ever find sessions the TUI happened to write.
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    send(&mut ws, json!({"type": "prompt", "text": "hello"})).await;
    let _ = until(&mut ws, "agent_settled").await;

    let root = dir.path().join("sessions");
    let cwd = Workspace::new(dir.path().to_path_buf())
        .unwrap()
        .root()
        .to_path_buf();
    // The save runs right after the settled frame is written, so poll briefly.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let loaded = daedalus_core::session::load_previous(&root, &cwd).unwrap();
        if let Some(history) = loaded {
            assert!(
                history
                    .iter()
                    .any(|m| matches!(m, daedalus_core::provider::Message::Assistant { .. })),
                "the saved session must contain the answer: {history:?}"
            );
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the server never persisted the settled session"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn prompt_streams_events_and_settles() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    send(&mut ws, json!({"type": "prompt", "text": "hello"})).await;
    let events = until(&mut ws, "agent_settled").await;

    assert!(events.iter().any(|e| e["type"] == "turn_start"));
    let settled = events
        .iter()
        .find(|e| e["type"] == "agent_settled")
        .unwrap();
    assert_eq!(settled["text"], "done");
    assert_eq!(settled["interrupted"], false);
}

#[tokio::test]
async fn set_model_and_get_state_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    send(
        &mut ws,
        json!({"type": "set_model", "model": "other-model"}),
    )
    .await;
    send(&mut ws, json!({"type": "get_state"})).await;

    // A fresh session emits an initial state_changed on connect, so wait for
    // the one carrying the new model rather than the first one.
    loop {
        let event = next_event(&mut ws).await;
        if event["type"] == "state_changed" && event["model"] == "other-model" {
            break;
        }
    }
}

#[tokio::test]
async fn malformed_command_yields_error_event() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    ws.send(Message::Text("{not json".into())).await.unwrap();
    // The worker emits agent_start (and the session an initial state_changed)
    // first; read until the error event.
    let events = until(&mut ws, "error").await;
    let event = events.last().unwrap();
    assert_eq!(event["type"], "error");
    assert!(event["message"]
        .as_str()
        .unwrap()
        .contains("malformed command"));
}

#[tokio::test]
async fn id_carrying_command_gets_a_response_frame() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    send(&mut ws, json!({"type": "get_state", "id": "req-1"})).await;
    let events = until(&mut ws, "response").await;
    let response = events.last().unwrap();
    assert_eq!(response["type"], "response");
    assert_eq!(response["id"], "req-1");
    assert_eq!(response["ok"], true);
}

async fn state_workspace(ws: &mut Client) -> PathBuf {
    send(ws, json!({"type": "get_state"})).await;
    let event = until(ws, "state_changed").await.pop().unwrap();
    PathBuf::from(event["workspace"].as_str().unwrap())
}

#[tokio::test]
async fn identity_header_selects_the_mapped_workspace() {
    let server_dir = tempfile::tempdir().unwrap();
    let alice = tempfile::tempdir().unwrap();
    let config = fake_config_with(server_dir.path(), &[("alice@example.com", alice.path())]);
    let workspace = Workspace::new(server_dir.path().to_path_buf()).unwrap();
    let state = AppState::new(config, workspace, server_dir.path().join("sessions"));
    let addr = start_state(state).await;

    let mut ws = connect_raw(
        addr,
        "/ws",
        &[("tailscale-user-login", "alice@example.com")],
    )
    .await;
    assert_eq!(
        state_workspace(&mut ws).await,
        alice.path().canonicalize().unwrap()
    );
}

#[tokio::test]
async fn identity_mapping_wins_over_the_query_override() {
    let server_dir = tempfile::tempdir().unwrap();
    let alice = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let config = fake_config_with(server_dir.path(), &[("alice@example.com", alice.path())]);
    let workspace = Workspace::new(server_dir.path().to_path_buf()).unwrap();
    let state = AppState::new(config, workspace, server_dir.path().join("sessions"));
    let addr = start_state(state).await;

    let query = format!("/ws?workspace={}", other.path().to_string_lossy());
    let mut ws = connect_raw(
        addr,
        &query,
        &[("tailscale-user-login", "alice@example.com")],
    )
    .await;
    assert_eq!(
        state_workspace(&mut ws).await,
        alice.path().canonicalize().unwrap()
    );
}

#[tokio::test]
async fn unmapped_identity_is_rejected_when_a_map_exists() {
    let server_dir = tempfile::tempdir().unwrap();
    let alice = tempfile::tempdir().unwrap();
    let config = fake_config_with(server_dir.path(), &[("alice@example.com", alice.path())]);
    let workspace = Workspace::new(server_dir.path().to_path_buf()).unwrap();
    let state = AppState::new(config, workspace, server_dir.path().join("sessions"));
    let addr = start_state(state).await;

    let mut ws = connect_raw(addr, "/ws", &[("tailscale-user-login", "bob@example.com")]).await;
    let event = next_event(&mut ws).await;
    assert_eq!(event["type"], "error");
    assert!(event["message"]
        .as_str()
        .unwrap()
        .contains("no workspace mapped"));
}

#[tokio::test]
async fn mapped_identity_cannot_switch_workspace() {
    // DAE-107: the identity map is not a filesystem sandbox, but a mapped
    // session must not be able to repoint its workspace at will.
    let server_dir = tempfile::tempdir().unwrap();
    let alice = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let config = fake_config_with(server_dir.path(), &[("alice@example.com", alice.path())]);
    let workspace = Workspace::new(server_dir.path().to_path_buf()).unwrap();
    let state = AppState::new(config, workspace, server_dir.path().join("sessions"));
    let addr = start_state(state).await;

    let mut ws = connect_raw(
        addr,
        "/ws",
        &[("tailscale-user-login", "alice@example.com")],
    )
    .await;
    assert_eq!(
        state_workspace(&mut ws).await,
        alice.path().canonicalize().unwrap()
    );

    send(
        &mut ws,
        json!({"type": "switch_workspace", "path": other.path().to_string_lossy(), "id": "sw"}),
    )
    .await;
    let events = until(&mut ws, "error").await;
    assert!(
        events.last().unwrap()["message"]
            .as_str()
            .unwrap()
            .contains("fixed"),
        "{events:?}"
    );
    assert_eq!(
        state_workspace(&mut ws).await,
        alice.path().canonicalize().unwrap(),
        "the mapped workspace must not change"
    );
}

#[tokio::test]
async fn identity_header_is_ignored_without_a_map() {
    // A single-workspace server (no `[identities]`) accepts any identity and
    // uses the default workspace.
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect_raw(addr, "/ws", &[("tailscale-user-login", "bob@example.com")]).await;
    assert_eq!(
        state_workspace(&mut ws).await,
        dir.path().canonicalize().unwrap()
    );
}

#[tokio::test]
async fn steer_abort_and_switch_workspace_are_dispatched() {
    let dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect(addr).await;

    // steer while idle starts a turn; the fake provider settles with "done".
    send(&mut ws, json!({"type": "steer", "text": "go"})).await;
    let events = until(&mut ws, "agent_settled").await;
    assert_eq!(events.last().unwrap()["text"], "done");

    // switch_workspace: the ack and the worker's state_changed can arrive in
    // either order, so collect until both are seen.
    send(
        &mut ws,
        json!({"type": "switch_workspace", "path": other.path().to_string_lossy(), "id": "sw"}),
    )
    .await;
    let wanted = other.path().canonicalize().unwrap();
    let mut saw_response = false;
    let mut saw_state = false;
    while !(saw_response && saw_state) {
        let event = next_event(&mut ws).await;
        if event["type"] == "response" && event["id"] == "sw" {
            saw_response = true;
        }
        if event["type"] == "state_changed"
            && Path::new(event["workspace"].as_str().unwrap()) == wanted
        {
            saw_state = true;
        }
    }

    // abort is accepted and acknowledged even while idle.
    send(&mut ws, json!({"type": "abort", "id": "ab"})).await;
    let events = until(&mut ws, "response").await;
    assert_eq!(events.last().unwrap()["id"], "ab");
}

#[tokio::test]
async fn sessions_are_isolated_by_workspace() {
    let server_dir = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let addr = start(server_dir.path()).await;

    let mut a = connect(addr).await;
    send(&mut a, json!({"type": "get_state"})).await;
    let a_state = until(&mut a, "state_changed").await.pop().unwrap();

    let mut b = connect_workspace(addr, other.path()).await;
    send(&mut b, json!({"type": "get_state"})).await;
    let b_state = until(&mut b, "state_changed").await.pop().unwrap();

    assert_ne!(
        a_state["workspace"], b_state["workspace"],
        "each session must report its own workspace"
    );
    assert_eq!(
        PathBuf::from(b_state["workspace"].as_str().unwrap()),
        other.path().canonicalize().unwrap()
    );
}

#[tokio::test]
async fn bad_workspace_override_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let addr = start(dir.path()).await;
    let mut ws = connect_workspace(addr, Path::new("/definitely/not/a/dir")).await;
    let event = next_event(&mut ws).await;
    assert_eq!(event["type"], "error");
    assert!(event["message"].as_str().unwrap().contains("bad workspace"));
}
