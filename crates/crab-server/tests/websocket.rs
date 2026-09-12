//! Offline integration tests for the CRAB-124 WebSocket server: a real axum
//! server on an ephemeral loopback port, a real WebSocket client, and the
//! scripted fake provider — no network, no external services.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use crab_core::config::{provider_by_name, Config};
use crab_core::workspace::Workspace;
use crab_server::{build_router, AppState};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type Client = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

fn fake_config(workspace: &Path) -> Config {
    Config {
        provider: provider_by_name("fake").unwrap(),
        model: "fake-model".into(),
        workspace: workspace.to_path_buf(),
        ..Config::defaults(workspace.to_path_buf())
    }
}

/// Start the router on an ephemeral port; return its address.
async fn start(dir: &Path) -> SocketAddr {
    let workspace = Workspace::new(dir.to_path_buf()).unwrap();
    let state = AppState::new(fake_config(dir), workspace, dir.join("sessions"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, build_router(state)).await;
    });
    addr
}

async fn connect(addr: SocketAddr) -> Client {
    let (ws, _) = connect_async(format!("ws://{addr}/ws")).await.unwrap();
    ws
}

async fn connect_workspace(addr: SocketAddr, workspace: &Path) -> Client {
    let path = workspace.to_string_lossy().replace(' ', "%20");
    let (ws, _) = connect_async(format!("ws://{addr}/ws?workspace={path}"))
        .await
        .unwrap();
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
