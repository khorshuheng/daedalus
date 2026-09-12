//! The headless WebSocket server (CRAB-124): the same `Event`/`Command`
//! vocabulary as stdio RPC (CRAB-120) over one JSON object per WebSocket frame.
//!
//! Each connection gets its own [`AgentRuntime`] on a dedicated thread (the
//! CRAB-116 threading model), so concurrent sessions are fully isolated in
//! workspace and cancellation. The server never exposes a public listener by
//! itself: it binds an address the operator chooses (loopback by default) and
//! is meant to sit behind `tailscale serve`, which adds HTTPS + tailnet
//! identity. No `tailscale funnel` — public exposure is out of scope.
//!
//! # Deployment (tailnet)
//!
//! Run `crab-server --bind 127.0.0.1:8787` and front it with
//! `tailscale serve --bg 8787`, which terminates HTTPS on the tailnet and
//! injects the `Tailscale-User-Login` header. Configure `[identities]` in the
//! crab config to map those logins to workspaces; when a map is configured, an
//! unmapped identity is rejected rather than given another user's workspace.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use crab_core::config::Config;
use crab_core::provider;
use crab_core::runtime::{AgentRuntime, Command, CommandKind, Event};
use crab_core::session;
use crab_core::tools::resolver::ToolSet;
use crab_core::workspace::Workspace;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;

/// Shared, per-server state. The `Config` is a template: each session clones it
/// and resolves its own workspace (query override or the default).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub workspace: Arc<Workspace>,
    /// Where `/resume` sessions live (CRAB-109).
    pub session_root: Arc<PathBuf>,
}

impl AppState {
    pub fn new(config: Config, workspace: Workspace, session_root: PathBuf) -> Self {
        Self {
            config: Arc::new(config),
            workspace: Arc::new(workspace),
            session_root: Arc::new(session_root),
        }
    }
}

/// Optional per-session workspace: `/ws?workspace=/abs/path` (validated as an
/// existing directory). Tailnet-identity -> workspace mapping is a follow-up.
#[derive(Debug, Default, Deserialize)]
struct WsQuery {
    workspace: Option<String>,
}

/// The HTTP surface: `/health` (liveness) and `/ws` (the RPC protocol).
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_handler))
        .with_state(state)
}

async fn health() -> &'static str {
    "{\"status\":\"ok\"}"
}

/// The identity header injected by `tailscale serve` (HTTPS + tailnet
/// identity). Other fronters can set an equivalent header; the server trusts
/// it only because it binds loopback behind such a proxy.
const IDENTITY_HEADER: &str = "tailscale-user-login";

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<WsQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let identity = headers
        .get(IDENTITY_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    ws.on_upgrade(move |socket| session_loop(socket, state, query, identity))
}

/// Drive one connected session: pump runtime events out, client commands in.
async fn session_loop(
    socket: WebSocket,
    state: AppState,
    query: WsQuery,
    identity: Option<String>,
) {
    let (mut sink, mut stream) = socket.split();

    // Resolve the session workspace, rejecting an invalid mapping/override.
    let workspace = match resolve_workspace(&state, &query, identity.as_deref()) {
        Ok(ws) => ws,
        Err(e) => {
            let event = Event::Error { message: e };
            let _ = sink
                .send(Message::Text(serde_json::to_string(&event).unwrap().into()))
                .await;
            return;
        }
    };

    let mut config = (*state.config).clone();
    config.workspace = workspace.root().to_path_buf();
    let provider = provider::from_config(&config);
    let (tools, _mcp_warnings) = ToolSet::from_config(&config, config.max_output_bytes);
    let (rt, mut events) = AgentRuntime::new(config, provider, tools, workspace);

    let worker = rt.clone();
    let handle = std::thread::spawn(move || worker.run_forever());
    rt.get_state(); // emit the initial state_changed

    loop {
        tokio::select! {
            event = events.recv() => {
                let Some(event) = event else { break };
                let text = serde_json::to_string(&event).unwrap_or_else(|e| {
                    format!("{{\"type\":\"error\",\"message\":\"serialize: {e}\"}}")
                });
                if sink.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            incoming = stream.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        match dispatch(&rt, &text, &state.session_root).await {
                            // An id-carrying request gets a terminal response frame,
                            // matching the stdio RPC adapter (CRAB-120). The WS loop
                            // stays concurrent so steering can arrive mid-turn, so
                            // this is an acceptance ack rather than a post-turn one.
                            Ok(Some(id)) => {
                                let response = serde_json::json!({
                                    "type": "response",
                                    "id": id,
                                    "ok": true
                                });
                                if sink
                                    .send(Message::Text(response.to_string().into()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Ok(None) => {}
                            Err(message) => {
                                let event = Event::Error { message };
                                let json = serde_json::to_string(&event).unwrap();
                                if sink.send(Message::Text(json.into())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {} // ping/pong/binary ignored
                    Some(Err(_)) => break,
                }
            }
        }
    }

    // Abort first: a disconnect mid-turn would otherwise leave `shutdown`
    // waiting for the turn to settle.
    rt.abort();
    rt.shutdown();
    let _ = handle.join();
}

/// Pick the session workspace. Precedence: mapped identity (from the proxy
/// header, CRAB-124) > explicit `?workspace=` override > server default. When
/// an identity map is configured, an identity the proxy vouched for but the
/// map does not cover is **rejected** rather than falling back — otherwise a
/// tailnet user could reach the admin's default workspace. A request with no
/// identity header (local/direct use) may still use the override or default.
fn resolve_workspace(
    state: &AppState,
    query: &WsQuery,
    identity: Option<&str>,
) -> Result<Workspace, String> {
    if let Some(identity) = identity {
        if let Some(path) = state.config.identity_workspaces.get(identity) {
            return Workspace::new(path.clone())
                .map_err(|e| format!("workspace for identity '{identity}' is invalid: {e}"));
        }
        if !state.config.identity_workspaces.is_empty() {
            return Err(format!("no workspace mapped for identity '{identity}'"));
        }
    }
    match query.workspace.as_deref() {
        Some(path) if !path.trim().is_empty() => {
            Workspace::new(PathBuf::from(path)).map_err(|e| format!("bad workspace: {e}"))
        }
        _ => Ok((*state.workspace).clone()),
    }
}

/// Apply one client `Command`. Returns the command's `id` (for a terminal
/// `response` frame) or an error message (sent back as an `error` event) for a
/// malformed frame or a failed resume.
async fn dispatch(
    rt: &AgentRuntime,
    text: &str,
    session_root: &Path,
) -> Result<Option<String>, String> {
    let command: Command =
        serde_json::from_str(text).map_err(|e| format!("malformed command: {e}"))?;
    let id = command.id.clone();
    match command.kind {
        CommandKind::Prompt { text } => rt.prompt(&text),
        CommandKind::Steer { text } => rt.steer(&text),
        CommandKind::FollowUp { text } => rt.follow_up(&text),
        CommandKind::Abort {} => rt.abort(),
        CommandKind::SetModel { model } => rt.set_model(&model),
        CommandKind::SetEffort { effort } => rt.set_effort(effort),
        CommandKind::SwitchWorkspace { path } => rt.switch_workspace(&path),
        CommandKind::Clear {} => rt.clear(),
        CommandKind::GetState {} => rt.get_state(),
        CommandKind::Resume => match session::load_previous(session_root, &rt.workspace_root()) {
            Ok(Some(history)) => rt.replace_history(history),
            Ok(None) => {}
            Err(e) => return Err(format!("could not resume: {e}")),
        },
    }
    Ok(id)
}
