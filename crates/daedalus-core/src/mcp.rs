//! MCP tool servers: attach external tools over the Model Context
//! Protocol so daedalus's tool surface is no longer limited to the four built-ins.
//!
//! The spec originally named "rig-rmcp"; rig-core 0.42 has no MCP integration
//! and no such crate exists, so this uses the official Rust SDK (`rmcp`)
//! directly. The design:
//!
//! - Each configured `[[mcp_servers]]` entry (stdio child process or
//!   streamable-http endpoint) is connected at construction, non-fatally: a
//!   server that fails to start is reported as a warning and the agent runs
//!   with the remaining tools.
//! - Every remote tool becomes a [`McpTool`] implementing the same [`Tool`]
//!   trait as the built-ins, named `mcp__<server>__<tool>`, so the existing
//!   `ToolSet`/resolver and the `tool_start`/`tool_end` events carry the
//!   qualified name with no runtime changes.
//! - MCP schemas are JSON Schema, a superset of daedalus's internal validation
//!   subset, so argument validation for external tools is delegated to the
//!   server.
//! - A cancelled call (session `CancellationToken`) sends the MCP
//!   `notifications/cancelled` for the in-flight request (request teardown) and
//!   returns [`ToolError::Cancelled`]. Server processes are torn down when the
//!   owning runtime is dropped (rmcp's child cleanup), not per call.
//!
//! The [`McpClient`] seam keeps the bridge testable without child processes:
//! offline tests inject a scripted in-process stub, and an end-to-end test
//! wires a real rmcp server over an in-process duplex transport.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotification, CancelledNotificationParam,
    ClientRequest, ContentBlock, ServerResult,
};
use rmcp::service::{serve_client, PeerRequestOptions, RoleClient, RunningService};
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::tools::{truncate_tail, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// How to reach an MCP server. `stdio` (default) spawns a child process;
/// `http` talks to a streamable-http endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransportKind {
    #[default]
    Stdio,
    Http,
}

fn default_true() -> bool {
    true
}

/// One `[[mcp_servers]]` config entry.
///
/// ```toml
/// [[mcp_servers]]
/// name = "fs"
/// transport = "stdio"            # default
/// command = "npx"
/// args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
/// env = { RUST_LOG = "warn" }
///
/// [[mcp_servers]]
/// name = "remote"
/// transport = "http"
/// url = "http://localhost:8000/mcp"
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub transport: McpTransportKind,
    /// stdio: executable to spawn.
    #[serde(default)]
    pub command: Option<String>,
    /// stdio: arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// stdio: extra environment variables for the child.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// http: streamable-http endpoint.
    #[serde(default)]
    pub url: Option<String>,
}

/// Validate the MCP server table: unique non-empty names, and the transport's
/// required fields. Called from `PartialConfig::resolve`.
pub fn validate_servers(servers: &[McpServerConfig]) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for s in servers {
        if s.name.trim().is_empty() {
            return Err("mcp_servers entries need a non-empty name".into());
        }
        if !seen.insert(s.name.as_str()) {
            return Err(format!("duplicate mcp server name '{}'", s.name));
        }
        match s.transport {
            McpTransportKind::Stdio => {
                if s.command.as_deref().map(str::trim).unwrap_or("").is_empty() {
                    return Err(format!("mcp server '{}' (stdio) needs a 'command'", s.name));
                }
            }
            McpTransportKind::Http => {
                if s.url.as_deref().map(str::trim).unwrap_or("").is_empty() {
                    return Err(format!("mcp server '{}' (http) needs a 'url'", s.name));
                }
            }
        }
    }
    Ok(())
}

/// A tool advertised by an MCP server, before it is wrapped for the toolset.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

/// The client seam: call a remote tool by its unqualified name. Implemented by
/// the rmcp adapter and by offline test stubs. Implementations must honor
/// `cancel`: return [`ToolError::Cancelled`] promptly when it fires (the rmcp
/// adapter also sends the MCP `notifications/cancelled`).
pub trait McpClient: Send + Sync {
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<String, ToolError>>;
}

/// One remote tool, named `mcp:<server>:<tool>` for the provider and events.
pub struct McpTool {
    qualified: String,
    remote: String,
    description: String,
    schema: Value,
    client: Arc<dyn McpClient>,
    max_output: usize,
}

impl McpTool {
    pub fn new(
        server: &str,
        descriptor: McpToolDescriptor,
        client: Arc<dyn McpClient>,
        max_output: usize,
    ) -> Self {
        Self {
            qualified: qualified_name(server, &descriptor.name),
            remote: descriptor.name,
            description: descriptor.description,
            schema: descriptor.schema,
            client,
            max_output,
        }
    }

    /// The one-line description shown in `/tools` and the provider `tools`.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// `mcp__<server>__<tool>` — the qualified name used everywhere downstream.
/// Provider tool/function names are restricted to `[A-Za-z0-9_-]` (OpenAI and
/// Anthropic both enforce `^[a-zA-Z0-9_-]{1,64}$`), so a colon separator (as
/// the original spec suggested) would be rejected on the wire. Each segment
/// is sanitized and the whole name is capped at 64 chars.
pub fn qualified_name(server: &str, tool: &str) -> String {
    let mut name = format!(
        "mcp__{}__{}",
        sanitize_segment(server),
        sanitize_segment(tool)
    );
    if name.len() > MAX_TOOL_NAME {
        name.truncate(MAX_TOOL_NAME);
        while !name.is_char_boundary(name.len()) {
            name.pop();
        }
    }
    name
}

/// Provider tool-name limit (`^[a-zA-Z0-9_-]{1,64}$`).
const MAX_TOOL_NAME: usize = 64;

/// Replace any character outside `[A-Za-z0-9_-]` with `_` so a server/tool
/// name is a legal provider function name segment.
fn sanitize_segment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    fn run<'a>(
        &'a self,
        _workspace: &'a Workspace,
        args: &'a Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            if !args.is_object() {
                return Err(ToolError::Argument(
                    "MCP tool arguments must be a JSON object".into(),
                ));
            }
            let content = self
                .client
                .call_tool(&self.remote, args.clone(), cancel)
                .await?;
            let (content, _truncated) = truncate_tail(&content, self.max_output);
            Ok(ToolOutput { content })
        })
    }
}

/// A connected server: its name, the client, and the tools it advertised.
pub struct ConnectedServer {
    name: String,
    client: Arc<dyn McpClient>,
    tools: Vec<McpToolDescriptor>,
    max_output: usize,
}

impl ConnectedServer {
    /// Wrap each advertised tool as an [`McpTool`].
    pub fn into_tools(self) -> Vec<McpTool> {
        self.tools
            .into_iter()
            .map(|d| McpTool::new(&self.name, d, Arc::clone(&self.client), self.max_output))
            .collect()
    }
}

/// Connect every enabled server in `configs`, blocking on a dedicated
/// single-worker tokio runtime that stays alive for the returned tools (the
/// rmcp service tasks run on it). Connection failures are non-fatal: returned
/// as human-readable warnings, and the rest of the toolset is unaffected.
pub fn connect_all(configs: &[McpServerConfig], max_output: usize) -> (Vec<McpTool>, Vec<String>) {
    let enabled: Vec<&McpServerConfig> = configs.iter().filter(|c| c.enabled).collect();
    if enabled.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
    {
        Ok(rt) => Arc::new(rt),
        Err(e) => return (Vec::new(), vec![format!("cannot start MCP runtime: {e}")]),
    };

    let mut tools = Vec::new();
    let mut warnings = Vec::new();
    for cfg in enabled {
        match runtime.block_on(connect_one(cfg, Arc::clone(&runtime), max_output)) {
            Ok(server) => tools.extend(server.into_tools()),
            Err(e) => warnings.push(format!("mcp server '{}' unavailable: {e}", cfg.name)),
        }
    }
    (tools, warnings)
}

/// Connect one server and list its tools. Async: the caller owns the runtime.
async fn connect_one(
    config: &McpServerConfig,
    runtime: Arc<tokio::runtime::Runtime>,
    max_output: usize,
) -> Result<ConnectedServer, String> {
    match config.transport {
        McpTransportKind::Stdio => {
            let command = config.command.clone().unwrap_or_default();
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&config.args).envs(&config.env);
            let transport = rmcp::transport::TokioChildProcess::new(cmd)
                .map_err(|e| format!("spawn '{command}': {e}"))?;
            connect_transport(&config.name, transport, Some(runtime), max_output).await
        }
        McpTransportKind::Http => {
            let url = config.url.clone().unwrap_or_default();
            let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url);
            connect_transport(&config.name, transport, Some(runtime), max_output).await
        }
    }
}

/// Finish a connection once a client transport is available: initialize the
/// MCP session, list every tool (following pagination), and wrap them. Split
/// from [`connect_one`] so tests can drive it over an in-process duplex
/// transport instead of a child process.
async fn connect_transport<T, E, A>(
    name: &str,
    transport: T,
    runtime: Option<Arc<tokio::runtime::Runtime>>,
    max_output: usize,
) -> Result<ConnectedServer, String>
where
    T: rmcp::transport::IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let service: RunningService<RoleClient, ()> = serve_client((), transport)
        .await
        .map_err(|e| format!("initialize: {e}"))?;

    // `list_all_tools` follows `next_cursor` to the end.
    let listed = service
        .list_all_tools()
        .await
        .map_err(|e| format!("list_tools: {e}"))?;

    let tools = listed
        .into_iter()
        .map(|t| McpToolDescriptor {
            name: t.name.to_string(),
            description: t.description.map(|d| d.to_string()).unwrap_or_default(),
            schema: Value::Object((*t.input_schema).clone()),
        })
        .collect();

    let client = Arc::new(RmcpClient {
        service: Arc::new(service),
        _runtime: runtime,
    });
    Ok(ConnectedServer {
        name: name.to_string(),
        client,
        tools,
        max_output,
    })
}

/// The rmcp-backed client. Holds the running service and (optionally) the
/// runtime that drives it; `call_tool` sends a cancellable request so a session
/// cancellation notifies the server and stops waiting.
struct RmcpClient {
    service: Arc<RunningService<RoleClient, ()>>,
    /// Keeps the runtime (and thus the service task) alive. `None` when the
    /// caller already owns the driving runtime (in-process tests).
    _runtime: Option<Arc<tokio::runtime::Runtime>>,
}

impl McpClient for RmcpClient {
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let arguments: Map<String, Value> = args.as_object().cloned().unwrap_or_default();
            let params = CallToolRequestParams::new(name.to_string()).with_arguments(arguments);
            // A raw cancellable request (rather than the MRTR-driving
            // `RunningService::call_tool`) so a session cancellation can send
            // `notifications/cancelled`. daedalus's client handler is `()`, so it
            // cannot service MRTR input-required rounds anyway — only the
            // direct `CallToolResult` is handled.
            let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));

            let handle = self
                .service
                .send_cancellable_request(request, PeerRequestOptions::no_options())
                .await
                .map_err(|e| ToolError::Command(format!("MCP call '{name}' failed: {e}")))?;
            // Keep what we need to cancel: `await_response` consumes the handle.
            let peer = handle.peer.clone();
            let id = handle.id.clone();

            let response = tokio::select! {
                res = handle.await_response() => res
                    .map_err(|e| ToolError::Command(format!("MCP call '{name}' failed: {e}")))?,
                _ = cancel.cancelled() => {
                    // Tell the server to stop, then report the cancellation.
                    let notification = CancelledNotification::new(
                        CancelledNotificationParam::new(
                            Some(id),
                            Some("session cancelled".to_string()),
                        ),
                    );
                    let _ = peer.send_notification(notification.into()).await;
                    return Err(ToolError::Cancelled);
                }
            };

            let result = match response {
                ServerResult::CallToolResult(result) => result,
                _ => {
                    return Err(ToolError::Command(format!(
                        "MCP tool '{name}' returned an unexpected response"
                    )))
                }
            };

            let mut text = String::new();
            for block in &result.content {
                if let ContentBlock::Text(t) = block {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&t.text);
                }
            }
            if text.is_empty() {
                if let Some(structured) = &result.structured_content {
                    text = structured.to_string();
                }
            }
            if result.is_error == Some(true) {
                return Err(ToolError::Command(format!(
                    "MCP tool '{name}' returned an error: {text}"
                )));
            }
            if text.is_empty() {
                text = "(no output)".to_string();
            }
            Ok(text)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A scripted in-process client: no child processes, no network.
    struct StubClient {
        replies: Mutex<BTreeMap<String, Result<String, ToolError>>>,
    }

    impl StubClient {
        fn new(pairs: &[(&str, Result<&str, &str>)]) -> Self {
            let replies = pairs
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        Ok(s) => Ok((*s).to_string()),
                        Err(e) => Err(ToolError::Command((*e).to_string())),
                    };
                    ((*k).to_string(), val)
                })
                .collect();
            Self {
                replies: Mutex::new(replies),
            }
        }
    }

    impl McpClient for StubClient {
        fn call_tool<'a>(
            &'a self,
            name: &'a str,
            _args: Value,
            _cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<String, ToolError>> {
            let reply = self.replies.lock().unwrap().get(name).cloned();
            Box::pin(async move {
                reply.unwrap_or_else(|| Err(ToolError::Command(format!("no stub for '{name}'"))))
            })
        }
    }

    fn descriptor(name: &str) -> McpToolDescriptor {
        McpToolDescriptor {
            name: name.into(),
            description: "does a thing".into(),
            schema: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    fn workspace() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        (dir, ws)
    }

    #[test]
    fn connect_all_skips_disabled_and_missing_servers_without_spawning() {
        let (tools, warnings) = connect_all(&[], 1000);
        assert!(tools.is_empty());
        assert!(warnings.is_empty());
        let disabled = McpServerConfig {
            name: "x".into(),
            enabled: false,
            command: Some("definitely-not-spawned".into()),
            ..Default::default()
        };
        let (tools, warnings) = connect_all(&[disabled], 1000);
        assert!(tools.is_empty());
        assert!(warnings.is_empty(), "disabled servers are skipped silently");
    }

    #[test]
    fn qualified_name_is_provider_safe() {
        assert_eq!(qualified_name("fs", "read_file"), "mcp__fs__read_file");
        // Dots/spaces/colons are not legal in provider function names.
        assert_eq!(
            qualified_name("my.server", "read file:v2"),
            "mcp__my_server__read_file_v2"
        );
        // Long names are capped at the provider limit.
        let long = qualified_name(&"s".repeat(80), &"t".repeat(80));
        assert!(long.len() <= 64);
    }

    #[test]
    fn config_parses_stdio_and_http() {
        let cfg: McpServerConfig = toml::from_str(
            r#"
            name = "fs"
            command = "npx"
            args = ["-y", "server"]
            env = { A = "1" }
            "#,
        )
        .unwrap();
        assert_eq!(cfg.transport, McpTransportKind::Stdio);
        assert!(cfg.enabled, "enabled defaults to true");
        assert_eq!(cfg.command.as_deref(), Some("npx"));
        assert_eq!(cfg.env.get("A").map(String::as_str), Some("1"));

        let cfg: McpServerConfig = toml::from_str(
            r#"
            name = "remote"
            transport = "http"
            url = "http://localhost:8000/mcp"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.transport, McpTransportKind::Http);
        assert_eq!(cfg.url.as_deref(), Some("http://localhost:8000/mcp"));
    }

    #[test]
    fn validation_rejects_missing_fields_and_duplicates() {
        let missing_cmd = McpServerConfig {
            name: "a".into(),
            ..Default::default()
        };
        assert!(validate_servers(&[missing_cmd])
            .unwrap_err()
            .contains("needs a 'command'"));

        let missing_url = McpServerConfig {
            name: "b".into(),
            transport: McpTransportKind::Http,
            ..Default::default()
        };
        assert!(validate_servers(&[missing_url])
            .unwrap_err()
            .contains("needs a 'url'"));

        let dup = vec![
            McpServerConfig {
                name: "x".into(),
                command: Some("true".into()),
                ..Default::default()
            },
            McpServerConfig {
                name: "x".into(),
                command: Some("true".into()),
                ..Default::default()
            },
        ];
        assert!(validate_servers(&dup).unwrap_err().contains("duplicate"));
    }

    #[tokio::test]
    async fn mcp_tool_forwards_and_truncates() {
        let (_dir, ws) = workspace();
        let client: Arc<dyn McpClient> = Arc::new(StubClient::new(&[("echo", Ok("hello"))]));
        let tool = McpTool::new("srv", descriptor("echo"), client, 1000);
        assert_eq!(tool.name(), "mcp__srv__echo");
        assert_eq!(tool.description(), "does a thing");
        let out = tool
            .run(
                &ws,
                &serde_json::json!({"msg": "hi"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "hello");
    }

    #[tokio::test]
    async fn mcp_tool_surfaces_server_errors() {
        let (_dir, ws) = workspace();
        let client: Arc<dyn McpClient> =
            Arc::new(StubClient::new(&[("boom", Err("server exploded"))]));
        let tool = McpTool::new("srv", descriptor("boom"), client, 1000);
        let err = tool
            .run(&ws, &serde_json::json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("server exploded"));
    }

    #[tokio::test]
    async fn mcp_tool_rejects_non_object_args() {
        let (_dir, ws) = workspace();
        let client: Arc<dyn McpClient> = Arc::new(StubClient::new(&[]));
        let tool = McpTool::new("srv", descriptor("x"), client, 1000);
        let err = tool
            .run(&ws, &serde_json::json!([1, 2]), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("JSON object"));
    }

    #[tokio::test]
    async fn mcp_tool_cancellation_reports_cancelled() {
        /// A client whose call never resolves.
        struct Hanging;
        impl McpClient for Hanging {
            fn call_tool<'a>(
                &'a self,
                _name: &'a str,
                _args: Value,
                cancel: CancellationToken,
            ) -> BoxFuture<'a, Result<String, ToolError>> {
                Box::pin(async move {
                    cancel.cancelled().await;
                    Err(ToolError::Cancelled)
                })
            }
        }
        let (_dir, ws) = workspace();
        let tool = McpTool::new("srv", descriptor("slow"), Arc::new(Hanging), 1000);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = tool
            .run(&ws, &serde_json::json!({}), cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }

    /// End-to-end over a real rmcp client <-> real rmcp server on an
    /// in-process duplex transport: no child process, no network. Exercises
    /// `connect_transport` (list_all_tools + wrapping) and a real tool call
    /// through the bridge.
    #[tokio::test]
    async fn real_in_process_mcp_server_end_to_end() {
        use rmcp::model::{
            CallToolResponse, CallToolResult, ListToolsResult, ServerCapabilities, ServerInfo, Tool,
        };
        use rmcp::service::{RequestContext, RoleServer};
        use rmcp::ServerHandler;

        struct EchoServer;
        impl ServerHandler for EchoServer {
            fn get_info(&self) -> ServerInfo {
                ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            }
            async fn list_tools(
                &self,
                _request: Option<rmcp::model::PaginatedRequestParams>,
                _context: RequestContext<RoleServer>,
            ) -> Result<ListToolsResult, rmcp::ErrorData> {
                Ok(ListToolsResult {
                    tools: vec![Tool::new("echo", "Echo a message", serde_json::Map::new())],
                    ..Default::default()
                })
            }
            async fn call_tool(
                &self,
                request: CallToolRequestParams,
                _context: RequestContext<RoleServer>,
            ) -> Result<CallToolResponse, rmcp::ErrorData> {
                let msg = request
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("msg"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Ok(CallToolResult::success(vec![ContentBlock::text(msg)]).into())
            }
        }

        let (server_io, client_io) = tokio::io::duplex(8192);
        // The server must run concurrently: `serve_server` waits for the
        // client's initialize, so it cannot be awaited before the client
        // exists.
        let server_task = tokio::spawn(rmcp::serve_server(EchoServer, server_io));

        let connected = connect_transport("srv", client_io, None, 1000)
            .await
            .unwrap();
        let mut tools = connected.into_tools();
        assert_eq!(tools.len(), 1);
        let tool = tools.pop().unwrap();
        assert_eq!(tool.name(), "mcp__srv__echo");
        assert_eq!(tool.description(), "Echo a message");

        let (_dir, ws) = workspace();
        let out = tool
            .run(
                &ws,
                &serde_json::json!({"msg": "hi from mcp"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "hi from mcp");

        server_task.abort();
    }
}
