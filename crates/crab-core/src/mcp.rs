//! MCP tool servers (CRAB-133): attach external tools over the Model Context
//! Protocol so crab's tool surface is no longer limited to the four built-ins.
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
//!   trait as the built-ins, named `mcp:<server>:<tool>`, so the existing
//!   `ToolSet`/resolver and the `tool_start`/`tool_end` events carry the
//!   qualified name with no runtime changes.
//! - MCP schemas are JSON Schema, a superset of crab's internal validation
//!   subset, so argument validation for external tools is delegated to the
//!   server.
//! - A hanging call honors the session [`CancellationToken`]: the in-flight
//!   request future is dropped (request teardown) and the tool reports
//!   [`ToolError::Cancelled`]. Server processes are torn down when the owning
//!   runtime is dropped (rmcp's child cleanup), not per call.
//!
//! The [`McpClient`] seam keeps the bridge testable without child processes:
//! offline tests inject a scripted in-process stub.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::BoxFuture;
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
/// the rmcp adapter and by offline test stubs.
pub trait McpClient: Send + Sync {
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: Value,
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

/// `mcp:<server>:<tool>` — the qualified name used everywhere downstream.
pub fn qualified_name(server: &str, tool: &str) -> String {
    format!("mcp:{server}:{tool}")
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
            let call = self.client.call_tool(&self.remote, args.clone());
            let result = tokio::select! {
                res = call => res,
                _ = cancel.cancelled() => Err(ToolError::Cancelled),
            };
            let content = result?;
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
    use rmcp::service::{serve_client, RoleClient, RunningService};

    let service: RunningService<RoleClient, ()> = match config.transport {
        McpTransportKind::Stdio => {
            let command = config.command.clone().unwrap_or_default();
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&config.args).envs(&config.env);
            let transport = rmcp::transport::TokioChildProcess::new(cmd)
                .map_err(|e| format!("spawn '{command}': {e}"))?;
            serve_client((), transport)
                .await
                .map_err(|e| format!("initialize: {e}"))?
        }
        McpTransportKind::Http => {
            let url = config.url.clone().unwrap_or_default();
            let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url);
            serve_client((), transport)
                .await
                .map_err(|e| format!("initialize: {e}"))?
        }
    };

    let listed = service
        .list_tools(None)
        .await
        .map_err(|e| format!("list_tools: {e}"))?;

    let tools = listed
        .tools
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
        name: config.name.clone(),
        client,
        tools,
        max_output,
    })
}

/// The rmcp-backed client. Holds the running service and the runtime that
/// drives it; `call_tool` sends a request and awaits the reply (the service
/// loop keeps running on `_runtime`).
struct RmcpClient {
    service: Arc<rmcp::service::RunningService<rmcp::service::RoleClient, ()>>,
    /// Keeps the runtime (and thus the service task) alive.
    _runtime: Arc<tokio::runtime::Runtime>,
}

impl McpClient for RmcpClient {
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: Value,
    ) -> BoxFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            use rmcp::model::CallToolRequestParams;
            let arguments: Map<String, Value> = args.as_object().cloned().unwrap_or_default();
            let params = CallToolRequestParams::new(name.to_string()).with_arguments(arguments);
            let result = self
                .service
                .call_tool(params)
                .await
                .map_err(|e| ToolError::Command(format!("MCP call '{name}' failed: {e}")))?;

            let mut text = String::new();
            for block in &result.content {
                if let rmcp::model::ContentBlock::Text(t) = block {
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
    fn qualified_name_matches_the_event_contract() {
        assert_eq!(qualified_name("fs", "read_file"), "mcp:fs:read_file");
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
        assert_eq!(tool.name(), "mcp:srv:echo");
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
            ) -> BoxFuture<'a, Result<String, ToolError>> {
                Box::pin(std::future::pending())
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
}
