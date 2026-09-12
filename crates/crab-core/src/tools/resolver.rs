//! The resolver maps a `{name, args}` pair to the correct tool executor and
//! surfaces unknown-tool / bad-args / not-found errors clearly.

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use super::{
    bash::BashTool, edit::EditTool, read::ReadTool, search::SearchTool, write::WriteTool, Tool,
    ToolError, ToolOutput,
};
use crate::workspace::Workspace;

/// A registered tool: its name, a one-line description for the provider's
/// `tools` field, and its executor. Built-ins are fixed; MCP servers
/// (CRAB-133) contribute additional entries at construction.
struct ToolEntry {
    name: String,
    description: String,
    tool: Box<dyn Tool>,
    /// External (MCP) tools carry server-authored JSON Schema, which is not
    /// the subset [`super::validate_args`] understands, so argument validation
    /// is delegated to the server (CRAB-133).
    external: bool,
}

/// The built-in tools plus any configured MCP tools, sharing one output size
/// cap.
pub struct ToolSet {
    tools: Vec<ToolEntry>,
}

impl ToolSet {
    /// The built-ins, in a stable order (no default bash timeout).
    pub fn new(max_output: usize) -> Self {
        Self::with_bash_timeout(max_output, None)
    }

    /// Like [`new`](Self::new), but applies `bash_timeout_secs` as the `bash`
    /// default when the model omits `timeout` (CRAB-139 review), so an
    /// unbounded command cannot run forever.
    pub fn with_bash_timeout(max_output: usize, bash_timeout_secs: Option<u64>) -> Self {
        let mut set = Self { tools: Vec::new() };
        set.push(
            "read",
            "Read a file (optionally a line range) in the workspace.",
            Box::new(ReadTool { max_output }),
        );
        set.push(
            "bash",
            "Run a shell command in the workspace directory.",
            Box::new(BashTool {
                max_output,
                default_timeout_secs: bash_timeout_secs,
            }),
        );
        set.push(
            "search",
            "Search file contents with a regex (ripgrep-style: respects .gitignore, skips hidden and binary files).",
            Box::new(SearchTool { max_output }),
        );
        set.push(
            "edit",
            "Apply precise, validated text replacements to a file.",
            Box::new(EditTool),
        );
        set.push(
            "write",
            "Create or overwrite a file in the workspace.",
            Box::new(WriteTool),
        );
        set
    }

    /// Build the tool set from config: the built-ins plus every enabled MCP
    /// server (CRAB-133). Returns the set and non-fatal warnings for
    /// servers that failed to start — the agent runs with the tools that did.
    pub fn from_config(config: &crate::config::Config, max_output: usize) -> (Self, Vec<String>) {
        let mut set = Self::with_bash_timeout(max_output, config.bash_default_timeout());
        let (mcp_tools, warnings) = crate::mcp::connect_all(&config.mcp_servers, max_output);
        for tool in mcp_tools {
            let entry = ToolEntry {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                tool: Box::new(tool),
                external: true,
            };
            set.tools.push(entry);
        }
        (set, warnings)
    }

    fn push(&mut self, name: &str, description: &str, tool: Box<dyn Tool>) {
        self.tools.push(ToolEntry {
            name: name.to_string(),
            description: description.to_string(),
            tool,
            external: false,
        });
    }

    /// JSON Schemas describing each tool's arguments, wrapped with the tool's
    /// name and description for the provider `tools` field.
    pub fn tool_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|e| {
                json!({
                    "name": e.name,
                    "description": e.description,
                    "parameters": e.tool.schema()
                })
            })
            .collect()
    }

    pub fn tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.tool.as_ref())
    }

    /// `(name, description)` for every registered tool — the `/tools` listing.
    pub fn listing(&self) -> Vec<(String, String)> {
        self.tools
            .iter()
            .map(|e| (e.name.clone(), e.description.clone()))
            .collect()
    }

    /// Route `name`+`args` to the matching executor, or a clear error. Built-in
    /// args are validated against the tool's declared JSON Schema first
    /// (CRAB-107 #9) so malformed model calls never reach an executor; external
    /// (MCP) tools are exempt because their server-authored schemas are not the
    /// subset `validate_args` understands (CRAB-133) — the server validates.
    pub async fn execute(
        &self,
        workspace: &Workspace,
        name: &str,
        args: &Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        match self.tools.iter().find(|e| e.name == name) {
            Some(entry) => {
                if !entry.external {
                    super::validate_args(&entry.tool.schema(), args)?;
                }
                entry.tool.run(workspace, args, cancel).await
            }
            None => {
                let names: Vec<&str> = self.tools.iter().map(|e| e.name.as_str()).collect();
                Err(ToolError::Argument(format!(
                    "unknown tool '{name}' (expected {})",
                    names.join(", ")
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;

    fn workspace(_name: &str) -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        let ws = Workspace::new(root).unwrap();
        (dir, ws)
    }

    /// A trivial remote tool used to prove dynamic registration works.
    struct EchoTool {
        name: String,
    }

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn schema(&self) -> Value {
            json!({"type": "object", "properties": {"x": {"type": "string"}}})
        }
        fn run<'a>(
            &'a self,
            _workspace: &'a Workspace,
            args: &'a Value,
            _cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
            let x = args
                .get("x")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutput { content: x }) })
        }
    }

    #[tokio::test]
    async fn routes_to_correct_executor() {
        let (_dir, ws) = workspace("bash");
        let ts = ToolSet::new(1000);

        let out = ts
            .execute(
                &ws,
                "bash",
                &json!({"command": "echo hi"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("hi"));
    }

    #[tokio::test]
    async fn unknown_tool_is_clear_error() {
        let (_dir, ws) = workspace("unknown");
        let ts = ToolSet::new(1000);
        let err = ts
            .execute(&ws, "frobnicate", &json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown tool 'frobnicate'"));
        assert!(err.to_string().contains("search"));
    }

    #[test]
    fn exposes_builtin_tool_schemas() {
        let ts = ToolSet::new(1000);
        let schemas = ts.tool_schemas();
        assert_eq!(schemas.len(), 5);
        for s in &schemas {
            assert!(s.get("name").is_some());
            assert!(s.get("description").is_some());
            assert!(s.get("parameters").is_some());
        }
    }

    #[tokio::test]
    async fn rejects_bad_args_before_running() {
        let (_dir, ws) = workspace("badargs");
        let ts = ToolSet::new(1000);
        let err = ts
            .execute(&ws, "read", &json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing required 'path'"));
        // A non-string path is rejected too.
        let err = ts
            .execute(&ws, "read", &json!({"path": 7}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("'path' must be a string"));
    }

    /// A dynamically registered tool (as an MCP server would contribute) is
    /// discoverable, listed, schema-exposed, and executable.
    #[tokio::test]
    async fn dynamic_tools_join_the_set() {
        let (_dir, ws) = workspace("dynamic");
        let mut ts = ToolSet::new(1000);
        ts.push(
            "mcp__echo__echo",
            "remote echo",
            Box::new(EchoTool {
                name: "mcp__echo__echo".into(),
            }),
        );
        assert_eq!(ts.tool_schemas().len(), 6);
        assert!(ts
            .listing()
            .iter()
            .any(|(n, d)| n == "mcp__echo__echo" && d == "remote echo"));
        let out = ts
            .execute(
                &ws,
                "mcp__echo__echo",
                &json!({"x": "remote!"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "remote!");
        // The unknown-tool error now lists the dynamic name too.
        let err = ts
            .execute(&ws, "nope", &json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("mcp__echo__echo"));
    }

    /// A tool whose schema is not crab's subset (e.g. no `"type": "object"`,
    /// which real MCP servers may omit) must bypass the built-in validator
    /// when registered as external, or every call would be rejected.
    struct NoTypeSchemaTool;

    impl Tool for NoTypeSchemaTool {
        fn name(&self) -> &str {
            "mcp__x__y"
        }
        fn schema(&self) -> Value {
            json!({"properties": {"q": {"type": "string"}}})
        }
        fn run<'a>(
            &'a self,
            _workspace: &'a Workspace,
            _args: &'a Value,
            _cancel: CancellationToken,
        ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
            Box::pin(async move {
                Ok(ToolOutput {
                    content: "ok".into(),
                })
            })
        }
    }

    #[tokio::test]
    async fn external_tools_skip_builtin_schema_validation() {
        let (_dir, ws) = workspace("external");
        let mut ts = ToolSet::new(1000);
        ts.tools.push(ToolEntry {
            name: "mcp__x__y".into(),
            description: "external".into(),
            tool: Box::new(NoTypeSchemaTool),
            external: true,
        });
        let out = ts
            .execute(
                &ws,
                "mcp__x__y",
                &json!({"q": "hi"}),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "ok");

        // The same schema registered as a *built-in* is rejected, proving the
        // exemption is scoped to external tools.
        let mut strict = ToolSet::new(1000);
        strict.tools.push(ToolEntry {
            name: "mcp__x__y".into(),
            description: "external".into(),
            tool: Box::new(NoTypeSchemaTool),
            external: false,
        });
        let err = strict
            .execute(&ws, "mcp__x__y", &json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("schema must describe an object"));
    }

    #[test]
    fn from_config_without_mcp_keeps_the_builtins() {
        let (_dir, ws) = workspace("fromconfig");
        let cfg = crate::config::Config::defaults(ws.root().to_path_buf());
        let (ts, warnings) = ToolSet::from_config(&cfg, 1000);
        assert!(warnings.is_empty());
        assert_eq!(ts.tool_schemas().len(), 5);
    }
}
