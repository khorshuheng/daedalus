//! The resolver maps a `{name, args}` pair to the correct tool executor and
//! surfaces unknown-tool / bad-args / not-found errors clearly.

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use super::{
    bash::BashTool, edit::EditTool, read::ReadTool, write::WriteTool, Tool, ToolError, ToolOutput,
};
use crate::workspace::Workspace;

/// A registered tool: its name, a one-line description for the provider's
/// `tools` field, and its executor. Built-ins are fixed; MCP servers
/// (CRAB-133) contribute additional entries at construction.
struct ToolEntry {
    name: String,
    description: String,
    tool: Box<dyn Tool>,
}

/// The four built-in tools plus any configured MCP tools, sharing one output
/// cap.
pub struct ToolSet {
    tools: Vec<ToolEntry>,
}

impl ToolSet {
    /// The four built-ins, in a stable order.
    pub fn new(max_output: usize) -> Self {
        let mut set = Self { tools: Vec::new() };
        set.push(
            "read",
            "Read a file (optionally a line range) in the workspace.",
            Box::new(ReadTool { max_output }),
        );
        set.push(
            "bash",
            "Run a shell command in the workspace directory.",
            Box::new(BashTool { max_output }),
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

    /// Build the tool set from config: the four built-ins plus every enabled
    /// MCP server (CRAB-133). Returns the set and non-fatal warnings for
    /// servers that failed to start — the agent runs with the tools that did.
    pub fn from_config(config: &crate::config::Config, max_output: usize) -> (Self, Vec<String>) {
        let mut set = Self::new(max_output);
        let (mcp_tools, warnings) = crate::mcp::connect_all(&config.mcp_servers, max_output);
        for tool in mcp_tools {
            let entry = ToolEntry {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                tool: Box::new(tool),
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

    /// Route `name`+`args` to the matching executor, or a clear error. Args
    /// are validated against the tool's declared JSON Schema first (CRAB-107
    /// #9), so malformed model calls never reach an executor. MCP tools carry
    /// server-authored JSON Schema, which `validate_args` handles as a
    /// superset of the subset the built-ins use.
    pub async fn execute(
        &self,
        workspace: &Workspace,
        name: &str,
        args: &Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        match self.tool(name) {
            Some(tool) => {
                super::validate_args(&tool.schema(), args)?;
                tool.run(workspace, args, cancel).await
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
        assert!(err.to_string().contains("read, bash, edit, write"));
    }

    #[test]
    fn exposes_four_tool_schemas() {
        let ts = ToolSet::new(1000);
        let schemas = ts.tool_schemas();
        assert_eq!(schemas.len(), 4);
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
            "mcp:echo:echo",
            "remote echo",
            Box::new(EchoTool {
                name: "mcp:echo:echo".into(),
            }),
        );
        assert_eq!(ts.tool_schemas().len(), 5);
        assert!(ts
            .listing()
            .iter()
            .any(|(n, d)| n == "mcp:echo:echo" && d == "remote echo"));
        let out = ts
            .execute(
                &ws,
                "mcp:echo:echo",
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
        assert!(err.to_string().contains("mcp:echo:echo"));
    }
}
