//! The resolver maps a `{name, args}` pair to the correct tool executor and
//! surfaces unknown-tool / bad-args / not-found errors clearly.

use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::{
    bash::BashTool, edit::EditTool, read::ReadTool, write::WriteTool, Tool, ToolError, ToolOutput,
};
use crate::workspace::Workspace;

/// A registered tool: its name, a one-line description for the provider's
/// `tools` field, and its executor. This table is the single source of truth
/// for the closed tool set.
type ToolEntry = (&'static str, &'static str, Box<dyn Tool>);

/// The fixed, closed set of four tools, all sharing a single output cap.
pub struct ToolSet {
    tools: Vec<ToolEntry>,
}

impl ToolSet {
    pub fn new(max_output: usize) -> Self {
        Self {
            tools: vec![
                (
                    "read",
                    "Read a file (optionally a line range) in the workspace.",
                    Box::new(ReadTool { max_output }),
                ),
                (
                    "bash",
                    "Run a shell command in the workspace directory.",
                    Box::new(BashTool { max_output }),
                ),
                (
                    "edit",
                    "Apply precise, validated text replacements to a file.",
                    Box::new(EditTool),
                ),
                (
                    "write",
                    "Create or overwrite a file in the workspace.",
                    Box::new(WriteTool),
                ),
            ],
        }
    }

    /// JSON Schemas describing each tool's arguments, wrapped with the tool's
    /// name and description for the provider `tools` field.
    pub fn tool_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|(name, description, tool)| {
                json!({
                    "name": *name,
                    "description": *description,
                    "parameters": tool.schema()
                })
            })
            .collect()
    }

    pub fn tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, _, t)| t.as_ref())
    }

    /// Route `name`+`args` to the matching executor, or a clear error.
    pub fn execute(
        &self,
        workspace: &Workspace,
        name: &str,
        args: &Value,
        cancel: &AtomicBool,
    ) -> Result<ToolOutput, ToolError> {
        match self.tool(name) {
            Some(tool) => tool.run(workspace, args, cancel),
            None => Err(ToolError::Argument(format!(
                "unknown tool '{name}' (expected read, bash, edit, write)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace(_name: &str) -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        let ws = Workspace::new(root).unwrap();
        (dir, ws)
    }

    #[test]
    fn routes_to_correct_executor() {
        let (_dir, ws) = workspace("bash");
        let ts = ToolSet::new(1000);

        let out = ts
            .execute(
                &ws,
                "bash",
                &json!({"command": "echo hi"}),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(out.content.contains("hi"));
    }

    #[test]
    fn unknown_tool_is_clear_error() {
        let (_dir, ws) = workspace("unknown");
        let ts = ToolSet::new(1000);
        let err = ts
            .execute(&ws, "frobnicate", &json!({}), &AtomicBool::new(false))
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
}
