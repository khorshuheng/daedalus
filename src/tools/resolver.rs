//! The resolver maps a `{name, args}` pair to the correct tool executor and
//! surfaces unknown-tool / bad-args / not-found errors clearly.

use serde_json::{json, Value};

use super::{bash::BashTool, edit::EditTool, read::ReadTool, write::WriteTool, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// The fixed, closed set of four tools, all sharing a single output cap.
pub struct ToolSet {
    read: ReadTool,
    bash: BashTool,
    edit: EditTool,
    write: WriteTool,
}

impl ToolSet {
    pub fn new(max_output: usize) -> Self {
        Self {
            read: ReadTool { max_output },
            bash: BashTool { max_output },
            edit: EditTool { max_output },
            write: WriteTool { max_output },
        }
    }

    /// The names of the four built-in tools, in canonical order.
    pub fn names(&self) -> [&'static str; 4] {
        ["read", "bash", "edit", "write"]
    }

    /// JSON Schemas describing each tool's arguments, wrapped with the tool's
    /// name for the provider `tools` field.
    pub fn tool_schemas(&self) -> Vec<Value> {
        self.names()
            .iter()
            .map(|&n| {
                let s = self.tool(n).expect("tool exists").schema();
                json!({ "name": n, "description": self.description(n), "parameters": s })
            })
            .collect()
    }

    fn description(&self, name: &str) -> &'static str {
        match name {
            "read" => "Read a file (optionally a line range) in the workspace.",
            "bash" => "Run a shell command in the workspace directory.",
            "edit" => "Apply precise, validated text replacements to a file.",
            "write" => "Create or overwrite a file in the workspace.",
            _ => "A built-in crab tool.",
        }
    }

    pub fn tool(&self, name: &str) -> Option<&dyn Tool> {
        match name {
            "read" => Some(&self.read),
            "bash" => Some(&self.bash),
            "edit" => Some(&self.edit),
            "write" => Some(&self.write),
            _ => None,
        }
    }

    /// Route `name`+`args` to the matching executor, or a clear error.
    pub fn execute(
        &self,
        workspace: &Workspace,
        name: &str,
        args: &Value,
    ) -> Result<ToolOutput, ToolError> {
        match self.tool(name) {
            Some(tool) => tool.run(workspace, args),
            None => Err(ToolError::Argument(format!(
                "unknown tool '{name}' (expected read, bash, edit, write)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_to_correct_executor() {
        let dir = std::env::temp_dir().join(format!("crab-resolver-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        let ts = ToolSet::new(1000);

        let out = ts.execute(&ws, "bash", &json!({"command": "echo hi"})).unwrap();
        assert!(out.content.contains("hi"));
    }

    #[test]
    fn unknown_tool_is_clear_error() {
        let dir = std::env::temp_dir().join(format!("crab-resolver2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        let ts = ToolSet::new(1000);
        let err = ts.execute(&ws, "frobnicate", &json!({})).unwrap_err();
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
            assert!(s.get("parameters").is_some());
        }
    }
}
