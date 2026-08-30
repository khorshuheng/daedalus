//! The `write` tool: create or overwrite a file in the workspace.

use std::path::Path;

use serde_json::{json, Value};

use super::{arg_string, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to create/overwrite." },
                "content": { "type": "string", "description": "Full contents to write." }
            },
            "required": ["path", "content"]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Argument("'content' must be a string".into()))?;

        let resolved = resolve(workspace, Path::new(&path))?;
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
        }
        std::fs::write(&resolved, content).map_err(|e| ToolError::Io(e.to_string()))?;

        Ok(ToolOutput {
            content: format!("wrote {} bytes to {}", content.len(), path),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-write-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    #[test]
    fn creates_file() {
        let (ws, dir) = setup("create");
        let tool = WriteTool;
        tool.run(&ws, &json!({"path": "b.txt", "content": "data"}))
            .unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("b.txt")).unwrap(), "data");
    }

    #[test]
    fn overwrites_file() {
        let (ws, dir) = setup("overwrite");
        std::fs::write(dir.join("b.txt"), "old").unwrap();
        let tool = WriteTool;
        tool.run(&ws, &json!({"path": "b.txt", "content": "new"}))
            .unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("b.txt")).unwrap(), "new");
    }

    #[test]
    fn creates_parent_dirs() {
        let (ws, dir) = setup("parents");
        let tool = WriteTool;
        tool.run(&ws, &json!({"path": "a/b/c.txt", "content": "deep"}))
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("a/b/c.txt")).unwrap(),
            "deep"
        );
    }
}
