//! The `write` tool: create or overwrite a file in the workspace.

use futures::future::BoxFuture;
use serde_json::{json, Value};
use std::path::Path;
use tokio_util::sync::CancellationToken;

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

    fn run<'a>(
        &'a self,
        workspace: &'a Workspace,
        args: &'a Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || WriteTool.run_sync(&ws, &args))
                .await
                .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

impl WriteTool {
    /// The synchronous body, executed on the blocking pool (CRAB-130).
    fn run_sync(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
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

    fn setup(_name: &str) -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        (Workspace::new(root).unwrap(), dir)
    }

    #[tokio::test]
    async fn creates_file() {
        let (ws, dir) = setup("create");
        let tool = WriteTool;
        tool.run(
            &ws,
            &json!({"path": "b.txt", "content": "data"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
            "data"
        );
    }

    #[tokio::test]
    async fn overwrites_file() {
        let (ws, dir) = setup("overwrite");
        std::fs::write(dir.path().join("b.txt"), "old").unwrap();
        let tool = WriteTool;
        tool.run(
            &ws,
            &json!({"path": "b.txt", "content": "new"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
            "new"
        );
    }

    #[tokio::test]
    async fn creates_parent_dirs() {
        let (ws, dir) = setup("parents");
        let tool = WriteTool;
        tool.run(
            &ws,
            &json!({"path": "a/b/c.txt", "content": "deep"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "deep"
        );
    }
}
