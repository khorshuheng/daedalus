//! The `write` tool: create or overwrite a file in the workspace.

use futures::future::BoxFuture;
use serde_json::{json, Value};
use std::path::Path;
use tokio_util::sync::CancellationToken;

use super::mutation::with_file_mutation;
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
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || WriteTool.run_sync(&ws, &args, &cancel))
                .await
                .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

impl WriteTool {
    /// The synchronous body, executed on the blocking pool (CRAB-130).
    fn run_sync(
        &self,
        workspace: &Workspace,
        args: &Value,
        cancel: &CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Argument("'content' must be a string".into()))?;

        let resolved = resolve(workspace, Path::new(&path))?;
        // Serialize with any concurrent write/edit of the same file (CRAB-146).
        with_file_mutation(&resolved, || write_at(&resolved, &path, content, cancel))
    }
}

/// The locked body of `write`: reject non-regular targets, create parents, and
/// write the content.
fn write_at(
    resolved: &Path,
    path: &str,
    content: &str,
    cancel: &CancellationToken,
) -> Result<ToolOutput, ToolError> {
    // Check before touching the filesystem so an aborted turn cannot write
    // (CRAB-152); the single `fs::write` below cannot be interrupted anyway.
    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }
    // A FIFO/device/socket at this path would block `fs::write` on open, so
    // reject an existing non-regular file. A missing path is a normal
    // create (CRAB-139 review).
    if let Ok(meta) = std::fs::metadata(resolved) {
        if !meta.is_file() {
            return Err(ToolError::Invalid(format!(
                "refusing to write '{path}': not a regular file"
            )));
        }
    }
    if let Some(parent) = resolved.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
    }
    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }
    std::fs::write(resolved, content).map_err(|e| ToolError::Io(e.to_string()))?;

    Ok(ToolOutput {
        content: format!("wrote {} bytes to {}", content.len(), path),
    })
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

    /// CRAB-152: a pre-cancelled token means the write never happens.
    #[tokio::test]
    async fn cancelled_write_leaves_no_file() {
        let (ws, dir) = setup("cancel");
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let err = WriteTool
            .run(&ws, &json!({"path": "b.txt", "content": "x"}), cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled), "{err}");
        assert!(!dir.path().join("b.txt").exists());
    }

    /// CRAB-139 review: writing to an existing FIFO would block on open.
    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_writing_to_non_regular_files() {
        use std::ffi::CString;
        let (ws, dir) = setup("fifo");
        let fifo = dir.path().join("pipe");
        let c = CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: mkfifo with a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
        let err = WriteTool
            .run(
                &ws,
                &json!({"path": "pipe", "content": "x"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
    }
}
