//! The `read` tool: read a file, optionally a line range.

use std::path::Path;

use serde_json::{json, Value};

use super::{arg_string, arg_usize, resolve, truncate, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct ReadTool {
    pub max_output: usize,
}

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to read, relative to the workspace." },
                "offset": { "type": "integer", "minimum": 0, "description": "0-based starting line." },
                "limit": { "type": "integer", "minimum": 1, "description": "Number of lines to return." }
            },
            "required": ["path"]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let offset = arg_usize(args, "offset")?.unwrap_or(0);
        let limit = arg_usize(args, "limit")?;

        let resolved = resolve(workspace, Path::new(&path))?;
        let content = match std::fs::read_to_string(&resolved) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path));
            }
            Err(e) => return Err(ToolError::Io(e.to_string())),
        };

        let lines: Vec<&str> = content.lines().collect();

        let selected: String = if let Some(limit) = limit {
            if offset >= lines.len() {
                String::new()
            } else {
                lines[offset..std::cmp::min(offset + limit, lines.len())]
                    .to_vec()
                    .join("\n")
            }
        } else if offset == 0 {
            lines.join("\n")
        } else if offset >= lines.len() {
            String::new()
        } else {
            lines[offset..].join("\n")
        };

        let body = truncate(selected, self.max_output);
        let content = if body.contains("[truncated") {
            format!("{body}\n[file has {} lines]", lines.len())
        } else {
            body
        };
        Ok(ToolOutput { content })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str, contents: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-read-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), contents).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    #[test]
    fn reads_full_file() {
        let (ws, _dir) = setup("full", "line1\nline2\nline3\n");
        let tool = ReadTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"path": "a.txt"})).unwrap();
        assert_eq!(out.content, "line1\nline2\nline3");
    }

    #[test]
    fn respects_offset_and_limit() {
        let (ws, _dir) = setup("range", "l0\nl1\nl2\nl3\nl4\n");
        let tool = ReadTool { max_output: 1000 };
        let out = tool
            .run(&ws, &json!({"path": "a.txt", "offset": 1, "limit": 2}))
            .unwrap();
        assert_eq!(out.content, "l1\nl2");
    }

    #[test]
    fn missing_file_is_not_found() {
        let (ws, _dir) = setup("missing", "x\n");
        let tool = ReadTool { max_output: 1000 };
        assert!(matches!(
            tool.run(&ws, &json!({"path": "nope.txt"})),
            Err(ToolError::NotFound(_))
        ));
    }

    #[test]
    fn caps_output() {
        let (ws, _dir) = setup("cap", &"y".repeat(5000));
        let tool = ReadTool { max_output: 64 };
        let out = tool.run(&ws, &json!({"path": "a.txt"})).unwrap();
        assert!(out.content.contains("[truncated"));
    }
}
