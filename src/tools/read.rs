//! The `read` tool: read a file, optionally a line range.

use std::path::Path;

use serde_json::{json, Value};

use super::{arg_string, arg_usize, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// Maximum number of lines returned per read (matches pi's default).
const MAX_LINES: usize = 2000;

pub struct ReadTool {
    pub max_output: usize,
}

/// Keep whole lines from the head that fit within `max_lines` and `max_bytes`
/// (the byte count accounts for the newlines that `join("\n")` will insert).
fn head_truncate<'a>(
    lines: &[&'a str],
    max_lines: usize,
    max_bytes: usize,
) -> (Vec<&'a str>, bool) {
    let mut kept = Vec::new();
    let mut bytes = 0usize;
    let mut truncated = false;
    for (i, line) in lines.iter().enumerate() {
        if i >= max_lines {
            truncated = true;
            break;
        }
        let add = line.len() + usize::from(i > 0);
        if bytes + add > max_bytes {
            truncated = true;
            break;
        }
        kept.push(*line);
        bytes += add;
    }
    (kept, truncated)
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
                "offset": { "type": "integer", "minimum": 1, "description": "1-indexed starting line." },
                "limit": { "type": "integer", "minimum": 1, "description": "Number of lines to return." }
            },
            "required": ["path"]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let offset = arg_usize(args, "offset")?.unwrap_or(1);
        if offset == 0 {
            return Err(ToolError::Argument("'offset' must be >= 1".into()));
        }
        let limit = arg_usize(args, "limit")?;
        if limit == Some(0) {
            return Err(ToolError::Argument("'limit' must be >= 1".into()));
        }

        let resolved = resolve(workspace, Path::new(&path))?;
        let content = match std::fs::read_to_string(&resolved) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path));
            }
            Err(e) => return Err(ToolError::Io(e.to_string())),
        };

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let start = offset - 1; // 0-based

        if start >= total_lines {
            return Err(ToolError::Argument(format!(
                "offset {offset} is beyond end of file ({total_lines} lines total)"
            )));
        }

        let end = match limit {
            Some(l) => std::cmp::min(start + l, total_lines),
            None => total_lines,
        };
        let selected = &lines[start..end];

        let (kept, truncated) = head_truncate(selected, MAX_LINES, self.max_output);
        let shown_start = start + 1; // 1-based
        let shown_end = start + kept.len();

        // A single line larger than the whole cap gets a targeted hint instead
        // of an empty result.
        if kept.is_empty() {
            let line = selected[0];
            return Ok(ToolOutput {
                content: format!(
                    "[Line {shown_start} is {} bytes, exceeds {} limit. Use bash: sed -n '{}p' {} | head -c {}]",
                    line.len(),
                    self.max_output,
                    shown_start,
                    path,
                    self.max_output
                ),
            });
        }

        let body = kept.join("\n");
        let mut out = body;
        if truncated {
            let next_offset = shown_end + 1;
            out = format!(
                "{out}\n\n[Showing lines {shown_start}-{shown_end} of {total_lines}. Use offset={next_offset} to continue.]"
            );
        } else if end < total_lines {
            let remaining = total_lines - end;
            let next_offset = end + 1;
            out = format!(
                "{out}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            );
        }

        Ok(ToolOutput { content: out })
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
    fn respects_one_based_offset_and_limit() {
        let (ws, _dir) = setup("range", "l0\nl1\nl2\nl3\nl4\n");
        let tool = ReadTool { max_output: 1000 };
        // offset=4 starts at "l3"; limit=2 reaches the end of the file.
        let out = tool
            .run(&ws, &json!({"path": "a.txt", "offset": 4, "limit": 2}))
            .unwrap();
        assert_eq!(out.content, "l3\nl4");
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
    fn caps_output_and_reports_continuation_offset() {
        let (ws, _dir) = setup("cap", &"y\n".repeat(5000));
        let tool = ReadTool { max_output: 64 };
        let out = tool.run(&ws, &json!({"path": "a.txt"})).unwrap();
        assert!(out.content.contains("[Showing lines 1-"));
        assert!(out.content.contains("Use offset="));
    }

    #[test]
    fn reports_remaining_lines_after_limit() {
        let (ws, _dir) = setup("limit", "l0\nl1\nl2\nl3\nl4\n");
        let tool = ReadTool { max_output: 1000 };
        let out = tool
            .run(&ws, &json!({"path": "a.txt", "offset": 1, "limit": 2}))
            .unwrap();
        assert_eq!(
            out.content,
            "l0\nl1\n\n[3 more lines in file. Use offset=3 to continue.]"
        );
    }

    #[test]
    fn huge_single_line_gets_targeted_hint() {
        let (ws, _dir) = setup("hugeline", &"z".repeat(5000));
        let tool = ReadTool { max_output: 64 };
        let out = tool.run(&ws, &json!({"path": "a.txt"})).unwrap();
        assert!(out.content.contains("exceeds 64 limit"));
        assert!(out.content.contains("sed -n '1p'"));
    }
}
