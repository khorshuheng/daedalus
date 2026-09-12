//! The `read` tool: read a file, optionally a line range.

use futures::future::BoxFuture;
use serde_json::{json, Value};
use std::io::BufRead;
use std::path::Path;
use tokio_util::sync::CancellationToken;

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

    fn run<'a>(
        &'a self,
        workspace: &'a Workspace,
        args: &'a Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        let max_output = self.max_output;
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || ReadTool { max_output }.run_sync(&ws, &args))
                .await
                .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

/// Files at or below this size are read whole (the continuation hint keeps
/// exact totals); larger files are streamed so a huge file is never loaded
/// into memory just to return a capped slice.
const MAX_FULL_READ: u64 = 4 * 1024 * 1024;

impl ReadTool {
    /// The synchronous body, executed on the blocking pool (CRAB-130).
    fn run_sync(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
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
        // Refuse anything that is not a regular file: opening a FIFO, device,
        // or socket blocks until a peer appears, which hangs the turn (and the
        // blocking task cannot be cancelled). Directories are not readable
        // text either. `metadata` stats without opening, so it cannot block.
        let meta = match std::fs::metadata(&resolved) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path));
            }
            Err(e) => return Err(ToolError::Io(e.to_string())),
        };
        if !meta.is_file() {
            return Err(ToolError::Invalid(format!(
                "refusing to read '{path}': not a regular file"
            )));
        }

        if meta.len() <= MAX_FULL_READ {
            self.read_whole(&resolved, &path, offset, limit)
        } else {
            self.read_streaming(&resolved, &path, offset, limit)
        }
    }

    /// Read a small file whole; the continuation hint keeps exact totals.
    fn read_whole(
        &self,
        resolved: &Path,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<ToolOutput, ToolError> {
        let content = match std::fs::read_to_string(resolved) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path.to_string()));
            }
            Err(e) => return Err(ToolError::Io(e.to_string())),
        };

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let start = offset - 1; // 0-based

        if total_lines == 0 {
            return Ok(ToolOutput {
                content: "(empty file)".to_string(),
            });
        }

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

    /// Stream a large file, stopping at `limit`/`MAX_LINES`/`max_output`
    /// instead of loading it whole. The hint omits the exact total (which
    /// would require reading the rest of the file).
    fn read_streaming(
        &self,
        resolved: &Path,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<ToolOutput, ToolError> {
        let file = std::fs::File::open(resolved).map_err(|e| ToolError::Io(e.to_string()))?;
        let reader = std::io::BufReader::new(file);
        let max_lines = limit.unwrap_or(MAX_LINES).min(MAX_LINES);
        let start = offset - 1; // 0-based lines to skip
        let mut kept: Vec<String> = Vec::new();
        let mut bytes = 0usize;
        let mut line_no = 0usize;
        let mut more = false;

        for line in reader.lines() {
            let line = line.map_err(|e| ToolError::Io(e.to_string()))?;
            line_no += 1;
            if line_no <= start {
                continue;
            }
            if kept.len() >= max_lines {
                more = true;
                break;
            }
            let add = line.len() + usize::from(!kept.is_empty());
            if bytes + add > self.max_output {
                if kept.is_empty() {
                    return Ok(ToolOutput {
                        content: format!(
                            "[Line {line_no} is {} bytes, exceeds {} limit. Use bash: sed -n '{}p' {} | head -c {}]",
                            line.len(),
                            self.max_output,
                            line_no,
                            path,
                            self.max_output
                        ),
                    });
                }
                more = true;
                break;
            }
            kept.push(line);
            bytes += add;
        }

        if line_no == 0 {
            return Ok(ToolOutput {
                content: "(empty file)".to_string(),
            });
        }
        if kept.is_empty() {
            return Err(ToolError::Argument(format!(
                "offset {offset} is beyond end of file ({line_no} lines total)"
            )));
        }

        let shown_start = offset;
        let shown_end = offset + kept.len() - 1;
        let body = kept.join("\n");
        let out = if more {
            format!(
                "{body}\n\n[Showing lines {shown_start}-{shown_end}. Use offset={} to continue.]",
                shown_end + 1
            )
        } else {
            body
        };
        Ok(ToolOutput { content: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block on a tool future (tests are sync; CRAB-130).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }
    use crate::workspace::Workspace;

    fn setup(_name: &str, contents: &str) -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("a.txt"), contents).unwrap();
        (Workspace::new(root).unwrap(), dir)
    }

    #[tokio::test]
    async fn reads_full_file() {
        let (ws, _dir) = setup("full", "line1\nline2\nline3\n");
        let tool = ReadTool { max_output: 1000 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "line1\nline2\nline3");
    }

    #[tokio::test]
    async fn respects_one_based_offset_and_limit() {
        let (ws, _dir) = setup("range", "l0\nl1\nl2\nl3\nl4\n");
        let tool = ReadTool { max_output: 1000 };
        // offset=4 starts at "l3"; limit=2 reaches the end of the file.
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "offset": 4, "limit": 2}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "l3\nl4");
    }

    #[test]
    fn missing_file_is_not_found() {
        let (ws, _dir) = setup("missing", "x\n");
        let tool = ReadTool { max_output: 1000 };
        assert!(matches!(
            block_on(tool.run(
                &ws,
                &json!({"path": "nope.txt"}),
                tokio_util::sync::CancellationToken::new()
            )),
            Err(ToolError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn caps_output_and_reports_continuation_offset() {
        let (ws, _dir) = setup("cap", &"y\n".repeat(5000));
        let tool = ReadTool { max_output: 64 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("[Showing lines 1-"));
        assert!(out.content.contains("Use offset="));
    }

    #[tokio::test]
    async fn reports_remaining_lines_after_limit() {
        let (ws, _dir) = setup("limit", "l0\nl1\nl2\nl3\nl4\n");
        let tool = ReadTool { max_output: 1000 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "offset": 1, "limit": 2}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            out.content,
            "l0\nl1\n\n[3 more lines in file. Use offset=3 to continue.]"
        );
    }

    #[tokio::test]
    async fn reads_empty_file() {
        let (ws, _dir) = setup("empty", "");
        let tool = ReadTool { max_output: 1000 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.content, "(empty file)");
    }

    #[tokio::test]
    async fn huge_single_line_gets_targeted_hint() {
        let (ws, _dir) = setup("hugeline", &"z".repeat(5000));
        let tool = ReadTool { max_output: 64 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "a.txt"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("exceeds 64 limit"));
        assert!(out.content.contains("sed -n '1p'"));
    }

    /// CRAB-139 review: a FIFO (or any non-regular file) must be rejected
    /// instead of blocking forever on open.
    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_special_files_instead_of_blocking() {
        use std::ffi::CString;
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let fifo = root.join("pipe");
        let c_path = CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: mkfifo with a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) }, 0);
        let ws = Workspace::new(root).unwrap();
        let tool = ReadTool { max_output: 1000 };
        let err = tool
            .run(
                &ws,
                &json!({"path": "pipe"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
    }

    /// A file over the whole-read threshold is streamed, so the output stays
    /// bounded instead of materialising the whole file.
    #[tokio::test]
    async fn large_files_are_streamed_and_bounded() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let mut body = String::new();
        while body.len() < (5 * 1024 * 1024) {
            body.push_str("abcdefghij\n");
        }
        std::fs::write(root.join("big.txt"), &body).unwrap();
        let ws = Workspace::new(root).unwrap();
        let tool = ReadTool { max_output: 64 };
        let out = tool
            .run(
                &ws,
                &json!({"path": "big.txt"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(out.content.len() < 512, "bounded: {}", out.content.len());
        assert!(out.content.contains("Use offset="), "{}", out.content);
    }
}
