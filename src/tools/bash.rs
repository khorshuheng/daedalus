//! The `bash` tool: run a shell command in the workspace directory.
//!
//! Known limitation: a command that backgrounds a process while keeping
//! stdout/stderr open (e.g. `sh -c "sleep 100 &"`) will block until that
//! process exits, because the readers wait for the pipes to reach EOF.

use std::path::Path;
use std::process::{Command, Output, Stdio};

use serde_json::{json, Value};

use super::{arg_string, arg_usize, truncate_tail, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct BashTool {
    pub max_output: usize,
}

/// Run `sh -c <command>` capturing stdout/stderr, with an optional timeout in
/// seconds. Both pipes are drained on background threads so a chatty child
/// cannot deadlock the parent on a full pipe buffer while we wait.
fn run_command(command: &str, cwd: &Path, timeout_secs: Option<u64>) -> Result<Output, ToolError> {
    use std::io::Read;
    use std::time::{Duration, Instant};

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ToolError::Io(e.to_string()))?;

    let mut stdout_pipe = child.stdout.take().unwrap();
    let mut stderr_pipe = child.stderr.take().unwrap();

    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    let deadline = timeout_secs.map(|s| Instant::now() + Duration::from_secs(s));
    let status = loop {
        match child.try_wait().map_err(|e| ToolError::Io(e.to_string()))? {
            Some(status) => break status,
            None => {
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        // Do NOT join the reader threads here: a grandchild of
                        // `sh` (e.g. `sh -c "a && b"`) may still hold the pipe
                        // write end open, so joining would block past the
                        // deadline. The threads are abandoned and finish once
                        // every writer has exited.
                        return Err(ToolError::Timeout(format!(
                            "after {} seconds",
                            timeout_secs.unwrap()
                        )));
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };

    let _ = child.wait();
    let stdout = out_handle
        .join()
        .map_err(|_| ToolError::Io("stdout reader panicked".into()))?;
    let stderr = err_handle
        .join()
        .map_err(|_| ToolError::Io("stderr reader panicked".into()))?;

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// Write the full command output to a temp file and return its path, so a
/// truncated result still lets the model retrieve the complete output.
fn write_full_output(full: &str) -> Result<std::path::PathBuf, ToolError> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("crab-bash-{}-{nanos}.log", std::process::id()));
    std::fs::write(&path, full).map_err(|e| ToolError::Io(e.to_string()))?;
    Ok(path)
}

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run in the workspace." },
                "timeout": { "type": "integer", "minimum": 1, "description": "Timeout in seconds (optional, no default)." }
            },
            "required": ["command"]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let command = arg_string(args, "command")?;
        let timeout = match arg_usize(args, "timeout")? {
            Some(0) => return Err(ToolError::Argument("'timeout' must be >= 1".into())),
            Some(s) => Some(s as u64),
            None => None,
        };

        let output = run_command(&command, workspace.root(), timeout)?;

        let mut text = String::new();
        if !output.stdout.is_empty() {
            text.push_str("stdout:\n");
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push('\n');
        }
        if !output.stderr.is_empty() {
            text.push_str("stderr:\n");
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            text.push('\n');
        }

        let (truncated, was_truncated) = truncate_tail(&text, self.max_output);
        let mut result = truncated;
        if was_truncated {
            let full_path = write_full_output(&text)?;
            result.push_str(&format!("\n\n[full output: {}]", full_path.display()));
        }

        match output.status.code() {
            Some(0) => {
                if result.trim().is_empty() {
                    Ok(ToolOutput {
                        content: "(no output)".to_string(),
                    })
                } else {
                    Ok(ToolOutput {
                        content: format!("{result}\nexit code: 0"),
                    })
                }
            }
            Some(code) => Err(ToolError::Command(format!(
                "{result}\n\nCommand exited with code {code}"
            ))),
            None => Err(ToolError::Command(format!(
                "{result}\n\nCommand terminated by signal"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-bash-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    #[test]
    fn captures_stdout_and_exit_code() {
        let (ws, _dir) = setup("out");
        let tool = BashTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"command": "echo hello"})).unwrap();
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("exit code: 0"));
    }

    #[test]
    fn nonzero_exit_is_a_tool_error() {
        let (ws, _dir) = setup("err");
        let tool = BashTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"command": "echo boo 1>&2; exit 3"}))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("stderr:"));
        assert!(msg.contains("boo"));
        assert!(msg.contains("Command exited with code 3"));
    }

    #[test]
    fn runs_in_workspace_dir() {
        let (ws, dir) = setup("pwd");
        std::fs::write(dir.join("marker.txt"), "x").unwrap();
        let tool = BashTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"command": "ls"})).unwrap();
        assert!(out.content.contains("marker.txt"));
    }

    #[test]
    fn caps_output_from_the_tail() {
        let (ws, _dir) = setup("cap");
        let tool = BashTool { max_output: 64 };
        // 10000 '1's: the tail (last bytes) is kept, the head is dropped.
        let out = tool
            .run(&ws, &json!({"command": "printf '%.0s1' {1..10000}"}))
            .unwrap();
        assert!(out.content.contains("[truncated"));
        assert!(out.content.contains("[full output:"));
    }

    #[test]
    fn timeout_kills_command() {
        let (ws, _dir) = setup("timeout");
        let tool = BashTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"command": "sleep 5", "timeout": 1}))
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }

    #[test]
    fn timeout_returns_promptly_when_command_forks() {
        let (ws, _dir) = setup("timeout-fork");
        let tool = BashTool { max_output: 1000 };
        let start = std::time::Instant::now();
        let err = tool
            .run(
                &ws,
                &json!({"command": "sleep 2 && echo done", "timeout": 1}),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
        assert!(start.elapsed() < std::time::Duration::from_millis(1500));
    }
}
