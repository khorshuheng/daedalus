//! The `bash` tool: run a shell command in the workspace directory.
//!
//! The shell runs in its own process group (Unix) so a timeout or cancel can
//! kill the whole tree — including grandchildren — not just the direct child
//! (CRAB-107 #14).
//!
//! Known limitation: a command that backgrounds a process while keeping
//! stdout/stderr open (e.g. `sh -c "sleep 100 &"`) will block until that
//! process exits, because the readers wait for the pipes to reach EOF.

use futures::future::BoxFuture;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use serde_json::{json, Value};

use super::{arg_string, arg_usize, truncate_tail, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct BashTool {
    pub max_output: usize,
}

/// Run `sh -c <command>` capturing stdout/stderr, with an optional timeout in
/// seconds. Both pipes are drained on background threads so a chatty child
/// cannot deadlock the parent on a full pipe buffer while we wait.
fn run_command(
    command: &str,
    cwd: &Path,
    timeout_secs: Option<u64>,
    cancel: &CancellationToken,
) -> Result<Output, ToolError> {
    use std::io::Read;
    use std::time::{Duration, Instant};

    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Run the shell in its own session/process group so we can kill the
    // whole tree on timeout/cancel (pi semantics).
    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            // setsid: detach into a new session whose process group id == pid.
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| ToolError::Io(e.to_string()))?;

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
                if cancel.is_cancelled() {
                    kill_tree(&mut child);
                    return Err(ToolError::Cancelled);
                }
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        kill_tree(&mut child);
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

/// Kill the whole process tree rooted at `child`. On Unix the child was
/// started in its own session (setsid), so its process-group id equals its
/// pid and `kill(-pid, SIGKILL)` reaches every descendant; the direct child
/// is then reaped. On other platforms only the direct child is killed.
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as libc::pid_t;
        // Negative pid => the whole process group.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Write the full command output to a temp file and return its path, so a
/// truncated result still lets the model retrieve the complete output. The
/// file is kept after the write (the caller embeds its path in the tool
/// result), so the temp file is created with `keep` semantics.
fn write_full_output(full: &str) -> Result<std::path::PathBuf, ToolError> {
    let mut file = tempfile::Builder::new()
        .prefix("crab-bash-")
        .suffix(".log")
        .tempfile()
        .map_err(|e| ToolError::Io(format!("cannot create temp file: {e}")))?;
    std::io::Write::write_all(&mut file, full.as_bytes())
        .map_err(|e| ToolError::Io(e.to_string()))?;
    let (_, path) = file.keep().map_err(|e| ToolError::Io(e.to_string()))?;
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

    fn run<'a>(
        &'a self,
        workspace: &'a Workspace,
        args: &'a Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        let max_output = self.max_output;
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || {
                BashTool { max_output }.run_sync(&ws, &args, cancel)
            })
            .await
            .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

impl BashTool {
    /// The synchronous body, executed on the blocking pool (CRAB-130).
    fn run_sync(
        &self,
        workspace: &Workspace,
        args: &Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        let command = arg_string(args, "command")?;
        let timeout = match arg_usize(args, "timeout")? {
            Some(0) => return Err(ToolError::Argument("'timeout' must be >= 1".into())),
            Some(s) => Some(s as u64),
            None => None,
        };

        let output = run_command(&command, workspace.root(), timeout, &cancel)?;

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

    fn setup(_name: &str) -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        (Workspace::new(root).unwrap(), dir)
    }

    /// Block on a tool future (tests are sync; CRAB-130).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    #[tokio::test]
    async fn captures_stdout_and_exit_code() {
        let (ws, _dir) = setup("out");
        let tool = BashTool { max_output: 1000 };
        let out = tool
            .run(&ws, &json!({"command": "echo hello"}), token())
            .await
            .unwrap();
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("exit code: 0"));
    }

    #[tokio::test]
    async fn nonzero_exit_is_a_tool_error() {
        let (ws, _dir) = setup("err");
        let tool = BashTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"command": "echo boo 1>&2; exit 3"}), token())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("stderr:"));
        assert!(msg.contains("boo"));
        assert!(msg.contains("Command exited with code 3"));
    }

    #[tokio::test]
    async fn runs_in_workspace_dir() {
        let (ws, dir) = setup("pwd");
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let tool = BashTool { max_output: 1000 };
        let out = tool
            .run(&ws, &json!({"command": "ls"}), token())
            .await
            .unwrap();
        assert!(out.content.contains("marker.txt"));
    }

    #[tokio::test]
    async fn caps_output_from_the_tail() {
        let (ws, _dir) = setup("cap");
        let tool = BashTool { max_output: 64 };
        // 10000 '1's: the tail (last bytes) is kept, the head is dropped.
        let out = tool
            .run(
                &ws,
                &json!({"command": "printf '%.0s1' {1..10000}"}),
                token(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("[truncated"));
        assert!(out.content.contains("[full output:"));
    }

    #[tokio::test]
    async fn timeout_kills_command() {
        let (ws, _dir) = setup("timeout");
        let tool = BashTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"command": "sleep 5", "timeout": 1}), token())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }

    #[tokio::test]
    async fn timeout_returns_promptly_when_command_forks() {
        let (ws, _dir) = setup("timeout-fork");
        let tool = BashTool { max_output: 1000 };
        let start = std::time::Instant::now();
        let err = tool
            .run(
                &ws,
                &json!({"command": "sleep 2 && echo done", "timeout": 1}),
                token(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
        assert!(start.elapsed() < std::time::Duration::from_millis(1500));
    }

    #[test]
    fn cancel_kills_running_command() {
        let (ws, _dir) = setup("cancel");
        let tool = BashTool { max_output: 1000 };
        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        let handle = std::thread::spawn(move || {
            block_on(tool.run(&ws, &json!({"command": "sleep 5"}), cancel2))
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        cancel.cancel();
        let result = handle.join().unwrap();
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_grandchild_processes() {
        let (ws, dir) = setup("grandchild");
        let tool = BashTool { max_output: 1000 };
        // Start a backgrounded grandchild that writes its PID and sleeps; the
        // direct `sh` exits immediately but the grandchild must not survive
        // the group kill.
        let marker = dir.path().join("gc.pid");
        // The direct `sh` backgrounds a grandchild, then waits on it, so the
        // timeout fires while both the direct child and the grandchild are
        // alive. The group kill must reap both.
        let cmd = format!("sleep 30 & echo $! > '{}'; wait", marker.display());
        let err = tool
            .run(&ws, &json!({"command": cmd, "timeout": 1}), token())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
        // Give the kill a moment to land, then confirm the grandchild is gone.
        std::thread::sleep(std::time::Duration::from_millis(300));
        let pid: i32 = std::fs::read_to_string(&marker)
            .map(|s| s.trim().parse().unwrap_or(0))
            .unwrap_or(0);
        assert!(
            pid > 0,
            "grandchild should have started and recorded its pid"
        );
        // kill(pid, 0) returns 0 while the process exists; -1 (ESRCH) means gone.
        let alive = unsafe { libc::kill(pid, 0) == 0 };
        assert!(
            !alive,
            "grandchild {pid} should have been killed with the group"
        );
    }
}
