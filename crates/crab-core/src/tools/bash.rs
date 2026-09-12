//! The `bash` tool: run a shell command in the workspace directory.
//!
//! The shell runs in its own process group (Unix) so a timeout or cancel can
//! kill the whole tree — including grandchildren — not just the direct child
//! (CRAB-107 #14).
//!
//! stdout/stderr are drained on reader threads. A command that leaves a
//! background process holding the pipes open (`sh -c "sleep 100 &"`) does not
//! block the tool: after the direct child exits the readers get a short grace
//! to drain, then are stopped (CRAB-147, mirroring pi's `EXIT_STDIO_GRACE_MS`).

use futures::future::BoxFuture;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use serde_json::{json, Value};

use super::{arg_string, arg_usize, truncate_tail, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct BashTool {
    pub max_output: usize,
    /// Applied when the model omits `timeout`, so no command can run forever.
    /// `None` disables the default (tests, or `bash_timeout_secs = 0`).
    pub default_timeout_secs: Option<u64>,
}

/// How long after the direct child exits to let the stdout/stderr readers reach
/// EOF before proceeding without them (CRAB-147). A descendant that keeps a
/// pipe write end open must not hold the tool hostage.
const EXIT_STDIO_GRACE: Duration = Duration::from_millis(150);

/// Upper bound for a model-supplied `timeout`, matching pi's `MAX_TIMEOUT_MS`
/// (2_147_483_647 ms). Keeps `Instant + Duration` far from overflow and rejects
/// nonsense values with a clear error instead of a panic in the blocking task
/// (CRAB-153).
const MAX_TIMEOUT_SECS: u64 = i32::MAX as u64;

/// State shared between `run_command` and its reader threads. stdout and
/// stderr append to one buffer in arrival order (CRAB-155), matching pi's
/// single `OutputAccumulator`.
struct StreamCapture {
    buf: Mutex<Vec<u8>>,
    /// Set once the parent has stopped caring; readers then stop appending, so
    /// a chatty descendant cannot grow the buffer after the result is read.
    stop: AtomicBool,
    /// Number of reader threads that have reached EOF.
    done: AtomicUsize,
}

impl StreamCapture {
    fn new() -> Self {
        Self {
            buf: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            done: AtomicUsize::new(0),
        }
    }
}

/// Drain `pipe` into the shared buffer until EOF or the `stop` flag is set
/// (CRAB-147/155). The thread exits on EOF; a descendant that keeps the pipe
/// open can leave it blocked in `read`, which is deliberate — the parent never
/// joins it.
fn drain(mut pipe: impl std::io::Read + Send + 'static, capture: Arc<StreamCapture>) {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if capture.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    capture
                        .buf
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend_from_slice(&chunk[..n]);
                }
                Err(_) => break,
            }
        }
        capture.done.fetch_add(1, Ordering::SeqCst);
    });
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

    let capture = Arc::new(StreamCapture::new());
    if let Some(stdout) = child.stdout.take() {
        drain(stdout, Arc::clone(&capture));
    }
    if let Some(stderr) = child.stderr.take() {
        drain(stderr, Arc::clone(&capture));
    }

    // `checked_add` so an absurd timeout is a bad argument, not a panic
    // (defense in depth behind the argument/schema clamp in CRAB-153).
    let deadline = match timeout_secs {
        Some(s) => Some(
            Instant::now()
                .checked_add(Duration::from_secs(s))
                .ok_or_else(|| ToolError::Argument(format!("'timeout' is too large: {s}")))?,
        ),
        None => None,
    };
    let status = loop {
        match child.try_wait().map_err(|e| ToolError::Io(e.to_string()))? {
            Some(status) => break status,
            None => {
                if cancel.is_cancelled() {
                    capture.stop.store(true, Ordering::SeqCst);
                    kill_tree(&mut child);
                    return Err(ToolError::Cancelled);
                }
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        capture.stop.store(true, Ordering::SeqCst);
                        kill_tree(&mut child);
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
    // Let the readers drain what the child already wrote, then stop them, so a
    // descendant holding the pipe cannot block the join or grow the buffer
    // (CRAB-147).
    let drain_deadline = Instant::now() + EXIT_STDIO_GRACE;
    while capture.done.load(Ordering::SeqCst) < 2 && Instant::now() < drain_deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    capture.stop.store(true, Ordering::SeqCst);
    let merged = std::mem::take(&mut *capture.buf.lock().unwrap_or_else(|e| e.into_inner()));

    Ok(Output {
        status,
        stdout: merged,
        stderr: Vec::new(),
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
                "timeout": { "type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECS, "description": "Timeout in seconds (optional; a server default applies otherwise)." }
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
        let default_timeout_secs = self.default_timeout_secs;
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || {
                BashTool {
                    max_output,
                    default_timeout_secs,
                }
                .run_sync(&ws, &args, cancel)
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
            Some(s) if s as u64 > MAX_TIMEOUT_SECS => {
                return Err(ToolError::Argument(format!(
                    "'timeout' must be <= {MAX_TIMEOUT_SECS}"
                )))
            }
            Some(s) => Some(s as u64),
            None => self.default_timeout_secs,
        };

        let output = run_command(&command, workspace.root(), timeout, &cancel)?;

        // stdout and stderr are merged in arrival order (CRAB-155); no stream
        // labels, matching pi's single accumulator.
        let text = String::from_utf8_lossy(&output.stdout).to_string();

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
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
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
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let err = tool
            .run(&ws, &json!({"command": "echo boo 1>&2; exit 3"}), token())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("boo"));
        assert!(!msg.contains("stderr:"));
        assert!(msg.contains("Command exited with code 3"));
    }

    /// CRAB-155: stdout and stderr share one buffer, so their relative order is
    /// preserved and no `stdout:`/`stderr:` labels are emitted.
    #[tokio::test]
    async fn interleaves_stdout_and_stderr_in_arrival_order() {
        let (ws, _dir) = setup("interleave");
        let tool = BashTool {
            max_output: 10_000,
            default_timeout_secs: None,
        };
        let cmd = "echo o1; sleep 0.05; echo e1 1>&2; sleep 0.05; \
                   echo o2; sleep 0.05; echo e2 1>&2";
        let out = tool
            .run(&ws, &json!({"command": cmd}), token())
            .await
            .unwrap();
        let idx = |needle: &str| {
            out.content
                .find(needle)
                .unwrap_or_else(|| panic!("missing {needle:?} in {:?}", out.content))
        };
        assert!(idx("o1") < idx("e1"), "{}", out.content);
        assert!(idx("e1") < idx("o2"), "{}", out.content);
        assert!(idx("o2") < idx("e2"), "{}", out.content);
        assert!(!out.content.contains("stdout:"), "{}", out.content);
        assert!(!out.content.contains("stderr:"), "{}", out.content);
    }

    #[tokio::test]
    async fn runs_in_workspace_dir() {
        let (ws, dir) = setup("pwd");
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let out = tool
            .run(&ws, &json!({"command": "ls"}), token())
            .await
            .unwrap();
        assert!(out.content.contains("marker.txt"));
    }

    #[tokio::test]
    async fn caps_output_from_the_tail() {
        let (ws, _dir) = setup("cap");
        let tool = BashTool {
            max_output: 64,
            default_timeout_secs: None,
        };
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

    /// The configured default timeout applies when the model omits `timeout`,
    /// so an unbounded command cannot run forever (CRAB-139 review).
    #[tokio::test]
    async fn default_timeout_applies_when_omitted() {
        let (ws, _dir) = setup("default-timeout");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: Some(1),
        };
        let err = tool
            .run(&ws, &json!({"command": "sleep 5"}), token())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }

    /// CRAB-153: an absurd timeout is a bad argument, not an `Instant` overflow
    /// panic inside the blocking task.
    #[tokio::test]
    async fn huge_timeout_is_a_bad_argument() {
        let (ws, _dir) = setup("huge-timeout");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let err = tool
            .run(
                &ws,
                &json!({"command": "true", "timeout": i64::MAX}),
                token(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err}");
        assert!(err.to_string().contains("must be <="), "{err}");
    }

    #[tokio::test]
    async fn timeout_kills_command() {
        let (ws, _dir) = setup("timeout");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let err = tool
            .run(&ws, &json!({"command": "sleep 5", "timeout": 1}), token())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }

    /// CRAB-147: a backgrounded descendant that keeps the pipes open must not
    /// hang the tool when no default timeout applies.
    #[cfg(unix)]
    #[tokio::test]
    async fn background_pipe_holder_does_not_hang() {
        let (ws, _dir) = setup("pipe-holder");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let start = std::time::Instant::now();
        let out = tool
            .run(&ws, &json!({"command": "sleep 5 & echo hi"}), token())
            .await
            .unwrap();
        assert!(out.content.contains("hi"), "{}", out.content);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "returned too slowly: {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn timeout_returns_promptly_when_command_forks() {
        let (ws, _dir) = setup("timeout-fork");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
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
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
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
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
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
