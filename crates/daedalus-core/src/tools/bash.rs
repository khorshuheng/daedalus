//! The `bash` tool: run a shell command in the workspace directory.
//!
//! The shell runs in its own process group (Unix) so a timeout or cancel can
//! kill the whole tree — including grandchildren — not just the direct child.
//!
//! stdout/stderr are drained on reader threads. A command that leaves a
//! background process holding the pipes open (`sh -c "sleep 100 &"`) does not
//! block the tool: after the direct child exits the readers get a short grace
//! to drain, then are stopped (mirroring pi's `EXIT_STDIO_GRACE_MS`).

use futures::future::BoxFuture;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use serde_json::{json, Value};

use super::{arg_string, arg_usize, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct BashTool {
    pub max_output: usize,
    /// Applied when the model omits `timeout`, so no command can run forever.
    /// `None` disables the default (tests, or `bash_timeout_secs = 0`).
    pub default_timeout_secs: Option<u64>,
}

/// How long after the direct child exits to let the stdout/stderr readers reach
/// EOF before proceeding without them. A descendant that keeps a
/// pipe write end open must not hold the tool hostage.
const EXIT_STDIO_GRACE: Duration = Duration::from_millis(150);

/// Upper bound for a model-supplied `timeout`, matching pi's `MAX_TIMEOUT_MS`
/// (2_147_483_647 ms). Keeps `Instant + Duration` far from overflow and rejects
/// nonsense values with a clear error instead of a panic in the blocking task.
const MAX_TIMEOUT_SECS: u64 = i32::MAX as u64;

/// How many truncated-output logs to keep in the temp directory. Older ones
/// are pruned when a new log is created, so `/tmp` cannot grow without bound.
const MAX_BASH_LOGS: usize = 20;

/// State shared between `run_command` and its reader threads. stdout and
/// stderr append to one tail in arrival order, matching pi's single
/// `OutputAccumulator`; once the tail is full the whole stream also spills to a
/// temp log, so a chatty command cannot grow daedalus's memory without bound.
struct CaptureInner {
    /// The last `max` bytes seen.
    tail: Vec<u8>,
    /// Every byte seen (for the truncation note).
    total: usize,
    /// Full-output log, created lazily when the tail first overflows.
    spill: Option<std::fs::File>,
    spill_path: Option<PathBuf>,
}

struct StreamCapture {
    inner: Mutex<CaptureInner>,
    max: usize,
    /// Set once the parent has stopped caring; readers then stop appending, so
    /// a chatty descendant cannot grow the buffer after the result is read.
    stop: AtomicBool,
    /// Number of reader threads that have reached EOF.
    done: AtomicUsize,
}

impl StreamCapture {
    fn new(max: usize) -> Self {
        Self {
            inner: Mutex::new(CaptureInner {
                tail: Vec::new(),
                total: 0,
                spill: None,
                spill_path: None,
            }),
            max,
            stop: AtomicBool::new(false),
            done: AtomicUsize::new(0),
        }
    }

    /// Append a chunk, keeping only the tail in memory and spilling the whole
    /// stream to a log once the tail overflows.
    fn append(&self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.total += chunk.len();
        if inner.spill.is_none() && inner.tail.len() + chunk.len() > self.max {
            if let Some((mut file, path)) = create_spill_file() {
                // Seed the log with the bytes already buffered.
                let _ = std::io::Write::write_all(&mut file, &inner.tail);
                inner.spill = Some(file);
                inner.spill_path = Some(path);
            }
        }
        if let Some(file) = inner.spill.as_mut() {
            let _ = std::io::Write::write_all(file, chunk);
        }
        inner.tail.extend_from_slice(chunk);
        if inner.tail.len() > self.max {
            let drop = inner.tail.len() - self.max;
            inner.tail.drain(..drop);
        }
    }

    /// `(tail, total bytes, full-output log path)`.
    fn finish(&self) -> (Vec<u8>, usize, Option<PathBuf>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut file) = inner.spill.take() {
            let _ = std::io::Write::flush(&mut file);
        }
        (
            std::mem::take(&mut inner.tail),
            inner.total,
            inner.spill_path.take(),
        )
    }
}

/// Create a kept temp log and prune older ones so `/tmp` stays bounded.
fn create_spill_file() -> Option<(std::fs::File, PathBuf)> {
    prune_old_spill_logs();
    let file = tempfile::Builder::new()
        .prefix("daedalus-bash-")
        .suffix(".log")
        .tempfile()
        .ok()?;
    file.keep().ok()
}

/// Delete all but the newest [`MAX_BASH_LOGS`] `daedalus-bash-*.log` files in
/// the temp directory. Best-effort: listing/removal failures are ignored.
fn prune_old_spill_logs() {
    let dir = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut logs: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("daedalus-bash-") && name.ends_with(".log")
        })
        .filter_map(|e| {
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, e.path()))
        })
        .collect();
    if logs.len() < MAX_BASH_LOGS {
        return;
    }
    logs.sort_by_key(|(modified, _)| *modified);
    let remove = logs.len() - MAX_BASH_LOGS + 1;
    for (_, path) in logs.into_iter().take(remove) {
        let _ = std::fs::remove_file(path);
    }
}

/// Decode the retained tail and, when bytes were dropped, prefix the note. The
/// tail may start mid-UTF-8, so skip any leading continuation bytes first.
fn tail_string(tail: &[u8], total: usize) -> String {
    let mut start = 0usize;
    while start < tail.len() && (tail[start] & 0xC0) == 0x80 {
        start += 1;
    }
    let text = String::from_utf8_lossy(&tail[start..]).into_owned();
    if total > tail.len() {
        format!(
            "…[truncated: {total} bytes total, showing last {} bytes]\n{text}",
            text.len()
        )
    } else {
        text
    }
}

/// Drain `pipe` into the shared capture until EOF or the `stop` flag is set.
/// The thread exits on EOF; a descendant that keeps the pipe
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
                    capture.append(&chunk[..n]);
                }
                Err(_) => break,
            }
        }
        capture.done.fetch_add(1, Ordering::SeqCst);
    });
}

/// How a command stopped.
enum WaitOutcome {
    Exited(ExitStatus),
    TimedOut,
    Cancelled,
}

/// A finished command: its exit status and the bounded output tail.
struct CommandOutput {
    status: ExitStatus,
    /// Merged stdout/stderr, truncated to the tail when it exceeded the cap.
    text: String,
    /// True when `text` is only the tail of the stream.
    truncated: bool,
    /// The full-output log, when the stream was truncated.
    full_log: Option<PathBuf>,
}

/// Resolve the shell used by the `bash` tool, mirroring pi: `/bin/bash`, then
/// `bash` on `PATH`, then `sh`. Cached because it is pure.
fn resolve_shell() -> PathBuf {
    static SHELL: OnceLock<PathBuf> = OnceLock::new();
    SHELL
        .get_or_init(|| {
            pick_shell(
                Path::new("/bin/bash").is_file(),
                std::env::var_os("PATH").as_deref(),
            )
        })
        .clone()
}

/// Pure shell selection, split out so it can be tested without touching the
/// real filesystem or environment.
fn pick_shell(bin_bash: bool, path: Option<&OsStr>) -> PathBuf {
    if bin_bash {
        return PathBuf::from("/bin/bash");
    }
    if let Some(path) = path {
        for dir in std::env::split_paths(path) {
            let candidate = dir.join("bash");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from("sh")
}

/// Run the resolved shell with `-c <command>` capturing stdout/stderr, with an
/// optional timeout in seconds. Both pipes are drained on background threads so
/// a chatty child cannot deadlock the parent on a full pipe buffer while we
/// wait.
fn run_command(
    command: &str,
    cwd: &Path,
    timeout_secs: Option<u64>,
    cancel: &CancellationToken,
    max_output: usize,
) -> Result<CommandOutput, ToolError> {
    let mut cmd = Command::new(resolve_shell());
    cmd.arg("-c")
        .arg(command)
        .current_dir(cwd)
        // Never inherit daedalus's stdin: in `--mode rpc` it is the command
        // stream and in the TUI it is the keystroke source, so a child that
        // reads stdin (`cat`, `read`, a pager, `ssh`) could swallow them (pi
        // uses `/dev/null` too).
        .stdin(Stdio::null())
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

    let capture = Arc::new(StreamCapture::new(max_output));
    if let Some(stdout) = child.stdout.take() {
        drain(stdout, Arc::clone(&capture));
    }
    if let Some(stderr) = child.stderr.take() {
        drain(stderr, Arc::clone(&capture));
    }

    // `checked_add` so an absurd timeout is a bad argument, not a panic
    // (defense in depth behind the argument/schema clamp).
    let deadline = match timeout_secs {
        Some(s) => Some(
            Instant::now()
                .checked_add(Duration::from_secs(s))
                .ok_or_else(|| ToolError::Argument(format!("'timeout' is too large: {s}")))?,
        ),
        None => None,
    };

    // Wait for exit without a busy-poll on `try_wait`: a waiter thread owns the
    // child and reports its status through a channel, while this thread blocks
    // on the channel (with the deadline / cancellation as its wakeups).
    let status = {
        let pid = child.id();
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<ExitStatus>>();
        let waiter = std::thread::spawn(move || {
            let status = child.wait();
            let _ = tx.send(status);
            // `child` is dropped here, after it has been reaped.
        });
        match wait_loop(&rx, deadline, cancel) {
            WaitOutcome::Exited(status) => {
                let _ = waiter.join();
                status
            }
            outcome @ (WaitOutcome::TimedOut | WaitOutcome::Cancelled) => {
                kill_group(pid);
                // On Unix the group kill makes the waiter's `wait()` return, so
                // it can be joined. Other platforms have no group kill; do not
                // block the caller on a child that may still be running.
                #[cfg(unix)]
                let _ = waiter.join();
                #[cfg(not(unix))]
                drop(waiter);
                capture.stop.store(true, Ordering::SeqCst);
                let (tail, total, spill) = capture.finish();
                let text = tail_string(&tail, total);
                let mut detail = text;
                if let Some(path) = spill {
                    detail.push_str(&format!("\n\n[full output: {}]", path.display()));
                }
                return Err(match outcome {
                    WaitOutcome::TimedOut => ToolError::Timeout(format!(
                        "after {} seconds\n{detail}",
                        timeout_secs.unwrap_or(0)
                    )),
                    _ => ToolError::Cancelled,
                });
            }
        }
    };

    // Let the readers drain what the child already wrote, then stop them, so a
    // descendant holding the pipe cannot block the join or grow the buffer.
    let drain_deadline = Instant::now() + EXIT_STDIO_GRACE;
    while capture.done.load(Ordering::SeqCst) < 2 && Instant::now() < drain_deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    capture.stop.store(true, Ordering::SeqCst);
    let (tail, total, full_log) = capture.finish();
    let truncated = total > tail.len();
    Ok(CommandOutput {
        status,
        text: tail_string(&tail, total),
        truncated,
        full_log,
    })
}

/// Block on the waiter channel until the child exits, the deadline elapses, or
/// cancellation fires. It wakes at least every 50ms to check the cancel
/// token rather than blocking for the whole remaining deadline, so an abort
/// is noticed promptly even when a long timeout is set.
fn wait_loop(
    rx: &std::sync::mpsc::Receiver<std::io::Result<ExitStatus>>,
    deadline: Option<Instant>,
    cancel: &CancellationToken,
) -> WaitOutcome {
    use std::sync::mpsc::RecvTimeoutError;
    const POLL: Duration = Duration::from_millis(50);
    loop {
        if cancel.is_cancelled() {
            return WaitOutcome::Cancelled;
        }
        let timeout = match deadline {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    return WaitOutcome::TimedOut;
                }
                (deadline - now).min(POLL)
            }
            None => POLL,
        };
        match rx.recv_timeout(timeout) {
            Ok(Ok(status)) => return WaitOutcome::Exited(status),
            Ok(Err(_)) | Err(RecvTimeoutError::Disconnected) => return WaitOutcome::Cancelled,
            Err(RecvTimeoutError::Timeout) => {
                // A poll-interval wakeup, not necessarily the real deadline
                // (the wait is capped to `POLL` so cancellation gets
                // checked promptly); the top of the loop re-checks both the
                // cancel token and `now >= deadline` before waiting again.
            }
        }
    }
}

/// Kill the process group rooted at `pid`. The child was started with
/// `setsid`, so its process-group id equals its pid and `kill(-pid, SIGKILL)`
/// reaches every descendant.
#[cfg(unix)]
fn kill_group(pid: u32) {
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
    }
}

/// On non-Unix there is no process group; killing the direct child is the best
/// available. The waiter reaps it, so this only signals.
#[cfg(not(unix))]
fn kill_group(_pid: u32) {}

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
    /// The synchronous body, executed on the blocking pool.
    fn run_sync(
        &self,
        workspace: &Workspace,
        args: &Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        let command = arg_string(args, "command")?;

        // Steer the shell search commands back to the bounded tools before the
        // shell runs anything: the system prompt already bans them, but only
        // this makes the rule stick (a `grep -r` of a 163 G home tree wedged a
        // turn). Head-of-statement only, and fails open — it is guidance, not a
        // sandbox; see [`super::guard`].
        if let Some(refusal) = super::guard::refuse_search(&command) {
            return Err(ToolError::Denied(refusal));
        }

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

        let output = run_command(
            &command,
            workspace.root(),
            timeout,
            &cancel,
            self.max_output,
        )?;

        // stdout and stderr are merged in arrival order; no stream
        // labels, matching pi's single accumulator. `output.text` is already
        // the bounded tail.
        let mut result = output.text;
        if output.truncated {
            if let Some(path) = &output.full_log {
                result.push_str(&format!("\n\n[full output: {}]", path.display()));
            }
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

    /// Block on a tool future (tests are sync).
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

    /// Shell resolution prefers /bin/bash, then bash on PATH, then sh.
    #[test]
    fn shell_resolution_order() {
        use std::ffi::OsStr;
        assert_eq!(
            super::pick_shell(true, None),
            std::path::PathBuf::from("/bin/bash")
        );
        assert_eq!(
            super::pick_shell(false, None),
            std::path::PathBuf::from("sh")
        );
        assert_eq!(
            super::pick_shell(false, Some(OsStr::new("/nonexistent-dir"))),
            std::path::PathBuf::from("sh")
        );
        // A `bash` on PATH is chosen when /bin/bash is absent.
        let dir = tempfile::tempdir().expect("temp dir");
        let bash = dir.path().join("bash");
        std::fs::write(&bash, "").unwrap();
        assert_eq!(super::pick_shell(false, Some(dir.path().as_os_str())), bash);
    }

    /// The tool now runs bash, so a bash-only construct works.
    #[cfg(unix)]
    #[tokio::test]
    async fn runs_bash_only_syntax() {
        let (ws, _dir) = setup("bash-only");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let out = tool
            .run(&ws, &json!({"command": "[[ 1 == 1 ]] && echo ok"}), token())
            .await
            .unwrap();
        assert!(out.content.contains("ok"), "{}", out.content);
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

    /// A child that reads stdin gets EOF, not daedalus's command stream or
    /// keystrokes, and exits promptly.
    #[tokio::test]
    async fn child_stdin_is_closed_not_inherited() {
        let (ws, _dir) = setup("stdin-null");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: Some(2),
        };
        let out = tool
            .run(&ws, &json!({"command": "cat"}), token())
            .await
            .unwrap();
        assert_eq!(out.content, "(no output)", "{}", out.content);
    }

    /// A shell search command is refused up front, before the shell spawns,
    /// and the error names the bounded replacement tool.
    #[tokio::test]
    async fn refuses_shell_search_commands_before_running_them() {
        let (ws, _dir) = setup("guard");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        // Marker proves the shell never ran: `grep` here would create the file
        // only if the guard let the command through.
        let dir = ws.root().to_path_buf();
        let err = tool
            .run(
                &ws,
                &json!({"command": "grep -ril needle . ; touch ran.txt"}),
                token(),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::Denied(msg) => {
                assert!(msg.contains("`grep` tool"), "{msg}");
                assert!(msg.contains("paths"), "{msg}");
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert!(!dir.join("ran.txt").exists(), "the command must not run");

        let err = tool
            .run(&ws, &json!({"command": "find . -name x"}), token())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Denied(_)), "{err:?}");
    }

    /// Filtering an explicit list is bounded work, so a search tool as a pipe
    /// stage still runs.
    #[tokio::test]
    async fn allows_a_search_tool_as_a_pipe_stage() {
        let (ws, _dir) = setup("guard-pipe");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let out = tool
            .run(
                &ws,
                &json!({"command": "printf 'a\\nb\\n' | grep b"}),
                token(),
            )
            .await
            .unwrap();
        assert!(out.content.contains('b'), "{}", out.content);
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

    /// Stdout and stderr share one buffer, so their relative order is
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

    /// A large output keeps only a bounded tail in memory but spills the full
    /// stream to the log it references (DAE-115).
    #[tokio::test]
    async fn large_output_spills_the_full_stream_to_the_log() {
        let (ws, _dir) = setup("spill");
        let tool = BashTool {
            max_output: 64,
            default_timeout_secs: Some(5),
        };
        let out = tool
            .run(
                &ws,
                &json!({"command": "echo HEAD_MARKER; printf '%.0s1' {1..5000}; echo; echo TAIL_MARKER"}),
                token(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("TAIL_MARKER"), "{}", out.content);
        assert!(
            !out.content.contains("HEAD_MARKER"),
            "only the tail is returned to the model: {}",
            out.content
        );
        let path = out
            .content
            .split("[full output: ")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .expect("the full-output path is included");
        let full = std::fs::read_to_string(path).unwrap();
        assert!(full.contains("HEAD_MARKER"), "the head is in the log");
        assert!(full.contains("TAIL_MARKER"), "the tail is in the log");
        let _ = std::fs::remove_file(path);
    }

    /// The configured default timeout applies when the model omits `timeout`,
    /// so an unbounded command cannot run forever.
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

    /// An absurd timeout is a bad argument, not an `Instant` overflow
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

    /// A backgrounded descendant that keeps the pipes open must not
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

    /// Cancellation must be noticed quickly even when a long timeout is also
    /// set: the deadline used to make `wait_loop` block for the whole
    /// remaining timeout before it re-checked the cancel token.
    #[test]
    fn cancel_is_prompt_even_with_a_long_timeout() {
        let (ws, _dir) = setup("cancel-with-timeout");
        let tool = BashTool {
            max_output: 1000,
            default_timeout_secs: None,
        };
        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        let start = std::time::Instant::now();
        let handle = std::thread::spawn(move || {
            block_on(tool.run(&ws, &json!({"command": "sleep 30", "timeout": 60}), cancel2))
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        cancel.cancel();
        let result = handle.join().unwrap();
        assert!(matches!(result, Err(ToolError::Cancelled)), "{result:?}");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "cancel took too long: {:?}",
            start.elapsed()
        );
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
