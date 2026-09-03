//! Session persistence (CRAB-109).
//!
//! The interactive conversation is persisted to disk so `/resume` can reload
//! the previous session for the current working directory. Storage follows
//! pi / Claude Code: each session is an **append-only JSONL log** — a header
//! line followed by one line per message — keyed by working directory under
//! `~/.local/share/crab/sessions/<cwd-encoded>/<id>.jsonl`.
//!
//! `save_session` writes a brand-new session file **atomically** (temp file +
//! rename, fsynced before the rename commits), so a crash mid-save can never
//! corrupt an existing session. `load_previous` picks the most recent file for
//! the cwd; a torn final line (a partial append) is dropped and the file is
//! repaired in place instead of failing (pi's torn-tail recovery). Any other
//! parse failure is real corruption and surfaces as `SessionError::Corrupt`.
//!
//! SQLite / full-text search is a non-goal (CRAB-111 defers it); the JSONL log
//! is the source of truth.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::provider::{Message, ToolCall};

/// The on-disk format version stored in every header line. Bump when the
/// entry schema changes; loaders refuse files with a different version.
pub const FORMAT_VERSION: u32 = 1;

/// An error while persisting or loading a session.
#[derive(Debug)]
pub enum SessionError {
    /// Filesystem failure (create dir, open, read, write, rename).
    Io(std::io::Error),
    /// A session file could not be parsed and cannot be recovered. A torn
    /// final line is recovered automatically; reaching this means real
    /// corruption (bad line in the middle, missing header, unsupported
    /// version, ...).
    Corrupt {
        path: PathBuf,
        /// 1-based line number; 0 when the file has no usable header.
        line: usize,
        reason: String,
    },
    /// Serialization failure while writing (should be impossible for the
    /// fixed entry schema).
    Serde(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Io(e) => write!(f, "session I/O error: {e}"),
            SessionError::Corrupt { path, line, reason } => {
                write!(
                    f,
                    "corrupt session file '{}' at line {line}: {reason}",
                    path.display()
                )
            }
            SessionError::Serde(e) => write!(f, "session serialization error: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// One JSON object per line: the header or a single conversation message.
/// The `kind` tag mirrors pi's JSONL session shape
/// (`packages/agent/src/harness/session/jsonl`): a header line
/// `{ kind: "header", version, id, createdAt, cwd, parentSessionId? }`
/// followed by one line per entry (assistant text, tool calls, tool results).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Entry {
    Header {
        version: u32,
        id: String,
        #[serde(rename = "createdAt")]
        created_at: u64,
        cwd: String,
        #[serde(rename = "parentSessionId", skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<String>,
    },
    System {
        text: String,
    },
    User {
        text: String,
    },
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        tool_call_id: String,
        result: String,
    },
}

fn entry_from_message(m: &Message) -> Entry {
    match m {
        Message::System(text) => Entry::System { text: text.clone() },
        Message::User(text) => Entry::User { text: text.clone() },
        Message::Assistant { text, tool_calls } => Entry::Assistant {
            text: text.clone(),
            tool_calls: tool_calls.clone(),
        },
        Message::ToolResult {
            tool_call_id,
            result,
        } => Entry::ToolResult {
            tool_call_id: tool_call_id.clone(),
            result: result.clone(),
        },
    }
}

fn message_from_entry(entry: Entry) -> Message {
    match entry {
        Entry::System { text } => Message::System(text),
        Entry::User { text } => Message::User(text),
        Entry::Assistant { text, tool_calls } => Message::Assistant { text, tool_calls },
        Entry::ToolResult {
            tool_call_id,
            result,
        } => Message::ToolResult {
            tool_call_id,
            result,
        },
        Entry::Header { .. } => unreachable!("the header is handled before message lines"),
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A session id unique within this process: created-at timestamp + pid + a
/// counter. Filename-safe and sortable.
fn new_id(created_at: u64) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{created_at:x}-{}-{n}", std::process::id())
}

/// Encode a working directory into a filesystem-safe directory name, following
/// pi's scheme: leading separators are stripped, separators and `:` become
/// `-`, wrapped in `--…--` (`/home/user/proj` -> `--home-user-proj--`).
///
/// `pub(crate)` so the memory store (CRAB-113) keys its per-project
/// `lessons.jsonl` under the same encoding; CRAB-119 will consolidate the
/// shared storage helpers.
pub(crate) fn encode_cwd(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let inner = s
        .trim_start_matches(['/', '\\'])
        .replace(['/', '\\', ':'], "-");
    format!("--{inner}--")
}

fn session_dir(root: &Path, cwd: &Path) -> PathBuf {
    root.join(encode_cwd(cwd))
}

/// Write `contents` to `path` atomically: populate a sibling `.tmp` file,
/// fsync it, then rename it over the destination. A crash mid-write leaves
/// only the ignored `.tmp` file; the destination is untouched.
fn write_file_atomic(path: &Path, contents: &str) -> Result<(), SessionError> {
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f = File::create(&tmp).map_err(SessionError::Io)?;
        f.write_all(contents.as_bytes()).map_err(SessionError::Io)?;
        f.sync_all().map_err(SessionError::Io)?;
    }
    std::fs::rename(&tmp, path).map_err(SessionError::Io)
}

/// The most recent `.jsonl` session file for `cwd` under `root`, or `None`
/// when there is no session yet. `.tmp` files and subdirectories are ignored.
fn previous_session_file(root: &Path, cwd: &Path) -> Result<Option<PathBuf>, SessionError> {
    let dir = session_dir(root, cwd);
    if !dir.is_dir() {
        return Ok(None);
    }
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).map_err(SessionError::Io)? {
        let entry = entry.map_err(SessionError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let mtime = entry
            .metadata()
            .map_err(SessionError::Io)?
            .modified()
            .map_err(SessionError::Io)?;
        let better = match &best {
            None => true,
            Some((best_mtime, best_path)) => {
                mtime > *best_mtime || (mtime == *best_mtime && path > *best_path)
            }
        };
        if better {
            best = Some((mtime, path));
        }
    }
    Ok(best.map(|(_, path)| path))
}

/// Parse a session file into its messages. The first line must be a `header`
/// entry with a supported version; each following line is one message. A
/// *final* line that is truncated (EOF-while-parsing, i.e. a partial append
/// left by a crash) is treated as a torn tail: it is dropped and the file is
/// repaired atomically to the valid prefix, mirroring pi's torn-tail
/// recovery. Garbage and schema errors are corruption — a crash can only
/// truncate a line, never complete an invalid one — and always fail loudly.
fn load_file(path: &Path) -> Result<Vec<Message>, SessionError> {
    let file = File::open(path).map_err(SessionError::Io)?;
    let mut entries: Vec<(usize, String)> = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(SessionError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        entries.push((index + 1, line));
    }

    if entries.is_empty() {
        return Err(SessionError::Corrupt {
            path: path.to_path_buf(),
            line: 0,
            reason: "missing header line".into(),
        });
    }
    let corrupt = |line: usize, reason: String| SessionError::Corrupt {
        path: path.to_path_buf(),
        line,
        reason,
    };

    let (header_line, header_raw) = &entries[0];
    let header: Entry = serde_json::from_str(header_raw)
        .map_err(|e| corrupt(*header_line, format!("invalid header: {e}")))?;
    let version = match &header {
        Entry::Header { version, .. } => *version,
        _ => {
            return Err(corrupt(
                *header_line,
                "first line is not a session header".into(),
            ))
        }
    };
    if version != FORMAT_VERSION {
        return Err(corrupt(
            *header_line,
            format!("unsupported session version {version} (expected {FORMAT_VERSION})"),
        ));
    }

    let mut messages = Vec::new();
    let mut torn_tail = false;
    for index in 1..entries.len() {
        let (line, raw) = &entries[index];
        let is_last = index == entries.len() - 1;
        match serde_json::from_str::<Entry>(raw) {
            Ok(Entry::Header { .. }) => {
                return Err(corrupt(*line, "unexpected header line".into()))
            }
            Ok(entry) => messages.push(message_from_entry(entry)),
            // Only a truncated line (crash mid-append) is a torn tail:
            // serde_json classifies EOF-while-parsing as `Category::Eof`.
            // Garbage (`Syntax`) and schema errors (`Data`) are corruption
            // and must fail loudly — a crash can only truncate a line, never
            // complete an invalid one.
            Err(e) if is_last && e.classify() == serde_json::error::Category::Eof => {
                torn_tail = true
            }
            Err(e) => return Err(corrupt(*line, format!("invalid entry: {e}"))),
        }
    }

    if torn_tail {
        // Drop the unacknowledged partial append and repair the file to the
        // valid prefix (pi's torn-tail recovery).
        let valid_prefix: String = entries[..entries.len() - 1]
            .iter()
            .map(|(_, raw)| format!("{raw}\n"))
            .collect();
        write_file_atomic(path, &valid_prefix)?;
    }
    Ok(messages)
}

/// The default sessions root: `~/.local/share/crab/sessions`.
pub fn default_root() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".local/share/crab/sessions")
}

/// Persist `history` as a new session for `cwd`, returning the file path.
/// The file is written atomically; a later `load_previous` for the same `cwd`
/// returns this session's messages until a newer one is saved.
pub fn save_session(root: &Path, cwd: &Path, history: &[Message]) -> Result<PathBuf, SessionError> {
    let dir = session_dir(root, cwd);
    std::fs::create_dir_all(&dir).map_err(SessionError::Io)?;

    let created_at = now_millis();
    let id = new_id(created_at);
    let path = dir.join(format!("{created_at}_{id}.jsonl"));
    let header = Entry::Header {
        version: FORMAT_VERSION,
        id,
        created_at,
        cwd: cwd.to_string_lossy().into_owned(),
        parent_session_id: None,
    };

    let mut out = String::new();
    let push = |entry: &Entry, out: &mut String| -> Result<(), SessionError> {
        let line = serde_json::to_string(entry).map_err(|e| SessionError::Serde(e.to_string()))?;
        out.push_str(&line);
        out.push('\n');
        Ok(())
    };
    push(&header, &mut out)?;
    for message in history {
        push(&entry_from_message(message), &mut out)?;
    }
    write_file_atomic(&path, &out)?;
    Ok(path)
}

/// Load the most recent session for `cwd`, or `None` when there is no saved
/// session yet. A torn final line is recovered (dropped + repaired).
pub fn load_previous(root: &Path, cwd: &Path) -> Result<Option<Vec<Message>>, SessionError> {
    match previous_session_file(root, cwd)? {
        None => Ok(None),
        Some(path) => Ok(Some(load_file(&path)?)),
    }
}

/// Delete the most recent saved session for `cwd`. Available for callers
/// that want to forget the previous conversation; `/clear` itself resets only
/// the in-memory context and leaves saved sessions in place. Idempotent:
/// no session -> no-op.
pub fn clear_previous(root: &Path, cwd: &Path) -> Result<(), SessionError> {
    if let Some(path) = previous_session_file(root, cwd)? {
        std::fs::remove_file(&path).map_err(SessionError::Io)?;
    }
    Ok(())
}

/// Read the session `id` recorded in a saved file's header line. Used as
/// lesson provenance when reflecting over a just-saved session (CRAB-112);
/// `None` when the file has no parseable header.
pub fn file_id(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let first = BufReader::new(file).lines().next()?.ok()?;
    let value: serde_json::Value = serde_json::from_str(first.trim()).ok()?;
    value.get("id")?.as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    /// RAII guard that removes its dir on drop.
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir(name: &str) -> (TempDir, PathBuf) {
        let base = std::env::temp_dir().join(format!("crab-session-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        (TempDir(base.clone()), base)
    }

    fn cwd() -> PathBuf {
        PathBuf::from("/tmp/crab-proj")
    }

    fn sample_history() -> Vec<Message> {
        vec![
            Message::System("you are crab in /ws".into()),
            Message::User("make a greeting".into()),
            Message::Assistant {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({"command": "echo hi"}),
                }],
            },
            Message::ToolResult {
                tool_call_id: "c1".into(),
                result: "hi\n".into(),
            },
            Message::Assistant {
                text: Some("done".into()),
                tool_calls: vec![],
            },
        ]
    }

    #[test]
    fn round_trips_a_multi_turn_session() {
        let (_guard, root) = tempdir("roundtrip");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("jsonl"));
        assert_eq!(
            load_previous(&root, &cwd()).unwrap().unwrap(),
            sample_history()
        );
    }

    #[test]
    fn no_previous_session_returns_none() {
        let (_guard, root) = tempdir("none");
        assert!(load_previous(&root, &cwd()).unwrap().is_none());
    }

    #[test]
    fn sessions_are_keyed_by_working_directory() {
        let (_guard, root) = tempdir("keyed");
        save_session(&root, &cwd(), &sample_history()).unwrap();
        let other = PathBuf::from("/tmp/other-project");
        assert!(load_previous(&root, &other).unwrap().is_none());
        assert!(load_previous(&root, &cwd()).unwrap().is_some());
    }

    #[test]
    fn most_recent_session_is_loaded() {
        let (_guard, root) = tempdir("recent");
        save_session(&root, &cwd(), &sample_history()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let mut second = sample_history();
        second.push(Message::User("another turn".into()));
        save_session(&root, &cwd(), &second).unwrap();
        assert_eq!(load_previous(&root, &cwd()).unwrap().unwrap(), second);
    }

    #[test]
    fn torn_final_line_is_dropped_and_repaired() {
        let (_guard, root) = tempdir("torn");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();

        // Simulate a crash mid-append: a partial JSON line with no newline.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{\"kind\":\"assistant\",\"text\":\"partial").unwrap();
        drop(f);

        let loaded = load_previous(&root, &cwd())
            .unwrap()
            .expect("torn tail is recovered");
        assert_eq!(loaded, sample_history());

        // The file was repaired: the torn line is gone and it loads cleanly.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("partial"));
        assert_eq!(
            load_previous(&root, &cwd()).unwrap().unwrap(),
            sample_history()
        );
    }

    #[test]
    fn schema_error_on_last_line_is_corruption_not_torn_tail() {
        let (_guard, root) = tempdir("schema");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();

        // A complete-but-invalid line (unknown kind) is schema corruption,
        // not a torn append: it must fail loudly, never be silently dropped.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{\"kind\":\"bogus\",\"text\":\"x\"}}").unwrap();
        drop(f);

        let err = load_previous(&root, &cwd()).unwrap_err();
        assert!(matches!(err, SessionError::Corrupt { .. }));
        assert!(err.to_string().contains("unknown variant"));
    }

    #[test]
    fn corrupt_middle_line_fails() {
        let (_guard, root) = tempdir("corrupt");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();

        // Replace the second line (a user message) with garbage.
        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        let mut out = lines[0].clone();
        out.push('\n');
        out.push_str("this is not json\n");
        out.push_str(&lines[2..].join("\n"));
        out.push('\n');
        std::fs::write(&path, out).unwrap();

        let err = load_previous(&root, &cwd()).unwrap_err();
        assert!(matches!(err, SessionError::Corrupt { .. }));
        assert!(err.to_string().contains("line 2"));
    }

    #[test]
    fn stray_tmp_files_are_ignored() {
        let (_guard, root) = tempdir("tmp");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();
        // A crashed atomic write left a temp file; it must not be picked as
        // the previous session.
        std::fs::write(path.with_extension("jsonl.tmp"), "garbage").unwrap();
        assert_eq!(
            load_previous(&root, &cwd()).unwrap().unwrap(),
            sample_history()
        );
    }

    #[test]
    fn clear_previous_removes_the_session() {
        let (_guard, root) = tempdir("clear");
        save_session(&root, &cwd(), &sample_history()).unwrap();
        clear_previous(&root, &cwd()).unwrap();
        assert!(load_previous(&root, &cwd()).unwrap().is_none());
        // Idempotent: clearing when nothing is saved is fine.
        clear_previous(&root, &cwd()).unwrap();
    }

    #[test]
    fn file_id_reads_the_saved_header() {
        let (_guard, root) = tempdir("fileid");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();
        let id = file_id(&path).expect("saved session has an id");
        assert!(!id.is_empty());
        assert_eq!(file_id(&path).unwrap(), id, "id is stable across reads");
    }

    #[test]
    fn file_id_is_none_for_unparseable_files() {
        let (_guard, root) = tempdir("fileid-none");
        let dir = session_dir(&root, &cwd());
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("garbage.jsonl");
        std::fs::write(&bad, "this is not json\n").unwrap();
        assert!(file_id(&bad).is_none());
        let missing = dir.join("nope.jsonl");
        assert!(file_id(&missing).is_none());
    }

    #[test]
    fn unsupported_version_fails() {
        let (_guard, root) = tempdir("version");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let first_line_end = raw.find('\n').unwrap();
        let mut header: serde_json::Value = serde_json::from_str(&raw[..first_line_end]).unwrap();
        header["version"] = serde_json::json!(99);
        std::fs::write(&path, format!("{header}\n{}", &raw[first_line_end + 1..])).unwrap();

        let err = load_previous(&root, &cwd()).unwrap_err();
        assert!(err.to_string().contains("version 99"));
    }

    #[test]
    fn header_line_matches_spec_shape() {
        let (_guard, root) = tempdir("header");
        let path = save_session(&root, &cwd(), &sample_history()).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let first = raw.lines().next().unwrap();
        let v: serde_json::Value = serde_json::from_str(first).unwrap();
        assert_eq!(v["kind"], "header");
        assert_eq!(v["version"], FORMAT_VERSION);
        assert_eq!(v["cwd"], "/tmp/crab-proj");
        assert!(v["id"].is_string());
        assert!(v["createdAt"].is_number());
        // parentSessionId is optional and omitted when absent.
        assert!(v.get("parentSessionId").is_none());
    }

    #[test]
    fn cwd_encoding_is_filesystem_safe() {
        assert_eq!(
            encode_cwd(Path::new("/home/user/proj")),
            "--home-user-proj--"
        );
        assert_eq!(encode_cwd(Path::new("/home/user/a b")), "--home-user-a b--");
        assert_eq!(
            encode_cwd(Path::new("C:\\Users\\me\\proj")),
            "--C--Users-me-proj--"
        );
    }
}
