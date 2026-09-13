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
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// Filesystem failure (create dir, open, read, write, rename).
    #[error("session I/O error: {0}")]
    Io(std::io::Error),
    /// A session file could not be parsed and cannot be recovered. A torn
    /// final line is recovered automatically; reaching this means real
    /// corruption (bad line in the middle, missing header, unsupported
    /// version, ...).
    #[error("corrupt session file '{}' at line {line}: {reason}", path.display())]
    Corrupt {
        path: PathBuf,
        /// 1-based line number; 0 when the file has no usable header.
        line: usize,
        reason: String,
    },
    /// Serialization failure while writing (should be impossible for the
    /// fixed entry schema).
    #[error("session serialization error: {0}")]
    Serde(String),
}

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
    // A temp file in the destination directory, fsynced, then renamed over
    // the target (tempfile, CRAB-119). A crash mid-write leaves only the
    // temp file; the destination is untouched.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent).map_err(SessionError::Io)?;
    tmp.write_all(contents.as_bytes())
        .map_err(SessionError::Io)?;
    tmp.as_file().sync_all().map_err(SessionError::Io)?;
    tmp.persist(path).map_err(|e| SessionError::Io(e.error))?;
    Ok(())
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
    crate::paths::data_dir().join("sessions")
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

/// Read the session `id` recorded in a saved file's header line; `None`
/// when the file has no parseable header.
pub fn file_id(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let first = BufReader::new(file).lines().next()?.ok()?;
    let value: serde_json::Value = serde_json::from_str(first.trim()).ok()?;
    value.get("id")?.as_str().map(|s| s.to_string())
}

/// A saved session offered by the `/resume` picker. Sessions store no title of
/// their own (unlike pi), so the label is derived from the first user message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    pub path: PathBuf,
    /// Creation time from the header (epoch millis); the sort key.
    pub created_at: u64,
    /// First user message on one clipped line, or a placeholder when empty.
    pub title: String,
}

/// Longest derived session title, in characters.
const MAX_TITLE_CHARS: usize = 80;

/// Cap the title scan so listing many large sessions stays cheap. The first
/// user message is always near the top, so this is never hit in practice.
const MAX_SUMMARY_LINES: usize = 500;

/// Load a specific saved session file. Used by the `/resume` picker after the
/// user chooses a row (unlike [`load_previous`], which takes the newest).
pub fn load_at(path: &Path) -> Result<Vec<Message>, SessionError> {
    load_file(path)
}

/// Every saved session for `cwd`, newest first, for the `/resume` picker.
/// Unreadable files are skipped rather than failing the whole listing.
pub fn list_sessions(root: &Path, cwd: &Path) -> Result<Vec<SessionSummary>, SessionError> {
    let dir = session_dir(root, cwd);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<SessionSummary> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(SessionError::Io)? {
        let entry = entry.map_err(SessionError::Io)?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if let Some(summary) = summarize(&path) {
            out.push(summary);
        }
    }
    // Newest first; ties broken by path (descending) so the order is stable.
    out.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| b.path.cmp(&a.path))
    });
    Ok(out)
}

/// Delete all but the `keep` most recent sessions for `cwd`, returning how many
/// files were removed. `keep == 0` disables pruning. Recency matches the
/// `/resume` picker (the header's `createdAt`, newest first); a file whose
/// header cannot be read falls back to its modification time so it stays
/// eligible. Non-`.jsonl` files (e.g. leftover atomic-write temps) are left
/// alone.
pub fn prune_sessions(root: &Path, cwd: &Path, keep: usize) -> Result<usize, SessionError> {
    if keep == 0 {
        return Ok(0);
    }
    let dir = session_dir(root, cwd);
    if !dir.is_dir() {
        return Ok(0);
    }
    let mut sessions: Vec<(u64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(SessionError::Io)? {
        let entry = entry.map_err(SessionError::Io)?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let created = header_created_at(&path).or_else(|| {
            entry
                .metadata()
                .ok()?
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
        });
        sessions.push((created.unwrap_or(0), path));
    }
    // Newest first, ties broken by path (descending), matching `list_sessions`.
    sessions.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let mut removed = 0;
    for (_, path) in sessions.into_iter().skip(keep) {
        std::fs::remove_file(&path).map_err(SessionError::Io)?;
        removed += 1;
    }
    Ok(removed)
}

/// The `createdAt` from a session file's first header line, or `None` when it
/// has no parseable header (the caller falls back to the file's mtime).
fn header_created_at(path: &Path) -> Option<u64> {
    let file = File::open(path).ok()?;
    for line in BufReader::new(file).lines() {
        let line = line.ok()?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line).ok()?;
        return value.get("createdAt").and_then(serde_json::Value::as_u64);
    }
    None
}

/// Read just the header (`created_at`) and the first user message from a
/// session file. `None` when the file has no usable header.
fn summarize(path: &Path) -> Option<SessionSummary> {
    let file = File::open(path).ok()?;
    let mut created_at = 0u64;
    let mut have_header = false;
    let mut title = None;
    for line in BufReader::new(file).lines().take(MAX_SUMMARY_LINES) {
        let line = line.ok()?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Entry>(&line) {
            Ok(Entry::Header { created_at: ts, .. }) if !have_header => {
                created_at = ts;
                have_header = true;
            }
            Ok(_) if !have_header => return None,
            Ok(Entry::User { text }) => {
                // Skip user messages that yield no text (e.g. whitespace only)
                // and keep looking: they make an empty picker row.
                let line_title = first_line(&text);
                if !line_title.is_empty() {
                    title = Some(line_title);
                    break;
                }
            }
            Ok(_) => {}
            // A corrupt/unreadable line costs us the title, not the session:
            // keep the summary so a torn tail does not hide a resumable file.
            Err(_) => break,
        }
    }
    if !have_header {
        return None;
    }
    Some(SessionSummary {
        path: path.to_path_buf(),
        created_at,
        title: title.unwrap_or_else(|| "(empty session)".to_string()),
    })
}

/// The first non-empty line of `text`, trimmed and clipped for a picker row.
/// Control characters are stripped (a user message must not be able to inject
/// escape sequences into the terminal) and runs of whitespace collapse to one
/// space, so the label is always a single clean line.
fn first_line(text: &str) -> String {
    let line = text
        .lines()
        .map(|l| {
            l.chars()
                // Keep whitespace (it becomes a separator below); drop other
                // control characters so no escape sequence reaches the screen.
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .find(|l| !l.is_empty())
        .unwrap_or_default();
    let mut out: String = line.chars().take(MAX_TITLE_CHARS).collect();
    if line.chars().count() > MAX_TITLE_CHARS {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    /// A unique temp dir cleaned up on drop (tempfile, CRAB-119).
    fn tempdir(_name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let base = dir.path().to_path_buf();
        (dir, base)
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
    fn lists_sessions_newest_first_with_derived_titles() {
        let (_guard, root) = tempdir("list");
        save_session(&root, &cwd(), &sample_history()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = vec![Message::User("second session title\nand a body".into())];
        let second_path = save_session(&root, &cwd(), &second).unwrap();

        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions.len(), 2);
        // Newest first, and the title is the first line of the first user
        // message (not the whole message).
        assert_eq!(sessions[0].path, second_path);
        assert_eq!(sessions[0].title, "second session title");
        assert_eq!(sessions[1].title, "make a greeting");

        // A different cwd has its own list.
        assert!(list_sessions(&root, &PathBuf::from("/tmp/other-project"))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn session_without_a_user_message_gets_a_placeholder_title() {
        let (_guard, root) = tempdir("empty");
        save_session(&root, &cwd(), &[Message::System("sys".into())]).unwrap();
        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "(empty session)");
    }

    #[test]
    fn blank_user_messages_are_skipped_for_the_title() {
        let (_guard, root) = tempdir("blank");
        let history = vec![
            Message::User("   \n\t ".into()),
            Message::User("real title".into()),
        ];
        save_session(&root, &cwd(), &history).unwrap();
        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions[0].title, "real title");
    }

    #[test]
    fn titles_strip_control_characters_and_collapse_whitespace() {
        let (_guard, root) = tempdir("sanitize");
        let history = vec![Message::User("a\x1b[31mt\tb   c".into())];
        save_session(&root, &cwd(), &history).unwrap();
        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions[0].title, "a[31mt b c");
    }

    #[test]
    fn a_torn_tail_does_not_hide_the_session() {
        let (_guard, root) = tempdir("torn-list");
        // No user message, so the scan runs to the corrupt final line.
        let path = save_session(&root, &cwd(), &[Message::System("sys".into())]).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        std::io::Write::write_all(&mut f, b"{\"kind\":\"user\",\"text\":\"partial").unwrap();
        drop(f);

        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions.len(), 1, "a torn tail must not hide the session");
        assert_eq!(sessions[0].path, path);
        assert_eq!(sessions[0].title, "(empty session)");
    }

    #[test]
    fn load_at_loads_the_chosen_file() {
        let (_guard, root) = tempdir("load-at");
        let first = sample_history();
        let first_path = save_session(&root, &cwd(), &first).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        save_session(&root, &cwd(), &[Message::User("newer".into())]).unwrap();
        // `load_at` ignores recency: it loads exactly the file given.
        assert_eq!(load_at(&first_path).unwrap(), first);
    }

    #[test]
    fn long_titles_are_clipped() {
        let (_guard, root) = tempdir("clip");
        let long = "x".repeat(MAX_TITLE_CHARS + 50);
        save_session(&root, &cwd(), &[Message::User(long)]).unwrap();
        let sessions = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(sessions[0].title.chars().count(), MAX_TITLE_CHARS + 1);
        assert!(sessions[0].title.ends_with('…'));
    }

    #[test]
    fn prune_keeps_only_the_newest_n() {
        let (_guard, root) = tempdir("prune");
        let mut paths = Vec::new();
        for i in 0..4 {
            paths.push(save_session(&root, &cwd(), &[Message::User(format!("s{i}"))]).unwrap());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(prune_sessions(&root, &cwd(), 2).unwrap(), 2);
        let left = list_sessions(&root, &cwd()).unwrap();
        assert_eq!(left.len(), 2);
        // The newest two survive; the older two are gone.
        assert_eq!(left[0].path, paths[3]);
        assert_eq!(left[1].path, paths[2]);
        assert!(!paths[0].exists() && !paths[1].exists());
    }

    #[test]
    fn prune_zero_disables_pruning() {
        let (_guard, root) = tempdir("prune-zero");
        for _ in 0..3 {
            save_session(&root, &cwd(), &sample_history()).unwrap();
        }
        assert_eq!(prune_sessions(&root, &cwd(), 0).unwrap(), 0);
        assert_eq!(list_sessions(&root, &cwd()).unwrap().len(), 3);
    }

    #[test]
    fn prune_is_scoped_to_one_workspace_and_leaves_tmp_files() {
        let (_guard, root) = tempdir("prune-scope");
        for _ in 0..3 {
            save_session(&root, &cwd(), &sample_history()).unwrap();
        }
        let other = PathBuf::from("/tmp/other-project");
        let other_session = save_session(&root, &other, &sample_history()).unwrap();
        // A leftover atomic-write temp is not a session and must survive.
        let tmp = session_dir(&root, &cwd()).join("stale.tmp");
        std::fs::write(&tmp, b"partial").unwrap();

        assert_eq!(prune_sessions(&root, &cwd(), 1).unwrap(), 2);
        assert!(tmp.exists(), "non-jsonl files are not sessions");
        assert!(other_session.exists(), "other workspaces are untouched");
        assert_eq!(list_sessions(&root, &cwd()).unwrap().len(), 1);
        assert_eq!(list_sessions(&root, &other).unwrap().len(), 1);
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
