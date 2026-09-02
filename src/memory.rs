//! Lesson memory store (CRAB-113).
//!
//! Persists the steering lessons extracted from past sessions (CRAB-112) as an
//! **append-only JSONL log** — the authoritative source of truth for agent
//! memory. Layout mirrors the session store (CRAB-109): one log per project
//! under `~/.local/share/crab/memory/<cwd-encoded>/lessons.jsonl`.
//!
//! One JSON object per line; each line is a lesson record
//! `{ id, text, kind, tags, cwd, sourceSessionId, createdAt }`.
//!
//! The log is append-only and doubles as the audit trail:
//! - **New lesson**: append a record with a fresh `id`.
//! - **Edit**: append a *superseding* record with the same `id` and updated
//!   fields. `list_lessons` shows only the last record per id.
//! - **Retraction**: append a superseding record with the same `id` and
//!   `retracted: true` (a tombstone). `list_lessons` drops the id entirely.
//!
//! Ordinary records omit the `retracted` key, so a `cat`/`grep` over the log
//! shows exactly the spec's lesson schema; tombstones carry the extra flag.
//!
//! Writes are appended with an fsync; a torn final line (partial append after
//! a crash) is dropped and the file repaired in place on the next read —
//! CRAB-109's torn-tail recovery. Garbage or schema errors anywhere else are
//! real corruption and surface as `MemoryError::Corrupt`. No SQLite/header
//! here: the log is the source of truth; CRAB-114 derives an index from it.
//!
//! API (per spec): `append_lesson(lesson)`, `list_lessons(cwd)`.

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::session::encode_cwd;

/// A lesson record as it appears on one line of the log. `retracted` is the
/// tombstone flag: always `false` on ordinary records (and omitted from the
/// serialized line), `true` on a retraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    /// Stable identity across superseding records. The extractor (CRAB-112)
    /// assigns ids; an edit/retraction reuses the id of the record it
    /// supersedes.
    pub id: String,
    /// The steering lesson text (e.g. "build with `make`, never `cargo build`").
    pub text: String,
    /// Lesson classification (CRAB-112 emits rule/tip/warning). Stored as a
    /// free string so unknown future kinds never break the log.
    pub kind: String,
    pub tags: Vec<String>,
    /// Working directory the lesson applies to (matches the log's own key).
    pub cwd: String,
    #[serde(
        rename = "sourceSessionId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source_session_id: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub retracted: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// An error while persisting or reading lesson memory.
#[derive(Debug)]
pub enum MemoryError {
    /// Filesystem failure (create dir, open, read, write, rename).
    Io(std::io::Error),
    /// A memory file could not be parsed and cannot be recovered. A torn
    /// final line is recovered automatically; reaching this means real
    /// corruption (garbage/schema error anywhere in the file).
    Corrupt {
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        reason: String,
    },
    /// Serialization failure while writing (should be impossible for the
    /// fixed lesson schema).
    Serde(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Io(e) => write!(f, "memory I/O error: {e}"),
            MemoryError::Corrupt { path, line, reason } => {
                write!(
                    f,
                    "corrupt memory file '{}' at line {line}: {reason}",
                    path.display()
                )
            }
            MemoryError::Serde(e) => write!(f, "memory serialization error: {e}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// The default memory root: `~/.local/share/crab/memory`.
pub fn default_root() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".local/share/crab/memory")
}

/// The log file name inside each cwd-encoded directory.
const LOG_FILE: &str = "lessons.jsonl";

/// Write `contents` to `path` atomically: populate a sibling `.tmp` file,
/// fsync it, then rename it over the destination. A crash mid-write leaves
/// only the ignored `.tmp` file; the destination is untouched.
fn write_file_atomic(path: &Path, contents: &str) -> Result<(), MemoryError> {
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f = File::create(&tmp).map_err(MemoryError::Io)?;
        f.write_all(contents.as_bytes()).map_err(MemoryError::Io)?;
        f.sync_all().map_err(MemoryError::Io)?;
    }
    std::fs::rename(&tmp, path).map_err(MemoryError::Io)
}

/// The memory log path for `cwd` under `root`: the same cwd-encoded
/// directory scheme as sessions (CRAB-109), holding one `lessons.jsonl`.
fn lessons_file(root: &Path, cwd: &Path) -> PathBuf {
    root.join(encode_cwd(cwd)).join(LOG_FILE)
}

/// Fold raw log records into the current lessons: the last record per id
/// wins (append-only edits/retractions supersede earlier ones) and retracted
/// ids disappear. Output keeps the file order of each id's winning record.
fn fold_lessons(records: Vec<Lesson>) -> Vec<Lesson> {
    let mut last: HashMap<String, usize> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        last.insert(r.id.clone(), i);
    }
    records
        .into_iter()
        .enumerate()
        .filter(|(i, r)| last.get(&r.id) == Some(i) && !r.retracted)
        .map(|(_, r)| r)
        .collect()
}

/// Parse a memory log into raw lesson records. Lines are one JSON object
/// each. A *final* line that is truncated (EOF-while-parsing, i.e. a partial
/// append left by a crash) is treated as a torn tail: it is dropped and the
/// file is repaired atomically to the valid prefix. Garbage and schema
/// errors are corruption — a crash can only truncate a line, never complete
/// an invalid one — and always fail loudly.
fn load_file(path: &Path) -> Result<Vec<Lesson>, MemoryError> {
    let file = File::open(path).map_err(MemoryError::Io)?;
    let mut lines: Vec<(usize, String)> = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(MemoryError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        lines.push((index + 1, line));
    }

    let corrupt = |line: usize, reason: String| MemoryError::Corrupt {
        path: path.to_path_buf(),
        line,
        reason,
    };

    let mut records = Vec::new();
    let mut torn_tail = false;
    for (i, (line, raw)) in lines.iter().enumerate() {
        let is_last = i == lines.len() - 1;
        match serde_json::from_str::<Lesson>(raw) {
            Ok(lesson) => records.push(lesson),
            // Only a truncated line (crash mid-append) is a torn tail:
            // serde_json classifies EOF-while-parsing as `Category::Eof`.
            // Garbage (`Syntax`) and schema errors (`Data`) are corruption.
            Err(e) if is_last && e.classify() == serde_json::error::Category::Eof => {
                torn_tail = true
            }
            Err(e) => return Err(corrupt(*line, format!("invalid lesson: {e}"))),
        }
    }

    if torn_tail {
        // Drop the unacknowledged partial append and repair the file to the
        // valid prefix (CRAB-109's torn-tail recovery).
        let valid_prefix: String = lines[..lines.len() - 1]
            .iter()
            .map(|(_, raw)| format!("{raw}\n"))
            .collect();
        write_file_atomic(path, &valid_prefix)?;
    }
    Ok(records)
}

/// Append `lesson` to the memory log for `cwd`, creating the log and its
/// directory on first use. A new lesson uses a fresh id; an edit or
/// retraction appends a superseding record with the same id (see module
/// docs). The line is written with an fsync so a completed append survives
/// a crash; a partial append is a torn tail that reads recover.
pub fn append_lesson(root: &Path, cwd: &Path, lesson: &Lesson) -> Result<(), MemoryError> {
    let path = lessons_file(root, cwd);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(MemoryError::Io)?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(MemoryError::Io)?;
    // One line in a single write: with O_APPEND each append is a single
    // atomic syscall, so a concurrent writer cannot interleave mid-record.
    let mut line = serde_json::to_string(lesson).map_err(|e| MemoryError::Serde(e.to_string()))?;
    line.push('\n');
    f.write_all(line.as_bytes()).map_err(MemoryError::Io)?;
    f.sync_all().map_err(MemoryError::Io)?;
    Ok(())
}

/// Load the current lessons for `cwd` from the log, folded to the latest
/// record per id with retracted lessons removed. Returns an empty list when
/// no memory exists yet. A torn final line is dropped and repaired.
pub fn list_lessons(root: &Path, cwd: &Path) -> Result<Vec<Lesson>, MemoryError> {
    let path = lessons_file(root, cwd);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let records = load_file(&path)?;
    Ok(fold_lessons(records))
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
        let base = std::env::temp_dir().join(format!("crab-memory-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        (TempDir(base.clone()), base)
    }

    fn cwd() -> PathBuf {
        PathBuf::from("/tmp/crab-proj")
    }

    fn lesson(id: &str, text: &str) -> Lesson {
        Lesson {
            id: id.into(),
            text: text.into(),
            kind: "rule".into(),
            tags: vec!["project:crab".into()],
            cwd: "/tmp/crab-proj".into(),
            source_session_id: Some("sess-1".into()),
            created_at: 1000,
            retracted: false,
        }
    }

    #[test]
    fn round_trips_lessons_exactly() {
        let (_guard, root) = tempdir("roundtrip");
        let a = lesson("l1", "always run make first");
        let b = lesson("l2", "never commit to main directly");
        append_lesson(&root, &cwd(), &a).unwrap();
        append_lesson(&root, &cwd(), &b).unwrap();
        assert_eq!(list_lessons(&root, &cwd()).unwrap(), vec![a, b]);
    }

    #[test]
    fn no_memory_returns_empty_list() {
        let (_guard, root) = tempdir("empty");
        assert_eq!(list_lessons(&root, &cwd()).unwrap(), Vec::<Lesson>::new());
    }

    #[test]
    fn empty_log_file_returns_empty_list() {
        let (_guard, root) = tempdir("emptyfile");
        let dir = root.join(encode_cwd(&cwd()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lessons.jsonl"), "").unwrap();
        assert_eq!(list_lessons(&root, &cwd()).unwrap(), Vec::<Lesson>::new());
    }

    #[test]
    fn lessons_are_keyed_by_working_directory() {
        let (_guard, root) = tempdir("keyed");
        append_lesson(&root, &cwd(), &lesson("l1", "some rule")).unwrap();
        let other = PathBuf::from("/tmp/other-project");
        assert_eq!(list_lessons(&root, &other).unwrap(), Vec::<Lesson>::new());
        assert_eq!(list_lessons(&root, &cwd()).unwrap().len(), 1);
    }

    #[test]
    fn file_layout_mirrors_session_store() {
        let (_guard, root) = tempdir("layout");
        append_lesson(&root, &cwd(), &lesson("l1", "rule text")).unwrap();
        let path = root.join(encode_cwd(&cwd())).join("lessons.jsonl");
        assert!(path.is_file(), "expected log at {}", path.display());
    }

    #[test]
    fn record_schema_matches_spec() {
        let (_guard, root) = tempdir("schema");
        let no_source = Lesson {
            source_session_id: None,
            ..lesson("l1", "no provenance")
        };
        append_lesson(&root, &cwd(), &no_source).unwrap();
        append_lesson(&root, &cwd(), &lesson("l2", "with provenance")).unwrap();

        let raw =
            std::fs::read_to_string(root.join(encode_cwd(&cwd())).join("lessons.jsonl")).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON object per line");

        // Optional keys are omitted when absent.
        let v1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v1["id"], "l1");
        assert_eq!(v1["text"], "no provenance");
        assert_eq!(v1["kind"], "rule");
        assert_eq!(v1["tags"], serde_json::json!(["project:crab"]));
        assert_eq!(v1["cwd"], "/tmp/crab-proj");
        assert_eq!(v1["createdAt"], 1000);
        assert!(v1.get("sourceSessionId").is_none());
        assert!(v1.get("retracted").is_none());

        let v2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v2["sourceSessionId"], "sess-1");
    }

    #[test]
    fn torn_final_line_is_dropped_and_repaired() {
        let (_guard, root) = tempdir("torn");
        append_lesson(&root, &cwd(), &lesson("l1", "keep me")).unwrap();

        // Simulate a crash mid-append: a partial JSON line with no newline.
        let path = root.join(encode_cwd(&cwd())).join("lessons.jsonl");
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{\"id\":\"l2\",\"text\":\"partial").unwrap();
        drop(f);

        let loaded = list_lessons(&root, &cwd()).unwrap();
        assert_eq!(loaded, vec![lesson("l1", "keep me")]);

        // The file was repaired: the torn line is gone and it loads cleanly.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("partial"));
        assert_eq!(
            list_lessons(&root, &cwd()).unwrap(),
            vec![lesson("l1", "keep me")]
        );
    }

    #[test]
    fn complete_but_invalid_final_line_is_corruption_not_torn_tail() {
        let (_guard, root) = tempdir("schemaerr");
        append_lesson(&root, &cwd(), &lesson("l1", "ok")).unwrap();

        // A complete JSON object missing required fields is schema corruption,
        // not a torn append: it must fail loudly, never be silently dropped.
        let path = root.join(encode_cwd(&cwd())).join("lessons.jsonl");
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{\"id\":\"l2\",\"text\":\"x\"}}").unwrap();
        drop(f);

        let err = list_lessons(&root, &cwd()).unwrap_err();
        assert!(matches!(err, MemoryError::Corrupt { .. }));
        assert!(err.to_string().contains("missing field"));
    }

    #[test]
    fn corrupt_middle_line_fails() {
        let (_guard, root) = tempdir("corrupt");
        append_lesson(&root, &cwd(), &lesson("l1", "first")).unwrap();
        append_lesson(&root, &cwd(), &lesson("l2", "second")).unwrap();

        // Replace the second line with garbage.
        let path = root.join(encode_cwd(&cwd())).join("lessons.jsonl");
        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        let repaired = format!("{}\nthis is not json\n", lines[0]);
        std::fs::write(&path, repaired).unwrap();

        let err = list_lessons(&root, &cwd()).unwrap_err();
        assert!(matches!(err, MemoryError::Corrupt { .. }));
        assert!(err.to_string().contains("line 2"));
    }

    #[test]
    fn superseding_edit_keeps_log_but_shows_latest() {
        let (_guard, root) = tempdir("edit");
        append_lesson(&root, &cwd(), &lesson("l1", "old text")).unwrap();
        let mut updated = lesson("l1", "new text");
        updated.created_at = 2000;
        append_lesson(&root, &cwd(), &updated).unwrap();

        // list_lessons folds to the latest record per id ...
        let listed = list_lessons(&root, &cwd()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].text, "new text");

        // ... while the log stays append-only with both records.
        let raw =
            std::fs::read_to_string(root.join(encode_cwd(&cwd())).join("lessons.jsonl")).unwrap();
        assert_eq!(raw.lines().count(), 2);
        assert!(raw.contains("old text") && raw.contains("new text"));
    }

    #[test]
    fn retraction_appends_tombstone_and_removes_lesson() {
        let (_guard, root) = tempdir("retract");
        append_lesson(&root, &cwd(), &lesson("l1", "obsolete rule")).unwrap();
        append_lesson(&root, &cwd(), &lesson("l2", "still valid")).unwrap();

        // Retract l1 with a superseding tombstone record.
        let tombstone = Lesson {
            retracted: true,
            ..lesson("l1", "")
        };
        append_lesson(&root, &cwd(), &tombstone).unwrap();

        let listed = list_lessons(&root, &cwd()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "l2");

        // The log keeps the tombstone line for the audit trail.
        let raw =
            std::fs::read_to_string(root.join(encode_cwd(&cwd())).join("lessons.jsonl")).unwrap();
        assert_eq!(raw.lines().count(), 3);
        assert!(raw.contains("\"retracted\":true"));
    }

    #[test]
    fn memory_is_greppable() {
        let (_guard, root) = tempdir("grep");
        append_lesson(&root, &cwd(), &lesson("l1", "build with make")).unwrap();
        let raw =
            std::fs::read_to_string(root.join(encode_cwd(&cwd())).join("lessons.jsonl")).unwrap();
        assert!(raw.contains("build with make"));
        assert!(raw.contains("project:crab"));
    }
}
