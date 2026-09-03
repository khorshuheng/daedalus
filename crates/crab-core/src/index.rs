//! Memory indexing & retrieval (CRAB-114).
//!
//! A SQLite FTS5 index over the lessons log (CRAB-113), derived the way pi's
//! `session-backends/sqlite-node` derives a search index from its JSONL:
//! the **JSONL log remains the source of truth** — the index is rebuildable
//! from it at any time, and deleting `index.sqlite` loses nothing.
//!
//! Layout: `~/.local/share/crab/memory/<cwd-encoded>/index.sqlite` sits next
//! to `lessons.jsonl`. The schema keeps one content table (`lessons`) with
//! the full record and an FTS5 virtual table (`lessons_fts`) over
//! text/kind/tags with aligned rowids; a `meta` row stores a signature of
//! the log file so `sync` rebuilds only when the log actually changed.
//!
//! Retrieval ranks lessons by lexical relevance (FTS5 MATCH + `bm25`) and
//! returns top-k within a token budget; `injection_block` formats the
//! winners for the system prompt (CRAB-114's injection half, wired into
//! `Agent::session_system_prompt`).

use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::memory::Lesson;

/// Errors while indexing or searching lessons.
#[derive(Debug)]
pub enum IndexError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Memory(crate::memory::MemoryError),
    /// A lesson row stored in the index could not be deserialized (internal
    /// corruption; rebuild from the JSONL log fixes it).
    Corrupt(String),
}

/// Bring the index up to date with the lesson log for `cwd` under `root`:
/// rebuilds from JSONL when the index is missing or the log changed since
/// the last sync (the log is the source of truth). Returns the number of
/// lessons indexed; 0 when there is no memory yet (and creates nothing).
pub fn sync(root: &Path, cwd: &Path) -> Result<usize, IndexError> {
    let log = crate::memory::lessons_file(root, cwd);
    if !log.is_file() {
        return Ok(0); // no memory yet; nothing to index, nothing to create
    }
    let db = db_path(root, cwd);
    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let signature = log_signature(&log)?;
    let needs_rebuild = !db.is_file() || stored_signature(&db)? != signature;
    if !needs_rebuild {
        // Index is current; report the lesson count without touching it.
        let conn = Connection::open(&db)?;
        return lesson_count(&conn);
    }
    let lessons = crate::memory::list_lessons(root, cwd)?;
    rebuild_from_lessons(&db, &lessons, &signature)?;
    Ok(lessons.len())
}

/// Rank lessons for `cwd` by lexical relevance to `task` and return the top
/// `top_k`. The index is synced first (lazily built when missing), so a
/// caller never has to remember to sync. Returns an empty vec when there is
/// no memory yet or nothing matches.
pub fn search(
    root: &Path,
    cwd: &Path,
    task: &str,
    top_k: usize,
) -> Result<Vec<crate::memory::Lesson>, IndexError> {
    let _n = sync(root, cwd)?;
    let db = db_path(root, cwd);
    if !db.is_file() {
        return Ok(Vec::new());
    }
    let Some(query) = fts_query(task) else {
        return Ok(Vec::new());
    };
    let conn = Connection::open(&db)?;
    let mut stmt = conn.prepare(
        "SELECT l.json FROM lessons_fts f JOIN lessons l ON l.rowid = f.rowid \
         WHERE lessons_fts MATCH ?1 ORDER BY bm25(lessons_fts) LIMIT ?2",
    )?;
    // SQLite treats a negative LIMIT as "no limit"; injection requests an
    // effectively unbounded k and trims to its token budget itself.
    let limit: i64 = top_k.try_into().unwrap_or(-1);
    let rows = stmt.query_map(params![query, limit], |row| row.get::<_, String>(0))?;
    let mut lessons = Vec::new();
    for row in rows {
        let json = row?;
        let lesson: crate::memory::Lesson = serde_json::from_str(&json)
            .map_err(|e| IndexError::Corrupt(format!("invalid lesson row: {e}")))?;
        lessons.push(lesson);
    }
    Ok(lessons)
}

/// Search the top relevant lessons for `task` and format them as a block for
/// the system prompt, e.g. "Project lessons:\n- [rule] ...". Returns ""
/// when there is no memory or nothing matches. Never injects more than
/// `token_budget` tokens (greedy top-k by rank, budget in chars/4).
pub fn injection_block(
    root: &Path,
    cwd: &Path,
    task: &str,
    token_budget: usize,
) -> Result<String, IndexError> {
    if token_budget == 0 {
        return Ok(String::new());
    }
    // Greedy top-k by relevance, stopping once the budget is exhausted.
    let mut block = String::from("Project lessons learned in this workspace:\n");
    let mut used = estimate_tokens(&block);
    // Search without a hard k and trim here, so budget logic lives in one place.
    let mut rank = search(root, cwd, task, usize::MAX)?;
    // search() can only bound results after ranking; cap defensively.
    rank.truncate(50);
    for lesson in rank {
        let line = format!("- [{}] {}\n", lesson.kind, lesson.text);
        let line_tokens = estimate_tokens(&line);
        if used + line_tokens > token_budget {
            break;
        }
        block.push_str(&line);
        used += line_tokens;
    }
    if block.lines().count() <= 1 {
        return Ok(String::new()); // no lesson fit; nothing to inject
    }
    Ok(block)
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Io(e) => write!(f, "index I/O error: {e}"),
            IndexError::Sqlite(e) => write!(f, "index sqlite error: {e}"),
            IndexError::Memory(e) => write!(f, "index memory error: {e}"),
            IndexError::Corrupt(m) => write!(f, "corrupt index row: {m}"),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<std::io::Error> for IndexError {
    fn from(e: std::io::Error) -> Self {
        IndexError::Io(e)
    }
}

impl From<rusqlite::Error> for IndexError {
    fn from(e: rusqlite::Error) -> Self {
        IndexError::Sqlite(e)
    }
}

impl From<crate::memory::MemoryError> for IndexError {
    fn from(e: crate::memory::MemoryError) -> Self {
        IndexError::Memory(e)
    }
}

/// The sqlite file name inside each cwd-encoded directory.
const DB_FILE: &str = "index.sqlite";

/// The index path for `cwd` under `root`: same cwd-encoded directory scheme
/// as the lesson log, holding `index.sqlite`.
fn db_path(root: &Path, cwd: &Path) -> PathBuf {
    root.join(crate::session::encode_cwd(cwd)).join(DB_FILE)
}

/// Rough token estimate: ~4 characters per token (same heuristic as the
/// agent's context budgeting; fine for bounding injection size).
fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    chars.div_ceil(4)
}

/// Common English stopwords too frequent to discriminate lessons (also
/// covers the 3-letter test cases). Short technical words like `git`/`fix`
/// stay searchable via the length filter.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "how", "what", "why", "when", "where", "from", "that", "this",
    "you", "your", "are", "was", "were", "have", "has", "had", "not", "but", "all", "can", "get",
    "our", "its", "one", "two", "out", "about", "into", "them", "they", "will", "would", "there",
    "their", "please", "do", "does", "did", "just", "like", "then", "than", "over", "here", "also",
    "very", "want",
];

/// Turn a free-text task into an FTS5 MATCH expression: lowercase words of
/// at least 3 characters that are not stopwords, joined by OR, so any of
/// them can hit a lesson. Returns `None` when there is nothing searchable
/// (empty task, or only stopword-sized tokens), in which case the caller
/// returns no results.
fn fts_query(task: &str) -> Option<String> {
    let mut words: Vec<String> = Vec::new();
    for word in task.split(|c: char| !c.is_alphanumeric()) {
        let lower = word.to_lowercase();
        if lower.chars().count() >= 3
            && !STOPWORDS.contains(&lower.as_str())
            && !words.contains(&lower)
        {
            words.push(lower);
        }
    }
    if words.is_empty() {
        None
    } else {
        Some(words.join(" OR "))
    }
}

/// A fingerprint of the log file: `len:mtime_nanos`. Append-only JSONL means
/// any change (new lesson, superseding edit, tombstone) grows or rewrites the
/// file, so this reliably detects "index is stale".
fn log_signature(log: &Path) -> Result<String, IndexError> {
    let meta = std::fs::metadata(log)?;
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(format!("{}:{mtime_nanos}", meta.len()))
}

/// The signature stored in the index's `meta` table, or an empty string
/// when the db has not been indexed yet (fresh file, no meta row).
fn stored_signature(db: &Path) -> Result<String, IndexError> {
    let conn = Connection::open(db)?;
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = 'log_sig'")?;
    let mut rows = stmt.query([])?;
    match rows.next()? {
        Some(row) => Ok(row.get(0)?),
        None => Ok(String::new()),
    }
}

/// Number of rows currently in the index.
fn lesson_count(conn: &Connection) -> Result<usize, IndexError> {
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM lessons")?;
    let count: i64 = stmt.query_row([], |r| r.get(0))?;
    Ok(count as usize)
}

/// Drop and recreate the schema, then insert every `lesson` (from the JSONL
/// log — the source of truth) and record `signature` in `meta`.
fn rebuild_from_lessons(db: &Path, lessons: &[Lesson], signature: &str) -> Result<(), IndexError> {
    let conn = Connection::open(db)?;
    conn.execute_batch(
        "DROP TABLE IF EXISTS lessons_fts;
         DROP TABLE IF EXISTS lessons;
         DROP TABLE IF EXISTS meta;
         CREATE TABLE lessons (
             rowid INTEGER PRIMARY KEY AUTOINCREMENT,
             id TEXT NOT NULL,
             json TEXT NOT NULL
         );
         CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    // FTS5 needs its own rowids aligned with `lessons.rowid`; insert into the
    // content table first, then into the index with the same rowid.
    conn.execute_batch("CREATE VIRTUAL TABLE lessons_fts USING fts5(id, text, kind, tags);")?;
    for lesson in lessons {
        let json = serde_json::to_string(lesson).map_err(|e| {
            IndexError::Sqlite(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;
        conn.execute(
            "INSERT INTO lessons (id, json) VALUES (?1, ?2)",
            params![lesson.id, json],
        )?;
        let rowid = conn.last_insert_rowid();
        let text = lesson.text.clone();
        let kind = lesson.kind.clone();
        let tags = lesson.tags.join(" ");
        conn.execute(
            "INSERT INTO lessons_fts (rowid, id, text, kind, tags) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![rowid, lesson.id, text, kind, tags],
        )?;
    }
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('log_sig', ?1)",
        params![signature],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_extracts_significant_words() {
        let q = fts_query("How do I build the project with make?").unwrap();
        assert!(q.contains("build"));
        assert!(q.contains("project"));
        assert!(q.contains("make"));
        assert!(q.contains(" OR "));
        // Stopword-sized tokens and punctuation are dropped.
        assert!(!q.contains("how"));
        assert!(!q.contains("do"));
        assert!(!q.contains("the"));
        assert!(!q.contains("with"));
    }

    #[test]
    fn fts_query_dedupes_and_lowercases() {
        let q = fts_query("Make MAKE Makefile make").unwrap();
        assert_eq!(q, "make OR makefile");
    }

    #[test]
    fn fts_query_is_none_for_empty_or_stopwords() {
        assert!(fts_query("").is_none());
        assert!(fts_query("!!  ").is_none());
        assert!(fts_query("the of it").is_none());
    }

    #[test]
    fn estimate_tokens_matches_chars_over_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn db_path_mirrors_the_lesson_log_layout() {
        let root = Path::new("/tmp/mem");
        let cwd = Path::new("/home/user/proj");
        assert_eq!(
            db_path(root, cwd),
            Path::new("/tmp/mem/--home-user-proj--/index.sqlite")
        );
    }

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_mem(name: &str) -> (TempDir, PathBuf) {
        let base = std::env::temp_dir().join(format!("crab-index-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        (TempDir(base.clone()), base)
    }

    fn cwd() -> PathBuf {
        PathBuf::from("/tmp/crab-proj")
    }

    static SEED_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn seed(root: &Path, cwd: &Path, lessons: &[(&str, &str, &str)]) {
        // (text, kind, tags-comma-joined); ids unique across seed() calls so
        // appending in a later seed() never supersedes an earlier lesson.
        for (text, kind, tags) in lessons {
            let n = SEED_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let lesson = crate::memory::Lesson {
                id: format!("l{n}"),
                text: text.to_string(),
                kind: kind.to_string(),
                tags: tags.split(',').map(|s| s.to_string()).collect(),
                cwd: cwd.to_string_lossy().into_owned(),
                source_session_id: Some("s1".into()),
                created_at: 1000 + n,
                retracted: false,
            };
            crate::memory::append_lesson(root, cwd, &lesson).unwrap();
        }
    }

    #[test]
    fn sync_builds_index_from_the_log() {
        let (_guard, root) = temp_mem("sync-build");
        seed(&root, &cwd(), &[("build with make", "rule", "build")]);
        let n = sync(&root, &cwd()).unwrap();
        assert_eq!(n, 1);
        let db = db_path(&root, &cwd());
        assert!(db.is_file(), "index file created at {}", db.display());
    }

    #[test]
    fn sync_with_no_memory_is_a_no_op() {
        let (_guard, root) = temp_mem("sync-empty");
        assert_eq!(sync(&root, &cwd()).unwrap(), 0);
        assert!(!db_path(&root, &cwd()).exists());
    }

    #[test]
    fn search_ranks_relevant_lessons_first() {
        let (_guard, root) = temp_mem("search-rank");
        seed(
            &root,
            &cwd(),
            &[
                ("always build with make", "rule", "build"),
                ("build the docs with mdbook", "tip", "docs"),
                ("deploy to prod on friday", "warning", "deploy"),
            ],
        );
        sync(&root, &cwd()).unwrap();

        // The task mentions both build and make; the first lesson matches
        // both terms, so it must rank above the one matching only "build".
        let results = search(&root, &cwd(), "how do i build with make", 5).unwrap();
        assert_eq!(
            results.len(),
            2,
            "unrelated deploy lesson is not returned: {results:?}"
        );
        assert_eq!(results[0].text, "always build with make");
        assert_eq!(results[1].text, "build the docs with mdbook");
        // A deploy-only query returns only the deploy lesson.
        let deploy = search(&root, &cwd(), "deploy to production", 5).unwrap();
        assert_eq!(deploy.len(), 1);
        assert!(deploy[0].text.contains("deploy"));
    }

    #[test]
    fn search_auto_builds_when_no_index_yet() {
        // A fresh memory dir with lessons but no explicit sync: search must
        // build the index lazily rather than return nothing.
        let (_guard, root) = temp_mem("search-lazy");
        seed(&root, &cwd(), &[("build with make", "rule", "build")]);
        let results = search(&root, &cwd(), "build", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("make"));
    }

    #[test]
    fn search_respects_top_k() {
        let (_guard, root) = temp_mem("search-k");
        seed(
            &root,
            &cwd(),
            &[
                ("build with make", "rule", "build"),
                ("build with cargo", "rule", "build"),
                ("build script is slow", "tip", "build"),
            ],
        );
        assert_eq!(search(&root, &cwd(), "build", 2).unwrap().len(), 2);
    }

    #[test]
    fn sync_rebuilds_after_the_log_changes() {
        let (_guard, root) = temp_mem("sync-change");
        seed(&root, &cwd(), &[("build with make", "rule", "build")]);
        sync(&root, &cwd()).unwrap();
        // Append another lesson; sync must pick it up (log changed).
        seed(
            &root,
            &cwd(),
            &[("never deploy on friday", "warning", "deploy")],
        );
        let n = sync(&root, &cwd()).unwrap();
        assert_eq!(n, 2);
        let results = search(&root, &cwd(), "deploy", 5).unwrap();
        assert!(results.iter().any(|l| l.text.contains("deploy")));
    }

    #[test]
    fn deleting_the_db_and_rebuilding_yields_identical_results() {
        let (_guard, root) = temp_mem("rebuild");
        seed(
            &root,
            &cwd(),
            &[
                ("build with make", "rule", "build"),
                ("deploy carefully", "tip", "deploy"),
            ],
        );
        sync(&root, &cwd()).unwrap();
        let before = search(&root, &cwd(), "build deploy", 5).unwrap();

        // Simulate a lost index: delete the sqlite file, sync again.
        std::fs::remove_file(db_path(&root, &cwd())).unwrap();
        sync(&root, &cwd()).unwrap();
        let after = search(&root, &cwd(), "build deploy", 5).unwrap();

        assert_eq!(before, after, "rebuild from JSONL must be identical");
    }

    #[test]
    fn injection_block_is_empty_without_memory_or_matches() {
        let (_guard, root) = temp_mem("inj-empty");
        assert_eq!(injection_block(&root, &cwd(), "build", 100).unwrap(), "");
        seed(&root, &cwd(), &[("deploy carefully", "tip", "deploy")]);
        assert_eq!(injection_block(&root, &cwd(), "build", 100).unwrap(), "");
    }

    #[test]
    fn injection_block_lists_relevant_lessons() {
        let (_guard, root) = temp_mem("inj-list");
        seed(
            &root,
            &cwd(),
            &[
                ("always build with make, never cargo", "rule", "build"),
                ("deploy to prod on friday", "warning", "deploy"),
            ],
        );
        let block = injection_block(&root, &cwd(), "how do i build", 200).unwrap();
        assert!(block.contains("make"), "{block}");
        assert!(
            !block.contains("deploy"),
            "unrelated lesson omitted: {block}"
        );
        assert!(block.starts_with("Project lessons"), "{block}");
    }

    #[test]
    fn injection_block_respects_the_token_budget() {
        let (_guard, root) = temp_mem("inj-budget");
        seed(
            &root,
            &cwd(),
            &[
                ("always build with make, never cargo build", "rule", "build"),
                ("build the docs with mdbook", "tip", "build"),
            ],
        );
        // Query matches lesson A on both build+make; A ranks first. Budget is
        // enough for the header + A (~22 tokens) but not A + B (~30), so only
        // the top-ranked lesson may be injected.
        let block = injection_block(&root, &cwd(), "build with make", 25).unwrap();
        assert!(!block.is_empty());
        assert!(
            block.contains("make"),
            "highest-ranked lesson included: {block}"
        );
        assert!(
            !block.contains("mdbook"),
            "budget stopped after top lesson: {block}"
        );
        assert!(estimate_tokens(&block) <= 25, "budget respected: {block}");
    }

    #[test]
    fn injection_block_round_trips_lesson_text() {
        let (_guard, root) = temp_mem("inj-roundtrip");
        seed(&root, &cwd(), &[("build with make", "rule", "build")]);
        let block = injection_block(&root, &cwd(), "build", 200).unwrap();
        assert!(block.contains("build with make"));
    }
}
