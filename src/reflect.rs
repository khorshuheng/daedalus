//! Lesson extraction / reflection (CRAB-112).
//!
//! Turns a finished session transcript into concise steering lessons by
//! asking the LLM a dedicated summarization question (like pi's
//! `harness/compaction`). The pipeline is:
//!
//! 1. render the transcript (bounded) for the prompt;
//! 2. call the provider for a JSON array of lessons;
//! 3. parse into `DraftLesson`s, dropping low-value/unknown ones;
//! 4. dedupe against existing memory (CRAB-113) and within the batch;
//! 5. enrich drafts with provenance (id, cwd, source session, timestamp)
//!    and append to the memory store.
//!
//! Extraction never writes to memory itself: parsing/deduping happens before
//! any append, so an LLM failure or malformed response surfaces as an error
//! and can never corrupt existing memory.

use crate::provider::Message;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A lesson as emitted by the model, before provenance is attached. `kind`
/// is validated to `rule | tip | warning` at parse time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DraftLesson {
    pub text: String,
    pub kind: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Errors while reflecting on a session.
#[derive(Debug)]
pub enum ReflectError {
    /// Nothing to reflect on (no assistant/user conversation in the history).
    EmptyHistory,
    /// The provider failed (auth, timeout, malformed response, ...).
    Provider(crate::provider::ProviderError),
    /// The model's answer was not parseable as a lesson list.
    Malformed(String),
    /// Reading or writing the memory store failed.
    Memory(crate::memory::MemoryError),
}

impl std::fmt::Display for ReflectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReflectError::EmptyHistory => write!(f, "no conversation to reflect on"),
            ReflectError::Provider(e) => write!(f, "reflection LLM error: {e}"),
            ReflectError::Malformed(m) => write!(f, "malformed lesson response: {m}"),
            ReflectError::Memory(e) => write!(f, "memory error: {e}"),
        }
    }
}

impl std::error::Error for ReflectError {}

impl From<crate::provider::ProviderError> for ReflectError {
    fn from(e: crate::provider::ProviderError) -> Self {
        ReflectError::Provider(e)
    }
}

impl From<crate::memory::MemoryError> for ReflectError {
    fn from(e: crate::memory::MemoryError) -> Self {
        ReflectError::Memory(e)
    }
}

/// Canonical text used for dedupe and content-addressed ids: trim, collapse
/// runs of whitespace to a single space, lowercase.
pub fn normalize_text(s: &str) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.to_lowercase()
}

/// Content-addressed lesson id: a 16-hex-digit FNV-1a 64-bit hash of the
/// *normalized* text. The same lesson learned again (in another session, with
/// different phrasing spacing/case) maps to the same id, which is what makes
/// memory dedupe and last-wins updates stable across reflections.
pub fn lesson_id(normalized_text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in normalized_text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Parse the model's response into lessons. Accepts a bare JSON array of
/// lesson objects, optionally wrapped in ```json fences (models often add
/// them). Each object is `{ text, kind, tags? }`. Empty/whitespace-only
/// text and unknown kinds are dropped as low-value; structural failures
/// (no JSON, not an array) are `ReflectError::Malformed`.
pub fn parse_lessons(text: &str) -> Result<Vec<DraftLesson>, ReflectError> {
    let trimmed = text.trim();
    // Models frequently wrap the array in a ```json fence; strip it.
    let inner = trimmed
        .strip_prefix("```")
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed)
        .trim_start_matches("json")
        .trim();
    let value: serde_json::Value =
        serde_json::from_str(inner).map_err(|e| ReflectError::Malformed(e.to_string()))?;
    let items = match value {
        serde_json::Value::Array(items) => items,
        _ => return Err(ReflectError::Malformed("expected a JSON array".into())),
    };
    let mut lessons = Vec::new();
    for item in items {
        let Ok(lesson) = serde_json::from_value::<DraftLesson>(item) else {
            continue; // malformed element: skip rather than fail the batch
        };
        if lesson.text.trim().is_empty() {
            continue;
        }
        let kind = lesson.kind.trim().to_ascii_lowercase();
        if !matches!(kind.as_str(), "rule" | "tip" | "warning") {
            continue;
        }
        lessons.push(DraftLesson {
            text: lesson.text,
            kind,
            tags: lesson.tags,
        });
    }
    Ok(lessons)
}

/// Render a session history into the transcript shown to the reflection
/// model: each user/assistant text line prefixed by its role, tool calls
/// summarized as `[tool: name]`, and tool results capped to keep the prompt
/// bounded. The system prompt is not part of the transcript.
pub fn render_transcript(history: &[Message]) -> String {
    const RESULT_CAP: usize = 400;
    let mut out = String::new();
    for msg in history {
        match msg {
            Message::System(_) => {}
            Message::User(text) => {
                out.push_str("user: ");
                out.push_str(text);
                out.push('\n');
            }
            Message::Assistant { text, tool_calls } => {
                if let Some(text) = text {
                    out.push_str("assistant: ");
                    out.push_str(text);
                    out.push('\n');
                }
                if !tool_calls.is_empty() {
                    let names: Vec<&str> = tool_calls.iter().map(|c| c.name.as_str()).collect();
                    out.push_str(&format!("[tool: {}]\n", names.join(", ")));
                }
            }
            Message::ToolResult { result, .. } => {
                out.push_str("tool result: ");
                if result.chars().count() > RESULT_CAP {
                    let capped: String = result.chars().take(RESULT_CAP).collect();
                    out.push_str(&capped);
                    out.push('…');
                } else {
                    out.push_str(result);
                }
                out.push('\n');
            }
        }
    }
    out
}

/// Filter `candidates` to only lessons not already in `existing` memory and
/// not duplicated within the batch itself. Comparison uses `normalize_text`,
/// so re-learned lessons with different spacing/case are recognized as the
/// same lesson.
pub fn dedupe(candidates: &[DraftLesson], existing: &[crate::memory::Lesson]) -> Vec<DraftLesson> {
    let existing_norm: std::collections::HashSet<String> =
        existing.iter().map(|l| normalize_text(&l.text)).collect();
    let mut seen = std::collections::HashSet::new();
    candidates
        .iter()
        .filter(|c| {
            let norm = normalize_text(&c.text);
            !existing_norm.contains(&norm) && seen.insert(norm)
        })
        .cloned()
        .collect()
}

/// The messages sent to the reflection model: a system instruction naming the
/// project and asking for a JSON array of lessons, plus the rendered
/// transcript as the user turn.
pub fn reflection_messages(history: &[Message], cwd: &Path) -> Vec<Message> {
    let system = format!(
        "You are reflecting on a coding session that took place in the project '{}'.\n\
         Extract at most 5 concise, reusable steering lessons a future agent session in this\n\
         project should follow (e.g. \"build with `make`, never `cargo build` here\").\n\
         Respond with ONLY a JSON array, no prose or markdown fence. Each element is an object\n\
         with exactly: \"text\" (the lesson, one short sentence), \"kind\" (one of\n\
         \"rule\" | \"tip\" | \"warning\"), and \"tags\" (an array of short strings, e.g.\n\
         [\"build\"], may be empty). Skip trivia and one-off instructions.",
        cwd.display()
    );
    vec![
        Message::System(system),
        Message::User(render_transcript(history)),
    ]
}

/// Ask the provider to reflect on `history` and return the parsed lessons.
/// Empty (system-only) histories are rejected before any LLM call. A provider
/// failure or malformed answer surfaces as `ReflectError`; nothing is written
/// to memory here — that happens only after dedupe, in the caller.
pub fn extract(
    provider: &dyn crate::provider::Provider,
    cancel: &std::sync::atomic::AtomicBool,
    history: &[Message],
    cwd: &Path,
) -> Result<Vec<DraftLesson>, ReflectError> {
    let has_conversation = history
        .iter()
        .any(|m| matches!(m, Message::User(_) | Message::Assistant { .. }));
    if !has_conversation {
        return Err(ReflectError::EmptyHistory);
    }
    let messages = reflection_messages(history, cwd);
    let completion = provider
        .complete(&messages, &[], cancel, &mut |_| {})
        .map_err(ReflectError::Provider)?;
    match completion.response {
        crate::provider::Response::Text(text) => parse_lessons(&text),
        _ => Err(ReflectError::Malformed(
            "expected a text response from the reflection LLM".into(),
        )),
    }
}

/// Orchestrate one reflection: extract fresh lessons from `history` via the
/// provider, dedupe against lessons already stored for `cwd` under `root`,
/// attach provenance (`cwd`, `source_session_id`, `created_at`), and append
/// the survivors to the memory log. Returns the number of lessons added.
///
/// Nothing is written until parsing and dedupe succeed, so a provider
/// failure or malformed answer leaves existing memory untouched.
pub fn reflect_and_store(
    root: &Path,
    cwd: &Path,
    provider: &dyn crate::provider::Provider,
    cancel: &std::sync::atomic::AtomicBool,
    history: &[Message],
    source_session_id: Option<String>,
    created_at: u64,
) -> Result<usize, ReflectError> {
    // Reject before any LLM call or write when there is nothing to reflect on.
    let has_conversation = history
        .iter()
        .any(|m| matches!(m, Message::User(_) | Message::Assistant { .. }));
    if !has_conversation {
        return Err(ReflectError::EmptyHistory);
    }
    // Extract (LLM) first: a failure here must not touch memory.
    let drafts = extract(provider, cancel, history, cwd)?;
    // Dedupe against what is already stored, then append only the survivors.
    let existing = crate::memory::list_lessons(root, cwd).map_err(ReflectError::Memory)?;
    let fresh = dedupe(&drafts, &existing);
    let lessons = lessons_from_drafts(&fresh, cwd, source_session_id, created_at);
    let mut added = 0;
    for lesson in &lessons {
        crate::memory::append_lesson(root, cwd, lesson).map_err(ReflectError::Memory)?;
        added += 1;
    }
    Ok(added)
}

/// Attach provenance to fresh drafts: a content-addressed `id`, the `cwd`,
/// the `source_session_id` of the session reflected on, and a `created_at`
/// timestamp. Kinds and tags are carried over as parsed.
pub fn lessons_from_drafts(
    drafts: &[DraftLesson],
    cwd: &Path,
    source_session_id: Option<String>,
    created_at: u64,
) -> Vec<crate::memory::Lesson> {
    drafts
        .iter()
        .map(|d| crate::memory::Lesson {
            id: lesson_id(&normalize_text(&d.text)),
            text: d.text.clone(),
            kind: d.kind.clone(),
            tags: d.tags.clone(),
            cwd: cwd.to_string_lossy().into_owned(),
            source_session_id: source_session_id.clone(),
            created_at,
            retracted: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_trims_and_collapses_whitespace() {
        assert_eq!(normalize_text("  use   make first  "), "use make first");
        assert_eq!(normalize_text("\tbuild\nwith cargo\n"), "build with cargo");
    }

    #[test]
    fn normalize_text_lowercases() {
        assert_eq!(normalize_text("Always run MAKE"), "always run make");
    }

    #[test]
    fn normalize_text_of_empty_string_is_empty() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("   "), "");
    }

    #[test]
    fn lesson_id_is_deterministic_and_content_addressed() {
        assert_eq!(lesson_id("use make first"), lesson_id("use make first"));
        // Normalization happens before hashing, so the caller must pass the
        // normalized form; this just documents that different text differs.
        assert_ne!(lesson_id("use make first"), lesson_id("use cargo first"));
    }

    #[test]
    fn lesson_id_is_hex_and_fixed_width() {
        let id = lesson_id("always run make");
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn parses_a_lesson_array() {
        let lessons = parse_lessons(
            r#"[{"text":"use make","kind":"rule","tags":["build"]},
                 {"text":"never force push","kind":"warning"}]"#,
        )
        .unwrap();
        assert_eq!(lessons.len(), 2);
        assert_eq!(lessons[0].text, "use make");
        assert_eq!(lessons[0].kind, "rule");
        assert_eq!(lessons[0].tags, vec!["build"]);
        assert_eq!(lessons[1].tags, Vec::<String>::new()); // tags default
    }

    #[test]
    fn parses_a_json_fenced_array() {
        let raw = "```json\n[{\"text\":\"run tests first\",\"kind\":\"tip\"}]\n```";
        let lessons = parse_lessons(raw).unwrap();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, "run tests first");
    }

    #[test]
    fn drops_empty_and_unknown_kind_lessons() {
        let lessons = parse_lessons(
            r#"[{"text":"  ","kind":"rule"},
                 {"text":"valid one","kind":"tip"},
                 {"text":"mystery kind","kind":"poem"}]"#,
        )
        .unwrap();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, "valid one");
    }

    #[test]
    fn kind_case_is_normalized() {
        let lessons = parse_lessons(
            r#"[{"text":"upper kind","kind":"Rule"},
                 {"text":"shouty","kind":"WARNING"}]"#,
        )
        .unwrap();
        assert_eq!(lessons.len(), 2);
        assert_eq!(lessons[0].kind, "rule");
        assert_eq!(lessons[1].kind, "warning");
    }

    #[test]
    fn non_json_response_is_malformed() {
        let err = parse_lessons("sure, here are some tips: 1. use make").unwrap_err();
        assert!(matches!(err, ReflectError::Malformed(_)));
    }

    #[test]
    fn empty_array_parses_to_no_lessons() {
        assert_eq!(parse_lessons("[]").unwrap(), Vec::<DraftLesson>::new());
    }

    fn text_msg(role: &str, text: &str) -> Message {
        match role {
            "user" => Message::User(text.to_string()),
            "system" => Message::System(text.to_string()),
            _ => Message::Assistant {
                text: Some(text.to_string()),
                tool_calls: vec![],
            },
        }
    }

    fn tool_result(result: &str) -> Message {
        Message::ToolResult {
            tool_call_id: "c1".into(),
            result: result.to_string(),
        }
    }

    #[test]
    fn renders_roles_and_text() {
        let history = vec![
            Message::System("you are crab".into()),
            text_msg("user", "fix the build"),
            text_msg("assistant", "I fixed it"),
        ];
        let out = render_transcript(&history);
        assert!(!out.contains("you are crab"), "system excluded: {out}");
        assert!(out.contains("user: fix the build"), "{out}");
        assert!(out.contains("assistant: I fixed it"), "{out}");
    }

    #[test]
    fn renders_tool_calls_as_summary() {
        let history = vec![
            text_msg("user", "check the repo"),
            Message::Assistant {
                text: None,
                tool_calls: vec![
                    crate::provider::ToolCall {
                        id: "a".into(),
                        name: "bash".into(),
                        args: serde_json::json!({"command": "ls"}),
                    },
                    crate::provider::ToolCall {
                        id: "b".into(),
                        name: "read".into(),
                        args: serde_json::json!({"path": "Cargo.toml"}),
                    },
                ],
            },
            tool_result("ok"),
        ];
        let out = render_transcript(&history);
        assert!(out.contains("bash") && out.contains("read"), "{out}");
        assert!(out.contains("tool result: ok"), "{out}");
    }

    #[test]
    fn caps_long_tool_results() {
        let long = "x".repeat(5000);
        let out = render_transcript(&[tool_result(&long)]);
        assert!(out.contains("…"), "must mark truncation");
        assert!(out.len() < 2000, "result capped");
    }

    #[test]
    fn empty_history_renders_empty() {
        assert_eq!(render_transcript(&[]), "");
        assert_eq!(render_transcript(&[Message::System("sys".into())]), "");
    }

    fn lesson_in_memory(text: &str) -> crate::memory::Lesson {
        crate::memory::Lesson {
            id: crate::reflect::lesson_id(&normalize_text(text)),
            text: text.to_string(),
            kind: "rule".into(),
            tags: vec![],
            cwd: "/tmp/proj".into(),
            source_session_id: None,
            created_at: 1,
            retracted: false,
        }
    }

    fn draft(text: &str, kind: &str) -> DraftLesson {
        DraftLesson {
            text: text.to_string(),
            kind: kind.to_string(),
            tags: vec![],
        }
    }

    #[test]
    fn keeps_new_lessons() {
        let candidates = vec![draft("use make", "rule"), draft("commit often", "tip")];
        let fresh = dedupe(&candidates, &[]);
        assert_eq!(fresh, candidates);
    }

    #[test]
    fn drops_lessons_already_in_memory() {
        let existing = vec![lesson_in_memory("always run make first")];
        // Same meaning, different spacing/case -> recognized as duplicate.
        let candidates = vec![
            draft("ALWAYS  run Make first", "rule"),
            draft("new one", "tip"),
        ];
        let fresh = dedupe(&candidates, &existing);
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].text, "new one");
    }

    #[test]
    fn drops_duplicates_within_the_batch() {
        let candidates = vec![draft("use make", "rule"), draft("USE make", "tip")];
        let fresh = dedupe(&candidates, &[]);
        assert_eq!(fresh.len(), 1);
    }

    #[test]
    fn reflection_messages_include_project_and_transcript() {
        let history = vec![
            Message::System("sys".into()),
            text_msg("user", "fix build"),
            text_msg("assistant", "done"),
        ];
        let msgs = reflection_messages(&history, Path::new("/tmp/proj"));
        assert_eq!(msgs.len(), 2); // system instruction + user transcript
        match &msgs[0] {
            Message::System(s) => {
                assert!(s.contains("lessons"));
                assert!(s.contains("/tmp/proj"));
                assert!(s.contains("rule"));
            }
            _ => panic!("first message should be the system instruction"),
        }
        match &msgs[1] {
            Message::User(u) => {
                assert!(u.contains("user: fix build"));
                assert!(!u.contains("sys")); // system excluded from transcript
            }
            _ => panic!("second message should be the user transcript"),
        }
    }

    #[test]
    fn extract_calls_provider_and_returns_lessons() {
        use crate::provider::{fake::FakeProvider, Response};
        let history = vec![
            Message::System("sys".into()),
            text_msg("user", "fix build"),
            text_msg("assistant", "done"),
        ];
        let fake = FakeProvider::new(vec![Response::Text(
            r#"[{"text":"run make","kind":"rule"}]"#.into(),
        )]);
        let lessons = extract(
            &fake,
            &std::sync::atomic::AtomicBool::new(false),
            &history,
            Path::new("/tmp/proj"),
        )
        .unwrap();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, "run make");
    }

    #[test]
    fn extract_rejects_empty_history_without_calling_provider() {
        use crate::provider::{fake::FakeProvider, Response};
        let fake = FakeProvider::new(vec![Response::Text("[]".into())]);
        let err = extract(
            &fake,
            &std::sync::atomic::AtomicBool::new(false),
            &[Message::System("sys".into())],
            Path::new("/tmp/proj"),
        )
        .unwrap_err();
        assert!(matches!(err, ReflectError::EmptyHistory));
        assert_eq!(fake.calls(), 0, "no LLM call for empty history");
    }

    #[test]
    fn extract_surfaces_provider_failure() {
        use crate::provider::fake::FakeProvider;
        // Exhausted fake returns final text "done" which is not JSON -> malformed.
        let fake = FakeProvider::new(vec![]);
        let history = vec![text_msg("user", "q"), text_msg("assistant", "a")];
        let err = extract(
            &fake,
            &std::sync::atomic::AtomicBool::new(false),
            &history,
            Path::new("/tmp/proj"),
        )
        .unwrap_err();
        assert!(matches!(err, ReflectError::Malformed(_)));
    }

    #[test]
    fn lessons_from_drafts_attach_provenance() {
        let drafts = vec![draft("Always run make", "rule")];
        let lessons =
            lessons_from_drafts(&drafts, Path::new("/tmp/proj"), Some("sess-9".into()), 42);
        assert_eq!(lessons.len(), 1);
        let l = &lessons[0];
        assert_eq!(l.id, lesson_id(&normalize_text("Always run make")));
        assert_eq!(l.cwd, "/tmp/proj");
        assert_eq!(l.source_session_id.as_deref(), Some("sess-9"));
        assert_eq!(l.created_at, 42);
        assert!(!l.retracted);
        assert_eq!(l.kind, "rule");
    }

    struct TempDir(std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_mem(name: &str) -> (TempDir, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!("crab-reflect-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        (TempDir(base.clone()), base)
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message::System("sys".into()),
            text_msg("user", "fix the build"),
            text_msg("assistant", "I ran make and fixed it"),
        ]
    }

    fn lessons_json(n: usize) -> String {
        let mut body = String::new();
        for i in 0..n {
            if i > 0 {
                body.push(',');
            }
            body.push_str(&format!("{{\"text\":\"lesson {i}\",\"kind\":\"rule\"}}"));
        }
        format!("[{body}]")
    }

    #[test]
    fn reflect_and_store_appends_new_lessons_with_provenance() {
        use crate::provider::{fake::FakeProvider, Response};
        let (_guard, mem) = temp_mem("store1");
        let fake = FakeProvider::new(vec![Response::Text(lessons_json(2))]);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let added = reflect_and_store(
            &mem,
            Path::new("/tmp/proj"),
            &fake,
            &cancel,
            &conversation(),
            Some("sess-7".into()),
            100,
        )
        .unwrap();
        assert_eq!(added, 2);
        let stored = crate::memory::list_lessons(&mem, Path::new("/tmp/proj")).unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].text, "lesson 0");
        assert_eq!(stored[0].source_session_id.as_deref(), Some("sess-7"));
        assert_eq!(stored[0].created_at, 100);
    }

    #[test]
    fn reflect_and_store_dedupes_against_existing_memory() {
        use crate::provider::{fake::FakeProvider, Response};
        let (_guard, mem) = temp_mem("store-dedupe");
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // First reflection stores two lessons.
        let fake = FakeProvider::new(vec![Response::Text(lessons_json(2))]);
        assert_eq!(
            reflect_and_store(
                &mem,
                Path::new("/tmp/proj"),
                &fake,
                &cancel,
                &conversation(),
                None,
                100,
            )
            .unwrap(),
            2
        );
        // A second reflection over the same transcript yields the same
        // lessons -> nothing new to add.
        let fake2 = FakeProvider::new(vec![Response::Text(lessons_json(2))]);
        assert_eq!(
            reflect_and_store(
                &mem,
                Path::new("/tmp/proj"),
                &fake2,
                &cancel,
                &conversation(),
                None,
                200,
            )
            .unwrap(),
            0
        );
        assert_eq!(
            crate::memory::list_lessons(&mem, Path::new("/tmp/proj"))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn reflect_and_store_error_never_touches_memory() {
        use crate::provider::{fake::FakeProvider, Response};
        let (_guard, mem) = temp_mem("store-err");
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // A malformed answer (non-JSON) must error and leave memory empty.
        let fake = FakeProvider::new(vec![Response::Text("not json at all".into())]);
        let err = reflect_and_store(
            &mem,
            Path::new("/tmp/proj"),
            &fake,
            &cancel,
            &conversation(),
            None,
            100,
        )
        .unwrap_err();
        assert!(matches!(err, ReflectError::Malformed(_)));
        assert_eq!(
            crate::memory::list_lessons(&mem, Path::new("/tmp/proj")).unwrap(),
            Vec::<crate::memory::Lesson>::new()
        );
        // An empty history must be rejected before any LLM call.
        let fake2 = FakeProvider::new(vec![Response::Text(lessons_json(1))]);
        let err = reflect_and_store(
            &mem,
            Path::new("/tmp/proj"),
            &fake2,
            &cancel,
            &[Message::System("sys".into())],
            None,
            100,
        )
        .unwrap_err();
        assert!(matches!(err, ReflectError::EmptyHistory));
        assert_eq!(fake2.calls(), 0);
    }
}
