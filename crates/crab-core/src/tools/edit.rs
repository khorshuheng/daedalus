//! The `edit` tool: apply precise, validated text replacements to a file.
//!
//! Every `oldText` must match exactly once; zero or multiple matches are an
//! error. Matching tolerates CRLF line endings, a UTF-8 BOM, and common
//! confusables (smart quotes, Unicode dashes, Unicode spaces) via a fuzzy
//! fallback that preserves the file's original bytes on unchanged text. Edits
//! are matched against the original file (not incrementally) and applied
//! together, so disjoint replacements cannot interfere.

use futures::future::BoxFuture;
use serde_json::{json, Value};
use std::path::Path;
use tokio_util::sync::CancellationToken;

use super::mutation::with_file_mutation;
use super::{arg_string, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct EditTool;

/// Detect the dominant line ending: `\r\n` only when the first `\r\n` precedes
/// the first bare `\n` (same rule as pi).
fn detect_line_ending(s: &str) -> &'static str {
    match (s.find("\r\n"), s.find('\n')) {
        (Some(c), Some(l)) if c < l => "\r\n",
        _ => "\n",
    }
}

fn normalize_lf(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

fn restore_line_endings(s: &str, ending: &str) -> String {
    if ending == "\r\n" {
        s.replace('\n', "\r\n")
    } else {
        s.to_string()
    }
}

/// Map a single character to its ASCII equivalent for fuzzy matching, after
/// NFKC normalization. Length-preserving (one char in, one char out).
fn fuzzy_char(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        other => other,
    }
}

/// A normalized string plus, for each normalized char, the byte range it
/// came from in the original. NFKC is applied per extended grapheme cluster
/// (CRAB-149): it can compose across code points ("e" + U+0301 -> "é") and
/// expand one cluster into several chars (ligatures, compat forms), so a
/// match in normalized space maps back to the original through this table.
/// Trailing whitespace on each line is trimmed and excluded from the map
/// (pi's `normalizeForFuzzyMatch`).
fn normalize_for_fuzzy(s: &str) -> (String, Vec<(usize, usize)>) {
    use unicode_normalization::UnicodeNormalization;
    use unicode_segmentation::UnicodeSegmentation;
    // Normalize whole grapheme clusters rather than single chars: NFKC
    // composition only happens within a combining sequence, i.e. within one
    // extended grapheme cluster, so this equals whole-string NFKC while still
    // letting each output char know its source byte range.
    let mut norm_chars: Vec<(char, (usize, usize))> = Vec::new();
    for (byte_start, grapheme) in s.grapheme_indices(true) {
        let byte_end = byte_start + grapheme.len();
        for n in grapheme.nfkc() {
            norm_chars.push((fuzzy_char(n), (byte_start, byte_end)));
        }
    }
    // Trim trailing whitespace on each line (pi's normalizeForFuzzyMatch), so
    // a model's oldText without trailing whitespace matches file lines that
    // have it (spaces, tabs, any Unicode whitespace — `char::is_whitespace`,
    // CRAB-150). Trimming drops entries from the map, keeping indices aligned.
    let mut result = String::new();
    let mut result_map: Vec<(usize, usize)> = Vec::new();
    let mut line: Vec<(char, (usize, usize))> = Vec::new();
    for item in norm_chars {
        if item.0 == '\n' {
            let mut keep = line.len();
            while keep > 0 && line[keep - 1].0.is_whitespace() {
                keep -= 1;
            }
            // Truncate *before* draining: `drain(..keep)` alone leaves the
            // trimmed whitespace in `line`, which would carry it onto the next
            // line.
            line.truncate(keep);
            for (ch, range) in line.drain(..) {
                result.push(ch);
                result_map.push(range);
            }
            result.push('\n');
            result_map.push(item.1);
        } else {
            line.push(item);
        }
    }
    let mut keep = line.len();
    while keep > 0 && line[keep - 1].0.is_whitespace() {
        keep -= 1;
    }
    line.truncate(keep);
    for (ch, range) in line.drain(..) {
        result.push(ch);
        result_map.push(range);
    }
    (result, result_map)
}

fn describe_edit(idx: usize, total: usize) -> String {
    if total == 1 {
        "oldText".to_string()
    } else {
        format!("edits[{idx}].oldText")
    }
}

/// Locate `old` in `content`, trying an exact match first and then a fuzzy
/// match, returning the byte range in `content`. Errors on empty, missing, or
/// ambiguous matches.
fn locate_match(
    content: &str,
    old: &str,
    path: &str,
    idx: usize,
    total: usize,
) -> Result<(usize, usize), ToolError> {
    if old.is_empty() {
        return Err(ToolError::Invalid(format!(
            "{} must not be empty in '{path}'",
            describe_edit(idx, total)
        )));
    }

    let exact: Vec<usize> = content.match_indices(old).map(|(i, _)| i).collect();
    match exact.len() {
        1 => {
            let start = exact[0];
            return Ok((start, start + old.len()));
        }
        0 => {}
        n => {
            return Err(ToolError::Invalid(format!(
                "{} matched {n} times in '{path}' (expected exactly 1)",
                describe_edit(idx, total)
            )))
        }
    }

    let (fuzzy_content, map) = normalize_for_fuzzy(content);
    let (fuzzy_old, _old_map) = normalize_for_fuzzy(old);
    // Byte index of a normalized char = its position in the String; but the
    // match range in the *String* is byte-based while `map` is per-char. Walk
    // char indices to locate the normalized match, then translate both ends.
    let fuzzy: Vec<usize> = fuzzy_content
        .match_indices(&fuzzy_old)
        .map(|(i, _)| i)
        .collect();
    match fuzzy.len() {
        0 => Err(ToolError::Invalid(format!(
            "{} not found in '{path}'",
            describe_edit(idx, total)
        ))),
        1 => {
            let char_start = fuzzy_content[..fuzzy[0]].chars().count();
            let char_end = char_start + fuzzy_old.chars().count();
            // map[i] = original byte range of the i-th normalized char. The
            // matched region spans chars [char_start, char_end): its start is
            // the start of the first matched char and its end the end of the
            // last matched char.
            let start = map.get(char_start).map(|r| r.0).unwrap_or(content.len());
            let end = map
                .get(char_end.saturating_sub(1))
                .map(|r| r.1)
                .unwrap_or(content.len());
            Ok((start, end))
        }
        n => Err(ToolError::Invalid(format!(
            "{} matched {n} times in '{path}' (expected exactly 1)",
            describe_edit(idx, total)
        ))),
    }
}

/// Minimal unified diff (no context) between two texts, sufficient to show the
/// model what changed. Delegates the line diff to `similar` (CRAB-119),
/// rendered with no surrounding context.
fn unified_diff(old: &str, new: &str) -> String {
    similar::TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(0)
        .to_string()
}

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to edit." },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    },
                    "description": "Disjoint replacements to apply in order."
                },
                "oldText": { "type": "string", "description": "Alternative to 'edits' for a single replacement." },
                "newText": { "type": "string" }
            },
            "required": ["path"],
            "oneOf": [
                { "required": ["edits"] },
                { "required": ["oldText", "newText"] }
            ]
        })
    }

    fn run<'a>(
        &'a self,
        workspace: &'a Workspace,
        args: &'a Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>> {
        Box::pin(async move {
            let ws = workspace.clone();
            let args = args.clone();
            tokio::task::spawn_blocking(move || EditTool.run_sync(&ws, &args))
                .await
                .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

impl EditTool {
    /// The synchronous body, executed on the blocking pool (CRAB-130).
    fn run_sync(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let edits: Vec<(String, String)> =
            if let Some(edits) = args.get("edits").and_then(|v| v.as_array()) {
                if edits.is_empty() {
                    return Err(ToolError::Argument("'edits' must not be empty".into()));
                }
                edits
                    .iter()
                    .map(|e| {
                        let old = arg_string(e, "oldText")?;
                        let new = arg_string(e, "newText")?;
                        Ok((old, new))
                    })
                    .collect::<Result<_, ToolError>>()?
            } else {
                vec![(arg_string(args, "oldText")?, arg_string(args, "newText")?)]
            };

        let resolved = resolve(workspace, Path::new(&path))?;
        // Serialize with any concurrent write/edit of the same file (CRAB-146).
        with_file_mutation(&resolved, || edit_at(&resolved, &path, &edits))
    }
}

/// The locked body of `edit`: reject non-regular files, read, match, and write.
fn edit_at(
    resolved: &Path,
    path: &str,
    edits: &[(String, String)],
) -> Result<ToolOutput, ToolError> {
    // Reject non-regular files before opening: a FIFO/device/socket would
    // block the read forever, and blocking tasks cannot be cancelled
    // (CRAB-139 review).
    let meta = match std::fs::metadata(resolved) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ToolError::NotFound(path.to_string()));
        }
        Err(e) => return Err(ToolError::Io(e.to_string())),
    };
    if !meta.is_file() {
        return Err(ToolError::Invalid(format!(
            "refusing to edit '{path}': not a regular file"
        )));
    }
    let bytes = match std::fs::read(resolved) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ToolError::NotFound(path.to_string()));
        }
        Err(e) => return Err(ToolError::Io(e.to_string())),
    };

    let (has_bom, body) = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        (true, &bytes[3..])
    } else {
        (false, bytes.as_slice())
    };
    let content = match std::str::from_utf8(body) {
        Ok(s) => s.to_string(),
        Err(_) => {
            return Err(ToolError::Invalid(format!(
                "file '{path}' is not valid UTF-8"
            )))
        }
    };

    let ending = detect_line_ending(&content);
    let lf = normalize_lf(&content);

    let mut replacements: Vec<(usize, usize, String)> = Vec::with_capacity(edits.len());
    for (idx, (old, new)) in edits.iter().enumerate() {
        let old_lf = normalize_lf(old);
        let new_lf = normalize_lf(new);
        let (start, end) = locate_match(&lf, &old_lf, path, idx, edits.len())?;
        replacements.push((start, end, new_lf));
    }

    replacements.sort_by_key(|r| r.0);
    for window in replacements.windows(2) {
        if window[0].1 > window[1].0 {
            return Err(ToolError::Invalid(format!(
                "edits overlap in '{path}': merge overlapping edits or target disjoint regions"
            )));
        }
    }

    let mut new_lf = lf.clone();
    for (start, end, new) in replacements.iter().rev() {
        new_lf.replace_range(*start..*end, new);
    }
    if new_lf == lf {
        return Err(ToolError::Invalid(format!(
            "no change made to '{path}': replacements produced identical content"
        )));
    }

    let diff = unified_diff(&lf, &new_lf);

    let restored = restore_line_endings(&new_lf, ending);
    let mut out = Vec::with_capacity(restored.len() + 3);
    if has_bom {
        out.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
    }
    out.extend_from_slice(restored.as_bytes());
    std::fs::write(resolved, &out).map_err(|e| ToolError::Io(e.to_string()))?;

    Ok(ToolOutput {
        content: format!(
            "Successfully replaced {} block(s) in {path}\n\n{diff}",
            edits.len()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(_name: &str, contents: &str) -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("a.txt"), contents).unwrap();
        (Workspace::new(root).unwrap(), dir)
    }

    fn read(dir: &tempfile::TempDir) -> String {
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap()
    }

    #[tokio::test]
    async fn unique_match_replaces() {
        let (ws, dir) = setup("unique", "hello world");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "world", "newText": "there"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "hello there");
    }

    #[tokio::test]
    async fn zero_match_is_error() {
        let (ws, _dir) = setup("zero", "hello");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "oldText": "zzz", "newText": "x"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[tokio::test]
    async fn multiple_match_is_error() {
        let (ws, _dir) = setup("multi", "a a a");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "oldText": "a", "newText": "b"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[tokio::test]
    async fn multiple_disjoint_edits() {
        let (ws, dir) = setup("disjoint", "one two three");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "edits": [
                {"oldText": "one", "newText": "1"},
                {"oldText": "three", "newText": "3"}
            ]}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "1 two 3");
    }

    #[tokio::test]
    async fn edits_crlf_file_and_preserves_line_endings() {
        let (ws, dir) = setup("crlf", "line one\r\nline two\r\nline three\r\n");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "line two", "newText": "LINE TWO"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "line one\r\nLINE TWO\r\nline three\r\n");
    }

    #[tokio::test]
    async fn fuzzy_matches_smart_quotes() {
        // The file uses a curly quote; the model sends an ASCII quote.
        let (ws, dir) = setup("fuzzy", "don\u{2019}t panic");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "don't", "newText": "do not"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "do not panic");
    }

    #[tokio::test]
    async fn fuzzy_matches_nfkc_fullwidth_and_ligatures() {
        // Fullwidth Latin (NFKC -> ASCII) in the model's oldText.
        let (ws, dir) = setup("nfkc", "use HelloWorld here");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "ＨｅｌｌｏＷｏｒｌｄ", "newText": "hi"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "use hi here");
    }

    #[tokio::test]
    async fn fuzzy_tolerates_trailing_whitespace_on_the_line() {
        // The model's oldText has no trailing spaces, but the file line does;
        // the match must still succeed (pi's normalizeForFuzzyMatch trims each
        // line). The trailing spaces themselves are not part of oldText, so
        // they remain after the replacement.
        let (ws, dir) = setup("trailing", "alpha   \nbeta   \n");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "beta", "newText": "gamma"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "alpha   \ngamma   \n");
    }

    /// CRAB-149: NFKC composition across a code-point boundary must match
    /// (precomposed file text vs a decomposed model oldText).
    #[tokio::test]
    async fn fuzzy_matches_decomposed_against_precomposed() {
        // "café" with a precomposed U+00E9.
        let (ws, dir) = setup("decomposed", "caf\u{00E9}\n");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "cafe\u{0301}", "newText": "tea"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "tea\n");
    }

    /// CRAB-149: bytes around a normalized match are preserved exactly.
    #[tokio::test]
    async fn fuzzy_match_preserves_surrounding_bytes() {
        let (ws, dir) = setup("surround", "before cafe\u{0301} after\n");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "caf\u{00E9}", "newText": "X"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(read(&dir), "before X after\n");
    }

    /// CRAB-150: trailing whitespace other than a space (tabs, etc.) is trimmed
    /// in the fuzzy view too, and bytes outside the match are preserved.
    #[tokio::test]
    async fn fuzzy_trims_trailing_tabs() {
        let (ws, dir) = setup("trailtab", "alpha\t\nbeta\t\n");
        let tool = EditTool;
        // No exact match: the interior tab after "alpha" forces the fuzzy path.
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "alpha\nbeta", "newText": "ALPHA\nBETA"}),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        // The trailing tab after BETA is outside the match and survives.
        assert_eq!(read(&dir), "ALPHA\nBETA\t\n");
    }

    #[tokio::test]
    async fn overlapping_edits_are_rejected() {
        let (ws, _dir) = setup("overlap", "hello world");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "edits": [
                    {"oldText": "hello", "newText": "hi"},
                    {"oldText": "hello world", "newText": "bye"}
                ]}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    /// CRAB-146: concurrent edits to the same file must not lose updates.
    #[tokio::test]
    async fn concurrent_edits_to_same_file_all_apply() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let original: String = (0..8).map(|i| format!("line{i}\n")).collect();
        std::fs::write(root.join("a.txt"), &original).unwrap();
        let ws = Workspace::new(root).unwrap();

        let edits = (0..8).map(|i| {
            let ws = ws.clone();
            async move {
                EditTool
                    .run(
                        &ws,
                        &json!({
                            "path": "a.txt",
                            "oldText": format!("line{i}"),
                            "newText": format!("LINE{i}")
                        }),
                        tokio_util::sync::CancellationToken::new(),
                    )
                    .await
            }
        });
        for result in futures::future::join_all(edits).await {
            result.unwrap();
        }

        let expected: String = (0..8).map(|i| format!("LINE{i}\n")).collect();
        let got = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(got, expected);
    }

    /// CRAB-139 review: a FIFO must be rejected before the read, which would
    /// otherwise block forever.
    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_non_regular_files() {
        use std::ffi::CString;
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().to_path_buf();
        let fifo = root.join("pipe");
        let c = CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: mkfifo with a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
        let ws = Workspace::new(root).unwrap();
        let err = EditTool
            .run(
                &ws,
                &json!({"path": "pipe", "oldText": "a", "newText": "b"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
    }
}
