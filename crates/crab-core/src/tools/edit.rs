//! The `edit` tool: apply precise, validated text replacements to a file.
//!
//! Every `oldText` must match exactly once; zero or multiple matches are an
//! error. Matching tolerates CRLF line endings, a UTF-8 BOM, and common
//! confusables (smart quotes, Unicode dashes, Unicode spaces) via a fuzzy
//! fallback that preserves the file's original bytes on unchanged text. Edits
//! are matched against the original file (not incrementally) and applied
//! together, so disjoint replacements cannot interfere.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

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

/// Map a single character to its ASCII equivalent for fuzzy matching. This is
/// length-preserving (one char in, one char out), which lets fuzzy-space char
/// offsets be mapped back onto the original content.
fn fuzzy_char(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        other => other,
    }
}

fn fuzzy_normalize(s: &str) -> String {
    s.chars().map(fuzzy_char).collect()
}

/// Byte offset of the `char_idx`-th character in `s`.
fn char_index_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
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

    let fuzzy_content = fuzzy_normalize(content);
    let fuzzy_old = fuzzy_normalize(old);
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
            let char_idx = fuzzy_content[..fuzzy[0]].chars().count();
            let start = char_index_to_byte(content, char_idx);
            let end = char_index_to_byte(content, char_idx + fuzzy_old.chars().count());
            Ok((start, end))
        }
        n => Err(ToolError::Invalid(format!(
            "{} matched {n} times in '{path}' (expected exactly 1)",
            describe_edit(idx, total)
        ))),
    }
}

/// Minimal unified diff (no context) between two texts, sufficient to show the
/// model what changed without a heavy diff dependency.
fn unified_diff(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.split('\n').collect();
    let new_lines: Vec<&str> = new.split('\n').collect();

    let mut prefix = 0;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < old_lines.len() - prefix
        && suffix < new_lines.len() - prefix
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let old_start = prefix;
    let old_count = old_lines.len() - prefix - suffix;
    let new_start = prefix;
    let new_count = new_lines.len() - prefix - suffix;

    let mut out = format!(
        "@@ -{},{} +{},{} @@\n",
        if old_count > 0 {
            old_start + 1
        } else {
            old_start
        },
        old_count,
        if new_count > 0 {
            new_start + 1
        } else {
            new_start
        },
        new_count
    );
    for line in &old_lines[old_start..old_start + old_count] {
        out.push_str(&format!("-{line}\n"));
    }
    for line in &new_lines[new_start..new_start + new_count] {
        out.push_str(&format!("+{line}\n"));
    }
    out
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

    fn run(
        &self,
        workspace: &Workspace,
        args: &Value,
        _cancel: &AtomicBool,
    ) -> Result<ToolOutput, ToolError> {
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
        let bytes = match std::fs::read(&resolved) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path));
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
            let (start, end) = locate_match(&lf, &old_lf, &path, idx, edits.len())?;
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
        std::fs::write(&resolved, &out).map_err(|e| ToolError::Io(e.to_string()))?;

        Ok(ToolOutput {
            content: format!(
                "Successfully replaced {} block(s) in {path}\n\n{diff}",
                edits.len()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str, contents: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-edit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), contents).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    fn read(dir: &std::path::Path) -> String {
        std::fs::read_to_string(dir.join("a.txt")).unwrap()
    }

    #[test]
    fn unique_match_replaces() {
        let (ws, dir) = setup("unique", "hello world");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "world", "newText": "there"}),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(read(&dir), "hello there");
    }

    #[test]
    fn zero_match_is_error() {
        let (ws, _dir) = setup("zero", "hello");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "oldText": "zzz", "newText": "x"}),
                &std::sync::atomic::AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[test]
    fn multiple_match_is_error() {
        let (ws, _dir) = setup("multi", "a a a");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "oldText": "a", "newText": "b"}),
                &std::sync::atomic::AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[test]
    fn multiple_disjoint_edits() {
        let (ws, dir) = setup("disjoint", "one two three");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "edits": [
                {"oldText": "one", "newText": "1"},
                {"oldText": "three", "newText": "3"}
            ]}),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(read(&dir), "1 two 3");
    }

    #[test]
    fn edits_crlf_file_and_preserves_line_endings() {
        let (ws, dir) = setup("crlf", "line one\r\nline two\r\nline three\r\n");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "line two", "newText": "LINE TWO"}),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(read(&dir), "line one\r\nLINE TWO\r\nline three\r\n");
    }

    #[test]
    fn fuzzy_matches_smart_quotes() {
        // The file uses a curly quote; the model sends an ASCII quote.
        let (ws, dir) = setup("fuzzy", "don\u{2019}t panic");
        let tool = EditTool;
        tool.run(
            &ws,
            &json!({"path": "a.txt", "oldText": "don't", "newText": "do not"}),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(read(&dir), "do not panic");
    }

    #[test]
    fn overlapping_edits_are_rejected() {
        let (ws, _dir) = setup("overlap", "hello world");
        let tool = EditTool;
        let err = tool
            .run(
                &ws,
                &json!({"path": "a.txt", "edits": [
                    {"oldText": "hello", "newText": "hi"},
                    {"oldText": "hello world", "newText": "bye"}
                ]}),
                &std::sync::atomic::AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }
}
