//! The `search` tool (CRAB-144): ripgrep-style content search over the
//! workspace.
//!
//! Traversal uses the `ignore` crate — the same walker ripgrep uses — so the
//! defaults skip hidden files/directories (`.git`, `.cache`, …) and respect
//! `.gitignore`/`.ignore`. Matching uses `regex`. Everything is bounded: a
//! total match cap, a per-line display cap, an output cap, a per-file size cap
//! and a total scanned-byte budget, plus cooperative cancellation — so a search
//! cannot run away the way `grep -r ~` did.
//!
//! The argument set mirrors ripgrep's common flags (`-i/-S/-F/-w/-v/-l/-c/-C/
//! -A/-B/-m/-g/-t/--hidden/--no-ignore/-L/-o`).

use futures::future::BoxFuture;
use ignore::overrides::OverrideBuilder;
use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::Path;
use tokio_util::sync::CancellationToken;

use super::{arg_string, arg_usize, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// Total matches returned unless `max_results` says otherwise.
const DEFAULT_MAX_RESULTS: usize = 100;
/// Hard ceiling on `max_results`.
const HARD_MAX_RESULTS: usize = 1_000;
/// A single matched/context line is clipped to this many characters.
const MAX_LINE_DISPLAY: usize = 200;
/// Files larger than this are skipped (matches `read`'s whole-read threshold).
const MAX_SEARCH_FILE: u64 = 4 * 1024 * 1024;
/// Total bytes read across all files before the search stops.
const MAX_SCAN_BYTES: u64 = 256 * 1024 * 1024;

pub struct SearchTool {
    pub max_output: usize,
}

impl Tool for SearchTool {
    fn name(&self) -> &'static str {
        "search"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex to search for (Rust regex syntax)." },
                "path": { "type": "string", "description": "File or directory to search (default: the workspace root)." },
                "glob": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Include/exclude globs; a leading '!' excludes (rg -g). Repeatable."
                },
                "type": { "type": "string", "description": "Only search files of this type, e.g. rust, py, js (rg -t)." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive (rg -i)." },
                "smart_case": { "type": "boolean", "description": "Case-insensitive unless the pattern has an uppercase letter (rg -S)." },
                "fixed_strings": { "type": "boolean", "description": "Treat the pattern as a literal string (rg -F)." },
                "word": { "type": "boolean", "description": "Match whole words only (rg -w)." },
                "invert": { "type": "boolean", "description": "Select lines that do NOT match (rg -v)." },
                "only_matching": { "type": "boolean", "description": "Print only the matched text, one per occurrence (rg -o)." },
                "files_with_matches": { "type": "boolean", "description": "Print only paths with at least one match (rg -l)." },
                "count": { "type": "boolean", "description": "Print a per-file match count (rg -c)." },
                "context": { "type": "integer", "minimum": 0, "description": "Lines of context before and after a match (rg -C)." },
                "after_context": { "type": "integer", "minimum": 0, "description": "Lines of context after a match (rg -A)." },
                "before_context": { "type": "integer", "minimum": 0, "description": "Lines of context before a match (rg -B)." },
                "max_count": { "type": "integer", "minimum": 1, "description": "Stop after this many matches per file (rg -m)." },
                "hidden": { "type": "boolean", "description": "Also search hidden files and directories (rg --hidden)." },
                "no_ignore": { "type": "boolean", "description": "Do not respect .gitignore/.ignore (rg --no-ignore)." },
                "follow": { "type": "boolean", "description": "Follow symbolic links (rg -L)." },
                "max_results": { "type": "integer", "minimum": 1, "description": "Total match cap for this call (default 100, max 1000)." }
            },
            "required": ["pattern"]
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
                SearchTool { max_output }.run_sync(&ws, &args, &cancel)
            })
            .await
            .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

/// Options that shape one file's output.
struct Opts {
    invert: bool,
    files_with_matches: bool,
    count: bool,
    only_matching: bool,
    before: usize,
    after: usize,
    max_count: Option<usize>,
}

/// The rendered result for one file.
struct FileOutcome {
    output: String,
    /// Result records produced (match lines, occurrences, or one per file).
    emitted: usize,
    /// A cap was reached, so the walk should stop.
    hit_cap: bool,
}

fn get_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// `glob` accepts a single string or an array of strings.
fn get_globs(args: &Value) -> Result<Vec<String>, ToolError> {
    match args.get("glob") {
        None => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ToolError::Argument("'glob' entries must be strings".into()))
            })
            .collect(),
        Some(_) => Err(ToolError::Argument(
            "'glob' must be a string or an array of strings".into(),
        )),
    }
}

/// Clip a line for display, char-boundary safe.
fn clip(s: &str) -> String {
    let mut out: String = s.chars().take(MAX_LINE_DISPLAY).collect();
    if s.chars().count() > MAX_LINE_DISPLAY {
        out.push('…');
    }
    out
}

/// Emit rg's `--` separator when the next line is not adjacent to the last one.
/// Only used when context is requested (rg omits it otherwise).
fn group_sep(out: &mut String, last: &mut Option<usize>, next: usize, enabled: bool) {
    if !enabled {
        return;
    }
    if let Some(prev) = *last {
        if next > prev + 1 {
            out.push_str("--\n");
        }
    }
}

impl SearchTool {
    fn run_sync(
        &self,
        workspace: &Workspace,
        args: &Value,
        cancel: &CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        let pattern = arg_string(args, "pattern")?;
        let fixed = get_bool(args, "fixed_strings");
        let word = get_bool(args, "word");
        let ignore_case = get_bool(args, "ignore_case");
        let smart_case = get_bool(args, "smart_case");
        let hidden = get_bool(args, "hidden");
        let no_ignore = get_bool(args, "no_ignore");
        let follow = get_bool(args, "follow");

        let context = arg_usize(args, "context")?.unwrap_or(0);
        let opts = Opts {
            invert: get_bool(args, "invert"),
            files_with_matches: get_bool(args, "files_with_matches"),
            count: get_bool(args, "count"),
            only_matching: get_bool(args, "only_matching"),
            before: arg_usize(args, "before_context")?.unwrap_or(context),
            after: arg_usize(args, "after_context")?.unwrap_or(context),
            max_count: arg_usize(args, "max_count")?,
        };
        let max_results = arg_usize(args, "max_results")?
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, HARD_MAX_RESULTS);

        // Build the matcher. `word`/`fixed` transform the pattern; smart-case is
        // judged from the original pattern (like rg).
        let mut source = pattern.clone();
        if fixed {
            source = regex::escape(&source);
        }
        if word {
            source = format!(r"\b(?:{source})\b");
        }
        let case_insensitive =
            ignore_case || (smart_case && !pattern.chars().any(char::is_uppercase));
        let re = RegexBuilder::new(&source)
            .case_insensitive(case_insensitive)
            .build()
            .map_err(|e| ToolError::Argument(format!("invalid regex '{pattern}': {e}")))?;

        let root = match args.get("path").and_then(Value::as_str) {
            Some(p) => resolve(workspace, Path::new(p))?,
            None => workspace.root().to_path_buf(),
        };
        if !root.exists() {
            return Err(ToolError::NotFound(
                args.get("path")
                    .and_then(Value::as_str)
                    .unwrap_or(".")
                    .to_string(),
            ));
        }

        // The `ignore` walker: defaults already skip hidden entries and honor
        // .gitignore/.ignore, exactly like ripgrep.
        let mut wb = WalkBuilder::new(&root);
        if hidden {
            wb.hidden(false);
        }
        if follow {
            wb.follow_links(true);
        }
        if no_ignore {
            wb.ignore(false)
                .git_ignore(false)
                .git_global(false)
                .git_exclude(false)
                .parents(false);
        }
        let globs = get_globs(args)?;
        if !globs.is_empty() {
            let mut ob = OverrideBuilder::new(&root);
            for g in &globs {
                ob.add(g)
                    .map_err(|e| ToolError::Argument(format!("bad glob '{g}': {e}")))?;
            }
            wb.overrides(
                ob.build()
                    .map_err(|e| ToolError::Argument(format!("bad globs: {e}")))?,
            );
        }
        if let Some(t) = args.get("type").and_then(Value::as_str) {
            let mut tb = TypesBuilder::new();
            tb.add_defaults();
            tb.select(t);
            wb.types(
                tb.build()
                    .map_err(|e| ToolError::Argument(format!("bad type '{t}': {e}")))?,
            );
        }

        let mut out = String::new();
        let mut emitted = 0usize;
        let mut scanned = 0u64;
        let mut skipped_binary = 0usize;
        let mut skipped_large = 0usize;
        let mut truncated = false;

        'walk: for result in wb.build() {
            if cancel.is_cancelled() {
                return Err(ToolError::Cancelled);
            }
            let Ok(entry) = result else { continue };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.len() > MAX_SEARCH_FILE {
                skipped_large += 1;
                continue;
            }
            scanned += meta.len();
            if scanned > MAX_SCAN_BYTES {
                truncated = true;
                break 'walk;
            }
            let bytes = match std::fs::read(entry.path()) {
                Ok(b) => b,
                Err(_) => continue,
            };
            if bytes.iter().take(8192).any(|&b| b == 0) {
                skipped_binary += 1;
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);

            let display = entry
                .path()
                .strip_prefix(workspace.root())
                .unwrap_or(entry.path())
                .display()
                .to_string();

            let match_budget = max_results.saturating_sub(emitted);
            let out_budget = self.max_output.saturating_sub(out.len());
            let outcome = render_file(&re, &display, &text, &opts, match_budget, out_budget);
            if !outcome.output.is_empty() {
                out.push_str(&outcome.output);
            }
            emitted += outcome.emitted;
            if outcome.hit_cap || emitted >= max_results {
                truncated = true;
                break 'walk;
            }
        }

        if out.is_empty() {
            return Ok(ToolOutput {
                content: format!("no matches for '{pattern}'"),
            });
        }

        let mut content = out.trim_end().to_string();
        if truncated {
            content.push_str(&format!(
                "\n\n[truncated at {emitted} results; narrow the pattern, add a glob/type, or raise max_results]"
            ));
        }
        if skipped_binary + skipped_large > 0 {
            content.push_str(&format!(
                "\n[skipped {} binary and {} oversize file(s)]",
                skipped_binary, skipped_large
            ));
        }
        Ok(ToolOutput { content })
    }
}

/// Render one file's matches/context. `match_budget` caps result records and
/// `out_budget` caps bytes; reaching either sets `hit_cap`.
fn render_file(
    re: &Regex,
    display: &str,
    text: &str,
    opts: &Opts,
    match_budget: usize,
    out_budget: usize,
) -> FileOutcome {
    let mut out = String::new();
    let mut emitted = 0usize;
    let mut hit_cap = false;

    // `-l` and `-c` emit one record per matched file.
    if opts.files_with_matches || opts.count {
        let mut n = 0usize;
        for line in text.lines() {
            if re.is_match(line) != opts.invert {
                n += 1;
            }
        }
        if n > 0 {
            if emitted >= match_budget || out.len() >= out_budget {
                hit_cap = true;
            } else if opts.files_with_matches {
                out.push_str(display);
                out.push('\n');
                emitted += 1;
            } else {
                out.push_str(&format!("{display}:{n}\n"));
                emitted += 1;
            }
        }
        return FileOutcome {
            output: out,
            emitted,
            hit_cap,
        };
    }

    let mut before: VecDeque<(usize, String)> = VecDeque::new();
    let mut after_remaining = 0usize;
    let mut last_emitted: Option<usize> = None;
    // rg prints `--` between non-contiguous groups only with context.
    let separators = opts.before > 0 || opts.after > 0;

    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let is_match = re.is_match(line) != opts.invert;
        if is_match {
            if let Some(mc) = opts.max_count {
                if emitted >= mc {
                    break;
                }
            }
            if opts.before > 0 {
                for (n, t) in before.drain(..) {
                    group_sep(&mut out, &mut last_emitted, n, separators);
                    out.push_str(&format!("{display}-{n}-{}\n", clip(&t)));
                    last_emitted = Some(n);
                }
            } else {
                before.clear();
            }
            let records: Vec<String> = if opts.only_matching {
                re.find_iter(line).map(|m| m.as_str().to_string()).collect()
            } else {
                vec![line.to_string()]
            };
            for record in records {
                if emitted >= match_budget || out.len() >= out_budget {
                    hit_cap = true;
                    break;
                }
                group_sep(&mut out, &mut last_emitted, line_no, separators);
                out.push_str(&format!("{display}:{line_no}:{}\n", clip(&record)));
                last_emitted = Some(line_no);
                emitted += 1;
            }
            if hit_cap {
                break;
            }
            if !opts.only_matching {
                after_remaining = opts.after;
            }
        } else if after_remaining > 0 {
            if out.len() >= out_budget {
                hit_cap = true;
                break;
            }
            group_sep(&mut out, &mut last_emitted, line_no, separators);
            out.push_str(&format!("{display}-{line_no}-{}\n", clip(line)));
            last_emitted = Some(line_no);
            after_remaining -= 1;
        } else if opts.before > 0 {
            before.push_back((line_no, line.to_string()));
            if before.len() > opts.before {
                before.pop_front();
            }
        }
    }

    FileOutcome {
        output: out,
        emitted,
        hit_cap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup(name: &str) -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect(name);
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        (ws, dir)
    }

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    async fn run(ws: &Workspace, args: Value) -> Result<ToolOutput, ToolError> {
        SearchTool {
            max_output: 100_000,
        }
        .run(ws, &args, CancellationToken::new())
        .await
    }

    #[tokio::test]
    async fn finds_and_formats_matches() {
        let (ws, dir) = setup("basic");
        write(dir.path(), "a.txt", "hello\nworld\nhello again\n");
        let out = run(&ws, json!({"pattern": "hello"})).await.unwrap();
        assert_eq!(out.content, "a.txt:1:hello\na.txt:3:hello again");
    }

    #[tokio::test]
    async fn skips_hidden_and_gitignored() {
        let (ws, dir) = setup("hidden");
        write(dir.path(), "visible.txt", "needle\n");
        write(dir.path(), ".cache/hidden.txt", "needle\n");
        write(dir.path(), "target/built.txt", "needle\n");
        // `ignore` only applies .gitignore inside a git repo.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
        let out = run(&ws, json!({"pattern": "needle"})).await.unwrap();
        assert!(out.content.contains("visible.txt"), "{}", out.content);
        assert!(!out.content.contains("hidden.txt"), "{}", out.content);
        assert!(!out.content.contains("built.txt"), "{}", out.content);
    }

    #[tokio::test]
    async fn caps_results_with_a_note() {
        let (ws, dir) = setup("cap");
        write(dir.path(), "a.txt", &"x\n".repeat(50));
        let out = run(&ws, json!({"pattern": "^x$", "max_results": 5}))
            .await
            .unwrap();
        assert!(
            out.content.contains("truncated at 5 results"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn invalid_regex_is_an_argument_error() {
        let (ws, _dir) = setup("badre");
        let err = run(&ws, json!({"pattern": "("})).await.unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err:?}");
    }

    #[tokio::test]
    async fn fixed_strings_are_literal() {
        let (ws, dir) = setup("fixed");
        write(dir.path(), "a.txt", "a.c\nabc\n");
        let out = run(&ws, json!({"pattern": "a.c", "fixed_strings": true}))
            .await
            .unwrap();
        assert!(out.content.contains("a.c"));
        assert!(!out.content.contains("abc"));
    }

    #[tokio::test]
    async fn case_flags() {
        let (ws, dir) = setup("case");
        write(dir.path(), "a.txt", "Hello\n");
        let out = run(&ws, json!({"pattern": "hello", "ignore_case": true}))
            .await
            .unwrap();
        assert!(out.content.contains("Hello"));
        let out = run(&ws, json!({"pattern": "hello", "smart_case": true}))
            .await
            .unwrap();
        assert!(out.content.contains("Hello"));
        let out = run(&ws, json!({"pattern": "HELLO", "smart_case": true}))
            .await
            .unwrap();
        assert!(out.content.contains("no matches"), "{}", out.content);
    }

    #[tokio::test]
    async fn word_matches_whole_words_only() {
        let (ws, dir) = setup("word");
        write(dir.path(), "a.txt", "cat\ncatalog\n");
        let out = run(&ws, json!({"pattern": "cat", "word": true}))
            .await
            .unwrap();
        assert!(out.content.contains("a.txt:1:cat"));
        assert!(!out.content.contains("catalog"));
    }

    #[tokio::test]
    async fn invert_selects_non_matching_lines() {
        let (ws, dir) = setup("invert");
        write(dir.path(), "a.txt", "keep\ndrop\n");
        let out = run(&ws, json!({"pattern": "drop", "invert": true}))
            .await
            .unwrap();
        assert!(out.content.contains("a.txt:1:keep"));
        assert!(!out.content.contains("drop"));
    }

    #[tokio::test]
    async fn list_and_count_modes() {
        let (ws, dir) = setup("modes");
        write(dir.path(), "a.txt", "x\nx\n");
        write(dir.path(), "b.txt", "y\n");
        let out = run(&ws, json!({"pattern": "x", "files_with_matches": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt");
        let out = run(&ws, json!({"pattern": "x", "count": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt:2");
    }

    #[tokio::test]
    async fn context_lines_use_dash_separators() {
        let (ws, dir) = setup("ctx");
        write(dir.path(), "a.txt", "one\ntwo\nthree\nfour\n");
        let out = run(
            &ws,
            json!({"pattern": "three", "before_context": 1, "after_context": 1}),
        )
        .await
        .unwrap();
        assert!(out.content.contains("a.txt-2-two"), "{}", out.content);
        assert!(out.content.contains("a.txt:3:three"), "{}", out.content);
        assert!(out.content.contains("a.txt-4-four"), "{}", out.content);
    }

    #[tokio::test]
    async fn glob_and_type_filters() {
        let (ws, dir) = setup("filter");
        write(dir.path(), "a.rs", "needle\n");
        write(dir.path(), "b.txt", "needle\n");
        let out = run(&ws, json!({"pattern": "needle", "glob": ["*.rs"]}))
            .await
            .unwrap();
        assert!(out.content.contains("a.rs"));
        assert!(!out.content.contains("b.txt"));
        let out = run(&ws, json!({"pattern": "needle", "type": "rust"}))
            .await
            .unwrap();
        assert!(out.content.contains("a.rs"));
        assert!(!out.content.contains("b.txt"));
    }

    #[tokio::test]
    async fn binary_files_are_skipped() {
        let (ws, dir) = setup("bin");
        std::fs::write(dir.path().join("bin.dat"), b"needle\0\x01\x02").unwrap();
        write(dir.path(), "a.txt", "needle\n");
        let out = run(&ws, json!({"pattern": "needle"})).await.unwrap();
        assert!(out.content.contains("a.txt"));
        assert!(!out.content.contains("bin.dat"));
    }

    #[tokio::test]
    async fn cancellation_is_honored() {
        let (ws, dir) = setup("cancel");
        write(dir.path(), "a.txt", "x\n");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = SearchTool { max_output: 1000 }
            .run(&ws, &json!({"pattern": "x"}), cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }
}
