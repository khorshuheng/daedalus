//! The `grep` tool: ripgrep-style content search over the
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
//! -A/-B/-m/-g/-t/-I/-N/-o/-r/--hidden/--no-ignore/-L`), plus `offset`/`unique`/
//! `total_count` for the `rg | sed -n 'A,Bp'`, `rg | sort -u` and `rg | wc -l`
//! idioms.

use futures::future::BoxFuture;
use ignore::overrides::OverrideBuilder;
use ignore::types::TypesBuilder;
use ignore::{WalkBuilder, WalkState};
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::path::Path;
use tokio_util::sync::CancellationToken;

use super::{arg_string, arg_usize, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// Total matches returned unless `max_results` says otherwise.
const DEFAULT_MAX_RESULTS: usize = 100;
/// Hard ceiling on `max_results`.
const HARD_MAX_RESULTS: usize = 1_000;
/// Default clip for a matched/context line (`max_columns` overrides it).
const DEFAULT_MAX_COLUMNS: usize = 200;
/// Files larger than this are skipped (matches `read`'s whole-read threshold).
const MAX_SEARCH_FILE: u64 = 4 * 1024 * 1024;
/// Total bytes read across all files before the search stops.
const MAX_SCAN_BYTES: u64 = 256 * 1024 * 1024;

pub struct GrepTool {
    pub max_output: usize,
}

impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex to search for (Rust regex syntax)." },
                "path": { "type": "string", "description": "File or directory to search (default: the workspace root)." },
                "glob": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ],
                    "description": "Include/exclude glob(s); a leading '!' excludes (rg -g). One glob or a list."
                },
                "type": { "type": "string", "description": "Only search files of this type, e.g. rust, py, js (rg -t)." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive (rg -i)." },
                "smart_case": { "type": "boolean", "description": "Case-insensitive unless the pattern has an uppercase letter (rg -S)." },
                "fixed_strings": { "type": "boolean", "description": "Treat the pattern as a literal string (rg -F)." },
                "word": { "type": "boolean", "description": "Match whole words only (rg -w)." },
                "invert": { "type": "boolean", "description": "Select lines that do NOT match (rg -v)." },
                "only_matching": { "type": "boolean", "description": "Print only the matched text, one per occurrence (rg -o)." },
                "replace": { "type": "string", "description": "Replace matches in the output with this template, using $1/${name} capture refs (rg -r)." },
                "files_with_matches": { "type": "boolean", "description": "Print only paths with at least one match (rg -l)." },
                "count": { "type": "boolean", "description": "Print a per-file match count (rg -c)." },
                "total_count": { "type": "boolean", "description": "Print only the total number of matches, like `rg | wc -l`." },
                "context": { "type": "integer", "minimum": 0, "description": "Lines of context before and after a match (rg -C)." },
                "after_context": { "type": "integer", "minimum": 0, "description": "Lines of context after a match (rg -A)." },
                "before_context": { "type": "integer", "minimum": 0, "description": "Lines of context before a match (rg -B)." },
                "max_count": { "type": "integer", "minimum": 1, "description": "Stop after this many matches per file (rg -m)." },
                "offset": { "type": "integer", "minimum": 0, "description": "Skip the first N result records before returning any, for paging (`rg | sed -n 'A,Bp'`)." },
                "unique": { "type": "boolean", "description": "Drop duplicate result lines, like `rg | sort -u`." },
                "no_filename": { "type": "boolean", "description": "Omit the file path prefix (rg -I)." },
                "no_line_number": { "type": "boolean", "description": "Omit line numbers (rg -N)." },
                "sort": { "type": "string", "description": "Sort results by: none (default), path, modified, accessed, or created (rg --sort). Sorting buffers the file list." },
                "sort_reverse": { "type": "boolean", "description": "Reverse the sort order (rg --sortr)." },
                "max_columns": { "type": "integer", "minimum": 1, "description": "Clip each displayed line to this many characters (rg -M; default 200)." },
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
                GrepTool { max_output }.run_sync(&ws, &args, &cancel)
            })
            .await
            .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

/// Options that shape the output.
struct Opts {
    invert: bool,
    files_with_matches: bool,
    count: bool,
    only_matching: bool,
    /// True when a whole-buffer `is_match` prefilter is sound: the pattern has
    /// no `^`/`$` anchors, whose per-line meaning differs from whole-text.
    prefilter: bool,
    before: usize,
    after: usize,
    max_count: Option<usize>,
    offset: usize,
    replace: Option<String>,
    no_filename: bool,
    no_line_number: bool,
    max_columns: usize,
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
fn clip(s: &str, max_columns: usize) -> String {
    let mut out: String = s.chars().take(max_columns).collect();
    if s.chars().count() > max_columns {
        out.push('…');
    }
    out
}

/// Prefix for a result line, honoring `-I`/`-N`. `:` separates a match, `-` a
/// context line (rg's convention).
fn prefix(display: &str, line_no: usize, opts: &Opts, is_context: bool) -> String {
    let sep = if is_context { '-' } else { ':' };
    match (opts.no_filename, opts.no_line_number) {
        (false, false) => format!("{display}{sep}{line_no}{sep}"),
        (false, true) => format!("{display}{sep}"),
        (true, false) => format!("{line_no}{sep}"),
        (true, true) => String::new(),
    }
}

/// rg's `--` separator between non-contiguous groups, only with context.
fn maybe_sep(out: &mut String, last: &mut Option<usize>, next: usize, enabled: bool) {
    if !enabled {
        return;
    }
    if let Some(prev) = *last {
        if next > prev + 1 {
            out.push_str("--\n");
        }
    }
}

/// Matching lines (or occurrences) in one file, used for `-c` and
/// `total_count`.
fn count_matches(
    re: &Regex,
    text: &str,
    invert: bool,
    only_matching: bool,
    max: Option<usize>,
    prefilter: bool,
) -> usize {
    // Cheap whole-buffer reject before touching individual lines. Only sound
    // for non-invert matching: an invert search matches every line when the
    // pattern is absent.
    if prefilter && !invert && !re.is_match(text) {
        return 0;
    }
    let mut n = 0usize;
    for line in text.lines() {
        if re.is_match(line) != invert {
            n += 1;
            if let Some(m) = max {
                if n >= m {
                    break;
                }
            }
        }
    }
    if only_matching && !invert {
        let mut occ = 0usize;
        for line in text.lines() {
            occ += re.find_iter(line).count();
            if let Some(m) = max {
                if occ >= m {
                    return occ;
                }
            }
        }
        return occ;
    }
    n
}

impl GrepTool {
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
        let total_count = get_bool(args, "total_count");
        let unique = get_bool(args, "unique");
        let sort = args
            .get("sort")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_ascii_lowercase();
        let sort_reverse = get_bool(args, "sort_reverse");

        let context = arg_usize(args, "context")?.unwrap_or(0);
        let mut opts = Opts {
            invert: get_bool(args, "invert"),
            files_with_matches: get_bool(args, "files_with_matches"),
            count: get_bool(args, "count"),
            only_matching: get_bool(args, "only_matching"),
            // Filled in below once the pattern source is known.
            prefilter: false,
            before: arg_usize(args, "before_context")?.unwrap_or(context),
            after: arg_usize(args, "after_context")?.unwrap_or(context),
            max_count: arg_usize(args, "max_count")?,
            offset: arg_usize(args, "offset")?.unwrap_or(0),
            replace: args
                .get("replace")
                .and_then(Value::as_str)
                .map(str::to_string),
            no_filename: get_bool(args, "no_filename"),
            no_line_number: get_bool(args, "no_line_number"),
            max_columns: arg_usize(args, "max_columns")?
                .unwrap_or(DEFAULT_MAX_COLUMNS)
                .clamp(1, 10_000),
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
        // A whole-buffer `is_match` rejects a non-matching file in one pass
        // (regex literal optimizations) instead of a regex call per line. It is
        // only equivalent for patterns without anchors, whose `^`/`$` would
        // otherwise mean "start/end of the whole buffer".
        opts.prefilter = !source.contains('^') && !source.contains('$');

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
        let mut seen = 0usize; // records seen (pre-offset), across files
        let mut emitted = 0usize; // records emitted, across files
        let mut total = 0usize; // total_count accumulator
        let mut scanned = 0u64;
        let mut skipped_binary = 0usize;
        let mut skipped_large = 0usize;
        let mut truncated = false;

        // Parallel directory walk: collect the candidate file paths (respecting
        // every ignore/glob/type setting), then process them in a deterministic
        // order. Only paths are buffered, so memory is bounded by the file
        // count, not content.
        let collected: std::sync::Mutex<Vec<std::path::PathBuf>> =
            std::sync::Mutex::new(Vec::new());
        let walker = wb.build_parallel();
        walker.run(|| {
            let collected = &collected;
            Box::new(move |result| {
                if cancel.is_cancelled() {
                    return WalkState::Quit;
                }
                if let Ok(entry) = result {
                    if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                        collected
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(entry.into_path());
                    }
                }
                WalkState::Continue
            })
        });
        let mut paths = collected.into_inner().unwrap_or_else(|e| e.into_inner());
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }
        match sort.as_str() {
            // `none` sorts too, so the output does not depend on the parallel
            // walk order.
            "none" | "" | "path" => paths.sort(),
            "modified" | "accessed" | "created" => paths.sort_by_key(|p| {
                let meta = std::fs::metadata(p).ok();
                match sort.as_str() {
                    "modified" => meta.and_then(|m| m.modified().ok()),
                    "accessed" => meta.and_then(|m| m.accessed().ok()),
                    _ => meta.and_then(|m| m.created().ok()),
                }
            }),
            other => {
                return Err(ToolError::Argument(format!(
                    "unknown sort '{other}' (none|path|modified|accessed|created)"
                )))
            }
        }
        if sort_reverse {
            paths.reverse();
        }

        'walk: for path in paths {
            if cancel.is_cancelled() {
                return Err(ToolError::Cancelled);
            }
            let meta = match std::fs::metadata(&path) {
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
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            if bytes.iter().take(8192).any(|&b| b == 0) {
                skipped_binary += 1;
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);

            let display = path
                .strip_prefix(workspace.root())
                .unwrap_or(&path)
                .display()
                .to_string();

            if total_count {
                total += count_matches(
                    &re,
                    &text,
                    opts.invert,
                    opts.only_matching,
                    opts.max_count,
                    opts.prefilter,
                );
                continue;
            }

            let match_budget = max_results.saturating_sub(emitted);
            let out_budget = self.max_output.saturating_sub(out.len());
            let (file_out, hit_cap) = render_file(
                &re,
                &display,
                &text,
                &opts,
                &mut seen,
                &mut emitted,
                match_budget,
                out_budget,
            );
            out.push_str(&file_out);
            // `hit_cap` is set only when a further result existed but was not
            // emitted, so the truncation note is accurate.
            if hit_cap {
                truncated = true;
                break 'walk;
            }
        }

        if total_count {
            return Ok(ToolOutput {
                content: total.to_string(),
            });
        }

        let mut content = out.trim_end().to_string();
        if unique {
            let mut deduped = String::new();
            let mut seen_lines: HashSet<&str> = HashSet::new();
            for line in content.lines() {
                if line == "--" {
                    continue;
                }
                if seen_lines.insert(line) {
                    deduped.push_str(line);
                    deduped.push('\n');
                }
            }
            content = deduped.trim_end().to_string();
        }
        if content.is_empty() {
            return Ok(ToolOutput {
                content: format!("no matches for '{pattern}'"),
            });
        }
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

/// Render one file. `seen`/`emitted` are global counters; `match_budget`/`out_budget`
/// bound this call. Returns the file's output and whether a cap was hit.
#[allow(clippy::too_many_arguments)]
fn render_file(
    re: &Regex,
    display: &str,
    text: &str,
    opts: &Opts,
    seen: &mut usize,
    emitted: &mut usize,
    match_budget: usize,
    out_budget: usize,
) -> (String, bool) {
    let mut out = String::new();

    // Whole-buffer reject before the per-line pass: files with no match at all
    // are the common case in a repo-wide search.
    if opts.prefilter && !opts.invert && !re.is_match(text) {
        return (out, false);
    }

    // `-l` and `-c` emit one record per matched file.
    if opts.files_with_matches || opts.count {
        let n = count_matches(re, text, opts.invert, false, None, opts.prefilter);
        if n == 0 {
            return (out, false);
        }
        let idx = *seen;
        *seen += 1;
        if idx >= opts.offset {
            if *emitted >= match_budget || out.len() >= out_budget {
                return (out, true);
            }
            if opts.files_with_matches {
                out.push_str(display);
            } else {
                out.push_str(&format!("{display}:{n}"));
            }
            out.push('\n');
            *emitted += 1;
        }
        return (out, false);
    }

    let mut before: VecDeque<(usize, String)> = VecDeque::new();
    let mut after_remaining = 0usize;
    let mut last_emitted: Option<usize> = None;
    let mut file_matches = 0usize;
    let separators = opts.before > 0 || opts.after > 0;

    for (i, line) in text.lines().enumerate() {
        let line_no = i + 1;
        let is_match = re.is_match(line) != opts.invert;
        if is_match {
            file_matches += 1;
            if let Some(mc) = opts.max_count {
                if file_matches > mc {
                    break;
                }
            }
            // Records for this line: one per occurrence in `-o` mode, else one.
            let records: Vec<String> = if opts.only_matching {
                re.find_iter(line)
                    .map(|m| match &opts.replace {
                        Some(r) => re.replace_all(m.as_str(), r.as_str()).to_string(),
                        None => m.as_str().to_string(),
                    })
                    .collect()
            } else {
                vec![match &opts.replace {
                    Some(r) => re.replace_all(line, r.as_str()).to_string(),
                    None => line.to_string(),
                }]
            };
            let base = *seen;
            *seen += records.len();
            let mut emitted_any = false;
            for (j, record) in records.iter().enumerate() {
                if base + j < opts.offset {
                    continue;
                }
                if !emitted_any {
                    if opts.before > 0 {
                        for (n, t) in before.drain(..) {
                            if *emitted >= match_budget || out.len() >= out_budget {
                                return (out, true);
                            }
                            maybe_sep(&mut out, &mut last_emitted, n, separators);
                            out.push_str(&prefix(display, n, opts, true));
                            out.push_str(&clip(&t, opts.max_columns));
                            out.push('\n');
                            last_emitted = Some(n);
                            *emitted += 1;
                        }
                    } else {
                        before.clear();
                    }
                    emitted_any = true;
                }
                if *emitted >= match_budget || out.len() >= out_budget {
                    return (out, true);
                }
                maybe_sep(&mut out, &mut last_emitted, line_no, separators);
                out.push_str(&prefix(display, line_no, opts, false));
                out.push_str(&clip(record, opts.max_columns));
                out.push('\n');
                last_emitted = Some(line_no);
                *emitted += 1;
            }
            if emitted_any {
                after_remaining = opts.after;
            }
        } else if after_remaining > 0 {
            if out.len() >= out_budget {
                return (out, true);
            }
            maybe_sep(&mut out, &mut last_emitted, line_no, separators);
            out.push_str(&prefix(display, line_no, opts, true));
            out.push_str(&clip(line, opts.max_columns));
            out.push('\n');
            last_emitted = Some(line_no);
            *emitted += 1;
            after_remaining -= 1;
        } else if opts.before > 0 {
            before.push_back((line_no, line.to_string()));
            if before.len() > opts.before {
                before.pop_front();
            }
        }
    }

    (out, false)
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
        GrepTool {
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

    /// The parallel walk must yield a deterministic (path-sorted) order even
    /// when no `sort` is requested.
    #[tokio::test]
    async fn default_order_is_deterministic_across_files() {
        let (ws, dir) = setup("deterministic");
        write(dir.path(), "b.txt", "x\n");
        write(dir.path(), "a.txt", "x\n");
        write(dir.path(), "c.txt", "x\n");
        let out = run(&ws, json!({"pattern": "x"})).await.unwrap();
        assert_eq!(out.content, "a.txt:1:x\nb.txt:1:x\nc.txt:1:x");
    }

    /// The whole-buffer prefilter must not reject a file whose `^`-anchored
    /// pattern matches a later line (DAE-116).
    #[tokio::test]
    async fn anchored_patterns_still_match_later_lines() {
        let (ws, dir) = setup("anchor");
        write(dir.path(), "a.txt", "bar\nfoo\n");
        let out = run(&ws, json!({"pattern": "^foo"})).await.unwrap();
        assert_eq!(out.content, "a.txt:2:foo");
    }

    /// Invert must not use the "no match at all" short-circuit: every line is a
    /// match when the pattern is absent.
    #[tokio::test]
    async fn invert_still_returns_lines_without_the_pattern() {
        let (ws, dir) = setup("invert-prefilter");
        write(dir.path(), "a.txt", "one\ntwo\n");
        let out = run(&ws, json!({"pattern": "absent", "invert": true}))
            .await
            .unwrap();
        assert!(out.content.contains("a.txt:1:one"), "{}", out.content);
        assert!(out.content.contains("a.txt:2:two"), "{}", out.content);
    }

    #[tokio::test]
    async fn skips_hidden_and_gitignored() {
        let (ws, dir) = setup("hidden");
        write(dir.path(), "visible.txt", "needle\n");
        write(dir.path(), ".cache/hidden.txt", "needle\n");
        write(dir.path(), "target/built.txt", "needle\n");
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
    async fn offset_pages_results() {
        let (ws, dir) = setup("offset");
        write(dir.path(), "a.txt", "m1\nm2\nm3\nm4\nm5\n");
        let out = run(&ws, json!({"pattern": "^m", "offset": 1, "max_results": 2}))
            .await
            .unwrap();
        // The two requested records first, then the (accurate) truncation note.
        assert!(
            out.content.starts_with("a.txt:2:m2\na.txt:3:m3"),
            "{}",
            out.content
        );
        assert!(!out.content.contains("m1"), "{}", out.content);
        assert!(!out.content.contains("m4"), "{}", out.content);
    }

    #[tokio::test]
    async fn replace_rewrites_matches() {
        let (ws, dir) = setup("replace");
        write(dir.path(), "a.txt", "foo=1\nbar=2\n");
        let out = run(
            &ws,
            json!({"pattern": r"(\w+)=(\d+)", "replace": "$2 -> $1"}),
        )
        .await
        .unwrap();
        assert!(out.content.contains("a.txt:1:1 -> foo"), "{}", out.content);
        assert!(out.content.contains("a.txt:2:2 -> bar"), "{}", out.content);
    }

    #[tokio::test]
    async fn no_filename_and_no_line_number() {
        let (ws, dir) = setup("plain");
        write(dir.path(), "a.txt", "match here\n");
        let out = run(
            &ws,
            json!({"pattern": "here", "no_filename": true, "no_line_number": true}),
        )
        .await
        .unwrap();
        assert_eq!(out.content, "match here");
        let out = run(&ws, json!({"pattern": "here", "no_line_number": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt:match here");
    }

    #[tokio::test]
    async fn unique_dedupes_occurrences() {
        let (ws, dir) = setup("unique");
        write(dir.path(), "a.txt", "foo\nfoo\nbar\n");
        let out = run(&ws, json!({"pattern": "foo", "only_matching": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt:1:foo\na.txt:2:foo");
        let out = run(
            &ws,
            json!({"pattern": "foo", "only_matching": true, "no_filename": true, "no_line_number": true, "unique": true}),
        )
        .await
        .unwrap();
        assert_eq!(out.content, "foo");
    }

    #[tokio::test]
    async fn total_count_returns_a_number() {
        let (ws, dir) = setup("total");
        write(dir.path(), "a.txt", "x\nx\n");
        write(dir.path(), "b.txt", "x\n");
        let out = run(&ws, json!({"pattern": "x", "total_count": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "3");
    }

    #[tokio::test]
    async fn max_columns_clips_long_lines() {
        let (ws, dir) = setup("cols");
        write(
            dir.path(),
            "a.txt",
            &format!("needle {}\n", "z".repeat(500)),
        );
        let out = run(&ws, json!({"pattern": "needle", "max_columns": 20}))
            .await
            .unwrap();
        let line = out.content.lines().next().unwrap();
        // 20 chars of content plus the "…" marker.
        let text = line.splitn(3, ':').nth(2).unwrap();
        assert!(text.ends_with('…'), "{line}");
        assert_eq!(text.chars().count(), 21, "{line}");
    }

    #[tokio::test]
    async fn sorts_by_path_and_reverse() {
        let (ws, dir) = setup("sort");
        write(dir.path(), "b.txt", "x\n");
        write(dir.path(), "a.txt", "x\n");
        write(dir.path(), "c.txt", "x\n");
        let out = run(&ws, json!({"pattern": "x", "sort": "path"}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt:1:x\nb.txt:1:x\nc.txt:1:x");
        let out = run(
            &ws,
            json!({"pattern": "x", "sort": "path", "sort_reverse": true}),
        )
        .await
        .unwrap();
        assert_eq!(out.content, "c.txt:1:x\nb.txt:1:x\na.txt:1:x");
    }

    #[tokio::test]
    async fn unknown_sort_is_rejected() {
        let (ws, dir) = setup("badsort");
        write(dir.path(), "a.txt", "x\n");
        let err = run(&ws, json!({"pattern": "x", "sort": "sideways"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err:?}");
    }

    #[tokio::test]
    async fn cancellation_is_honored() {
        let (ws, dir) = setup("cancel");
        write(dir.path(), "a.txt", "x\n");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = GrepTool { max_output: 1000 }
            .run(&ws, &json!({"pattern": "x"}), cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }
}
