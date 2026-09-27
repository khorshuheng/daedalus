//! The `find` tool: file-path search by name/glob, mirroring `fd`.
//!
//! Traversal uses the same `ignore` crate walker as `grep`, so the defaults
//! skip hidden files/directories (`.git`, `target`, …) and respect
//! `.gitignore`/`.ignore`. `pattern` is a regex matched against the file name
//! by default (fd's convention); `full_path` matches it against the path
//! relative to the search root instead. `glob`/`type` filter which files are
//! considered at all, the same way they do in `grep`. Everything is bounded: a
//! total result cap, an output cap, and cooperative cancellation.
//!
//! Alternatively, `paths` supplies an explicit list of candidates (e.g. the
//! output of `git ls-files` or `git diff --name-only`). The list is filtered
//! by `kind`/`glob`/`type`/`pattern` directly, without walking the filesystem
//! and without hidden/ignore rules — the caller named these paths, so they are
//! wanted even inside `.git` or `node_modules`. This is the reason to reach
//! for the tool instead of piping into a shell `find`, whose walk would comb
//! those directories unfiltered.

use futures::future::BoxFuture;
use ignore::overrides::{Override, OverrideBuilder};
use ignore::types::{Types, TypesBuilder};
use ignore::{Match, WalkBuilder, WalkState};
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

use super::{arg_usize, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

/// Total paths returned unless `max_results` says otherwise.
const DEFAULT_MAX_RESULTS: usize = 100;
/// Hard ceiling on `max_results`.
const HARD_MAX_RESULTS: usize = 1_000;

pub struct FindTool {
    pub max_output: usize,
}

impl Tool for FindTool {
    fn name(&self) -> &'static str {
        "find"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex matched against the file name (fd's PATTERN); omit to match every entry." },
                "path": { "type": "string", "description": "Directory to search (default: the workspace root). Ignored when `paths` is set." },
                "paths": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ],
                    "description": "Explicit candidate paths to filter instead of walking (e.g. from `git ls-files`). Hidden/ignore rules do not apply. One path or a list."
                },
                "glob": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } }
                    ],
                    "description": "Include/exclude glob(s); a leading '!' excludes. One glob or a list, e.g. '*.rs'."
                },
                "type": { "type": "string", "description": "Only match files of this type, e.g. rust, py, js (fd/rg -t)." },
                "kind": { "type": "string", "description": "Entry kind to return: file (default), dir, symlink, or any (fd -t f/d/l)." },
                "full_path": { "type": "boolean", "description": "Match `pattern` against the path relative to the search root instead of just the file name (fd -p)." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive pattern match (fd -i)." },
                "smart_case": { "type": "boolean", "description": "Case-insensitive unless the pattern has an uppercase letter (fd -S)." },
                "fixed_strings": { "type": "boolean", "description": "Treat `pattern` as a literal string instead of regex (fd -F)." },
                "hidden": { "type": "boolean", "description": "Also match hidden files and directories (fd -H)." },
                "no_ignore": { "type": "boolean", "description": "Do not respect .gitignore/.ignore (fd -I)." },
                "follow": { "type": "boolean", "description": "Follow symbolic links (fd -L)." },
                "max_depth": { "type": "integer", "minimum": 1, "description": "Limit recursion to this many directory levels (fd -d)." },
                "sort": { "type": "string", "description": "Sort results by: none (default; path order), path, modified, accessed, or created. Sorting buffers the path list." },
                "sort_reverse": { "type": "boolean", "description": "Reverse the sort order." },
                "offset": { "type": "integer", "minimum": 0, "description": "Skip the first N results, for paging." },
                "max_results": { "type": "integer", "minimum": 1, "description": "Total result cap for this call (default 100, max 1000)." }
            }
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
                FindTool { max_output }.run_sync(&ws, &args, &cancel)
            })
            .await
            .unwrap_or_else(|e| Err(ToolError::Io(format!("blocking task failed: {e}"))))
        })
    }
}

fn get_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// `glob` accepts a single string or an array of strings (mirrors `grep`).
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

/// `paths` accepts a single string or an array of strings; absent means walk.
fn get_paths(args: &Value) -> Result<Option<Vec<String>>, ToolError> {
    match args.get("paths") {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(vec![s.clone()])),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ToolError::Argument("'paths' entries must be strings".into()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(ToolError::Argument(
            "'paths' must be a string or an array of strings".into(),
        )),
    }
}

/// Build the glob override matcher for `root`, or `None` when no globs were
/// given. Shared by the filesystem walk and the explicit-`paths` filter so the
/// two agree on include/exclude semantics.
fn build_overrides(root: &Path, globs: &[String]) -> Result<Option<Override>, ToolError> {
    if globs.is_empty() {
        return Ok(None);
    }
    let mut ob = OverrideBuilder::new(root);
    for g in globs {
        ob.add(g)
            .map_err(|e| ToolError::Argument(format!("bad glob '{g}': {e}")))?;
    }
    ob.build()
        .map(Some)
        .map_err(|e| ToolError::Argument(format!("bad globs: {e}")))
}

/// Build the file-type matcher for `t`, or fail with the same error the walk
/// path used to raise.
fn build_types(t: &str) -> Result<Types, ToolError> {
    let mut tb = TypesBuilder::new();
    tb.add_defaults();
    tb.select(t);
    tb.build()
        .map_err(|e| ToolError::Argument(format!("bad type '{t}': {e}")))
}

/// The per-path filter both the walker and the explicit-`paths` mode apply:
/// first the `kind` (file/dir/symlink/any), then the regex — against the file
/// name, or the root-relative path when `full_path` is set.
struct PathFilter<'a> {
    re: Option<&'a Regex>,
    full_path: bool,
    root: &'a Path,
    kind: &'a str,
}

impl PathFilter<'_> {
    fn matches(&self, path: &Path, file_type: std::fs::FileType) -> bool {
        let kind_ok = match self.kind {
            "file" => file_type.is_file(),
            "dir" => file_type.is_dir(),
            "symlink" => file_type.is_symlink(),
            _ => true,
        };
        if !kind_ok {
            return false;
        }
        let Some(re) = self.re else {
            return true;
        };
        let subject = if self.full_path {
            path.strip_prefix(self.root)
                .unwrap_or(path)
                .display()
                .to_string()
        } else {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        re.is_match(&subject)
    }
}

impl FindTool {
    fn run_sync(
        &self,
        workspace: &Workspace,
        args: &Value,
        cancel: &CancellationToken,
    ) -> Result<ToolOutput, ToolError> {
        let pattern = args.get("pattern").and_then(Value::as_str);
        let fixed = get_bool(args, "fixed_strings");
        let ignore_case = get_bool(args, "ignore_case");
        let smart_case = get_bool(args, "smart_case");
        let full_path = get_bool(args, "full_path");
        let hidden = get_bool(args, "hidden");
        let no_ignore = get_bool(args, "no_ignore");
        let follow = get_bool(args, "follow");
        let kind = args
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("file")
            .to_ascii_lowercase();
        let sort = args
            .get("sort")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_ascii_lowercase();
        let sort_reverse = get_bool(args, "sort_reverse");
        let offset = arg_usize(args, "offset")?.unwrap_or(0);
        let max_results = arg_usize(args, "max_results")?
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, HARD_MAX_RESULTS);
        let max_depth = arg_usize(args, "max_depth")?;

        let re = match pattern {
            Some(pattern) => {
                let mut source = pattern.to_string();
                if fixed {
                    source = regex::escape(&source);
                }
                let case_insensitive =
                    ignore_case || (smart_case && !pattern.chars().any(char::is_uppercase));
                Some(
                    RegexBuilder::new(&source)
                        .case_insensitive(case_insensitive)
                        .build()
                        .map_err(|e| {
                            ToolError::Argument(format!("invalid regex '{pattern}': {e}"))
                        })?,
                )
            }
            None => None,
        };

        if !matches!(kind.as_str(), "file" | "dir" | "symlink" | "any") {
            return Err(ToolError::Argument(format!(
                "unknown kind '{kind}' (file|dir|symlink|any)"
            )));
        }

        let explicit = get_paths(args)?;
        let root = match args.get("path").and_then(Value::as_str) {
            Some(p) if explicit.is_none() => resolve(workspace, Path::new(p))?,
            _ => workspace.root().to_path_buf(),
        };
        if explicit.is_none() && !root.exists() {
            return Err(ToolError::NotFound(
                args.get("path")
                    .and_then(Value::as_str)
                    .unwrap_or(".")
                    .to_string(),
            ));
        }

        let globs = get_globs(args)?;
        let overrides = build_overrides(&root, &globs)?;
        let types = match args.get("type").and_then(Value::as_str) {
            Some(t) => Some(build_types(t)?),
            None => None,
        };

        let filter = PathFilter {
            re: re.as_ref(),
            full_path,
            root: &root,
            kind: kind.as_str(),
        };

        let mut paths: Vec<PathBuf> = if let Some(list) = explicit {
            // Explicit candidate paths: filter the list in place. The walker's
            // hidden/ignore rules do not apply — the caller named these paths,
            // so they are wanted even under `.git` or `node_modules`.
            let mut out = Vec::new();
            for p in list {
                if cancel.is_cancelled() {
                    return Err(ToolError::Cancelled);
                }
                let path = resolve(workspace, Path::new(&p))?;
                let Ok(meta) = std::fs::symlink_metadata(&path) else {
                    continue;
                };
                let ft = meta.file_type();
                if let Some(ovr) = &overrides {
                    if matches!(ovr.matched(&path, ft.is_dir()), Match::Ignore(_)) {
                        continue;
                    }
                }
                if let Some(types) = &types {
                    if matches!(types.matched(&path, ft.is_dir()), Match::Ignore(_)) {
                        continue;
                    }
                }
                if filter.matches(&path, ft) {
                    out.push(path);
                }
            }
            out
        } else {
            // The `ignore` walker: defaults already skip hidden entries and
            // honor .gitignore/.ignore, exactly like `fd`.
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
            if let Some(depth) = max_depth {
                wb.max_depth(Some(depth));
            }
            if let Some(overrides) = overrides.clone() {
                wb.overrides(overrides);
            }
            if let Some(types) = types.clone() {
                wb.types(types);
            }

            // Parallel directory walk: filter by kind and pattern up front, so
            // only matching paths are ever buffered.
            let collected: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());
            let walker = wb.build_parallel();
            walker.run(|| {
                let collected = &collected;
                let filter = &filter;
                Box::new(move |result| {
                    if cancel.is_cancelled() {
                        return WalkState::Quit;
                    }
                    let Ok(entry) = result else {
                        return WalkState::Continue;
                    };
                    // The root entry itself (depth 0) is never a result.
                    if entry.depth() == 0 {
                        return WalkState::Continue;
                    }
                    let Some(file_type) = entry.file_type() else {
                        return WalkState::Continue;
                    };
                    if !filter.matches(entry.path(), file_type) {
                        return WalkState::Continue;
                    }
                    collected
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(entry.into_path());
                    WalkState::Continue
                })
            });
            collected.into_inner().unwrap_or_else(|e| e.into_inner())
        };
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        match sort.as_str() {
            // `none` sorts too, so the output does not depend on the parallel
            // walk order.
            "none" | "" | "path" => paths.sort(),
            "modified" | "accessed" | "created" => paths.sort_by_key(|p| {
                let meta = std::fs::symlink_metadata(p).ok();
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

        let total = paths.len();
        let mut out = String::new();
        let mut emitted = 0usize;
        let mut truncated = false;
        for path in paths.into_iter().skip(offset) {
            if emitted >= max_results || out.len() >= self.max_output {
                truncated = true;
                break;
            }
            let display = path
                .strip_prefix(workspace.root())
                .unwrap_or(&path)
                .display()
                .to_string();
            out.push_str(&display);
            out.push('\n');
            emitted += 1;
        }

        let mut content = out.trim_end().to_string();
        if content.is_empty() {
            return Ok(ToolOutput {
                content: "no matches".to_string(),
            });
        }
        if truncated || offset + emitted < total {
            content.push_str(&format!(
                "\n\n[truncated at {emitted} results; narrow the pattern, add a glob/type, or raise max_results]"
            ));
        }
        Ok(ToolOutput { content })
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

    fn write(root: &Path, rel: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, "x").unwrap();
    }

    async fn run(ws: &Workspace, args: Value) -> Result<ToolOutput, ToolError> {
        FindTool {
            max_output: 100_000,
        }
        .run(ws, &args, CancellationToken::new())
        .await
    }

    #[tokio::test]
    async fn lists_all_files_by_default() {
        let (ws, dir) = setup("all");
        write(dir.path(), "a.txt");
        write(dir.path(), "sub/b.txt");
        let out = run(&ws, json!({})).await.unwrap();
        assert_eq!(out.content, "a.txt\nsub/b.txt");
    }

    #[tokio::test]
    async fn pattern_matches_file_name_only_by_default() {
        let (ws, dir) = setup("name");
        write(dir.path(), "foo.rs");
        write(dir.path(), "foodir/bar.rs");
        let out = run(&ws, json!({"pattern": "^foo\\."})).await.unwrap();
        assert_eq!(out.content, "foo.rs");
    }

    #[tokio::test]
    async fn full_path_matches_the_relative_path() {
        let (ws, dir) = setup("fullpath");
        write(dir.path(), "sub/needle.txt");
        write(dir.path(), "other.txt");
        let out = run(&ws, json!({"pattern": "^sub/", "full_path": true}))
            .await
            .unwrap();
        assert_eq!(out.content, "sub/needle.txt");
    }

    #[tokio::test]
    async fn glob_filters_by_extension() {
        let (ws, dir) = setup("glob");
        write(dir.path(), "a.rs");
        write(dir.path(), "b.txt");
        let out = run(&ws, json!({"glob": "*.rs"})).await.unwrap();
        assert_eq!(out.content, "a.rs");
    }

    #[tokio::test]
    async fn type_filter_matches_rg_types() {
        let (ws, dir) = setup("type");
        write(dir.path(), "a.rs");
        write(dir.path(), "b.py");
        let out = run(&ws, json!({"type": "rust"})).await.unwrap();
        assert_eq!(out.content, "a.rs");
    }

    #[tokio::test]
    async fn kind_selects_directories() {
        let (ws, dir) = setup("kind");
        write(dir.path(), "sub/a.txt");
        let out = run(&ws, json!({"kind": "dir"})).await.unwrap();
        assert_eq!(out.content, "sub");
    }

    #[tokio::test]
    async fn unknown_kind_is_rejected() {
        let (ws, _dir) = setup("badkind");
        let err = run(&ws, json!({"kind": "socket"})).await.unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err:?}");
    }

    #[tokio::test]
    async fn skips_hidden_and_gitignored() {
        let (ws, dir) = setup("hidden");
        write(dir.path(), "visible.txt");
        write(dir.path(), ".cache/hidden.txt");
        write(dir.path(), "built/output.txt");
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "built/\n").unwrap();
        let out = run(&ws, json!({})).await.unwrap();
        assert!(out.content.contains("visible.txt"), "{}", out.content);
        assert!(!out.content.contains("hidden.txt"), "{}", out.content);
        assert!(!out.content.contains("output.txt"), "{}", out.content);
    }

    #[tokio::test]
    async fn hidden_flag_includes_dotfiles() {
        let (ws, dir) = setup("hiddenflag");
        write(dir.path(), ".env");
        let out = run(&ws, json!({"hidden": true})).await.unwrap();
        assert_eq!(out.content, ".env");
    }

    #[tokio::test]
    async fn caps_results_with_a_note() {
        let (ws, dir) = setup("cap");
        for i in 0..20 {
            write(dir.path(), &format!("f{i:02}.txt"));
        }
        let out = run(&ws, json!({"max_results": 5})).await.unwrap();
        assert!(
            out.content.contains("truncated at 5 results"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn offset_pages_results() {
        let (ws, dir) = setup("offset");
        write(dir.path(), "a.txt");
        write(dir.path(), "b.txt");
        write(dir.path(), "c.txt");
        // One more result (c.txt) remains beyond the page, so the truncation
        // note is expected alongside it.
        let out = run(&ws, json!({"offset": 1, "max_results": 1}))
            .await
            .unwrap();
        assert!(out.content.starts_with("b.txt"), "{}", out.content);
        assert!(!out.content.contains("a.txt"), "{}", out.content);
        assert!(!out.content.contains("c.txt"), "{}", out.content);
        assert!(
            out.content.contains("truncated at 1 results"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn sorts_reverse() {
        let (ws, dir) = setup("sort");
        write(dir.path(), "a.txt");
        write(dir.path(), "b.txt");
        let out = run(&ws, json!({"sort_reverse": true})).await.unwrap();
        assert_eq!(out.content, "b.txt\na.txt");
    }

    #[tokio::test]
    async fn max_depth_limits_recursion() {
        let (ws, dir) = setup("depth");
        write(dir.path(), "a.txt");
        write(dir.path(), "sub/b.txt");
        write(dir.path(), "sub/deeper/c.txt");
        let out = run(&ws, json!({"max_depth": 1})).await.unwrap();
        assert_eq!(out.content, "a.txt");
    }

    #[tokio::test]
    async fn invalid_regex_is_an_argument_error() {
        let (ws, _dir) = setup("badre");
        let err = run(&ws, json!({"pattern": "("})).await.unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err:?}");
    }

    #[tokio::test]
    async fn no_matches_message() {
        let (ws, _dir) = setup("nomatch");
        let out = run(&ws, json!({"pattern": "zzz-nope"})).await.unwrap();
        assert_eq!(out.content, "no matches");
    }

    #[tokio::test]
    async fn cancellation_is_honored() {
        let (ws, dir) = setup("cancel");
        write(dir.path(), "a.txt");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = FindTool { max_output: 1000 }
            .run(&ws, &json!({}), cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }

    #[tokio::test]
    async fn paths_filters_the_list_without_walking() {
        let (ws, dir) = setup("paths");
        write(dir.path(), "a.rs");
        write(dir.path(), "b.txt");
        // Not listed: must not appear even though it matches the pattern.
        write(dir.path(), "c.rs");
        let out = run(&ws, json!({"paths": ["a.rs", "b.txt"], "pattern": "\\.rs$"}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.rs");
    }

    #[tokio::test]
    async fn paths_ignores_hidden_and_gitignore_rules() {
        let (ws, dir) = setup("paths-hidden");
        write(dir.path(), ".git/objects/aa");
        write(dir.path(), "built/output.txt");
        std::fs::write(dir.path().join(".gitignore"), "built/\n").unwrap();
        let out = run(
            &ws,
            json!({"paths": [".git/objects/aa", "built/output.txt"]}),
        )
        .await
        .unwrap();
        assert_eq!(out.content, ".git/objects/aa\nbuilt/output.txt");
    }

    #[tokio::test]
    async fn paths_accepts_a_single_string() {
        let (ws, dir) = setup("paths-single");
        write(dir.path(), "a.txt");
        let out = run(&ws, json!({"paths": "a.txt"})).await.unwrap();
        assert_eq!(out.content, "a.txt");
    }

    #[tokio::test]
    async fn paths_applies_glob() {
        let (ws, dir) = setup("paths-glob");
        write(dir.path(), "a.rs");
        write(dir.path(), "b.txt");
        let out = run(&ws, json!({"paths": ["a.rs", "b.txt"], "glob": "*.rs"}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.rs");
    }

    #[tokio::test]
    async fn paths_skips_missing_entries() {
        let (ws, dir) = setup("paths-missing");
        write(dir.path(), "a.txt");
        let out = run(&ws, json!({"paths": ["a.txt", "nope.txt"]}))
            .await
            .unwrap();
        assert_eq!(out.content, "a.txt");
    }

    #[tokio::test]
    async fn paths_rejects_non_string_entries() {
        let (ws, _dir) = setup("paths-bad");
        let err = run(&ws, json!({"paths": [1, 2]})).await.unwrap_err();
        assert!(matches!(err, ToolError::Argument(_)), "{err:?}");
    }
}
