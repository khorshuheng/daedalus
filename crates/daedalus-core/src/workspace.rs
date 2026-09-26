//! Workspace scoping (CRAB-105).
//!
//! The workspace is the default root that relative `read`/`write`/`edit` paths
//! and `bash` commands resolve against. Absolute paths and `..` are allowed:
//! tools can reach anywhere the user can (as `bash` always could), so the
//! workspace is a default location, not a security boundary.

use std::path::{Path, PathBuf};

/// A workspace rooted at a single canonical directory.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf, // canonicalized absolute path
}

/// Normalize common path confusables that models sometimes emit — Unicode
/// spaces, a leading `@`, and a leading `~` — so e.g. `foo\u{00A0}bar`
/// resolves to `foo bar` and `~/notes.md` resolves under the home directory
/// (mirrors pi's `normalizePath`/`resolveToCwd`). `~user` is not expanded.
fn normalize_path(rel: &Path, home: Option<&Path>) -> PathBuf {
    let mut s = rel.to_string_lossy().to_string();
    if let Some(stripped) = s.strip_prefix('@') {
        s = stripped.to_string();
    }
    let normalized: String = s
        .chars()
        .map(|c| match c {
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect();
    PathBuf::from(expand_tilde(&normalized, home))
}

/// Expand a leading `~` or `~/…` to `home` (CRAB-148). `~user` and interior
/// `~` are left untouched; when no home directory is known the path is returned
/// unchanged, so a missing home never turns a relative path absolute.
fn expand_tilde(s: &str, home: Option<&Path>) -> String {
    let Some(home) = home else {
        return s.to_string();
    };
    if s == "~" || s == "~/" {
        return home.to_string_lossy().to_string();
    }
    match s.strip_prefix("~/") {
        Some(rest) => home.join(rest).to_string_lossy().to_string(),
        None => s.to_string(),
    }
}

/// The user's home directory, if it can be determined (CRAB-148). `BaseDirs`
/// handles the platform conventions; `$HOME` is the Unix fallback.
fn home_dir() -> Option<PathBuf> {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().to_path_buf())
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
}

impl Workspace {
    /// Construct a workspace from `root`, canonicalizing it and verifying it is
    /// a directory.
    pub fn new(root: PathBuf) -> Result<Self, String> {
        let canon = root
            .canonicalize()
            .map_err(|e| format!("workspace '{}' cannot be resolved: {e}", root.display()))?;
        if !canon.is_dir() {
            return Err(format!(
                "workspace '{}' is not a directory",
                canon.display()
            ));
        }
        Ok(Self { root: canon })
    }

    /// The canonical workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve `rel` to a path: relative paths join the workspace root, while
    /// absolute paths and `..` are honored as given.
    ///
    /// For paths that do not yet exist (e.g. a `write` target), the deepest
    /// existing ancestor is canonicalized (resolving symlinks) and the
    /// non-existing remainder is re-appended.
    pub fn resolve(&self, rel: &Path) -> Result<PathBuf, String> {
        self.resolve_with_home(rel, home_dir().as_deref())
    }

    /// Like [`resolve`](Self::resolve) but with an explicit home directory, so
    /// the `~` expansion is testable without touching the process `$HOME`
    /// (CRAB-148).
    fn resolve_with_home(&self, rel: &Path, home: Option<&Path>) -> Result<PathBuf, String> {
        let normalized = normalize_path(rel, home);
        let joined = if normalized.is_absolute() {
            normalized
        } else {
            self.root.join(&normalized)
        };

        // Find the deepest existing ancestor and canonicalize it (resolving
        // symlinks), then re-append the non-existing remainder.
        let mut existing = joined.clone();
        let mut suffix: Vec<PathBuf> = Vec::new();
        while !existing.exists() {
            match (existing.parent(), existing.file_name()) {
                (Some(p), Some(name)) if p != existing => {
                    suffix.push(PathBuf::from(name));
                    existing = p.to_path_buf();
                }
                _ => break,
            }
        }
        let canon_existing = existing
            .canonicalize()
            .map_err(|e| format!("cannot resolve '{}': {e}", rel.display()))?;

        suffix.reverse();
        let mut out = canon_existing;
        for s in suffix {
            out.push(s);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    /// A unique temp dir cleaned up on drop (tempfile, CRAB-119).
    fn tempdir(_name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let base = dir.path().to_path_buf();
        (dir, base)
    }

    #[test]
    fn resolves_relative_paths() {
        let (_guard, dir) = tempdir("rel");
        std::fs::write(dir.join("a.txt"), "hi").unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        let p = ws.resolve(Path::new("a.txt")).unwrap();
        assert_eq!(p, dir.join("a.txt"));
    }

    #[test]
    fn allows_dotdot_outside_the_workspace() {
        let (_guard, dir) = tempdir("dotdot");
        let ws = Workspace::new(dir.clone()).unwrap();
        // `..` is honored: the workspace's parent resolves to a real path.
        let p = ws.resolve(Path::new("../")).unwrap();
        assert_eq!(p, dir.parent().unwrap().canonicalize().unwrap());
    }

    #[test]
    fn allows_absolute_outside_the_workspace() {
        let (_guard, dir) = tempdir("abs");
        let ws = Workspace::new(dir.clone()).unwrap();
        let p = ws.resolve(Path::new("/etc")).unwrap();
        assert_eq!(p, Path::new("/etc").canonicalize().unwrap());
    }

    #[test]
    fn follows_symlinks_outside_the_workspace() {
        let (_guard, dir) = tempdir("symlink");
        let outside = std::env::temp_dir().join(format!("daedalus-outside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, dir.join("link")).unwrap();

        let ws = Workspace::new(dir.clone()).unwrap();
        let p = ws.resolve(Path::new("link/secret.txt")).unwrap();
        assert_eq!(p, outside.canonicalize().unwrap().join("secret.txt"));
        std::fs::remove_dir_all(&outside).ok();
    }

    #[test]
    fn resolves_new_file_under_root() {
        let (_guard, dir) = tempdir("newfile");
        let ws = Workspace::new(dir.clone()).unwrap();
        let p = ws.resolve(Path::new("sub/deep/new.txt")).unwrap();
        assert_eq!(p, dir.join("sub/deep/new.txt"));
        assert!(p.starts_with(&dir));
    }

    #[test]
    fn normalizes_unicode_spaces_in_paths() {
        let (_guard, dir) = tempdir("nbsp");
        std::fs::write(dir.join("a b.txt"), "x").unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        // A non-breaking space in the filename is treated as a regular space.
        let p = ws.resolve(Path::new("a\u{00A0}b.txt")).unwrap();
        assert_eq!(p, dir.join("a b.txt"));
    }

    /// CRAB-148: a leading `~`/`~/` expands to the home directory; `~user` and
    /// interior `~` are left alone, and an unknown home leaves the path as-is.
    #[test]
    fn expands_leading_tilde_only() {
        let home = Path::new("/home/u");
        assert_eq!(expand_tilde("~", Some(home)), "/home/u");
        assert_eq!(expand_tilde("~/", Some(home)), "/home/u");
        assert_eq!(expand_tilde("~/a/b", Some(home)), "/home/u/a/b");
        assert_eq!(expand_tilde("~user/x", Some(home)), "~user/x");
        assert_eq!(expand_tilde("a/~/b", Some(home)), "a/~/b");
        assert_eq!(expand_tilde("~/a", None), "~/a");
    }

    /// CRAB-148: resolution honors the expanded home and canonicalizes it.
    #[test]
    fn expands_tilde_to_home_when_resolving() {
        let (_guard, dir) = tempdir("tilde");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join("a.txt"), "x").unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        let canon_home = home.canonicalize().unwrap();

        assert_eq!(
            ws.resolve_with_home(Path::new("~/a.txt"), Some(&home))
                .unwrap(),
            canon_home.join("a.txt")
        );
        assert_eq!(
            ws.resolve_with_home(Path::new("~"), Some(&home)).unwrap(),
            canon_home
        );
        // Without a known home, `~` stays a literal component under the root.
        assert_eq!(
            ws.resolve_with_home(Path::new("~/a.txt"), None).unwrap(),
            dir.canonicalize().unwrap().join("~").join("a.txt")
        );
    }
}
