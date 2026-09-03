//! Workspace scoping (CRAB-105).
//!
//! The workspace is the single root that all `read`/`write`/`edit` paths and
//! `bash` commands resolve against. `resolve` rejects any path that escapes the
//! workspace after resolving `..` and symlinks (the canonical path must remain
//! under the canonical workspace root).

use std::path::{Path, PathBuf};

/// A workspace rooted at a single canonical directory.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf, // canonicalized absolute path
}

/// Normalize common path confusables that models sometimes emit — Unicode
/// spaces and a leading `@` — so e.g. `foo\u{00A0}bar` resolves to `foo bar`
/// (mirrors pi's `normalizeToolPath`).
fn normalize_path(rel: &Path) -> PathBuf {
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
    PathBuf::from(normalized)
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

    /// Resolve `rel` (absolute or relative) to a path inside the workspace,
    /// rejecting anything that escapes the root via `..` or symlinks.
    ///
    /// For paths that do not yet exist (e.g. a `write` target), the deepest
    /// existing ancestor is canonicalized and checked, then the non-existing
    /// remainder is re-appended.
    pub fn resolve(&self, rel: &Path) -> Result<PathBuf, String> {
        let normalized = normalize_path(rel);
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

        if !canon_existing.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes the workspace '{}'",
                rel.display(),
                self.root.display()
            ));
        }

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
    fn rejects_dotdot_escape() {
        let (_guard, dir) = tempdir("dotdot");
        std::fs::write(dir.join("a.txt"), "hi").unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        assert!(ws.resolve(Path::new("../a.txt")).is_err());
        assert!(ws.resolve(Path::new("../../etc/passwd")).is_err());
    }

    #[test]
    fn rejects_absolute_outside() {
        let (_guard, dir) = tempdir("abs");
        let ws = Workspace::new(dir.clone()).unwrap();
        assert!(ws.resolve(Path::new("/etc/passwd")).is_err());
    }

    #[test]
    fn rejects_symlink_escape() {
        let (_guard, dir) = tempdir("symlink");
        let outside = std::env::temp_dir().join(format!("crab-outside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, dir.join("link")).unwrap();

        let ws = Workspace::new(dir.clone()).unwrap();
        let err = ws.resolve(Path::new("link/secret.txt")).unwrap_err();
        assert!(err.contains("escapes the workspace"));
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
}
