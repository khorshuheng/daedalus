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
        let joined = if rel.is_absolute() {
            rel.to_path_buf()
        } else {
            self.root.join(rel)
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

    /// RAII guard that removes its dir on drop.
    struct Dir(std::path::PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir(name: &str) -> (Dir, PathBuf) {
        let base = std::env::temp_dir().join(format!("crab-ws-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        (Dir(base.clone()), base)
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
}
