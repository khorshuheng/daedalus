//! User instructions: a standing, always-on supplement to the system prompt.
//!
//! Skills already let a user extend the agent's instructions without
//! touching code, but a skill is **load-on-demand**: the model only sees it
//! after a `/skill <name>`. Some instructions should apply to every session
//! without being asked for — house style, tool preferences, a standing "never
//! touch X". This module reads those from a file:
//!
//! - `~/.config/daedalus/APPEND_SYSTEM.md` (the config dir from [`paths`])
//!
//! The contents are appended verbatim to the system prompt, after the built-in
//! prompt and before the skills catalog. The file is re-read on every turn (the
//! seed prompt is rebuilt in `push_user`), so edits take effect without a
//! restart. A missing, unreadable, or blank file contributes nothing.
//!
//! The name matches pi's `APPEND_SYSTEM.md`, so the muscle memory carries over.
//! It is deliberately user-level only: a workspace file would be project
//! content, which belongs to the (unimplemented) project-context work rather
//! than to a user's standing config (see `docs/open-items.md`).

use std::path::{Path, PathBuf};

use crate::paths;

/// The instructions file name, relative to the config dir.
pub const FILE_NAME: &str = "APPEND_SYSTEM.md";

/// The user-level instructions file (`~/.config/daedalus/APPEND_SYSTEM.md`).
pub fn user_file() -> PathBuf {
    paths::config_dir().join(FILE_NAME)
}

/// Read `path` as user instructions. `None` when the file is missing,
/// unreadable, or blank after trimming — an empty file must not add prompt
/// noise.
pub fn read_from(path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Append `instructions` to `base` under a short lead-in so the model reads
/// them as user-authored, or return `base` unchanged when there are none.
pub fn append(base: &str, instructions: Option<&str>) -> String {
    match instructions.map(str::trim) {
        Some(text) if !text.is_empty() => {
            format!("{base}\n\nAdditional instructions from the user:\n{text}")
        }
        _ => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn missing_file_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_from(&dir.path().join(FILE_NAME)), None);
    }

    #[test]
    fn blank_file_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), FILE_NAME, "  \n\t\n");
        assert_eq!(read_from(&path), None);
    }

    #[test]
    fn file_reads_trimmed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), FILE_NAME, "\nAlways run `cargo fmt`.\n\n");
        assert_eq!(read_from(&path).as_deref(), Some("Always run `cargo fmt`."));
    }

    #[test]
    fn append_adds_instructions_under_a_lead_in() {
        let base = "You are daedalus.";
        let out = append(base, Some("Prefer rg over git grep."));
        assert!(out.starts_with(base));
        assert!(out.contains("Additional instructions from the user:"));
        assert!(out.ends_with("Prefer rg over git grep."));
    }

    #[test]
    fn append_without_instructions_is_identity() {
        let base = "You are daedalus.";
        assert_eq!(append(base, None), base);
        assert_eq!(append(base, Some("   ")), base);
    }

    #[test]
    fn user_file_lives_in_the_config_dir() {
        let path = user_file();
        assert!(path.ends_with(FILE_NAME));
        assert_eq!(path.parent(), Some(paths::config_dir().as_path()));
    }
}
