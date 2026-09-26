//! XDG base directories (CRAB-119): one source of truth for where daedalus keeps
//! its config, sessions, and memory, replacing hand-rolled `$HOME/.config/...`
//! and `$HOME/.local/share/...` joins with the `directories` crate.
//!
//! On Linux this resolves `$XDG_CONFIG_HOME`/`$XDG_DATA_HOME` (defaulting to
//! `~/.config` and `~/.local/share`); other platforms follow their own
//! conventions. When no home directory can be determined the current
//! directory is used, matching the previous `HOME` fallback.

use std::path::PathBuf;

use directories::ProjectDirs;

const QUALIFIER: &str = "";
const ORGANIZATION: &str = "";
const APPLICATION: &str = "daedalus";

fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from(QUALIFIER, ORGANIZATION, APPLICATION)
}

/// The XDG config root for daedalus, e.g. `~/.config/daedalus`. The config file lives
/// at `config_dir()/config.toml`.
pub fn config_dir() -> PathBuf {
    project_dirs()
        .map(|d| d.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The XDG data root for daedalus, e.g. `~/.local/share/daedalus`. Sessions and
/// memory live under per-project subdirectories of `data_dir()`.
pub fn data_dir() -> PathBuf {
    project_dirs()
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ProjectDirs::from("", "", "daedalus")` places the application dir at the
    /// XDG root with the app name as the last component (config and data dirs
    /// are distinct).
    #[test]
    fn roots_are_distinct_and_application_scoped() {
        let Some(dirs) = ProjectDirs::from(QUALIFIER, ORGANIZATION, APPLICATION) else {
            return; // no home dir in this environment; fallbacks already cover it
        };
        assert!(dirs.config_dir().ends_with("daedalus"));
        assert!(dirs.data_dir().ends_with("daedalus"));
        assert_ne!(dirs.config_dir(), dirs.data_dir());
        // config_dir()/data_dir() must match the same ProjectDirs resolution.
        assert_eq!(config_dir(), dirs.config_dir());
        assert_eq!(data_dir(), dirs.data_dir());
    }
}
