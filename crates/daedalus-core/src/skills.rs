//! Skills: pi-style load-on-demand instruction files.
//!
//! With the memory/self-improving loop gone, skills are the one
//! mechanism for users to extend the agent's instructions without touching
//! code. A skill is a markdown file `<name>.md` discovered in two places:
//!
//! - **user level** — `~/.config/daedalus/skills/`
//! - **workspace level** — `<workspace>/.daedalus/skills/`, which wins on a name
//!   clash (so a project can pin its own version of a user skill).
//!
//! The first paragraph is the **description**; the whole file is the
//! instruction **content** loaded into the conversation. There is no
//! frontmatter parser in v1.
//!
//! The runtime appends a catalog block to the system prompt
//! ([`with_catalog`]) so the model knows what exists. A human loads one with
//! the TUI `/skill <name>` command, which pushes the content as a user
//! message. The model can also self-serve by reading the file with the
//! existing `read` tool — but only for **workspace** skills, whose path is
//! inside the workspace (a user-level skill lives outside the workspace and is
//! therefore `/skill`-only; see [`catalog_block`]). No new tool, no extra
//! model call.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::paths;
use crate::workspace::Workspace;

/// Longest description embedded in the system-prompt catalog, in chars. Keeps
/// a runaway first paragraph from bloating the prompt on every turn.
const MAX_DESCRIPTION: usize = 200;

/// Where a skill was discovered. Workspace skills shadow user skills of the
/// same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillLevel {
    /// `~/.config/daedalus/skills/`.
    User,
    /// `<workspace>/.daedalus/skills/`.
    Workspace,
}

impl SkillLevel {
    /// Short label for the catalog and the `/skills` listing.
    pub fn label(self) -> &'static str {
        match self {
            SkillLevel::User => "user",
            SkillLevel::Workspace => "workspace",
        }
    }
}

/// A discovered skill: its name (file stem), first-paragraph description, the
/// full file content, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub path: PathBuf,
    pub level: SkillLevel,
}

impl Skill {
    /// The user message that loads this skill into the conversation.
    pub fn prompt(&self) -> String {
        format!("Follow these instructions:\n\n{}", self.content)
    }
}

/// The user-level skills directory (`~/.config/daedalus/skills`).
pub fn user_dir() -> PathBuf {
    paths::config_dir().join("skills")
}

/// The workspace-level skills directory (`<workspace>/.daedalus/skills`).
pub fn workspace_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".daedalus").join("skills")
}

/// Discover skills for `workspace`: user level first, then workspace level (so
/// workspace files overwrite same-named user files). Results are sorted by
/// name for deterministic output; a missing or empty directory contributes
/// nothing.
pub fn discover(workspace: &Workspace) -> Vec<Skill> {
    discover_in(&user_dir(), workspace)
}

/// Like [`discover`], but with an explicit user-level directory. Split out so
/// tests (and callers that override the config root) do not depend on the
/// ambient `~/.config/daedalus/skills`.
pub fn discover_in(user_dir: &Path, workspace: &Workspace) -> Vec<Skill> {
    let mut skills: BTreeMap<String, Skill> = BTreeMap::new();
    read_dir_into(user_dir, SkillLevel::User, None, &mut skills);
    read_dir_into(
        &workspace_dir(workspace.root()),
        SkillLevel::Workspace,
        Some(workspace),
        &mut skills,
    );
    skills.into_values().collect()
}

/// True when `path` canonicalizes to a location inside `workspace`.
fn within(workspace: &Workspace, path: &Path) -> Option<PathBuf> {
    let canon = path.canonicalize().ok()?;
    canon.starts_with(workspace.root()).then_some(canon)
}

/// Read every markdown file in `dir` into `out`, keyed by file stem (later
/// inserts, i.e. the workspace level, replace earlier ones).
///
/// Skills are workspace-based: when `sandbox` is set, both the directory and
/// every file must canonicalize inside that workspace, so a symlinked skill — or
/// a symlinked `.daedalus/skills` — cannot pull content from outside it. This is a
/// skills-specific rule; it does not reintroduce the general tool path guard,
/// which was removed. User-level skills are exempt (they live outside by design).
fn read_dir_into(
    dir: &Path,
    level: SkillLevel,
    sandbox: Option<&Workspace>,
    out: &mut BTreeMap<String, Skill>,
) {
    let dir = match sandbox {
        Some(ws) => match within(ws, dir) {
            Some(p) => p,
            None => return, // symlinked/escaping skills dir
        },
        None => dir.to_path_buf(),
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return; // missing/unreadable directory: no skills
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !has_markdown_extension(&path) {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let resolved = match sandbox {
            Some(ws) => match within(ws, &path) {
                Some(p) => p,
                None => continue, // symlink escaping the workspace
            },
            None => path.clone(),
        };
        let Ok(raw) = std::fs::read_to_string(&resolved) else {
            continue; // unreadable, or a directory named `*.md`
        };
        let content = raw.trim().to_string();
        if content.is_empty() {
            continue; // nothing to load or describe
        }
        out.insert(
            name.to_string(),
            Skill {
                name: name.to_string(),
                description: describe(&content),
                content,
                path: resolved,
                level,
            },
        );
    }
}

/// True when `path` has a `.md` extension, case-insensitively.
fn has_markdown_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

/// The description of a skill: its first paragraph (block of non-empty
/// lines), with a leading ATX heading marker stripped and lines joined by a
/// space. Truncated to [`MAX_DESCRIPTION`] chars. Falls back to the first line
/// when the file has no blank line.
fn describe(content: &str) -> String {
    let mut parts: Vec<String> = content
        .lines()
        .map(str::trim)
        .skip_while(|l| l.is_empty())
        .take_while(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    // Strip a leading ATX heading marker ("# Title", "## Title") but not a
    // '#' that is part of the text ("#1 priority").
    if let Some(first) = parts.first_mut() {
        let hashes = first.len() - first.trim_start_matches('#').len();
        if hashes > 0 {
            let rest = &first[hashes..];
            if rest.is_empty() || rest.starts_with(' ') {
                *first = rest.trim_start().to_string();
            }
        }
    }
    let joined = parts.join(" ");
    let joined = joined.trim();
    if joined.is_empty() {
        "(no description)".to_string()
    } else {
        truncate_chars(joined, MAX_DESCRIPTION)
    }
}

/// Truncate `s` to at most `max` chars on a char boundary, appending an
/// ellipsis when it was cut.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The system-prompt catalog block for `skills`, or `None` when there are
/// none (so a missing skills directory leaves the prompt untouched).
///
/// Only workspace skills advertise a file path: the model's `read` tool is
/// confined to the workspace, so a user-level path would be a dead reference.
/// User skills are listed as `/skill`-loadable only.
pub fn catalog_block(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from("Available skills (load one with /skill <name>):\n");
    for s in skills {
        out.push_str(&format!("- {} — {}", s.name, s.description));
        if s.level == SkillLevel::Workspace {
            out.push_str(&format!(" (also readable at {})", s.path.display()));
        }
        out.push('\n');
    }
    Some(out)
}

/// Append the skills catalog to a base system prompt. Returns `base`
/// unchanged when there are no skills.
pub fn with_catalog(base: &str, skills: &[Skill]) -> String {
    match catalog_block(skills) {
        Some(block) => format!("{base}\n\n{block}"),
        None => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_at(root: &Path) -> Workspace {
        Workspace::new(root.to_path_buf()).expect("workspace")
    }

    fn write(dir: &Path, name: &str, content: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), content).unwrap();
    }

    fn skill(name: &str, level: SkillLevel) -> Skill {
        Skill {
            name: name.into(),
            description: "does a demo".into(),
            content: "body".into(),
            path: PathBuf::from("/ws/.daedalus/skills/demo.md"),
            level,
        }
    }

    #[test]
    fn missing_directories_yield_no_skills() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        assert!(discover_in(user.path(), &workspace_at(ws.path())).is_empty());
    }

    #[test]
    fn workspace_skill_is_discovered_with_description_and_content() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        write(
            &workspace_dir(ws.path()),
            "review.md",
            "Reviews Rust code.\n\nCheck for idiomatic patterns.\n",
        );
        let skills = discover_in(user.path(), &workspace_at(ws.path()));
        assert_eq!(skills.len(), 1);
        let s = &skills[0];
        assert_eq!(s.name, "review");
        assert_eq!(s.description, "Reviews Rust code.");
        assert_eq!(
            s.content,
            "Reviews Rust code.\n\nCheck for idiomatic patterns."
        );
        assert_eq!(s.level, SkillLevel::Workspace);
        assert_eq!(s.level.label(), "workspace");
        assert!(s.prompt().contains("Check for idiomatic patterns."));
    }

    #[test]
    fn workspace_shadows_user_and_order_is_deterministic() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        write(&user.path().join("skills"), "shared.md", "user version");
        write(&user.path().join("skills"), "zeta.md", "zeta");
        write(&workspace_dir(ws.path()), "shared.md", "workspace version");
        write(&workspace_dir(ws.path()), "alpha.md", "alpha");

        let skills = discover_in(&user.path().join("skills"), &workspace_at(ws.path()));
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "shared", "zeta"]);
        let shared = skills.iter().find(|s| s.name == "shared").unwrap();
        assert_eq!(shared.content, "workspace version");
        assert_eq!(shared.level, SkillLevel::Workspace);
    }

    #[test]
    fn non_markdown_and_empty_files_are_ignored() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let dir = workspace_dir(ws.path());
        write(&dir, "notes.txt", "not a skill");
        write(&dir, "blank.md", "   \n\n");
        assert!(discover_in(user.path(), &workspace_at(ws.path())).is_empty());
    }

    #[test]
    fn markdown_extension_is_case_insensitive() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        write(
            &workspace_dir(ws.path()),
            "UPPER.MD",
            "Upper skill.\n\nbody",
        );
        let skills = discover_in(user.path(), &workspace_at(ws.path()));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "UPPER");
    }

    #[test]
    fn heading_is_stripped_from_the_description() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        write(
            &workspace_dir(ws.path()),
            "deploy.md",
            "# Deploy checklist\n\nRun the tests first.\n",
        );
        let skills = discover_in(user.path(), &workspace_at(ws.path()));
        assert_eq!(skills[0].description, "Deploy checklist");
    }

    #[test]
    fn hash_not_followed_by_space_is_kept() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        write(
            &workspace_dir(ws.path()),
            "note.md",
            "#1 priority\n\nbody\n",
        );
        let skills = discover_in(user.path(), &workspace_at(ws.path()));
        assert_eq!(skills[0].description, "#1 priority");
    }

    #[test]
    fn long_description_is_truncated() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let long = "x".repeat(500);
        write(
            &workspace_dir(ws.path()),
            "long.md",
            &format!("{long}\n\nbody\n"),
        );
        let skills = discover_in(user.path(), &workspace_at(ws.path()));
        let d = &skills[0].description;
        assert!(d.chars().count() <= MAX_DESCRIPTION);
        assert!(d.ends_with('…'));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_skill_outside_the_workspace_is_ignored() {
        use std::os::unix::fs::symlink;
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.md"), "Secret.\n\nsecret body").unwrap();
        let dir = workspace_dir(ws.path());
        std::fs::create_dir_all(&dir).unwrap();
        symlink(outside.path().join("secret.md"), dir.join("secret.md")).unwrap();
        assert!(discover_in(user.path(), &workspace_at(ws.path())).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_skills_directory_outside_the_workspace_is_ignored() {
        use std::os::unix::fs::symlink;
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.md"), "Secret.\n\nsecret body").unwrap();
        std::fs::create_dir_all(ws.path().join(".daedalus")).unwrap();
        symlink(outside.path(), ws.path().join(".daedalus").join("skills")).unwrap();
        assert!(discover_in(user.path(), &workspace_at(ws.path())).is_empty());
    }

    #[test]
    fn catalog_block_is_none_for_no_skills_and_omits_user_paths() {
        assert!(catalog_block(&[]).is_none());
        let base = "base prompt";
        assert_eq!(with_catalog(base, &[]), base);

        // A workspace skill advertises its in-workspace read path.
        let workspace_skill = skill("demo", SkillLevel::Workspace);
        let block = catalog_block(std::slice::from_ref(&workspace_skill)).unwrap();
        assert!(block.contains("Available skills"));
        assert!(block.contains("demo — does a demo"));
        assert!(block.contains("/ws/.daedalus/skills/demo.md"));
        assert!(with_catalog(base, &[workspace_skill]).starts_with(base));

        // A user skill is /skill-only: its out-of-workspace path is not shown.
        let mut user_skill = skill("global", SkillLevel::User);
        user_skill.path = PathBuf::from("/home/u/.config/daedalus/skills/global.md");
        let block = catalog_block(std::slice::from_ref(&user_skill)).unwrap();
        assert!(block.contains("global — does a demo"));
        assert!(!block.contains("/home/u/.config/daedalus/skills/global.md"));
    }
}
