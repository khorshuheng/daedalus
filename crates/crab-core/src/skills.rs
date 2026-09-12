//! Skills (CRAB-138): pi-style load-on-demand instruction files.
//!
//! With the memory/self-improving loop gone (CRAB-137), skills are the one
//! mechanism for users to extend the agent's instructions without touching
//! code. A skill is a markdown file `<name>.md` discovered in two places:
//!
//! - **user level** — `~/.config/crab/skills/`
//! - **workspace level** — `<workspace>/.crab/skills/`, which wins on a name
//!   clash (so a project can pin its own version of a user skill).
//!
//! The first paragraph is the **description**; the whole file is the
//! instruction **content** loaded into the conversation. There is no
//! frontmatter parser in v1.
//!
//! The runtime appends a catalog block to the system prompt
//! ([`with_catalog`]) so the model knows what exists and can load one itself
//! with the existing `read` tool; a human loads one with the TUI `/skill
//! <name>` command, which pushes the content as a user message. No new tool
//! and no extra model call.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::paths;

/// Where a skill was discovered. Workspace skills shadow user skills of the
/// same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillLevel {
    /// `~/.config/crab/skills/`.
    User,
    /// `<workspace>/.crab/skills/`.
    Workspace,
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

/// The user-level skills directory (`~/.config/crab/skills`).
pub fn user_dir() -> PathBuf {
    paths::config_dir().join("skills")
}

/// The workspace-level skills directory (`<workspace>/.crab/skills`).
pub fn workspace_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".crab").join("skills")
}

/// Discover skills for `workspace_root`: user level first, then workspace
/// level (so workspace files overwrite same-named user files). Results are
/// sorted by name for deterministic output; a missing or empty directory
/// contributes nothing.
pub fn discover(workspace_root: &Path) -> Vec<Skill> {
    discover_in(&user_dir(), workspace_root)
}

/// Like [`discover`], but with an explicit user-level directory. Split out so
/// tests (and callers that override the config root) do not depend on the
/// ambient `~/.config/crab/skills`.
pub fn discover_in(user_dir: &Path, workspace_root: &Path) -> Vec<Skill> {
    let mut skills: BTreeMap<String, Skill> = BTreeMap::new();
    read_into(user_dir, SkillLevel::User, &mut skills);
    read_into(
        &workspace_dir(workspace_root),
        SkillLevel::Workspace,
        &mut skills,
    );
    skills.into_values().collect()
}

/// Read every `*.md` in `dir` into `out`, keyed by file stem (later inserts,
/// i.e. the workspace level, replace earlier ones).
fn read_into(dir: &Path, level: SkillLevel, out: &mut BTreeMap<String, Skill>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return; // missing/unreadable directory: no skills
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
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
                path,
                level,
            },
        );
    }
}

/// The description of a skill: its first paragraph (block of non-empty
/// lines), with a leading ATX heading marker stripped and lines joined by a
/// space. Falls back to the first line when the file has no blank line.
fn describe(content: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for line in content.lines().map(str::trim).skip_while(|l| l.is_empty()) {
        if line.is_empty() {
            break;
        }
        parts.push(line);
    }
    let joined = parts.join(" ");
    let trimmed = joined.trim_start_matches('#').trim();
    if trimmed.is_empty() {
        "(no description)".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The system-prompt catalog block for `skills`, or `None` when there are
/// none (so a missing skills directory leaves the prompt untouched).
pub fn catalog_block(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out =
        String::from("Available skills (load one with /skill <name>, or read its file):\n");
    for s in skills {
        out.push_str(&format!(
            "- {} — {} ({})\n",
            s.name,
            s.description,
            s.path.display()
        ));
    }
    Some(out)
}

/// Append the skills catalog to a base system prompt. Returns `base`
/// unchanged when there are no skills.
pub fn with_catalog(base: &str, skills: &[Skill]) -> String {
    match catalog_block(skills) {
        Some(block) => format!("{base}\n{block}"),
        None => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, content: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn missing_directories_yield_no_skills() {
        let ws = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        assert!(discover_in(user.path(), ws.path()).is_empty());
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
        let skills = discover_in(user.path(), ws.path());
        assert_eq!(skills.len(), 1);
        let s = &skills[0];
        assert_eq!(s.name, "review");
        assert_eq!(s.description, "Reviews Rust code.");
        assert_eq!(
            s.content,
            "Reviews Rust code.\n\nCheck for idiomatic patterns."
        );
        assert_eq!(s.level, SkillLevel::Workspace);
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

        let skills = discover_in(&user.path().join("skills"), ws.path());
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
        assert!(discover_in(user.path(), ws.path()).is_empty());
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
        let skills = discover_in(user.path(), ws.path());
        assert_eq!(skills[0].description, "Deploy checklist");
    }

    #[test]
    fn catalog_block_is_none_for_no_skills_and_lists_otherwise() {
        assert!(catalog_block(&[]).is_none());
        let base = "base prompt";
        assert_eq!(with_catalog(base, &[]), base);

        let skill = Skill {
            name: "demo".into(),
            description: "does a demo".into(),
            content: "body".into(),
            path: PathBuf::from("/ws/.crab/skills/demo.md"),
            level: SkillLevel::Workspace,
        };
        let block = catalog_block(std::slice::from_ref(&skill)).unwrap();
        assert!(block.contains("demo"));
        assert!(block.contains("does a demo"));
        assert!(block.contains("/ws/.crab/skills/demo.md"));
        let composed = with_catalog(base, &[skill]);
        assert!(composed.starts_with(base));
        assert!(composed.contains("Available skills"));
    }
}
