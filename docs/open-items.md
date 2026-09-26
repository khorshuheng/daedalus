# Open items

Candidate improvements that are not scheduled yet. Give each one a ticket
(`DAE-…`) before implementing.

## Environment / context conventions in the system prompt

daedalus injects only the skills catalog. It has no environment or
project-context conventions, unlike the agents it was reviewed against:

- **pi** appends `<project_context>` blocks (`<project_instructions path="…">`)
  and a trailing `Current working directory: …` line.
- **opencode** auto-loads `AGENTS.md` / `CLAUDE.md` from the repo.
- **Claude Code** emits an `<env>` block (cwd, OS, git status/branch).
- **deepseek-harness** materializes dynamic context as a durable snapshot on the
  turn, and keeps a mode-independent tool catalog for request-cache stability.

Possible directions:

- Inject a working-directory / OS / git-state line.
- Auto-load `AGENTS.md` (or `DAEDALUS.md`) from the workspace when present, as
  project instructions.
- Decide the caching story: the prompt is currently a static string (cache
  friendly); per-turn env lines trade that for freshness.

Partially addressed by: a user-level
`~/.config/daedalus/APPEND_SYSTEM.md` is appended to the system prompt on every
turn (matching pi's `APPEND_SYSTEM.md`). That covers standing user instructions;
the workspace/project-context and env/cwd/git halves above remain open.

Context: this came out of reviewing the system prompt against pi / opencode /
Claude Code / deepseek-harness. The tool-routing "translation" guidance was
moved into the tool descriptions at the same time; the environment/context
conventions were deferred.
