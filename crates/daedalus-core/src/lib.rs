//! Daedalus core library: the agent engine, split from the frontends.
//!
//! The crate is split into these layers:
//! - `config`/`workspace`: configuration and workspace scoping.
//! - `credential`: API-key resolution (flag > provider env > keyring).
//! - `instructions`: user-level `APPEND_SYSTEM.md`, appended to the prompt.
//! - `tools`: the five built-in tools (read, search, bash, edit, write).
//! - `provider`: LLM providers behind an extensible `Provider` trait.
//! - `session`: on-disk conversation persistence for `/resume`.
//! - `runtime`: stateful AgentRuntime engine + Event/Command surface.
//! - `skills`: load-on-demand instruction files (user + workspace).
//! - `mcp`: external MCP tool servers bridged onto the Tool trait.
//! - `paths`: XDG config/data base directories.
//!
//! The frontends (`daedalus` binary: main/modes/tui; `daedalus-server`) live in
//! sibling crates, so this core never depends on terminal or stdio concerns. It
//! is async (tokio) but stays runtime-agnostic: the worker thread owns its
//! current-thread runtime, and nothing here uses `#[tokio::main]`.

pub mod config;
pub mod credential;
pub mod instructions;
pub mod mcp;
pub mod paths;
pub mod provider;
pub mod runtime;
pub mod session;
pub mod skills;
pub mod theme;
pub mod tools;
pub mod workspace;
