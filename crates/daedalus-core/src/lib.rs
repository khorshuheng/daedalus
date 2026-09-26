//! Daedalus core library: the synchronous engine.
//!
//! The crate is split into these layers:
//! - `config`/`workspace`: configuration and workspace scoping.
//! - `credential`: API-key resolution (flag > provider env > keyring).
//! - `tools`: the four built-in tools (read, bash, edit, write).
//! - `provider`: LLM providers behind an extensible `Provider` trait.
//! - `session`: on-disk conversation persistence for `/resume`.
//! - `runtime`: stateful AgentRuntime engine + Event/Command surface.
//! - `skills`: load-on-demand instruction files (user + workspace).
//! - `mcp`: external MCP tool servers bridged onto the Tool trait.
//! - `paths`: XDG config/data base directories.
//!
//! The frontends (`daedalus` binary: main/modes/term) live in the sibling crate so
//! this core never depends on terminal or stdio concerns, and stays free of
//! async dependencies.

pub mod config;
pub mod credential;
pub mod mcp;
pub mod paths;
pub mod provider;
pub mod runtime;
pub mod session;
pub mod skills;
pub mod theme;
pub mod tools;
pub mod workspace;
