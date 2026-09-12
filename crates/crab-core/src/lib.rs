//! Crab core library (CRAB-117 workspace split): the synchronous engine.
//!
//! The crate is split into layers that mirror the CRAB tickets:
//! - `config`/`workspace` (CRAB-105/118): configuration and workspace scoping.
//! - `credential` (CRAB-118): API-key resolution (flag > provider env > keyring).
//! - `tools` (CRAB-102): the four built-in tools (read, bash, edit, write).
//! - `provider` (CRAB-103): LLM providers behind an extensible `Provider` trait.
//! - `session` (CRAB-109): on-disk conversation persistence for `/resume`.
//! - `runtime` (CRAB-116): stateful AgentRuntime engine + Event/Command surface.
//! - `skills` (CRAB-138): load-on-demand instruction files (user + workspace).
//! - `mcp` (CRAB-133): external MCP tool servers bridged onto the Tool trait.
//! - `paths` (CRAB-119): XDG config/data base directories.
//!
//! The frontends (`crab` binary: main/modes/term) live in the sibling crate so
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
pub mod tools;
pub mod workspace;
