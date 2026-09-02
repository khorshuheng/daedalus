//! Crab — a minimal coding agent.
//!
//! The crate is split into layers that mirror the CRAB tickets:
//! - `config`/`workspace` (CRAB-105): configuration and workspace scoping.
//! - `tools` (CRAB-102): the four built-in tools (read, bash, edit, write).
//! - `provider` (CRAB-103): LLM providers behind an extensible `Provider` trait.
//! - `session` (CRAB-109): on-disk conversation persistence for `/resume`.
//! - `index` (CRAB-114): SQLite FTS5 search over lessons + system-prompt injection.
//! - `memory` (CRAB-113): append-only JSONL lesson store for agent memory.
//! - `reflect` (CRAB-112): LLM-backed lesson extraction from session transcripts.
//! - `runtime` (CRAB-116): stateful AgentRuntime engine + Event/Command surface.
//! - `main` (CRAB-101): the CLI entrypoint.

pub mod config;
pub mod index;
pub mod memory;
pub mod provider;
pub mod reflect;
pub mod runtime;
pub mod session;
pub mod term;
pub mod tools;
pub mod workspace;
