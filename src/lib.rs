//! Crab — a minimal coding agent.
//!
//! The crate is split into layers that mirror the CRAB tickets:
//! - `config`/`workspace` (CRAB-105): configuration and workspace scoping.
//! - `tools` (CRAB-102): the four built-in tools (read, bash, edit, write).
//! - `provider` (CRAB-103): LLM providers behind an extensible `Provider` trait.
//! - `agent` (CRAB-104): the bounded orchestration loop.
//! - `session` (CRAB-109): on-disk conversation persistence for `/resume`.
//! - `memory` (CRAB-113): append-only JSONL lesson store for agent memory.
//! - `main` (CRAB-101): the CLI entrypoint.

pub mod agent;
pub mod config;
pub mod memory;
pub mod provider;
pub mod session;
pub mod term;
pub mod tools;
pub mod workspace;
