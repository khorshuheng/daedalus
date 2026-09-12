//! crab-server (CRAB-124): the headless WebSocket frontend for crab.
//!
//! The crate exposes [`server::build_router`] / [`AppState`] so the server can
//! be embedded and tested in-process; the `crab-server` binary wires config,
//! workspace and bind address to it.

pub mod server;

pub use server::{build_router, AppState};
