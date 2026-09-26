//! daedalus-server: the headless WebSocket frontend for daedalus.
//!
//! The crate exposes [`server::build_router`] / [`AppState`] so the server can
//! be embedded and tested in-process; the `daedalus-server` binary wires config,
//! workspace and bind address to it.

pub mod server;

pub use server::{build_router, AppState};
