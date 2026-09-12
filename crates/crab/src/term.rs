//! Process-wide cancellation flag.
//!
//! The REPL (CRAB-110) and its raw-mode line editor were removed in CRAB-135;
//! what survives here is the process-wide cancel flag checked by the
//! providers and by reflection, used by the headless json/rpc modes until
//! CRAB-130 replaces it with a proper cancellation token. The TUI drives
//! cancellation through `AgentRuntime::abort` (crossterm key events).

use std::sync::atomic::AtomicBool;

static CANCEL: AtomicBool = AtomicBool::new(false);

/// The process-wide cancel flag, checked by the providers and tools.
pub fn cancel_flag() -> &'static AtomicBool {
    &CANCEL
}
