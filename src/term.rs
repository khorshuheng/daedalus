//! Terminal + cancellation handling.
//!
//! Ctrl-C (via the `ctrlc` crate) and Esc (via raw terminal mode) both request
//! *cancellation* of the current action instead of terminating the process.
//! Exiting is done explicitly with `/exit` at the prompt.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

static CANCEL: AtomicBool = AtomicBool::new(false);

/// The process-wide cancel flag, checked by the providers and tools.
pub fn cancel_flag() -> &'static AtomicBool {
    &CANCEL
}

pub fn request_cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

pub fn clear_cancel() {
    CANCEL.store(false, Ordering::SeqCst);
}

/// Install a Ctrl-C handler that requests cancellation instead of terminating.
pub fn install_ctrl_c() {
    let _ = ctrlc::set_handler(request_cancel);
}

/// RAII guard that switches the terminal into raw mode (no line buffering,
/// echo, or signal generation) and restores it on drop.
#[cfg(unix)]
pub struct RawMode {
    orig: libc::termios,
}

#[cfg(unix)]
impl RawMode {
    pub fn enable() -> std::io::Result<Self> {
        unsafe {
            let mut raw: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut raw) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let orig = raw;
            libc::cfmakeraw(&mut raw);
            raw.c_cc[libc::VMIN] = 0;
            raw.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(RawMode { orig })
        }
    }
}

#[cfg(unix)]
impl Drop for RawMode {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.orig);
        }
    }
}

/// Watch stdin for Esc (0x1B) or Ctrl-C (0x03) and request cancellation, until
/// `stop` is set. In raw mode these arrive as plain bytes.
#[cfg(unix)]
pub fn spawn_esc_watcher(stop: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut byte = [0u8; 1];
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            match std::io::stdin().read(&mut byte) {
                Ok(0) => std::thread::sleep(Duration::from_millis(20)),
                Ok(_) => {
                    if byte[0] == 0x1B || byte[0] == 0x03 {
                        request_cancel();
                    }
                }
                Err(_) => break,
            }
        }
    })
}

// Non-Unix fallbacks: no raw mode, no Esc (Ctrl-C still works via ctrlc).
#[cfg(not(unix))]
pub struct RawMode;

#[cfg(not(unix))]
impl RawMode {
    pub fn enable() -> std::io::Result<Self> {
        Ok(RawMode)
    }
}

#[cfg(not(unix))]
pub fn spawn_esc_watcher(_stop: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(|| {})
}
