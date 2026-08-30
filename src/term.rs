//! Terminal + cancellation handling.
//!
//! In interactive mode the terminal is put into raw *input* mode so Esc and
//! Ctrl-C arrive as bytes (output processing is kept, so `\n` still works).
//! A single input thread reads stdin: while the agent is busy, Esc/Ctrl-C
//! request cancellation; while idle, it line-edits and emits `Line`/`Cancel`/
//! `Eof` events.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
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

/// An event produced by the interactive input thread.
pub enum InputEvent {
    /// A completed line of input (Enter pressed).
    Line(String),
    /// Esc or Ctrl-C while idle (at the prompt).
    Cancel,
    /// Ctrl-D on an empty line.
    Eof,
}

/// RAII guard that switches the terminal into raw input mode and restores it
/// on drop. Output line processing (OPOST/ONLCR) is preserved so `println!`
/// output still renders normally.
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
            raw.c_oflag |= libc::OPOST | libc::ONLCR;
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

/// Spawn the input thread. While `busy`, Esc/Ctrl-C request cancellation and
/// other input is discarded; while idle, it performs line editing and emits
/// `InputEvent`s.
#[cfg(unix)]
pub fn spawn_input(busy: Arc<AtomicBool>) -> (Receiver<InputEvent>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || input_loop(busy, tx));
    (rx, handle)
}

#[cfg(unix)]
fn input_loop(busy: Arc<AtomicBool>, tx: mpsc::Sender<InputEvent>) {
    let mut buf: Vec<u8> = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        if std::io::stdin().read(&mut byte).unwrap_or(0) == 0 {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        let b = byte[0];
        let idle = !busy.load(Ordering::Relaxed);
        match b {
            0x0A | 0x0D => {
                if idle {
                    let line = String::from_utf8_lossy(&buf).into_owned();
                    buf.clear();
                    let _ = std::io::stdout().write_all(b"\n");
                    let _ = std::io::stdout().flush();
                    if tx.send(InputEvent::Line(line)).is_err() {
                        return;
                    }
                }
            }
            0x1B | 0x03 => {
                if idle {
                    if tx.send(InputEvent::Cancel).is_err() {
                        return;
                    }
                } else {
                    request_cancel();
                }
            }
            0x04 => {
                if idle && buf.is_empty() && tx.send(InputEvent::Eof).is_err() {
                    return;
                }
            }
            0x7F | 0x08 => {
                if idle && !buf.is_empty() {
                    buf.pop();
                    let _ = std::io::stdout().write_all(b"\x08 \x08");
                    let _ = std::io::stdout().flush();
                }
            }
            b if idle && ((0x20..=0x7E).contains(&b) || b >= 0x80) => {
                buf.push(b);
                let _ = std::io::stdout().write_all(&[b]);
                let _ = std::io::stdout().flush();
            }
            _ => {}
        }
    }
}

// Non-Unix fallback: no raw mode, no Esc. Ctrl-C keeps the default SIGINT
// behavior (terminate); input is a plain line reader.
#[cfg(not(unix))]
pub struct RawMode;

#[cfg(not(unix))]
impl RawMode {
    pub fn enable() -> std::io::Result<Self> {
        Ok(RawMode)
    }
}

#[cfg(not(unix))]
pub fn spawn_input(_busy: Arc<AtomicBool>) -> (Receiver<InputEvent>, JoinHandle<()>) {
    use std::io::BufRead;
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    if tx.send(InputEvent::Line(l)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    (rx, handle)
}
