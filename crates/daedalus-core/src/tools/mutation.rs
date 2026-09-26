//! Per-file mutation serialization.
//!
//! Tool calls within one assistant turn run concurrently (`runtime.rs`
//! `join_all`), and `write`/`edit` are read-modify-write operations. Without a
//! lock, two mutations of the same file both read the original content and the
//! later `fs::write` wins, silently discarding the other edit. pi serializes
//! these with a per-file queue (`core/tools/file-mutation-queue.js`); this is
//! the same idea for daedalus's blocking tool bodies.
//!
//! The lock is keyed on the canonical path, so two symlinks to one file share a
//! lock, while distinct files still mutate in parallel. A single tool call holds
//! at most one file lock, so lock ordering cannot deadlock.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

type Lock = Arc<Mutex<()>>;

/// Process-wide registry of per-path locks.
fn registry() -> &'static Mutex<HashMap<PathBuf, Lock>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Lock>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The lock key for `path`: the fully canonical path when it exists (so two
/// symlinks to the same file share a lock), otherwise the path as given —
/// workspace resolution has already canonicalized its deepest existing
/// ancestor.
fn mutation_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Write `contents` to `path` atomically: a sibling temp file is written,
/// fsynced, and renamed over the target. A crash, kill or ENOSPC leaves the
/// original file intact instead of a truncated one. An existing file's
/// permissions are preserved; a new file gets the conventional `0644`.
pub fn write_atomic(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let existing = std::fs::metadata(path).ok().map(|m| m.permissions());

    let mut builder = tempfile::Builder::new();
    builder.prefix(".daedalus-write-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(
            existing
                .clone()
                .unwrap_or_else(|| std::fs::Permissions::from_mode(0o644)),
        );
    }
    let mut tmp = builder.tempfile_in(parent)?;
    #[cfg(not(unix))]
    if let Some(permissions) = existing {
        tmp.as_file().set_permissions(permissions)?;
    }
    std::io::Write::write_all(&mut tmp, contents)?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    // Best-effort directory fsync so the rename itself is durable.
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Run `f` while holding the mutation lock for `path`.
///
/// Serializes read-modify-write operations targeting the same file; operations
/// on different files still run in parallel. A panic in `f` only poisons this
/// path's lock, which is recovered so later mutations still run.
pub fn with_file_mutation<T>(path: &Path, f: impl FnOnce() -> T) -> T {
    let key = mutation_key(path);
    let lock = {
        let mut map = registry().lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(map.entry(key.clone()).or_default())
    };

    let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let result = f();
    drop(guard);

    // Drop the entry once nobody else holds or waits on it, so the map cannot
    // grow without bound. `strong_count == 2` means only the map and our local
    // clone reference it (a waiter or holder keeps its own clone, so the count
    // stays >= 3).
    let mut map = registry().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = map.get(&key) {
        if Arc::ptr_eq(existing, &lock) && Arc::strong_count(existing) == 2 {
            map.remove(&key);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    /// The whole point: two mutations of one path must never be inside the
    /// critical section at the same time.
    #[test]
    fn same_path_never_overlaps() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("f");
        let inside = Arc::new(AtomicBool::new(false));
        let overlap = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let path = path.clone();
            let inside = Arc::clone(&inside);
            let overlap = Arc::clone(&overlap);
            handles.push(std::thread::spawn(move || {
                with_file_mutation(&path, || {
                    if inside.swap(true, Ordering::SeqCst) {
                        overlap.store(true, Ordering::SeqCst);
                    }
                    std::thread::sleep(Duration::from_millis(20));
                    inside.store(false, Ordering::SeqCst);
                });
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread");
        }
        assert!(
            !overlap.load(Ordering::SeqCst),
            "two same-path critical sections overlapped"
        );
    }

    /// Distinct paths must not be serialized behind one another.
    #[test]
    fn different_paths_do_not_block_each_other() {
        let dir = tempfile::tempdir().expect("temp dir");
        let a = dir.path().join("a");
        let b = dir.path().join("b");

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder_path = a.clone();
        let holder = std::thread::spawn(move || {
            with_file_mutation(&holder_path, || {
                entered_tx.send(()).expect("signal entered");
                release_rx.recv().expect("wait for release");
            });
        });
        entered_rx.recv().expect("A is held");

        // B must acquire promptly while A is still held.
        let (done_tx, done_rx) = mpsc::channel();
        let other = b.clone();
        std::thread::spawn(move || {
            with_file_mutation(&other, || {});
            done_tx.send(()).expect("signal done");
        });
        assert!(
            done_rx.recv_timeout(Duration::from_secs(2)).is_ok(),
            "a different path was blocked by an unrelated lock"
        );

        release_tx.send(()).expect("release A");
        holder.join().expect("holder thread");
    }

    /// The registry must not leak entries for paths that are no longer in use.
    #[test]
    fn registry_entry_is_dropped_after_use() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("f");
        with_file_mutation(&path, || {});
        let map = registry().lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !map.contains_key(&mutation_key(&path)),
            "queue entry leaked for {}",
            path.display()
        );
    }
}
