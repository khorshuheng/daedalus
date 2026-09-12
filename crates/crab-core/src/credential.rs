//! API-key credentials (CRAB-118): secrets never live in the config file.
//!
//! Resolution order, applied when building the config:
//!
//! 1. `--api-key` flag (highest);
//! 2. provider-native environment variable (`OPENAI_API_KEY`,
//!    `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`);
//! 3. the OS keyring (secret-service on Linux), where `/login` stores the key.
//!
//! The keyring is best-effort: a headless run without a Secret Service
//! falls back to flag/env and simply reports that no stored key exists.

use crate::config::ProviderInfo;

/// The keyring service name under which crab stores provider API keys.
const KEYRING_SERVICE: &str = "crab";

/// Store `key` for `provider` in the OS keyring (used by `/login`).
pub fn store_api_key(provider: &ProviderInfo, key: &str) -> Result<(), String> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, provider.name)
        .map_err(|e| format!("keyring unavailable: {e}"))?;
    entry
        .set_password(key)
        .map_err(|e| format!("could not store API key: {e}"))
}

/// Read the stored key for `provider` from the OS keyring, or `None` when
/// nothing is stored (or no Secret Service is available).
pub fn stored_api_key(provider: &ProviderInfo) -> Option<String> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, provider.name).ok()?;
    entry.get_password().ok()
}

/// Delete the stored key for `provider`, if any.
pub fn delete_api_key(provider: &ProviderInfo) -> Result<(), String> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, provider.name)
        .map_err(|e| format!("keyring unavailable: {e}"))?;
    entry
        .delete_credential()
        .map_err(|e| format!("could not delete API key: {e}"))
}

/// Resolve the API key for `provider`: `flag` > provider-native env >
/// keyring. Returns the key to attach to the config, or `None` when nothing
/// is configured (providers then fail with a clear auth error at request
/// time).
pub fn resolve_api_key(provider: &ProviderInfo, flag: Option<String>) -> Option<String> {
    if let Some(k) = flag {
        return Some(k);
    }
    if let Some(name) = provider.api_key_env {
        if let Ok(k) = std::env::var(name) {
            if !k.is_empty() {
                return Some(k);
            }
        }
    }
    stored_api_key(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::provider_by_name;

    fn test_provider(name: &str) -> &'static ProviderInfo {
        provider_by_name(name).unwrap()
    }

    #[test]
    fn flag_beats_env_beats_keyring() {
        // flag wins over env.
        std::env::set_var("OPENAI_API_KEY", "env-key");
        assert_eq!(
            resolve_api_key(test_provider("openai"), Some("flag-key".into())).as_deref(),
            Some("flag-key")
        );
        // env wins when no flag.
        assert_eq!(
            resolve_api_key(test_provider("openai"), None).as_deref(),
            Some("env-key")
        );
        std::env::remove_var("OPENAI_API_KEY");
        // No env/flag: keyring (empty in tests) -> None.
        assert_eq!(resolve_api_key(test_provider("openai"), None), None);
    }

    #[test]
    fn fake_provider_has_no_env_name() {
        assert_eq!(test_provider("fake").api_key_env, None);
        assert_eq!(test_provider("openai").api_key_env, Some("OPENAI_API_KEY"));
    }
}

#[cfg(test)]
mod keyring_tests {
    use super::*;

    /// A synthetic provider so these tests can never read, overwrite, or
    /// delete a real provider's stored key. They previously used `openai` and
    /// `deepseek`, so `cargo test` destroyed a developer's credentials on
    /// every run (CRAB-143). Each test gets its own account so the parallel
    /// test runner cannot race on a shared entry.
    fn probe(name: &'static str) -> ProviderInfo {
        ProviderInfo {
            name,
            preset_base_url: "",
            api_key_env: None,
            effort: crate::config::EffortStyle::None,
        }
    }

    // These tests exercise the real OS keyring (secret-service on this host).
    // They are gated on the environment actually exposing one; where none is
    // available the store functions fail cleanly and we skip assertions.
    fn has_keyring(p: &ProviderInfo) -> bool {
        let stored = keyring::Entry::new(KEYRING_SERVICE, p.name)
            .and_then(|e| e.set_password("probe"))
            .is_ok();
        let _ = keyring::Entry::new(KEYRING_SERVICE, p.name).and_then(|e| e.delete_credential());
        stored
    }

    #[test]
    fn keyring_round_trips_a_probe_key_when_available() {
        let p = probe("crab-keyring-probe-rt");
        if !has_keyring(&p) {
            eprintln!("skipping: no OS keyring in this environment");
            return;
        }
        store_api_key(&p, "probe-secret").unwrap();
        assert_eq!(stored_api_key(&p).as_deref(), Some("probe-secret"));
        delete_api_key(&p).unwrap();
        assert_eq!(stored_api_key(&p), None);
    }

    #[test]
    fn missing_key_returns_none() {
        // A provider we never stored a key for must resolve to None even when
        // a keyring exists (delete first to be safe).
        let p = probe("crab-keyring-probe-missing");
        let _ = delete_api_key(&p);
        assert_eq!(stored_api_key(&p), None);
    }
}
