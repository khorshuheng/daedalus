//! Configuration (CRAB-105, revised CRAB-118).
//!
//! Keys: provider, base URL, model, temperature, max_iterations, output size
//! caps, workspace. Sources merge with documented precedence:
//! `flags > config file > defaults` (the `CRAB_*` env layer was removed in
//! CRAB-118). API keys are **not** a config key: they are resolved separately
//! via `credential` (`--api-key` > provider-native env > OS keyring), so a
//! secret can never be stored in the config file.
//!
//! Supported providers: `openai`, `anthropic`, `deepseek`. DeepSeek reuses the
//! OpenAI-compatible client via its own base URL + model (see CRAB-103). The
//! `provider` key selects the client implementation; unknown values fail fast
//! listing the supported set.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The supported provider kinds. `Deepseek` is served through the same
/// OpenAI-compatible wire client as `Openai` (only base URL + model differ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Deepseek,
    Fake,
}

impl ProviderKind {
    /// Parse a provider name from config/env/flags, failing fast on unknown
    /// values with the list of supported providers.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            "deepseek" => Ok(Self::Deepseek),
            "fake" => Ok(Self::Fake),
            other => Err(format!(
                "unknown provider '{other}' (supported: openai, anthropic, deepseek, fake)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Deepseek => "deepseek",
            Self::Fake => "fake",
        }
    }

    /// Default base URL for the provider preset.
    pub fn preset_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://api.openai.com",
            Self::Deepseek => "https://api.deepseek.com",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Fake => "",
        }
    }

    /// Default model for the provider preset.
    pub fn preset_model(self) -> &'static str {
        match self {
            Self::Openai => "gpt-4o-mini",
            Self::Deepseek => "deepseek-chat",
            Self::Anthropic => "claude-3-5-sonnet-latest",
            Self::Fake => "fake-model",
        }
    }

    /// Default context-window size (input tokens) of the provider's default
    /// model. Used to derive the history budget unless it is overridden.
    pub fn preset_context_window(self) -> usize {
        match self {
            Self::Openai => 128_000,
            Self::Deepseek => 64_000,
            Self::Anthropic => 200_000,
            Self::Fake => 128_000,
        }
    }

    /// The provider-native environment variable that carries this provider's
    /// API key (CRAB-118: secrets never live in the config file).
    pub fn api_key_env(self) -> Option<&'static str> {
        match self {
            Self::Openai => Some("OPENAI_API_KEY"),
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::Deepseek => Some("DEEPSEEK_API_KEY"),
            Self::Fake => None,
        }
    }
}

/// Fully-resolved configuration passed to the provider and the agent loop.
#[derive(Debug, Clone)]
pub struct Config {
    pub provider: ProviderKind,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub max_iterations: usize,
    /// Cap for any single tool result / file read, in bytes.
    pub max_output_bytes: usize,
    /// Maximum tokens requested per completion (Anthropic requires this).
    pub max_tokens: usize,
    /// Per-request timeout, in seconds.
    pub timeout_secs: u64,
    /// Number of retries for transient failures (timeouts, 429, 5xx).
    pub max_retries: usize,
    /// Rough token budget for the conversation history. The oldest tool turns
    /// are dropped once exceeded so a session cannot blow the model context.
    pub max_context_tokens: usize,
    /// The single root directory the agent is allowed to touch.
    pub workspace: PathBuf,
}

impl Config {
    /// Built-in defaults, before any file/env/flag override.
    pub fn defaults(workspace: PathBuf) -> Self {
        let provider = ProviderKind::Openai;
        Self {
            provider,
            base_url: provider.preset_base_url().to_string(),
            api_key: None,
            model: provider.preset_model().to_string(),
            temperature: 0.7,
            max_iterations: 30,
            max_output_bytes: 32_000,
            max_tokens: 2048,
            timeout_secs: 60,
            max_retries: 2,
            // Leave headroom for the completion output (max_tokens).
            max_context_tokens: provider.preset_context_window().saturating_sub(4_096),
            workspace,
        }
    }
}

/// Partial configuration from one source (config file or CLI flags). All
/// fields are optional; unresolved fields fall back to defaults or to the
/// next-higher-precedence source. Doubles as the schema of the TOML config
/// file and as the flag overrides — `api_key` is deliberately absent so a
/// secret can never be stored in the config file (CRAB-118).
#[derive(Debug, Default, Clone, Deserialize)]
pub struct PartialConfig {
    pub provider: Option<ProviderKind>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_iterations: Option<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_tokens: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub max_retries: Option<usize>,
    pub max_context_tokens: Option<usize>,
    pub workspace: Option<PathBuf>,
}

impl PartialConfig {
    /// Overlay `higher` on top of `self`: every field set in `higher` wins.
    /// Used to merge the config file under CLI flags (`flags > file`).
    pub fn overlay(&mut self, higher: &PartialConfig) {
        macro_rules! take {
            ($f:ident) => {
                if higher.$f.is_some() {
                    self.$f = higher.$f.clone();
                }
            };
        }
        take!(provider);
        take!(base_url);
        take!(model);
        take!(temperature);
        take!(max_iterations);
        take!(max_output_bytes);
        take!(max_tokens);
        take!(timeout_secs);
        take!(max_retries);
        take!(max_context_tokens);
        take!(workspace);
    }

    /// Resolve to a complete `Config`: fill defaults (provider presets) and
    /// validate, attaching the resolved `api_key`.
    pub fn resolve(
        self,
        default_workspace: PathBuf,
        api_key: Option<String>,
    ) -> Result<Config, String> {
        let provider = self.provider.unwrap_or(ProviderKind::Openai);
        // base_url defaults to the provider preset unless explicitly set.
        let base_url = self
            .base_url
            .unwrap_or_else(|| provider.preset_base_url().to_string());
        let model = self
            .model
            .unwrap_or_else(|| provider.preset_model().to_string());
        let workspace = self.workspace.unwrap_or(default_workspace);

        if workspace.as_os_str().is_empty() {
            return Err("workspace directory must not be empty".into());
        }
        if self.max_iterations.unwrap_or(30) == 0 {
            return Err("max_iterations must be >= 1".into());
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(format!("temperature {t} out of range (0.0..=2.0)"));
            }
        }
        if let Some(t) = self.timeout_secs {
            if t == 0 {
                return Err("timeout_secs must be >= 1".into());
            }
        }
        let max_context_tokens = self
            .max_context_tokens
            .unwrap_or_else(|| provider.preset_context_window().saturating_sub(4_096));
        if max_context_tokens == 0 {
            return Err("max_context_tokens must be >= 1".into());
        }

        Ok(Config {
            provider,
            base_url,
            api_key,
            model,
            temperature: self.temperature.unwrap_or(0.7),
            max_iterations: self.max_iterations.unwrap_or(30),
            max_output_bytes: self.max_output_bytes.unwrap_or(32_000),
            max_tokens: self.max_tokens.unwrap_or(2048),
            timeout_secs: self.timeout_secs.unwrap_or(60),
            max_retries: self.max_retries.unwrap_or(2),
            max_context_tokens,
            workspace,
        })
    }
}

impl Config {
    /// Build a `Config` by merging the config `file`, then CLI `flags` over
    /// defaults — `flags > config file > defaults` (CRAB-118 removes the env
    /// layer). `api_key` is resolved separately by the caller via
    /// `crate::credential::resolve_api_key` and passed in; the config file
    /// cannot carry a secret. `default_workspace` is used unless an override
    /// supplies one.
    /// Build a `Config` by merging the config `file`, then CLI `flags` over
    /// defaults — `flags > config file > defaults` (CRAB-118 removes the env
    /// layer). `api_key_flag` feeds the resolution chain
    /// (`--api-key` > provider-native env > keyring); the config file cannot
    /// carry a secret, so an `api_key` key in it is rejected. `default_workspace`
    /// is used unless an override supplies one.
    pub fn load(
        default_workspace: PathBuf,
        file: Option<&Path>,
        flags: PartialConfig,
        api_key_flag: Option<String>,
    ) -> Result<Config, String> {
        let mut merged = PartialConfig::default();
        if let Some(path) = file {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read config '{}': {e}", path.display()))?;
            // Reject a secret in the config file explicitly: parse the raw
            // TOML too, because typed deserialization would silently drop an
            // unknown `api_key` key.
            let raw: toml::Value = toml::from_str(&text)
                .map_err(|e| format!("invalid config '{}': {e}", path.display()))?;
            if raw.get("api_key").is_some() {
                return Err(format!(
                    "invalid config '{}': api_key in the config file is not supported; set it via --api-key, a provider-native environment variable, or /login (keyring)",
                    path.display()
                ));
            }
            let fc: PartialConfig = toml::from_str(&text)
                .map_err(|e| format!("invalid config '{}': {e}", path.display()))?;
            merged.overlay(&fc);
        }
        merged.overlay(&flags);
        // Resolve the API key against the final provider (flag > env > keyring).
        let api_key = crate::credential::resolve_api_key(
            merged.provider.unwrap_or(ProviderKind::Openai),
            api_key_flag,
        );
        merged.resolve(default_workspace, api_key)
    }
}

/// Deserialize `ProviderKind` from a lowercase string (config file / TOML).
impl<'de> serde::Deserialize<'de> for ProviderKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ProviderKind::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ws() -> PathBuf {
        PathBuf::from("/tmp/crab-config-test")
    }

    fn load(ws: PathBuf, file: Option<&Path>, flags: PartialConfig) -> Result<Config, String> {
        Config::load(ws, file, flags, None)
    }

    #[test]
    fn defaults() {
        let c = load(ws(), None, PartialConfig::default()).unwrap();
        assert_eq!(c.provider, ProviderKind::Openai);
        assert_eq!(c.base_url, "https://api.openai.com");
        assert_eq!(c.api_key, None);
        assert_eq!(c.max_iterations, 30);
        assert_eq!(c.max_context_tokens, 128_000 - 4_096);
    }

    #[test]
    fn deepseek_preset_applies_base_url_and_model() {
        let flags = PartialConfig {
            provider: Some(ProviderKind::Deepseek),
            ..Default::default()
        };
        let c = load(ws(), None, flags).unwrap();
        assert_eq!(c.provider, ProviderKind::Deepseek);
        assert_eq!(c.base_url, "https://api.deepseek.com");
        assert_eq!(c.model, "deepseek-chat");
        assert_eq!(c.max_context_tokens, 64_000 - 4_096);
    }

    #[test]
    fn explicit_base_url_wins_over_preset() {
        let flags = PartialConfig {
            provider: Some(ProviderKind::Deepseek),
            base_url: Some("http://localhost:9000".into()),
            ..Default::default()
        };
        let c = load(ws(), None, flags).unwrap();
        assert_eq!(c.base_url, "http://localhost:9000");
    }

    #[test]
    fn unknown_provider_fails() {
        let err = ProviderKind::parse("wat").unwrap_err();
        assert!(err.contains("openai, anthropic, deepseek"));
    }

    #[test]
    fn unknown_provider_in_file_fails() {
        let dir = std::env::temp_dir().join("crab-config-bad-provider");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "provider = \"nope\"\n").unwrap();
        let err = load(ws(), Some(&path), PartialConfig::default()).unwrap_err();
        assert!(err.contains("unknown provider"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flags_override_file() {
        let dir = std::env::temp_dir().join("crab-config-flag-file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "model = \"file-model\"\n").unwrap();
        let flags = PartialConfig {
            model: Some("flag-model".into()),
            ..Default::default()
        };
        let c = load(ws(), Some(&path), flags).unwrap();
        assert_eq!(c.model, "flag-model");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_file_sets_values() {
        let dir = std::env::temp_dir().join("crab-config-file-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "provider = \"anthropic\"\nmodel = \"claude-x\"\nmax_iterations = 7\n",
        )
        .unwrap();
        let c = load(ws(), Some(&path), PartialConfig::default()).unwrap();
        assert_eq!(c.provider, ProviderKind::Anthropic);
        assert_eq!(c.base_url, "https://api.anthropic.com");
        assert_eq!(c.model, "claude-x");
        assert_eq!(c.max_iterations, 7);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn api_key_in_config_file_is_rejected() {
        let dir = std::env::temp_dir().join("crab-config-secret");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "api_key = \"sk-secret\"\n").unwrap();
        let err = load(ws(), Some(&path), PartialConfig::default()).unwrap_err();
        assert!(err.contains("api_key in the config file is not supported"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn overlay_flags_win_over_file() {
        let file = PartialConfig {
            model: Some("file-model".into()),
            max_iterations: Some(5),
            ..Default::default()
        };
        let mut merged = file.clone();
        let flags = PartialConfig {
            model: Some("flag-model".into()),
            ..Default::default()
        };
        merged.overlay(&flags);
        assert_eq!(merged.model.as_deref(), Some("flag-model"));
        assert_eq!(merged.max_iterations, Some(5));
    }
}
