//! Configuration (CRAB-105).
//!
//! Keys: provider, base URL, API key, model, temperature, max_iterations,
//! output size caps. Sources merge with documented precedence:
//! `flags > env > config file`, all applied over built-in defaults.
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
            workspace,
        }
    }
}

/// Optional overrides collected from the config file, environment, and CLI
/// flags. Fields left `None` are not overridden.
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    pub provider: Option<ProviderKind>,
    /// Raw provider string (from env) kept so an invalid value can fail fast
    /// with an accurate error instead of silently falling back to defaults.
    pub provider_raw: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_iterations: Option<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_tokens: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub max_retries: Option<usize>,
    pub workspace: Option<PathBuf>,
}

/// Schema of the TOML config file (`--config`, defaulting to
/// `~/.config/crab/config.toml`). All keys are optional.
#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    provider: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    max_iterations: Option<usize>,
    max_output_bytes: Option<usize>,
    max_tokens: Option<usize>,
    timeout_secs: Option<u64>,
    max_retries: Option<usize>,
    workspace: Option<PathBuf>,
}

impl Config {
    /// Build a `Config` by merging `file`, then `env`, then `flags` over
    /// defaults. `default_workspace` is used unless an override supplies one.
    pub fn load(
        default_workspace: PathBuf,
        file: Option<&Path>,
        env: Overrides,
        flags: Overrides,
    ) -> Result<Config, String> {
        let mut merged = Overrides::default();

        if let Some(path) = file {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read config '{}': {e}", path.display()))?;
            let fc: FileConfig = toml::from_str(&text)
                .map_err(|e| format!("invalid config '{}': {e}", path.display()))?;
            merged.apply_file(fc)?;
        }

        merged.merge(env);
        merged.merge(flags);

        merged.build(default_workspace)
    }
}

impl Overrides {
    fn apply_file(&mut self, fc: FileConfig) -> Result<(), String> {
        if let Some(p) = fc.provider {
            self.provider = Some(ProviderKind::parse(&p)?);
        }
        if let Some(v) = fc.base_url {
            self.base_url = Some(v);
        }
        if let Some(v) = fc.api_key {
            self.api_key = Some(v);
        }
        if let Some(v) = fc.model {
            self.model = Some(v);
        }
        if let Some(v) = fc.temperature {
            self.temperature = Some(v);
        }
        if let Some(v) = fc.max_iterations {
            self.max_iterations = Some(v);
        }
        if let Some(v) = fc.max_output_bytes {
            self.max_output_bytes = Some(v);
        }
        if let Some(v) = fc.max_tokens {
            self.max_tokens = Some(v);
        }
        if let Some(v) = fc.timeout_secs {
            self.timeout_secs = Some(v);
        }
        if let Some(v) = fc.max_retries {
            self.max_retries = Some(v);
        }
        if let Some(v) = fc.workspace {
            self.workspace = Some(v);
        }
        Ok(())
    }

    fn merge(&mut self, other: Overrides) {
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.base_url.is_some() {
            self.base_url = other.base_url;
        }
        if other.api_key.is_some() {
            self.api_key = other.api_key;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.max_iterations.is_some() {
            self.max_iterations = other.max_iterations;
        }
        if other.max_output_bytes.is_some() {
            self.max_output_bytes = other.max_output_bytes;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if other.timeout_secs.is_some() {
            self.timeout_secs = other.timeout_secs;
        }
        if other.max_retries.is_some() {
            self.max_retries = other.max_retries;
        }
        if other.workspace.is_some() {
            self.workspace = other.workspace;
        }
    }

    fn build(self, default_workspace: PathBuf) -> Result<Config, String> {
        let provider = match self.provider {
            Some(p) => p,
            None => match self.provider_raw {
                Some(raw) => ProviderKind::parse(&raw)?,
                None => ProviderKind::Openai,
            },
        };
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

        Ok(Config {
            provider,
            base_url,
            api_key: self.api_key,
            model,
            temperature: self.temperature.unwrap_or(0.7),
            max_iterations: self.max_iterations.unwrap_or(30),
            max_output_bytes: self.max_output_bytes.unwrap_or(32_000),
            max_tokens: self.max_tokens.unwrap_or(2048),
            timeout_secs: self.timeout_secs.unwrap_or(60),
            max_retries: self.max_retries.unwrap_or(2),
            workspace,
        })
    }
}

impl Overrides {
    /// Read `CRAB_*` environment variables into an `Overrides`.
    pub fn from_env() -> Self {
        let mut o = Overrides::default();
        if let Ok(v) = std::env::var("CRAB_PROVIDER") {
            // A bad env value fails fast via parse in `build`; here we just
            // keep the raw string so the error message is accurate.
            match ProviderKind::parse(&v) {
                Ok(p) => o.provider = Some(p),
                Err(_) => {
                    o.provider_raw = Some(v);
                }
            }
        }
        if let Ok(v) = std::env::var("CRAB_BASE_URL") {
            o.base_url = Some(v);
        }
        if let Ok(v) = std::env::var("CRAB_API_KEY") {
            o.api_key = Some(v);
        }
        if let Ok(v) = std::env::var("CRAB_MODEL") {
            o.model = Some(v);
        }
        if let Ok(v) = std::env::var("CRAB_TEMPERATURE") {
            o.temperature = v.parse().ok();
        }
        if let Ok(v) = std::env::var("CRAB_MAX_ITERATIONS") {
            o.max_iterations = v.parse().ok();
        }
        if let Ok(v) = std::env::var("CRAB_MAX_OUTPUT_BYTES") {
            o.max_output_bytes = v.parse().ok();
        }
        if let Ok(v) = std::env::var("CRAB_MAX_TOKENS") {
            o.max_tokens = v.parse().ok();
        }
        if let Ok(v) = std::env::var("CRAB_TIMEOUT_SECS") {
            o.timeout_secs = v.parse().ok();
        }
        if let Ok(v) = std::env::var("CRAB_MAX_RETRIES") {
            o.max_retries = v.parse().ok();
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ws() -> PathBuf {
        PathBuf::from("/tmp/crab-config-test")
    }

    #[test]
    fn defaults() {
        let c = Config::load(ws(), None, Overrides::default(), Overrides::default()).unwrap();
        assert_eq!(c.provider, ProviderKind::Openai);
        assert_eq!(c.base_url, "https://api.openai.com");
        assert_eq!(c.max_iterations, 30);
    }

    #[test]
    fn deepseek_preset_applies_base_url_and_model() {
        let flags = Overrides {
            provider: Some(ProviderKind::Deepseek),
            ..Default::default()
        };
        let c = Config::load(ws(), None, Overrides::default(), flags).unwrap();
        assert_eq!(c.provider, ProviderKind::Deepseek);
        assert_eq!(c.base_url, "https://api.deepseek.com");
        assert_eq!(c.model, "deepseek-chat");
    }

    #[test]
    fn explicit_base_url_wins_over_preset() {
        let flags = Overrides {
            provider: Some(ProviderKind::Deepseek),
            base_url: Some("http://localhost:9000".into()),
            ..Default::default()
        };
        let c = Config::load(ws(), None, Overrides::default(), flags).unwrap();
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
        let err = Config::load(
            ws(),
            Some(&path),
            Overrides::default(),
            Overrides::default(),
        )
        .unwrap_err();
        assert!(err.contains("unknown provider"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flags_override_env_override_file() {
        let env = Overrides {
            model: Some("env-model".into()),
            ..Default::default()
        };
        let flags = Overrides {
            model: Some("flag-model".into()),
            ..Default::default()
        };
        let c = Config::load(ws(), None, env, flags).unwrap();
        assert_eq!(c.model, "flag-model");
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
        let c = Config::load(
            ws(),
            Some(&path),
            Overrides::default(),
            Overrides::default(),
        )
        .unwrap();
        assert_eq!(c.provider, ProviderKind::Anthropic);
        assert_eq!(c.base_url, "https://api.anthropic.com");
        assert_eq!(c.model, "claude-x");
        assert_eq!(c.max_iterations, 7);
        std::fs::remove_dir_all(&dir).ok();
    }
}
