//! Configuration.
//!
//! Keys: provider, base URL, model, temperature, max_iterations, output size
//! caps, workspace, session retention. Sources merge with documented
//! precedence: `flags > config file > defaults` (the `DAEDALUS_*` env layer was
//! removed). API keys are **not** a config key: they are resolved
//! separately via `credential` (`--api-key` > provider-native env > OS
//! keyring), so a secret can never be stored in the config file.
//!
//! Supported providers: a registry (`PROVIDERS`) grown from the original
//! `openai`/`anthropic`/`deepseek` — provider + model +
//! optional base URL are config-driven strings validated against the registry,
//! and unknown values fail fast listing the supported set. The adapter
//! (`provider/rig.rs`) maps each registry entry onto rig-core's client for
//! that provider; adding a provider is one table row, no client code.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::mcp::{validate_servers, McpServerConfig};
use crate::theme::{Theme, ThemePartial};

/// How a provider's canonical `Effort` level maps to wire parameters.
/// `None` = the provider gets no effort params (unsupported or
/// unknown semantics — capability honesty over guesswork).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortStyle {
    /// OpenAI `reasoning_effort` (OpenAI-family chat-completions wires).
    OpenaiEffort,
    /// Anthropic `thinking` block with a token budget.
    AnthropicThinking,
    /// No effort parameters.
    None,
}

/// One registry entry: the wire facts daedalus knows about a provider —
/// endpoint, credential surface, and effort style. Deliberately **no model
/// presets**: model catalogs go stale and local
/// providers have no meaningful default — the user must choose a model.
/// Adding a provider is a row here plus (if rig has a dedicated client) one
/// match arm in the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub preset_base_url: &'static str,
    /// Provider-native env var carrying the API key. `None` for
    /// local providers and the fake.
    pub api_key_env: Option<&'static str>,
    pub effort: EffortStyle,
}

/// The provider registry. The first row is the default provider
/// (`openai`) — used when no `provider` is configured.
pub const PROVIDERS: &[ProviderInfo] = &[
    ProviderInfo {
        name: "openai",
        preset_base_url: "https://api.openai.com",
        api_key_env: Some("OPENAI_API_KEY"),
        effort: EffortStyle::OpenaiEffort,
    },
    ProviderInfo {
        name: "deepseek",
        preset_base_url: "https://api.deepseek.com",
        api_key_env: Some("DEEPSEEK_API_KEY"),
        effort: EffortStyle::OpenaiEffort,
    },
    ProviderInfo {
        name: "anthropic",
        preset_base_url: "https://api.anthropic.com",
        api_key_env: Some("ANTHROPIC_API_KEY"),
        effort: EffortStyle::AnthropicThinking,
    },
    ProviderInfo {
        name: "gemini",
        preset_base_url: "https://generativelanguage.googleapis.com",
        api_key_env: Some("GEMINI_API_KEY"),
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "mistral",
        preset_base_url: "https://api.mistral.ai",
        api_key_env: Some("MISTRAL_API_KEY"),
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "groq",
        preset_base_url: "https://api.groq.com/openai/v1",
        api_key_env: Some("GROQ_API_KEY"),
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "xai",
        preset_base_url: "https://api.x.ai",
        api_key_env: Some("XAI_API_KEY"),
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "openrouter",
        preset_base_url: "https://openrouter.ai/api/v1",
        api_key_env: Some("OPENROUTER_API_KEY"),
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "ollama",
        preset_base_url: "http://localhost:11434",
        api_key_env: None,
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "lmstudio",
        preset_base_url: "http://localhost:1234",
        api_key_env: None,
        // Whatever model the user has loaded in LM Studio — no presets.
        effort: EffortStyle::None,
    },
    ProviderInfo {
        name: "fake",
        preset_base_url: "",
        api_key_env: None,
        effort: EffortStyle::None,
    },
];

/// Conservative context-window fallback (input tokens) used when the config
/// does not set `max_context_tokens`. It is model-independent on purpose:
/// daedalus does not know which model the user picked. Set `max_context_tokens`
/// to match a larger model.
pub const DEFAULT_CONTEXT_WINDOW: usize = 32_000;

/// The default provider (`openai`, the first registry row).
pub fn default_provider() -> &'static ProviderInfo {
    &PROVIDERS[0]
}

/// Look up a provider by name (case-insensitive), failing with the supported
/// list — the registry is the single source of truth for valid names.
pub fn provider_by_name(s: &str) -> Result<&'static ProviderInfo, String> {
    let wanted = s.trim().to_ascii_lowercase();
    PROVIDERS.iter().find(|p| p.name == wanted).ok_or_else(|| {
        format!(
            "unknown provider '{s}' (supported: {})",
            PROVIDERS
                .iter()
                .map(|p| p.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

impl ProviderInfo {
    /// True when the provider needs an API key. Local providers (Ollama,
    /// LM Studio) and the fake do not.
    pub fn requires_key(&self) -> bool {
        self.api_key_env.is_some()
    }
}

/// Fully-resolved configuration passed to the provider and the agent loop.
#[derive(Debug, Clone)]
pub struct Config {
    pub provider: &'static ProviderInfo,
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
    /// Default timeout for `bash` when the model omits one, in seconds.
    /// `0` disables the default (commands may then run indefinitely).
    pub bash_timeout_secs: u64,
    /// Number of retries for transient failures (timeouts, 429, 5xx).
    pub max_retries: usize,
    /// Rough token budget for the conversation history. The oldest tool turns
    /// are dropped once exceeded so a session cannot blow the model context.
    pub max_context_tokens: usize,
    /// Sessions to keep per workspace; older ones are pruned at startup.
    /// `0` disables pruning (keep every session).
    pub session_retention: usize,
    /// The single root directory the agent is allowed to touch.
    pub workspace: PathBuf,
    /// External MCP tool servers, started at runtime construction.
    pub mcp_servers: Vec<McpServerConfig>,
    /// Optional identity -> workspace map: a request identity (e.g.
    /// a Tailscale login header) selects a session workspace. Empty = use
    /// `workspace` for every session.
    pub identity_workspaces: BTreeMap<String, PathBuf>,
    /// The resolved TUI theme.
    pub theme: Theme,
}

impl Config {
    /// Built-in defaults, before any file/env/flag override.
    pub fn defaults(workspace: PathBuf) -> Self {
        Self::from_provider(default_provider(), workspace)
    }

    /// Defaults derived from one provider's registry row. The model is
    /// deliberately empty: choosing a model is the user's decision and is
    /// enforced by [`PartialConfig::resolve`].
    pub fn from_provider(provider: &'static ProviderInfo, workspace: PathBuf) -> Self {
        Self {
            provider,
            base_url: provider.preset_base_url.to_string(),
            api_key: None,
            model: String::new(),
            temperature: 0.7,
            max_iterations: 30,
            max_output_bytes: 32_000,
            max_tokens: 2048,
            timeout_secs: 60,
            bash_timeout_secs: 120,
            max_retries: 2,
            // Leave headroom for the completion output (max_tokens).
            max_context_tokens: DEFAULT_CONTEXT_WINDOW.saturating_sub(4_096),
            session_retention: 10,
            workspace,
            mcp_servers: Vec::new(),
            identity_workspaces: BTreeMap::new(),
            theme: Theme::dark(),
        }
    }

    /// The `bash` default timeout, or `None` when disabled (`0`).
    pub fn bash_default_timeout(&self) -> Option<u64> {
        (self.bash_timeout_secs > 0).then_some(self.bash_timeout_secs)
    }
}

/// Partial configuration from one source (config file or CLI flags). All
/// fields are optional; unresolved fields fall back to defaults or to the
/// next-higher-precedence source. Doubles as the schema of the TOML config
/// file and as the flag overrides — `api_key` is deliberately absent so a
/// secret can never be stored in the config file.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct PartialConfig {
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_iterations: Option<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_tokens: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub bash_timeout_secs: Option<u64>,
    pub max_retries: Option<usize>,
    pub max_context_tokens: Option<usize>,
    /// Sessions to keep per workspace at startup (default 10; `0` = keep all).
    pub session_retention: Option<usize>,
    pub workspace: Option<PathBuf>,
    /// `[[mcp_servers]]` tables.
    #[serde(default)]
    pub mcp_servers: Option<Vec<McpServerConfig>>,
    /// `[identities]` table: identity -> workspace directory.
    #[serde(default, rename = "identities")]
    pub identity_workspaces: Option<BTreeMap<String, PathBuf>>,
    /// `[theme]` table.
    #[serde(default)]
    pub theme: Option<ThemePartial>,
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
        take!(bash_timeout_secs);
        take!(max_retries);
        take!(max_context_tokens);
        take!(session_retention);
        take!(workspace);
        take!(mcp_servers);
        take!(identity_workspaces);
        // The theme table merges per field: a flag name must not discard the
        // file's per-token overrides.
        if let Some(higher_theme) = &higher.theme {
            match &mut self.theme {
                Some(t) => t.overlay(higher_theme),
                None => self.theme = Some(higher_theme.clone()),
            }
        }
    }

    /// Resolve to a complete `Config`: fill defaults (provider presets) and
    /// validate, attaching the resolved `api_key`.
    pub fn resolve(
        self,
        default_workspace: PathBuf,
        api_key: Option<String>,
    ) -> Result<Config, String> {
        let provider = match self.provider.as_deref() {
            Some(name) => provider_by_name(name)?,
            None => default_provider(),
        };
        // base_url defaults to the provider preset unless explicitly set.
        let base_url = self
            .base_url
            .unwrap_or_else(|| provider.preset_base_url.to_string());
        // The model is the user's choice — there is no preset.
        let model = match self.model.as_deref() {
            Some(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => {
                return Err(format!(
                    "no model configured for provider '{}': set `model` in the config file or pass --model",
                    provider.name
                ))
            }
        };
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
            .unwrap_or_else(|| DEFAULT_CONTEXT_WINDOW.saturating_sub(4_096));
        if max_context_tokens == 0 {
            return Err("max_context_tokens must be >= 1".into());
        }
        let bash_timeout_secs = self.bash_timeout_secs.unwrap_or(120);
        let mcp_servers = self.mcp_servers.unwrap_or_default();
        validate_servers(&mcp_servers)?;
        let identity_workspaces = self.identity_workspaces.unwrap_or_default();
        for (identity, path) in &identity_workspaces {
            if identity.trim().is_empty() {
                return Err("identities entries need a non-empty identity".into());
            }
            if path.as_os_str().is_empty() {
                return Err(format!(
                    "identity '{identity}' maps to an empty workspace path"
                ));
            }
        }

        let theme_partial = self.theme.clone().unwrap_or_default();
        let theme = crate::theme::resolve(&theme_partial, crate::theme::detect_scheme())?;

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
            bash_timeout_secs,
            max_retries: self.max_retries.unwrap_or(2),
            max_context_tokens,
            session_retention: self.session_retention.unwrap_or(10),
            workspace,
            mcp_servers,
            identity_workspaces,
            theme,
        })
    }
}

impl Config {
    /// Build a `Config` by merging the config `file`, then CLI `flags` over
    /// defaults — `flags > config file > defaults` (the env layer was
    /// removed). `api_key` is resolved separately by the caller via
    /// `crate::credential::resolve_api_key` and passed in; the config file
    /// cannot carry a secret. `default_workspace` is used unless an override
    /// supplies one.
    /// Build a `Config` by merging the config `file`, then CLI `flags` over
    /// defaults — `flags > config file > defaults` (the env layer was
    /// removed). `api_key_flag` feeds the resolution chain
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
        let info = match merged.provider.as_deref() {
            Some(name) => provider_by_name(name)?,
            None => default_provider(),
        };
        let api_key = crate::credential::resolve_api_key(info, api_key_flag);
        merged.resolve(default_workspace, api_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ws() -> PathBuf {
        PathBuf::from("/tmp/daedalus-config-test")
    }

    fn load(ws: PathBuf, file: Option<&Path>, flags: PartialConfig) -> Result<Config, String> {
        Config::load(ws, file, flags, None)
    }

    #[test]
    fn defaults() {
        let flags = PartialConfig {
            model: Some("test-model".into()),
            ..Default::default()
        };
        let c = load(ws(), None, flags).unwrap();
        assert_eq!(c.provider.name, "openai");
        assert_eq!(c.base_url, "https://api.openai.com");
        assert_eq!(c.api_key, None);
        assert_eq!(c.max_iterations, 30);
        assert_eq!(c.max_context_tokens, DEFAULT_CONTEXT_WINDOW - 4_096);
        assert_eq!(c.session_retention, 10);
    }

    #[test]
    fn session_retention_is_overridable_and_zero_is_allowed() {
        let flags = PartialConfig {
            model: Some("test-model".into()),
            session_retention: Some(3),
            ..Default::default()
        };
        assert_eq!(load(ws(), None, flags).unwrap().session_retention, 3);
        // `0` is the documented "keep everything" value, not an error.
        let flags = PartialConfig {
            model: Some("test-model".into()),
            session_retention: Some(0),
            ..Default::default()
        };
        assert_eq!(load(ws(), None, flags).unwrap().session_retention, 0);
    }

    #[test]
    fn missing_model_fails_with_guidance() {
        // The model is the user's choice — there is no preset.
        let err = load(ws(), None, PartialConfig::default()).unwrap_err();
        assert!(err.contains("no model configured"), "{err}");
        assert!(err.contains("openai"), "{err}");
    }

    #[test]
    fn bash_timeout_defaults_and_can_be_disabled() {
        let mut c = Config::defaults(ws());
        assert_eq!(c.bash_timeout_secs, 120);
        assert_eq!(c.bash_default_timeout(), Some(120));
        c.bash_timeout_secs = 0;
        assert_eq!(c.bash_default_timeout(), None);
        // A config-file/flag value resolves through.
        let flags = PartialConfig {
            model: Some("m".into()),
            bash_timeout_secs: Some(7),
            ..Default::default()
        };
        assert_eq!(load(ws(), None, flags).unwrap().bash_timeout_secs, 7);
    }

    #[test]
    fn theme_table_resolves_and_overlays() {
        let dir = std::env::temp_dir().join("daedalus-config-theme");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "model = \"m\"\n[theme]\nname = \"light\"\n[theme.colors]\nuser = \"red\"\n",
        )
        .unwrap();
        let c = load(ws(), Some(&path), PartialConfig::default()).unwrap();
        assert_eq!(c.theme.name, "light");
        assert_eq!(
            c.theme.token(crate::theme::Token::User).fg,
            crate::theme::ThemeColor::Indexed(1)
        );
        // A `--theme` name merges over the file but keeps its token overrides.
        let flags = PartialConfig {
            theme: Some(crate::theme::ThemePartial {
                name: Some("dark".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let c = load(ws(), Some(&path), flags).unwrap();
        assert_eq!(c.theme.name, "dark");
        assert_eq!(
            c.theme.token(crate::theme::Token::User).fg,
            crate::theme::ThemeColor::Indexed(1)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deepseek_preset_applies_base_url() {
        let flags = PartialConfig {
            provider: Some("deepseek".into()),
            model: Some("deepseek-chat".into()),
            ..Default::default()
        };
        let c = load(ws(), None, flags).unwrap();
        assert_eq!(c.provider.name, "deepseek");
        assert_eq!(c.base_url, "https://api.deepseek.com");
        assert_eq!(c.model, "deepseek-chat");
        assert_eq!(c.max_context_tokens, DEFAULT_CONTEXT_WINDOW - 4_096);
    }

    #[test]
    fn explicit_base_url_wins_over_preset() {
        let flags = PartialConfig {
            provider: Some("deepseek".into()),
            base_url: Some("http://localhost:9000".into()),
            model: Some("m".into()),
            ..Default::default()
        };
        let c = load(ws(), None, flags).unwrap();
        assert_eq!(c.base_url, "http://localhost:9000");
    }

    #[test]
    fn unknown_provider_fails() {
        let err = provider_by_name("wat").unwrap_err();
        assert!(err.contains("openai") && err.contains("ollama"));
    }

    #[test]
    fn unknown_provider_in_file_fails() {
        let dir = std::env::temp_dir().join("daedalus-config-bad-provider");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "provider = \"nope\"\n").unwrap();
        let err = load(ws(), Some(&path), PartialConfig::default()).unwrap_err();
        assert!(err.contains("unknown provider"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flags_override_file() {
        let dir = std::env::temp_dir().join("daedalus-config-flag-file");
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
        let dir = std::env::temp_dir().join("daedalus-config-file-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "provider = \"anthropic\"\nmodel = \"claude-x\"\nmax_iterations = 7\n",
        )
        .unwrap();
        let c = load(ws(), Some(&path), PartialConfig::default()).unwrap();
        assert_eq!(c.provider.name, "anthropic");
        assert_eq!(c.base_url, "https://api.anthropic.com");
        assert_eq!(c.model, "claude-x");
        assert_eq!(c.max_iterations, 7);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn api_key_in_config_file_is_rejected() {
        let dir = std::env::temp_dir().join("daedalus-config-secret");
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

    #[test]
    fn config_file_parses_mcp_servers() {
        let dir = std::env::temp_dir().join("daedalus-config-mcp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"
model = "m"
[[mcp_servers]]
name = "fs"
command = "npx"
args = ["-y", "server"]
[[mcp_servers]]
name = "remote"
transport = "http"
url = "http://localhost:8000/mcp"
"#,
        )
        .unwrap();
        let c = load(ws(), Some(&path), PartialConfig::default()).unwrap();
        assert_eq!(c.mcp_servers.len(), 2);
        assert_eq!(c.mcp_servers[0].name, "fs");
        assert!(c.mcp_servers[0].enabled);
        assert_eq!(
            c.mcp_servers[1].transport,
            crate::mcp::McpTransportKind::Http
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalid_mcp_server_in_file_fails() {
        let dir = std::env::temp_dir().join("daedalus-config-mcp-bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        // A stdio server with no command is a validation error.
        std::fs::write(&path, "model = \"m\"\n[[mcp_servers]]\nname = \"fs\"\n").unwrap();
        let err = load(ws(), Some(&path), PartialConfig::default()).unwrap_err();
        assert!(err.contains("needs a 'command'"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_file_parses_identity_workspaces() {
        let dir = std::env::temp_dir().join("daedalus-config-identities");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "model = \"m\"\n[identities]\n\"alice@example.com\" = \"/home/alice/projects\"\n",
        )
        .unwrap();
        let c = load(ws(), Some(&path), PartialConfig::default()).unwrap();
        assert_eq!(
            c.identity_workspaces
                .get("alice@example.com")
                .map(PathBuf::as_path),
            Some(Path::new("/home/alice/projects"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
