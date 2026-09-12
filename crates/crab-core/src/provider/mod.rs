//! LLM provider integration (CRAB-103), async on rig-core (CRAB-130).
//!
//! Providers implement the `Provider` trait; the agent loop is
//! provider-agnostic. The real HTTP clients live in `provider/rig.rs`, a thin
//! adapter over rig-core 0.42's unified provider contracts (OpenAI
//! chat-completions serving OpenAI + DeepSeek via a configurable base URL,
//! and Anthropic Messages); `fake` is an in-memory scripted provider for
//! offline tests. Adding a provider means extending the adapter and
//! registering it in `from_config`.

pub mod fake;
pub mod rig;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::Config;

/// A tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value, // JSON object of arguments
}

/// A canonical message in the agent history. Provider implementations convert
/// these to and from their own wire format.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System(String),
    User(String),
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        tool_call_id: String,
        result: String,
    },
}

/// The outcome of a completion: final text, or one or more tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Text(String),
    ToolCalls(Vec<ToolCall>),
    /// Tool calls from a response that was cut off by the output token limit;
    /// their arguments may be incomplete and must not be executed.
    TruncatedToolCalls(Vec<ToolCall>),
}

/// A provider completion: the response plus the exact prompt/input token count
/// reported by the provider. The token count anchors context budgeting; it is
/// `None` when the provider does not report it (e.g. the fake provider).
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub response: Response,
    pub prompt_tokens: Option<usize>,
    /// True when the generation was interrupted mid-stream (user steering).
    pub aborted: bool,
}

/// Typed provider errors, surfaced clearly at the loop boundary.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("authentication error: {0}")]
    Auth(String),
    #[error("request timed out: {0}")]
    Timeout(String),
    #[error("malformed response: {0}")]
    Malformed(String),
    #[error("http error: {0}")]
    Http(String),
}

/// A chat + tool-calling backend. Implementations must be `Send + Sync` so the
/// loop can hold them behind `Box<dyn Provider>`.
///
/// `complete` is async (CRAB-130) but the trait stays object-safe: it returns
/// a boxed future rather than using `async_trait`. `cancel` is checked
/// between streamed chunks (via `select!`); when cancelled, the request is
/// aborted and `Completion.aborted` is set. `on_text` receives text deltas as
/// they stream in. `effort_params` carries the provider-flavored wire
/// parameters for the current thinking level (`provider_effort`, CRAB-116) —
/// an empty object means "nothing to add".
pub trait Provider: Send + Sync {
    fn complete<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [Value],
        effort_params: &'a Value,
        cancel: CancellationToken,
        on_text: &'a mut (dyn FnMut(&str) + Send),
    ) -> BoxFuture<'a, Result<Completion, ProviderError>>;
}

/// Build the provider selected by `config`. OpenAI and DeepSeek share the
/// OpenAI chat-completions adapter (only base URL + model differ); Anthropic
/// uses its own Messages client; `fake` is the scripted offline provider.
/// Fails fast when a hosted provider has no key, instead of sending an empty
/// bearer token (CRAB-143).
pub fn from_config(config: &Config) -> Result<Box<dyn Provider>, String> {
    crate::config::ensure_api_key(config)?;
    Ok(match config.provider.name {
        "fake" => Box::new(fake::FakeProvider::new(vec![])),
        _ => Box::new(rig::RigProvider::new(config)),
    })
}

/// Map an HTTP status code to a typed provider error.
pub(crate) fn map_status_error(code: u16, text: String) -> ProviderError {
    if is_quota_or_billing(&text) {
        ProviderError::Http(format!("quota or billing limit (status {code}): {text}"))
    } else if code == 401 || code == 403 {
        ProviderError::Auth(format!("status {code}: {text}"))
    } else if code == 408 || code == 429 {
        ProviderError::Timeout(format!("status {code}: {text}"))
    } else {
        ProviderError::Http(format!("status {code}: {text}"))
    }
}

/// True when a status code indicates a transient failure worth retrying.
pub(crate) fn is_transient_status(code: u16) -> bool {
    code == 408 || code == 429 || (500..=599).contains(&code)
}

/// True when an error body indicates a quota/billing limit rather than a
/// transient failure, so it is never retried (mirrors pi's non-retryable
/// provider-limit patterns in `ai/src/utils/retry.ts`).
pub(crate) fn is_quota_or_billing(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    [
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
        "usage limit",
        "available balance",
        "free usage limit",
        "monthly usage",
    ]
    .iter()
    .any(|p| t.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_errors_are_not_labeled_timeout() {
        let e = map_status_error(429, "insufficient_quota".into());
        assert!(matches!(e, ProviderError::Http(_)));
        assert!(e.to_string().contains("quota or billing"));
    }

    #[test]
    fn transient_429_still_maps_to_timeout() {
        let e = map_status_error(429, "rate limit exceeded".into());
        assert!(matches!(e, ProviderError::Timeout(_)));
    }

    #[test]
    fn from_config_builds_each_kind() {
        for info in crate::config::PROVIDERS {
            let mut config = Config::defaults(std::env::temp_dir());
            config.provider = info;
            // Hosted providers require a key (CRAB-143).
            config.api_key = Some("test-key".into());
            drop(from_config(&config).unwrap());
        }
    }

    #[test]
    fn from_config_rejects_a_missing_key() {
        let mut config = Config::defaults(std::env::temp_dir());
        config.model = "gpt-x".into();
        config.api_key = None;
        match from_config(&config) {
            Ok(_) => panic!("a key-requiring provider must not build without a key"),
            Err(e) => assert!(e.contains("no API key for provider 'openai'"), "{e}"),
        }
    }
}
