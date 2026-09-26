//! LLM provider integration, async on rig-core.
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

/// One piece of a provider reasoning/thinking block. Mirrors the shapes the
/// supported wires emit: signed text (Anthropic), summary text, and opaque
/// encrypted/redacted payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningPart {
    /// Plain reasoning text with an optional provider signature.
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Provider-generated reasoning summary text.
    Summary(String),
    /// Provider-encrypted reasoning payload.
    Encrypted(String),
    /// Redacted reasoning payload preserved as opaque data.
    Redacted { data: String },
}

/// A provider reasoning/thinking block, preserved in history so it can be
/// replayed on the next request. Anthropic requires the signed thinking block
/// back when a `tool_use` loop continues, and DeepSeek's thinking mode expects
/// `reasoning_content` on assistant messages in a tool loop, so dropping these
/// makes the second request of a tool-using turn fail.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ReasoningBlock {
    /// Provider-issued durable handle, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Ordered parts of the block.
    #[serde(default)]
    pub content: Vec<ReasoningPart>,
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
        /// Provider thinking/reasoning to replay with this message.
        reasoning: Vec<ReasoningBlock>,
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
    /// Thinking/reasoning captured from the stream, for replay on the next
    /// request (see [`ReasoningBlock`]).
    pub reasoning: Vec<ReasoningBlock>,
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
    /// The request was cancelled by the caller. Distinct from an error so the
    /// runtime can report a clean interruption instead of a provider failure.
    #[error("cancelled")]
    Cancelled,
    /// The provider does not support the requested capability.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The provider rejected the configured model name. `supported` holds the
    /// ids the provider itself reported, in its own order (empty when it named
    /// none) — the provider is the only authority on what it serves, so this is
    /// never filled from a hard-coded catalog.
    #[error("unknown model '{requested}': {detail}")]
    InvalidModel {
        requested: String,
        supported: Vec<String>,
        detail: String,
    },
}

/// A chat + tool-calling backend. Implementations must be `Send + Sync` so the
/// loop can hold them behind `Box<dyn Provider>`.
///
/// `complete` is async but the trait stays object-safe: it returns
/// a boxed future rather than using `async_trait`. `cancel` is checked
/// between streamed chunks (via `select!`); when cancelled, the request is
/// aborted and `Completion.aborted` is set. `on_delta` receives streamed
/// fragments — assistant text and, for providers that surface it, model
/// reasoning (`StreamDelta`). `effort_params` carries the
/// provider-flavored wire parameters for the current thinking level
/// (`provider_effort`) — an empty object means "nothing to add".
pub trait Provider: Send + Sync {
    fn complete<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [Value],
        effort_params: &'a Value,
        cancel: CancellationToken,
        on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
    ) -> BoxFuture<'a, Result<Completion, ProviderError>>;

    /// Discover the provider's available model ids. The default reports the
    /// capability as unsupported, so only providers that can list need to
    /// implement it.
    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, ProviderError>> {
        Box::pin(async {
            Err(ProviderError::Unsupported(
                "provider does not support model listing".into(),
            ))
        })
    }
}

/// A streamed fragment of a completion: assistant `Text` or model `Thinking`
/// (reasoning). Providers that do not surface reasoning only emit `Text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    Text(String),
    Thinking(String),
}

/// Build the provider selected by `config`. OpenAI and DeepSeek share the
/// OpenAI chat-completions adapter (only base URL + model differ); Anthropic
/// uses its own Messages client; `fake` is the scripted offline provider.
///
/// A missing API key is not rejected here: the real provider refuses to make a
/// request without one, so the interactive frontends still start
/// and `/login` stays reachable.
pub fn from_config(config: &Config) -> Box<dyn Provider> {
    match config.provider.name {
        "fake" => Box::new(fake::FakeProvider::new(vec![])),
        _ => Box::new(rig::RigProvider::new(config)),
    }
}

/// Map an HTTP status code to a typed provider error. `configured` is the
/// model daedalus asked for, used when the provider's rejection does not name
/// it; the classification itself comes from the body.
pub(crate) fn map_status_error(code: u16, text: String, configured: Option<&str>) -> ProviderError {
    if let Some(e) = invalid_model_error(code, &text, configured) {
        return e;
    }
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

/// Recognise a provider rejection of an unknown model.
///
/// A model rejection has no structural signal — a 400 is also what a malformed
/// request gets — so the body has to be read. DeepSeek states it outright and
/// names the ids it does serve:
///
/// ```text
/// {"error":{"message":"The supported API model names are deepseek-flash,
///   deepseek-v4-pro, but you passed totally-bogus-xyz-42. (request_id: …)",
///   "type":"invalid_request_error","code":"invalid_request_error"}}
/// ```
///
/// Recognition is deliberately narrow (an explicit model-existence phrase), so
/// an unrelated 400 — a malformed tool schema, an over-long prompt — stays a
/// plain `Http` error, and the unmatched body is kept verbatim in `detail`, so
/// a failed guess loses nothing.
///
/// Note the provider's list is not the same as what it *accepts*: DeepSeek
/// serves `deepseek-chat` and `deepseek-reasoner` while naming neither, so this
/// list is a hint for a frontend, not a validation gate.
pub(crate) fn invalid_model_error(
    code: u16,
    text: &str,
    configured: Option<&str>,
) -> Option<ProviderError> {
    if code != 400 && code != 404 {
        return None;
    }
    const SUPPORTED: &str = "supported api model names are";
    const PASSED: &str = "but you passed";
    // Phrases marking a model-existence rejection; kept explicit so a 400 about
    // anything else cannot land here.
    const MARKERS: [&str; 5] = [
        "does not exist",
        "model not exist",
        "unknown model",
        "unsupported model",
        "model_not_found",
    ];
    let lower = text.to_ascii_lowercase();
    if !(lower.contains(SUPPORTED)
        || lower.contains(PASSED)
        || MARKERS.iter().any(|m| lower.contains(m)))
    {
        return None;
    }
    // `to_ascii_lowercase` leaves non-ASCII bytes untouched, so an index found
    // in `lower` is a valid char boundary in `text`.
    let supported = match lower.find(SUPPORTED) {
        Some(i) => {
            let rest = &text[i + SUPPORTED.len()..];
            let end = rest.to_ascii_lowercase().find(PASSED).unwrap_or(rest.len());
            rest[..end]
                .split(',')
                .map(|s| s.trim().trim_end_matches('.').trim())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        }
        None => Vec::new(),
    };
    let requested = (|| {
        let i = lower.find(PASSED)?;
        let rest =
            text[i + PASSED.len()..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
        let token = rest.split_whitespace().next()?;
        let token = token.trim_matches(|c: char| c == '\'' || c == '"' || c == '`');
        let token = token.trim_end_matches(['.', ',', ';']);
        (!token.is_empty()).then(|| token.to_string())
    })()
    .or_else(|| first_quoted(text))
    .or_else(|| configured.map(str::to_string))
    .unwrap_or_default();
    Some(ProviderError::InvalidModel {
        requested,
        supported,
        detail: text.to_string(),
    })
}

/// Words that appear quoted inside an error envelope but never name a model,
/// so a quoted envelope value is not mistaken for the requested one.
const NON_MODEL_QUOTED: [&str; 10] = [
    "error",
    "message",
    "type",
    "param",
    "code",
    "status",
    "invalid_request_error",
    "not_found_error",
    "model_not_found",
    "authentication_error",
];

/// The first `'…'` / `"…"` / `` `…` `` token in `text` that could be a model
/// name, for providers that quote the offending model instead of saying "but
/// you passed".
///
/// `text` is usually a whole JSON error envelope (`{"error":{"message":…}}`),
/// so a quoted *key* — recognised by the `:` that follows it — and the handful
/// of fixed envelope values are skipped: they are far more common than a quoted
/// model, and returning `"error"` as the requested model would be worse than
/// returning nothing.
fn first_quoted(text: &str) -> Option<String> {
    for q in ['\'', '"', '`'] {
        let mut from = 0usize;
        while let Some(i) = text[from..].find(q) {
            let start = from + i + q.len_utf8();
            let Some(j) = text[start..].find(q) else {
                break;
            };
            let token = &text[start..start + j];
            // Resume after this token's closing quote.
            from = start + j + q.len_utf8();
            // A quoted *name*, not a quoted sentence, a key, or envelope noise.
            if token.is_empty() || token.contains(char::is_whitespace) {
                continue;
            }
            if text[from..].trim_start().starts_with(':') {
                continue; // a JSON key, e.g. "error":
            }
            if NON_MODEL_QUOTED
                .iter()
                .any(|w| token.eq_ignore_ascii_case(w))
            {
                continue;
            }
            return Some(token.to_string());
        }
    }
    None
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
        let e = map_status_error(429, "insufficient_quota".into(), None);
        assert!(matches!(e, ProviderError::Http(_)));
        assert!(e.to_string().contains("quota or billing"));
    }

    #[test]
    fn transient_429_still_maps_to_timeout() {
        let e = map_status_error(429, "rate limit exceeded".into(), None);
        assert!(matches!(e, ProviderError::Timeout(_)));
    }

    #[test]
    fn unknown_model_carries_the_providers_supported_list() {
        // Verbatim body from a live DeepSeek 400 for a model it does not serve.
        let body = r#"{"error":{"message":"The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed totally-bogus-xyz-42. (request_id: 28c024b2-b4df-4e56-854e-e7750881e9da)","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}"#;
        let e = map_status_error(400, body.to_string(), Some("deepseek-chat"));
        match e {
            ProviderError::InvalidModel {
                requested,
                supported,
                ..
            } => {
                assert_eq!(requested, "totally-bogus-xyz-42");
                assert_eq!(supported, vec!["deepseek-flash", "deepseek-v4-pro"]);
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn unrelated_bad_request_is_not_an_unknown_model() {
        // A 400 about the body must stay a plain HTTP error: the model is fine.
        let body = r#"{"error":{"message":"invalid tool schema"}}"#;
        let e = map_status_error(400, body.to_string(), Some("deepseek-flash"));
        assert!(matches!(e, ProviderError::Http(_)), "got {e:?}");
    }

    #[test]
    fn unknown_model_falls_back_to_the_configured_name() {
        // No "but you passed", no provider list: the configured model names it.
        let body = r#"{"error":{"message":"The model does not exist"}}"#;
        let e = map_status_error(404, body.to_string(), Some("my-model"));
        match e {
            ProviderError::InvalidModel {
                requested,
                supported,
                ..
            } => {
                assert_eq!(requested, "my-model");
                assert!(supported.is_empty());
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn quoted_model_is_recognised_without_but_you_passed() {
        let body = r#"{"error":{"message":"The model 'gpt-99' does not exist"}}"#;
        let e = map_status_error(404, body.to_string(), Some("gpt-4"));
        match e {
            ProviderError::InvalidModel { requested, .. } => assert_eq!(requested, "gpt-99"),
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn quoted_envelope_keys_are_not_mistaken_for_the_model() {
        // `model_not_found` is a rejection marker *and* a quoted envelope
        // value: the configured model names the request, not the provider's
        // error code.
        let body = r#"{"error":{"message":"The model does not exist","code":"model_not_found"}}"#;
        let e = map_status_error(404, body.to_string(), Some("deepseek-v4-pro"));
        match e {
            ProviderError::InvalidModel { requested, .. } => {
                assert_eq!(requested, "deepseek-v4-pro")
            }
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn from_config_builds_each_kind() {
        for info in crate::config::PROVIDERS {
            let mut config = Config::defaults(std::env::temp_dir());
            config.provider = info;
            drop(from_config(&config));
        }
    }
}
