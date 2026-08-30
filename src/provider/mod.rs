//! LLM provider integration (CRAB-103).
//!
//! Providers implement the `Provider` trait; the agent loop is
//! provider-agnostic. `openai` (OpenAI chat-completions + tools, also serving
//! DeepSeek via a configurable base URL) and `anthropic` (Messages API) are
//! real HTTP clients; `fake` is an in-memory scripted provider for offline
//! tests. Adding a provider means implementing `Provider` and registering it in
//! `from_config`.

pub mod anthropic;
pub mod fake;
pub mod openai;

use std::fmt;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{Config, ProviderKind};

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
#[derive(Debug)]
pub enum ProviderError {
    Auth(String),
    Timeout(String),
    Malformed(String),
    Http(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Auth(m) => write!(f, "authentication error: {m}"),
            ProviderError::Timeout(m) => write!(f, "request timed out: {m}"),
            ProviderError::Malformed(m) => write!(f, "malformed response: {m}"),
            ProviderError::Http(m) => write!(f, "http error: {m}"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// A chat + tool-calling backend. Implementations must be `Send + Sync` so the
/// loop can hold them behind `Box<dyn Provider>`.
pub trait Provider: Send + Sync {
    /// Send `history` plus the tool schemas and return the completion (text or
    /// tool calls) together with any reported prompt token usage. `cancel` is
    /// checked between streamed chunks; when set, the request is aborted and
    /// `Completion.aborted` is set.
    fn complete(
        &self,
        history: &[Message],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<Completion, ProviderError>;
}

/// Build the provider selected by `config`. DeepSeek reuses the OpenAI client
/// (only base URL + model differ); Anthropic is its own client.
pub fn from_config(config: &Config) -> Box<dyn Provider> {
    match config.provider {
        ProviderKind::Openai | ProviderKind::Deepseek => {
            Box::new(openai::OpenAIProvider::new(config))
        }
        ProviderKind::Anthropic => Box::new(anthropic::AnthropicProvider::new(config)),
        ProviderKind::Fake => Box::new(fake::FakeProvider::new(vec![])),
    }
}

/// Map an HTTP status code to a typed provider error.
fn map_status_error(code: u16, text: String) -> ProviderError {
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
fn is_transient_status(code: u16) -> bool {
    code == 408 || code == 429 || (500..=599).contains(&code)
}

/// True when an error body indicates a quota/billing limit rather than a
/// transient failure, so it is never retried (mirrors pi's non-retryable
/// provider-limit patterns in `ai/src/utils/retry.ts`).
fn is_quota_or_billing(text: &str) -> bool {
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

/// Classify a transport error, separating timeouts from generic HTTP failures.
fn map_transport(t: &ureq::Transport) -> ProviderError {
    let msg = t.to_string();
    if t.kind() == ureq::ErrorKind::Io && msg.to_ascii_lowercase().contains("timed") {
        ProviderError::Timeout(msg)
    } else {
        ProviderError::Http(msg)
    }
}

fn backoff(attempt: usize) {
    let ms = 250u64 << attempt.min(6);
    std::thread::sleep(Duration::from_millis(ms));
}

/// Send a JSON body and return the response body as a blocking reader for
/// streaming (SSE), with the same timeout + retry policy as the old
/// non-streaming `post_json`.
fn post_stream(
    url: &str,
    headers: &[(&str, &str)],
    body: Value,
    timeout_secs: u64,
    max_retries: usize,
) -> Result<Box<dyn Read + Send + Sync>, ProviderError> {
    let mut attempt = 0usize;
    loop {
        let mut req = ureq::post(url).set("Content-Type", "application/json");
        for (k, v) in headers {
            req = req.set(k, v);
        }
        req = req.timeout(Duration::from_secs(timeout_secs));

        match req.send_json(body.clone()) {
            Ok(resp) => return Ok(resp.into_reader()),
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                if is_quota_or_billing(&text)
                    || !is_transient_status(code)
                    || attempt >= max_retries
                {
                    return Err(map_status_error(code, text));
                }
                attempt += 1;
                backoff(attempt);
            }
            Err(ureq::Error::Transport(t)) => {
                if attempt >= max_retries {
                    return Err(map_transport(&t));
                }
                attempt += 1;
                backoff(attempt);
            }
        }
    }
}

/// Read the next SSE `data:` payload as a parsed JSON value, returning `None`
/// at end of stream or `[DONE]`, or when `cancel` is set. `event:` lines,
/// comments, and empty lines are skipped.
fn next_sse_event(
    reader: &mut impl std::io::BufRead,
    cancel: &AtomicBool,
) -> Result<Option<Value>, ProviderError> {
    let mut line = String::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| ProviderError::Http(e.to_string()))?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if let Some(data) = trimmed.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                return Ok(None);
            }
            if data.is_empty() {
                continue;
            }
            let value: Value =
                serde_json::from_str(data).map_err(|e| ProviderError::Malformed(e.to_string()))?;
            return Ok(Some(value));
        }
    }
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
}
