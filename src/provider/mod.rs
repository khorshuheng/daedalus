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
}

/// Typed provider errors, surfaced clearly at the loop boundary.
#[derive(Debug)]
pub enum ProviderError {
    Auth(String),
    Timeout(String),
    Malformed(String),
    Http(String),
    Message(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Auth(m) => write!(f, "authentication error: {m}"),
            ProviderError::Timeout(m) => write!(f, "request timed out: {m}"),
            ProviderError::Malformed(m) => write!(f, "malformed response: {m}"),
            ProviderError::Http(m) => write!(f, "http error: {m}"),
            ProviderError::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// A chat + tool-calling backend. Implementations must be `Send + Sync` so the
/// loop can hold them behind `Box<dyn Provider>`.
pub trait Provider: Send + Sync {
    /// Send `history` plus the tool schemas and return text or tool calls.
    fn complete(&self, history: &[Message], tools: &[Value]) -> Result<Response, ProviderError>;
}

/// Build the provider selected by `config`. DeepSeek reuses the OpenAI client
/// (only base URL + model differ); Anthropic is its own client.
pub fn from_config(config: &Config) -> Box<dyn Provider> {
    match config.provider {
        ProviderKind::Openai | ProviderKind::Deepseek => {
            Box::new(openai::OpenAIProvider::new(config))
        }
        ProviderKind::Anthropic => Box::new(anthropic::AnthropicProvider::new(config)),
    }
}
