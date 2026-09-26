//! The rig-core adapter: the only module in daedalus-core that names
//! rig. Maps daedalus's canonical message/response types to rig's provider
//! contracts and back, for the OpenAI chat-completions wire (serving both the
//! OpenAI and DeepSeek presets, as before) and the Anthropic Messages wire.
//!
//! Hard requirements carried over from the hand-written clients:
//! - `prompt_tokens` from provider-reported usage anchors context budgeting;
//!   missing usage degrades explicitly to `None` (rig's zero-valued sentinel).
//! - Quota/billing errors are never retried; transient statuses back off
//!   (the error classification, now over rig errors).
//! - Truncated tool calls (`finish_reason: length`) must not be executed.
//! - Effort/thinking wire parameters (`provider_effort`) flow into
//!   the request via `additional_params`.

use std::collections::HashSet;
use std::time::Duration;

use futures::StreamExt;
use rig_core::client::{CompletionClient, ModelListingClient};
use rig_core::completion::message::ToolCall as RigToolCall;
use rig_core::completion::message::{
    AssistantContent, Reasoning, ReasoningContent, Text, ToolCallId, ToolFunction, ToolResult,
    ToolResultContent, UserContent,
};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, FinishReason, ToolDefinition,
};
use rig_core::providers::{anthropic, gemini, groq, mistral, ollama, openai, openrouter, xai};
use rig_core::streaming::StreamedAssistantContent;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{
    is_quota_or_billing, is_transient_status, map_status_error, Completion, Message, Provider,
    ProviderError, ReasoningBlock, ReasoningPart, Response, StreamDelta, ToolCall,
};

/// Extra output-token headroom above the extended-thinking budget: Anthropic
/// requires `max_tokens` strictly greater than `budget_tokens`.
const THINKING_HEADROOM: usize = 1_024;

enum Backend {
    /// OpenAI chat-completions; also serves DeepSeek and LM Studio via a
    /// configurable base URL.
    OpenAiCompatible(openai::completion::GenericCompletionModel<openai::OpenAICompletionsExt>),
    /// Anthropic Messages API.
    Anthropic(anthropic::completion::CompletionModel),
    Gemini(gemini::completion::CompletionModel),
    Mistral(mistral::completion::CompletionModel),
    /// Groq rides rig's shared OpenAI-compatible transport.
    Groq(groq::CompletionModel),
    Xai(xai::completion::CompletionModel),
    OpenRouter(openrouter::completion::CompletionModel),
    Ollama(ollama::CompletionModel),
}

/// The rig-backed provider. Construct via [`RigProvider::openai_compatible`]
/// or [`RigProvider::anthropic`].
pub struct RigProvider {
    backend: Backend,
    model: String,
    /// Registry row: drives the key requirement and env-var name.
    provider: &'static crate::config::ProviderInfo,
    /// Resolved API key (empty when none was configured).
    api_key: String,
    /// Resolved base URL (for model listing; the backend is prebuilt).
    base_url: String,
    /// Configured completion cap. Anthropic requires it on every request.
    max_tokens: usize,
    /// Per-request timeout: bounds both connection/stream establishment and the
    /// gap between streamed items, so a stalled socket cannot wedge a turn.
    timeout: Duration,
    /// Transient-failure retries, from config (was a hardcoded constant).
    max_retries: usize,
}

impl RigProvider {
    /// Build the backend for `config.provider`. Each registry name
    /// maps onto rig's client for that provider; the configured `base_url` is
    /// always forwarded so a user override wins. Names with no dedicated rig
    /// client ride the OpenAI-compatible transport.
    pub fn new(config: &crate::config::Config) -> Self {
        let key = config.api_key.clone().unwrap_or_default();
        let base = config.base_url.trim_end_matches('/').to_string();
        let backend = match config.provider.name {
            "openai" | "deepseek" | "lmstudio" => {
                // These presets are API roots: rig expects the `/v1` prefix
                // on the base, so append it unless the user already did.
                let base = if base.ends_with("/v1") {
                    base.clone()
                } else {
                    format!("{base}/v1")
                };
                let client = openai::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid OpenAI-compatible client");
                Backend::OpenAiCompatible(
                    client
                        .completions_api()
                        .completion_model(config.model.clone()),
                )
            }
            "anthropic" => {
                let client = anthropic::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid Anthropic client");
                Backend::Anthropic(client.completion_model(config.model.clone()))
            }
            "gemini" => {
                let client = gemini::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid Gemini client");
                Backend::Gemini(client.completion_model(config.model.clone()))
            }
            "mistral" => {
                let client = mistral::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid Mistral client");
                Backend::Mistral(client.completion_model(config.model.clone()))
            }
            "groq" => {
                let client = groq::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid Groq client");
                Backend::Groq(client.completion_model(config.model.clone()))
            }
            "xai" => {
                let client = xai::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid xAI client");
                Backend::Xai(client.completion_model(config.model.clone()))
            }
            "openrouter" => {
                let client = openrouter::Client::builder()
                    .api_key(key.clone())
                    .base_url(base.clone())
                    .build()
                    .expect("valid OpenRouter client");
                Backend::OpenRouter(client.completion_model(config.model.clone()))
            }
            "ollama" => {
                // Ollama needs no key; an empty key means "no auth header".
                let client = ollama::Client::builder()
                    .api_key(String::new())
                    .base_url(base.clone())
                    .build()
                    .expect("valid Ollama client");
                Backend::Ollama(client.completion_model(config.model.clone()))
            }
            other => panic!(
                "no rig backend for provider '{other}' (config should have been rejected at load)"
            ),
        };
        Self {
            backend,
            model: config.model.clone(),
            provider: config.provider,
            api_key: key,
            base_url: base,
            max_tokens: config.max_tokens,
            timeout: Duration::from_secs(config.timeout_secs.max(1)),
            max_retries: config.max_retries,
        }
    }

    /// Map daedalus history to rig messages. `ToolResult` names are resolved from
    /// the originating tool call in the preceding assistant message (daedalus's
    /// canonical `ToolResult` carries only the call id; rig's replay wires key
    /// on the executed tool's name).
    fn to_rig_messages(
        history: &[Message],
        replay_reasoning: bool,
    ) -> Vec<rig_core::completion::Message> {
        // Build the tool-call id -> name map once (used by the tool-result arms
        // below), instead of reverse-scanning the whole history per result.
        let mut call_names: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        for m in history {
            if let Message::Assistant { tool_calls, .. } = m {
                for call in tool_calls {
                    call_names.insert(call.id.as_str(), call.name.as_str());
                }
            }
        }
        let mut out = Vec::with_capacity(history.len());
        for m in history {
            match m {
                Message::System(s) => {
                    out.push(rig_core::completion::Message::System { content: s.clone() })
                }
                Message::User(s) => out.push(rig_core::completion::Message::User {
                    content: vec![UserContent::text(s.clone())],
                }),
                Message::Assistant {
                    text,
                    tool_calls,
                    reasoning,
                } => {
                    let mut content = Vec::new();
                    // Replay reasoning first (Anthropic requires the thinking
                    // block ahead of the tool_use it belongs to). Skipped when
                    // the current request has thinking disabled — Anthropic
                    // rejects thinking blocks in that case.
                    if replay_reasoning {
                        for block in reasoning {
                            let rig = reasoning_to_rig(block);
                            if !rig.content.is_empty() {
                                content.push(AssistantContent::Reasoning(rig));
                            }
                        }
                    }
                    if let Some(t) = text {
                        if !t.is_empty() {
                            content.push(AssistantContent::Text(Text::new(t.clone())));
                        }
                    }
                    for call in tool_calls {
                        content.push(AssistantContent::ToolCall(RigToolCall::new(
                            ToolCallId::new_or_mint(call.id.clone()),
                            ToolFunction {
                                name: call.name.clone(),
                                arguments: call.args.clone(),
                            },
                        )));
                    }
                    out.push(rig_core::completion::Message::Assistant { id: None, content });
                }
                Message::ToolResult {
                    tool_call_id,
                    result,
                } => {
                    let name = call_names
                        .get(tool_call_id.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| tool_call_id.to_string());
                    out.push(rig_core::completion::Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            call: ToolCallId::new_or_mint(tool_call_id.clone()),
                            provider: None,
                            name,
                            content: vec![ToolResultContent::text(result.clone())],
                        })],
                    });
                }
            }
        }
        out
    }

    fn to_tool_definitions(tools: &[Value]) -> Vec<ToolDefinition> {
        tools
            .iter()
            .map(|t| ToolDefinition {
                name: t["name"].as_str().unwrap_or_default().to_string(),
                description: t["description"].as_str().unwrap_or_default().to_string(),
                parameters: t
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default())),
            })
            .collect()
    }

    fn build_request(
        &self,
        history: &[Message],
        tools: &[Value],
        effort_params: &Value,
    ) -> CompletionRequest {
        let extra = effort_params
            .as_object()
            .filter(|m| !m.is_empty())
            .map(|m| Value::Object(m.clone()));
        // Anthropic rejects a request with no `max_tokens`, and rig's model
        // table has no fallback for ids it does not know (every Claude id newer
        // than its table). Send an explicit value, raised above the thinking
        // budget when extended thinking is on. Other providers keep rig's
        // default rather than risk an unsupported parameter.
        let max_tokens = (self.provider.name == "anthropic").then(|| {
            let budget = extra
                .as_ref()
                .and_then(|v| v.get("thinking"))
                .and_then(|t| t.get("budget_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            self.max_tokens
                .max(budget.saturating_add(THINKING_HEADROOM)) as u64
        });
        // Anthropic only accepts replayed thinking blocks while thinking is
        // enabled in the same request; a later effort change to `off` must not
        // smuggle the old blocks back. Other wires (DeepSeek) always accept
        // `reasoning_content`.
        let thinking_enabled = extra
            .as_ref()
            .and_then(|v| v.get("thinking"))
            .and_then(|t| t.get("type"))
            .and_then(Value::as_str)
            == Some("enabled");
        let replay_reasoning = self.provider.name != "anthropic" || thinking_enabled;
        CompletionRequest {
            model: Some(self.model.clone()),
            preamble: None,
            chat_history: Self::to_rig_messages(history, replay_reasoning),
            documents: Vec::new(),
            tools: Self::to_tool_definitions(tools),
            temperature: None,
            max_tokens,
            tool_choice: None,
            additional_params: extra,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// Classify a rig completion error, preserving that policy:
    /// quota/billing is never retried, auth is auth, timeouts are timeouts,
    /// a rejected model is never retried, transient statuses (and bare
    /// transport failures) stay retryable. `model` is the id daedalus asked
    /// for, so a rejection can name it even when the provider does not.
    fn map_error(err: &CompletionError, model: &str) -> (ProviderError, bool) {
        let body = err.provider_response_body().unwrap_or_default().to_string();
        let status = err.provider_response_status().map(|s| s.as_u16());
        if let Some(code) = status {
            let mapped = map_status_error(code, body.clone(), Some(model));
            let retryable = !matches!(mapped, ProviderError::InvalidModel { .. })
                && !is_quota_or_billing(&body)
                && is_transient_status(code);
            (mapped, retryable)
        } else if is_quota_or_billing(&body) {
            (
                ProviderError::Http(format!("quota or billing limit: {body}")),
                false,
            )
        } else {
            // A status-less failure is either a transport problem (retry) or a
            // client-side request/validation error (never retry: the same
            // request will fail identically). rig reports the latter as
            // `RequestError`/`JsonError`/`UrlError`.
            let client_side = matches!(
                err,
                CompletionError::RequestError(_)
                    | CompletionError::JsonError(_)
                    | CompletionError::UrlError(_)
            );
            let msg = err.to_string();
            let retryable = !client_side;
            (ProviderError::Http(msg), retryable)
        }
    }

    fn with_retries(
        &self,
        request: CompletionRequest,
        cancel: CancellationToken,
    ) -> futures::future::BoxFuture<
        '_,
        Result<rig_core::streaming::StreamingCompletionResponse, ProviderError>,
    > {
        let backend = self.backend();
        Box::pin(async move {
            let mut attempt = 0usize;
            loop {
                // Bound connection/stream establishment. A stalled TCP connect
                // or a server that never sends the first event would otherwise
                // hang the turn until a human aborts.
                let result = tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                    r = tokio::time::timeout(self.timeout, backend.stream(request.clone())) => r,
                };
                match result {
                    Ok(Ok(stream)) => return Ok(stream),
                    Ok(Err(e)) => {
                        let (mapped, retryable) = Self::map_error(&e, &self.model);
                        if !retryable || attempt >= self.max_retries {
                            return Err(mapped);
                        }
                        attempt += 1;
                        tokio::time::sleep(Duration::from_millis(250u64 << attempt.min(6))).await;
                    }
                    Err(_) => {
                        let timeout = ProviderError::Timeout(format!(
                            "no response within {}s",
                            self.timeout.as_secs()
                        ));
                        if attempt >= self.max_retries {
                            return Err(timeout);
                        }
                        attempt += 1;
                        tokio::time::sleep(Duration::from_millis(250u64 << attempt.min(6))).await;
                    }
                }
            }
        })
    }

    fn backend(&self) -> &Backend {
        &self.backend
    }
}

impl Backend {
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> futures::future::BoxFuture<
        '_,
        Result<rig_core::streaming::StreamingCompletionResponse, CompletionError>,
    > {
        match self {
            Backend::OpenAiCompatible(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Anthropic(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Gemini(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Mistral(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Groq(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Xai(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::OpenRouter(m) => Box::pin(CompletionModel::stream(m, request)),
            Backend::Ollama(m) => Box::pin(CompletionModel::stream(m, request)),
        }
    }
}

/// Find the tool name that matches `tool_call_id`, scanning the history for
/// the assistant message that issued the call. Test-only; the adapter builds a
/// single map per request instead.
#[cfg(test)]
fn tool_name_for_call(history: &[Message], tool_call_id: &str) -> Option<String> {
    history.iter().rev().find_map(|m| match m {
        Message::Assistant { tool_calls, .. } => tool_calls
            .iter()
            .find(|c| c.id == tool_call_id)
            .map(|c| c.name.clone()),
        _ => None,
    })
}

/// Convert a captured [`ReasoningBlock`] back into rig's replay shape.
fn reasoning_to_rig(block: &ReasoningBlock) -> Reasoning {
    Reasoning {
        id: block.id.clone(),
        content: block
            .content
            .iter()
            .map(|part| match part {
                ReasoningPart::Text { text, signature } => ReasoningContent::Text {
                    text: text.clone(),
                    signature: signature.clone(),
                },
                ReasoningPart::Summary(s) => ReasoningContent::Summary(s.clone()),
                ReasoningPart::Encrypted(d) => ReasoningContent::Encrypted(d.clone()),
                ReasoningPart::Redacted { data } => {
                    ReasoningContent::Redacted { data: data.clone() }
                }
            })
            .collect(),
    }
}

/// Capture a streamed rig [`Reasoning`] block in the canonical form so it can
/// be persisted and replayed.
fn reasoning_from_rig(reasoning: &Reasoning) -> ReasoningBlock {
    ReasoningBlock {
        id: reasoning.id.clone(),
        content: reasoning
            .content
            .iter()
            .map(|part| match part {
                ReasoningContent::Text { text, signature } => ReasoningPart::Text {
                    text: text.clone(),
                    signature: signature.clone(),
                },
                ReasoningContent::Summary(s) => ReasoningPart::Summary(s.clone()),
                ReasoningContent::Encrypted(d) => ReasoningPart::Encrypted(d.clone()),
                ReasoningContent::Redacted { data } => {
                    ReasoningPart::Redacted { data: data.clone() }
                }
            })
            .collect(),
    }
}

/// The displayable text of a rig reasoning block: concatenated `Text` and
/// `Summary` content. Encrypted and redacted payloads carry no displayable
/// text.
fn reasoning_text(reasoning: &Reasoning) -> String {
    let mut out = String::new();
    for content in &reasoning.content {
        match content {
            ReasoningContent::Text { text, .. } => out.push_str(text),
            ReasoningContent::Summary(summary) => out.push_str(summary),
            ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => {}
        }
    }
    out
}

impl Provider for RigProvider {
    fn complete<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [Value],
        effort_params: &'a Value,
        cancel: CancellationToken,
        on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
    ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
        Box::pin(async move {
            // Never send an empty bearer token: refuse with an actionable
            // message when a key-requiring provider has no key.
            // Enforced at request time, not at startup, so the interactive
            // frontends still launch and `/login` stays reachable.
            if self.provider.requires_key() && self.api_key.is_empty() {
                return Err(ProviderError::Auth(format!(
                    "no API key configured; set {}, pass --api-key, or run /login",
                    self.provider.api_key_env.unwrap_or("<PROVIDER>_API_KEY")
                )));
            }
            let request = self.build_request(history, tools, effort_params);
            // An abort while connecting/backing off is a clean interruption,
            // not a provider failure: report an aborted completion so the
            // runtime marks the turn interrupted and stops draining follow-ups.
            let mut stream = match self.with_retries(request, cancel.clone()).await {
                Ok(stream) => stream,
                Err(ProviderError::Cancelled) => {
                    return Ok(Completion {
                        response: Response::Text(String::new()),
                        prompt_tokens: None,
                        aborted: true,
                        reasoning: Vec::new(),
                    })
                }
                Err(e) => return Err(e),
            };

            let mut text = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut truncated = false;
            let mut prompt_tokens: Option<usize> = None;
            let mut aborted = false;
            // Complete reasoning blocks captured for replay. The streamed
            // deltas are display-only; the complete block carries the
            // provider signature.
            let mut captured_reasoning: Vec<ReasoningBlock> = Vec::new();
            // rig correlator ids whose reasoning deltas already streamed, so
            // the superseding complete `Reasoning` block is not emitted twice.
            let mut seen_reasoning: HashSet<String> = HashSet::new();

            loop {
                let item = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        stream.cancel();
                        aborted = true;
                        None
                    }
                    item = tokio::time::timeout(self.timeout, stream.next()) => match item {
                        Ok(item) => item,
                        // A silent stream (no delta for the whole timeout) is a
                        // stall, not a reason to kill the process: surface it
                        // as a provider timeout so the user sees why.
                        Err(_) => {
                            return Err(ProviderError::Timeout(format!(
                                "no stream data for {}s",
                                self.timeout.as_secs()
                            )))
                        }
                    },
                };
                let Some(item) = item else { break };
                match item {
                    Ok(StreamedAssistantContent::Text(Text { text: t, .. })) => {
                        text.push_str(&t);
                        on_delta(StreamDelta::Text(t));
                    }
                    Ok(StreamedAssistantContent::ReasoningDelta { id, reasoning, .. }) => {
                        if !reasoning.is_empty() {
                            seen_reasoning.insert(id);
                            on_delta(StreamDelta::Thinking(reasoning));
                        }
                    }
                    Ok(StreamedAssistantContent::Reasoning { reasoning, id }) => {
                        // Preserve the complete block (with any signature) for
                        // the next request before deciding what to display.
                        captured_reasoning.push(reasoning_from_rig(&reasoning));
                        // The complete block supersedes its deltas, so emit it
                        // only when nothing streamed for this correlator.
                        if !seen_reasoning.contains(&id) {
                            let thinking = reasoning_text(&reasoning);
                            if !thinking.is_empty() {
                                on_delta(StreamDelta::Thinking(thinking));
                            }
                        }
                    }
                    Ok(StreamedAssistantContent::Final(final_record)) => {
                        if final_record
                            .finish_reason
                            .as_ref()
                            .is_some_and(|r| matches!(r, FinishReason::Length))
                        {
                            truncated = true;
                        }
                        if final_record.usage.input_tokens > 0 {
                            prompt_tokens = Some(final_record.usage.input_tokens as usize);
                        }
                    }
                    Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                        calls.push(ToolCall {
                            id: tool_call.id.to_string(),
                            name: tool_call.function.name.clone(),
                            args: tool_call.function.arguments.clone(),
                        });
                    }
                    // Partial tool-call fragments and other unmodeled items.
                    Ok(_) => {}
                    Err(e) => return Err(Self::map_error(&e, &self.model).0),
                }
            }

            if cancel.is_cancelled() {
                aborted = true;
            }

            let response = if aborted {
                Response::Text(text)
            } else if !calls.is_empty() {
                if truncated {
                    Response::TruncatedToolCalls(calls)
                } else {
                    Response::ToolCalls(calls)
                }
            } else {
                Response::Text(text)
            };
            Ok(Completion {
                response,
                prompt_tokens,
                aborted,
                reasoning: captured_reasoning,
            })
        })
    }

    /// Discover model ids via rig's `ModelListingClient`. Providers without a
    /// listing client return `Unsupported`.
    fn list_models<'a>(
        &'a self,
    ) -> futures::future::BoxFuture<'a, Result<Vec<String>, ProviderError>> {
        Box::pin(async move {
            let key = self.api_key.clone();
            let base = self.base_url.trim_end_matches('/').to_string();
            let list = match self.provider.name {
                "openai" | "deepseek" | "lmstudio" => {
                    let base = if base.ends_with("/v1") {
                        base.clone()
                    } else {
                        format!("{base}/v1")
                    };
                    let client = openai::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "anthropic" => {
                    let client = anthropic::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "gemini" => {
                    let client = gemini::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "mistral" => {
                    let client = mistral::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "groq" => {
                    let client = groq::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "openrouter" => {
                    let client = openrouter::Client::builder()
                        .api_key(key)
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                "ollama" => {
                    let client = ollama::Client::builder()
                        .api_key(String::new())
                        .base_url(base)
                        .build()
                        .map_err(|e| ProviderError::Http(e.to_string()))?;
                    client.list_models().await
                }
                other => {
                    return Err(ProviderError::Unsupported(format!(
                        "model listing is not supported for provider '{other}'"
                    )))
                }
            };
            let list = list.map_err(|e| ProviderError::Http(e.to_string()))?;
            let mut ids: Vec<String> = list.data.into_iter().map(|m| m.id).collect();
            ids.sort();
            ids.dedup();
            Ok(ids)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_key_fails_before_any_request() {
        let mut config = crate::config::Config::defaults(std::env::temp_dir());
        config.provider = crate::config::provider_by_name("deepseek").unwrap();
        config.model = "deepseek-chat".into();
        config.api_key = None;
        let provider = RigProvider::new(&config);
        let mut on_delta = |_: StreamDelta| {};
        let err = provider
            .complete(
                &[],
                &[],
                &serde_json::json!({}),
                CancellationToken::new(),
                &mut on_delta,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Auth(_)));
        assert!(err.to_string().contains("DEEPSEEK_API_KEY"), "{err}");
    }

    #[test]
    fn configured_timeout_and_retries_reach_the_adapter() {
        // Regression DAE-105: these config knobs were previously ignored.
        let mut config = crate::config::Config::defaults(std::env::temp_dir());
        config.provider = crate::config::provider_by_name("deepseek").unwrap();
        config.model = "deepseek-chat".into();
        config.timeout_secs = 7;
        config.max_retries = 5;
        let provider = RigProvider::new(&config);
        assert_eq!(provider.timeout, Duration::from_secs(7));
        assert_eq!(provider.max_retries, 5);
    }

    #[test]
    fn anthropic_requests_carry_max_tokens_for_unknown_models() {
        // Regression DAE-102: rig's `max_tokens_for_model` table knows only
        // claude-opus-4*, claude-sonnet-4* and claude-haiku-4-5; a newer id used
        // to be rejected client-side before any HTTP request.
        let mut config = crate::config::Config::defaults(std::env::temp_dir());
        config.provider = crate::config::provider_by_name("anthropic").unwrap();
        config.model = "claude-sonnet-5".into();
        config.api_key = Some("dummy".into());
        let provider = RigProvider::new(&config);
        let req = provider.build_request(&[], &[], &serde_json::json!({}));
        assert_eq!(req.max_tokens, Some(config.max_tokens as u64));
    }

    #[test]
    fn anthropic_raises_max_tokens_above_the_thinking_budget() {
        let mut config = crate::config::Config::defaults(std::env::temp_dir());
        config.provider = crate::config::provider_by_name("anthropic").unwrap();
        config.model = "claude-sonnet-5".into();
        config.api_key = Some("dummy".into());
        let provider = RigProvider::new(&config);
        let params = serde_json::json!({"thinking": {"type": "enabled", "budget_tokens": 8_192}});
        let req = provider.build_request(&[], &[], &params);
        assert!(req.max_tokens.unwrap() > 8_192, "{req:?}");
    }

    #[test]
    fn client_side_request_errors_are_not_retried() {
        let err = CompletionError::RequestError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "max_tokens must be set for Anthropic",
        )));
        let (mapped, retryable) = RigProvider::map_error(&err, "claude-sonnet-5");
        assert!(!retryable, "{mapped}");
    }

    #[test]
    fn anthropic_omits_reasoning_when_thinking_is_disabled() {
        // A later effort change to `off` must not replay old thinking blocks:
        // Anthropic rejects them when thinking is disabled.
        let mut config = crate::config::Config::defaults(std::env::temp_dir());
        config.provider = crate::config::provider_by_name("anthropic").unwrap();
        config.model = "claude-sonnet-5".into();
        config.api_key = Some("dummy".into());
        let provider = RigProvider::new(&config);
        let history = vec![Message::Assistant {
            text: Some("calling".into()),
            tool_calls: vec![],
            reasoning: vec![ReasoningBlock {
                id: None,
                content: vec![ReasoningPart::Text {
                    text: "thought".into(),
                    signature: Some("sig".into()),
                }],
            }],
        }];
        let params = serde_json::json!({"thinking": {"type": "disabled"}});
        let req = provider.build_request(&history, &[], &params);
        let rig_core::completion::Message::Assistant { content, .. } = &req.chat_history[0] else {
            panic!("expected an assistant message");
        };
        assert!(!content
            .iter()
            .any(|c| matches!(c, AssistantContent::Reasoning(_))));
    }

    #[test]
    fn assistant_reasoning_replays_ahead_of_tool_calls() {
        // Regression DAE-103: the signed thinking block must be sent back with
        // the assistant turn that issued a tool call, ahead of the tool_use.
        let history = vec![Message::Assistant {
            text: Some("calling".into()),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "ls"}),
            }],
            reasoning: vec![ReasoningBlock {
                id: None,
                content: vec![ReasoningPart::Text {
                    text: "thought".into(),
                    signature: Some("sig".into()),
                }],
            }],
        }];
        let messages = RigProvider::to_rig_messages(&history, true);
        let rig_core::completion::Message::Assistant { content, .. } = &messages[0] else {
            panic!("expected an assistant message");
        };
        assert!(matches!(content[0], AssistantContent::Reasoning(_)));
        assert!(matches!(content[1], AssistantContent::Text(_)));
        assert!(matches!(content[2], AssistantContent::ToolCall(_)));
    }

    #[test]
    fn reasoning_round_trips_through_the_canonical_form() {
        let signed = Reasoning::new_with_signature("chain of thought", Some("sig-1".into()));
        let canonical = reasoning_from_rig(&signed);
        assert_eq!(canonical.content.len(), 1);
        let back = reasoning_to_rig(&canonical);
        assert_eq!(back, signed);
    }

    #[test]
    fn reasoning_text_concatenates_text_and_summary() {
        use rig_core::completion::message::{Reasoning, ReasoningContent};
        assert_eq!(reasoning_text(&Reasoning::new("plain")), "plain");
        let r = Reasoning {
            id: None,
            content: vec![
                ReasoningContent::Summary("sum".into()),
                ReasoningContent::Text {
                    text: " txt".into(),
                    signature: None,
                },
                ReasoningContent::Encrypted("opaque".into()),
            ],
        };
        assert_eq!(reasoning_text(&r), "sum txt");
    }

    #[test]
    fn tool_names_resolve_from_the_issuing_assistant_message() {
        let history = vec![
            Message::System("sys".into()),
            Message::User("run it".into()),
            Message::Assistant {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({"command": "ls"}),
                }],
                reasoning: vec![],
            },
            Message::ToolResult {
                tool_call_id: "call-1".into(),
                result: "out".into(),
            },
        ];
        assert_eq!(
            tool_name_for_call(&history, "call-1").as_deref(),
            Some("bash")
        );
        assert_eq!(tool_name_for_call(&history, "missing"), None);
    }

    #[test]
    fn tool_definitions_map_from_daedalus_schemas() {
        let schemas = vec![serde_json::json!({
            "name": "read",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}}
        })];
        let defs = RigProvider::to_tool_definitions(&schemas);
        assert_eq!(defs[0].name, "read");
        assert_eq!(defs[0].description, "Read a file.");
        assert_eq!(defs[0].parameters["type"], "object");
    }
}
