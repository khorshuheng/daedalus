//! The rig-core adapter (CRAB-130): the only module in crab-core that names
//! rig. Maps crab's canonical message/response types to rig's provider
//! contracts and back, for the OpenAI chat-completions wire (serving both the
//! OpenAI and DeepSeek presets, as before) and the Anthropic Messages wire.
//!
//! Hard requirements carried over from the hand-written clients:
//! - `prompt_tokens` from provider-reported usage anchors context budgeting;
//!   missing usage degrades explicitly to `None` (rig's zero-valued sentinel).
//! - Quota/billing errors are never retried; transient statuses back off
//!   (the CRAB-107 #14 classification, now over rig errors).
//! - Truncated tool calls (`finish_reason: length`) must not be executed.
//! - Effort/thinking wire parameters (`provider_effort`, CRAB-116) flow into
//!   the request via `additional_params`.

use futures::StreamExt;
use rig_core::client::CompletionClient;
use rig_core::completion::message::ToolCall as RigToolCall;
use rig_core::completion::message::{
    AssistantContent, Text, ToolCallId, ToolFunction, ToolResult, ToolResultContent, UserContent,
};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, FinishReason, ToolDefinition,
};
use rig_core::providers::{anthropic, openai};
use rig_core::streaming::StreamedAssistantContent;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{
    is_quota_or_billing, is_transient_status, map_status_error, Completion, Message, Provider,
    ProviderError, Response, ToolCall,
};

/// How many times a transient failure is retried with exponential backoff
/// (same policy as the old hand-written client, CRAB-103).
const MAX_RETRIES: usize = 3;

enum Backend {
    /// OpenAI chat-completions; also serves DeepSeek via a configurable base
    /// URL (as the old hand-written client did).
    OpenAiCompatible(openai::completion::GenericCompletionModel<openai::OpenAICompletionsExt>),
    /// Anthropic Messages API.
    Anthropic(anthropic::completion::CompletionModel),
}

/// The rig-backed provider. Construct via [`RigProvider::openai_compatible`]
/// or [`RigProvider::anthropic`].
pub struct RigProvider {
    backend: Backend,
    model: String,
}

impl RigProvider {
    /// OpenAI chat-completions wire for the OpenAI/DeepSeek presets. crab's
    /// configured `base_url` historically pointed at the API root (the client
    /// appended `/v1/chat/completions`), while rig expects the base to carry
    /// the `/v1` prefix — normalized here, preserving the old wire URL.
    pub fn openai_compatible(config: &crate::config::Config) -> Self {
        let base = format!("{}/v1", config.base_url.trim_end_matches('/'));
        let client = openai::Client::builder()
            .api_key(config.api_key.clone().unwrap_or_default())
            .base_url(base)
            .build()
            .expect("valid OpenAI-compatible client");
        let model = client
            .completions_api()
            .completion_model(config.model.clone());
        Self {
            backend: Backend::OpenAiCompatible(model),
            model: config.model.clone(),
        }
    }

    /// Anthropic Messages wire. The configured base URL is passed through;
    /// rig normalizes it (stripping a `/v1/messages` suffix if present) and
    /// appends the Messages path.
    pub fn anthropic(config: &crate::config::Config) -> Self {
        let client = anthropic::Client::builder()
            .api_key(config.api_key.clone().unwrap_or_default())
            .base_url(config.base_url.clone())
            .build()
            .expect("valid Anthropic client");
        let model = client.completion_model(config.model.clone());
        Self {
            backend: Backend::Anthropic(model),
            model: config.model.clone(),
        }
    }

    /// Map crab history to rig messages. `ToolResult` names are resolved from
    /// the originating tool call in the preceding assistant message (crab's
    /// canonical `ToolResult` carries only the call id; rig's replay wires key
    /// on the executed tool's name).
    fn to_rig_messages(history: &[Message]) -> Vec<rig_core::completion::Message> {
        let mut out = Vec::with_capacity(history.len());
        for m in history {
            match m {
                Message::System(s) => {
                    out.push(rig_core::completion::Message::System { content: s.clone() })
                }
                Message::User(s) => out.push(rig_core::completion::Message::User {
                    content: vec![UserContent::text(s.clone())],
                }),
                Message::Assistant { text, tool_calls } => {
                    let mut content = Vec::new();
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
                    let name = tool_name_for_call(history, tool_call_id)
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
        CompletionRequest {
            model: Some(self.model.clone()),
            preamble: None,
            chat_history: Self::to_rig_messages(history),
            documents: Vec::new(),
            tools: Self::to_tool_definitions(tools),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: extra,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// Classify a rig completion error, preserving the CRAB-107 #14 policy:
    /// quota/billing is never retried, auth is auth, timeouts are timeouts,
    /// transient statuses (and bare transport failures) stay retryable.
    fn map_error(err: &CompletionError) -> (ProviderError, bool) {
        let body = err.provider_response_body().unwrap_or_default().to_string();
        let status = err.provider_response_status().map(|s| s.as_u16());
        if let Some(code) = status {
            let mapped = map_status_error(code, body.clone());
            let retryable = !is_quota_or_billing(&body) && is_transient_status(code);
            (mapped, retryable)
        } else if is_quota_or_billing(&body) {
            (
                ProviderError::Http(format!("quota or billing limit: {body}")),
                false,
            )
        } else {
            let msg = err.to_string();
            let retryable = true; // transport-level failure: back off and retry
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
                let result = tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Timeout("cancelled".into())),
                    r = backend.stream(request.clone()) => r,
                };
                match result {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        let (mapped, retryable) = Self::map_error(&e);
                        if !retryable || attempt >= MAX_RETRIES {
                            return Err(mapped);
                        }
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            250u64 << attempt.min(6),
                        ))
                        .await;
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
        }
    }
}

/// Find the tool name that matches `tool_call_id`, scanning the history for
/// the assistant message that issued the call.
fn tool_name_for_call(history: &[Message], tool_call_id: &str) -> Option<String> {
    history.iter().rev().find_map(|m| match m {
        Message::Assistant { tool_calls, .. } => tool_calls
            .iter()
            .find(|c| c.id == tool_call_id)
            .map(|c| c.name.clone()),
        _ => None,
    })
}

impl Provider for RigProvider {
    fn complete<'a>(
        &'a self,
        history: &'a [Message],
        tools: &'a [Value],
        effort_params: &'a Value,
        cancel: CancellationToken,
        on_text: &'a mut (dyn FnMut(&str) + Send),
    ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
        Box::pin(async move {
            let request = self.build_request(history, tools, effort_params);
            let mut stream = self.with_retries(request, cancel.clone()).await?;

            let mut text = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut truncated = false;
            let mut prompt_tokens: Option<usize> = None;
            let mut aborted = false;

            loop {
                let item = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        stream.cancel();
                        aborted = true;
                        None
                    }
                    item = stream.next() => item,
                };
                let Some(item) = item else { break };
                match item {
                    Ok(StreamedAssistantContent::Text(Text { text: t, .. })) => {
                        on_text(&t);
                        text.push_str(&t);
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
                    // Partial tool-call fragments precede the complete call;
                    // reasoning deltas are not surfaced to the UI today.
                    Ok(_) => {}
                    Err(e) => return Err(Self::map_error(&e).0),
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
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn tool_definitions_map_from_crab_schemas() {
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
