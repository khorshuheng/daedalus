//! Anthropic provider (Messages API + tool use), with streaming (SSE).

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use super::{
    next_sse_event, post_stream, Completion, Message, Provider, ProviderError, Response, ToolCall,
};
use crate::config::Config;

pub struct AnthropicProvider {
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: usize,
    timeout_secs: u64,
    max_retries: usize,
}

impl AnthropicProvider {
    pub fn new(config: &Config) -> Self {
        Self {
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            timeout_secs: config.timeout_secs,
            max_retries: config.max_retries,
        }
    }

    fn url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    fn request(&self, history: &[Message], tools: &[Value]) -> Value {
        // Anthropic takes system content as a top-level field.
        let system: Vec<&str> = history
            .iter()
            .filter_map(|m| match m {
                Message::System(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let messages: Vec<Value> = history
            .iter()
            .filter(|m| !matches!(m, Message::System(_)))
            .map(to_anthropic_message)
            .collect();

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": true,
            "messages": messages,
        });
        if !system.is_empty() {
            body["system"] = Value::String(system.join("\n\n"));
        }
        if !tools.is_empty() {
            body["tools"] = to_anthropic_tools(tools);
        }
        body
    }
}

fn to_anthropic_message(m: &Message) -> Value {
    match m {
        Message::System(_) => unreachable!("system messages are extracted before conversion"),
        Message::User(s) => json!({"role": "user", "content": s}),
        Message::Assistant { text, tool_calls } => {
            let mut blocks = Vec::new();
            if let Some(t) = text {
                if !t.is_empty() {
                    blocks.push(json!({"type": "text", "text": t}));
                }
            }
            for tc in tool_calls {
                blocks.push(json!({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args
                }));
            }
            json!({"role": "assistant", "content": Value::Array(blocks)})
        }
        Message::ToolResult {
            tool_call_id,
            result,
        } => json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": result}]
        }),
    }
}

/// Anthropic tool schema is `{name, description, input_schema}`.
fn to_anthropic_tools(tools: &[Value]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"]
                })
            })
            .collect(),
    )
}

/// Parse a streamed Anthropic Messages SSE body into a `Completion`. Tool-use
/// arguments arrive as `input_json_delta` fragments and are reassembled.
fn parse_anthropic_stream(
    reader: &mut impl BufRead,
    cancel: &AtomicBool,
) -> Result<Completion, ProviderError> {
    let mut texts = Vec::new();
    // (id, name, args_json), indexed by content-block index.
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut prompt_tokens = None;
    let mut truncated = false;

    while let Some(ev) = next_sse_event(reader, cancel)? {
        match ev.get("type").and_then(|v| v.as_str()) {
            Some("message_start") => {
                if let Some(u) = ev.pointer("/message/usage/input_tokens") {
                    prompt_tokens = u.as_u64().map(|n| n as usize);
                }
            }
            Some("content_block_start") => {
                let index = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(block) = ev.get("content_block") {
                    if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                        if calls.len() <= index {
                            calls.resize(index + 1, (String::new(), String::new(), String::new()));
                        }
                        calls[index].0 = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        calls[index].1 = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                    }
                }
            }
            Some("content_block_delta") => {
                let index = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(delta) = ev.get("delta") {
                    match delta.get("type").and_then(|v| v.as_str()) {
                        Some("text_delta") => {
                            if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                        Some("input_json_delta") => {
                            if calls.len() <= index {
                                calls.resize(
                                    index + 1,
                                    (String::new(), String::new(), String::new()),
                                );
                            }
                            if let Some(p) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                calls[index].2.push_str(p);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("message_delta") => {
                if let Some(d) = ev.get("delta") {
                    if let Some(sr) = d.get("stop_reason").and_then(|v| v.as_str()) {
                        if sr == "max_tokens" {
                            truncated = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return Ok(Completion {
            response: Response::Text(texts.join("\n")),
            prompt_tokens,
            aborted: true,
        });
    }

    if !calls.is_empty() {
        let tool_calls = calls
            .into_iter()
            .map(|(id, name, args_json)| {
                let args = serde_json::from_str(&args_json).unwrap_or_else(|_| json!({}));
                ToolCall { id, name, args }
            })
            .collect();
        let response = if truncated {
            Response::TruncatedToolCalls(tool_calls)
        } else {
            Response::ToolCalls(tool_calls)
        };
        Ok(Completion {
            response,
            prompt_tokens,
            aborted: false,
        })
    } else {
        Ok(Completion {
            response: Response::Text(texts.join("\n")),
            prompt_tokens,
            aborted: false,
        })
    }
}

impl Provider for AnthropicProvider {
    fn complete(
        &self,
        history: &[Message],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<Completion, ProviderError> {
        let body = self.request(history, tools);
        let mut headers: Vec<(&str, &str)> = vec![("anthropic-version", "2023-06-01")];
        if let Some(key) = self.api_key.as_deref() {
            headers.push(("x-api-key", key));
        }
        let reader = post_stream(
            &self.url(),
            &headers,
            body,
            self.timeout_secs,
            self.max_retries,
        )?;
        let mut buf = BufReader::new(reader);
        parse_anthropic_stream(&mut buf, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_text_response() {
        let cancel = AtomicBool::new(false);
        let sse = "event: message_start\n\
                   data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12}}}\n\n\
                   event: content_block_start\n\
                   data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                   event: content_block_delta\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                   event: message_delta\n\
                   data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
                   event: message_stop\n\
                   data: {\"type\":\"message_stop\"}\n\n";
        let mut data = sse.as_bytes();
        let r = parse_anthropic_stream(&mut data, &cancel).unwrap();
        assert!(!r.aborted);
        assert_eq!(r.response, Response::Text("hi".into()));
        assert_eq!(r.prompt_tokens, Some(12));
    }

    #[test]
    fn streams_and_reassembles_tool_use() {
        let cancel = AtomicBool::new(false);
        let sse = "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"read\",\"input\":{}}}\n\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}\n\n\
                   data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n\
                   data: {\"type\":\"message_stop\"}\n\n";
        let mut data = sse.as_bytes();
        let r = parse_anthropic_stream(&mut data, &cancel).unwrap();
        match r.response {
            Response::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "t_1");
                assert_eq!(calls[0].name, "read");
                assert_eq!(calls[0].args["path"], "a.txt");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[test]
    fn streams_truncated_tool_use() {
        let cancel = AtomicBool::new(false);
        let sse = "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"read\",\"input\":{}}}\n\n\
                   data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}\n\n\
                   data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n\
                   data: {\"type\":\"message_stop\"}\n\n";
        let mut data = sse.as_bytes();
        let r = parse_anthropic_stream(&mut data, &cancel).unwrap();
        assert!(matches!(r.response, Response::TruncatedToolCalls(_)));
    }

    #[test]
    fn converts_tool_result_to_user_message() {
        let m = Message::ToolResult {
            tool_call_id: "t".into(),
            result: "ok".into(),
        };
        let v = to_anthropic_message(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"][0]["type"], "tool_result");
        assert_eq!(v["content"][0]["tool_use_id"], "t");
    }

    #[test]
    fn extracts_system_from_messages() {
        let p = AnthropicProvider {
            base_url: "https://api.anthropic.com".into(),
            api_key: None,
            model: "m".into(),
            temperature: 0.5,
            max_tokens: 100,
            timeout_secs: 60,
            max_retries: 0,
        };
        let hist = vec![
            Message::System("be nice".into()),
            Message::User("hi".into()),
        ];
        let body = p.request(&hist, &[]);
        assert_eq!(body["system"], "be nice");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["stream"], true);
    }
}
