//! OpenAI provider (also serves DeepSeek via a configurable base URL + model).
//!
//! Uses the OpenAI chat-completions protocol with streaming (SSE) + tool
//! calling. Streamed responses are normalized into the internal
//! `Response`/`ToolCall` types; `cancel` aborts the stream mid-generation.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use super::{
    next_sse_event, post_stream, Completion, Message, Provider, ProviderError, Response, ToolCall,
};
use crate::config::Config;

pub struct OpenAIProvider {
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: usize,
    timeout_secs: u64,
    max_retries: usize,
}

impl OpenAIProvider {
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
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn request(&self, history: &[Message], tools: &[Value]) -> Value {
        let messages: Vec<Value> = history.iter().map(to_openai_message).collect();
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        if !tools.is_empty() {
            body["tools"] = to_openai_tools(tools);
        }
        body
    }
}

fn to_openai_message(m: &Message) -> Value {
    match m {
        Message::System(s) => json!({"role": "system", "content": s}),
        Message::User(s) => json!({"role": "user", "content": s}),
        Message::Assistant { text, tool_calls } => {
            let mut obj = json!({"role": "assistant", "content": text.clone().unwrap_or_default()});
            if !tool_calls.is_empty() {
                let calls: Vec<Value> = tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.args)
                                    .unwrap_or_else(|_| "{}".into())
                            }
                        })
                    })
                    .collect();
                obj["tool_calls"] = Value::Array(calls);
            }
            obj
        }
        Message::ToolResult {
            tool_call_id,
            result,
        } => json!({"role": "tool", "tool_call_id": tool_call_id, "content": result}),
    }
}

/// Wrap internal `{name, description, parameters}` schemas into OpenAI's
/// `{type, function:{name, description, parameters}}` shape.
fn to_openai_tools(tools: &[Value]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"]
                    }
                })
            })
            .collect(),
    )
}

/// Parse a streamed OpenAI chat-completions SSE body into a `Completion`.
/// Tool-call arguments arrive as JSON-string fragments keyed by `index` and are
/// reassembled before parsing.
fn parse_openai_stream(
    reader: &mut impl BufRead,
    cancel: &AtomicBool,
) -> Result<Completion, ProviderError> {
    let mut text = String::new();
    // (id, name, concatenated arguments JSON).
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut prompt_tokens = None;
    let mut truncated = false;

    while let Some(ev) = next_sse_event(reader, cancel)? {
        if let Some(u) = ev.get("usage") {
            prompt_tokens = u
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
        }
        if let Some(choice) = ev
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
        {
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                if fr == "length" || fr == "max_tokens" {
                    truncated = true;
                }
            }
            if let Some(delta) = choice.get("delta") {
                if let Some(t) = delta.get("content").and_then(|v| v.as_str()) {
                    text.push_str(t);
                }
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        if calls.len() <= index {
                            calls.resize(index + 1, (String::new(), String::new(), String::new()));
                        }
                        let c = &mut calls[index];
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            c.0 = id.to_string();
                        }
                        if let Some(name) = tc.pointer("/function/name").and_then(|v| v.as_str()) {
                            c.1 = name.to_string();
                        }
                        if let Some(arg) =
                            tc.pointer("/function/arguments").and_then(|v| v.as_str())
                        {
                            c.2.push_str(arg);
                        }
                    }
                }
            }
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return Ok(Completion {
            response: Response::Text(text),
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
            response: Response::Text(text),
            prompt_tokens,
            aborted: false,
        })
    }
}

impl Provider for OpenAIProvider {
    fn complete(
        &self,
        history: &[Message],
        tools: &[Value],
        cancel: &AtomicBool,
    ) -> Result<Completion, ProviderError> {
        let body = self.request(history, tools);
        let auth = self.api_key.as_ref().map(|k| format!("Bearer {k}"));
        let mut headers: Vec<(&str, &str)> = Vec::new();
        if let Some(a) = &auth {
            headers.push(("Authorization", a.as_str()));
        }
        let reader = post_stream(
            &self.url(),
            &headers,
            body,
            self.timeout_secs,
            self.max_retries,
        )?;
        let mut buf = BufReader::new(reader);
        parse_openai_stream(&mut buf, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn text_sse() -> &'static str {
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: [DONE]\n\n"
    }

    fn tool_sse() -> &'static str {
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"a.txt\\\"}\"}}]}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
         data: [DONE]\n\n"
    }

    #[test]
    fn streams_text_response() {
        let cancel = AtomicBool::new(false);
        let mut data = text_sse().as_bytes();
        let r = parse_openai_stream(&mut data, &cancel).unwrap();
        assert!(!r.aborted);
        assert_eq!(r.response, Response::Text("hello world".into()));
    }

    #[test]
    fn streams_and_reassembles_tool_calls() {
        let cancel = AtomicBool::new(false);
        let mut data = tool_sse().as_bytes();
        let r = parse_openai_stream(&mut data, &cancel).unwrap();
        match r.response {
            Response::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "read");
                assert_eq!(calls[0].args["path"], "a.txt");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[test]
    fn streams_truncated_tool_calls() {
        let cancel = AtomicBool::new(false);
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}]}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
                   data: [DONE]\n\n";
        let mut data = sse.as_bytes();
        let r = parse_openai_stream(&mut data, &cancel).unwrap();
        assert!(matches!(r.response, Response::TruncatedToolCalls(_)));
    }

    #[test]
    fn cancel_yields_aborted_partial_text() {
        let cancel = AtomicBool::new(false);
        let data = text_sse().as_bytes();
        // Yield one byte at a time and set cancel after the first few bytes, so
        // the stream aborts partway through the first event.
        let reader = ToggleReader {
            data,
            pos: 0,
            cancel_after: 8,
            cancel: &cancel,
        };
        let mut buf = BufReader::new(reader);
        let r = parse_openai_stream(&mut buf, &cancel).unwrap();
        assert!(r.aborted);
        if let Response::Text(t) = r.response {
            assert!(t.contains("hello"));
        } else {
            panic!("expected partial text");
        }
    }

    #[test]
    fn message_conversion_produces_openai_shape() {
        let m = Message::ToolResult {
            tool_call_id: "c".into(),
            result: "ok".into(),
        };
        let v = to_openai_message(&m);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "c");
        assert_eq!(v["content"], "ok");
    }

    /// Reader that yields `data` one byte at a time and sets `cancel` once it
    /// has yielded `cancel_after` bytes.
    struct ToggleReader<'a> {
        data: &'a [u8],
        pos: usize,
        cancel_after: usize,
        cancel: &'a AtomicBool,
    }

    impl Read for ToggleReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            if self.pos >= self.cancel_after {
                self.cancel.store(true, Ordering::Relaxed);
            }
            Ok(1)
        }
    }
}
