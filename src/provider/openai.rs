//! OpenAI provider (also serves DeepSeek via a configurable base URL + model).
//!
//! Uses the OpenAI chat-completions protocol with function/tool calling. The
//! four tool schemas are wrapped into OpenAI's `tools` field, and responses are
//! normalized into the internal `Response`/`ToolCall` types.

use serde_json::{json, Value};

use super::{Completion, Message, Provider, ProviderError, Response, ToolCall};
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
                                "arguments": serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into())
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

fn parse_response(resp: Value) -> Result<Response, ProviderError> {
    let choices = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ProviderError::Malformed("missing 'choices' array".into()))?;
    let choice = choices
        .first()
        .ok_or_else(|| ProviderError::Malformed("missing choice[0]".into()))?;
    let message = choice
        .get("message")
        .ok_or_else(|| ProviderError::Malformed("missing choice[0].message".into()))?;
    let truncated = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(|r| r == "length" || r == "max_tokens")
        .unwrap_or(false);

    let tool_calls = message
        .get("tool_calls")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();

    if !tool_calls.is_empty() {
        let mut calls = Vec::with_capacity(tool_calls.len());
        for tc in &tool_calls {
            let name = tc
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::Malformed("tool call missing function.name".into()))?
                .to_string();
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let args_str = tc
                .pointer("/function/arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
            calls.push(ToolCall { id, name, args });
        }
        if truncated {
            Ok(Response::TruncatedToolCalls(calls))
        } else {
            Ok(Response::ToolCalls(calls))
        }
    } else {
        let text = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        Ok(Response::Text(text))
    }
}

impl Provider for OpenAIProvider {
    fn complete(&self, history: &[Message], tools: &[Value]) -> Result<Completion, ProviderError> {
        let body = self.request(history, tools);
        let auth = self.api_key.as_ref().map(|k| format!("Bearer {k}"));
        let mut headers: Vec<(&str, &str)> = Vec::new();
        if let Some(a) = &auth {
            headers.push(("Authorization", a.as_str()));
        }
        let value = super::post_json(
            &self.url(),
            &headers,
            body,
            self.timeout_secs,
            self.max_retries,
        )?;
        let prompt_tokens = value
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let response = parse_response(value)?;
        Ok(Completion {
            response,
            prompt_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Message, Response};

    fn text_msg(s: &str) -> Value {
        json!({"choices": [{"message": {"content": s}}]})
    }

    fn tool_msg() -> Value {
        json!({
            "choices": [{"message": {
                "content": null,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read", "arguments": "{\"path\":\"a.txt\"}"}}
                ]
            }}]
        })
    }

    #[test]
    fn parses_text_response() {
        let r = parse_response(text_msg("hi there")).unwrap();
        assert_eq!(r, Response::Text("hi there".into()));
    }

    #[test]
    fn parses_tool_call_response() {
        let r = parse_response(tool_msg()).unwrap();
        match r {
            Response::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "read");
                assert_eq!(calls[0].args["path"], "a.txt");
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parses_truncated_tool_call_response() {
        let truncated = json!({
            "choices": [{"finish_reason": "length", "message": {
                "content": null,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read", "arguments": "{\"path\":\"a.txt\"}"}}
                ]
            }}]
        });
        let r = parse_response(truncated).unwrap();
        assert!(matches!(r, Response::TruncatedToolCalls(_)));
    }

    #[test]
    fn malformed_missing_choices_is_error() {
        assert!(parse_response(json!({})).is_err());
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

    #[test]
    fn assistant_tool_calls_are_wrapped() {
        let m = Message::Assistant {
            text: None,
            tool_calls: vec![ToolCall {
                id: "x".into(),
                name: "write".into(),
                args: json!({"path":"p"}),
            }],
        };
        let v = to_openai_message(&m);
        assert_eq!(v["tool_calls"][0]["function"]["name"], "write");
        assert_eq!(
            v["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"p\"}"
        );
    }
}
