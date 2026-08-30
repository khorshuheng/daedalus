//! Anthropic provider (Messages API + tool use).

use serde_json::{json, Value};

use super::{Message, Provider, ProviderError, Response, ToolCall};
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

fn parse_response(resp: Value) -> Result<Response, ProviderError> {
    let content = resp
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ProviderError::Malformed("missing 'content' array".into()))?;

    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for block in content {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    texts.push(t.to_string());
                }
            }
            Some("tool_use") => calls.push(ToolCall {
                id: block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                name: block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ProviderError::Malformed("tool_use missing name".into()))?
                    .to_string(),
                args: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }),
            _ => {}
        }
    }

    if !calls.is_empty() {
        Ok(Response::ToolCalls(calls))
    } else {
        Ok(Response::Text(texts.join("\n")))
    }
}

impl Provider for AnthropicProvider {
    fn complete(&self, history: &[Message], tools: &[Value]) -> Result<Response, ProviderError> {
        let body = self.request(history, tools);
        let mut headers: Vec<(&str, &str)> = vec![("anthropic-version", "2023-06-01")];
        if let Some(key) = self.api_key.as_deref() {
            headers.push(("x-api-key", key));
        }
        let value = super::post_json(
            &self.url(),
            &headers,
            body,
            self.timeout_secs,
            self.max_retries,
        )?;
        parse_response(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_response() {
        let r = parse_response(json!({"content": [{"type": "text", "text": "hi"}]})).unwrap();
        assert_eq!(r, Response::Text("hi".into()));
    }

    #[test]
    fn parses_tool_use_response() {
        let r = parse_response(json!({
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "t_1", "name": "read", "input": {"path": "a.txt"}}
            ]
        }))
        .unwrap();
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
    }
}
