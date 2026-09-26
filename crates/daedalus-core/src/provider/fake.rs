//! Fake provider: an in-memory, scripted backend for deterministic offline
//! tests. It pops a scripted `Response` for each call and
//! records every history it receives so tests can assert on result feedback.
//! Async behind the same seam since then; cancellation is honored the way
//! a real provider would report it (`Completion.aborted`).

use std::collections::VecDeque;
use std::sync::Mutex;

use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{Completion, Message, Provider, ProviderError, Response, StreamDelta};

pub struct FakeProvider {
    responses: Mutex<VecDeque<Response>>,
    /// Scripted reasoning fragments, one per `complete` call.
    thinking: Mutex<VecDeque<String>>,
    /// Model ids reported by `list_models`.
    models: Vec<String>,
    /// History of every `complete` call, in order, for assertions.
    histories: Mutex<Vec<Vec<Message>>>,
}

impl FakeProvider {
    /// A provider that plays `responses` in order; once exhausted it returns
    /// final text "done".
    pub fn new(responses: Vec<Response>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            thinking: Mutex::new(VecDeque::new()),
            models: Vec::new(),
            histories: Mutex::new(Vec::new()),
        }
    }

    /// Models reported by `list_models`.
    pub fn with_models(mut self, models: Vec<String>) -> Self {
        self.models = models;
        self
    }

    /// Script one reasoning fragment per `complete` call, streamed before the
    /// scripted response.
    pub fn with_thinking(mut self, thinking: Vec<String>) -> Self {
        self.thinking = Mutex::new(thinking.into());
        self
    }

    /// Number of `complete` calls made so far.
    pub fn calls(&self) -> usize {
        self.histories.lock().unwrap().len()
    }

    /// The history passed to the n-th `complete` call (0-based).
    pub fn history(&self, index: usize) -> Vec<Message> {
        self.histories.lock().unwrap()[index].clone()
    }

    /// True if the last history fed back a tool result for `tool_call_id`.
    pub fn saw_tool_result(&self, tool_call_id: &str) -> bool {
        let h = self.histories.lock().unwrap();
        if let Some(last) = h.last() {
            last.iter().any(|m| match m {
                Message::ToolResult {
                    tool_call_id: id, ..
                } => id == tool_call_id,
                _ => false,
            })
        } else {
            false
        }
    }
}

impl Provider for FakeProvider {
    fn complete<'a>(
        &'a self,
        history: &'a [Message],
        _tools: &'a [Value],
        _effort_params: &'a Value,
        cancel: CancellationToken,
        on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
    ) -> BoxFuture<'a, Result<Completion, ProviderError>> {
        Box::pin(async move {
            self.histories.lock().unwrap().push(history.to_vec());
            let mut q = self.responses.lock().unwrap();
            let response = match q.pop_front() {
                Some(r) => r,
                None => Response::Text("done".into()),
            };
            drop(q);
            if let Some(thinking) = self.thinking.lock().unwrap().pop_front() {
                on_delta(StreamDelta::Thinking(thinking));
            }
            if cancel.is_cancelled() {
                let text = match response {
                    Response::Text(t) => t,
                    _ => String::new(),
                };
                return Ok(Completion {
                    response: Response::Text(text),
                    prompt_tokens: None,
                    aborted: true,
                });
            }
            Ok(Completion {
                response,
                prompt_tokens: None,
                aborted: false,
            })
        })
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, ProviderError>> {
        Box::pin(async move { Ok(self.models.clone()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;

    fn blank() -> (&'static Value, CancellationToken) {
        static EMPTY: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        (
            EMPTY.get_or_init(|| Value::Object(Default::default())),
            CancellationToken::new(),
        )
    }

    #[tokio::test]
    async fn plays_scripted_responses_then_final_text() {
        let (effort, cancel) = blank();
        let p = FakeProvider::new(vec![
            Response::ToolCalls(vec![ToolCall {
                id: "c".into(),
                name: "bash".into(),
                args: serde_json::json!({"command":"echo hi"}),
            }]),
            Response::Text("final".into()),
        ]);
        let mut on_delta = |_: StreamDelta| {};
        assert!(matches!(
            p.complete(&[], &[], effort, cancel.clone(), &mut on_delta)
                .await
                .unwrap()
                .response,
            Response::ToolCalls(_)
        ));
        assert_eq!(
            p.complete(&[], &[], effort, cancel.clone(), &mut on_delta)
                .await
                .unwrap()
                .response,
            Response::Text("final".into())
        );
        // Exhausted -> final "done".
        assert_eq!(
            p.complete(&[], &[], effort, cancel, &mut on_delta)
                .await
                .unwrap()
                .response,
            Response::Text("done".into())
        );
    }

    #[tokio::test]
    async fn records_history_and_tool_results() {
        let (effort, cancel) = blank();
        let p = FakeProvider::new(vec![Response::Text("ok".into())]);
        let hist = vec![Message::ToolResult {
            tool_call_id: "abc".into(),
            result: "r".into(),
        }];
        let mut on_delta = |_: StreamDelta| {};
        p.complete(&hist, &[], effort, cancel, &mut on_delta)
            .await
            .unwrap();
        assert_eq!(p.calls(), 1);
        assert!(p.saw_tool_result("abc"));
    }

    #[tokio::test]
    async fn cancelled_call_reports_aborted() {
        let (effort, cancel) = blank();
        cancel.cancel();
        let p = FakeProvider::new(vec![Response::Text("never seen".into())]);
        let mut on_delta = |_: StreamDelta| {};
        let c = p
            .complete(&[], &[], effort, cancel, &mut on_delta)
            .await
            .unwrap();
        // Like a real provider, the scripted (partial) text is preserved and
        // only the `aborted` flag signals the interruption.
        assert!(c.aborted);
        assert_eq!(c.response, Response::Text("never seen".into()));
    }
}
