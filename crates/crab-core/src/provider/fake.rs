//! Fake provider: an in-memory, scripted backend for deterministic offline
//! tests (CRAB-103/104/106). It pops a scripted `Response` for each call and
//! records every history it receives so tests can assert on result feedback.

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use serde_json::Value;

use super::{Completion, Message, Provider, ProviderError, Response};

pub struct FakeProvider {
    responses: Mutex<VecDeque<Response>>,
    /// History of every `complete` call, in order, for assertions.
    histories: Mutex<Vec<Vec<Message>>>,
}

impl FakeProvider {
    /// A provider that plays `responses` in order; once exhausted it returns
    /// final text "done".
    pub fn new(responses: Vec<Response>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            histories: Mutex::new(Vec::new()),
        }
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
    fn complete(
        &self,
        history: &[Message],
        _tools: &[Value],
        _cancel: &AtomicBool,
        _on_text: &mut dyn FnMut(&str),
    ) -> Result<Completion, ProviderError> {
        self.histories.lock().unwrap().push(history.to_vec());
        let mut q = self.responses.lock().unwrap();
        let response = match q.pop_front() {
            Some(r) => r,
            None => Response::Text("done".into()),
        };
        Ok(Completion {
            response,
            prompt_tokens: None,
            aborted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;

    #[test]
    fn plays_scripted_responses_then_final_text() {
        let p = FakeProvider::new(vec![
            Response::ToolCalls(vec![ToolCall {
                id: "c".into(),
                name: "bash".into(),
                args: serde_json::json!({"command":"echo hi"}),
            }]),
            Response::Text("final".into()),
        ]);
        assert!(matches!(
            p.complete(&[], &[], &AtomicBool::new(false), &mut |_| {})
                .unwrap()
                .response,
            Response::ToolCalls(_)
        ));
        assert_eq!(
            p.complete(&[], &[], &AtomicBool::new(false), &mut |_| {})
                .unwrap()
                .response,
            Response::Text("final".into())
        );
        // Exhausted -> final "done".
        assert_eq!(
            p.complete(&[], &[], &AtomicBool::new(false), &mut |_| {})
                .unwrap()
                .response,
            Response::Text("done".into())
        );
    }

    #[test]
    fn records_history_and_tool_results() {
        let p = FakeProvider::new(vec![Response::Text("ok".into())]);
        let hist = vec![Message::ToolResult {
            tool_call_id: "abc".into(),
            result: "r".into(),
        }];
        p.complete(&hist, &[], &AtomicBool::new(false), &mut |_| {})
            .unwrap();
        assert_eq!(p.calls(), 1);
        assert!(p.saw_tool_result("abc"));
    }
}
