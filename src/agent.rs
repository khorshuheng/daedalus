//! The orchestration loop (CRAB-104).
//!
//! Alternates between the LLM and the tool executor: seed history with the
//! system + user prompt, call the provider, execute any tool calls, feed
//! results back verbatim (capped), and repeat until a final answer or the
//! iteration cap. Tool errors are returned to the model rather than treated as
//! fatal, so the model can correct itself. The loop is bounded and
//! deterministic.

use std::fmt;

use crate::config::Config;
use crate::provider::{Message, Provider, ProviderError, Response, ToolCall};
use crate::tools::resolver::ToolSet;
use crate::workspace::Workspace;

/// Terminal failures of a session.
#[derive(Debug)]
pub enum AgentError {
    /// The model never produced a final answer within the iteration budget.
    IterationCap(usize),
    Provider(ProviderError),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::IterationCap(n) => {
                write!(
                    f,
                    "iteration cap exceeded: no final answer after {n} iterations"
                )
            }
            AgentError::Provider(e) => write!(f, "provider error: {e}"),
        }
    }
}

impl std::error::Error for AgentError {}

/// Rough token estimate: ~4 characters per token (fine for budgeting, not
/// meant to match a real tokenizer).
fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    chars.div_ceil(4)
}

fn estimate_message_tokens(m: &Message) -> usize {
    match m {
        Message::System(s) | Message::User(s) => estimate_tokens(s) + 4,
        Message::Assistant { text, tool_calls } => {
            let text_tokens = text.as_deref().map(estimate_tokens).unwrap_or(0);
            let call_tokens: usize = tool_calls
                .iter()
                .map(|tc| 8 + estimate_tokens(&tc.name) + estimate_tokens(&tc.args.to_string()))
                .sum();
            text_tokens + call_tokens + 4
        }
        Message::ToolResult { result, .. } => estimate_tokens(result) + 4,
    }
}

fn total_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Drop the oldest assistant-turn blocks (an `Assistant` tool-call message
/// followed by its `ToolResult` messages) while the history exceeds `budget`.
/// The system + user seed (`seed_len`) is never dropped.
fn trim_history(history: &mut Vec<Message>, seed_len: usize, budget: usize) {
    while total_tokens(history) > budget && history.len() > seed_len {
        let Some(first_assistant) = history
            .iter()
            .position(|m| matches!(m, Message::Assistant { .. }))
        else {
            break;
        };
        let mut end = first_assistant + 1;
        while end < history.len() && matches!(history[end], Message::ToolResult { .. }) {
            end += 1;
        }
        history.drain(first_assistant..end);
    }
}

pub struct Agent<'a> {
    provider: &'a dyn Provider,
    tools: &'a ToolSet,
    workspace: &'a Workspace,
    config: &'a Config,
}

impl<'a> Agent<'a> {
    pub fn new(
        provider: &'a dyn Provider,
        tools: &'a ToolSet,
        workspace: &'a Workspace,
        config: &'a Config,
    ) -> Self {
        Self {
            provider,
            tools,
            workspace,
            config,
        }
    }

    /// The system prompt seeding every session: workspace + tool rules.
    pub fn system_prompt(&self) -> String {
        format!(
            "You are crab, a minimal coding agent working in the directory '{}'.\n\
             You may only use these four tools: read, bash, edit, write.\n\
             All file paths are relative to the workspace and must stay inside it.\n\
             Use the tools to inspect and modify the code, then give a final answer.",
            self.workspace.root().display()
        )
    }

    /// Execute a batch of tool calls (or, when `fail_all`, report each as not
    /// executed because the response was truncated) and append the assistant
    /// message plus one tool-result per call to `history`.
    fn push_tool_results(&self, history: &mut Vec<Message>, calls: Vec<ToolCall>, fail_all: bool) {
        history.push(Message::Assistant {
            text: None,
            tool_calls: calls.clone(),
        });
        for call in &calls {
            let result_str = if fail_all {
                format!(
                    "Tool call \"{}\" was not executed: the response was truncated by the output token limit, so its arguments may be incomplete. Re-issue the tool call with complete arguments.",
                    call.name
                )
            } else {
                match self.tools.execute(self.workspace, &call.name, &call.args) {
                    Ok(out) => out.content,
                    Err(e) => format!("tool error: {e}"),
                }
            };
            history.push(Message::ToolResult {
                tool_call_id: call.id.clone(),
                result: result_str,
            });
        }
    }

    /// Run a bounded agentic session for `prompt`, returning the final answer.
    pub fn run(&self, prompt: &str) -> Result<String, AgentError> {
        let mut history = vec![
            Message::System(self.system_prompt()),
            Message::User(prompt.to_string()),
        ];
        let seed_len = history.len();
        let schemas = self.tools.tool_schemas();
        let mut iterations = 0usize;

        loop {
            if iterations >= self.config.max_iterations {
                return Err(AgentError::IterationCap(self.config.max_iterations));
            }
            trim_history(&mut history, seed_len, self.config.max_context_tokens);
            let response = self
                .provider
                .complete(&history, &schemas)
                .map_err(AgentError::Provider)?;

            match response {
                Response::Text(text) => return Ok(text),
                Response::ToolCalls(calls) => self.push_tool_results(&mut history, calls, false),
                Response::TruncatedToolCalls(calls) => {
                    self.push_tool_results(&mut history, calls, true)
                }
            }
            iterations += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::provider::{fake::FakeProvider, ToolCall};

    fn workspace(name: &str) -> (std::path::PathBuf, Workspace) {
        let dir = std::env::temp_dir().join(format!("crab-agent-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        (dir, ws)
    }

    fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            args,
        }
    }

    #[test]
    fn runs_multi_step_session_and_prints_final_answer() {
        let (_dir, ws) = workspace("multi");
        let fake = FakeProvider::new(vec![
            Response::ToolCalls(vec![call(
                "c1",
                "bash",
                serde_json::json!({"command": "echo hi"}),
            )]),
            Response::Text("done here".into()),
        ]);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let agent = Agent::new(&fake, &tools, &ws, &cfg);
        let answer = agent.run("greet me").unwrap();
        assert_eq!(answer, "done here");
        assert_eq!(fake.calls(), 2);
        // The bash result was fed back before the final call.
        assert!(fake.saw_tool_result("c1"));
        let last = fake.history(1);
        assert!(last
            .iter()
            .any(|m| matches!(m, Message::ToolResult { result, .. } if result.contains("hi"))));
    }

    #[test]
    fn stops_after_max_iterations() {
        let (_dir, ws) = workspace("cap");
        let fake = FakeProvider::new(vec![
            Response::ToolCalls(vec![call(
                "c1",
                "bash",
                serde_json::json!({"command": "true"}),
            )]),
            Response::ToolCalls(vec![call(
                "c2",
                "bash",
                serde_json::json!({"command": "true"}),
            )]),
            Response::ToolCalls(vec![call(
                "c3",
                "bash",
                serde_json::json!({"command": "true"}),
            )]),
        ]);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 2,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let agent = Agent::new(&fake, &tools, &ws, &cfg);
        let err = agent.run("keep going").unwrap_err();
        assert!(matches!(err, AgentError::IterationCap(2)));
    }

    #[test]
    fn tool_errors_are_returned_to_model() {
        let (_dir, ws) = workspace("toolerr");
        let fake = FakeProvider::new(vec![
            Response::ToolCalls(vec![call("c1", "frobnicate", serde_json::json!({}))]),
            Response::Text("ok".into()),
        ]);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 5,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let agent = Agent::new(&fake, &tools, &ws, &cfg);
        assert_eq!(agent.run("x").unwrap(), "ok");
        // The error was handed back as a tool result, not fatal.
        let last = fake.history(1);
        assert!(last.iter().any(
            |m| matches!(m, Message::ToolResult { result, .. } if result.contains("unknown tool"))
        ));
    }

    #[test]
    fn truncated_tool_calls_are_not_executed() {
        let (dir, ws) = workspace("truncated");
        let fake = FakeProvider::new(vec![
            Response::TruncatedToolCalls(vec![call(
                "c1",
                "bash",
                serde_json::json!({"command": "touch marker.txt"}),
            )]),
            Response::Text("ok".into()),
        ]);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 5,
            workspace: dir.clone(),
            ..Config::defaults(dir.clone())
        };
        let agent = Agent::new(&fake, &tools, &ws, &cfg);
        assert_eq!(agent.run("x").unwrap(), "ok");
        // The truncated call was NOT executed.
        assert!(!dir.join("marker.txt").exists());
        let last = fake.history(1);
        assert!(last.iter().any(
            |m| matches!(m, Message::ToolResult { result, .. } if result.contains("was not executed"))
        ));
    }

    #[test]
    fn trim_history_drops_oldest_turns() {
        let mut h = vec![
            Message::System("s".into()),
            Message::User("p".into()),
            Message::Assistant {
                text: None,
                tool_calls: vec![call("a", "bash", serde_json::json!({}))],
            },
            Message::ToolResult {
                tool_call_id: "a".into(),
                result: "x".repeat(400),
            },
            Message::Assistant {
                text: None,
                tool_calls: vec![call("b", "bash", serde_json::json!({}))],
            },
            Message::ToolResult {
                tool_call_id: "b".into(),
                result: "x".repeat(400),
            },
        ];
        trim_history(&mut h, 2, 130);
        assert_eq!(h.len(), 4);
        assert!(matches!(&h[0], Message::System(_)));
        assert!(matches!(&h[1], Message::User(_)));
        match &h[2] {
            Message::Assistant { tool_calls, .. } => assert_eq!(tool_calls[0].id, "b"),
            _ => panic!("expected assistant message"),
        }
    }
}
