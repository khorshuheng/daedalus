//! The orchestration loop (CRAB-104).
//!
//! Alternates between the LLM and the tool executor: seed history with the
//! system + user prompt, call the provider, execute any tool calls, feed
//! results back verbatim (capped), and repeat until a final answer or the
//! iteration cap. Tool errors are returned to the model rather than treated as
//! fatal, so the model can correct itself. The loop is bounded and
//! deterministic.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

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

/// Estimate the total tokens of `history`, using `anchor_tokens` as the exact
/// count of `history[..anchor_len]` (reported by the last provider completion)
/// plus a chars/4 estimate of anything appended since.
fn anchored_total(history: &[Message], anchor_tokens: usize, anchor_len: usize) -> usize {
    if anchor_len > history.len() {
        // The anchor was invalidated by trimming; fall back to a full estimate.
        return total_tokens(history);
    }
    anchor_tokens + total_tokens(&history[anchor_len..])
}

/// Drop the oldest assistant-turn blocks (an `Assistant` tool-call message
/// followed by its `ToolResult` messages) while the history exceeds `budget`.
/// The system + user seed (`seed_len`) is never dropped. Removing part of the
/// measured prefix invalidates the usage anchor, which the next completion
/// re-anchors with an exact count.
fn trim_history(
    history: &mut Vec<Message>,
    seed_len: usize,
    budget: usize,
    anchor_tokens: &mut usize,
    anchor_len: &mut usize,
) {
    while anchored_total(history, *anchor_tokens, *anchor_len) > budget && history.len() > seed_len
    {
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
        if first_assistant < *anchor_len {
            *anchor_tokens = 0;
            *anchor_len = 0;
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

/// Shared steering state between the stdin reader thread and the agent loop.
/// `steer` posts a message and cancels the in-flight generation; the loop
/// consumes the message on the next turn and resumes.
pub struct Steering {
    cancel: AtomicBool,
    message: Mutex<Option<String>>,
}

impl Default for Steering {
    fn default() -> Self {
        Self::new()
    }
}

impl Steering {
    pub fn new() -> Self {
        Self {
            cancel: AtomicBool::new(false),
            message: Mutex::new(None),
        }
    }

    /// Post a steering message and request cancellation of the current turn.
    pub fn steer(&self, msg: String) {
        *self.message.lock().unwrap() = Some(msg);
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// The cancel flag checked by providers between streamed chunks.
    pub fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }

    /// Take the pending steering message, clearing the cancel flag if present.
    pub fn take_message(&self) -> Option<String> {
        let msg = self.message.lock().unwrap().take();
        if msg.is_some() {
            self.cancel.store(false, Ordering::SeqCst);
        }
        msg
    }
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
            "You are crab, a minimal coding agent. You inspect and modify files in the workspace '{}' by calling tools.\n\
             You have exactly four tools and no others: read, bash, edit, write.\n\
             - read: read a file or a line range (use offset to page through long files).\n\
             - bash: run a shell command in the workspace; check results before trusting them.\n\
             - edit: apply precise text replacements; each oldText must match exactly once.\n\
             - write: create or overwrite a file.\n\
             Rules:\n\
             - All file paths are relative to the workspace and must stay inside it.\n\
             - Read before editing; verify changes with bash.\n\
             - Make the smallest change that satisfies the request.\n\
             - When finished, give a concise final answer.",
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
        self.run_impl(prompt, None)
    }

    /// Run a session that can be steered: the `Steering` handle is checked for
    /// a pending user message and cancellation between turns and mid-stream.
    pub fn run_steered(&self, prompt: &str, steering: &Steering) -> Result<String, AgentError> {
        self.run_impl(prompt, Some(steering))
    }

    fn run_impl(&self, prompt: &str, steering: Option<&Steering>) -> Result<String, AgentError> {
        let mut history = vec![
            Message::System(self.system_prompt()),
            Message::User(prompt.to_string()),
        ];
        let seed_len = history.len();
        let schemas = self.tools.tool_schemas();
        let mut iterations = 0usize;
        // Exact token count of history[..anchor_len], reported by the last
        // provider completion. Before the first completion we estimate all.
        let mut anchor_tokens = 0usize;
        let mut anchor_len = 0usize;
        // A dummy cancel flag for non-interactive runs.
        let no_cancel = AtomicBool::new(false);
        let cancel = steering.map(|s| s.cancel_flag()).unwrap_or(&no_cancel);

        loop {
            if iterations >= self.config.max_iterations {
                return Err(AgentError::IterationCap(self.config.max_iterations));
            }
            trim_history(
                &mut history,
                seed_len,
                self.config.max_context_tokens,
                &mut anchor_tokens,
                &mut anchor_len,
            );
            let completion = self
                .provider
                .complete(&history, &schemas, cancel)
                .map_err(AgentError::Provider)?;
            if let Some(tokens) = completion.prompt_tokens {
                anchor_tokens = tokens;
                anchor_len = history.len();
            }

            if completion.aborted {
                let partial = match completion.response {
                    Response::Text(t) => t,
                    _ => String::new(),
                };
                // Keep the partial text so the model sees what it was saying.
                if !partial.is_empty() {
                    history.push(Message::Assistant {
                        text: Some(partial.clone()),
                        tool_calls: vec![],
                    });
                }
                if let Some(s) = steering {
                    if let Some(msg) = s.take_message() {
                        history.push(Message::User(msg));
                        continue;
                    }
                }
                // Cancelled with no steering message: stop and return what we had.
                return Ok(partial);
            }

            match completion.response {
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
        let mut anchor_tokens = 0;
        let mut anchor_len = 0;
        trim_history(&mut h, 2, 130, &mut anchor_tokens, &mut anchor_len);
        assert_eq!(h.len(), 4);
        assert!(matches!(&h[0], Message::System(_)));
        assert!(matches!(&h[1], Message::User(_)));
        match &h[2] {
            Message::Assistant { tool_calls, .. } => assert_eq!(tool_calls[0].id, "b"),
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn anchored_total_prefers_exact_anchor() {
        let h = vec![Message::System("s".into()), Message::User("p".into())];
        // The exact anchor (5 tokens for the whole 2-message prefix) is used
        // directly rather than re-estimated.
        assert_eq!(anchored_total(&h, 5, 2), 5);
        // An anchor pointing past the history is invalid, so we estimate all.
        assert!(anchored_total(&h, 5, 99) > 0);
    }

    /// A provider that plays a scripted sequence of `Completion`s.
    struct ScriptedProvider {
        completions: Mutex<std::collections::VecDeque<crate::provider::Completion>>,
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
        ) -> Result<crate::provider::Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            Ok(self
                .completions
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| crate::provider::Completion {
                    response: Response::Text("done".into()),
                    prompt_tokens: None,
                    aborted: false,
                }))
        }
    }

    #[test]
    fn steering_injects_message_and_resumes() {
        let (_dir, ws) = workspace("steer");
        let steering = Steering::new();

        let provider = ScriptedProvider {
            completions: Mutex::new(std::collections::VecDeque::from([
                crate::provider::Completion {
                    response: Response::Text("going the wrong way".into()),
                    prompt_tokens: None,
                    aborted: true,
                },
                crate::provider::Completion {
                    response: Response::Text("fixed answer".into()),
                    prompt_tokens: None,
                    aborted: false,
                },
            ])),
            histories: Mutex::new(Vec::new()),
        };

        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let agent = Agent::new(&provider, &tools, &ws, &cfg);

        steering.steer("stop, do this instead".into());
        let answer = agent.run_steered("initial prompt", &steering).unwrap();
        assert_eq!(answer, "fixed answer");

        // The second completion saw the partial assistant text + steering line.
        let histories = provider.histories.lock().unwrap();
        assert_eq!(histories.len(), 2);
        let second = &histories[1];
        assert!(second.iter().any(
            |m| matches!(m, Message::Assistant { text: Some(t), .. } if t == "going the wrong way")
        ));
        assert!(second
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "stop, do this instead")));
    }
}
