//! The orchestration loop (CRAB-104).
//!
//! Alternates between the LLM and the tool executor: seed history with the
//! system + user prompt, call the provider, execute any tool calls, feed
//! results back verbatim (capped), and repeat until a final answer or the
//! iteration cap. Tool errors are returned to the model rather than treated as
//! fatal, so the model can correct itself. The loop is bounded and
//! deterministic.

use std::fmt;
use std::sync::atomic::AtomicBool;

use crate::config::Config;
use crate::provider::{Message, Provider, ProviderError, Response, ToolCall};
use crate::tools::resolver::ToolSet;
use crate::tools::ToolError;
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
    /// message plus one tool-result per call to `history`. Returns true if a
    /// tool was cancelled.
    fn push_tool_results(
        &self,
        history: &mut Vec<Message>,
        calls: Vec<ToolCall>,
        fail_all: bool,
        cancel: &AtomicBool,
    ) -> bool {
        history.push(Message::Assistant {
            text: None,
            tool_calls: calls.clone(),
        });
        let mut cancelled = false;
        for call in &calls {
            let result_str = if fail_all {
                format!(
                    "Tool call \"{}\" was not executed: the response was truncated by the output token limit, so its arguments may be incomplete. Re-issue the tool call with complete arguments.",
                    call.name
                )
            } else {
                match self
                    .tools
                    .execute(self.workspace, &call.name, &call.args, cancel)
                {
                    Ok(out) => out.content,
                    Err(e) => {
                        cancelled = matches!(e, ToolError::Cancelled);
                        format!("tool error: {e}")
                    }
                }
            };
            history.push(Message::ToolResult {
                tool_call_id: call.id.clone(),
                result: result_str,
            });
        }
        cancelled
    }

    /// Run a one-shot, non-interactive session and return the final answer.
    pub fn run(&self, prompt: &str) -> Result<String, AgentError> {
        let cancel = AtomicBool::new(false);
        let mut session = Session::new(self, prompt, &cancel);
        match session.run_turn(&mut |_| {})? {
            Turn::Final(text) => Ok(text),
            Turn::Cancelled(_) => Ok(String::new()),
        }
    }
}

/// The outcome of a single agent turn.
pub enum Turn {
    /// A final answer from the model.
    Final(String),
    /// The turn was cancelled; `String` holds any partial text (kept).
    Cancelled(String),
}

/// A displayable event emitted while a turn runs.
pub enum Stream {
    /// A streamed text delta from the model.
    Text(String),
    /// Tool names about to execute.
    Tools(Vec<String>),
}

/// A persistent agent session that owns the message history, so a cancelled
/// turn can be steered and resumed (the REPL model).
pub struct Session<'a, 'inner> {
    agent: &'a Agent<'inner>,
    history: Vec<Message>,
    schemas: Vec<serde_json::Value>,
    anchor_tokens: usize,
    anchor_len: usize,
    cancel: &'a AtomicBool,
}

impl<'a, 'inner> Session<'a, 'inner> {
    pub fn new(agent: &'a Agent<'inner>, prompt: &str, cancel: &'a AtomicBool) -> Self {
        let history = vec![
            Message::System(agent.system_prompt()),
            Message::User(prompt.to_string()),
        ];
        let schemas = agent.tools.tool_schemas();
        Self {
            agent,
            history,
            schemas,
            anchor_tokens: 0,
            anchor_len: 0,
            cancel,
        }
    }

    /// Inject a follow-up user message for the next turn.
    pub fn resume(&mut self, msg: String) {
        self.history.push(Message::User(msg));
    }

    /// Run one turn (complete -> tools -> repeat) until a final answer,
    /// cancellation, or error. `emit` receives streamed text and tool markers
    /// for display.
    pub fn run_turn(&mut self, emit: &mut dyn FnMut(Stream)) -> Result<Turn, AgentError> {
        let seed_len = 2; // [System, first User] are never trimmed.
        let mut iterations = 0usize;
        loop {
            if iterations >= self.agent.config.max_iterations {
                return Err(AgentError::IterationCap(self.agent.config.max_iterations));
            }
            trim_history(
                &mut self.history,
                seed_len,
                self.agent.config.max_context_tokens,
                &mut self.anchor_tokens,
                &mut self.anchor_len,
            );
            let completion = self
                .agent
                .provider
                .complete(&self.history, &self.schemas, self.cancel, &mut |t| {
                    emit(Stream::Text(t.to_string()))
                })
                .map_err(AgentError::Provider)?;
            if let Some(tokens) = completion.prompt_tokens {
                self.anchor_tokens = tokens;
                self.anchor_len = self.history.len();
            }

            if completion.aborted {
                let partial = match completion.response {
                    Response::Text(t) => t,
                    _ => String::new(),
                };
                // Keep the partial text so the model sees what it was saying.
                if !partial.is_empty() {
                    self.history.push(Message::Assistant {
                        text: Some(partial.clone()),
                        tool_calls: vec![],
                    });
                }
                return Ok(Turn::Cancelled(partial));
            }

            match completion.response {
                Response::Text(text) => {
                    // Record the final answer so follow-up turns have context.
                    self.history.push(Message::Assistant {
                        text: Some(text.clone()),
                        tool_calls: vec![],
                    });
                    return Ok(Turn::Final(text));
                }
                Response::ToolCalls(calls) => {
                    emit(Stream::Tools(
                        calls.iter().map(|c| c.name.clone()).collect(),
                    ));
                    if self
                        .agent
                        .push_tool_results(&mut self.history, calls, false, self.cancel)
                    {
                        return Ok(Turn::Cancelled(String::new()));
                    }
                }
                Response::TruncatedToolCalls(calls) => {
                    self.agent
                        .push_tool_results(&mut self.history, calls, true, self.cancel);
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
    use std::sync::Mutex;

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
            _on_text: &mut dyn FnMut(&str),
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
    fn session_cancel_then_resume() {
        let (_dir, ws) = workspace("session");
        let cancel = AtomicBool::new(false);

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

        let mut session = Session::new(&agent, "initial prompt", &cancel);
        let first = session.run_turn(&mut |_| {}).unwrap();
        assert!(matches!(first, Turn::Cancelled(_)));

        session.resume("stop, do this instead".into());
        let second = session.run_turn(&mut |_| {}).unwrap();
        match second {
            Turn::Final(text) => assert_eq!(text, "fixed answer"),
            _ => panic!("expected final answer"),
        }

        // The second completion saw the partial assistant text + steering line.
        let histories = provider.histories.lock().unwrap();
        assert_eq!(histories.len(), 2);
        let second_history = &histories[1];
        assert!(second_history.iter().any(
            |m| matches!(m, Message::Assistant { text: Some(t), .. } if t == "going the wrong way")
        ));
        assert!(second_history
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "stop, do this instead")));
    }

    #[test]
    fn final_answer_is_recorded_for_followups() {
        let (_dir, ws) = workspace("finalrec");
        let cancel = AtomicBool::new(false);
        let provider = ScriptedProvider {
            completions: Mutex::new(std::collections::VecDeque::from([
                crate::provider::Completion {
                    response: Response::Text("first answer".into()),
                    prompt_tokens: None,
                    aborted: false,
                },
                crate::provider::Completion {
                    response: Response::Text("second answer".into()),
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
        let mut session = Session::new(&agent, "q1", &cancel);
        assert!(matches!(
            session.run_turn(&mut |_| {}).unwrap(),
            Turn::Final(_)
        ));
        session.resume("q2".into());
        session.run_turn(&mut |_| {}).unwrap();

        // The follow-up turn saw the previous final answer in history.
        let histories = provider.histories.lock().unwrap();
        let second = &histories[1];
        assert!(second
            .iter()
            .any(|m| matches!(m, Message::Assistant { text: Some(t), .. } if t == "first answer")));
    }
}
