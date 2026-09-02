//! Agent runtime (CRAB-116): the stateful, UI-agnostic engine.
//!
//! Replaces the callback-driven turn loop (`Session::run_turn(emit)`) with a
//! single `AgentRuntime` that owns the core state — config, provider, toolset,
//! workspace, and the conversation — and exposes a typed **Event** stream plus
//! a **Command** surface. Every frontend (print, json, rpc, tui, server)
//! becomes a client of this one object: it sends commands and consumes events;
//! it never touches the loop internals.
//!
//! # Threading model
//!
//! The turn loop stays synchronous and single-threaded, running on a worker
//! thread owned by the runtime. Adapters hold a cheap cloneable handle and an
//! event `Receiver`:
//!
//! ```text
//!   adapter ── Command (queue + notify) ──▶ AgentRuntime worker (turn loop)
//!   adapter ◀───── Event (unbounded channel) ── AgentRuntime
//! ```
//!
//! Commands sent while a turn is running are queued and delivered at the pi
//! boundaries: `steer` after the current assistant message finishes its tool
//! calls, `followUp` when the agent stops, `abort` immediately (partial text
//! is kept). Cancellation is per-session (`Arc<AtomicBool>` threaded into
//! providers and tools), so one client's abort never affects another session.
//!
//! # Wire format
//!
//! `Event` and `Command` are serde-tagged on `type` with `snake_case`
//! discriminators and snake_case fields (pi's RPC vocabulary), so the same
//! objects cross stdio RPC (CRAB-120), the TUI (CRAB-121) and the WebSocket
//! server (CRAB-122) unchanged. The vocabulary:
//!
//! ```text
//! Event:   agent_start, text_delta, thinking_delta, tool_start, tool_end,
//!          turn_start, turn_end, usage, queue_update, state_changed,
//!          agent_settled, error
//! Command: prompt, steer, follow_up, abort, set_model, set_effort,
//!          switch_workspace, clear, resume, get_state
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use serde::{Deserialize, Serialize};

use crate::config::{Config, ProviderKind};
use crate::provider::{Message, Provider, Response};
use crate::tools::resolver::ToolSet;
use crate::workspace::Workspace;

/// Canonical thinking level, mapped per provider (CRAB-116). Off disables
/// thinking; the other levels request progressively more reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    #[default]
    Medium,
    Off,
    Minimal,
    Low,
    High,
}

impl Effort {
    /// Parse a canonical level name (`off|minimal|low|medium|high`), or
    /// `None` for anything else.
    pub fn parse(s: &str) -> Option<Effort> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Effort::Off),
            "minimal" => Some(Effort::Minimal),
            "low" => Some(Effort::Low),
            "medium" => Some(Effort::Medium),
            "high" => Some(Effort::High),
            _ => None,
        }
    }

    /// The canonical level name (serialized form).
    pub fn name(self) -> &'static str {
        match self {
            Effort::Off => "off",
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }

    /// OpenAI `reasoning_effort` value for this level; `None` for `off`
    /// (OpenAI has no "off" — a low-effort model is simply called without a
    /// reasoning parameter).
    pub fn openai_reasoning_effort(self) -> Option<&'static str> {
        match self {
            Effort::Off => None,
            Effort::Minimal => Some("minimal"),
            Effort::Low => Some("low"),
            Effort::Medium => Some("medium"),
            Effort::High => Some("high"),
        }
    }

    /// Anthropic thinking budget in tokens for this level (`None` for `off`,
    /// which disables the thinking block). Hardcoded capability table — no
    /// discovery endpoint, per CRAB-116.
    pub fn anthropic_thinking_budget(self) -> Option<usize> {
        match self {
            Effort::Off => None,
            Effort::Minimal => Some(1_024),
            Effort::Low => Some(4_096),
            Effort::Medium => Some(8_192),
            Effort::High => Some(16_384),
        }
    }
}

impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Hardcoded per-provider capability table (CRAB-116): maps a canonical
/// effort level to the wire value the provider understands. No discovery.
pub fn provider_effort(provider: ProviderKind, effort: Effort) -> serde_json::Value {
    match provider {
        ProviderKind::Openai | ProviderKind::Deepseek => match effort.openai_reasoning_effort() {
            Some(v) => serde_json::json!({ "reasoning_effort": v }),
            None => serde_json::json!({}),
        },
        ProviderKind::Anthropic => match effort.anthropic_thinking_budget() {
            Some(budget) => serde_json::json!({
                "thinking": { "type": "enabled", "budget_tokens": budget }
            }),
            None => serde_json::json!({ "thinking": { "type": "disabled" } }),
        },
        ProviderKind::Fake => serde_json::json!({}),
    }
}

/// One event emitted by the runtime, serde-tagged on `type` with snake_case
/// discriminators and fields (pi's RPC event vocabulary). The vocabulary is
/// documented in the module docs and is stable: adapters depend on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// The runtime started; carries the initial session identity.
    AgentStart {
        model: String,
        effort: Effort,
        workspace: String,
    },
    /// A new user message is about to be processed.
    TurnStart {},
    /// A streamed fragment of assistant text.
    TextDelta { text: String },
    /// A streamed fragment of model thinking (effort > off).
    ThinkingDelta { text: String },
    /// A tool call is about to execute.
    ToolStart {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// A tool call finished.
    ToolEnd {
        name: String,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// An assistant turn finished with a final answer (or an empty string
    /// when interrupted).
    TurnEnd {},
    /// Token usage reported by the last completion.
    Usage {
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_tokens: Option<usize>,
    },
    /// The command queue changed (a steer/followUp was queued while busy).
    QueueUpdate { queued: usize, kind: String },
    /// Runtime state changed (model/effort/workspace).
    StateChanged {
        model: String,
        effort: Effort,
        workspace: String,
    },
    /// The agent settled: a final answer (or empty when cancelled).
    AgentSettled { text: String, interrupted: bool },
    /// A non-fatal error surfaced by the runtime.
    Error { message: String },
}

impl Event {
    /// The `type` discriminator of this event (snake_case, stable wire name).
    pub fn type_name(&self) -> &'static str {
        match self {
            Event::AgentStart { .. } => "agent_start",
            Event::TurnStart {} => "turn_start",
            Event::TextDelta { .. } => "text_delta",
            Event::ThinkingDelta { .. } => "thinking_delta",
            Event::ToolStart { .. } => "tool_start",
            Event::ToolEnd { .. } => "tool_end",
            Event::TurnEnd {} => "turn_end",
            Event::Usage { .. } => "usage",
            Event::QueueUpdate { .. } => "queue_update",
            Event::StateChanged { .. } => "state_changed",
            Event::AgentSettled { .. } => "agent_settled",
            Event::Error { .. } => "error",
        }
    }
}

/// A command sent to the runtime, serde-tagged on `type`. The optional `id`
/// lets a request/response adapter correlate replies (e.g. stdio RPC).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Command {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub kind: CommandKind,
}

/// The command kinds, serde-tagged on `type` with snake_case names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandKind {
    /// Start a new turn with `text` as the user message.
    Prompt { text: String },
    /// Queue a steering message: delivered after the current assistant
    /// message finishes its tool calls (pi semantics).
    Steer { text: String },
    /// Queue a follow-up: delivered when the agent stops.
    FollowUp { text: String },
    /// Cancel the in-flight turn immediately, keeping partial text.
    Abort {},
    /// Change the model used for subsequent completions.
    SetModel { model: String },
    /// Change the thinking level.
    SetEffort { effort: Effort },
    /// Change the workspace (and re-seed the system prompt).
    SwitchWorkspace { path: String },
    /// Reset the conversation to a fresh context (system prompt only).
    Clear {},
    /// Ask the runtime to report its current state.
    GetState {},
}

/// Internal queue item: a wire command or a worker shutdown request.
#[derive(Debug, Clone, PartialEq)]
enum Control {
    Command(CommandKind),
    Shutdown,
}

/// Terminal errors surfaced by the synchronous `run_once` path. The worker
/// path reports these as `Event::Error` instead.
#[derive(Debug, PartialEq)]
pub enum RuntimeError {
    /// The model never produced a final answer within the iteration budget.
    IterationCap(usize),
    /// A provider failure.
    Provider(String),
}

impl RuntimeError {
    fn from_message(msg: String) -> Self {
        if let Some(rest) = msg.strip_prefix("iteration cap exceeded: no final answer after ") {
            if let Some((n, _)) = rest.split_once(' ') {
                if let Ok(n) = n.parse() {
                    return RuntimeError::IterationCap(n);
                }
            }
        }
        RuntimeError::Provider(msg)
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::IterationCap(n) => write!(
                f,
                "iteration cap exceeded: no final answer after {n} iterations"
            ),
            RuntimeError::Provider(e) => write!(f, "provider error: {e}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// A snapshot of the runtime's mutable state, reported via `get_state` and
/// carried in `state_changed` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RuntimeState {
    pub model: String,
    pub effort: Effort,
    pub workspace: String,
    pub busy: bool,
}

/// Rough token estimate: ~4 characters per token (fine for budgeting).
fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4)
    }
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
/// count of `history[..anchor_len]` plus a chars/4 estimate of what follows.
fn anchored_total(history: &[Message], anchor_tokens: usize, anchor_len: usize) -> usize {
    if anchor_len > history.len() {
        return total_tokens(history);
    }
    anchor_tokens + total_tokens(&history[anchor_len..])
}

/// Drop the oldest assistant-turn blocks while the history exceeds `budget`.
/// The seed (`history[..seed_len]`) is never dropped.
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

/// The shared state behind an `AgentRuntime` handle. The worker thread owns
/// the turn loop and mutates the conversation; adapters send commands through
/// the queue and read events from the channel. `history` is behind a mutex so
/// adapters can snapshot it (session save / reflect) while a turn runs.
struct Inner {
    config: Config,
    provider: Arc<dyn Provider>,
    tools: ToolSet,
    /// Canonical workspace; `switch_workspace` replaces it (re-seeding the
    /// system prompt). Read by adapters (session/memory keying).
    workspace: Mutex<Workspace>,
    memory_root: Option<PathBuf>,
    /// Mutable runtime state (model/effort), guarded for adapter `get_state`.
    state: Mutex<RuntimeState>,
    /// Per-session cancel, threaded into providers and tools (replaces the
    /// process-global `term::cancel_flag`).
    cancel: AtomicBool,
    busy: AtomicBool,
    history: Mutex<Vec<Message>>,
    anchor: Mutex<(usize, usize)>,
    /// Tool argument schemas for the provider `tools` field (built once).
    schemas: Vec<serde_json::Value>,
    /// Incoming commands; the worker blocks on `cond` when idle and drains
    /// the queue at turn boundaries while busy. `Shutdown` stops the worker.
    queue: Mutex<VecDeque<Control>>,
    cond: Condvar,
    events: Sender<Event>,
}

/// A cloneable handle to a running agent. Construct with `AgentRuntime::new`
/// to get the handle plus its event `Receiver`; drive it with `run_forever`
/// on a thread the adapter chooses (the REPL spawns one; CRAB-122 gives each
/// connected session its own thread).
#[derive(Clone)]
pub struct AgentRuntime {
    inner: Arc<Inner>,
}

impl AgentRuntime {
    /// Build a runtime owning all core state and return it with the event
    /// receiver. No thread is spawned yet; call `run_forever` to start the
    /// worker loop.
    pub fn new(
        config: Config,
        provider: Box<dyn Provider>,
        tools: ToolSet,
        workspace: Workspace,
        memory_root: Option<PathBuf>,
    ) -> (AgentRuntime, Receiver<Event>) {
        let (events, rx) = mpsc::channel();
        let schemas = tools.tool_schemas();
        let model = config.model.clone();
        let effort = Effort::Medium;
        let ws_path = workspace.root().to_string_lossy().into_owned();
        let runtime = AgentRuntime {
            inner: Arc::new(Inner {
                config,
                provider: Arc::from(provider),
                tools,
                workspace: Mutex::new(workspace),
                memory_root,
                state: Mutex::new(RuntimeState {
                    model,
                    effort,
                    workspace: ws_path,
                    busy: false,
                }),
                cancel: AtomicBool::new(false),
                busy: AtomicBool::new(false),
                history: Mutex::new(Vec::new()),
                anchor: Mutex::new((0, 0)),
                schemas,
                queue: Mutex::new(VecDeque::new()),
                cond: Condvar::new(),
                events,
            }),
        };
        (runtime, rx)
    }

    // --- command surface (thread-safe, non-blocking) ---

    fn push(&self, kind: CommandKind) {
        let mut q = self.inner.queue.lock().unwrap();
        q.push_back(Control::Command(kind));
        self.inner.cond.notify_all();
    }

    /// Start a turn with `text` as the user message (queued if busy, then
    /// delivered when the current turn settles).
    pub fn prompt(&self, text: &str) {
        self.push(CommandKind::Prompt {
            text: text.to_string(),
        });
    }

    /// Queue a steering message: while busy it is delivered after the current
    /// assistant message finishes its tool calls; while idle it starts a turn.
    pub fn steer(&self, text: &str) {
        self.push(CommandKind::Steer {
            text: text.to_string(),
        });
    }

    /// Queue a follow-up, delivered when the agent stops (after `settle`).
    pub fn follow_up(&self, text: &str) {
        self.push(CommandKind::FollowUp {
            text: text.to_string(),
        });
    }

    /// Abort the in-flight turn immediately, keeping partial text. Safe to
    /// call from any thread while a turn runs.
    pub fn abort(&self) {
        self.inner.cancel.store(true, Ordering::SeqCst);
        self.push(CommandKind::Abort {});
    }

    /// Change the model for subsequent completions.
    pub fn set_model(&self, model: &str) {
        self.push(CommandKind::SetModel {
            model: model.to_string(),
        });
    }

    /// Change the thinking level.
    pub fn set_effort(&self, effort: Effort) {
        self.push(CommandKind::SetEffort { effort });
    }

    /// Switch the workspace, re-seeding the system prompt.
    pub fn switch_workspace(&self, path: &str) {
        self.push(CommandKind::SwitchWorkspace {
            path: path.to_string(),
        });
    }

    /// Reset the conversation to a fresh context (system prompt only).
    pub fn clear(&self) {
        self.push(CommandKind::Clear {});
    }

    /// Synchronously reset the conversation to a fresh seed. Intended for
    /// handlers that run while the worker is idle (e.g. the REPL `/clear`);
    /// adapters driving through the queue use `clear()`.
    pub fn reset_sync(&self) {
        self.reset_to_seed();
    }

    /// Ask the runtime to emit a `state_changed` event with current state.
    pub fn get_state(&self) {
        self.push(CommandKind::GetState {});
    }

    /// Signal the worker to exit its `run_forever` loop after any in-flight
    /// turn settles.
    pub fn shutdown(&self) {
        let mut q = self.inner.queue.lock().unwrap();
        q.push_back(Control::Shutdown);
        self.inner.cond.notify_all();
    }

    // --- adapter accessors ---

    /// A snapshot of the current history (system prompt onward).
    pub fn history(&self) -> Vec<Message> {
        self.inner.history.lock().unwrap().clone()
    }

    /// Replace the conversation wholesale (e.g. `/resume` loads a previous
    /// session). The first message must be the system prompt.
    pub fn replace_history(&self, history: Vec<Message>) {
        let mut h = self.inner.history.lock().unwrap();
        *h = history;
        *self.inner.anchor.lock().unwrap() = (0, 0);
    }

    /// The canonical workspace root (for session/memory keying).
    pub fn workspace_root(&self) -> PathBuf {
        self.inner.workspace.lock().unwrap().root().to_path_buf()
    }

    /// The provider backing this runtime (reflection reuses it, CRAB-112).
    pub fn provider(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.inner.provider)
    }

    /// Current runtime state snapshot.
    pub fn state(&self) -> RuntimeState {
        self.inner.state.lock().unwrap().clone()
    }

    /// True while a turn is running (for busy detection in adapters).
    pub fn is_busy(&self) -> bool {
        self.inner.busy.load(Ordering::Relaxed)
    }

    // ------------------------------------------------------------------
    // Worker loop
    // ------------------------------------------------------------------

    /// Drive the worker loop forever (until `shutdown`). Blocks the calling
    /// thread; adapters spawn this on a thread of their choosing (the REPL
    /// spawns one, CRAB-122 gives each connected session its own).
    pub fn run_forever(&self) {
        {
            let st = self.inner.state.lock().unwrap();
            let agent_start = Event::AgentStart {
                model: st.model.clone(),
                effort: st.effort,
                workspace: self.ws_path(),
            };
            drop(st);
            self.emit(agent_start);
        }
        while let Some(kind) = self.wait_for_command() {
            match kind {
                CommandKind::Prompt { text } => self.drive_until_settled(&text),
                // While idle a steer/follow-up is simply a new turn.
                CommandKind::Steer { text } | CommandKind::FollowUp { text } => {
                    self.drive_until_settled(&text)
                }
                CommandKind::Abort {} => self.cancel_clear(),
                CommandKind::GetState {} => self.emit_state_changed(),
                CommandKind::Clear {} => self.reset_to_seed(),
                CommandKind::SetModel { .. }
                | CommandKind::SetEffort { .. }
                | CommandKind::SwitchWorkspace { .. } => self.apply_state_command(kind),
            }
        }
    }

    /// Run a single prompt to completion on this thread (no worker), returning
    /// the final answer. Used by the piped/one-shot CLI path and integration
    /// tests. Errors (iteration cap, provider failure) are returned as
    /// `Err(RuntimeError)`.
    pub fn run_once(&self, prompt: &str) -> Result<String, RuntimeError> {
        self.set_busy(true);
        self.cancel_clear();
        let mut saved = VecDeque::new();
        let (text, _interrupted, error) = self.run_one_user_message(prompt, &mut saved);
        self.emit(Event::AgentSettled {
            text: text.clone(),
            interrupted: false,
        });
        self.set_busy(false);
        match error {
            Some(msg) => Err(RuntimeError::from_message(msg)),
            None => Ok(text),
        }
    }

    // ------------------------------------------------------------------
    // Turn engine (runs on the worker thread)
    // ------------------------------------------------------------------

    /// Emit an event to the adapter channel.
    fn emit(&self, event: Event) {
        let _ = self.inner.events.send(event);
    }

    fn workspace_path(&self) -> PathBuf {
        self.inner.workspace.lock().unwrap().root().to_path_buf()
    }

    fn ws_path(&self) -> String {
        self.inner
            .workspace
            .lock()
            .unwrap()
            .root()
            .to_string_lossy()
            .into_owned()
    }

    fn set_busy(&self, busy: bool) {
        self.inner.busy.store(busy, Ordering::SeqCst);
        let mut st = self.inner.state.lock().unwrap();
        st.busy = busy;
        st.workspace = self.workspace_path().to_string_lossy().into_owned();
    }

    fn emit_state_changed(&self) {
        let st = self.inner.state.lock().unwrap().clone();
        self.emit(Event::StateChanged {
            model: st.model,
            effort: st.effort,
            workspace: st.workspace,
        });
    }

    /// The seed system prompt for the current workspace. Memory injection
    /// (CRAB-114) appends the top lessons ranked against `task`.
    fn system_prompt(&self, task: &str) -> String {
        let base = format!(
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
            self.workspace_path().display()
        );
        let Some(root) = &self.inner.memory_root else {
            return base;
        };
        let ws = self.workspace_path();
        let budget = (self.inner.config.max_context_tokens / 20).max(64);
        match crate::index::injection_block(root, &ws, task, budget) {
            Ok(block) if !block.is_empty() => format!("{base}\n\n{block}"),
            _ => base,
        }
    }

    /// Reset the conversation to a fresh seed: system prompt only, ranked
    /// against nothing yet (the first user message re-seeds via the turn
    /// engine).
    fn reset_to_seed(&self) {
        let mut h = self.inner.history.lock().unwrap();
        h.clear();
        h.push(Message::System(self.system_prompt("")));
        *self.inner.anchor.lock().unwrap() = (0, 0);
    }

    fn cancel_clear(&self) {
        self.inner.cancel.store(false, Ordering::SeqCst);
    }

    /// Apply a set_model/set_effort/switch_workspace command to shared state.
    fn apply_state_command(&self, kind: CommandKind) {
        match kind {
            CommandKind::SetModel { model } => {
                self.inner.state.lock().unwrap().model = model;
                self.emit_state_changed();
            }
            CommandKind::SetEffort { effort } => {
                self.inner.state.lock().unwrap().effort = effort;
                self.emit_state_changed();
            }
            CommandKind::SwitchWorkspace { path } => match Workspace::new(PathBuf::from(&path)) {
                Ok(ws) => {
                    *self.inner.workspace.lock().unwrap() = ws;
                    self.reset_to_seed();
                    self.emit_state_changed();
                }
                Err(e) => self.emit(Event::Error {
                    message: format!("cannot switch workspace to '{path}': {e}"),
                }),
            },
            _ => {}
        }
    }

    /// Pop the oldest queued command without blocking, or `None` (also when
    /// the next item is a shutdown request).
    fn pop_queued(&self) -> Option<CommandKind> {
        let mut q = self.inner.queue.lock().unwrap();
        match q.pop_front() {
            Some(Control::Command(kind)) => Some(kind),
            Some(Control::Shutdown) | None => None,
        }
    }

    /// Block until the queue has a command, then pop it. Returns `None` on a
    /// shutdown request so the worker loop can exit.
    fn wait_for_command(&self) -> Option<CommandKind> {
        let mut q = self.inner.queue.lock().unwrap();
        loop {
            match q.pop_front() {
                Some(Control::Command(kind)) => return Some(kind),
                Some(Control::Shutdown) => return None,
                None => q = self.inner.cond.wait(q).unwrap(),
            }
        }
    }

    /// Drain all currently queued commands and classify them into the pending
    /// steer / follow-up messages plus immediate actions (pi semantics):
    /// - `Steer` is delivered after the current assistant message finishes
    ///   its tool calls;
    /// - `FollowUp` is delivered when the agent stops;
    /// - `Abort` cancels immediately;
    /// - set_*/get_state/clear apply immediately where safe.
    fn drain_queue(&self) -> (Vec<String>, Vec<String>) {
        let mut steers = Vec::new();
        let mut follow_ups = Vec::new();
        while let Some(kind) = self.pop_queued() {
            match kind {
                CommandKind::Steer { text } => steers.push(text),
                CommandKind::FollowUp { text } => follow_ups.push(text),
                CommandKind::Prompt { text } => follow_ups.push(text), // busy prompt = follow-up
                CommandKind::Abort {} => {
                    self.inner.cancel.store(true, Ordering::SeqCst);
                }
                CommandKind::GetState {} => self.emit_state_changed(),
                CommandKind::Clear {} => self.reset_to_seed(),
                CommandKind::SetModel { .. }
                | CommandKind::SetEffort { .. }
                | CommandKind::SwitchWorkspace { .. } => self.apply_state_command(kind),
            }
        }
        (steers, follow_ups)
    }

    /// Append `text` as the next user message, replacing the seed system
    /// prompt with one ranked against it when memory is enabled.
    fn push_user(&self, text: &str) {
        let mut h = self.inner.history.lock().unwrap();
        // Refresh the seed system prompt (history[0]) for this task.
        if h.is_empty() {
            h.push(Message::System(self.system_prompt(text)));
        } else if let Some(Message::System(first)) = h.first_mut() {
            *first = self.system_prompt(text);
        }
        h.push(Message::User(text.to_string()));
    }

    /// Run the whole loop for one user message (the initial prompt of a turn)
    /// until the model gives a final answer, is cancelled, or errors. Tool
    /// results are fed back verbatim; text deltas and tool lifecycle emit
    /// events. Runs entirely on the worker thread.
    ///
    /// Returns the terminal text, whether the turn was interrupted, and an
    /// error description when the turn ended on an error (iteration cap or
    /// provider failure — also emitted as `Event::Error`).
    fn run_one_user_message(
        &self,
        user_text: &str,
        saved: &mut VecDeque<String>,
    ) -> (String, bool, Option<String>) {
        self.emit(Event::TurnStart {});
        self.push_user(user_text);
        let seed_len = 2; // [System, first User] are never trimmed.
        let mut iterations = 0usize;
        let mut final_text = String::new();
        let mut interrupted = false;
        let mut error: Option<String> = None;
        'steps: loop {
            // Drain anything queued between LLM calls: abort cancels, steer is
            // delivered only after an assistant tool phase (handled below),
            // follow-ups wait for settle (kept in `saved`).
            let (steers, followups) = self.drain_queue();
            saved.extend(followups);
            if !steers.is_empty() {
                // A steer arrived after the assistant's tool phase: append it
                // as a user message and keep looping (no settle).
                for steer in steers {
                    self.push_user(&steer);
                    iterations = 0; // fresh turn budget for the steer
                }
            }

            if self.inner.cancel.load(Ordering::SeqCst) {
                interrupted = true;
                break 'steps;
            }
            if iterations >= self.inner.config.max_iterations {
                let msg = format!(
                    "iteration cap exceeded: no final answer after {} iterations",
                    self.inner.config.max_iterations
                );
                self.emit(Event::Error {
                    message: msg.clone(),
                });
                error = Some(msg);
                interrupted = true;
                break 'steps;
            }

            {
                let mut h = self.inner.history.lock().unwrap();
                let (mut a0, mut a1) = *self.inner.anchor.lock().unwrap();
                trim_history(
                    &mut h,
                    seed_len,
                    self.inner.config.max_context_tokens,
                    &mut a0,
                    &mut a1,
                );
                *self.inner.anchor.lock().unwrap() = (a0, a1);
            }

            let completion = {
                let h = self.inner.history.lock().unwrap();
                let schemas = &self.inner.schemas;
                let cancel = &self.inner.cancel;
                let emit = &self.inner.events;
                self.inner.provider.complete(&h, schemas, cancel, &mut |t| {
                    let _ = emit.send(Event::TextDelta {
                        text: t.to_string(),
                    });
                })
            };
            match completion {
                Err(e) => {
                    self.emit(Event::Error {
                        message: format!("provider error: {e}"),
                    });
                    break 'steps;
                }
                Ok(completion) => {
                    if let Some(tokens) = completion.prompt_tokens {
                        self.emit(Event::Usage {
                            prompt_tokens: Some(tokens),
                        });
                    }
                    if completion.aborted {
                        // Keep the partial text so the model sees what it was
                        // saying; mark interrupted.
                        let partial = match completion.response {
                            Response::Text(t) => t,
                            _ => String::new(),
                        };
                        if !partial.is_empty() {
                            let mut h = self.inner.history.lock().unwrap();
                            h.push(Message::Assistant {
                                text: Some(partial.clone()),
                                tool_calls: vec![],
                            });
                            final_text = partial;
                        }
                        interrupted = true;
                        break 'steps;
                    }
                    match completion.response {
                        Response::Text(text) => {
                            let mut h = self.inner.history.lock().unwrap();
                            h.push(Message::Assistant {
                                text: Some(text.clone()),
                                tool_calls: vec![],
                            });
                            final_text = text;
                            break 'steps;
                        }
                        Response::ToolCalls(calls) => {
                            let mut h = self.inner.history.lock().unwrap();
                            h.push(Message::Assistant {
                                text: None,
                                tool_calls: calls.clone(),
                            });
                            drop(h);
                            let mut tool_cancelled = false;
                            for call in &calls {
                                self.emit(Event::ToolStart {
                                    name: call.name.clone(),
                                    id: Some(call.id.clone()),
                                });
                                let result_str = {
                                    let ws = self.inner.workspace.lock().unwrap();
                                    let cancel = &self.inner.cancel;
                                    match self
                                        .inner
                                        .tools
                                        .execute(&ws, &call.name, &call.args, cancel)
                                    {
                                        Ok(out) => out.content,
                                        Err(e) => {
                                            tool_cancelled =
                                                matches!(e, crate::tools::ToolError::Cancelled);
                                            format!("tool error: {e}")
                                        }
                                    }
                                };
                                self.emit(Event::ToolEnd {
                                    name: call.name.clone(),
                                    ok: !result_str.starts_with("tool error:")
                                        && !result_str.contains("was not executed"),
                                    error: result_str
                                        .strip_prefix("tool error:")
                                        .map(|s| s.trim().to_string()),
                                });
                                let mut h = self.inner.history.lock().unwrap();
                                h.push(Message::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    result: result_str,
                                });
                            }
                            if tool_cancelled {
                                interrupted = true;
                                break 'steps;
                            }
                            // Loop: feed results back for another completion,
                            // unless a steer was queued meanwhile.
                            iterations += 1;
                            continue 'steps;
                        }
                        Response::TruncatedToolCalls(calls) => {
                            let mut h = self.inner.history.lock().unwrap();
                            h.push(Message::Assistant {
                                text: None,
                                tool_calls: calls.clone(),
                            });
                            drop(h);
                            for call in &calls {
                                let result_str = format!(
                                    "Tool call \"{}\" was not executed: the response was truncated by the output token limit, so its arguments may be incomplete. Re-issue the tool call with complete arguments.",
                                    call.name
                                );
                                let mut h = self.inner.history.lock().unwrap();
                                h.push(Message::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    result: result_str,
                                });
                            }
                            iterations += 1;
                            continue 'steps;
                        }
                    }
                }
            }
        }
        self.emit(Event::TurnEnd {});
        (final_text, interrupted, error)
    }

    /// Drive a full conversation from a first user message until the agent
    /// settles: emit events, honor queued steers/follow-ups at the pi
    /// boundaries, honor abort, and loop into follow-ups automatically.
    fn drive_until_settled(&self, first: &str) {
        self.set_busy(true);
        self.cancel_clear();
        let mut saved = VecDeque::new();
        // First message runs immediately.
        let (mut final_text, mut interrupted, _) = self.run_one_user_message(first, &mut saved);
        // Follow-ups queued while busy are delivered when the agent stops.
        loop {
            if interrupted {
                // Cancelled mid-turn: stop delivering queued messages.
                break;
            }
            let (steers, followups) = self.drain_queue();
            saved.extend(followups);
            saved.extend(steers);
            let Some(next) = saved.pop_front() else { break };
            let (text, was_interrupted, _) = self.run_one_user_message(&next, &mut saved);
            final_text = text;
            interrupted = was_interrupted;
        }
        self.emit(Event::AgentSettled {
            text: final_text,
            interrupted,
        });
        self.set_busy(false);
        self.cancel_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::provider::{Completion, ProviderError, ToolCall};
    use std::sync::Mutex;

    /// A provider whose responses are a scripted queue of `Completion`s.
    struct GateProvider {
        completions: Mutex<VecDeque<Completion>>,
    }

    impl GateProvider {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                completions: Mutex::new(
                    responses
                        .into_iter()
                        .map(|response| Completion {
                            response,
                            prompt_tokens: Some(1),
                            aborted: false,
                        })
                        .collect(),
                ),
            }
        }
    }

    impl Provider for GateProvider {
        fn complete(
            &self,
            _history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<Completion, ProviderError> {
            let mut q = self.completions.lock().unwrap();
            Ok(q.pop_front().unwrap_or(Completion {
                response: Response::Text("done".into()),
                prompt_tokens: None,
                aborted: false,
            }))
        }
    }

    fn workspace(name: &str) -> (std::path::PathBuf, Workspace) {
        let dir = std::env::temp_dir().join(format!("crab-runtime-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = Workspace::new(dir.clone()).unwrap();
        (dir, ws)
    }

    fn runtime_with(
        name: &str,
        provider: Box<dyn Provider>,
    ) -> (
        AgentRuntime,
        Receiver<Event>,
        std::thread::JoinHandle<()>,
        Workspace,
    ) {
        let (_dir, ws) = workspace(name);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let (rt, rx) = AgentRuntime::new(cfg, provider, tools, ws.clone(), None);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        (rt, rx, handle, ws)
    }

    /// Collect events until `agent_settled` (or a short timeout) and return
    /// them plus whether the run settled.
    fn collect_until_settled(rx: &Receiver<Event>) -> (Vec<Event>, bool) {
        let mut events = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(e) => {
                    let settled = matches!(e, Event::AgentSettled { .. });
                    events.push(e);
                    if settled {
                        return (events, true);
                    }
                }
                Err(_) if std::time::Instant::now() > deadline => return (events, false),
                Err(_) => continue,
            }
        }
    }

    fn text(text: &str) -> String {
        text.to_string()
    }

    #[test]
    fn iteration_cap_emits_error_and_settles() {
        // Two tool-call completions then no final text; max_iterations = 2.
        let (_dir, ws) = workspace("cap");
        let provider = Box::new(GateProvider::new(vec![
            Response::ToolCalls(vec![ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "true"}),
            }]),
            Response::ToolCalls(vec![ToolCall {
                id: "c2".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "true"}),
            }]),
            Response::ToolCalls(vec![ToolCall {
                id: "c3".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "true"}),
            }]),
        ]));
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 2,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let (rt, rx) = AgentRuntime::new(cfg, provider, tools, ws.clone(), None);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("keep going");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled);
        assert!(events.iter().any(|e| matches!(e, Event::Error { .. })));
        let settled_event = events.iter().find_map(|e| match e {
            Event::AgentSettled { interrupted, .. } => Some(*interrupted),
            _ => None,
        });
        assert_eq!(settled_event, Some(true), "cap should settle interrupted");
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn clear_resets_conversation_and_set_effort_emits_state() {
        let (rt, rx, handle, _ws) = runtime_with(
            "clear",
            Box::new(GateProvider::new(vec![
                Response::Text(text("a")),
                Response::Text(text("b")),
            ])),
        );
        rt.prompt("first");
        collect_until_settled(&rx);
        rt.clear();
        rt.set_effort(Effort::High);
        rt.prompt("second");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled);
        // After /clear the first exchange is gone from history: only the
        // seed, the second user message, and its answer remain.
        let h = rt.history();
        assert_eq!(h.len(), 3, "seed + second exchange after clear: {h:?}");
        assert!(!h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "first")));
        assert!(!h
            .iter()
            .any(|m| matches!(m, Message::Assistant { text: Some(t), .. } if t == "a")));
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "second")));
        // set_effort was applied and reported via state_changed.
        let state = rt.state();
        assert_eq!(state.effort, Effort::High);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::StateChanged { .. })));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn two_runtimes_are_isolated() {
        // A cancel on one runtime must not affect the other.
        let (rt_a, rx_a, handle_a, _ws_a) = runtime_with("iso-a", Box::new(AbortProvider));
        let (rt_b, rx_b, handle_b, _ws_b) = runtime_with(
            "iso-b",
            Box::new(GateProvider::new(vec![Response::Text(text("b ok"))])),
        );
        rt_a.abort(); // cancels a's in-flight turn only
        rt_b.prompt("hi");
        let (events_b, settled_b) = collect_until_settled(&rx_b);
        assert!(settled_b, "runtime b must be unaffected by a's abort");
        assert!(events_b.iter().any(|e| matches!(
            e,
            Event::AgentSettled {
                text,
                interrupted: false
            } if text == "b ok"
        )));
        rt_a.shutdown();
        rt_b.shutdown();
        handle_a.join().unwrap_or(());
        handle_b.join().unwrap_or(());
        let _ = (rx_a, _ws_a, _ws_b);
    }

    #[test]
    fn prompt_runs_to_settled_and_emits_events() {
        let (rt, rx, handle, _ws) = runtime_with(
            "basic",
            Box::new(GateProvider::new(vec![Response::Text(text("hello world"))])),
        );
        rt.prompt("greet me");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled, "must settle: {events:?}");
        assert!(events.iter().any(|e| matches!(e, Event::AgentStart { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::TurnStart {})));
        let settled_event = events.iter().find_map(|e| match e {
            Event::AgentSettled { text, interrupted } => Some((text.clone(), *interrupted)),
            _ => None,
        });
        assert_eq!(settled_event, Some(("hello world".to_string(), false)));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn tool_calls_execute_and_emit_lifecycle_events() {
        let (_dir, ws) = workspace("tools");
        let builder = GateProvider::new(vec![
            Response::ToolCalls(vec![ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "echo hi"}),
            }]),
            Response::Text(text("done")),
        ]);
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let (rt, rx) = AgentRuntime::new(cfg, Box::new(builder), tools, ws.clone(), None);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("run a command");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolStart { name, .. } if name == "bash")));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { name, ok: true, .. } if name == "bash")));
        // The bash result was fed back before the final completion.
        let h = rt.history();
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::ToolResult { result, .. } if result.contains("hi"))));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn abort_keeps_partial_text_and_settles_interrupted() {
        // A scripted abort: the provider reports an aborted completion with
        // partial text, so the runtime keeps it and settles interrupted.
        let (rt, rx, handle, _ws) = runtime_with("abort", Box::new(AbortProvider));
        rt.prompt("tell me a story");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled);
        let settled_event = events.iter().find_map(|e| match e {
            Event::AgentSettled { text, interrupted } => Some((text.clone(), *interrupted)),
            _ => None,
        });
        assert_eq!(settled_event, Some(("partial story".to_string(), true)));
        // The partial text was recorded so a follow-up has context.
        let h = rt.history();
        assert!(h.iter().any(
            |m| matches!(m, Message::Assistant { text: Some(t), .. } if t == "partial story")
        ));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    /// A provider scripted to pause on each call so the test can queue
    /// commands mid-turn deterministically.
    struct PausableProvider {
        completions: Mutex<VecDeque<Completion>>,
        /// Receives one "release" signal per provider call from the test.
        release_rx: Mutex<Receiver<()>>,
        /// Sent to the test each time a call starts (flow control).
        started_tx: Sender<()>,
    }

    impl Provider for PausableProvider {
        fn complete(
            &self,
            _history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<Completion, ProviderError> {
            let _ = self.started_tx.send(());
            let _ = self
                .release_rx
                .lock()
                .unwrap()
                .recv_timeout(std::time::Duration::from_secs(5));
            let mut q = self.completions.lock().unwrap();
            Ok(q.pop_front().unwrap_or(Completion {
                response: Response::Text("done".into()),
                prompt_tokens: None,
                aborted: false,
            }))
        }
    }

    #[test]
    fn steer_delivered_after_assistant_finishes_tool_calls() {
        let (_dir, ws) = workspace("steer");
        let mut completions = VecDeque::new();
        completions.push_back(Completion {
            response: Response::ToolCalls(vec![ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                args: serde_json::json!({"command": "echo hi"}),
            }]),
            prompt_tokens: None,
            aborted: false,
        });
        completions.push_back(Completion {
            response: Response::Text("steered answer".into()),
            prompt_tokens: None,
            aborted: false,
        });
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let provider = PausableProvider {
            completions: Mutex::new(completions),
            release_rx: Mutex::new(release_rx),
            started_tx,
        };
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let (rt, rx) = AgentRuntime::new(cfg, Box::new(provider), tools, ws.clone(), None);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());

        // Let call 1 (tool phase) start; wait until it is in flight.
        rt.prompt("do the thing");
        let _ = started_rx.recv_timeout(std::time::Duration::from_secs(5));
        // The assistant is mid tool phase; queue a steer, then release call 1.
        rt.steer("no, do it the other way");
        let _ = release_tx.send(());
        // Release call 2 (produces the final text after the steer).
        let _ = started_rx.recv_timeout(std::time::Duration::from_secs(5));
        let _ = release_tx.send(());
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled, "steer must eventually settle: {events:?}");
        let settled_event = events.iter().find_map(|e| match e {
            Event::AgentSettled { text, .. } => Some(text.clone()),
            _ => None,
        });
        assert_eq!(settled_event.as_deref(), Some("steered answer"));
        // The steer text was delivered as a user message after the tool phase.
        let h = rt.history();
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "no, do it the other way")));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn follow_up_runs_after_agent_settles() {
        let (_dir, ws) = workspace("followup");
        let provider = Box::new(GateProvider::new(vec![
            Response::Text(text("first answer")),
            Response::Text(text("second answer")),
        ]));
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: _dir.clone(),
            ..Config::defaults(_dir.clone())
        };
        let (rt, rx) = AgentRuntime::new(cfg, provider, tools, ws.clone(), None);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("first question");
        // Queue the follow-up while the first turn is still running.
        rt.follow_up("second question");
        let (events, settled) = collect_until_settled(&rx);
        assert!(settled, "must settle after follow-up: {events:?}");
        let settled_text = events
            .iter()
            .filter_map(|e| match e {
                Event::AgentSettled { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            settled_text.last().map(String::as_str),
            Some("second answer")
        );
        let h = rt.history();
        assert!(h
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "second question")));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn memory_lessons_are_injected_into_the_system_prompt() {
        let (dir, ws) = workspace("mem-inject");
        let memory_root = dir.join("memory");
        // Seed a lesson about building with make.
        let lesson = crate::memory::Lesson {
            id: "l1".into(),
            text: "always build with make, never cargo".into(),
            kind: "rule".into(),
            tags: vec!["build".into()],
            cwd: ws.root().to_string_lossy().into_owned(),
            source_session_id: Some("s1".into()),
            created_at: 1,
            retracted: false,
        };
        crate::memory::append_lesson(&memory_root, ws.root(), &lesson).unwrap();

        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.clone(),
            ..Config::defaults(dir.clone())
        };
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(RecordingProvider::default()),
            tools,
            ws.clone(),
            Some(memory_root),
        );
        let answer = rt.run_once("how do i build").unwrap();
        assert_eq!(answer, "done");
        // The seed system prompt included the retrieved lesson.
        let h = rt.history();
        let Some(Message::System(system)) = h.first() else {
            panic!("history must start with a system prompt");
        };
        assert!(
            system.contains("always build with make"),
            "lesson injected: {system}"
        );
    }

    /// A provider that answers "done" and records histories.
    #[derive(Default)]
    struct RecordingProvider {
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for RecordingProvider {
        fn complete(
            &self,
            history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            Ok(Completion {
                response: Response::Text("done".into()),
                prompt_tokens: None,
                aborted: false,
            })
        }
    }

    /// A provider that always reports an aborted stream with partial text.
    struct AbortProvider;
    impl Provider for AbortProvider {
        fn complete(
            &self,
            _history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<Completion, ProviderError> {
            Ok(Completion {
                response: Response::Text("partial story".into()),
                prompt_tokens: None,
                aborted: true,
            })
        }
    }
}
