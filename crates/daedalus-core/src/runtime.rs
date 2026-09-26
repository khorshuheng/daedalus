//! Agent runtime: the stateful, UI-agnostic engine.
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
//! The turn loop is async and single-threaded, driven by a tokio
//! current-thread runtime owned by the worker thread. Adapters hold a cheap
//! cloneable handle and an event `Receiver`:
//!
//! ```text
//!   adapter ── Command (tokio unbounded channel) ──▶ AgentRuntime worker (async turn loop)
//!   adapter ◀───── Event (tokio unbounded channel) ── AgentRuntime
//! ```
//!
//! Commands sent while a turn is running are queued and delivered at the pi
//! boundaries: `steer` after the current assistant message finishes its tool
//! calls, `followUp` when the agent stops, `abort` immediately (partial text
//! is kept). Cancellation is per-session (a `CancellationToken` threaded into
//! providers and tools), so one client's abort never affects another session.
//!
//! # Wire format
//!
//! `Event` and `Command` are serde-tagged on `type` with `snake_case`
//! discriminators and snake_case fields (pi's RPC vocabulary), so the same
//! objects cross stdio RPC, the TUI and the WebSocket
//! server unchanged. The vocabulary:
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
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, EffortStyle, ProviderInfo};
use crate::instructions;
use crate::provider::{Message, Provider, Response, StreamDelta};
use crate::skills::{self, Skill};
use crate::tools::resolver::ToolSet;
use crate::workspace::Workspace;

/// Canonical thinking level, mapped per provider. Off disables
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
    /// discovery endpoint.
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

/// Whether a stored tool-result string represents success. Failures
/// are formatted as `tool error: ...`; a call cut off by the output token limit
/// records that it "was not executed". Shared so the runtime's `tool_end`
/// events and the TUI's resumed-transcript replay agree on the marker.
pub fn tool_result_ok(result: &str) -> bool {
    !result.starts_with("tool error:") && !result.contains("was not executed")
}

/// Hardcoded per-provider capability table: maps a canonical
/// effort level to the wire value the provider understands. No discovery.
pub fn provider_effort(provider: &ProviderInfo, effort: Effort) -> serde_json::Value {
    match provider.effort {
        EffortStyle::OpenaiEffort => match effort.openai_reasoning_effort() {
            Some(v) => serde_json::json!({ "reasoning_effort": v }),
            None => serde_json::json!({}),
        },
        EffortStyle::AnthropicThinking => match effort.anthropic_thinking_budget() {
            Some(budget) => serde_json::json!({
                "thinking": { "type": "enabled", "budget_tokens": budget }
            }),
            None => serde_json::json!({ "thinking": { "type": "disabled" } }),
        },
        EffortStyle::None => serde_json::json!({}),
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
        /// Raw model-supplied tool arguments, so a frontend can show what the
        /// call will do (e.g. the bash command). Optional and omitted on the
        /// wire when absent, so older event consumers keep working.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<Value>,
    },
    /// A tool call finished.
    ToolEnd {
        name: String,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// The tool result text, so a frontend can render the output under the
        /// call line. Optional and omitted on the wire when empty,
        /// so older event consumers keep working. Bounded by the
        /// tool's own `max_output` (32 KB by default), so the JSON/RPC frames
        /// stay small.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
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
        /// Active provider name; defaults for older clients.
        #[serde(default)]
        provider: String,
    },
    /// The agent settled: a final answer (or empty when cancelled).
    AgentSettled { text: String, interrupted: bool },
    /// The provider's available model ids, in reply to `list_models`.
    ModelsListed { models: Vec<String> },
    /// A non-fatal error surfaced by the runtime.
    Error {
        message: String,
        /// Machine-readable classification, when the runtime has one. Optional
        /// and omitted on the wire when `None`, so older event consumers keep
        /// working — and so a consumer never has to match on `message` prose.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kind: Option<ErrorKind>,
    },
}

/// What an [`Event::Error`] actually was, so a frontend can react (offer a
/// model picker, prompt for a key) without parsing the message text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum ErrorKind {
    /// The provider rejected the configured model. `supported` is the list the
    /// provider itself reported — not a hard-coded catalog, and not necessarily
    /// exhaustive, since a provider may also serve aliases it does not name.
    /// A frontend should offer these as candidates, not treat them as a gate.
    InvalidModel {
        requested: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        supported: Vec<String>,
    },
    /// Authentication failed for the active provider (no key, or a rejected
    /// one). The fix is a credential, not a retry.
    Auth,
    /// The iteration budget ran out before a final answer.
    IterationCap,
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
            Event::ModelsListed { .. } => "models_listed",
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
    /// Switch the active provider at runtime.
    SetProvider { provider: String },
    /// Change the thinking level.
    SetEffort { effort: Effort },
    /// Change the workspace (and re-seed the system prompt).
    SwitchWorkspace { path: String },
    /// Reset the conversation to a fresh context (system prompt only).
    Clear {},
    /// Ask the runtime to report its current state.
    GetState {},
    /// Discover the configured provider's available models.
    ListModels {},
    /// Reload the previous saved session for the workspace. Handled by the
    /// adapter (the runtime does not own the session store).
    Resume,
}

/// Internal queue item: a wire command or a worker shutdown request.
#[derive(Debug, Clone, PartialEq)]
enum Control {
    Command(CommandKind),
    Shutdown,
}

/// Terminal errors surfaced by the synchronous `run_once` path. The worker
/// path reports these as `Event::Error` instead.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RuntimeError {
    /// The model never produced a final answer within the iteration budget.
    #[error("iteration cap exceeded: no final answer after {0} iterations")]
    IterationCap(usize),
    /// A provider failure.
    #[error("provider error: {0}")]
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

/// A snapshot of the runtime's mutable state, reported via `get_state` and
/// carried in `state_changed` events.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RuntimeState {
    pub model: String,
    /// Active provider name.
    pub provider: String,
    pub effort: Effort,
    pub workspace: String,
    pub busy: bool,
    /// Token budget for the conversation history: the denominator a frontend
    /// uses for its context-usage percentage. `0` when unknown.
    #[serde(default)]
    pub max_context_tokens: usize,
    /// The active model's context window (input tokens), for a context-usage
    /// percentage. `0` when daedalus has no figure for the model, in which
    /// case frontends omit the percentage rather than invent a denominator.
    #[serde(default)]
    pub context_window: usize,
}

/// Classify a provider failure for `Event::Error.kind`, so frontends can react
/// to the failure rather than to its prose. `None` means "render the message":
/// a timeout or a malformed response has no special handling anywhere.
fn error_kind(e: &crate::provider::ProviderError) -> Option<ErrorKind> {
    match e {
        crate::provider::ProviderError::InvalidModel {
            requested,
            supported,
            ..
        } => Some(ErrorKind::InvalidModel {
            requested: requested.clone(),
            supported: supported.clone(),
        }),
        crate::provider::ProviderError::Auth(_) => Some(ErrorKind::Auth),
        _ => None,
    }
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
/// adapters can snapshot it (session save) while a turn runs.
struct Inner {
    config: Config,
    /// Config of the *live* provider. `switch_provider`/`SetModel` rewrite and
    /// rebuild from this, so a runtime model change actually reaches the wire:
    /// the provider bakes `model` into its backend at construction.
    provider_config: Mutex<Config>,
    /// The active provider; swappable at runtime.
    provider: Mutex<Arc<dyn Provider>>,
    /// Registry row of the active provider.
    provider_info: Mutex<&'static ProviderInfo>,
    tools: Arc<ToolSet>,
    /// Canonical workspace; `switch_workspace` replaces it (re-seeding the
    /// system prompt). Read by adapters (session keying).
    workspace: Mutex<Workspace>,
    /// Mutable runtime state (model/effort), guarded for adapter `get_state`.
    state: Mutex<RuntimeState>,
    /// Per-session cancel, threaded into providers and tools (replaces the
    /// process-global `term::cancel_flag`). Guarded so the worker
    /// can swap in a fresh token when a turn settles (abort is one-shot:
    /// `tokio_util::sync::CancellationToken` has no reset). Callers snapshot
    /// the current token with [`AgentRuntime::cancel_token`].
    cancel: Mutex<CancellationToken>,
    /// Interactive adapters (REPL/TUI) set this: a human is present and can
    /// abort with Ctrl+C, so the iteration cap is not enforced and long
    /// exploration (many tool steps before a final answer) is allowed.
    interactive: AtomicBool,
    busy: AtomicBool,
    history: Mutex<Vec<Message>>,
    anchor: Mutex<(usize, usize)>,
    /// Tool argument schemas for the provider `tools` field (built once).
    schemas: Vec<serde_json::Value>,
    /// Incoming commands; the worker awaits the receiver when idle and drains
    /// it at turn boundaries while busy. `Shutdown` stops the worker. Only the
    /// worker touches the receiver, so it sits behind a plain mutex to keep
    /// `Inner` `Sync` (the lock is never held across an `await`).
    commands_tx: tokio::sync::mpsc::UnboundedSender<Control>,
    commands_rx: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Control>>,
    events: tokio::sync::mpsc::UnboundedSender<Event>,
    /// Cached model ids discovered for the current provider.
    models: Mutex<Vec<String>>,
    /// Where user instructions (`APPEND_SYSTEM.md`) are read from. Defaults to
    /// the standard config dir; a seam so tests do not depend on the ambient
    /// `~/.config`.
    instructions_path: PathBuf,
}

/// A cloneable handle to a running agent. Construct with `AgentRuntime::new`
/// to get the handle plus its event `Receiver`; drive it with `run_forever`
/// on a thread the adapter chooses (the REPL spawns one; the server gives each
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
    ) -> (AgentRuntime, tokio::sync::mpsc::UnboundedReceiver<Event>) {
        Self::with_instructions_path(
            config,
            provider,
            tools,
            workspace,
            instructions::user_file(),
        )
    }

    /// Like [`new`](Self::new), but reads user instructions from `path`. The
    /// path is otherwise fixed at `~/.config/daedalus/APPEND_SYSTEM.md`; tests
    /// use this to keep the seed prompt independent of the ambient config dir.
    fn with_instructions_path(
        config: Config,
        provider: Box<dyn Provider>,
        tools: ToolSet,
        workspace: Workspace,
        instructions_path: PathBuf,
    ) -> (AgentRuntime, tokio::sync::mpsc::UnboundedReceiver<Event>) {
        let (events, rx) = tokio::sync::mpsc::unbounded_channel();
        let (commands_tx, commands_rx) = tokio::sync::mpsc::unbounded_channel();
        let schemas = tools.tool_schemas();
        let tools = Arc::new(tools);
        let model = config.model.clone();
        let provider_name = config.provider.name.to_string();
        let provider_info = config.provider;
        let provider_config = config.clone();
        let effort = config.effort;
        let max_context_tokens = config.max_context_tokens;
        let context_window = crate::catalog::context_window(&provider_name, &model).unwrap_or(0);
        let ws_path = workspace.root().to_string_lossy().into_owned();
        let runtime = AgentRuntime {
            inner: Arc::new(Inner {
                config,
                provider_config: Mutex::new(provider_config),
                provider: Mutex::new(Arc::from(provider)),
                provider_info: Mutex::new(provider_info),
                tools,
                workspace: Mutex::new(workspace),
                state: Mutex::new(RuntimeState {
                    model,
                    provider: provider_name,
                    effort,
                    workspace: ws_path,
                    busy: false,
                    max_context_tokens,
                    context_window,
                }),
                cancel: Mutex::new(CancellationToken::new()),
                interactive: AtomicBool::new(false),
                busy: AtomicBool::new(false),
                history: Mutex::new(Vec::new()),
                anchor: Mutex::new((0, 0)),
                schemas,
                commands_tx,
                commands_rx: tokio::sync::Mutex::new(commands_rx),
                events,
                models: Mutex::new(Vec::new()),
                instructions_path,
            }),
        };
        (runtime, rx)
    }

    // --- command surface (thread-safe, non-blocking) ---

    fn push(&self, kind: CommandKind) {
        let _ = self.inner.commands_tx.send(Control::Command(kind));
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

    /// Mark this runtime as interactive (REPL/TUI). Interactive runs skip the
    /// iteration cap: the human is the backstop and can abort with Ctrl+C, so
    /// legitimate long explorations are not cut off at `max_iterations`.
    /// Headless adapters (json/rpc) leave this off and keep the cap.
    pub fn set_interactive(&self, on: bool) {
        self.inner.interactive.store(on, Ordering::SeqCst);
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
        self.cancel_current();
        self.push(CommandKind::Abort {});
    }

    /// Snapshot the current cancellation token (providers and tools hold
    /// their clone for the duration of one call).
    pub fn cancel_token(&self) -> CancellationToken {
        self.inner.cancel.lock().unwrap().clone()
    }

    /// Cancel the current token (worker-side drain of an `Abort`).
    fn cancel_current(&self) {
        self.inner.cancel.lock().unwrap().cancel();
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
    /// handlers that run while the worker is idle; adapters driving through
    /// the queue use `clear()`.
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
        let _ = self.inner.commands_tx.send(Control::Shutdown);
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

    /// The provider backing this runtime.
    pub fn provider(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.inner.provider.lock().unwrap())
    }

    /// The provider kind this runtime was built with (for /login and the
    /// model picker).
    pub fn provider_kind(&self) -> &'static ProviderInfo {
        *self.inner.provider_info.lock().unwrap()
    }

    /// Cached model ids discovered for the current provider.
    pub fn models(&self) -> Vec<String> {
        self.inner.models.lock().unwrap().clone()
    }

    /// Ask the worker to refresh the provider's model list; the result arrives
    /// as `Event::ModelsListed` (or `Event::Error`).
    pub fn refresh_models(&self) {
        let _ = self
            .inner
            .commands_tx
            .send(Control::Command(CommandKind::ListModels {}));
    }

    /// Switch the active provider at runtime. The worker rebuilds
    /// the provider and emits `StateChanged` (or `Error` when it refuses).
    pub fn set_provider(&self, provider: &str) {
        let _ = self
            .inner
            .commands_tx
            .send(Control::Command(CommandKind::SetProvider {
                provider: provider.to_string(),
            }));
    }

    /// Rebuild the active provider for `name`: resolve the registry row and
    /// key, refuse a keyless hosted provider, swap the provider (keeping the
    /// conversation), then refresh the model list.
    async fn switch_provider(&self, name: &str) {
        let info = match crate::config::provider_by_name(name) {
            Ok(i) => i,
            Err(e) => {
                self.emit(Event::Error {
                    message: e,
                    kind: None,
                });
                return;
            }
        };
        let key = crate::credential::resolve_api_key(info, None);
        if info.requires_key() && key.as_deref().unwrap_or("").is_empty() {
            let env = info.api_key_env.unwrap_or("<PROVIDER>_API_KEY");
            self.emit(Event::Error {
                message: format!(
                    "no API key for provider '{}': set {env} or pass --api-key; note /login stores for the current provider",
                    info.name
                ),
                kind: Some(ErrorKind::Auth),
            });
            return;
        }
        let mut new_config = self.inner.provider_config.lock().unwrap().clone();
        new_config.provider = info;
        new_config.base_url = info.preset_base_url.to_string();
        new_config.api_key = key;
        new_config.model = self.inner.state.lock().unwrap().model.clone();
        let provider = crate::provider::from_config(&new_config);
        *self.inner.provider.lock().unwrap() = Arc::from(provider);
        *self.inner.provider_info.lock().unwrap() = info;
        *self.inner.provider_config.lock().unwrap() = new_config;
        {
            let mut st = self.inner.state.lock().unwrap();
            st.provider = info.name.to_string();
            // A different provider can change whether the window is known.
            st.context_window =
                crate::catalog::context_window(&st.provider, &st.model).unwrap_or(0);
        }
        // A different provider has a different model catalog.
        *self.inner.models.lock().unwrap() = Vec::new();
        self.emit_state_changed();
        let provider = Arc::clone(&self.inner.provider.lock().unwrap());
        match provider.list_models().await {
            Ok(models) => {
                *self.inner.models.lock().unwrap() = models.clone();
                self.emit(Event::ModelsListed { models });
            }
            Err(e) => self.emit(Event::Error {
                message: format!("could not list models: {e}"),
                kind: None,
            }),
        }
    }

    /// Skills discovered for the current workspace (user + workspace levels,
    /// workspace wins on a name clash). Re-read on every call so a
    /// file dropped into `<workspace>/.daedalus/skills/` is picked up on the next
    /// turn. Workspace skills are resolved through the sandbox, so a symlinked
    /// skill cannot smuggle content from outside the workspace.
    pub fn skills(&self) -> Vec<Skill> {
        let workspace = self.inner.workspace.lock().unwrap().clone();
        skills::discover(&workspace)
    }

    /// `(name, description)` for every registered tool — built-ins plus any
    /// MCP tools — for the TUI `/tools` listing.
    pub fn tool_listing(&self) -> Vec<(String, String)> {
        self.inner.tools.listing()
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
    /// thread; adapters spawn this on a thread of their choosing (the TUI
    /// spawns one, the server gives each connected session its own). The thread
    /// owns a tokio current-thread runtime that drives the async turn engine.
    pub fn run_forever(&self) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("worker tokio runtime");
        rt.block_on(self.run_forever_async());
    }

    /// The async body of the worker loop.
    async fn run_forever_async(&self) {
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
        while let Some(kind) = self.wait_for_command().await {
            match kind {
                CommandKind::Prompt { text } => self.drive_until_settled(&text).await,
                // While idle a steer/follow-up is simply a new turn.
                CommandKind::Steer { text } | CommandKind::FollowUp { text } => {
                    self.drive_until_settled(&text).await
                }
                CommandKind::Abort {} => self.cancel_clear(),
                CommandKind::GetState {} => self.emit_state_changed(),
                CommandKind::Clear {} => {
                    self.reset_to_seed();
                    self.emit_state_changed();
                }
                CommandKind::Resume => {}
                CommandKind::ListModels {} => {
                    let provider = Arc::clone(&self.inner.provider.lock().unwrap());
                    match provider.list_models().await {
                        Ok(models) => {
                            *self.inner.models.lock().unwrap() = models.clone();
                            self.emit(Event::ModelsListed { models });
                        }
                        Err(e) => self.emit(Event::Error {
                            message: format!("could not list models: {e}"),
                            kind: None,
                        }),
                    }
                }
                CommandKind::SetProvider { provider } => self.switch_provider(&provider).await,
                CommandKind::SetModel { .. }
                | CommandKind::SetEffort { .. }
                | CommandKind::SwitchWorkspace { .. } => self.apply_state_command(kind).await,
            }
        }
    }

    /// Run a single prompt to completion on this thread (no worker), returning
    /// the final answer. Used by the piped/one-shot CLI path and integration
    /// tests. Errors (iteration cap, provider failure) are returned as
    /// `Err(RuntimeError)`.
    pub fn run_once(&self, prompt: &str) -> Result<String, RuntimeError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("run_once tokio runtime");
        rt.block_on(self.run_once_async(prompt))
    }

    /// The async body of `run_once`.
    async fn run_once_async(&self, prompt: &str) -> Result<String, RuntimeError> {
        self.set_busy(true);
        self.cancel_clear();
        let mut saved = VecDeque::new();
        let (text, _interrupted, error) = self.run_one_user_message(prompt, &mut saved).await;
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
        // Re-read the workspace from the canonical source: `switch_workspace`
        // replaces it without going through `set_busy`, so a cached
        // `state.workspace` would report the previous directory.
        let workspace = self.workspace_path().to_string_lossy().into_owned();
        let mut guard = self.inner.state.lock().unwrap();
        guard.workspace = workspace.clone();
        let st = guard.clone();
        drop(guard);
        self.emit(Event::StateChanged {
            model: st.model,
            effort: st.effort,
            workspace,
            provider: st.provider,
        });
    }

    /// The seed system prompt for the current workspace: the built-in prompt,
    /// then any user instructions (`APPEND_SYSTEM.md`), then the skills catalog.
    fn system_prompt(&self) -> String {
        let base = format!(
            "You are daedalus, a minimal coding agent. You inspect and modify files in the workspace '{}' by calling tools.\n\
             You have exactly five tools and no others: read, search, bash, edit, write.\n\
             - read: read one or more files (a path, a list of paths, or a directory + glob) with an optional line range; use offset to page through long files.\n\
             - search: search file contents with a regex; respects .gitignore, skips hidden and binary files, and is bounded. Use this for all content searches.\n\
             - bash: run a shell command in the workspace; check results before trusting them.\n\
             - edit: apply precise text replacements; each oldText must match exactly once.\n\
             - write: create or overwrite a file.\n\
             Rules:\n\
             - Paths are relative to the workspace by default; absolute paths and `..` are allowed.\n\
             - Read before editing; verify changes with bash.\n\
             - For content searches always use the `search` tool; never call `grep`/`rg`/`find` through bash.\n\
             - Prefer the `read` tool over `cat`/`head`; use bash for commands, not for dumping files.\n\
             - Make the smallest change that satisfies the request.\n\
             - When finished, give a concise final answer.",
            self.workspace_path().display()
        );
        let base = instructions::append(
            &base,
            instructions::read_from(&self.inner.instructions_path).as_deref(),
        );
        skills::with_catalog(&base, &self.skills())
    }

    /// Reset the conversation to a fresh seed: system prompt only, ranked
    /// against nothing yet (the first user message re-seeds via the turn
    /// engine).
    fn reset_to_seed(&self) {
        let seed = self.system_prompt();
        let mut h = self.inner.history.lock().unwrap();
        h.clear();
        h.push(Message::System(seed));
        *self.inner.anchor.lock().unwrap() = (0, 0);
    }

    fn cancel_clear(&self) {
        *self.inner.cancel.lock().unwrap() = CancellationToken::new();
    }

    /// Apply a set_model/set_effort/switch_workspace command to shared state.
    async fn apply_state_command(&self, kind: CommandKind) {
        match kind {
            CommandKind::SetModel { model } => {
                {
                    let mut st = self.inner.state.lock().unwrap();
                    st.model = model.clone();
                    // The window is a property of the model, so switching the
                    // model re-derives it; an unknown model clears it.
                    st.context_window =
                        crate::catalog::context_window(&st.provider, &model).unwrap_or(0);
                }
                // The provider captures the model id at construction, so a
                // runtime switch must rebuild it; otherwise the request (and
                // the error that names it) keeps reporting the old model.
                let mut cfg = self.inner.provider_config.lock().unwrap().clone();
                cfg.model = model;
                let provider = crate::provider::from_config(&cfg);
                *self.inner.provider.lock().unwrap() = Arc::from(provider);
                *self.inner.provider_config.lock().unwrap() = cfg;
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
                    kind: None,
                }),
            },
            _ => {}
        }
    }

    /// Pop the oldest queued command without blocking, or `None` (also when
    /// the next item is a shutdown request).
    async fn pop_queued(&self) -> Option<CommandKind> {
        let mut rx = self.inner.commands_rx.lock().await;
        match rx.try_recv() {
            Ok(Control::Command(kind)) => Some(kind),
            Ok(Control::Shutdown) | Err(_) => None,
        }
    }

    /// Block until the queue has a command, then pop it. Returns `None` on a
    /// shutdown request so the worker loop can exit.
    async fn wait_for_command(&self) -> Option<CommandKind> {
        let mut rx = self.inner.commands_rx.lock().await;
        match rx.recv().await {
            Some(Control::Command(kind)) => Some(kind),
            Some(Control::Shutdown) | None => None,
        }
    }

    /// Drain all currently queued commands and classify them into the pending
    /// steer / follow-up messages plus immediate actions (pi semantics):
    /// - `Steer` is delivered after the current assistant message finishes
    ///   its tool calls;
    /// - `FollowUp` is delivered when the agent stops;
    /// - `Abort` cancels immediately;
    /// - set_*/get_state/clear apply immediately where safe.
    async fn drain_queue(&self) -> (Vec<String>, Vec<String>) {
        let mut steers = Vec::new();
        let mut follow_ups = Vec::new();
        while let Some(kind) = self.pop_queued().await {
            match kind {
                CommandKind::Steer { text } => steers.push(text),
                CommandKind::FollowUp { text } => follow_ups.push(text),
                CommandKind::Prompt { text } => follow_ups.push(text), // busy prompt = follow-up
                CommandKind::Abort {} => {
                    self.cancel_current();
                }
                CommandKind::GetState {} => self.emit_state_changed(),
                CommandKind::Clear {} => {
                    self.reset_to_seed();
                    self.emit_state_changed();
                }
                CommandKind::Resume => {}
                CommandKind::ListModels {} => {
                    let provider = Arc::clone(&self.inner.provider.lock().unwrap());
                    match provider.list_models().await {
                        Ok(models) => {
                            *self.inner.models.lock().unwrap() = models.clone();
                            self.emit(Event::ModelsListed { models });
                        }
                        Err(e) => self.emit(Event::Error {
                            message: format!("could not list models: {e}"),
                            kind: None,
                        }),
                    }
                }
                CommandKind::SetProvider { provider } => self.switch_provider(&provider).await,
                CommandKind::SetModel { .. }
                | CommandKind::SetEffort { .. }
                | CommandKind::SwitchWorkspace { .. } => self.apply_state_command(kind).await,
            }
        }
        (steers, follow_ups)
    }

    /// Append `text` as the next user message, replacing the seed system
    /// prompt with one ranked against it when memory is enabled.
    fn push_user(&self, text: &str) {
        let seed = self.system_prompt();
        let mut h = self.inner.history.lock().unwrap();
        // Refresh the seed system prompt (history[0]) for this task.
        if h.is_empty() {
            h.push(Message::System(seed));
        } else if let Some(Message::System(first)) = h.first_mut() {
            *first = seed;
        }
        h.push(Message::User(text.to_string()));
    }

    /// Run the whole loop for one user message (the initial prompt of a turn)
    /// until the model gives a final answer, is cancelled, or errors. Tool
    /// results are fed back verbatim; text deltas and tool lifecycle emit
    /// events. Runs entirely on the worker thread.
    ///
    /// Keep `history` within `budget` tokens. Prefers **compaction**: the
    /// oldest removable turn blocks are summarized via a
    /// provider call and replaced with a compact System summary, so the agent
    /// keeps a compressed memory instead of silently losing old turns. When
    /// compaction is unavailable (e.g. the provider is fake) or fails, falls
    /// back to dropping the oldest blocks (trim), which always keeps the
    /// session under budget.
    async fn manage_context(
        &self,
        history: &mut Vec<Message>,
        seed_len: usize,
        budget: usize,
        anchor_tokens: &mut usize,
        anchor_len: &mut usize,
    ) {
        // Try compaction first when over budget.
        if anchored_total(history, *anchor_tokens, *anchor_len) > budget {
            self.compact_history(history, seed_len, budget).await;
        }
        // Whatever remains over budget is trimmed (compaction is best-effort:
        // a summary may itself be long, or the provider may be unavailable).
        trim_history(history, seed_len, budget, anchor_tokens, anchor_len);
    }

    /// Compact the oldest removable turns (past the seed) into a single
    /// summary System message, repeated until the history fits `budget` or
    /// nothing more can be removed. Best-effort: returns without changing
    /// anything when there is nothing summarizable or the provider call
    /// fails (the caller then trims).
    async fn compact_history(&self, history: &mut Vec<Message>, seed_len: usize, budget: usize) {
        // Find the oldest removable turn block (an Assistant message and its
        // following ToolResults). Compaction summarizes from the seed onward.
        loop {
            // Everything from the seed up to (and including) the oldest
            // assistant block + its tool results is compacted into a summary.
            let Some(first_removable) = history
                .iter()
                .enumerate()
                .skip(seed_len)
                .find(|(_, m)| matches!(m, Message::Assistant { .. }))
                .map(|(i, _)| i)
            else {
                return; // no assistant turns yet (nothing worth compacting)
            };
            let mut block_end = first_removable + 1;
            while block_end < history.len()
                && matches!(history[block_end], Message::ToolResult { .. })
            {
                block_end += 1;
            }
            // Compact from the first non-system message (index 1, past the
            // system prompt) through the oldest assistant block, so the User
            // that prompted the assistant is summarized too.
            let compact_from = 1usize.max(seed_len.saturating_sub(1));
            let compact_range = compact_from..block_end;
            let to_compact: Vec<Message> = history[compact_range.clone()].to_vec();
            if to_compact.len() <= 1 {
                return; // nothing meaningful to compress
            }
            let summary = match self.request_summary(&to_compact).await {
                Some(s) if !s.is_empty() => s,
                _ => return, // compaction unavailable/failed; caller trims
            };
            // Replace the compacted range with a summary System message.
            history.drain(compact_range);
            history.insert(
                compact_from,
                Message::System(format!("Summary of earlier conversation: {summary}")),
            );
            if anchored_total(history, 0, 0) <= budget {
                return;
            }
        }
    }

    /// Ask the provider to compress `messages` into a short summary. Returns
    /// `None` when the provider cannot (fake provider has no scripted
    /// response) or the call fails. The request is fire-and-forget: no events
    /// are emitted for it.
    async fn request_summary(&self, messages: &[Message]) -> Option<String> {
        let prompt = format!(
            "Compress the following conversation into a concise summary that preserves \
             the key instructions, decisions, and constraints, so a follow-up turn \
             can continue without the full transcript. Keep it under 200 words.\n\n{}",
            messages
                .iter()
                .map(|m| match m {
                    Message::User(u) => format!("user: {u}"),
                    Message::Assistant { text, .. } => {
                        format!("assistant: {}", text.as_deref().unwrap_or("(tool call)"))
                    }
                    Message::ToolResult { result, .. } => format!("result: {result}"),
                    Message::System(s) => format!("system: {s}"),
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
        let history = vec![Message::User(prompt)];
        static EMPTY: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        let empty = EMPTY.get_or_init(|| serde_json::Value::Object(Default::default()));
        let mut on_delta = |_: StreamDelta| {}; // do not stream a compaction into the UI
        let provider = Arc::clone(&self.inner.provider.lock().unwrap());
        match provider
            .complete(&history, &[], empty, self.cancel_token(), &mut on_delta)
            .await
        {
            Ok(completion) => match completion.response {
                Response::Text(t) => Some(t),
                _ => None,
            },
            Err(_) => None,
        }
    }

    /// Returns the terminal text, whether the turn was interrupted, and an
    /// error description when the turn ended on an error (iteration cap or
    /// provider failure — also emitted as `Event::Error`).
    async fn run_one_user_message(
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
            let (steers, followups) = self.drain_queue().await;
            saved.extend(followups);
            if !steers.is_empty() {
                // A steer arrived after the assistant's tool phase: append it
                // as a user message and keep looping (no settle).
                for steer in steers {
                    self.push_user(&steer);
                    iterations = 0; // fresh turn budget for the steer
                }
            }

            if self.inner.cancel.lock().unwrap().is_cancelled() {
                interrupted = true;
                break 'steps;
            }
            if iterations >= self.inner.config.max_iterations
                && !self.inner.interactive.load(Ordering::SeqCst)
            {
                let msg = format!(
                    "iteration cap exceeded: no final answer after {} iterations",
                    self.inner.config.max_iterations
                );
                self.emit(Event::Error {
                    message: msg.clone(),
                    kind: Some(ErrorKind::IterationCap),
                });
                error = Some(msg);
                interrupted = true;
                break 'steps;
            }

            let (mut h, (mut a0, mut a1)) = {
                let h = self.inner.history.lock().unwrap().clone();
                let anchor = *self.inner.anchor.lock().unwrap();
                (h, anchor)
            };
            self.manage_context(
                &mut h,
                seed_len,
                self.inner.config.max_context_tokens,
                &mut a0,
                &mut a1,
            )
            .await;
            {
                // Write the compacted/trimmed history back wholesale.
                let mut hh = self.inner.history.lock().unwrap();
                *hh = h;
                *self.inner.anchor.lock().unwrap() = (a0, a1);
            }

            let completion = {
                // Clone the history so the lock is not held across the await
                // (the vector is small and this is the only mutation window).
                let h = self.inner.history.lock().unwrap().clone();
                let effort_params = {
                    let st = self.inner.state.lock().unwrap();
                    provider_effort(*self.inner.provider_info.lock().unwrap(), st.effort)
                };
                let cancel = self.cancel_token();
                let emit = self.inner.events.clone();
                let mut on_delta = |delta: StreamDelta| match delta {
                    StreamDelta::Text(text) => {
                        let _ = emit.send(Event::TextDelta { text });
                    }
                    StreamDelta::Thinking(text) => {
                        let _ = emit.send(Event::ThinkingDelta { text });
                    }
                };
                let provider = Arc::clone(&self.inner.provider.lock().unwrap());
                provider
                    .complete(
                        &h,
                        &self.inner.schemas,
                        &effort_params,
                        cancel,
                        &mut on_delta,
                    )
                    .await
            };
            match completion {
                Err(e) => {
                    // Auth failures carry the provider name and remediation, so a
                    // revoked or incorrect key is actionable rather than a bare
                    // provider status.
                    let message = match &e {
                        crate::provider::ProviderError::Auth(_) => {
                            let info = *self.inner.provider_info.lock().unwrap();
                            format!(
                                "authentication failed for provider '{}': check the API key ({}, --api-key, or /login) — {e}",
                                info.name,
                                info.api_key_env.unwrap_or("<PROVIDER>_API_KEY"),
                            )
                        }
                        _ => format!("provider error: {e}"),
                    };
                    self.emit(Event::Error {
                        message: message.clone(),
                        kind: error_kind(&e),
                    });
                    // The turn ended without an answer: report it, so one-shot
                    // callers (`run_once`) and the adapters do not treat this as
                    // a completed turn.
                    error = Some(message);
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
                            self.inner.history.lock().unwrap().push(Message::Assistant {
                                text: None,
                                tool_calls: calls.clone(),
                            });
                            // Run tool calls in parallel, like
                            // pi. Each thread locks the workspace and checks
                            // the shared cancel flag; results are collected in
                            // call order so history stays deterministic.
                            let mut cancelled = false;
                            let ws = self.inner.workspace.lock().unwrap().clone();
                            let results: Vec<(String, Result<String, crate::tools::ToolError>)> =
                                futures::future::join_all(calls.iter().map(|call| {
                                    self.emit(Event::ToolStart {
                                        name: call.name.clone(),
                                        id: Some(call.id.clone()),
                                        args: Some(call.args.clone()),
                                    });
                                    let tools = Arc::clone(&self.inner.tools);
                                    let ws = ws.clone();
                                    let cancel = self.cancel_token();
                                    let name = call.name.clone();
                                    let args = call.args.clone();
                                    let id = call.id.clone();
                                    async move {
                                        let result = tools
                                            .execute(&ws, &name, &args, cancel)
                                            .await
                                            .map(|out| out.content);
                                        (id, result)
                                    }
                                }))
                                .await;
                            for (call, (_id, result)) in calls.iter().zip(&results) {
                                let result_str = match result {
                                    Ok(content) => content.clone(),
                                    Err(e) => {
                                        cancelled |=
                                            matches!(e, crate::tools::ToolError::Cancelled);
                                        format!("tool error: {e}")
                                    }
                                };
                                self.emit(Event::ToolEnd {
                                    name: call.name.clone(),
                                    ok: tool_result_ok(&result_str),
                                    error: result_str
                                        .strip_prefix("tool error:")
                                        .map(|s| s.trim().to_string()),
                                    output: (!result_str.is_empty()).then(|| result_str.clone()),
                                });
                                let mut h = self.inner.history.lock().unwrap();
                                h.push(Message::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    result: result_str,
                                });
                            }
                            if cancelled {
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
    async fn drive_until_settled(&self, first: &str) {
        self.set_busy(true);
        self.cancel_clear();
        let mut saved = VecDeque::new();
        // First message runs immediately.
        let (mut final_text, mut interrupted, _) =
            self.run_one_user_message(first, &mut saved).await;
        // Follow-ups queued while busy are delivered when the agent stops.
        loop {
            if interrupted {
                // Cancelled mid-turn: stop delivering queued messages.
                break;
            }
            let (steers, followups) = self.drain_queue().await;
            saved.extend(followups);
            saved.extend(steers);
            let Some(next) = saved.pop_front() else { break };
            let (text, was_interrupted, _) = self.run_one_user_message(&next, &mut saved).await;
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
    use crate::provider::fake::FakeProvider;
    use crate::provider::{Completion, ProviderError, ToolCall};
    use std::sync::mpsc;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

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
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async move {
                let mut q = self.completions.lock().unwrap();
                Ok(q.pop_front().unwrap_or(Completion {
                    response: Response::Text("done".into()),
                    prompt_tokens: None,
                    aborted: false,
                }))
            })
        }
    }

    fn workspace(_name: &str) -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let root = dir.path().to_path_buf();
        let ws = Workspace::new(root).unwrap();
        (dir, ws)
    }

    /// A skill in `<workspace>/.daedalus/skills/` shows up in the
    /// system-prompt catalog. The user-level dir is not injected here, so a
    /// real `~/.config/daedalus/skills` can only add entries, never remove the
    /// workspace one this asserts on.
    #[test]
    fn system_prompt_lists_workspace_skills() {
        let (dir, ws) = workspace("skills");
        let skills_dir = dir.path().join(".daedalus").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("demo.md"),
            "Demo skill.\n\nDo the demo thing.",
        )
        .unwrap();
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(GateProvider::new(vec![])),
            ToolSet::new(1000),
            ws,
        );
        let prompt = rt.system_prompt();
        assert!(prompt.contains("Available skills"), "catalog block missing");
        assert!(prompt.contains("demo — Demo skill."));
        assert!(prompt.contains("demo.md"));
        // The model can self-serve with the existing read tool rather than a
        // new skill tool. Assert on the specific skill, not the count: a real
        // `~/.config/daedalus/skills` may contribute extra user-level entries.
        let skills = rt.skills();
        assert!(skills
            .iter()
            .any(|s| s.name == "demo" && s.prompt().contains("Do the demo thing.")));
    }

    /// A user-level `APPEND_SYSTEM.md` is appended to the seed prompt, after the
    /// built-in text and before the skills catalog.
    #[test]
    fn system_prompt_appends_user_instructions() {
        let (dir, ws) = workspace("instructions");
        let config_home = tempfile::tempdir().unwrap();
        let instructions_path = config_home.path().join("APPEND_SYSTEM.md");
        std::fs::write(
            &instructions_path,
            "Always run `cargo fmt` before committing.\n",
        )
        .unwrap();
        // A skill too, so the ordering (instructions before catalog) is checked.
        let skills_dir = dir.path().join(".daedalus").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(skills_dir.join("demo.md"), "Demo skill.\n\nbody").unwrap();

        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::with_instructions_path(
            cfg,
            Box::new(GateProvider::new(vec![])),
            ToolSet::new(1000),
            ws,
            instructions_path,
        );
        let prompt = rt.system_prompt();

        assert!(
            prompt.contains("Additional instructions from the user:"),
            "lead-in missing: {prompt}"
        );
        assert!(prompt.contains("Always run `cargo fmt` before committing."));
        let instructions_at = prompt.find("Additional instructions").unwrap();
        let catalog_at = prompt.find("Available skills").unwrap();
        assert!(
            instructions_at < catalog_at,
            "user instructions must precede the skills catalog"
        );
    }

    /// No instructions file (the common case) leaves the seed prompt unchanged:
    /// no lead-in, same built-in text as before the feature.
    #[test]
    fn system_prompt_without_user_instructions_is_unmarked() {
        let (dir, ws) = workspace("no-instructions");
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::with_instructions_path(
            cfg,
            Box::new(GateProvider::new(vec![])),
            ToolSet::new(1000),
            ws,
            dir.path().join("missing.md"),
        );
        assert!(!rt.system_prompt().contains("Additional instructions"));
    }

    fn runtime_with(
        name: &str,
        provider: Box<dyn Provider>,
    ) -> (
        AgentRuntime,
        tokio::sync::mpsc::UnboundedReceiver<Event>,
        std::thread::JoinHandle<()>,
        Workspace,
    ) {
        let (dir, ws) = workspace(name);
        let root = dir.path().to_path_buf();
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: root.clone(),
            ..Config::defaults(root)
        };
        let (rt, rx) = AgentRuntime::new(cfg, provider, tools, ws.clone());
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        (rt, rx, handle, ws)
    }

    /// Collect events until `agent_settled` (or a short timeout) and return
    /// them plus whether the run settled.
    fn collect_until_settled(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
    ) -> (Vec<Event>, bool) {
        let mut events = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match rx.try_recv() {
                Ok(e) => {
                    let settled = matches!(e, Event::AgentSettled { .. });
                    events.push(e);
                    if settled {
                        return (events, true);
                    }
                }
                Err(_) if std::time::Instant::now() > deadline => return (events, false),
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
    }

    fn text(text: &str) -> String {
        text.to_string()
    }

    #[test]
    fn iteration_cap_emits_error_and_settles() {
        // Two tool-call completions then no final text; max_iterations = 2.
        let (dir, ws) = workspace("cap");
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
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, provider, tools, ws.clone());
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("keep going");
        let (events, settled) = collect_until_settled(&mut rx);
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

    /// A provider that always fails the way a live provider rejects a model.
    struct FailProvider;

    impl Provider for FailProvider {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async move {
                Err(ProviderError::InvalidModel {
                    requested: "totally-bogus-xyz-42".into(),
                    supported: vec!["deepseek-flash".into()],
                    detail: "The model 'totally-bogus-xyz-42' does not exist".into(),
                })
            })
        }
    }

    /// A rejected model reaches the frontend as a typed `ErrorKind`, so an
    /// adapter can offer the provider's list instead of parsing the message.
    #[test]
    fn provider_failure_carries_a_machine_readable_kind() {
        let (dir, ws) = workspace("provider-kind");
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, Box::new(FailProvider), ToolSet::new(1000), ws);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("hello");
        let (events, settled) = collect_until_settled(&mut rx);
        assert!(settled, "a failed turn still settles");
        let kind = events.iter().find_map(|e| match e {
            Event::Error { kind, .. } => Some(kind.clone()),
            _ => None,
        });
        assert_eq!(
            kind,
            Some(Some(ErrorKind::InvalidModel {
                requested: "totally-bogus-xyz-42".into(),
                supported: vec!["deepseek-flash".into()],
            })),
            "events: {events:?}"
        );
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    /// The one-shot path reports a provider failure as a failure rather than
    /// returning the (empty) text of a turn that never produced an answer.
    #[test]
    fn provider_failure_makes_run_once_fail() {
        let (dir, ws) = workspace("provider-error-once");
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(cfg, Box::new(FailProvider), ToolSet::new(1000), ws);
        let err = rt.run_once("hello").unwrap_err();
        assert!(matches!(err, RuntimeError::Provider(_)), "got {err:?}");
    }

    #[test]
    fn interactive_mode_is_not_capped_by_max_iterations() {
        // Interactive adapters (REPL/TUI) opt out of the iteration cap: a
        // human is present and can abort with Ctrl+C, so legitimate long
        // exploration (many reads/checks before a final answer) must not be
        // killed at the configured cap. Same provider as the cap test but
        // with interactive=true: all three tool calls run, then the provider
        // falls back to a final text and the turn settles cleanly.
        let (dir, ws) = workspace("interactive");
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
            max_iterations: 2, // deliberately below the 3 tool calls
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, provider, tools, ws.clone());
        rt.set_interactive(true);
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("keep going");
        let (events, settled) = collect_until_settled(&mut rx);
        assert!(settled);
        assert!(
            !events.iter().any(|e| matches!(e, Event::Error { .. })),
            "interactive mode must not emit the iteration-cap error"
        );
        let settled_event = events.iter().find_map(|e| match e {
            Event::AgentSettled {
                text, interrupted, ..
            } => Some((text.clone(), *interrupted)),
            _ => None,
        });
        assert_eq!(
            settled_event,
            Some(("done".to_string(), false)),
            "all tool calls run and the turn settles with the final text"
        );
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn clear_resets_conversation_and_set_effort_emits_state() {
        let (rt, mut rx, handle, _ws) = runtime_with(
            "clear",
            Box::new(GateProvider::new(vec![
                Response::Text(text("a")),
                Response::Text(text("b")),
            ])),
        );
        rt.prompt("first");
        collect_until_settled(&mut rx);
        rt.clear();
        rt.set_effort(Effort::High);
        rt.prompt("second");
        let (events, settled) = collect_until_settled(&mut rx);
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
    fn set_model_rebuilds_the_provider_so_the_new_id_reaches_the_wire() {
        // The rig provider bakes the model id into its backend at construction,
        // so a state-only `SetModel` would keep sending the old model (and a
        // rejected one would be re-sent forever). The provider must be swapped.
        let (rt, mut rx, handle, _ws) = runtime_with(
            "set-model",
            Box::new(GateProvider::new(vec![Response::Text(text("ok"))])),
        );
        let before = rt.provider();
        rt.set_model("deepseek-flash");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let mut applied = false;
        while !applied && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(Event::StateChanged { model, .. }) if model == "deepseek-flash" => {
                    applied = true
                }
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        assert!(applied, "set_model must ack with state_changed");
        assert_eq!(rt.state().model, "deepseek-flash");
        let after = rt.provider();
        assert!(
            !Arc::ptr_eq(&before, &after),
            "the provider must be rebuilt so the new model id is what gets sent"
        );
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn state_tracks_the_active_models_context_window() {
        // The footer percentage needs a real denominator: the model's window
        // when daedalus knows it, and no denominator at all otherwise.
        let (dir, ws) = workspace("context-window");
        let mut cfg = Config::defaults(dir.path().to_path_buf());
        cfg.provider = crate::config::provider_by_name("deepseek").unwrap();
        cfg.model = "deepseek-chat".into();
        let tools = ToolSet::new(1000);
        let (rt, mut rx) = AgentRuntime::new(
            cfg,
            Box::new(GateProvider::new(vec![Response::Text(text("ok"))])),
            tools,
            ws,
        );
        assert_eq!(
            rt.state().context_window,
            64_000,
            "a known model reports its window"
        );

        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.set_model("deepseek-unlisted");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let mut applied = false;
        while !applied && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(Event::StateChanged { model, .. }) if model == "deepseek-unlisted" => {
                    applied = true
                }
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        assert!(applied, "set_model must ack with state_changed");
        assert_eq!(
            rt.state().context_window,
            0,
            "an unknown model clears the window instead of guessing one"
        );
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn queued_clear_acknowledges_with_state_changed() {
        let (rt, mut rx, handle, _ws) = runtime_with(
            "clear-ack",
            Box::new(GateProvider::new(vec![Response::Text(text("a"))])),
        );
        rt.prompt("first");
        collect_until_settled(&mut rx);
        // A queued /clear must emit state_changed so an rpc adapter has a
        // deterministic ack boundary.
        rt.clear();
        let mut seen_state = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !seen_state && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(e) => seen_state = matches!(e, Event::StateChanged { .. }),
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
            }
        }
        assert!(seen_state, "clear must ack with state_changed");
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn two_runtimes_are_isolated() {
        // A cancel on one runtime must not affect the other.
        let (rt_a, rx_a, handle_a, _ws_a) = runtime_with("iso-a", Box::new(AbortProvider));
        let (rt_b, mut rx_b, handle_b, _ws_b) = runtime_with(
            "iso-b",
            Box::new(GateProvider::new(vec![Response::Text(text("b ok"))])),
        );
        rt_a.abort(); // cancels a's in-flight turn only
        rt_b.prompt("hi");
        let (events_b, settled_b) = collect_until_settled(&mut rx_b);
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
        let (rt, mut rx, handle, _ws) = runtime_with(
            "basic",
            Box::new(GateProvider::new(vec![Response::Text(text("hello world"))])),
        );
        rt.prompt("greet me");
        let (events, settled) = collect_until_settled(&mut rx);
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
    fn switch_provider_updates_state_and_keeps_history() {
        let (rt, mut rx, handle, _ws) = runtime_with(
            "switch",
            Box::new(GateProvider::new(vec![Response::Text(text("answer"))])),
        );
        // Have a conversation first.
        rt.prompt("hi");
        let _ = collect_until_settled(&mut rx);
        let before = rt.history().len();
        // Ollama needs no key, so the switch is allowed.
        rt.set_provider("ollama");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut switched = false;
        while !switched && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(Event::StateChanged { provider, .. }) if provider == "ollama" => switched = true,
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        assert!(switched, "expected a state change for the new provider");
        assert_eq!(rt.provider_kind().name, "ollama");
        // The conversation is kept across a provider switch.
        assert_eq!(rt.history().len(), before);
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn switch_provider_refuses_without_a_key() {
        std::env::remove_var("GEMINI_API_KEY");
        let (rt, mut rx, handle, _ws) = runtime_with("refuse", Box::new(GateProvider::new(vec![])));
        rt.set_provider("gemini");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut err = None;
        while err.is_none() && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(Event::Error { message, kind }) => err = Some((message, kind)),
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        let (msg, kind) = err.expect("expected a refusal");
        assert!(msg.contains("no API key for provider 'gemini'"), "{msg}");
        assert_eq!(
            kind,
            Some(ErrorKind::Auth),
            "the refusal must be classifiable so a UI can prompt for a key"
        );
        assert_eq!(rt.provider_kind().name, "openai");
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn default_list_models_reports_unsupported() {
        // GateProvider implements only `complete`, so it inherits the default.
        let g = GateProvider::new(vec![]);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt.block_on(g.list_models()).unwrap_err();
        assert!(matches!(err, ProviderError::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn refresh_models_caches_and_emits() {
        let provider = FakeProvider::new(vec![]).with_models(vec!["m2".into(), "m1".into()]);
        let (rt, mut rx, handle, _ws) = runtime_with("models", Box::new(provider));
        rt.refresh_models();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let mut listed: Option<Vec<String>> = None;
        while listed.is_none() && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(Event::ModelsListed { models }) => listed = Some(models),
                Ok(_) => {}
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        assert_eq!(
            listed.as_deref(),
            Some(&["m2".to_string(), "m1".to_string()][..])
        );
        assert_eq!(rt.models(), vec!["m2".to_string(), "m1".to_string()]);
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn thinking_deltas_are_emitted_and_stay_out_of_history() {
        let (rt, mut rx, handle, _ws) = runtime_with(
            "thinking",
            Box::new(
                FakeProvider::new(vec![Response::Text(text("answer"))])
                    .with_thinking(vec!["hmm".into()]),
            ),
        );
        rt.prompt("what model are you");
        let (events, settled) = collect_until_settled(&mut rx);
        assert!(settled);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ThinkingDelta { text } if text == "hmm")));
        // Reasoning is display-only: it never enters the canonical history.
        assert!(!rt
            .history()
            .iter()
            .any(|m| matches!(m, Message::Assistant { text: Some(t), .. } if t.contains("hmm"))));
        rt.shutdown();
        handle.join().unwrap_or(());
    }

    #[test]
    fn tool_calls_execute_and_emit_lifecycle_events() {
        let (dir, ws) = workspace("tools");
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
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, Box::new(builder), tools, ws.clone());
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("run a command");
        let (events, settled) = collect_until_settled(&mut rx);
        assert!(settled);
        // The event carries the call's arguments so a frontend can
        // show the executed command.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::ToolStart { name, args: Some(a), .. }
                if name == "bash" && a.get("command").and_then(|v| v.as_str()) == Some("echo hi")
        )));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { name, ok: true, .. } if name == "bash")));
        // The same event carries the result text so a frontend can
        // render the output under the call line.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::ToolEnd { name, ok: true, output: Some(out), .. }
                if name == "bash" && out.contains("hi")
        )));
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
        let (rt, mut rx, handle, _ws) = runtime_with("abort", Box::new(AbortProvider));
        rt.prompt("tell me a story");
        let (events, settled) = collect_until_settled(&mut rx);
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
        release_rx: Mutex<mpsc::Receiver<()>>,
        /// Sent to the test each time a call starts (flow control).
        started_tx: mpsc::Sender<()>,
    }

    impl Provider for PausableProvider {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async move {
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
            })
        }
    }

    #[test]
    fn steer_delivered_after_assistant_finishes_tool_calls() {
        let (dir, ws) = workspace("steer");
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
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, Box::new(provider), tools, ws.clone());
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
        let (events, settled) = collect_until_settled(&mut rx);
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
        let (dir, ws) = workspace("followup");
        let provider = Box::new(GateProvider::new(vec![
            Response::Text(text("first answer")),
            Response::Text(text("second answer")),
        ]));
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, mut rx) = AgentRuntime::new(cfg, provider, tools, ws.clone());
        let worker = rt.clone();
        let handle = std::thread::spawn(move || worker.run_forever());
        rt.prompt("first question");
        // Queue the follow-up while the first turn is still running.
        rt.follow_up("second question");
        let (events, settled) = collect_until_settled(&mut rx);
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

    /// A provider that always reports an aborted stream with partial text.
    struct AbortProvider;
    impl Provider for AbortProvider {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async move {
                Ok(Completion {
                    response: Response::Text("partial story".into()),
                    prompt_tokens: None,
                    aborted: true,
                })
            })
        }
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;
    use crate::provider::ToolCall;

    #[test]
    fn tool_calls_execute_in_parallel() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let tools = ToolSet::new(1000);
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(cfg, Box::new(ParallelCallProvider), tools, ws.clone());
        // Two bash calls each sleep 400ms; parallel wall time is ~400ms, not
        // ~800ms.
        let start = std::time::Instant::now();
        let answer = rt.run_once("run both").unwrap();
        let elapsed = start.elapsed();
        assert_eq!(answer, "done");
        assert!(
            elapsed < std::time::Duration::from_millis(750),
            "parallel tools should finish well under the sequential sum: {elapsed:?}"
        );
    }

    /// A provider that first emits two parallel bash `sleep 0.4` calls, then
    /// a final answer.
    struct ParallelCallProvider;
    impl crate::provider::Provider for ParallelCallProvider {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            _effort_params: &'a serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<
            'a,
            Result<crate::provider::Completion, crate::provider::ProviderError>,
        > {
            Box::pin(async move {
                use std::sync::atomic::Ordering as O;
                static CALLS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = CALLS.fetch_add(1, O::SeqCst);
                let response = if n == 0 {
                    Response::ToolCalls(vec![
                        ToolCall {
                            id: "p1".into(),
                            name: "bash".into(),
                            args: serde_json::json!({"command": "sleep 0.4"}),
                        },
                        ToolCall {
                            id: "p2".into(),
                            name: "bash".into(),
                            args: serde_json::json!({"command": "sleep 0.4"}),
                        },
                    ])
                } else {
                    Response::Text("done".into())
                };
                Ok(crate::provider::Completion {
                    response,
                    prompt_tokens: None,
                    aborted: false,
                })
            })
        }
    }
}

#[cfg(test)]
mod compaction_tests {
    use super::*;
    use crate::provider::fake::FakeProvider;

    /// Block on an async runtime method (tests are sync).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }

    fn runtime_with_summarizer(summary: Option<String>) -> AgentRuntime {
        let dir = tempfile::tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let tools = ToolSet::new(1000);
        let responses = summary.map(|s| vec![Response::Text(s)]).unwrap_or_default();
        let cfg = Config {
            max_iterations: 5,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(cfg, Box::new(FakeProvider::new(responses)), tools, ws);
        rt
    }

    #[test]
    fn compaction_replaces_old_turns_with_a_summary() {
        let rt = runtime_with_summarizer(Some("compact summary here".into()));
        // Build a history that is over budget: seed + a long old turn.
        let mut history = vec![
            Message::System("seed prompt that is reasonably long".into()),
            Message::User("first long question that fills space".into()),
            Message::Assistant {
                text: Some("a".repeat(5000)),
                tool_calls: vec![],
            },
            Message::User("second question".into()),
        ];
        let mut a0 = 0usize;
        let mut a1 = 0usize;
        block_on(rt.manage_context(&mut history, 2, 120, &mut a0, &mut a1));
        // The old assistant text was replaced by the summary.
        assert!(
            history
                .iter()
                .any(|m| matches!(m, Message::System(s) if s.contains("compact summary here"))),
            "expected a summary message: {history:?}"
        );
        assert!(
            !history
                .iter()
                .any(|m| matches!(m, Message::Assistant { text: Some(t), .. } if t.len() > 100)),
            "long old turn should be gone"
        );
    }

    #[test]
    fn compaction_falls_back_to_trimming_without_summarizer() {
        // A provider with no scripted summary: FakeProvider returns "done" for
        // the compaction call (not a useful summary is still a text). We rely
        // on the fallback path keeping the session bounded.
        let rt = runtime_with_summarizer(None);
        let mut history = vec![
            Message::System("seed".into()),
            Message::User("q".into()),
            Message::Assistant {
                text: Some("x".repeat(300)),
                tool_calls: vec![],
            },
        ];
        let mut a0 = 0usize;
        let mut a1 = 0usize;
        block_on(rt.manage_context(&mut history, 2, 50, &mut a0, &mut a1));
        // History is now within budget (trimming happened).
        assert!(crate::runtime::anchored_total(&history, a0, a1) <= 50);
    }
}

#[cfg(test)]
mod accessor_tests {
    use super::*;

    #[test]
    fn provider_kind_is_exposed_for_frontends() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let tools = ToolSet::new(1000);
        let cfg = Config {
            provider: crate::config::provider_by_name("anthropic").unwrap(),
            max_iterations: 5,
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(crate::provider::fake::FakeProvider::new(vec![])),
            tools,
            ws,
        );
        assert_eq!(rt.provider_kind().name, "anthropic");
    }
}
#[cfg(test)]
mod effort_tests {
    use super::*;
    use crate::provider::{Completion, ProviderError};
    use std::sync::Mutex as StdMutex;

    /// Records the provider-flavored effort params of every completion call.
    struct EffortRecorder {
        seen: StdMutex<Vec<serde_json::Value>>,
    }

    impl EffortRecorder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                seen: StdMutex::new(Vec::new()),
            })
        }

        fn seen(&self) -> Vec<serde_json::Value> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Provider for EffortRecorder {
        fn complete<'a>(
            &'a self,
            _history: &'a [Message],
            _tools: &'a [serde_json::Value],
            effort_params: &'a serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            Box::pin(async move {
                self.seen.lock().unwrap().push(effort_params.clone());
                Ok(Completion {
                    response: Response::Text("done".into()),
                    prompt_tokens: None,
                    aborted: false,
                })
            })
        }
    }

    /// `AgentRuntime::new` takes `Box<dyn Provider>`, so hand it a shared
    /// handle to the one recorder the test keeps inspecting.
    struct SharedRecorder(Arc<EffortRecorder>);

    impl Provider for SharedRecorder {
        fn complete<'a>(
            &'a self,
            history: &'a [Message],
            tools: &'a [serde_json::Value],
            effort_params: &'a serde_json::Value,
            cancel: tokio_util::sync::CancellationToken,
            on_delta: &'a mut (dyn FnMut(StreamDelta) + Send),
        ) -> futures::future::BoxFuture<'a, Result<Completion, ProviderError>> {
            self.0
                .complete(history, tools, effort_params, cancel, on_delta)
        }
    }

    fn runtime_with_effort(
        effort: Effort,
    ) -> (AgentRuntime, Arc<EffortRecorder>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let workspace = Workspace::new(dir.path().to_path_buf()).unwrap();
        let cfg = Config {
            provider: crate::config::default_provider(),
            max_iterations: 5,
            workspace: dir.path().to_path_buf(),
            effort,
            ..Config::defaults(dir.path().to_path_buf())
        };
        let recorder = EffortRecorder::new();
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(SharedRecorder(Arc::clone(&recorder))),
            ToolSet::new(1000),
            workspace,
        );
        (rt, recorder, dir)
    }

    /// The runtime computes the provider-flavored effort parameters
    /// (`provider_effort`) and hands them to every completion call.
    #[test]
    fn effort_params_flow_into_provider_calls() {
        let (rt, recorder, _dir) = runtime_with_effort(Effort::Medium);
        rt.set_effort(Effort::High);
        rt.run_once("task").unwrap();
        let seen = recorder.seen();
        assert_eq!(seen.len(), 1, "one completion call expected");
        assert_eq!(
            seen[0],
            serde_json::json!({ "reasoning_effort": "high" }),
            "openai effort mapping must reach the provider"
        );
    }

    /// A configured effort (persisted by the TUI, read back at startup) seeds
    /// the runtime state and reaches the first completion — the level must not
    /// reset to the default on restart.
    #[test]
    fn configured_effort_is_used_without_a_runtime_set() {
        let (rt, recorder, _dir) = runtime_with_effort(Effort::High);
        assert_eq!(rt.state().effort, Effort::High);
        rt.run_once("task").unwrap();
        let seen = recorder.seen();
        assert_eq!(seen.len(), 1, "one completion call expected");
        assert_eq!(
            seen[0],
            serde_json::json!({ "reasoning_effort": "high" }),
            "the configured effort must be sent, not the Medium default"
        );
    }
}
