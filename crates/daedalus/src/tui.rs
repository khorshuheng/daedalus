//! Ratatui TUI, modeled on pi's interactive mode.
//!
//! The TUI is a **client of `AgentRuntime`** — it never owns the agent loop.
//! It sends commands (`prompt`/`steer`/`abort`/`set_model`/`set_effort`/
//! `switch_workspace`/`clear`/`resume`) and renders the runtime's Event
//! stream. Layout: transcript on top, a line-editing input with a visible
//! caret at the bottom, a centered picker overlay for /model,
//! /effort, /provider and /resume, and a footer/status line (provider, model, effort,
//! animated spinner while busy). `/skills` lists skills and
//! `/skill <name>` (load an instruction file as a user message). Dragging over
//! the transcript highlights text; Ctrl-Y copies the selection with OSC 52.
//!
//! This module is split so the behavior is testable without a terminal:
//! the pure model (`parse_slash`, `LineAction` routing, `apply_event`
//! transcript/status updates) lives here with unit tests, and the
//! `run_tui` shell (crossterm raw mode + alternate screen, event poll
//! loop, ratatui draw) is a thin wrapper over it. The interactive path
//! never uses `println!`/`print!` — ratatui owns the screen.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use daedalus_core::config::{persist_edit, ConfigEdit, PROVIDERS};
use daedalus_core::provider::Message;
use daedalus_core::runtime::{AgentRuntime, Effort, ErrorKind, Event, RuntimeState};
use daedalus_core::theme::{Modifiers, StyleSpec, Theme, ThemeColor, Token};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line as TLine, Span};
use ratatui::widgets::Clear;
use ratatui::Frame;
use ratatui::Terminal;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A line submitted in the input editor, classified into the action the TUI
/// should take. Pure routing: what the runtime does with it depends on
/// whether the agent is busy (pi: type-while-running steers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineAction {
    /// A normal user message: prompt when idle, steer when busy.
    Message(String),
    /// A slash command (`/resume`, `/clear`, `/help`, ...).
    Command(SlashCommand),
}

/// The slash commands the TUI understands (pi-style).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    /// Store an API key for the configured provider in the OS keyring.
    Login,
    /// Change the model (opens the model picker when no argument).
    Model(String),
    /// Change the provider (opens the provider picker when no argument).
    Provider(String),
    /// Change the thinking effort (opens the effort picker when no argument).
    Effort(String),
    /// Change the workspace.
    Workspace(PathBuf),
    /// List the discovered skills (user + workspace).
    Skills,
    /// Load a skill's content into the conversation as a user message.
    Skill(String),
    /// List the registered tools (built-ins + MCP servers).
    Tools,
    /// Continue the previous session for this workspace.
    Resume,
    /// Reset the conversation to a fresh context.
    Clear,
    /// Show the command list.
    Help,
    /// Quit (saving the session).
    Exit,
}

/// Metadata for one slash command: the single source of truth for the parser,
/// `/help`, and the completion dropdown.
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// Argument hint shown after the name, e.g. `Some("<name>")`.
    pub args: Option<&'static str>,
    pub description: &'static str,
    build: fn(String) -> SlashCommand,
}

/// Every slash command, in display order.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "login",
        aliases: &[],
        args: None,
        description: "Store a provider API key in the keyring",
        build: |_| SlashCommand::Login,
    },
    CommandSpec {
        name: "model",
        aliases: &[],
        args: Some("<name>"),
        description: "Change the model (picker when omitted)",
        build: SlashCommand::Model,
    },
    CommandSpec {
        name: "provider",
        aliases: &[],
        args: Some("<name>"),
        description: "Change the provider (picker when omitted)",
        build: SlashCommand::Provider,
    },
    CommandSpec {
        name: "effort",
        aliases: &[],
        args: Some("[level]"),
        description: "Change thinking effort (picker when omitted)",
        build: SlashCommand::Effort,
    },
    CommandSpec {
        name: "workspace",
        aliases: &[],
        args: Some("<path>"),
        description: "Change the workspace directory",
        build: |a| SlashCommand::Workspace(PathBuf::from(a)),
    },
    CommandSpec {
        name: "skills",
        aliases: &[],
        args: None,
        description: "List discovered skills",
        build: |_| SlashCommand::Skills,
    },
    CommandSpec {
        name: "skill",
        aliases: &[],
        args: Some("<name>"),
        description: "Load a skill's instructions",
        build: SlashCommand::Skill,
    },
    CommandSpec {
        name: "tools",
        aliases: &[],
        args: None,
        description: "List registered tools",
        build: |_| SlashCommand::Tools,
    },
    CommandSpec {
        name: "resume",
        aliases: &[],
        args: None,
        description: "Pick a previous session",
        build: |_| SlashCommand::Resume,
    },
    CommandSpec {
        name: "clear",
        aliases: &[],
        args: None,
        description: "Reset the conversation",
        build: |_| SlashCommand::Clear,
    },
    CommandSpec {
        name: "help",
        aliases: &[],
        args: None,
        description: "Show this list",
        build: |_| SlashCommand::Help,
    },
    CommandSpec {
        name: "exit",
        aliases: &["quit", "q"],
        args: None,
        description: "Quit (saving the session)",
        build: |_| SlashCommand::Exit,
    },
];

/// Slash-command candidates for the input's leading `/token`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// Indexes into [`COMMANDS`].
    pub matches: Vec<usize>,
    pub selected: usize,
}

/// Live-filter slash commands by the input's leading `/token`. `None` unless
/// the input is a bare command token with at least one match.
pub fn slash_completions(input: &str) -> Option<Completion> {
    let rest = input.strip_prefix('/')?;
    if rest.contains(char::is_whitespace) {
        return None;
    }
    let prefix = rest.to_ascii_lowercase();
    let matches: Vec<usize> = COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name.starts_with(&prefix))
        .map(|(i, _)| i)
        .collect();
    if matches.is_empty() {
        None
    } else {
        Some(Completion {
            matches,
            selected: 0,
        })
    }
}

/// The text inserted when accepting `spec`: `/name ` with a trailing space so
/// an argument can follow.
pub fn accept_completion(spec: &CommandSpec) -> String {
    format!("/{} ", spec.name)
}

/// Classify a raw submitted line. Lines that do not start with `/` are user
/// messages; `/name` is parsed as a command, with `/name value` carrying an
/// argument.
pub fn parse_line(line: &str) -> LineAction {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix('/') {
        let (name, arg) = match rest.split_once(' ') {
            Some((n, a)) => (n, a.trim()),
            None => (rest, ""),
        };
        let spec = COMMANDS
            .iter()
            .find(|c| c.name == name || c.aliases.contains(&name));
        match spec {
            Some(c) => LineAction::Command((c.build)(arg.to_string())),
            None => LineAction::Command(SlashCommand::Help), // unknown -> help
        }
    } else {
        LineAction::Message(trimmed.to_string())
    }
}

/// One-line, human-readable summary of a tool call's arguments, shown after
/// the tool name in the transcript. Returns `None` when there is no
/// useful single-line summary, so the call line stays just `⚙ <name>`.
pub fn tool_detail(name: &str, args: Option<&serde_json::Value>) -> Option<String> {
    let args = args?;
    let raw = match name {
        "bash" => format!("$ {}", args.get("command")?.as_str()?),
        "read" | "edit" | "write" => {
            let path = args
                .get("path")
                .or_else(|| args.get("file_path"))?
                .as_str()?;
            format!("{name} {path}")
        }
        _ => return None,
    };
    Some(one_line(&raw))
}

/// Longest detail rendered after the tool name, to keep a heredoc or a `write`
/// body from flooding the transcript.
const TOOL_DETAIL_MAX: usize = 120;

/// Collapse to a single line (the first line plus ` …` when more follow) and
/// cap at [`TOOL_DETAIL_MAX`] characters, char-boundary safe.
fn one_line(raw: &str) -> String {
    let cleaned = raw.replace('\r', "");
    let multi = cleaned.contains('\n');
    let first = cleaned.lines().next().unwrap_or("");
    let mut out = first.trim_end().to_string();
    if multi {
        out.push_str(" …");
    }
    if out.chars().count() > TOOL_DETAIL_MAX {
        out = out.chars().take(TOOL_DETAIL_MAX - 1).collect();
        out.push('…');
    }
    out
}

/// Transcript scrollback. `top` is the first visible wrapped row; `follow`
/// keeps the view pinned to the newest content as it grows; `viewport` is the
/// last rendered height, so key handling can page without knowing the terminal
/// size. Pure state, unit-testable without a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptScroll {
    top: usize,
    follow: bool,
    viewport: usize,
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            top: 0,
            follow: true,
            viewport: 0,
        }
    }
}

impl TranscriptScroll {
    /// Clamp for the current content height and viewport, returning the first
    /// row to render. Re-enables follow once the bottom is reached.
    pub fn resolve(&mut self, total: usize, viewport: usize) -> usize {
        self.viewport = viewport;
        let max_top = total.saturating_sub(viewport);
        if self.follow {
            self.top = max_top;
        } else {
            self.top = self.top.min(max_top);
            if self.top >= max_top {
                self.follow = true;
            }
        }
        self.top
    }

    /// Page up one viewport and stop following the tail.
    pub fn page_up(&mut self) {
        self.follow = false;
        self.top = self.top.saturating_sub(self.viewport.max(1));
    }

    /// Page down one viewport; `resolve` re-enables follow at the bottom.
    pub fn page_down(&mut self) {
        self.top = self.top.saturating_add(self.viewport.max(1));
        self.follow = false;
    }

    /// Scroll by `lines` (negative = up); scrolling up stops following.
    pub fn scroll_by(&mut self, lines: isize) {
        if lines < 0 {
            self.follow = false;
            self.top = self.top.saturating_sub(lines.unsigned_abs());
        } else {
            self.top = self.top.saturating_add(lines as usize);
            self.follow = false;
        }
    }

    /// Jump to the oldest content.
    pub fn jump_to_top(&mut self) {
        self.follow = false;
        self.top = 0;
    }

    /// Resume following the newest content.
    pub fn follow_tail(&mut self) {
        self.follow = true;
    }
}

/// A single line of the transcript (user or assistant content). Assistant
/// text streams in via `text_delta` events and is appended to the current
/// assistant line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptLine {
    User(String),
    Assistant(String),
    /// Streamed model reasoning, rendered dim and italic.
    Thinking(String),
    Tool(String),
    /// A finished tool call: the `✓`/`✗` marker line (theme tokens),
    /// followed by the result text when present.
    ToolResult {
        name: String,
        ok: bool,
        output: Option<String>,
    },
    Notice(String),
}

/// Pure UI model: the transcript and the status footer, updated from runtime
/// Events. No terminal I/O here, so it is unit-testable.
#[derive(Debug, Default, Clone)]
pub struct UiModel {
    pub transcript: Vec<TranscriptLine>,
    pub assistant_buf: String,
    /// Streamed model reasoning, flushed as a `Thinking` line.
    pub thinking_buf: String,
    pub state: RuntimeState,
    pub usage: Option<usize>,
    pub iterations: usize,
    /// True when the last event was a turn end (used to reset stats display).
    pub settled: bool,
    /// Transcript scrollback (follow the tail unless the user scrolled up).
    pub scroll: TranscriptScroll,
    /// Global tool-output expansion (Ctrl+O). When false, tool
    /// results render as a collapsed preview. Private so it is only ever
    /// changed through `toggle_verbose`, which invalidates the render cache.
    verbose: bool,
    /// Bumped whenever the transcript changes, to key the render cache.
    pub revision: u64,
    /// Cache of the rendered transcript, keyed by (revision, area width).
    pub md_cache: Option<(u64, u16, Vec<ratatui::text::Line<'static>>)>,
    /// Active slash-command completion, if any.
    pub completion: Option<Completion>,
    /// True while a `/model` model-list fetch is in flight.
    pub model_fetch_pending: bool,
    /// Mouse text selection over the transcript (absolute screen cells), for
    /// Ctrl-Y copy. Cleared by scrolling and by a plain click.
    pub selection: Option<Selection>,
    /// The transcript viewport's inner screen rect from the last draw.
    pub transcript_area: Rect,
    /// The transcript viewport's visible text rows from the last draw, one
    /// string per screen row (wrapping included), for selection extraction.
    pub transcript_rows: Vec<String>,
    /// Where a model/provider chosen at runtime is remembered (set by
    /// `run_tui`; `None` writes nothing, so tests never touch a real config).
    pub persist: Option<PathBuf>,
}

impl UiModel {
    pub fn new(state: RuntimeState) -> Self {
        Self {
            transcript: Vec::new(),
            assistant_buf: String::new(),
            thinking_buf: String::new(),
            state,
            usage: None,
            iterations: 0,
            settled: false,
            scroll: TranscriptScroll::default(),
            verbose: false,
            revision: 0,
            md_cache: None,
            completion: None,
            model_fetch_pending: false,
            selection: None,
            transcript_area: Rect::default(),
            transcript_rows: Vec::new(),
            persist: None,
        }
    }

    /// Fold a runtime Event into the model. Returns nothing; the caller just
    /// re-renders. This is the single place events become display state.
    pub fn apply_event(&mut self, event: &Event) {
        let before = self.transcript.len();
        match event {
            Event::AgentStart { .. } => {}
            Event::TurnStart {} => {
                self.settled = false;
                self.iterations += 1;
            }
            Event::TextDelta { text } => {
                self.flush_thinking();
                self.assistant_buf.push_str(text);
            }
            Event::ThinkingDelta { text } => {
                self.thinking_buf.push_str(text);
            }
            Event::ToolStart { name, args, .. } => {
                self.flush_thinking();
                self.flush_assistant();
                let line = match tool_detail(name, args.as_ref()) {
                    Some(detail) => format!("⚙ {name} {detail}"),
                    None => format!("⚙ {name}"),
                };
                self.transcript.push(TranscriptLine::Tool(line));
            }
            Event::ToolEnd {
                name, ok, output, ..
            } => {
                self.flush_thinking();
                self.transcript.push(TranscriptLine::ToolResult {
                    name: name.clone(),
                    ok: *ok,
                    output: output.clone(),
                });
            }
            Event::TurnEnd {} => {
                self.flush_thinking();
                self.flush_assistant();
            }
            Event::Usage { prompt_tokens } => {
                self.usage = *prompt_tokens;
            }
            Event::QueueUpdate { .. } => {}
            Event::StateChanged {
                model,
                effort,
                workspace,
                provider,
            } => {
                self.state.model = model.clone();
                self.state.effort = *effort;
                self.state.workspace = workspace.clone();
                self.state.provider = provider.clone();
            }
            Event::AgentSettled {
                text,
                interrupted: _,
            } => {
                self.settled = true;
                self.flush_thinking();
                // `turn_end` already flushed streamed text, so `assistant_buf`
                // is empty here even when the answer streamed — the previous
                // `!assistant_buf.contains(text)` guard therefore re-appended
                // it, duplicating the answer for every provider that streams.
                // Only add `text` when it is not already the last line (i.e. a
                // provider that did not stream it).
                let already_shown = matches!(
                    self.transcript.last(),
                    Some(TranscriptLine::Assistant(t)) if t == text
                );
                if !text.is_empty() && !already_shown && !self.assistant_buf.contains(text) {
                    self.assistant_buf.push_str(text);
                }
                self.flush_assistant();
            }
            Event::ModelsListed { .. } => {} // handled by the shell
            Event::Error { message, .. } => {
                self.flush_thinking();
                self.flush_assistant();
                self.transcript
                    .push(TranscriptLine::Notice(format!("error: {message}")));
            }
        }
        // Only a changed transcript invalidates the render cache. Streamed
        // thinking/assistant deltas touch their buffers, not `transcript`, and
        // the live `assistant_buf` is rendered directly each frame.
        if self.transcript.len() != before {
            self.revision += 1;
        }
    }

    /// Push accumulated model reasoning (if any) as transcript lines.
    /// Always called before `flush_assistant`, so thinking renders before the
    /// answer it precedes. Split per reasoning line so a large
    /// block wraps line-by-line instead of as one huge paragraph every frame.
    fn flush_thinking(&mut self) {
        if !self.thinking_buf.is_empty() {
            let buffered = std::mem::take(&mut self.thinking_buf);
            for line in buffered.lines() {
                self.transcript
                    .push(TranscriptLine::Thinking(line.to_string()));
            }
        }
    }

    /// Push the accumulated assistant text (if any) as a transcript line.
    fn flush_assistant(&mut self) {
        if !self.assistant_buf.is_empty() {
            self.transcript
                .push(TranscriptLine::Assistant(std::mem::take(
                    &mut self.assistant_buf,
                )));
        }
    }

    /// Show a user message as a transcript line (called on submit).
    pub fn push_user(&mut self, text: &str) {
        // A new message should be visible: resume following the tail.
        self.scroll.follow_tail();
        self.transcript.push(TranscriptLine::User(text.to_string()));
        self.revision += 1;
    }

    /// Show a notice line (slash command results, etc.).
    pub fn push_notice(&mut self, text: &str) {
        self.transcript
            .push(TranscriptLine::Notice(text.to_string()));
        self.revision += 1;
    }

    /// Flip the global tool-output expansion (Ctrl+O) and invalidate
    /// the cached transcript so the change is visible on the next frame.
    pub fn toggle_verbose(&mut self) {
        self.verbose = !self.verbose;
        self.revision += 1;
    }

    /// Replace the visible transcript with a resumed conversation. Only user
    /// and assistant text is shown; tool traffic stays in the runtime history,
    /// which the model still sees.
    pub fn load_history(&mut self, history: &[Message]) {
        self.transcript.clear();
        self.assistant_buf.clear();
        self.thinking_buf.clear();
        // Tool results carry only the call id; map ids back to names so a
        // resumed session renders `✓ bash` rather than an anonymous marker.
        let mut names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for message in history {
            match message {
                Message::User(text) => {
                    self.transcript.push(TranscriptLine::User(text.clone()));
                }
                Message::Assistant { text, tool_calls } => {
                    for call in tool_calls {
                        names.insert(call.id.as_str(), call.name.as_str());
                    }
                    if let Some(text) = text {
                        if !text.is_empty() {
                            self.transcript
                                .push(TranscriptLine::Assistant(text.clone()));
                        }
                    }
                }
                Message::ToolResult {
                    tool_call_id,
                    result,
                } => {
                    let name = names.get(tool_call_id.as_str()).copied().unwrap_or("tool");
                    self.transcript.push(TranscriptLine::ToolResult {
                        name: name.to_string(),
                        ok: daedalus_core::runtime::tool_result_ok(result),
                        output: (!result.is_empty()).then(|| result.clone()),
                    });
                }
                _ => {}
            }
        }
        self.scroll.follow_tail();
        self.revision += 1;
    }

    /// Begin a selection at a screen cell. A press outside the transcript
    /// viewport just clears any existing selection.
    pub fn selection_start(&mut self, col: u16, row: u16) {
        if self.transcript_area.contains(Position { x: col, y: row }) {
            self.selection = Some(Selection {
                anchor: (col, row),
                head: (col, row),
            });
        } else {
            self.selection = None;
        }
    }

    /// Extend the active selection, clamped to the transcript viewport.
    pub fn selection_drag(&mut self, col: u16, row: u16) {
        if let Some(sel) = &mut self.selection {
            let area = self.transcript_area;
            sel.head = (
                col.clamp(area.left(), area.right().saturating_sub(1)),
                row.clamp(area.top(), area.bottom().saturating_sub(1)),
            );
        }
    }

    /// Finish a selection. A click that never moved clears it; a real drag is
    /// kept so Ctrl-Y can copy it.
    pub fn selection_end(&mut self) {
        if self.selection.is_some_and(|s| s.is_empty()) {
            self.selection = None;
        }
    }

    /// Drop the selection (used when scrolling, so the highlight never covers
    /// text it was not made over).
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }
}

/// The set of effort choices offered by the `/effort` picker.
pub const EFFORT_CHOICES: &[Effort] = &[
    Effort::Off,
    Effort::Minimal,
    Effort::Low,
    Effort::Medium,
    Effort::High,
];

/// A single-line text editor for the input box: the text plus a
/// byte cursor that always sits on a UTF-8 char boundary. Pure logic — no
/// terminal I/O — so cursor movement, insertion, deletion and the visible
/// window are unit-testable. Insert-style (arrow keys); vim modal editing is
/// an explicit non-goal.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InputEditor {
    text: String,
    /// Byte offset into `text` on a char boundary (`== text.len()` at end).
    cursor: usize,
}

// The `new`/`text`/`cursor` accessors are exercised only by unit tests — the
// shell reads the visible window/caret instead — hence the targeted allow.
#[allow(dead_code)]
impl InputEditor {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            cursor: text.len(),
            text,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Take the whole text and reset for the next line.
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    /// Replace the whole text; the cursor moves to the end.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.text.len();
    }

    /// Insert at the cursor. The editor is single-line, so pasted control
    /// characters (CR/LF from bracketed paste) are flattened to spaces.
    pub fn insert(&mut self, s: &str) {
        let cleaned: String = s
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let at = self.cursor;
        self.text.insert_str(at, &cleaned);
        self.cursor = at + cleaned.len();
    }

    pub fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Delete the char before the cursor (Backspace).
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.prev_boundary(self.cursor);
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
        }
    }

    /// Delete the char after the cursor (Delete / Ctrl-D).
    pub fn delete(&mut self) {
        if self.cursor < self.text.len() {
            let next = self.next_boundary(self.cursor);
            self.text.replace_range(self.cursor..next, "");
        }
    }

    /// Move the caret one char left; no-op at the start of the line.
    pub fn left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.prev_boundary(self.cursor);
        }
    }

    /// Move the caret one char right; no-op at the end of the line.
    pub fn right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor = self.next_boundary(self.cursor);
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Kill back to the start of the previous word (Ctrl-W, readline-style:
    /// the word and the whitespace before it).
    pub fn kill_prev_word(&mut self) {
        let start = self.word_start(self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Kill everything before the cursor (Ctrl-U).
    pub fn kill_to_start(&mut self) {
        self.text.replace_range(0..self.cursor, "");
        self.cursor = 0;
    }

    /// Kill everything from the cursor to the end (Ctrl-K).
    pub fn kill_to_end(&mut self) {
        self.text.replace_range(self.cursor.., "");
    }

    /// Byte index of the char boundary before `i`. Precondition: `i` is a
    /// char boundary with `i > 0` (callers guard the edges; stepping below 0
    /// would underflow, and `is_char_boundary` is false for index > len, so
    /// an unguarded call could loop forever).
    fn prev_boundary(&self, mut i: usize) -> usize {
        debug_assert!(i > 0 && i <= self.text.len());
        i -= 1;
        while !self.text.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    /// Byte index of the char boundary after `i`. Precondition: `i` is a
    /// char boundary with `i < text.len()` (see `prev_boundary`).
    fn next_boundary(&self, mut i: usize) -> usize {
        debug_assert!(i < self.text.len());
        i += 1;
        while !self.text.is_char_boundary(i) {
            i += 1;
        }
        i
    }

    /// Start byte of the previous word before `from` (readline Ctrl-W: the
    /// word under/ending at the cursor plus the whitespace before it).
    /// `from` must be a char boundary.
    fn word_start(&self, from: usize) -> usize {
        let bytes = self.text.as_bytes();
        let mut i = from;
        while i > 0 && !bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        while i > 0 && bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        i
    }

    /// The visible window of the text for a viewport `width` columns wide,
    /// returned as `(window, cursor_column)`. The caret behaves like a block
    /// cursor and occupies a cell inside the window, so the cursor column is
    /// always `< width` (never clipped onto the border): when the caret sits
    /// at the end of a full window, the leftmost char scrolls out of view.
    /// The whole text fits when short, else the tail is shown and the window
    /// follows the cursor. Widths come from unicode-width so wide (CJK)
    /// chars keep the caret aligned.
    pub fn window(&self, width: usize) -> (String, usize) {
        if width == 0 {
            return (String::new(), 0);
        }
        let chars: Vec<char> = self.text.chars().collect();
        // Cumulative column of each char boundary, plus the cursor's char index.
        let mut prefix = Vec::with_capacity(chars.len() + 1);
        prefix.push(0usize);
        for c in &chars {
            let w = UnicodeWidthChar::width(*c).unwrap_or(1);
            prefix.push(prefix.last().unwrap() + w);
        }
        let mut cursor_char = chars.len();
        for (i, (b, _)) in self.text.char_indices().enumerate() {
            if b == self.cursor {
                cursor_char = i;
                break;
            }
        }
        let total = *prefix.last().unwrap();
        let cursor_col = prefix[cursor_char];
        // Show the tail by default; scroll so the cursor stays in view.
        let mut start = total.saturating_sub(width);
        if cursor_col < start {
            start = cursor_col;
        } else if cursor_col >= start + width {
            start = cursor_col + 1 - width;
        }
        // Snap the start column back to a char boundary, keeping zero-width
        // chars attached to their base glyph.
        let mut start_char = prefix.partition_point(|&p| p <= start).saturating_sub(1);
        while start_char > 0 && prefix[start_char] == prefix[start_char - 1] {
            start_char -= 1;
        }
        let base = prefix[start_char];
        let mut end_char = start_char;
        while end_char < chars.len() && prefix[end_char + 1] - base <= width {
            end_char += 1;
        }
        let window: String = chars[start_char..end_char].iter().collect();
        (window, cursor_col.saturating_sub(base))
    }
}

/// Braille spinner frames, advanced by elapsed time so the
/// animation runs at a steady cadence independent of the render loop.
const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The spinner frame for a point in time (one step per 100ms).
fn spinner_frame(elapsed: Duration) -> char {
    SPINNER_FRAMES[(elapsed.as_millis() / 100) as usize % SPINNER_FRAMES.len()]
}

/// First visible row of a picker list so `selected` stays inside the viewport.
/// The list only scrolls when the selection leaves the window.
fn picker_offset(selected: usize, viewport: usize) -> usize {
    if viewport == 0 {
        return selected;
    }
    if selected < viewport {
        0
    } else {
        selected + 1 - viewport
    }
}

/// The model picker to open when the provider rejects the configured model.
///
/// The list is what the provider itself reported it serves — a hint, not a
/// gate, since a provider may also accept aliases it never names. `None` for
/// every other error, and for a rejection that named no alternatives: a picker
/// with no rows is a dead end, and the message is the remedy on its own.
fn model_picker_for_rejection(kind: &Option<ErrorKind>) -> Option<Picker> {
    match kind {
        Some(ErrorKind::InvalidModel {
            requested,
            supported,
        }) if !supported.is_empty() => Some(Picker::Model {
            selected: 0,
            models: supported.clone(),
            rejected_model: Some(requested.clone()),
        }),
        _ => None,
    }
}

/// A `width x height` rectangle centered inside `area` (clamped to it).
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// Recompute the slash-command completion for the current input, preserving
/// the highlighted row when the candidate set is unchanged.
fn refresh_completion(
    model: &mut UiModel,
    input: &InputEditor,
    picker: &Option<Picker>,
    login_pending: bool,
) {
    let next = if login_pending || picker.is_some() {
        None
    } else {
        slash_completions(input.text())
    };
    model.completion = match (model.completion.take(), next) {
        (Some(old), Some(mut new)) if old.matches == new.matches => {
            new.selected = old.selected.min(new.matches.len().saturating_sub(1));
            Some(new)
        }
        (_, next) => next,
    };
}

/// The model/provider the run started with — the baseline `remember_choice`
/// diffs against, so only a change the runtime actually adopted is saved.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Remembered {
    provider: String,
    model: String,
}

impl Remembered {
    fn of(state: &RuntimeState) -> Self {
        Self {
            provider: state.provider.clone(),
            model: state.model.clone(),
        }
    }
}

/// Remember a model/provider the runtime **actually adopted**, once per
/// change, by writing it into the config file at `persist`.
///
/// Diffing against the run's starting values (rather than hooking the pickers)
/// is what makes this correct for providers: a switch the runtime refused — an
/// unknown name, or no API key — never reaches `StateChanged`, so a refused
/// choice is never written back and the next run cannot start broken. The same
/// single hook covers the picker, `/model`, and `/provider`.
///
/// A failed write is a notice, never fatal: a read-only config must not take
/// the session down.
fn remember_choice(model: &mut UiModel, remembered: &mut Remembered) {
    let Some(path) = model.persist.clone() else {
        return;
    };
    if model.state.provider == remembered.provider && model.state.model == remembered.model {
        return;
    }
    let mut edit = ConfigEdit::default();
    if model.state.provider != remembered.provider {
        edit.provider = Some(model.state.provider.clone());
    }
    if model.state.model != remembered.model {
        edit.model = Some(model.state.model.clone());
    }
    // Move the baseline either way: the user has been told once, and a config
    // that keeps failing must not repeat the same notice on every event.
    remembered.provider = model.state.provider.clone();
    remembered.model = model.state.model.clone();
    let saved = edit_summary(&edit);
    match persist_edit(&path, &edit) {
        Ok(()) => model.push_notice(&format!("{saved} saved to {}", path.display())),
        Err(e) => model.push_notice(&format!("could not save to {}: {e}", path.display())),
    }
}

/// Render an edit as the TOML it wrote: `model = "m"`, `provider = "p"`, or
/// `provider = "p", model = "m"`.
fn edit_summary(edit: &ConfigEdit) -> String {
    let mut parts = Vec::new();
    if let Some(provider) = &edit.provider {
        parts.push(format!("provider = {provider:?}"));
    }
    if let Some(model) = &edit.model {
        parts.push(format!("model = {model:?}"));
    }
    parts.join(", ")
}

/// Ratatui rendering + event loop shell. Owns the screen (crossterm raw
/// mode + alternate screen); all *state* lives in the pure model above.
/// Never prints to stdout directly — ratatui owns the terminal. On exit the
/// session is auto-saved via `crate::auto_save`. A model/provider the user
/// changes at runtime is remembered in `persist` (the config file), or nowhere
/// when that is `None`.
pub fn run_tui(
    rt: &AgentRuntime,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
    initial: &str,
    session_root: &Path,
    theme: &Theme,
    persist: Option<PathBuf>,
) -> Result<i32, String> {
    enable_raw_mode().map_err(|e| format!("cannot enable raw mode: {e}"))?;
    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )
    .map_err(|e| format!("cannot enter alt screen: {e}"))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;

    let mut model = UiModel::new(rt.state());
    model.persist = persist;
    let mut remembered = Remembered::of(&model.state);
    let mut input = InputEditor::default();
    let mut picker: Option<Picker> = None;
    let mut login_pending = false;
    let mut should_exit = false;

    let result = (|| -> Result<i32, String> {
        // One-time editing-key hint so the line editor is discoverable.
        model.push_notice(
            "editing: ←→ Home End Del · Ctrl-W word · Ctrl-U/Ctrl-K line · PgUp/PgDn/↑↓ scroll · drag to select · Ctrl-Y copy · Ctrl-O tool output · /help commands",
        );
        // Seed the transcript with the initial prompt, then start the turn.
        if !initial.trim().is_empty() {
            model.push_user(initial);
            rt.prompt(initial);
        }

        let clock = Instant::now();
        let mut last_render = Instant::now();
        loop {
            if should_exit {
                break;
            }
            // Drain runtime events into the model.
            while let Ok(ev) = rx.try_recv() {
                // A pending `/model` fetch resolves into the model picker, or a
                // notice explaining why it failed.
                if model.model_fetch_pending {
                    match &ev {
                        Event::ModelsListed { models } if !models.is_empty() => {
                            picker = Some(Picker::Model {
                                selected: 0,
                                models: models.clone(),
                                rejected_model: None,
                            });
                            model.model_fetch_pending = false;
                        }
                        Event::ModelsListed { .. } => {
                            model.push_notice("no models reported; use /model <name>");
                            model.model_fetch_pending = false;
                        }
                        Event::Error { message, .. } => {
                            model.push_notice(&format!("{message}; use /model <name>"));
                            model.model_fetch_pending = false;
                        }
                        _ => {}
                    }
                }
                // The provider rejected the configured model. The transcript
                // already carries the message (which lists what it does
                // serve); turn that list into a picker rather than leaving the
                // user to retype a name from prose.
                if let Event::Error { kind, .. } = &ev {
                    if let Some(p) = model_picker_for_rejection(kind) {
                        picker = Some(p);
                    }
                }
                let state_changed = matches!(ev, Event::StateChanged { .. });
                model.apply_event(&ev);
                if state_changed {
                    model.state = rt.state();
                    remember_choice(&mut model, &mut remembered);
                }
            }
            // Render at ~30fps (also drives the busy spinner).
            if last_render.elapsed() >= Duration::from_millis(33) {
                let provider = rt.provider_kind();
                let spinner = if rt.is_busy() {
                    spinner_frame(clock.elapsed())
                } else {
                    ' '
                };
                terminal
                    .draw(|f| {
                        draw(
                            f,
                            &mut model,
                            theme,
                            &input,
                            &picker,
                            rt.is_busy(),
                            spinner,
                            provider.name,
                        )
                    })
                    .map_err(|e| e.to_string())?;
                last_render = Instant::now();
            }
            // Poll for a key (short timeout keeps the spinner/event drain live).
            if event::poll(Duration::from_millis(33)).map_err(|e| e.to_string())? {
                match event::read().map_err(|e| e.to_string())? {
                    TermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                        // Ctrl-Y copies the transcript selection via OSC 52.
                        // Ctrl-C stays cancel/exit, so copy gets its
                        // own chord.
                        if key.code == KeyCode::Char('y')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            match selection_text(&model) {
                                Some(text) => {
                                    let chars = text.chars().count();
                                    match osc52_copy(terminal.backend_mut(), &text) {
                                        Ok(()) => {
                                            model.clear_selection();
                                            model.push_notice(&format!("copied {chars} char(s)"));
                                        }
                                        Err(e) => {
                                            model.push_notice(&format!("could not copy: {e}"))
                                        }
                                    }
                                }
                                None => model.push_notice(
                                    "nothing selected — drag over the transcript, then Ctrl-Y",
                                ),
                            }
                        } else {
                            handle_key(
                                rt,
                                &mut model,
                                &mut input,
                                &mut picker,
                                &mut login_pending,
                                &mut should_exit,
                                session_root,
                                key.code,
                                key.modifiers,
                            );
                            refresh_completion(&mut model, &input, &picker, login_pending);
                        }
                    }
                    // Bracketed paste: insert the whole pasted text at the
                    // caret (so Ctrl+Shift+V works for keys / long inputs);
                    // CR/LF is flattened because the editor is single-line.
                    TermEvent::Paste(text) => {
                        input.insert(&text);
                        refresh_completion(&mut model, &input, &picker, login_pending);
                    }
                    // Mouse: wheel scrolls the transcript (3 rows per notch);
                    // a left-button drag selects text for Ctrl-Y copy.
                    TermEvent::Mouse(me) => match me.kind {
                        MouseEventKind::ScrollUp => {
                            model.clear_selection();
                            model.scroll.scroll_by(-3);
                        }
                        MouseEventKind::ScrollDown => {
                            model.clear_selection();
                            model.scroll.scroll_by(3);
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            model.selection_start(me.column, me.row);
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            model.selection_drag(me.column, me.row);
                        }
                        MouseEventKind::Up(MouseButton::Left) => model.selection_end(),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        Ok(0)
    })();

    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();
    // Auto-save at session end *after* the terminal is restored,
    // so its stderr output lands on the normal screen. Failures are warnings
    // only — quitting must never be blocked by persistence.
    crate::auto_save(rt, session_root);
    rt.shutdown();
    result
}

/// A modal picker overlay (model / effort / provider / session selection).
enum Picker {
    Effort {
        selected: usize,
    },
    /// Model choices fetched from the provider, or handed over by the provider
    /// itself when it rejected the configured model.
    Model {
        selected: usize,
        models: Vec<String>,
        /// The model the provider rejected, when that rejection is why the
        /// picker opened (as opposed to the user typing `/model`). The title
        /// then says so, since the overlay appeared uninvited.
        rejected_model: Option<String>,
    },
    /// Provider choices from the registry.
    Provider {
        selected: usize,
    },
    /// Saved sessions for `/resume`, newest first, with titles derived from
    /// each session's first user message.
    Session {
        selected: usize,
        sessions: Vec<daedalus_core::session::SessionSummary>,
    },
}

/// Route one key press. Pure decisions delegated to the model where possible;
/// runtime calls happen here (the shell owns the AgentRuntime handle).
#[allow(clippy::too_many_arguments)] // shell glue: one call site
fn handle_key(
    rt: &AgentRuntime,
    model: &mut UiModel,
    input: &mut InputEditor,
    picker: &mut Option<Picker>,
    login_pending: &mut bool,
    should_exit: &mut bool,
    session_root: &Path,
    code: KeyCode,
    modifiers: KeyModifiers,
) {
    if let Some(p) = picker {
        // Picker navigation consumes keys. Copy the current selection out so
        // we do not hold a borrow across the mutations below.
        let max = match p {
            Picker::Effort { .. } => EFFORT_CHOICES.len(),
            Picker::Model { models, .. } => models.len(),
            Picker::Provider { .. } => PROVIDERS.len(),
            Picker::Session { sessions, .. } => sessions.len(),
        };
        match code {
            // Ctrl-C cancels the overlay (like Esc); a second Ctrl-C at the
            // prompt then exits. Without this the picker would swallow it.
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => *picker = None,
            KeyCode::Esc => *picker = None,
            KeyCode::Down | KeyCode::Char('j') => match p {
                Picker::Effort { selected } => {
                    *selected = (*selected + 1).min(max - 1);
                }
                Picker::Model { selected, .. } => {
                    *selected = (*selected + 1).min(max.saturating_sub(1));
                }
                Picker::Provider { selected } => {
                    *selected = (*selected + 1).min(max.saturating_sub(1));
                }
                Picker::Session { selected, .. } => {
                    *selected = (*selected + 1).min(max.saturating_sub(1));
                }
            },
            KeyCode::Up | KeyCode::Char('k') => match p {
                Picker::Effort { selected } => {
                    *selected = selected.saturating_sub(1);
                }
                Picker::Model { selected, .. } => {
                    *selected = selected.saturating_sub(1);
                }
                Picker::Provider { selected } => {
                    *selected = selected.saturating_sub(1);
                }
                Picker::Session { selected, .. } => {
                    *selected = selected.saturating_sub(1);
                }
            },
            KeyCode::Enter => match p {
                Picker::Effort { selected } => {
                    let effort = EFFORT_CHOICES[*selected];
                    rt.set_effort(effort);
                    model.state.effort = effort;
                    *picker = None;
                }
                Picker::Model {
                    selected, models, ..
                } => {
                    if let Some(name) = models.get(*selected) {
                        rt.set_model(name);
                        model.state.model = name.clone();
                    }
                    *picker = None;
                }
                Picker::Provider { selected } => {
                    if let Some(info) = PROVIDERS.get(*selected) {
                        rt.set_provider(info.name);
                        // The new provider lists its models; prompt for one.
                        model.model_fetch_pending = true;
                    }
                    *picker = None;
                }
                Picker::Session { selected, sessions } => {
                    // Clone the choice before clearing the overlay so the
                    // borrow on `picker` ends before the assignment.
                    let choice = sessions.get(*selected).cloned();
                    *picker = None;
                    if let Some(summary) = choice {
                        match daedalus_core::session::load_at(&summary.path) {
                            Ok(history) => {
                                model.load_history(&history);
                                rt.replace_history(history);
                            }
                            Err(e) => model.push_notice(&format!("could not resume: {e}")),
                        }
                    }
                }
            },
            _ => {}
        }
        return;
    }

    // Slash-command completion: only navigation/accept keys are
    // intercepted, so typing still edits the input (which the shell then
    // re-filters). Up/Down no longer scroll the transcript while it is open.
    if model.completion.is_some() {
        let (sel, len, idx) = {
            let c = model.completion.as_ref().unwrap();
            (
                c.selected,
                c.matches.len(),
                c.matches.get(c.selected).copied(),
            )
        };
        match code {
            KeyCode::Up => {
                model.completion.as_mut().unwrap().selected = sel.saturating_sub(1);
                return;
            }
            KeyCode::Down => {
                if sel + 1 < len {
                    model.completion.as_mut().unwrap().selected = sel + 1;
                }
                return;
            }
            KeyCode::Tab => {
                if let Some(i) = idx {
                    let text = accept_completion(&COMMANDS[i]);
                    input.set_text(text);
                }
                model.completion = None;
                return;
            }
            KeyCode::Esc => {
                model.completion = None;
                return;
            }
            KeyCode::Enter => {
                // Enter accepts a partial token, but submits an exact command.
                if let Some(i) = idx {
                    if input.text() != format!("/{}", COMMANDS[i].name) {
                        let text = accept_completion(&COMMANDS[i]);
                        input.set_text(text);
                        model.completion = None;
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    // Transcript scrollback: PageUp/PageDown page, Up/Down by one line, and
    // Ctrl+Home/Ctrl+End jump to the oldest/newest content. Scrolling up pauses
    // follow; returning to the bottom resumes it.
    match code {
        KeyCode::PageUp => {
            model.scroll.page_up();
            return;
        }
        KeyCode::PageDown => {
            model.scroll.page_down();
            return;
        }
        KeyCode::Up => {
            model.scroll.scroll_by(-1);
            return;
        }
        KeyCode::Down => {
            model.scroll.scroll_by(1);
            return;
        }
        KeyCode::Home if modifiers.contains(KeyModifiers::CONTROL) => {
            model.scroll.jump_to_top();
            return;
        }
        KeyCode::End if modifiers.contains(KeyModifiers::CONTROL) => {
            model.scroll.follow_tail();
            return;
        }
        _ => {}
    }

    match code {
        KeyCode::Esc | KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
            if rt.is_busy() {
                rt.abort();
                model.push_notice("(interrupted)");
            } else {
                *should_exit = true;
            }
        }
        KeyCode::Enter => {
            let line = input.take();
            if submit_line(rt, model, picker, session_root, login_pending, &line) {
                *should_exit = true;
            }
        }
        // Line editing at the caret: movement, insert, delete,
        // word/line kill. Insert-style; no vim modal editing.
        KeyCode::Left => input.left(),
        KeyCode::Right => input.right(),
        KeyCode::Home => input.home(),
        KeyCode::End => input.end(),
        KeyCode::Backspace => input.backspace(),
        KeyCode::Delete => input.delete(),
        KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => input.home(),
        KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => input.end(),
        KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => input.delete(),
        // Ctrl+O expands/collapses every tool result.
        KeyCode::Char('o') if modifiers.contains(KeyModifiers::CONTROL) => model.toggle_verbose(),
        KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => input.kill_prev_word(),
        KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => input.kill_to_start(),
        KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => input.kill_to_end(),
        KeyCode::Char(c) => input.insert_char(c),
        _ => {}
    }
}

/// Handle a submitted line (a message or a slash command). Returns true
/// when the command asks to quit (/exit, /quit, /q).
fn submit_line(
    rt: &AgentRuntime,
    model: &mut UiModel,
    picker: &mut Option<Picker>,
    session_root: &Path,
    login_pending: &mut bool,
    line: &str,
) -> bool {
    // A pending /login reads the next non-slash line as the API key.
    if *login_pending {
        *login_pending = false;
        let key = line.trim();
        if key.is_empty() {
            model.push_notice("login cancelled");
            return false;
        }
        let provider = rt.provider_kind();
        match daedalus_core::credential::store_api_key(provider, key) {
            // The running runtime built its provider at startup, so the stored
            // key applies from the next launch.
            Ok(()) => model.push_notice(&format!(
                "stored API key for {} (restart daedalus to use it)",
                provider.name
            )),
            Err(e) => model.push_notice(&format!("could not store key: {e}")),
        }
        return false;
    }
    match parse_line(line) {
        LineAction::Message(text) => {
            if !text.is_empty() {
                model.push_user(&text);
                if rt.is_busy() {
                    rt.steer(&text); // pi: type-while-running steers
                    model.push_notice("(steer queued)");
                } else {
                    rt.prompt(&text);
                }
            }
            false
        }
        LineAction::Command(cmd) => {
            run_command(rt, model, picker, session_root, login_pending, cmd)
        }
    }
}

/// Execute a slash command. `/model`/`/effort` without an argument open a
/// picker; `/login` arms the next submitted line to be read as an API key.
fn run_command(
    rt: &AgentRuntime,
    model: &mut UiModel,
    picker: &mut Option<Picker>,
    session_root: &Path,
    login_pending: &mut bool,
    cmd: SlashCommand,
) -> bool {
    match cmd {
        SlashCommand::Help => {
            for c in COMMANDS {
                let args = c.args.map(|a| format!(" {a}")).unwrap_or_default();
                model.push_notice(&format!("/{}{} — {}", c.name, args, c.description));
            }
            model.push_notice(
                "keys: Enter send · Esc/Ctrl-C abort (busy) or quit (prompt) · PgUp/PgDn/↑↓ scroll · Ctrl-O tool output · Ctrl-Y copy",
            );
        }
        SlashCommand::Clear => {
            rt.clear();
            model.transcript.clear();
            model.push_notice("conversation cleared");
        }
        SlashCommand::Exit => {
            // Auto-save happens in run_tui after the loop exits.
            return true;
        }
        SlashCommand::Resume => {
            // List every session for this cwd and let the user pick one, like
            // pi's resume picker. Titles are derived
            // from the first user message; sessions store no name of their own.
            match daedalus_core::session::list_sessions(session_root, &rt.workspace_root()) {
                Ok(sessions) if sessions.is_empty() => {
                    model.push_notice("no previous session");
                }
                Ok(sessions) => {
                    *picker = Some(Picker::Session {
                        selected: 0,
                        sessions,
                    });
                }
                Err(e) => model.push_notice(&format!("could not list sessions: {e}")),
            }
        }
        SlashCommand::Login => {
            model.push_notice("/login: type your API key, then Enter");
            *login_pending = true;
        }
        SlashCommand::Model(arg) => {
            if arg.is_empty() {
                // Pick from the provider's live model list; fetch it
                // on first use (the result arrives as `Event::ModelsListed`).
                let models = rt.models();
                if models.is_empty() {
                    rt.refresh_models();
                    model.push_notice("fetching models…");
                    model.model_fetch_pending = true;
                } else {
                    *picker = Some(Picker::Model {
                        selected: 0,
                        models,
                        rejected_model: None,
                    });
                }
            } else {
                rt.set_model(&arg);
                model.state.model = arg;
            }
        }
        SlashCommand::Provider(arg) => {
            if arg.is_empty() {
                *picker = Some(Picker::Provider { selected: 0 });
            } else {
                rt.set_provider(&arg);
                // The new provider lists its models; open the picker when they
                // arrive.
                model.model_fetch_pending = true;
            }
        }
        SlashCommand::Effort(arg) => {
            if arg.is_empty() {
                *picker = Some(Picker::Effort { selected: 0 });
            } else if let Some(e) = Effort::parse(&arg) {
                rt.set_effort(e);
                model.state.effort = e;
            } else {
                model.push_notice(&format!(
                    "unknown effort '{arg}' (off|minimal|low|medium|high)"
                ));
            }
        }
        SlashCommand::Workspace(path) => {
            rt.switch_workspace(&path.to_string_lossy());
            model.state.workspace = path.to_string_lossy().into_owned();
        }
        SlashCommand::Skills => {
            let skills = rt.skills();
            if skills.is_empty() {
                model.push_notice(
                    "no skills found (add <workspace>/.daedalus/skills/<name>.md or \
                     ~/.config/daedalus/skills/<name>.md)",
                );
            } else {
                for s in &skills {
                    model.push_notice(&format!(
                        "{} [{}] — {} ({})",
                        s.name,
                        s.level.label(),
                        s.description,
                        s.path.display()
                    ));
                }
            }
        }
        SlashCommand::Skill(name) => {
            if name.is_empty() {
                model.push_notice("usage: /skill <name> (see /skills)");
            } else {
                match rt.skills().into_iter().find(|s| s.name == name) {
                    Some(skill) => {
                        model.push_user(&format!("/skill {}", skill.name));
                        if rt.is_busy() {
                            rt.steer(&skill.prompt());
                            model.push_notice("(skill queued as steer)");
                        } else {
                            rt.prompt(&skill.prompt());
                        }
                    }
                    None => model.push_notice(&format!("unknown skill '{name}' (see /skills)")),
                }
            }
        }
        SlashCommand::Tools => {
            for (name, description) in rt.tool_listing() {
                model.push_notice(&format!("{name} — {description}"));
            }
        }
    }
    false
}

/// Map a theme style spec onto a ratatui style.
pub fn style(spec: StyleSpec) -> Style {
    Style::default()
        .fg(color(spec.fg))
        .bg(color(spec.bg))
        .add_modifier(modifiers(spec.modifiers))
}

fn color(c: ThemeColor) -> ratatui::style::Color {
    match c {
        ThemeColor::Default => ratatui::style::Color::Reset,
        ThemeColor::Indexed(n) => ratatui::style::Color::Indexed(n),
        ThemeColor::Rgb(r, g, b) => ratatui::style::Color::Rgb(r, g, b),
    }
}

fn modifiers(m: Modifiers) -> Modifier {
    let mut out = Modifier::empty();
    if m.bold {
        out |= Modifier::BOLD;
    }
    if m.dim {
        out |= Modifier::DIM;
    }
    if m.italic {
        out |= Modifier::ITALIC;
    }
    if m.underlined {
        out |= Modifier::UNDERLINED;
    }
    if m.reversed {
        out |= Modifier::REVERSED;
    }
    if m.crossed_out {
        out |= Modifier::CROSSED_OUT;
    }
    out
}

/// Markdown styles pulled from the active theme.
fn markdown_style(theme: &Theme) -> crate::markdown::MarkdownStyle {
    crate::markdown::MarkdownStyle {
        text: style(theme.token(Token::Assistant)),
        heading: style(theme.token(Token::MdHeading)),
        code: style(theme.token(Token::MdCode)),
        code_block: style(theme.token(Token::MdCodeBlock)),
        link: style(theme.token(Token::MdLink)),
        quote: style(theme.token(Token::MdQuote)),
        bullet: style(theme.token(Token::MdBullet)),
    }
}

/// Tool results render collapsed to a preview by default: at most
/// this many lines...
const TOOL_PREVIEW_LINES: usize = 5;
/// ...and at most this many characters, before the "Ctrl+O to expand" hint.
const TOOL_PREVIEW_CHARS: usize = 600;

/// Normalize a tool result for display: CRLF to LF and drop trailing newlines
/// (bash output almost always ends with one, which would otherwise render an
/// empty line under every result).
fn normalize_tool_output(content: &str) -> String {
    content
        .replace("\r\n", "\n")
        .trim_end_matches('\n')
        .to_string()
}

/// A collapsed tool result: the preview text plus what was withheld.
struct ToolPreview {
    text: String,
    /// Trailing source lines not shown.
    hidden_lines: usize,
    /// True when the character cap cut `text` short.
    char_truncated: bool,
}

impl ToolPreview {
    /// The "Ctrl+O to expand" hint, or `None` when nothing was hidden.
    fn hint(&self) -> Option<String> {
        if self.hidden_lines == 0 && !self.char_truncated {
            return None;
        }
        let mut what = String::new();
        if self.hidden_lines > 0 {
            what.push_str(&format!("{} more line(s)", self.hidden_lines));
        }
        if self.char_truncated {
            if !what.is_empty() {
                what.push_str(" and ");
            }
            what.push_str("more text");
        }
        Some(format!("    … {what} — Ctrl+O to expand"))
    }
}

/// Collapse a tool result to a preview, tracking both the lines dropped and
/// whether the character cap cut the text short (mirroring
/// ICARUS-113). Either kind of hiding must surface the expand hint.
fn tool_preview(content: &str) -> ToolPreview {
    let norm = normalize_tool_output(content);
    let lines: Vec<&str> = norm.split('\n').collect();
    let shown = lines.len().min(TOOL_PREVIEW_LINES);
    let mut text = lines[..shown].join("\n");
    let char_truncated = text.chars().count() > TOOL_PREVIEW_CHARS;
    if char_truncated {
        text = text.chars().take(TOOL_PREVIEW_CHARS).collect();
    }
    ToolPreview {
        text,
        hidden_lines: lines.len().saturating_sub(shown),
        char_truncated,
    }
}

/// Lay a tool result out as indented transcript lines: the first gets `⎿`, the
/// rest align under it. One `TLine` per source line, so wrapping stays sane.
fn tool_output_lines(content: &str, spec: StyleSpec) -> Vec<TLine<'static>> {
    let norm = normalize_tool_output(content);
    if norm.is_empty() {
        return Vec::new();
    }
    norm.split('\n')
        .enumerate()
        .map(|(i, line)| {
            let prefix = if i == 0 { "  ⎿ " } else { "    " };
            TLine::from(Span::styled(format!("{prefix}{line}"), style(spec)))
        })
        .collect()
}

/// Build the ratatui lines for the flushed transcript, rendering assistant
/// messages as markdown and tool output under its call line,
/// collapsed unless `verbose` (Ctrl+O).
fn transcript_lines(
    transcript: &[TranscriptLine],
    theme: &Theme,
    width: u16,
    verbose: bool,
) -> Vec<TLine<'static>> {
    let md = markdown_style(theme);
    let mut out: Vec<TLine<'static>> = Vec::new();
    for line in transcript {
        match line {
            TranscriptLine::User(t) => out.push(TLine::from(Span::styled(
                format!("▶ {t}"),
                style(theme.token(Token::User)),
            ))),
            TranscriptLine::Assistant(t) => out.extend(crate::markdown::render(t, width, &md)),
            TranscriptLine::Thinking(t) => out.push(TLine::from(Span::styled(
                format!("  {t}"),
                style(theme.token(Token::Thinking)),
            ))),
            TranscriptLine::Tool(t) => out.push(TLine::from(Span::styled(
                format!("  {t}"),
                style(theme.token(Token::Tool)),
            ))),
            TranscriptLine::ToolResult { name, ok, output } => {
                let marker = if *ok { "✓" } else { "✗" };
                let token = if *ok { Token::ToolOk } else { Token::ToolErr };
                out.push(TLine::from(Span::styled(
                    format!("  {marker} {name}"),
                    style(theme.token(token)),
                )));
                if let Some(content) = output {
                    let spec = theme.token(token);
                    if verbose {
                        out.extend(tool_output_lines(content, spec));
                    } else {
                        let preview = tool_preview(content);
                        out.extend(tool_output_lines(&preview.text, spec));
                        if let Some(hint) = preview.hint() {
                            out.push(TLine::from(Span::styled(
                                hint,
                                style(theme.token(Token::Notice)),
                            )));
                        }
                    }
                }
            }
            TranscriptLine::Notice(t) => out.push(TLine::from(Span::styled(
                format!("• {t}"),
                style(theme.token(Token::Notice)),
            ))),
        }
    }
    out
}

/// Render a frame: transcript on top, input editor (with a visible caret) at
/// the bottom, footer with an animated spinner while busy. The model/effort
/// picker renders as a centered overlay sized to its choices.
#[allow(clippy::too_many_arguments)] // shell glue: one call site
fn draw(
    f: &mut Frame,
    model: &mut UiModel,
    theme: &Theme,
    input: &InputEditor,
    picker: &Option<Picker>,
    busy: bool,
    spinner: char,
    provider: &str,
) {
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

    let area = f.area();
    let cmd_h = model
        .completion
        .as_ref()
        .map(|c| (c.matches.len().min(6) + 2) as u16)
        .unwrap_or(0);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(cmd_h),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    // Transcript: markdown-rendered and cached by (revision, width), plus the
    // live streaming answer as a trailing block.
    let inner_w = chunks[0].width.saturating_sub(2);
    if !matches!(&model.md_cache, Some((r, w, _)) if *r == model.revision && *w == chunks[0].width)
    {
        let rendered = transcript_lines(&model.transcript, theme, inner_w, model.verbose);
        model.md_cache = Some((model.revision, chunks[0].width, rendered));
    }
    let mut lines: Vec<TLine> = model.md_cache.as_ref().unwrap().2.clone();
    if !model.assistant_buf.is_empty() {
        lines.extend(crate::markdown::render(
            &model.assistant_buf,
            inner_w,
            &markdown_style(theme),
        ));
    }
    // `line_count` returns the wrapped height including the block's border
    // rows, so the full chunk height is the matching viewport. Scrolling by the
    // logical line count instead left the newest (wrapped) lines off-screen.
    let transcript = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style(theme.token(Token::Border)))
                .title(" daedalus ")
                .title_style(style(theme.token(Token::Title))),
        )
        .wrap(Wrap { trim: false });
    let total = transcript.line_count(inner_w);
    let top = model.scroll.resolve(total, chunks[0].height as usize);
    let transcript = transcript.scroll(((top.min(u16::MAX as usize)) as u16, 0));
    f.render_widget(transcript, chunks[0]);

    // Remember the transcript viewport for mouse selection, then paint the
    // active selection over the rendered cells (so the copy matches the
    // screen, wrapped rows included).
    let inner = chunks[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    model.transcript_area = inner;
    model.transcript_rows = viewport_rows(f.buffer_mut(), inner);
    if let Some(sel) = model.selection {
        paint_selection(f.buffer_mut(), inner, sel, theme);
    }

    // Slash-command completion dropdown.
    if let Some(c) = &model.completion {
        draw_completions(f, chunks[1], c, theme);
    }

    // Input editor (or picker overlay).
    match picker {
        Some(Picker::Effort { selected }) => {
            let title = " effort — ↑/↓ · Enter apply · Esc cancel ";
            let names: Vec<String> = EFFORT_CHOICES
                .iter()
                .map(|e| e.name().to_string())
                .collect();
            draw_picker(f, &names, *selected, title, theme);
        }
        Some(Picker::Model {
            selected,
            models,
            rejected_model,
        }) => {
            // When the picker opened because the provider turned the current
            // model down, say which one in the title.
            let title = match rejected_model {
                Some(m) => format!(" model '{m}' rejected — ↑/↓ · Enter apply · Esc cancel "),
                None => " model — ↑/↓ · Enter apply · Esc cancel ".to_string(),
            };
            draw_picker(f, models, *selected, &title, theme);
        }
        Some(Picker::Provider { selected }) => {
            let names: Vec<String> = PROVIDERS.iter().map(|p| p.name.to_string()).collect();
            draw_picker(
                f,
                &names,
                *selected,
                " provider — ↑/↓ · Enter apply · Esc cancel ",
                theme,
            );
        }
        Some(Picker::Session { selected, sessions }) => {
            let titles: Vec<String> = sessions.iter().map(|s| s.title.clone()).collect();
            draw_picker(
                f,
                &titles,
                *selected,
                " resume — ↑/↓ · Enter open · Esc cancel ",
                theme,
            );
        }
        None => {
            // Show the window of the text that contains the caret so long
            // lines stay editable, and place the terminal cursor
            // on the caret so typing position is visible.
            let inner_w = chunks[2].width.saturating_sub(2) as usize;
            let (window, cursor_col) = input.window(inner_w);
            let editor = Paragraph::new(window)
                .style(style(theme.token(Token::Input)))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(style(theme.token(Token::Border)))
                        .title(" input ")
                        .title_style(style(theme.token(Token::Title))),
                );
            f.render_widget(editor, chunks[2]);
            let x = chunks[2].x + 1 + cursor_col as u16;
            f.set_cursor_position(Position {
                x: x.min(chunks[2].x + chunks[2].width.saturating_sub(1)),
                y: chunks[2].y + 1,
            });
        }
    }

    // Footer/status.
    let usage = model
        .usage
        .map(|t| t.to_string())
        .unwrap_or_else(|| "?".to_string());
    let status = format!(
        " {} ({}) | effort {} | {} | {} | {} tokens{}",
        model.state.model,
        provider,
        model.state.effort.name(),
        model.state.workspace,
        if busy { "busy" } else { "idle" },
        usage,
        if model.settled { " · settled" } else { "" },
    );
    let footer = Paragraph::new(TLine::from(vec![
        Span::styled(spinner.to_string(), style(theme.token(Token::Spinner))),
        Span::styled(status, style(theme.token(Token::Text))),
    ]));
    f.render_widget(footer, chunks[3]);
}

/// Render the slash-command completion dropdown: a bordered list directly above
/// the input, scrolled to keep the selection visible.
fn draw_completions(f: &mut Frame, area: Rect, completion: &Completion, theme: &Theme) {
    use ratatui::widgets::{Block, Borders, Paragraph};
    if area.height == 0 {
        return;
    }
    let viewport = area.height.saturating_sub(2) as usize;
    let offset = picker_offset(completion.selected, viewport);
    let lines: Vec<TLine> = completion
        .matches
        .iter()
        .enumerate()
        .skip(offset)
        .take(viewport)
        .map(|(i, &ci)| {
            let spec = &COMMANDS[ci];
            let args = spec.args.map(|a| format!(" {a}")).unwrap_or_default();
            let marker = if i == completion.selected {
                "❯ "
            } else {
                "  "
            };
            let text = format!("{marker}/{}{} — {}", spec.name, args, spec.description);
            let line = TLine::from(Span::raw(text));
            if i == completion.selected {
                line.style(style(theme.token(Token::Selection)))
            } else {
                line.style(style(theme.token(Token::Text)))
            }
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style(theme.token(Token::Border)))
        .title(" commands ")
        .title_style(style(theme.token(Token::Title)));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// Render the picker overlay: a centered window sized to the choices (not the
/// 1-row input slot), cleared behind, with a viewport that keeps the selected
/// row visible and the current selection highlighted.
fn draw_picker(f: &mut Frame, items: &[String], selected: usize, title: &str, theme: &Theme) {
    use ratatui::widgets::{Block, Borders, Paragraph};

    let item_w = items.iter().map(|s| s.width()).max().unwrap_or(0) as u16;
    let inner_w = (item_w + 2).max(UnicodeWidthStr::width(title) as u16);
    let area = centered_rect(inner_w + 2, items.len() as u16 + 2, f.area());
    f.render_widget(Clear, area);

    let viewport = area.height.saturating_sub(2) as usize;
    let offset = picker_offset(selected, viewport);
    let lines: Vec<TLine> = items
        .iter()
        .enumerate()
        .skip(offset)
        .take(viewport)
        .map(|(i, c)| {
            let marker = if i == selected { "❯ " } else { "  " };
            let line = TLine::from(Span::raw(format!("{marker}{c}")));
            if i == selected {
                line.style(style(theme.token(Token::Selection)))
            } else {
                line.style(style(theme.token(Token::Text)))
            }
        })
        .collect();
    let picker = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(style(theme.token(Token::Border)))
            .title(title)
            .title_style(style(theme.token(Token::Title))),
    );
    f.render_widget(picker, area);
}

/// A mouse text selection over the transcript viewport. Coordinates are
/// absolute screen cells `(column, row)`; `anchor` is where the drag began and
/// `head` where it is now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub anchor: (u16, u16),
    pub head: (u16, u16),
}

impl Selection {
    /// The selection ends ordered top-left to bottom-right (row first).
    pub fn ordered(&self) -> ((u16, u16), (u16, u16)) {
        if (self.anchor.1, self.anchor.0) <= (self.head.1, self.head.0) {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }

    /// True when the selection covers no cell (a plain click).
    pub fn is_empty(&self) -> bool {
        self.anchor == self.head
    }
}

/// The visible text of each row in `area`, read back from the rendered buffer,
/// so a selection copies exactly what is on screen (wrapping included).
fn viewport_rows(buf: &ratatui::buffer::Buffer, area: Rect) -> Vec<String> {
    (area.top()..area.bottom())
        .map(|y| {
            let mut row = String::new();
            for x in area.left()..area.right() {
                if let Some(cell) = buf.cell(Position { x, y }) {
                    row.push_str(cell.symbol());
                }
            }
            row
        })
        .collect()
}

/// Overlay the selection style on the selected cells of the transcript view.
fn paint_selection(buf: &mut ratatui::buffer::Buffer, area: Rect, sel: Selection, theme: &Theme) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let ((sc, sr), (ec, er)) = sel.ordered();
    let last_row = area.bottom() - 1;
    let last_col = area.right() - 1;
    let sr = sr.clamp(area.top(), last_row);
    let er = er.clamp(area.top(), last_row);
    // Honor a themed selection color; both presets only set `bold` and leave
    // fg/bg at the terminal default, in which case reverse video is the
    // portable highlight (and keeps the text's own colors). `style()` must not
    // be used here: it force-sets fg/bg to Reset, which would erase them.
    let spec = theme.token(Token::Selection);
    let mut style = Style::default().add_modifier(modifiers(spec.modifiers));
    if spec.fg != ThemeColor::Default {
        style = style.fg(color(spec.fg));
    }
    if spec.bg != ThemeColor::Default {
        style = style.bg(color(spec.bg));
    } else {
        style = style.add_modifier(Modifier::REVERSED);
    }
    for y in sr..=er {
        let left = if y == sr {
            sc.clamp(area.left(), last_col)
        } else {
            area.left()
        };
        let right = if y == er {
            ec.clamp(area.left(), last_col)
        } else {
            last_col
        };
        for x in left..=right {
            if let Some(cell) = buf.cell_mut(Position { x, y }) {
                cell.set_style(style);
            }
        }
    }
}

/// Char index within `chars` for a terminal column, honoring wide characters
/// (a CJK glyph occupies two columns). Columns past the end clamp to the end.
fn char_index_at_col(chars: &[char], col: usize) -> usize {
    let mut width = 0usize;
    for (i, ch) in chars.iter().enumerate() {
        if width >= col {
            return i;
        }
        width += UnicodeWidthChar::width(*ch).unwrap_or(0);
    }
    chars.len()
}

/// The text of the current selection, read from the last drawn transcript
/// viewport. `None` when nothing is selected. The end cell is inclusive.
fn selection_text(model: &UiModel) -> Option<String> {
    let sel = model.selection?;
    if sel.is_empty() {
        return None;
    }
    let area = model.transcript_area;
    let rows = &model.transcript_rows;
    if rows.is_empty() {
        return None;
    }
    let ((sc, sr), (ec, er)) = sel.ordered();
    let first = area.y;
    let last = area.y + rows.len() as u16 - 1;
    let sr = sr.clamp(first, last);
    let er = er.clamp(first, last);
    let col = |c: u16| c.saturating_sub(area.x) as usize;
    let mut out = String::new();
    for row in sr..=er {
        let chars: Vec<char> = rows[(row - first) as usize].chars().collect();
        let from = if row == sr {
            char_index_at_col(&chars, col(sc))
        } else {
            0
        };
        let to = if row == er {
            char_index_at_col(&chars, col(ec).saturating_add(1))
        } else {
            chars.len()
        };
        if from < to {
            out.extend(&chars[from..to]);
        }
        if row < er {
            out.push('\n');
        }
    }
    let trimmed = out.trim_end();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Copy `text` to the terminal's clipboard with OSC 52. Most terminals support
/// it (inside tmux, set `set-clipboard on`); no external tool is needed, so it
/// also works over SSH.
fn osc52_copy(out: &mut impl std::io::Write, text: &str) -> std::io::Result<()> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = text.as_bytes();
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        encoded.push(ALPHABET[(n >> 18) as usize & 63] as char);
        encoded.push(ALPHABET[(n >> 12) as usize & 63] as char);
        encoded.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    write!(out, "\x1b]52;c;{encoded}\x07")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daedalus_core::config::Config;
    use daedalus_core::provider::fake::FakeProvider;
    use daedalus_core::tools::resolver::ToolSet;
    use daedalus_core::workspace::Workspace;

    /// A runtime backed by the fake provider so `handle_key` (picker routing,
    /// Ctrl-C semantics) can be exercised without a terminal.
    fn test_rt() -> AgentRuntime {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(FakeProvider::new(vec![])),
            ToolSet::new(1000),
            ws,
        );
        rt
    }

    fn ctrl_c() -> (KeyCode, KeyModifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    #[test]
    fn ctrl_c_while_picker_open_cancels_the_picker_without_exiting() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut input = InputEditor::default();
        let mut picker = Some(Picker::Effort { selected: 2 });
        let mut login_pending = false;
        let mut should_exit = false;
        let (code, mods) = ctrl_c();
        handle_key(
            &rt,
            &mut model,
            &mut input,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            code,
            mods,
        );
        assert!(picker.is_none(), "picker must be cancelled");
        assert!(!should_exit, "the first Ctrl-C must not quit");
    }

    #[test]
    fn ctrl_c_at_an_idle_prompt_quits() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut input = InputEditor::default();
        let mut picker = None;
        let mut login_pending = false;
        let mut should_exit = false;
        let (code, mods) = ctrl_c();
        handle_key(
            &rt,
            &mut model,
            &mut input,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            code,
            mods,
        );
        assert!(should_exit);
    }

    #[test]
    fn plain_lines_are_messages() {
        assert_eq!(
            parse_line("  hello world  "),
            LineAction::Message("hello world".to_string())
        );
        assert_eq!(parse_line(""), LineAction::Message("".to_string()));
    }

    #[test]
    fn slash_commands_parse() {
        assert_eq!(
            parse_line("/resume"),
            LineAction::Command(SlashCommand::Resume)
        );
        assert_eq!(
            parse_line("/clear"),
            LineAction::Command(SlashCommand::Clear)
        );
        assert_eq!(parse_line("/help"), LineAction::Command(SlashCommand::Help));
        assert_eq!(parse_line("/exit"), LineAction::Command(SlashCommand::Exit));
        assert_eq!(parse_line("/quit"), LineAction::Command(SlashCommand::Exit));
        assert_eq!(
            parse_line("/login"),
            LineAction::Command(SlashCommand::Login)
        );
    }

    #[test]
    fn slash_commands_with_args_parse() {
        assert_eq!(
            parse_line("/model gpt-4o"),
            LineAction::Command(SlashCommand::Model("gpt-4o".to_string()))
        );
        assert_eq!(
            parse_line("/effort high"),
            LineAction::Command(SlashCommand::Effort("high".to_string()))
        );
        assert_eq!(
            parse_line("/workspace /tmp/proj"),
            LineAction::Command(SlashCommand::Workspace(PathBuf::from("/tmp/proj")))
        );
    }

    #[test]
    fn skill_commands_parse() {
        assert_eq!(
            parse_line("/skills"),
            LineAction::Command(SlashCommand::Skills)
        );
        assert_eq!(
            parse_line("/skill review"),
            LineAction::Command(SlashCommand::Skill("review".to_string()))
        );
        // A bare `/skill` still routes to the command; run_command prints usage.
        assert_eq!(
            parse_line("/skill"),
            LineAction::Command(SlashCommand::Skill(String::new()))
        );
        assert_eq!(
            parse_line("/tools"),
            LineAction::Command(SlashCommand::Tools)
        );
    }

    /// `/tools` lists the registered toolset (the built-ins
    /// when no MCP servers are configured).
    #[test]
    fn tools_command_lists_builtins() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut picker = None;
        let mut login_pending = false;
        run_command(
            &rt,
            &mut model,
            &mut picker,
            Path::new("/tmp"),
            &mut login_pending,
            SlashCommand::Tools,
        );
        for builtin in ["read", "bash", "search", "edit", "write"] {
            assert!(
                model
                    .transcript
                    .iter()
                    .any(|l| matches!(l, TranscriptLine::Notice(n) if n.starts_with(builtin))),
                "missing tool listing for {builtin}"
            );
        }
    }

    /// `/skills` lists discovered skills and `/skill <name>` loads
    /// one as a user message (unknown names get a notice instead).
    #[test]
    fn skill_command_loads_body_and_lists_catalog() {
        let dir = tempfile::tempdir().expect("temp dir");
        let skills_dir = dir.path().join(".daedalus").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(skills_dir.join("demo.md"), "Demo skill.\n\nDo the demo.").unwrap();
        let ws = Workspace::new(dir.path().to_path_buf()).unwrap();
        let cfg = Config {
            workspace: dir.path().to_path_buf(),
            ..Config::defaults(dir.path().to_path_buf())
        };
        let (rt, _rx) = AgentRuntime::new(
            cfg,
            Box::new(FakeProvider::new(vec![])),
            ToolSet::new(1000),
            ws,
        );
        let mut model = UiModel::new(rt.state());
        let mut picker = None;
        let mut login_pending = false;
        let session_root = Path::new("/tmp");

        run_command(
            &rt,
            &mut model,
            &mut picker,
            session_root,
            &mut login_pending,
            SlashCommand::Skills,
        );
        assert!(model
            .transcript
            .iter()
            .any(|l| matches!(l, TranscriptLine::Notice(n) if n.contains("demo") && n.contains("Demo skill."))));

        run_command(
            &rt,
            &mut model,
            &mut picker,
            session_root,
            &mut login_pending,
            SlashCommand::Skill("demo".to_string()),
        );
        assert!(model
            .transcript
            .iter()
            .any(|l| matches!(l, TranscriptLine::User(u) if u == "/skill demo")));

        run_command(
            &rt,
            &mut model,
            &mut picker,
            session_root,
            &mut login_pending,
            SlashCommand::Skill("nope".to_string()),
        );
        assert!(model
            .transcript
            .iter()
            .any(|l| matches!(l, TranscriptLine::Notice(n) if n.contains("unknown skill"))));
    }

    #[test]
    fn unknown_slash_falls_back_to_help() {
        assert_eq!(parse_line("/nope"), LineAction::Command(SlashCommand::Help));
    }

    #[test]
    fn effort_choices_cover_all_levels() {
        let all = [
            Effort::Off,
            Effort::Minimal,
            Effort::Low,
            Effort::Medium,
            Effort::High,
        ];
        assert_eq!(EFFORT_CHOICES, all);
    }

    fn state() -> RuntimeState {
        RuntimeState {
            model: "gpt-4o".into(),
            provider: "openai".into(),
            effort: Effort::Medium,
            workspace: "/ws".into(),
            busy: false,
        }
    }

    #[test]
    fn text_deltas_stream_into_one_assistant_line() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::TextDelta { text: "Hel".into() });
        m.apply_event(&Event::TextDelta { text: "lo".into() });
        assert!(m.assistant_buf == "Hello");
        // A tool start flushes the partial assistant text.
        m.apply_event(&Event::ToolStart {
            name: "bash".into(),
            id: None,
            args: None,
        });
        assert!(m.assistant_buf.is_empty());
        assert_eq!(m.transcript[0], TranscriptLine::Assistant("Hello".into()));
        assert_eq!(m.transcript[1], TranscriptLine::Tool("⚙ bash".into()));
    }

    #[test]
    fn bash_tool_start_shows_the_command() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ToolStart {
            name: "bash".into(),
            id: None,
            args: Some(serde_json::json!({ "command": "ls -la" })),
        });
        assert_eq!(
            m.transcript[0],
            TranscriptLine::Tool("⚙ bash $ ls -la".into())
        );
    }

    #[test]
    fn tool_detail_rules() {
        use serde_json::json;
        assert_eq!(
            tool_detail("bash", Some(&json!({"command": "ls -la"}))).as_deref(),
            Some("$ ls -la")
        );
        // Multi-line input keeps only the first line, with an ellipsis.
        assert_eq!(
            tool_detail("bash", Some(&json!({"command": "echo a\necho b"}))).as_deref(),
            Some("$ echo a …")
        );
        // read/edit/write summarize the path.
        assert_eq!(
            tool_detail("read", Some(&json!({"path": "src/main.rs"}))).as_deref(),
            Some("read src/main.rs")
        );
        assert_eq!(
            tool_detail("write", Some(&json!({"file_path": "a.txt"}))).as_deref(),
            Some("write a.txt")
        );
        // No summary: absent args, missing/non-string keys, unknown tools.
        assert_eq!(tool_detail("bash", None), None);
        assert_eq!(tool_detail("bash", Some(&json!({"timeout": 5}))), None);
        assert_eq!(tool_detail("mcp__x", Some(&json!({"command": "ls"}))), None);
        // Long input is capped at 120 chars, char-boundary safe.
        let long = "é".repeat(200);
        let detail = tool_detail("bash", Some(&json!({"command": long}))).unwrap();
        assert_eq!(detail.chars().count(), 120);
        assert!(detail.ends_with('…'));
    }

    /// Concatenate each rendered line's span text, for assertions.
    fn line_texts(lines: &[TLine<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn tool_preview_caps_lines_and_flags_char_truncation() {
        // Short results pass through unchanged, with no hint.
        let p = tool_preview("one\ntwo");
        assert_eq!(p.text, "one\ntwo");
        assert_eq!(p.hidden_lines, 0);
        assert!(!p.char_truncated);
        assert_eq!(p.hint(), None);
        // A trailing newline (ubiquitous for bash output) is not a blank line.
        assert_eq!(tool_preview("hi\n").text, "hi");
        assert_eq!(tool_preview("hi\n").hidden_lines, 0);
        // Long results keep the first five lines and count the remainder.
        let long = (0..9).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let p = tool_preview(&long);
        assert_eq!(p.text, "0\n1\n2\n3\n4");
        assert_eq!(p.hidden_lines, 4);
        assert_eq!(p.hint().unwrap(), "    … 4 more line(s) — Ctrl+O to expand");
        // CRLF is normalized.
        assert_eq!(tool_preview("a\r\nb").text, "a\nb");
        // One very long line: the character cap stops it on a boundary, but
        // the truncation must still surface a hint.
        let p = tool_preview(&"é".repeat(1000));
        assert_eq!(p.text.chars().count(), 600);
        assert!(p.char_truncated);
        assert_eq!(p.hidden_lines, 0);
        assert_eq!(p.hint().unwrap(), "    … more text — Ctrl+O to expand");
        // Both cuts at once read coherently.
        let p = tool_preview(
            &(0..9)
                .map(|_| "y".repeat(500))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        assert!(p.char_truncated && p.hidden_lines == 4);
        assert_eq!(
            p.hint().unwrap(),
            "    … 4 more line(s) and more text — Ctrl+O to expand"
        );
    }

    #[test]
    fn char_truncated_preview_still_shows_the_expand_hint() {
        // A wide single-line result (e.g. a long path dump) is cut by the
        // character cap; the user must still be told output is hidden.
        let theme = Theme::dark();
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ToolEnd {
            name: "read".into(),
            ok: true,
            error: None,
            output: Some("x".repeat(1000)),
        });
        let lines = line_texts(&transcript_lines(&m.transcript, &theme, 80, m.verbose));
        assert_eq!(lines[0], "  ✓ read");
        assert_eq!(lines[1].chars().count(), 4 + 600);
        assert_eq!(lines[2], "    … more text — Ctrl+O to expand");
    }

    #[test]
    fn trailing_newline_does_not_render_a_blank_line() {
        // Bash output nearly always ends in a newline; it must not add a
        // spurious padded line under the result.
        let theme = Theme::dark();
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ToolEnd {
            name: "bash".into(),
            ok: true,
            error: None,
            output: Some("hi\n".into()),
        });
        assert_eq!(
            line_texts(&transcript_lines(&m.transcript, &theme, 80, m.verbose)),
            vec!["  ✓ bash", "  ⎿ hi"]
        );
    }

    #[test]
    fn ctrl_o_key_toggles_verbose_and_invalidates_the_cache() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut input = InputEditor::default();
        let mut picker = None;
        let mut login_pending = false;
        let mut should_exit = false;
        let base = model.revision;
        handle_key(
            &rt,
            &mut model,
            &mut input,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            KeyCode::Char('o'),
            KeyModifiers::CONTROL,
        );
        assert!(model.verbose, "Ctrl+O must expand tool output");
        assert!(
            model.revision > base,
            "the toggle must invalidate the render cache"
        );
        assert!(!should_exit);
    }

    #[test]
    fn tool_result_renders_a_preview_then_expands_on_ctrl_o() {
        let theme = Theme::dark();
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ToolStart {
            name: "bash".into(),
            id: None,
            args: None,
        });
        m.apply_event(&Event::ToolEnd {
            name: "bash".into(),
            ok: true,
            error: None,
            output: Some("l0\nl1\nl2\nl3\nl4\nl5\nl6\n".into()),
        });
        // Collapsed by default: marker, first five indented lines, then a hint.
        assert_eq!(
            line_texts(&transcript_lines(&m.transcript, &theme, 80, m.verbose)),
            vec![
                "  ⚙ bash",
                "  ✓ bash",
                "  ⎿ l0",
                "    l1",
                "    l2",
                "    l3",
                "    l4",
                "    … 2 more line(s) — Ctrl+O to expand",
            ]
        );
        // Ctrl+O expands every tool result, with no hint.
        let base = m.revision;
        m.toggle_verbose();
        assert!(m.verbose);
        assert!(
            m.revision > base,
            "the toggle must invalidate the render cache"
        );
        let expanded = line_texts(&transcript_lines(&m.transcript, &theme, 80, m.verbose));
        assert_eq!(expanded.len(), 2 + 7);
        assert_eq!(expanded[8], "    l6");
        assert!(expanded.iter().all(|l| !l.contains("Ctrl+O")));
    }

    #[test]
    fn tool_result_without_output_renders_just_the_marker() {
        let theme = Theme::dark();
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ToolEnd {
            name: "bash".into(),
            ok: false,
            error: Some("boom".into()),
            output: None,
        });
        assert_eq!(
            line_texts(&transcript_lines(&m.transcript, &theme, 80, m.verbose)),
            vec!["  ✗ bash"]
        );
    }

    #[test]
    fn state_changed_updates_footer_state() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::StateChanged {
            model: "claude-x".into(),
            effort: Effort::High,
            workspace: "/other".into(),
            provider: "anthropic".into(),
        });
        assert_eq!(m.state.model, "claude-x");
        assert_eq!(m.state.effort, Effort::High);
        assert_eq!(m.state.workspace, "/other");
        assert_eq!(m.state.provider, "anthropic");
    }

    /// A model whose chosen model/provider is written to `path`, plus the
    /// baseline `run_tui` starts it with (the values the run loaded).
    fn persistent(s: RuntimeState, path: &Path) -> (UiModel, Remembered) {
        let mut m = UiModel::new(s);
        m.persist = Some(path.to_path_buf());
        let remembered = Remembered::of(&m.state);
        (m, remembered)
    }

    fn notices(m: &UiModel) -> Vec<String> {
        m.transcript
            .iter()
            .filter_map(|l| match l {
                TranscriptLine::Notice(n) => Some(n.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn an_adopted_model_and_provider_are_remembered_in_the_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "provider = \"openai\"\nmodel = \"gpt-4o\" # mine\n").unwrap();
        let (mut m, mut remembered) = persistent(state(), &path);

        // An unchanged state (every other StateChanged) writes nothing.
        remember_choice(&mut m, &mut remembered);
        assert!(m.transcript.is_empty(), "nothing changed, nothing to say");

        m.state.provider = "deepseek".into();
        m.state.model = "deepseek-chat".into();
        remember_choice(&mut m, &mut remembered);

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("provider = \"deepseek\""), "{text}");
        assert!(text.contains("model = \"deepseek-chat\""), "{text}");
        assert!(
            !text.contains("gpt-4o"),
            "the replaced value is gone:\n{text}"
        );
        assert!(
            text.contains("# mine"),
            "hand-written notes survive:\n{text}"
        );
        let said = notices(&m);
        assert_eq!(said.len(), 1, "one change, one notice: {said:?}");
        assert!(
            said[0].contains("deepseek-chat") && said[0].contains("config.toml"),
            "{said:?}"
        );
    }

    #[test]
    fn a_config_file_is_created_when_there_is_none_yet() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        let (mut m, mut remembered) = persistent(state(), &path);
        m.state.model = "o3".into();
        remember_choice(&mut m, &mut remembered);
        assert!(std::fs::read_to_string(&path).unwrap().contains("o3"));
    }

    #[test]
    fn one_change_is_remembered_once_and_switching_back_counts_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let (mut m, mut remembered) = persistent(state(), &path);
        m.state.model = "o3".into();
        remember_choice(&mut m, &mut remembered);
        // More StateChanged events for the same state must not re-write (or
        // re-notice): only a real change is news.
        remember_choice(&mut m, &mut remembered);
        assert_eq!(notices(&m).len(), 1);
        m.state.model = "gpt-4o".into();
        remember_choice(&mut m, &mut remembered);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("model = \"gpt-4o\""), "{text}");
        assert_eq!(notices(&m).len(), 2);
    }

    #[test]
    fn an_unwritable_config_is_a_notice_not_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        // A directory where the config file should be: the write must fail.
        let path = dir.path().join("config.toml");
        std::fs::create_dir(&path).unwrap();
        let (mut m, mut remembered) = persistent(state(), &path);
        m.state.model = "o3".into();
        remember_choice(&mut m, &mut remembered);
        let said = notices(&m);
        assert_eq!(said.len(), 1);
        assert!(said[0].starts_with("could not save to"), "{said:?}");
    }

    #[test]
    fn without_a_target_nothing_is_written_or_said() {
        let mut m = UiModel::new(state());
        let mut remembered = Remembered::of(&m.state);
        m.state.model = "o3".into();
        m.state.provider = "deepseek".into();
        remember_choice(&mut m, &mut remembered);
        assert!(m.transcript.is_empty());
    }

    #[test]
    fn a_remembered_provider_is_written_by_registry_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let (mut m, mut remembered) = persistent(state(), &path);
        // Providers are matched case-insensitively; the file gets the
        // canonical name so it reloads cleanly.
        m.state.provider = "DeepSeek".into();
        remember_choice(&mut m, &mut remembered);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.trim(), "provider = \"deepseek\"", "{text}");
    }

    #[test]
    fn agent_settled_flushes_final_text() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::TextDelta {
            text: "answer".into(),
        });
        m.apply_event(&Event::AgentSettled {
            text: "answer".into(),
            interrupted: false,
        });
        assert!(m.settled);
        assert_eq!(
            m.transcript.last(),
            Some(&TranscriptLine::Assistant("answer".into()))
        );
    }

    #[test]
    fn agent_settled_does_not_duplicate_streamed_text() {
        // Real providers stream via text_delta, so turn_end flushes the answer
        // before agent_settled arrives; it must not be appended again.
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::TextDelta {
            text: "answer".into(),
        });
        m.apply_event(&Event::TurnEnd {});
        m.apply_event(&Event::AgentSettled {
            text: "answer".into(),
            interrupted: false,
        });
        let copies = m
            .transcript
            .iter()
            .filter(|l| matches!(l, TranscriptLine::Assistant(t) if t == "answer"))
            .count();
        assert_eq!(copies, 1, "transcript: {:?}", m.transcript);
    }

    #[test]
    fn agent_settled_adds_text_when_nothing_streamed() {
        // Providers that do not stream deliver the answer only in agent_settled.
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::TurnEnd {});
        m.apply_event(&Event::AgentSettled {
            text: "answer".into(),
            interrupted: false,
        });
        assert_eq!(
            m.transcript,
            vec![TranscriptLine::Assistant("answer".into())]
        );
    }

    #[test]
    fn thinking_delta_renders_before_assistant_text() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::ThinkingDelta { text: "hmm".into() });
        m.apply_event(&Event::ThinkingDelta {
            text: " more".into(),
        });
        m.apply_event(&Event::TextDelta {
            text: "answer".into(),
        });
        m.apply_event(&Event::TurnEnd {});
        assert_eq!(
            m.transcript,
            vec![
                TranscriptLine::Thinking("hmm more".into()),
                TranscriptLine::Assistant("answer".into()),
            ]
        );
    }

    #[test]
    fn thinking_flushes_on_turn_end_and_error() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ThinkingDelta {
            text: "reason".into(),
        });
        m.apply_event(&Event::TurnEnd {});
        assert_eq!(
            m.transcript,
            vec![TranscriptLine::Thinking("reason".into())]
        );
    }

    #[test]
    fn thinking_splits_into_one_line_per_reasoning_line() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::ThinkingDelta {
            text: "first\nsecond\n".into(),
        });
        m.apply_event(&Event::TurnEnd {});
        assert_eq!(
            m.transcript,
            vec![
                TranscriptLine::Thinking("first".into()),
                TranscriptLine::Thinking("second".into()),
            ]
        );
    }

    #[test]
    fn streaming_deltas_do_not_invalidate_the_render_cache() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        let base = m.revision;
        m.apply_event(&Event::ThinkingDelta { text: "a".into() });
        assert_eq!(
            m.revision, base,
            "buffered thinking is not a transcript line"
        );
        // Flushing the buffer appends a transcript line, which does invalidate.
        m.apply_event(&Event::TextDelta { text: "b".into() });
        assert!(m.revision > base);
    }

    #[test]
    fn transcript_scroll_follows_and_pages() {
        let mut s = TranscriptScroll::default();
        // Following: resolve pins to the newest content.
        assert_eq!(s.resolve(100, 10), 90);
        // Paging up pauses follow and keeps the view put as content grows.
        s.page_up();
        assert_eq!(s.resolve(100, 10), 80);
        assert_eq!(s.resolve(120, 10), 80);
        // Paging down to the bottom resumes following: newer content then
        // pins to the tail again.
        s.page_down();
        s.page_down();
        s.page_down();
        assert_eq!(s.resolve(120, 10), 110);
        assert_eq!(s.resolve(200, 10), 190);
        // Jump to the oldest, then resume the tail.
        s.jump_to_top();
        assert_eq!(s.resolve(120, 10), 0);
        s.follow_tail();
        assert_eq!(s.resolve(120, 10), 110);
        // Content shorter than the viewport never scrolls.
        assert_eq!(TranscriptScroll::default().resolve(3, 10), 0);
    }

    #[test]
    fn error_surfaces_as_notice() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::Error {
            message: "boom".into(),
            kind: None,
        });
        assert_eq!(
            m.transcript.last(),
            Some(&TranscriptLine::Notice("error: boom".into()))
        );
    }

    #[test]
    fn iterations_and_usage_are_tracked() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::TurnStart {});
        m.apply_event(&Event::Usage {
            prompt_tokens: Some(1200),
        });
        assert_eq!(m.iterations, 1);
        assert_eq!(m.usage, Some(1200));
    }

    #[test]
    fn push_user_and_notice_append_lines() {
        let mut m = UiModel::new(state());
        m.push_user("hello");
        m.push_notice("resumed");
        assert_eq!(m.transcript[0], TranscriptLine::User("hello".into()));
        assert_eq!(m.transcript[1], TranscriptLine::Notice("resumed".into()));
    }

    // --- slash-command dropdown + model picker ---

    #[test]
    fn completions_filter_by_prefix() {
        assert_eq!(
            slash_completions("/").unwrap().matches.len(),
            COMMANDS.len()
        );
        let names: Vec<&str> = slash_completions("/sk")
            .unwrap()
            .matches
            .iter()
            .map(|&i| COMMANDS[i].name)
            .collect();
        assert_eq!(names, vec!["skills", "skill"]);
        assert_eq!(slash_completions("/MO").unwrap().matches.len(), 1);
        assert!(slash_completions("/model ").is_none());
        assert!(slash_completions("/zzz").is_none());
        assert!(slash_completions("hello").is_none());
    }

    #[test]
    fn accept_completion_appends_a_space() {
        let spec = COMMANDS.iter().find(|c| c.name == "skill").unwrap();
        assert_eq!(accept_completion(spec), "/skill ");
    }

    #[test]
    fn commands_table_and_parser_agree() {
        for c in COMMANDS {
            for name in std::iter::once(c.name).chain(c.aliases.iter().copied()) {
                match parse_line(&format!("/{name}")) {
                    LineAction::Command(cmd) => assert_eq!(cmd, (c.build)(String::new()), "{name}"),
                    other => panic!("{name} parsed as {other:?}"),
                }
            }
        }
    }

    #[test]
    fn dropdown_navigates_and_accepts() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.completion = slash_completions("/sk");
        let mut editor = InputEditor::new("/sk");
        let mut picker = None;
        let mut login_pending = false;
        let mut should_exit = false;
        handle_key(
            &rt,
            &mut model,
            &mut editor,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            KeyCode::Down,
            KeyModifiers::NONE,
        );
        assert_eq!(model.completion.as_ref().unwrap().selected, 1);
        handle_key(
            &rt,
            &mut model,
            &mut editor,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            KeyCode::Tab,
            KeyModifiers::NONE,
        );
        assert_eq!(editor.text(), "/skill ");
        assert!(model.completion.is_none());
    }

    #[test]
    fn model_picker_enter_applies_the_model() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut editor = InputEditor::default();
        let mut picker = Some(Picker::Model {
            selected: 1,
            models: vec!["a".into(), "b".into()],
            rejected_model: None,
        });
        let mut login_pending = false;
        let mut should_exit = false;
        handle_key(
            &rt,
            &mut model,
            &mut editor,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            KeyCode::Enter,
            KeyModifiers::NONE,
        );
        assert!(picker.is_none());
        assert_eq!(model.state.model, "b");
    }

    #[test]
    fn a_rejected_model_opens_a_picker_of_the_providers_alternatives() {
        let kind = ErrorKind::InvalidModel {
            requested: "bogus".into(),
            supported: vec!["deepseek-flash".into(), "deepseek-v4-pro".into()],
        };
        match model_picker_for_rejection(&Some(kind)) {
            Some(Picker::Model {
                selected,
                models,
                rejected_model,
            }) => {
                assert_eq!(selected, 0);
                assert_eq!(models, vec!["deepseek-flash", "deepseek-v4-pro"]);
                assert_eq!(rejected_model.as_deref(), Some("bogus"));
            }
            other => panic!("expected a model picker, got one: {:?}", other.is_some()),
        }
    }

    #[test]
    fn a_rejection_without_alternatives_or_another_error_opens_nothing() {
        // The provider named no model: the message alone is the remedy, and a
        // zero-row picker would be a dead end.
        let bare = ErrorKind::InvalidModel {
            requested: "bogus".into(),
            supported: Vec::new(),
        };
        assert!(model_picker_for_rejection(&Some(bare)).is_none());
        // Auth and the iteration cap have their own remedies, not a model list.
        assert!(model_picker_for_rejection(&Some(ErrorKind::Auth)).is_none());
        assert!(model_picker_for_rejection(&Some(ErrorKind::IterationCap)).is_none());
        assert!(model_picker_for_rejection(&None).is_none());
    }

    #[test]
    fn the_rejection_picker_names_the_model_it_replaced() {
        use ratatui::backend::TestBackend;
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let theme = Theme::default();
        let input = InputEditor::default();
        let picker = Some(Picker::Model {
            selected: 0,
            models: vec!["deepseek-flash".into()],
            rejected_model: Some("bogus".into()),
        });
        // Wide enough that the title is not clipped.
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|f| draw(f, &mut model, &theme, &input, &picker, false, ' ', "fake"))
            .unwrap();
        // Flatten the screen so the model name and its title can be searched.
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(screen.contains("'bogus' rejected"), "{screen}");
        assert!(screen.contains("deepseek-flash"), "{screen}");
    }

    #[test]
    fn provider_picker_enter_switches() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        let mut editor = InputEditor::default();
        let mut picker = Some(Picker::Provider { selected: 0 });
        let mut login_pending = false;
        let mut should_exit = false;
        handle_key(
            &rt,
            &mut model,
            &mut editor,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            Path::new("/tmp"),
            KeyCode::Enter,
            KeyModifiers::NONE,
        );
        assert!(picker.is_none());
        // The switch triggers a model fetch so the picker opens on results.
        assert!(model.model_fetch_pending);
    }

    #[test]
    fn load_history_repaints_user_assistant_and_tool_traffic() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.load_history(&[
            Message::System("sys".into()),
            Message::User("hello".into()),
            Message::Assistant {
                text: None,
                tool_calls: vec![daedalus_core::provider::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({"command": "echo hi"}),
                }],
            },
            Message::ToolResult {
                tool_call_id: "c1".into(),
                result: "hi\n".into(),
            },
            Message::Assistant {
                text: Some("done".into()),
                tool_calls: vec![],
            },
        ]);
        // The system prompt is not shown; user/assistant text and the resumed
        // tool result (with its call name) are.
        assert_eq!(
            model.transcript,
            vec![
                TranscriptLine::User("hello".into()),
                TranscriptLine::ToolResult {
                    name: "bash".into(),
                    ok: true,
                    output: Some("hi\n".into()),
                },
                TranscriptLine::Assistant("done".into()),
            ]
        );
    }

    #[test]
    fn resumed_tool_result_expands_with_ctrl_o() {
        let theme = Theme::dark();
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.load_history(&[
            Message::Assistant {
                text: None,
                tool_calls: vec![daedalus_core::provider::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({}),
                }],
            },
            Message::ToolResult {
                tool_call_id: "c1".into(),
                result: "l0\nl1\nl2\nl3\nl4\nl5\nl6\n".into(),
            },
        ]);
        // Collapsed on resume, and expandable — the point of the replay.
        let collapsed = line_texts(&transcript_lines(
            &model.transcript,
            &theme,
            80,
            model.verbose,
        ));
        assert_eq!(collapsed[0], "  ✓ bash");
        assert!(collapsed
            .iter()
            .any(|l| l == "    … 2 more line(s) — Ctrl+O to expand"));
        model.toggle_verbose();
        let expanded = line_texts(&transcript_lines(
            &model.transcript,
            &theme,
            80,
            model.verbose,
        ));
        assert_eq!(expanded.len(), 1 + 7);
        assert_eq!(expanded[7], "    l6");
    }

    #[test]
    fn resumed_failed_tool_result_marks_the_error() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.load_history(&[
            Message::Assistant {
                text: None,
                tool_calls: vec![daedalus_core::provider::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({}),
                }],
            },
            Message::ToolResult {
                tool_call_id: "c1".into(),
                result: "tool error: boom".into(),
            },
            // An unknown call id falls back to a generic label rather than
            // panicking.
            Message::ToolResult {
                tool_call_id: "missing".into(),
                result: String::new(),
            },
        ]);
        assert_eq!(
            model.transcript,
            vec![
                TranscriptLine::ToolResult {
                    name: "bash".into(),
                    ok: false,
                    output: Some("tool error: boom".into()),
                },
                TranscriptLine::ToolResult {
                    name: "tool".into(),
                    ok: true,
                    output: None,
                },
            ]
        );
    }

    #[test]
    fn session_picker_enter_loads_and_repaints_a_session() {
        let rt = test_rt();
        let root = tempfile::tempdir().expect("sessions root");
        let cwd = rt.workspace_root();
        let history = vec![
            Message::System("sys".into()),
            Message::User("first question".into()),
            Message::Assistant {
                text: Some("first answer".into()),
                tool_calls: vec![],
            },
        ];
        daedalus_core::session::save_session(root.path(), &cwd, &history).unwrap();
        let sessions = daedalus_core::session::list_sessions(root.path(), &cwd).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "first question");

        let mut model = UiModel::new(rt.state());
        let mut editor = InputEditor::default();
        let mut picker = Some(Picker::Session {
            selected: 0,
            sessions,
        });
        let mut login_pending = false;
        let mut should_exit = false;
        handle_key(
            &rt,
            &mut model,
            &mut editor,
            &mut picker,
            &mut login_pending,
            &mut should_exit,
            root.path(),
            KeyCode::Enter,
            KeyModifiers::NONE,
        );
        assert!(picker.is_none());
        // History is restored into the runtime *and* painted into the
        // transcript (the bug: resume used to leave the view blank).
        assert_eq!(rt.history(), history);
        assert_eq!(
            model.transcript,
            vec![
                TranscriptLine::User("first question".into()),
                TranscriptLine::Assistant("first answer".into()),
            ]
        );
    }

    // --- transcript selection + OSC 52 copy ---

    #[test]
    fn selection_orders_by_row_then_column() {
        let a = Selection {
            anchor: (5, 2),
            head: (1, 1),
        };
        assert_eq!(a.ordered(), ((1, 1), (5, 2)));
        // Same row: the column decides.
        let c = Selection {
            anchor: (9, 3),
            head: (2, 3),
        };
        assert_eq!(c.ordered(), ((2, 3), (9, 3)));
    }

    #[test]
    fn selection_start_outside_the_transcript_clears() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.transcript_area = Rect::new(1, 1, 10, 5);
        model.selection = Some(Selection {
            anchor: (2, 2),
            head: (3, 3),
        });
        model.selection_start(50, 50); // outside the viewport
        assert!(model.selection.is_none());

        model.selection_start(2, 2);
        model.selection_drag(4, 3);
        assert_eq!(model.selection.unwrap().head, (4, 3));
        model.selection_end();
        // A real drag survives release; a click that never moved does not.
        assert!(model.selection.is_some());
        model.selection_start(2, 2);
        model.selection_end();
        assert!(model.selection.is_none());
    }

    #[test]
    fn selection_text_reads_the_highlighted_rows() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.transcript_area = Rect::new(0, 0, 20, 3);
        model.transcript_rows = vec!["hello world".into(), "second line".into(), String::new()];

        // Columns 0..=4 of row 0 (the end cell is inclusive).
        model.selection = Some(Selection {
            anchor: (0, 0),
            head: (4, 0),
        });
        assert_eq!(selection_text(&model).as_deref(), Some("hello"));

        // Across rows, a reversed anchor/head still reads top-left first.
        model.selection = Some(Selection {
            anchor: (3, 1),
            head: (6, 0),
        });
        assert_eq!(selection_text(&model).as_deref(), Some("world\nseco"));

        // Nothing selected, or a zero-width selection, copies nothing.
        model.selection = None;
        assert_eq!(selection_text(&model), None);
        model.selection = Some(Selection {
            anchor: (3, 0),
            head: (3, 0),
        });
        assert_eq!(selection_text(&model), None);
    }

    #[test]
    fn selection_text_handles_wide_characters() {
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.transcript_area = Rect::new(0, 0, 10, 1);
        model.transcript_rows = vec!["a字b".into()];
        // Columns 1..=2 are the two cells of the wide glyph.
        model.selection = Some(Selection {
            anchor: (1, 0),
            head: (2, 0),
        });
        assert_eq!(selection_text(&model).as_deref(), Some("字"));
    }

    #[test]
    fn osc52_copy_frames_base64_payload() {
        let mut out = Vec::new();
        osc52_copy(&mut out, "hello").unwrap();
        assert_eq!(out, b"\x1b]52;c;aGVsbG8=\x07");
        // 2-byte tail pads with one '='.
        let mut out = Vec::new();
        osc52_copy(&mut out, "hi").unwrap();
        assert_eq!(out, b"\x1b]52;c;aGk=\x07");
        // Empty text still frames cleanly.
        let mut out = Vec::new();
        osc52_copy(&mut out, "").unwrap();
        assert_eq!(out, b"\x1b]52;c;\x07");
    }

    #[test]
    fn draw_captures_the_transcript_and_paints_the_selection() {
        use ratatui::backend::TestBackend;
        let rt = test_rt();
        let mut model = UiModel::new(rt.state());
        model.push_user("hello selection");
        let theme = Theme::default();
        let input = InputEditor::default();
        let picker = None;
        let mut terminal = Terminal::new(TestBackend::new(40, 8)).unwrap();
        let render = |terminal: &mut Terminal<TestBackend>, model: &mut UiModel| {
            terminal
                .draw(|f| draw(f, model, &theme, &input, &picker, false, ' ', "fake"))
                .unwrap();
        };
        render(&mut terminal, &mut model);

        // The viewport readback holds the rendered rows.
        assert!(
            model
                .transcript_rows
                .iter()
                .any(|r| r.contains("hello selection")),
            "{:?}",
            model.transcript_rows
        );
        let area = model.transcript_area;
        assert!(area.height > 0);

        // A selection repaints cells in the rendered buffer.
        let before = terminal.backend().buffer().content.clone();
        model.selection = Some(Selection {
            anchor: (area.x, area.y),
            head: (area.x + 4, area.y),
        });
        render(&mut terminal, &mut model);
        let after = terminal.backend().buffer().content.clone();
        assert_ne!(
            before, after,
            "the selection highlight should repaint cells"
        );
        // The default theme leaves the selection fg/bg unset, so the highlight
        // is reverse video (kept bold from the token) and must not reset the
        // text's own color.
        let cell = terminal
            .backend()
            .buffer()
            .cell(Position {
                x: area.x,
                y: area.y,
            })
            .unwrap()
            .clone();
        assert!(
            cell.modifier.contains(ratatui::style::Modifier::REVERSED),
            "selected cell should be reversed, got {:?}",
            cell.modifier
        );
        assert_ne!(cell.fg, ratatui::style::Color::Reset);
    }

    // --- line-editing input editor ---

    #[test]
    fn input_editor_inserts_at_cursor_and_moves() {
        let mut e = InputEditor::new("abcd");
        assert_eq!(e.cursor(), 4);
        e.left();
        e.left();
        e.insert_char('X');
        assert_eq!(e.text(), "abXcd");
        assert_eq!(e.cursor(), 3);
        e.home();
        assert_eq!(e.cursor(), 0);
        e.end();
        assert_eq!(e.cursor(), 5);
        e.backspace();
        assert_eq!(e.text(), "abXc");
        // Delete at the end is a no-op.
        e.delete();
        assert_eq!(e.text(), "abXc");
        // Delete after the cursor removes the following char.
        e.left();
        e.delete();
        assert_eq!(e.text(), "abX");
    }

    #[test]
    fn input_editor_arrow_keys_are_noops_at_the_edges() {
        // Regression: unguarded left()/right() at the edges looped forever —
        // prev_boundary(0) underflowed and next_boundary(len) scanned past the
        // end, where is_char_boundary is false for every index.
        let mut e = InputEditor::new("hi");
        e.home();
        e.left();
        e.left();
        e.left();
        assert_eq!(e.cursor(), 0);
        e.end();
        e.right();
        e.right();
        e.right();
        assert_eq!(e.cursor(), 2);
        // Empty input too.
        let mut e = InputEditor::default();
        e.left();
        e.right();
        assert_eq!(e.cursor(), 0);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn input_editor_arrow_keys_move_by_char_not_byte() {
        let mut e = InputEditor::new("éx");
        e.home();
        e.right();
        // é is two bytes; the cursor must land past the whole char.
        assert_eq!(e.cursor(), 2);
        e.backspace();
        assert_eq!(e.text(), "x");
    }

    #[test]
    fn input_editor_kill_commands() {
        // Ctrl-W: previous word plus the whitespace before it (readline-style).
        let mut e = InputEditor::new("foo bar   baz");
        e.end();
        e.kill_prev_word();
        assert_eq!(e.text(), "foo bar");
        assert_eq!(e.cursor(), 7);
        // Ctrl-U: kill to line start.
        e.kill_to_start();
        assert_eq!(e.text(), "");
        // Ctrl-K: kill to line end.
        e.insert("keep");
        e.left();
        e.kill_to_end();
        assert_eq!(e.text(), "kee");
    }

    #[test]
    fn input_editor_take_resets_for_the_next_line() {
        let mut e = InputEditor::new("hello");
        assert_eq!(e.take(), "hello");
        assert_eq!(e.text(), "");
        assert_eq!(e.cursor(), 0);
        e.insert("next");
        assert_eq!(e.text(), "next");
    }

    #[test]
    fn input_editor_flattens_pasted_control_chars() {
        let mut e = InputEditor::default();
        e.insert("line1\nline2\r\n");
        assert_eq!(e.text(), "line1 line2  ");
        assert_eq!(e.cursor(), e.text().len());
    }

    #[test]
    fn input_editor_window_shows_tail_and_keeps_cursor_visible() {
        let mut e = InputEditor::new("hello world");
        e.end();
        let (w, col) = e.window(5);
        // Block-cursor model: 4 chars + the caret cell fill the viewport.
        assert_eq!((w.as_str(), col), ("orld", 4));
        // Five lefts put the cursor at column 6; the window follows it so the
        // caret stays visible (at the left edge of the viewport).
        for _ in 0..5 {
            e.left();
        }
        let (w, col) = e.window(5);
        assert_eq!((w.as_str(), col), ("world", 0));
        e.home();
        let (w, col) = e.window(5);
        assert_eq!((w.as_str(), col), ("hello", 0));
    }

    #[test]
    fn input_editor_window_fits_short_text_whole() {
        let e = InputEditor::new("hi");
        let (w, col) = e.window(10);
        assert_eq!(w, "hi");
        assert_eq!(col, 2);
        // Degenerate zero-width viewport.
        let (w, col) = e.window(0);
        assert_eq!((w, col), (String::new(), 0));
    }

    // --- picker overlay helpers ---

    #[test]
    fn picker_offset_keeps_selection_visible() {
        assert_eq!(picker_offset(0, 3), 0);
        assert_eq!(picker_offset(2, 3), 0);
        assert_eq!(picker_offset(3, 3), 1);
        assert_eq!(picker_offset(4, 3), 2);
        // Degenerate zero-height viewport.
        assert_eq!(picker_offset(9, 0), 9);
    }

    // --- theme → ratatui style mapping ---

    #[test]
    fn theme_style_maps_colors_and_modifiers() {
        use daedalus_core::theme::{Modifiers, StyleSpec, ThemeColor};
        let s = style(StyleSpec {
            fg: ThemeColor::Rgb(1, 2, 3),
            bg: ThemeColor::Indexed(4),
            modifiers: Modifiers {
                bold: true,
                italic: true,
                ..Default::default()
            },
        });
        assert_eq!(s.fg, Some(ratatui::style::Color::Rgb(1, 2, 3)));
        assert_eq!(s.bg, Some(ratatui::style::Color::Indexed(4)));
        assert!(s.add_modifier.contains(Modifier::BOLD));
        assert!(s.add_modifier.contains(Modifier::ITALIC));
        // The terminal default maps to Reset.
        assert_eq!(
            style(StyleSpec::default()).fg,
            Some(ratatui::style::Color::Reset)
        );
    }

    // --- animated spinner ---

    #[test]
    fn spinner_frame_advances_with_elapsed_time() {
        assert_eq!(spinner_frame(Duration::from_millis(0)), '⠋');
        assert_eq!(spinner_frame(Duration::from_millis(100)), '⠙');
        assert_eq!(spinner_frame(Duration::from_millis(900)), '⠏');
        // One full cycle wraps back to the first frame.
        assert_eq!(spinner_frame(Duration::from_millis(1000)), '⠋');
    }
}
