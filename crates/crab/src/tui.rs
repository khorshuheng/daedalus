//! Ratatui TUI (CRAB-121), modeled on pi's interactive mode.
//!
//! The TUI is a **client of `AgentRuntime`** — it never owns the agent loop.
//! It sends commands (`prompt`/`steer`/`abort`/`set_model`/`set_effort`/
//! `switch_workspace`/`clear`/`resume`) and renders the runtime's Event
//! stream. Layout: transcript on top, a line-editing input with a visible
//! caret at the bottom (CRAB-126), a centered picker overlay for /model and
//! /effort (CRAB-127), and a footer/status line (provider, model, effort,
//! animated spinner while busy, CRAB-128). CRAB-138 adds `/skills` (list) and
//! `/skill <name>` (load an instruction file as a user message).
//!
//! This module is split so the behavior is testable without a terminal:
//! the pure model (`parse_slash`, `LineAction` routing, `apply_event`
//! transcript/status updates) lives here with unit tests, and the
//! `run_tui` shell (crossterm raw mode + alternate screen, event poll
//! loop, ratatui draw) is a thin wrapper over it. The interactive path
//! never uses `println!`/`print!` — ratatui owns the screen.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crab_core::runtime::{AgentRuntime, Effort, Event, RuntimeState};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
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
    /// Change the thinking effort (opens the effort picker when no argument).
    Effort(String),
    /// Change the workspace.
    Workspace(PathBuf),
    /// List the discovered skills (user + workspace).
    Skills,
    /// Load a skill's content into the conversation as a user message.
    Skill(String),
    /// List the registered tools (built-ins + MCP servers, CRAB-133).
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
        let cmd = match name {
            "login" => Some(SlashCommand::Login),
            "model" => Some(SlashCommand::Model(arg.to_string())),
            "effort" => Some(SlashCommand::Effort(arg.to_string())),
            "workspace" => Some(SlashCommand::Workspace(PathBuf::from(arg))),
            "skills" => Some(SlashCommand::Skills),
            "skill" => Some(SlashCommand::Skill(arg.to_string())),
            "tools" => Some(SlashCommand::Tools),
            "resume" => Some(SlashCommand::Resume),
            "clear" => Some(SlashCommand::Clear),
            "help" => Some(SlashCommand::Help),
            "exit" | "quit" | "q" => Some(SlashCommand::Exit),
            _ => None,
        };
        match cmd {
            Some(c) => LineAction::Command(c),
            None => LineAction::Command(SlashCommand::Help), // unknown -> help
        }
    } else {
        LineAction::Message(trimmed.to_string())
    }
}

/// One-line, human-readable summary of a tool call's arguments, shown after
/// the tool name in the transcript (CRAB-139). Returns `None` when there is no
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
    /// Streamed model reasoning (CRAB-139), rendered dim and italic.
    Thinking(String),
    Tool(String),
    Notice(String),
}

/// Pure UI model: the transcript and the status footer, updated from runtime
/// Events. No terminal I/O here, so it is unit-testable.
#[derive(Debug, Default, Clone)]
pub struct UiModel {
    pub transcript: Vec<TranscriptLine>,
    pub assistant_buf: String,
    /// Streamed model reasoning, flushed as a `Thinking` line (CRAB-139).
    pub thinking_buf: String,
    pub state: RuntimeState,
    pub usage: Option<usize>,
    pub iterations: usize,
    /// True when the last event was a turn end (used to reset stats display).
    pub settled: bool,
    /// Transcript scrollback (follow the tail unless the user scrolled up).
    pub scroll: TranscriptScroll,
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
        }
    }

    /// Fold a runtime Event into the model. Returns nothing; the caller just
    /// re-renders. This is the single place events become display state.
    pub fn apply_event(&mut self, event: &Event) {
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
            Event::ToolEnd { name, ok, .. } => {
                self.flush_thinking();
                let marker = if *ok { "✓" } else { "✗" };
                self.transcript
                    .push(TranscriptLine::Tool(format!("{marker} {name}")));
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
            } => {
                self.state.model = model.clone();
                self.state.effort = *effort;
                self.state.workspace = workspace.clone();
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
            Event::Error { message } => {
                self.flush_thinking();
                self.flush_assistant();
                self.transcript
                    .push(TranscriptLine::Notice(format!("error: {message}")));
            }
        }
    }

    /// Push accumulated model reasoning (if any) as a transcript line.
    /// Always called before `flush_assistant`, so thinking renders before the
    /// answer it precedes (CRAB-139).
    fn flush_thinking(&mut self) {
        if !self.thinking_buf.is_empty() {
            self.transcript
                .push(TranscriptLine::Thinking(std::mem::take(
                    &mut self.thinking_buf,
                )));
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
    }

    /// Show a notice line (slash command results, etc.).
    pub fn push_notice(&mut self, text: &str) {
        self.transcript
            .push(TranscriptLine::Notice(text.to_string()));
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

/// A single-line text editor for the input box (CRAB-126): the text plus a
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

/// Braille spinner frames, advanced by elapsed time (CRAB-128) so the
/// animation runs at a steady cadence independent of the render loop.
const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The spinner frame for a point in time (one step per 100ms).
fn spinner_frame(elapsed: Duration) -> char {
    SPINNER_FRAMES[(elapsed.as_millis() / 100) as usize % SPINNER_FRAMES.len()]
}

/// First visible row of a picker list so `selected` stays inside the viewport
/// (CRAB-127): the list only scrolls when the selection leaves the window.
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

/// Ratatui rendering + event loop shell. Owns the screen (crossterm raw
/// mode + alternate screen); all *state* lives in the pure model above.
/// Never prints to stdout directly — ratatui owns the terminal. On exit the
/// session is auto-saved via `crate::auto_save`.
pub fn run_tui(
    rt: &AgentRuntime,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
    initial: &str,
    session_root: &Path,
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
    let mut input = InputEditor::default();
    let mut picker: Option<Picker> = None;
    let mut login_pending = false;
    let mut should_exit = false;

    let result = (|| -> Result<i32, String> {
        // One-time editing-key hint so the line editor is discoverable.
        model.push_notice(
            "editing: ←→ Home End Del · Ctrl-W word · Ctrl-U/Ctrl-K line · PgUp/PgDn/↑↓ scroll · /help commands",
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
                let state_changed = matches!(ev, Event::StateChanged { .. });
                model.apply_event(&ev);
                if state_changed {
                    model.state = rt.state();
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
                    TermEvent::Key(key) if key.kind == KeyEventKind::Press => handle_key(
                        rt,
                        &mut model,
                        &mut input,
                        &mut picker,
                        &mut login_pending,
                        &mut should_exit,
                        session_root,
                        key.code,
                        key.modifiers,
                    ),
                    // Bracketed paste: insert the whole pasted text at the
                    // caret (so Ctrl+Shift+V works for keys / long inputs);
                    // CR/LF is flattened because the editor is single-line.
                    TermEvent::Paste(text) => input.insert(&text),
                    // Mouse wheel scrolls the transcript (3 rows per notch).
                    TermEvent::Mouse(me) => match me.kind {
                        MouseEventKind::ScrollUp => model.scroll.scroll_by(-3),
                        MouseEventKind::ScrollDown => model.scroll.scroll_by(3),
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
    // Auto-save at session end (CRAB-109) *after* the terminal is restored,
    // so its stderr output lands on the normal screen. Failures are warnings
    // only — quitting must never be blocked by persistence.
    crate::auto_save(rt, session_root);
    rt.shutdown();
    result
}

/// A modal picker overlay (model / effort selection).
enum Picker {
    Effort { selected: usize },
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
            },
            KeyCode::Up | KeyCode::Char('k') => match p {
                Picker::Effort { selected } => {
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
            },
            _ => {}
        }
        return;
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
        // Line editing at the caret (CRAB-126): movement, insert, delete,
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
        match crab_core::credential::store_api_key(provider, key) {
            // The running runtime built its provider at startup, so the stored
            // key applies from the next launch (CRAB-143 review).
            Ok(()) => model.push_notice(&format!(
                "stored API key for {} (restart crab to use it)",
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
            model.push_notice(
                "/login /model /effort /workspace /resume /clear /skills /skill /tools /help /exit",
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
            match crab_core::session::load_previous(session_root, &rt.workspace_root()) {
                Ok(Some(h)) => {
                    rt.replace_history(h);
                    model.transcript.clear();
                    model.push_notice("resumed previous session");
                }
                Ok(None) => model.push_notice("no previous session"),
                Err(e) => model.push_notice(&format!("could not resume: {e}")),
            }
        }
        SlashCommand::Login => {
            model.push_notice("/login: type your API key, then Enter");
            *login_pending = true;
        }
        SlashCommand::Model(arg) => {
            if arg.is_empty() {
                // CRAB-132: no static model lists — the user types the model.
                model.push_notice("usage: /model <model>");
            } else {
                rt.set_model(&arg);
                model.state.model = arg;
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
                    "no skills found (add <workspace>/.crab/skills/<name>.md or \
                     ~/.config/crab/skills/<name>.md)",
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

/// Render a frame: transcript on top, input editor (with a visible caret) at
/// the bottom, footer with an animated spinner while busy. The model/effort
/// picker renders as a centered overlay sized to its choices (CRAB-127).
fn draw(
    f: &mut Frame,
    model: &mut UiModel,
    input: &InputEditor,
    picker: &Option<Picker>,
    busy: bool,
    spinner: char,
    provider: &str,
) {
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::text::{Line as TLine, Span};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    // Transcript.
    let lines: Vec<TLine> = model
        .transcript
        .iter()
        .map(|tl| match tl {
            TranscriptLine::User(t) => TLine::from(Span::styled(
                format!("▶ {t}"),
                ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
            )),
            TranscriptLine::Assistant(t) => TLine::from(Span::styled(
                t.to_string(),
                ratatui::style::Style::default().fg(ratatui::style::Color::White),
            )),
            TranscriptLine::Thinking(t) => TLine::from(Span::styled(
                format!("  {t}"),
                ratatui::style::Style::default()
                    .fg(ratatui::style::Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )),
            TranscriptLine::Tool(t) => TLine::from(Span::styled(
                format!("  {t}"),
                ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
            )),
            TranscriptLine::Notice(t) => TLine::from(Span::styled(
                format!("• {t}"),
                ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
            )),
        })
        .collect();
    // `line_count` returns the wrapped height including the block's border
    // rows, so the full chunk height is the matching viewport. Scrolling by the
    // logical line count instead left the newest (wrapped) lines off-screen.
    let inner_w = chunks[0].width.saturating_sub(2);
    let transcript = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" crab "))
        .wrap(Wrap { trim: false });
    let total = transcript.line_count(inner_w);
    let top = model.scroll.resolve(total, chunks[0].height as usize);
    let transcript = transcript.scroll(((top.min(u16::MAX as usize)) as u16, 0));
    f.render_widget(transcript, chunks[0]);

    // Input editor (or picker overlay).
    match picker {
        Some(Picker::Effort { selected }) => {
            let title = " effort — ↑/↓ · Enter apply · Esc cancel ";
            let names: Vec<String> = EFFORT_CHOICES
                .iter()
                .map(|e| e.name().to_string())
                .collect();
            draw_picker(f, &names, *selected, title);
        }
        None => {
            // Show the window of the text that contains the caret so long
            // lines stay editable (CRAB-126), and place the terminal cursor
            // on the caret so typing position is visible.
            let inner_w = chunks[1].width.saturating_sub(2) as usize;
            let (window, cursor_col) = input.window(inner_w);
            let editor = Paragraph::new(window)
                .block(Block::default().borders(Borders::ALL).title(" input "));
            f.render_widget(editor, chunks[1]);
            let x = chunks[1].x + 1 + cursor_col as u16;
            f.set_cursor_position(Position {
                x: x.min(chunks[1].x + chunks[1].width.saturating_sub(1)),
                y: chunks[1].y + 1,
            });
        }
    }

    // Footer/status.
    let status = format!(
        "{} {} ({}) | effort {} | {} | {} | {:?} tokens | {} turn(s){}",
        spinner,
        model.state.model,
        provider,
        model.state.effort.name(),
        model.state.workspace,
        if busy { "busy" } else { "idle" },
        model.usage,
        model.iterations,
        if model.settled { " · settled" } else { "" },
    );
    let footer = Paragraph::new(TLine::from(Span::raw(status)));
    f.render_widget(footer, chunks[2]);
}

/// Render the picker overlay: a centered window sized to the choices (not the
/// 1-row input slot), cleared behind, with a viewport that keeps the selected
/// row visible and the current selection highlighted.
fn draw_picker(f: &mut Frame, items: &[String], selected: usize, title: &str) {
    use ratatui::text::{Line as TLine, Span};
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
                line.style(Style::default().add_modifier(Modifier::BOLD))
            } else {
                line
            }
        })
        .collect();
    let picker = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(picker, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_core::config::Config;
    use crab_core::provider::fake::FakeProvider;
    use crab_core::tools::resolver::ToolSet;
    use crab_core::workspace::Workspace;

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

    /// CRAB-133: `/tools` lists the registered toolset (the four built-ins
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
        for builtin in ["read", "bash", "edit", "write"] {
            assert!(
                model
                    .transcript
                    .iter()
                    .any(|l| matches!(l, TranscriptLine::Notice(n) if n.starts_with(builtin))),
                "missing tool listing for {builtin}"
            );
        }
    }

    /// CRAB-138: `/skills` lists discovered skills and `/skill <name>` loads
    /// one as a user message (unknown names get a notice instead).
    #[test]
    fn skill_command_loads_body_and_lists_catalog() {
        let dir = tempfile::tempdir().expect("temp dir");
        let skills_dir = dir.path().join(".crab").join("skills");
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

    #[test]
    fn state_changed_updates_footer_state() {
        let mut m = UiModel::new(state());
        m.apply_event(&Event::StateChanged {
            model: "claude-x".into(),
            effort: Effort::High,
            workspace: "/other".into(),
        });
        assert_eq!(m.state.model, "claude-x");
        assert_eq!(m.state.effort, Effort::High);
        assert_eq!(m.state.workspace, "/other");
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

    // --- CRAB-126: line-editing input editor ---

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

    // --- CRAB-127: picker overlay helpers ---

    #[test]
    fn picker_offset_keeps_selection_visible() {
        assert_eq!(picker_offset(0, 3), 0);
        assert_eq!(picker_offset(2, 3), 0);
        assert_eq!(picker_offset(3, 3), 1);
        assert_eq!(picker_offset(4, 3), 2);
        // Degenerate zero-height viewport.
        assert_eq!(picker_offset(9, 0), 9);
    }

    // --- CRAB-128: animated spinner ---

    #[test]
    fn spinner_frame_advances_with_elapsed_time() {
        assert_eq!(spinner_frame(Duration::from_millis(0)), '⠋');
        assert_eq!(spinner_frame(Duration::from_millis(100)), '⠙');
        assert_eq!(spinner_frame(Duration::from_millis(900)), '⠏');
        // One full cycle wraps back to the first frame.
        assert_eq!(spinner_frame(Duration::from_millis(1000)), '⠋');
    }
}
