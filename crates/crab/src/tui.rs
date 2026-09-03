//! Ratatui TUI (CRAB-121), modeled on pi's interactive mode.
//!
//! The TUI is a **client of `AgentRuntime`** — it never owns the agent loop.
//! It sends commands (`prompt`/`steer`/`abort`/`set_model`/`set_effort`/
//! `switch_workspace`/`clear`/`resume`) and renders the runtime's Event
//! stream. Layout: input editor at the bottom, transcript above, a
//! footer/status line (provider, model, effort, spinner while busy).
//!
//! This module is split so the behavior is testable without a terminal:
//! the pure model (`parse_slash`, `LineAction` routing, `apply_event`
//! transcript/status updates) lives here with unit tests, and the
//! `run_tui` shell (crossterm raw mode + alternate screen, event poll
//! loop, ratatui draw) is a thin wrapper over it. The interactive path
//! never uses `println!`/`print!` — ratatui owns the screen.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use crab_core::config::ProviderKind;
use crab_core::runtime::{AgentRuntime, Effort, Event, RuntimeState};
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Frame;
use ratatui::Terminal;

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

/// A single line of the transcript (user or assistant content). Assistant
/// text streams in via `text_delta` events and is appended to the current
/// assistant line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptLine {
    User(String),
    Assistant(String),
    Tool(String),
    Notice(String),
}

/// Pure UI model: the transcript and the status footer, updated from runtime
/// Events. No terminal I/O here, so it is unit-testable.
#[derive(Debug, Default, Clone)]
pub struct UiModel {
    pub transcript: Vec<TranscriptLine>,
    pub assistant_buf: String,
    pub state: RuntimeState,
    pub usage: Option<usize>,
    pub iterations: usize,
    /// True when the last event was a turn end (used to reset stats display).
    pub settled: bool,
}

impl UiModel {
    pub fn new(state: RuntimeState) -> Self {
        Self {
            transcript: Vec::new(),
            assistant_buf: String::new(),
            state,
            usage: None,
            iterations: 0,
            settled: false,
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
                self.assistant_buf.push_str(text);
            }
            Event::ThinkingDelta { .. } => {} // rendered inline by the shell if desired
            Event::ToolStart { name, .. } => {
                self.flush_assistant();
                self.transcript
                    .push(TranscriptLine::Tool(format!("⚙ {name}")));
            }
            Event::ToolEnd { name, ok, .. } => {
                let marker = if *ok { "✓" } else { "✗" };
                self.transcript
                    .push(TranscriptLine::Tool(format!("{marker} {name}")));
            }
            Event::TurnEnd {} => {
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
                // The settled text may already be in assistant_buf (streamed);
                // if not, show it as the final line.
                if !text.is_empty() && !self.assistant_buf.contains(text) {
                    self.assistant_buf.push_str(text);
                }
                self.flush_assistant();
            }
            Event::Error { message } => {
                self.flush_assistant();
                self.transcript
                    .push(TranscriptLine::Notice(format!("error: {message}")));
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

/// Candidate models offered by the `/model` picker (provider presets; the
/// user can also type a custom model).
pub fn model_choices(provider: ProviderKind) -> &'static [&'static str] {
    match provider {
        ProviderKind::Openai => &["gpt-4o-mini", "gpt-4o", "o3-mini"],
        ProviderKind::Anthropic => &["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        ProviderKind::Deepseek => &["deepseek-chat", "deepseek-reasoner"],
        ProviderKind::Fake => &["fake-model"],
    }
}

/// Ratatui rendering + event loop shell. Owns the screen (crossterm raw
/// mode + alternate screen); all *state* lives in the pure model above.
/// Never prints to stdout directly — ratatui owns the terminal.
pub fn run_tui(
    rt: &AgentRuntime,
    rx: &Receiver<Event>,
    initial: &str,
    session_root: &Path,
) -> Result<i32, String> {
    enable_raw_mode().map_err(|e| format!("cannot enable raw mode: {e}"))?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| format!("cannot enter alt screen: {e}"))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;

    let mut model = UiModel::new(rt.state());
    let mut input = String::new();
    let mut picker: Option<Picker> = None;
    let mut login_pending = false;
    let mut should_exit = false;

    let result = (|| -> Result<i32, String> {
        // Seed the transcript with the initial prompt, then start the turn.
        if !initial.trim().is_empty() {
            model.push_user(initial);
            rt.prompt(initial);
        }

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
                terminal
                    .draw(|f| draw(f, &model, &input, &picker, rt.is_busy(), provider.name()))
                    .map_err(|e| e.to_string())?;
                last_render = Instant::now();
            }
            // Poll for a key (short timeout keeps the spinner/event drain live).
            if event::poll(Duration::from_millis(33)).map_err(|e| e.to_string())? {
                if let TermEvent::Key(key) = event::read().map_err(|e| e.to_string())? {
                    if key.kind == KeyEventKind::Press {
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
                    }
                }
            }
        }
        // Save the session on exit (auto-save at session end).
        let history = rt.history();
        let _ = crab_core::session::save_session(session_root, &rt.workspace_root(), &history);
        rt.shutdown();
        Ok(0)
    })();

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    result
}

/// A modal picker overlay (model / effort selection).
enum Picker {
    Model {
        choices: Vec<String>,
        selected: usize,
    },
    Effort {
        selected: usize,
    },
}

/// Route one key press. Pure decisions delegated to the model where possible;
/// runtime calls happen here (the shell owns the AgentRuntime handle).
#[allow(clippy::too_many_arguments)] // shell glue: one call site
fn handle_key(
    rt: &AgentRuntime,
    model: &mut UiModel,
    input: &mut String,
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
            Picker::Model { choices, .. } => choices.len(),
            Picker::Effort { .. } => EFFORT_CHOICES.len(),
        };
        match code {
            KeyCode::Esc => *picker = None,
            KeyCode::Down | KeyCode::Char('j') => match p {
                Picker::Model { selected, .. } | Picker::Effort { selected } => {
                    *selected = (*selected + 1).min(max - 1);
                }
            },
            KeyCode::Up | KeyCode::Char('k') => match p {
                Picker::Model { selected, .. } | Picker::Effort { selected } => {
                    *selected = selected.saturating_sub(1);
                }
            },
            KeyCode::Enter => match p {
                Picker::Model { choices, selected } => {
                    let model_name = choices[*selected].clone();
                    rt.set_model(&model_name);
                    model.state.model = model_name;
                    *picker = None;
                }
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
            let line = std::mem::take(input);
            if submit_line(rt, model, picker, session_root, login_pending, &line) {
                *should_exit = true;
            }
        }
        KeyCode::Backspace => {
            input.pop();
        }
        KeyCode::Char(c) => {
            input.push(c);
        }
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
        let provider = provider_for(rt);
        match crab_core::credential::store_api_key(provider, key) {
            Ok(()) => model.push_notice(&format!("stored API key for {}", provider.name())),
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
            model.push_notice("/login /model /effort /workspace /resume /clear /help /exit");
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
                let provider = provider_for(rt);
                let choices = model_choices(provider)
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                *picker = Some(Picker::Model {
                    choices,
                    selected: 0,
                });
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
    }
    false
}

/// Render a frame: transcript on top, input editor at the bottom, footer.
fn draw(
    f: &mut Frame,
    model: &UiModel,
    input: &str,
    picker: &Option<Picker>,
    busy: bool,
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
    let viewport = (chunks[0].height as usize).saturating_sub(2); // borders
    let scrollback = lines.len().saturating_sub(viewport);
    let transcript = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" crab "))
        .scroll((scrollback as u16, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(transcript, chunks[0]);

    // Input editor (or picker overlay).
    match picker {
        Some(Picker::Model { choices, selected }) => {
            let items: Vec<TLine> = choices
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let marker = if i == *selected { "❯ " } else { "  " };
                    TLine::from(Span::raw(format!("{marker}{c}")))
                })
                .collect();
            let p = Paragraph::new(items)
                .block(Block::default().borders(Borders::ALL).title(" model "))
                .scroll((*selected as u16, 0));
            f.render_widget(p, chunks[1]);
        }
        Some(Picker::Effort { selected }) => {
            let items: Vec<TLine> = EFFORT_CHOICES
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let marker = if i == *selected { "❯ " } else { "  " };
                    TLine::from(Span::raw(format!("{marker}{}", e.name())))
                })
                .collect();
            let p = Paragraph::new(items)
                .block(Block::default().borders(Borders::ALL).title(" effort "))
                .scroll((*selected as u16, 0));
            f.render_widget(p, chunks[1]);
        }
        None => {
            let editor = Paragraph::new(input)
                .block(Block::default().borders(Borders::ALL).title(" input "))
                .scroll((0, 0));
            f.render_widget(editor, chunks[1]);
        }
    }

    // Footer/status.
    let spinner = if busy {
        "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".chars().next().unwrap_or(' ')
    } else {
        ' '
    };
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

/// The provider currently configured for this runtime (for /login + /model
/// picker). Reads it from the runtime's shared state via the config the
/// runtime was built with — exposed through `state` indirectly; for the
/// picker we infer from the model preset.
fn provider_for(rt: &AgentRuntime) -> ProviderKind {
    // The runtime state carries the workspace/model but not the provider
    // kind; infer from the model name's preset (best effort for the picker).
    let m = rt.state().model;
    for p in [
        ProviderKind::Openai,
        ProviderKind::Anthropic,
        ProviderKind::Deepseek,
    ] {
        if p.preset_model() == m {
            return p;
        }
    }
    ProviderKind::Openai
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn model_choices_are_provider_specific() {
        assert!(model_choices(ProviderKind::Openai).contains(&"gpt-4o"));
        assert!(model_choices(ProviderKind::Deepseek).contains(&"deepseek-reasoner"));
        assert_eq!(model_choices(ProviderKind::Fake), &["fake-model"]);
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
        });
        assert!(m.assistant_buf.is_empty());
        assert_eq!(m.transcript[0], TranscriptLine::Assistant("Hello".into()));
        assert_eq!(m.transcript[1], TranscriptLine::Tool("⚙ bash".into()));
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
}
