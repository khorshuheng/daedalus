//! Crab CLI entrypoint (CRAB-101).
//!
//! Takes a positional `<prompt>` and a workspace directory (`--dir`, defaulting
//! to cwd) plus `--model`, `--provider`, `--max-iterations`, and `--config`.
//! Builds the app from config (CRAB-105), constructs the agent loop (CRAB-104)
//! with the selected provider (CRAB-103) and tools (CRAB-102), runs it, and
//! prints the final answer to stdout.
//!
//! Exit codes:
//!   0 — clean final answer
//!   1 — hard error (bad config/flags, provider unreachable)
//!   2 — iteration cap exceeded

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crab::agent::{Agent, AgentError, Session, Stream, Turn};
use crab::config::{Config, Overrides, ProviderKind};
use crab::memory;
use crab::provider::{self, Message};
use crab::reflect;
use crab::session;
use crab::term;
use crab::tools::resolver::ToolSet;
use crab::workspace::Workspace;

const USAGE: &str = "\
crab — a minimal coding agent

USAGE:
    crab <prompt> [OPTIONS]

ARGS:
    <prompt>    The instruction to give the model.

OPTIONS:
    --dir <path>            Workspace directory (default: current directory)
    --provider <name>       openai | anthropic | deepseek | fake (default: openai)
    --model <name>          Model identifier (provider-specific default)
    --max-iterations <n>    Iteration cap for the agent loop (default: 30)
    --config <path>         Config file (default: ~/.config/crab/config.toml)
    -h, --help              Print this help.

Precedence for config values: flags > env (CRAB_*) > config file > defaults.
";

enum ParseOutcome {
    Run(Box<Cli>),
    Help,
}

struct Cli {
    prompt: String,
    dir: Option<PathBuf>,
    config_path: Option<PathBuf>,
    flags: Overrides,
}

fn next_value<'a>(it: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or_else(|| format!("flag '{flag}' requires a value"))
}

fn parse_args(args: &[String]) -> Result<ParseOutcome, String> {
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut dir: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut flags = Overrides::default();

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dir" => dir = Some(PathBuf::from(next_value(&mut it, "--dir")?)),
            "--model" => flags.model = Some(next_value(&mut it, "--model")?),
            "--provider" => {
                let v = next_value(&mut it, "--provider")?;
                flags.provider = Some(ProviderKind::parse(&v)?);
            }
            "--max-iterations" => {
                let v = next_value(&mut it, "--max-iterations")?;
                flags.max_iterations = Some(
                    v.parse()
                        .map_err(|_| format!("invalid --max-iterations '{v}'"))?,
                );
            }
            "--config" => config_path = Some(PathBuf::from(next_value(&mut it, "--config")?)),
            "-h" | "--help" => return Ok(ParseOutcome::Help),
            s if s.starts_with("--dir=") => dir = Some(PathBuf::from(&s["--dir=".len()..])),
            s if s.starts_with('-') => return Err(format!("unknown flag '{s}'")),
            s => prompt_parts.push(s.to_string()),
        }
    }

    if prompt_parts.is_empty() {
        return Err("no prompt given".into());
    }

    Ok(ParseOutcome::Run(Box::new(Cli {
        prompt: prompt_parts.join(" "),
        dir,
        config_path,
        flags,
    })))
}

/// If `--config` was not given, use `~/.config/crab/config.toml` when present.
fn resolve_config_path(explicit: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".config/crab/config.toml");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn run(cli: Cli) -> Result<i32, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot determine cwd: {e}"))?;
    let default_workspace = cli.dir.clone().unwrap_or(cwd);
    let config_path = resolve_config_path(cli.config_path);

    let config = Config::load(
        default_workspace,
        config_path.as_deref(),
        Overrides::from_env(),
        cli.flags,
    )?;

    let workspace = Workspace::new(config.workspace.clone())?;
    let provider = provider::from_config(&config);
    let tools = ToolSet::new(config.max_output_bytes);
    let agent = Agent::new(provider.as_ref(), &tools, &workspace, &config);

    // Interactive terminal -> REPL (Ctrl-C/Esc cancel while busy, terminate at
    // the prompt, /exit to quit). Piped stdin -> one-shot.
    if !std::io::stdin().is_terminal() {
        return match agent.run(&cli.prompt) {
            Ok(answer) => {
                println!("{answer}");
                Ok(0)
            }
            Err(AgentError::IterationCap(n)) => {
                eprintln!("crab: iteration cap exceeded: no final answer after {n} iterations");
                Ok(2)
            }
            Err(e) => Err(e.to_string()),
        };
    }

    run_repl(&agent, &cli.prompt)
}

/// Persist the current session history to disk (auto-save on exit, CRAB-109)
/// and, when the conversation produced output, reflect it into memory
/// (auto-reflect at session end, CRAB-112). Failures are warnings only —
/// quitting must never be blocked by persistence or reflection.
fn auto_save(agent: &Agent, session: &Session, root: &Path, memory_root: &Path) {
    let saved = match session::save_session(root, agent.workspace_root(), session.history()) {
        Ok(path) => {
            eprintln!("session saved: {}", path.display());
            Some(path)
        }
        Err(e) => {
            eprintln!("crab: warning: could not save session: {e}");
            None
        }
    };
    if !has_conversation(session.history()) {
        return;
    }
    // The session was just saved, so lessons can carry its real id.
    let source = saved.as_deref().and_then(session::file_id);
    match reflect::reflect_and_store(
        memory_root,
        agent.workspace_root(),
        agent.provider(),
        term::cancel_flag(),
        session.history(),
        source,
        now_millis(),
    ) {
        Ok(0) => {}
        Ok(n) => eprintln!("memory: reflected {n} new lesson(s) from this session"),
        Err(e) => eprintln!("crab: warning: could not reflect lessons: {e}"),
    }
}

/// How the REPL reacts to a slash command.
enum CommandOutcome {
    /// `/exit`/`/quit` — leave the REPL (after auto-saving).
    Exit,
    /// Show this message to the user.
    Message(String),
    /// Show this error to the user.
    Error(String),
}

/// The built-in command list shown by `/help` and after an unknown command.
const COMMANDS: &str = "\
Commands:
  /resume   Continue the previous session for this workspace
  /clear    Reset the conversation to a fresh context
  /reflect  Extract steering lessons from this session into memory
  /help     Show this help
  /exit     Quit and save the session (also /quit, /q)";

/// True when the history contains at least one assistant message — i.e. the
/// conversation actually produced output. A session holding only the system
/// prompt (or an unanswered user message) is not worth resuming.
fn has_conversation(history: &[Message]) -> bool {
    history
        .iter()
        .any(|m| matches!(m, Message::Assistant { .. }))
}

/// Dispatch a `/`-prefixed line to its handler (CRAB-110). Adding a command
/// is one match arm plus a handler; there is no plugin mechanism.
fn dispatch_command<'a, 'inner>(
    agent: &'a Agent<'inner>,
    session: &mut Session<'a, 'inner>,
    root: &Path,
    memory_root: &Path,
    cmd: &str,
) -> CommandOutcome {
    match cmd {
        "/exit" | "/quit" | "/q" => CommandOutcome::Exit,
        "/help" => CommandOutcome::Message(COMMANDS.to_string()),
        "/resume" => CommandOutcome::Message(handle_resume(agent, session, root)),
        "/clear" => CommandOutcome::Message(handle_clear(agent, session)),
        "/reflect" => CommandOutcome::Message(handle_reflect(agent, session, root, memory_root)),
        other => CommandOutcome::Error(format!("unknown command '{other}'\n{COMMANDS}")),
    }
}

/// `/resume`: replace the running history with the previous session's, so the
/// next turn continues where that session left off. Reports clearly when
/// there is no previous session (or only an empty one) for this workspace.
fn handle_resume<'a, 'inner>(
    agent: &'a Agent<'inner>,
    session: &mut Session<'a, 'inner>,
    root: &Path,
) -> String {
    match session::load_previous(root, agent.workspace_root()) {
        Ok(Some(history)) if has_conversation(&history) => {
            let discarding = has_conversation(session.history());
            let n = history.len();
            *session = Session::with_history(agent, history, term::cancel_flag());
            if discarding {
                format!("resumed previous session ({n} messages); current conversation discarded")
            } else {
                format!("resumed previous session ({n} messages)")
            }
        }
        Ok(Some(_)) => "previous session has no conversation to resume".to_string(),
        Ok(None) => "no previous session for this workspace".to_string(),
        Err(e) => format!("could not load previous session: {e}"),
    }
}

/// `/clear`: reset the conversation to a fresh context — just the system
/// prompt, so the next message becomes the first user turn. Saved sessions
/// are left untouched (clear is a context reset, not a deletion).
fn handle_clear<'a, 'inner>(agent: &'a Agent<'inner>, session: &mut Session<'a, 'inner>) -> String {
    let fresh = vec![Message::System(agent.system_prompt())];
    *session = Session::with_history(agent, fresh, term::cancel_flag());
    "conversation cleared".to_string()
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `/reflect`: reflect on the running conversation (auto-saving it first so
/// lessons carry a real session id as provenance) and append new lessons to
/// memory. On-demand trigger (CRAB-112).
fn handle_reflect<'a, 'inner>(
    agent: &'a Agent<'inner>,
    session: &Session<'a, 'inner>,
    root: &Path,
    memory_root: &Path,
) -> String {
    if !has_conversation(session.history()) {
        return "no conversation to reflect on".to_string();
    }
    let path = match session::save_session(root, agent.workspace_root(), session.history()) {
        Ok(p) => p,
        Err(e) => return format!("could not save session: {e}"),
    };
    match reflect::reflect_and_store(
        memory_root,
        agent.workspace_root(),
        agent.provider(),
        term::cancel_flag(),
        session.history(),
        session::file_id(&path),
        now_millis(),
    ) {
        Ok(0) => "reflected: no new lessons".to_string(),
        Ok(n) => format!("reflected: added {n} new lesson(s)"),
        Err(e) => format!("reflection failed: {e}"),
    }
}

/// The interactive REPL. Raw input mode is on for the whole session; a single
/// input thread line-edits and reports Line/Cancel/Eof. While a turn runs,
/// Esc/Ctrl-C cancel it; at the prompt, they terminate. A line beginning
/// with `/` is a slash command (CRAB-110); anything else is a follow-up
/// message. The session is auto-saved on exit (CRAB-109).
fn run_repl(agent: &Agent, initial: &str) -> Result<i32, String> {
    let _raw = term::RawMode::enable().map_err(|e| format!("cannot enable raw mode: {e}"))?;
    let busy = Arc::new(AtomicBool::new(false));
    let (rx, _input) = term::spawn_input(Arc::clone(&busy));
    let root = session::default_root();
    let memory_root = memory::default_root();

    let mut session = Session::new(agent, initial, term::cancel_flag());
    loop {
        term::clear_cancel();
        // A turn runs only when the conversation has an unanswered user
        // message (the initial prompt or a follow-up). Commands like /resume
        // and /clear change the history without adding one, so the loop just
        // waits for the next input instead of firing a spurious turn.
        if session
            .history()
            .last()
            .is_some_and(|m| matches!(m, Message::User(_)))
        {
            busy.store(true, Ordering::SeqCst);

            let mut emit = |ev: Stream| match ev {
                Stream::Text(t) => {
                    print!("{t}");
                    let _ = std::io::stdout().flush();
                }
                Stream::Tools(names) => {
                    print!("\n⚙ {}", names.join(", "));
                    let _ = std::io::stdout().flush();
                }
            };
            match session.run_turn(&mut emit) {
                Ok(Turn::Final(answer)) => println!("{answer}"),
                Ok(Turn::Cancelled(_)) => println!("\n(interrupted)"),
                Err(AgentError::IterationCap(n)) => {
                    eprintln!("\ncrab: iteration cap exceeded after {n} iterations")
                }
                Err(e) => eprintln!("\ncrab: {e}"),
            }
            busy.store(false, Ordering::SeqCst);
        }

        print!("> ");
        let _ = std::io::stdout().flush();
        let event = rx.recv().map_err(|_| "input closed".to_string())?;
        match event {
            term::InputEvent::Line(line) => {
                let trimmed = line.trim();
                if trimmed.starts_with('/') {
                    // A line beginning with `/` is a command, known or not.
                    match dispatch_command(agent, &mut session, &root, &memory_root, trimmed) {
                        CommandOutcome::Exit => {
                            auto_save(agent, &session, &root, &memory_root);
                            return Ok(0);
                        }
                        CommandOutcome::Message(m) => println!("{m}"),
                        CommandOutcome::Error(e) => eprintln!("{e}"),
                    }
                } else if trimmed.is_empty() {
                    continue;
                } else {
                    session.resume(trimmed.to_string());
                }
            }
            term::InputEvent::Cancel | term::InputEvent::Eof => {
                auto_save(agent, &session, &root, &memory_root);
                return Ok(0);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_args(&args) {
        Ok(ParseOutcome::Help) => {
            print!("{USAGE}");
            std::process::exit(0);
        }
        Ok(ParseOutcome::Run(cli)) => *cli,
        Err(e) => {
            eprintln!("crab: {e}\n\n{USAGE}");
            std::process::exit(1);
        }
    };

    let code = match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("crab: {e}");
            1
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab::config::Config;
    use crab::provider::{Provider, ProviderError, Response};
    use crab::tools::resolver::ToolSet;
    use crab::workspace::Workspace;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    /// A provider that records every history it is given and answers "done".
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
        ) -> Result<crab::provider::Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            Ok(crab::provider::Completion {
                response: Response::Text("done".into()),
                prompt_tokens: None,
                aborted: false,
            })
        }
    }

    /// Build a temp workspace + sessions root + a recording provider.
    fn setup(
        name: &str,
    ) -> (
        PathBuf,
        PathBuf,
        RecordingProvider,
        ToolSet,
        Workspace,
        Config,
    ) {
        let dir = std::env::temp_dir().join(format!("crab-main-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let root = dir.join("sessions");
        let provider = RecordingProvider {
            histories: Mutex::new(Vec::new()),
        };
        let tools = ToolSet::new(1000);
        let ws = Workspace::new(dir.clone()).unwrap();
        let cfg = Config {
            max_iterations: 10,
            workspace: dir.clone(),
            ..Config::defaults(dir.clone())
        };
        (dir, root, provider, tools, ws, cfg)
    }

    fn run_turn(session: &mut Session) {
        assert!(matches!(
            session.run_turn(&mut |_| {}).unwrap(),
            Turn::Final(_)
        ));
    }

    #[test]
    fn clear_resets_history_for_the_next_message() {
        let (_dir, _root, provider, tools, ws, cfg) = setup("clear");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "initial prompt", term::cancel_flag());
        run_turn(&mut session); // establishes a conversation

        let msg = handle_clear(&agent, &mut session);
        assert!(msg.contains("cleared"));
        assert_eq!(session.history().len(), 1); // just the system prompt

        // The next message starts a fresh conversation the model cannot
        // confuse with the cleared one.
        session.resume("follow-up after clear".into());
        run_turn(&mut session);

        let histories = provider.histories.lock().unwrap();
        let second = &histories[1];
        assert!(second
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "follow-up after clear")));
        assert!(!second
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "initial prompt")));
    }

    #[test]
    fn resume_loads_the_previous_session() {
        let (_dir, root, provider, tools, ws, cfg) = setup("resume");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let prior = vec![
            Message::System("sys".into()),
            Message::User("q1".into()),
            Message::Assistant {
                text: Some("a1".into()),
                tool_calls: vec![],
            },
            Message::User("q2".into()),
            Message::Assistant {
                text: Some("a2".into()),
                tool_calls: vec![],
            },
        ];
        session::save_session(&root, agent.workspace_root(), &prior).unwrap();

        let mut session = Session::new(&agent, "trigger prompt", term::cancel_flag());
        let msg = handle_resume(&agent, &mut session, &root);
        assert!(msg.contains("resumed previous session (5 messages)"));
        // The running history is replaced; the trigger prompt is gone.
        assert_eq!(session.history(), prior.as_slice());

        // The next turn shows the provider the resumed history.
        session.resume("q3".into());
        run_turn(&mut session);
        let histories = provider.histories.lock().unwrap();
        let seen = &histories[0];
        assert!(seen
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "q1")));
        assert!(!seen
            .iter()
            .any(|m| matches!(m, Message::User(u) if u == "trigger prompt")));
    }

    #[test]
    fn resume_notes_when_the_current_conversation_is_discarded() {
        let (_dir, root, provider, tools, ws, cfg) = setup("resume-discard");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        session::save_session(
            &root,
            agent.workspace_root(),
            &[
                Message::System("sys".into()),
                Message::User("q1".into()),
                Message::Assistant {
                    text: Some("a1".into()),
                    tool_calls: vec![],
                },
            ],
        )
        .unwrap();

        // The current session already produced a conversation; resuming must
        // say it is being discarded.
        let mut session = Session::new(&agent, "initial", term::cancel_flag());
        run_turn(&mut session);
        let msg = handle_resume(&agent, &mut session, &root);
        assert!(msg.contains("resumed previous session (3 messages)"));
        assert!(msg.contains("current conversation discarded"));
        assert_eq!(session.history().len(), 3); // the loaded session
    }

    #[test]
    fn resume_with_no_previous_session_reports_clearly() {
        let (_dir, root, provider, tools, ws, cfg) = setup("resume-none");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "x", term::cancel_flag());
        let msg = handle_resume(&agent, &mut session, &root);
        assert!(msg.contains("no previous session for this workspace"));
        assert_eq!(session.history().len(), 2); // untouched
    }

    #[test]
    fn resume_skips_a_session_with_no_conversation() {
        let (_dir, root, provider, tools, ws, cfg) = setup("resume-empty");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        // A saved session that is only the system prompt (what /clear followed
        // by exit leaves behind) is not resumable.
        session::save_session(
            &root,
            agent.workspace_root(),
            &[Message::System("sys".into())],
        )
        .unwrap();

        let mut session = Session::new(&agent, "x", term::cancel_flag());
        let msg = handle_resume(&agent, &mut session, &root);
        assert!(msg.contains("no conversation to resume"));
        assert_eq!(session.history().len(), 2); // untouched
    }

    #[test]
    fn unknown_command_is_reported_with_the_command_list() {
        let (dir, root, provider, tools, ws, cfg) = setup("unknown");
        let memory_root = dir.join("memory");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "x", term::cancel_flag());
        match dispatch_command(&agent, &mut session, &root, &memory_root, "/nope") {
            CommandOutcome::Error(e) => {
                assert!(e.contains("unknown command '/nope'"));
                assert!(e.contains("/resume"));
                assert!(e.contains("/help"));
            }
            _ => panic!("expected an error outcome"),
        }
    }

    #[test]
    fn help_lists_the_commands_and_exit_returns_exit() {
        for cmd in ["/resume", "/clear", "/reflect", "/help", "/exit", "/quit"] {
            assert!(COMMANDS.contains(cmd), "missing {cmd} in help");
        }
        let (dir, root, provider, tools, ws, cfg) = setup("help");
        let memory_root = dir.join("memory");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "x", term::cancel_flag());
        match dispatch_command(&agent, &mut session, &root, &memory_root, "/exit") {
            CommandOutcome::Exit => {}
            _ => panic!("expected exit outcome"),
        }
        match dispatch_command(&agent, &mut session, &root, &memory_root, "/help") {
            CommandOutcome::Message(m) => assert_eq!(m, COMMANDS),
            _ => panic!("expected message outcome"),
        }
    }

    /// A provider that plays a scripted `Response` per call (for /reflect).
    struct ScriptedProvider {
        responses: Mutex<std::collections::VecDeque<crab::provider::Response>>,
        histories: Mutex<Vec<Vec<Message>>>,
    }

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            history: &[Message],
            _tools: &[serde_json::Value],
            _cancel: &AtomicBool,
            _on_text: &mut dyn FnMut(&str),
        ) -> Result<crab::provider::Completion, ProviderError> {
            self.histories.lock().unwrap().push(history.to_vec());
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(crab::provider::Response::Text("done".into()));
            Ok(crab::provider::Completion {
                response,
                prompt_tokens: None,
                aborted: false,
            })
        }
    }

    #[test]
    fn reflect_dispatching_adds_lessons_with_session_provenance() {
        let (dir, root, _provider, tools, ws, cfg) = setup("reflect");
        let memory_root = dir.join("memory");
        let provider = ScriptedProvider {
            responses: Mutex::new(std::collections::VecDeque::from([
                crab::provider::Response::Text("done with the task".into()),
                crab::provider::Response::Text(
                    r#"[{"text":"always run make first","kind":"rule"}]"#.into(),
                ),
            ])),
            histories: Mutex::new(Vec::new()),
        };
        // First provider call: the turn's final answer; second call: the
        // reflection LLM's lesson list.
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "fix the build", term::cancel_flag());
        run_turn(&mut session);

        let msg = match dispatch_command(&agent, &mut session, &root, &memory_root, "/reflect") {
            CommandOutcome::Message(m) => m,
            _ => panic!("expected a message outcome"),
        };
        assert!(msg.contains("added 1 new lesson"), "{msg}");

        // The lesson landed in memory with provenance back to the session.
        let lessons = crab::memory::list_lessons(&memory_root, ws.root()).unwrap();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, "always run make first");
        assert_eq!(lessons[0].cwd, ws.root().to_string_lossy());
        assert!(lessons[0].source_session_id.is_some());
        // The session was auto-saved so provenance has a real id.
        assert!(session::load_previous(&root, ws.root()).unwrap().is_some());
    }

    #[test]
    fn reflect_with_no_conversation_reports_clearly() {
        let (dir, root, provider, tools, ws, cfg) = setup("reflect-empty");
        let memory_root = dir.join("memory");
        let agent = Agent::new(&provider, &tools, &ws, &cfg);
        let mut session = Session::new(&agent, "x", term::cancel_flag());
        let msg = match dispatch_command(&agent, &mut session, &root, &memory_root, "/reflect") {
            CommandOutcome::Message(m) => m,
            _ => panic!("expected a message outcome"),
        };
        assert!(msg.contains("no conversation to reflect on"), "{msg}");
    }
}
