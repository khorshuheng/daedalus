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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crab::agent::{Agent, AgentError, Session, Stream, Turn};
use crab::config::{Config, Overrides, ProviderKind};
use crab::provider;
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

/// Persist the current session history to disk (auto-save on exit, CRAB-109).
/// A failure to save is a warning only — quitting must never be blocked by
/// persistence.
fn auto_save(agent: &Agent, session: &Session) {
    let root = session::default_root();
    match session::save_session(&root, agent.workspace_root(), session.history()) {
        Ok(path) => eprintln!("session saved: {}", path.display()),
        Err(e) => eprintln!("crab: warning: could not save session: {e}"),
    }
}

/// The interactive REPL. Raw input mode is on for the whole session; a single
/// input thread line-edits and reports Line/Cancel/Eof. While a turn runs,
/// Esc/Ctrl-C cancel it; at the prompt, they terminate.
fn run_repl(agent: &Agent, initial: &str) -> Result<i32, String> {
    let _raw = term::RawMode::enable().map_err(|e| format!("cannot enable raw mode: {e}"))?;
    let busy = Arc::new(AtomicBool::new(false));
    let (rx, _input) = term::spawn_input(Arc::clone(&busy));

    let mut session = Session::new(agent, initial, term::cancel_flag());
    loop {
        term::clear_cancel();
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

        print!("> ");
        let _ = std::io::stdout().flush();
        let event = rx.recv().map_err(|_| "input closed".to_string())?;
        match event {
            term::InputEvent::Line(line) => match line.trim() {
                "/exit" | "/quit" | "/q" => {
                    auto_save(agent, &session);
                    return Ok(0);
                }
                "" => continue,
                msg => session.resume(msg.to_string()),
            },
            term::InputEvent::Cancel | term::InputEvent::Eof => {
                auto_save(agent, &session);
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
