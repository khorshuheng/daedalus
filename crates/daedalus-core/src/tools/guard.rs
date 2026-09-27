//! A static guard that refuses the shell search commands the system prompt
//! already bans, before `bash` runs them.
//!
//! The prompt says *"For content searches always use the `grep` tool and for
//! filename searches always use the `find` tool; never call
//! `grep`/`rg`/`find`/`fd` through bash"* — but nothing enforced it, so the
//! model kept shelling out (e.g. `grep -ril parthenon ~`, a whole-home scan
//! that hangs a turn). This guard makes the written rule machine-checked: if a
//! statement's *head* is a search tool, refuse it and name the replacement.
//!
//! Scope is deliberately narrow:
//!
//! - **Head of each statement only.** `git log | grep -i fix` heads with `git`
//!   and is bounded, so it is allowed: the rule bans running the tools, not
//!   filtering through them.
//! - **Fail open.** Anything the splitter cannot classify confidently runs. A
//!   wrapper the tokenizer does not know (a shell function, `sh -c '…'`, a
//!   checked-in script, a here-doc) is a gap by design.
//! - **Not a security boundary.** It steers the common reflex back to the
//!   bounded tools; it does not sandbox the shell.

/// Content-search tools: replaced by the `grep` tool.
const CONTENT_SEARCH: &[&str] = &["grep", "egrep", "fgrep", "rg", "ag", "ack"];
/// Filename-search tools: replaced by the `find` tool.
const NAME_SEARCH: &[&str] = &["find", "fd"];

/// Commands that merely wrap another command, dropped before the head is
/// classified (`sudo grep -r x` is still a search).
///
/// `xargs` is here on purpose, which costs one over-block: `git ls-files |
/// xargs grep foo` is refused even though a pipe stage would be allowed, and
/// the refusal points at the `grep` tool's `paths` argument — the shape we want
/// that command in anyway. The alternative, letting `xargs` through, would
/// also let `xargs fd` (recursive, unbounded) through.
const WRAPPERS: &[&str] = &[
    "sudo", "doas", "env", "nice", "ionice", "time", "timeout", "nohup", "exec", "xargs", "stdbuf",
    "setsid",
];

/// Shell keywords that introduce a statement, so the command after them is
/// still the head (`! grep -r x`, `if grep -q x f; then`).
const KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!",
];

/// Which bounded tool replaces the refused command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Content,
    Name,
}

/// One token of the command: a word with quotes resolved, or a statement
/// separator (`;`, `&&`, `||`, `&`, newline). A single `|` is *not* a
/// separator: a pipe stage is a bounded filter, and the guard judges only
/// each statement's head.
#[derive(Debug, PartialEq, Eq)]
enum Seg {
    Word(String),
    Sep,
}

/// The refusal for `command`, or `None` when it should run.
///
/// The message is the load-bearing part: it names the replacement tool and its
/// arguments, so a refusal re-routes the model rather than prompting a retry.
pub fn refuse_search(command: &str) -> Option<String> {
    for statement in statements(&segments(command)) {
        if let Some((tool, kind)) = head_search_tool(&statement) {
            return Some(refusal_message(tool, kind));
        }
    }
    None
}

/// Split the command into statements at separator tokens.
fn statements(segs: &[Seg]) -> Vec<Vec<&str>> {
    let mut out: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for seg in segs {
        match seg {
            Seg::Sep => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            Seg::Word(w) => current.push(w.as_str()),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// `(tool, kind)` when the statement's head is a banned search command.
///
/// Returns the constant name from [`CONTENT_SEARCH`]/[`NAME_SEARCH`], not the
/// model's spelling: `grep` and `/usr/bin/grep` refuse identically.
fn head_search_tool(words: &[&str]) -> Option<(&'static str, Kind)> {
    for word in words {
        // Skip leading assignments (`FOO=1 grep …`), wrappers, keyword
        // introductions and stray option flags. The first remaining word is
        // the command being run.
        if is_assignment(word)
            || is_wrapper(word)
            || is_keyword(word)
            || word.starts_with('-')
            || is_wrapper_value(word)
        {
            continue;
        }
        let base = word.rsplit('/').next().unwrap_or(word);
        if let Some(tool) = CONTENT_SEARCH.iter().find(|t| **t == base) {
            return Some((tool, Kind::Content));
        }
        if let Some(tool) = NAME_SEARCH.iter().find(|t| **t == base) {
            return Some((tool, Kind::Name));
        }
        // A non-search head: the statement runs.
        return None;
    }
    None
}

fn is_assignment(word: &str) -> bool {
    match word.split_once('=') {
        Some((name, _)) => {
            !name.is_empty()
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
}

fn is_wrapper(word: &str) -> bool {
    WRAPPERS.contains(&word)
}

fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word)
}

/// A bare value passed to a wrapper's option or taken positionally (`nice -n 5
/// grep`, `timeout 5 grep`). Command names are words, not digit-led numbers
/// (`7z` is the one real exception, and it is never followed by a search
/// tool), so skipping these cannot hide a head we mean to classify.
fn is_wrapper_value(word: &str) -> bool {
    if !word.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    // A bare number (`5`), or a number with a unit suffix (`5s`, `2M`).
    word.chars().all(|c| c.is_ascii_digit())
        || word
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .chars()
            .all(|c| c.is_ascii_alphabetic())
}

fn refusal_message(tool: &str, kind: Kind) -> String {
    match kind {
        Kind::Content => format!(
            "shell `{tool}` is not allowed. Run content searches with the `grep` tool \
             (`pattern`, plus optional `glob`/`paths`) instead. To search an explicit list of \
             files produced by another command (e.g. `git ls-files`), pass them to its `paths` \
             argument rather than piping into shell `grep`."
        ),
        Kind::Name => format!(
            "shell `{tool}` is not allowed. Run filename searches with the `find` tool \
             (`pattern`, plus optional `glob`/`paths`) instead. To filter an explicit list of \
             paths produced by another command, pass them to its `paths` argument rather than \
             piping into shell `find`."
        ),
    }
}

/// Tokenize `command` quote-aware: separators outside quotes split statements,
/// and quoting keeps its contents in the same word, so `echo "grep -r x"`
/// heads with `echo` and is allowed.
fn segments(command: &str) -> Vec<Seg> {
    // 0 = unquoted, 1 = single-quoted, 2 = double-quoted.
    let mut state = 0u8;
    let mut out = Vec::new();
    let mut word = String::new();
    let mut chars = command.chars().peekable();

    fn flush(word: &mut String, out: &mut Vec<Seg>) {
        if !word.is_empty() {
            out.push(Seg::Word(std::mem::take(word)));
        }
    }

    while let Some(c) = chars.next() {
        match state {
            1 => {
                if c == '\'' {
                    state = 0;
                } else {
                    word.push(c);
                }
            }
            2 => {
                if c == '"' {
                    state = 0;
                } else if c == '\\' {
                    if let Some(next) = chars.next() {
                        word.push(next);
                    }
                } else {
                    word.push(c);
                }
            }
            _ => match c {
                '\'' => {
                    state = 1;
                }
                '"' => {
                    state = 2;
                }
                '\\' => {
                    if let Some(next) = chars.next() {
                        word.push(next);
                    }
                }
                c if c.is_whitespace() => {
                    flush(&mut word, &mut out);
                    if c == '\n' {
                        out.push(Seg::Sep);
                    }
                }
                ';' => {
                    flush(&mut word, &mut out);
                    out.push(Seg::Sep);
                }
                // `&&` separates statements, and so does a bare `&`. A `&`
                // that belongs to a redirection does not: in `2>&1` and
                // `>&2` it follows a `>`, and in `&>file` it precedes one.
                '&' if word.ends_with('>') => word.push('&'),
                '&' if chars.peek() == Some(&'>') => word.push('&'),
                '&' => {
                    flush(&mut word, &mut out);
                    // `&&` is one separator, not two.
                    if chars.peek() == Some(&'&') {
                        chars.next();
                    }
                    out.push(Seg::Sep);
                }
                // A pipe stage is a filter, not a new command: `git log |
                // grep -i fix` stays one statement headed by `git`. `||` is a
                // real separator, so the command after it is judged.
                '|' => {
                    flush(&mut word, &mut out);
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        out.push(Seg::Sep);
                    }
                }
                _ => word.push(c),
            },
        }
    }
    flush(&mut word, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not refused: no statement's head is a shell search tool.
    fn allowed(command: &str) {
        assert_eq!(
            refuse_search(command),
            None,
            "expected `{command}` to run, but it was refused"
        );
    }

    /// Refused, with a message that names the replacement tool.
    fn refused(command: &str, mentions: &str) {
        let msg = refuse_search(command)
            .unwrap_or_else(|| panic!("expected `{command}` to be refused, but it was allowed"));
        assert!(
            msg.contains(mentions),
            "refusal for `{command}` should name `{mentions}`: {msg}"
        );
    }

    /// The command from the log that hung a turn: a home-directory scan as the
    /// first statement of a longer pipeline.
    #[test]
    fn refuses_the_logged_home_scan() {
        refused(
            "grep -ril parthenon /home/yggdrasil 2>/dev/null | head -20; echo \"---home---\"; ls -la /home/yggdrasil",
            "`grep` tool",
        );
    }

    /// The earlier logged incident: a filesystem-root walk.
    #[test]
    fn refuses_the_logged_root_walk() {
        refused(
            "find / -maxdepth 4 -name 'parthenon' -type d 2>/dev/null",
            "`find` tool",
        );
    }

    /// `rg`/`fd` recurse by default, so no `-r` flag is needed to refuse them.
    #[test]
    fn refuses_recursive_by_default_tools() {
        refused("rg parthenon", "`grep` tool");
        refused("fd parthenon", "`find` tool");
        refused("fd -e rs", "`find` tool");
    }

    #[test]
    fn refuses_every_banned_verb() {
        for cmd in ["grep x", "egrep x", "fgrep x", "rg x", "ag x", "ack x"] {
            refused(cmd, "`grep` tool");
        }
        for cmd in ["find . -name x", "fd x"] {
            refused(cmd, "`find` tool");
        }
    }

    /// A later statement is still checked, and quoted search words are not.
    #[test]
    fn checks_every_statement() {
        refused("cd /tmp && grep -rn x ~", "`grep` tool");
        refused("ls -la\nrg parthenon\n", "`grep` tool");
        refused("cat notes.txt | head -5; find / -name x", "`find` tool");
    }

    #[test]
    fn strips_wrappers_assignments_and_keywords() {
        refused("sudo grep -r x .", "`grep` tool");
        refused("env FOO=1 grep -r x .", "`grep` tool");
        refused("nice -n 5 rg x", "`grep` tool");
        refused("timeout 5s grep -r x .", "`grep` tool");
        refused("FOO=bar find /tmp -name x", "`find` tool");
        refused("xargs -0 fd x", "`find` tool");
        refused("! grep -r x .", "`grep` tool");
        refused("/usr/bin/grep -r x .", "`grep` tool");
    }

    /// Head-only: a bounded pipe stage is a legitimate filter, not a search.
    #[test]
    fn allows_search_tools_as_a_pipe_stage() {
        allowed("git ls-files | grep foo");
        allowed("git log | grep -i fix");
        allowed("git diff --name-only | rg '^src/'");
        allowed("cargo tree | grep -c '^├'");
    }

    /// A redirection's `&` is not a statement separator, so the pipe stage
    /// after `2>&1` is still a filter over the first command's output.
    #[test]
    fn allows_a_pipe_stage_after_a_stream_redirect() {
        allowed("cargo test 2>&1 | grep -i warning");
        allowed("ls -la 2>&1 | tail -5");
    }

    /// The redirection exception must not swallow real separators: `&&`, a
    /// bare `&` and `;` still start a new statement whose head is judged.
    #[test]
    fn still_splits_on_real_separators() {
        refused("cd /tmp && grep -r x .", "`grep` tool");
        refused("sleep 1 & grep -r x .", "`grep` tool");
        refused("true; find . -name x", "`find` tool");
    }

    /// Quoting and evaluation order keep the head something other than a tool.
    #[test]
    fn allows_quoted_or_evaluated_mentions() {
        allowed("echo grep -r x");
        allowed("echo 'grep -r x'");
        allowed("echo \"find / -name x\"");
        allowed("printf '%s\\n' rg");
        allowed("man grep");
        allowed("command -v grep");
    }

    /// Anything the splitter cannot classify runs (fail open, by design).
    #[test]
    fn fails_open_on_wrappers_it_cannot_see_through() {
        allowed("sh -c 'grep -r x .'");
        allowed("$(which grep) -r x .");
        allowed("bash ./scripts/scan.sh");
    }

    #[test]
    fn allows_ordinary_commands() {
        allowed("cargo test --workspace");
        allowed("ls -la && cat Cargo.toml");
        allowed("");
    }

    /// The refusal must point at the bounded tool, not just say "denied".
    #[test]
    fn refusal_names_the_replacement_and_its_arguments() {
        let msg = refuse_search("grep -ril needle /").unwrap();
        assert!(msg.contains("`grep` tool"), "{msg}");
        assert!(msg.contains("paths"), "{msg}");
        assert!(refuse_search("find / -name x")
            .unwrap()
            .contains("`find` tool"));
    }
}
