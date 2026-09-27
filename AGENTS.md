# AGENTS.md

Guidance for agents working in this repository.

Daedalus is a minimal coding agent, modeled on the pi coding agent. It is a
Cargo workspace of three crates: an async agent engine, a CLI/TUI frontend, and
a headless server.

## Layout

- `crates/daedalus-core` — the library `daedalus_core`: config, workspace,
  credential, the built-in tools, the provider layer, session persistence,
  the `AgentRuntime` engine, skills, MCP, and paths.
- `crates/daedalus` — the CLI binary `dl`: argument parsing, the headless
  `--mode json` / `--mode rpc` adapters, and the ratatui TUI.
- `crates/daedalus-server` — the headless WebSocket server frontend over the
  same `AgentRuntime` protocol.

## Build, test, format

```sh
make check     # cargo check --workspace --all-targets
make test      # cargo test --workspace
make fmt       # cargo fmt --all
make build     # release binaries; `make link` symlinks dl + daedalus-server into ~/.local/bin
```

Run `cargo fmt --all` and `cargo check --workspace --all-targets` before
handing off a change. Tests live in-module under `#[cfg(test)]` and in
`crates/*/tests/`.

## Complexity metrics

`cargo clippy` is the linter and the build gate, but it deliberately does not
enforce cognitive complexity here. That check lives in
[rust-code-analysis][rca], applied through `make metrics`, which ranks functions
by cognitive and cyclomatic complexity, and `make metrics-check`, which fails
when any function exceeds `COGNITIVE_THRESHOLD` (default 100). The threshold
currently clears the tree, so ratchet it down as the worst functions are split
up. `make rca-install` drops the pinned release binary into `$(BINDIR)`; the
metrics tool is optional and never required to build or test.

[rca]: https://github.com/mozilla/rust-code-analysis

## Conventions

- **MSRV is Rust 1.94** (`rust-version` in the workspace manifest), edition
  2021. Keep dependency floors compatible with it.
- **Dependency direction:** frontends depend on `daedalus-core`; the core
  never depends on a frontend crate (`clap`, `ratatui`/`crossterm`, `axum`).
  A new frontend gets its own crate.
- **The core is async** (tokio, with a current-thread runtime owned by the
  worker thread). The library never uses `#[tokio::main]`; the crate stays
  runtime-agnostic at its boundaries.
- **Comments are prose.** Doc and line comments explain behavior and *why*;
  they do not cite internal ticket ids. Avoid adding comments which can be
  inferred from the code.
- **Tool descriptions are user-facing** (they ship to the model). Keep them
  accurate and terse.
- The built-in tools are `read`, `grep`, `find`, `bash`, `edit`, and `write`.
- The TUI is the interactive frontend, and
  headless use requires an explicit `--mode`.
