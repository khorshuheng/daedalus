# Makefile for the Daedalus Rust workspace.
#
# `make` builds both release binaries and points the symlinks in ~/.local/bin
# at them, matching the existing layout:
#
#   ~/.local/bin/dl              -> target/release/dl              (crates/daedalus)
#   ~/.local/bin/daedalus-server -> target/release/daedalus-server (crates/daedalus-server)
#
# Override as needed, e.g. `make BINDIR=/usr/local/bin` or
# `make CARGO_FLAGS=--locked`.

CARGO       ?= cargo
PYTHON      ?= python3
PREFIX      ?= $(HOME)/.local
BINDIR      ?= $(PREFIX)/bin
CARGO_FLAGS ?=
RELEASE_DIR := target/release
BINS        := dl daedalus-server

# rust-code-analysis CLI and the per-function cognitive-complexity ceiling
# enforced by `make metrics-check`; this is the only cognitive-complexity
# check. The ceiling passes on the current tree; ratchet it down as complex
# functions are split up.
RCA                ?= rust-code-analysis-cli
RCA_VERSION        ?= 0.0.25
COGNITIVE_THRESHOLD ?= 100

.PHONY: all build link unlink clean test check fmt lint \
        rca-install metrics metrics-check

## Build the binaries and refresh the symlinks (default target).
all: build link

## Build both workspace release binaries.
build:
	$(CARGO) build --release $(CARGO_FLAGS)

## Point $(BINDIR)/<bin> at the freshly built release binaries.
link: build
	@mkdir -p "$(BINDIR)"
	@for bin in $(BINS); do \
		ln -sfn "$(CURDIR)/$(RELEASE_DIR)/$$bin" "$(BINDIR)/$$bin"; \
		printf '  %s -> %s\n' "$(BINDIR)/$$bin" "$(CURDIR)/$(RELEASE_DIR)/$$bin"; \
	done
	@case ":$$PATH:" in \
		*":$(BINDIR):"*) ;; \
		*) echo "warning: $(BINDIR) is not on PATH" >&2 ;; \
	esac

## Remove the symlinks created by `link` (does not touch the binaries).
unlink:
	@for bin in $(BINS); do \
		if [ -L "$(BINDIR)/$$bin" ]; then rm -v "$(BINDIR)/$$bin"; \
		else echo "skip: $(BINDIR)/$$bin is not a symlink"; fi; \
	done

## Remove build artifacts.
clean:
	$(CARGO) clean

## Run the workspace test suite.
test:
	$(CARGO) test --workspace

## Type-check everything, including tests.
check:
	$(CARGO) check --workspace --all-targets

## Format the workspace.
fmt:
	$(CARGO) fmt --all

## Lint the workspace.
lint:
	$(CARGO) clippy --workspace --all-targets --all-features --tests -- -D warnings

## Install the rust-code-analysis CLI into $(BINDIR).
rca-install:
	@mkdir -p "$(BINDIR)"
	@case "$$(uname -s)-$$(uname -m)" in \
		Linux-x86_64)  url="https://github.com/mozilla/rust-code-analysis/releases/download/v$(RCA_VERSION)/rust-code-analysis-linux-cli-x86_64.tar.gz" ;; \
		Darwin-*)      url="https://github.com/mozilla/rust-code-analysis/releases/download/v$(RCA_VERSION)/rust-code-analysis-macos-cli-x86_64.tar.gz" ;; \
		*) echo "unsupported platform $$(uname -s)-$$(uname -m); install manually with 'cargo install rust-code-analysis'" >&2; exit 1 ;; \
	esac; \
	curl -fsSL "$$url" | tar -xz -C "$(BINDIR)"; \
	"$(BINDIR)/rust-code-analysis-cli" --version

## Report the most complex functions (rust-code-analysis).
metrics:
	$(PYTHON) scripts/cognitive-complexity.py --rca "$(RCA)"

## Fail if any function exceeds $(COGNITIVE_THRESHOLD) cognitive complexity.
metrics-check:
	$(PYTHON) scripts/cognitive-complexity.py --rca "$(RCA)" --threshold $(COGNITIVE_THRESHOLD)
