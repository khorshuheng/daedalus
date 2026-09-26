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
PREFIX      ?= $(HOME)/.local
BINDIR      ?= $(PREFIX)/bin
CARGO_FLAGS ?=
RELEASE_DIR := target/release
BINS        := dl daedalus-server

.PHONY: all build link unlink clean test check fmt

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
