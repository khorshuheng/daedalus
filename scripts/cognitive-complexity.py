#!/usr/bin/env python3
"""Report per-function code metrics for a Rust tree via rust-code-analysis.

rust-code-analysis ("rca") walks the source and emits one JSON object per
analysed file, each a tree of "spaces" (impls/traits) containing functions with
their metrics. This script flattens that tree, prints the functions that rank
highest by cognitive complexity, and optionally fails when one exceeds a
threshold.

It is a *metrics* companion to `cargo clippy`, not a replacement: clippy is the
linter that gates the build, this reports where the tree is getting hard to
read and can enforce a ceiling.

Examples:
    # print the 20 most complex functions
    scripts/cognitive-complexity.py --threshold 0

    # fail if any function is more complex than 40
    scripts/cognitive-complexity.py --threshold 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def fail(message: str) -> "NoReturn":  # noqa: F821 - kept terse on purpose
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def run_rca(rca: str, roots: list[str]) -> str:
    """Run rca over `roots` and return its JSON-lines output."""
    cmd = [
        rca,
        "-p",
        *roots,
        "-m",
        "-I",
        "*.rs",
        "-X",
        "*/target/*",
        "-O",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        fail(
            f"{rca!r} not found on PATH; run `make rca-install` or pass --rca"
        )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"exit status {proc.returncode}"
        fail(f"{rca} failed: {detail}")
    return proc.stdout


def walk(space: dict, file: str, prefix: str, out: list[dict]) -> None:
    """Recursively collect function metrics into `out`."""
    for child in space.get("spaces", []):
        name = child.get("name") or "?"
        qualified = f"{prefix}.{name}" if prefix else name
        if child.get("kind") == "function":
            metrics = child.get("metrics", {})
            out.append(
                {
                    "file": file,
                    "name": qualified,
                    "line": child.get("start_line", 0),
                    "cognitive": metrics.get("cognitive", {}).get("sum", 0.0),
                    "cyclomatic": metrics.get("cyclomatic", {}).get("sum", 0.0),
                    "sloc": metrics.get("loc", {}).get("sloc", 0.0),
                }
            )
        walk(child, file, qualified, out)


def parse(text: str) -> list[dict]:
    """Parse rca's JSON-lines output into a flat list of function records."""
    functions: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            root = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"could not parse rca output on line {lineno}: {exc}")
        walk(root, root.get("name", "?"), "", functions)
    return functions


def render(functions: list[dict], top: int) -> None:
    functions.sort(key=lambda f: (f["cognitive"], f["cyclomatic"]), reverse=True)
    shown = functions if top <= 0 else functions[:top]
    width = max((len(f["name"]) for f in shown), default=0)
    header = f"{'COGNITIVE':>9}  {'CYCLOMATIC':>10}  {'SLOC':>5}  FUNCTION"
    print(header)
    print("-" * len(header))
    for f in shown:
        print(
            f"{f['cognitive']:>9.0f}  {f['cyclomatic']:>10.0f}  {f['sloc']:>5.0f}"
            f"  {f['name']:<{width}}  ({f['file']}:{f['line']})"
        )
    print()
    print(f"{len(functions)} functions analysed; showing {len(shown)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rca",
        default=os.environ.get("RCA", "rust-code-analysis-cli"),
        help="rust-code-analysis CLI (default: $RCA or PATH lookup)",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=None,
        help="files or directories to analyse (default: crates)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="read rca JSON-lines from FILE instead of running the tool",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="exit non-zero if any function exceeds this cognitive complexity",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of functions to print; 0 prints all (default: 20)",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="also write the flattened metrics to FILE as JSON",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as handle:
            text = handle.read()
    else:
        roots = args.roots or ["crates"]
        text = run_rca(args.rca, roots)

    functions = parse(text)
    render(functions, args.top)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(functions, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.threshold > 0:
        over = [f for f in functions if f["cognitive"] > args.threshold]
        if over:
            print(
                f"\nerror: {len(over)} function(s) exceed the cognitive "
                f"complexity threshold of {args.threshold:g}:",
                file=sys.stderr,
            )
            for f in over:
                print(
                    f"  {f['cognitive']:.0f}  {f['name']}  "
                    f"({f['file']}:{f['line']})",
                    file=sys.stderr,
                )
            return 1
        print(f"\nall functions within the threshold of {args.threshold:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
