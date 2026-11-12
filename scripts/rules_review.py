#!/usr/bin/env python3
"""rules_review.py: write RULES-review.md, the rules of RULES.md that a script cannot check.

    python3 scripts/rules_review.py [--root REPO]           write RULES-review.md
    python3 scripts/rules_review.py --check [--root REPO]   warn (never fail) when it is stale or missing

RULES.md has three columns, ID, Rule, Check. The Check cell says who enforces the rule: `lint`, `test`
or `review`. A row stays when its Check cell contains the word `review`; every other row is dropped and
its id goes on one line under its table. Kept rows are copied as the original lines, never rebuilt, so
no rule is reworded. A pipe written `\\|` inside a cell is part of the cell, not a separator.

Exit codes: 0 done (also for --check, with or without a warning), 2 RULES.md missing when writing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GENERATED = (
    "Generated from RULES.md by scripts/rules_review.py. Do not edit. "
    "Rules checked by script are not listed; just lint prints the failing line."
)
STALE = "WARNING RULES-review.md is stale: run just rules-review"
REVIEW_RE = re.compile(r"\breview\b")
SPLIT_RE = re.compile(r"(?<!\\)\|")


def cells(line: str) -> list[str]:
    """The cells of a table row, split on pipes that are not escaped."""
    inner = re.sub(r"^\||(?<!\\)\|$", "", line.strip())
    return [c.strip() for c in SPLIT_RE.split(inner)]


def is_row(line: str) -> bool:
    return line.startswith("| ")


def generate(rules_text: str) -> str:
    """The text of RULES-review.md for the text of RULES.md."""
    lines = rules_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out: list[str] = []
    dropped: list[str] = []
    in_table = False
    i = 0

    def close_table() -> None:
        out.extend(["", "Checked by script, not listed: " + (", ".join(dropped) or "none") + "."])
        dropped.clear()

    while i < len(lines):
        line = lines[i]
        if in_table and not is_row(line):
            close_table()
            in_table = False
        if not in_table and is_row(line):
            in_table = True
            out.append(line)  # header row
            if i + 1 < len(lines) and lines[i + 1].startswith("|"):
                out.append(lines[i + 1])  # separator
                i += 1
        elif in_table:
            parts = cells(line)
            if len(parts) < 3 or REVIEW_RE.search(parts[2]):
                out.append(line)
            else:
                dropped.append(parts[0])
        else:
            out.append(line)
        i += 1
    if in_table:
        close_table()
    if out and out[0].startswith("# "):
        out[1:1] = ["", GENERATED]
    else:
        out[0:0] = [GENERATED, ""]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--check", action="store_true", help="warn when RULES-review.md is stale; exit 0")
    args = ap.parse_args()
    rules, target = args.root / "RULES.md", args.root / "RULES-review.md"
    if not rules.exists():
        if args.check:
            return 0
        print(f"rules_review: {rules} does not exist", file=sys.stderr)
        return 2
    fresh = generate(rules.read_bytes().decode("utf-8"))
    if args.check:
        if not target.exists() or target.read_bytes().decode("utf-8") != fresh:
            print(STALE)
        return 0
    target.write_bytes(fresh.encode("utf-8"))
    print(f"RULES-review.md: {len(fresh.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
