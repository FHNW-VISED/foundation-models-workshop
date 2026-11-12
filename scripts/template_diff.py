"""Compare a notebook repo's scripts with the vault template, byte for byte.

A repo gets its own copy of the helper scripts. Copies drift (the repos' night.sh all still named Opus),
so night.sh runs this first and refuses to start from a copy that differs. Only the files in MUST_MATCH
are compared: build.py, lint_notebook.py, version.py, justfile and AGENTS.md legitimately differ per repo.
Standard library only.

    template_diff.py [--repo DIR] [--template DIR]

Prints SAME, DIFF or MISSING per file. Exit 1 on any DIFF or MISSING; exit 0 with a warning when the
template folder does not exist (the check cannot run, it should not block).
"""

import argparse
import os
import sys
from pathlib import Path

MUST_MATCH = [
    "scripts/night.sh",
    "scripts/subagent_guard.py",
    "scripts/helper.sh",
    "scripts/handoff.py",
    "scripts/context_watch.py",
    "scripts/port_her.py",
    "scripts/verify_her.py",
    "scripts/sections_info.py",
    "scripts/rules_review.py",
    "scripts/template_diff.py",
    "scripts/facilitation_row.py",
    ".claude/settings.json",
]
DEFAULT_TEMPLATE = "/Users/simone.bonato/Desktop/second-brain/.claude/skills/notebook/template"


def compare(repo, template):
    """Return [(path, 'SAME'|'DIFF'|'MISSING')] for every file in MUST_MATCH."""
    out = []
    for rel in MUST_MATCH:
        mine, theirs = Path(repo) / rel, Path(template) / rel
        if not mine.is_file():
            out.append((rel, "MISSING"))
        elif not theirs.is_file():
            out.append((rel, "MISSING"))
        else:
            out.append((rel, "SAME" if mine.read_bytes() == theirs.read_bytes() else "DIFF"))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--template", default=os.environ.get("NB_TEMPLATE") or DEFAULT_TEMPLATE)
    args = parser.parse_args(argv)
    if not Path(args.template).is_dir():
        print(
            f"template_diff: WARNING template folder not found: {args.template}; check skipped",
            file=sys.stderr,
        )
        return 0
    results = compare(args.repo, args.template)
    for rel, state in results:
        print(f"{state:7} {rel}")
    bad = [rel for rel, state in results if state != "SAME"]
    if bad:
        print(
            f"template_diff: {len(bad)} of {len(results)} files differ from {args.template}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
