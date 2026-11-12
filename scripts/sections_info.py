#!/usr/bin/env python3
"""sections_info.py: look at the section files without a throwaway script. Reads only, writes nothing.

    python3 scripts/sections_info.py cells NN [--root REPO]   one line per cell of sections/NN_*.py
    python3 scripts/sections_info.py words [--root REPO]      visible words per section, and the total

`cells` prints `k kind bytes flags first-line`. k is the cell's position in the file, the same k build.py
puts in the cell id. kind is md, code or raw. flags (`-` for none): helper (first line `# @title`), box (a
`<!-- callout: -->` cell), diagram (a `<!-- diagram: -->` cell, the workshop's built HTML), ed=<edition>
(the cell belongs to one edition only; k counts both). It needs no jupytext.

`words` takes the counting rule and the budget from this repo's lint_notebook.py (visible_words,
total_visible_words, BUDGET_RE), so it cannot disagree with `just lint`. It needs jupytext (use `uv run`).

Exit codes: 0 done, 2 refused (unknown or duplicated section number, lint_notebook.py too old).
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from port_her import read_section_cells  # noqa: E402

KIND = {"markdown": "md", "code": "code", "raw": "raw"}
BOX_RE = re.compile(r"<!--\s*callout:")
DIAGRAM_RE = re.compile(r"<!--\s*diagram:")
CUT = 70


def find_section(root: Path, num: str) -> Path | str:
    """The one file sections/NN_*.py, or the one-line refusal."""
    nn = num.zfill(2) if num.isdigit() else num
    found = sorted((root / "sections").glob(f"{nn}_*.py"))
    if not found:
        return f"no section {nn} under {root / 'sections'}"
    if len(found) > 1:
        return f"section {nn} is not unique: " + ", ".join(p.name for p in found)
    return found[0]


def first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def cell_line(cell) -> str:
    first = first_line(cell.text)
    flags = []
    if first.startswith(("# @title", "#@title")):
        flags.append("helper")
    if BOX_RE.match(first):
        flags.append("box")
    if DIAGRAM_RE.match(first):
        flags.append("diagram")
    if cell.edition:
        flags.append(f"ed={cell.edition}")
    nbytes = len(cell.text.encode("utf-8"))
    kind = KIND.get(cell.kind, "raw")
    return f"{cell.k:>3} {kind:<4} {nbytes:>6} {','.join(flags) or '-'} {first[:CUT]}"


def cmd_cells(root: Path, num: str) -> int:
    path = find_section(root, num)
    if isinstance(path, str):
        print(path)
        return 2
    for cell in read_section_cells(path):
        print(cell_line(cell))
    return 0


def section_title(cells) -> str:
    for cell in cells:
        if cell.cell_type == "markdown":
            return first_line(cell.source).lstrip("# ").strip()
    return ""


def cmd_words(root: Path) -> int:
    import jupytext

    lint = importlib.import_module("lint_notebook")
    if not hasattr(lint, "visible_words"):
        print("lint_notebook.py has no visible_words: update it from the template (issue N14)")
        return 2
    all_cells = []
    for path in sorted((root / "sections").glob("[0-9][0-9]_*.py")):
        cells = jupytext.read(path, fmt="py:percent").cells
        all_cells += cells
        print(f"{path.name[:2]} {section_title(cells)}: {lint.visible_words(cells)}")
    budget_re, total_fn = getattr(lint, "BUDGET_RE", None), getattr(lint, "total_visible_words", None)
    if budget_re is None or total_fn is None:
        print("no total gate in lint_notebook.py")
        return 0
    spec = root / "SPEC.md"
    m = budget_re.search(spec.read_text(encoding="utf-8")) if spec.exists() else None
    if not m:
        print("no Word budget in SPEC.md")
        return 0
    budget, total = int(m.group(1).replace(",", "")), total_fn(all_cells)
    diff = f"+{total - budget} over" if total > budget else f"{budget - total} left"
    print(f"total: {total} of budget {budget} ({diff})")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", type=Path, default=HERE.parent, help="the notebook repo (default: this one)"
    )
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cells = sub.add_parser("cells", parents=[common], help="the cells of one section")
    p_cells.add_argument("nn", help="section number, e.g. 00")
    sub.add_parser("words", parents=[common], help="visible words per section against the Word budget")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    return cmd_cells(root, args.nn) if args.cmd == "cells" else cmd_words(root)


if __name__ == "__main__":
    sys.exit(main())
