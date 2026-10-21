"""Mechanical checks for the rules in RULES.md that a script can decide.

Prints one line per problem (compact on purpose: agents read this output) and exits 1 on any FAIL.
Rule IDs refer to RULES.md.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import jupytext

from build import CALLOUT_KINDS, parse_marker

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
SPEC = ROOT / "SPEC.md"
SETUP_SRC = ROOT / "setup_check.py"
ANY_MARKER_RE = re.compile(r"<!--\s*callout\b")
TABLE_BOX_RE = re.compile(r"<table\b[^>]*\bbgcolor", re.I | re.S)

Q_RE = re.compile(r"\*\*Q(\d+)\.?\*\*")
LEVELS = ("🟢", "🟡", "🔴")
HELPER_MIN_LINES = 8

problems: list[str] = []


def fail(where: str, msg: str) -> None:
    problems.append(f"FAIL {where}: {msg}")


def is_helper_cell(src: str) -> bool:
    """A code cell that only imports / defines functions or classes (no top-level work)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    body = [n for n in tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    return (
        bool(body)
        and all(isinstance(n, defs) for n in body)
        and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in body)
    )


def check_callouts(where: str, src: str) -> bool:
    """N-034 checks on one markdown cell. True if the cell is a (well-placed) callout."""
    if TABLE_BOX_RE.search(src):
        fail(where, "coloured box as <table bgcolor>: Colab cuts it; use a callout marker (N-034)")
    marker = parse_marker(src)
    if marker is None:
        if ANY_MARKER_RE.search(src):
            fail(where, "callout marker must be the first line of its own markdown cell (N-034)")
        return False
    kind, title, _ = marker
    if kind not in CALLOUT_KINDS:
        fail(where, f"unknown callout kind '{kind}' (known: {', '.join(CALLOUT_KINDS)}) (N-034)")
    if "<" in title or "*" in title:
        fail(where, "callout title must be plain text: no '<' or '*' (N-034)")
    if len(ANY_MARKER_RE.findall(src)) > 1:
        fail(where, "one callout per cell (N-034)")
    return True


def has_objective(cells: list) -> bool:
    """N-003: '**Objective:**' in the heading cell, or an objective callout as the second cell
    (third when the second is the title banner)."""
    if "**Objective:**" in cells[0].source:
        return True
    kinds = [
        (m[0] if c.cell_type == "markdown" and (m := parse_marker(c.source)) else None) for c in cells[1:3]
    ]
    return kinds[:1] == ["objective"] or kinds[:2] == ["banner", "objective"]


def spec_rows() -> dict[str, str]:
    """sections/NN_x.py -> status, from SPEC.md's sections table."""
    if not SPEC.exists():
        fail("SPEC.md", "missing")
        return {}
    rows = {}
    for line in SPEC.read_text().splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*(sections/\d\d_[\w-]+\.py)\s*\|.*\|\s*([\w-]+)\s*\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2).lower()
    return rows


def main() -> int:
    files = sorted(SECTIONS.glob("[0-9][0-9]_*.py"))
    on_disk = {f"sections/{p.name}" for p in files}
    rows = spec_rows()
    listed = set(rows)
    for missing in sorted(listed - on_disk):
        if rows[missing] != "todo":  # unbuilt sections may be missing
            fail(missing, f"status '{rows[missing]}' in SPEC.md but the file does not exist (N-016)")
    for extra in sorted(on_disk - listed):
        fail(extra, "file exists but is not listed in SPEC.md sections table (N-016)")

    expected_q = 1
    smoke_defined = False
    for path in files:
        rel = f"sections/{path.name}"
        nb = jupytext.read(path, fmt="py:percent")
        cells = nb.cells
        if not cells or cells[0].cell_type != "markdown" or not cells[0].source.lstrip().startswith("#"):
            fail(rel, "first cell must be a markdown heading (N-003)")
        elif not has_objective(cells):
            fail(rel, "needs '**Objective:**' in the heading cell or an objective callout after it (N-003)")

        for i, cell in enumerate(cells):
            where = f"{rel} cell {i}"
            src = cell.source
            if cell.cell_type == "code":
                if re.search(r"^\s*SMOKE\s*=", src, re.M):
                    smoke_defined = True
                if src.lstrip().startswith("!pip"):
                    fail(where, "use %pip, not !pip (N-021)")
                n_lines = len(src.strip().splitlines())
                if (
                    n_lines >= HELPER_MIN_LINES
                    and is_helper_cell(src)
                    and not src.lstrip().startswith("# @title")
                ):
                    fail(
                        where,
                        f"helper-only cell ({n_lines} lines) needs '# @title ...' to collapse (N-008)",
                    )
                continue
            is_callout = check_callouts(where, src)
            if not is_callout and re.search(r"\bstyle\s*=", src):  # the build owns callout HTML
                fail(where, "markdown uses style= — Colab strips it; use a callout marker (N-025, N-034)")
            for line in src.splitlines():
                for m in Q_RE.finditer(line):
                    q = int(m.group(1))
                    if q != expected_q:
                        fail(where, f"question Q{q} out of sequence, expected Q{expected_q} (N-006)")
                    expected_q = q + 1
                    if not any(lv in line for lv in LEVELS):
                        fail(where, f"Q{q} has no level marker 🟢/🟡/🔴 on its line (N-005)")
                    if "```python" in src:
                        nxt = next((c for c in cells[i + 1 :] if c.cell_type == "code"), None)
                        if nxt is None or not nxt.source.lstrip().startswith(f"# Q{q}"):
                            fail(
                                where,
                                f"Q{q} shows code: next code cell must start with '# Q{q}' (N-007)",
                            )

    if SETUP_SRC.exists():
        for i, cell in enumerate(jupytext.read(SETUP_SRC, fmt="py:percent").cells):
            if cell.cell_type == "markdown":
                check_callouts(f"setup_check.py cell {i}", cell.source)

    if files and not smoke_defined:
        fail("sections", "no cell defines SMOKE = ... (N-014)")

    for p in problems:
        print(p)
    print(f"lint: {len(files)} sections, {expected_q - 1} questions, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
