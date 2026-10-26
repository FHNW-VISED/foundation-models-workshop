"""Mechanical checks for the rules in RULES.md that a script can decide.

Prints one line per problem (compact on purpose: agents read this output) and exits 1 on any FAIL.
Rule IDs refer to RULES.md. Mentions of a question before it is asked (N-052) only print WARNING lines:
they never change the exit code.
"""

from __future__ import annotations

import ast
import builtins
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
Q_MENTION_RE = re.compile(r"\bQ(\d+)\b")
LEVELS = ("🟢", "🟡", "🔴")
HELPER_MIN_LINES = 8

problems: list[str] = []
warnings: list[str] = []


def fail(where: str, msg: str) -> None:
    problems.append(f"FAIL {where}: {msg}")


def forward_question_refs(line: str, next_q: int) -> list[int]:
    """N-052 (4): the question numbers a markdown line mentions before they are asked. `next_q` is the first
    question not asked yet; the line's own **Qn.** markers count as asked."""
    for m in Q_RE.finditer(line):
        next_q = max(next_q, int(m.group(1)) + 1)
    return [int(m.group(1)) for m in Q_MENTION_RE.finditer(Q_RE.sub(" ", line)) if int(m.group(1)) >= next_q]


def yourcode_stubs(src: str) -> list[ast.FunctionDef]:
    """The top-level functions of a '# ✍️ YOUR CODE' cell (the stubs the learner fills)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


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


SETTING_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")
SETTING_EXEMPT = {"RESULTS"}  # the smoke test's hook (SPEC §9), not a setting
MUTATORS = {
    "append",
    "extend",
    "insert",
    "update",
    "setdefault",
    "pop",
    "popitem",
    "clear",
    "remove",
    "add",
    "discard",
    "sort",
    "reverse",
}
BUILTIN_NAMES = set(dir(builtins))


def is_setting(name: str) -> bool:
    return bool(SETTING_RE.match(name)) and name not in SETTING_EXEMPT


def module_level(body: list[ast.stmt], path: tuple = ()):
    """(statement, branch path) for what runs at notebook level: the cell body and the inside of
    if/for/while/with/try, not of def/class. The path records the branch of each if/try a statement is in."""
    for node in body:
        yield node, path
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from module_level(getattr(node, field, []) or [], path + ((id(node), field),))
        for k, handler in enumerate(getattr(node, "handlers", []) or []):
            yield from module_level(handler.body, path + ((id(node), f"handler{k}"),))


def exclusive(a: tuple, b: tuple) -> bool:
    """True when two branch paths can never both run: they split at one if/try into different branches."""
    for (node_a, field_a), (node_b, field_b) in zip(a, b, strict=False):
        if node_a != node_b:
            return False
        if field_a != field_b:
            return "finalbody" not in (field_a, field_b)
    return False


def assigned_names(node: ast.stmt) -> list[str]:
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else [node.target]
        if isinstance(node, (ast.AnnAssign, ast.AugAssign))
        else []
    )
    return [
        n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    ]


def computed_from(value: ast.expr, imported: set[str]) -> str | None:
    """The first name a setting's value reads that makes it computed data rather than a setting: anything
    other than another setting, a builtin, an imported module/name, or a comprehension's own variable."""
    bound = {
        n.id
        for c in ast.walk(value)
        if isinstance(c, ast.comprehension)
        for n in ast.walk(c.target)
        if isinstance(n, ast.Name)
    }
    for n in ast.walk(value):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if not (is_setting(n.id) or n.id in bound or n.id in BUILTIN_NAMES or n.id in imported):
                return n.id
    return None


def check_settings(files: list[Path]) -> None:
    """N-058: UPPER_CASE names are settings, decided once. Not reassigned, not changed in place, not computed
    from the notebook's data or functions; a learner's ✏️ / @param setting is read only in its own section."""
    parsed = []  # (rel, cell index, cell source, tree)
    for path in files:
        for i, cell in enumerate(jupytext.read(path, fmt="py:percent").cells):
            if cell.cell_type != "code":
                continue
            # magics become blank lines, so line numbers still match the cell
            src = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line for line in cell.source.splitlines()
            )
            try:
                parsed.append((f"sections/{path.name}", i, cell.source, ast.parse(src)))
            except SyntaxError:
                continue
    imported = {
        (a.asname or a.name).split(".")[0]
        for _, _, _, tree in parsed
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for a in node.names
    }
    first_set: dict[str, tuple[str, int, tuple]] = {}  # setting -> (where, cell, branch path)
    learner_set: dict[str, str] = {}  # setting -> section of its ✏️ cell
    for rel, i, src, tree in parsed:
        where = f"{rel} cell {i}"
        src_lines = src.splitlines()
        for node, path in module_level(tree.body):
            # the learner's settings: everything in a ✏️ cell, or a Colab form line marked '# @param'
            learner = src.lstrip().startswith("# ✏️") or "# @param" in src_lines[node.lineno - 1]
            for name in [n for n in assigned_names(node) if is_setting(n)]:
                first = first_set.get(name)
                if first and first[1] == id(tree) and exclusive(first[2], path):
                    continue  # if/else alternatives in one cell: still one decision
                if isinstance(node, ast.AugAssign) or first:
                    fail(
                        where,
                        f"{name} is set again (first set in {first[0] if first else 'this cell'}); a "
                        "setting is decided once (N-058)",
                    )
                    continue
                first_set[name] = (where, id(tree), path)
                if learner:
                    learner_set[name] = rel
                elif node.value is not None and (culprit := computed_from(node.value, imported)):
                    fail(where, f"{name} is computed (reads {culprit}); computed data is lowercase (N-058)")
        for node in ast.walk(tree):
            changed = None
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                changed = node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATORS
            ):
                changed = node.func.value
            if isinstance(changed, ast.Name) and is_setting(changed.id):
                fail(where, f"{changed.id} is changed in place; a setting is decided once (N-058)")
    for rel, i, _, tree in parsed:
        for name in sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & set(learner_set)):
            if learner_set[name] != rel:
                fail(
                    f"{rel} cell {i}",
                    f"{name} is the learner's ✏️ setting in {learner_set[name]}; another "
                    "section needs its own setting (N-058)",
                )


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
                if src.lstrip().startswith("# ✍️ YOUR CODE"):
                    for fn in yourcode_stubs(src):
                        doc = ast.get_docstring(fn) or ""
                        if not re.search(r"\be\.g\.", doc):
                            fail(where, f"✍️ {fn.name} docstring needs an example return 'e.g. …' (N-050)")
                continue
            is_callout = check_callouts(where, src)
            if not is_callout and re.search(r"\bstyle\s*=", src):  # the build owns callout HTML
                fail(where, "markdown uses style= — Colab strips it; use a callout marker (N-025, N-034)")
            for line in src.splitlines():
                for q in forward_question_refs(line, expected_q):
                    warnings.append(f"WARNING {where}: mentions Q{q} before it is asked (N-052)")
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

    check_settings(files)

    for p in problems:
        print(p)
    for w in warnings:
        print(w)
    print(f"lint: {len(files)} sections, {expected_q - 1} questions, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
