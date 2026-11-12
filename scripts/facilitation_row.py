"""Copy the night agent's facilitation row from reports/latest-task.md into FACILITATION.md.

The night agent does not open FACILITATION.md (it is 23-40 KB). When an issue finishes a section it writes
that section's row, one markdown table line, under a `## Facilitation row` heading in reports/latest-task.md,
and night.sh calls this script. Standard library only.

    facilitation_row.py <latest-task.md> <FACILITATION.md>

The row replaces the line of the section table whose first cell matches (spaces stripped), or is appended
after the table's last row. Every other byte of FACILITATION.md stays as it was; no row is ever deleted.
Exit 0: nothing to do (no heading) or the file was written. Exit 1 with one line: the row is malformed
(wrong column count, no row, empty first cell) or the file or its section table is missing; nothing
is written.
"""

import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^##\s+Facilitation row\s*$", re.IGNORECASE)
ANY_HEADING_RE = re.compile(r"^#{1,6}\s")
SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")
CELL_SPLIT = re.compile(r"(?<!\\)\|")


class Fail(Exception):
    pass


def cells(line):
    """Cells of a table line, stripped. Raises Fail when it does not start and end with a pipe."""
    s = line.strip()
    if len(s) < 2 or not (s.startswith("|") and s.endswith("|")):
        raise Fail("facilitation row is not a table line (it must start and end with |)")
    return [c.strip() for c in CELL_SPLIT.split(s[1:-1])]


def extract_row(task_text):
    """The first data line of the table under `## Facilitation row`; None when the heading is absent."""
    lines = task_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if HEADING_RE.match(ln.strip())), None)
    if start is None:
        return None
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if ANY_HEADING_RE.match(s):
            break
        if not s.startswith("|") or SEPARATOR_RE.match(s):
            continue
        if cells(s)[0].lower() == "section":  # the agent copied the header line: not a data row
            continue
        return s
    raise Fail("no table row under '## Facilitation row'")


def find_table(lines):
    """(index of header line, index of last row) of the first table whose header starts with 'Section'."""
    for i, ln in enumerate(lines[:-1]):
        s = ln.strip()
        if not s.startswith("|") or not SEPARATOR_RE.match(lines[i + 1].strip()):
            continue
        try:
            head = cells(s)
        except Fail:
            continue
        if head[0].lower() != "section":
            continue
        last = i + 1
        while last + 1 < len(lines) and lines[last + 1].strip().startswith("|"):
            last += 1
        return i, last, len(head)
    return None


def apply(task_text, fac_text):
    """New text for FACILITATION.md, or None when there is no heading."""
    row = extract_row(task_text)
    if row is None:
        return None
    new_cells = cells(row)
    lines = re.findall(r"[^\n]*\n|[^\n]+", fac_text)  # \n only: no other separator may split a row
    bare = [ln.rstrip("\r\n") for ln in lines]
    table = find_table(bare)
    if table is None:
        raise Fail("no section table (header starting '| Section') in FACILITATION.md")
    head, last, width = table
    if len(new_cells) != width:
        raise Fail(f"facilitation row has {len(new_cells)} columns, the table has {width}")
    if not new_cells[0]:
        raise Fail("facilitation row has an empty first cell")
    for i in range(head + 2, last + 1):
        try:
            same = cells(bare[i])[0] == new_cells[0]
        except Fail:  # a hand-edited row without closing pipe: leave it alone
            same = False
        if same:
            eol = lines[i][len(bare[i]) :]
            lines[i] = row + eol
            return "".join(lines)
    own_eol = lines[last][len(bare[last]) :]
    if own_eol:
        lines.insert(last + 1, row + own_eol)
    else:  # the table ends the file without a newline: keep that ending, move the newline up one line
        lines[last] += "\n"
        lines.insert(last + 1, row)
    return "".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: facilitation_row.py <latest-task.md> <FACILITATION.md>", file=sys.stderr)
        return 1
    task, fac = Path(argv[0]), Path(argv[1])
    try:
        if not task.is_file():
            return 0
        task_text = task.read_text(encoding="utf-8")
        if not any(HEADING_RE.match(ln.strip()) for ln in task_text.splitlines()):
            return 0
        if not fac.is_file():
            raise Fail(f"{fac} not found")
        with open(fac, encoding="utf-8", newline="") as fh:
            fac_text = fh.read()
        new_text = apply(task_text, fac_text)
        if new_text is None or new_text == fac_text:
            return 0
        with open(fac, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print(f"facilitation row written to {fac}")
        return 0
    except (Fail, UnicodeDecodeError, OSError) as exc:
        print(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
