#!/usr/bin/env python3
"""port_her.py: port her edited copy of the notebook into sections/*.py, mechanically.

Her copy is diffed against the frozen notebook she was sent (the base). A change that can be mapped to one
section cell without a guess is written into the section file; anything else is listed as a conflict. Her
text always wins and her structure and style are never changed by a guess. Standard library only: it never
imports build, jupytext or nbformat, never builds anything and never writes an .ipynb.

    python3 scripts/port_her.py --her HER.ipynb --base versions/vNN-.../NAME.ipynb [--dry-run]

The map from a base cell back to its section cell is the id build.py stamps on every built cell:
`<section-stem[:50]>-<k>`, k being the cell's position in its section file (every cell counts, also the ones
of the other edition), plus `metadata.section`. Commented magics (`# %pip ...`) are not un-commented here, so
a section that uses them shows up as diverged.

Exit codes: 0 nothing in conflict, 1 conflicts remain, 2 refused (sections moved on, or the written files
failed the read-back), 3 her copy shares too little with the base (wrong base, or a rewrite).
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_BUDGET = 26_000  # bytes per report file: the size of the real 2026-10-28 list of visible changes
CUT = 600  # characters of a before / after text shown in a report
PAIR_EQUAL_BLOCK = 0.5  # similarity to pair cells in order inside a replace block of equal length
PAIR_UNEQUAL_BLOCK = 0.6  # ... and inside an unequal block, greedily by best similarity
TOO_LITTLE = (
    "her copy shares too little with the base: probably the wrong base version, or a rewrite. "
    "Adopt it as the new base, or pass the right --base."
)

HEADER_RE = re.compile(r"^# %%(?:\s|$)")
EDITION_RE = re.compile(r"""\bedition=["']([^"']*)["']""")
MARKDOWN_RE = re.compile(r"\[(?:markdown|md)\]")
RAW_RE = re.compile(r"\[raw\]")
BOX_RE = re.compile(r"^<!--\s*(?:callout|diagram):")  # a marker the build turns into an HTML code cell
SKIP_MARKER_RE = re.compile(r"^# needs: (boxes|masks)[ \t]*\n?", re.M)  # build.py strip_skip_markers
DIRECTIVE_LINE_RE = re.compile(r"^# needs: (?:boxes|masks)[ \t]*$")


# ---- normalisation and readers ----------------------------------------------------------------------


def norm(text: str) -> str:
    """For comparing only: her text is written exactly as she typed it."""
    lines = [ln.rstrip(" ") for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].startswith("#@title"):
        lines[0] = "# @title" + lines[0][len("#@title"):]
    return "\n".join(lines)


@dataclass
class SectionCell:
    k: int  # position in the file, counting every cell from 0
    kind: str  # "markdown", "code" or "raw"
    header: str  # the `# %% ...` line, verbatim ("" for text before the first header)
    text: str  # markdown without its `# ` prefixes; code as written
    span: tuple[int, int]  # first line (the header) and last line of the cell, 0-based, inclusive
    edition: str | None = None
    body_start: int = 0  # first line of the body
    block_end: int = 0  # first line after the cell and the blank lines that follow it


def _strip_comment(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line.rstrip() == "#":
        return ""
    return line


def parse_section_text(text: str) -> list[SectionCell]:
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if HEADER_RE.match(ln)]
    cells: list[SectionCell] = []
    first = starts[0] if starts else len(lines)
    pre = 0
    if lines and lines[0].rstrip() == "# ---":  # a jupytext YAML header is not a cell
        for i in range(1, first):
            if lines[i].rstrip() == "# ---":
                pre = i + 1
                break
    body = [i for i in range(pre, first) if lines[i].strip()]
    bounds: list[tuple[int, int | None]] = []
    if body:  # jupytext makes text before the first header a code cell
        bounds.append((-1, body[0]))
    for s in starts:
        bounds.append((s, None))
    for k, (header_idx, content_start) in enumerate(bounds):
        if header_idx == -1:
            lo = content_start
            nxt = starts[0] if starts else len(lines)
            header = ""
            kind = "code"
            edition = None
            body_start = lo
        else:
            lo = header_idx
            nxt = next((s for s in starts if s > header_idx), len(lines))
            header = lines[header_idx]
            kind = "markdown" if MARKDOWN_RE.search(header) else "raw" if RAW_RE.search(header) else "code"
            m = EDITION_RE.search(header)
            edition = m.group(1) if m else None
            body_start = header_idx + 1
        last = nxt - 1
        while last > lo and not lines[last].strip():
            last -= 1
        body_lines = lines[body_start:last + 1] if last >= body_start else []
        if kind == "markdown":
            body_lines = [_strip_comment(ln) for ln in body_lines]
        cell_text = "\n".join(ln.rstrip("\r") for ln in body_lines)
        cells.append(SectionCell(k, kind, header, cell_text, (lo, last), edition, body_start, nxt))
    return cells


def read_section_cells(path) -> list[SectionCell]:
    """The cells of one percent-format section file, with their line ranges (imported by N06 and N09)."""
    return parse_section_text(Path(path).read_bytes().decode("utf-8"))


# ---- section cell as the build makes it -------------------------------------------------------------
# The text transforms between a section cell and the cell build.py writes into the notebook. Keep them
# here, in one place, and keep them in step with scripts/build.py (template, EDA, YOLO and workshop all
# do the same):
#   1. jupytext (read_cells / assemble) un-comments magics and shell lines of code cells: `# %pip ...`
#      becomes `%pip ...`. The rules below mirror jupytext/magics.py (percent format, python).
#   2. build.py `strip_skip_markers` (Teaching Edition): `# needs: boxes|masks` lines are removed.
#   3. read_cells rstrips code cells (norm covers it).
# Markers that turn a whole markdown cell into an HTML code cell (`<!-- callout: ... -->`, the workshop's
# `<!-- diagram: ... -->`) are not text transforms: preflight marks those cells "built".

_C = r"(?:# |#)*"
_MAGIC_FORCE_ESC = re.compile(r"^\s*" + _C + r"(%|%%|%%%)[a-zA-Z](.*)#\s*escape")
_MAGIC_NOT_ESC = re.compile(r"^\s*" + _C + r"(%|%%|%%%)[a-zA-Z](.*)#\s*noescape")
_MAGIC = re.compile(r"^\s*" + _C + r"(%|%%|%%%)[a-zA-Z]")
_HELP_OR_BASH = re.compile(r"^\s*(?:(?:# |#)+\s*)?(\?|!)\s*[A-Za-z\.\~\$\\\/\{\}]")
_BASH_WORDS = "cat|cd|cp|mv|rm|rmdir|mkdir|copy|ddir|echo|ls|ldir|ren"
_MAGIC_CMD = re.compile(r"^" + _C + "(" + _BASH_WORDS + r")($|\s$|\s[^=,])")
_MAGIC_HELP = re.compile(r"^\s*(# )*[^\s]*\?\s*$")
_MAGIC_ASSIGN = re.compile(r"^" + _C + r"\s*([a-zA-Z_][a-zA-Z_$0-9]*)\s*=\s*(%|%%|%%%|!)[a-zA-Z](.*)")
_CONTINUATION = re.compile(r".*\\\s*$")


def _is_magic(line: str) -> bool:
    if _MAGIC_FORCE_ESC.match(line):
        return True
    if _MAGIC_NOT_ESC.match(line):
        return False
    return bool(
        _MAGIC.match(line)
        or _HELP_OR_BASH.match(line)
        or _MAGIC_ASSIGN.match(line)
        or _MAGIC_HELP.match(line)
        or _MAGIC_CMD.match(line)
    )


class _Quotes:
    """Which lines start inside a string (jupytext's StringParser, python only); magics there stay."""

    def __init__(self):
        self.single = None
        self.triple = None
        self.triple_start = -1

    def quoted(self) -> bool:
        return bool(self.single or self.triple)

    def read(self, line: str) -> None:
        if not self.quoted() and line.lstrip().startswith("#"):
            return
        self.triple_start = -1
        for i, ch in enumerate(line):
            if self.single is None and self.triple is None and ch == "#":
                break
            if ch not in "\"'":
                continue
            if line[i - 1 : i] == "\\":
                continue
            if self.single == ch:
                self.single = None
                continue
            if self.single is not None:
                continue
            if line[i - 2 : i + 1] == 3 * ch and i >= self.triple_start + 3:
                if self.triple == ch:
                    self.triple, self.triple_start = None, i
                elif self.triple is None:
                    self.triple, self.triple_start = ch, i
                continue
            if self.triple is not None:
                continue
            self.single = ch
        self.single = None


def _magic_flags(lines: list[str]) -> list[bool]:
    flags, quotes, nxt = [], _Quotes(), False
    for line in lines:
        hit = not quotes.quoted() and (nxt or _is_magic(line))
        flags.append(hit)
        if hit:
            nxt = bool(_CONTINUATION.match(line))
        quotes.read(line)
    return flags


def uncomment_magic(lines: list[str]) -> list[str]:
    """`# %pip x` -> `%pip x`, as jupytext does when it reads a percent-format section."""
    out = list(lines)
    for n, hit in enumerate(_magic_flags(lines)):
        if hit:
            body = lines[n].lstrip()
            indent = lines[n][: len(lines[n]) - len(body)]
            if body.startswith("# "):
                out[n] = indent + body[2:]
            elif body.startswith("#"):
                out[n] = indent + body[1:]
    return out


def comment_magic(lines: list[str]) -> list[str]:
    """`%pip x` -> `# %pip x`: how a code cell with magics is stored in a section file."""
    out = list(lines)
    quotes, nxt = _Quotes(), False
    for n, line in enumerate(lines):
        if not quotes.quoted() and (nxt or _is_magic(line)):
            if nxt:
                out[n] = "# " + line
            else:
                body = line.lstrip()
                out[n] = line[: len(line) - len(body)] + "# " + body
            nxt = bool(_CONTINUATION.match(line))
        quotes.read(line)
    return out


def as_built(cell: SectionCell, edition: str = "teaching") -> str:
    """The text build.py puts in the notebook for this section cell (before norm). Markdown is unchanged."""
    if cell.kind != "code":
        return cell.text
    text = "\n".join(uncomment_magic(cell.text.split("\n")))
    if edition == "teaching":
        text = SKIP_MARKER_RE.sub("", text)
    return text



@dataclass
class NbCell:
    index: int  # position among the markdown and code cells of the notebook
    cell_type: str
    text: str
    id: str | None = None
    section: str | None = None
    attachments: bool = False


def read_notebook_cells(path) -> list[NbCell]:
    """The markdown and code cells of an .ipynb as plain data. Raw cells are skipped."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for raw in nb.get("cells", []):
        ctype = raw.get("cell_type")
        if ctype not in ("markdown", "code"):
            continue
        src = raw.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        cells.append(
            NbCell(
                len(cells),
                ctype,
                text,
                raw.get("id"),
                (raw.get("metadata") or {}).get("section"),
                bool(raw.get("attachments")),
            )
        )
    return cells


# ---- alignment --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _ratio(a: str, b: str) -> float:
    """Similarity of two cell texts on word tokens (cheap enough for a whole-notebook rewrite)."""
    ta, tb = _TOKEN_RE.findall(a), _TOKEN_RE.findall(b)
    if not ta and not tb:
        return 1.0
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    return sm.ratio()


def _pair_block(base_keys, her_keys, i1, i2, j1, j2) -> list[tuple[int, int]]:
    nb, nh = i2 - i1, j2 - j1
    if nb == nh:
        return [
            (i1 + n, j1 + n)
            for n in range(nb)
            if _ratio(base_keys[i1 + n][1], her_keys[j1 + n][1]) >= PAIR_EQUAL_BLOCK
        ]
    scored = []
    for i in range(i1, i2):
        for j in range(j1, j2):
            r = _ratio(base_keys[i][1], her_keys[j][1])
            if r >= PAIR_UNEQUAL_BLOCK:
                scored.append((r, i, j))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    taken: list[tuple[int, int]] = []
    for _r, i, j in scored:  # best first; never cross a pair already taken, so order is kept
        if all((i < ti and j < tj) or (i > ti and j > tj) for ti, tj in taken):
            taken.append((i, j))
    return sorted(taken)


def align_cells(base_keys, her_keys) -> list[tuple[int | None, int | None]]:
    """Align two lists of keys `(cell_type, norm(text))`. Returns ordered pairs `(base index, her index)`:
    both set = the same cell (equal or edited), only base = deleted, only her = inserted."""
    base_keys, her_keys = list(base_keys), list(her_keys)
    sm = difflib.SequenceMatcher(None, base_keys, her_keys, autojunk=False)
    out: list[tuple[int | None, int | None]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend((i1 + n, j1 + n) for n in range(i2 - i1))
        elif tag == "delete":
            out.extend((i, None) for i in range(i1, i2))
        elif tag == "insert":
            out.extend((None, j) for j in range(j1, j2))
        else:
            pairs = _pair_block(base_keys, her_keys, i1, i2, j1, j2)
            ci, cj = i1, j1
            for pi, pj in pairs + [(i2, j2)]:
                out.extend((i, None) for i in range(ci, pi))
                out.extend((None, j) for j in range(cj, pj))
                if pi < i2:
                    out.append((pi, pj))
                ci, cj = pi + 1, pj + 1
    return out


# ---- the plan ---------------------------------------------------------------------------------------


@dataclass
class BaseInfo:
    cell: NbCell
    status: str  # "same", "built" (a box the build turned into HTML), "diverged", "unmapped"
    path: Path | None = None
    sec: SectionCell | None = None


@dataclass
class Edit:
    cls: str  # modified, deleted, inserted
    path: Path
    k: int  # section cell k; for an insert the cell it goes after
    before: str
    after: str
    kind: str  # cell kind of the written cell
    order: int = 0  # position inside a run of inserted cells


@dataclass
class Conflict:
    cls: str
    where: str
    her: str
    base: str
    need: str


@dataclass
class Plan:
    infos: list[BaseInfo]
    her: list[NbCell]
    base: list[NbCell]
    aligned: int = 0
    edits: list[Edit] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    diverged: list[BaseInfo] = field(default_factory=list)

    @property
    def share(self) -> float:
        if not self.her:
            return 1.0 if not self.base else 0.0
        return self.aligned / len(self.her)

    def count(self, cls: str) -> int:
        return sum(1 for e in self.edits if e.cls == cls)


NEEDS = {
    "box": "box: rewrite the box source to match her visible text",
    "type": "type: she changed a cell between markdown and code; say if intended, then edit it by hand",
    "boundary": "boundary: say which section the new cells belong to",
    "moved": "moved: say if the cell moved or was deleted and added again; neither change was applied",
    "attachment": "attachment: her markdown cell carries an image; save the image and write the cell by hand",
    "clash": "clash: section and her copy both changed this cell; her text was applied, check the section's",
    "unmapped": "unmapped: the section cell of this base cell is gone; place her change by hand",
}


def load_sections(root: Path) -> dict[str, tuple[Path, list[SectionCell]]]:
    out = {}
    for p in sorted((root / "sections").glob("[0-9][0-9]_*.py")):
        out[p.stem[:50]] = (p, read_section_cells(p))
    return out


def preflight(base: list[NbCell], root: Path, edition: str) -> list[BaseInfo]:
    sections = load_sections(root)
    infos = []
    for cell in base:
        info = BaseInfo(cell, "unmapped")
        m = re.fullmatch(r"(.+)-(\d+)", cell.id or "")
        if m:
            stem = (cell.section or m.group(1))[:50]
            found = sections.get(stem)
            k = int(m.group(2))
            if found and found[0].stem[:50] + f"-{k}" == cell.id and k < len(found[1]):
                sec = found[1][k]
                if sec.edition in (None, edition):
                    info.path, info.sec = found[0], sec
                    if norm(as_built(sec, edition)) == norm(cell.text):
                        info.status = "same"
                    elif sec.kind == "markdown" and BOX_RE.match(sec.text.lstrip("\n")):
                        info.status = "built"
                    else:
                        info.status = "diverged"
        infos.append(info)
    return infos


def _where(info: BaseInfo, root: Path) -> str:
    if info.path is None or info.sec is None:
        return f"{info.cell.id or '(no id)'}"
    return f"sections/{info.path.name} cell {info.sec.k}"


def build_plan(base, her, infos, root: Path) -> Plan:
    plan = Plan(infos, her, base)
    plan.diverged = [i for i in infos if i.status in ("diverged", "unmapped")]
    pairs = align_cells(
        [(c.cell_type, norm(c.text)) for c in base], [(c.cell_type, norm(c.text)) for c in her]
    )
    her_to_base: dict[int, int] = {j: i for i, j in pairs if i is not None and j is not None}
    deleted = [i for i, j in pairs if j is None]
    inserted = [j for i, j in pairs if i is None]
    plan.aligned = len(her_to_base)

    def conflict(cls, where, her_text, base_text):
        plan.conflicts.append(Conflict(cls, where, her_text, base_text, NEEDS[cls]))

    for j, i in sorted(her_to_base.items()):
        b, h, info = base[i], her[j], infos[i]
        if norm(b.text) == norm(h.text) and b.cell_type == h.cell_type:
            continue
        where = _where(info, root)
        if b.cell_type != h.cell_type:
            conflict("type", where, h.text, b.text)
        elif info.status == "built":
            conflict("box", where, h.text, b.text)
        elif h.attachments:
            conflict("attachment", where, h.text, b.text)
        elif info.sec is None or info.path is None:
            conflict("unmapped", where, h.text, b.text)
        else:
            plan.edits.append(Edit("modified", info.path, info.sec.k, info.sec.text, h.text, info.sec.kind))
            if info.status == "diverged":
                conflict("clash", where, h.text, info.sec.text)

    moved_ins: set[int] = set()
    moved_del: set[int] = set()
    for j in inserted:
        t = norm(her[j].text)
        if not t:
            continue
        for i in deleted:
            if i not in moved_del and norm(base[i].text) == t:
                moved_del.add(i)
                moved_ins.add(j)
                conflict("moved", _where(infos[i], root), her[j].text, base[i].text)
                break

    for i in deleted:
        if i in moved_del:
            continue
        info = infos[i]
        where = _where(info, root)
        if info.sec is None or info.path is None:
            conflict("unmapped", where, "", base[i].text)
            continue
        plan.edits.append(Edit("deleted", info.path, info.sec.k, info.sec.text, "", info.sec.kind))
        if info.status == "diverged":
            conflict("clash", where, "", info.sec.text)

    _plan_inserts(plan, her, base, infos, her_to_base, inserted, moved_ins, root, conflict)
    plan.edits.sort(key=lambda e: (e.path.name, e.k, e.order, e.cls))
    return plan


def _plan_inserts(plan, her, base, infos, her_to_base, inserted, moved_ins, root, conflict) -> None:
    runs: list[list[int]] = []
    for j in inserted:
        if runs and runs[-1][-1] == j - 1:
            runs[-1].append(j)
        else:
            runs.append([j])
    for run in runs:
        anchor_j = next((j for j in range(run[0] - 1, -1, -1) if j in her_to_base), None)
        anchor_i = her_to_base[anchor_j] if anchor_j is not None else None
        keep = []
        for j in run:
            if j in moved_ins:
                continue
            if her[j].attachments and her[j].cell_type == "markdown":
                conflict("attachment", "inserted cell", her[j].text, "")
            else:
                keep.append(j)
        if not keep:
            continue
        her_text = "\n\n".join(her[j].text for j in keep)
        if anchor_i is None:
            conflict("boundary", "before the first cell of the notebook", her_text, "")
            continue
        info = infos[anchor_i]
        where = f"after {_where(info, root)}"
        nxt = infos[anchor_i + 1] if anchor_i + 1 < len(infos) else None
        if info.sec is None or info.path is None:
            conflict("unmapped", where, her_text, base[anchor_i].text)
        elif nxt is not None and (nxt.cell.section or "") != (info.cell.section or ""):
            conflict("boundary", where, her_text, "")
        else:
            for n, j in enumerate(keep):
                plan.edits.append(
                    Edit("inserted", info.path, info.sec.k, "", her[j].text, her[j].cell_type, order=n)
                )


# ---- writing ----------------------------------------------------------------------------------------


def _body_lines(kind: str, text: str, keep: list[tuple[int, str]] = ()) -> list[str]:
    """The section-file lines of a cell written from her text. Markdown gets its `# ` prefixes; code gets its
    magic lines commented again (`%pip x` -> `# %pip x`) and the directive lines `keep` (index, line) that
    her cell cannot contain put back where the original section cell had them."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        return [("# " + ln) if ln.strip() else "#" for ln in lines]
    lines = comment_magic(lines)
    for idx, line in keep:
        lines.insert(min(idx, len(lines)), line)
    return lines


def _directives(cell: SectionCell, edition: str) -> list[tuple[int, str]]:
    if edition != "teaching" or cell.kind != "code":
        return []
    return [(i, ln) for i, ln in enumerate(cell.text.split("\n")) if DIRECTIVE_LINE_RE.match(ln)]


def edit_section_text(
    text: str, edits: list[Edit], edition: str = "teaching"
) -> tuple[str, list[tuple[str, str, str]]]:
    """The new text of one section file and the (header, kind, norm text) of every cell it must then have."""
    cells = parse_section_text(text)
    lines = text.split("\n")
    mods = {e.k: e for e in edits if e.cls == "modified"}
    dels = {e.k for e in edits if e.cls == "deleted"}
    ins: dict[int, list[Edit]] = {}
    for e in edits:
        if e.cls == "inserted":
            ins.setdefault(e.k, []).append(e)
    ops = []  # (start, priority, lo, hi, new lines); applied from the bottom of the file up
    expected = []
    tail = len(cells)  # cells from here to the end of the file are all deleted
    while tail > 0 and (tail - 1) in dels:
        tail -= 1
    if tail < len(cells):  # one cut from the end of the last kept cell, so the file still ends in a newline
        lo = cells[tail - 1].span[1] + 1 if tail else 0
        hi = len(lines) - 1 if lines and lines[-1] == "" else len(lines)
        ops.append((lo, 0, lo, hi, []))
    for cell in cells:
        if cell.k in dels:
            if cell.k < tail:
                ops.append((cell.span[0], 0, cell.span[0], cell.block_end, []))
        else:
            if cell.k in mods:
                e = mods[cell.k]
                new_body = _body_lines(cell.kind, e.after, _directives(cell, edition))
                ops.append((cell.body_start, 2, cell.body_start, cell.span[1] + 1, new_body))
                expected.append((cell.header, cell.kind, norm(e.after)))
            else:
                expected.append((cell.header, cell.kind, norm(as_built(cell, edition))))
            if cell.k in ins:
                new: list[str] = []
                for e in sorted(ins[cell.k], key=lambda x: x.order):
                    header = "# %% [markdown]" if e.kind == "markdown" else "# %%"
                    new += ["", header, *_body_lines(e.kind, e.after)]
                    expected.append((header, e.kind, norm(e.after)))
                at = cell.span[1] + 1
                ops.append((at, 1, at, at, new))
    for _start, prio, lo, hi, new in sorted(ops, key=lambda o: (-o[0], o[1])):
        if prio == 1 and lo < len(lines) and lines[lo].strip():
            new = new + [""]  # keep one blank line before the cell that follows
        lines[lo:hi] = new
    return "\n".join(lines), expected


def verify_written(expected: dict[Path, list[tuple[str, str, str]]], edition: str = "teaching") -> list[str]:
    """Read every written file back with the section reader: its cells, as built, must be the expected."""
    problems = []
    for path, want in expected.items():
        got = [(c.header, c.kind, norm(as_built(c, edition))) for c in read_section_cells(path)]
        if got != want:
            pairs = enumerate(zip(got, want, strict=False))
            n = next((i for i, (a, b) in pairs if a != b), min(len(got), len(want)))
            problems.append(f"{path.name}: read-back differs at cell {n} ({len(got)} read, {len(want)} due)")
    return problems


def write_sections(
    new_texts: dict[Path, str],
    expected: dict[Path, list[tuple[str, str, str]]],
    verify: Callable[[dict[Path, list[tuple[str, str, str]]]], list[str]] = verify_written,
) -> list[str]:
    """Write all files through temporary names, rename, read back. On any failure the originals come back."""
    originals = {p: p.read_bytes() for p in new_texts}
    tmps: dict[Path, Path] = {}
    problems: list[str] = []
    try:
        for p, t in new_texts.items():
            tmp = p.with_name(p.name + ".port_her.tmp")
            tmp.write_bytes(t.encode("utf-8"))
            tmps[p] = tmp
        for p, tmp in tmps.items():
            os.replace(tmp, p)
        tmps = {}
        problems = verify(expected)
    except OSError as err:
        problems = [f"write failed: {err}"]
    for tmp in tmps.values():
        tmp.unlink(missing_ok=True)
    if problems:
        for p, data in originals.items():
            p.write_bytes(data)
    return problems


# ---- reports ----------------------------------------------------------------------------------------


def cut_text(text: str, limit: int = CUT) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    return text[:limit] + f"\n(+{len(text) - limit} characters)", len(text) - limit


def _fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    mark = "`" * max(3, longest + 1)
    return f"{mark}\n{text}\n{mark}"


def _block(title: str, parts: list[tuple[str, str]], tail: str = "") -> tuple[str, int]:
    out, cut_total = [f"### {title}", ""], 0
    for label, text in parts:
        shown, cut = cut_text(text)
        cut_total += cut
        out += [f"{label}:", _fence(shown), ""]
    if tail:
        out += [tail, ""]
    return "\n".join(out), cut_total


def _listed(header_for: Callable[[int, int, int], str], blocks: list[tuple[str, int]]) -> str:
    """The header plus as many blocks as fit the budget; the header says how many were left out."""
    reserve = len(header_for(len(blocks), len(blocks), 10**7).encode("utf-8")) + 64
    used, n = reserve, 0
    for text, _cut in blocks:
        size = len(text.encode("utf-8")) + 1
        if used + size > REPORT_BUDGET:
            break
        used += size
        n += 1
    cut = sum(c for _t, c in blocks[:n])
    return header_for(len(blocks), n, cut) + "\n".join(t for t, _c in blocks[:n])


def render_reports(plan: Plan, base_file, her_file, edition: str, root: Path) -> tuple[str, str]:
    stat = (
        f"base: {base_file}\nher copy: {her_file}\nedition: {edition}\n"
        f"aligned: {plan.aligned} of {len(plan.her)} her cells ({plan.share:.0%})\n"
    )
    mod, dele, ins = plan.count("modified"), plan.count("deleted"), plan.count("inserted")
    cls_counts: dict[str, int] = {}
    for c in plan.conflicts:
        cls_counts[c.cls] = cls_counts.get(c.cls, 0) + 1
    conf_line = ", ".join(f"{n} {k}" for k, n in sorted(cls_counts.items())) or "none"

    diff_blocks = []
    for n, e in enumerate(plan.edits, 1):
        at = f"after cell {e.k}" if e.cls == "inserted" else f"cell {e.k}"
        parts = [("before", e.before), ("after", e.after)]
        diff_blocks.append(_block(f"A-{n:02d} {e.cls} sections/{e.path.name} {at}", parts))

    def diff_header(total: int, listed: int, cut: int) -> str:
        left = "all listed"
        if listed < total:
            left = f"not listed: {total - listed} (over the {REPORT_BUDGET:,}-byte budget)"
        return (
            f"# Her copy: applied changes\n\n{stat}applied: {mod} modified, {dele} deleted, {ins} inserted\n"
            f"conflicts: {conf_line} (see her-conflicts.md)\n"
            f"listed: {listed} of {total}; {left}\n"
            f"characters cut from before and after texts (at {CUT} each): {cut}\n\n"
        )

    conf_blocks = []
    for n, c in enumerate(plan.conflicts, 1):
        conf_blocks.append(
            _block(f"C-{n:02d} {c.cls} {c.where}", [("her", c.her), ("base", c.base)], f"needed: {c.need}")
        )

    def conf_header(total: int, listed: int, cut: int) -> str:
        left = f"not listed: {total - listed} (over the budget)" if listed < total else "all listed"
        return (
            f"# Her copy: conflicts, none of them applied except clashes\n\n{stat}"
            f"conflicts: {conf_line}\nlisted: {listed} of {total}; {left}\n"
            f"characters cut (at {CUT} each): {cut}\n\n"
        )

    return _listed(diff_header, diff_blocks), _listed(conf_header, conf_blocks)


# ---- command line -----------------------------------------------------------------------------------


def summary(plan: Plan) -> str:
    cls_counts: dict[str, int] = {}
    for c in plan.conflicts:
        cls_counts[c.cls] = cls_counts.get(c.cls, 0) + 1
    conf = ", ".join(f"{n} {k}" for k, n in sorted(cls_counts.items())) or "0"
    return (
        f"aligned: {plan.aligned} of {len(plan.her)} her cells ({plan.share:.0%}); "
        f"base cells: {len(plan.base)}\n"
        f"applied: {plan.count('modified')} modified, {plan.count('deleted')} deleted, "
        f"{plan.count('inserted')} inserted\nconflicts: {conf}"
    )


def main(argv=None, *, verify=None) -> int:
    ap = argparse.ArgumentParser(description="Port her edited copy into sections/*.py")
    ap.add_argument("--her", required=True, help="her edited .ipynb")
    ap.add_argument("--base", required=True, help="the frozen .ipynb she was sent (versions/vNN-.../*.ipynb)")
    ap.add_argument("--root", default=str(ROOT), help="the notebook repo (default: this one)")
    ap.add_argument("--edition", default="teaching", choices=["teaching", "own"])
    ap.add_argument("--allow-diverged", action="store_true", help="go on although sections moved on")
    ap.add_argument("--dry-run", action="store_true", help="print the reports, write nothing")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="YYYY-MM-DD, reports folder")
    ap.add_argument("--min-aligned", type=float, default=0.6)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    try:
        base = read_notebook_cells(args.base)
        her = read_notebook_cells(args.her)
    except (OSError, ValueError) as err:
        print(f"cannot read a notebook (nothing written): {err}")
        return 2
    infos = preflight(base, root, args.edition)
    stale = [i for i in infos if i.status in ("diverged", "unmapped")]
    if stale and not args.allow_diverged:
        print(f"the section files changed since the base was made: {len(stale)} of {len(base)} base cells")
        for n, i in enumerate(stale[:20], 1):
            print(f"C-{n:02d} diverged {i.cell.id or '(no id)'}")
        if len(stale) > 20:
            print(f"... and {len(stale) - 20} more (nothing written)")
        print("Pass the base her copy came from, or --allow-diverged.")
        return 2

    plan = build_plan(base, her, infos, root)
    if plan.share < args.min_aligned:
        print(summary(plan))
        print(TOO_LITTLE)
        return 3

    by_file: dict[Path, list[Edit]] = {}
    for e in plan.edits:
        by_file.setdefault(e.path, []).append(e)
    new_texts, expected = {}, {}
    for path, edits in by_file.items():
        old = path.read_bytes().decode("utf-8")
        new_texts[path], expected[path] = edit_section_text(old, edits, args.edition)

    diff_md, conf_md = render_reports(plan, args.base, args.her, args.edition, root)
    if args.dry_run:
        print(summary(plan))
        print("\n----- her-diff.md -----\n" + diff_md)
        print("\n----- her-conflicts.md -----\n" + conf_md)
        return 1 if plan.conflicts else 0

    if verify is None:

        def verify(exp):
            return verify_written(exp, args.edition)

    problems = write_sections(new_texts, expected, verify) if new_texts else []
    if problems:
        print("the written section files failed the read-back; the originals are back in place:")
        for p in problems:
            print("  " + p)
        return 2
    out = root / "reviews" / f"{args.date}-her-copy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "her-diff.md").write_text(diff_md, encoding="utf-8")
    (out / "her-conflicts.md").write_text(conf_md, encoding="utf-8")
    print(summary(plan))
    print(f"reports: {out}/her-diff.md, {out}/her-conflicts.md")
    return 1 if plan.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
