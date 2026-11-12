#!/usr/bin/env python3
"""verify_her.py: prove that every difference between her copy and the new build is an approved item.

Her edited copy (an .ipynb) is compared cell by cell with the freshly built teaching notebook. Every
difference (a cell that changed, a cell of hers the build no longer has, a cell the build added) must be
matched by one line of `approved.md` that cites a real item of the meeting list `items.md`. The script only
counts. Whether an item really approves a difference is a person's judgement. Standard library only; it
builds nothing, edits nothing and writes only its report, on standard output.

    python3 scripts/verify_her.py --her HER.ipynb --built NAME.ipynb --folder reviews/<date>-her-copy [--list]

The folder holds `items.md` (the meeting items) and `approved.md`, one line per difference:

    <built cell id, or her:N> | <item id> | <reason of at least three words>

`her:N` is her cell number N, counted from 1 over her markdown and code cells; the id of a built cell is
`<section-stem>-<k>`. An item id is a token such as `I-03` or `C-24` (one to three capital letters, a dash,
one to three digits) that stands at the start of a line or after `|`, `·`, `/` or `,` in `items.md`. Several
item ids on one line may be given, separated by `/` or `,`. Lines starting with `#` are comments. A missing
`approved.md` means that nothing is approved.

Exit codes: 0 no unapproved difference and no unknown item, 1 otherwise, 2 unreadable input (or an
`items.md` without item ids).
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from port_her import BOX_RE, NbCell, align_cells, norm, read_notebook_cells  # noqa: E402

LIMITATION = (
    "Limitation: this compares the text of the cells. It does not compare run outputs; those need "
    "Simone's Colab run."
)
CUT = 300  # characters of a text shown under --list
SHOWN = 60  # characters of a text shown on an UNAPPROVED line
MIN_REASON_WORDS = 3

ID = r"[A-Z]{1,3}-\d{1,3}"
ITEM_RE = re.compile(rf"(?:^|[|·/,])[ \t]*(?:[-*][ \t]+)?\**({ID})\b")
ID_ONLY_RE = re.compile(ID)
TAG_RE = re.compile(r"<[^>]*>")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
HTML_CELL_RE = re.compile(r'HTML\(r?"""(.*?)"""\)', re.S)
HTML_START_RE = re.compile(r"^\s*<(?:div|details|table|section|span|p)\b", re.I)


class InputError(Exception):
    pass


@dataclass
class Difference:
    key: str  # built cell id, or `her:N`
    kind: str  # changed | her-only | built-only
    cell_type: str
    her_text: str | None
    built_text: str | None


def is_box(cell: NbCell) -> bool:
    """A callout or diagram box: a marker cell, or the HTML the build made of it."""
    text = cell.text
    return bool(BOX_RE.match(text.lstrip()) or HTML_START_RE.match(text) or HTML_CELL_RE.search(text))


def visible_text(text: str) -> str:
    """What a student sees of a box: tags and comments removed, entities decoded, whitespace collapsed."""
    found = HTML_CELL_RE.search(text)
    if found:
        text = found.group(1)
    text = TAG_RE.sub(" ", COMMENT_RE.sub(" ", text))
    return " ".join(html.unescape(text).split())


def cell_key(cell: NbCell) -> tuple[str, str]:
    """The comparison key. A box is compared by its visible text alone, its cell type does not count."""
    if is_box(cell):
        return ("box", visible_text(cell.text))
    return (cell.cell_type, norm(cell.text))


def read_cells(path: str | Path, what: str) -> list[NbCell]:
    try:
        return read_notebook_cells(path)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise InputError(f"cannot read {what} notebook {path}: {exc}") from exc


def find_differences(her: list[NbCell], built: list[NbCell]) -> tuple[list[Difference], int]:
    """Align her cells with the built cells; return the differences and the number of identical cells."""
    pairs = align_cells([cell_key(c) for c in her], [cell_key(c) for c in built])
    diffs: list[Difference] = []
    identical = 0
    for hi, bi in pairs:
        if hi is not None and bi is not None:
            h, b = her[hi], built[bi]
            if cell_key(h) == cell_key(b):
                identical += 1
            else:
                diffs.append(Difference(built_key(b, bi), "changed", b.cell_type, h.text, b.text))
        elif hi is not None:
            h = her[hi]
            diffs.append(Difference(f"her:{hi + 1}", "her-only", h.cell_type, h.text, None))
        elif bi is not None:
            b = built[bi]
            diffs.append(Difference(built_key(b, bi), "built-only", b.cell_type, None, b.text))
    return diffs, identical


def built_key(cell: NbCell, position: int) -> str:
    return cell.id or f"built:{position + 1}"


def read_items(folder: Path) -> set[str]:
    path = folder / "items.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    ids = {m.group(1) for line in text.splitlines() for m in ITEM_RE.finditer(line)}
    if not ids:
        raise InputError("items.md has no item ids")
    return ids


@dataclass
class Approval:
    line: int
    key: str
    items: list[str]
    reason: str


def read_approved(folder: Path, items: set[str]) -> tuple[list[Approval], list[str], list[str]]:
    """The approved lines that can cover a difference, the UNKNOWN ITEM notes and the BAD LINE notes."""
    path = folder / "approved.md"
    if not path.is_file():
        return [], [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    good: list[Approval] = []
    unknown: list[str] = []
    bad: list[str] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3 or not parts[0]:
            bad.append(f"BAD LINE {number}: expected `<cell id> | <item id> | <reason>`")
            continue
        key, item_field, reason = parts
        ids = [t.strip("*") for t in re.split(r"[\s/,]+", item_field) if t.strip("*")]
        if not ids or not all(ID_ONLY_RE.fullmatch(t) for t in ids) or any(t not in items for t in ids):
            unknown.append(f"UNKNOWN ITEM {number}: {item_field or '(empty)'} is not an item in items.md")
            continue
        if len(reason.split()) < MIN_REASON_WORDS:
            bad.append(f"BAD LINE {number}: the reason needs at least {MIN_REASON_WORDS} words")
            continue
        good.append(Approval(number, key, ids, reason))
    return good, unknown, bad


def one_line(text: str | None, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def report(her, built, diffs, identical, approvals, unknown, bad, show_list: bool) -> tuple[str, int]:
    approved_keys = {a.key for a in approvals}
    covered = [d for d in diffs if d.key in approved_keys]
    unapproved = [d for d in diffs if d.key not in approved_keys]
    out = [
        LIMITATION,
        f"her cells: {len(her)}",
        f"built cells: {len(built)}",
        f"identical: {identical}",
        f"differences: {len(diffs)}",
        f"approved: {len(covered)}",
        f"unapproved: {len(unapproved)}",
    ]
    for d in unapproved:
        shown = one_line(d.built_text if d.built_text is not None else d.her_text, SHOWN)
        out.append(f"UNAPPROVED {d.key} {d.kind}/{d.cell_type} {shown}")
    out.extend(unknown)
    out.extend(bad)
    diff_keys = {d.key for d in diffs}
    out.extend(f"UNUSED {a.line}: {a.key} matches no difference" for a in approvals if a.key not in diff_keys)
    if show_list:
        for d in unapproved:
            out.append("")
            out.append(f"--- {d.key} ({d.kind}, {d.cell_type})")
            out.append(f"her:   {one_line(d.her_text, CUT) if d.her_text is not None else '(none)'}")
            out.append(f"built: {one_line(d.built_text, CUT) if d.built_text is not None else '(none)'}")
            out.append(f"{d.key} | ITEM-ID | REASON IN AT LEAST THREE WORDS")
    return "\n".join(out) + "\n", 1 if unapproved or unknown else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--her", required=True, help="her edited copy (.ipynb)")
    parser.add_argument("--built", required=True, help="the freshly built teaching notebook (.ipynb)")
    parser.add_argument("--folder", required=True, help="folder with items.md and approved.md")
    parser.add_argument("--list", action="store_true", help="list each unapproved difference in full")
    args = parser.parse_args(argv)
    try:
        folder = Path(args.folder)
        items = read_items(folder)
        her = read_cells(args.her, "her")
        built = read_cells(args.built, "built")
        approvals, unknown, bad = read_approved(folder, items)
    except InputError as exc:
        print(f"verify_her: {exc}", file=sys.stderr)
        return 2
    diffs, identical = find_differences(her, built)
    text, code = report(her, built, diffs, identical, approvals, unknown, bad, args.list)
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
