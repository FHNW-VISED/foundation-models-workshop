"""Assemble the Jupytext percent sources into the two notebooks participants open.

Agents and humans edit only the small source files; the .ipynb files are build artifacts.
Usage: uv run python scripts/build.py
    sections/NN_*.py  -> foundation-models-workshop.ipynb  (the workshop)
    setup_check.py    -> setup_check.ipynb                 (run the week before, SPEC §6 / N-030)
Outputs are stripped, except the saved HTML of callout cells (below); cell ids are deterministic so the
same sources give byte-identical notebooks.

Callouts (N-034, SPEC L-001): a markdown cell whose first line is `<!-- callout: kind | Title -->` becomes a
hidden form code cell (`# @title`) that displays the box, and the box HTML is also saved as that cell's
output, so it shows before anything runs. Colab cuts text inside markdown tables, so boxes are never tables.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import jupytext
import markdown
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
NAME = "foundation-models-workshop"
OUT = ROOT / f"{NAME}.ipynb"
SETUP_SRC = ROOT / "setup_check.py"
SETUP_OUT = ROOT / "setup_check.ipynb"

# kind -> (icon, default title, background, bar + text colour). SPEC L-001. `banner` is drawn apart.
KINDS = {
    "objective": ("🎯", "Objective", "#E8F5E9", "#1B5E20"),
    "takeaway": ("✅", "Takeaway", "#E0F2F1", "#004D40"),
    "theory": ("📘", "Theory", "#EDE7F6", "#311B92"),
    "your-turn": ("✏️", "Your turn", "#FFE8D6", "#7A2E00"),
    "question": ("❓", "Question", "#F3E5F5", "#4A148C"),
    "across": ("🔀", "Across approaches", "#E8EAF6", "#1A237E"),
    "pitfall": ("⚠️", "Pitfall", "#FDECEA", "#8B1A10"),
    "note": ("🧾", "Note", "#FFF8E1", "#5D4037"),
    "go-deeper": ("🔍", "Go deeper", "#F5F5F5", "#616161"),
}
BANNER_BG = "#4A148C"
CALLOUT_KINDS = ("banner", *KINDS)
MARKER_RE = re.compile(r"^<!--\s*callout:\s*([\w-]+)\s*(?:\|\s*(.*?))?\s*-->\s*$")
# A title "starts with an emoji" when its first character is in the symbol / pictograph blocks.
EMOJI_START_RE = re.compile(
    r"^([⌀-⯿\U0001F000-\U0001FAFF][️‍\U0001F3FB-\U0001F3FF]*)\s*(.*)$"
)


def parse_marker(source: str) -> tuple[str, str, str] | None:
    """(kind, title, body) if the cell's first line is a callout marker, else None."""
    first, _, body = source.partition("\n")
    m = MARKER_RE.match(first.strip())
    if not m:
        return None
    return m.group(1), (m.group(2) or "").strip(), body.strip("\n")


def render_body(body: str) -> str:
    """Markdown -> HTML. Markdown inside a <details> block is rendered too (md_in_html, part of `extra`)."""
    body = re.sub(r"<details(?![^>]*markdown=)", '<details markdown="1"', body)
    out = markdown.markdown(body, extensions=["extra", "sane_lists"]).strip()
    return out.replace("<a href=", '<a style="color:inherit;text-decoration:underline" href=')


def callout_html(kind: str, title: str, body: str) -> str:
    inner = render_body(body)
    if kind == "banner":
        return (
            f'<div style="background:{BANNER_BG};color:#FFFFFF;border-radius:6px;padding:18px 20px;'
            "margin:6px 0;width:100%;box-sizing:border-box;font-family:inherit;line-height:1.5\">"
            '<div style="font-size:2em;font-weight:bold;line-height:1.2;margin-bottom:6px">'
            f"{html.escape(title)}</div>{inner}</div>"
        )
    icon, default_title, bg, fg = KINDS[kind]
    title = title or default_title
    m = EMOJI_START_RE.match(title)
    if m:  # the title's own emoji replaces the kind's icon (👥 pair moments keep the question colour)
        icon, title = m.group(1), m.group(2)
    head = f"<b>{icon} {html.escape(title)}</b>"
    box = (
        f'<div style="background:{bg};border-left:6px solid {fg};border-radius:6px;padding:12px 14px;'
        f'margin:6px 0;color:{fg};font-family:inherit;line-height:1.5">'
    )
    if kind == "go-deeper":
        return f"{box}<details><summary>{head}</summary>{inner}</details></div>"
    return f"{box}{head}{inner}</div>"


def callout_cell(kind: str, title: str, body: str, where: str) -> nbformat.NotebookNode:
    """The hidden form code cell that draws the box, with the box HTML saved as its output."""
    if kind not in CALLOUT_KINDS:
        sys.exit(f"{where}: unknown callout kind '{kind}' (known: {', '.join(CALLOUT_KINDS)})")
    box = callout_html(kind, title, body)
    if '"""' in box or box.endswith("\\"):
        sys.exit(f"{where}: callout text cannot contain three double quotes or end with a backslash")
    src = f'# @title\nfrom IPython.display import HTML, display\ndisplay(HTML(r"""{box}"""))'
    cell = nbformat.v4.new_code_cell(src)
    cell.metadata["cellView"] = "form"
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"text/html": box, "text/plain": "<IPython.core.display.HTML object>"},
            metadata={},
        )
    ]
    return cell


def section_files() -> list[Path]:
    files = sorted(p for p in SECTIONS.glob("[0-9][0-9]_*.py"))
    if not files:
        sys.exit("no sections/NN_*.py files found")
    return files


def assemble(sources: list[Path], out: Path) -> Path:
    """Concatenate the cells of `sources` (percent format) into one notebook at `out`."""
    cells = []
    for path in sources:
        nb = jupytext.read(path, fmt="py:percent")
        for k, cell in enumerate(nb.cells):
            marker = parse_marker(cell.source) if cell.cell_type == "markdown" else None
            if marker:
                cell = callout_cell(*marker, where=f"{path.name} cell {k}")
            cell.id = f"{path.stem[:50]}-{k}"  # deterministic ids: same sources -> byte-identical .ipynb
            cell.metadata.pop("lines_to_next_cell", None)
            cell.metadata["section"] = path.stem
            # Colab collapses a code cell whose first line is "# @title ..." into a form.
            if cell.cell_type == "code" and cell.source.lstrip().startswith("# @title"):
                cell.metadata["cellView"] = "form"
            cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


def build() -> Path:
    return assemble(section_files(), OUT)


def build_setup_check() -> Path:
    if not SETUP_SRC.exists():
        sys.exit(f"{SETUP_SRC.name} not found")
    return assemble([SETUP_SRC], SETUP_OUT)


if __name__ == "__main__":
    out = build()
    print(f"built {out.name}: {len(section_files())} sections")
    print(f"built {build_setup_check().name}")
