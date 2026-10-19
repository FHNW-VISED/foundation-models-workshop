"""Assemble sections/*.py (Jupytext percent format) into the single student notebook.

Agents and humans edit only the small section files; the .ipynb is a build artifact.
Usage: uv run python scripts/build.py            -> writes foundation-models-workshop.ipynb (outputs stripped)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
NAME = "foundation-models-workshop"
OUT = ROOT / f"{NAME}.ipynb"


def section_files() -> list[Path]:
    files = sorted(p for p in SECTIONS.glob("[0-9][0-9]_*.py"))
    if not files:
        sys.exit("no sections/NN_*.py files found")
    return files


def build() -> Path:
    cells = []
    for path in section_files():
        nb = jupytext.read(path, fmt="py:percent")
        for k, cell in enumerate(nb.cells):
            cell.id = f"{path.stem[:50]}-{k}"  # deterministic ids: same sections -> byte-identical .ipynb
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
    nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"built {out.name}: {len(section_files())} sections")
