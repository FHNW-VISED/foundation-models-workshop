"""Assemble the Jupytext percent sources into the two notebooks participants open.

Agents and humans edit only the small source files; the .ipynb files are build artifacts.
Usage: uv run python scripts/build.py
    sections/NN_*.py  -> foundation-models-workshop.ipynb  (the workshop)
    setup_check.py    -> setup_check.ipynb                 (run the week before, SPEC §6 / N-030)
Outputs are stripped; cell ids are deterministic so the same sources give byte-identical notebooks.
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
SETUP_SRC = ROOT / "setup_check.py"
SETUP_OUT = ROOT / "setup_check.ipynb"


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
