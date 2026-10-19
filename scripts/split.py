"""Reverse of build.py: after a hand edit of the .ipynb (Jupyter or a Colab download), write
each cell back to its section file using the `section` cell metadata that build.py stamps.

Cells without that metadata (new cells added by hand) join the section of the cell above them.
Always review `git diff sections/` afterwards.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "foundation-models-workshop.ipynb"


def main() -> int:
    nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else NOTEBOOK, as_version=4)
    groups: OrderedDict[str, list] = OrderedDict()
    current = None
    for cell in nb.cells:
        current = cell.metadata.get("section", current)
        if current is None:
            print("first cell has no `section` metadata; cannot split", file=sys.stderr)
            return 1
        cell.outputs = [] if cell.cell_type == "code" else None
        if cell.cell_type != "code":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        else:
            cell.execution_count = None
        for key in ("section", "cellView", "id"):
            cell.metadata.pop(key, None)
        groups.setdefault(current, []).append(cell)
    for stem, cells in groups.items():
        part = nbformat.v4.new_notebook(cells=cells)
        part.metadata["jupytext"] = {"notebook_metadata_filter": "-all"}  # no YAML header in section files
        jupytext.write(part, ROOT / "sections" / f"{stem}.py", fmt="py:percent")
    print(f"split into {len(groups)} section files; review `git diff sections/`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
