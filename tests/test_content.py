"""Content assertions on the executed smoke notebook: does it TEACH what SPEC.md promises?

"It runs" is not "it teaches" — and an agent will satisfy a weak test without doing the work.
These assertions are written by Simone (or with him, during /notebook grill), never by the
night agent: scripts/night.sh reverts any agent edit to tests/.

Runs after tests/test_smoke.py, which writes build/smoke-executed.ipynb.
Helpers below: cells_of(section), outputs_text(section), images_in(section), source_of(section).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

EXECUTED = Path(__file__).resolve().parent.parent / "build" / "smoke-executed.ipynb"


@pytest.fixture(scope="module")
def nb():
    if not EXECUTED.exists():
        pytest.skip("no executed notebook — run tests/test_smoke.py first (just smoke)")
    return nbformat.read(EXECUTED, as_version=4)


def cells_of(nb, section: str):
    return [c for c in nb.cells if c.metadata.get("section", "").startswith(section)]


def source_of(nb, section: str) -> str:
    return "\n".join(c.source for c in cells_of(nb, section))


def outputs_text(nb, section: str) -> str:
    parts = []
    for c in cells_of(nb, section):
        for o in c.get("outputs", []):
            parts.append(o.get("text", "") or str(o.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def images_in(nb, section: str) -> int:
    return sum(
        1 for c in cells_of(nb, section) for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    )


def test_setup_states_learning_outcomes(nb):
    src = source_of(nb, "00_")
    assert "At the end of this notebook you will be able to" in src


# Examples of the assertions to write per section during grilling:
# def test_class_distribution_is_plotted(nb):
#     assert images_in(nb, "03_") >= 1
# def test_planted_duplicates_are_found(nb):
#     assert "duplicates found: 5" in outputs_text(nb, "05_")
