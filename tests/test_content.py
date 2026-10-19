"""Content assertions: does the workshop TEACH what SPEC.md promises? (W-1…W-17, 2026-10-19)

⛔ The night agent never edits this file (scripts/night.sh reverts it).
Sections write named results into RESULTS; the smoke test dumps them to build/smoke-results.json.
Progressive gate: a test runs once every section it needs has a status other than "todo" in SPEC §6.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "build" / "smoke-results.json"
SPEC = ROOT / "SPEC.md"
NUMERIC_COLUMNS = ("iou", "effort_min", "reliability", "seconds")


def _statuses() -> dict[str, str]:
    rows = {}
    for line in SPEC.read_text().splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*sections/(\d\d)_[\w-]+\.py\s*\|.*\|\s*([\w-]+)\s*\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2).lower()
    return rows


def requires(*sections: str):
    st = _statuses()
    wanted = list(st) if sections == ("all",) else list(sections)
    todo = [n for n in wanted if st.get(n, "todo") == "todo"]
    return pytest.mark.skipif(bool(todo), reason=f"sections not built yet: {todo}")


@pytest.fixture(scope="module")
def R():
    if not RESULTS_FILE.exists():
        pytest.fail("build/smoke-results.json missing: 00_setup.py must create RESULTS = {} (SPEC §9)")
    return json.loads(RESULTS_FILE.read_text())


def need(R, key):
    assert key in R, f"RESULTS['{key}'] was never set — the section that owns it is missing or incomplete"
    return R[key]


def iou(v):
    assert isinstance(v, (int, float)) and 0.0 <= v <= 1.0, f"not an IoU in [0, 1]: {v!r}"
    return v


@requires("01")
def test_s01_pets_work_out_of_the_box(R):
    iou(need(R, "pets_zero_shot_iou"))
    iou(need(R, "pets_head_iou"))


@requires("01", "02")
def test_s02_the_story_holds_specialised_is_worse(R):
    pets, spec = iou(need(R, "pets_zero_shot_iou")), iou(need(R, "specialised_zero_shot_iou"))
    assert pets > spec, f"the foundation model must do worse on the specialised task: pets {pets} vs {spec}"
    ladder = need(R, "sam_prompt_ladder")
    assert len(ladder) >= 3, "SAM must be probed with escalating prompts (≥ 3 settings)"


@requires("03", "05")
def test_s05_missing_is_not_background(R):
    naive, masked = need(R, "naive_recall_hidden_fg"), need(R, "masked_recall_hidden_fg")
    assert masked > naive, f"treating missing as unknown must recover hidden foreground: {masked} vs {naive}"


@requires("04")
def test_s04_ladder_has_its_rungs(R):
    ladder = need(R, "ladder")
    assert {"zero_shot", "one_shot", "scribbles"} <= set(ladder)
    for v in ladder.values():
        iou(v)


@requires("06")
def test_s06_recovered_supervision_measured(R):
    iou(need(R, "pseudo_label_iou"))
    assert "gradcam_iou" in R, "Grad-CAM path must report a result (or None with a stated reason)"


@requires("07")
def test_s07_direct_prompting_measured(R):
    iou(need(R, "sam_direct_iou"))


@requires("08")
def test_s08_comparison_table_complete(R):
    rows = need(R, "comparison_table")
    assert len(rows) >= 6, f"one row per approach (≥ 6), got {len(rows)}"
    for row in rows:
        missing = [c for c in NUMERIC_COLUMNS if not isinstance(row.get(c), (int, float))]
        assert not missing, f"row {row.get('approach')!r} lacks numeric {missing}"


@requires("09")
def test_s09_own_images_path_runs_headless(R):
    assert need(R, "own_images_ok") is True


@requires("10")
def test_s10_principles_written(R):
    assert len(need(R, "principles")) >= 4


@requires("all")
def test_every_participant_task_falls_back_cleanly(R):
    fallbacks = need(R, "use_fallbacks")  # {task_name: True} recorded by use() when the cell is untouched
    assert fallbacks and all(v is True for v in fallbacks.values()), f"use() fallbacks: {fallbacks}"
