"""Run setup_check.ipynb headless against a local release built from data/smoke/ (no Drive, no release URL).

The models come from the machine's own Hugging Face cache (downloaded on first run, ~260 MB, ungated).
`make_source()` also documents the release layout the real hosting must follow: manifest.json plus the files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "setup_check.ipynb"
SMOKE = ROOT / "data" / "smoke"
EXECUTED = ROOT / "build" / "setup-check-executed.ipynb"
REQUIRED = [
    "Cache folder",
    "Packages",
    "data/pets",
    "data/specialised",
    "DINOv3 ViT-S/16",
    "SAM 2.1 hiera-small",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(root: Path) -> Path:
    """A release folder as setup_check.py expects it: data/, precomputed/, specialised.json, manifest.json."""
    shutil.copytree(SMOKE / "pets", root / "data" / "pets")
    shutil.copytree(SMOKE / "specialised", root / "data" / "specialised")
    shutil.copytree(SMOKE / "precomputed", root / "precomputed")
    shutil.copyfile(ROOT / "data" / "specialised.json", root / "specialised.json")
    files = [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    (root / "manifest.json").write_text(json.dumps({"version": "smoke", "files": files}, indent=1))
    return root


def run(nb_path: Path, env: dict[str, str]) -> nbformat.NotebookNode:
    nb = nbformat.read(nb_path, as_version=4)
    os.environ.update(env)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    try:
        client.execute()
    except CellExecutionError as err:
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(err))
        pytest.fail("setup check failed:\n" + "\n".join(plain.strip().splitlines()[-25:]), pytrace=False)
    finally:
        EXECUTED.parent.mkdir(exist_ok=True)
        nbformat.write(nb, EXECUTED)
    return nb


def test_setup_check_runs_locally_and_everything_passes(tmp_path):
    assert NOTEBOOK.exists(), "run `just build` first"
    source = make_source(tmp_path / "release")
    cache = tmp_path / "cache"
    run(
        NOTEBOOK,
        {
            "FM_SETUP_SOURCE": str(source),
            "FM_SETUP_CACHE": str(cache),
            "FM_TRY_SAM3": "0",
            "MPLBACKEND": "Agg",
        },
    )
    result = json.loads((cache / "setup-check.json").read_text())
    checks = result["checks"]
    assert result["ok"], checks
    for item in REQUIRED:
        assert checks[item]["icon"] == "✅", (item, checks[item])
    assert not [k for k, v in checks.items() if v["icon"] == "❌"], checks
    assert checks["SAM 3 (optional)"]["icon"] == "⏭️"
    # The cache mirrors the release, byte for byte, and the main notebook's inputs are all there.
    manifest = json.loads((source / "manifest.json").read_text())["files"]
    for entry in manifest:
        assert sha256_of(cache / entry["path"]) == entry["sha256"], entry["path"]
    assert (cache / "data" / "pets" / "index.json").exists()
    assert (cache / "precomputed" / "heads" / "index.json").exists()
    assert (cache / "specialised.json").exists()
    # The second run finds everything cached: nothing downloaded, nothing failed.
    nb = run(NOTEBOOK, {"FM_SETUP_SOURCE": str(source), "FM_SETUP_CACHE": str(cache)})
    text = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
    )
    assert "already cached" in text and "❌" not in text
