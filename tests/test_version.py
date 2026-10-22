"""Versioning (Simone, 2026-10-22): versions/ keeps a snapshot of SPEC.md + the built notebook per version,
and the newest snapshot always equals what is on main. scripts/version.py makes and checks them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "version.py"


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), *args, "--root", str(root)]
    return subprocess.run(cmd, capture_output=True, text=True)


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "SPEC.md").write_text("spec one\n")
    (tmp_path / "demo.ipynb").write_text('{"cells": []}\n')
    return tmp_path


def test_new_version_copies_spec_and_notebook(tmp_path):
    root = make_repo(tmp_path)
    r = run(root, "new", "First cut", "--date", "2026-10-22")
    assert r.returncode == 0, r.stderr
    v = root / "versions" / "v01-2026-10-22-first-cut"
    assert (v / "SPEC.md").read_text() == "spec one\n"
    assert (v / "demo.ipynb").read_text() == '{"cells": []}\n'
    index = (root / "versions" / "README.md").read_text()
    assert "| v01 | 2026-10-22 | First cut |" in index


def test_unchanged_files_make_no_new_version(tmp_path):
    root = make_repo(tmp_path)
    run(root, "new", "First", "--date", "2026-10-22")
    r = run(root, "new", "Again", "--date", "2026-10-23")
    assert r.returncode == 1
    assert "nothing changed" in r.stdout + r.stderr
    assert len(list((root / "versions").glob("v*"))) == 1


def test_next_version_number_and_latest_is_last(tmp_path):
    root = make_repo(tmp_path)
    run(root, "new", "First", "--date", "2026-10-22")
    (root / "SPEC.md").write_text("spec two\n")
    r = run(root, "new", "Second", "--date", "2026-10-23")
    assert r.returncode == 0, r.stderr
    assert (root / "versions" / "v02-2026-10-23-second" / "SPEC.md").read_text() == "spec two\n"
    index = (root / "versions" / "README.md").read_text()
    assert index.index("| v01 |") < index.index("| v02 |")


def test_check_passes_only_when_latest_equals_current(tmp_path):
    root = make_repo(tmp_path)
    assert run(root, "check").returncode == 1  # no version yet
    run(root, "new", "First", "--date", "2026-10-22")
    assert run(root, "check").returncode == 0
    (root / "demo.ipynb").write_text('{"cells": [1]}\n')
    r = run(root, "check")
    assert r.returncode == 1
    assert "just version" in r.stdout + r.stderr


def test_every_root_notebook_is_versioned(tmp_path):
    root = make_repo(tmp_path)
    (root / "setup_check.ipynb").write_text('{"cells": ["setup"]}\n')
    run(root, "new", "Two notebooks", "--date", "2026-10-22")
    v = root / "versions" / "v01-2026-10-22-two-notebooks"
    assert (v / "demo.ipynb").exists() and (v / "setup_check.ipynb").exists()
    assert run(root, "check").returncode == 0
    (root / "setup_check.ipynb").write_text('{"cells": ["changed"]}\n')
    assert run(root, "check").returncode == 1
