"""Structure checks: RULES.md rules that a script can decide (see scripts/lint_notebook.py)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_lint_passes():
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lint_notebook.py")], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout
