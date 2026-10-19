"""Compare this notebook's runtime dependencies with what Google Colab preinstalls.

Source: github.com/googlecolab/backend-info (pip-freeze.txt = CPU runtime, pip-freeze.gpu.txt = GPU).
Prints one line per dependency; exits 1 if a pin in pyproject.toml differs from Colab's version.
Usage: uv run python scripts/colab_pins.py [cpu|gpu]
"""

from __future__ import annotations

import re
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://raw.githubusercontent.com/googlecolab/backend-info/main/"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    runtime = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    url = BASE + ("pip-freeze.gpu.txt" if runtime == "gpu" else "pip-freeze.txt")
    freeze = urllib.request.urlopen(url, timeout=30).read().decode()
    colab = {}
    for line in freeze.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
        if m:
            colab[norm(m.group(1))] = m.group(2)
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    bad = 0
    print(f"Colab {runtime} runtime ({url})")
    for dep in deps:
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[.*\])?\s*(==\s*([^\s;,]+))?", dep)
        name, pinned = norm(m.group(1)), m.group(3)
        have = colab.get(name)
        if have is None:
            status = "NOT PREINSTALLED → needs a %pip install cell in Colab"
        elif pinned and pinned != have.split("+")[0] and pinned != have:
            status = f"CONFLICT: pinned {pinned}, Colab has {have}"
            bad += 1
        else:
            status = f"ok (Colab {have})" + ("" if pinned else " — consider pinning ==" + have.split("+")[0])
        print(f"  {name:28s} {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
