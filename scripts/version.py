"""Versions of the notebook (Simone, 2026-10-22): one folder per version in versions/, holding SPEC.md and the
built notebook(s) at the repo root as they were. The newest folder always equals what is on main.

    python scripts/version.py new "What changed"   # snapshot SPEC.md + the notebook as the next version
    python scripts/version.py check                 # fail if the newest version differs from them
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

INDEX_HEAD = """# Versions

One folder per version of the notebook: `SPEC.md` and the built notebook exactly as they were.
The newest folder is always the notebook on `main`. Make one with `just version "what changed"`
before merging to main; `just version-check` fails if `main` has moved on without one.
Never edit a version folder.

| Version | Date | What changed | Commit |
|---|---|---|---|
"""


def notebooks(root: Path) -> list[Path]:
    found = sorted(root.glob("*.ipynb"))
    if not found:
        sys.exit("no notebook (*.ipynb) at the repo root")
    return found


def versions(root: Path) -> list[Path]:
    return sorted(p for p in (root / "versions").glob("v[0-9][0-9]-*") if p.is_dir())


def same(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def current_matches(root: Path, v: Path) -> bool:
    return same(root / "SPEC.md", v / "SPEC.md") and all(same(nb, v / nb.name) for nb in notebooks(root))


def commit(root: Path) -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "—"


def new(root: Path, label: str, date: str | None) -> int:
    vs = versions(root)
    if vs and current_matches(root, vs[-1]):
        print(f"nothing changed since {vs[-1].name}: no new version made")
        return 1
    n = len(vs) + 1
    slug = ""
    for word in re.findall(r"[a-z0-9]+", label.lower()):  # whole words, at most ~32 characters
        if slug and len(slug) + 1 + len(word) > 32:
            break
        slug = f"{slug}-{word}" if slug else word
    date = date or dt.date.today().isoformat()
    dest = root / "versions" / f"v{n:02d}-{date}-{slug}"
    dest.mkdir(parents=True)
    shutil.copy2(root / "SPEC.md", dest / "SPEC.md")
    for nb in notebooks(root):
        shutil.copy2(nb, dest / nb.name)
    index = root / "versions" / "README.md"
    if not index.exists():
        index.write_text(INDEX_HEAD)
    with index.open("a") as f:
        f.write(f"| v{n:02d} | {date} | {label} | {commit(root)} |\n")
    print(f"made {dest.relative_to(root)}")
    return 0


def check(root: Path) -> int:
    vs = versions(root)
    if not vs:
        print("no version in versions/ yet: run `just version \"what changed\"`")
        return 1
    if not current_matches(root, vs[-1]):
        print(f"SPEC.md or the notebook differ from the newest version ({vs[-1].name}):")
        print('run `just version "what changed"`')
        return 1
    print(f"newest version {vs[-1].name} equals SPEC.md and the notebook")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "check"])
    ap.add_argument("label", nargs="?", default="")
    ap.add_argument("--date")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args()
    if a.cmd == "new":
        if not a.label:
            sys.exit('give a label: version.py new "what changed"')
        return new(a.root, a.label, a.date)
    return check(a.root)


if __name__ == "__main__":
    sys.exit(main())
