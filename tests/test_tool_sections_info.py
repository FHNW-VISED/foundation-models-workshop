"""sections_info.py on a synthetic repo: `cells NN` and `words`. Skipped when jupytext cannot be imported
(the module itself needs only the standard library, but `words` and the lint import need jupytext)."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import jupytext  # noqa: F401

    HAVE_JUPYTEXT = True
except ImportError:
    HAVE_JUPYTEXT = False

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

SECTION_00 = """# %% [markdown]
# # Section zero title
#
# Short intro text with a few words.

# %% [markdown]
# <!-- callout: tip | A tip -->
# Tip body here.

# %% [markdown] edition="own"
# Own-data only paragraph.

# %% [markdown]
# <!-- diagram: flow | Flow of data -->

# %% [raw]
# raw line

# %%
# @title Helper cell
def helper():
    return 1

# %%
x = 1
print(x)
"""

SECTION_01 = """# %% [markdown]
# # Section one
#
# Another intro with some more words in it.
"""

MANY_WORDS = "# " + " ".join(f"word{i}" for i in range(60)) + "\n"


def load():
    spec = importlib.util.spec_from_file_location("sections_info", SCRIPTS / "sections_info.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = mod.main(list(argv))
    return code, out.getvalue()


def make_repo(tmp: Path, budget: str | None = "**Word budget:** 40", sections=None) -> Path:
    (tmp / "sections").mkdir()
    for name, text in (sections or {"00_intro.py": SECTION_00, "01_next.py": SECTION_01}).items():
        (tmp / "sections" / name).write_text(text)
    (tmp / "SPEC.md").write_text(f"# SPEC\n\n{budget}\n" if budget else "# SPEC\n")
    return tmp


def _lint_has_total_gate():
    """True when this repo's lint_notebook.py has the total word gate (EDA and the workshop get it in N14)."""
    if not HAVE_JUPYTEXT:
        return False
    sys.path.insert(0, str(SCRIPTS))
    try:
        import lint_notebook
    except Exception:
        return False
    return hasattr(lint_notebook, "total_visible_words") and hasattr(lint_notebook, "BUDGET_RE")


LINT_HAS_TOTAL_GATE = _lint_has_total_gate()
NO_GATE = "this repo's lint_notebook.py has no total word gate yet (issue N14)"


@unittest.skipUnless(HAVE_JUPYTEXT, "jupytext is not installed")
class SectionsInfo(unittest.TestCase):
    def setUp(self):
        self.mod = load()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def snapshot(self):
        return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.tmp.rglob("*") if p.is_file()}

    def test_cells_format_and_flags(self):
        make_repo(self.tmp)
        before = self.snapshot()
        code, out = run(self.mod, "cells", "00", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), 7)
        self.assertEqual([ln.split()[0] for ln in lines], [str(k) for k in range(7)])
        self.assertEqual([ln.split()[1] for ln in lines], ["md", "md", "md", "md", "raw", "code", "code"])
        self.assertEqual(
            lines[0],
            "  0 md       "
            + str(len("# Section zero title\n\nShort intro text with a few words.".encode()))
            + " - # Section zero title",
        )
        self.assertIn(" box ", lines[1])
        self.assertIn(" ed=own ", lines[2])
        self.assertIn(" diagram ", lines[3])
        self.assertIn(" helper ", lines[5])
        self.assertTrue(lines[5].endswith("# @title Helper cell"))
        self.assertNotIn("helper", lines[6])
        self.assertTrue(all(len(ln.split(" ", 4)[-1]) <= 70 for ln in lines))
        self.assertEqual(self.snapshot(), before)

    def test_cells_cut_at_70_and_number_without_zero(self):
        text = "# %%\n" + "# " + "a" * 100 + "\nx = 1\n"
        make_repo(self.tmp, sections={"02_long.py": text})
        code, out = run(self.mod, "cells", "2", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        self.assertEqual(len(out.splitlines()), 1)
        self.assertTrue(out.rstrip("\n").endswith("# " + "a" * 68))

    def test_unknown_section_exits_2_with_one_line(self):
        make_repo(self.tmp)
        code, out = run(self.mod, "cells", "07", "--root", str(self.tmp))
        self.assertEqual(code, 2)
        self.assertEqual(len(out.splitlines()), 1)
        self.assertIn("07", out)

    def test_duplicated_section_number_lists_both_and_exits_2(self):
        make_repo(self.tmp, sections={"01_a.py": SECTION_01, "01_b.py": SECTION_01})
        code, out = run(self.mod, "cells", "01", "--root", str(self.tmp))
        self.assertEqual(code, 2)
        self.assertIn("01_a.py", out)
        self.assertIn("01_b.py", out)

    @unittest.skipUnless(LINT_HAS_TOTAL_GATE, NO_GATE)
    def test_words_under_budget(self):
        make_repo(self.tmp)
        before = self.snapshot()
        code, out = run(self.mod, "words", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("00 Section zero title: "))
        self.assertTrue(lines[1].startswith("01 Section one: "))
        import lint_notebook

        cells = []
        for name in ("00_intro.py", "01_next.py"):
            cells += jupytext.read(self.tmp / "sections" / name, fmt="py:percent").cells
        total = lint_notebook.total_visible_words(cells)
        self.assertLess(total, 40)
        self.assertEqual(lines[-1], f"total: {total} of budget 40 ({40 - total} left)")
        self.assertEqual(int(lines[0].rsplit(": ", 1)[1]), lint_notebook.visible_words(cells[:7]))
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(LINT_HAS_TOTAL_GATE, NO_GATE)
    def test_words_over_budget(self):
        extra = "# %% [markdown]\n" + MANY_WORDS
        make_repo(self.tmp, sections={"00_intro.py": SECTION_00 + "\n" + extra})
        code, out = run(self.mod, "words", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        last = out.splitlines()[-1]
        self.assertRegex(last, r"^total: \d+ of budget 40 \(\+\d+ over\)$")
        total = int(last.split()[1])
        self.assertEqual(last, f"total: {total} of budget 40 (+{total - 40} over)")

    @unittest.skipUnless(LINT_HAS_TOTAL_GATE, NO_GATE)
    def test_words_without_budget_line(self):
        make_repo(self.tmp, budget=None)
        code, out = run(self.mod, "words", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[-1], "no Word budget in SPEC.md")
        self.assertEqual(len(out.splitlines()), 3)

    def test_words_lint_without_visible_words_exits_2(self):
        make_repo(self.tmp)
        with mock.patch.dict(sys.modules, {"lint_notebook": types.SimpleNamespace()}):
            code, out = run(self.mod, "words", "--root", str(self.tmp))
        self.assertEqual(code, 2)
        self.assertEqual(
            out.splitlines(),
            ["lint_notebook.py has no visible_words: update it from the template (issue N14)"],
        )

    def test_words_lint_without_total_gate_exits_0(self):
        make_repo(self.tmp)
        fake = types.SimpleNamespace(visible_words=lambda cells: 5)
        with mock.patch.dict(sys.modules, {"lint_notebook": fake}):
            code, out = run(self.mod, "words", "--root", str(self.tmp))
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "00 Section zero title: 5")
        self.assertEqual(lines[-1], "no total gate in lint_notebook.py")


if __name__ == "__main__":
    unittest.main()
