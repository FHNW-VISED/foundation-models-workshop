"""port_her.py on synthetic repos: sections written as text, a base notebook derived from them the way
build.py stamps cells (id `<stem>-<k>`, metadata.section), her copy written as a modified list of cells.
No test opens a real notebook."""

import ast
import contextlib
import copy
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "port_her.py"
SECTIONS_DIR = Path(__file__).resolve().parent.parent / "sections"

spec = importlib.util.spec_from_file_location("port_her", SCRIPT)
port_her = importlib.util.module_from_spec(spec)
sys.modules["port_her"] = port_her
spec.loader.exec_module(port_her)

INTRO = """# %% [markdown]
# # Title of the notebook
#
# This notebook shows how to look at a small dataset step by step.

# %% [markdown] tags=["x"]
# ## Setup
#
# Run the cell below once; it prepares everything the notebook needs later.

# %%
import os

print("setup done", os.name)
"""

DATA = """# %% [markdown]
# ## Data
#
# We load the data and look at the first rows before doing anything else.

# %% [markdown]
# <!-- callout: theory | Theory -->
# A box body that explains the idea in a couple of sentences for the learner.

# %%
# @title Load the data
data = [1, 2, 3]

# %% [markdown] edition="own"
# This paragraph is only for the own-data edition of the notebook.

# %%
total = sum(data)
print(total)

# %% [markdown]
# ## The end
#
# That is all for this notebook, thank you for following along today.
"""

MAGIC = """# %% [markdown]
# ## Install and look

# %%
# @title Install { display-mode: "form" }
# needs: boxes
import sys

if "google.colab" in sys.modules:
    # %pip install -q imagehash
    pass
print("ok")

# %%
# %matplotlib inline
# !ls data
x = 1

# %% [markdown]
# <!-- diagram: s00-map | The map -->
# A caption that goes under the picture for the learner.

# %%
z = 3
"""


# ---- an independent mini builder: the same stamping rule as build.py, nothing shared with port_her ----


def split_cells(text):
    chunks, cur = [], None
    for line in text.split("\n"):
        if line.startswith("# %%"):
            cur = [line, []]
            chunks.append(cur)
        elif cur is not None:
            cur[1].append(line)
    return [(h, "\n".join(b).strip("\n")) for h, b in chunks]


def build_base(root, edition="teaching"):
    cells = []
    for path in sorted((root / "sections").glob("[0-9][0-9]_*.py")):
        for k, (header, body) in enumerate(split_cells(path.read_text())):
            if 'edition="' in header and f'edition="{edition}"' not in header:
                continue
            if "[markdown]" in header:
                body = "\n".join(
                    ln[2:] if ln.startswith("# ") else "" if ln == "#" else ln for ln in body.split("\n")
                )
                if body.startswith(("<!-- callout:", "<!-- diagram:")):
                    html = '<div class="callout">' + body.split("\n", 1)[1] + "</div>"
                    cell = {
                        "cell_type": "code",
                        "source": f'# @title\nfrom IPython.display import HTML; HTML(r"""{html}""")',
                    }
                else:
                    cell = {"cell_type": "markdown", "source": body}
            else:
                # what jupytext and build.py do to a code cell: un-comment magics, drop `# needs:` lines
                body = re.sub(r"(?m)^(\s*)# (?=[%!])", r"\1", body)
                body = re.sub(r"(?m)^# needs: (boxes|masks)\n?", "", body)
                cell = {"cell_type": "code", "source": body}
            cell["id"] = f"{path.stem[:50]}-{k}"
            cell["metadata"] = {"section": path.stem}
            cells.append(cell)
    return cells


def lines_list(text):
    return text.splitlines(keepends=True)


def write_nb(path, cells):
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb), encoding="utf-8")


def find(cells, snippet):
    hits = [i for i, c in enumerate(cells) if snippet in "".join(c["source"])]
    assert len(hits) == 1, (snippet, hits)
    return hits[0]


def md(text):
    return {"cell_type": "markdown", "source": text, "metadata": {}}


def code(text):
    return {"cell_type": "code", "source": text, "metadata": {}, "outputs": [], "execution_count": None}


class PortTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "repo"
        (self.root / "sections").mkdir(parents=True)
        self.write_section("00_intro.py", INTRO)
        self.write_section("01_data.py", DATA)
        self.base = build_base(self.root)

    def write_section(self, name, text):
        (self.root / "sections" / name).write_text(text, encoding="utf-8")

    def section(self, name):
        return (self.root / "sections" / name).read_text(encoding="utf-8")

    def snapshot(self):
        return {p.name: p.read_bytes() for p in sorted((self.root / "sections").iterdir())}

    def her(self):
        return copy.deepcopy(self.base)

    def run_port(self, her_cells, *extra, base_cells=None, verify=None):
        base_path, her_path = self.tmp / "base.ipynb", self.tmp / "her.ipynb"
        write_nb(base_path, self.base if base_cells is None else base_cells)
        write_nb(her_path, her_cells)
        argv = [
            "--her",
            str(her_path),
            "--base",
            str(base_path),
            "--root",
            str(self.root),
            "--date",
            "2026-01-02",
            *extra,
        ]
        out = io.StringIO()
        kwargs = {} if verify is None else {"verify": verify}
        with contextlib.redirect_stdout(out):
            code_ = port_her.main(argv, **kwargs)
        return code_, out.getvalue()

    def reports(self):
        d = self.root / "reviews" / "2026-01-02-her-copy"
        return (d / "her-diff.md").read_text(encoding="utf-8"), (d / "her-conflicts.md").read_text(
            encoding="utf-8"
        )


class ReaderTests(unittest.TestCase):
    def test_section_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "00_x.py"
            p.write_text(DATA, encoding="utf-8")
            cells = port_her.read_section_cells(p)
        self.assertEqual([c.k for c in cells], [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            [c.kind for c in cells], ["markdown", "markdown", "code", "markdown", "code", "markdown"]
        )
        self.assertEqual(cells[3].header, '# %% [markdown] edition="own"')
        self.assertEqual(cells[3].edition, "own")
        self.assertEqual(
            cells[0].text,
            "## Data\n\nWe load the data and look at the first rows before doing anything else.",
        )
        self.assertEqual(cells[0].span, (0, 3))
        self.assertEqual(cells[2].text, "# @title Load the data\ndata = [1, 2, 3]")

    def test_md_alias_and_preamble(self):
        cells = port_her.parse_section_text("x = 1\n\n# %% [md]\n# text\n#\n# more\n")
        self.assertEqual([(c.kind, c.header) for c in cells], [("code", ""), ("markdown", "# %% [md]")])
        self.assertEqual(cells[1].text, "text\n\nmore")

    def test_matches_jupytext_when_available(self):
        try:
            import jupytext
        except ImportError:
            self.skipTest("jupytext not installed")
        path = sorted(SECTIONS_DIR.glob("*.py"))[0]  # the first section; names differ per repo
        mine = port_her.read_section_cells(path)
        theirs = jupytext.read(path, fmt="py:percent").cells
        self.assertEqual(len(mine), len(theirs))
        for a, b in zip(mine, theirs, strict=True):
            self.assertEqual(a.kind, b.cell_type)
            self.assertEqual(port_her.norm(port_her.as_built(a)), port_her.norm(b.source))

    def test_notebook_reader_skips_raw_and_joins_lists(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.ipynb"
            write_nb(
                p,
                [
                    {"cell_type": "raw", "source": "r"},
                    {"cell_type": "markdown", "source": ["a\n", "b"]},
                    {"cell_type": "code", "source": "c", "id": "x-1", "metadata": {"section": "x"}},
                ],
            )
            cells = port_her.read_notebook_cells(p)
        self.assertEqual([(c.cell_type, c.text) for c in cells], [("markdown", "a\nb"), ("code", "c")])
        self.assertEqual((cells[1].id, cells[1].section), ("x-1", "x"))

    def test_norm(self):
        self.assertEqual(port_her.norm("\n\n#@title Hi  \r\nx = 1   \n\n"), "# @title Hi\nx = 1")
        self.assertEqual(port_her.norm("a\n  b"), "a\n  b")  # nothing else is touched

    def test_align_cells(self):
        keys = [("markdown", "alpha beta gamma delta"), ("code", "x = 1")]
        self.assertEqual(port_her.align_cells(keys, keys), [(0, 0), (1, 1)])
        her = [("markdown", "alpha beta gamma epsilon"), ("code", "x = 1"), ("code", "new")]
        self.assertEqual(port_her.align_cells(keys, her), [(0, 0), (1, 1), (None, 2)])
        self.assertEqual(port_her.align_cells(keys, keys[:1]), [(0, 0), (1, None)])

    def test_standard_library_only(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module.split(".")[0])
        self.assertLessEqual(names, set(sys.stdlib_module_names))
        text = SCRIPT.read_text(encoding="utf-8").lower()
        self.assertNotIn("meeting", text)
        self.assertNotIn("susanne", text)


class ApplyTests(PortTest):
    def test_unchanged_writes_nothing(self):
        before = self.snapshot()
        her = self.her()
        her.insert(2, {"cell_type": "raw", "source": "raw cell", "metadata": {}})
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertEqual(self.snapshot(), before)
        self.assertIn("0 modified, 0 deleted, 0 inserted", out)
        diff, conflicts = self.reports()
        self.assertIn("applied: 0 modified, 0 deleted, 0 inserted", diff)
        self.assertIn("conflicts: none", conflicts)

    def test_markdown_edit(self):
        her = self.her()
        i = find(her, "look at the first rows")
        her[i]["source"] = "## Data\n\nWe load the data and glance at the first rows.\n\nSecond paragraph."
        before = self.snapshot()
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertEqual(
            self.section("01_data.py").split("\n")[:8],
            [
                "# %% [markdown]",
                "# ## Data",
                "#",
                "# We load the data and glance at the first rows.",
                "#",
                "# Second paragraph.",
                "",
                "# %% [markdown]",
            ],
        )
        self.assertEqual(self.snapshot()["00_intro.py"], before["00_intro.py"])
        self.assertEqual(self.section("01_data.py").split("\n")[6:], DATA.split("\n")[4:])

    def test_code_edit_and_title_header_written_as_typed(self):
        her = self.her()
        i = find(her, "Load the data")
        self.assertEqual(her[i]["source"], "# @title Load the data\ndata = [1, 2, 3]")
        her[i]["source"] = "#@title Load the data\ndata = [1, 2, 3, 4]"
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertIn("#@title Load the data\ndata = [1, 2, 3, 4]\n", self.section("01_data.py"))

    def test_colab_title_and_trailing_space_noise_is_ignored(self):
        before = self.snapshot()
        her = self.her()
        i = find(her, "Load the data")
        her[i]["source"] = "#@title Load the data  \r\ndata = [1, 2, 3]   \n\n"
        j = find(her, "look at the first rows")
        her[j]["source"] = lines_list(her[j]["source"].replace("\n", "  \n"))
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertEqual(self.snapshot(), before)

    def test_header_with_tags_survives_byte_for_byte(self):
        her = self.her()
        i = find(her, "Run the cell below")
        her[i]["source"] = "## Setup\n\nRun the cell below once."
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertIn(
            '# %% [markdown] tags=["x"]\n# ## Setup\n#\n# Run the cell below once.\n\n# %%\nimport os',
            self.section("00_intro.py"),
        )

    def test_deleted_cell(self):
        her = self.her()
        del her[find(her, "Run the cell below")]
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertEqual(
            self.section("00_intro.py"),
            INTRO.split("# %% [markdown] tags")[0] + '# %%\nimport os\n\nprint("setup done", os.name)\n',
        )
        self.assertIn("deleted sections/00_intro.py cell 1", self.reports()[0])

    def test_deleted_last_cells_keep_final_newline(self):
        her = self.her()
        del her[find(her, "total = sum")]
        del her[find(her, "That is all")]
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        text = self.section("01_data.py")
        self.assertTrue(
            text.endswith(
                '# %% [markdown] edition="own"\n# This paragraph is only for the own-data edition'
                " of the notebook.\n"
            ),
            text[-120:],
        )

    def test_inserted_mid_section_and_edition_numbering(self):
        her = self.her()
        i = find(her, "look at the first rows")
        her.insert(i + 1, md("A new paragraph.\n\nWith two parts."))
        her.insert(i + 2, code("print('new')"))
        j = find(her, "total = sum")  # its section k is 4: the own-edition cell at k=3 counts
        her[j]["source"] = "total = sum(data)\nprint('total', total)"
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        text = self.section("01_data.py")
        self.assertIn(
            "first rows before doing anything else.\n\n# %% [markdown]\n# A new paragraph.\n#\n"
            "# With two parts.\n"
            "\n# %%\nprint('new')\n\n# %% [markdown]\n# <!-- callout",
            text,
        )
        self.assertIn("# %%\ntotal = sum(data)\nprint('total', total)\n\n# %% [markdown]\n# ## The end", text)
        self.assertIn('edition="own"', text)

    def test_insert_after_last_cell_of_notebook_is_applied(self):
        her = self.her()
        her.append(md("One more closing remark."))
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertTrue(
            self.section("01_data.py").endswith(
                "along today.\n\n# %% [markdown]\n# One more closing remark.\n"
            )
        )

    def test_boundary_runs_are_conflicts(self):
        before = self.snapshot()
        her = self.her()
        her.insert(0, md("A cell before everything else."))
        her.insert(find(her, "setup done") + 1, md("A cell between two sections."))
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        self.assertEqual(self.snapshot(), before)
        conflicts = self.reports()[1]
        self.assertEqual(conflicts.count(" boundary "), 2)
        self.assertIn("boundary: say which section the new cells belong to", conflicts)

    def test_type_change_is_a_conflict(self):
        before = self.snapshot()
        her = self.her()
        i = find(her, "look at the first rows")
        her[i] = code(her[i]["source"])
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        self.assertEqual(self.snapshot(), before)
        self.assertIn("### C-01 type sections/01_data.py cell 0", self.reports()[1])

    def test_edited_box_is_a_conflict_and_other_edits_still_apply(self):
        her = self.her()
        i = find(her, "A box body")
        her[i]["source"] = her[i]["source"].replace("learner", "student")
        j = find(her, "That is all")
        her[j]["source"] = "## The end\n\nThat is all, thank you for following along."
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        text = self.section("01_data.py")
        self.assertIn("for the learner.", text)
        self.assertIn("# That is all, thank you for following along.", text)
        self.assertIn("### C-01 box sections/01_data.py cell 1", self.reports()[1])
        self.assertIn("box: rewrite the box source", self.reports()[1])

    def test_moved_cell_is_a_conflict(self):
        before = self.snapshot()
        her = self.her()
        cell = her.pop(find(her, "That is all"))
        her.insert(find(her, "total = sum"), cell)
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        self.assertEqual(self.snapshot(), before)
        self.assertIn(" moved sections/01_data.py cell ", self.reports()[1])

    def test_attachment_is_a_conflict(self):
        before = self.snapshot()
        her = self.her()
        her.insert(
            find(her, "look at the first rows") + 1,
            {**md("![pic](attachment:image.png)"), "attachments": {"image.png": {"image/png": "AAAA"}}},
        )
        i = find(her, "That is all")
        her[i] = {
            **md("## The end\n\nThat is all for this notebook, thank you ![pic](attachment:x.png)"),
            "attachments": {"x.png": {"image/png": "BB"}},
        }
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.reports()[1].count(" attachment "), 2)

    def test_two_sections_at_once(self):
        her = self.her()
        her[find(her, "Run the cell below")]["source"] = "## Setup\n\nRun the cell below once, then go on."
        her[find(her, "That is all")]["source"] = (
            "## The end\n\nThat is all for this notebook, see you next time."
        )
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertIn("go on.", self.section("00_intro.py"))
        self.assertIn("see you next time.", self.section("01_data.py"))
        diff = self.reports()[0]
        self.assertIn("modified sections/00_intro.py cell 1", diff)
        self.assertIn("modified sections/01_data.py cell 5", diff)

    def test_dry_run_writes_nothing(self):
        before = self.snapshot()
        her = self.her()
        her[find(her, "Run the cell below")]["source"] = "## Setup\n\nRun the cell below once, then go on."
        code_, out = self.run_port(her, "--dry-run")
        self.assertEqual(code_, 0, out)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / "reviews").exists())
        self.assertIn("### A-01 modified sections/00_intro.py cell 1", out)
        self.assertIn("her-conflicts.md", out)

    def test_diverged_sections_are_refused_and_allow_lists_a_clash(self):
        self.write_section("01_data.py", DATA.replace("first rows", "first lines"))
        her = self.her()
        before = self.snapshot()
        code_, out = self.run_port(her)
        self.assertEqual(code_, 2, out)
        self.assertIn("diverged 01_data-0", out)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / "reviews").exists())
        # allowed: an unchanged copy keeps the section text; a cell both changed is a clash, her text applied
        code_, out = self.run_port(her, "--allow-diverged")
        self.assertEqual(code_, 0, out)
        self.assertEqual(self.snapshot(), before)
        her[find(her, "look at the first rows")]["source"] = (
            "## Data\n\nWe load the data and look at the first rows."
        )
        code_, out = self.run_port(her, "--allow-diverged")
        self.assertEqual(code_, 1, out)
        self.assertIn("# We load the data and look at the first rows.\n", self.section("01_data.py"))
        self.assertIn(" clash sections/01_data.py cell 0", self.reports()[1])

    def test_too_little_aligned_is_refused(self):
        before = self.snapshot()
        her = [
            md(f"Completely other words number {i} appear here only in her copy {i * 7}.") for i in range(8)
        ]
        code_, out = self.run_port(her)
        self.assertEqual(code_, 3, out)
        self.assertIn("her copy shares too little with the base", out)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / "reviews").exists())

    def test_failed_read_back_restores_the_originals(self):
        before = self.snapshot()
        her = self.her()
        her[find(her, "Run the cell below")]["source"] = "## Setup\n\nRun the cell below once, then go on."
        her[find(her, "That is all")]["source"] = (
            "## The end\n\nThat is all for this notebook, see you next time."
        )
        code_, out = self.run_port(her, verify=lambda expected: ["forced failure"])
        self.assertEqual(code_, 2, out)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(list((self.root / "sections").glob("*.tmp")))
        self.assertFalse((self.root / "reviews").exists())

    def test_real_read_back_catches_a_cell_that_would_split(self):
        before = self.snapshot()
        her = self.her()
        her[find(her, "Load the data")]["source"] = "# @title Load\n# %% fake header inside the cell\nx = 1"
        code_, out = self.run_port(her)
        self.assertEqual(code_, 2, out)
        self.assertEqual(self.snapshot(), before)

    def test_report_budget_and_not_listed_count(self):
        words = lambda i, n: " ".join(f"w{i}_{x}" for x in range(n))  # noqa: E731
        sec = "".join(f"# %% [markdown]\n# {words(i, 130)}\n\n" for i in range(60))
        self.write_section("00_intro.py", sec)
        (self.root / "sections" / "01_data.py").unlink()
        self.base = build_base(self.root)
        her = self.her()
        for i, c in enumerate(her):
            c["source"] = c["source"].replace(f"w{i}_3 ", "changed ").replace(f"w{i}_9 ", "other ")
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        diff = (self.root / "reviews" / "2026-01-02-her-copy" / "her-diff.md").read_bytes()
        self.assertLessEqual(len(diff), 26_000)
        text = diff.decode()
        listed = text.count("\n### A-")
        self.assertIn("applied: 60 modified", text)
        self.assertIn(f"listed: {listed} of 60; not listed: {60 - listed}", text)
        self.assertLess(listed, 60)
        self.assertGreater(listed, 10)

    def test_texts_are_cut_at_600_characters(self):
        long_before = "alpha " * 400
        sec = f"# %% [markdown]\n# {long_before.strip()}\n"
        self.write_section("00_intro.py", sec)
        (self.root / "sections" / "01_data.py").unlink()
        self.base = build_base(self.root)
        her = self.her()
        her[0]["source"] = long_before.strip() + " beta"
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        diff = self.reports()[0]
        before_len, after_len = len(long_before.strip()), len(long_before.strip() + " beta")
        self.assertIn(f"(+{before_len - 600} characters)", diff)
        self.assertIn(f"(+{after_len - 600} characters)", diff)
        self.assertIn(f"(at 600 each): {before_len - 600 + after_len - 600}", diff)
        self.assertNotIn("alpha " * 130, diff)

    def test_cli_runs_as_a_script(self):
        base_path, her_path = self.tmp / "base.ipynb", self.tmp / "her.ipynb"
        write_nb(base_path, self.base)
        write_nb(her_path, self.her())
        run = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--her",
                str(her_path),
                "--base",
                str(base_path),
                "--root",
                str(self.root),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("aligned: 8 of 8", run.stdout)


class TransformTests(PortTest):
    """Every text transform between a section cell and the built cell, both directions."""

    def setUp(self):
        super().setUp()
        self.write_section("02_magic.py", MAGIC)
        self.base = build_base(self.root)

    def test_built_cells_equal_their_sections(self):
        before = self.snapshot()
        i = find(self.base, "pip install -q imagehash")
        self.assertIn("\n    %pip install -q imagehash", self.base[i]["source"])
        self.assertNotIn("needs", self.base[i]["source"])
        code_, out = self.run_port(self.her())
        self.assertEqual(code_, 0, out)
        self.assertEqual(self.snapshot(), before)

    def test_edited_magic_line_is_written_commented_again(self):
        her = self.her()
        i = find(her, "pip install -q imagehash")
        her[i]["source"] = her[i]["source"].replace("imagehash", "imagehash timm")
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        text = self.section("02_magic.py")
        self.assertIn('    # %pip install -q imagehash timm\n    pass\nprint("ok")', text)
        self.assertIn('# @title Install { display-mode: "form" }\n# needs: boxes\nimport sys', text)

    def test_edit_next_to_a_directive_keeps_the_directive_in_place(self):
        her = self.her()
        i = find(her, "pip install -q imagehash")
        her[i]["source"] = her[i]["source"].replace('print("ok")', 'print("ready")')
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        text = self.section("02_magic.py")
        self.assertIn("# needs: boxes\nimport sys", text)
        self.assertIn('print("ready")', text)

    def test_magic_lines_she_adds_are_commented(self):
        her = self.her()
        i = find(her, "matplotlib inline")
        her[i]["source"] = "%matplotlib inline\n!ls data\n%pip install -q numpy\nx = 2"
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertIn(
            "# %matplotlib inline\n# !ls data\n# %pip install -q numpy\nx = 2\n", self.section("02_magic.py")
        )

    def test_inserted_code_cell_with_magic(self):
        her = self.her()
        her.insert(find(her, "matplotlib inline") + 1, code("%%time\ny = 1"))
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertIn("x = 1\n\n# %%\n# %%time\ny = 1\n", self.section("02_magic.py"))

    def test_string_content_is_not_a_magic(self):
        text = 'a = 1\ntext = """\n# %pip stays inside the string\n"""\n# %pip is_real\n'
        cell = port_her.parse_section_text("# %%\n" + text)[0]
        self.assertEqual(port_her.as_built(cell), text.replace("# %pip is_real", "%pip is_real").rstrip("\n"))
        self.assertEqual(
            "\n".join(port_her.comment_magic(port_her.as_built(cell).split("\n"))), text.rstrip("\n")
        )

    def test_deleting_a_directive_cell(self):
        her = self.her()
        del her[find(her, "pip install -q imagehash")]
        code_, out = self.run_port(her)
        self.assertEqual(code_, 0, out)
        self.assertNotIn("needs: boxes", self.section("02_magic.py"))

    def test_diagram_cell_is_a_built_box(self):
        her = self.her()
        i = find(her, "A caption")
        her[i]["source"] = her[i]["source"].replace("learner", "student")
        code_, out = self.run_port(her)
        self.assertEqual(code_, 1, out)
        self.assertIn(" box sections/02_magic.py cell 3", self.reports()[1])

    def test_as_built_units(self):
        read = port_her.parse_section_text(MAGIC)
        code_cell = read[1]
        self.assertEqual(code_cell.text.count("# needs: boxes"), 1)
        built = port_her.as_built(code_cell)
        self.assertIn("    %pip install -q imagehash", built)
        self.assertNotIn("needs:", built)
        self.assertIn(
            "# needs: boxes", port_her.as_built(code_cell, "own")
        )  # only the Teaching Edition strips
        self.assertEqual(port_her.as_built(read[0]), read[0].text)  # markdown is untouched

    def test_magic_rules_match_jupytext_when_available(self):
        try:
            import jupytext
        except ImportError:
            self.skipTest("jupytext not installed")
        extra = "\n".join(
            [
                "# %%",
                "# %%capture",
                "# x = !ls",
                "# ?help",
                "# echo hello",
                "# cat photos are fun",
                "run = 1",
                "# !pip install \\",
                "#   -q numpy",
                "# %%",
                'text = """',
                "# %pip inside",
                '"""',
                "# %pip real",
                "",
            ]
        )
        for name, text in (("magic", MAGIC), ("extra", extra)):
            path = self.tmp / f"{name}.py"
            path.write_text(text, encoding="utf-8")
            mine = port_her.read_section_cells(path)
            theirs = jupytext.read(path, fmt="py:percent").cells
            self.assertEqual(len(mine), len(theirs), name)
            for a, b in zip(mine, theirs, strict=True):
                built = port_her.as_built(a, "own")
                self.assertEqual(port_her.norm(built), port_her.norm(b.source), name)
                if a.kind == "code":  # writing the built text back gives the section text
                    again = "\n".join(port_her.comment_magic(built.split("\n")))
                    self.assertEqual(port_her.norm(again), port_her.norm(a.text), name)


if __name__ == "__main__":
    unittest.main()
