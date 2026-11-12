"""verify_her.py on synthetic notebooks written with json.dumps in a temporary folder. No test opens a real
notebook."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_her  # noqa: E402

ITEMS = """# Meeting items
| id | where | what |
|---|---|---|
I-03 | section 4 | the plot title is shortened
I-04 | section 5 | a box is reworded
- C-24 / R-01 | chapter 10 | the park sample
"""


def md(text, cid=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    if cid:
        cell["id"] = cid
    return cell


def code(text, cid=None):
    cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.splitlines(keepends=True),
    }
    if cid:
        cell["id"] = cid
    return cell


def html_box(body, cid=None):
    return code(
        f'# @title\nfrom IPython.display import HTML; HTML(r"""<div class="callout">{body}</div>""")', cid
    )


BUILT = [
    md("# Title of the notebook", "intro-0"),
    md("Run the cell below once.", "intro-1"),
    code("import os\nprint(os.name)", "setup-0"),
    code("x = 1\nprint(x)", "setup-1"),
    md("Last words of the chapter.", "setup-2"),
]


class VerifyHerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.folder = self.root / "review"
        self.folder.mkdir()
        (self.folder / "items.md").write_text(ITEMS, encoding="utf-8")
        (self.folder / "approved.md").write_text("# nothing yet\n", encoding="utf-8")

    def notebook(self, name, cells):
        path = self.root / name
        path.write_text(
            json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
            encoding="utf-8",
        )
        return path

    def approve(self, *lines):
        (self.folder / "approved.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_main(self, her_cells, built_cells=BUILT, extra=()):
        her = self.notebook("her.ipynb", her_cells)
        built = self.notebook("built.ipynb", built_cells)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code_ = verify_her.main(
                ["--her", str(her), "--built", str(built), "--folder", str(self.folder), *extra]
            )
        return code_, out.getvalue(), err.getvalue()

    def count(self, out, label):
        line = next(ln for ln in out.splitlines() if ln.startswith(f"{label}: "))
        return int(line.split(": ")[1])

    def changed_her(self):
        cells = [dict(c) for c in BUILT]
        cells[3] = code("x = 2\nprint(x)")
        return cells

    def test_identical_notebooks_exit_0(self):
        code_, out, _ = self.run_main(BUILT)
        self.assertEqual(code_, 0)
        self.assertEqual(self.count(out, "her cells"), 5)
        self.assertEqual(self.count(out, "built cells"), 5)
        self.assertEqual(self.count(out, "identical"), 5)
        self.assertEqual(self.count(out, "differences"), 0)
        self.assertEqual(self.count(out, "unapproved"), 0)

    def test_report_header_names_the_limitation_and_counts_come_in_order(self):
        _, out, _ = self.run_main(BUILT)
        self.assertIn("run outputs", out.splitlines()[0])
        labels = [ln.split(":")[0] for ln in out.splitlines()[1:7]]
        self.assertEqual(
            labels, ["her cells", "built cells", "identical", "differences", "approved", "unapproved"]
        )

    def test_modified_cell_is_unapproved(self):
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertEqual(self.count(out, "differences"), 1)
        self.assertEqual(self.count(out, "unapproved"), 1)
        self.assertIn("UNAPPROVED setup-1 changed/code x = 1 print(x)", out)

    def test_modified_cell_approved_by_a_real_item(self):
        self.approve("setup-1 | I-03 | she wants the value set to two")
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 0)
        self.assertEqual(self.count(out, "approved"), 1)
        self.assertEqual(self.count(out, "unapproved"), 0)

    def test_two_item_ids_in_one_line_both_must_exist(self):
        self.approve("setup-1 | C-24 / R-01 | the park sample feeds this cell")
        code_, _, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 0)
        self.approve("setup-1 | C-24 / R-77 | the park sample feeds this cell")
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertIn("UNKNOWN ITEM 1", out)

    def test_unknown_item_id_is_reported_and_covers_nothing(self):
        self.approve("setup-1 | I-99 | she wants the value set to two")
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertIn("UNKNOWN ITEM 1", out)
        self.assertEqual(self.count(out, "approved"), 0)
        self.assertEqual(self.count(out, "unapproved"), 1)

    def test_unknown_item_alone_fails_even_without_differences(self):
        self.approve("setup-1 | I-99 | she wants the value set to two")
        code_, out, _ = self.run_main(BUILT)
        self.assertEqual(code_, 1)
        self.assertIn("UNKNOWN ITEM 1", out)

    def test_two_word_reason_covers_nothing(self):
        self.approve("setup-1 | I-03 | she asked")
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertEqual(self.count(out, "approved"), 0)
        self.assertIn("BAD LINE 1", out)

    def test_approved_line_for_another_cell_does_not_cover(self):
        self.approve("setup-0 | I-03 | she wants the value set to two")
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertIn("UNUSED 1", out)

    def test_deleted_by_the_build_is_keyed_by_her_position(self):
        her = [dict(c) for c in BUILT]
        her.insert(2, md("A paragraph of hers that the build no longer has at all."))
        code_, out, _ = self.run_main(her)
        self.assertEqual(code_, 1)
        self.assertEqual(self.count(out, "her cells"), 6)
        self.assertIn("UNAPPROVED her:3 her-only/markdown", out)
        self.approve("her:3 | I-04 | the item removes this paragraph")
        code_, out, _ = self.run_main(her)
        self.assertEqual(code_, 0)
        self.assertEqual(self.count(out, "approved"), 1)

    def test_cell_the_build_added_is_keyed_by_its_built_id(self):
        her = [c for i, c in enumerate(BUILT) if i != 1]
        code_, out, _ = self.run_main(her)
        self.assertEqual(code_, 1)
        self.assertIn("UNAPPROVED intro-1 built-only/markdown", out)
        self.approve("intro-1 | I-04 | the item adds this line")
        code_, _, _ = self.run_main(her)
        self.assertEqual(code_, 0)

    def test_one_changed_one_deleted_one_inserted_each_listed(self):
        her = [dict(c) for c in BUILT]
        her[3] = code("x = 2\nprint(x)")
        del her[1]
        her.append(md("Something she wrote at the very end of the notebook."))
        code_, out, _ = self.run_main(her)
        self.assertEqual(code_, 1)
        self.assertEqual(self.count(out, "unapproved"), 3)
        self.assertEqual(self.count(out, "differences"), 3)

    def test_box_is_compared_by_visible_text(self):
        built = [*BUILT, html_box("<b>Tip</b>\n<p>Use a   small sample first.</p>", "box-0")]
        her_same = [*BUILT, html_box('<p style="x"><b>Tip</b> Use a small sample first.</p>')]
        code_, out, _ = self.run_main(her_same, built)
        self.assertEqual(code_, 0)
        self.assertEqual(self.count(out, "identical"), 6)
        her_other = [*BUILT, html_box("<b>Tip</b> Use a big sample first.")]
        code_, out, _ = self.run_main(her_other, built)
        self.assertEqual(code_, 1)
        self.assertIn("UNAPPROVED box-0 changed/code", out)

    def test_box_marker_markdown_cell_matches_by_visible_text_not_type(self):
        built = [*BUILT, html_box("Tip: be careful here", "box-0")]
        her = [*BUILT, md('<div class="callout">Tip:   be careful here</div>')]
        code_, _, _ = self.run_main(her, built)
        self.assertEqual(code_, 0)

    def test_list_prints_both_texts_cut_and_a_ready_to_fill_line(self):
        long_text = "word " * 120
        her = [dict(c) for c in BUILT]
        her[1] = md("short " + long_text)
        built = [dict(c) for c in BUILT]
        built[1] = md("other " + long_text, "intro-1")
        code_, out, _ = self.run_main(her, built, extra=["--list"])
        self.assertEqual(code_, 1)
        her_line = next(ln for ln in out.splitlines() if ln.startswith("her:   "))
        built_line = next(ln for ln in out.splitlines() if ln.startswith("built: "))
        self.assertEqual(len(her_line), len("her:   ") + verify_her.CUT)
        self.assertEqual(len(built_line), len("built: ") + verify_her.CUT)
        self.assertIn("intro-1 | ITEM-ID | REASON", out)

    def test_without_list_no_texts_are_printed_beyond_the_short_line(self):
        _, out, _ = self.run_main(self.changed_her())
        self.assertNotIn("ITEM-ID", out)
        self.assertNotIn("her:   ", out)

    def test_missing_items_md_exits_2(self):
        (self.folder / "items.md").unlink()
        code_, out, err = self.run_main(BUILT)
        self.assertEqual(code_, 2)
        self.assertEqual(out, "")
        self.assertIn("items.md", err)

    def test_items_md_without_ids_exits_2(self):
        (self.folder / "items.md").write_text("nothing numbered here\n", encoding="utf-8")
        code_, _, err = self.run_main(BUILT)
        self.assertEqual(code_, 2)
        self.assertIn("items.md has no item ids", err)

    def test_unreadable_notebook_exits_2(self):
        her = self.root / "her.ipynb"
        her.write_text("not json", encoding="utf-8")
        built = self.notebook("built.ipynb", BUILT)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code_ = verify_her.main(["--her", str(her), "--built", str(built), "--folder", str(self.folder)])
        self.assertEqual(code_, 2)
        self.assertIn("cannot read her notebook", err.getvalue())

    def test_item_ids_are_found_in_the_shapes_the_real_lists_use(self):
        (self.folder / "items.md").write_text(
            "- Group 1 · the cell stopped · C-24 / R-01\nR-02, R-03 in a row\n| N-13 | renames |\n",
            encoding="utf-8",
        )
        self.assertEqual(verify_her.read_items(self.folder), {"C-24", "R-01", "R-02", "R-03", "N-13"})

    def test_missing_approved_md_means_nothing_approved(self):
        (self.folder / "approved.md").unlink()
        code_, out, _ = self.run_main(self.changed_her())
        self.assertEqual(code_, 1)
        self.assertEqual(self.count(out, "approved"), 0)

    def test_script_does_not_write_into_the_folder(self):
        before = sorted(p.name for p in self.folder.iterdir())
        self.run_main(self.changed_her())
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
