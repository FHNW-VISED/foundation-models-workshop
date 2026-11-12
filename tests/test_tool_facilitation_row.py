"""facilitation_row.py: copies the night agent's row from latest-task.md into FACILITATION.md.

Replace by first cell, append after the last row, byte-exact elsewhere, and exit 1 (writing nothing)
on a malformed row or a missing file or table.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "facilitation_row.py"

HEADER = (
    "| Section | You present (what to say) | They do | Pause / interaction | Min |\n|---|---|---|---|---|\n"
)
FAC = (
    "# Facilitation guide\n\nSession: x  \n\n## Section-by-section\n\n"
    + HEADER
    + "| 0 · Setup | old zero | a | b | 5 |\n"
    + "| 1 · Data | old one | a | b | 10 |\n"
    + "\n## Where people get stuck\n\n| Not | the | table |\n|---|---|---|\n| 1 · Data | x | y |\n"
)


def task(*body):
    return "# Task\n\nDone.\n\n## Facilitation row\n\n" + "\n".join(body) + "\n\n## Next\n\nmore\n"


class FacilitationRowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.task = self.dir / "latest-task.md"
        self.fac = self.dir / "FACILITATION.md"

    def run_script(self, task_text, fac_text=FAC, newline=None):
        if task_text is not None:
            self.task.write_text(task_text, encoding="utf-8")
        if fac_text is not None:
            with open(self.fac, "w", encoding="utf-8", newline="") as fh:
                fh.write(fac_text)
        res = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.task), str(self.fac)], capture_output=True, text=True
        )
        after = self.fac.read_bytes().decode() if self.fac.exists() else None
        return res, after

    def test_no_heading_is_a_no_op(self):
        res, after = self.run_script("# Task\n\n| 1 · Data | a | b | c | 1 |\n")
        self.assertEqual((res.returncode, after), (0, FAC))

    def test_missing_task_file_is_a_no_op(self):
        self.fac.write_text(FAC)
        res = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.dir / "nope.md"), str(self.fac)], capture_output=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertEqual(self.fac.read_text(), FAC)

    def test_replaces_the_matching_row_only(self):
        res, after = self.run_script(task("| 1 · Data | new one | x | y | 12 |"))
        self.assertEqual(res.returncode, 0)
        self.assertEqual(
            after,
            FAC.replace("| 1 · Data | old one | a | b | 10 |\n", "| 1 · Data | new one | x | y | 12 |\n", 1),
        )
        self.assertIn("| 1 · Data | x | y |\n", after)  # the other table is untouched

    def test_first_cell_compared_with_spaces_stripped(self):
        res, after = self.run_script(task("|1 · Data|new|x|y|12|"))
        self.assertEqual(res.returncode, 0)
        self.assertIn("|1 · Data|new|x|y|12|\n", after)
        self.assertNotIn("old one", after)
        self.assertEqual(after.count("1 · Data"), 2)  # one in the section table, one in the other table

    def test_appends_after_the_last_row_when_no_match(self):
        res, after = self.run_script(task("| 2 · Plots | new | x | y | 7 |"))
        self.assertEqual(res.returncode, 0)
        self.assertEqual(
            after,
            FAC.replace(
                "| 1 · Data | old one | a | b | 10 |\n",
                "| 1 · Data | old one | a | b | 10 |\n| 2 · Plots | new | x | y | 7 |\n",
                1,
            ),
        )

    def test_append_when_table_ends_the_file_without_newline(self):
        fac = "## S\n\n" + HEADER + "| 0 · Setup | a | b | c | 1 |"
        res, after = self.run_script(task("| 1 · Data | n | x | y | 2 |"), fac)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(after, fac + "\n| 1 · Data | n | x | y | 2 |")

    def test_crlf_file_keeps_its_line_endings(self):
        fac = FAC.replace("\n", "\r\n")
        res, after = self.run_script(task("| 0 · Setup | z | x | y | 3 |"), fac)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(
            after, fac.replace("| 0 · Setup | old zero | a | b | 5 |", "| 0 · Setup | z | x | y | 3 |")
        )

    def test_same_row_changes_nothing(self):
        res, after = self.run_script(task("| 1 · Data | old one | a | b | 10 |"))
        self.assertEqual((res.returncode, after), (0, FAC))

    def test_header_and_separator_lines_are_skipped(self):
        res, after = self.run_script(task(HEADER.rstrip("\n"), "| 2 · Plots | new | x | y | 7 |"))
        self.assertEqual(res.returncode, 0)
        self.assertIn("| 2 · Plots | new | x | y | 7 |\n", after)

    def test_only_the_first_row_is_used(self):
        res, after = self.run_script(task("| 2 · A | a | b | c | 1 |", "| 3 · B | a | b | c | 1 |"))
        self.assertIn("2 · A", after)
        self.assertNotIn("3 · B", after)

    def test_escaped_pipe_is_one_cell(self):
        res, after = self.run_script(task(r"| 2 · P | a \| b | x | y | 7 |"))
        self.assertEqual(res.returncode, 0)
        self.assertIn(r"| 2 · P | a \| b | x | y | 7 |", after)

    def assert_fails_and_writes_nothing(self, task_text, fac_text=FAC):
        res, after = self.run_script(task_text, fac_text)
        self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
        self.assertEqual(len((res.stdout + res.stderr).strip().splitlines()), 1)
        if fac_text is not None:
            self.assertEqual(after, fac_text)

    def test_wrong_column_count(self):
        self.assert_fails_and_writes_nothing(task("| 2 · Plots | only | three |"))
        self.assert_fails_and_writes_nothing(task("| 2 · Plots | a | b | c | d | 7 |"))

    def test_heading_without_a_row(self):
        self.assert_fails_and_writes_nothing(task("no table here"))

    def test_empty_first_cell(self):
        self.assert_fails_and_writes_nothing(task("|  | a | b | c | 1 |"))

    def test_missing_facilitation_file(self):
        self.task.write_text(task("| 2 · P | a | b | c | 1 |"))
        res = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.task), str(self.dir / "missing.md")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertFalse((self.dir / "missing.md").exists())

    def test_missing_section_table(self):
        self.assert_fails_and_writes_nothing(task("| 2 · P | a | b | c | 1 |"), "# Guide\n\nNo table.\n")

    def test_never_deletes_a_row(self):
        res, after = self.run_script(task("| 0 · Setup | z | x | y | 3 |"))
        self.assertEqual(after.count("\n| "), FAC.count("\n| "))


if __name__ == "__main__":
    unittest.main()
