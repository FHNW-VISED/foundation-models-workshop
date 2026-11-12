"""rules_review.py: RULES-review.md keeps the `review` rules byte for byte, lists script-checked ids."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rules_review.py"

ROWS_BASE = [
    "| N-001 | Plain review rule. | review |",
    "| N-002 | Script rule that says review in its text but is checked by lint. | lint |",
    "| N-003 | Smoke rule. | test |",
    "| N-004 | Colour words (🟢 🟡). | lint (colour words) + review |",
    "| N-005 | Question numbers match. | review; lint warns on (4) for question numbers |",
    "| N-006 | Pipe rule: use a \\| b inside a cell, never a bare one. | review |",
    "| N-007 | Escaped pipe but a script checks it: a \\| b. | lint |",
]
ROWS_OVERLAY = [
    "| SN-001 | Her rule. | review |",
    "| SN-002 | Her script rule. | lint |",
    "| SN-003 | Her smoke rule. | test |",
]
RULES = (
    "# RULES — every notebook\n\n"
    "Copied from the ledger.\n"
    "Check: `lint` = script · `review` = read for it.\n\n"
    "| ID | Rule | Check |\n|---|---|---|\n" + "\n".join(ROWS_BASE) + "\n\n"
    "## Rules for notebooks made with Susanne Suter\n\n"
    "These beat a conflicting rule above.\n\n"
    "| ID | Rule | Check |\n|---|---|---|\n" + "\n".join(ROWS_OVERLAY) + "\n"
)


class RulesReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rules = self.tmp / "RULES.md"
        self.out = self.tmp / "RULES-review.md"
        self.rules.write_text(RULES, encoding="utf-8")

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.tmp), *args], capture_output=True, text=True
        )

    def generated(self):
        self.assertEqual(self.run_script().returncode, 0)
        return self.out.read_text(encoding="utf-8")

    def test_keeps_exactly_the_review_rows(self):
        lines = self.generated().split("\n")
        kept = [x for x in lines if x.startswith("| N-") or x.startswith("| SN-")]
        self.assertEqual([x.split(" ")[1] for x in kept], ["N-001", "N-004", "N-005", "N-006", "SN-001"])

    def test_kept_rows_are_byte_identical(self):
        text = self.generated()
        original = RULES.split("\n")
        for line in text.split("\n"):
            if line.startswith("| N-") or line.startswith("| SN-"):
                self.assertIn(line, original)
        for row in (ROWS_BASE[3], ROWS_BASE[4], ROWS_BASE[5]):
            self.assertIn(row + "\n", text)

    def test_escaped_pipe_row_is_kept_and_classified_on_the_check_cell(self):
        text = self.generated()
        self.assertIn("| N-006 | Pipe rule: use a \\| b inside a cell, never a bare one. | review |\n", text)
        self.assertNotIn("N-007 |", text)
        self.assertIn("N-007", text.split("Checked by script, not listed:")[1].split("\n")[0])

    def test_titles_headings_headers_and_generated_line(self):
        text = self.generated()
        lines = text.split("\n")
        self.assertEqual(lines[0], "# RULES — every notebook")
        self.assertTrue(
            lines[2].startswith("Generated from RULES.md by scripts/rules_review.py. Do not edit.")
        )
        self.assertIn("## Rules for notebooks made with Susanne Suter", lines)
        self.assertEqual(lines.count("| ID | Rule | Check |"), 2)
        self.assertEqual(lines.count("|---|---|---|"), 2)
        self.assertIn("Copied from the ledger.", lines)
        self.assertTrue(text.endswith("\n") and not text.endswith("\n\n"))

    def test_not_listed_line_per_table(self):
        notes = [x for x in self.generated().split("\n") if x.startswith("Checked by script")]
        self.assertEqual(
            notes,
            [
                "Checked by script, not listed: N-002, N-003, N-007.",
                "Checked by script, not listed: SN-002, SN-003.",
            ],
        )

    def test_nothing_lost(self):
        text = self.generated()
        listed = ", ".join(x for x in text.split("\n") if x.startswith("Checked by script"))
        for row in ROWS_BASE + ROWS_OVERLAY:
            rid = row.split(" ")[1]
            kept = (row + "\n") in text
            self.assertNotEqual(kept, rid in listed, rid)  # exactly one of kept or listed

    def test_check_missing_stale_current(self):
        warning = "WARNING RULES-review.md is stale: run just rules-review\n"
        missing = self.run_script("--check")
        self.assertEqual((missing.returncode, missing.stdout), (0, warning))
        self.out.write_text("old\n", encoding="utf-8")
        stale = self.run_script("--check")
        self.assertEqual((stale.returncode, stale.stdout), (0, warning))
        self.run_script()
        current = self.run_script("--check")
        self.assertEqual((current.returncode, current.stdout), (0, ""))
        self.rules.write_text(RULES + "| SN-004 | New. | review |\n", encoding="utf-8")
        self.assertEqual(self.run_script("--check").stdout, warning)

    def test_check_without_rules_is_silent(self):
        self.rules.unlink()
        done = self.run_script("--check")
        self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_empty_drop_list_says_none(self):
        self.rules.write_text("# T\n\n| ID | Rule | Check |\n|---|---|---|\n| X-1 | r | review |\n")
        self.assertIn("Checked by script, not listed: none.\n", self.generated())


if __name__ == "__main__":
    unittest.main()
