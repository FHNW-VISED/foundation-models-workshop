"""template_diff.py: byte comparison of the must-match scripts against the vault template."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import template_diff  # noqa: E402


class TemplateDiffTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.repo, self.template = root / "repo", root / "template"
        for base in (self.repo, self.template):
            for rel in template_diff.MUST_MATCH:
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_bytes(b"same\n")

    def run_main(self, template=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = template_diff.main(
                ["--repo", str(self.repo), "--template", str(template or self.template)]
            )
        return code, out.getvalue(), err.getvalue()

    def test_list_names_the_scripts_the_night_run_depends_on(self):
        for rel in (
            "scripts/night.sh",
            "scripts/subagent_guard.py",
            "scripts/helper.sh",
            "scripts/handoff.py",
            "scripts/facilitation_row.py",
            "scripts/template_diff.py",
            ".claude/settings.json",
        ):
            self.assertIn(rel, template_diff.MUST_MATCH)
        for rel in (
            "scripts/build.py",
            "scripts/lint_notebook.py",
            "scripts/version.py",
            "justfile",
            "AGENTS.md",
        ):
            self.assertNotIn(rel, template_diff.MUST_MATCH)

    def test_all_same_exits_0(self):
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("SAME"), len(template_diff.MUST_MATCH))

    def test_a_differing_file_is_reported_and_exits_1(self):
        (self.repo / "scripts/night.sh").write_bytes(b"same\n ")  # one extra byte
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("DIFF    scripts/night.sh", out)
        self.assertIn("SAME    scripts/helper.sh", out)

    def test_a_file_missing_in_the_repo_exits_1(self):
        (self.repo / "scripts/helper.sh").unlink()
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MISSING scripts/helper.sh", out)

    def test_a_file_missing_in_the_template_exits_1(self):
        (self.template / ".claude/settings.json").unlink()
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MISSING .claude/settings.json", out)

    def test_missing_template_folder_warns_and_exits_0(self):
        code, out, err = self.run_main(template=self.template / "nope")
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("WARNING", err)


if __name__ == "__main__":
    unittest.main()
