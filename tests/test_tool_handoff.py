"""handoff.py: new, check, start and night, in a throwaway git repo."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "handoff.py"
VAULT = "/tmp/vault-for-tests"


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp()).resolve() / "my-notebook"
        self.repo.mkdir()
        self.addCleanup(shutil.rmtree, self.repo.parent, ignore_errors=True)
        for cmd in (["init", "-q", "-b", "main"], ["config", "user.name", "T"],
                    ["config", "user.email", "t@example.com"]):
            self.git(*cmd)
        (self.repo / "SPEC.md").write_text("spec text\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")
        self.env = {**os.environ, "NB_VAULT": VAULT}

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, check=True)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args, "--root", str(self.repo)],
                              capture_output=True, text=True, env=self.env)

    def new(self, *extra):
        return self.run_cli("new", "--phase", "susanne-copy/01-port", "--workflow", "susanne-copy",
                            "--model", "sonnet", *extra)

    @property
    def handoff(self):
        return self.repo / "reports" / "handoff.md"

    def fill(self, replace_lines=None):
        text = self.handoff.read_text(encoding="utf-8")
        out, section = [], None
        for line in text.splitlines():
            if line.startswith("## "):
                section = line[3:]
            if "<fill in>" in line:
                if section == "Files that matter":
                    line = "- SPEC.md — the spec"
                elif section == "Start the next session":
                    line = line.replace("<fill in>", "01-port")
                else:
                    line = "- finished, abc1234"
            out.append(line)
        self.handoff.write_text("\n".join(out) + "\n", encoding="utf-8")

    def edit(self, old, new):
        text = self.handoff.read_text(encoding="utf-8")
        self.handoff.write_text(text.replace(old, new, 1), encoding="utf-8")

    def check(self):
        return self.run_cli("check")

    def filled(self):
        self.assertEqual(self.new().returncode, 0)
        self.fill()
        done = self.check()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_new_writes_frontmatter_and_sections_in_order(self):
        self.assertEqual(self.new().returncode, 0)
        text = self.handoff.read_text(encoding="utf-8")
        keys = re.findall(r"^(\w+):", text.split("---")[1], re.M)
        self.assertEqual(keys, ["phase", "workflow", "written", "branch", "worktree", "model_next"])
        self.assertIn("branch: main", text)
        self.assertIn(f"worktree: {self.repo}", text)
        self.assertIn("# Handoff — my-notebook — after susanne-copy/01-port", text)
        heads = re.findall(r"^## (.+)$", text, re.M)
        self.assertEqual(heads, ["Done", "Left", "Files that matter", "Decisions taken",
                                 "Open questions for Simone", "Start the next session"])
        self.assertIn(f"cd {self.repo} && claude --model sonnet --add-dir {VAULT}", text)
        self.assertIn(f"{VAULT}/.claude/skills/notebook/phases/susanne-copy/<fill in>.md", text)

    def test_fresh_file_fails_check_naming_every_placeholder(self):
        self.new()
        n = self.handoff.read_text(encoding="utf-8").count("<fill in>")
        done = self.check()
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout.count("not replaced"), n)
        self.assertEqual(n, 6)

    def test_filled_file_passes(self):
        self.filled()

    def test_block_quote_fails(self):
        self.filled()
        self.edit("- finished, abc1234", "> she said so")
        self.assertIn("block-quote", self.check().stdout)

    def test_long_quoted_span_fails_short_passes(self):
        self.filled()
        self.edit("- finished, abc1234", '- she wrote "one two three four five six seven" ok')
        self.assertEqual(self.check().returncode, 0)  # seven words
        self.edit('seven" ok', 'seven eight" ok')
        self.assertIn("quoted span of 8 words", self.check().stdout)
        self.edit('"one two three four five six seven eight"', "“one two three four five six seven eight”")
        self.assertIn("quoted span of 8 words", self.check().stdout)

    def test_missing_section_fails(self):
        self.filled()
        self.edit("## Decisions taken\n", "")
        out = self.check()
        self.assertEqual(out.returncode, 1)
        self.assertIn("Decisions taken", out.stdout)

    def test_wrong_order_fails(self):
        self.filled()
        text = self.handoff.read_text(encoding="utf-8")
        text = text.replace("## Left", "## TMP").replace("## Done", "## Left").replace("## TMP", "## Done")
        self.handoff.write_text(text, encoding="utf-8")
        out = self.check()
        self.assertEqual(out.returncode, 1)
        self.assertIn("required order", out.stdout)

    def test_missing_frontmatter_key_fails(self):
        self.filled()
        self.edit("model_next: sonnet\n", "")
        self.assertIn("model_next", self.check().stdout)

    def test_too_long_fails(self):
        self.filled()
        self.edit("## Left\n", "## Left\n" + "- more\n" * 50)
        self.assertIn("at most 60", self.check().stdout)

    def test_missing_path_fails_absolute_path_passes(self):
        self.filled()
        self.edit("- SPEC.md — the spec", "- nope/missing.md")
        self.assertIn("path does not exist: nope/missing.md", self.check().stdout)
        self.edit("- nope/missing.md", f"- {self.repo}/SPEC.md")
        self.assertEqual(self.check().returncode, 0)

    def test_start_block_without_claude_command_fails(self):
        self.filled()
        self.edit(f"cd {self.repo} && claude", f"cd {self.repo} && echo")
        self.assertIn("cd … && claude", self.check().stdout)

    def test_text_fence_needs_one_line(self):
        self.filled()
        self.edit("```text\n", "```text\nextra line\n")
        self.assertIn("exactly one line", self.check().stdout)

    def test_refuses_uncommitted_overwrite_then_force(self):
        self.new()
        again = self.new()
        self.assertEqual(again.returncode, 1)
        self.assertIn("uncommitted", again.stderr)
        self.assertEqual(self.new("--force").returncode, 0)

    def test_overwrites_committed_file(self):
        self.new()
        self.git("add", "reports/handoff.md")
        self.git("commit", "-q", "-m", "handoff")
        self.assertEqual(self.new().returncode, 0)

    def test_start_prints_announcement(self):
        self.filled()
        out = self.run_cli("start")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertEqual(
            lines[0], "Phase susanne-copy/01-port is done. Start a new session for the next phase:"
        )
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], f"  cd {self.repo} && claude --model sonnet --add-dir {VAULT}")
        self.assertEqual(lines[3:6], ["", "Then type:", ""])
        self.assertEqual(lines[6], f"  Read reports/handoff.md and {VAULT}/.claude/skills/notebook/phases/"
                                   "susanne-copy/01-port.md and do that phase only.")
        total = self.handoff.stat().st_size + (self.repo / "SPEC.md").stat().st_size
        self.assertEqual(lines[8], f"The next session reads about {total / 4000:.1f}k tokens first: "
                                   f"{total} bytes in handoff.md plus the files it names.")

    def night(self, report_text):
        report = self.repo / "reports" / "night-2026-11-12.md"
        report.parent.mkdir(exist_ok=True)
        report.write_text(report_text, encoding="utf-8")
        return self.run_cli("night", str(report))

    def test_night_with_headings(self):
        done = self.night("# Night\n\n## ✅ #12 — a\n\ntext\n## ✅ #13 — b\n## ❌ #14 — c (stopped here)\n"
                          "## ⏸ #15 — d (paused)\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        text = self.handoff.read_text(encoding="utf-8")
        self.assertIn("issues done: #12, #13", text)
        self.assertIn("#14, #15", text)
        self.assertIn("review the report, decide merge or fix", text)
        self.assertIn("git diff main...main --stat", text)
        self.assertIn("phase: new-notebook/night", text)
        self.assertIn(f"{VAULT}/.claude/skills/notebook/phases/new-notebook/03-morning.md", text)
        self.assertIn("- reports/night-2026-11-12.md", text)
        out = self.check()
        self.assertEqual(out.returncode, 0, out.stdout)

    def test_night_without_headings_still_valid(self):
        done = self.night("# Night\n\nnothing ran\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("no done issue found", self.handoff.read_text(encoding="utf-8"))
        self.assertEqual(self.check().returncode, 0)


if __name__ == "__main__":
    unittest.main()
