"""night.sh in a synthetic repo with stub gh, claude, just, uv and caffeinate; nothing real is called.

Covers the model line and warning, the template-drift stop, the RULES-review.md prompt, the protected list,
the strings shared with helper.sh, the facilitation row and the morning handoff on the success path, and
that the flags and variables of the original script still parse. Runs under /bin/bash (3.2 on macOS).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent
SCRIPTS = TEMPLATE / "scripts"
BASH = "/bin/bash"
DATE = "2026-01-01"
PROTECTED_LINE = (
    "PROTECTED=(tests scripts RULES.md RULES-review.md justfile .github AGENTS.md CLAUDE.md .claude versions)"
)
STUB_ONLY = (
    "scripts/subagent_guard.py",
    "scripts/helper.sh",
    "scripts/context_watch.py",
    "scripts/port_her.py",
    "scripts/verify_her.py",
    "scripts/sections_info.py",
    "scripts/rules_review.py",
    ".claude/settings.json",
)
REAL = ("scripts/night.sh", "scripts/handoff.py", "scripts/template_diff.py", "scripts/facilitation_row.py")

GH = r"""#!/bin/bash
case "$1 $2" in
  "issue list") echo 7 ;;
  "issue view") case "$*" in *title*) echo "Stub title" ;; *) printf 'Stub body\nFILES: x\n' ;; esac ;;
  "pr view") exit 1 ;;
  *) exit 0 ;;
esac
"""
CLAUDE = r"""#!/bin/bash
echo called >> "$RECORD.calls"
printf '%s\0' "$@" > "$RECORD.args"
for last; do :; done
printf '%s' "$last" > "$RECORD.prompt"
mkdir -p reports
[ -n "${TASK_NOTE:-}" ] && printf '%b' "$TASK_NOTE" > reports/latest-task.md
echo '{"result":"ok"}'
"""
OK_STUB = "#!/bin/bash\nexit 0\n"
FAC = (
    "# Guide\n\n## Section-by-section\n\n"
    "| Section | You present (what to say) | They do | Pause / interaction | Min |\n|---|---|---|---|---|\n"
    "| 0 · Setup | old | a | b | 5 |\n\n## Where people get stuck\n"
)


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True
    )


class NightBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name).resolve()
        self.root, self.repo, self.tpl = root, root / "repo", root / "template"
        self.bin, self.record = root / "bin", root / "record"
        self.bin.mkdir()
        for name, body in (
            ("gh", GH),
            ("claude", CLAUDE),
            ("just", OK_STUB),
            ("uv", OK_STUB),
            ("caffeinate", OK_STUB),
        ):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        for rel in STUB_ONLY:
            self.put(self.repo / rel, "stub\n")
        for rel in REAL:
            self.put(self.repo / rel, (TEMPLATE / rel).read_text())
        (self.repo / "FACILITATION.md").write_text(FAC)
        (self.repo / ".gitignore").write_text("build/\n")
        shutil.copytree(self.repo / "scripts", self.tpl / "scripts")
        shutil.copytree(self.repo / ".claude", self.tpl / ".claude")
        origin = root / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.repo)], check=True)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "init")
        git(self.repo, "remote", "add", "origin", str(origin))
        git(self.repo, "push", "-q", "origin", "main")

    @staticmethod
    def put(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit_file(self, rel, text):
        self.put(self.repo / rel, text)
        git(self.repo, "add", rel)
        git(self.repo, "commit", "-q", "-m", f"add {rel}")

    def night(self, *args, env=None):
        base = {k: v for k, v in os.environ.items() if not k.startswith(("NIGHT_", "NB_"))}
        base.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            NB_TEMPLATE=str(self.tpl),
            NB_VAULT=str(self.root / "vault"),
            NIGHT_DATE=DATE,
            RECORD=str(self.record),
            # night.sh commits; CI runners have no git identity (seen on ubuntu CI, 2026-11-12)
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@t",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@t",
        )
        base.update(env or {})
        return subprocess.run(
            [BASH, str(self.repo / "scripts/night.sh"), *args],
            cwd=self.repo,
            env=base,
            capture_output=True,
            text=True,
        )

    def prompt(self):
        return Path(f"{self.record}.prompt").read_text()


class DryRunTest(NightBase):
    def test_default_model_line_before_the_prompt(self):
        res = self.night("--dry-run")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("model: claude-sonnet-5-5 (from default)", res.stdout)
        self.assertLess(res.stdout.index("model: "), res.stdout.index("--- prompt for #7"))
        self.assertNotIn("WARNING", res.stderr)

    def test_opus_from_the_variable_warns_and_continues(self):
        res = self.night("--dry-run", env={"NIGHT_MODEL": "claude-opus-5-5"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("model: claude-opus-5-5 (from NIGHT_MODEL)", res.stdout)
        self.assertIn(
            "WARNING: not Sonnet. The efficiency rule says Sonnet builds, "
            "Opus only for the SPEC and the final review.",
            res.stderr,
        )
        self.assertIn("--- prompt for #7", res.stdout)

    def test_flag_wins_over_the_variable(self):
        res = self.night("--dry-run", "--model", "claude-sonnet-5-5", env={"NIGHT_MODEL": "claude-opus-5-5"})
        self.assertIn("model: claude-sonnet-5-5 (from --model)", res.stdout)
        self.assertNotIn("WARNING", res.stderr)
        self.assertIn("--model claude-sonnet-5-5", res.stdout)

    def test_flag_opus_warns(self):
        res = self.night("--dry-run", "--model", "claude-opus-5-5")
        self.assertIn("(from --model)", res.stdout)
        self.assertIn("WARNING: not Sonnet", res.stderr)

    def test_other_flags_and_variables_still_parse(self):
        res = self.night("--dry-run", "--label", "night", "--max", "1", "--retries", "0")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("issues to run: 7", res.stdout)
        res = self.night("--dry-run", env={"NIGHT_UNTIL": "1", "NIGHT_SUBAGENT_MODEL": "haiku"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.night("--bogus").returncode, 2)
        self.assertEqual(self.night("--dry-run", env={"NIGHT_SUBAGENT_MODEL": "fable"}).returncode, 2)

    def test_prompt_tells_the_agent_the_night_rules(self):
        out = self.night("--dry-run").stdout
        self.assertIn("do not open FACILITATION.md", out)
        self.assertIn("## Facilitation row", out)
        self.assertIn("| Section | You present (what to say) | They do | Pause / interaction | Min |", out)
        self.assertIn("Any Agent call must name model haiku or sonnet.", out)

    def test_prompt_names_rules_review_only_when_it_exists(self):
        out = self.night("--dry-run").stdout
        self.assertIn("read SPEC.md, RULES.md, this issue,", out)
        self.assertNotIn("RULES-review.md", out.split("--- prompt")[1].split("--- ISSUE")[0])
        self.commit_file("RULES-review.md", "short\n")
        out = self.night("--dry-run").stdout
        self.assertIn("read SPEC.md, RULES-review.md (RULES.md is the full source), this issue,", out)


class TemplateCheckTest(NightBase):
    def test_a_differing_script_stops_the_run_with_exit_4(self):
        (self.tpl / "scripts/helper.sh").write_text("newer\n")
        res = self.night("--dry-run")
        self.assertEqual(res.returncode, 4)
        self.assertIn("DIFF    scripts/helper.sh", res.stdout)
        self.assertNotIn("--- prompt", res.stdout)

    def test_a_missing_script_stops_the_run_with_exit_4(self):
        (self.tpl / "scripts/verify_her.py").unlink()
        res = self.night("--dry-run")
        self.assertEqual(res.returncode, 4)
        self.assertIn("MISSING scripts/verify_her.py", res.stdout)

    def test_skip_variable_bypasses_the_check(self):
        (self.tpl / "scripts/helper.sh").write_text("newer\n")
        res = self.night("--dry-run", env={"NIGHT_SKIP_TEMPLATE_CHECK": "1"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("--- prompt", res.stdout)

    def test_missing_template_folder_warns_and_continues(self):
        res = self.night("--dry-run", env={"NB_TEMPLATE": str(self.root / "nowhere")})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("WARNING", res.stderr)
        self.assertIn("--- prompt", res.stdout)


class StringsTest(unittest.TestCase):
    def setUp(self):
        self.night = (SCRIPTS / "night.sh").read_text()

    def test_protected_line(self):
        self.assertIn("\n" + PROTECTED_LINE + "\n", self.night)

    def test_tools_and_guard_match_helper_sh(self):
        helper_path = SCRIPTS / "helper.sh"
        if not helper_path.is_file():
            self.skipTest("helper.sh not present")
        helper = helper_path.read_text()
        tools = re.search(r'^TOOLS="([^"]*)"$', self.night, re.M).group(1)
        h_tools = re.search(r'^TOOLS="([^"]*)"$', helper, re.M).group(1)
        self.assertIn("Bash(uv add:*)", tools)
        self.assertEqual(tools.replace(",Bash(uv add:*)", ""), h_tools)
        guard = re.search(r'^GUARD="([^"]*)"$', self.night, re.M).group(1)
        self.assertEqual(guard, re.search(r'^GUARD="([^"]*)"$', helper, re.M).group(1))
        settings = re.search(r'--settings "(.*)"\)$', self.night, re.M).group(1)
        self.assertEqual(settings, re.search(r'^SETTINGS="(.*)"$', helper, re.M).group(1))
        common = (
            '--permission-mode acceptEdits --allowedTools "$TOOLS" --strict-mcp-config '
            "--mcp-config '{\"mcpServers\":{}}' --output-format json)"
        )
        self.assertIn(common, self.night)
        self.assertIn(common, helper)

    def test_lock_limit_file_and_resume_check_are_kept(self):
        for text in (
            'LOCK="build/night.lock"',
            'LIMITFILE="build/night.limit"; rm -f "$LIMITFILE"',
            'if grep -q "^night: #$n " <<<"$(git log --format=%s main..HEAD)"',
            'revert_protected "$n" "$body"',
        ):
            self.assertIn(text, self.night)


class FullRunTest(NightBase):
    ROW = "| 0 · Setup | new | x | y | 9 |"

    def run_full(self, note, **kw):
        res = self.night(env={"TASK_NOTE": note, **kw.pop("env", {})})
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        return res

    def tree(self, rev, path):
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"], cwd=self.repo, capture_output=True, text=True
        ).stdout

    def test_row_is_committed_with_the_issue_and_handoff_with_the_report(self):
        self.run_full(f"Did the thing.\n\n## Facilitation row\n\n{self.ROW}\n")
        branch = f"night/{DATE}"
        self.assertIn("| 0 · Setup | new | x | y | 9 |", self.tree(branch, "FACILITATION.md"))
        self.assertNotIn("| old |", self.tree(branch, "FACILITATION.md"))
        subjects = subprocess.run(
            ["git", "log", "--format=%s", "main.." + branch], cwd=self.repo, capture_output=True, text=True
        ).stdout.split("\n")
        self.assertIn("night: #7 Stub title", subjects)
        changed = subprocess.run(
            ["git", "show", "--name-only", "--format=", f"{branch}~1"],
            cwd=self.repo,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn("FACILITATION.md", changed)
        report = self.tree(branch, f"reports/night-{DATE}.md")
        self.assertIn("## ✅ #7 — Stub title", report)
        self.assertNotIn("facilitation row not written", report)
        self.assertIn("phase:", self.tree(branch, "reports/handoff.md"))
        self.assertIn("#7", self.tree(branch, "reports/handoff.md"))

    def test_malformed_row_is_noted_and_does_not_stop_the_run(self):
        self.run_full("Did it.\n\n## Facilitation row\n\n| 0 · Setup | only | three |\n")
        branch = f"night/{DATE}"
        self.assertEqual(self.tree(branch, "FACILITATION.md"), FAC)
        report = self.tree(branch, f"reports/night-{DATE}.md")
        self.assertIn("## ✅ #7", report)
        self.assertRegex(report, r"facilitation row not written: .*3 columns")

    def test_no_heading_leaves_the_file_alone(self):
        self.run_full("Did it.\n")
        branch = f"night/{DATE}"
        self.assertEqual(self.tree(branch, "FACILITATION.md"), FAC)
        self.assertNotIn("facilitation row not written", self.tree(branch, f"reports/night-{DATE}.md"))

    def test_without_the_two_scripts_the_run_still_finishes(self):
        for rel in ("scripts/facilitation_row.py", "scripts/handoff.py"):
            git(self.repo, "rm", "-q", rel)
        git(self.repo, "commit", "-q", "-m", "drop scripts")
        git(self.repo, "push", "-q", "origin", "main")  # the night branch starts from origin/main
        res = self.night(
            env={"TASK_NOTE": f"x\n\n## Facilitation row\n\n{self.ROW}\n", "NIGHT_SKIP_TEMPLATE_CHECK": "1"}
        )
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        branch = f"night/{DATE}"
        self.assertEqual(self.tree(branch, "FACILITATION.md"), FAC)
        self.assertEqual(self.tree(branch, "reports/handoff.md"), "")
        self.assertIn("## ✅ #7", self.tree(branch, f"reports/night-{DATE}.md"))

    # night.sh formats the cut-off with BSD `date -r <epoch>`; GNU date (Linux CI) reads -r as a file
    @unittest.skipUnless(sys.platform == "darwin", "night.sh runs on macOS only")
    def test_cut_off_variable_still_stops_before_the_first_issue(self):
        res = self.night(env={"NIGHT_UNTIL": "1", "TASK_NOTE": "x"})
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertFalse(Path(f"{self.record}.calls").exists())
        self.assertIn("stopped at #7 (cut-off", self.tree(f"night/{DATE}", f"reports/night-{DATE}.md"))


if __name__ == "__main__":
    unittest.main()
