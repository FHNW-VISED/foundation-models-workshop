"""helper.sh: starts one headless helper inside the repo root, on Sonnet, with a turn cap that is always set.

A fake `claude` records its working directory, environment and arguments. The script runs under
/bin/bash (3.2 on macOS) so a bash-4-only feature breaks the test.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent
BASH = "/bin/bash"
RESULT_JSON = (
    '{"result":"line1\\nline2","num_turns":3,"usage":{"cache_read_input_tokens":10,"output_tokens":2}}'
)
FAKE = """#!/bin/bash
pwd -P > "$RECORD.cwd"
printf '%s' "$CLAUDE_CODE_SUBAGENT_MODEL" > "$RECORD.env"
for a in "$@"; do printf '%s\\0' "$a"; done >> "$RECORD.args"
echo called >> "$RECORD.calls"
if read -r _line; then echo "stdin had data" >> "$RECORD.stdin"; fi
cat <<'JSON'
__JSON__
JSON
"""


class HelperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name).resolve()
        self.repo = tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("helper.sh", "subagent_guard.py"):
            shutil.copy(TEMPLATE / "scripts" / name, self.repo / "scripts" / name)
        self.record = tmp / "record"
        self.fake = tmp / "fake-claude"
        self.write_fake(RESULT_JSON)
        self.brief = tmp / "brief.md"
        self.brief.write_text("Do the thing.\nSecond line.\n")
        self.elsewhere = tmp / "elsewhere"
        self.elsewhere.mkdir()

    def write_fake(self, payload):
        self.fake.write_text(FAKE.replace("__JSON__", payload))
        self.fake.chmod(0o755)

    def run_helper(self, *args, cwd=None, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k not in ("HELPER_MODEL", "CLAUDE_BIN")}
        env.update({"CLAUDE_BIN": str(self.fake), "RECORD": str(self.record)})
        env.update(env_extra or {})
        return subprocess.run(
            [BASH, str(self.repo / "scripts" / "helper.sh"), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd or self.elsewhere),
            env=env,
        )

    def called(self):
        return Path(str(self.record) + ".calls").exists()

    def recorded_args(self):
        raw = Path(str(self.record) + ".args").read_bytes().decode()
        return raw.split("\0")[:-1]

    def value_of(self, args, flag):
        return args[args.index(flag) + 1]

    def test_stdin_is_not_passed_to_claude(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HELPER_MODEL", "CLAUDE_BIN")}
        env.update({"CLAUDE_BIN": str(self.fake), "RECORD": str(self.record)})
        subprocess.run(
            [BASH, str(self.repo / "scripts" / "helper.sh"), "sin", str(self.brief)],
            input="junk on stdin\n",
            capture_output=True,
            text=True,
            cwd=str(self.elsewhere),
            env=env,
        )
        self.assertTrue(self.called())
        self.assertFalse(Path(str(self.record) + ".stdin").exists())

    def test_cap_hit_says_why(self):
        self.write_fake('{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":3}')
        out = self.run_helper("cap", str(self.brief))
        self.assertIn("helper cap: stopped: error_max_turns", out.stdout)
        self.assertNotIn("None", out.stdout)
        self.assertEqual((self.repo / "build/helpers/cap.md").read_text().strip(), "")

    # --- the call -------------------------------------------------------------------------------

    def test_default_call(self):
        done = self.run_helper("probe", str(self.brief))
        self.assertEqual(done.returncode, 0, done.stderr)
        args = self.recorded_args()
        self.assertEqual(self.value_of(args, "--model"), "sonnet")
        self.assertEqual(self.value_of(args, "--max-turns"), "25")
        self.assertEqual(self.value_of(args, "--permission-mode"), "acceptEdits")
        self.assertEqual(self.value_of(args, "--output-format"), "json")
        self.assertIn("-p", args)
        self.assertIn("--strict-mcp-config", args)
        self.assertEqual(json.loads(self.value_of(args, "--mcp-config")), {"mcpServers": {}})
        self.assertEqual(Path(str(self.record) + ".env").read_text(), "sonnet")

    def test_tools_are_nights_minus_uv_add(self):
        night = (TEMPLATE / "scripts" / "night.sh").read_text().splitlines()
        night_tools = next(line for line in night if line.startswith("TOOLS=")).split('"')[1]
        self.assertIn("Bash(uv add:*)", night_tools)
        expected = night_tools.replace(",Bash(uv add:*)", "")
        self.run_helper("probe", str(self.brief))
        args = self.recorded_args()
        self.assertEqual(self.value_of(args, "--allowedTools"), expected)
        self.assertNotIn("uv add", self.value_of(args, "--allowedTools"))

    def test_guard_settings_name_the_repo_guard(self):
        self.run_helper("probe", str(self.brief))
        settings = json.loads(self.value_of(self.recorded_args(), "--settings"))
        entry = settings["hooks"]["PreToolUse"][0]
        self.assertEqual(entry["matcher"], "Agent|Task")
        hook = entry["hooks"][0]
        self.assertEqual(hook["type"], "command")
        self.assertEqual(hook["command"], f"python3 {self.repo}/scripts/subagent_guard.py")

    def test_explicit_model_and_cap(self):
        self.run_helper("probe", str(self.brief), "--model", "haiku", "--max-turns", "7")
        args = self.recorded_args()
        self.assertEqual(self.value_of(args, "--model"), "haiku")
        self.assertEqual(self.value_of(args, "--max-turns"), "7")

    def test_model_from_environment(self):
        self.run_helper("probe", str(self.brief), env_extra={"HELPER_MODEL": "opus"})
        self.assertEqual(self.value_of(self.recorded_args(), "--model"), "opus")

    def test_prompt_is_brief_then_cap_footer(self):
        self.run_helper("probe", str(self.brief), "--max-turns", "12")
        prompt = self.recorded_args()[-1]
        self.assertTrue(prompt.startswith("Do the thing.\nSecond line.\n\n"))
        last = prompt.splitlines()[-1]
        self.assertIn("at most 12 turns", last)
        self.assertIn("at most 15 lines", last)
        self.assertIn("Do not run git commit, push or branch.", last)
        # N01: the guard blocks an Agent call with no model, so the helper is told to name one.
        self.assertTrue(last.endswith("Any Agent call must name model haiku or sonnet."))

    def test_runs_in_repo_root_even_when_started_elsewhere(self):
        self.run_helper("probe", str(self.brief), cwd=self.elsewhere)
        self.assertEqual(Path(str(self.record) + ".cwd").read_text().strip(), str(self.repo))

    # --- refusals -------------------------------------------------------------------------------

    def refused(self, *args, **kw):
        done = self.run_helper(*args, **kw)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertFalse(self.called(), "claude must not be called")
        self.assertEqual(len(done.stderr.strip().splitlines()), 1, done.stderr)

    def test_fable_refused_any_case(self):
        for model in ("fable", "Fable-5", "claude-FABLE-x"):
            with self.subTest(model=model):
                self.refused("probe", str(self.brief), "--model", model)

    def test_fable_refused_from_environment(self):
        self.refused("probe", str(self.brief), env_extra={"HELPER_MODEL": "Fable"})

    def test_bad_names(self):
        for name in ("Probe", "a_b", "a b", "../x", "-x", "a/b"):
            with self.subTest(name=name):
                self.refused(name, str(self.brief))

    def test_brief_missing_empty_or_too_big(self):
        self.refused("probe", str(self.brief.parent / "nope.md"))
        empty = self.brief.parent / "empty.md"
        empty.write_text("")
        self.refused("probe", str(empty))
        big = self.brief.parent / "big.md"
        big.write_text("x" * 25000)
        self.refused("probe", str(big))

    def test_brief_at_limit_is_accepted(self):
        ok = self.brief.parent / "ok.md"
        ok.write_text("x" * 20000)
        self.assertEqual(self.run_helper("probe", str(ok)).returncode, 0)

    def test_cap_out_of_range_or_not_a_number(self):
        for cap in ("61", "0", "-1", "abc", ""):
            with self.subTest(cap=cap):
                self.refused("probe", str(self.brief), "--max-turns", cap)
        self.assertEqual(self.run_helper("probe", str(self.brief), "--max-turns", "60").returncode, 0)

    def test_unknown_flag_or_missing_args(self):
        self.refused("probe")
        self.refused("probe", str(self.brief), "--turbo")

    # --- dry run and output ---------------------------------------------------------------------

    def test_dry_run_calls_nothing(self):
        done = self.run_helper("probe", str(self.brief), "--dry-run")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse(self.called())
        self.assertFalse((self.repo / "build").exists())
        self.assertIn("model: sonnet", done.stdout)
        self.assertIn("max-turns: 25", done.stdout)
        self.assertIn("Bash(just:*)", done.stdout)
        self.assertIn("subagent_guard.py", done.stdout)
        prompt = done.stdout.split("--- prompt (first 10 lines) ---\n", 1)[1]
        self.assertLessEqual(len(prompt.splitlines()), 10)
        self.assertIn("Do the thing.", prompt)

    def test_dry_run_prompt_shows_only_first_ten_lines(self):
        long_brief = self.brief.parent / "long.md"
        long_brief.write_text("".join(f"row {i}\n" for i in range(30)))
        done = self.run_helper("probe", str(long_brief), "--dry-run")
        prompt = done.stdout.split("--- prompt (first 10 lines) ---\n", 1)[1]
        self.assertEqual(prompt.splitlines(), [f"row {i}" for i in range(10)])

    def test_output_files_and_short_print(self):
        lines = "\\n".join(f"r{i}" for i in range(100))
        self.write_fake(
            '{"result":"' + lines + '","num_turns":3,'
            '"usage":{"cache_read_input_tokens":10,"output_tokens":2}}'
        )
        done = self.run_helper("probe", str(self.brief))
        self.assertEqual(done.returncode, 0, done.stderr)
        raw = self.repo / "build" / "helpers" / "probe.json"
        md = self.repo / "build" / "helpers" / "probe.md"
        self.assertEqual(json.loads(raw.read_text())["num_turns"], 3)
        self.assertEqual(len(md.read_text().strip().splitlines()), 100)
        out = done.stdout.splitlines()
        self.assertLessEqual(len(out), 41)
        self.assertEqual(out[-2], "r99")
        self.assertEqual(out[-1], "helper probe: model sonnet, turns 3, cache_read 10, output 2")

    def test_missing_usage_fields_print_unknown(self):
        self.write_fake('{"result":"ok"}')
        done = self.run_helper("probe", str(self.brief))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            done.stdout.splitlines()[-1],
            "helper probe: model sonnet, turns unknown, cache_read unknown, output unknown",
        )

    def test_exit_code_is_claudes(self):
        self.fake.write_text('#!/bin/bash\necho \'{"result":"boom"}\'\nexit 3\n')
        done = self.run_helper("probe", str(self.brief))
        self.assertEqual(done.returncode, 3)
        self.assertIn("boom", done.stdout)

    def test_non_json_output_does_not_crash(self):
        self.fake.write_text("#!/bin/bash\necho 'Error: nope'\nexit 1\n")
        done = self.run_helper("probe", str(self.brief))
        self.assertEqual(done.returncode, 1)
        self.assertIn("Error: nope", done.stdout)

    # --- script hygiene -------------------------------------------------------------------------

    def test_script_text(self):
        text = (TEMPLATE / "scripts" / "helper.sh").read_text()
        for banned in ("pkill", "killall", "kill ", ",,}", "mapfile", "declare -A", "readarray"):
            self.assertNotIn(banned, text)

    @unittest.skipUnless(Path(BASH).exists(), "no /bin/bash")
    def test_runs_under_system_bash(self):
        version = subprocess.run([BASH, "-c", "echo $BASH_VERSION"], capture_output=True, text=True)
        self.assertEqual(self.run_helper("probe", str(self.brief)).returncode, 0, version.stdout)


if __name__ == "__main__":
    unittest.main()
