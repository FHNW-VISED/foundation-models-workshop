"""context_watch.py: one notice at the limit, the next only after another 50k, never a non-zero exit."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "context_watch.py"


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # the script keeps its state in <repo>/build, so run a copy inside a throwaway repo layout
        (self.tmp / "scripts").mkdir()
        shutil.copy(SCRIPT, self.tmp / "scripts" / "context_watch.py")
        self.transcript = self.tmp / "t.jsonl"

    def write(self, tokens, noise=True):
        lines = []
        if noise:
            lines.append(json.dumps({"type": "user", "message": {"content": "hi"}}))
        usage = {
            "input_tokens": 10,
            "cache_creation_input_tokens": 90,
            "cache_read_input_tokens": tokens - 100,
        }
        lines.append(json.dumps({"type": "assistant", "message": {"usage": usage}}))
        lines.append(json.dumps({"type": "user", "message": {"content": "next"}}))
        self.transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_hook(self, session="s1", transcript=None, stdin=None, env=None):
        payload = stdin if stdin is not None else json.dumps(
            {"session_id": session, "transcript_path": str(transcript or self.transcript)})
        return subprocess.run(
            [sys.executable, str(self.tmp / "scripts" / "context_watch.py")],
            input=payload, capture_output=True, text=True, env={**os.environ, **(env or {})},
        )

    def test_below_limit_prints_nothing(self):
        self.write(149_999)
        done = self.run_hook()
        self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_notices_follow_the_step(self):
        self.write(150_000)
        first = self.run_hook()
        self.assertEqual(first.returncode, 0)
        out = json.loads(first.stdout)["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "UserPromptSubmit")
        self.assertIn("reports/handoff.md", out["additionalContext"])
        self.assertIn("150k", out["additionalContext"])
        self.write(160_000)
        self.assertEqual(self.run_hook().stdout, "")
        self.write(200_000)
        second = self.run_hook()
        self.assertIn("200k", json.loads(second.stdout)["hookSpecificOutput"]["additionalContext"])
        self.write(210_000)
        self.assertEqual(self.run_hook().stdout, "")

    def test_sessions_are_independent(self):
        self.write(150_000)
        self.assertNotEqual(self.run_hook(session="a").stdout, "")
        self.assertNotEqual(self.run_hook(session="b").stdout, "")

    def test_limit_from_env(self):
        self.write(5_000)
        done = self.run_hook(env={"CONTEXT_WATCH_LIMIT": "4000"})
        self.assertIn("hookSpecificOutput", done.stdout)

    def test_state_file_is_written(self):
        self.write(150_000)
        self.run_hook()
        state = json.loads((self.tmp / "build" / "context-watch.json").read_text())
        self.assertEqual(state, {"s1": 150_000})

    def test_corrupt_transcript_prints_nothing(self):
        self.transcript.write_text("not json\n{broken", encoding="utf-8")
        done = self.run_hook()
        self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_missing_transcript_prints_nothing(self):
        done = self.run_hook(transcript=self.tmp / "absent.jsonl")
        self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_bad_stdin_prints_nothing(self):
        for payload in ("", "[]", '{"session_id": "x"}'):
            with self.subTest(payload=payload):
                done = self.run_hook(stdin=payload)
                self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_reads_only_the_tail(self):
        big = json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 900_000}}})
        filler = json.dumps({"type": "user", "message": {"content": "x" * 1000}})
        usage = {"input_tokens": 5, "cache_read_input_tokens": 1000}
        last = json.dumps({"type": "assistant", "message": {"usage": usage}})
        body = "\n".join([big, *([filler] * 400), last]) + "\n"
        self.transcript.write_text(body, encoding="utf-8")
        self.assertEqual(self.run_hook().stdout, "")  # the 900k line is beyond the last 300 KB


if __name__ == "__main__":
    unittest.main()
