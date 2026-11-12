"""subagent_guard.py: blocks Fable, forks and a call with no model (exit 2); everything else passes."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "subagent_guard.py"


def run(call):
    return subprocess.run(
        [sys.executable, str(GUARD)], input=json.dumps(call), capture_output=True, text=True
    )


def agent(**tool_input):
    return {"tool_name": "Agent", "tool_input": {"prompt": "x", **tool_input}}


class GuardTest(unittest.TestCase):
    def test_other_tool_passes(self):
        self.assertEqual(run({"tool_name": "Bash", "tool_input": {"command": "ls"}}).returncode, 0)

    def test_allowed_models_pass(self):
        for model in ("sonnet", "haiku", "opus"):
            with self.subTest(model=model):
                self.assertEqual(run(agent(model=model)).returncode, 0)

    def test_fable_blocked(self):
        done = run(agent(model="fable"))
        self.assertEqual(done.returncode, 2)
        self.assertIn("Fable", done.stderr)

    def test_fork_blocked(self):
        self.assertEqual(run(agent(model="sonnet", subagent_type="fork")).returncode, 2)

    def test_missing_model_blocked(self):
        done = run(agent())
        self.assertEqual(done.returncode, 2)
        self.assertIn("haiku", done.stderr)
        self.assertIn("sonnet", done.stderr)

    def test_empty_model_blocked(self):
        self.assertEqual(run(agent(model="")).returncode, 2)


if __name__ == "__main__":
    unittest.main()
