"""Structure checks: RULES.md rules that a script can decide (see scripts/lint_notebook.py)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_lint_passes():
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lint_notebook.py")], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout


def test_forward_question_reference_is_found():
    sys.path.insert(0, str(ROOT / "scripts"))
    from lint_notebook import forward_question_refs

    assert forward_question_refs("You run it in Q9 below.", next_q=9) == [9]
    assert forward_question_refs("As in Q8, count the rows (Q12 later).", next_q=9) == [12]
    assert forward_question_refs("**Q9.** 🟢 Run it; Q9 asks for the count.", next_q=9) == []
    assert forward_question_refs("Worked answer (Q24): which check finds which kind", next_q=25) == []
    assert forward_question_refs("QA and Q-learning are not questions.", next_q=1) == []


def test_settings_rule_n058(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import lint_notebook

    def section(name: str, *cells: str) -> Path:
        path = tmp_path / name
        path.write_text("".join(f"# %%\n{c}\n\n" for c in cells))
        return path

    files = [
        section(
            "01_a.py",
            "THRESHOLD = 8\nRESULTS = {}\nRESULTS['n'] = 1\n"
            "if THRESHOLD > 4:\n    MODE = 'a'\nelse:\n    MODE = 'b'",
            "TABLE = load_table(THRESHOLD)",
            "CONFIG = {'a': 1}\nCONFIG['b'] = 2",
            "# ✏️ YOUR TURN\nCLASS_TO_SHOW = 'dog'",
            "FLAG = True  # @param {type:'boolean'}\nSTEPS = 30",
        ),
        section("02_b.py", "THRESHOLD = 9", "print(CLASS_TO_SHOW, STEPS)"),
    ]
    lint_notebook.problems.clear()
    lint_notebook.check_settings(files)
    found = "\n".join(lint_notebook.problems)
    assert "TABLE is computed (reads load_table)" in found
    assert "CONFIG is changed in place" in found
    assert "THRESHOLD is set again" in found
    assert "CLASS_TO_SHOW is the learner's ✏️ setting in sections/01_a.py" in found
    for fine in ("RESULTS", "MODE", "STEPS", "FLAG"):
        assert f" {fine} " not in found, found
    assert len(lint_notebook.problems) == 4, found
