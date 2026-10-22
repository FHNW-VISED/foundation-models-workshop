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
