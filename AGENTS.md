# When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation — agent instructions

This repo builds **one teaching notebook**, `foundation-models-workshop.ipynb`. Learners open it in Google Colab.
The `.ipynb` is a **build artifact**. The source is the small Jupytext files in `sections/`.

## Read first, and only this

1. `SPEC.md`: what the notebook must do (purpose, audience, sections table, environment, scope, L-rules).
2. `RULES.md`: rules every notebook respects (N-rules). L-rules in SPEC override them.
3. The issue you are working on, then **only the section file(s) it names**.
4. To learn what earlier sections define, run `just api` (every def/class with file:line) and read just those lines.

⛔ **Never open or edit `foundation-models-workshop.ipynb` or `build/`.** They are large JSON. Edit `sections/NN_*.py` and run `just build`.
⛔ Do not read the whole `sections/` folder "for context". `just api` plus the SPEC table is the context.
⛔ Do not edit `SPEC.md`, except the **Status** column of the sections table for the section you finished.
⛔ Do not edit `RULES.md`, `scripts/`, `tests/`, `pyproject.toml` or `uv.lock` unless the issue names them.
`FACILITATION.md` (N-033): when you finish a section, add or update its row in the section-by-section table.

## Commands

| Command | What it does |
|---|---|
| `just check` | **The acceptance gate**: rule lint + ruff + build + full headless smoke run. Must pass before you finish. |
| `just build` | Assemble `sections/*.py` → `foundation-models-workshop.ipynb` |
| `just lint` | Rule checks a script can decide + ruff |
| `just smoke` | Build, then execute the whole notebook with `NB_SMOKE=1` (tiny data, CPU) |
| `just api` | List every function and class defined in the notebook |

Environment: `uv` only (`uv add <pkg>` if an issue allows a new dependency). Never `pip install` into the system.

⛔ **Subagents are never Fable.** If you use the Agent tool, pass `model: "haiku"` (searches, lookups) or
`"sonnet"` (the default); `"opus"` only for the heaviest task. Never `"fable"`, never `subagent_type: "fork"`
(a fork inherits your model). The runner blocks both.

## How the sections work

- Each file in `sections/` is Jupytext **percent** format. `# %% [markdown]` starts a markdown cell (lines prefixed `# `), and `# %%` starts a code cell.
- Files are concatenated in filename order into **one kernel**, so a section may use names defined by earlier ones (use `just api` to find them).
- A section starts with a markdown heading and an `**Objective:**` line (N-003).
- Helper-only cells start with `# @title Short name { display-mode: "form" }` so Colab collapses them (N-008).
- Questions: `**Q7.** 🟡 …` with continuous numbering across the notebook (N-005/N-006). If the learner must run code, the next code cell starts with `# Q7` (N-007).
- Every expensive step reads `SMOKE` (defined in `00_setup.py`) and shrinks to seconds on CPU while running the same code (N-014). Never skip logic in smoke mode, only shrink it.

## Finishing a task

1. `just check` passes. If it fails, read only the failure lines it prints, fix, and re-run.
2. Set the section's Status in the SPEC table to `done`.
3. Write 3–6 lines to `reports/latest-task.md`: what you built, any decision you took that the SPEC did not dictate, anything left open.
4. If the SPEC is silent or contradictory on something that changes what learners see, **do not guess**. Do the part that is clear, record the question under "Open" in `reports/latest-task.md`, and stop.

Git is handled by the runner (`scripts/night.sh`): do not commit, push, branch or open PRs.
