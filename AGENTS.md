# When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation — agent instructions

This repo builds **one teaching notebook**, `foundation-models-workshop.ipynb`. Learners open it in Google Colab.
The `.ipynb` is a **build artifact**. The source is the small Jupytext files in `sections/`.

## Read first, and only this

1. `SPEC.md`: what the notebook must do (purpose, audience, sections table, environment, scope, L-rules).
2. `RULES-review.md`: the N-rules a script cannot check. `RULES.md` is the full source. L-rules in SPEC override them.
   The rules a script checks fail `just lint` and print the failing line.
3. The issue you are working on, then **only the section file(s) it names**.
4. To learn what earlier sections define, run `just api` (every def/class with file:line) and read just those lines.

⛔ **Never open or edit `foundation-models-workshop.ipynb` or `build/`.** They are large JSON. Edit `sections/NN_*.py` and run `just build`.
⛔ Do not read the whole `sections/` folder "for context". `just api` plus the SPEC table is the context.
⛔ Do not edit `SPEC.md`, except the **Status** column of the sections table for the section you finished.
⛔ Do not edit `RULES.md`, `scripts/`, `tests/`, `pyproject.toml` or `uv.lock` unless the issue names them.
⛔ Never edit `versions/`: each folder is a frozen version (SPEC.md + notebooks). The newest one always equals `main`;
`just version "what changed"` adds the next one, and only the person or runner merging to main runs it.
`FACILITATION.md` (N-033): when you finish a section, add or update its row in the section-by-section table. In a night run, do not open FACILITATION.md: put the row under `## Facilitation row` in reports/latest-task.md and night.sh writes it.

## Phase sessions

- If the session's first sentence names `reports/handoff.md`, read that file and the phase file it names, then only the files it lists.
- Do that phase only.
- At the end: `just handoff-new <phase> <workflow> <model>`, fill it in, `just handoff-check`, commit it, `just handoff-start`, and show Simone its output word for word.
- A message starting `[context-watch]`: finish the current step, write the handoff, tell Simone to start a new session.
- Helpers start only with `scripts/helper.sh <name> <brief-file>` (Sonnet, turn cap set), never with the Agent tool from a vault session. An Agent call must name model haiku or sonnet; the guard refuses one without a model.

## Commands

| Command | What it does |
|---|---|
| `just check` | **The acceptance gate**: rule lint + ruff + build + full headless smoke run. Must pass before you finish. |
| `just build` | Assemble `sections/*.py` → `foundation-models-workshop.ipynb` |
| `just lint` | Rule checks a script can decide + ruff |
| `just smoke` | Build, then execute the whole notebook with `NB_SMOKE=1` (tiny data, CPU) |
| `just api` | List every function and class defined in the notebook |
| `just version "…"` | Snapshot SPEC.md + the notebooks as the next folder in `versions/` (before merging to main) |
| `just version-check` | Fails if the newest version differs from SPEC.md or the built notebooks |
| `just cells NN` | Cells of section NN: position, md/code, bytes, flags, first line |
| `just words` | Visible words per section against the Word budget in SPEC.md |
| `just rules-review` | Regenerate RULES-review.md after RULES.md changes |
| `just handoff-new phase workflow model` | Start `reports/handoff.md` (fill it in, then check it) |
| `just handoff-check` | Validate `reports/handoff.md` |
| `just handoff-start` | Print the command and first sentence that start the next session |
| `just helper name brief` | One headless helper in this repo (Sonnet, turn cap): `scripts/helper.sh` |

Environment: `uv` only (`uv add <pkg>` if an issue allows a new dependency). Never `pip install` into the system.

⛔ **Subagents are never Fable.** If you use the Agent tool, pass `model: "haiku"` (searches, lookups) or
`"sonnet"` (the default); `"opus"` only for the heaviest task. Never `"fable"`, never `subagent_type: "fork"`
(a fork inherits your model). The runner blocks both.

## Standing rules

- **One build, never variants.** No parallel candidate notebooks (three were built, two thrown away). An open choice
   (dataset, model) is settled in the grilling with Simone, or by ONE desk-research agent (Sonnet, ≤ 20 tool calls,
   no downloads over 100 MB, never trains) that returns a ranked table. (source: .claude/skills/notebook/SKILL.md)
- ⛔ A collaborator's structure is never the thing cut (Simone, 2026-11-09): in a `With: Susanne` notebook every
   task label, box and front-matter block stays; shorten sentences inside them, and ask him before touching structure. (source: .claude/skills/notebook/SKILL.md)
- Quotes from meetings with Susanne (or any collaborator) stay in the private vault. A repo she can read (FHNW-VISED notebook
repos) gets a paraphrased work list and rule text only, never "she said …" verbatim, never the transcript. (source: wiki/work/CLAUDE.md)
- when she edits a shared artefact herself (her copy of a notebook), **her edits are ported first, the meeting items second,
conflicts shown to Simone** (source: wiki/work/CLAUDE.md)
- **What she removes stays removed.** A part missing from her edited copy (the Challenge, the Wrap-up, the objective boxes, the Q-numbers) is intended; it is not restored, re-proposed or put to her as a question. Only what her removal *breaks* (a name a later cell still uses) is fixed. (source: wiki/work/styles/susanne-suter/susanne-suter-notebooks.md)
- When Simone has hand-edited a draft (proposal, letter, any document), change only the lines he
asks about. Every other edit, including fixes I think are improvements, is shown as a before/after
and applied only after his yes. (source: memory/feedback_hand_edits_are_his.md)
- Before creating a branch or editing files in a repo under `~/Work/notebooks/` (or any repo another Claude session
might use), run `git status` and check `ListAgents`. If the tree has changes that aren't mine, or a peer session is
busy on that project, **don't touch that clone**. Work in a separate `git worktree add <scratchpad>/… -b <branch>
origin/<base>`, and tell the peer session before doing anything to its clone. (source: memory/feedback_shared_repo_worktree.md)
- ⛔ **Versions (Simone, 2026-10-22: "you MUST keep a versioning of the notebook from now on").** Every repo has `versions/`:
one frozen folder per version (`SPEC.md` + the built notebook), and the newest folder always equals `main`. Before any
merge to main, run `just version "what changed"` on the branch being merged; after it, `just version-check` must pass on
main. Stacked PRs: retarget the upper PR to main before deleting its base branch. (source: .claude/skills/notebook/SKILL.md)
- Do not read RULES.md; read RULES-review.md. The rules it leaves out are checked by just lint, which prints the failing line.

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
