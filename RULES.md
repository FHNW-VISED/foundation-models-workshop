# RULES — every notebook

Copied from the second-brain ledger `wiki/work/notebooks/Notebook Rules.md` on 2026-10-19.
Do not edit here; SPEC.md › Notebook-specific rules (L-xxx) override these for this notebook.
Check: `lint` = scripts/lint_notebook.py · `test` = smoke test · `review` = read for it.

| ID | Rule | Check |
|---|---|---|
| N-001 | One notebook, one storyline. Advanced material goes in clearly marked optional 🔴 parts, never in a second notebook. | review |
| N-002 | State 2–3 learning outcomes at the top ("At the end you will be able to …") and restate them in a closing wrap-up with the takeaways. | review |
| N-003 | Every section opens with a markdown heading and an `**Objective:**` line. | lint |
| N-004 | Each section follows task → short explanation → *predict* → run → result → reflect. Ask for the prediction before the interactive cell. | review |
| N-005 | Every question carries a level marker: 🟢 beginner · 🟡 intermediate · 🔴 advanced (optional). The default path is mostly 🟢/🟡; parameter tuning is 🔴. | lint |
| N-006 | Questions are numbered **Q1…Qn** continuously and sit inside the section they belong to. | lint |
| N-007 | When a question asks the learner to run code, the next code cell starts with `# Qn` (the "Put Qn code here" cell). | lint |
| N-008 | Helper-only code cells (defs or imports with no learner-facing work) start with `# @title …` so Colab collapses them. Learners see only the code they are meant to read or change. | lint |
| N-009 | Hands-on early: the learner runs or changes something within the first ~10 minutes. Show first, then explain. No long front-loaded theory, no repetition. | review |
| N-010 | Introduce a concept with a visual before going deeper into code or theory. Keep one consistent colour scheme across figures. | review |
| N-011 | Use few metrics and explain each one. Prefer looking at correct vs incorrect examples over metric tables. | review |
| N-012 | A question must not presuppose a particular result. It must stay answerable whatever the seed, defaults or hardware. | review |
| N-013 | Re-running a section from its first cell resets its state: models are rebuilt, not continued, and a fixed `SEED` is used. | review |
| N-014 | One `SMOKE` flag (set from `NB_SMOKE=1`) shrinks every expensive step (data size, epochs, model size) to seconds on CPU while running the *same* code path. | lint |
| N-015 | The smoke run finishes within its budget (default 900 s). | test |
| N-016 | `SPEC.md`'s sections table and the files in `sections/` match one to one. | lint |
| N-017 | Learner instructions assume Colab on the web. Local setup is not a supported path in the text. | review |
| N-018 | Warn before the first slow step (e.g. "the first epoch is slow while Colab caches the images"). | review |
| N-019 | Data never goes into git if it is over 5 MB. A cell downloads it idempotently from a stable source and survives a half-finished download. | review |
| N-020 | Everything learner-facing is in English. | review |
| N-021 | Install with `%pip`, not `!pip`, and only what Colab does not already have. | lint |
| N-022 | Parameter (control-panel) cells validate their values and fail with a plain message, not a deep stack trace. | review |
| N-023 | When the notebook follows a known recipe or paper, say so near the top and link it. | review |
| N-024 | Questions that end in a written answer get a hidden worked answer (`<details>`), unless the SPEC says the answers are shown openly. | review |
