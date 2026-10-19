# SPEC — When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation

The single source of truth for **what** this notebook is. `AGENTS.md` says how to work,
`RULES.md` says what every notebook must respect, and this file says what *this* one must do.
It is filled in by the `/notebook grill` session. Agents read it at the start of every task and
never change it, except the `Status` column of the sections table.

## 1. Purpose
<!-- One paragraph: who uses it, in which course/event, and the single question it answers. -->

## 2. Audience and level
<!-- Who the learners are, what they already know, and what they do NOT know. Default path level. -->

## 2b. Prerequisites
<!-- What learners must already know or have (accounts, laptop, Python level). -->

## 3. Learning outcomes (2–3)
<!-- "At the end you will be able to …". These appear verbatim in sections/00_setup.py. -->
1.
2.

## 4. Use case and data
<!-- The concrete scenario that makes the answers and metrics meaningful. Dataset: name, source URL,
     licence, size, how it is fetched (never committed if > 5 MB), what the SMOKE subset is. -->

## 5. Environment
| Item | Value |
|---|---|
| Runs on | Google Colab (web). Local Jupyter only for development |
| Colab runtime | CPU / T4 GPU |
| Python | 3.12 |
| Packages not preinstalled in Colab | |
| Pinned versions (Colab, checked YYYY-MM-DD) | |
| Smoke run | `NB_SMOKE=1`, CPU, budget 900 s |

## 6. Sections
Time = minutes of participant time. Status: todo · in-progress · done · blocked.

| # | File | Title | Objective (one line) | Level | Time | Status |
|---|---|---|---|---|---|---|
| 0 | sections/00_setup.py | Setup | Get running; see the outcomes | 🟢 | 5 | done |

## 6b. Pitfalls and extensions
<!-- Common learner mistakes the notebook should anticipate, and optional 🔴 extensions for fast learners. -->

## 7. Out of scope / limitations
<!-- What this notebook deliberately does not cover (e.g. "no training — that is next week"). -->

## 8. Notebook-specific rules
<!-- Rules that apply only here, each with its source. They take precedence over RULES.md where they
     conflict. Format: L-001 — rule — source. -->

## 9. Acceptance (definition of done for the whole notebook)
- `just check` passes, including the content assertions in `tests/test_content.py` (written with Simone, never by the night agent).
- Runs top to bottom on a fresh kernel; no hidden state; re-running a section from its start gives the same result (N-013).
- Every section: markdown → small code cell → interpretation of the output (no wall of code without a sentence on what to look at).
- At least one exercise per section with an answer scaffold.
- Outputs are tidy: no warnings spam, no 1,000-row prints, figures labelled.
- A full (non-smoke) run in Colab by Simone, with the runtime above, finishes with no errors in ≤ __ min.
- 

## 10. Open questions
<!-- Things only Simone or Susanne can answer. An agent that hits one stops and lists it in its report. -->
