# RULES — every notebook

Copied from the second-brain ledger `wiki/work/notebooks/Notebook Rules.md` on 2026-10-26.
Do not edit here; SPEC.md › Notebook-specific rules (L-xxx) override these for this notebook.
Check: `lint` = scripts/lint_notebook.py · `test` = smoke test · `review` = read for it.

| ID | Rule | Check |
|---|---|---|
| N-001 | One notebook, one storyline. Advanced material goes in clearly marked optional 🔴 parts, never in a second notebook. | review |
| N-002 | State 2–3 learning outcomes at the top ("At the end you will be able to …") and restate them in a closing wrap-up with the takeaways. | review |
| N-003 | Every section opens with a markdown heading and an `**Objective:**` line. | lint |
| N-004 | Each section follows task → short explanation → *predict* → run → result → reflect. Ask for the prediction before the interactive cell. | review |
| N-005 | Every question carries a level marker: 🟢 beginner · 🟡 intermediate · 🔴 advanced (optional). The default path is mostly 🟢/🟡; parameter tuning is 🔴. How to calibrate: 🟢 = the answer can be read straight off a plot or a printed output; 🟡 = it needs reasoning beyond what is shown (e.g. how a box or mask changes shape when the image is resized); 🔴 = an optional extension, rare when the topics are intuitive. | lint |
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
| N-024 | Every worked answer sits in a collapsed toggle (`<details>`), hidden until the learner opens it, answer figures and printed answer keys included (a figure is drawn as HTML output inside the toggle). No SPEC override. | review |
| N-025 | Every kind of part sits in a Callout, and each kind keeps one colour and icon for the whole notebook. Theory Callouts name their topic. | review |
| N-026 | Every place where a learner writes code or changes a value is unmistakable: an action Callout right above the cell, and stubs that run without errors before they are touched. Not foolproof: learners may write code. | review |
| N-027 | Each notebook has its own Theme (palette, title banner, figure style), and a colour means the same thing everywhere in it (e.g. one colour per class across every plot and overlay). | review |
| N-028 | The SPEC states the learner Freedom level (low / medium / high), and the mix of run-and-read, change-a-value and write-code cells follows it. | review |
| N-029 | Learner-code cells never block a full run: the learner fills a stub with a given signature, a `use(fn, reference=…)` cell checks it against the reference solution on the Play Dataset ("✅ matches" / "❌ cat: 12 vs 14"), and the notebook continues with the reference either way. | review |
| N-030 | Workshop notebooks: everything heavy (models, data, packages) is downloaded and installed **before** the session by a setup-check notebook shared in advance. Nothing installs during the workshop. | review |
| N-031 | Every code cell a participant runs is preceded by one sentence saying what it does and why. Understanding over execution. | review |
| N-032 | The material (notebook, setup check, data) is shared before the session, and the required skills and prerequisites are stated at the top of the notebook. | review |
| N-033 | Every notebook ships a facilitation guide (`FACILITATION.md`) for whoever presents it: per section, what to say, what learners do, minutes, where to pause, the show-of-hands or pair moments. It is generated from the SPEC. | review |
| N-034 | Callouts and the title banner are HTML **output** of a hidden code cell (`# @title` with no text, form view) whose output the build saves into the `.ipynb`, so the box shows before anything runs and re-running redraws it. Never markdown tables for coloured boxes: Colab sets `white-space: nowrap; overflow: hidden` on table cells and cuts the text. Coloured text in a markdown blockquote is the fallback. | lint |
| N-035 | Every cell the learner only runs (no value to change, no code to write) is collapsed with `# @title <what it does>`, plotting and demo cells included. Only cells the learner edits show code. | lint |
| N-036 | No walls of text. A markdown cell holds at most three short paragraphs or one list; enumerations and questions are bullet lists, link collections a compact list or table with one line each, and blank space separates blocks. | review |
| N-037 | Show before asking: learners see example images or data before any question that asks them to judge that data. | review |
| N-038 | Every theory or checklist point carries a concrete example from the notebook's own story (e.g. the park's cameras in winter and summer), and every criterion a question asks about must be answerable from what the notebook shows. | review |
| N-039 | Every statistic is defined in one plain line (how it is computed) where it first appears, and shown on real images: the minimum, a typical (median) and the maximum side by side. | review |
| N-040 | Distributions are drawn as violin plots, not box plots. | lint |
| N-041 | A question gives the learner every input it needs (invent concrete numbers if necessary), and a question about the shape of a plot shows tiny example plots of the possible shapes right under it. | review |
| N-042 | A non-obvious mechanism gets a small visual and, where it helps, its formula (bilinear interpolation, perceptual-hash bit distance). When a result would look identical to the reference in the same encoding, show it in a different one (e.g. a heatmap of the interpolated mask). | review |
| N-043 | No cryptic sentences: every abstract claim is anchored in a concrete scenario, and a scenario is written as short numbered steps, each with its consequence, plus a small visual of what the learner should look for. | review |
| N-044 | Few Callout kinds, each with its own job, and no Callout colour that can be mistaken for a level dot (🟢🟡🔴). Kinds a learner cannot tell apart at a glance are merged. *(Rewritten 2026-10-23; was: "Objective lines and Takeaways each have their own colour. A Takeaway is a list of 'When you see X → do Y' lines." The Takeaway format survives as a notebook's choice.)* | review |
| N-045 | No licence or attribution text in learner-facing notebooks; credits live in the repo (README, `data/ATTRIBUTION.md`). | review |
| N-046 | The reference solution of a learner-code cell sits in a collapsed helper cell above the stub, never visible right under it; `use(…)` still falls back to it so Run all never blocks (N-029). | review |
| N-047 | The notebook fits its slot. Each section has a minute budget in the SPEC; after every build a cutting review proposes what to drop, and only Simone approves removals. | review |
| N-048 | Main points are visually dominant; depth is visibly secondary (collapsed `<details>` or a muted "Go deeper" callout), so a learner who only reads the coloured parts still gets the lesson. | review |
| N-049 | An abstract pitfall is shown with a tiny, slightly playful simulation on the notebook's own data (e.g. a fake model that always answers "sheep"), not only described. | review |
| N-050 | Every ✍️ YOUR CODE docstring ends with a made-up example of the return value (`e.g. {"train": ["a.jpg", …], "test": ["b.jpg", …]}`), so the learner sees the shape before writing. | lint |
| N-051 | A figure that shows only part of a set (the first 3 of 12 folds, 5 of 40 photos) says so in the image itself: dots (⋮ or …) after the last shown item and an "N more not shown" label. | review |
| N-052 | The order follows the learner. (1) A hunt or multi-step task opens with what the learner will do, in order (a numbered roadmap). (2) Everything a question needs (pictures, definitions) comes before it, and nothing that answers it does: no figure, caption, number, example or box that gives the answer away (a picture that *shows* a case the question asks about gives it away; show a correct row and, under it, the same picture with an error, and let the question name each picture it refers to by its title, e.g. Picture A). (3) A question sits right after its material; only its own ✏️/✍️ box may stand between the question and its run cell, and side boxes (go-deeper, "try this afterwards") come after. (4) Nothing refers to a question, cell or picture before it appears ("run it in Q9 below"). (5) The worked answer comes after the cell that shows the result. | review; lint warns on (4) for question numbers |
| N-053 | A mechanism that recurs (the ✍️ stub with its reference fallback, a control panel) is introduced once, as a short demo that says what it is and what the learner will see, before its first real use. Do not announce how many times it will come. | review |
| N-054 | Callout text is the same size as the notebook's markdown body text; the HTML of a Callout never sets a smaller font. | review |
| N-055 | A question's (or Task's) level dot is in its heading, so the table of contents shows where the tasks are and how hard they are. | review |
| N-056 | Section numbers in headings and in the table of contents are generated by the build from the section order, never typed by hand, so they cannot go stale. | review |
| N-057 | An image gallery or example view lets the learner browse: pick the class, or step next/previous, instead of always showing the same first N images. | review |

## Rules for notebooks made with Susanne Suter

Copied from `wiki/work/styles/susanne-suter/susanne-suter-notebooks.md`. **These beat a conflicting rule above.**
Where one of them contradicts an L-rule in SPEC.md, it wins too, and the SPEC is brought in line at the next pass.

| ID | Rule | Check |
|---|---|---|
| SN-001 | Few boxes. Use a Callout only where it has its own job; the kinds she keeps are the 💡 answer toggle, the blue 📘 theory/info box, the grey 🔍 go-deeper box and the Task box (SN-002). Anything else is plain markdown or merges into one of these. Overrides N-025's "every kind of part sits in a Callout". | review |
| SN-002 | Questions and code exercises are one kind: a **Task**, in one style, marked 🟢 beginner · 🟡 intermediate · 🔴 advanced. The legend reads "Tasks are marked beginner, intermediate, advanced." | review |
| SN-003 | A section's objective is plain markdown right under the section heading, not a box. Overrides the objective half of the old N-044. | review |
| SN-004 | Quoted reference material (the Karpathy *Recipe* quotes) uses the theory blue; it does not get a colour of its own. | review |
| SN-005 | Learning objectives are written with Bloom's-taxonomy verbs, so each one states the level asked: describe, explain, apply, implement, justify. | review |
| SN-006 | Screen every photo a student can see before the notebook is handed over: no nudity, swimwear or bare torsos. Search the dataset's captions or tags for such words where they exist, look at every match, and list the excluded ids with a reason in the data-building script, removed after sampling so the rest of the selection stays the same. A caption search does not see pixels, so anything she or a reviewer spots later is excluded the same way. | review |
