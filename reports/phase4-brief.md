# Phase 4 brief: apply the reviewers' non-removal suggestions (2026-10-21)

The three reviews are `reports/review-facilitator.md`, `reports/review-participant.md` and `reports/review-cuts.md`.
Read the parts about your sections; their line numbers match the current files. All three agree: the notebook runs
~230–270 min against 175, and the cause is **repetition and scrolling**, not compute.

## Allowed in this phase
Rewording, tightening, reordering cells inside your files, merging cells that say the same thing, moving secondary
text into a 🔍 `go-deeper` callout, collapsing run-only cells, fixing errors, and small visual or rendering changes.

## Not allowed (proposals only: list them under "Cut ideas" in your reply)
Removing a section, part, question, figure, exercise or code cell. Changing a 🟡 to 🔴. Moving the break. Adding
`@param` sliders (they are proposals for Simone). Changing what code computes or any `RESULTS` key. Renumbering
questions.

## Cross-cutting decisions (apply in your files)
1. **One colour for "do code", another for "think".** `your-turn` (orange) is only for cells where the learner edits
   code or a value. Every "Your prediction" box becomes kind `question` with the title `🤔 Your prediction` (the
   title emoji replaces the icon; keep any "(1 min)" suffix).
2. **No empty "Question" boxes.** A `question` callout whose body is only "Run the next cell…" is merged with the
   `**Qn.**` text that follows: one `question` callout holding the instruction line and the Qn line(s). Where a
   prediction, a "Question" box and the Qn are about the same cell, merge them into **one** callout (prediction
   first, then Qn). The lint still sees `**Qn.** 🟡` inside callouts. Keep the `# Qn` code-cell header when the Qn
   shows code (N-007).
3. **Cell intros: one sentence** (N-031). The markdown line before a collapsed cell says what it shows in one
   sentence and does not repeat the figure title.
4. **Takeaways: at most 4 "When … → …" lines**, none repeating a pitfall of the same section word for word.
5. **"Over-trusting SAM's confident masks"** is a full pitfall only in **02** (first time) and **06** (with the
   blind-spot simulation). In 01 and 07 it shrinks to one sentence (07: inside the Takeaway).
6. **Error-map colours** are the same everywhere: violet = right, **orange = false alarm** (marked but not true),
   **green = missed** (true but not marked), as in 01/04. Fix any text (and plot) that says otherwise (03:378-380).
7. **Define at first use, one line each:** fundus (02), zero-shot (01), balanced loss (01, "each class gets half the
   total weight"), 1024 px crops of a 3504×2336 fundus (02, where the data is introduced), PU = positive-unlabelled
   (05), teacher (06). Don't define them again later.
8. **Citations and literature** go into `go-deeper` callouts, not main theory boxes.
9. **Skipping a 🔴 part must never crash a later cell** (FACILITATION lists them as skippable). Guard the consumer:
   a later cell that reads a name or `RESULTS` key from a 🔴 cell checks it exists and skips that row/print with
   one short note.

## Reply (short)
What you applied (one line each, with file), what you did not and why, cut ideas (with minutes), anything you
changed in code. You still have no shell; the orchestrator runs `just check`.
