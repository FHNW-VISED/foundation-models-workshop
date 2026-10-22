# Phase 1 brief: transfer T-02 … T-12 to your sections (2026-10-21)

Shared by every Phase 1 owner. Your own prompt lists your files and section-specific items.

## Read, in this order
1. `AGENTS.md`, then `RULES.md` (N-034 … N-049 are the new rules; read their wording).
2. `reviews/2026-10-21-transfer.md`, the "Transfer items" list.
3. `reviews/eda-2026-10-21-pass-1.reference.md`: what Simone asked for in the *other* notebook. Use it to see the
   **kind** of fix he wants (R-08 … R-29 are concrete). Do not copy EDA text: this workshop is about pets, then retina
   vessels, foundation models (DINOv3, SAM) and missing labels.
4. `reports/callout-system.md`: how callouts are written (already converted in your files).
5. `SPEC.md` §6 rows for your sections and §8 (L-rules). `just api` for names defined in earlier sections.
6. Only your own section files.

## What to do in your files
- **T-02 (N-035)** Every code cell the learner only runs starts with `# @title <what it does> { display-mode: "form" }`,
  plot and demo cells included. Only cells with a value to change or code to write show code.
- **T-03 (N-044)** Objective is already a 🎯 callout. Every section ends with (or has, at its natural end) a ✅
  `takeaway` callout written as a list of "When you see X → do Y" lines grounded in what this section showed. Convert an
  existing summary/takeaway paragraph rather than adding a second one.
- **T-04 (N-036)** No walls of text: at most three short paragraphs or one list per markdown cell; enumerations and
  questions as bullets; links one per line; split long cells.
- **T-05 (N-037)** Every question or prediction about data or a result comes after the learner has seen examples of it.
  Reorder cells if needed (keep question numbers continuous; renumber only inside your files and only if the order of
  questions actually changes — tell the orchestrator if it does).
- **T-06 (N-038)** Each theory point has a concrete example from this workshop's story (pets, then the retina vessels).
  Every criterion a question asks about must be answerable from what the notebook shows.
- **T-07 (N-039)** Define each metric/statistic in one plain line where it first appears (IoU, clDice, recall, PCA
  colours of features, pseudo-label quality numbers…). Where it makes sense, show it on real images: worst / typical
  (median) / best side by side. Only where it first appears; later sections just use it.
- **T-08 (N-041, N-042)** Non-obvious mechanisms get a small visual and, where it helps, a one-line formula (the linear
  1×1 head on patch features, the masked/PU loss, Grad-CAM, the prompt ladder). When a result would look identical to
  the reference, use another encoding (heatmap, difference map). Questions carry concrete numbers; a question about a
  plot's shape gets tiny example plots of the possible shapes under it.
- **T-09 (N-040, N-043, N-048, N-049)** Rewrite cryptic sentences as short numbered steps, each with its consequence.
  Abstract pitfalls get a tiny playful simulation on our own data where cheap. Violin plots, never box plots. Depth
  that is not needed for the main line goes into a 🔍 `go-deeper` callout (collapsed): a learner who reads only the
  coloured parts still gets the lesson.
- **T-10 (N-045, L-004)** Remove licence and attribution talk from body text. Model loads keep **one short muted
  line** of source + licence (no paragraphs). The DINOv3 notice moves into a collapsed 🔍 "Credits & licences" go-deeper
  callout at the end of section 10 (owner of 10 only).
- **T-11 (N-046)** Each `_ref_*` reference solution moves into its own collapsed helper cell
  (`# @title Reference solution (open only if stuck) { display-mode: "form" }`) placed **above** the learner's stub
  cell. The stub cell keeps the `# ✍️ YOUR CODE` header, signature, docstring, `...` body and the `use(…)` call, so
  `use()` still falls back and Run all never blocks.
- **T-12 (N-047)** Tighten prose everywhere: shorter sentences, fewer words, same meaning. **Never remove** a section,
  part, question, figure or exercise: if you think something should go, write it under "Cut ideas" in your reply.

## Hard limits
- Edit only your files. Never open `.ipynb` or `build/`. Do not edit SPEC, RULES, scripts, tests, pyproject.
- Do not change what code computes, the names it defines, or keys written to `RESULTS` (tests read them).
  New visuals are fine; every new expensive step reads `SMOKE` and shrinks (N-014). New plots: L-003 palette.
- Keep question numbering and level markers (`**Q7.** 🟡`), and `# Q7` code-cell headers (N-005 … N-007).
- Callout titles plain text (no `<` or `*`); one callout per cell; marker on the first line.
- Subagents have **no shell** in this session. Instead of `just api`, use Grep with pattern `^(def|class) |^    def `
  on `sections/`. You cannot run lint, ruff or smoke: check your own work by careful reading (imports present, names
  defined earlier, no syntax errors, SMOKE respected). List the code you added; the orchestrator runs `just check`
  after all owners finish and sends failures back to you.
- Do not commit.

## Reply (keep it short, the orchestrator reads all five)
- Per T-item: done / partly / n.a. + one line, for your files.
- Code you added or changed (cell + what it does).
- Cut ideas (proposals only) with rough minutes saved.
- Anything ambiguous you decided, and open questions.
