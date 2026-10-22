# Facilitator review (live, AMLD 2027, 180 min + 30 break, Colab T4)

Read: SPEC, RULES, FACILITATION, sections 00-10. Paths are `sections/NN_*.py:line`. Bash was denied, so nothing was run.
My timings are estimates from counting cells: 17 "Your prediction" callouts, 36 questions, ~10 write-code stubs, ~75 output cells.

## 0. The three things that matter most

1. **The break is at minute 60, not 90.** Sections 0-3 budget 5+15+20+20 = 60 min, so the break (03:537) lands there. Block 2 then holds 115 min with no pause. SPEC §6 says "Block 1 = 0-3 plus the start of 4", but the notebook stops at 3. Live, I expect sections 0-3 to run ~80 min, which is fine, but the two blocks are 80 / 135 on paper.
   - Fix (a): move the break to just before 4.3 (04:380). The annotator "with a new instruction" is the literal answer to Q13, so that is still a cliffhanger. Block 1 then runs ~72 budget min, closer to the 90-min block.
2. **FACILITATION's "if short on time" cuts crash later sections.** The notebook never guards these:
   - Skipping 🔴 Q12 (03:499) leaves `balanced_head` undefined, and 05:298 uses it (NameError).
   - Skipping the 🔴 PU cell Q20 (05:467-517) breaks 08:65 and 08:76, which read `path_reports[pu]` and `RESULTS["pu_head"]`.
   - Skipping "prototype → SAM" (04:329) breaks 08:73 (`proto_sam_sec`) and the `one_shot_sam` row.
   - FACILITATION rows 3, 4 and 5 tell the presenter to skip exactly these. Fix: make those cells always-run collapsed helpers (each is ~10 s), or let 08 skip missing rows.
3. **Section 5 repeats section 4.3.** 04:503 already trains a head with unlabelled = -1, which is the masked loss. 04:420-427 and 05:22 print the same formula. 05 then says "treat missing as unknown" as if new, and the 04 Takeaway even says it is "the answer to the cliffhanger". At 05:4 you will stumble ("didn't we just do this?"). Fix (a): at 05:4 say "same trick, now measured on the section-3 file". Fix (b): merge 5.1-5.2 into 4.3 (see section 3).

## 1. Section by section

**00 Setup (5 min).**
- First 4 text cells, then a Theory cell, before any picture or run (00:4-54, 232-240). That is ~60 lines, and the first figure is at ~00:346. Fix (a): fold "What you need" and "The plan" (00:33-47) into the banner, and move "Recipe and sources" (00:50-54) to go-deeper.
- Prediction + "Answer in your head" + Q1 are 3 cells for one thought (00:349-368). Merge into one. The Q1 answer (00:363-366) says thin, branching, low-contrast, then 02:128 and 02:186 say it again. Cut the Q1 answer to one line.
- The model load is the only dead time (00:468). The Pitfall about slow first steps comes after it (00:474). Move it above the load cell (N-018). Use the load seconds to say the DINOv3 and SAM one-liners now (see 01).
- Takeaway at 00:479 is troubleshooting, not a lesson. It steals the Takeaway colour. Make it a Pitfall.

**01 Pets (15 min, I would plan ~22).**
- The three tools are explained three times: bullets 01:9-13, Theory 01:19-28, and 00:50-54. Delete the bullets, keep the Theory.
- Move that Theory below the first figure (01:140). Today the learner has read ~45 lines before the first run (N-009).
- Standalone "Question" callouts say only "Run the next cell" (01:125-127). That pattern repeats at 02:145, 02:606, 03:399, 06:375, 06:839, 07:580, 09:426. Merge each into the prediction or Q cell that follows, which saves about 8 cells of scrolling.
- The linear head is explained three times: 01:288-293 (formula), 01:421-441 (score heatmap), and 03:205-257 (step-by-step figure). The 03 version is the best picture. Move it to 01:288, replace the formula and the heatmap with it, and remove it from 03 (saves ~3 min, and the explanation arrives when the head is first used).
- The 👥 at 01:469 is dead as written. Everyone has `PETS_EVAL[0]` and the same seed, so the neighbour sees the identical picture. Say "pick different indices" (as 02:698 does).
- The floor-click experiment (01:476-492) is the best surprise in the section but comes last, after Q4 and the pair moment. Move it right after the Q2 figure (01:155). The Pitfall at 01:495 then lands immediately.

**02 Falls short (20 min, I would plan ~35).**
- This is the busiest Block 1 section: ~11 output cells, 5 questions, 3 predictions, a ✍️, a 👥 and an own-image cell.
- The punchline chart "Same tools, two tasks" (02:719-743) sits after the 6-figure gallery and a 5-min pair. Move it to right after the 512/1024 scoring cell (02:586) so the gap is on screen at about minute 25. Keep the gallery and 👥 as the "now look closer" part.
- The 2.5 Pitfall (02:587-595) is attention arithmetic (1M, 17M, 2.3B pairs). Move it to go-deeper (N-048); the lesson is one line.
- 2.4 PCA on the retina (02:459-490, Q7) repeats 1.3 and tells nothing the head result does not. Collapse it to a one-line Go deeper (b: ~3 min).
- Pair-moment bug: 02:698 says "pick a different `GALLERY_START` or `demo` index". Changing `demo` after the ladder (02:366 stores `ladder_demo` for the old demo) pastes the old masks on the new image in 02:411, and later sections reuse `demo` via `k_demo`. Say "use `GALLERY_START` only".
- Add a visual: in Q5 (02:172), add the section-1 pet click mask as a first panel so "pets vs retina" is one picture.

**03 Hide the labels (20 min).**
- Strongest story section (the annotator analogy at 03:9-17). The Theory at 03:20-29 repeats it (half of it is the same sentence). Cut it to 2 lines.
- Q10 (03:125-139) has the rhetorical answer "no, you can't tell". Fold its bullet into the 3.1 text, or drop (b: ~2 min).
- 03:145-152 (recall/precision theory) is basic for this audience. Make it one line plus the examples.
- The ~90 lines of text before the break (03:446-535) are the longest un-interactive run in the notebook, at the worst moment for energy. In order they are: Theory 5 steps, details, Pitfall, 3.4 🔴, details, a 5-bullet Takeaway. Fixes (a):
  - Turn the 5-step Theory (03:447-457) into a tiny picture (two look-alike patches, one labelled 1 and one 0, so p ≈ 0.5) and keep the steps as its caption.
  - Move the Pitfall (03:473) and the Q12 details to go-deeper.
  - Cut the Takeaway (03:525-535) to 3 bullets and end on the Q13 cliffhanger.
- The "believer" fake model (03:264-274) is one print line, so the N-049 joke doesn't show. Add it as a first bar in the 03:365 chart, or add a panel showing the mask equals the file.

**04 Ladder (25 min, I would plan ~40).**
- The opening bullets (04:8-10) and the Theory (04:20-29) both list the three rungs. Delete the bullets.
- "What each rung receives" (04:178-209) is the best picture in the section but comes after the Theory and a 60-line helper. Put it first, before the Theory (N-010).
- The rung "prototype → SAM" (04:322-377) and Q15 are a fourth rung the SPEC does not name (SPEC's rungs: zero-shot, one-shot, scribbles). They exist because section 8 has 12 rows. For a live room that is 4 min of extra cognitive load. See removals.
- 4.4 (04:540-632) is a table, a 16-bar chart with the same numbers, error maps and a difference map. The bar chart repeats the table. Keep the table and the error maps, and move the difference maps (04:609-632) to go-deeper (or drop, ~1.5 min).
- The own-image prototype cell (04:752-803) is a 50-line code cell the learner never edits. Collapse it as `# @title` and put the callout in go-deeper. The same applies to the copies at 02:767 and 07:667.
- Pair moment missing: Q17 begs for "you take 20 px, I take 80 px, compare the 'vessel pixels called background' line". Add a 👥 there.

**05 Learn from less (20 min).**
- See point 3 in section 0 for the overlap with 04.3.
- Q21 (05:562) is 🟡 but needs the 🔴 PU head. It is also the most transferable discussion question. Move it to right after Q19 (05:390) and word it for the masked head only.
- The effort cell (05:548-559) prints one line and interrupts Q20 → Q21. Move it to the end.
- The legend for the 3 label maps comes after the figure (05:178-179). Put the colour key before it, or into the titles.
- The FACILITATION pair moment "pairs pick different LOOK_IMAGE" (FACILITATION row 5) is not in the notebook. Add it, or remove it from FACILITATION.

**06 Recover supervision (20 min, I would plan ~40).**
- At 990 lines it is the longest section for the second-smallest clock. It has two methods, two ✍️-like cells (pseudo targets and the Q23 tuning), a fake teacher, a Grad-CAM tutorial, and Q22-Q26.
- Grad-CAM is called "an awkward fit" and "a poor fit" by the notebook itself (06:548-557, 06:969). We spend ~10 min on it: tile classifier, 3-shape sketch (06:684-704), 6-panel step-by-step (06:706-749), pseudo-label cell, Q24, Q25. Fix (a): keep the pseudo-label cell (06:756-817) and Q24; move the step-by-step figure and the sketch into go-deeper (~4 min saved).
- Pitfall "over-trusting SAM" (06:465-476) plus the blind-spot sim (06:478-536) are good (N-049). Shorten the Pitfall to its 3 numbered steps and let the sim carry the rest.
- Q23 (06:379) is the best pair/slider moment in the notebook (see section 4). The 4-min pair is in FACILITATION only; add a 👥 callout at 06:446.
- The 6.2 cut/band picture (06:214-233) is a good visual. Keep.

**07 Path 3 (10 min, I would plan ~22).**
- "Over-trusting SAM's confident masks" is the title of three Pitfalls (02:451, 06:465, 07:559) and the idea of a fourth (01:495). You will say it four times. Keep 02 (first time) and 06 (with the sim). Turn 07:559-567 into one sentence inside the Takeaway.
- Q27 (07:233-257) re-teaches 02.3 ("more clicks make the mask bigger"). Keep as a slider (section 4) or drop.
- 7.4 SAM 3 🔴 (07:570-651) is 80 lines for most learners who will see "not in your cache". Collapse it, move the framing to go-deeper.
- The third "Try it on your image" (07:667-721) duplicates section 9. Move to go-deeper.

**08 Compare (15 min, I would plan ~25).**
- The printed table (08:251-261) has 10 columns in monospace (~110 chars) and wraps on a projector. This is the climax of the workshop and it is plain text. Fix (a): render it as an HTML or pandas Styler table with the approach-colour chips and the best value in each column in bold. The "how each one scales" list (08:268-270) is 12 sentences of text under it: move to go-deeper.
- Order: table → trade-off plot (08:336-379) → Q32 first; the violin (08:291-310) and the winner's worst/typical/best (08:318-326) can follow or sit in go-deeper. Today the central picture is the third one.
- The plot reading guide is in a collapsed go-deeper callout (08:381-389) although it is the point (N-048 inverted). Show 2 bullets, or label the corners on the plot ("dream", "cheap and weak", "hours away").
- Q31 asks learners to type short names (08:174-175). FACILITATION row 8 lists "typing the long name" as a known stumble. A dropdown fixes it (not one of my 3 proposals; a plain `@param` list would work).
- The 👥 (08:416-422) is weak: with the same seed and defaults, all tables are identical. Replace it with "A defends row X for situation (a), B defends row Y for (c), 1 min each, swap", using Q32.
- `my_spread` (08:201-210) is `np.std`, which this audience knows. Make it optional-collapsed or skip.

**09 Your images (20 min, I would plan ~28).**
- Strong interaction; the canvas will be the memorable part. Risk: the default example is two pets (09:117-131), so a learner who only runs cells sees "all three rungs agree", and 9.4's lesson (look where they disagree) is flat. Fix (a): make one bundled example a retina held-out image, which has real disagreement.
- Tell people to pick an image before the session (setup email) or to use the phone camera roll in 9.1, or the canvas time becomes image-hunting time for 40-200 people.
- Q33's cell description is 15 lines of text (09:430-444). Move the definitions of "thing ✓ / bg ✓" below the table.
- Q35 🔴 (09:609-625) asks them to re-run 3 cells; fine as an extension.

**10 Wrap-up (5 min).**
- After the 6 generated principles (good, from the learner's own numbers) there are 4 more summaries: Theory with 6 questions (10:183-194), outcomes (10:254-261), a 7-bullet Takeaway (10:264-278), and an Across callout (10:297-306), plus a 10.3 list. That is ~100 lines for 5 minutes at the end.
- Fix (a): merge the 6 questions of 10:187-194 into the Takeaway, shorten 10.3, and keep the outcomes (N-002) and the Across closing.
- The principles print as one paragraph block (10:180). Render each as a card: bold rule on top, evidence below in smaller text.
- Q36 is good: it ends on the learner's own problem.

## 2. Scrolling and visual balance
- Longest runs of text with no learner action: 00:4-54 (first screens), 03:446-535 (before the break), 06:538-575 (Grad-CAM intro, pitfall, theory, go-deeper in a row), 10:182-306 (end).
- Text with no picture: 03:445-478 (mechanism with no figure, N-042), 05:19-28 (masked loss formula), 08:268-270, 10:180.
- Picture with weak point: 01:421-441 (heatmap), 02:405-426 (difference map), 04:609-632 (difference maps), 06:706-749 (6-panel Grad-CAM), 08:291-310 (violin before the trade-off plot).
- The pattern prediction → "Question" callout → Qn → "this cell does…" → code is used ~15 times. Collapsing it to prediction+Q → code removes ~25 cells with no content loss.

## 3. Proposals for @param controls (for Simone; max 3)
1. **04:684 `BG_BEYOND_PX`** (Q17). `BG_BEYOND_PX = 40 #@param {type:"slider", min:0, max:160, step:10}`, with 0 meaning None (convert after validation). The learner sees the "% true vessels called background" and recall fall as the slider grows, and the orange panel (04:722) spreads. It makes Q17 a 20-second experiment and pairs well with a new 👥.
2. **06:400-401 `PSEUDO_THRESHOLD` and `UNSURE_BAND`** (Q23). Sliders 0.45-0.75 step 0.025 and 0-0.10 step 0.01. SAM's probabilities crowd between 0.50 and 0.68 (FACILITATION row 6), so the vessel→bg / bg→vessel columns flip from "everything" to "nothing" within a few notches. This is the clearest cliff in the notebook and gives the optional 👥 something concrete.
3. **07:418 `BOX_PAD_TRY`** (Q28). Slider 0-48 step 8 (`WITH_CLICK_TRY` as a checkbox). The learner sees recall rise and precision fall as the boxes grow, and the box recipe turn back into the section-2 big box at the top of the range.
- Keep all code cells visible (not form-only) so the `raise SystemExit` validation and the "Put Qn here" convention (N-007) still read.

## 4. Pair moments and questions in a live room
- Pair moments sit at min ~15 (01, dead as written), ~35 (02, good), ~150 (08, weak) and ~170 (09, good). Nothing between ~35 and ~150 (the whole of sections 3-7, plus the break), where the room is passive. Add three cheap ones (a): Q17 (distances), Q23 (thresholds), Q29 (prompt or train, argue opposite sides).
- Questions are mostly well placed after a figure. Misplaced or weak: Q10 (rhetorical), Q15 (details give the answer), Q21 (needs 🔴), Q27 (repeats 02). Q13 (coffee) and Q36 (own problem) are the best.
- Show-of-hands moments exist only in FACILITATION (rows 0-10), and only 10:246 is in the notebook. They are good; consider a one-line grey marker `🖐 show of hands` in the cell so the presenter does not forget.

## 5. (a) Non-removal summary (ordered by value)
Break placement (03:537 → 04:380); guard or stop skipping the 🔴 cells that later sections need; merge 5.1-5.2 into 4.3 or reframe 05:4; move the step-by-step head figure to 01:288; move the pets-vs-specialised chart to 02:586; HTML table and trade-off plot first in 08; render principles as cards; example in 09 = a retina image; merge the standalone "Question" callouts; collapse the three "Try it on your image" cells and 7.4.

## 6. (b) Removal ideas (Simone approves; rough minutes saved in the room)
| Idea | Where | Min |
|---|---|---|
| 2.4 retina PCA + Q7 | 02:459-490 | ~3 |
| Score heatmap in 1.4 (if the head figure moves) | 01:421-441 | ~1.5 |
| Q10 | 03:125-139 | ~2 |
| Difference maps (2.3 and 4.4) | 02:405-426, 04:609-632 | ~3 |
| Rung "prototype → SAM" + Q15 (and its row in 08) | 04:322-377 | ~4 |
| Grad-CAM step-by-step + 3-shape sketch + Q25 | 06:684-749, 06:838-898 | ~5 |
| SAM 3 🔴, Q27, third own-image cell | 07:233-257, 07:570-721 | ~5 |
| `my_spread` and `my_hidden_recall` (trivial one-liners) | 08:201-210, 03:155-190 | ~2 |
| Theory 10:183-194 (fold into Takeaway) | 10:183-194 | ~1 |
| Merge 5.1-5.2 into 4.3 | 05:132-261 | ~8-10 |
Total if all taken: ~35-40 min, which matches the gap between my estimate and the budget.
