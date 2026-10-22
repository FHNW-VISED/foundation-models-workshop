# Participant review: ML practitioner, not a CV specialist (all 11 sections read)

Short version: the story works (pets fine, retina falls short, then three ways out), the "When you see X -> do Y"
takeaways are good, and the invented-number questions (Q9, Q13, Q32) are the best moments. The problem is volume.
Doing every cell as written I estimate about 270 min against the SPEC's 175. Most of the extra is repetition of
ideas already landed, not new content. File:line below are `sections/NN_*.py`.

## Cross-cutting (read these first)

- **Colour semantics flip between sections.** 03:378-380 says orange = "hidden vessel it missed". 04:222 and 01:229
  say orange = false alarm, green = missed. I learned the 03 reading, then misread the 04 error maps. Pick one.
- **"Your prediction" and "Your turn (code)" share the orange box.** There are ~20 prediction boxes and ~12 code
  boxes, all #FFE8D6. At a glance I can't tell "write a guess" from "write code", and the orange box is the most
  frequent colour in the notebook, so it outshouts the takeaways (N-048).
- **The purple "Question" box carries only an instruction; the real question is plain text below it.** See 00:354,
  01:125, 02:146, 03:399, 06:375, 07:580, 09:426. Put Qn inside the box, or drop the box.
- **Three nearly identical "Pitfall: over-trusting SAM's confident masks" boxes** (02:451, 06:465, 07:559) plus a
  fourth in 01:495. By the third I skip them. Keep one; make the others a one-line reminder.
- **"Balanced" loss is used from 01 but only explained at 03:483.** The 01 helper description (01:296) says "balanced
  cross-entropy" with no meaning, and 03:195-199 introduces "plain" cross-entropy as a change. Result: the headline
  chart in 3.3 (section-2 head vs naive head) mixes two changes, and I only learn that at 3.4, which is red/optional.
  Fix: one line in 1.4 ("each class gets half the total weight"), and say in 3.3 which of the two causes the gap.
- **Terms never defined for a non-CV reader:** "fundus" (02:12, 02:16; only the figure title 02:105 hints at it),
  "zero-shot" (01 objective), "PU" (05:10; never expanded in main text, only in the 05:454 go-deeper), "teacher"
  (06:26), "prototype" is fine. One-line definitions at first use.
- **"1024 px crop" appears from nowhere** (02:498, 04:684, 07:353). Nobody told me the images are 1024 px crops of a
  3504 px fundus until the pitfall at 02:592. Say it once in 2.0 when the data is introduced.
- **CPU timing claims disagree for the same SAM call:** ~20 s (00:476), "up to half a minute" (06:92), "about a
  second per image" (07:202), "10-20 s" (09:438).
- **Coordinates for `MY_CLICKS` / `MY_FG_CLICKS` / `MY_BOXES`** (02:779, 04:766, 07:685): I have no way to read pixel
  (x, y) off an uploaded photo in Colab. Three mini "try on your image" cells are unusable without a tool, and
  section 9 does it properly with a canvas.

## Section by section

### 00 Setup (5 min budget)
- **Lost:** 00:53 "the MIT 'DINOv2 + scribbles' recipe, a tiny head on frozen patch features". Rewrite: "Scribble
  idea: train a tiny classifier on frozen DINO patch features, using only the pixels a person marked."
- **Bored:** four prompts on the same images in a row: Your prediction (349), Question (354), Q1 (358), plus the
  Objective (30) that repeats the banner. Q1's answer already uses "patch features (16 px squares)" (366) before
  patches exist.
- **Main vs secondary:** fine. The Takeaway is only troubleshooting, which is right for setup.
- Non-removal: merge 349-351 and Q1 into one prompt. Removal: the "Recipe and sources" note (50-54), 0.5 min.

### 01 Out of the box (15 min)
- **Good:** "three tools" intro, floor-click experiment, Across box. IoU defined with a worked example.
- **Lost:** "385 parameters" (283) is stated before `w.f + b` is explained at 288; swap the order. "balanced" (296),
  see above.
- **Bored:** the linear head is explained at 288-293, again in the 03 Theory (203-211) and a third time in the 03
  step-by-step cell. The per-patch score heatmap (421-441) adds little after the mask.
- **Order:** the 👥 callout (469) asks "did one click fail?" before the floor-click experiment (475), which is the
  better thing to compare.
- **Exercise `my_iou`:** clear, stub enough. The 4x4 play arrays live in a collapsed cell, so I can't see the example
  the callout mentions. Show the 4x4 arrays in the callout.
- Removal: score heatmap 1.5 min; 👥 optional 3 min (already optional).

### 02 Where it falls short (20 min budget, ~35 real)
- **Lost:**
  - 02:13 "to see which of the two limits the result" (which two?). Rewrite: "...and give DINOv3 twice the
    resolution, to see whether the limit is the prompt or the patch size."
  - 02:249 "with four fake predictions": the cell makes three fakes plus the truth panel.
  - 02:499-500 "roughly four to sixteen times the compute" vs the pitfall at 02:591 "4x the patches, 16x the cost".
    Pick one.
  - 02:592-593 "3504 px fundus: about 48 000 patches, 2.3 billion pairs": 3504x2336 is ~32 000 patches, ~1 billion
    pairs (the figure assumes a square). A sharp participant will check.
  - 02:191-193 the clDice sentence is a mouthful, and `tprec`/`tsens` are never spelled out. Rewrite: "tprec (how much
    of what you drew is real vessel) and tsens (how much of the real vessel you drew), combined by harmonic mean."
  - 02:152 "(a CPU session has 12)" -> "(on a CPU session there are 12 held-out images)".
  - 02:433-434 the worked "how to read a row" example sits in the middle of a list of questions; move it above.
  - 02:812 takeaway "Perfect masks cost 3-5 hours per fundus" is new in the takeaway; 02 never says it (03:9 does).
- **Q7 (478):** "name the colour of the optic disc / border / thick vessels" is not answerable in a useful way: PCA
  colours are arbitrary per image (the answer says so). Ask "do the thick vessels share a colour at all?".
- **Q9 (746):** "You now have a checklist" but the checklist is the Takeaway after it. Good question otherwise.
- **Predict-before-run overload:** 2.1, 2.3, 2.5 each have a prediction box, then a Q. That is 5 prompts in the
  section's first half. Merge each prediction into its Q.
- **Main vs secondary:** the SAM-ladder bar chart and the "pets vs specialised" chart carry the lesson; the difference
  map (405) and the median/best gallery (672) are secondary and not marked as such.
- **Exercise `my_cldice`:** task clear, formula given, stub sufficient. But three metric-writing exercises (iou,
  cldice, hidden_recall) in the first hour, with the real conceptual one (masked_loss) only in section 5.
- Removal: difference map 1.5; Q7 + retina PCA 3; contrast gallery 1.5; whole-fundus row 1; end "try your image"
  3; `my_cldice` as exercise -> reference only 3. About 12 min.

### 03 Hide the labels (20 min budget, ~28 real)
- **Good:** the annotator story, the "believer" fake model, Q13 as a coffee cliffhanger.
- **Lost:** 03:195-199 (see "balanced"). 03:533-535 "none is fixed by a bigger model" is asserted; no bigger model
  was ever tried. Rewrite: "none of these is a model-size problem we have tested; each needs different supervision."
- **Bored:** linear-head Theory (203-211) and step-by-step cell (217-257) repeat 01.
- **Main vs secondary:** the main number (recall on hidden vessels) is in a bar chart, good. The "worst/typical/best"
  strip is secondary but looks equal.
- **Exercise `my_hidden_recall`:** a one-liner (mean of the prediction over the mask). Instructions clear, but too small
  to justify an exercise box.
- Removal: head step-by-step + Theory 3; `my_hidden_recall` 1. Q12 (3.4) is the "balanced" explanation and should move
  earlier, not be cut.

### 04 The ladder (25 min budget, ~38 real)
- **Lost:** 04:384 "this time the file says so (value 0)". Value 0 meant "background" in 03 and means "unlabelled" in
  the stroke map here, while the targets use -1. Three encodings (stroke map 0/1/2, targets -1/0/1, section-3 file
  0/1) are never put side by side. One tiny table would fix it. Also "four rows" vs "three rungs" (rung 1 has two
  variants), say so.
- **Citations in the main Theory box** (20-29: PerSAM, Tang et al., "MIT DINOv2 + scribbles"). Belongs in go-deeper.
- **Bored:** four visual blocks to compare rungs (table, bar chart, error maps, difference maps). The difference
  maps (604-632) say "several look alike" and then show something almost as alike.
- **Main vs secondary:** 4.5 (the "far = background" rule) is a real lesson, but it is an extended exercise with its
  own Q17 and a second training run; it competes with the ladder lesson.
- **Exercise `my_scribble_targets`:** clear, hint included. The `bg_beyond_px` argument is not needed until 4.5, which
  makes the first attempt look harder than it is. Start with the 2-argument version.
- Removal: difference maps 2; prototype -> SAM clicks + Q15 4 (but section 8 uses the row; keep if so); 4.5 compressed
  to the Pitfall plus one fixed-distance run 4; end "try your image" 3.

### 05 Path 1 (20 min budget, ~30 real)
- **Lost:** 05:10 "PU" (expand). 05:263-266 "a stricter rule (the text it prints says how)": the checkpoint was made
  with the hidden truth (05:280-283), so `TRAIN_YOURSELF = False` silently changes the lesson. Say it in the
  markdown, not only in the printout. 5.1: where do the "3-5 background strokes" come from in the half-painted-file
  story? One sentence: "the annotator also drew a few strokes on clear background".
- **Bored:** "all-vessel trivial solution" is covered four times: Theory "Catch" (26), Q18 answer (193), Pitfall (407),
  fake-head cell (416). Keep Theory + fake head.
- **Main vs secondary:** the PU block (5.3) is 🔴 but is longer than the masked block. Its formula
  `L_b = (mean_all(-log(1-p)) - pi.mean_painted(-log(1-p)))/(1-pi)` (05:442) is where I lose the thread; an analogy
  ("count everything as background, then subtract the vessels you know hide in there") should come first.
- **Exercise `masked_loss`:** the best exercise in the notebook. Signature, formula, three numbered steps, 2x3 play
  example. Stub is enough to start.
- Removal: 5.3 PU to go-deeper / appendix 7 min.

### 06 Path 2 (20 min budget, ~35 real)
- **Lost:** 06:556 "Our bake-off judged it a poor fit": I don't know what a bake-off is. Then 8 minutes of
  Grad-CAM work in a section that has told me it is a poor fit. Motivation drops. 06:154-161 explains how to read
  the 6.1 printout after the cell; move it before. 06:952-954 Q26 ("a SAM checkpoint fine-tuned on retinas") asks me
  to compare options with no numbers on either, so I can only guess (N-041).
- **Contradiction:** Q23 (379-387) asks me to tune the threshold and compare held-out numbers; the details (459-460)
  then say "never on the test images". Make that the explicit point of Q23.
- **Bored:** Grad-CAM gets sketch of three shapes (689), 6-panel step-by-step (711), pseudo-label cell, Q24, Q25.
- **Good:** the colouring-in analogy (6-9), the "blind spot teacher" fake model (479-536).
- **Exercise `my_pseudo_targets`:** the rules are listed in order, stub is enough.
- Removal: Grad-CAM step-by-step + sketch 3, Q25 2; blind spot sim 2 (nice but optional). About 5-7 min.

### 07 Path 3 (10 min budget, ~22 real)
- **Lost:** 07:195-198 the prediction asks what happens when "we add twenty not-here clicks", but 6.1 just did
  exactly that. 7.1 is 6.1 minus the head, on held-out images. I answer from memory and the cell feels repeated.
  The fairness point (07:192, Path 3 gets strokes on every held-out image) is the real news; lead with it.
- **Exercise `my_stroke_box`:** a bounding box with padding. Clear, with a good "rows before columns" warning, but low
  value for an ML audience.
- **Main vs secondary:** SAM 3 (7.4) is a 100-line visible cell that, for most people, prints "not in your cache".
- Removal: 7.1 + Q27 folded into 7.3 4; `my_stroke_box` 2; SAM 3 to an appendix 3; end "try your image" 2.

### 08 Compare (15 min budget, ~22 real)
- **Lost:** 12 approaches with snake_case keys (`one_shot_sam`, `gradcam_pseudo`). The gallery (134-153) has 14 tiny
  panels titled by key. I can't hold twelve things in my head. Group by path colour, or name the 4-5 that matter.
- **Duplicate prompts:** "Your prediction (1 min)" (156) and Q31 (166) ask the same thing back to back. Q31 also
  needs me to type a key from a list that scrolled away.
- **Main vs secondary:** the trade-off plot (log x, error bars, 12 labels) is the section's one picture, and the text
  that lets me read it is in a collapsed go-deeper (382-389). If I read only boxes and figures, I get a plot I can't
  interpret. Put two lines of "how to read" visibly under the plot.
- 08:417-418 "(Q17, Q23, Q27, Q28, the 🔴 parts)": nobody remembers those numbers. Rewrite: "if you changed any setting
  earlier (threshold, number of clicks, box size)".
- **Exercise `my_spread`:** `np.std`. Not worth a box for this audience.
- Removal: `my_spread` 2; winner's worst/typical/best 1; one of the two table printouts (console table + "how each
  one scales" list) 1.

### 09 Your images (20 min budget, ~25 real)
- **Lost:** 09:282-283 "then run the cell after the next text" is cryptic. Rewrite: "then run the 'collect strokes'
  cell further down."
- **Big miss:** the default path (no upload) runs on two pets. On pets all three rungs agree, so the disagreement
  map shows only a thin rim and the section's lesson ("whole orange parts = where to stroke next") is never seen by
  anyone who skips the upload. Add one thin-structure example (a retina held-out image with generated strokes) as a
  second default, so the contrast is visible without effort.
- **Main vs secondary:** the privacy pitfall comes first and is worth it. The stroke-check columns `thing ✓` / `bg ✓`
  are explained inside the Q33 markdown cell; they deserve their own two-line legend.
- **Exercise `my_disagreement`:** clear, with a toy 3-mask table and a hint. Good.
- **Canvas task (9.2):** instructions are clear and the PNG fallback is well placed.
- Removal: none big. Q35 is already 🔴.

### 10 Principles (5 min budget)
- **Main vs secondary:** the six generated principles, the one thing I should take home, are printed as grey console
  text (`textwrap.fill`) inside a collapsed cell. They are the least visible thing in the section, while the coloured
  takeaway below them is a second, similar summary. Render the principles in a coloured box.
- **Repetition:** the banner outcomes, "10.2 What you can do now", the 6-question Theory (186-194), the 7-line
  takeaway and the closing Across box all restate the same lesson. Keep the principles and the takeaway.
- **Duplicate prompts:** "Your turn (2 min)" (197) and Q36 (207) both ask problem + first step.
- **Cryptic:** 10:86-87 "...scored IoU 0.xx at 512 px and 0.xx at 1024 px (clDice a -> b), against 0.xx on pets"
  ("against" what, a different head?). Rewrite: "...; on pets the same kind of head scored 0.xx."
- Removal: 10.2 and the six-question Theory 2 min; merge Your turn into Q36.

## Where the notebook is strong
Q9, Q13 and Q32 give concrete invented numbers and a decision, which is exactly right. The fake models (trunk-only,
believer, blind-spot teacher, always-vessel head, thickened strokes) make abstract pitfalls tangible. The
`masked_loss` exercise and `use()` fallback are well designed. The takeaways are standalone.

## Suggested cuts (rough participant minutes; only Simone approves, N-047)
- 02: ~12, 03: ~4, 04: ~13, 05: ~8, 06: ~7, 07: ~11, 08: ~4, 10: ~4, 01: ~2. Total about 65 min, which would bring me to
  ~205 min. Reaching 175 needs one of: appendix the PU block and SAM 3 (already counted), plus dropping the three
  trivial exercises (iou stays), plus merging the 6.1 / 7.1 duplicates (counted).

## Non-removal fixes (cheap, high value)
1. Unify colour meaning of orange/green; separate prediction vs code box colours.
2. Move the Q text into its box; delete the empty "Run the next cell" boxes.
3. One line each: fundus, zero-shot, PU, balanced loss, 1024 px crops, the three label encodings.
4. Visible reading guide under the section-8 trade-off plot; coloured box for the section-10 principles.
5. Add a thin-structure default example to section 9.
6. Fix the factual slips: 02:249 "four fakes", 02:499 vs 02:591 compute, 02:592 patch count, CPU timings.
