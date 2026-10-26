# Round-2 audit: alignment, cuts, plots, exercises (2026-10-26)

Reviewer report, read-only. Design source: the accepted AMLD 2027 proposal and its flow chart. Line numbers for
`04`–`10` are exact; `00`–`03` were being edited while this was written, so their numbers are a snapshot and the
cell title is the anchor. Items already being fixed by other agents (playground truth toggle, SAM input on the pets
figure, clDice explainer, "marked" definition, gallery bug, section-3 figures and legends, believer removal) are not
proposed again here.

**In short.** The notebook runs about **310 participant-minutes** with every optional part, about 265 on the main
path alone, against a budget of 175. Most of the excess is the same picture drawn again (worst / typical / best
panels in seven places, a bar chart that repeats the printed table in four sections), vessel-specific depth the
proposal never promises, and five ✍️ stubs where two would do. The cuts below bring it to about **185**, with PU on
the main path and the new interactive parts included; the last ~10 minutes need one decision from Simone (section 6).

How the minutes were estimated, one line: words of markdown ÷ 200 per minute, plus 1 min per figure cell, 0.3 min
per other code cell, 2 min per question, plus the stated minutes of each ✍️, 👥 and playground. It is crude (±15 %),
but it is the same ruler before and after.

---

## 1. Alignment

### Proposal and flow chart → notebook

| Promise / step | Where | Status |
|---|---|---|
| Start with a VFM on a simple task, see what it already does | 1.1–1.4 | Main. **Over-weighted**: SAM on pets is shown four times (Q2 figure, Q3 playground, worst/median/best, click on the floor) |
| Move to a specialised task, see where it falls short | 2.1–2.7 | Main. **Over-weighted**: clDice in every title, resolution (2 go-deeper boxes + a zoom figure), retina PCA, whole fundus, two galleries |
| Progressively add task information, watch predictions change | 4 (zero-shot → one-shot → scribbles), sliders in 3, 6, 7 | Main. Over-weighted by three 🔴 side trips (prototype → SAM, difference maps, distance rule) |
| Annotation is expensive and incomplete; vary amount and type of supervision | 3 (`HIDE_PERCENT`), 4 (type), 8 (effort column) | Main, well delivered |
| Complete ground truth → hide annotations → incomplete supervision | 3.1 | Main |
| Naive baseline (missing = background) | 3.3 | Main |
| Observe the failure | 3.3 chart, Q12, Q13 | Main, slightly over (three figures plus the 3.4 detour) |
| (1) Learn from less: missing = unknown | 5.2 masked loss | Main |
| (1) … uncertainty-aware | only the unsure band in 6.2 | Partial: say once in 5 that "unknown" is the simplest uncertainty-aware label; no new code |
| (1) … **PU learning** | 5.3 | **Optional (🔴): must move to the main path** (section 5 of this report) |
| (2) Recover supervision: explanations, e.g. Grad-CAM | 6.3 | Main. **Over-weighted**: formula box, step-by-step figure, three-shapes sketch, Grad-CAM → SAM (Q27) |
| (2) … foundation-model pseudo-labels | 6.1–6.2 | Main. Over: ✍️ `my_pseudo_targets`, blind-spot simulation, Q28 |
| (3) Skip training, use the FM directly | 7.1–7.3 | Main. SAM 3 text sits in 🔴 7.4: **move the retina text prompt into the main path when SAM 3 is cached** (Simone asked for it; it is the purest "out of the box" test) |
| Evaluate on the same hidden ground truth | every section, 8 | Main |
| Compare trade-offs: accuracy, effort, reliability, speed, compute, scalability | 8 table, trade-off plot, violin, scalability list | Main, complete |
| What can AI solve out of the box, when is ML engineering needed | 10 principles + story picture | Main |
| A reusable experimental framework | "six questions for any task" in a collapsed 🔍 box (10) | **Hidden**: promote the six questions to the main path as the framework card |
| Bring your own images, scribble, compare with FM predictions | 9 | Main. **Over-weighted**: ✍️ disagreement, Q37, stroke-check box |
| Guided coding; optional challenges modify losses | 5 ✍️ stubs | Too many and too hard (section 4). The proposal's "modify losses" fits a 🔴 one-line PU fill exactly |

### Notebook parts that serve no promise (cut candidates)

clDice beyond one explainer (the metric printed in every title of 2–7) · the attention arithmetic and "why 512 not
518" boxes · the retina PCA (2.4) · the whole fundus (2.8) · the balanced-loss detour (3.4) · prototype → SAM clicks
(4) · difference maps (4) · the distance rule (4.5–4.6) · the blind-spot simulation (6) · Grad-CAM step by step and
the sketch (6) · Grad-CAM as a SAM prompt (Q27) · the invented-pilot question (Q28) · `my_scribble_targets`,
`my_pseudo_targets`, `my_disagreement` · Q37.

---

## 2. Cut list

E = essential to reach the budget; N = nice-to-have. Minutes = estimated participant-minutes saved.

| id | file:lines | what | why | action | min | E/N |
|---|---|---|---|---|---|---|
| C1 | 01:222–247 (Q2 cell) + helper `show_sam_demo` | Q2 static click + box figure | repeat of the Q3 playground, which does click and box live | delete; move Q2's two questions into Q3; keep `DEMO_INDEX` out | 4 | E |
| C2 | 01:662–694 | 👥 "pick a DEMO_INDEX", floor-click cell, its pitfall | Q3 already asks to click the floor | delete; one pitfall sentence under Q3 | 5 | E |
| C3 | 01:288–335 | ✍️ `my_iou` | not the workshop's topic | 1-line fill, or drop (Simone, see §4) | 1 | N |
| C4 | 01:606–635 | head worst/median/best + 3 truth panels | same layout as the SAM worst/median/best | merge: one figure, row 1 SAM box, row 2 head, same three pets | 2 | N |
| C5 | 02:208–276 (helper + Q6) | Q6 static click + box on retina | repeat of the Q7 playground and of 2.3's 1-point and box | delete; set `demo = spec_eval[0]` in the load cell (sections 4–8 use `demo`) | 4 | E |
| C6 | 02:616–653 | 2.4 🔴 retina PCA + Q9 | repeat of 1.3 | delete | 4 | E |
| C7 | 02:666–669 | go-deeper "why 512 and not 518" | trivia | delete | 0.5 | N |
| C8 | 02:813–824 | go-deeper attention arithmetic | over-weighted resolution depth | one clause in the 2.5 intro ("4× the patches, ~16× the cost") | 1.5 | E |
| C9 | 02:729–777 (zoom part) | two-resolution zoom figure | the numbers and the 2.0 grid figure say it | keep the printed lines, drop the figure | 1.5 | N |
| C10 | 02:900–908 | "For contrast: typical and best" | the `GALLERY_START` slider pages the same ranking | delete | 2 | E |
| C11 | 02:910–926 | 👥 "whose image fails worst" + details | same task as Q10 | merge: Q10 becomes the pair moment | 3 | E |
| C12 | 02:953–989 | 2.8 🔴 whole fundus | repeat gallery; point made by resolution | delete | 2.5 | E |
| C13 | 03:639–690 + 05:299–305, 05:37–46 | 3.4 🔴 balanced loss, Q14, its row in 5 | not in the description; PU makes the same "wrong proportion" point | delete; one sentence in the PU text. ⚠️ another agent may be polishing 3.4's one-liner right now | 5 | E |
| C14 | 03:476–530 | `show_naive_on_image` + Q12 + details | held-out failure is already in the 3.3 chart; same figure style as the 2.6 gallery | delete the cell; move the details' two lessons (stingy vs generous, close IoU hides it) under the 3.3 chart | 4 | E |
| C15 | 03:455–470 | worst / typical / best training image | three panels where one shows it | keep the typical image only (after the section-3 agent's fix) | 1.5 | N |
| C16 | 03:532–538 | go-deeper "a precise model looks trustworthy" | already in the takeaway | delete | 0.5 | N |
| C17 | 04:212–235 (+ helper `show_extremes` 194–209) | rung-0 worst/typical/best error maps + metric list | repeat of the 2.6 gallery's SAM column; Simone: "what do the colours even mean?" | keep the one printed line, move the metric list to 4.4 | 2.5 | E |
| C18 | 04:338–398; rows at 04:87,93,99,623,629; 08:52,79–80,109–110; 10:25 | 🔴 prototype → SAM clicks + Q16 | Path 3 and Q27 do "map → SAM clicks" again | delete, with its row in 4.4, 8 and 10 | 5 | E |
| C19 | 04:482–547 | ✍️ `my_scribble_targets` + reference + play | hard (see §4); its lesson is the masked loss's | replace by a dropdown `UNLABELLED_TARGET` (−1 ignore / 0 background) on the training cell; reference collapsed | 3 | E |
| C20 | 04:707–737 | 🔴 difference maps | repeat of the 4.4 error maps | delete | 2.5 | E |
| C21 | 04:739–848 | 4.5 pitfall, 4.6 🔴 distance rule, Q19 | section 3's lesson again ("through the side door") | delete; one takeaway bullet; drop `bg_beyond_px` from the reference | 8 | E |
| C22 | 04:654–669 | 4.4 bar chart | repeats the printed table; section 8 has the final charts | delete, keep table + error maps | 1 | N |
| C23 | 04:430–437 | three-encodings table | only needed by the ✍️ | one line ("−1 = ignored") | 1 | N |
| C24 | 05:350–427 | `compare_heads`, Q20, your-turn, details | its two figures are Q23's rows again | merge into one 2 × 3 figure after PU (see §5); Q20's questions join Q23 | 5 | E |
| C25 | 05:470–487 | "What the PU loss sees" | same picture as panel 4 of 5.1 | delete | 1.5 | E |
| C26 | 05:348 | first `path_chart` call (masked only) | drawn again after PU | one call, after PU, naive / masked / PU (+ full masks as reference) | 1 | E |
| C27 | 05:429–445 | fake "vessel everywhere" head | fine lesson, own cell | print its one number inside the pitfall cell | 1 | N |
| C28 | 05:667–685 | effort cell | bookkeeping with its own markdown | fold into the scoring cell | 0.5 | N |
| C29 | 06:229–296 | ✍️ `my_pseudo_targets` + reference + play | hard (see §4); the Q25 sliders teach the same | delete stub; reference collapsed; sliders are the exercise | 4 | E |
| C30 | 06:493–551 | blind-spot simulation | cool, but 6.1 already shows a head copying SAM's region | delete (Simone's rule: cool ones may stay; I would cut) | 4 | E |
| C31 | 06:779–843 | Grad-CAM step by step + three-shapes sketch | Grad-CAM depth beyond the description | delete | 3 | E |
| C32 | 06:582–591 | Grad-CAM formula box | same | one line + citation | 0.5 | N |
| C33 | 06:845–913 | Q27 🔴 Grad-CAM as a SAM prompt | prompt route again (2.3, 7) | delete | 4 | E |
| C34 | 06:960–983 | Q28 invented pilot | heavy reading, invented numbers | move to FACILITATION as a discussion prompt | 3 | N |
| C35 | 06:933–947 | 6.4 bar chart | repeats the table | delete | 1 | N |
| C36 | 07:197–226 + `try_clicks` 138–150 | Q29 🔴 click-count slider | section 2.3 showed "more clicks, bigger region" | delete | 4 | E |
| C37 | 07:303–322 | box recipe worst / typical / best | the pattern again | delete | 1.5 | E |
| C38 | 07:406–426 | 7.3 bar chart | repeats the table | delete | 1 | N |
| C39 | 07:440–529 | 7.4 SAM 3 batch cell + Q31 + box | duplicates the Q32 playground | keep one SAM 3 playground on the retina (main path when cached); the batch score becomes an optional row in 8 | 4 | E |
| C40 | 08:396–401 | go-deeper "more on reading the plot" | merge into the note above it | merge | 0.5 | N |
| C41 | 09:593–643 | disagreement theory + ✍️ `my_disagreement` | section 9's minutes belong to their own images | keep the map (reference code), 2-line theory, no stub | 3 | E |
| C42 | 09:684–701 | Q37 🔴 redraw and rerun | one sentence does it | one sentence under Q36 | 3 | E |
| C43 | 09:575–585 | go-deeper stroke checks | long | 2 bullets | 0.5 | N |
| C44 | all `<details>` "what to look for" (~30) | answer boxes of 100–150 words | reading load | ≤ 3 bullets, ≤ 60 words each | 8 | E |
| C45 | all "This helper cell defines…" lines (~28) | repeat the cell title and docstring | reading load | ≤ 15 words each (N-031 needs one sentence only before learner-run cells) | 2 | E |
| C46 | 02:91–97, 04:172–209, 05:101–108, 06:370–379 | five-number reports (IoU · clDice · recall · precision · marked) in every title and printout | vessel-metric depth everywhere | IoU · recall · marked in 3–7; clDice only in 2.2 and the section-8 table | 3 | E |
| C47 | 06:561–591 | Grad-CAM pitfall + theory + go-deeper | three boxes for one idea | one theory box, ≤ 80 words | 1.5 | E |
| C48 | 02:595–606, 06:179–191, 07:20–27, 07:212–226 | "SAM looks for one object containing all clicks", said four times | repeat | say it once in 2.3, one back-reference later | 2 | E |
| C49 | 04:59–69 + 04:238–246 | rung theory, then the same steps numbered | repeat | merge | 1 | N |
| C50 | 03:8–25, 03:217–231, 03:444–452 | three theory boxes on the naive failure | repeat | one numbered explanation (after the section-3 agent finishes) | 1.5 | N |

**Totals.** Essential rows save ~110 min, nice-to-have rows ~20 more.

| | minutes |
|---|---|
| Now, every part including 🔴 | ~310 |
| Now, main path only | ~265 |
| New parts added this round (truth toggle, clDice explainer, SAM 3 on retina; PU was already counted) | +4 |
| After the essential cuts | ~205 |
| After all cuts | **~185** |

The last ~10 minutes need one of the decisions in section 6.

---

## 3. Plot inventory

Legend = a key for every colour inside the figure. Desc / look = a sentence before the cell / a "what to look for"
line. "In fix" = being handled this round by another agent. For 00–03 the legend column is the snapshot state:
other agents are adding `legend=` right now, and the shared helper they built (`show_masks(..., legend=[...])` in 01,
`show_keyed` in 03) should be reused for every "fix legend" in 04–09, which makes those fixes one argument each.

| # | file:line | cell title | shows (≤ 8 words) | legend | desc / look | duplicate of | verdict |
|---|---|---|---|---|---|---|---|
| F1 | 00:343 | Load both datasets… | two pets, two retinas, truth in green | n | y / y (Q1) | — | fix legend |
| F2 | 00:548 | Interactive SAM (headless fallback) | fixed click and box, two masks | added | n / n | Q2, Q6 figures | keep (smoke only) |
| F3 | 01:244 | Q2 — SAM one click, one box | click mask, box mask, truth | added | y / y | F4, F2 | **cut** (C1) |
| F4 | 01:260 | Q3 — drive SAM yourself | live SAM playground | in fix | y / y | — | keep |
| F5 | 01:366 | Worst, median and best pet by IoU | SAM box error maps, three pets | added | y / y | F8 | keep; merge F8 in (C4) |
| F6 | 01:389 | Show the patch features as colours | DINOv3 PCA as RGB | n | y / y (Q4) | F16 | fix legend (figure text "colours = main feature directions, not classes") |
| F7 | 01:554 | The linear head step by step | patches → p → upsampled → mask | partial (no colourbar, teal box unexplained) | y / n | — | fix legend |
| F8 | 01:606 | Score the head on the held-out pets | head masks + truth, three pets | added | y / y | F5 | merge into F5 |
| F9 | 01:674 | Click on the floor, not the pet | SAM mask from a floor click | added | y / y | F4 (Q3 asks it) | **cut** (C2) |
| F10 | 02:165 | Meet the retina images | photo, vessels, zoom on patch grid | partial (titles) | y / y | F17 | keep |
| F11 | 02:273 | Q6 — click and box on retina | SAM click mask, box mask, truth | added | y / y | F12, F15 | **cut** (C5) |
| F12 | 02:289 | Q7 — make SAM mark only vessels | live playground, retina | in fix | y / y | — | keep |
| F13 | 02:475 | See clDice on a made-up tree | truth, prediction, tprec, tsens, bars | partial | in fix | — | keep (explainer in fix) |
| F14 | 02:513 | SAM prompt ladder (bar chart) | IoU per prompt vs "everything" | y | y / y (Q8) | printed table | keep |
| F15 | 02:513 | SAM prompt ladder (four prompts) | four prompts and SAM's masks | added | y / y | — | keep |
| F16 | 02:623 | DINOv3 PCA on three retinas | PCA as RGB, retina | n | y / y (Q9) | F6 | **cut** (C6) |
| F17 | 02:729 | Score both heads, zoom | head @512 vs @1024, centre zoom | added | y / n | F10 | **cut** the figure (C9) |
| F18 | 02:780 | Pets vs vessels | same tools, IoU on both tasks | y | y / y | — | keep |
| F19 | 02:895 | Q10 — failure gallery | photo, SAM box, head, truth × 3 | added | y / y | — | keep (bug in fix) |
| F20 | 02:904 | For contrast: typical and best | same rows, median and best | added | y / n | F19 | **cut** (C10) |
| F21 | 02:960 | Failure gallery on the whole fundus | gallery row on the whole fundus | added | y / n | F19 | **cut** (C12) |
| F22 | 03:123 | Hide half of the vessel segments | photo, truth, painted, hidden | in fix | y / in fix | — | keep (in fix) |
| F23 | 03:~200 | Worked example: believer | the file as a prediction | — | — | — | being removed |
| F24 | 03:233 | Two look-alike patches | two crops, two targets | in fix | y / in fix | — | in fix |
| F25 | 03:380 | Score both heads… (bars) | share of vessels found, per head | y | y / in fix | — | keep (in fix) |
| F26 | 03:459 | Worst, typical, best training image | found / missed hidden vessels | in fix | y / in fix | F28 | keep one panel (C15) |
| F27 | 03:514 | Q12 — naive head on held-out image | both heads' masks + truth | partial | y / y | F19 style | **cut** (C14) |
| F28 | 03:634 | Q13 — hide more or fewer | painted, hidden, found / missed | in fix | in fix | F26 | keep (in fix) |
| F29 | 04:18 | What each rung receives | box, one full mask, strokes | n | y / n | — | fix legend, add look-for |
| F30 | 04:226 | Score rung 0… | SAM box error maps × 3 | n | y / n | F19 SAM column | **cut** (C17) |
| F31 | 04:321 | Q15 — one-shot | annotated image, similarity, mask, truth | n (no colourbar) | y / y | — | fix legend (purple = like vessel, orange = like background) |
| F32 | 04:350 | Prototype → SAM clicks | clicks, SAM mask, truth | n | y / y | Path 3 | **cut** (C18) |
| F33 | 04:444 | Show the scribbles… | photo, strokes, what loss sees, truth | partial | y / n | — | fix legend, add look-for |
| F34 | 04:619 | Table, bar chart… (bars) | four metrics per rung | y | y / y | table, 8 | **cut** (C22) |
| F35 | 04:619 | Table, bar chart… (error maps) | each rung's error map, demo image | n | y / y (Q18) | — | fix legend |
| F36 | 04:714 | Difference maps | what changes between rungs | y | y / n | F35 | **cut** (C20) |
| F37 | 04:831 | Q19 — distance rule | rule's targets, vessels called bg | partial | y / y | — | **cut** (C21) |
| F38 | 05:149 | What each recipe tells the loss | naive / masked / PU targets | n (key in markdown only) | y / n | F42, F44 top row | fix legend |
| F39 | 05:295 | Score the masked head… (chart) | hidden recall, recall, precision | y | y / n | F43 | merge into F43 (C26) |
| F40 | 05:404 | Q20 — hidden found | found / missed per head, training image | n | y / y | F44 | merge into F44 (C24) |
| F41 | 05:404 | Q20 — error maps | two heads, demo image | n | y / y | F44 bottom row | merge into F44 (C24) |
| F42 | 05:475 | What the PU loss sees | painted vs unlabelled | partial | y / n | F38 panel 4 | **cut** (C25) |
| F43 | 05:562 | Q21 — PU (chart) | naive / masked / PU recalls, precision | y | y / y | — | keep (the section's chart) |
| F44 | 05:658 | Q23 — three recipes | what loss heard, mistake made | partial (suptitle only) | y / y | F38, F40, F41 | keep as the merged 2 × 3; fix legend |
| F45 | 06:168 | Run SAM on the stroke clicks | clicks, SAM probability, pseudo-label errors | partial (no colourbar) | y / y (Q24) | — | fix legend |
| F46 | 06:204 | The cut and the unsure band | probability axis: 0 / −1 / 1 | y (self-labelled) | y / n.a. | — | keep |
| F47 | 06:499 | Playful pitfall: blind spot | blob, clean head, biased head | n | y / y | — | **cut** (C30) |
| F48 | 06:698 | Grad-CAM pseudo-labels | heat map, two pseudo-labels vs truth | partial | y / y (Q26) | — | fix legend |
| F49 | 06:786 | Grad-CAM step by step | A_k, α, weighted sum, heat map | partial | y / n | — | **cut** (C31) |
| F50 | 06:828 | Sketch: three shapes | tree, blobs, noise | n | n / n | — | **cut** (C31) |
| F51 | 06:898 | Q27 — Grad-CAM as SAM prompt | clicks, SAM mask vs truth | n | y / y | F45 | **cut** (C33) |
| F52 | 06:923 | Path 2 side by side (bars) | four metrics, three heads | y | y / n | table, 8 | **cut** (C35) |
| F53 | 06:923 | Path 2 side by side (error maps) | three heads, demo image | n | y / n | — | fix legend, add look-for |
| F54 | 07:274 | One box and click per stroke | stroke boxes, SAM, thickened strokes | n | y / y (Q30) | — | fix legend |
| F55 | 07:308 | Box recipe: worst, typical, best | box recipe error maps × 3 | n | y / n | F54 | **cut** (C37) |
| F56 | 07:362 | Path 3 side by side (bars) | four metrics, five approaches | y | y / n | table, 8 | **cut** (C38) |
| F57 | 07:362 | Path 3 side by side (error maps) | five approaches, demo image | n | y / n | — | fix legend, add look-for |
| F58 | 07:465 | Q31 — SAM 3 from text | SAM 3 vs stroke boxes, errors | n | y / y | F59 | merge (C39) |
| F59 | 07:539 | Q32 — Ask SAM 3 or click | live playground, retina + pets | n.a. | y / y | — | keep, main path when SAM 3 is cached |
| F60 | 08:159 | All approaches on one image | every approach's mask, truth | partial (title colours) | y / y (Q33) | — | **keep** (final); one figure line "mask colour = approach" |
| F61 | 08:345 | Trade-off plot | IoU vs effort; compute bars | y (direct labels) | y / y (note) | — | **keep** (final) |
| F62 | 08:453 | Violin plot | per-image IoU per approach | partial | y / n | — | **keep** (final); add look-for |
| F63 | 08:479 | Worst, typical, best of the winner | winner's error maps × 3 | n | y / n | — | **keep** (final); fix legend |
| F64 | 09:348 | Choose your images | the chosen photos | n.a. | y / n.a. | — | keep |
| F65 | 09:375 | Draw your strokes | drawing canvases | n.a. | y / y | — | keep |
| F66 | 09:396 | Collect your strokes | photo + strokes | n | y / n | — | fix legend |
| F67 | 09:407 | Prompt SAM on your images | live playground | n.a. | y / n | — | keep |
| F68 | 09:568 | Q35 — the ladder on your images | strokes, three masks, truth | n | y / y | — | fix legend |
| F69 | 09:650 | Disagreement map | strokes, disagreement in orange | n | y / y (Q36) | — | fix legend |
| F70 | 10:21 | The story in one picture | IoU per stage of the story | y | y / n | — | keep; add look-for |

Count: 70 figure cells now; 23 cut or merged; about 47 remain, of which ~20 still need a legend.

---

## 4. ✍️ exercises

| stub (`use()` target) | learner writes | why it is hard for this audience | simpler remake | keep? |
|---|---|---|---|---|
| `my_iou` (`iou`), 01:322 | intersection ÷ union of two boolean masks, empty case | not hard (2 lines), but IoU is not the workshop's topic | 1-line fill: `inter = (pred & truth).sum()`, union and return given | Simone decides: it is the only code moment in block 1 |
| `my_scribble_targets` (`scribble_targets`), 04:531 | stroke map → targets 1 / 0 / −1, region last, plus a distance mode with `distance_transform_edt` | three label encodings (the table at 04:430), assignment order matters, boolean indexing, a second mode tied to cut 4.6; ~6 lines | a dropdown `UNLABELLED_TARGET = -1  # @param [-1, 0]` on the scribble-head cell: predict, run; 0 brings section 3's failure back | **drop** the stub |
| `masked_loss` (`masked_loss`), 05:231 | per-pixel BCE, separate means over known vessel and known background, average | the balancing (two means, empty-class clamp) is the hard part and is not the lesson; any-shape tensors; ~4 lines | `bce` line and the balanced `return` given; the learner writes one line, `known = ~ignore_mask` (🟡); writing the balanced return is a 🔴 extra | **keep** (core) |
| PU loss (no stub today), 05:503 | nothing (read-only, 🔴) | — | show `pu_loss` as a short visible cell with its three numbered steps; 🔴 fill one line: `bg_loss = (all_as_bg - prior * painted_as_bg) / (1 - prior)` | **keep** as the optional challenge ("modify losses", in the proposal) |
| `my_pseudo_targets` (`pseudo_targets`), 06:281 | threshold ± band → 1 / 0 / −1, strokes overrule, region last | ordering rules, int8, three conditions; ~6 lines; the sliders below teach the same thing | none: `PSEUDO_THRESHOLD` and `UNSURE_BAND` are already "change one value and predict" | **drop** the stub |
| `my_disagreement` (`disagreement`), 09:628 | marked by some masks, not by all | `np.stack` + reduce along an axis; easy, but it takes time from their own images | none needed: keep the map as reference code | **drop** the stub |

After this, block 2 has one core ✍️ (masked loss, one line) and one 🔴 challenge (PU, one line); block 1 has
`my_iou` only if Simone keeps it. That still meets "freedom medium" with predictions, sliders and the canvas as the
other hands-on parts.

---

## 5. PU on the main path

**What changes** (all in `05` unless noted):

- 5.3 heading "🔴 Optional: a PU-inspired loss" (05:448) → "5.3 A PU loss"; intro bullet "🔴 PU" (05:11) and the
  objective "then 🔴 a PU-inspired loss" (05:17) lose the 🔴.
- Delete "What the PU loss sees" (05:470–487, C25).
- Split the helper cell (05:502): `pu_loss` becomes a short visible cell with steps (1)–(3) as comments;
  `train_pu_head` stays collapsed.
- Your-turn (05:553–560) "optional, 🔴" → 🟡 "change one value"; the one-line fill becomes the 🔴 part.
- **Q21** 🔴 → 🟡 (05:568); its details (05:573–589) shortened to three bullets, the 3.4 comparison sentence goes
  (C13). **Q22** loses "(🔴, if you ran 5.3)" (05:605).
- Dead branches go: "Run 5.3 first" (05:624, 05:648–649), the `LABELLED_SHARE in globals()` fallback (05:673–674),
  `skipped["pu"]` (08:81–82), "The 🔴 parts you skipped: PU loss (5)" (10:293).
- Outside the sections (integrator): SPEC §6 row 5 level 🟡🔴 → 🟡 and its contents line; §6b "Extensions: PU
  loss" removed; FACILITATION row 5; optionally a content test that `RESULTS["pu_head"]` exists.
- Minutes: ~7 on the main path (read 2, run the slider twice 2, Q21 2, figure 1), plus 3 for the 🔴 fill.

**How to present it simply.**

One paragraph: *"The masked loss threw away every unpainted pixel. The PU loss keeps them, as 'unlabelled': a mix
of background and the vessels nobody painted. The annotator tells us roughly what share they painted, here about
half. Say the painted vessels cover 1.5 % of the fundus: then about 3 % is vessel, and another 1.5 % hides among
the unlabelled pixels. The loss counts every unpainted pixel as background, then subtracts that 1.5 %,
so it stops punishing the hidden vessels."* (The cell prints the real painted share and prior.)

One figure: the merged 2 × 3 after PU (C24): columns naive, masked, PU; row 1 the hidden vessels each head found on a
training image (violet found, orange missed, teal painted); row 2 its error map on the demo image (violet found,
orange invented, green missed); a legend under each row.

One knob: `LABELLED_SHARE` (0.3 / 0.5 / 0.8). Predict first: "If the annotator says 'I did a third', will the head
mark more or less?" The printout shows the prior next to the result.

---

## 6. Section plan after the changes

"Est." = my estimate after all cuts; "Budget" = what I recommend the SPEC table says. The ~8-minute gap closes with one
of the decisions below.

| # | Title | Objective (one line) | Est. | Budget |
|---|---|---|---|---|
| 0 | Setup & the question | Load the cache; read the question and the outcomes | 8 | 7 |
| 1 | Out of the box: pets | SAM (playground, truth toggle), DINOv3 PCA, a head on five pets | 20 | 18 |
| 2 | Where it falls short | Same tools on retina; escalating prompts; clDice once; resolution in two numbers; gallery; checklist | 26 | 24 |
| 3 | Hide the labels: the naive baseline | Hide half the vessels, train "missing = background", see what it misses; `HIDE_PERCENT` slider | 10 | 12 |
| 4 | The adaptation ladder | Zero-shot → one-shot → scribbles; `UNLABELLED_TARGET` choice | 11 | 12 |
| 5 | Path 1 · Learn from less | Masked loss (one-line ✍️), then the PU loss with `LABELLED_SHARE`; one 2 × 3 picture | 21 | 22 |
| 6 | Path 2 · Recover supervision | SAM pseudo-labels with threshold and band sliders; Grad-CAM, lean | 20 | 18 |
| 7 | Path 3 · Skip specialist training | Stroke clicks and stroke boxes vs thickened strokes; SAM 3 text on retina if cached | 14 | 12 |
| 8 | Compare everything | Predict, then the table, trade-off plot, violin, what happened | 18 | 16 |
| 9 | Your own images | Upload, scribble, playground, three rungs, disagreement map | 26 | 25 |
| 10 | Principles & the framework | Story picture, six principles, the six questions as the reusable framework | 8 | 9 |
| | **Total** | | **182** | **175** |

**Decisions for Simone**

1. **The break.** With these cuts, block 1 ends at ~60 min if the break stays before 4.3, and block 2 runs ~115. A
   balanced split puts the break **before 5.3 (PU)**: block 1 ≈ 85 (pets → failure → naive → ladder → masked loss),
   block 2 ≈ 90. Q17 would then become: "the masked head over-marks; the annotator says they did about half; how could
   that number help?", which leads straight into PU. This reverses today's S0 decision.
2. **The last ~8 minutes**, one of: (a) the one-shot prototype rung (4.2) moves to a go-deeper box, so the ladder is
   zero-shot → scribbles (~6 min); (b) Q11's cracks checklist becomes a facilitator discussion (~3); (c) section 9
   drops its zero-shot box rung, since the playground already gives SAM a box (~2).
3. **`my_iou`**: keep it as a 1-line warm-up, so block 1 has one code moment, or drop it.
4. **Cool but cut** (your "if it is cool, keep it" rule): the blind-spot simulation (C30) and the clDice in every
   title (C46). I would cut both; they are yours to keep.
