# Cutting review (N-047) — 2026-10-21, after the transfer pass

The short version: the notebook no longer fits. At participant pace I estimate about **256 minutes** of
work against a 180-minute slot (SPEC budget 175). Compute is not the problem: on a T4 the whole run costs
roughly 3–4 minutes of model time. The problem is volume. There are about 1,550 visible markdown lines,
about 85 figures and 36 questions, and reading alone takes around 105 minutes. Cutting a few cells will
not close a gap of about 80 minutes. The package below does it by removing repetition. The same point is
often made three or four times: a pitfall, a takeaway, an "across" callout and a worst/typical/best figure.
I also move several second-order exercises to 🔴, and one SPEC-level item (Grad-CAM) needs your call.
These are proposals only. Simone decides.

## 1. Estimate per section

**Method.** Reading counts the visible markdown at 220 words per minute. Folded `<details>` and go-deeper
text count at 50 %, and existing 🔴 parts at 25 %. Each run cell adds 0.2 min, each figure 0.4 min, each
🟡 question 1.5 min (+1 if it asks for a re-run), each "Your prediction" 0.75 min, each ✍️ exercise half
its stated minutes (about half the room tries it), and 👥 pair moments their stated minutes. On top of
that comes the T4 compute read off the code: 25 held-out 1024 px crops, 60 held-out pets, 200-step heads,
and one SAM processor + encoder pass of roughly 0.2 s per `sam.probability` call. Facilitator talk is
assumed to overlap with reading.

| Section | SPEC budget | My estimate | Delta | After package |
|---|---|---|---|---|
| 00 Setup | 5 | 10 | +5 | 8 |
| 01 Out of the box | 15 | 21 | +6 | 18 |
| 02 Falls short | 20 | 40 | +20 | 26 |
| 03 Hidden labels | 20 | 18 | −2 | 12 |
| 04 Ladder | 25 | 30 | +5 | 18 |
| 05 Learn from less | 20 | 19 | −1 | 15 |
| 06 Recover supervision | 20 | 32 | +12 | 16 |
| 07 Use FM directly | 10 | 23 | +13 | 8 |
| 08 Compare | 15 | 21 | +6 | 19 |
| 09 Your images | 20 | 31 | +11 | 27 |
| 10 Principles | 5 | 11 | +6 | 8 |
| **Total** | **175** | **≈ 256** (range 220–290) | **+81** | **≈ 175** |

Even the optimistic end of the range (220) is 40 minutes over the slot. Compute per section on a T4 stays
under a minute everywhere. The largest costs are 02's SAM ladder (about 100 SAM calls, ~20 s) and 01's
120 SAM calls (~25 s).

The **06 confirmation-bias simulation** (06:478-536) costs about 5–10 s on a T4. It trains two 200-step
linear heads on five 1024² label maps (~1–2 s each). Most of the rest is CPU time in `bright_blob`: a
Gaussian with σ ≈ 17 px on six 1024² images, ~0.4 s each. So the GPU does not decide its fate. Its
reading and looking time does, about 2–3 min.

## 2. Cut proposals, ordered by minutes saved per unit of learning lost

1. **Cell intros down to one sentence**, notebook-wide (for example 02:343-345, 04:532-537, 06:906-911,
   07:449-457, 08:213-216) · shrink · **8 min**. N-031 asks for one sentence per cell. About 60 intros run
   to 2–4 lines and restate the figure title. Nothing is lost.
2. **Merge theory callouts that restate an earlier one** · merge · **5 min**. In 05:17-28 the masked loss
   is the same formula as 04:416-427, so leave one line pointing to 4.3. In 04:19-29 the rungs are
   described three times (intro, this callout, the 4.2 steps), so keep only the citations. 06:18-26 repeats
   the intro's pseudo-label definition. 07:19-31 repeats section 2 and 6.1. 01:18-28 repeats the intro
   bullets. Every concept still has one home.
3. **Remove repeated pitfalls, "across" callouts and the closing takeaway** · cut · **4.5 min**. The
   "over-trusting SAM" pitfall appears a third and fourth time (06:464-476, 07:558-567). The across
   callouts 06:981-990 and 07:734-745 restate the next section's intro. The whole-workshop takeaway
   10:263-278 repeats the six printed principles. 01:494 and 02:450 keep the pitfall, and 06's simulation
   shows it on data.
4. **The repeated visuals of this pass in 02** · cut / 🔴 / go-deeper · **5.5 min**.
   - Cut the difference map (02:405-426): the "marked" column and Q6 make the same point.
   - Move the trunk-only simulation (02:248-298) to 🔴: the `my_cldice` play print already contrasts IoU
     and clDice on a wide trace and a blob.
   - Merge the prompt preview (02:310-331) into the ladder figure, which already draws the clicks.
   - Move the whole-fundus row (02:647-670) to 🔴.
   - Turn the resolution pitfall (02:586-595) into go-deeper.
5. **Collapse 7.1 into a computed row** (07:182-273) · shrink, with Q27 to 🔴 · **5 min**. "Stroke clicks
   give a region" is already shown in 2.3 (vessel clicks) and 6.1 (the same 20 + 20 clicks). Keep one line
   plus a collapsed cell so the `sam_clicks` row still reaches 08. The new idea of section 7, one box per
   stroke, stays whole.
6. **Duplicate worst/typical/best figures** · cut / merge · **4.5 min**. Cut 02:672-694 (the gallery is
   already ranked, and 04's rung 0 shows SAM's extremes again), 06:164-182 and 07:381-400. In 03, drop
   Q11's first row (03:420-432), which repeats the figure just above (03:377-396). N-039 still holds where
   each statistic first appears: IoU in 01, recall in 03, precision/marked in 04, spread in 08.
7. **04 · 4.5 "choose a distance" + Q17 to 🔴** (04:658-749, keeping the 3-step pitfall 04:665-673) ·
   🔴 · **4.5 min**. Its lesson, "the section-3 mistake through the side door", is told by the pitfall
   alone. It costs a second training run plus a re-run loop.
8. **Framing trims** · merge / go-deeper · **4.5 min**. In 00, merge the recipe note, pitfall and takeaway
   (00:49-54, 473-485). Drop 01's optional 👥 (01:468-473), which FACILITATION already lists as "skip".
   Make 10's theory callout and "where to go next" (10:182-194, 250-294) go-deeper.
9. **06 · 6.3 Grad-CAM to 🔴** (06:538-898) · 🔴 · **9 min** · ⚠️ SPEC §6 (06) and outcome 2
   ("explanations") name it, so **Simone decides**. Keep one collapsed cell that still produces
   `gradcam_iou` (tests/test_content.py:86 needs it) and the 08 row. The pitfall itself says it is "a poor
   fit". **Lighter variant: 4 min.** Move only the shape sketch + prediction + step-by-step
   (06:679-749) to go-deeper and keep the theory callout, the pseudo-label cell and Q24.
10. **Collapse the "Try it on your image" cells into one-line invitations** pointing to section 9
    (02:766-801, 04:751-803, 07:666-721) · shrink · **3.5 min**. Section 9 does this properly. SPEC W-10
    asks only for "short invitations", so keep the sentence and drop the ~150 lines of code. The real-room
    cost is higher than counted, because people go looking for photos.
11. **Move 02 · 2.4 retina PCA + Q7 to 🔴** (02:458-490) · 🔴 · **3 min**. 01 · 1.3 taught PCA, and the
    "meet the retina" zoom already shows vessel against patch. FACILITATION already marks it "skip".
12. **Move 04 · prototype → SAM clicks + Q15 to 🔴** (04:321-377) · 🔴 · **3 min**. The cell stays
    collapsed so the `one_shot_sam` row reaches 08. The PerSAM trick comes back in 06's Q25.
13. **Move 03's linear head theory + step-by-step figure to go-deeper** (03:201-257) · **2.5 min**.
    01 · 1.4 already gives `w · f + b`, and its per-patch score heatmap (01:420-441) shows the same
    mechanism.
14. **Takeaways down to ≤ 3 "When → do" lines** (02:803-812, 03:524-535, 05:584-596, 06:968-979,
    07:723-732) · shrink · **2 min**. Several lines repeat the pitfalls of the same section.
15. **Cut Q29** (07:538-556) · cut · **2 min**. It is the same "1,000 new images: prompt or train?"
    question as 08's Q32(a), one section later.
16. **Cut Q10 and fold Q18 into Q19** (03:124-139, 05:177-196) · **2.5 min**. Q10's point, "you can't see
    what's missing", is made by the panel title and the 3.3 theory. Q18's point is made again by the
    fake-head print (05:415-426).
17. **Move the 06 confirmation-bias simulation to 🔴** (06:478-536) · **2 min**. The real SAM pseudo-label
    head in 6.2 already shows a head copying its teacher's habit. Keep it if you want N-049 met in 06 by a
    simulation rather than by real data.
18. **09: one image by default, `my_disagreement` → reference, Q35 out** (09:535-568, 608-625) · shrink ·
    **3 min**. The disagreement theory, the map, Q34 and the 👥 stay. This loses some hands-on in the
    section that matters most to people, so it is medium loss.
19. **Move 04's difference maps to 🔴** (04:604-632) · **1.3 min**. The error maps just above already
    show each rung, and Q16 can drop the reference.
20. **Cut 07's prompt-types figure** (07:141-180) · **1.2 min**. 6.1 drew the clicks and 7.2 draws the
    boxes.
21. **Low-value ✍️ to reference** · **2.5 min**. `my_hidden_recall` (03:154-190) is one line,
    `my_spread` (08:181-210) is `np.std`, and `my_stroke_box` (07:287-336) is min/max. Six ✍️ stay
    (`my_iou`, `my_cldice`, `my_scribble_targets`, `masked_loss`, `my_pseudo_targets`,
    `my_disagreement`), so Freedom stays medium. Bind the reference to the same name so Run all works.
22. **Cut the winner's worst/typical/best figure in 08** (08:312-326) · **1.2 min**. The violin already
    shows the spread. Medium loss: it is where N-039 meets "spread".
23. **Drop the ✍️ `my_cldice` exercise** (02:199-241) and keep the theory and the play print with the
    reference · **1.5 min**. FACILITATION already lists it as "skip". Medium loss: it is the only metric
    participants write for thin structures.

Safety notes for whoever applies these:
- Cells moved to 🔴 that feed RESULTS keep running collapsed: `one_shot_sam`, `sam_clicks`, the
  Grad-CAM row, and `balanced_head` (05 uses it).
- 02:255 imports `ndi`. If the trunk simulation is deleted instead of 🔴-labelled, move the import.

## 3. Recommended package

**Proposals 1–21 plus 23, with 9 as the full Grad-CAM move.** This saves about 81 min, so 256 → **≈ 175
min**, which meets the SPEC budget with 5 min of buffer inside 180. Only proposal 22 is left out, so
N-039's spread example stays.

If you keep Grad-CAM in the default path (variant 9, 4 min instead of 9), the package lands at about
**180 min**, right at the slot limit with no buffer. The facilitator's "if short on time" list then
becomes the buffer.

The package changes these SPEC lines, all of which are yours to decide:
- 06 Grad-CAM → 🔴 (only with the full move).
- W-10: the try-it invitations become one line each.
- §6 "06 ✍️ tune the threshold" is unaffected. 04's "✍️ choose which pixels to ignore" keeps
  `my_scribble_targets`, but its distance experiment (Q17) becomes 🔴.

The budgets should also be rebalanced to match reality: 00 5→8, 02 20→26, 03 20→12, 07 10→8, 09 20→27.

**Block balance after the package:** sections 0–3 ≈ 64 min, 04 ≈ 18 min, so block 1 (0–3 + 04) ≈ 82
min, and block 2 (05–10) ≈ 93 min. The SPEC's "block 1 = 0–3 + the start of 4" now means nearly all of 04
before the break. Block 2 runs about 3 min over, which the facilitation cuts absorb.
