# SPEC — When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation

**With:** Susanne Suter

The single source of truth for **what** this notebook is. `AGENTS.md` says how to work,
`RULES.md` says what every notebook must respect, and this file says what *this* one must do.
Written 2026-10-19 from Simone's grilling (W-1…W-17 on the vault page
`wiki/work/projects/Foundation Models Workshop.md`) and the accepted AMLD 2027 proposal (the only
design source; the 2025 `AMDL_workshop` repo is **discarded**). Agents never change it, except the
`Status` column of §6. **[OPEN]** marks a default taken while Simone's answer is pending.

## 1. Purpose
A hands-on workshop at **AMLD 2027** (EPFL, Thu 11 Mar 2027) and **SDS2027** (Zurich, 1 Jun 2027).
Participants start from a vision foundation model that works out of the box on a simple
segmentation task. They watch it fall short on a specialised one, then add task-specific
information step by step: scribbles, missing labels, pseudo-labels, explanations, direct prompting.
Every approach is compared on the same hidden ground truth.
The question it answers: **"What can modern AI solve out of the box, when do its limitations start to matter, and what should we do next?"**
They leave with a reusable experimental framework and guiding principles that transfer to other tasks.

## 2. Audience and level
AMLD audience: practitioners and researchers with practical ML experience; no computer-vision
background needed (proposal, "Intermediate level"). 40–200 people in one room, first come first served.
Default path 🟡; only one 🔴 part is left (the PU one-liner in 5.3).
**Freedom (N-028): medium** (G-14, W-5):
- Participants **write their own code and do manual work** (scribbling).
- Every such task is **strongly suggested but never mandatory**: a default answer exists via `use()` (N-029), while the value comes from doing it themselves (W-5).
- **It must be a practical workshop.** AMLD feedback: "less slides, more action", and "explanations … rather than just executing a Notebook" (N-031).

## 2b. Prerequisites
Basic ML (training, validation, objectives, evaluation); basic Python helpful; a laptop with a browser
and a Google account (Colab). The **setup-check notebook** must be run in the week before (W-12, N-030).

## 3. Learning outcomes
These appear verbatim in the title banner.
1. Judge when a vision foundation model is good enough out of the box, and recognise the signs that it is not.
2. Adapt it with cheap, incomplete supervision (scribbles, missing labels, pseudo-labels, explanations, prompts), and explain what each approach uses and assumes.
3. Compare approaches on accuracy, annotation effort, reliability and compute, and decide what to do next on a new problem, including your own images.

## 4. Use case and data

### Simple task: Oxford-IIIT Pet (W-2)
- Pet vs background from the trimap: pet = class 1. The trimap's "not sure" band (class 3, ~11 % of an image) is
  excluded from scoring and from the heads' loss, and drawn grey (Simone's review 2026-10-26: counting it as pet made the
  ground truth spill onto the rug). It is the workshop's idea in miniature: an annotator's "unknown" is not "background".
- Licence **CC BY-SA 4.0**: we may host a resized subset with attribution and share-alike. ~100 images in the workshop, 4 in SMOKE.

### Specialised task: **[OPEN — chosen by the bake-off, issue #1]**
- Simone prefers **retinal imaging** if a public dataset does the job; cell microscopy or industrial defects are acceptable (W-2).
- **Bake-off candidates** (research 2026-10-19, `docs/work/research/2026-10-19-fm-specialised-datasets.md` in the vault):
  - **Retina — HRF:** 45 fundus images, 3504×2336, vessel masks, CC BY 4.0, `https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip` (76 MB). FIVES (800 images, CC BY 4.0, 1.76 GB `.rar` from figshare) is the larger follow-up. The hook: "annotating one image would take approximately 3–5 hours" (FIVES paper).
  - **Industrial — KolektorSDD2:** 3,335 images, 356 defective, masks, CC BY-NC-SA 4.0, `https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip` (853 MB). Image-level defect labels make Grad-CAM natural.
  - **Microscopy — DSB2018 stage1_train:** nuclei, CC0, 83 MB (LIVECell is the larger follow-up, CC BY-NC 4.0).
- **Provisional pick until Simone decides: retina (HRF)** (W-8: the agent may keep building on its pick).
- Sections after 02 must not hard-code the dataset. Use a `SPECIALISED` config dict (name, loader, preprocessing size, scribble generator, metric extras, e.g. clDice for vessels).

### Incomplete supervision, generated from the full ground truth (hidden for evaluation)
- **Missing labels:** hide ~50 % of foreground components (or vessel segments).
- **Auto-scribbles:** the ground-truth skeleton with ~70 % of segments dropped, plus 3–5 background strokes (W-6).
- **Own images:** scribbles from the canvas.
- The full masks are kept **only for evaluation**.

### Smoke subset (N-014)
4 pet images and 4 specialised images at reduced resolution, CPU, budget **1,200 s**.

## 5. Environment

| Item | Value |
|---|---|
| Runs on | Google Colab, **T4 GPU expected** (W-13), and **every step must complete on CPU** in reasonable time using precomputed features and fallback checkpoints (W-7) |
| Python | 3.12; Colab preinstalls torch, transformers 5.18, timm 1.0.30 |
| **DINOv3** | `timm` **`vit_small_patch16_dinov3.lvd1689m`**: ungated on HF (verified 2026-10-19 "gated: False"), DINOv3 licence attached. Show the licence notice. Participants need no account (W-4) |
| **SAM** | **[OPEN, W18]** Default: **SAM 2.1 hiera-small** (`facebook/sam2.1-hiera-small`, Apache-2.0, ungated, verified). SAM 3 is an **optional upgrade** if the participant has Meta's approval on HF, or if FHNW re-hosts it after legal sign-off. SAM 3 runs in fp16 via transformers, never Meta's bf16 path, on a T4 |
| Grad-CAM | `grad-cam` (pytorch-grad-cam, MIT); the only `%pip install` |
| Licence checks | Re-check the DINOv3/SAM 3 terms ~4 weeks before each event (Meta can change them) |
| Precomputed | DINOv3 patch features for every workshop image, plus a checkpoint for every training step ("not done in 3 min? load ours") (W-7), downloaded by the setup check |
| Setup check | A separate small notebook `setup_check.ipynb`, shared a week before (W-12, N-030), with an email that also asks each participant to **bring one image of their own** for section 9 (S8, Simone 2026-10-26). It downloads models, data and caches into the participant's Google Drive and prints ✅. The main notebook loads from that cache |
| Smoke run | `NB_SMOKE=1`, CPU, budget 1,200 s; local test images only, no Drive |

## 6. Sections
Two blocks of about 85 and 90 minutes around the **30-min coffee break**, which sits **just before 5.3** (W-1, W-9; moved
by S0 and again on 2026-10-26). Time = participant minutes, summing to ~175. 👥 = pairs compare results for ~4–5 min (W-11), in
sections 2, 6, 8 and 9. ✍️ = participant code via `use()`. 🖍️ = draw on the canvas.

| # | File | Title | Objective (one line) | Level | Time | Status |
|---|---|---|---|---|---|---|
| 0 | sections/00_setup.py | Setup & the question | Load the cached models and data; see the question and outcomes | 🟢 | 7 | done |
| 1 | sections/01_out_of_the_box.py | Out of the box: pets | See a foundation model segment pets almost perfectly, zero-shot | 🟡 | 18 | done |
| 2 | sections/02_falls_short.py | Where it falls short | Run the same tools on the specialised task, probe SAM with escalating prompts, and name the signs that it is not good enough | 🟡 | 24 | done |
| 3 | sections/03_hidden_labels.py | Hide the labels: the naive baseline | Train a head with "missing = background" and watch it fail differently | 🟡 | 12 | done |
| 4 | sections/04_ladder.py | The adaptation ladder | Zero-shot → scribbles: see how predictions change at each rung (one-shot in a go-deeper box) | 🟡 | 12 | done |
| 5 | sections/05_learn_from_less.py | Path 1 · Learn from less | Treat missing labels as unknown (masked loss), then tell the loss how many vessels hide among them (PU loss) | 🟡🔴 | 22 | done |
| 6 | sections/06_recover_supervision.py | Path 2 · Recover supervision | Turn SAM pseudo-labels and Grad-CAM maps into training signal, and know when to distrust them | 🟡 | 18 | done |
| 7 | sections/07_use_fm_directly.py | Path 3 · Skip specialist training | Prompt the foundation model directly with scribble points and boxes | 🟡 | 12 | done |
| 8 | sections/08_compare.py | Compare everything | Fill the comparison table, predicting the winner first | 🟡 | 16 | done |
| 9 | sections/09_your_images.py | Your own images | Scribble on your own images and run the ladder | 🟡 | 25 | done |
| 10 | sections/10_principles.py | Guiding principles & the framework | Turn the results into rules and six questions you can reuse | 🟢 | 9 | done |

**Block 1** (about 85 min) = sections 0–4 and 5.1–5.2, "when it works, when it doesn't, and the first way to read missing labels".
Section 5 is split by the break: 5.1–5.2 (the masked loss) take about 12 min, 5.3 (the PU loss) about 10.
The coffee break sits **just before 5.3** (Simone 2026-10-26; before that it was before 4.3, and before that at the end of 03).
It carries a no-code question to think about over coffee (Q10: "the masked head over-marks; the annotator says they painted about
half of the vessels; how could that one number help the training?", with a hint for after the break); block 2 opens by collecting
answers, then 5.3 starts with "Welcome back" and the PU loss.
**Block 2** (about 90 min) = 5.3 (the PU loss) to section 10, "the rest of the ways out, the comparison, and your own images".

Questions are numbered **Q1–Q23** straight through the notebook (Q1 in 00; Q2–Q4 in 01; Q5–Q7 in 02; Q8 in 03; Q9 in 04; Q10–Q13 in 05;
Q14–Q16 in 06; Q17–Q18 in 07; Q19–Q20 in 08; Q21–Q22 in 09; Q23 in 10). Two former questions are facilitator discussions in
`FACILITATION.md` and not in the notebook: Q11's "cracks in concrete" checklist (section 2) and the invented pilot (section 6).

### Section contents
- **00 Setup & the question**
  - Title banner (Theme L-003) with the title, outcomes, level legend, and "run `setup_check.ipynb` first".
  - Setup cell: IN_COLAB, SMOKE, DEVICE (cuda / cpu with a plain message), SEED, `RESULTS = {}`, `use()` (N-029), `SPECIALISED` config, the Theme style.
  - Loading models (timm DINOv3 ViT-S/16, SAM per §5) and data from the Drive cache, or a local cache in SMOKE.
  - The DINOv3 licence notice.
  - One sentence per cell on what and why (N-031). Q1 🟢 is a written prediction on the two tasks.
- **01 Out of the box: pets**
  - SAM with 1 point and with a box on pets (it shines).
  - DINOv3 patch features shown as a 3-colour PCA image.
  - A linear head trained on 5 pets with full masks (seconds), with IoU on held-out pets, and a step-by-step figure of the head
    (patches, one probability per patch, upsampled map, mask) with one patch worked by hand.
  - ✍️ optional one-line `my_union` warm-up in 1.2: the intersection is given and the learner writes the union; reference collapsed, `use()` fallback unchanged. No 👥 in this section.
  - **Interactive SAM** (`sam_playground`, defined in 00, Q2): click, shift-click for background, drag a box, or type a word
    for SAM 3 if cached; IoU and "marked" update live; 👁 Show ground truth colours found / invented / missed / not scored. Also in sections 2, 7.4 and 9 (Simone 2026-10-26: "way more
    interactive"). Headless it shows one fixed click and one box.
- **02 Where it falls short**
  - The same three tools on the specialised task, plus a failure gallery (Q7, 👥 "whose image fails worst?").
  - **SAM probed with escalating prompts** (1 → 5 → 20 points; box) (W-4 research); one figure shows the four prompts next to SAM's masks (Q6).
  - Resolution effect: features at 512 vs 1024 for thin structures, in two numbers, next to a pets-vs-vessels chart.
  - clDice appears **only here (2.2) and in the section-8 table**: collapsed reference code with an interactive visual (`CASE`: too wide, capillaries missed, blob). Elsewhere titles and printouts show IoU · recall · marked. No
    ✍️ here: the workshop is not about vessel segmentation (Simone 2026-10-26, reverses the earlier decision to keep a clDice exercise).
  - The playground: "try to get SAM to mark only the vessels" (Q5).
  - The signs that a model is not good enough (the "everything" line, resolution, IoU vs clDice) are closed by a **facilitator discussion** (the "cracks in concrete" checklist, ex-Q11), not a notebook question.
  - Cut 2026-10-26: the DINOv3 PCA on retina images, the whole-fundus view and the "thick trunk" simulation.
  - Predict before you run (N-004).
- **03 Hide the labels: the naive baseline**
  - Generate incomplete supervision (§4).
  - Train the head with **missing = background**, evaluate on the hidden ground truth, and show where it fails ("it learns that vessels are background").
  - No ✍️ here: `hidden_recall` is reference code. The 🔴 balanced-loss head was cut 2026-10-26.
  - "Hidden" = vessels in the full ground truth that the label file left unpainted; defined with a picture in 3.1–3.2.
  - The naive baseline is explained as numbered steps, the guess comes before the "why" (N-004).
  - A slider (`HIDE_PERCENT`, Q8) rebuilds the label file with more or fewer hidden segments and retrains the naive head.
  - Leads straight into section 4; the break comes later, before 5.3.
- **04 The adaptation ladder** (W-3)
  - Main path: zero-shot (02) → **scribbles** (auto-scribbles, partial cross-entropy that ignores unlabelled pixels: the MIT DINOv2-scribble idea applied to DINOv3). Q9 reads the side-by-side table and error maps.
  - **One-shot** (one annotated image, as a nearest-feature prototype) sits in a collapsed 🔍 go-deeper box at the end, with a plain prompt (no question number) and an `ONE_SHOT_IMAGE` field; it still runs, so `RESULTS["ladder"]` holds `zero_shot`, `one_shot`, `scribbles`.
  - Show **how the predictions differ**, not "quality rises" (Simone: quality need not improve).
  - No ✍️ here: a dropdown `UNLABELLED_TARGET` (-1 ignored, 0 background) on the scribble-head cell replaces the old target-building exercise; 0 brings section 3's failure back.
  - Cut 2026-10-26: prototype → SAM clicks, the difference maps between rungs, the "surely background" distance (`BG_BEYOND_PX`) and its question.
- **05 Path 1 · Learn from less**
  - Treat missing as **unknown**: a masked loss (5.1–5.2, block 1), then, after the break, the **PU loss on the main path** (5.3, block 2; no longer an extension).
  - ✍️ optional one-line `known_pixels(ignore_mask)` (masked loss; the balanced loss is given, returning all-ones is the naive loss) via `use()`, so participants write the part that matters (W-5).
  - 5.3: one plain paragraph, a three-step theory box, the helper collapsed, one knob `LABELLED_SHARE` (0.2–1, default 0.5), and the figure below.
  - 🔴 challenge: one-line `pu_background` (`L_b = (all_as_bg − prior · painted_as_bg) / (1 − prior)`), reference collapsed, `use()` fallback unchanged.
  - Compare with section 03's naive baseline.
  - One 2 × 3 figure after PU: columns naive, masked, PU; row 1 the hidden vessels each head found on a training image, row 2 its error map on the demo image; plus a recall / precision chart (Q11–Q12). Q13 asks for a real annotation habit that breaks the random-hiding assumption.
  - The break question Q10 sits between 5.2 and 5.3.
- **06 Path 2 · Recover supervision**
  - **SAM pseudo-labels** from scribble points, used as training targets for the head.
  - **Grad-CAM** from a small classifier on patch-level labels, thresholded into a pseudo-mask (lean: no step-by-step box, no peaks-as-clicks).
  - No ✍️ here: `pseudo_targets` is collapsed reference code. The pseudo-label threshold and the unsure band are Colab sliders (`PSEUDO_THRESHOLD`, `UNSURE_BAND`, Q15, with an optional 👥).
  - The **blind-spot simulation** stays, with a legend and a "what to look for" line.
  - The invented pilot (ex-Q28) is a **facilitator discussion** in `FACILITATION.md`, not a notebook question.
  - **[OPEN]** If the specialised task is retina, Grad-CAM needs an image-level task. Default: classify patches as "contains vessel / not", and say plainly that this is an awkward fit. The bake-off reports whether Grad-CAM is meaningful.
- **07 Path 3 · Skip specialist training**
  - SAM prompted directly with points from the scribbles (one computed row, no click-count slider), plus one small box and a centre click per stroke. No training.
  - The box size and the centre click are a slider (`BOX_PAD_TRY`) and a checkbox (`WITH_CLICK_TRY`, Q17). The thickened strokes (no model) are the baseline to beat.
  - 7.4 SAM 3 text prompts if available (Q18): a playground with free text on retina images, needing SAM 3 access and otherwise skippable; an optional go-deeper cell scores one phrase on all images (`SAM3_TEXT` dropdown with free text).
- **08 Compare everything**
  - First, a ✏️ "your prediction: which approach wins, which is steadiest?" cell with two dropdowns (`MY_WINNER`, `MY_STEADIEST`; W-11; Q19).
  - Then the **comparison table** (W-15), one row per approach:
    - accuracy = IoU / Dice (+ clDice for vessels) on the hidden ground truth;
    - annotation effort = labelled pixels + strokes → estimated minutes, with the minutes-per-stroke assumption stated;
    - reliability = spread over 3 seeds or images;
    - speed / compute = seconds of training + inference;
    - scalability = discussion only.
  - A small plot, then a violin plot of the per-image IoU and the winner's worst, typical and best image.
  - 👥 defend a row for a given situation (Q20).
- **09 Your own images**
  - Upload 1–3 images. Draw scribbles on a **canvas inside Colab** (HTML canvas + `google.colab.output`), with an **upload-a-PNG fallback** (W-6).
  - Run the ladder on them: zero-shot SAM, the scribble head, SAM from scribble points (Q21); an optional playground first.
  - No ✍️ here: the `disagreement` map (9.4, Q22) is collapsed reference code; sliders (`N_OWN_CLICKS`, `OWN_BOX_PAD`) and the canvas are the hands-on part.
  - In SMOKE / without Colab, use a bundled example image and generated scribbles.
  - ⚠️ Never use Gradio's public share link (privacy).
  - 👥 compare own-image results.
  - Sections 2, 4 and 7 end with a one-line pointer to this section; the own-image cells that W-10 first put there were cut (2026-10-26), so section 9 is the only place for participants' own photos. Never mandatory.
  - The setup email asks each participant to bring one image (S8).
- **10 Guiding principles & the framework**
  - 4–6 principles derived from *this run's* table (e.g. "thin structures defeat patch features: check the resolution first").
  - **The six questions for any task** are a main-path card (10.2), not a go-deeper box: the reusable experimental framework the proposal promises. Each question carries a number you can measure.
  - Outcomes restated ("At the end you can"); "where to go next" is a go-deeper box; licences and credits.
  - A closing cell for their own problem (`MY_PROBLEM`, `MY_PRINCIPLE`, `MY_FIRST_STEP`, Q23); no separate whole-workshop takeaway.

### Separate deliverable: `setup_check.py` → `setup_check.ipynb`
- Mounts Google Drive and downloads the DINOv3 timm weights, SAM weights, data subsets, precomputed features and checkpoints into `MyDrive/fm-workshop-cache/`.
- Verifies each with a checksum, and prints ✅ per item plus the total time.
- If SAM 3 access exists, it also caches SAM 3 (optional).
- ⛔ It needs `scripts/build.py` to build a second notebook (issue scope).

## 6b. Pitfalls and extensions
- Pitfalls to anticipate:
  - a CPU-only Colab session;
  - Drive not mounted;
  - training "too slow" → load our checkpoint;
  - canvas not rendering → upload fallback;
  - over-trusting SAM's confident masks.
- Extensions (🔴): the PU one-line challenge `pu_background` (the PU loss itself is on the main path since 2026-10-26), the U-Net-from-scratch baseline, a Frangi vesselness filter (if retina): "a 1998 filter vs a 2025 foundation model". SAM 3 text prompts (7.4) are not an extension but are skippable when access is missing.

## 7. Out of scope / limitations
- No full fine-tuning of the foundation models (heads only; a light decoder at most).
- No local-environment setup path in the participant text (N-017), although the proposal mentions it.
- No commercial services.
- Participants' own images never leave their Colab session.

## 8. Notebook-specific rules
**L-001 — Callouts (N-034):** a markdown cell whose first line is `<!-- callout: <kind> | <title> -->`, followed by markdown. `scripts/build.py` turns it into a hidden form code cell (`# @title`) that draws the box as HTML and saves the same HTML as the cell's output, so the box shows before anything runs. Never a `<table bgcolor>`: Colab cuts text in table cells, and the lint fails on one. Contract: `reports/callout-system.md`. Kinds, each written as background / bar and text colour:
- 🎯 Objective `#E8F5E9` / `#1B5E20`;
- ✅ Takeaway `#E0F2F1` / `#004D40` (a "When you see X → do Y" list, N-044);
- 📘 Theory `#EDE7F6` / `#311B92`;
- ✏️ Your turn `#FFE8D6` / `#7A2E00`;
- ❓ Questions `#F3E5F5` / `#4A148C` (👥 pair moments keep this colour; the title's emoji replaces the icon);
- 🔀 Across approaches `#E8EAF6` / `#1A237E`;
- ⚠️ Pitfall `#FDECEA` / `#8B1A10`;
- 🧾 Note / source `#FFF8E1` / `#5D4037`;
- 🔍 Go deeper `#F5F5F5` / `#616161`: muted and collapsed by default, for secondary depth (N-048).

**L-002 — Code participants write or change:** `# ✍️ YOUR CODE` / `# ✏️ YOUR TURN` headers, a signature and docstring, an `...` body, and `use(fn, reference=_ref_fn)` (N-029). **Never mandatory** (W-5).

**L-003 — Theme violet → orange** (W-16): cool violet = "it works", warm orange = "it fails / we adapt".
- Title banner: a `banner` callout (L-001), full width, background `#4A148C`, white text.
- Figure palette: works `#6A1B9A`, fails `#E65100`, ground truth `#2E7D32`, scribbles `#00838F`, background overlay grey.
- One colour per *approach*, kept identical across the ladder, the paths and the comparison table.

**L-004 — Model provenance and licence** (shrunk 2026-10-21 for N-045; open question for Simone): every model load prints one short line, `model · source · licence`, and nothing more. No licence or attribution talk in the body text. All credits and licences, including the DINOv3 notice, sit in one collapsed 🔍 "Credits & licences" callout at the end of section 10; the repo keeps them in `README.md` / `data/ATTRIBUTION.md`.

**L-005 — Ground truth is for evaluation only.** Training code may use only the incomplete supervision of the current rung. Checked in review: no test can prove it, so the morning review reads the training cells for it.

**L-006 — CPU path:** every training cell checks the elapsed time and offers the precomputed checkpoint. Under SMOKE, tiny steps on CPU.

**L-007 — Figures carry their legend (N-059):** every figure draws a key for every colour and marker it uses, inside the figure; an error map always names found / invented / missed / not scored (violet / orange / green / light grey). Model masks are violet, ground truth green, annotations teal; orange means "invented" in images and nowhere else. Shared helpers: `add_key` and `show_masks(..., legend=, note=)` in 01, `show_clicks` in 02, `show_keyed` in 03. Each figure is preceded by one sentence saying what it shows and one line on what to look for.

## 9. Acceptance
- `just check` passes, including `tests/test_content.py` (the night agent never edits it).
- **Results contract:**
  - `RESULTS = {}` is created in 00, and sections write named results into it.
  - The smoke test dumps them.
  - Progressive gate: a test runs once the sections it needs have a status other than `todo`.
- Content tests:
  - **pets zero-shot IoU > specialised zero-shot IoU** (the story holds);
  - the naive baseline's recall on hidden foreground < the masked-loss head's recall (missing ≠ background, the lesson holds);
  - the comparison table has one row per approach with all five numeric columns filled;
  - each `use()` falls back cleanly when the participant cell is untouched;
  - the canvas section runs headless with the bundled example.
- Every step completes on CPU in SMOKE (W-13). Simone runs it fully on a Colab T4 by hand.
- `FACILITATION.md` covers both blocks and the break before 5.3, with timings, 👥 moments, show-of-hands moments, "if short on time" cuts, and the two facilitator discussions (the cracks checklist and the invented pilot) (N-033, W-14).
- The 19-Nov AMLD agenda draft can be read off §6 (W-14).

## 10. Open questions
- **W18**: how participants get SAM 3. Default: SAM 2.1, plus SAM 3 if the participant already has access. FHNW legal is to be asked about re-hosting.
- **Specialised dataset**: the bake-off (issue #1) → Simone picks. Provisional: retina (HRF).
- Grad-CAM on retina (section 06): is it meaningful? The bake-off reports.
- Minutes-per-stroke assumption for "annotation effort" (section 08): default 2 s per scribble stroke and ~3–5 h per full vessel map (FIVES paper). Stated in the notebook as assumptions.
