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
Default path 🟡, with 🔴 extensions.
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
Two **90-minute blocks around the 30-min coffee break** (W-1, W-9). Time = participant minutes.
👥 = pairs compare results for ~5 min (W-11). ✍️ = participant code via `use()`. 🖍️ = draw on the canvas.

| # | File | Title | Objective (one line) | Level | Time | Status |
|---|---|---|---|---|---|---|
| 0 | sections/00_setup.py | Setup & the question | Load the cached models and data; see the question and outcomes | 🟢 | 5 | done |
| 1 | sections/01_out_of_the_box.py | Out of the box: pets | See a foundation model segment pets almost perfectly, zero-shot | 🟡 | 15 | done |
| 2 | sections/02_falls_short.py | Where it falls short | Run the same tools on the specialised task and probe SAM with escalating prompts | 🟡 | 20 | done |
| 3 | sections/03_hidden_labels.py | Hide the labels: the naive baseline | Train a head with "missing = background" and watch it fail differently | 🟡 | 20 | done |
| 4 | sections/04_ladder.py | The adaptation ladder | Zero-shot → one-shot → scribbles: see how predictions change at each rung | 🟡 | 25 | done |
| 5 | sections/05_learn_from_less.py | Path 1 · Learn from less | Treat missing labels as unknown; write the masked / PU loss | 🟡🔴 | 20 | done |
| 6 | sections/06_recover_supervision.py | Path 2 · Recover supervision | Turn SAM pseudo-labels and Grad-CAM maps into training signal | 🟡🔴 | 20 | done |
| 7 | sections/07_use_fm_directly.py | Path 3 · Skip specialist training | Prompt the foundation model directly with scribble points and boxes | 🟡 | 10 | done |
| 8 | sections/08_compare.py | Compare everything | Fill the comparison table, predicting the winner first | 🟡 | 15 | done |
| 9 | sections/09_your_images.py | Your own images | Scribble on your own images and run the ladder | 🟡 | 20 | done |
| 10 | sections/10_principles.py | Guiding principles & wrap-up | Turn the results into rules you can reuse | 🟢 | 5 | done |

**Block 1** = sections 0–3 and section 4 up to the scribble rung (4.3), "when it works and when it doesn't". The coffee break
sits **just before 4.3** (S0, Simone 2026-10-26; was: end of 03). It carries a no-code question to think about over coffee
(Q17: "what could you tell the training about a pixel nobody labelled?", with a hint for after the break); block 2 opens by
collecting answers, then 4.3 starts with "Welcome back" and the annotator's new instruction.
**Block 2** = 4.3 (scribbles) to section 10, "ways out".

### Section contents
- **00 Setup & the question**
  - Title banner (Theme L-003) with the title, outcomes, level legend, and "run `setup_check.ipynb` first".
  - Setup cell: IN_COLAB, SMOKE, DEVICE (cuda / cpu with a plain message), SEED, `RESULTS = {}`, `use()` (N-029), `SPECIALISED` config, the Theme style.
  - Loading models (timm DINOv3 ViT-S/16, SAM per §5) and data from the Drive cache, or a local cache in SMOKE.
  - The DINOv3 licence notice.
  - One sentence per cell on what and why (N-031).
- **01 Out of the box: pets**
  - SAM with 1 point and with a box on pets (it shines).
  - DINOv3 patch features shown as a 3-colour PCA image.
  - A linear head trained on 5 pets with full masks (seconds), with IoU on held-out pets, and a step-by-step figure of the head
    (patches, one probability per patch, upsampled map, mask) with one patch worked by hand (moved here from section 3).
  - ✍️ optional `my_iou`. 👥 optional.
  - **Interactive SAM** (`sam_playground`, defined in 00): click, shift-click for background, drag a box, or type a word
    for SAM 3 if cached; IoU and "marked" update live. Also in sections 2, 7.4 and 9 (Simone 2026-10-26: "way more
    interactive"). Headless it shows one fixed click and one box.
- **02 Where it falls short**
  - The same three tools on the specialised task, plus a failure gallery.
  - **SAM probed with escalating prompts** (1 → 5 → 20 points; box) (W-4 research); one figure shows the four prompts next to SAM's masks.
  - Resolution effect: features at 512 vs 1024 for thin structures (the attention arithmetic is a go-deeper box).
  - clDice is collapsed reference code with an interactive visual (three cases: too wide, capillaries missed, blob). No
    ✍️ here: the workshop is not about vessel segmentation (Simone 2026-10-26, reverses the earlier "keep `my_cldice`").
  - The playground: "try to get SAM to mark only the vessels".
  - 🔴 optional: the DINOv3 PCA on retina images, the whole-fundus view (the "thick trunk" simulation was cut 2026-10-26).
  - Predict before you run (N-004).
  - 👥 "whose image fails worst?" (a show-of-hands moment, in FACILITATION).
- **03 Hide the labels: the naive baseline**
  - Generate incomplete supervision (§4).
  - Train the head with **missing = background**, evaluate on the hidden ground truth, and show where it fails ("it learns that vessels are background").
  - No ✍️ here: `hidden_recall` is reference code. 🔴 optional: the balanced-loss head ("is it the labels, or the loss?").
  - "Hidden" = vessels in the full ground truth that the label file left unpainted; defined with a picture in 3.1–3.2.
  - The naive baseline is explained as numbered steps, the guess comes before the "why" (N-004).
  - A slider (`HIDE_PERCENT`) rebuilds the label file with more or fewer hidden segments and retrains the naive head.
  - Leads straight into section 4's first rungs; the break comes later, before 4.3 (S0).
- **04 The adaptation ladder** (W-3)
  - Rungs: zero-shot (02) → **one-shot**: one annotated image, as a nearest-feature prototype (🔴 optional: the same map used as SAM prompts) → **scribbles** (auto-scribbles, partial cross-entropy that ignores unlabelled pixels: the MIT DINOv2-scribble idea applied to DINOv3).
  - Show **how the predictions differ**, not "quality rises" (Simone: quality need not improve).
  - ✍️ optional `my_scribble_targets`: choose which unlabelled pixels to ignore.
  - The coffee break and its question sit before 4.3 (see §6 above).
  - 🔴 optional: the difference maps between rungs, and "choose a distance" (the "surely background" shortcut, with `BG_BEYOND_PX`).
- **05 Path 1 · Learn from less**
  - Treat missing as **unknown**: a masked loss, then 🔴 a PU-inspired loss.
  - ✍️ `masked_loss(logits, labels, ignore_mask) -> torch.Tensor` via `use()`, so participants write the loss (W-5).
  - Compare with section 03's naive baseline.
  - The three recipes side by side on one image: what each loss hears (row 1) and the mistake it produces (row 2).
- **06 Path 2 · Recover supervision**
  - **SAM pseudo-labels** from scribble points, used as training targets for the head.
  - **Grad-CAM** from a small classifier on image-level or patch-level labels, thresholded into a pseudo-mask *or* used as a SAM prompt.
  - ✍️ optional `my_pseudo_targets`. The pseudo-label threshold and the unsure band are Colab sliders (`PSEUDO_THRESHOLD`, `UNSURE_BAND`).
  - The Grad-CAM sketch and its step-by-step sit in a go-deeper box; 🔴 optional: Grad-CAM peaks as SAM clicks.
  - **[OPEN]** If the specialised task is retina, Grad-CAM needs an image-level task. Default: classify patches as "contains vessel / not", and say plainly that this is an awkward fit. The bake-off reports whether Grad-CAM is meaningful.
- **07 Path 3 · Skip specialist training**
  - SAM prompted directly with points from the scribbles (one computed row; the number-of-clicks variation is 🔴), plus one small box and a centre click per stroke. No training.
  - The box size and the centre click are a slider (`BOX_PAD_TRY`) and a checkbox (`WITH_CLICK_TRY`). The thickened strokes (no model) are the baseline to beat.
  - 🔴 SAM 3 text prompts if available: a dropdown with free text, plus the playground on retina and pets images.
- **08 Compare everything**
  - First, a ✏️ "your prediction: which approach wins, which is steadiest?" cell with two dropdowns (`MY_WINNER`, `MY_STEADIEST`; W-11).
  - Then the **comparison table** (W-15), one row per approach:
    - accuracy = IoU / Dice (+ clDice for vessels) on the hidden ground truth;
    - annotation effort = labelled pixels + strokes → estimated minutes, with the minutes-per-stroke assumption stated;
    - reliability = spread over 3 seeds or images;
    - speed / compute = seconds of training + inference;
    - scalability = discussion only.
  - A small plot, then a violin plot of the per-image IoU.
  - 👥 defend a row for a given situation.
- **09 Your own images**
  - Upload 1–3 images. Draw scribbles on a **canvas inside Colab** (HTML canvas + `google.colab.output`), with an **upload-a-PNG fallback** (W-6).
  - Run the ladder on them: zero-shot SAM, the scribble head, SAM from scribble points.
  - In SMOKE / without Colab, use a bundled example image and generated scribbles.
  - ⚠️ Never use Gradio's public share link (privacy).
  - 👥 compare own-image results.
  - Sections 2, 4 and 7 end with a one-line pointer to this section; the own-image cells that W-10 first put there were cut (2026-10-26), so section 9 is the only place for participants' own photos. Never mandatory.
  - The setup email asks each participant to bring one image (S8).
- **10 Guiding principles & wrap-up**
  - 4–6 principles derived from *this run's* table (e.g. "thin structures defeat patch features: check the resolution first").
  - Outcomes restated ("At the end you can"); "six questions for any task" and "where to go next" are go-deeper boxes; licences and credits.
  - A closing cell for their own problem (`MY_PROBLEM`, `MY_PRINCIPLE`, `MY_FIRST_STEP`); no separate whole-workshop takeaway.

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
- Extensions (🔴): PU loss, SAM 3 text prompts, the U-Net-from-scratch baseline, a Frangi vesselness filter (if retina): "a 1998 filter vs a 2025 foundation model".

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
- `FACILITATION.md` covers both blocks, with timings, 👥 moments, show-of-hands moments, and "if short on time" cuts (N-033, W-14).
- The 19-Nov AMLD agenda draft can be read off §6 (W-14).

## 10. Open questions
- **W18**: how participants get SAM 3. Default: SAM 2.1, plus SAM 3 if the participant already has access. FHNW legal is to be asked about re-hosting.
- **Specialised dataset**: the bake-off (issue #1) → Simone picks. Provisional: retina (HRF).
- Grad-CAM on retina (section 06): is it meaningful? The bake-off reports.
- Minutes-per-stroke assumption for "annotation effort" (section 08): default 2 s per scribble stroke and ~3–5 h per full vessel map (FIVES paper). Stated in the notebook as assumptions.
