# %% [markdown]
# ## 10. Guiding principles & wrap-up
#
# A table is good for one decision; a rule is what you take to the next problem. This last section reads the
# numbers *your* session produced in sections 1 to 8 and writes them up as a handful of principles, each
# with its evidence.
#
# They are not general truths about foundation models. They are what this run, on this task, with these
# settings, showed. If your neighbour's session ran with other settings, some sentences may read
# differently, and that is fine.

# %% [markdown]
# <!-- callout: objective | Objective -->
# turn the results into rules you can reuse.

# %% [markdown]
# This cell draws the whole story in one picture: the pets one box got almost right, the same box on the
# specialised task, and the best approach of every path on the same hidden ground truth.

# %%
# @title The story in one picture { display-mode: "form" }
TABLE = {r["approach"]: r for r in RESULTS["comparison_table"]}  # section 8's rows, by short name
EVERYTHING = RESULTS["specialised_everything_iou"]
PATHS = {  # the rows of section 8, grouped the way the notebook introduced them
    "ladder (section 4)": ("one_shot", "one_shot_sam", "scribbles"),
    "path 1 (section 5)": ("naive", "masked", "pu"),
    "path 2 (section 6)": ("sam_pseudo", "gradcam_pseudo"),
    "path 3 (section 7)": ("sam_clicks", "sam_boxes"),
}
story = [("pets · one box (section 1)", RESULTS["pets_zero_shot_iou"], THEME["works"])]
story.append(
    (f"{SPECIALISED['name']} · one box (section 2)", TABLE["zero_shot"]["iou"], APPROACH_COLOURS["zero_shot"])
)
for path, keys in PATHS.items():
    best = max(keys, key=lambda k: TABLE[k]["iou"])
    story.append((f"best of {path}: {best}", TABLE[best]["iou"], APPROACH_COLOURS[best]))
story.append(("complete masks (section 2)", TABLE["full_masks"]["iou"], APPROACH_COLOURS["full_masks"]))

fig, ax = plt.subplots(figsize=(8, 3.4))
labels, values, colours = zip(*story[::-1], strict=True)
ax.barh(labels, values, color=colours)
for y, v in enumerate(values):
    ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color="#311B92")
ax.axvline(EVERYTHING, color=THEME["background"], ls="--", label='IoU of "everything" (specialised task)')
ax.set_xlim(0, 1.08)
ax.set_xlabel("mean IoU on the held-out images")
ax.set_title("Where it worked, where it fell short, how far each path got back")
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 10.1 The principles, from your numbers
#
# The next cell writes six rules as cards, each with the evidence from your session (a rule with your own
# number attached tells you how strongly to believe it), and says so where a lesson did not hold in your run.

# %%
# @title Principles from this run { display-mode: "form" }
import textwrap


def p_check_first() -> str:
    """Outcome 1: is the foundation model good enough out of the box?"""
    pets, spec = RESULTS["pets_zero_shot_iou"], RESULTS["specialised_zero_shot_iou"]
    rule = "Test the foundation model on your own task first, and read its score against marking everything."
    fact = (
        f"One box scored IoU {pets:.2f} on pets and {spec:.2f} on {SPECIALISED['name']}, where a mask over "
        f"the whole image scores {EVERYTHING:.2f}."
    )
    if spec < EVERYTHING + 0.1:
        return f"{rule} {fact} Out of the box it was close to doing nothing useful here."
    return f"{rule} {fact} It did part of the job out of the box; adapt only if the rest matters."


def p_resolution() -> str:
    """Section 2: the patch is 16 px; does a larger input help?"""
    res = sorted((int(k.split()[0]), v) for k, v in RESULTS["resolution_effect"].items())
    (lo, a), (hi, b) = res[0], res[-1]
    if SPECIALISED.get("thin_structures"):
        rule = "Thin structures defeat patch features: check the resolution first."
    else:
        rule = "Check the resolution first: a patch feature describes a 16 px block, not a pixel."
    fact = (
        f"The same head on complete masks scored IoU {a['iou']:.2f} at {lo} px and {b['iou']:.2f} at {hi} px "
        f"(clDice {a['cldice']:.2f} → {b['cldice']:.2f}); on pets the same kind of head scored "
        f"{RESULTS['pets_head_iou']:.2f}."
    )
    if b["iou"] - a["iou"] < 0.05:
        return f"{rule} {fact} More pixels helped little, so the limit is the patch: plan for cheap labels."
    return f"{rule} {fact} More pixels helped, so try the larger input before anything more expensive."


def p_missing_is_unknown() -> str:
    """Sections 3 and 5: missing labels are unknown, not background."""
    naive, masked = RESULTS["naive_recall_hidden_fg"], RESULTS["masked_recall_hidden_fg"]
    rule = "An unlabelled pixel is unknown, not background: tell the loss so."
    fact = (
        f"On the same half-painted maps, the head that read the gaps as background found {naive:.0%} of the "
        f"hidden foreground, the masked loss {masked:.0%}."
    )
    if masked <= naive:
        return f"{rule} {fact} Here ignoring the gaps did not recover more; check what the strokes cover."
    pn, pm = TABLE["naive"]["precision"], TABLE["masked"]["precision"]
    if pm < pn:
        return f"{rule} {fact} The price was precision ({pn:.2f} → {pm:.2f}): fewer misses, more alarms."
    return f"{rule} {fact} And it kept its precision ({pn:.2f} → {pm:.2f})."


def p_cheapest_baseline() -> str:
    """Section 7: compare SAM with the hints alone."""
    alone = RESULTS["sam_direct_strokes_alone"]
    sam_rows = [k for k in ("zero_shot", "one_shot_sam", "sam_clicks", "sam_boxes") if k in TABLE]
    best = max(sam_rows, key=lambda k: TABLE[k]["iou"])
    gain = TABLE[best]["iou"] - alone["iou"]
    rule = "Compare the foundation model with the cheapest baseline built from the same hints."
    fact = (
        f"The strokes thickened by {alone['thicken_px']} px, no model at all, scored IoU {alone['iou']:.2f}; "
        f"the best SAM recipe, {best}, scored {TABLE[best]['iou']:.2f}."
    )
    if gain <= 0:
        return f"{rule} {fact} The model added nothing the strokes did not already say."
    if gain < 0.05:
        return f"{rule} {fact} The model added only {gain:.2f}."
    return f"{rule} {fact} The model earned its place, by {gain:.2f}."


def p_one_number() -> str:
    """Section 8: no single column decides."""
    rows = RESULTS["comparison_table"]
    by_iou = max(rows, key=lambda r: r["iou"])
    by_recall = max(rows, key=lambda r: r["recall"])
    steady = min(rows, key=lambda r: r["reliability"])
    rule = "Choose by what a mistake costs, not by one number."
    fact = (
        f"Highest IoU: {by_iou['approach']} ({by_iou['iou']:.2f}); highest recall: {by_recall['approach']} "
        f"({by_recall['recall']:.2f}); smallest spread: {steady['approach']} ({steady['reliability']:.2f})."
    )
    leaders = {by_iou["approach"], by_recall["approach"], steady["approach"]}
    if len(leaders) == 1:
        return f"{rule} {fact} This time one approach led all three; check that before assuming it."
    lead = f"{len(leaders)} different approaches lead the three columns"
    return f"{rule} {fact} {lead}, so the right row depends on which mistake costs most."


def p_minutes_before_hours() -> str:
    """Section 8: annotation effort, once and per new image."""
    full = TABLE["full_masks"]
    cheap_rows = [r for r in RESULTS["comparison_table"] if r["effort_min"] < full["effort_min"] / 10]
    cheap = max(cheap_rows or RESULTS["comparison_table"], key=lambda r: r["iou"])
    per_image = max(RESULTS["comparison_table"], key=lambda r: r["effort_min_per_new_image"])
    rule = "Spend minutes before hours: start with strokes, and buy complete masks only if the table says so."
    fact = (
        f"Complete masks cost {full['effort_min']:.0f} min of annotation for IoU {full['iou']:.2f}; "
        f"{cheap['approach']} needed {cheap['effort_min']:.1f} min for IoU {cheap['iou']:.2f}."
    )
    if cheap["iou"] >= full["iou"]:
        verdict = "The cheap labels matched or beat the expensive ones."
    else:
        ratio = full["effort_min"] / max(cheap["effort_min"], 0.01)
        verdict = f"The hours bought {full['iou'] - cheap['iou']:.2f} IoU for {ratio:.0f}× the effort."
    scale = (
        f"And count per new image: {per_image['approach']} costs "
        f"{per_image['effort_min_per_new_image']:.2f} min on every one, a trained head none."
    )
    return f"{rule} {fact} {verdict} {scale}"


PRINCIPLES = [
    p_check_first(),
    p_resolution(),
    p_missing_is_unknown(),
    p_cheapest_baseline(),
    p_one_number(),
    p_minutes_before_hours(),
]
RESULTS["principles"] = PRINCIPLES
headline = f"Principles from this session ({SPECIALISED['name']}, {len(SPEC_EVAL)} held-out images, {DEVICE})"
try:
    from html import escape

    from IPython.display import HTML, display

    cards = ""
    for n, text in enumerate(PRINCIPLES, start=1):
        rule, _, evidence = text.partition(". ")  # the rule is the first sentence, the rest is the evidence
        cards += (
            f'<div style="flex:1 1 300px;background:#EDE7F6;border-left:6px solid {THEME["works"]};'
            'border-radius:6px;padding:10px 12px;color:#311B92;line-height:1.45">'
            f'<div style="font-weight:bold;font-size:15px">{n}. {escape(rule.rstrip("."))}.</div>'
            f'<div style="font-size:12px;margin-top:5px;color:#4527A0">{escape(evidence)}</div></div>'
        )
    display(
        HTML(
            f'<div style="font-weight:bold;color:#311B92;margin:4px 0">{escape(headline)}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px">{cards}</div>'
        )
    )
except ImportError:  # no IPython: plain text
    print(headline + ":\n")
    for n, text in enumerate(PRINCIPLES, start=1):
        print(textwrap.fill(text, width=100, initial_indent=f"{n}. ", subsequent_indent="   "), end="\n\n")

# %% [markdown]
# <!-- callout: your-turn | Your turn (2 min) -->
# **Q36.** 🟢 Think of one segmentation or detection problem of your own (work, research, or the images of
# section 9). Which of the principles above would have changed what you did first, and what number would you
# measure to check it?
#
# - Write the problem and your first step on Monday, as a rule with a number, in the next cell.
# - The cell prints them next to the principle you picked and checks nothing.

# %%
# Q36 — your problem and your first step (leave "" to skip)
MY_PROBLEM = ""  # e.g. "cracks in drone photos of bridges"
MY_PRINCIPLE = 0  # the number of the principle above that changes your plan (1–6), or 0 for none
MY_FIRST_STEP = ""  # e.g. "one box on 20 images, compared with the 'everything' line"
if not isinstance(MY_PRINCIPLE, int) or not 0 <= MY_PRINCIPLE <= len(PRINCIPLES):
    raise SystemExit(f"MY_PRINCIPLE must be a whole number from 0 to {len(PRINCIPLES)}.")
if not (isinstance(MY_PROBLEM, str) and isinstance(MY_FIRST_STEP, str)):
    raise SystemExit('MY_PROBLEM and MY_FIRST_STEP must be text in quotes, e.g. "my problem".')
if MY_PROBLEM or MY_FIRST_STEP:
    print(f"problem: {MY_PROBLEM or '(not written)'}\nfirst step: {MY_FIRST_STEP or '(not written)'}")
    if MY_PRINCIPLE:
        print(textwrap.fill(PRINCIPLES[MY_PRINCIPLE - 1], width=100, initial_indent="because: "))
else:
    print("nothing written; fine. Keep the question for the train home: what would you measure first?")

# %% [markdown]
# <details><summary>A possible answer</summary>
#
# Take cracks in drone photos of bridges. They are thin, like vessels, so principle 2 changes the plan.
#
# 1. Run one box and one feature head on ten images at two input sizes.
# 2. See whether the larger size moves clDice at all.
# 3. If it does not, the patch is the limit: spend the first afternoon on strokes for twenty images, not
#    on a bigger model.
#
# The number to measure is the gap between the head and the "everything" line on a small test set you
# label completely, a few images, once. A different problem gives a different first step; what carries
# over is writing the step with the number that would prove it wrong.
#
# </details>

# %% [markdown]
# <!-- callout: question | Show of hands -->
# - Which principle would you have got wrong this morning, before section 1?
# - Which one would you most like to check on your own data next week?

# %% [markdown]
# ### 10.2 What you can do now

# %% [markdown]
# At the end you can:
#
# 1. **Judge when a vision foundation model is good enough out of the box**, and recognise the signs that it
#    is not (sections 1 and 2).
# 2. **Adapt it with cheap, incomplete supervision**, and explain what each approach uses and assumes
#    (sections 3 to 7).
# 3. **Compare approaches** on accuracy, annotation effort, reliability and compute, and decide what to do
#    next on a new problem, including your own images (sections 8 and 9).

# %% [markdown]
# <!-- callout: takeaway | When you see X, do Y: the whole workshop -->
# - **When** one box scores about the same as "mark everything", or the structure is thinner than a 16 px
#   patch **→** compare with the "everything" line, trust clDice over IoU, and try a larger input
#   (sections 1 and 2).
# - **When** labels are partly missing, or you only have strokes **→** treat an unlabelled pixel as unknown,
#   turn strokes into pseudo-labels or SAM prompts, and compare with the thickened strokes alone
#   (sections 3 and 5 to 7).
# - **When** two approaches differ by less than their spread, or a method needs strokes on every one of
#   thousands of new images **→** call it a tie and choose by annotation minutes; train a head once
#   (section 8).
# - **When** there is no ground truth **→** look where the approaches disagree, add a stroke there, and
#   label a small test set before deciding (section 9).
#
# The same six questions carry over to any task:
#
# - Is the out-of-the-box result above the "everything" line?
# - Is the structure smaller than a 16 px patch?
# - What do the gaps in the labels mean?
# - Does the model beat the hints alone?
# - Which mistake costs most?
# - How many minutes does every new image cost?

# %% [markdown]
# ### 10.3 Where to go next
#
# - **Label a small test set first**: a few fully annotated images turn every later decision into a
#   measurement, as the hidden ground truth did here.
# - **Let the model choose the next stroke**: draw where the approaches disagree (section 9), retrain, repeat.
# - **A light decoder instead of a linear head**: a small upsampling decoder on the same frozen features
#   comes before any fine-tuning.
# - **The 🔴 parts you skipped**: PU loss (5), Grad-CAM peaks as clicks (6), SAM 3 from a word (7), extra
#   strokes (9).
# - **Sources**: [DINOv3](https://github.com/facebookresearch/dinov3) ·
#   [SAM 2](https://github.com/facebookresearch/sam2) ·
#   [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

# %% [markdown]
# <!-- callout: across | Across approaches, one last time -->
# The question on the first screen was what modern AI solves out of the box, when its limits start to
# matter, and what to do next.
#
# Your session's answer is in the six principles above, with numbers. The foundation model was a starting
# point, not an answer, and every way out paid for its gain in a different currency: minutes of strokes, a
# wrong assumption about the gaps, a second model's mistakes, or a person pointing at every image.
#
# On your next problem, measure first, spend minutes before hours, and keep the table. Thank you for taking
# part.

# %%
# @title Credits: models and datasets of this session { display-mode: "form" }
cfg = SPECIALISED
print(f"DINOv3 ViT-S/16 · {DINO_SOURCE} · {DINO_LICENCE}")
print(f"SAM 2.1 hiera-small · {SAM_ID} · {SAM_LICENCE}")
print(f"{PETS['name']} · {PETS['licence']}")
print(f"{cfg['name']} · {cfg['licence']} ({cfg['licence_url']})")
print(textwrap.fill(cfg["citation"], width=100, initial_indent="  ", subsequent_indent="  "))
RESULTS["credits"] = {
    "models": {DINO_SOURCE: DINO_LICENCE, SAM_ID: SAM_LICENCE},
    "datasets": {PETS["name"]: PETS["licence"], SPECIALISED["name"]: SPECIALISED["licence"]},
}

# %% [markdown]
# <!-- callout: go-deeper | Credits & licences -->
# - **DINOv3** ViT-S/16 (Meta), via the `timm` re-host
#   [timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m).
# - DINOv3 is licensed under the
#   [DINOv3 License](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), Copyright © Meta
#   Platforms, Inc. All Rights Reserved. It allows research and commercial use under Meta's conditions and
#   acceptable-use policy; keep this notice if you redistribute the weights or anything derived from them.
#   SAM 2.1 is Apache-2.0.
# - **SAM 2.1** hiera-small (Meta),
#   [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small). SAM 3, if you used
#   it, comes under Meta's own SAM 3 licence, which you accepted on Hugging Face.
# - **Oxford-IIIT Pet**: Parkhi, Vedaldi, Zisserman and Jawahar, *Cats and Dogs*, CVPR 2012,
#   [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The subset shipped with this workshop
#   is resized and shared under the same licence.
# - **pytorch-grad-cam** (Jacob Gildenblat and contributors), MIT.
# - The specialised dataset is printed by the credits cell above, read from the dataset config.
# - Every derived file (scribbles, hidden labels, pseudo-labels, features) carries its source's licence.
#   The full list is in `data/ATTRIBUTION.md`.
