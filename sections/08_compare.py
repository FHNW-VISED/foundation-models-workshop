# %% [markdown]
# ## 8. Compare everything
#
# Every section since 2 ended with its own small table, each against a different neighbour. Here they all
# sit in one table, scored on the same held-out images against the same hidden ground truth.
#
# The question becomes the one you face on a real project: not "which approach has the highest number" but
# "which approach would I pay for". A recipe can win on accuracy and lose on annotation hours, or win on
# average and fail badly on one image in five.

# %% [markdown]
# <!-- callout: objective | Objective -->
# fill the comparison table, predicting the winner first.

# %% [markdown]
# <!-- callout: theory | Theory: the five columns -->
# Each row of the table is read through five columns:
#
# - **Accuracy**: mean IoU over the held-out images (0 = no overlap with the truth, 1 = perfect), plus
#   clDice (does the mask follow the vessel centre lines?).
# - **Annotation effort**: what a person had to draw, in minutes. A stroke or a box is seconds, a complete
#   vessel map is hours.
# - **Reliability, or spread**: the standard deviation of the per-image IoU, i.e. how far a typical image
#   sits from the average. IoUs of 0.60, 0.62, 0.58 give a spread of 0.02: trust the average. IoUs of
#   0.90, 0.30, 0.60 have the same average but a spread of 0.25: some images fail badly.
# - **Compute**: seconds of training plus inference in this session.
# - **Scalability**: a sentence, not a number. What does the next 1,000 images cost?
#
# The table also shows **recall** (the share of the true vessel pixels the mask finds) and **worst** (the
# lowest IoU on any single image).

# %% [markdown]
# <!-- callout: note | Assumptions behind the minutes -->
# - A scribble stroke, a click or a box takes about **2 seconds**.
# - A complete vessel map of one fundus image takes **3 to 5 hours** (FIVES paper, 2022). We count the
#   midpoint, 4 hours, and a half-painted map as half of that.
# - Both numbers come from the dataset config, not from a measurement in this room. Change them there and
#   every row moves.
# - Training minutes are paid once. The per-image column is what every *new* image costs.

# %% [markdown]
# This helper cell collects each approach's held-out predictions, numbers, seconds and annotation minutes,
# and gives each one a sentence on how it scales.

# %%
# @title Comparison helpers: every approach, its predictions, seconds and effort { display-mode: "form" }
APPROACH_COLOURS |= {"full_masks": THEME["works"]}
APPROACH_NAMES |= {"full_masks": "full masks: complete maps (section 2)"}
COMPARE_ORDER = [
    "zero_shot",
    "one_shot",
    "one_shot_sam",
    "scribbles",
    "naive",
    "masked",
    "pu",
    "sam_pseudo",
    "gradcam_pseudo",
    "sam_clicks",
    "sam_boxes",
    "full_masks",
]
COMPARE_REPORTS = (  # mean IoU, clDice, recall, precision, marked share and held-out predictions per approach
    {k: ladder_report[k] | {"held_preds": ladder_preds[k]} for k in ladder_report}
    | {k: path_reports[APPROACH_NAMES[k]] for k in ("naive", "masked", "pu")}
    | {k: path2_report[k] | {"held_preds": path2_preds[k]} for k in ("sam_pseudo", "gradcam_pseudo")}
    | {k: path3_report[k] | {"held_preds": path3_preds[k]} for k in ("sam_clicks", "sam_boxes")}
    | {"full_masks": path_reports["full masks (section 2)"]}
)
COMPARE_SECONDS = {  # training + pseudo-labels + inference, as recorded by each section in this session
    "zero_shot": RESULTS["specialised_zero_shot_seconds"] / len(LADDER),  # the box: 1 of section 2's prompts
    "one_shot": proto_sec,
    "one_shot_sam": proto_sec + proto_sam_sec,
    "full_masks": spec_heads[side][1] + RESULTS["scribbles_head"]["infer_sec"],  # same head, same features
} | {
    k: sum(float(RESULTS[entry].get(part, 0.0)) for part in ("train_sec", "pseudo_sec", "infer_sec"))
    for k, entry in {
        "scribbles": "scribbles_head",
        "naive": "naive_head",
        "masked": "masked_head",
        "pu": "pu_head",
        "sam_pseudo": "sam_pseudo_head",
        "gradcam_pseudo": "gradcam_head",
        "sam_clicks": "sam_direct_clicks",
        "sam_boxes": "sam_direct_boxes",
    }.items()
}
COMPARE_EFFORT = (  # annotation minutes for this run (section 4's effort_minutes and its assumptions)
    RESULTS["ladder_effort_min"]
    | RESULTS["path1_effort_min"]
    | RESULTS["path2_effort_min"]
    | RESULTS["path3_effort_min"]
    | {"full_masks": round(effort_minutes(n_full_masks=len(spec_train)), 2)}
)
COMPARE_PER_IMAGE = {  # annotation minutes for every *new* image; trained heads need none
    k: 0.0 for k in COMPARE_ORDER
} | {
    "zero_shot": effort_minutes(n_strokes=1),  # one box
    "sam_clicks": RESULTS["sam_direct_clicks"]["effort_min_per_image"],
    "sam_boxes": RESULTS["sam_direct_boxes"]["effort_min_per_image"],
}
SCALES = {
    "zero_shot": "a box on every new image, SAM per image, nothing to train",
    "one_shot": "one complete mask once; new images cost one feature pass",
    "one_shot_sam": "one complete mask once; SAM per new image, clicks picked automatically",
    "scribbles": "minutes of strokes once; new images cost one feature pass",
    "naive": "half-painted maps once (hours); new images cost one feature pass",
    "masked": "half-painted maps plus a few strokes once; new images cost one feature pass",
    "pu": "half-painted maps once plus a guessed share; new images cost one feature pass",
    "sam_pseudo": "strokes once, SAM at training time only; new images cost one feature pass",
    "gradcam_pseudo": "strokes once, a small classifier at training time; new images cost one feature pass",
    "sam_clicks": "strokes on every new image, SAM per image; the clicking never stops",
    "sam_boxes": "strokes on every new image, SAM per box; the clicking never stops",
    "full_masks": "hours of painting per training image; new images cost one feature pass",
}
WITH_CLDICE = "cldice" in SPECIALISED.get("metric_extras", [])
print(f"{len(COMPARE_ORDER)} approaches, each scored on the same {len(SPEC_EVAL)} held-out images:")
for k in COMPARE_ORDER:
    print(f"  {k:<16}{APPROACH_NAMES[k]}")
cost = SPECIALISED["annotation_effort"]
print(
    f"effort assumptions ({SPECIALISED['name']}): {cost['seconds_per_stroke']} s per stroke, click or box · "
    f"{cost['hours_per_full_mask'][0]}–{cost['hours_per_full_mask'][-1]} h per complete mask "
    f"(we count {np.mean(cost['hours_per_full_mask']):.0f} h)"
)

# %% [markdown]
# ### 8.1 Look first, then predict
#
# This cell shows all twelve approaches on the same held-out image, next to the image and its hidden
# ground truth (green). Look for who finds the thin vessels, who paints blobs, and who misses most.

# %%
# @title All twelve approaches on one held-out image { display-mode: "form" }
k_show = 0  # which held-out image to look at
it_show = SPEC_EVAL[k_show]
gallery = [
    (f"{it_show['name']} · image", it_show["image"]),
    ("ground truth (scoring only)", overlay(it_show["image"], it_show["mask"], THEME["truth"])),
]
for k in COMPARE_ORDER:
    pred = COMPARE_REPORTS[k]["held_preds"][k_show] & region(it_show)
    gallery.append((k, overlay(it_show["image"], pred, APPROACH_COLOURS[k])))
g_cols = 5
g_rows = -(-len(gallery) // g_cols)
fig, axes = plt.subplots(g_rows, g_cols, figsize=(2.6 * g_cols, 2.6 * g_rows))
for ax in np.ravel(axes):
    ax.axis("off")
for ax, (title, img) in zip(np.ravel(axes), gallery, strict=False):
    ax.imshow(img)
    ax.set_title(title, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: your-turn | Your prediction (1 min) -->
# Before the table appears, write down two guesses, using the short names above.
#
# - Which approach will have the **highest mean IoU**?
# - Which will be the **steadiest**, with the smallest spread of IoU from image to image?
#
# You have seen these numbers one section at a time. Does your picture of the whole hold up when they sit
# side by side?

# %% [markdown]
# **Q31.** 🟡 Which approach wins on IoU, and which is the steadiest?
#
# - Write your two picks in the next cell and run it.
# - The table cell further down tells you where they landed.
# - The cell only checks that your picks are names from the list.

# %%
# Q31 — your two predictions (short names from the list above, or leave "" to skip)
MY_WINNER = ""  # e.g. "scribbles"
MY_STEADIEST = ""  # e.g. "full_masks"
for label, pick in (("MY_WINNER", MY_WINNER), ("MY_STEADIEST", MY_STEADIEST)):
    if not isinstance(pick, str) or (pick and pick not in COMPARE_ORDER):
        raise SystemExit(f'{label} must be "" or one of: {", ".join(COMPARE_ORDER)}.')
print(f"your picks · highest IoU: {MY_WINNER or '(none)'} · steadiest: {MY_STEADIEST or '(none)'}")

# %% [markdown]
# ### 8.2 What "reliable" means here

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min) -->
# Fill `my_spread`.
#
# - **In:** the IoU of one approach on each held-out image, a list of floats.
# - **Out:** their standard deviation (divide by the number of images, not by one less), a float.
# - **Example:** `[0.1, 0.3, 0.2, 0.6]` has mean 0.3 and spread 0.19.
#
# The cell checks your version on those four play values and continues with ours if you skip it.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_spread(per_image_iou: list[float]) -> float:
    """Reference: numpy's standard deviation, which divides by n by default."""
    return float(np.std(np.asarray(per_image_iou, dtype=float)))


# %%
# ✍️ YOUR CODE
def my_spread(per_image_iou: list[float]) -> float:
    """The population standard deviation of the per-image IoUs: sqrt(mean((x - mean(x))²))."""
    ...


_play_ious = [0.1, 0.3, 0.2, 0.6]  # four images: three middling, one much better
spread = use(my_spread, reference=_ref_spread, check=lambda f: round(float(f(_play_ious)), 4), name="spread")
print("play IoUs", _play_ious, "· spread", round(spread(_play_ious), 4))

# %% [markdown]
# ### 8.3 The table
#
# This cell scores every approach's held-out masks image by image against the hidden ground truth and fills
# one row per approach. It prints the table sorted by IoU and tells you where your two predictions landed.

# %%
# @title Score every approach and print the table { display-mode: "form" }
comparison = []
PER_IMAGE_IOU = {}  # approach -> its IoU on each held-out image (drawn as violins below)
for k in COMPARE_ORDER:
    rep = COMPARE_REPORTS[k]
    pairs = zip(rep["held_preds"], SPEC_EVAL, strict=True)
    per_image = [iou(p & region(it), it["mask"] & region(it)) for p, it in pairs]
    PER_IMAGE_IOU[k] = per_image
    comparison.append(
        {
            "approach": k,
            "name": APPROACH_NAMES[k],
            "iou": float(np.mean(per_image)),
            **({"cldice": float(rep["cldice"])} if WITH_CLDICE else {}),
            "recall": float(rep["recall"]),
            "precision": float(rep["precision"]),
            "effort_min": float(COMPARE_EFFORT[k]),
            "effort_min_per_new_image": round(float(COMPARE_PER_IMAGE[k]), 2),
            "reliability": float(spread(per_image)),
            "worst_iou": float(np.min(per_image)),
            "seconds": round(float(COMPARE_SECONDS[k]), 1),
            "scalability": SCALES[k],
        }
    )
RESULTS["comparison_table"] = comparison
RESULTS["comparison_assumptions"] = dict(SPECIALISED["annotation_effort"]) | {
    "reliability": f"standard deviation of the IoU over {len(SPEC_EVAL)} held-out images",
    "seconds": f"training + pseudo-labels + inference in this session, on {DEVICE}",
}

ranked = sorted(comparison, key=lambda r: r["iou"], reverse=True)
extra = f"{'clDice':>8}" if WITH_CLDICE else ""
print(
    f"{'approach':<46}{'IoU':>6}{extra}{'recall':>8}{'± spread':>10}{'worst':>7}{'effort min':>12}"
    f"{'/new img':>10}{'sec':>7}"
)
for r in ranked:
    cl = f"{r['cldice']:>8.3f}" if WITH_CLDICE else ""
    print(
        f"{r['name']:<46}{r['iou']:>6.3f}{cl}{r['recall']:>8.3f}{r['reliability']:>10.3f}"
        f"{r['worst_iou']:>7.3f}{r['effort_min']:>12.1f}{r['effort_min_per_new_image']:>10.2f}"
        f"{r['seconds']:>7.1f}"
    )
print(
    f"{'no model: strokes thickened (section 7)':<46}{RESULTS['sam_direct_strokes_alone']['iou']:>6.3f}\n"
    f"{'everything is foreground':<46}{RESULTS['specialised_everything_iou']:>6.3f}\n"
)
print("recall = share of true vessel pixels found · ± spread = std of the per-image IoU · worst = lowest")
print("per-image IoU · effort min = annotation minutes once · /new img = annotation minutes per new image\n")
print("how each one scales:")
for r in ranked:
    print(f"  {r['approach']:<16}{r['scalability']}")

winner = ranked[0]["approach"]
steadiest = min(comparison, key=lambda r: r["reliability"])["approach"]
by_spread = [r["approach"] for r in sorted(comparison, key=lambda r: r["reliability"])]
print(f"\nhighest IoU: {winner} · steadiest: {steadiest}")
if MY_WINNER:
    rank = [r["approach"] for r in ranked].index(MY_WINNER) + 1
    print(f"your IoU pick {MY_WINNER} ranks {rank} of {len(ranked)}")
if MY_STEADIEST:
    rank = by_spread.index(MY_STEADIEST) + 1
    print(f"your steadiness pick {MY_STEADIEST} ranks {rank} of {len(by_spread)}")
if not (MY_WINNER or MY_STEADIEST):
    print("no prediction written in Q31; fine, but next time guess first: it is the cheapest lesson here")

# %% [markdown]
# The spread column hides the shape. This cell draws, for each approach, one dot per held-out image and a
# violin around the dots: a wide violin means the IoU varies a lot from image to image. The short bar is the
# median image.

# %%
# @title Violin plot: each approach's IoU over the held-out images { display-mode: "form" }
fig, ax = plt.subplots(figsize=(10, 3.8))
for x, r in enumerate(ranked):
    vals = np.asarray(PER_IMAGE_IOU[r["approach"]], float)
    colour = APPROACH_COLOURS[r["approach"]]
    if len(vals) >= 2 and np.ptp(vals) > 1e-9:  # a violin needs some spread to draw
        body = ax.violinplot(vals, positions=[x], widths=0.8, showextrema=False)["bodies"][0]
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.3)
    ax.scatter(np.full(len(vals), x), vals, s=12, color=colour, zorder=3)
    ax.plot([x - 0.3, x + 0.3], [np.median(vals)] * 2, color=colour, lw=2.5, zorder=4)
line_all = RESULTS["specialised_everything_iou"]
ax.axhline(line_all, color=THEME["background"], ls="--", label='mean IoU of "everything"')
ax.set_xticks(range(len(ranked)), [r["approach"] for r in ranked], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("IoU on one held-out image")
ax.set_title("Per-image IoU, best mean on the left (dot = one image, bar = median)")
ax.legend(fontsize=7, frameon=False, loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# The same idea on real images. This cell takes the approach with the highest mean IoU and shows its
# **worst**, **typical** (median) and **best** held-out image. Violet = vessels found, orange = false
# alarms, green = vessels missed.

# %%
# @title Worst, typical and best image of the IoU winner { display-mode: "form" }
w_iou = np.asarray(PER_IMAGE_IOU[winner], float)
w_order = np.argsort(w_iou)
panels = []
w_picks = (("worst", w_order[0]), ("typical (median)", w_order[len(w_order) // 2]), ("best", w_order[-1]))
for tag, n in w_picks:
    it_n, pred_n = SPEC_EVAL[n], COMPARE_REPORTS[winner]["held_preds"][n]
    panels.append((f"{winner} · {tag} · IoU {w_iou[n]:.2f}", error_map(it_n["image"], pred_n, it_n)))
show_masks(panels, width=3.4)

# %% [markdown]
# ### 8.4 The trade-off in one picture
#
# This cell draws the table. On the left, IoU against annotation minutes (log scale, because a box and a
# complete mask differ by a factor of thousands), with a bar for each approach's spread. On the right, the
# seconds of compute. One colour per approach, as in every earlier figure.

# %%
# @title Trade-off plot: accuracy against annotation, and compute { display-mode: "form" }
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
top = min(1.0, 1.25 * max(r["iou"] + r["reliability"] for r in comparison) + 0.05)
gap = 0.045 * top  # labels of approaches with about the same effort are stacked at least this far apart
groups, last = [], None
for r in sorted(comparison, key=lambda r: r["effort_min"]):
    lx = np.log10(max(r["effort_min"], 0.01))
    if last is None or lx - last > 0.35:
        groups.append([])
    groups[-1].append(r)
    last = lx
label_y = {}
for group in groups:
    y_prev = -1.0
    for r in sorted(group, key=lambda r: r["iou"]):
        label_y[r["approach"]] = y_prev = max(r["iou"], y_prev + gap)
for r in comparison:
    x, colour = max(r["effort_min"], 0.01), APPROACH_COLOURS[r["approach"]]
    ax.errorbar(x, r["iou"], yerr=r["reliability"], fmt="o", ms=7, capsize=3, color=colour)
    ax.annotate(
        r["approach"],
        (x, r["iou"]),
        xytext=(x * 1.25, label_y[r["approach"]]),
        fontsize=7,
        color=colour,
        va="center",
    )
ax.axhline(
    RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--", label='IoU of "everything"'
)
ax.set_xscale("log")
ax.set_xlim(right=ax.get_xlim()[1] * 4)  # room for the labels of the right-most points
ax.set_xlabel("annotation effort, ≈ minutes (log scale)")
ax.set_ylabel("mean IoU on the hidden ground truth (± spread)")
ax.set_ylim(0, top)
ax.set_title("Accuracy against annotation")
ax.legend(fontsize=7, frameon=False, loc="upper left")
names = [r["approach"] for r in ranked][::-1]
bx.barh(names, [r["seconds"] for r in ranked][::-1], color=[APPROACH_COLOURS[k] for k in names])
bx.set_xlabel(f"seconds, training + inference ({DEVICE})")
bx.tick_params(axis="y", labelsize=7)
bx.set_title("Compute (sorted by IoU, best at the top)")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: go-deeper | How to read the trade-off plot -->
# - **Top-left** would be the dream: accurate and nearly free. On thin structures nobody is there.
# - **Bottom-left** is cheap and weak, often the prompt-only approaches.
# - **Far right** is the complete-mask head. Its minutes are hours, and the log axis hides how far away it is.
# - **Same x position** is the fairest comparison: same annotation, different use of it (the scribble head,
#   the two pseudo-label heads and the direct SAM recipes all start from the same kind of strokes).
# - **Vertical bars**: a slightly lower mean with a much smaller spread can be the safer choice.
# - **Right panel**: seconds matter little for the heads and a lot for anything that runs SAM on every image.

# %% [markdown]
# **Q32.** 🟡 Pick one approach for each situation and name the column that decided it.
#
# - (a) A screening study of 1,000 new images next month.
# - (b) One difficult image a clinician needs today.
# - (c) A project where a missed vessel is much worse than a false alarm.
#
# Columns you can use: IoU, recall, spread, worst, effort min, /new img. Did the IoU winner win any of them?
#
# <details><summary>A possible answer</summary>
#
# - (a) A thousand images make the **/new img** column decisive. Strokes or a box on every image multiply
#   their minutes by a thousand; a trained head pays its annotation once and then costs a feature pass per
#   image. Among the heads, pick by IoU and spread.
# - (b) For one image today the per-image rows are the natural candidates: no training, a minute of
#   strokes, and you check the result yourself. Compare them with the thickened-strokes line before
#   trusting SAM's share of the work.
# - (c) When misses cost more than false alarms, IoU is the wrong column. Look at **recall** and at
#   **worst**, and accept a head that over-marks if a person checks its output.
#
# The IoU winner can win none of the three. That is the point of a table with more than one column.
#
# </details>

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min: compare tables with your neighbour -->
# Your sessions ran with the same seed but possibly different settings (Q17, Q23, Q27, Q28, the 🔴 parts),
# and a CPU session has different image counts from a GPU one. Put your two tables side by side.
#
# - Which rows moved, and which stayed put?
# - Did you predict the same winner?
# - Agree on one approach you would try first on a new specialised task, and one sentence on why.

# %% [markdown]
# <details><summary>What pairs usually find</summary>
#
# - The order of the top few rows is less stable than it looks. A different threshold, number of clicks or
#   set of evaluation images can swap neighbours whose IoUs differ by a few hundredths. The spread column
#   says those differences are inside the noise.
# - The shape of the plot rarely moves: prompt-only rows stay cheap per image but weak or unsteady, the
#   complete-mask head stays hours away on the effort axis, and the stroke-based heads sit together in the
#   middle.
# - When two tables disagree on the winner but agree on the shape, trust the shape.
#
# </details>

# %% [markdown]
# <!-- callout: takeaway | When you see X, do Y: choosing an approach -->
# - **When** two approaches differ by less than their spread **→** call it a tie and pick the cheaper one.
# - **When** the best approach needs strokes or a box on every new image and you have thousands of images
#   **→** compare the /new img column first; a trained head pays its annotation once.
# - **When** a missed structure costs more than a false alarm **→** rank by recall and worst image, not by
#   mean IoU.
# - **When** the IoU winner is also the most expensive row **→** ask what the next-cheapest row loses
#   before you pay for it.
#
# Section 9 runs the cheapest rungs on your own images; section 10 turns this table into rules.
