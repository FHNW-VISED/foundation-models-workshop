# %% [markdown]
# ## 8. Compare everything
#
# Every section since 2 ended with its own small table, each against a different neighbour. Here they all
# sit in one table, scored on the same held-out images against the same hidden ground truth.
#
# The question is the one you face on a real project: not "which approach has the highest number" but
# "which approach would I pay for". A recipe can win on accuracy and lose on annotation hours, or win on
# average and fail badly on one image in five. Three pictures follow the table: a trade-off plot, a strip
# plot of the score on every image, and the winner's best and worst images.

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
# - **Compute**: seconds of training plus inference in this session (in the plot of 8.3).
# - **Scalability**: a sentence, not a number. What does the next 1,000 images cost?
#
# The table also shows **recall** (the share of the true vessel pixels the mask finds); the scalability
# sentences follow in a list under it.

# %% [markdown]
# <!-- callout: note | Assumptions behind the minutes -->
# - A scribble stroke, a click or a box takes about **2 seconds**.
# - A complete vessel map of one fundus image takes **3 to 5 hours** (FIVES paper, 2022). We count the
#   midpoint, 4 hours, and a half-painted map as half of that.
# - Both numbers come from the dataset config, not from a measurement in this room. Change them there and
#   every row moves.
# - Training minutes are paid once. The per-image column is what every *new* image costs.

# %% [markdown]
# This helper cell gathers every approach's predictions, seconds, annotation minutes and scaling note.

# %%
# @title Comparison helpers: every approach, its predictions, seconds and effort { display-mode: "form" }
approach_colours |= {"full_masks": THEME["works"]}
approach_names |= {"full_masks": "full masks: complete maps (section 2)"}
ALL_APPROACHES = [  # the approaches Q19 accepts (by their readable names in COMPARE_LABEL)
    "zero_shot",
    "one_shot",
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
COMPARE_LABEL = {  # readable names for figure titles and the table
    "zero_shot": "zero-shot box",
    "one_shot": "one-shot (go-deeper)",
    "scribbles": "scribble head",
    "naive": "naive (gaps = bg)",
    "masked": "masked loss",
    "pu": "PU loss",
    "sam_pseudo": "SAM pseudo-labels",
    "gradcam_pseudo": "Grad-CAM labels",
    "sam_clicks": "SAM from clicks",
    "sam_boxes": "SAM from boxes",
    "full_masks": "full masks",
    "sam3_text": "SAM 3 from a word",
}
# An optional extra row: section 7's SAM 3 phrase, only if that cell ran (it needs SAM 3 access).
has_sam3 = (
    isinstance(RESULTS.get("sam3_text"), dict)
    and "sam3_preds" in globals()
    and len(sam3_preds) == len(spec_eval)
)
compare_order = [*ALL_APPROACHES, *(["sam3_text"] if has_sam3 else [])]
compare_reports = (  # mean IoU, clDice, recall, precision, marked share and held-out predictions per approach
    {k: ladder_report[k] | {"held_preds": ladder_preds[k]} for k in ("zero_shot", "one_shot", "scribbles")}
    | {k: path_reports[approach_names[k]] for k in ("naive", "masked", "pu")}
    | {k: path2_report[k] | {"held_preds": path2_preds[k]} for k in ("sam_pseudo", "gradcam_pseudo")}
    | {k: path3_report[k] | {"held_preds": path3_preds[k]} for k in ("sam_clicks", "sam_boxes")}
    | {"full_masks": path_reports["full masks (section 2)"]}
    | ({"sam3_text": RESULTS["sam3_text"] | {"held_preds": sam3_preds}} if has_sam3 else {})
)
compare_seconds = {  # training + pseudo-labels + inference, as recorded by each section in this session
    "zero_shot": RESULTS["specialised_zero_shot_seconds"] / len(LADDER),  # the box: 1 of section 2's prompts
    "one_shot": RESULTS["one_shot"]["infer_sec"],
    "full_masks": spec_heads[side][1] + RESULTS["scribbles_head"]["infer_sec"],  # same head, same features
} | ({"sam3_text": RESULTS["sam3_text"]["infer_sec"]} if has_sam3 else {}) | {
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
compare_effort = (  # annotation minutes for this run (section 4's effort_minutes and its assumptions)
    RESULTS["ladder_effort_min"]
    | RESULTS["path1_effort_min"]
    | RESULTS["path2_effort_min"]
    | RESULTS["path3_effort_min"]
    | {"full_masks": round(effort_minutes(n_full_masks=len(spec_train)), 2)}
    | {"sam3_text": 0.0}  # a phrase typed once; no strokes
)
compare_per_image = {  # annotation minutes for every *new* image; trained heads need none
    k: 0.0 for k in compare_order
} | {
    "zero_shot": effort_minutes(n_strokes=1),  # one box
    "sam_clicks": RESULTS["sam_direct_clicks"]["effort_min_per_image"],
    "sam_boxes": RESULTS["sam_direct_boxes"]["effort_min_per_image"],
}
SCALES = {
    "zero_shot": "a box on every new image, SAM per image, nothing to train",
    "one_shot": "one complete mask once (go-deeper); new images cost one feature pass",
    "scribbles": "minutes of strokes once; new images cost one feature pass",
    "naive": "half-painted maps once (hours); new images cost one feature pass",
    "masked": "half-painted maps plus a few strokes once; new images cost one feature pass",
    "pu": "half-painted maps once plus a guessed share; new images cost one feature pass",
    "sam_pseudo": "strokes once, SAM at training time only; new images cost one feature pass",
    "gradcam_pseudo": "strokes once, a small classifier at training time; new images cost one feature pass",
    "sam_clicks": "strokes on every new image, SAM per image; the clicking never stops",
    "sam_boxes": "strokes on every new image, SAM per box; the clicking never stops",
    "full_masks": "hours of painting per training image; new images cost one feature pass",
    "sam3_text": "one phrase for every image, SAM 3 per image; no strokes at all",
}
use_cldice = "cldice" in specialised.get("metric_extras", [])
print(f"{len(compare_order)} approaches, each scored on the same {len(spec_eval)} held-out images:")
for k in compare_order:
    print(f"  {k:<16}{approach_names[k]}")
cost = specialised["annotation_effort"]
print(
    f"effort assumptions ({specialised['name']}): {cost['seconds_per_stroke']} s per stroke, click or box · "
    f"{cost['hours_per_full_mask'][0]}–{cost['hours_per_full_mask'][-1]} h per complete mask "
    f"(we count {np.mean(cost['hours_per_full_mask']):.0f} h)"
)

# %% [markdown]
# ### 8.1 Look first, then predict
#
# This cell shows every approach on one held-out image, one row per family. The first row is the photo, its
# hidden ground truth and the complete-mask head; the other rows are the approaches of sections 4 to 7.
# `SHOW_IMAGE` chooses which held-out image it shows: 0 is the first, the highest number the last.
#
# What to look for: who follows the thin vessels, who paints blobs, who marks almost nothing. In every
# approach's picture
# $\color{#6A1B9A}{\blacksquare}$ violet is a vessel it found, $\color{#E65100}{\blacksquare}$ orange is a
# mark that is not a vessel, and $\color{#2E7D32}{\blacksquare}$ green is a vessel it missed. A title's colour
# is the approach's colour in the charts below.

# %%
# @title All approaches on one held-out image { display-mode: "form" }
SHOW_IMAGE = 0  # @param {type:"slider", min:0, max:24, step:1}
if not isinstance(SHOW_IMAGE, int) or not 0 <= SHOW_IMAGE < len(spec_eval):
    raise SystemExit(f"SHOW_IMAGE must be a whole number from 0 to {len(spec_eval) - 1}.")
it_show = spec_eval[SHOW_IMAGE]
truth_img = overlay(it_show["image"], it_show["mask"], THEME["truth"], unscored=~region(it_show))


def approach_cell(k: str) -> tuple:
    """One approach's error map on the shown image: (title, picture, title colour, tag)."""
    pred = compare_reports[k]["held_preds"][SHOW_IMAGE]
    return (COMPARE_LABEL[k], error_map(it_show["image"], pred, it_show), approach_colours[k], "errors")


grid_rows = [  # (row label, the panels of that row)
    (
        "reference",
        [
            (it_show["name"], it_show["image"], "#311B92", "input"),
            ("ground truth (scoring only)", truth_img, THEME["truth"], "truth"),
            approach_cell("full_masks"),
        ],
    ),
    ("ladder · section 4", [approach_cell(k) for k in ("zero_shot", "one_shot", "scribbles")]),
    ("path 1 · section 5", [approach_cell(k) for k in ("naive", "masked", "pu")]),
    ("path 2 · section 6", [approach_cell(k) for k in ("sam_pseudo", "gradcam_pseudo")]),
    (
        "path 3 · section 7",
        [approach_cell(k) for k in ("sam_clicks", "sam_boxes", *(["sam3_text"] if has_sam3 else []))],
    ),
]
g_cols = max(len(cells) for _, cells in grid_rows)
fig, axes = plt.subplots(len(grid_rows), g_cols, figsize=(2.9 * g_cols, 3.0 * len(grid_rows)), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for row_axes, (row_label, cells) in zip(axes, grid_rows, strict=True):
    row_axes[0].text(
        -0.06, 0.5, row_label, transform=row_axes[0].transAxes, rotation=90, ha="right", va="center",
        fontsize=9, fontweight="bold", color="#311B92",
    )
    for ax, (title, img, colour, role) in zip(row_axes, cells, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, color=colour)
        tag_panel(ax, role)
add_key(
    fig,
    ["found", "invented", ("missed", "missed (truth only); the truth panel is all green"), "unscored"],
    "title colour = the approach's colour in the charts below",
)
plt.show()

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction (1 min) -->
# **Q19.** 🟡 Before the table appears: which approach will have the **highest mean IoU**, and which will be
# the **steadiest** (the smallest spread of IoU from image to image)?
#
# - Choose your two picks from the dropdowns in the next cell and run it; leave them blank to skip. The
#   names are the picture titles above. The table further down shows where they landed.

# %%
# Q19 — your two predictions (pick from the dropdowns, or leave blank to skip)
MY_WINNER = ""  # @param ["", "zero-shot box", "one-shot (go-deeper)", "scribble head", "naive (gaps = bg)", "masked loss", "PU loss", "SAM pseudo-labels", "Grad-CAM labels", "SAM from clicks", "SAM from boxes", "full masks"]
MY_STEADIEST = ""  # @param ["", "zero-shot box", "one-shot (go-deeper)", "scribble head", "naive (gaps = bg)", "masked loss", "PU loss", "SAM pseudo-labels", "Grad-CAM labels", "SAM from clicks", "SAM from boxes", "full masks"]
KEY_OF_NAME = {COMPARE_LABEL[k]: k for k in ALL_APPROACHES}  # the dropdown shows names; the code below uses keys
for label, pick in (("MY_WINNER", MY_WINNER), ("MY_STEADIEST", MY_STEADIEST)):
    if not isinstance(pick, str) or (pick and pick not in KEY_OF_NAME):
        raise SystemExit(f'{label} must be "" or one of: {", ".join(KEY_OF_NAME)}.')
PICK_WINNER, PICK_STEADIEST = KEY_OF_NAME.get(MY_WINNER, ""), KEY_OF_NAME.get(MY_STEADIEST, "")  # as keys
print(f"your picks · highest IoU: {MY_WINNER or '(none)'} · steadiest: {MY_STEADIEST or '(none)'}")

# %% [markdown]
# This helper cell defines `spread`, the standard deviation of one approach's per-image IoUs.

# %%
# @title spread: the standard deviation of the per-image IoU { display-mode: "form" }
def spread(per_image_iou: list[float]) -> float:
    """Reference: numpy's standard deviation, which divides by n by default.
    e.g. [0.1, 0.3, 0.2, 0.6] -> 0.19"""
    return float(np.std(np.asarray(per_image_iou, dtype=float)))


# %% [markdown]
# ### 8.2 The table
#
# This cell scores every approach image by image against the hidden ground truth and shows the table,
# sorted by IoU, with where your two picks landed under it.

# %%
# @title Score every approach and show the table { display-mode: "form" }
comparison = []
iou_per_image = {}  # approach -> its IoU on each held-out image (drawn as dots below)
for k in compare_order:
    rep = compare_reports[k]
    pairs = zip(rep["held_preds"], spec_eval, strict=True)
    per_image = [iou(p & region(it), it["mask"] & region(it)) for p, it in pairs]
    iou_per_image[k] = per_image
    comparison.append(
        {
            "approach": k,
            "name": approach_names[k],
            "iou": float(np.mean(per_image)),
            **({"cldice": float(rep["cldice"])} if use_cldice else {}),
            "recall": float(rep["recall"]),
            "precision": float(rep["precision"]),
            "effort_min": float(compare_effort[k]),
            "effort_min_per_new_image": round(float(compare_per_image[k]), 2),
            "reliability": float(spread(per_image)),
            "worst_iou": float(np.min(per_image)),
            "seconds": round(float(compare_seconds[k]), 1),
            "scalability": SCALES[k],
        }
    )
RESULTS["comparison_table"] = comparison
RESULTS["comparison_assumptions"] = dict(specialised["annotation_effort"]) | {
    "reliability": f"standard deviation of the IoU over {len(spec_eval)} held-out images",
    "seconds": f"training + pseudo-labels + inference in this session, on {DEVICE}",
}

ranked = sorted(comparison, key=lambda r: r["iou"], reverse=True)
thick_iou = RESULTS["sam_direct_strokes_alone"]["iou"]
winner = ranked[0]["approach"]
steadiest = min(comparison, key=lambda r: r["reliability"])["approach"]
by_spread = [r["approach"] for r in sorted(comparison, key=lambda r: r["reliability"])]
picks = []
if PICK_WINNER in by_spread:
    rank = [r["approach"] for r in ranked].index(PICK_WINNER) + 1
    picks.append(f"your IoU pick {MY_WINNER} ranks {rank} of {len(ranked)}")
if PICK_STEADIEST in by_spread:
    rank = by_spread.index(PICK_STEADIEST) + 1
    picks.append(f"your steadiness pick {MY_STEADIEST} ranks {rank} of {len(by_spread)}")
table_rows = [
    {
        **r,
        "name": COMPARE_LABEL[r["approach"]],
        "colour": approach_colours.get(r["approach"]),
        "spread": f"{r['reliability']:.2f}",
        "annotation": f"≈ {r['effort_min']:.0f} min once"
        + (f" + {r['effort_min_per_new_image']:.1f} min per new image" if r["effort_min_per_new_image"] else ""),
    }
    for r in ranked
]
show_table(
    "Every approach on the same held-out images, best IoU first",
    table_rows,
    [
        "iou",
        *(["cldice"] if use_cldice else []),
        "recall",
        ("spread", "spread", "how much IoU varies; lower = steadier", "text"),
        ("annotation", "annotation", "a person's drawing time", "text"),
    ],
    refs=[
        {"name": "no model: strokes thickened (section 7)", "iou": thick_iou},
        {"name": "everything is foreground", "iou": RESULTS["specialised_everything_iou"]},
    ],
    meta=f"{len(spec_eval)} held-out images · compute time is in the plot of 8.3",
    note=" · ".join(
        [f"highest IoU: {COMPARE_LABEL[winner]}", f"steadiest: {COMPARE_LABEL[steadiest]}", *picks]
    ),
)
show_list(
    "How each one scales",
    [f"{COMPARE_LABEL[r['approach']]}: {r['scalability']}" for r in ranked],
    numbered=False,
)
if not (PICK_WINNER or PICK_STEADIEST):
    print("no prediction written in Q19; fine, but next time guess first: it is the cheapest lesson here")

# %% [markdown]
# ### 8.3 The trade-off in one picture
#
# This cell draws the table as one plot: each approach is a dot, placed by how much annotation it needed
# (log scale) and how well it did (IoU), with a bar for its spread. When a name had to move away from its dot
# to stay readable, a thin line joins them.
#
# What to look for: the dots near the top-left are accurate and cheap. A short bar is a steady approach. The
# dashed line is the score of marking everything, so a dot near it has learned almost nothing.

# %%
# @title Trade-off plot: accuracy against annotation { display-mode: "form" }
fig, ax = plt.subplots(figsize=(9, 5))
top = min(1.0, 1.25 * max(r["iou"] + r["reliability"] for r in comparison) + 0.05)
gap = 0.05 * top  # names in one cluster are stacked at least this far apart
efforts = {r["approach"]: max(r["effort_min"], 0.01) for r in comparison}
clusters, last = [], None  # dots whose effort is within one factor of 10 of the next share a column of names
for r in sorted(comparison, key=lambda r: efforts[r["approach"]]):
    lx = np.log10(efforts[r["approach"]])
    if last is None or lx - last >= 1.0:
        clusters.append([])
    clusters[-1].append(r)
    last = lx
label_at = {}  # approach -> (x, y) of its name
for cluster in clusters:
    x_name = 1.5 * max(efforts[r["approach"]] for r in cluster)  # right of the cluster's right-most dot
    cluster = sorted(cluster, key=lambda r: r["iou"])
    ys = []
    for r in cluster:  # push up from the lowest so no two names are closer than `gap`
        ys.append(max(r["iou"], ys[-1] + gap) if ys else r["iou"])
    over = ys[-1] - (top - gap)
    if over > 0:  # the pile reached the top of the plot: shift it down, keeping the gaps
        ys = [max(y - over, 0.02 + gap * i) for i, y in enumerate(ys)]
    for r, y in zip(cluster, ys, strict=True):
        label_at[r["approach"]] = (x_name, y)
for r in comparison:
    k, colour = r["approach"], approach_colours[r["approach"]]
    x, (lx, ly) = efforts[k], label_at[k]
    ax.plot([x, lx], [r["iou"], ly], color="#BDBDBD", lw=0.8, zorder=1)  # leader line, dot to name
    ax.errorbar(x, r["iou"], yerr=r["reliability"], fmt="o", ms=7, capsize=3, color=colour, zorder=3)
    ax.text(lx * 1.06, ly, COMPARE_LABEL[k], fontsize=8, color=colour, va="center", ha="left", zorder=4)
ax.axhline(
    RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--", label='IoU of "everything"'
)
ax.plot([], [], "o", color="#616161", label="dot = mean IoU, bar = ± spread")
ax.set_xscale("log")
ax.set_xlim(min(efforts.values()) / 2, max(efforts.values()) * 25)  # room for the names right of the last dot
ax.set_xlabel("annotation effort, ≈ minutes (log scale)")
ax.set_ylabel("mean IoU on the hidden ground truth (± spread)")
ax.set_ylim(0, top)
ax.set_title("Accuracy against annotation")
ax.legend(fontsize=8, frameon=False, loc="upper left")
add_key(fig, None, "higher is more accurate · further left is cheaper · colour = approach, as in the table")
plt.show()

# %% [markdown]
# This cell shows the second cost: the seconds each approach took in this session, training and inference
# together. What to look for: seconds matter little for the trained heads and a lot for anything that runs
# SAM on every image.

# %%
# @title Compute plot: seconds per approach { display-mode: "form" }
fig, bx = plt.subplots(figsize=(7, 0.34 * len(ranked) + 1.2))
bar_names = [COMPARE_LABEL[r["approach"]] for r in ranked][::-1]
bar_secs = [r["seconds"] for r in ranked][::-1]
bars = bx.barh(bar_names, bar_secs, color=[approach_colours[r["approach"]] for r in ranked][::-1])
bx.bar_label(bars, labels=[f"{v:.0f} s" for v in bar_secs], fontsize=7, padding=3)
bx.set_xlim(0, 1.15 * max(bar_secs))
bx.set_xlabel(f"seconds, training + inference ({DEVICE})")
bx.tick_params(axis="y", labelsize=8)
bx.set_title("Compute (sorted by IoU, best at the top)")
add_key(fig, None, "colour = approach, as in the table · a shorter bar is faster")
plt.show()

# %% [markdown]
# <!-- diagram: s08-reading-tradeoff | How to read the trade-off plot -->
# A sketch of the trade-off plot and the compute bars above, not your numbers: where the good corner is, what a long bar means, and
# which dots are a fair comparison.

# %% [markdown]
# <!-- callout: note | Reading the plot -->
# <details><summary>The same in words</summary>
#
# - **Top-left** would be the dream (accurate and nearly free; on thin structures nobody is there),
#   **bottom-left** is cheap and weak, and the **far right** is the complete-mask head, hours away.
# - **The vertical bar is the spread**: a slightly lower dot with a much shorter bar can be the safer choice.
# - **Dots at the same x** are the fairest comparison: the same strokes, used in different ways.
# - **Compute bars**: seconds matter little for the heads and a lot for anything that runs SAM on every image.
#
# </details>

# %% [markdown]
# **Q20.** 🟡 Pick one approach for each situation and name the column that decided it.
#
# - (a) A screening study of 1,000 new images next month.
# - (b) One difficult image a clinician needs today.
# - (c) A project where a missed vessel is much worse than a false alarm.
#
# Columns you can use: IoU, recall, spread, worst, effort min, /new img. Did the IoU winner win any of them?
#
# <details><summary>A possible answer</summary>
#
# - (a) **/new img** decides. Strokes or a box on every image cost 1,000 times their minutes; a trained
#   head pays its annotation once. Among the heads, pick by IoU and spread.
# - (b) One image today: a per-image row needs no training. Compare it with the thickened-strokes line.
# - (c) IoU is the wrong column. Rank by **recall** and **worst**, and accept an over-marking head if a
#   person checks it.
#
# The IoU winner can win none of the three. That is the point of a table with more than one column.
#
# </details>

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 4 min: defend a row -->
# Use your Q20 answers. **A** defends one row for situation (a), **B** defends another row for situation
# (c), one minute each, naming the column that decides. Then swap: attack the other's row.
#
# - Which column did the defence rest on, and which column of the same row would hurt it most?
# - If you changed any setting earlier (threshold, number of clicks, box size), did a row of your table
#   move compared with your neighbour's?

# %% [markdown]
# <details><summary>What pairs usually find</summary>
#
# - A good defence names a column: **/new img** for (a), **recall** or **worst** for (c).
# - Every row has a weak column (hours of painting, clicking on every image, over-marking). Find it.
# - Two rows whose IoUs differ by less than their spread are a tie: choose by the cheaper column.
#
# </details>

# %% [markdown]
# ### 8.4 Who is steady, and who fails on some images
#
# This cell draws, for each approach, one dot per held-out image (spread sideways a little so they do not
# hide each other) and a thick bar at the median image. Approaches are sorted by mean IoU, best on the left.
#
# What to look for: dots spread far apart mean the approach is good on some images and bad on others.
# Compare two approaches with the same mean: the one with the lower bottom dots is the riskier one.

# %%
# @title Strip plot: each approach's IoU on every held-out image { display-mode: "form" }
fig, ax = plt.subplots(figsize=(10, 4.4))
jitter = np.random.default_rng(0)  # seeded: the dots sit in the same place on every run
for x, r in enumerate(ranked):
    vals = np.asarray(iou_per_image[r["approach"]], float)
    colour = approach_colours[r["approach"]]
    ax.scatter(x + jitter.uniform(-0.18, 0.18, len(vals)), vals, s=16, color=colour, alpha=0.8, zorder=3)
    median = np.median(vals)
    ax.plot([x - 0.3, x + 0.3], [median] * 2, color="white", lw=6, solid_capstyle="butt", zorder=4)
    ax.plot([x - 0.3, x + 0.3], [median] * 2, color=colour, lw=4, solid_capstyle="butt", zorder=5)
line_all = RESULTS["specialised_everything_iou"]
ax.axhline(line_all, color=THEME["background"], ls="--", label='mean IoU of "everything"')
ax.plot([], [], "o", ms=4, color="#616161", label="dot = one held-out image")
ax.plot([], [], "-", lw=4, color="#616161", label="bar = median image")
ax.set_xticks(range(len(ranked)), [COMPARE_LABEL[r["approach"]] for r in ranked], rotation=45, ha="right", fontsize=8)
ax.set_xlim(-0.6, len(ranked) - 0.4)
ax.set_ylabel("IoU on one held-out image")
ax.set_title("Per-image IoU, best mean on the left")
ax.legend(fontsize=8, frameon=False, loc="upper right")
add_key(fig, None, "colour = approach, as in the table · dots spread far apart = unsteady")
plt.show()

# %% [markdown]
# This cell shows the IoU winner on its **worst**, **typical** (median) and **best** held-out image, as error
# maps. What to look for: what the winner invents ($\color{#E65100}{\blacksquare}$ orange) and what it misses
# ($\color{#2E7D32}{\blacksquare}$ green) when it fails, against
# when it works.

# %%
# @title Worst, typical and best image of the IoU winner { display-mode: "form" }
w_iou = np.asarray(iou_per_image[winner], float)
w_order = np.argsort(w_iou)
panels = []
w_picks = (("worst", w_order[0]), ("typical (median)", w_order[len(w_order) // 2]), ("best", w_order[-1]))
for tag, n in w_picks:
    it_n, pred_n = spec_eval[n], compare_reports[winner]["held_preds"][n]
    panels.append((f"{COMPARE_LABEL[winner]}\n{tag} · IoU {w_iou[n]:.2f}", error_map(it_n["image"], pred_n, it_n)))
show_masks(panels, width=3.4, legend=["found", "invented", "missed", "unscored"])

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
