# %% [markdown]
# ## 8. Compare everything
#
# Every section since 2 ended with a small table of its own, each against a slightly different neighbour.
# This section puts all of them in one table, scored on the same held-out images and the same hidden
# ground truth, so that the question becomes the one you face on a real project: not "which approach has
# the highest number" but "which approach would I pay for". A recipe that wins on accuracy can lose on
# annotation hours, or win on average and fail badly on one image in five. The table has one row per
# approach and five things to read in each row; the plot underneath shows the trade-off at a glance.

# %% [markdown]
# <!-- callout: objective | Objective -->
# fill the comparison table, predicting the winner first.

# %% [markdown]
# <!-- callout: theory | Theory: the five columns -->
# **Accuracy**: the mean IoU over the held-out images, plus clDice for thin structures (does the mask
# follow the centre lines?). **Annotation effort**: what a person had to draw, turned into minutes:
# strokes and boxes at a few seconds each, complete or half-complete masks at hours each. **Reliability**:
# how much the IoU varies from one held-out image to the next (the standard deviation); a small spread
# means you can trust the average on a new image, a large one means some images fail badly. **Compute**:
# seconds of training plus inference in this session. **Scalability** is a sentence, not a number: what
# it costs to segment the next thousand images.

# %% [markdown]
# <!-- callout: note | Assumptions behind the minutes -->
# A scribble stroke, a click or a box takes about **2 seconds** to draw. A complete vessel
# map of one fundus image takes **3 to 5 hours** (the FIVES paper, Jin et al., *Scientific Data*
# 2022), so we count the midpoint, 4 hours, and a half-painted map as half of that. Both numbers come from
# the dataset config, not from a measurement in this room; change them there and every row moves with them.
# Training minutes are paid once; the per-image column is what every *new* image costs.

# %% [markdown]
# This helper cell collects, for every approach of sections 2 to 7, its predictions on the held-out images,
# its numbers, its seconds and its annotation minutes, and gives each approach one sentence on how it scales.

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
# ### 8.1 Your prediction first

# %% [markdown]
# <!-- callout: your-turn | Your prediction (1 min) -->
# Before the table appears, write down two guesses using the short names printed above. Which approach will
# have the **highest mean IoU**? And which one will be the **steadiest**, the smallest spread of IoU
# from image to image? You have seen most of these numbers one section at a time; the point is to see
# whether your picture of the whole holds up when they sit side by side.

# %% [markdown]
# **Q31.** 🟡 Which approach wins on IoU, and which is the steadiest? Write your two picks in the next cell
# and run it; the table cell further down tells you where they landed.
#
# This cell only checks that your picks are names from the list, so that the comparison below can find them.

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
# Fill `my_spread`. In: the IoU of one approach on each held-out image, as a list of floats. Out:
# their standard deviation, the population version (divide by the number of images, not by one less), as a
# float. A small spread means the mean IoU is a fair promise for the next image; a large one means a few
# images pull the mean up or down. The cell checks your version on four play values and continues with ours
# if you skip it.


# %%
# ✍️ YOUR CODE
def my_spread(per_image_iou: list[float]) -> float:
    """The population standard deviation of the per-image IoUs: sqrt(mean((x - mean(x))²))."""
    ...


def _ref_spread(per_image_iou: list[float]) -> float:
    """Reference: numpy's standard deviation, which divides by n by default."""
    return float(np.std(np.asarray(per_image_iou, dtype=float)))


_play_ious = [0.1, 0.3, 0.2, 0.6]  # four images: three middling, one much better
spread = use(my_spread, reference=_ref_spread, check=lambda f: round(float(f(_play_ious)), 4), name="spread")
print("play IoUs", _play_ious, "· spread", round(spread(_play_ious), 4))

# %% [markdown]
# ### 8.3 The table
#
# This cell scores every approach's held-out masks image by image against the hidden ground truth, fills
# one row per approach (IoU, clDice, minutes of annotation, spread, seconds, and how it scales), prints the
# table sorted by IoU, and tells you where your two predictions landed.

# %%
comparison = []
for k in COMPARE_ORDER:
    rep = COMPARE_REPORTS[k]
    pairs = zip(rep["held_preds"], SPEC_EVAL, strict=True)
    per_image = [iou(p & region(it), it["mask"] & region(it)) for p, it in pairs]
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
    f"{'approach':<46}{'IoU':>6}{extra}{'± spread':>10}{'worst':>7}{'effort min':>12}{'/new img':>10}"
    f"{'sec':>7}"
)
for r in ranked:
    cl = f"{r['cldice']:>8.3f}" if WITH_CLDICE else ""
    print(
        f"{r['name']:<46}{r['iou']:>6.3f}{cl}{r['reliability']:>10.3f}{r['worst_iou']:>7.3f}"
        f"{r['effort_min']:>12.1f}{r['effort_min_per_new_image']:>10.2f}{r['seconds']:>7.1f}"
    )
print(
    f"{'no model: strokes thickened (section 7)':<46}{RESULTS['sam_direct_strokes_alone']['iou']:>6.3f}\n"
    f"{'everything is foreground':<46}{RESULTS['specialised_everything_iou']:>6.3f}\n"
)
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
# ### 8.4 The trade-off in one picture
#
# This cell draws the table: on the left, IoU against annotation minutes (log scale, because a box and a
# complete mask differ by a factor of thousands), with a bar for each approach's spread over the images; on
# the right, the seconds of compute. One colour per approach, the same as in every earlier figure.

# %%
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
# <details><summary>What to look for</summary>
#
# Read the left panel by corners. Top-left would be the dream: accurate and nearly free; usually nobody is
# there on thin structures. Bottom-left is cheap and weak, often the prompt-only approaches. Far right is the
# complete-mask head: whatever its IoU, its minutes are hours, and the log axis hides how far away it is.
# The most interesting comparisons are between approaches at the same x position, because they used the
# same annotation and differ only in what was done with it (the scribble head, the two pseudo-label heads,
# and the direct SAM recipes all start from the same kind of strokes). Then look at the vertical bars: an
# approach with a slightly lower mean but a much smaller spread can be the safer choice. The right panel
# usually matters least for the heads (seconds) and most for anything that runs SAM on every image.
#
# </details>
#
# **Q32.** 🟡 Pick one approach for each situation and name the column that decided it: (a) a screening
# study of 1,000 new images next month, (b) one difficult image a clinician needs today, (c) a project
# where a missed structure is much worse than a false alarm. Did the IoU winner win any of them?
#
# <details><summary>A possible answer</summary>
#
# (a) A thousand images make the per-image column decisive: anything that needs strokes or a box on every
# image multiplies its minutes by a thousand, while a trained head pays its annotation once and then costs a
# feature pass per image; among the heads, pick by IoU and spread. (b) For one image today, the per-image
# rows are the natural candidates: no training, a minute of strokes, and you look at the result yourself
# (compare them with the thickened strokes line before trusting SAM's share of the work). (c) When misses
# cost more than false alarms, IoU is the wrong column: look at recall (in <code>RESULTS</code>, or the
# tables of sections 4–7) and at the worst image, and accept a head that over-marks if a person checks its
# output. The IoU winner can win none of the three; that is the point of a table with more than one column.
#
# </details>

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min: compare tables with your neighbour -->
# Your sessions ran with the same seed but possibly different settings (Q17, Q23,
# Q27, Q28, the 🔴 parts), and on a CPU session the image counts differ from a GPU one. Put your two tables
# side by side. Which rows moved, and which stayed put? Did you predict the same winner? Agree on one
# approach you would try first on a new specialised task, and one sentence on why.

# %% [markdown]
# <details><summary>What pairs usually find</summary>
#
# The order of the top few rows is often less stable than it looks: a different threshold, a different
# number of clicks or a different set of evaluation images can swap neighbours whose IoUs differ by a few
# hundredths, and the spread column tells you those differences are inside the noise. What rarely moves is
# the shape of the plot: the prompt-only rows stay cheap per image but weak or unsteady, the complete-mask
# head stays hours away on the effort axis, and the stroke-based heads sit together in the middle. When two
# tables disagree on the winner but agree on the shape, trust the shape.
#
# </details>

# %% [markdown]
# <!-- callout: across | Across approaches -->
# Every path of this notebook added a different kind of information: a prompt, one complete example, cheap
# strokes, an honest loss about missing labels, a second model's guesses, or a person pointing at every
# image. The table shows that none of them is free and none wins every column. Accuracy, annotation minutes,
# reliability and compute pull in different directions, and the right row depends on how many images you
# have, who checks the output, and what a mistake costs. Section 9 runs the cheapest rungs on your own
# images; section 10 turns this table into rules you can take to the next problem.
