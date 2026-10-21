# %% [markdown]
# ## 4. The adaptation ladder
#
# Welcome back. Before the break the head believed a label file that said "background" wherever nobody had
# looked. Block 2 is about the ways out, and the first one is a ladder: give the foundation model a little
# more task-specific information at each rung and look at what changes. Rung 0 is section 2, no training,
# just a prompt. Rung 1 is **one-shot**: a single fully annotated image. Rung 2 is **scribbles**: a few
# quick strokes on several images, and a loss that skips every pixel nobody labelled. Each rung costs a
# different kind of effort, and each one fails in its own way. Higher on the ladder is not automatically
# better, so we look at the masks, not only at the numbers.

# %% [markdown]
# <!-- callout: objective | Objective -->
# zero-shot → one-shot → scribbles: see how predictions change at each rung.

# %% [markdown]
# <!-- callout: theory | Theory: what each rung tells the model -->
# **Zero-shot** (section 2): SAM gets a box and nothing about the task; it returns its
# best guess of "the object". **One-shot**: one image with a complete mask. With frozen DINOv3 features
# the simplest use of it is a *prototype*: the mean feature of the vessel patches and the mean feature
# of the background patches; every patch of a new image goes to the nearer of the two. The same map can
# also pick clicks for SAM, the idea behind PerSAM (Zhang et al., "Personalize Segment Anything Model with
# One Shot", ICLR 2024). **Scribbles**: strokes on some vessels and some background in a few images,
# everything else *unlabelled*. The section-2 head is trained with a *partial* cross-entropy
# that sums only over the labelled pixels (Tang et al., ECCV 2018), the recipe the MIT "DINOv2 + scribbles"
# work used on frozen features, here on DINOv3. One complete mask costs hours; a scribble costs seconds.

# %% [markdown]
# This helper cell defines the tools of the ladder: one colour per approach (kept for the rest of the
# notebook), prototypes from patch features, clicks picked from a similarity map, an error map that shows
# what each prediction found, missed and invented, and a small report of the numbers we compare.

# %%
# @title Ladder helpers: approach colours, prototypes, clicks, error maps { display-mode: "form" }
APPROACH_COLOURS = {  # one colour per approach (L-003); later sections add theirs and keep these
    "zero_shot": "#E65100",
    "one_shot": "#C2185B",
    "one_shot_sam": "#F06292",
    "scribbles": "#3949AB",
}
APPROACH_NAMES = {
    "zero_shot": "zero-shot: SAM, one box",
    "one_shot": "one-shot: nearest prototype",
    "one_shot_sam": "one-shot: prototype → SAM clicks",
    "scribbles": "scribbles: partial cross-entropy",
}


def patch_share(mask: np.ndarray, grid: tuple[int, int]) -> torch.Tensor:
    """Share of each feature patch covered by a boolean pixel mask: [h, w] floats in [0, 1]."""
    m = torch.from_numpy(np.asarray(mask, np.float32))[None, None]
    return F.interpolate(m, size=grid, mode="area")[0, 0].to(DEVICE)


def prototypes(feat: torch.Tensor, w_fg: torch.Tensor, w_bg: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Foreground and background prototypes: the weighted means of the unit-length patch features."""
    f = F.normalize(feat.flatten(1), dim=0)  # [D, h*w]
    return tuple(F.normalize(f @ w.flatten().to(f), dim=0) for w in (w_fg, w_bg))


@torch.no_grad()
def prototype_score(feat: torch.Tensor, protos: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Per patch: cosine similarity to the foreground prototype minus to the background one ([h, w])."""
    return torch.einsum("dhw,d->hw", F.normalize(feat, dim=0), protos[0] - protos[1])


def score_to_mask(s: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Upsample a patch score map to the image and mark the pixels nearer the foreground prototype."""
    up = F.interpolate(s[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
    return (up > 0).cpu().numpy()


def heat_rgb(s: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """A score map as an RGB image: purple = like the foreground prototype, orange = like the background."""
    up = F.interpolate(s[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    up = 0.5 + 0.5 * up / (np.abs(up).max() + 1e-9)
    return (mpl.colormaps["PuOr"](up)[..., :3] * 255).astype(np.uint8)


def top_points(s: torch.Tensor, keep: torch.Tensor, k: int, shape: tuple[int, int], largest: bool = True):
    """k (x, y) clicks at the centres of the patches that score highest (or lowest) inside `keep`, at least
    3 patches apart: how a similarity map becomes SAM prompts (the PerSAM idea)."""
    s = (s if largest else -s).clone()
    s[keep < 0.5] = -float("inf")
    (h, w), (H, W) = s.shape, shape
    pts = []
    for _ in range(k):
        idx = int(s.argmax())
        i, j = divmod(idx, w)
        if not torch.isfinite(s[i, j]):
            break
        pts.append([(j + 0.5) * W / w, (i + 0.5) * H / h])
        s[max(0, i - 2) : i + 3, max(0, j - 2) : j + 3] = -float("inf")
    return pts


def error_map(image: np.ndarray, pred: np.ndarray, item: dict) -> np.ndarray:
    """Inside the evaluation region: vessels found in violet, false alarms in orange, missed vessels in
    green (the ground truth the prediction did not cover)."""
    r = region(item)
    truth, p = item["mask"] & r, pred & r
    out = overlay(image, truth & ~p, THEME["truth"])
    out = overlay(out, p & truth, THEME["works"])
    return overlay(out, p & ~truth, THEME["fails"])


def rung_report(preds: list[np.ndarray]) -> dict:
    """Mean IoU, clDice, recall, precision and marked share of the evaluation region, over SPEC_EVAL."""
    out = mean_scores([score(p, it) for p, it in zip(preds, SPEC_EVAL, strict=True)])
    out["recall"] = part_recall(preds, SPEC_EVAL, "mask")
    pairs = zip(preds, SPEC_EVAL, strict=True)
    out["precision"] = float(np.mean([hidden_recall(it["mask"], p & region(it)) for p, it in pairs]))
    out["marked"] = float(np.mean([p[region(it)].mean() for p, it in zip(preds, SPEC_EVAL, strict=True)]))
    return out


def effort_minutes(n_strokes: int = 0, n_full_masks: int = 0) -> float:
    """Annotation minutes under the dataset's stated assumptions (seconds per stroke, hours per full mask)."""
    cost = SPECIALISED["annotation_effort"]
    return n_strokes * cost["seconds_per_stroke"] / 60 + n_full_masks * 60 * float(
        np.mean(cost["hours_per_full_mask"])
    )


# %% [markdown]
# ### 4.1 Rung 0: zero-shot, from section 2
#
# Nothing new to run for rung 0: SAM with one box on every evaluation image, the section-2 headline. This
# cell picks up those masks and computes the numbers every rung of this section is compared on: IoU and
# clDice as before, plus the recall and precision of section 3, and the share of the fundus each mask marks.

# %%
k_demo = next(i for i, it in enumerate(SPEC_EVAL) if it is demo)  # the section-2 demo image
ladder_preds = {"zero_shot": sam_box_masks}  # section 2.3: SAM, one box, per evaluation image
ladder_report = {"zero_shot": rung_report(sam_box_masks)}
v = ladder_report["zero_shot"]
print(
    f"zero-shot (SAM, one box) on {len(SPEC_EVAL)} held-out images · IoU {v['iou']:.3f} · clDice "
    f"{v['cldice']:.3f} · recall {v['recall']:.3f} · precision {v['precision']:.3f} · marks {v['marked']:.0%}"
)

# %% [markdown]
# ### 4.2 Rung 1: one annotated image
#
# Someone spends an afternoon and traces every vessel of **one** training image. That complete mask is all
# this rung may use (L-005): the held-out ground truth stays for scoring. From it we build the two
# prototypes, "what a vessel patch looks like" and "what a background patch looks like", and label every
# patch of a held-out image by the nearer one. No training: one mean per class.

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# One complete example against SAM's box from rung 0. Will the prototype mark more of the fundus than SAM, or
# less? Will it find the thin vessels, or only the thick ones? Write one guess for each.

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell, then change which training image is the annotated one.

# %% [markdown]
# **Q14.** 🟡 Look at the similarity map: which parts of the held-out image look most like the annotated
# vessels (purple), and which like the background (orange)? Then pick another annotated image: does the
# held-out IoU move much? What does that say about relying on a single example?
#
# ```python
# ONE_SHOT_IMAGE = 0   # try 1, 2, ...
# ```
#
# This cell builds the two prototypes from the annotated image, scores every evaluation image against them
# (one DINOv3 feature grid each, seconds), and shows the annotated image, the similarity map on the demo
# image, and the resulting mask.

# %%
# Q14 — one-shot: the prototypes of one annotated image, applied to every held-out image
ONE_SHOT_IMAGE = 0  # which training image is the annotated one; try 1, 2, ...
if not isinstance(ONE_SHOT_IMAGE, int) or not 0 <= ONE_SHOT_IMAGE < len(SPECIALISED["fit"]):
    raise SystemExit(f"ONE_SHOT_IMAGE must be a whole number from 0 to {len(SPECIALISED['fit']) - 1}.")
side = SPECIALISED["feat_side"]
shot = load_item(SPECIALISED, SPECIALISED["fit"][ONE_SHOT_IMAGE])
shot_feat = features_for(SPECIALISED, shot, side)
grid, r = tuple(shot_feat.shape[1:]), region(shot)
# the rung's supervision: one complete mask of one *training* image (L-005)
protos = prototypes(shot_feat, patch_share(shot["mask"] & r, grid), patch_share(~shot["mask"] & r, grid))
t0 = time.monotonic()
proto_scores = [prototype_score(features_for(SPECIALISED, it, side), protos) for it in SPEC_EVAL]
ladder_preds["one_shot"] = [
    score_to_mask(s, it["mask"].shape) for s, it in zip(proto_scores, SPEC_EVAL, strict=True)
]
ladder_report["one_shot"] = rung_report(ladder_preds["one_shot"])
proto_sec = round(time.monotonic() - t0, 1)
v = ladder_report["one_shot"]
print(
    f"one-shot prototype from {shot['name']} · IoU {v['iou']:.3f} · clDice {v['cldice']:.3f} · recall "
    f"{v['recall']:.3f} · precision {v['precision']:.3f} · marks {v['marked']:.0%} · "
    f"{proto_sec} s on {DEVICE}"
)
print(f"for scale: the section-2 head on five complete masks · IoU {RESULTS['specialised_head_iou']:.3f}")
show_masks(
    [
        (f"the one annotated image · {shot['name']}", overlay(shot["image"], shot["mask"], THEME["truth"])),
        ("similarity: vessel − background", heat_rgb(proto_scores[k_demo], demo["mask"].shape)),
        (
            f"one-shot prototype · IoU {score(ladder_preds['one_shot'][k_demo], demo)['iou']:.2f}",
            overlay(demo["image"], ladder_preds["one_shot"][k_demo], THEME["fails"]),
        ),
        ("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"])),
    ],
    width=2.8,
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The prototype answers "which patches look like the annotated vessel patches", one 16 px square at a time.
# Expect the purple to follow the thick vessels and the dark, textured parts of the fundus, and the mask to
# be drawn as wide as a patch: the prototype has the same resolution limit as the section-2 head. Compare
# the two IoUs printed above: one example and a mean, against five complete masks and a trained head. More
# labels do not automatically give a better mask, and a simpler rule can fail less. How much the IoU moves
# when you swap the annotated image tells you how much this rung depends on <i>which</i> example you
# happened to annotate: with one image, a lucky or unlucky choice is the whole training set.
#
# </details>
#
# The same similarity map can do something else: point SAM at the right places. This cell takes, on every
# evaluation image, the five patches most like a vessel as positive clicks and the five most like background
# as negative clicks, and asks SAM for a mask (one SAM image encoding per image: seconds, also on a CPU).

# %%
N_PROTO_CLICKS = 5  # positive clicks, and as many negative ones, per image
t0 = time.monotonic()
ladder_preds["one_shot_sam"], proto_clicks = [], []
for it, s in zip(SPEC_EVAL, proto_scores, strict=True):
    keep = patch_share(region(it), tuple(s.shape))
    pos = top_points(s, keep, N_PROTO_CLICKS, it["mask"].shape, largest=True)
    neg = top_points(s, keep, N_PROTO_CLICKS, it["mask"].shape, largest=False)
    prob = sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
    ladder_preds["one_shot_sam"].append(prob >= 0.5)
    proto_clicks.append((pos, neg))
ladder_report["one_shot_sam"] = rung_report(ladder_preds["one_shot_sam"])
proto_sam_sec = round(time.monotonic() - t0, 1)
v = ladder_report["one_shot_sam"]
print(
    f"one-shot prototype → SAM clicks · IoU {v['iou']:.3f} · clDice {v['cldice']:.3f} · recall "
    f"{v['recall']:.3f} · precision {v['precision']:.3f} · marks {v['marked']:.0%} · {proto_sam_sec} s"
)
pos, neg = proto_clicks[k_demo]
demo_sam = overlay(demo["image"], ladder_preds["one_shot_sam"][k_demo], THEME["fails"])
show_clicks(
    [
        ("clicks chosen as vessel", demo["image"], pos),
        ("clicks chosen as background", demo["image"], neg),
        ("SAM from those clicks", demo_sam, pos),
        ("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]), None),
    ],
    width=2.8,
)

# %% [markdown]
# **Q15.** 🟡 Same annotated image, same similarity map, two ways to use it. Where do the prototype mask and
# the SAM mask differ: the width of the vessels, the regions they mark, the thin branches? Did the clicks
# land on vessels?
#
# <details><summary>What to look for</summary>
#
# The prototype labels <i>every</i> patch, so it covers the whole tree at patch resolution. SAM only sees
# ten clicks, and it does what section 2 showed it does with clicks: it looks for the object that contains
# the positive ones and avoids the negative ones. Its edges are sharper, but it can lose whole branches no
# click touched, or grow into a region around a thick vessel. If a "vessel" click landed on the optic disc
# or a dark spot, the mask follows it: SAM trusts its prompts. Two uses of the same information can fail in
# opposite directions.
#
# </details>
#
# ### 4.3 Rung 2: scribbles
#
# The annotator from section 3 again, with a different instruction: "don't trace everything; draw a few
# strokes along some vessels and a few on clear background, in each of five images, then stop". Every
# other pixel is unlabelled, and this time the file says so (value 0), so the loss can skip it. This cell
# shows the strokes on one training image: vessel strokes in teal, background strokes in grey, thickened
# for the figure only.

# %%
scrib_train = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:N_TRAIN_SPEC]]
labelled = float(np.mean([(it["scribbles"] > 0)[region(it)].mean() for it in scrib_train]))
scrib_meta = SPECIALISED["fit"][:N_TRAIN_SPEC]
n_strokes = sum(m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in scrib_meta)
print(
    f"{len(scrib_train)} training images · {n_strokes} strokes · {labelled:.2%} of the fundus pixels "
    f"labelled · ≈ {effort_minutes(n_strokes=n_strokes):.1f} min at "
    f"{SPECIALISED['annotation_effort']['seconds_per_stroke']} s per stroke"
)
ex = scrib_train[0]
thick = {v: ndi.binary_dilation(ex["scribbles"] == v, iterations=2) for v in (1, 2)}
strokes = overlay(overlay(ex["image"], thick[2], THEME["background"], 0.9), thick[1], THEME["scribble"], 0.9)
show_masks(
    [
        (ex["name"], ex["image"]),
        ("scribbles: vessel · background", strokes),
        ("ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
    ],
    width=3.0,
)

# %% [markdown]
# <!-- callout: theory | Theory: partial cross-entropy -->
# The loss of section 2 sums a cross-entropy term over every pixel. The partial version
# sums it only over the pixels a person labelled: a vessel stroke says "1", a background stroke says "0",
# and an unlabelled pixel contributes nothing at all, neither "vessel" nor "background". In code that is a
# target of `-1`, which `train_head` already skips (the border of the fundus has been
# skipped that way since section 2). The balanced weighting is computed over the labelled pixels only.
# Compare with section 3: there the unpainted pixels were targets of `0`.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 3 min): choose which pixels to ignore -->
# Fill `my_scribble_targets`. It turns a scribble map
# (0 unlabelled, 1 vessel stroke, 2 background stroke) into training targets: 1, 0, or −1 for "ignore".
# Pixels outside `region` are always −1. With `bg_beyond_px=None`, every unlabelled
# pixel is ignored. With a number, you decide that unlabelled pixels farther than that many pixels from any
# vessel stroke count as background (`ndi.distance_transform_edt(scribbles != 1)` gives each
# pixel's distance to the nearest vessel stroke). The cell checks your version on a 6 × 6 play map with both
# settings and continues with ours if you skip it. You will use the second setting in 4.5.

# %%
# ✍️ YOUR CODE
def my_scribble_targets(scribbles: np.ndarray, region: np.ndarray, bg_beyond_px: float | None = None):
    """Targets (int8) from a scribble map: 1 vessel stroke, 0 background stroke, -1 ignored. Outside
    `region`: -1. If bg_beyond_px is a number, unlabelled pixels farther than bg_beyond_px from every vessel
    stroke become 0 (background); the others stay -1."""
    ...


def _ref_scribble_targets(scribbles: np.ndarray, region: np.ndarray, bg_beyond_px: float | None = None):
    """Reference: strokes as given, unlabelled ignored (or background when far from every vessel stroke)."""
    targets = np.full(scribbles.shape, -1, np.int8)
    if bg_beyond_px is not None:
        targets[ndi.distance_transform_edt(scribbles != 1) > bg_beyond_px] = 0
    targets[scribbles == 2] = 0
    targets[scribbles == 1] = 1
    targets[~np.asarray(region, bool)] = -1
    return targets


_play_scrib = np.zeros((6, 6), np.uint8)
_play_scrib[1, :4] = 1  # a vessel stroke
_play_scrib[4, 1:3] = 2  # a background stroke
_play_scrib[5, 5] = 2  # a background stroke outside the region
_play_region = np.ones((6, 6), bool)
_play_region[:, 5] = False  # the last column is outside the evaluation region
scribble_targets = use(
    my_scribble_targets,
    reference=_ref_scribble_targets,
    check=lambda f: [np.asarray(f(_play_scrib, _play_region, d)).astype(int) for d in (None, 2)],
    name="scribble_targets",
)
print("play example · targets with bg_beyond_px=2 (−1 = ignored):")
print(scribble_targets(_play_scrib, _play_region, 2))

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# The head
# will see labels on fewer than two pixels in a hundred, and no vessel will ever be called "background".
# Against the one complete mask of rung 1, will it find more of the held-out vessels, or fewer? Will it
# mark more of the fundus, or less? Write two guesses.

# %% [markdown]
# This cell trains the section-2 head on the scribbles of the five training images, every unlabelled pixel
# ignored, and prints the seconds it took. Not done in a minute, or on a CPU and in a hurry? Set
# `TRAIN_YOURSELF = False` and run it again: it loads our checkpoint, trained the same way on all fit images.

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "scribbles" head instead of training
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

scrib_feats = [features_for(SPECIALISED, it, side) for it in scrib_train]
# the rung's supervision: strokes only; unlabelled pixels and the border are ignored (-1), never "background"
scrib_labels = [scribble_targets(it["scribbles"], region(it)) for it in scrib_train]
if TRAIN_YOURSELF:
    scrib_head, scrib_sec = train_head(scrib_feats, scrib_labels, HEAD_STEPS)
    src = f"trained here on {len(scrib_train)} images, {HEAD_STEPS} steps, partial cross-entropy"
else:
    scrib_head, entry = load_head(SPECIALISED, "scribbles", side)
    scrib_sec = entry["train_sec"]
    src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
print(f"scribble head @ {side} px · {src} · {scrib_sec:.1f} s on {DEVICE}")
if scrib_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the scribble head on the evaluation images with the same five numbers as the other rungs.

# %%
t0 = time.monotonic()
ladder_preds["scribbles"] = [
    predict_head(scrib_head, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
]
ladder_report["scribbles"] = rung_report(ladder_preds["scribbles"])
scrib_infer = round(time.monotonic() - t0, 1)
v = ladder_report["scribbles"]
print(
    f"scribble head · IoU {v['iou']:.3f} · clDice {v['cldice']:.3f} · recall {v['recall']:.3f} · precision "
    f"{v['precision']:.3f} · marks {v['marked']:.0%} · {scrib_infer} s on {DEVICE}"
)

# %% [markdown]
# ### 4.4 The ladder, side by side
#
# This cell puts the rungs next to each other: a table with the five numbers and the annotation each rung
# needed, a bar chart (one colour per approach, kept for the rest of the notebook), and the demo image's
# error maps: **violet** = vessels found, **orange** = false alarms, **green** = vessels missed.

# %%
effort = {
    "zero_shot": effort_minutes(n_strokes=len(SPEC_EVAL)),  # one box per image you segment
    "one_shot": effort_minutes(n_full_masks=1),
    "one_shot_sam": effort_minutes(n_full_masks=1),
    "scribbles": effort_minutes(n_strokes=n_strokes),
}
effort_text = {
    "zero_shot": f"1 box per image ({len(SPEC_EVAL)} here)",
    "one_shot": "1 complete mask",
    "one_shot_sam": "1 complete mask",
    "scribbles": f"{n_strokes} strokes, {len(scrib_train)} images",
}
RESULTS["ladder"] = {k: ladder_report[k]["iou"] for k in ("zero_shot", "one_shot", "scribbles")}  # SPEC §9
RESULTS["ladder_detail"] = ladder_report
RESULTS["ladder_effort_min"] = {k: round(v, 2) for k, v in effort.items()}
RESULTS["scribbles_head"] = ladder_report["scribbles"] | {
    "long_side": side,
    "n_train": len(scrib_train),
    "n_strokes": n_strokes,
    "labelled_fraction": labelled,
    "train_sec": round(float(scrib_sec), 2),
    "infer_sec": scrib_infer,
}
RESULTS["one_shot"] = ladder_report["one_shot"] | {"image": shot["name"], "infer_sec": proto_sec}

COLUMNS = f"{'IoU':>6}{'clDice':>8}{'recall':>8}{'precis.':>8}{'marked':>8}"
print(f"{'approach':<36}{COLUMNS}   annotation (≈ minutes)")
for k, v in ladder_report.items():
    print(
        f"{APPROACH_NAMES[k]:<36}{v['iou']:>6.3f}{v['cldice']:>8.3f}{v['recall']:>8.3f}{v['precision']:>8.3f}"
        f"{v['marked']:>8.0%}   {effort_text[k]} (≈ {effort[k]:.1f})"
    )
print(f"{'everything is foreground':<36}{RESULTS['specialised_everything_iou']:>6.3f}")

metrics = ["iou", "cldice", "recall", "precision"]
x = np.arange(len(metrics))
width = 0.8 / len(ladder_report)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for n, (k, v) in enumerate(ladder_report.items()):
    shift = (n - (len(ladder_report) - 1) / 2) * width
    ax.bar(x + shift, [v[m] for m in metrics], width, color=APPROACH_COLOURS[k], label=APPROACH_NAMES[k])
ax.axhline(
    RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--", label='IoU of "everything"'
)
ax.set_xticks(x, ["IoU", "clDice", "recall", "precision"])
ax.set_ylim(0, 1)
ax.set_title("The adaptation ladder: different information, different failures")
ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
plt.tight_layout()
plt.show()

show_masks(
    [
        (
            f"{APPROACH_NAMES[k].split(':')[0]} · IoU {score(ladder_preds[k][k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], ladder_preds[k][k_demo], demo),
        )
        for k in ladder_preds
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.6,
)

# %% [markdown]
# **Q16.** 🟡 Read the error maps rung by rung. Which rung misses the most vessels (green), which invents the
# most (orange), and is that the same rung that has the lowest IoU? If you had to pick one rung for a
# screening tool that must not miss a vessel, and one for a tool whose every mark must be right, would you
# pick the same?
#
# <details><summary>What to look for</summary>
#
# The rungs do not line up on a single "quality" axis. SAM's box marks a region or the whole eye: high
# recall, many false alarms, an IoU near the grey line. The prototype labels by patch, so it draws vessels
# as wide as a square and confuses dark, textured background with vessel. SAM from prototype clicks is
# sharper but only covers what its clicks touched. The scribble head saw the fewest labelled pixels, yet
# from several images, and it was never told a vessel was background, so its errors are not the section-3
# errors: look at its recall next to the naive head's. Which rung is "best" depends on whether missing or
# inventing a vessel costs you more, and on the minutes in the last column. On your run the order may differ
# from your neighbour's: that is the point of reading the maps, not only the IoU.
#
# </details>
#
# ### 4.5 Your choice: which unlabelled pixels to ignore
#
# Ignoring every unlabelled pixel is the safe choice, and also a wasteful one: more than 98 % of the fundus
# says nothing. A tempting shortcut is "far from every vessel stroke, it's surely background". That is the
# second setting of your `scribble_targets`.

# %% [markdown]
# <!-- callout: pitfall | Pitfall: "surely background" is a guess -->
# The strokes cover only some of the vessel segments. A vessel nobody scribbled can be
# far from every stroke, and the shortcut then labels it background: the section-3 mistake, back through
# the side door. The next cell measures how much of the true vessel area your rule mislabels, on the
# training images, using the hidden ground truth for that diagnosis only, never for training.

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell, then try a few distances.

# %% [markdown]
# **Q17.** 🟡 Set a distance and train a second scribble head with it. How many true vessel pixels did your
# rule call background? Did recall, precision and IoU on the held-out images move, and in which direction?
# Is there a distance that helps without hurting?
#
# ```python
# BG_BEYOND_PX = 40   # try 20, 80, None (= ignore every unlabelled pixel)
# ```
#
# This cell turns the scribbles into targets with your distance, trains the same head on them (seconds),
# and prints its numbers next to the scribble head of 4.3, which ignored every unlabelled pixel.

# %%
# Q17 — scribbles plus "far from every vessel stroke = background"
BG_BEYOND_PX = 40  # pixels, on the 1024 px crops; try 20, 80, None (= ignore every unlabelled pixel)
if BG_BEYOND_PX is not None and (not isinstance(BG_BEYOND_PX, (int, float)) or BG_BEYOND_PX <= 0):
    raise SystemExit("BG_BEYOND_PX must be a positive number of pixels, or None.")
bg_px = BG_BEYOND_PX / 4 if (SMOKE and BG_BEYOND_PX) else BG_BEYOND_PX  # smoke thumbnails are 4× smaller
far_labels = [scribble_targets(it["scribbles"], region(it), bg_px) for it in scrib_train]
# diagnosis only: how much of the true vessel area the rule called background (never used for training)
far_pairs = list(zip(far_labels, scrib_train, strict=True))
wrong = float(np.mean([((t == 0) & it["mask"]).sum() / max(it["mask"].sum(), 1) for t, it in far_pairs]))
far_labelled = float(np.mean([(t >= 0)[region(it)].mean() for t, it in far_pairs]))
far_head, far_sec = train_head(scrib_feats, far_labels, HEAD_STEPS)
far_preds = [
    predict_head(far_head, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
]
far_report = rung_report(far_preds)
RESULTS["scribbles_bg_beyond"] = far_report | {"bg_beyond_px": BG_BEYOND_PX, "vessel_called_bg": wrong}
print(
    f"rule: unlabelled pixels > {BG_BEYOND_PX} px from every vessel stroke = background · labelled "
    f"{far_labelled:.0%} of the fundus · {wrong:.1%} of the true vessel pixels called background · "
    f"trained in {far_sec:.1f} s"
)
print(f"{'scribble head':<28}{COLUMNS}")
for name, v in {"ignore every unlabelled": ladder_report["scribbles"], "your distance": far_report}.items():
    nums = f"{v['iou']:>6.3f}{v['cldice']:>8.3f}{v['recall']:>8.3f}{v['precision']:>8.3f}"
    print(f"{name:<28}{nums}{v['marked']:>8.0%}")

# %% [markdown]
# <details><summary>What to look for</summary>
#
# A large distance adds a lot of background that really is background, and a little that is not; a small
# distance adds more of both. More "background" targets push the head towards marking less, so watch recall
# fall, the direction of section 3, and check whether precision rose enough to pay for it. Whether IoU
# improves depends on which effect wins on your images, and the line "vessel pixels called background"
# tells you the price. The honest lesson is not a
# number: every rule that fills in unlabelled pixels is an assumption about where the annotator did not
# look, and it is only as good as that assumption. Section 5 makes "unknown" explicit in the loss; section 6
# fills unlabelled pixels with a second model's guesses instead of a distance.
#
# </details>

# %% [markdown]
# <!-- callout: across | Across approaches -->
# Three
# rungs, three kinds of information. Zero-shot: a prompt, so SAM's idea of an object decides. One-shot: one
# complete example, so *which* example you annotated decides, and the patch resolution limits the
# mask. Scribbles: many cheap, partial hints from several images, with "unknown" kept unknown. The rungs
# differ more in *how* they fail than in a single number, and the cheapest rung in minutes is not the
# worst. Keep the colours: these approaches come back in the comparison table of section 8.

# %% [markdown]
# <!-- callout: your-turn | Try it on your image (optional, 2 min) -->
# The one-shot idea works from clicks too: a few clicks on the thing you want and a few on
# what you do not want give two prototypes, and every similar patch of the image is marked. Upload a photo
# with the folder icon on the left, put its path in `MY_IMAGE`, and some `(x, y)`
# clicks in `MY_FG_CLICKS` and `MY_BG_CLICKS`. Left empty, it runs on the demo image
# with clicks along its scribbles. Section 9 runs the whole ladder on your own scribbles.

# %% [markdown]
# This cell computes DINOv3 features of your image (or the demo image), builds the two prototypes from the
# patches under your clicks, and shows the similarity map and the mask: one model pass, no training.

# %%
MY_IMAGE = ""  # e.g. "/content/my_photo.jpg"; empty = the demo image
MY_FG_CLICKS = []  # e.g. [(120, 80), (300, 210)] on the thing you want
MY_BG_CLICKS = []  # e.g. [(20, 20)] on what you do not want; empty = every other patch
if not isinstance(MY_IMAGE, str) or not isinstance(MY_FG_CLICKS, list) or not isinstance(MY_BG_CLICKS, list):
    raise SystemExit('MY_IMAGE must be a path in quotes (or "") and the clicks lists of (x, y) pairs.')
if MY_IMAGE and not Path(MY_IMAGE).is_file():
    raise SystemExit(f"{MY_IMAGE} not found. Upload it with the folder icon on the left and copy its path.")
if MY_IMAGE and not MY_FG_CLICKS:
    raise SystemExit("Add at least one (x, y) click on the thing you want to MY_FG_CLICKS.")
my_img = np.array(Image.open(MY_IMAGE).convert("RGB")) if MY_IMAGE else demo["image"]
my_h, my_w = my_img.shape[:2]
if MY_IMAGE:
    my_fg = [[float(x), float(y)] for x, y in MY_FG_CLICKS]
    my_bg = [[float(x), float(y)] for x, y in MY_BG_CLICKS]
else:
    ys, xs = np.nonzero(demo["scribbles"] == 2)
    picks = np.linspace(0, len(ys) - 1, 3).astype(int) if len(ys) else []
    my_fg, my_bg = scribble_points(demo, 3), [[float(xs[k]), float(ys[k])] for k in picks]
if not all(0 <= x < my_w and 0 <= y < my_h for x, y in my_fg + my_bg):
    raise SystemExit(f"every click must lie inside the image: 0 ≤ x < {my_w}, 0 ≤ y < {my_h}.")
my_feat = dino.features(my_img, min(side, max(my_h, my_w)))
gh, gw = my_feat.shape[1:]
w_fg, w_bg = torch.zeros(gh, gw, device=DEVICE), torch.zeros(gh, gw, device=DEVICE)
for weights, pts in ((w_fg, my_fg), (w_bg, my_bg)):
    for x, y in pts:
        weights[min(int(y * gh / my_h), gh - 1), min(int(x * gw / my_w), gw - 1)] = 1.0
if not my_bg:
    w_bg = 1.0 - w_fg  # no background clicks: every other patch is the background prototype
my_score = prototype_score(my_feat, prototypes(my_feat, w_fg, w_bg))
my_mask = score_to_mask(my_score, (my_h, my_w))
show_clicks(
    [
        (MY_IMAGE or f"demo · {demo['name']} · clicks on the thing", my_img, my_fg),
        ("clicks on what you do not want", my_img, my_bg),
        ("similarity: thing − not", heat_rgb(my_score, (my_h, my_w)), None),
        (f"prototype mask · marks {my_mask.mean():.0%}", overlay(my_img, my_mask, THEME["fails"]), my_fg),
    ],
    width=2.8,
)
