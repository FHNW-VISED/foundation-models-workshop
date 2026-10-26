# %% [markdown]
# ## 4. The adaptation ladder
#
# In section 3 the head believed a label file that said "background" wherever nobody
# had looked. Now the ways out, and the first is a ladder: give the foundation model a little
# more task information at each rung, and look at what changes. Higher is not automatically better, and
# each rung fails in its own way, so we look at the masks, not only at the numbers.

# %% [markdown]
# <!-- callout: objective | Objective -->
# zero-shot → one-shot → scribbles: see how predictions change at each rung.

# %% [markdown]
# The next cell draws what the model receives at each rung, with its annotation cost: the input is the only
# thing that changes.

# %%
# @title What each rung receives { display-mode: "form" }
def effort_minutes(n_strokes: int = 0, n_full_masks: int = 0) -> float:
    """Annotation minutes under the dataset's stated assumptions (seconds per stroke, hours per full mask)."""
    cost = SPECIALISED["annotation_effort"]
    return n_strokes * cost["seconds_per_stroke"] / 60 + n_full_masks * 60 * float(
        np.mean(cost["hours_per_full_mask"])
    )


_shot0 = load_item(SPECIALISED, SPECIALISED["fit"][0])
_scribs = [load_item(SPECIALISED, m) for m in SPECIALISED["fit"][:N_TRAIN_SPEC]]
_n_str = sum(m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in SPECIALISED["fit"][:N_TRAIN_SPEC])


def _cost(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.0f} h"


_ex = _scribs[0]
_thick = {v: ndi.binary_dilation(_ex["scribbles"] == v, iterations=2) for v in (1, 2)}
_strokes = overlay(_ex["image"], _thick[2], THEME["background"], 0.9)
_strokes = overlay(_strokes, _thick[1], THEME["scribble"], 0.9)
_x0, _y0, _x1, _y1 = box_prompt(demo)
_box = mpl.patches.Rectangle((_x0, _y0), _x1 - _x0, _y1 - _y0, fill=False, lw=2, ec=THEME["scribble"])
_c0, _c1 = _cost(effort_minutes(n_strokes=1)), _cost(effort_minutes(n_full_masks=1))
_c2 = _cost(effort_minutes(n_strokes=_n_str))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
axes[0].imshow(demo["image"])
axes[0].add_patch(_box)
axes[0].set_title(f"Rung 0 · zero-shot\n1 image + 1 box · ≈ {_c0}")
axes[1].imshow(overlay(_shot0["image"], _shot0["mask"], THEME["truth"]))
axes[1].set_title(f"Rung 1 · one-shot\n1 image + its complete mask · ≈ {_c1}")
axes[2].imshow(_strokes)
axes[2].set_title(f"Rung 2 · scribbles\n{len(_scribs)} images + {_n_str} strokes · ≈ {_c2}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: theory | Theory: what each rung tells the model -->
# - **Rung 0 · zero-shot** (section 2): SAM gets the image and one box, and nothing about the task. It
#   returns its best guess of "the object": on the retina, the eye or one thick vessel.
# - **Rung 1 · one-shot:** one image with a complete vessel mask. On frozen DINOv3 features it becomes a
#   *prototype*: the mean feature of its vessel patches and of its background patches. Every patch of a new
#   image goes to the nearer one. This rung has **two variants**: the prototype mask itself, and the same
#   similarity map choosing clicks for SAM. That is why the table in 4.4 has four rows for three rungs.
# - **Rung 2 · scribbles:** strokes on some vessels and some background in a few images; everything else is
#   *unlabelled*. The section-2 head is trained with a loss that counts only the labelled pixels (4.3).
# - **Cost:** one complete mask takes hours, one scribble seconds.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper: where the rungs come from -->
# - Clicks from a similarity map: PerSAM, Zhang et al., ICLR 2024.
# - Partial cross-entropy on scribbles: Tang et al., ECCV 2018.
# - The recipe of the MIT "DINOv2 + scribbles" work, here on DINOv3.

# %% [markdown]
# This helper cell defines the tools of the ladder: one colour per approach (kept for the rest of the
# notebook), prototypes, clicks from a similarity map, error and difference maps, and a report of the
# numbers we compare.

# %%
# @title Ladder helpers: colours, prototypes, clicks, error and difference maps { display-mode: "form" }
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
APPROACH_SHORT = {
    "zero_shot": "zero-shot",
    "one_shot": "prototype",
    "one_shot_sam": "proto → SAM",
    "scribbles": "scribbles",
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


def diff_map(image: np.ndarray, prev: np.ndarray, new: np.ndarray, k_prev: str, k_new: str, item: dict):
    """Where two rungs disagree: pixels only the earlier rung marks in its colour, only the later rung in
    its colour, both in grey."""
    r = region(item)
    prev, new = prev & r, new & r
    out = overlay(image, prev & new, THEME["background"])
    out = overlay(out, prev & ~new, APPROACH_COLOURS[k_prev])
    return overlay(out, new & ~prev, APPROACH_COLOURS[k_new])


def rung_report(preds: list[np.ndarray]) -> dict:
    """Mean IoU, clDice, recall, precision and marked share of the evaluation region, over SPEC_EVAL."""
    out = mean_scores([score(p, it) for p, it in zip(preds, SPEC_EVAL, strict=True)])
    out["recall"] = part_recall(preds, SPEC_EVAL, "mask")
    pairs = zip(preds, SPEC_EVAL, strict=True)
    out["precision"] = float(np.mean([hidden_recall(it["mask"], p & region(it)) for p, it in pairs]))
    out["marked"] = float(np.mean([p[region(it)].mean() for p, it in zip(preds, SPEC_EVAL, strict=True)]))
    return out


def image_numbers(pred: np.ndarray, item: dict) -> dict:
    """IoU, clDice, recall, precision and marked share of one prediction on one image."""
    r = region(item)
    p = pred & r
    return score(pred, item) | {
        "recall": hidden_recall(pred, item["mask"] & r),
        "precision": hidden_recall(item["mask"], p),
        "marked": float(p[r].mean()),
    }


def show_extremes(preds: list[np.ndarray]) -> None:
    """Error maps of the worst, the median and the best held-out image (by IoU), with their numbers."""
    nums = [image_numbers(p, it) for p, it in zip(preds, SPEC_EVAL, strict=True)]
    order = np.argsort([n["iou"] for n in nums])
    picks = [("worst", order[0]), ("typical (median)", order[len(order) // 2]), ("best", order[-1])]
    show_masks(
        [
            (
                f"{label} · IoU {nums[k]['iou']:.2f}\nrecall {nums[k]['recall']:.2f} · "
                f"precision {nums[k]['precision']:.2f}\nmarked {nums[k]['marked']:.0%}",
                error_map(SPEC_EVAL[k]["image"], preds[k], SPEC_EVAL[k]),
            )
            for label, k in picks
        ],
        width=3.3,
    )


# %% [markdown]
# ### 4.1 Rung 0: zero-shot, from section 2
#
# Nothing new to run: the next cell measures section 2's SAM-with-one-box masks with the numbers every rung
# is compared on.

# %% [markdown]
# - **IoU, clDice:** as before: overlap, and overlap along the centre line.
# - **Recall:** of the true vessel pixels, the share the mask finds.
# - **Precision:** of the pixels the mask marks, the share that really are vessel.
# - **Marked:** the share of the fundus the mask covers.
# - **Error-map colours:** violet = vessel found, orange = false alarm, green = vessel missed.

# %%
# @title Score rung 0 and show its worst, typical and best image { display-mode: "form" }
k_demo = next(i for i, it in enumerate(SPEC_EVAL) if it is demo)  # the section-2 demo image
ladder_preds = {"zero_shot": sam_box_masks}  # section 2.3: SAM, one box, per evaluation image
ladder_report = {"zero_shot": rung_report(sam_box_masks)}
v = ladder_report["zero_shot"]
print(
    f"zero-shot (SAM, one box) on {len(SPEC_EVAL)} held-out images · IoU {v['iou']:.3f} · clDice "
    f"{v['cldice']:.3f} · recall {v['recall']:.3f} · precision {v['precision']:.3f} · marks {v['marked']:.0%}"
)
show_extremes(sam_box_masks)

# %% [markdown]
# ### 4.2 Rung 1: one annotated image
#
# Someone spends an afternoon tracing every vessel of **one** training image. That complete mask is all
# this rung may use: the held-out ground truth stays for scoring. No training, just one mean per class:
#
# 1. Cut the annotated image into 16 px patches; DINOv3 gives one feature vector per patch.
# 2. Average the vessel patches into a vessel prototype, the background patches into a background prototype.
# 3. For a new image, score every patch: `cos(feature, vessel) − cos(feature, background)`.
# 4. Upsample the scores to pixels and mark those above 0.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# You saw SAM's box on the three images above (its "marked" share and its error maps). Before running the
# prototype, write one guess for each:
#
# - Will the prototype mark more of the fundus than SAM, or less?
# - Will it find the thin vessels, or only the thick ones?

# %% [markdown]
# <!-- callout: your-turn | Your turn: swap the annotated image -->
# - Run the next cell: it builds the two prototypes from one annotated image, scores every evaluation image
#   (seconds), and shows the similarity map and the mask on the demo image.
# - Then set `ONE_SHOT_IMAGE` to 1, 2, … and run it again, to answer Q12.

# %%
# Q12 — one-shot: the prototypes of one annotated image, applied to every held-out image
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
# **Q12.** 🟡 Look at the similarity map and the IoU you got:
#
# - Which parts of the held-out image look most like the annotated vessels (purple), and which like the
#   background (orange)?
# - Pick another annotated image: does the held-out IoU move much?
# - What does that say about relying on a single example?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The purple follows the thick vessels and the dark, textured parts of the fundus, and the mask is as wide
# as a 16 px patch: the prototype has the same resolution limit as the section-2 head. Compare the two IoUs
# printed above: one example and a mean, against five complete masks and a trained head. More labels do not
# automatically give a better mask.
#
# If the IoU moves when you swap the annotated image, this rung depends on *which* example you happened to
# annotate. With one image, a lucky or unlucky choice is the whole training set.
#
# </details>

# %% [markdown]
# ### 🔴 Optional: the same map, pointing SAM
#
# The same similarity map can also point SAM at the right places (rung 1, second variant). The cell below
# runs under "Run all" even if you skip the reading: its row appears in the table of 4.4. On every
# evaluation image:
#
# 1. The five patches most like a vessel become positive clicks.
# 2. The five most like background become negative clicks.
# 3. SAM turns the ten clicks into a mask (one SAM image encoding per image: seconds, also on a CPU).

# %%
# @title Prototype → SAM clicks on every evaluation image { display-mode: "form" }
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
# **Q13.** 🔴 Same annotated image, same similarity map, two ways to use it. Compare the prototype mask
# with the SAM mask:
#
# - Which one draws wider vessels, and which loses thin branches?
# - Did the "vessel" clicks land on vessels?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The prototype labels *every* patch, so it covers the whole tree at patch resolution. SAM only sees ten
# clicks and does what section 2 showed: it looks for the object that contains the positive ones and
# avoids the negative ones. Its edges are sharper, but it can lose whole branches that no click touched,
# or grow into a region around a thick vessel.
#
# If a "vessel" click landed on the optic disc or a dark spot, the mask follows it: SAM trusts its prompts.
# Two uses of the same information can fail in opposite directions.
#
# </details>

# %% [markdown]
# ### ☕ Before the break

# %% [markdown]
# <!-- callout: question | Question for the coffee break -->
# No code, just think about it over coffee.
#
# **Q14.** 🟡 You have 50 fundus images and a 4-hour annotation budget. Perfect labels cost 3 to 5 hours per
# image, and the cheap ones lie: the label file says "background" wherever nobody looked. Without buying more
# labels, what could you tell the training about a pixel nobody labelled? Think of at least two options before
# 4.3.
#
# <details><summary>A hint, for after the break</summary>
#
# Three families of answers come back in block 2:
#
# - Say "unknown" instead of "background", and let the loss skip those pixels (sections 4 and 5).
# - Guess the missing labels with another model, for example SAM prompted from the labels you have, and train
#   on the guesses (section 6).
# - Skip training and give the cheap labels to the foundation model directly as prompts (section 7).
#
# </details>

# %% [markdown]
# ### 4.3 Rung 2: scribbles
#
# Welcome back. The annotator from section 3 again, with a new instruction: "don't trace everything;
# draw a few strokes along some vessels and a few on clear background, in each of five images, then stop".
# Every other pixel is *unlabelled*, and this time the stroke map says so, so the loss can skip it.

# %% [markdown]
# Three label encodings meet here, and the number 0 means something different in each:
#
# | | 0 | 1 | 2 | −1 |
# |---|---|---|---|---|
# | Section-3 file | background (or just unpainted) | vessel | – | – |
# | Stroke map (this rung) | unlabelled | vessel stroke | background stroke | – |
# | Training targets | background | vessel | – | ignore this pixel |

# %% [markdown]
# The next cell shows the strokes on one training image (vessel in teal, background in grey, thickened for
# the figure only) and, in the last panel, what the loss will actually see.

# %%
# @title Show the scribbles of one training image { display-mode: "form" }
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
counted = ndi.binary_dilation(ex["scribbles"] > 0, iterations=2)  # thickened for display only
loss_sees = np.where(counted[..., None], strokes, (ex["image"] * 0.25).astype(np.uint8))
show_masks(
    [
        (ex["name"], ex["image"]),
        ("scribbles: vessel · background", strokes),
        ("what the loss sees (dark = ignored)", loss_sees),
        ("ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
    ],
    width=2.8,
)

# %% [markdown]
# <!-- callout: theory | Theory: partial cross-entropy -->
# The loss of section 2 adds a cross-entropy term for every pixel. The partial version adds it only over
# the set S of pixels a person labelled:
#
# `L = −(1/|S|) · Σ_{i∈S} [ y_i·log p_i + (1−y_i)·log(1−p_i) ]`
#
# - A vessel stroke says y = 1, a background stroke says y = 0.
# - An unlabelled pixel is not in S: it adds nothing. In code it is a target of `-1`, which `train_head`
#   already skips (it has skipped the fundus border that way since section 2).
# - In section 3 the unpainted pixels were targets of `0`: a thin vessel nobody painted counted as a
#   "background" error.

# %% [markdown]
# This cell holds our version of the function you can write below. Open it only if you get stuck.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_scribble_targets(scribbles: np.ndarray, region: np.ndarray, bg_beyond_px: float | None = None):
    """Reference: strokes as given, unlabelled ignored (or background when far from every vessel stroke)."""
    targets = np.full(scribbles.shape, -1, np.int8)
    if bg_beyond_px is not None:
        targets[ndi.distance_transform_edt(scribbles != 1) > bg_beyond_px] = 0
    targets[scribbles == 2] = 0
    targets[scribbles == 1] = 1
    targets[~np.asarray(region, bool)] = -1
    return targets


# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 3 min): choose which pixels to ignore -->
# Fill `my_scribble_targets`: it turns a stroke map into training targets (rows 2 and 3 of the table above).
#
# - In: 0 unlabelled, 1 vessel stroke, 2 background stroke. Out: 1, 0, or −1 for "ignore".
# - Pixels outside `region` are always −1.
# - Start with the default, `bg_beyond_px=None`: every unlabelled pixel is ignored.
# - **Ignore the number setting of `bg_beyond_px` until 4.6.** There, unlabelled pixels farther than that
#   from every vessel stroke count as background; `ndi.distance_transform_edt(scribbles != 1)` gives each
#   pixel's distance to the nearest vessel stroke.
#
# The cell checks your version on a 6 × 6 play map with both settings and continues with ours if you skip
# it or leave the number setting out.

# %%
# ✍️ YOUR CODE
def my_scribble_targets(scribbles: np.ndarray, region: np.ndarray, bg_beyond_px: float | None = None):
    """Targets (int8) from a scribble map: 1 vessel stroke, 0 background stroke, -1 ignored. Outside
    `region`: -1. If bg_beyond_px is a number, unlabelled pixels farther than bg_beyond_px from every vessel
    stroke become 0 (background); the others stay -1.
    e.g. on a 3×4 crop, same shape as `scribbles`: [[-1, -1, 1, -1], [0, -1, 1, -1], [0, 0, -1, -1]]"""
    ...


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
# <!-- callout: question | 🤔 Your prediction -->
# The head will see labels on fewer than two pixels in a hundred (see the printout above), and no vessel
# will ever be called "background". Against the one complete mask of rung 1, write two guesses:
#
# - Will it find more of the held-out vessels (recall), or fewer?
# - Will it mark more of the fundus, or less?

# %% [markdown]
# The next cell trains the section-2 head on the scribbles of the five training images, every unlabelled
# pixel ignored; with `TRAIN_YOURSELF = False` it loads our checkpoint, trained the same way on all fit
# images.

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
# @title Score the scribble head on the evaluation images { display-mode: "form" }
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
# The next cell puts the four rows (three rungs: rung 1 has two variants) side by side: a table with the
# five numbers and the annotation each needed, a bar chart, and the demo image's error maps (violet =
# vessel found, orange = false alarm, green = vessel missed).

# %%
# @title Table, bar chart and error maps of all rungs { display-mode: "form" }
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
# **Q15.** 🟡 Read the table and the error maps rung by rung:
#
# - Which rung has the lowest recall (misses the most, green)? Which has the lowest precision (invents the
#   most, orange)? Is it the rung with the lowest IoU?
# - You need a screening tool that must not miss a vessel, and a tool whose every mark must be right. Do you
#   pick the same rung for both?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The rungs do not line up on one "quality" axis. SAM's box tends to mark a region or the whole eye: high
# recall, many false alarms, an IoU near the grey line. The prototype labels by patch, so it draws vessels
# as wide as a square and confuses dark, textured background with vessel. SAM from prototype clicks is
# sharper but only covers what its clicks touched.
#
# The scribble head saw the fewest labelled pixels, yet from several images, and it was never told a vessel
# was background. Its errors are not the section-3 errors: compare its recall with the naive head's.
#
# Which rung is "best" depends on whether missing or inventing a vessel costs you more, and on the minutes
# in the last column. Your order may differ from your neighbour's: that is why we read the maps.
#
# </details>

# %% [markdown]
# ### 🔴 Optional: what changes from one rung to the next
#
# Several of those error maps look alike, so the next cell shows only what changes from one rung to the next
# on the demo image (grey = both mark it, a colour = only that one).

# %%
# @title Difference maps: what changes from one rung to the next { display-mode: "form" }
keys = list(ladder_preds)
fig, axes = plt.subplots(1, len(keys) - 1, figsize=(3.0 * (len(keys) - 1), 3.6))
for ax, a, b in zip(axes, keys[:-1], keys[1:], strict=True):
    pa, pb = ladder_preds[a][k_demo], ladder_preds[b][k_demo]
    reg = region(demo)
    differs = float(((pa ^ pb) & reg).sum() / reg.sum())
    ax.imshow(diff_map(demo["image"], pa, pb, a, b, demo))
    ax.set_title(f"{APPROACH_SHORT[a]} → {APPROACH_SHORT[b]}\n{differs:.0%} of fundus differs", fontsize=9)
    ax.axis("off")
    ax.legend(
        handles=[
            mpl.patches.Patch(color=APPROACH_COLOURS[a], label=f"only {APPROACH_SHORT[a]}"),
            mpl.patches.Patch(color=APPROACH_COLOURS[b], label=f"only {APPROACH_SHORT[b]}"),
            mpl.patches.Patch(color=THEME["background"], label="both"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=7,
        frameon=False,
        ncol=1,
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.5 A tempting shortcut: which unlabelled pixels to ignore
#
# Ignoring every unlabelled pixel is safe, and wasteful: over 98 % of the fundus says nothing. A tempting
# shortcut is "far from every vessel stroke, it's surely background". That is the second setting of your
# `scribble_targets`.

# %% [markdown]
# <!-- callout: pitfall | Pitfall: "surely background" is a guess -->
# 1. The strokes cover only some vessel segments. Say vessel B has no stroke.
# 2. B is more than 40 px from every stroke, so the rule calls it "background".
# 3. The loss now pushes the head to call B-like pixels background: the section-3 mistake, back through the
#    side door.

# %% [markdown]
# ### 4.6 🔴 Optional: choose a distance
#
# Want to see what the shortcut costs? The next cell tries it on the real scribbles.

# %% [markdown]
# <!-- callout: your-turn | Your turn: choose a distance -->
# - Set `BG_BEYOND_PX` (try 20, 80, `None` = ignore every unlabelled pixel) and run the next cell.
# - It turns the scribbles into targets with your distance, trains the same head on them (seconds), and
#   prints its numbers next to the scribble head of 4.3, which ignored every unlabelled pixel.
# - It also measures how much true vessel your rule mislabels (the hidden ground truth is used for that
#   diagnosis only, never for training).
# - Try a few distances before you answer Q16.

# %%
# Q16 — scribbles plus "far from every vessel stroke = background"
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
# what the rule did on one training image: its targets, and the true vessels it called background
tgt, ex17 = far_labels[0], scrib_train[0]
called_bg = (tgt == 0) & ex17["mask"]
show_masks(
    [
        (
            "targets: vessel · background · ignored",
            overlay(
                overlay(ex17["image"], tgt == 0, THEME["background"], 0.5),
                ndi.binary_dilation(tgt == 1, iterations=2),
                THEME["scribble"],
                0.9,
            ),
        ),
        (
            f"true vessels called background (orange): {called_bg.sum() / max(ex17['mask'].sum(), 1):.1%}",
            two_colour(ex17["image"], ex17["mask"] & ~called_bg, called_bg),
        ),
    ],
    width=3.8,
)

# %% [markdown]
# **Q16.** 🔴 Compare your distance with the scribble head of 4.3:
#
# - How many true vessel pixels did your rule call background?
# - Did recall, precision and IoU on the held-out images move, and in which direction?
# - Is there a distance that helps without hurting?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# A large distance adds a lot of background that really is background, and a little that is not; a small
# distance adds more of both. More "background" targets push the head to mark less, so watch recall fall
# (the direction of section 3) and check whether precision rose enough to pay for it. The line "vessel
# pixels called background" tells you the price.
#
# The lesson is not a number: every rule that fills in unlabelled pixels is an assumption about where the
# annotator did not look, and it is only as good as that assumption. Section 5 makes "unknown" explicit in
# the loss; section 6 fills unlabelled pixels with a second model's guesses instead of a distance.
#
# </details>

# %% [markdown]
# Want to try the one-shot idea on your own photo? Section 9 runs the whole ladder on your own images.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **Zero-shot IoU looks fine but the mask covers a whole region** → read the "marked" share and the error
#   map, not only IoU.
# - **You have one complete mask** → expect patch-wide vessels, and swap the annotated image to see how
#   much the result depends on your choice.
# - **You have cheap scribbles** → train with a loss that ignores unlabelled pixels, and measure what any
#   "far from a stroke = background" rule mislabels before you use it (4.5).
# - **Two rungs score alike** → compare their error maps and the minutes of annotation: they differ more
#   in how they fail than in one number. Keep the rung colours; they return in section 8.
