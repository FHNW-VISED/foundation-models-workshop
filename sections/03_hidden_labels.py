# %% [markdown]
# ## 3. Hide the labels: the naive baseline

# %% [markdown]
# <!-- callout: objective | Objective -->
# Train a head with "missing = background" and watch it fail differently.

# %% [markdown]
# A perfect vessel map of one fundus takes an expert 3 to 5 hours, so real label files are rarely complete.
#
# Picture an annotator who traced the big vessels, ran out of time halfway through the tree, and saved the
# file. Nothing in it says "I stopped here": every pixel they did not paint is stored as background, exactly
# like the pixels that really are background.
#
# We make such a file ourselves by hiding about half of the vessels from the ground truth (3.1), train the
# section-2 head on it the obvious way (3.3), and score it on the vessels we hid (3.2). From here on the full
# masks are used for scoring only.

# %% [markdown]
# <!-- callout: theory | Theory: incomplete supervision -->
# The cheap labels of the rest of the notebook are of two kinds, both leaving pixels nobody looked at:
# **missing labels** (this section and section 5) and **scribbles** (section 4).
#
# What should a model believe about those pixels? The obvious answer, and what a plain loss does, is
# "background".

# %% [markdown]
# This helper cell defines how we hide labels, the same way `scripts/prepare_data.py` built the cached files:
# cut the ground-truth skeleton into segments at its junctions, give every foreground pixel to its nearest
# segment, then hide a random share of the segments.

# %%
# @title Missing labels: cut into segments, hide a share { display-mode: "form" }
import zlib

from scipy import ndimage as ndi


def label_units(mask: np.ndarray, thin: bool) -> np.ndarray:
    """Label map of the units we hide (0 = background): skeleton segments for thin structures or a single
    object, connected components otherwise. Every foreground pixel gets the label of its nearest segment."""
    lab, n = ndi.label(mask)
    if not thin and n >= 2:
        return lab
    sk = skeleton(mask)
    neighbours = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    seg, _ = ndi.label(sk & ~(sk & (neighbours > 2)), structure=np.ones((3, 3)))  # cut at junctions
    if seg.max() == 0:
        return np.where(mask, 1, 0)
    _, (iy, ix) = ndi.distance_transform_edt(seg == 0, return_indices=True)
    return np.where(mask, seg[iy, ix], 0)


def hide_units(mask: np.ndarray, fraction: float, rng: np.random.Generator, thin: bool) -> np.ndarray:
    """The 'missing labels' file: the ground truth with ~fraction of its units removed. Returns the visible
    mask; the hidden foreground is mask & ~visible."""
    lab = label_units(mask, thin)
    ids = np.arange(1, int(lab.max()) + 1)
    rng.shuffle(ids)
    k = int(round(len(ids) * fraction))
    hidden_ids = ids[: max(1, k)] if len(ids) >= 2 else ids[:0]
    return mask & ~np.isin(lab, hidden_ids)


def hide_rng(name: str) -> np.random.Generator:
    """One fixed random generator per image, so the same segments are hidden on every run (N-013)."""
    return np.random.default_rng([SEED, zlib.crc32(f"{name}/hide".encode())])


def part_recall(preds: list[np.ndarray], items: list[dict], part: str) -> float:
    """Mean over images of the share of item[part] ("mask", "visible", "hidden"), inside the evaluation
    region, that the prediction marks. Uses `hidden_recall` from 3.2."""
    pairs = zip(preds, items, strict=True)
    return float(np.mean([hidden_recall(p, it[part] & region(it)) for p, it in pairs]))


def two_colour(
    image: np.ndarray, good: np.ndarray, bad: np.ndarray, bad_colour: str = THEME["fails"]
) -> np.ndarray:
    """Overlay `good` in violet ("works") and `bad` in `bad_colour` (default orange, "fails") on one image."""
    return overlay(overlay(image, good, THEME["works"]), bad, bad_colour)


# %% [markdown]
# ### 3.1 Make a half-finished label file
#
# This cell hides about half of the vessel segments (the share comes from the dataset config) of the five
# retina training images of section 2.5 and shows one of them. Two words from here on: **painted** vessels
# are the ones left in the label file, **hidden** vessels are in the full ground truth but the file leaves
# them unpainted, so it calls them background.

# %%
# @title Hide half of the vessel segments and look at the result { display-mode: "form" }
missing_labels = specialised["missing_labels"]  # {"hidden_fraction": 0.5, "unit": ...}: never hard-coded here
miss_train = [load_item(specialised, it) for it in specialised["fit"][:N_TRAIN_SPEC]]
same_as_cache = []
for it in miss_train:
    rng = hide_rng(it["name"])
    visible = hide_units(it["mask"], missing_labels["hidden_fraction"], rng, specialised["thin_structures"])
    same_as_cache.append(float((visible == it["visible"]).mean()))
    it["visible"], it["hidden"] = visible, it["mask"] & ~visible  # from now on: what training may see
hidden_share = float(np.mean([it["hidden"].sum() / max(it["mask"].sum(), 1) for it in miss_train]))
print(
    f"{len(miss_train)} training images · {hidden_share:.0%} of the vessel pixels hidden "
    f"(asked for {missing_labels['hidden_fraction']:.0%} of the segments) · pixels identical to the cached "
    f"labels our checkpoints used: {np.mean(same_as_cache):.2%}"
)

ex = miss_train[0]
show_masks(
    [
        (ex["name"], ex["image"]),
        ("full ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
        ("painted: what training may use", overlay(ex["image"], ex["visible"], THEME["scribble"])),
        ('hidden: the file says "background"', overlay(ex["image"], ex["hidden"], THEME["fails"])),
    ],
    width=2.8,
)

# %% [markdown]
# ### 3.2 Two numbers: hidden vessels found, and precision

# %% [markdown]
# <!-- callout: theory | Theory: hidden vessels found, and precision -->
# **Hidden vessels found** = of the hidden vessel pixels (3.1), the share the model marks as vessel. It is
# the usual *recall*, counted on the hidden pixels only. **Precision** = of the pixels the model marks, the
# share that really are vessel.
#
# - Hidden vessels found: 400 hidden vessel pixels, the model marks 120 of them: 120 ÷ 400 = 30%.
# - Precision: the model marked 150 pixels in total and 120 are vessels: 120 ÷ 150 = 80%.
#
# IoU mixes the two into one number. Here we want them apart, because the question is how many hidden
# vessels the model finds.

# %% [markdown]
# This helper cell defines `hidden_recall(pred, hidden)` (the share of the `hidden` pixels that the
# prediction marks, 0.0 when nothing is hidden), `marked_precision(pred, truth)`, and `hidden_panels`, the
# three-picture figure used in the rest of the section.


# %%
# @title hidden_recall, marked_precision and the "found / missed" figure { display-mode: "form" }
def hidden_recall(pred: np.ndarray, hidden: np.ndarray) -> float:
    """Share of the `hidden` pixels that `pred` marks as foreground; 0.0 when `hidden` is empty."""
    hidden = np.asarray(hidden, bool)
    return float(np.asarray(pred, bool)[hidden].mean()) if hidden.any() else 0.0


def marked_precision(pred: np.ndarray, truth: np.ndarray) -> float:
    """Share of the pixels `pred` marks that are foreground in `truth`; 0.0 when `pred` marks nothing."""
    pred = np.asarray(pred, bool)
    return float(np.asarray(truth, bool)[pred].mean()) if pred.any() else 0.0


def found_panel(item: dict, pred: np.ndarray, label: str = "") -> tuple[str, np.ndarray]:
    """One picture: painted vessels in teal, hidden vessels the model found in violet, missed in orange."""
    hidden = item["hidden"] & region(item)
    found, missed = hidden & pred, hidden & ~pred
    base = overlay(item["image"], item["visible"], THEME["scribble"])
    n = int(hidden.sum())
    title = f"{label}found {int(found.sum())} of {n} hidden px = {hidden_recall(pred, hidden):.0%}"
    return title, two_colour(base, found, missed)


def hidden_panels(item: dict, pred: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The three pictures: the label file (painted), the hidden vessels, and found / missed by `pred`."""
    base = overlay(item["image"], item["visible"], THEME["scribble"])
    hidden_pic = overlay(base, item["hidden"] & region(item), THEME["fails"])
    return [("label file: painted vessels", base), ("hidden: the file says background", hidden_pic)] + [
        found_panel(item, pred)
    ]


# %% [markdown]
# A first example, with no training: the "believer" is a fake model that believes the label file perfectly,
# so its prediction is the file itself. Colours in every picture of this section: **teal** = painted
# vessels, **orange** = hidden vessels (in the third picture: hidden and missed), **violet** = hidden and
# found by the model.

# %%
# @title Worked example: a model that copies the label file { display-mode: "form" }
believer = [it["visible"] for it in miss_train]  # its prediction is the file itself
b_rec_hidden = part_recall(believer, miss_train, "hidden")
b_rec_visible = part_recall(believer, miss_train, "visible")
b_pairs = list(zip(believer, miss_train, strict=True))
b_precision = float(np.mean([marked_precision(b & region(it), it["mask"]) for b, it in b_pairs]))
b_iou = float(np.mean([iou(b & region(it), it["mask"] & region(it)) for b, it in b_pairs]))
print(
    f"believer · hidden vessels found {b_rec_hidden:.0%} · painted vessels found {b_rec_visible:.0%} · "
    f"precision {b_precision:.0%} · IoU {b_iou:.2f}"
)
show_masks(hidden_panels(miss_train[0], believer[0]), width=2.9)

# %% [markdown]
# ### 3.3 The naive baseline: missing = background
#
# "Naive" means: pass the label file straight into the textbook loss, with no thought about what the
# unpainted pixels mean. Here is what that does to our file.

# %% [markdown]
# <!-- callout: theory | Theory: why missing = background goes wrong -->
# 1. The annotator painted about half of the vessels and stopped (the label file of 3.1).
# 2. The simplest training reads every unpainted pixel as "background", hidden vessels included.
# 3. So the head sees two vessel patches that look the same: one labelled "vessel", one labelled
#    "background" (the picture below shows such a pair).
# 4. The head must give both patches one answer. Which answer will it settle on? Guess below, then train it.
#
# What we measure: the hidden vessels found (3.2), next to the section-2 head's.

# %% [markdown]
# This cell shows two look-alike patches from a training image, one the annotator painted (target 1) and one
# they never reached (target 0), which is all the head sees of the problem.

# %%
# @title Picture: two look-alike patches, two targets { display-mode: "form" }
pic_side = int(2 * 16 * max(miss_train[0]["mask"].shape) / specialised["feat_side"])  # 2 × 2 patches, in px
pic_half = pic_side // 2
pic_pair = None
for pic_item in miss_train:
    pic_gray = pic_item["image"].astype(float).mean(axis=2)
    pic_h, pic_w = pic_gray.shape
    cands = []
    for pic_part in ("hidden", "visible"):
        pys, pxs = np.nonzero(skeleton(pic_item[pic_part]))
        inside = (pys >= pic_half) & (pys < pic_h - pic_half) & (pxs >= pic_half) & (pxs < pic_w - pic_half)
        cands.append(np.stack([pys[inside], pxs[inside]], 1))
    if len(cands[0]) and len(cands[1]):
        pic_pair = (pic_item, cands)
        break
if pic_pair is None:
    print("no hidden and visible vessel segment far enough from the border to draw the picture")
else:
    pic_item, (pic_hid, pic_vis) = pic_pair
    pic_gray = pic_item["image"].astype(float).mean(axis=2)

    def pic_crop(arr: np.ndarray, centre: np.ndarray) -> np.ndarray:
        return arr[centre[0] - pic_half : centre[0] + pic_half, centre[1] - pic_half : centre[1] + pic_half]

    hid_centre = pic_hid[len(pic_hid) // 2]
    vis_pool = pic_vis[:: max(1, len(pic_vis) // 300)]
    gaps = [float(np.abs(pic_crop(pic_gray, c) - pic_crop(pic_gray, hid_centre)).mean()) for c in vis_pool]
    vis_centre = vis_pool[int(np.argmin(gaps))]  # the visible patch that looks most like the hidden one
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.4))
    for ax, centre, title, colour in (
        (axes[0], vis_centre, "target 1: painted", THEME["scribble"]),
        (axes[1], hid_centre, "target 0: never reached", THEME["fails"]),
    ):
        ax.imshow(pic_crop(pic_item["image"], centre))
        ax.set_title(title, fontsize=9, color=colour)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)
    fig.suptitle("same features, two targets: the head can only say p ≈ 0.5", fontsize=9)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The head is told that about half of the vessels in these five images are background. Write two guesses.
#
# - Of the hidden vessel pixels, how many will it find? Near the section-2 head's share, about half of it, or
#   close to zero (like the believer of 3.2)?
# - On new images, will it mark more pixels than the section-2 head, or fewer?

# %% [markdown]
# This helper cell defines `fit_naive`: it reads the label file as it is (no pixel is ignored except the
# fundus border, as in 2.5) and trains the head with the plain loss, or loads our checkpoint, trained the
# same way on all fit images.


# %%
# @title fit_naive: train the naive head, or load our checkpoint { display-mode: "form" }
def fit_naive(train_yourself: bool) -> tuple[dict, float, list[torch.Tensor], list[np.ndarray]]:
    """Naive head on the label file: returns (head, seconds, features, labels). The hidden ground truth never
    enters training (L-005)."""
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_NAIVE_HEAD must be True or False.")
    side = specialised["feat_side"]
    feats = [features_for(specialised, it, side) for it in miss_train]
    labels = [np.where(region(it), it["visible"], -1).astype(np.int8) for it in miss_train]  # unpainted = 0
    if train_yourself:
        head, sec = train_head(feats, labels, HEAD_STEPS, balanced=False)
        src = f"trained here on {len(miss_train)} images, {HEAD_STEPS} steps, plain cross-entropy"
    else:
        head, entry = load_head(specialised, "hidden_bg", side)
        sec = entry["train_sec"]
        src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
    print(f"naive head @ {side} px · {src} · {sec:.1f} s on {DEVICE}")
    if sec > 60:
        print("ℹ️ Slow session? Untick TRAIN_NAIVE_HEAD above to load our checkpoint next time.")
    return head, sec, feats, labels


# %% [markdown]
# This cell trains the naive head and prints the seconds it took; untick `TRAIN_NAIVE_HEAD` and run it again
# to load our checkpoint.

# %%
# @title Train the naive head { display-mode: "form" }
TRAIN_NAIVE_HEAD = True  # @param {type:"boolean"}
side = specialised["feat_side"]
naive_head, naive_sec, miss_feats, naive_labels = fit_naive(TRAIN_NAIVE_HEAD)

# %% [markdown]
# This cell scores the naive head and the section-2 head (full masks) on the vessel pixels we hid, the ones
# training saw, and every vessel of the held-out images, and plots them next to the believer's.

# %%
# @title Score both heads on hidden, painted and held-out vessels { display-mode: "form" }
full_head = spec_heads[side][0]  # section 2.5: same features, trained on full masks
compared_heads = {"full masks (section 2)": full_head, "naive: missing = background": naive_head}
t0 = time.monotonic()
naive_report = {}
for name, tensors in compared_heads.items():
    fit_preds = [
        predict_head(tensors, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
    ]
    held_preds = [
        predict_head(tensors, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
    ]
    held = mean_scores([score(p, it) for p, it in zip(held_preds, spec_eval, strict=True)])
    held_pairs = zip(held_preds, spec_eval, strict=True)
    precision = [marked_precision(p & region(it), it["mask"]) for p, it in held_pairs]
    naive_report[name] = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
        "recall_held": part_recall(held_preds, spec_eval, "mask"),
        "precision_held": float(np.mean(precision)),
        "iou_held": held["iou"],
        "cldice_held": held["cldice"],
        "fit_preds": fit_preds,
        "held_preds": held_preds,
    }
infer_sec = round(time.monotonic() - t0, 1)

naive = naive_report["naive: missing = background"]
RESULTS["naive_recall_hidden_fg"] = naive["recall_hidden_fit"]  # recall on the vessels hidden in training
RESULTS["naive_head"] = {k: v for k, v in naive.items() if not k.endswith("_preds")} | {
    "long_side": side,
    "n_train": len(miss_train),
    "hidden_share": hidden_share,
    "train_sec": round(float(naive_sec), 2),
    "infer_sec": infer_sec,
}

GROUPS = {
    "recall_hidden_fit": "hidden vessels\n(told: background)",
    "recall_visible_fit": "painted vessels\n(told: vessel)",
    "recall_held": "held-out images\n(all vessels)",
}
print(f"{'head':<30}{'hidden found':>13}{'painted':>9}{'held-out':>10}   held-out IoU / clDice / precision")
for name, r in naive_report.items():
    print(
        f"{name:<30}{r['recall_hidden_fit']:>13.3f}{r['recall_visible_fit']:>9.3f}{r['recall_held']:>10.3f}"
        f"   {r['iou_held']:.3f} / {r['cldice_held']:.3f} / {r['precision_held']:.3f}"
    )

x = np.arange(len(GROUPS))
fig, ax = plt.subplots(figsize=(6.4, 3))
# the believer (its prediction is the file) only exists on the training images: no bar for the held-out group
believer_bars = {0: b_rec_hidden, 1: b_rec_visible}
ax.bar(
    [x[i] - 0.27 for i in believer_bars],
    list(believer_bars.values()),
    0.26,
    color=THEME["background"],
    label="believer: the file itself",
)
for shift, name, colour in zip((0.0, 0.27), compared_heads, (THEME["works"], THEME["fails"]), strict=True):
    ax.bar(x + shift, [naive_report[name][g] for g in GROUPS], 0.26, color=colour, label=name)
ax.set_xticks(x, list(GROUPS.values()), fontsize=8)
ax.set_ylabel("share of vessel pixels found")
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("Which vessels does each head find?")
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: theory | Theory: why the head settles there -->
# 1. For two identical patches with opposite labels, the loss is lowest when the head answers "about 50 %
#    vessel" for both.
# 2. 50 % sits on the threshold, so faint and thin vessels fall just below it and are marked as background.
# 3. The head misses vessels, even painted ones. What it does mark is mostly the clearest vessels, so it is
#    mostly right: it looks precise.
#
# Check this against your guess and the numbers above.

# %% [markdown]
# This cell shows the naive head's worst, typical (median) and best training image by hidden vessels found:
# violet is a hidden vessel pixel it found, orange one it missed, teal what was painted.

# %%
# @title Worst, typical and best training image by hidden vessels found { display-mode: "form" }
per_image = [hidden_recall(p, it["hidden"]) for p, it in zip(naive["fit_preds"], miss_train, strict=True)]
rank = np.argsort(per_image)
show_masks(
    [
        found_panel(miss_train[k], naive["fit_preds"][k], f"{label} · ")
        for label, k in (("worst", rank[0]), ("typical (median)", rank[len(rank) // 2]), ("best", rank[-1]))
    ],
    width=3.4,
)

# %% [markdown]
# This helper cell defines `show_naive_on_image(k)`: it draws both heads' predictions on held-out image `k`
# (no new training, no model calls), each titled with the share of the fundus it marked and its IoU.


# %%
# @title show_naive_on_image: both heads on one held-out image { display-mode: "form" }
def show_naive_on_image(k: int) -> None:
    """Both heads' predictions on spec_eval[k] next to its ground truth, with the share of fundus marked."""
    if not isinstance(k, int) or not 0 <= k < len(spec_eval):
        raise SystemExit(f"DEMO_IMAGE must be a whole number from 0 to {len(spec_eval) - 1}.")
    item = spec_eval[k]
    panels = []
    for name, r in naive_report.items():
        pred = r["held_preds"][k]
        marked = pred[region(item)].mean()
        panels.append(
            (
                f"{name}\nmarked {marked:.0%} · IoU {score(pred, item)['iou']:.2f}",
                overlay(item["image"], pred, THEME["fails"]),
            )
        )
    panels.append(("ground truth", overlay(item["image"], item["mask"], THEME["truth"])))
    show_masks(panels, width=3.0)


# %% [markdown]
# <!-- callout: question | Question -->
# **Q12.** 🟡 Look at the worst, typical and best images above: which hidden vessels did the naive head
# miss (orange), and which did it still find (violet)? Then run the next cell, which draws the naive head's
# predictions on a held-out image next to the section-2 head's (image 0 is the demo image of section 2; set
# `DEMO_IMAGE` to look at others): does it fail the same way (too wide, the disc, the whole eye) or
# differently?

# %%
# @title Q12 — the naive head on a held-out image { display-mode: "form" }
DEMO_IMAGE = 0  # @param {type:"integer"}
show_naive_on_image(DEMO_IMAGE)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - The naive head did what the labels told it: it kept the clearest vessels, on the hidden segments, on new
#   images, and often even on the painted segments it was shown.
# - Compare the precision column: what it does mark is mostly right. The section-2 head failed by being too
#   generous (vessels as wide as a patch); this one fails by being too stingy.
# - Look at the held-out IoU of the two heads: it can be close, because IoU rewards the precision the naive
#   head gained as much as it punishes the vessels it lost. One number would have hidden the failure.
# - Same features, same head: the only thing that changed is what we told it about the unpainted pixels.
#
# </details>

# %% [markdown]
# <!-- callout: go-deeper | Pitfall: a precise model looks trustworthy -->
# Every vessel the naive head marks is a real vessel, so a quick look at a few predictions, or a precision
# score, says "works". The failure is in what is *not* marked, and you can only measure that against labels
# that are complete.
#
# - On your own data, keep a small, fully labelled evaluation set, even if training uses cheap labels.

# %% [markdown]
# **Try it: hide more or fewer vessels.** The next helper rebuilds the label file with a different share of
# hidden segments, trains the naive head again on it (a few seconds), and prints the result next to the
# default one.

# %%
# @title naive_with_hidden_share: rebuild the label file, retrain, score { display-mode: "form" }
_HELD_FEATS: dict = {}


def naive_with_hidden_share(percent: int) -> None:
    """Hide `percent` % of the vessel segments (the 3.1 helper), train the plain head on that label file
    (the 3.3 helper), and print hidden vessels found and held-out precision next to the default run."""
    if not isinstance(percent, int) or not 0 <= percent <= 90:
        raise SystemExit("HIDE_PERCENT must be a whole number from 0 to 90.")
    thin = specialised["thin_structures"]
    visibles = [
        it["mask"] if percent == 0 else hide_units(it["mask"], percent / 100, hide_rng(it["name"]), thin)
        for it in miss_train
    ]
    labels = [np.where(region(it), v, -1).astype(np.int8) for it, v in zip(miss_train, visibles, strict=True)]
    head, sec = train_head(miss_feats, labels, HEAD_STEPS, balanced=False)
    trial = [dict(it, visible=v, hidden=it["mask"] & ~v) for it, v in zip(miss_train, visibles, strict=True)]
    fit_preds = [predict_head(head, f, it["mask"].shape) for f, it in zip(miss_feats, trial, strict=True)]
    held_preds = []
    for it in spec_eval:
        if it["name"] not in _HELD_FEATS:
            _HELD_FEATS[it["name"]] = features_for(specialised, it, specialised["feat_side"])
        held_preds.append(predict_head(head, _HELD_FEATS[it["name"]], it["mask"].shape))
    shown = next((it for it in trial if it["hidden"].any()), None)  # first image that has hidden vessels
    has_hidden = shown is not None
    rows = {
        f"default ({missing_labels['hidden_fraction']:.0%} of segments)": (
            naive["recall_hidden_fit"],
            naive["recall_visible_fit"],
            naive["precision_held"],
        ),
        f"this run ({percent}% of segments)": (
            part_recall(fit_preds, trial, "hidden") if has_hidden else float("nan"),
            part_recall(fit_preds, trial, "visible"),
            float(
                np.mean(
                    [
                        marked_precision(p & region(it), it["mask"])
                        for p, it in zip(held_preds, spec_eval, strict=True)
                    ]
                )
            ),
        ),
    }
    share = np.mean([it["hidden"].sum() / max(it["mask"].sum(), 1) for it in trial])
    print(f"trained in {sec:.1f} s on {DEVICE} · {share:.0%} of the vessel pixels hidden")
    print(f"{'':<28}{'hidden found':>13}{'painted found':>15}{'held-out precision':>20}")
    for name, (hid, vis, prec) in rows.items():
        print(f"{name:<28}{hid:>13.2f}{vis:>15.2f}{prec:>20.2f}")
    if has_hidden:
        show_masks(hidden_panels(shown, fit_preds[trial.index(shown)]), width=2.9)
    else:
        print(
            "Nothing is hidden at 0%, so there is no 'hidden found' to report: the head sees the full masks."
        )


# %% [markdown]
# <!-- callout: question | Question -->
# **Q13.** 🟡 Write down a guess first: how will "hidden found" and "precision" change as you hide more of
# the vessels? Then run the next cell with `HIDE_PERCENT` at 10, 50 and 90. For each of the two numbers,
# say what happened, and what it means for an annotator who stops early.

# %%
# @title Q13 — hide more or fewer vessels, retrain the naive head { display-mode: "form" }
HIDE_PERCENT = 70  # @param {type:"slider", min:0, max:90, step:10}
naive_with_hidden_share(HIDE_PERCENT)

# %% [markdown]
# ### 3.4 🔴 Optional: is it the labels, or the loss?
#
# Plain cross-entropy is one of the two changes from section 2.5: the heads of sections 1 and 2 used the
# balanced loss.

# %% [markdown]
# **Q14.** 🔴 Train the same naive labels with the balanced loss of section 2.5 (`balanced=True` in
# `train_head`) and compare its hidden vessels found with the plain head's. Does reweighting fix
# "missing = background"? What did it fix?
#
# The next cell trains that balanced variant on the same incomplete labels (seconds) and prints both.


# %%
# @title compare_balanced_loss: the balanced head next to the plain one { display-mode: "form" }
def compare_balanced_loss() -> dict:
    """Train the balanced head on the naive labels, print its recalls next to the plain head's, return it."""
    head, sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=True)
    preds = [predict_head(head, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)]
    hidden, painted = (part_recall(preds, miss_train, part) for part in ("hidden", "visible"))
    RESULTS["naive_balanced_recall_hidden_fg"] = hidden
    print(f"balanced head · trained in {sec:.1f} s on {DEVICE}")
    print(f"{'loss':<12}{'hidden found':>13}{'painted':>9}")
    print(f"{'plain':<12}{naive['recall_hidden_fit']:>13.3f}{naive['recall_visible_fit']:>9.3f}")
    print(f"{'balanced':<12}{hidden:>13.3f}{painted:>9.3f}")
    return head


# %%
# @title Q14 — 🔴 the same "missing = background" labels with the balanced loss { display-mode: "form" }
balanced_head = compare_balanced_loss()

# %% [markdown]
# <!-- callout: go-deeper | Q14: what to look for -->
# - Reweighting usually recovers a lot of the hidden vessels. It does not fix the labels, it fixes the
#   *proportion*: missing = background told the head that vessels are about half as common as they are, and
#   the balanced loss stops caring about proportions.
# - It works here for two reasons that need not hold on your data: we hid segments *at random*, so the hidden
#   ones look like the painted ones, and a linear head is too simple to memorise which segment was hidden.
# - If annotators systematically skip the thin capillaries, the head learns "thin = background", and no class
#   weight undoes that.
# - Section 5 fixes the cause instead: tell the loss which pixels are unknown.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **A label file with only vessel and background** → ask how it was made: an unpainted pixel may mean
#   "unknown", not "background".
# - **A head trained with "missing = background"** → expect a stingy model (precision stays high, the vessels
#   left out are mostly not found), so also measure the hidden vessels found on a small, fully labelled set.
# - **A balanced loss that wins the recall back** → do not trust it for systematic gaps: it works when the
#   gaps are random, not when thin vessels are skipped.
# - **Three failures so far** (SAM finds the eye, the head is limited by its 16 px patches, naive training
#   believes the file) → none of these is a model-size problem we have tested; each needs different
#   supervision. The hidden vessels found in this section is the baseline to beat; section 4 starts on how.
