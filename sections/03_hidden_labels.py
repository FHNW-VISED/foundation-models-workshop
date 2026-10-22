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
# We make such a file ourselves by hiding about half of the vessels from the ground truth, train the
# section-2 head on it the obvious way, and score it on the vessels we hid. From here on the full masks are
# used for scoring only.

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
# retina training images of section 2.5 and shows one of them: the full truth we keep for scoring, the labels
# training may use, and what the label file now wrongly calls background.

# %%
# @title Hide half of the vessel segments and look at the result { display-mode: "form" }
MISSING = SPECIALISED["missing_labels"]  # {"hidden_fraction": 0.5, "unit": ...}: never hard-coded here
miss_train = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:N_TRAIN_SPEC]]
same_as_cache = []
for it in miss_train:
    rng = hide_rng(it["name"])
    visible = hide_units(it["mask"], MISSING["hidden_fraction"], rng, SPECIALISED["thin_structures"])
    same_as_cache.append(float((visible == it["visible"]).mean()))
    it["visible"], it["hidden"] = visible, it["mask"] & ~visible  # from now on: what training may see
hidden_share = float(np.mean([it["hidden"].sum() / max(it["mask"].sum(), 1) for it in miss_train]))
print(
    f"{len(miss_train)} training images · {hidden_share:.0%} of the vessel pixels hidden "
    f"(asked for {MISSING['hidden_fraction']:.0%} of the segments) · pixels identical to the cached labels "
    f"our checkpoints used: {np.mean(same_as_cache):.2%}"
)

ex = miss_train[0]
show_masks(
    [
        (ex["name"], ex["image"]),
        ("ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
        ("labels training sees: vessel", overlay(ex["image"], ex["visible"], THEME["scribble"])),
        ('hidden: the file says "background"', overlay(ex["image"], ex["hidden"], THEME["fails"])),
    ],
    width=2.8,
)

# %% [markdown]
# **Q10.** 🟡 Look at the third panel on its own. The line above says what share of the vessel pixels is
# hidden. If a colleague handed you that label file, could you tell that this much is missing? What would you
# need to know to tell?
#
# <details><summary>What to look for</summary>
#
# - Usually you cannot. The visible vessels are clean and correct, they simply stop, and a vessel tree that
#   stops looks plausible.
# - The file has two values, vessel and background, and no third value for "nobody looked here".
# - To tell, you would need to know how the file was made (who annotated, for how long, what they were told
#   to skip): exactly the information that gets lost.
# - That is why we hid the labels on purpose: here we know which pixels are missing, so we can measure what
#   the model does with them.
#
# </details>

# %% [markdown]
# ### 3.2 A metric for the missing vessels: recall

# %% [markdown]
# <!-- callout: theory | Theory: recall and precision -->
# **Recall** = found ÷ there, **precision** = right ÷ marked; IoU mixes them, and here we want them apart
# (our question is recall on the hidden vessels).
#
# - Recall: 400 hidden vessel pixels, the prediction marks 120 of them: 120 ÷ 400 = 30%.
# - Precision: it marked 150 pixels and 120 are vessels: 120 ÷ 150 = 80%.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 1 min) -->
# Fill `my_hidden_recall`: a boolean prediction and a boolean mask of hidden pixels in, one float out.
#
# - The share of the hidden pixels that the prediction marks.
# - Return 0.0 when nothing is hidden.
# - The play example has 8 hidden pixels and the prediction finds 2 of them: expect 0.25.
#
# The cell checks your version on two play masks and continues with ours if you skip it. Stuck? Open the
# collapsed "Reference solution" cell above it.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_hidden_recall(pred: np.ndarray, hidden: np.ndarray) -> float:
    """Reference: the prediction's mean over the hidden pixels."""
    hidden = np.asarray(hidden, bool)
    return float(np.asarray(pred, bool)[hidden].mean()) if hidden.any() else 0.0


# %%
# ✍️ YOUR CODE
def my_hidden_recall(pred: np.ndarray, hidden: np.ndarray) -> float:
    """Share of the `hidden` pixels that `pred` marks as foreground; 0.0 when `hidden` is empty.
    e.g. 0.38 (38 % of the hidden vessel pixels found)"""
    ...


_play_hidden = np.zeros((8, 8), bool)
_play_hidden[2, :] = True  # a hidden vessel, 8 px long
_play_pred = np.zeros((8, 8), bool)
_play_pred[1:4, :2] = True  # the prediction finds its first 2 px, plus some background
hidden_recall = use(
    my_hidden_recall,
    reference=_ref_hidden_recall,
    check=lambda f: [round(float(f(_play_pred, h)), 4) for h in (_play_hidden, np.zeros((8, 8), bool))],
    name="hidden_recall",
)
print(f"play example · recall on the hidden vessel: {hidden_recall(_play_pred, _play_hidden):.2f}")

# %% [markdown]
# ### 3.3 The naive baseline: missing = background
#
# The naive recipe is what you get by passing the label file straight into the textbook loss: every pixel not
# painted as vessel is a background target, and plain binary cross-entropy counts every pixel equally.
#
# Two things change from section 2.5, the labels and the loss weighting. The hidden labels cause the gap in
# the chart below; the plain loss only makes the head believe them more strictly, and 3.4 separates the two.
# Everything else is the same: the same five images, DINOv3 features, linear head and steps.

# %% [markdown]
# <!-- callout: theory | Theory: the linear head in one line -->
# For every patch, the head computes one probability from its 384 DINOv3 numbers:
#
# **p = σ(w · f + b)**
#
# - `f`: the 384 (standardised) numbers of one patch.
# - `w`, `b`: 384 weights and one bias (385 learned numbers), the **same for every patch**: a 1 × 1
#   convolution.
# - `σ`: squeezes the score into 0 to 1. Example: w · f + b = 2.2 gives p = 0.90, and w · f + b = 0 gives 0.5.
# - The per-patch map is upsampled to the image, and p > 0.5 means "vessel".

# %% [markdown]
# This cell runs the section-2 head on one training image and shows each step (patches, one probability per
# patch, the upsampled map, the final mask), then works through one patch by hand.

# %%
# @title The linear head step by step on one image { display-mode: "form" }
hside = SPECIALISED["feat_side"]
hx = miss_train[0]
hparams = spec_heads[hside][0]  # section 2.5: trained on full masks
hfeat = features_for(SPECIALISED, hx, hside)  # [384, h, w]: 384 numbers per patch
dev = hfeat.device
with torch.no_grad():
    hz = (hfeat - hparams["feat_mean"].to(dev).view(-1, 1, 1)) / hparams["feat_std"].to(dev).view(-1, 1, 1)
    hscore = F.conv2d(hz[None], hparams["weight"].to(dev), hparams["bias"].to(dev))[0, 0]  # w·f + b per patch
    hprob = torch.sigmoid(hscore)
    hup = F.interpolate(hprob[None, None], size=hx["mask"].shape, mode="bilinear", align_corners=False)[0, 0]
gh, gw = hprob.shape
py, px = np.unravel_index(int(hprob.argmax()), hprob.shape)  # the most vessel-like patch
f_vec, w_vec = hz[:, py, px], hparams["weight"].reshape(-1).to(dev)
f_head = ", ".join(f"{v:.2f}" for v in f_vec[:4].tolist())
dot, bias = float((w_vec * f_vec).sum()), float(hparams["bias"])
total, p_patch = float(hscore[py, px]), float(hprob[py, px])
print(f"patch (row {py}, col {px}) · f = [{f_head}, … 384 numbers]")
print(f"w · f + b = {dot:.2f} + ({bias:.2f}) = {total:.2f} → p = σ({total:.2f}) = {p_patch:.2f}")

prob_cmap = mpl.colors.LinearSegmentedColormap.from_list("prob", ["#FFFFFF", THEME["works"]])
H, W = hx["mask"].shape
fig, axes = plt.subplots(1, 4, figsize=(11.6, 3))
axes[0].imshow(hx["image"])
axes[0].vlines(np.arange(0, W + 1, W / gw), 0, H, colors="white", linewidths=0.4)
axes[0].hlines(np.arange(0, H + 1, H / gh), 0, W, colors="white", linewidths=0.4)
cell_box = plt.Rectangle((px * W / gw, py * H / gh), W / gw, H / gh, fill=False, ec=THEME["fails"], lw=2)
axes[0].add_patch(cell_box)
axes[0].set_title("1 · image cut into patches", fontsize=9)
axes[1].imshow(hprob.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1, interpolation="nearest")
axes[1].add_patch(plt.Rectangle((px - 0.5, py - 0.5), 1, 1, fill=False, ec=THEME["fails"], lw=2))
axes[1].set_title("2 · one p per patch (white 0, violet 1)", fontsize=9)
axes[2].imshow(hup.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1)
axes[2].set_title("3 · upsampled to the image", fontsize=9)
axes[3].imshow(overlay(hx["image"], predict_head(hparams, hfeat, hx["mask"].shape), THEME["fails"]))
axes[3].set_title("4 · p > 0.5 → vessel", fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# A playful extreme first, to have a scale: this cell scores a fake model that believes the label file
# perfectly (its prediction is the file itself), and it comes back as the grey bars in the chart below.

# %%
# @title Playful test: a model that believes the label file perfectly { display-mode: "form" }
believer = [it["visible"] for it in miss_train]  # its prediction is the file itself
b_rec_hidden = part_recall(believer, miss_train, "hidden")
b_rec_visible = part_recall(believer, miss_train, "visible")
b_pairs = list(zip(believer, miss_train, strict=True))
b_precision = float(np.mean([hidden_recall(it["mask"], b & region(it)) for b, it in b_pairs]))
b_iou = float(np.mean([iou(b & region(it), it["mask"] & region(it)) for b, it in b_pairs]))
print(
    f"believes the file perfectly · recall on hidden vessels {b_rec_hidden:.0%} · on visible vessels "
    f"{b_rec_visible:.0%} · precision {b_precision:.0%} · IoU {b_iou:.2f}"
)

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The head was told that about half of the vessels in these five images are background. Write two guesses.
#
# - On those hidden vessels, will its recall be near the section-2 head's, about half of it, or close to
#   zero (like the believer above)?
# - On new images, will it mark more pixels than the section-2 head, or fewer?

# %% [markdown]
# This cell trains the naive head on the incomplete labels only (the fundus border is ignored, as in 2.5) and
# prints the seconds it took; if it is slow, set `TRAIN_YOURSELF = False` and run it again to load our
# checkpoint, trained the same way on all fit images.

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "hidden_bg" head instead of training
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

side = SPECIALISED["feat_side"]
miss_feats = [features_for(SPECIALISED, it, side) for it in miss_train]
# missing = background: the label file as it is. The hidden ground truth never enters training (L-005).
naive_labels = [np.where(region(it), it["visible"], -1).astype(np.int8) for it in miss_train]
if TRAIN_YOURSELF:
    naive_head, naive_sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=False)
    src = f"trained here on {len(miss_train)} images, {HEAD_STEPS} steps, plain cross-entropy"
else:
    naive_head, entry = load_head(SPECIALISED, "hidden_bg", side)
    naive_sec = entry["train_sec"]
    src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
print(f"naive head @ {side} px · {src} · {naive_sec:.1f} s on {DEVICE}")
if naive_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the naive head and the section-2 head (full masks) on the vessel pixels we hid, the ones
# training saw, and every vessel of the held-out images, and plots the recalls next to the believer's.

# %%
# @title Score both heads on hidden, visible and held-out vessels { display-mode: "form" }
full_head = spec_heads[side][0]  # section 2.5: same features, trained on full masks
HEADS = {"full masks (section 2)": full_head, "naive: missing = background": naive_head}
t0 = time.monotonic()
naive_report = {}
for name, tensors in HEADS.items():
    fit_preds = [
        predict_head(tensors, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
    ]
    held_preds = [
        predict_head(tensors, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
    ]
    held = mean_scores([score(p, it) for p, it in zip(held_preds, SPEC_EVAL, strict=True)])
    # precision = the share of the marked pixels that are true vessels
    held_pairs = zip(held_preds, SPEC_EVAL, strict=True)
    precision = [hidden_recall(it["mask"], p & region(it)) for p, it in held_pairs]
    naive_report[name] = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
        "recall_held": part_recall(held_preds, SPEC_EVAL, "mask"),
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
    "recall_visible_fit": "visible vessels\n(told: vessel)",
    "recall_held": "held-out images\n(all vessels)",
}
print(f"{'head':<30}{'recall: hidden':>15}{'visible':>9}{'held-out':>10}   held-out IoU / clDice / precision")
for name, r in naive_report.items():
    print(
        f"{name:<30}{r['recall_hidden_fit']:>15.3f}{r['recall_visible_fit']:>9.3f}{r['recall_held']:>10.3f}"
        f"   {r['iou_held']:.3f} / {r['cldice_held']:.3f} / {r['precision_held']:.3f}"
    )

x = np.arange(len(GROUPS))
fig, ax = plt.subplots(figsize=(6.4, 3))
# the believer (its prediction is the file) only exists on the training images: no bar for the held-out group
believer_bars = {0: b_rec_hidden, 1: b_rec_visible}
ax.bar(
    [x[i] - 0.27 for i in believer_bars], list(believer_bars.values()), 0.26, color=THEME["background"],
    label="believer: the file itself",
)
for shift, name, colour in zip((0.0, 0.27), HEADS, (THEME["works"], THEME["fails"]), strict=True):
    ax.bar(x + shift, [naive_report[name][g] for g in GROUPS], 0.26, color=colour, label=name)
ax.set_xticks(x, list(GROUPS.values()), fontsize=8)
ax.set_ylabel("recall (share of vessel pixels found)")
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("Which vessels does each head still find?")
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)
plt.tight_layout()
plt.show()

# %% [markdown]
# This cell shows the naive head's worst, typical (median) and best training image by recall on the hidden
# vessels: violet is a hidden vessel pixel it still found, green one it missed.

# %%
# @title Worst, typical and best training image by hidden recall { display-mode: "form" }
per_image = [hidden_recall(p, it["hidden"]) for p, it in zip(naive["fit_preds"], miss_train, strict=True)]
rank = np.argsort(per_image)
panels = []
for label, k in (("worst", rank[0]), ("typical (median)", rank[len(rank) // 2]), ("best", rank[-1])):
    it, pred = miss_train[k], naive["fit_preds"][k]
    found, missed = it["hidden"] & pred, it["hidden"] & ~pred
    panels.append(
        (
            f"{label} · found {int(found.sum())} of {int(it['hidden'].sum())} hidden px = {per_image[k]:.0%}",
            two_colour(it["image"], found, missed, THEME["truth"]),
        )
    )
show_masks(panels, width=3.4)

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell: it draws the naive head's failures on a training image, then on the held-out demo image
# of section 2, from predictions you already have (no new training, no model calls).
#
# **Q11.** 🟡 On the training image, which hidden vessels did the naive head miss (green), and which did it
# still find (violet)? Use the "found X%" in the title. On the held-out image, compare it with the section-2
# head: does it fail the same way (too wide, the disc, the whole eye) or differently? Try another training
# image:
#
# ```python
# FAIL_IMAGE = 0   # try 1, 2, ...
# ```

# %%
# Q11 — where the naive head fails: a training image, then the held-out demo image
FAIL_IMAGE = 0  # try 1, 2, ...
if not isinstance(FAIL_IMAGE, int) or not 0 <= FAIL_IMAGE < len(miss_train):
    raise SystemExit(f"FAIL_IMAGE must be a whole number from 0 to {len(miss_train) - 1}.")
it, pred = miss_train[FAIL_IMAGE], naive["fit_preds"][FAIL_IMAGE]
found, missed = it["hidden"] & pred, it["hidden"] & ~pred
show_masks(
    [
        ("labels training saw", overlay(it["image"], it["visible"], THEME["scribble"])),
        ("naive head", overlay(it["image"], pred, THEME["fails"])),
        (
            f"hidden vessels · found {hidden_recall(pred, it['hidden']):.0%}",
            two_colour(it["image"], found, missed, THEME["truth"]),
        ),
    ],
    width=3.0,
)
k = next(i for i, e in enumerate(SPEC_EVAL) if e is demo)
show_masks(
    [
        (
            f"{name} · IoU {score(r['held_preds'][k], demo)['iou']:.2f}",
            overlay(demo["image"], r["held_preds"][k], THEME["fails"]),
        )
        for name, r in naive_report.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=3.0,
)

# %% [markdown]
# This cell shows two look-alike patches from a training image, one the annotator painted (target 1) and one
# they never reached (target 0), which is all the head sees of the problem.

# %%
# @title Picture: two look-alike patches, two targets { display-mode: "form" }
pic_side = int(2 * 16 * max(miss_train[0]["mask"].shape) / SPECIALISED["feat_side"])  # 2 × 2 patches, in px
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
        (axes[1], hid_centre, "target 0: never reached", THEME["background"]),
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
# <!-- callout: theory | Theory: why missing = background fails, step by step -->
# 1. The annotator traces vessels A, B and C, then stops. Vessels D and E are never painted.
# 2. The file stores D and E as 0, the same value as real background.
# 3. Training sees patches that look like A with target 1, and patches that look like D (the same features)
#    with target 0.
# 4. For look-alike patches, plain cross-entropy settles on the share that was labelled. If half of the
#    vessels like A are hidden, p comes out near 0.5.
# 5. p near 0.5 sits on the threshold, so small differences tip many vessels to "background". The head keeps
#    the clearest, thickest vessels and drops the rest.
#
# Consequence: recall on the hidden vessels falls, while precision stays high.

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - The naive head did what the labels told it: it kept the clearest vessels, on the hidden segments, on new
#   images, and often even on the visible segments it was shown.
# - Compare the precision column: what it does mark is mostly right. The section-2 head failed by being too
#   generous (vessels as wide as a patch); this one fails by being too stingy.
# - Look at the held-out IoU of the two heads: it can be close, because IoU rewards the precision the naive
#   head gained as much as it punishes the recall it lost. One number would have hidden the failure.
# - Same features, same head: the only thing that changed is what we told it about the unlabelled pixels.
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
# ### 3.4 🔴 Optional: is it the labels, or the loss?
#
# Plain cross-entropy is one of the two changes from section 2.5: the heads of sections 1 and 2 used the
# balanced loss.

# %% [markdown]
# **Q12.** 🔴 Train the same naive labels with the balanced loss of section 2.5 and compare its recall on the
# hidden vessels with the plain head's. Does reweighting fix "missing = background"? What did it fix?
#
# ```python
# head, sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=True)
# ```
#
# This cell trains that balanced variant on the same incomplete labels (seconds) and prints its recall next
# to the plain head's.

# %%
# Q12 — the same "missing = background" labels with the balanced loss of section 2.5
balanced_head, balanced_sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=True)
bal_fit = [
    predict_head(balanced_head, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
]
bal_hidden, bal_visible = (part_recall(bal_fit, miss_train, part) for part in ("hidden", "visible"))
RESULTS["naive_balanced_recall_hidden_fg"] = bal_hidden
print(f"balanced head · trained in {balanced_sec:.1f} s on {DEVICE}")
print(f"{'loss':<12}{'recall: hidden':>15}{'visible':>9}")
print(f"{'plain':<12}{naive['recall_hidden_fit']:>15.3f}{naive['recall_visible_fit']:>9.3f}")
print(f"{'balanced':<12}{bal_hidden:>15.3f}{bal_visible:>9.3f}")

# %% [markdown]
# <!-- callout: go-deeper | Q12: what to look for -->
# - Reweighting usually recovers a lot of the hidden vessels. It does not fix the labels, it fixes the
#   *proportion*: missing = background told the head that vessels are about half as common as they are, and
#   the balanced loss stops caring about proportions.
# - It works here for two reasons that need not hold on your data: we hid segments *at random*, so the hidden
#   ones look like the visible ones, and a linear head is too simple to memorise which segment was hidden.
# - If annotators systematically skip the thin capillaries, the head learns "thin = background", and no class
#   weight undoes that.
# - Section 5 fixes the cause instead: tell the loss which pixels are unknown.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **A label file with only vessel and background** → ask how it was made: an unpainted pixel may mean
#   "unknown", not "background".
# - **A head trained with "missing = background"** → expect a stingy model (precision stays high, recall on
#   what was left out collapses), so also measure recall on a small, fully labelled set.
# - **A balanced loss that wins the recall back** → do not trust it for systematic gaps: it works when the
#   gaps are random, not when thin vessels are skipped.
# - **Three failures so far** (SAM finds the eye, the head is limited by its 16 px patches, naive training
#   believes the file) → none of these is a model-size problem we have tested; each needs different
#   supervision. This section's recall on the hidden vessels is the baseline to beat; Q13 asks how.

# %% [markdown]
# ### ☕ Before the break

# %% [markdown]
# <!-- callout: question | Question for the coffee break -->
# No code, just think about it over coffee.
#
# **Q13.** 🟡 You have 50 fundus images and a 4-hour annotation budget. Perfect labels cost 3 to 5 hours per
# image, and the cheap ones lie: the label file says "background" wherever nobody looked. Without buying more
# labels, what could you tell the training about a pixel nobody labelled? Think of at least two options before
# section 4.
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
