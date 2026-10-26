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
#
# The cheap labels of this notebook come in two kinds, both leaving pixels nobody looked at: **missing
# labels** (sections 3 and 5) and **scribbles** (section 4). What should a model believe about those pixels?
# A plain loss says "background".

# %% [markdown]
# This helper hides a random half of the vessel segments, the way the cached label files were made.

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
    """Overlay `good` in violet ("works") and `bad` in `bad_colour` (default orange; pass THEME["truth"] for
    missed vessels, which are green) on one image."""
    return overlay(overlay(image, good, THEME["works"]), bad, bad_colour)


# %% [markdown]
# This helper draws a row of pictures with a colour key under them, so no colour is left unexplained.

# %%
# @title show_keyed: pictures with a colour key { display-mode: "form" }
key_painted = (THEME["scribble"], "teal: painted (in the label file)")
key_hidden = (THEME["truth"], "green: hidden vessel")
key_found = (THEME["works"], "violet: hidden vessel the head found")
key_missed = (THEME["truth"], "green: hidden vessel it missed")


def show_keyed(panels: list[tuple[str, np.ndarray]], key: list[tuple[str, str]], width: float = 3.0) -> None:
    """One row of RGB panels (title, image) with a colour key under them: `key` = [(colour, meaning), ...].
    Long titles wrap with a newline so neighbours never overlap."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width + 0.7))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    handles = [mpl.patches.Patch(color=c, label=m) for c, m in key]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    plt.show()


# %% [markdown]
# ### 3.1 Make a half-finished label file
#
# This cell hides about half of the vessel segments (the share comes from the dataset config) of the five
# retina training images of section 2.4 and shows one of them. Two words from here on: **painted** vessels
# are the ones left in the label file, **hidden** vessels are in the full ground truth but the file leaves
# them unpainted, so it calls them background.
#
# The printout gives two shares that differ a little: the share of hidden *segments* is what we ask for, the
# share of hidden vessel *pixels* is what results, because long segments hold more pixels than short ones.
# What to look for: the $\color{#00838F}{\blacksquare}$ teal (painted) vessels in the third picture and the
# $\color{#2E7D32}{\blacksquare}$ green (hidden) vessels in the
# fourth together make the full ground truth in the second.

# %% [markdown]
# <!-- diagram: s03-half-finished-label | How a half-finished label file is made -->
# Read left to right: the full tree, the same tree cut into numbered segments with a random half hidden, and
# the file the annotator saves. In that file a hidden vessel is stored exactly like background.

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
show_keyed(
    [
        (ex["name"], ex["image"]),
        ("full ground truth\n(evaluation only)", overlay(ex["image"], ex["mask"], THEME["truth"])),
        ("painted\n(what training may use)", overlay(ex["image"], ex["visible"], THEME["scribble"])),
        ('hidden\n(the file says "background")', overlay(ex["image"], ex["hidden"], THEME["truth"])),
    ],
    [(THEME["truth"], "green: true vessels (all of them, or the hidden ones)"), key_painted],
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
# <details><summary>The same in words</summary>
#
# - Hidden vessels found: 400 hidden vessel pixels, the model marks 120 of them: 120 ÷ 400 = 30%.
# - Precision: the model marked 150 pixels in total and 120 are vessels: 120 ÷ 150 = 80%.
#
# </details>
#
# IoU mixes the two into one number. Here we want them apart, because the question is how many hidden
# vessels the model finds.

# %% [markdown]
# <!-- diagram: s03-recall-precision | Hidden vessels found and precision, with the same 120 pixels -->
# The same 120 correctly marked pixels answer two questions. Look at what each one divides by.

# %% [markdown]
# This helper defines the two numbers above and the picture of what a head found and missed.


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
    """One picture: painted vessels in teal, hidden vessels the model found in violet, missed in green."""
    hidden = item["hidden"] & region(item)
    found, missed = hidden & pred, hidden & ~pred
    base = overlay(item["image"], item["visible"], THEME["scribble"])
    n = int(hidden.sum())
    title = (f"{label}\n" if label else "") + (
        f"found {int(found.sum())} of {n} hidden px = {hidden_recall(pred, hidden):.0%}"
    )
    return title, two_colour(base, found, missed, THEME["truth"])


def hidden_panels(item: dict, pred: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The three pictures: the label file (painted), the hidden vessels, and found / missed by `pred`."""
    base = overlay(item["image"], item["visible"], THEME["scribble"])
    hidden_pic = overlay(base, item["hidden"] & region(item), THEME["truth"])
    return [("label file:\npainted vessels", base), ("hidden:\nthe file says background", hidden_pic)] + [
        found_panel(item, pred)
    ]


# %% [markdown]
# ### 3.3 The naive baseline: missing = background
#
# "Naive" means: pass the label file straight into the textbook loss, with no thought about what the
# unpainted pixels mean. We measure the hidden vessels found (3.2), next to the section-2 head's.

# %% [markdown]
# The next picture shows two vessel patches of a training image that look alike. The annotator painted one
# and never reached the other. Each row is one patch: the photo around it, the same photo with the label
# file on top, and the target the loss gets for the patch.
#
# What to look for: the left pictures are almost the same, the right ones are opposite.

# %%
# @title Picture: two look-alike patches, two targets { display-mode: "form" }
pic_patch = 16 * max(miss_train[0]["mask"].shape) / specialised["feat_side"]  # one head patch, in image px
pic_ph = max(2, int(round(pic_patch)))  # patch side, whole pixels
pic_big = 3 * pic_ph  # the crop reaches 3 patches to each side of the centre
pic_pair = None
for pic_item in miss_train:
    pic_h, pic_w = pic_item["mask"].shape
    cands = []
    for pic_part in ("hidden", "visible"):
        pys, pxs = np.nonzero(skeleton(pic_item[pic_part]))
        inside = (pys >= pic_big) & (pys < pic_h - pic_big) & (pxs >= pic_big) & (pxs < pic_w - pic_big)
        cands.append(np.stack([pys[inside], pxs[inside]], 1))
    if len(cands[0]) and len(cands[1]):
        pic_pair = (pic_item, cands)
        break
if pic_pair is None:
    print("no hidden and visible vessel segment far enough from the border to draw the picture")
else:
    pic_item, (pic_hid, pic_vis) = pic_pair
    pic_gray = pic_item["image"].astype(float).mean(axis=2)

    def pic_crop(arr: np.ndarray, centre: np.ndarray, half: int) -> np.ndarray:
        return arr[centre[0] - half : centre[0] + half, centre[1] - half : centre[1] + half]

    hid_centre = pic_hid[len(pic_hid) // 2]
    vis_pool = pic_vis[:: max(1, len(pic_vis) // 300)]
    gaps = [
        float(np.abs(pic_crop(pic_gray, c, pic_ph) - pic_crop(pic_gray, hid_centre, pic_ph)).mean())
        for c in vis_pool
    ]
    vis_centre = vis_pool[int(np.argmin(gaps))]  # the painted patch that looks most like the hidden one
    pic_seen = overlay(pic_item["image"], pic_item["hidden"], THEME["truth"])  # hidden vessels in green
    pic_seen = overlay(pic_seen, pic_item["visible"], THEME["scribble"])  # painted vessels in teal
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.2))
    pic_rows = (
        (vis_centre, "painted patch", "vessel = 1"),
        (hid_centre, "hidden patch", "background = 0"),
    )
    for (centre, pic_name, target), row in zip(pic_rows, axes, strict=True):
        for ax, pic in zip(row[:2], (pic_item["image"], pic_seen), strict=True):
            ax.imshow(pic_crop(pic, centre, pic_big))
            for colour, width in (("white", 3.6), (THEME["scribble"], 1.8)):  # teal square with a white edge
                corner = pic_big - pic_ph / 2
                ax.add_patch(
                    mpl.patches.Rectangle((corner, corner), pic_ph, pic_ph, fill=False, ec=colour, lw=width)
                )
        row[0].set_title(f"{pic_name}\nthe photo", fontsize=9)
        row[1].set_title(f"{pic_name}\nwith the label file", fontsize=9)
        lab_patch = pic_crop(pic_item["visible"], centre, pic_ph // 2 + 1).astype(float)
        row[2].imshow(
            np.where(lab_patch[..., None] > 0, hex_rgb(THEME["scribble"]), hex_rgb(THEME["background"])),
            interpolation="nearest",
        )
        row[2].set_title("the target the loss\ngets for the patch", fontsize=9)
        row[2].text(
            0.5,
            0.5,
            target,
            transform=row[2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round", "fc": "black", "alpha": 0.6},
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    add_key(
        fig,
        [
            ("teal: painted (target 1)", THEME["scribble"]),
            ("green: hidden vessel (target 0)", THEME["truth"]),
            ("grey: background target 0", THEME["background"]),
            ("box", "square: the patch the head sees"),
        ],
        note="Both patches show a vessel. The label file calls one vessel and the other background.",
    )
    plt.show()

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The head is told that about half of the vessels in these five images are background. Write two guesses.
#
# - Of the hidden vessel pixels, how many will it find? Near the section-2 head's share, about half of it, or
#   close to zero?
# - On new images, will it mark more pixels than the section-2 head, or fewer?

# %% [markdown]
# This helper trains the head on the label file as it is, or loads our checkpoint.


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
# This cell scores the naive head and the section-2 head (full masks) on three sets of vessel pixels: the
# ones we hid, the ones that were painted, and every vessel of the held-out images. Each bar is the share of
# that set the head finds. The printout adds the held-out IoU, the precision, and **marked**: the share of
# the fundus (the scored area) the head calls vessel.
#
# What to look for (the numbers are from our run, yours will differ a little):
#
# - The **hidden** bars: how far below the section-2 head is the naive head? On our run it finds about 6 %
#   of the hidden vessels, the section-2 head about 88 %.
# - The **painted** bars: does it find even the vessels it was told about? On our run it misses many.
# - The printout: does the naive head mark more or less of the fundus, and is its precision higher or lower?

# %%
# @title Score both heads on hidden, painted and held-out vessels { display-mode: "form" }
full_head = spec_heads[side][0]  # section 2.4: same features, trained on full masks
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
    held_pairs = list(zip(held_preds, spec_eval, strict=True))
    precision = [marked_precision(p & region(it), it["mask"]) for p, it in held_pairs]
    naive_report[name] = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
        "recall_held": part_recall(held_preds, spec_eval, "mask"),
        "precision_held": float(np.mean(precision)),
        "marked_held": float(np.mean([p[region(it)].mean() for p, it in held_pairs])),
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
print(f"{'head':<30}{'hidden found':>13}{'painted':>9}{'held-out':>10}   held-out IoU / precision / marked")
for name, r in naive_report.items():
    print(
        f"{name:<30}{r['recall_hidden_fit']:>13.3f}{r['recall_visible_fit']:>9.3f}{r['recall_held']:>10.3f}"
        f"   {r['iou_held']:.3f} / {r['precision_held']:.3f} / {r['marked_held']:.0%}"
    )

x = np.arange(len(GROUPS))
fig, ax = plt.subplots(figsize=(6.4, 3.2))
head_colours = (THEME["works"], "#BF360C")  # full-mask head, naive head: the naive colour of section 5
for shift, name, colour in zip((-0.14, 0.14), compared_heads, head_colours, strict=True):
    bars = ax.bar(x + shift, [naive_report[name][g] for g in GROUPS], 0.27, color=colour, label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xticks(x, list(GROUPS.values()), fontsize=8)
ax.set_ylabel("share of vessel pixels found")
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("Which vessels does each head find?")
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)
plt.tight_layout()
plt.show()

# %% [markdown]
# What the numbers say:
#
# - **Stingy, not generous.** The section-2 head failed by marking too much (vessels as wide as a patch).
#   This one fails by marking too little: it keeps the clearest vessels and often misses even the painted
#   ones.
# - **A close IoU can hide it.** IoU rewards the precision the naive head gained as much as it punishes the
#   vessels it lost, so the two heads' held-out IoU can be close. One number would have hidden the failure.

# %% [markdown]
# <!-- callout: theory | Theory: why missing = background goes wrong -->
# <details><summary>The same in words</summary>
#
# 1. The annotator painted about half of the vessels and stopped (the label file of 3.1).
# 2. The plain loss reads every unpainted pixel as "background", hidden vessels included.
# 3. So the head sees look-alike vessel patches with opposite labels (the picture above).
# 4. For two identical patches with opposite labels, the loss is lowest when the head answers "about 50 %
#    vessel" for both.
# 5. 50 % sits on the threshold, so faint and thin vessels fall just below it and are marked as background.
# 6. The head misses vessels, even painted ones. What it marks is mostly the clearest vessels, so it is
#    mostly right: it looks precise.
#
# </details>
#
# Check this against your guess and the numbers above.

# %% [markdown]
# <!-- diagram: s03-missing-is-background | Why missing = background makes the head stingy -->
# Two patches that look the same get opposite targets, so the head settles on 50 % for both, right on the
# threshold.

# %% [markdown]
# This cell shows the naive head on its typical training image (the median by hidden vessels found).
#
# What to look for: how much $\color{#2E7D32}{\blacksquare}$ green (missed) is left, and whether the
# $\color{#6A1B9A}{\blacksquare}$ violet (found) vessels are the thick,
# clear ones.

# %%
# @title The typical training image: label file, hidden vessels, found and missed { display-mode: "form" }
per_image = [hidden_recall(p, it["hidden"]) for p, it in zip(naive["fit_preds"], miss_train, strict=True)]
typical = int(np.argsort(per_image)[len(per_image) // 2])
show_keyed(
    hidden_panels(miss_train[typical], naive["fit_preds"][typical]),
    [
        key_painted,
        (THEME["truth"], "green: hidden vessel (middle), missed by the head (right)"),
        key_found,
    ],
    width=3.6,
)

# %% [markdown]
# **Try it: hide more or fewer vessels.** The next helper rebuilds the label file with a different share of
# hidden segments, trains the naive head again on it (a few seconds), and prints the result next to the
# default one.

# %%
# @title naive_with_hidden_share: rebuild the label file, retrain, score { display-mode: "form" }
_HELD_FEATS: dict = {}


def naive_with_hidden_share(percent: int) -> None:
    """Hide `percent` % of the vessel segments (the 3.1 helper), retrain the plain head from scratch on that
    label file (the 3.3 helper), and print hidden vessels found and held-out precision next to the default."""
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
    print(
        f"The naive head is retrained from scratch on the same {len(miss_train)} training images, with their "
        f"label files rebuilt so that {percent}% of the vessel segments are hidden (trained in {sec:.1f} s "
        f"on {DEVICE})."
    )
    print(
        f"That hides {share:.0%} of the vessel pixels: the two percentages differ because segments differ "
        f"in size, and long segments hold more pixels than short ones."
    )
    print(f"{'':<28}{'hidden found':>13}{'painted found':>15}{'held-out precision':>20}")
    for name, (hid, vis, prec) in rows.items():
        print(f"{name:<28}{hid:>13.2f}{vis:>15.2f}{prec:>20.2f}")
    if has_hidden:
        print(
            f"The three pictures show training image {trial.index(shown)}: its new label file, the vessels "
            f"hidden in it, and what the retrained head found of them."
        )
        show_keyed(
            hidden_panels(shown, fit_preds[trial.index(shown)]),
            [
                key_painted,
                (THEME["truth"], "green: hidden vessel (middle), missed by the head (right)"),
                key_found,
            ],
            width=3.0,
        )
    else:
        print(
            "Nothing is hidden at 0%, so there is no 'hidden found' to report: the head sees the full masks."
        )


# %% [markdown]
# <!-- callout: question | Question -->
# **Q8.** 🟡 Write down a guess first: how will "hidden found" and "precision" change as you hide more of
# the vessels? Then run the next cell with `HIDE_PERCENT` at 10, 50 and 90. For each of the two numbers,
# say what happened, and what it means for an annotator who stops early.
#
# What the cell trains: the naive head is retrained from scratch on the same training images, with their label
# files rebuilt so that `HIDE_PERCENT` % of the vessel segments are hidden. The three pictures show training
# image 0: its new label file, the hidden vessels, and what the retrained head found of them. The share of
# hidden vessel *pixels* is a little different from `HIDE_PERCENT`, because segments differ in size.

# %%
# @title Q8 — hide more or fewer vessels, retrain the naive head { display-mode: "form" }
HIDE_PERCENT = 70  # @param {type:"slider", min:0, max:90, step:10}
naive_with_hidden_share(HIDE_PERCENT)

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **A label file with only vessel and background** → ask how it was made: an unpainted pixel may mean
#   "unknown", not "background".
# - **A head trained with "missing = background"** → expect a stingy model (it marks less, precision stays
#   high, the vessels left out are mostly not found), so also measure the hidden vessels found on a small,
#   fully labelled set.
# - **Three failures so far** (SAM finds the eye, the head is limited by its 16 px patches, naive training
#   believes the file) → none of these is a model-size problem we have tested; each needs different
#   supervision. The hidden vessels found in this section is the baseline to beat; section 4 starts on how.
