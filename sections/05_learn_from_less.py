# %% [markdown]
# ## 5. Path 1 · Learn from less
#
# Back to the half-finished label file of section 3. The naive head failed because the loss got the file
# as it was, and the file calls every unpainted pixel "background". Path 1 keeps the same file, features
# and head. It changes only what the loss is told about the unpainted pixels:
#
# - **Masked:** "unknown". The loss skips them; the only background it hears about is a few strokes on
#   clear background.
# - **🔴 PU:** "unlabelled, and about this share of them is vessel". The loss uses every pixel and corrects
#   for the vessels hiding among them.

# %% [markdown]
# <!-- callout: objective | Objective -->
# treat missing labels as unknown; write the masked loss, then 🔴 a PU-inspired loss.

# %% [markdown]
# <!-- callout: theory | Theory: the masked loss -->
# A **masked loss** is any loss with a mask that removes pixels from it. Let S be the pixels that carry a
# label. An ignored pixel is not in S: it adds nothing to the loss and sends no gradient.
#
# `L = −(1/|S|) · Σ_{i∈S} [ y_i·log p_i + (1−y_i)·log(1−p_i) ]`
#
# - **Example, a thin vessel nobody painted.** Naive (section 3): its pixels are in S with y = 0, so the
#   loss punishes the head for calling them vessel. Masked: they are not in S, so nothing punishes it.
# - **Catch:** if S holds almost only vessel, "everything is vessel" scores a perfect zero. Some known
#   background must stay in S.
# - The partial cross-entropy of section 4 is the same idea; here you write it yourself.

# %% [markdown]
# This helper cell defines a training loop that takes the loss as an argument (the same linear head,
# features and steps as every head so far), a report with the numbers of section 3 plus the held-out ones
# of section 4, a view of what the loss hears, and one colour for each head of this path.

# %%
# @title Path 1 helpers: train with any loss, score a head, view the loss mask { display-mode: "form" }
APPROACH_COLOURS |= {
    "naive": "#BF360C",
    "naive_balanced": "#FF8A65",
    "masked": "#F9A825",
    "pu": "#6D4C41",
}
APPROACH_NAMES |= {
    "naive": "naive: missing = background",
    "naive_balanced": "naive, balanced loss (3.4)",
    "masked": "masked: missing = unknown",
    "pu": "PU: missing = unlabelled",
}


def train_head_with(
    feats: list[torch.Tensor], labels: list[np.ndarray], loss_fn, steps: int, lr: float = 0.05
):
    """The linear head of `train_head`, trained with `loss_fn(logits, labels, ignore_mask)` on every pixel of
    every training image at once. Labels per pixel: 1 vessel, 0 background (or unlabelled), -1 ignored.
    Returns the head's tensors and the training seconds."""
    torch.manual_seed(SEED)
    allf = torch.cat([f.flatten(1) for f in feats], 1)
    mu, sd = allf.mean(1), allf.std(1) + 1e-6
    feats = [(f - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1) for f in feats]
    lab = torch.cat([torch.from_numpy(np.asarray(x, dtype=np.int8)).flatten() for x in labels]).to(DEVICE)
    target, ignore = (lab == 1).float(), lab < 0
    shapes = [np.asarray(x).shape for x in labels]
    head = torch.nn.Conv2d(feats[0].shape[0], 1, 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.monotonic()
    for _ in range(steps):
        opt.zero_grad()
        logits = torch.cat(
            [
                F.interpolate(head(f[None]), size=s, mode="bilinear", align_corners=False).flatten()
                for f, s in zip(feats, shapes, strict=True)
            ]
        )
        loss_fn(logits, target, ignore).backward()
        opt.step()
    tensors = {
        "weight": head.weight.detach().cpu(),
        "bias": head.bias.detach().cpu(),
        "feat_mean": mu.cpu(),
        "feat_std": sd.cpu(),
    }
    return tensors, time.monotonic() - t0


def path_report(tensors: dict) -> dict:
    """Recall on the hidden and visible vessels of the training images (section 3) and the held-out
    numbers of section 4 (IoU, clDice, recall, precision, marked share), plus the predictions."""
    fit_preds = [
        predict_head(tensors, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
    ]
    held_preds = [
        predict_head(tensors, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
    ]
    out = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
    }
    return out | rung_report(held_preds) | {"fit_preds": fit_preds, "held_preds": held_preds}


def print_reports(reports: dict) -> None:
    """One line per head: recall on the hidden / visible training vessels, then the held-out numbers."""
    print(f"{'head':<34}{'recall: hidden':>15}{'visible':>9}   held-out IoU  recall  precision  marked")
    for name, r in reports.items():
        print(
            f"{name:<34}{r['recall_hidden_fit']:>15.3f}{r['recall_visible_fit']:>9.3f}   {r['iou']:>12.3f}"
            f"{r['recall']:>8.3f}{r['precision']:>11.3f}{r['marked']:>8.0%}"
        )


def public(r: dict) -> dict:
    """A report without its prediction arrays, for RESULTS."""
    return {k: v for k, v in r.items() if not k.endswith("_preds")}


def loss_view(item: dict, labels: np.ndarray, show_bg: bool = True) -> np.ndarray:
    """What the loss hears on one image: painted vessel in teal, background targets in grey, pixels the loss
    ignores (label -1) dark. Sparse labels are thickened for display only."""
    known = labels >= 0
    if known.mean() < 0.2:
        known = ndi.binary_dilation(known, iterations=4)
    out = np.where(known[..., None], item["image"], (item["image"] * 0.25).astype(np.uint8))
    if show_bg:
        bg = labels == 0
        if bg.mean() < 0.2:
            bg = ndi.binary_dilation(bg, iterations=4)
        out = overlay(out, bg, THEME["background"])
    return overlay(out, labels == 1, THEME["scribble"])


# %% [markdown]
# ### 5.1 Three ways to read the same label file
#
# The next cell builds what each recipe tells the loss, for the five training images of section 3. The
# visible vessels are the same every time; the rest differs:
#
# - **Naive:** every unpainted pixel is a background target.
# - **Masked:** unpainted pixels are ignored, except the 3–5 background strokes of section 4 on the same
#   images (seconds of work per image): the loss needs some known background.
# - **PU:** every unpainted pixel counts, as "unlabelled".
#
# No ground truth goes in: only the visible vessels, the strokes and the fundus border.

# %%
# @title What each recipe tells the loss { display-mode: "form" }
bg_strokes = [(it["scribbles"] == 2) & region(it) & ~it["visible"] for it in miss_train]
# masked: vessel where painted, background only on the strokes, everything else unknown (-1)
masked_labels = [
    np.where(it["visible"] & region(it), 1, np.where(bg, 0, -1)).astype(np.int8)
    for it, bg in zip(miss_train, bg_strokes, strict=True)
]
# PU: vessel where painted, "unlabelled" (0) everywhere else inside the fundus
pu_labels = [np.where(region(it), it["visible"], -1).astype(np.int8) for it in miss_train]
known = {
    "naive": np.mean([(lab >= 0).mean() for lab in naive_labels]),
    "masked": np.mean([(lab >= 0).mean() for lab in masked_labels]),
}
n_bg_strokes = sum(m.get("n_bg_strokes", 0) for m in SPECIALISED["fit"][: len(miss_train)])
print(
    f"pixels the loss hears about · naive {known['naive']:.1%} (all called background unless painted) · "
    f"masked {known['masked']:.1%} ({n_bg_strokes} background strokes over {len(miss_train)} images) · "
    "PU: all of them, as vessel or unlabelled"
)

ex, k_ex = miss_train[0], 0
show_masks(
    [
        (ex["name"], ex["image"]),
        ("naive: vessel / background", loss_view(ex, naive_labels[k_ex])),
        ("masked: vessel / strokes / ignored", loss_view(ex, masked_labels[k_ex])),
        ("PU: vessel / unlabelled", loss_view(ex, pu_labels[k_ex], show_bg=False)),
    ],
    width=2.8,
)

# %% [markdown]
# Teal = painted vessel, grey = a background target, dark = ignored by the loss, bright with no colour =
# counted as "unlabelled" (PU only).
#
# **Q18.** 🟡 Look at the third panel. The masked recipe tells the loss about few pixels, almost all vessel:
#
# - Which pixels may the head call vessel without any penalty?
# - What would the loss do if you also removed the background strokes?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# Every dark pixel in the third panel: the hidden vessels, which is the point, but also all the background
# between the strokes, including the background right next to every vessel. The strokes sit on clear
# background away from the vessels, so they say little about where a vessel ends.
#
# Without them the loss only ever hears "vessel", and a head that calls every pixel vessel scores a perfect
# zero: the trivial solution. A masked loss is only as good as the known background left in it.
#
# </details>
#
# ### 5.2 Write the masked loss

# %% [markdown]
# This cell holds our version of the function you can write below. Open it only if you get stuck.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_masked_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    """Reference: per-pixel BCE, then a mean per known class (an empty class counts as 0)."""
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    pos, neg = ~ignore_mask & (labels == 1), ~ignore_mask & (labels == 0)
    return 0.5 * bce[pos].sum() / pos.sum().clamp(min=1) + 0.5 * bce[neg].sum() / neg.sum().clamp(min=1)


# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 4 min) -->
# Fill `masked_loss`. In: `logits` (float tensor, any shape), `labels` (same shape, 1.0 = vessel,
# 0.0 = background) and `ignore_mask` (same shape, bool, True = unknown). Out: one scalar tensor. It is the
# balanced loss of every head so far, with one change: ignored pixels count for nothing.
#
# 1. Take the binary cross-entropy of each pixel: `F.binary_cross_entropy_with_logits(..., reduction="none")`.
# 2. Average it over the known vessel pixels and over the known background pixels, separately.
# 3. Return the mean of the two averages.
#
# `L = ½ · mean_known vessel(BCE) + ½ · mean_known background(BCE)`
#
# The cell checks your version on a 2 × 3 play example and continues with ours if you skip it.

# %%
# ✍️ YOUR CODE
def masked_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    """Balanced binary cross-entropy over the pixels where `ignore_mask` is False: the mean loss on the
    known vessel pixels and the mean loss on the known background pixels, averaged."""
    ...


_play_logits = torch.tensor([[2.0, -1.0, 0.5], [-2.0, 3.0, 0.0]])
_play_labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
_play_ignore = torch.tensor([[False, False, True], [False, True, False]])  # two unknown pixels
masked_loss_fn = use(
    masked_loss,
    reference=_ref_masked_loss,
    check=lambda f: [
        round(float(f(_play_logits, _play_labels, ig)), 4)
        for ig in (_play_ignore, torch.zeros(2, 3, dtype=bool))
    ],
    name="masked_loss",
)
_with, _without = (
    float(masked_loss_fn(_play_logits, _play_labels, ig))
    for ig in (_play_ignore, torch.zeros_like(_play_ignore))
)
print(f"play example · loss {_with:.4f} with two pixels unknown, {_without:.4f} with none")

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# Same label file as the naive head (you saw its hidden-vessel recall in section 3), but the unpainted pixels
# are now unknown. Write two guesses:
#
# - On the hidden vessels of the training images, will its recall be closer to the naive head's or to the
#   section-2 head's (full masks)?
# - On new images, will it mark more of the fundus than the naive head, or less?

# %% [markdown]
# This cell trains the head with your masked loss (or ours) on the masked labels of 5.1 and prints the
# seconds it took. Not done in a minute, or on a CPU and in a hurry? Set `TRAIN_YOURSELF = False` and run it
# again: it loads our "masked" checkpoint, which was trained with a stricter rule (the text it prints says
# how).

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "masked" head instead of training
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

# the supervision of this path: painted vessels + background strokes; the hidden truth never enters (L-005)
if TRAIN_YOURSELF:
    masked_head, masked_sec = train_head_with(miss_feats, masked_labels, masked_loss_fn, HEAD_STEPS)
    src = f"trained here on {len(miss_train)} images, {HEAD_STEPS} steps, masked loss"
else:
    masked_head, entry = load_head(SPECIALISED, "masked", side)
    masked_sec = entry["train_sec"]
    src = (
        f"our checkpoint {entry['file']} ({len(entry['train_names'])} images). ⚠️ It was told exactly which "
        "pixels next to the hidden vessels are unknown and that the rest is background: a rule that needs "
        "the hidden truth, so read its numbers as an upper bound for this path"
    )
print(f"masked head @ {side} px · {src} · {masked_sec:.1f} s on {DEVICE}")
if masked_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the masked head next to the heads of sections 2 and 3, on the same groups as in 3.3
# (hidden and visible vessels of the training images) and on the held-out images, then plots the
# recalls and precision.

# %%
# @title Score the masked head next to the earlier heads { display-mode: "form" }
t0 = time.monotonic()
path_reports = {APPROACH_NAMES["masked"]: path_report(masked_head)}
masked_infer = round(time.monotonic() - t0, 1)
for key, tensors in (("naive", naive_head), ("naive_balanced", balanced_head)):
    path_reports[APPROACH_NAMES[key]] = path_report(tensors)
path_reports["full masks (section 2)"] = path_report(full_head)
masked = path_reports[APPROACH_NAMES["masked"]]

RESULTS["masked_recall_hidden_fg"] = masked["recall_hidden_fit"]  # same definition as section 3 (SPEC §9)
RESULTS["masked_head"] = public(masked) | {
    "long_side": side,
    "n_train": len(miss_train),
    "known_share": float(known["masked"]),
    "n_bg_strokes": int(n_bg_strokes),
    "trained_here": TRAIN_YOURSELF,
    "train_sec": round(float(masked_sec), 2),
    "infer_sec": masked_infer,
}
print_reports(path_reports)
print(f"section 3 recorded for the naive head · hidden-vessel recall {RESULTS['naive_recall_hidden_fg']:.3f}")

PATH_COLOURS = {
    APPROACH_NAMES["naive"]: APPROACH_COLOURS["naive"],
    APPROACH_NAMES["naive_balanced"]: APPROACH_COLOURS["naive_balanced"],
    APPROACH_NAMES["masked"]: APPROACH_COLOURS["masked"],
    "full masks (section 2)": THEME["works"],
}
BARS = {
    "recall_hidden_fit": "hidden vessels\n(training images)",
    "recall": "recall\n(held-out)",
    "precision": "precision\n(held-out)",
}


def path_chart(names: list[str]) -> None:
    """Grouped bars: recall on the hidden training vessels, held-out recall and precision, per head."""
    x, width = np.arange(len(BARS)), 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.2, 3))
    for j, name in enumerate(names):
        shift = (j - (len(names) - 1) / 2) * width
        ax.bar(x + shift, [path_reports[name][b] for b in BARS], width, color=PATH_COLOURS[name], label=name)
    ax.set_xticks(x, list(BARS.values()), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Path 1: what each head finds, and how much of what it marks is right")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


path_chart(list(PATH_COLOURS))

# %% [markdown]
# <!-- callout: your-turn | Your turn: compare the heads on one image -->
# - Run the next cell. It shows the hidden vessels each head found on a training image, then the error maps
#   of section 4 on the held-out demo image (violet found, orange invented, green missed).
# - Then set `LOOK_IMAGE` to 1, 2, … and run it again, to answer Q19.
# - No new training and no model calls: it draws from the predictions you already have.

# %%
# Q19 — hidden vessels found on a training image, then error maps on the held-out demo image
LOOK_IMAGE = 0  # try 1, 2, ...
if not isinstance(LOOK_IMAGE, int) or not 0 <= LOOK_IMAGE < len(miss_train):
    raise SystemExit(f"LOOK_IMAGE must be a whole number from 0 to {len(miss_train) - 1}.")
it = miss_train[LOOK_IMAGE]
LOOK = [APPROACH_NAMES["naive"], APPROACH_NAMES["masked"]]
look_preds = {name: path_reports[name]["fit_preds"][LOOK_IMAGE] for name in LOOK}
show_masks(
    [
        (
            f"{name.split(':')[0]} · hidden found {hidden_recall(pred, it['hidden']):.0%}",
            two_colour(it["image"], it["hidden"] & pred, it["hidden"] & ~pred),
        )
        for name, pred in look_preds.items()
    ],
    width=3.4,
)
k_demo = next(i for i, e in enumerate(SPEC_EVAL) if e is demo)
show_masks(
    [
        (
            f"{name.split(':')[0]} · IoU {score(path_reports[name]['held_preds'][k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], path_reports[name]["held_preds"][k_demo], demo),
        )
        for name in LOOK
    ],
    width=3.4,
)

# %% [markdown]
# **Q19.** 🟡 Compare the naive and the masked head:
#
# - Which hidden vessels does the masked head recover that the naive head missed?
# - On the held-out image, what did each head pay for what it found? Compare the orange.
# - Does that hold on another training image?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The masked head was never told that a hidden vessel is background, so it treats a vessel-looking patch as
# a vessel wherever it is: it finds most of the hidden vessels, the ones the naive head dropped.
#
# The price shows on the held-out image. The only background the loss heard about is a handful of strokes
# far from any vessel, so nothing told it where a vessel ends: the area around each vessel, and sometimes
# darker background, turns orange.
#
# Compare the "marked" column: the naive head is too stingy, the masked head is often too generous, and IoU
# alone may rank them either way. Both failures come from what the loss was told about the unpainted
# pixels, in opposite directions.
#
# </details>

# %% [markdown]
# <!-- callout: pitfall | Pitfall: "ignore" is not free -->
# 1. The loss ignores most pixels, and the few it keeps are almost all vessel.
# 2. A head that answers "vessel" everywhere gets a near-perfect loss on them.
# 3. The training loss looks great while the mask covers the whole eye.
#
# Watch the share of the image the model marks, and give the loss some background close to the objects,
# not only easy background far away. The next cell shows the trick with a fake head.

# %%
# @title A fake head that always answers "vessel" { display-mode: "form" }
_lab = torch.cat([torch.from_numpy(x.astype(np.int8)).flatten() for x in masked_labels])
_y, _ignored = (_lab == 1).float(), _lab < 0
_logits = torch.full_like(_y, 8.0)  # "vessel" everywhere
_bce = F.binary_cross_entropy_with_logits(_logits, _y, reduction="none")
_with_bg = float(_bce[~_ignored].mean())
_no_bg = float(_bce[~_ignored & (_lab == 1)].mean())  # the same labels with the background strokes removed
print(
    f"mean loss of 'vessel everywhere' on the pixels the masked loss keeps · with the background strokes "
    f"{_with_bg:.3f} · without them {_no_bg:.5f} (≈ 0: a perfect score for a useless head)"
)

# %% [markdown]
# ### 5.3 🔴 Optional: a PU-inspired loss
#
# The masked head needed background strokes; the PU head needs none. It reads every unpainted pixel as
# *unlabelled*: a mix of background and the vessels nobody painted. If you know roughly what share of the
# vessels was painted, you know roughly how much vessel hides in that mix, and you can subtract it.

# %% [markdown]
# <!-- callout: theory | Theory: the PU trick in three steps -->
# The **prior** π is the share of fundus pixels that are vessel. We estimate it as (painted share of the
# fundus) ÷ (share of the vessels that was painted). The second number comes from the annotator ("I did
# about half"; here the dataset config). Example: painted 1.5 % of the fundus, annotator did half → π = 3 %.
#
# 1. Vessel loss on the painted pixels, target "vessel": `L_v = mean_painted(−log p)`.
# 2. Background loss. Calling every pixel "background" mixes π vessels with (1−π) background, so subtract
#    the vessels: `L_b = ( mean_all(−log(1−p)) − π · mean_painted(−log(1−p)) ) / (1−π)`.
# 3. Balance them: `L = ½·L_v + ½·max(L_b, 0)`. The max stops L_b dipping below zero when the head
#    overfits the few painted vessels.
#
# It assumes the painted vessels are a random sample of all vessels: true here, because we hid segments at
# random.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper: where the PU loss comes from -->
# - Elkan & Noto, KDD 2008: with labelled positives drawn at random from all positives, plus the share of
#   positives, you can estimate the loss you would get with complete labels.
# - Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NeurIPS 2017,
#   [arXiv:1703.00593](https://arxiv.org/abs/1703.00593): the clipping at zero in step 3.
# - We use their non-negative version on a single pixel classifier, which is why we call it PU-*inspired*.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 🔴): guess the annotator wrong -->
# - Read `pu_loss` in the next cell, then run it.
# - The prior rests on one number from the annotator, `LABELLED_SHARE`. Pretend the annotator was wrong:
#   set `LABELLED_SHARE` to 0.3, then 0.8, and run again.
# - The cell defines the PU loss, estimates the prior from the painted vessels and `LABELLED_SHARE`, trains
#   the head on the PU labels of 5.1 (seconds) and prints its numbers next to the other heads.

# %%
# Q20 — a PU-inspired loss: painted vessels + unlabelled pixels + one guess about the annotator
LABELLED_SHARE = 1 - MISSING["hidden_fraction"]  # the annotator's "I did about half"; try 0.3, 0.8
if not isinstance(LABELLED_SHARE, (int, float)) or not 0 < LABELLED_SHARE <= 1:
    raise SystemExit("LABELLED_SHARE must be a number above 0 and at most 1, e.g. 0.5.")


def pu_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor, prior: float
) -> torch.Tensor:
    """Balanced non-negative PU loss: labels 1 = painted vessel, 0 = unlabelled; `prior` = vessel share."""
    z, y = logits[~ignore_mask], labels[~ignore_mask]
    painted = z[y == 1]
    vessel_loss = F.softplus(-painted).mean()  # (1) painted vessels, target "vessel"
    all_as_bg = F.softplus(z).mean()  # every pixel, target "background"
    painted_as_bg = F.softplus(painted).mean()  # painted vessels, target "background"
    bg_loss = (all_as_bg - prior * painted_as_bg) / (1 - prior)  # (2) the true background, estimated
    return 0.5 * vessel_loss + 0.5 * bg_loss.clamp(min=0)  # (3) balanced, clipped at zero


painted_share = sum(int((lab == 1).sum()) for lab in pu_labels) / sum(
    int((lab >= 0).sum()) for lab in pu_labels
)
prior = painted_share / LABELLED_SHARE
if not 0 < prior < 1:
    raise SystemExit(
        f"painted share {painted_share:.3f} ÷ LABELLED_SHARE {LABELLED_SHARE} gives a prior of "
        f"{prior:.2f}, which must lie between 0 and 1: choose a larger LABELLED_SHARE."
    )
pu_head, pu_sec = train_head_with(
    miss_feats, pu_labels, lambda z, y, ig: pu_loss(z, y, ig, prior), HEAD_STEPS
)
t0 = time.monotonic()
path_reports[APPROACH_NAMES["pu"]] = pu = path_report(pu_head)
PATH_COLOURS[APPROACH_NAMES["pu"]] = APPROACH_COLOURS["pu"]
RESULTS["pu_recall_hidden_fg"] = pu["recall_hidden_fit"]
RESULTS["pu_head"] = public(pu) | {
    "long_side": side,
    "n_train": len(miss_train),
    "labelled_share": float(LABELLED_SHARE),
    "prior": float(prior),
    "train_sec": round(float(pu_sec), 2),
    "infer_sec": round(time.monotonic() - t0, 1),
}
print(
    f"PU head · painted {painted_share:.1%} of the fundus ÷ labelled share {LABELLED_SHARE:.2f} → prior "
    f"{prior:.1%} vessel · trained in {pu_sec:.1f} s on {DEVICE}"
)
print_reports(
    {k: path_reports[k] for k in (APPROACH_NAMES["naive"], APPROACH_NAMES["masked"], APPROACH_NAMES["pu"])}
)
path_chart([APPROACH_NAMES["naive"], APPROACH_NAMES["masked"], APPROACH_NAMES["pu"]])

# %% [markdown]
# **Q20.** 🔴 Train again with a smaller and a larger `LABELLED_SHARE`. Which way does the head move:
#
# - on recall?
# - on the marked share?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The PU head hears about every pixel, like the naive head, but it no longer takes "unpainted" for
# "background": it expects a share π of vessel in the mix and stops pushing once the mix looks like that.
# Where it lands between the naive and the masked head depends on the run, so read your numbers.
#
# Compare it with the balanced naive head of 3.4: on labels hidden at random the two can land close,
# because the damage of "missing = background" is mostly a wrong proportion, and both correct it, one by
# reweighting, the other by subtracting the vessels it expects in the mix.
#
# The prior is the lever. A smaller `LABELLED_SHARE` claims more hidden vessel, a larger prior, and the
# head marks more; at 1.0 it claims nothing is hidden and moves back towards the naive head. A wrong guess
# moves the head in a predictable direction rather than breaking it: with a small fully labelled check set,
# you can pick the share that matches it.
#
# </details>

# %% [markdown]
# This cell writes the annotation effort of the three Path 1 heads for the comparison table of section 8,
# with the assumptions of section 4: the painted half of each vessel map, plus seconds per background stroke.

# %%
# @title Annotation effort of the Path 1 heads { display-mode: "form" }
painted_masks = len(miss_train) * LABELLED_SHARE
RESULTS["path1_effort_min"] = {
    "naive": round(effort_minutes(n_full_masks=painted_masks), 2),
    "masked": round(effort_minutes(n_strokes=n_bg_strokes, n_full_masks=painted_masks), 2),
    "pu": round(effort_minutes(n_full_masks=painted_masks), 2),
}
print(
    " · ".join(
        f"{APPROACH_NAMES[k].split(':')[0]} {v:.0f} min" for k, v in RESULTS["path1_effort_min"].items()
    )
)

# %% [markdown]
# **Q21.** 🟡 Both fixes rest on one assumption: the hidden vessels look like the painted ones, because we
# hid segments at random. Name a real annotation habit that breaks it. What would the masked head and the
# PU head learn then?

# %% [markdown]
# <details><summary>A possible answer</summary>
#
# Annotators skip what is hard or tedious: the thinnest capillaries, the low-contrast periphery, the
# vessels inside a bleeding. Then the painted vessels are thick and clear, and the unpainted ones are thin
# and faint.
#
# - **Masked head:** never hears "background" on the thin vessels, but never hears "vessel" either. It learns
#   "vessel = thick and clear" and still misses most of them.
# - **PU head:** worse off. Its correction assumes the hidden vessels look like the painted ones, so it
#   pushes the thin ones towards background. The PU literature calls this a violation of the "selected
#   completely at random" assumption.
#
# The practical fix is the one of the ⚠️ in section 3: keep a small fully labelled set and check the recall
# on the kind of structure your annotators skip.
#
# </details>

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **A label file where unpainted = background** → mark the unpainted pixels "ignore" (−1), so the hidden
#   vessels stop being punished.
# - **The loss ignores almost everything and a head says "vessel" everywhere** → keep some known background
#   close to the objects, and watch the "marked" share, not the training loss.
# - **Hidden vessels come back but the orange area grows** → that is the price of missing background;
#   compare precision and "marked", not IoU alone.
# - **Only positives are labelled and you can estimate their share** → try the PU loss, and sweep
#   `LABELLED_SHARE` to see how much a wrong guess moves recall.
# - **Annotators skip the thin or faint cases** → neither fix works; keep a small fully labelled check set.
# - **None of this adds information about the hidden vessels** → to recover them, try a second model
#   (section 6).
