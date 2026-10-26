# %% [markdown]
# ## 5. Path 1 · Learn from less
#
# Same trick as 4.2 ("unlabelled = ignored"), now measured on the half-painted label file of section 3. The
# naive head failed because the loss got that file as it was, and the file calls every unpainted pixel
# "background". Path 1 keeps the same file, features and head, and changes only what the loss is told about
# the unpainted pixels:
#
# - **Masked:** "unknown". The loss skips them; the only background it hears about is a few strokes on
#   clear background. "Unknown" is the simplest way to make a loss uncertainty-aware.
# - **PU** (positive-unlabelled: you know some positives, everything else is unknown): "unlabelled, and
#   about this share of them is vessel". The loss uses every pixel and corrects for the vessels hiding
#   among them.

# %% [markdown]
# <!-- callout: objective | Objective -->
# treat missing labels as unknown (write the masked loss), then tell the loss how many vessels hide among the
# unpainted pixels (the PU loss).


# %% [markdown]
# <!-- diagram: s00-map-path1 | You are here: Path 1 -->
# Where this section sits on the map of section 0.

# %% [markdown]
# <!-- callout: theory | Theory: the masked loss -->
# A **masked loss** is the partial cross-entropy of 4.2: only the pixels that carry a label count, and an
# ignored pixel adds nothing and sends no gradient.
#
# - **Example, a thin vessel nobody painted.** Naive (section 3): its pixels count as background, so the
#   loss punishes the head for calling them vessel. Masked: they are ignored, so nothing punishes it.
# - **Catch:** if the known pixels are almost only vessel, "everything is vessel" scores a perfect zero. Some
#   known background must stay. The picture below shows why.
#
# <details><summary>The same in words</summary>
#
# If the known pixels are almost only vessel, a head that says "vessel" everywhere is never wrong on a known
# pixel, so its loss is zero. A few known background pixels are enough to catch it.
#
# </details>

# %% [markdown]
# <!-- diagram: s05-masked-catch | The catch of the masked loss -->
# The same useless head in both columns. Only the background strokes can make it pay.

# %% [markdown]
# The next cell defines a training loop that takes any loss, a scoring report, a view of what the loss
# hears, and one colour for each head of this path.

# %%
# @title Path 1 helpers: train with any loss, score a head, view the loss mask { display-mode: "form" }
approach_colours |= {
    "naive": "#BF360C",
    "masked": "#F9A825",
    "pu": "#6D4C41",
}
approach_names |= {
    "naive": "naive: missing = background",
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
    numbers of section 4 (IoU, recall, precision, marked share), plus the predictions."""
    fit_preds = [
        predict_head(tensors, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
    ]
    held_preds = [
        predict_head(tensors, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
    ]
    out = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
    }
    return out | rung_report(held_preds) | {"fit_preds": fit_preds, "held_preds": held_preds}


def print_reports(reports: dict) -> None:
    """One line per head: recall on the hidden / visible training vessels, then the held-out numbers."""
    print(f"{'head':<34}{'recall: hidden':>15}{'visible':>9}   held-out IoU  recall  marked")
    for name, r in reports.items():
        print(
            f"{name:<34}{r['recall_hidden_fit']:>15.3f}{r['recall_visible_fit']:>9.3f}   {r['iou']:>12.3f}"
            f"{r['recall']:>8.3f}{r['marked']:>8.0%}"
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
# The next cell shows what each recipe tells the loss on one of the five training images of section 3. The
# painted vessels are the same every time; the rest differs. The picture below shows the idea on a strip of
# ten pixels, with a vessel nobody painted.
#
# <details><summary>The same in words</summary>
#
# - **Naive:** every unpainted pixel is a background target.
# - **Masked:** unpainted pixels are ignored, except 3–5 background strokes per image (the annotator also
#   drew a few strokes on clear background, as in section 4). The loss needs some known background.
# - **PU:** every unpainted pixel counts, as "unlabelled".
#
# </details>
#
# What to look for: the dark areas of the third picture are free for the head to call vessel. No ground truth
# goes in, only the painted vessels, the strokes and the fundus border.

# %% [markdown]
# <!-- diagram: s05-three-readings | Three ways to read the same label file -->
# Follow the outlined pixels, a vessel nobody painted. The labels are the same in all three columns; only
# what the loss makes of the unpainted pixels changes.

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
n_bg_strokes = sum(m.get("n_bg_strokes", 0) for m in specialised["fit"][: len(miss_train)])
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
    legend=[
        ("painted vessel (target: vessel)", THEME["scribble"]),
        ("background target", THEME["background"]),
        ("ignored by the loss (dark)", "#404040"),
    ],
    note="PU panel: everything that is not teal counts as unlabelled, hidden vessels included.",
)

# %% [markdown]
# ### 5.2 Write the masked loss

# %% [markdown]
# This cell holds our version of the one line you can write. Open it only if you get stuck.

# %%
# @title Reference solution and play example (open only if stuck) { display-mode: "form" }
def _ref_known_pixels(ignore_mask: torch.Tensor) -> torch.Tensor:
    """Reference: a pixel is known when it is not ignored."""
    return ~ignore_mask


_play_logits = torch.tensor([[2.0, -1.0, 0.5], [-2.0, 3.0, 0.0]])
_play_labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
_play_ignore = torch.tensor([[False, False, True], [False, True, False]])  # two unknown pixels


def _check_known(f):
    """What `use` compares: which pixels the function calls known on the play example."""
    return torch.as_tensor(f(_play_ignore)).int().tolist()


def show_masked_play(fn) -> None:
    """Print the loss `fn` gives on the 2 x 3 play example, with and without unknown pixels."""
    with_unknown, without = (
        float(fn(_play_logits, _play_labels, ig)) for ig in (_play_ignore, torch.zeros_like(_play_ignore))
    )
    print(f"play example · loss {with_unknown:.4f} with two pixels unknown, {without:.4f} with none")


# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 1 min): one line of the masked loss -->
# The masked loss below is the balanced loss of every head so far, with one change: ignored pixels count for
# nothing. Everything is given except one line. Fill `known_pixels`: it gets `ignore_mask` (a bool tensor,
# True = unknown) and returns a bool tensor of the same shape that is True where the label is **known**.
#
# Return `torch.ones_like(ignore_mask)` and every pixel counts: that is the naive loss again.
#
# The cell checks your line on a 2 × 3 play example and continues with ours if you skip it.

# %%
# ✍️ YOUR CODE
def known_pixels(ignore_mask: torch.Tensor) -> torch.Tensor:
    """True where the label is known (the pixel is not ignored), same shape as `ignore_mask`.
    e.g. tensor([[True, True, False], [True, False, True]])"""
    ...


known_pixels_fn = use(known_pixels, reference=_ref_known_pixels, check=_check_known, name="known_pixels")

# %% [markdown]
# The masked loss itself, with your line (or ours) in the first row. The play example prints it with two
# pixels unknown and with none.

# %%
def masked_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    """Balanced cross-entropy over the known pixels: the mean loss on the known vessel pixels and the mean
    loss on the known background pixels, averaged. An empty group counts as 0."""
    known_mask = known_pixels_fn(ignore_mask)  # your line: which labels can be trusted
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")  # loss of each pixel
    pos, neg = known_mask & (labels == 1), known_mask & (labels == 0)  # known vessel, known background
    return 0.5 * bce[pos].sum() / pos.sum().clamp(min=1) + 0.5 * bce[neg].sum() / neg.sum().clamp(min=1)


show_masked_play(masked_loss)

# %% [markdown]
# This helper cell defines `train_masked_head`: it trains with the masked loss, or loads our checkpoint.

# %%
# @title Masked-head helper: train with the masked loss, or load ours { display-mode: "form" }
def train_masked_head(train_yourself: bool):
    """The masked head of Path 1. Returns the head's tensors and the training seconds."""
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_MASKED_HEAD must be True or False.")
    # the supervision of this path: painted vessels + background strokes; no hidden truth (L-005)
    if train_yourself:
        head, sec = train_head_with(miss_feats, masked_labels, masked_loss, HEAD_STEPS)
        src = f"trained here on {len(miss_train)} images, {HEAD_STEPS} steps, masked loss"
    else:
        head, entry = load_head(specialised, "masked", side)
        sec = entry["train_sec"]
        src = (
            f"our checkpoint {entry['file']} ({len(entry['train_names'])} images). ⚠️ It was told exactly "
            "which pixels next to the hidden vessels are unknown and that the rest is background: a rule "
            "that needs the hidden truth, so read its numbers as an upper bound for this path"
        )
    print(f"masked head @ {side} px · {src} · {sec:.1f} s on {DEVICE}")
    if sec > 60:
        print("ℹ️ Slow session? Set TRAIN_MASKED_HEAD = False above to load our checkpoint next time.")
    return head, sec


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Same label file as the naive head (you saw its hidden-vessel recall in section 3), but the unpainted pixels
# are now unknown. Write two guesses:
#
# - On the hidden vessels of the training images, will its recall be closer to the naive head's or to the
#   section-2 head's (full masks)?
# - On new images, will it mark more of the fundus than the naive head, or less?

# %% [markdown]
# The next cell trains the head with the masked loss on the labels of 5.1. With `TRAIN_MASKED_HEAD = False`
# it loads our checkpoint, made with a stricter rule that needs the hidden truth: its numbers are an upper
# bound for this path.

# %%
# @title Train the masked head (False = load our checkpoint) { display-mode: "form" }
TRAIN_MASKED_HEAD = True  # @param {type:"boolean"}
masked_head, masked_sec = train_masked_head(TRAIN_MASKED_HEAD)

# %% [markdown]
# The next cell scores the masked head next to the naive head and the full-mask head, on the hidden and
# visible vessels of the training images and on the held-out images. "Marked" is the share of the fundus a
# mask covers. What to look for: how much of the fundus each head marks, against the share that is really
# vessel.

# %%
# @title Score the masked head next to the earlier heads { display-mode: "form" }
t0 = time.monotonic()
masked = path_report(masked_head)
masked_infer = round(time.monotonic() - t0, 1)
path_reports = {
    approach_names["naive"]: path_report(naive_head),
    approach_names["masked"]: masked,
    "full masks (section 2)": path_report(full_head),
}
path_colours = {
    approach_names["naive"]: approach_colours["naive"],
    approach_names["masked"]: approach_colours["masked"],
    approach_names["pu"]: approach_colours["pu"],
    "full masks (section 2)": THEME["works"],
}

RESULTS["masked_recall_hidden_fg"] = masked["recall_hidden_fit"]  # same definition as section 3 (SPEC §9)
RESULTS["masked_head"] = public(masked) | {
    "long_side": side,
    "n_train": len(miss_train),
    "known_share": float(known["masked"]),
    "n_bg_strokes": int(n_bg_strokes),
    "trained_here": TRAIN_MASKED_HEAD,
    "train_sec": round(float(masked_sec), 2),
    "infer_sec": masked_infer,
}
print_reports(path_reports)
truth_share = float(np.mean([it["mask"][region(it)].mean() for it in spec_eval]))
print(
    f"the true vessels cover {truth_share:.1%} of the fundus · the masked head marks {masked['marked']:.1%} "
    f"· the naive head {path_reports[approach_names['naive']]['marked']:.1%}"
)

# annotation effort of the three Path 1 heads, for the comparison table of section 8: the painted half of
# each vessel map (the dataset's share), plus seconds per background stroke
painted_masks = len(miss_train) * (1 - missing_labels["hidden_fraction"])
RESULTS["path1_effort_min"] = {
    "naive": round(effort_minutes(n_full_masks=painted_masks), 2),
    "masked": round(effort_minutes(n_strokes=n_bg_strokes, n_full_masks=painted_masks), 2),
    "pu": round(effort_minutes(n_full_masks=painted_masks), 2),
}
print(
    "annotation effort · "
    + " · ".join(
        f"{approach_names[k].split(':')[0]} {v:.0f} min" for k, v in RESULTS["path1_effort_min"].items()
    )
)

# %% [markdown]
# <!-- callout: pitfall | Pitfall: "ignore" is not free -->
# Remove the background strokes and the masked loss hears only "vessel": a head that paints the whole eye
# scores a loss near zero. Watch the "marked" share, not the training loss.

# %%
# @title A fake head that always answers "vessel" { display-mode: "form" }
_lab = torch.cat([torch.from_numpy(x.astype(np.int8)).flatten() for x in masked_labels])
_y, _ignored = (_lab == 1).float(), _lab < 0
_all_vessel = torch.full_like(_y, 8.0)  # a head that says "vessel" everywhere
_bce = F.binary_cross_entropy_with_logits(_all_vessel, _y, reduction="none")
print(
    "mean loss of 'vessel everywhere' on the labels the masked loss keeps: "
    f"{float(_bce[~_ignored].mean()):.3f} with the background strokes, "
    f"{float(_bce[~_ignored & (_lab == 1)].mean()):.5f} without them "
    "(≈ 0: a perfect score for a useless head)"
)

# %% [markdown]
# ### ☕ Coffee break
#
# End of block 1. The masked head found hidden vessels, and it marks too much.

# %% [markdown]
# <!-- callout: question | Question for the coffee break -->
# No code, just think about it over coffee.
#
# **Q10.** 🟡 The masked head over-marks. The annotator says they painted about half of the vessels. How
# could that one number help the training?
#
# <details><summary>A hint, for after the break</summary>
#
# - Half of the vessels are painted, so about as many again are hiding among the unpainted pixels.
# - If the loss knows roughly how many vessels to expect in that unpainted mix, it can stop punishing them,
#   without any background strokes. That is 5.3.
# - Later in block 2: section 6 guesses the missing labels with SAM, section 7 skips training and prompts
#   the foundation model.
#
# </details>

# %% [markdown]
# ### 5.3 The PU loss
#
# Welcome back. The masked loss threw away every unpainted pixel. The PU loss keeps them, as *unlabelled*: a
# mix of background and the vessels nobody painted. The annotator tells us roughly what share of the vessels
# they painted, here about half. Say the painted vessels cover 1.5 % of the fundus. Then about 3 % of the
# fundus is vessel, and the other 1.5 % hides among the unlabelled pixels. The loss counts every unpainted
# pixel as background, then subtracts that 1.5 % again, so it stops punishing the hidden vessels (and fixes
# the wrong proportion of the naive head on the way). The cell below prints the real painted share and the
# vessel share it implies.

# %% [markdown]
# <!-- callout: theory | Theory: the PU loss in three steps -->
# The **prior** π is the share of fundus pixels that are vessel: (painted share of the fundus) ÷ (share of the
# vessels the annotator painted). Example: painted 1.5 %, annotator did half, so π = 3 %.
#
# The picture below shows the three steps on 200 pixels. The formulas are in it, and in words here:
#
# <details><summary>The same in words</summary>
#
# 1. Vessel loss on the painted pixels, target "vessel": `L_v = mean_painted(−log p)`.
# 2. Background loss. Calling every pixel "background" mixes π vessels with (1−π) background, so subtract
#    the vessels: `L_b = ( mean_all(−log(1−p)) − π · mean_painted(−log(1−p)) ) / (1−π)`.
# 3. Balance them: `L = ½·L_v + ½·max(L_b, 0)`. The max stops L_b dipping below zero when the head
#    overfits the few painted vessels.
#
# </details>
#
# It assumes the painted vessels are a random sample of all vessels: true here, because we hid segments at
# random.

# %% [markdown]
# <!-- diagram: s05-pu-loss | The PU loss in three steps -->
# Read the three panels left to right. In step 2 the painted vessels stand in for every vessel, because they
# are a random sample of all of them.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper: where the PU loss comes from -->
# - Elkan & Noto, KDD 2008: with positives drawn at random from all positives, plus their share, you can
#   estimate the loss you would get with complete labels.
# - Kiryo et al., NeurIPS 2017, [arXiv:1703.00593](https://arxiv.org/abs/1703.00593): the clipping at zero in
#   step 3. We use it on a single pixel classifier, so we call ours PU-*inspired*.

# %% [markdown]
# This cell holds our version of step 2. Open it only if you get stuck.

# %%
# @title Reference solution for step 2 (open only if stuck) { display-mode: "form" }
def _ref_pu_background(all_as_bg: torch.Tensor, painted_as_bg: torch.Tensor, prior: float) -> torch.Tensor:
    """Reference: the background loss with the expected vessels subtracted."""
    return (all_as_bg - prior * painted_as_bg) / (1 - prior)


def _check_pu_background(f):
    """What `use` compares: step 2 on two made-up pairs of numbers."""
    pairs = ((0.9, 0.4, 0.2), (0.5, 0.5, 0.3))
    return [round(float(f(torch.tensor(a), torch.tensor(b), p)), 4) for a, b, p in pairs]


# %% [markdown]
# <!-- callout: your-turn | 🔴 Optional challenge (2 min): one line of the PU loss -->
# Fill `pu_background`, step 2 above. In: `all_as_bg` (the mean loss of calling every pixel background),
# `painted_as_bg` (the same on the painted vessels only) and `prior` (the vessel share π). Out: the
# background loss `L_b`, one number. Skip it and the notebook continues with ours.

# %%
# ✍️ YOUR CODE
def pu_background(all_as_bg: torch.Tensor, painted_as_bg: torch.Tensor, prior: float) -> torch.Tensor:
    """Background loss: every pixel as background, minus the expected vessels, rescaled by (1 - prior).
    e.g. tensor(0.7500)"""
    ...


pu_background_fn = use(
    pu_background, reference=_ref_pu_background, check=_check_pu_background, name="pu_background"
)

# %% [markdown]
# The PU loss, with your step 2 (or ours). The play example prints its value on six made-up pixels.

# %%
def pu_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor, prior: float
) -> torch.Tensor:
    """Balanced non-negative PU loss: labels 1 = painted vessel, 0 = unlabelled; `prior` = vessel share."""
    z, y = logits[~ignore_mask], labels[~ignore_mask]
    painted = z[y == 1]
    vessel_loss = F.softplus(-painted).mean()  # (1) painted vessels, target "vessel"
    all_as_bg = F.softplus(z).mean()  # every pixel, target "background"
    painted_as_bg = F.softplus(painted).mean()  # painted vessels, target "background"
    bg_loss = pu_background_fn(all_as_bg, painted_as_bg, prior)  # (2) the true background, estimated
    return 0.5 * vessel_loss + 0.5 * torch.as_tensor(bg_loss).clamp(min=0)  # (3) balanced, clipped at zero


_pu_logits = torch.tensor([2.0, -1.0, 0.5, -2.0, 3.0, 0.0])
_pu_labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # two painted vessels, four unlabelled
print(f"play example · PU loss {float(pu_loss(_pu_logits, _pu_labels, torch.zeros(6, dtype=bool), 0.3)):.4f}")

# %% [markdown]
# This helper cell defines `train_pu_head` and `recipe_comparison`, the 2 × 3 figure of the three heads.

# %%
# @title PU helpers: train the head, compare the three recipes { display-mode: "form" }
BARS = {
    "recall_hidden_fit": "hidden vessels\n(training images)",
    "recall": "recall\n(held-out)",
    "precision": "precision\n(held-out)",
}


def path_chart(names: list[str] | None = None) -> None:
    """Grouped bars: recall on the hidden training vessels, held-out recall and precision, per head."""
    names = names or [approach_names[k] for k in ("naive", "masked", "pu")] + ["full masks (section 2)"]
    x, width = np.arange(len(BARS)), 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for j, name in enumerate(names):
        shift = (j - (len(names) - 1) / 2) * width
        ax.bar(x + shift, [path_reports[name][b] for b in BARS], width, color=path_colours[name], label=name)
    ax.set_xticks(x, list(BARS.values()), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share, 0 to 1", fontsize=8)
    ax.set_title("Path 1: what each head finds,\nand how much of what it marks is right")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def recipe_comparison() -> None:
    """2 x 3 figure for the naive, masked and PU head. Row 1: the hidden vessels each head found on training
    image 0. Row 2: its error map on the held-out demo image. Uses the stored predictions; no training."""
    k_demo = next(i for i, e in enumerate(spec_eval) if e is demo)
    train, truth = miss_train[0], demo["mask"] & region(demo)
    hidden = train["hidden"] & region(train)
    base = overlay(train["image"], train["visible"], THEME["scribble"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    lab = {"naive": "naive", "masked": "masked", "pu": "PU"}.get
    for col, key in enumerate(("naive", "masked", "pu")):
        report = path_reports[approach_names[key]]
        fit = report["fit_preds"][0]
        found, missed = hidden & fit, hidden & ~fit
        axes[0, col].imshow(overlay(overlay(base, found, THEME["works"]), missed, THEME["truth"]))
        axes[0, col].set_title(
            f"{lab(key)} head · training image\nhidden vessels found {hidden_recall(fit, hidden):.0%}",
            fontsize=9,
        )
        held = report["held_preds"][k_demo]
        axes[1, col].imshow(error_map(demo["image"], held, demo))
        axes[1, col].set_title(
            f"{lab(key)} head · held-out image\nIoU {score(held, demo)['iou']:.2f} · found "
            f"{hidden_recall(held, truth):.0%} · {marked_pct(held, demo)}",
            fontsize=9,
        )
    for ax in axes.ravel():
        ax.axis("off")
    add_key(
        fig,
        [
            ("painted, in the label file (row 1)", THEME["scribble"]),
            "found",
            "missed",
            ("invented", "invented (row 2: prediction only)"),
        ],
        note="Row 1: only the hidden vessels are scored. Row 2: only the round fundus is scored.",
    )
    plt.show()


def train_pu_head(labelled_share: float):
    """Prior = painted share ÷ `labelled_share`; trains the head on the PU labels of 5.1 and adds it to
    `path_reports`. Returns the head's tensors."""
    if not isinstance(labelled_share, (int, float)) or not 0 < labelled_share <= 1:
        raise SystemExit("LABELLED_SHARE must be a number above 0 and at most 1, e.g. 0.5.")
    painted_share = sum(int((lab == 1).sum()) for lab in pu_labels) / sum(
        int((lab >= 0).sum()) for lab in pu_labels
    )
    prior = painted_share / labelled_share
    if not 0 < prior < 1:
        raise SystemExit(
            f"painted share {painted_share:.3f} ÷ LABELLED_SHARE {labelled_share} gives a prior of "
            f"{prior:.2f}, which must lie between 0 and 1: choose a larger LABELLED_SHARE."
        )
    head, sec = train_head_with(miss_feats, pu_labels, lambda z, y, ig: pu_loss(z, y, ig, prior), HEAD_STEPS)
    t0 = time.monotonic()
    path_reports[approach_names["pu"]] = pu = path_report(head)
    RESULTS["pu_recall_hidden_fg"] = pu["recall_hidden_fit"]
    RESULTS["pu_head"] = public(pu) | {
        "long_side": side,
        "n_train": len(miss_train),
        "labelled_share": float(labelled_share),
        "prior": float(prior),
        "train_sec": round(float(sec), 2),
        "infer_sec": round(time.monotonic() - t0, 1),
    }
    print(
        f"PU head · painted {painted_share:.1%} of the fundus ÷ labelled share {labelled_share:.2f} → "
        f"{prior:.1%} of the fundus is vessel (the prior) · trained in {sec:.1f} s on {DEVICE}"
    )
    print_reports({k: path_reports[approach_names[k]] for k in ("naive", "masked", "pu")})
    return head


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The annotator says "I painted about half". Suppose they had said "a third" instead. Will the PU head mark
# more or less of the fundus than with "half"? Write one sentence why.

# %% [markdown]
# The next cell trains the PU head (seconds) with `LABELLED_SHARE`, the share of the vessels the annotator
# painted, then draws two pictures: the 2 × 3 comparison of the three heads, and one chart with their recall
# and precision. What to look for in the pictures: how many hidden vessels each head found (row 1), and how
# much $\color{#E65100}{\blacksquare}$ orange (invented) the finding cost on the held-out image (row 2).

# %%
# @title Train the PU head, then compare the three recipes { display-mode: "form" }
# the share of the vessels the annotator painted (the dataset hides half of them: 0.5); try 0.3, 0.8
LABELLED_SHARE = 0.5  # @param {type:"slider", min:0.2, max:1, step:0.05}
pu_head = train_pu_head(LABELLED_SHARE)
recipe_comparison()
path_chart()

# %% [markdown]
# **Q11.** 🟡 Read the figure column by column. For each head, how many of the hidden vessels did it find (row
# 1), and what did it pay for that on the held-out image (row 2)? Which mistake would cost more in your own
# problem: missing vessels, or inventing them?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - **Naive:** finds few hidden vessels. It was told they are background, so little
#   $\color{#E65100}{\blacksquare}$ orange, much $\color{#2E7D32}{\blacksquare}$ green.
# - **Masked:** finds most of them, but nothing told it where a vessel ends, so the area around each vessel
#   often turns $\color{#E65100}{\blacksquare}$ orange.
# - **PU:** it hears about every pixel, so it lands between the two. Where exactly depends on the run: read
#   your numbers.
#
# </details>

# %% [markdown]
# **Q12.** 🟡 Compare with your prediction. Set `LABELLED_SHARE` to 0.3, run the cell, then to 0.8 and run it
# again. Which way does the head move on recall, and on the marked share?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - A smaller `LABELLED_SHARE` claims more hidden vessels (a larger prior), and the head marks more.
# - At 1.0 it claims nothing is hidden, and moves back towards the naive head.
# - A wrong guess moves the head in a predictable direction instead of breaking it. With a small fully
#   labelled check set you can pick the share that matches it.
#
# </details>

# %% [markdown]
# **Q13.** 🟡 The masked head rests on one assumption: the hidden vessels look like the painted ones, because
# we hid segments at random. Name a real annotation habit that breaks it. What would the masked head learn
# then?

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
#   pushes the thin ones towards background (the PU literature calls this breaking "selected completely at
#   random").
# - **Fix:** keep a small fully labelled set and check the recall on the kind of structure your annotators
#   skip (the ⚠️ in section 3).
#
# </details>

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **A label file where unpainted = background** → mark the unpainted pixels "ignore" (−1), so the hidden
#   vessels stop being punished.
# - **The loss ignores almost everything** → keep some known background close to the objects, and watch the
#   "marked" share, not the training loss: hidden vessels come back, but the
#   <span style="color:#E65100">■</span> orange area grows.
# - **Only positives are labelled and you can estimate their share** → use the PU loss, and sweep
#   `LABELLED_SHARE` to see how much a wrong guess moves recall.
# - **Annotators skip the thin or faint cases** → neither fix works and none adds information about the
#   hidden vessels: keep a small fully labelled check set, or try a second model (section 6).
