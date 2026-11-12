# %% [markdown]
# ## 6. Path 2 · Recover supervision
#
# Path 1 kept the label file and changed the loss. Path 2 keeps the loss and fills in the labels.
#
# A second model, the **teacher** (the model whose output we use as labels), guesses every pixel nobody
# painted, and the head trains on those guesses as if a person had drawn them. A guess used this way is a
# **pseudo-label**. It is like handing your half-done colouring-in to a fast, confident colleague: the page
# comes back full, and every confident mistake is now in your training set.
#
# We start from the scribbles of section 4 (a few strokes per image, no more annotation than rung 2). Two
# teachers get the job: **SAM**, prompted with clicks taken from the strokes, and **Grad-CAM**, the heat
# map of a small classifier that only learnt whether a tile contains a vessel.

# %% [markdown]
# <!-- callout: objective | Objective -->
# Turn SAM pseudo-labels and Grad-CAM maps into training signal, and know when to distrust them.


# %% [markdown]
# <!-- diagram: s00-map-path2 | You are here: Path 2 -->
# Where this section sits on the map of section 0.

# %% [markdown]
# <!-- diagram: s06-teacher-student | A teacher fills in the labels -->
# Read left to right. Your strokes are the only human input. Every guess the teacher makes in step 3 is
# trained on in step 4, right or wrong.

# %% [markdown]
# <!-- callout: theory | Theory: when pseudo-labels work -->
# A pseudo-label works when the teacher's guesses are mostly right *and* its mistakes are not systematic.
#
# - Pets: SAM's mask of a cat is a fine pseudo-label.
# - Retina: SAM's "vessel" mask is a fat region, and a head trained on it learns the region, not the line.
# - Guard: keep only the guesses the teacher is sure of and leave the rest unlabelled (6.2). A person's
#   strokes always overrule the teacher.

# %% [markdown]
# <!-- callout: go-deeper | Pseudo-label papers and confirmation bias -->
# A model trained on its teacher's errors learns to repeat them. The literature calls this *confirmation
# bias*. The usual guard is a confidence threshold.
#
# - Lee, "Pseudo-Label", ICML 2013 workshop: the original recipe.
# - FixMatch, Sohn et al., NeurIPS 2020: [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)

# %% [markdown]
# This helper cell adds the colours of this path, a probability colour scale and the stroke-click sampler.

# %%
# @title Path 2 helpers: colours, probability scale, clicks along the strokes { display-mode: "form" }
approach_colours |= {
    "sam_pseudo": "#0277BD",
    "gradcam_pseudo": "#827717",
}
approach_names |= {
    "sam_pseudo": "pseudo-labels: SAM from stroke clicks",
    "gradcam_pseudo": "pseudo-labels: Grad-CAM of a tile classifier",
}

# every map of this section uses the shared score scale of section 0: blue = 1, white = 0.5, dark grey = 0
# (blue, so a score map never shares a colour with the violet "found" of the error maps)
prob_key = [
    ("score map: 1 = sure it is vessel", THEME["score"]),
    ("score map: 0.5 = unsure", "white"),
    ("score map: 0 = sure it is not", "#424242"),
]
error_key = ["found", "invented", "missed"]  # the error-map legend: violet, orange, green


def prob_rgb(prob: np.ndarray) -> np.ndarray:
    """A probability map as RGB on the shared score scale: blue = 1, white = 0.5 (unsure), dark grey = 0."""
    return (score_cmap(np.clip(prob, 0, 1))[..., :3] * 255).astype(np.uint8)


def stroke_points(item: dict, value: int, n: int) -> list[list[float]]:
    """n (x, y) clicks spread along the strokes of one kind (1 vessel, 2 background) by farthest-point
    sampling from the first stroke pixel: no randomness, no ground truth."""
    ys, xs = np.nonzero(item["scribbles"] == value)
    if len(ys) == 0:
        return []
    pts = np.stack([xs, ys], 1).astype(float)
    chosen = [0]
    dist = np.linalg.norm(pts - pts[0], axis=1)
    while len(chosen) < min(n, len(pts)):
        k = int(dist.argmax())
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return [[float(x), float(y)] for x, y in pts[chosen]]


def pseudo_numbers(masks: list[np.ndarray], items: list[dict]) -> dict:
    """Diagnosis only (uses the hidden truth): mean IoU, recall (share of the true vessels the mask covers)
    and marked (share of the fundus the mask covers) of pseudo-label masks, inside the fundus."""
    rows = []
    for m, it in zip(masks, items, strict=True):
        r = region(it)
        rows.append(
            {
                "iou": score(m, it)["iou"],
                "recall": hidden_recall(m & r, it["mask"] & r),
                "marked": float((m & r)[r].mean()),
            }
        )
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


# %% [markdown]
# ### 6.1 SAM fills in the gaps
#
# We take the scribbles of section 4 again: a few vessel strokes and a few background strokes on each of the
# same five training images.

# %% [markdown]
# <!-- diagram: s06-sam-clicks | SAM from stroke clicks -->
# Forty clicks are taken from your strokes and go to SAM together with the image. SAM's answer for every
# pixel is the pseudo-label.

# %% [markdown]
# <details><summary>The same in words</summary>
#
# - Twenty clicks spread along the vessel strokes go in as "vessel", twenty along the background strokes as
#   "not vessel".
# - SAM's answer for every pixel is the pseudo-label.
# - Only the strokes go in. The ground truth appears in the figure so you can judge the guesses, and never
#   in training.
#
# </details>

# %% [markdown]
# This helper cell defines `sam_pseudo_labels`, which asks SAM for probability maps (or loads ours).

# %%
# @title SAM pseudo-label helper: probability maps from the stroke clicks { display-mode: "form" }
def sam_pseudo_labels(run_sam: bool, n_clicks: int):
    """SAM's probability map for each training image from `n_clicks` + `n_clicks` clicks on the strokes
    (or our precomputed maps). Prints the numbers and draws one image. Returns the maps, the clicks, the
    seconds and SAM's quality at the 0.5 cut (IoU, recall, marked)."""
    if not isinstance(run_sam, bool):
        raise SystemExit("RUN_SAM must be True or False.")
    t0 = time.monotonic()
    probs, clicks_used = [], []  # per training image: SAM's probability map, (vessel, background) clicks
    if run_sam:
        for it in tqdm(scrib_train, desc="SAM on the training images"):
            pos, neg = stroke_points(it, 1, n_clicks), stroke_points(it, 2, n_clicks)
            labels = [1] * len(pos) + [0] * len(neg)
            probs.append(sam.probability(it["image"], points=pos + neg, labels=labels))
            clicks_used.append((pos, neg))
        src = f"SAM 2.1 run here, {n_clicks} + {n_clicks} clicks per image"
    else:
        ours = PRECOMPUTED / "pseudo_labels" / "specialised"
        if not (ours / "prompts.json").is_file():
            raise SystemExit(f"{ours} is missing. Run setup_check.ipynb first, or set RUN_SAM = True.")
        prompts = json.loads((ours / "prompts.json").read_text())["prompts"]
        for it in scrib_train:
            probs.append(read_png(ours / f"{it['name']}.png") / 255)
            p = prompts[it["name"]]
            pairs = list(zip(p["points_xy"], p["labels"], strict=True))
            clicks_used.append(tuple([xy for xy, lab in pairs if lab == k] for k in (1, 0)))
        src = "our precomputed maps (same recipe, clicks drawn at random from the strokes)"
    sec = round(time.monotonic() - t0, 1)

    # diagnosis only: SAM's guesses at the usual 0.5 cut, scored against the hidden truth (never for training)
    quality = pseudo_numbers([p >= 0.5 for p in probs], scrib_train)
    # how sure SAM is, inside the fundus (no truth needed): the spread of its probabilities
    spread = np.quantile(
        np.concatenate([p[region(it)] for p, it in zip(probs, scrib_train, strict=True)]), [0.05, 0.5, 0.95]
    )
    train_truth = float(np.mean([it["mask"][region(it)].mean() for it in scrib_train]))
    show_result(
        "SAM pseudo-labels on the training images",
        quality | {"spread": f"{spread[0]:.2f} to {spread[2]:.2f}"},
        [
            ("spread", "how sure SAM is", "range holding 90 % of the pixels", "text"),
            "iou",
            "recall",
            "marked",
        ],
        meta=f"{len(scrib_train)} images · {src} · {sec} s on {DEVICE} · cut at 0.5",
        note=f"Checked against the truth, never used for training. Middle probability: {spread[1]:.2f}.",
        truth_share=train_truth,
    )
    scrib_ex, (pos, neg) = scrib_train[0], clicks_used[0]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    axes[0].imshow(scrib_ex["image"])
    for (x, y), kind in [(xy, 1) for xy in pos] + [(xy, 0) for xy in neg]:
        if kind:
            axes[0].plot(x, y, marker="*", ms=10, color=THEME["scribble"], mec="white", ls="")
        else:
            axes[0].plot(x, y, marker="X", ms=8, color="#424242", mec="white", ls="")
    axes[0].set_title(
        f"{scrib_ex['name']} · what SAM gets:\n{len(pos)} vessel + {len(neg)} background clicks", fontsize=9
    )
    im = axes[1].imshow(np.clip(probs[0], 0, 1), cmap=score_cmap, vmin=0, vmax=1)
    axes[1].set_title("what SAM answers: probability of vessel\n(pale = unsure, near 0.5)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, ticks=[0, 0.5, 1]).ax.tick_params(labelsize=7)
    axes[2].imshow(error_map(scrib_ex["image"], probs[0] >= 0.5, scrib_ex))
    axes[2].set_title("pseudo-label (probability ≥ 0.5)\ncompared with the ground truth", fontsize=9)
    for ax, role in zip(axes, ("input", "map", "errors"), strict=True):
        ax.axis("off")
        tag_panel(ax, role)
    add_key(
        fig,
        [("click", "vessel click"), ("background click (X)", "#424242"), *error_key],
        note="The colour bar reads the middle panel: blue = sure it is vessel, white = unsure, "
        "dark grey = sure it is not.",
    )
    plt.show()
    return probs, clicks_used, sec, quality


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Section 2 showed what SAM does with clicks on a vessel tree. Now it gets forty clicks per image, half of
# them saying "not here". Will its mask follow the vessels, or cover a region around them? Write one guess.

# %% [markdown]
# The next cell draws three panels, left to right: what SAM gets (the photo and the clicks), what it answers
# (a probability per pixel), and the pseudo-label you get by cutting that probability at 0.5, checked
# against the truth. The card above the figure says the same in numbers:
#
# - **How sure SAM is:** 90 % of the pixels fall inside the range shown. Values near 0 or 1 mean sure, near
#   0.5 unsure.
# - **Calls vessel:** how much of the fundus the pseudo-label calls vessel, with how much really is under it.
# - **Colours in the last panel:** $\color{#6A1B9A}{\blacksquare}$ violet = a real vessel found,
#   $\color{#E65100}{\blacksquare}$ orange = a vessel SAM invented, $\color{#2E7D32}{\blacksquare}$ green = a
#   vessel SAM left out. Every $\color{#E65100}{\blacksquare}$ orange and
#   $\color{#2E7D32}{\blacksquare}$ green pixel becomes a wrong training label if we take the guesses as
#   they are.

# %% [markdown]
# This cell asks SAM for a probability map on each training image from the stroke clicks and shows one image
# (seconds per image on a T4, much slower on a CPU, where `RUN_SAM = False` loads our maps, made the same way
# with clicks drawn at random).

# %%
# @title Run SAM on the stroke clicks to get pseudo-labels (False = load our maps) { display-mode: "form" }
RUN_SAM = True  # @param {type:"boolean"}
N_PSEUDO_CLICKS = 20  # @param {type:"integer"}
pseudo_probs, pseudo_clicks, pseudo_sec, sam_quality = sam_pseudo_labels(RUN_SAM, N_PSEUDO_CLICKS)
scrib_ex = scrib_train[0]

# %% [markdown]
# **Q14.** 🟡 Look at the error map and the probability panel. Where does SAM's guess go wrong: around
# the thick vessels, along the thin ones, in the background between the strokes? Is SAM sure of itself (deep
# $\color{#1565C0}{\blacksquare}$ blue, $\color{#424242}{\blacksquare}$ dark grey) or unsure (pale)? What
# does that mean for a cut at exactly 0.5?
#
# <details><summary>What to look for</summary>
#
# - SAM looks for one object that contains the positive clicks (section 2.3). Forty clicks on a vessel tree
#   do not describe one object, so SAM settles for the object that holds most of them, often the whole
#   fundus. Its edge is careful, but the inside is a region ($\color{#E65100}{\blacksquare}$ orange), and
#   thin branches between the clicks can
#   drop out ($\color{#2E7D32}{\blacksquare}$ green).
# - If the probability panel is mostly pale, the values sit just around 0.5 and the exact cut decides almost
#   everything: a little below and the whole fundus is "vessel", a little above and nothing is. The crisp
#   mask hides that doubt.
# - Whatever you see on your run, the threshold is your choice, not a property of SAM.
#
# </details>

# %% [markdown]
# ### 6.2 From a probability map to training targets
#
# A **target** is the label the head trains on: 1 vessel, 0 background, −1 ignored (the loss skips it). Two
# numbers turn SAM's probability into a target: the **cut** and the **unsure band**. With a cut of 0.5 and a
# band of 0.1:
#
# 1. A pixel at 0.9 is above 0.6, so it becomes vessel (1).
# 2. A pixel at 0.55 is within 0.1 of the cut, so it is ignored (−1).
# 3. A pixel at 0.3 is below 0.4, so it becomes background (0).
#
# Two rules sit on top: a person's stroke always wins over SAM, and pixels outside the fundus are always −1.

# %% [markdown]
# This cell draws the three zones for a cut of 0.5 and a band of 0.1.

# %%
# @title The cut and the unsure band, drawn { display-mode: "form" }
def draw_cut_and_band(cut: float, band: float) -> None:
    """One bar over SAM's probability axis, split into the three target zones."""
    zones = [
        (0, cut - band, THEME["background"], "0 · background"),
        (cut - band, cut + band, "white", "−1 · ignored"),
        (cut + band, 1, THEME["works"], "1 · vessel"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    for left, right, colour, label in zones:
        ax.barh(0, right - left, left=left, color=colour, edgecolor="black", height=0.5)
        ax.text(
            (left + right) / 2, 0, label, ha="center", va="center", fontsize=8,
            color="black" if colour == "white" else "white",
        )
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("SAM's probability for a pixel")
    ax.set_title(
        f"cut {cut}, band {band}: below {cut - band:g} a pixel becomes background,\n"
        f"from {cut + band:g} vessel, and in between it is ignored",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()


draw_cut_and_band(0.5, 0.1)

# %% [markdown]
# This helper cell defines `pseudo_targets`, the rule above in code. Open it if you want to read it.

# %%
# @title pseudo_targets: the cut, the band and the two rules in code { display-mode: "form" }
def pseudo_targets(
    prob: np.ndarray, scribbles: np.ndarray, region: np.ndarray, threshold: float = 0.5, band: float = 0.0
) -> np.ndarray:
    """Targets (int8) from a pseudo-label probability map: 1 where prob >= threshold + band, 0 where
    prob < threshold - band, -1 in between. Strokes overrule the map (1 vessel stroke -> 1, 2 background
    stroke -> 0); outside `region` -> -1.
    e.g. threshold 0.5, band 0.1, no strokes: prob [[0.9, 0.55, 0.3, 0.05]] -> [[1, -1, 0, 0]]"""
    targets = np.full(prob.shape, -1, np.int8)
    targets[prob >= threshold + band] = 1
    targets[prob < threshold - band] = 0
    targets[scribbles == 1] = 1
    targets[scribbles == 2] = 0
    targets[~np.asarray(region, bool)] = -1
    return targets


# %% [markdown]
# This helper cell defines `train_sam_head`, which makes targets at the 0.5 cut and trains or loads a head.

# %%
# @title SAM-head helper: pseudo-label targets, then train or load { display-mode: "form" }
def train_sam_head(train_yourself: bool):
    """The head trained on SAM's pseudo-labels at threshold 0.5, no band. Returns the head's tensors, the
    training seconds and the targets."""
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_SAM_HEAD must be True or False.")
    # the supervision of this path: the strokes, and SAM's guesses from clicks on the strokes (L-005)
    targets = [
        pseudo_targets(p, it["scribbles"], region(it), 0.5, 0.0)
        for p, it in zip(pseudo_probs, scrib_train, strict=True)
    ]
    if train_yourself:
        head, sec = train_head(scrib_feats, targets, HEAD_STEPS)
        src = f"trained here on {len(scrib_train)} images, {HEAD_STEPS} steps, SAM pseudo-labels at 0.5"
    else:
        head, entry = load_head(specialised, "pseudo", side)
        sec = entry["train_sec"]
        src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
    print(f"SAM pseudo-label head @ {side} px · {src} · {sec:.1f} s on {DEVICE}")
    if sec > 60:
        print("ℹ️ Slow session? Set TRAIN_SAM_HEAD = False above to load our checkpoint next time.")
    return head, sec, targets


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The scribble head of section 4 heard about the strokes only. This head hears about every pixel of the
# fundus, most of it in SAM's words. Will it mark more of the held-out fundus than the scribble head, or
# less? Will its IoU be higher? Write two guesses.

# %% [markdown]
# This cell turns SAM's maps into targets at the usual cut (threshold 0.5, no band) and trains the section-2
# head on them (`TRAIN_SAM_HEAD = False` loads our "pseudo" checkpoint instead, trained the same way).

# %%
# @title Train the SAM pseudo-label head (False = load our checkpoint) { display-mode: "form" }
TRAIN_SAM_HEAD = True  # @param {type:"boolean"}
sam_head, sam_sec, sam_targets = train_sam_head(TRAIN_SAM_HEAD)

# %% [markdown]
# This cell scores the new head on the held-out images and shows it under the scribble head, which saw the
# same strokes but kept every other pixel unknown. Columns: **IoU** (overlap with the truth), **recall** (the
# share of the true vessels it finds) and **calls vessel** (the share of the fundus it calls vessel; the
# "really vessel" row says how much really is).

# %%
# @title Score the SAM pseudo-label head on the held-out images { display-mode: "form" }
t0 = time.monotonic()
path2_preds = {
    "scribbles": ladder_preds["scribbles"],
    "sam_pseudo": [
        predict_head(sam_head, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
    ],
}
sam_infer = round(time.monotonic() - t0, 1)
path2_report = {k: rung_report(v) for k, v in path2_preds.items()}
RESULTS["pseudo_label_iou"] = path2_report["sam_pseudo"]["iou"]  # held-out IoU of the head (SPEC §9)
RESULTS["sam_pseudo_head"] = path2_report["sam_pseudo"] | {
    "long_side": side,
    "n_train": len(scrib_train),
    "threshold": 0.5,
    "band": 0.0,
    "pseudo_label_iou_fit": sam_quality["iou"],
    "pseudo_sec": pseudo_sec,
    "sam_run_here": RUN_SAM,
    "trained_here": TRAIN_SAM_HEAD,
    "train_sec": round(float(sam_sec), 2),
    "infer_sec": sam_infer,
}


def path2_table(
    reports: dict, title: str, meta: str | None = None, note: str | None = None, refs: list | None = None
) -> None:
    """One row per head on the held-out images: IoU, recall and the share of the fundus it calls vessel,
    then the reference rows (`refs`, then the ground truth)."""
    rows = [
        {"name": approach_names.get(k, k), "colour": approach_colours.get(k), **v} for k, v in reports.items()
    ]
    truth = {"name": "really vessel (the ground truth)", "marked": fg_share, "truth": True}
    show_table(title, rows, ["iou", "recall", "marked"], refs=[*(refs or []), truth], meta=meta, note=note)


path2_table(
    path2_report,
    "The scribble head and the SAM pseudo-label head on the held-out images",
    meta=f"{len(spec_eval)} images · pseudo-label head scored in {sam_infer} s on {DEVICE}",
)

# %% [markdown]
# This helper cell defines `tune_pseudo_labels`, which trains a head on targets made with your cut and band.

# %%
# @title Pseudo-label tuning helper: targets, head, diagnosis columns { display-mode: "form" }
def target_errors(targets: list[np.ndarray]) -> dict:
    """Diagnosis only (uses the hidden truth): share of the fundus ignored, of the true vessel pixels called
    background, and of the true background pixels called vessel, over the training images."""
    pairs = list(zip(targets, scrib_train, strict=True))
    return {
        "ignored": float(np.mean([(t < 0)[region(it)].mean() for t, it in pairs])),
        "vessel_as_bg": float(
            np.mean([((t == 0) & it["mask"]).sum() / max(it["mask"].sum(), 1) for t, it in pairs])
        ),
        "bg_as_vessel": float(
            np.mean(
                [
                    ((t == 1) & ~it["mask"] & region(it)).sum() / max((~it["mask"] & region(it)).sum(), 1)
                    for t, it in pairs
                ]
            )
        ),
    }


def tune_pseudo_labels(threshold: float, band: float) -> None:
    """Targets from SAM's maps with this cut and unsure band, a head trained on them, and its numbers
    shown under the head at 0.5."""
    ok = all(isinstance(v, (int, float)) for v in (threshold, band))
    if not ok or band < 0 or threshold - band <= 0 or threshold + band > 1:
        raise SystemExit(
            "PSEUDO_THRESHOLD must lie between 0 and 1, and UNSURE_BAND ≥ 0 must keep both edges inside."
        )
    side = specialised["feat_side"]
    tuned_targets = [
        pseudo_targets(p, it["scribbles"], region(it), threshold, band)
        for p, it in zip(pseudo_probs, scrib_train, strict=True)
    ]
    tuned_head, tuned_sec = train_head(scrib_feats, tuned_targets, HEAD_STEPS)
    tuned_preds = [
        predict_head(tuned_head, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
    ]
    tuned_report = rung_report(tuned_preds)
    RESULTS["sam_pseudo_tuned"] = tuned_report | {"threshold": threshold, "band": band}
    yours = f"yours (cut {threshold:g}, band {band:g})"
    label_rows = []
    for name, t in {"usual (cut 0.5, no band)": sam_targets, yours: tuned_targets}.items():
        label_rows.append({"name": name, **target_errors(t)})
    show_table(
        "1 · The labels the head trains on",
        label_rows,
        [
            ("ignored", "ignored", "share of the fundus left out", "pct"),
            ("vessel_as_bg", "vessel labelled background", "share of the true vessel pixels", "pct"),
            ("bg_as_vessel", "background labelled vessel", "share of the true background", "pct"),
        ],
        name_label="setting",
        meta=f"{len(scrib_train)} training images, checked against the truth · lower is better for the "
        "last two",
    )
    path2_table(
        {"usual (cut 0.5, no band)": path2_report["sam_pseudo"], yours: tuned_report},
        "2 · The heads trained on those labels",
        meta=f"{len(spec_eval)} held-out images · your head trained in {tuned_sec:.1f} s on {DEVICE}",
    )
    if threshold == 0.5 and band == 0:
        print("\nℹ️ Your setting equals the usual one: move a slider, then run this cell again.")
    k_demo = next(i for i, it in enumerate(spec_eval) if it is demo)
    ex = scrib_train[0]

    def label_view(t: np.ndarray) -> np.ndarray:
        """Training labels as three flat colours: vessel teal, background dark grey, ignored light grey."""
        out = np.full((*t.shape, 3), 224, np.uint8)  # ignored (−1) and outside the fundus
        out[t == 0] = (hex_rgb("#616161") * 255).astype(np.uint8)
        out[t == 1] = (hex_rgb(THEME["scribble"]) * 255).astype(np.uint8)
        return out

    def label_title(name: str, t: np.ndarray) -> str:
        r = region(ex)
        return (f"labels, {name}\nvessel {(t == 1)[r].mean():.0%} · background {(t == 0)[r].mean():.0%} · "
                f"ignored {(t < 0)[r].mean():.0%}")

    show_masks(
        [
            (label_title("usual", sam_targets[0]), label_view(sam_targets[0])),
            (label_title("yours", tuned_targets[0]), label_view(tuned_targets[0])),
            (rung_title("head, usual", path2_preds["sam_pseudo"][k_demo], demo),
             error_map(demo["image"], path2_preds["sam_pseudo"][k_demo], demo)),
            (rung_title("head, yours", tuned_preds[k_demo], demo),
             error_map(demo["image"], tuned_preds[k_demo], demo)),
        ],
        width=2.9,
        roles=["input", "input", "errors", "errors"],
        legend=[("label: vessel", THEME["scribble"]), ("label: background", "#616161"),
                ("label: ignored (−1)", "#E0E0E0"), "found", "invented", "missed"],
        note=f"Left: what each head is told on training image {ex['name']}. "
        "Right: what each head then predicts on the held-out demo image.",
    )


# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell, then move the cut.
#
# **Q15.** 🟡 Tune the pseudo-labels on the **training images**, never on the test images. Raise or lower
# `PSEUDO_THRESHOLD` and give the doubtful pixels an `UNSURE_BAND` to ignore, picking values from the spread
# shown in 6.1 and judging them by how many training targets are wrong (the two error columns). Then look
# at the held-out table once: does it move the same way? Is there a setting that beats 0.5 on your run?
#
# - `PSEUDO_THRESHOLD`: try 0.55, 0.6, 0.7 (or the middle probability shown in 6.1).
# - `UNSURE_BAND`: try 0.03; pixels within 0.03 of the threshold are ignored.
#
# Both are sliders in the next cell: move one, then run it again.

# %% [markdown]
# This cell trains the same head on labels made with your setting and compares it with the usual one (cut
# 0.5, no band), in two steps:
#
# 1. **The labels**, checked against the hidden truth on the training images: the share of the fundus
#    **ignored** (−1), the share of the true vessel pixels **labelled background**, and the share of the true
#    background **labelled vessel**. Lower is better for the last two.
# 2. **The heads** trained on those labels, on the held-out images, and a picture: the labels of one training
#    image for both settings, then both heads' predictions on the demo image.
#
# Colab does not rerun the cell when you move a slider: move it, then press ▶.

# %%
# @title Q15 — tune the pseudo-label threshold and the unsure band { display-mode: "form" }
PSEUDO_THRESHOLD = 0.5  # @param {type:"slider", min:0.45, max:0.75, step:0.025}
UNSURE_BAND = 0.0  # @param {type:"slider", min:0, max:0.1, step:0.01}
tune_pseudo_labels(PSEUDO_THRESHOLD, UNSURE_BAND)

# %% [markdown]
# <!-- callout: question | 👥 Pair moment (optional, 4 min) -->
# One of you tries a low cut (`PSEUDO_THRESHOLD = 0.55`), the other a high one (`0.7`). Compare your two
# error columns (true vessel labelled background, true background labelled vessel): which kind of wrong
# label did each of you buy, and what happened to the held-out row?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - The threshold trades one kind of wrong label for the other. A low cut turns more of SAM's region into
#   "vessel" (bg → vessel rises). A high cut keeps only SAM's surest pixels as vessel and the rest becomes
#   background (vessel → bg rises): the section-3 mistake again, with SAM as the annotator who stopped early.
#   When the probabilities crowd around one value, a step of 0.05 can flip most of the fundus.
# - The band ignores the doubtful pixels instead of guessing, so both error columns can fall at once. The
#   price is fewer labelled pixels (the "ignored" column).
# - Whether the held-out numbers follow the training diagnosis depends on your run. In a real project you
#   pick the setting on a few fully labelled images, never on the test images.
#
# </details>

# %% [markdown]
# <!-- callout: pitfall | Pitfall: over-trusting SAM's confident masks -->
# 1. SAM draws a crisp edge and fills the whole region. **Consequence:** it looks finished, with no "I don't
#    know" anywhere.
# 2. The head cannot tell a guess from a drawing. **Consequence:** it learns SAM's habit (a region where you
#    wanted a line) as faithfully as the vessels.

# %% [markdown]
# This simulation invents a teacher that is right about the strokes but confidently calls the brightest blob
# of the fundus (usually the optic disc) vessel. It trains one head on the strokes alone and one with that
# blind spot, and shows both on the same held-out image (no ground truth, seconds per head). What to look
# for: how much $\color{#6A1B9A}{\blacksquare}$ violet each head puts inside the
# $\color{#E65100}{\blacksquare}$ orange blob.

# %%
# @title Playful pitfall: a teacher with one confident blind spot { display-mode: "form" }
def bright_blob(item: dict) -> np.ndarray:
    """The brightest smooth blob of the fundus (usually the optic disc), found from the image alone."""
    g = ndi.gaussian_filter(item["image"][..., 1].astype(float), sigma=max(item["mask"].shape) / 60)
    g[~region(item)] = 0
    peak = np.unravel_index(int(g.argmax()), g.shape)
    lab, _ = ndi.label(g >= 0.9 * g.max())
    return ndi.binary_dilation(lab == lab[peak], iterations=max(1, max(item["mask"].shape) // 100))


def stroke_targets(item: dict, blind_spot: bool) -> np.ndarray:
    """The strokes only (1 vessel, 0 background, -1 unknown); with `blind_spot`, the teacher also calls the
    bright blob vessel wherever no stroke says otherwise."""
    t = np.full(item["mask"].shape, -1, np.int8)
    t[item["scribbles"] == 1] = 1
    t[item["scribbles"] == 2] = 0
    if blind_spot:
        t[bright_blob(item) & (item["scribbles"] == 0)] = 1
    return t


clean_head, _ = train_head(scrib_feats, [stroke_targets(it, False) for it in scrib_train], HEAD_STEPS)
biased_head, _ = train_head(scrib_feats, [stroke_targets(it, True) for it in scrib_train], HEAD_STEPS)
feat_demo = features_for(specialised, demo, side)
p_clean = predict_head(clean_head, feat_demo, demo["mask"].shape)
p_biased = predict_head(biased_head, feat_demo, demo["mask"].shape)
blob_demo = bright_blob(demo)
show_table(
    "Heads trained with and without the teacher's blind spot",
    [
        {
            "name": "head on the strokes alone",
            "blob": p_clean[blob_demo].mean(),
            "marked": p_clean[region(demo)].mean(),
        },
        {
            "name": "head on the biased teacher",
            "blob": p_biased[blob_demo].mean(),
            "marked": p_biased[region(demo)].mean(),
        },
    ],
    [("blob", "blind spot marked", "share of the blind spot it calls vessel", "pct"), "marked"],
    refs=[{"name": "the biased teacher itself", "blob": 1.0}],
    name_label="head",
    meta=f"held-out demo image · the blind spot covers {blob_demo.mean():.1%} of the image",
)
blob_rim = blob_demo & ~ndi.binary_erosion(blob_demo, iterations=max(2, max(demo["mask"].shape) // 200))


def head_on_blob(pred: np.ndarray) -> np.ndarray:
    """A head's mask in violet with the blind spot's outline in orange on top, so both are visible."""
    return overlay(overlay(demo["image"], pred, THEME["works"]), blob_rim, THEME["fails"], alpha=1.0)


show_masks(
    [
        (
            "the teacher's labels:\nthe blind spot is called vessel",
            overlay(demo["image"], blob_demo, THEME["fails"]),
        ),
        (f"head trained on the strokes alone\nmarks {p_clean[blob_demo].mean():.0%} of the blind spot",
         head_on_blob(p_clean)),
        (f"head trained on the biased teacher\nmarks {p_biased[blob_demo].mean():.0%} of the blind spot",
         head_on_blob(p_biased)),
    ],
    width=3.0,
    roles=["input", "prediction", "prediction"],
    legend=[
        ("invented", "orange: the blind spot (filled on the left, outlined on the right)"),
        ("mask", "violet: what the head marks as vessel"),
    ],
    note="no ground truth in this picture: the teacher's mistake is invented on purpose",
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - Compare the blind-spot share in the table with the $\color{#6A1B9A}{\blacksquare}$ violet inside the
#   $\color{#E65100}{\blacksquare}$ orange blob. If the biased head
#   marks the blob far more than the clean head, it copied the teacher's mistake onto an image it never saw
#   labelled.
# - It may also mark other bright patches, because it learnt "bright and round means vessel", not "this exact
#   blob".
# - If both heads mark about the same share on your run, the strokes alone already outweighed the blind spot.
#
# </details>

# %% [markdown]
# ### 6.3 Grad-CAM: where a classifier looks
#
# The second teacher does not segment at all.

# %% [markdown]
# <!-- diagram: s06-gradcam | Grad-CAM in three steps -->
# Three steps from a yes/no classifier to a pseudo-mask. Look at step 3: the map has one value per patch, and
# a vessel is thinner than a patch.

# %% [markdown]
# <details><summary>The same in words</summary>
#
# 1. We train a small classifier to answer one yes/no question.
# 2. We ask it *where* in the image its "yes" came from. Grad-CAM turns that answer into a heat map.
# 3. A threshold turns the heat map into a pseudo-mask.
#
# </details>

# %% [markdown]
# <!-- callout: theory | Theory: Grad-CAM, and why it fits this task poorly -->
# Grad-CAM shows which parts of the input made a classifier say "yes"
# ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)). It needs a question with a real "no", such as
# "is there a defect in this image?". Every retina image has vessels, so we ask about 64 px tiles instead:
# "does this tile contain a vessel?", with labels read off the strokes. The map has one value per 16 px patch
# and cannot outline a thin line. Seeing it fail shows what to check.

# %% [markdown]
# This helper cell defines the tile classifier, the tiling, the tile labels and its Grad-CAM map.

# %%
# @title Grad-CAM helpers: tile classifier, tiles, labels from strokes, the map { display-mode: "form" }
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class TileClassifier(torch.nn.Module):
    """'Does this tile contain a vessel?' on frozen DINOv3 features: 3 x 3 conv (384 -> 32), ReLU, one logit
    per patch, averaged. Fully convolutional, so it also runs on a whole image."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.hidden = torch.nn.Conv2d(dim, hidden, 3, padding=1)
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.hidden(x))).mean((2, 3))  # [B, 1]


def to_tiles(grid: torch.Tensor, t: int) -> torch.Tensor:
    """[C, h, w] -> [n, C, t, t]: non-overlapping t x t tiles, row by row (a ragged edge is dropped)."""
    c, h, w = grid.shape
    nh, nw = h // t, w // t
    g = grid[:, : nh * t, : nw * t].reshape(c, nh, t, nw, t)
    return g.permute(1, 3, 0, 2, 4).reshape(nh * nw, c, t, t)


def tile_labels(item: dict, grid: tuple[int, int], t: int) -> torch.Tensor:
    """Per tile, from the strokes only: 1 = a vessel stroke crosses it, 0 = only background strokes,
    -1 = no stroke (not used)."""
    has = [to_tiles(patch_share(item["scribbles"] == v, grid)[None], t).amax((1, 2, 3)) > 0 for v in (1, 2)]
    return torch.where(has[0], 1, torch.where(has[1], 0, -1))


def train_tile_classifier(x: torch.Tensor, y: torch.Tensor, steps: int, lr: float = 0.01):
    """Full-batch Adam with the balanced cross-entropy of every head; returns the classifier and seconds."""
    torch.manual_seed(SEED)
    clf = TileClassifier(x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    target = (y == 1).float()
    w = torch.where(y == 1, 0.5 / max(int((y == 1).sum()), 1), 0.5 / max(int((y == 0).sum()), 1))
    t0 = time.monotonic()
    for _ in range(steps):
        opt.zero_grad()
        bce = F.binary_cross_entropy_with_logits(clf(x)[:, 0], target, reduction="none")
        (bce * w).sum().backward()
        opt.step()
    return clf.eval(), time.monotonic() - t0


def cam_map(clf: TileClassifier, feat: torch.Tensor) -> np.ndarray:
    """Grad-CAM of the 'contains a vessel' score on a whole (standardised) feature grid: [h, w] in [0, 1]."""
    with GradCAM(model=clf, target_layers=[clf.act]) as cam:
        return cam(input_tensor=feat[None], targets=[ClassifierOutputTarget(0)])[0]


def held_out_vessel_tile_share(tile_patches: int) -> float:
    """Evaluation only: the share of held-out fundus tiles that truly contain a vessel."""
    shares = []
    for held in spec_eval:
        grid = tuple(features_for(specialised, held, side).shape[1:])
        vessel_share = patch_share(held["mask"] & region(held), grid)
        has_vessel = to_tiles(vessel_share[None], tile_patches).amax((1, 2, 3))
        inside = to_tiles(patch_share(region(held), grid)[None], tile_patches).mean((1, 2, 3)) > 0.5
        shares.append(float((has_vessel[inside] > 0).float().mean()))
    return float(np.mean(shares))


def cam_prob_maps(clf: TileClassifier, feats: list[torch.Tensor], items: list[dict]) -> list[np.ndarray]:
    """The Grad-CAM map of each training image, upsampled to the image size: [H, W] in [0, 1]."""
    maps = []
    for it, f in zip(items, feats, strict=True):
        small = torch.from_numpy(cam_map(clf, f))[None, None]
        up = F.interpolate(small, size=it["mask"].shape, mode="bilinear", align_corners=False)
        maps.append(up[0, 0].numpy())
    return maps


# %% [markdown]
# This cell cuts the five training images into 64 px tiles, labels each from the strokes (a vessel stroke
# crosses it → "vessel", only background strokes → "no vessel", no stroke → unused), trains the classifier in
# seconds and counts the tiles, with the held-out truth showing how rare a vessel-free tile really is.

# %%
# @title Train the tile classifier and count the tiles { display-mode: "form" }
TILE_PATCHES = 1 if SMOKE else 4  # 64 px of the 1024 px crop either way (smoke thumbnails are 4× smaller)
CLS_STEPS = 30 if SMOKE else 200
all_feats = torch.cat([f.flatten(1) for f in scrib_feats], 1)
cls_mu, cls_sd = all_feats.mean(1).view(-1, 1, 1), all_feats.std(1).view(-1, 1, 1) + 1e-6
std_feats = [(f - cls_mu) / cls_sd for f in scrib_feats]
tile_x, tile_y = [], []
for tile_item, tile_feat in zip(scrib_train, std_feats, strict=True):
    tile_lab = tile_labels(tile_item, tuple(tile_feat.shape[1:]), TILE_PATCHES)
    tile_x.append(to_tiles(tile_feat, TILE_PATCHES)[tile_lab >= 0])
    tile_y.append(tile_lab[tile_lab >= 0])
tile_x, tile_y = torch.cat(tile_x), torch.cat(tile_y)
clf, cls_sec = train_tile_classifier(tile_x, tile_y, CLS_STEPS)
with torch.no_grad():
    fit_acc = float(((clf(tile_x)[:, 0] > 0).long() == tile_y).float().mean())
vessel_tile_share = held_out_vessel_tile_share(TILE_PATCHES)
show_result(
    "Tile classifier on the stroke tiles",
    {
        "yes": int((tile_y == 1).sum()),
        "no": int((tile_y == 0).sum()),
        "acc": fit_acc,
        "held": vessel_tile_share,
    },
    [
        ("yes", "'vessel' tiles", "a vessel stroke crosses them", "text"),
        ("no", "'no vessel' tiles", "only background strokes", "text"),
        ("acc", "fit accuracy", "on these training tiles", "pct"),
        ("held", "held-out tiles with a vessel", "from the truth, not used", "pct"),
    ],
    meta=f"trained in {cls_sec:.1f} s on {DEVICE}",
    note="The held-out share is evaluation only: it shows how rare a vessel-free tile really is.",
)

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The classifier only ever answered "vessel in this tile, yes or no". Will its heat map look like a vessel
# tree, like blobs, or like noise? Write one guess.

# %% [markdown]
# This cell cuts each Grad-CAM map at half its peak into a pseudo-mask (strokes still win), trains the same
# head on it, scores it on the held-out images and shows one training image next to SAM's pseudo-label.

# %%
# @title Grad-CAM pseudo-labels: map, cut, head, score { display-mode: "form" }
CAM_THRESHOLD = 0.5  # share of the map's peak; Grad-CAM maps are rescaled to [0, 1] per image
t0 = time.monotonic()
cam_probs = cam_prob_maps(clf, std_feats, scrib_train)
cam_sec = round(time.monotonic() - t0, 1)
# the supervision: strokes, and the tile classifier's heat map (trained on stroke-derived tile labels)
cam_targets = [
    pseudo_targets(p, it["scribbles"], region(it), CAM_THRESHOLD, 0.0)
    for p, it in zip(cam_probs, scrib_train, strict=True)
]
cam_quality = pseudo_numbers([t == 1 for t in cam_targets], scrib_train)
cam_head, cam_head_sec = train_head(scrib_feats, cam_targets, HEAD_STEPS)
t0 = time.monotonic()
path2_preds["gradcam_pseudo"] = [
    predict_head(cam_head, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
]
cam_infer = round(time.monotonic() - t0, 1)
path2_report["gradcam_pseudo"] = rung_report(path2_preds["gradcam_pseudo"])
RESULTS["gradcam_iou"] = path2_report["gradcam_pseudo"]["iou"]  # held-out IoU of the head (SPEC §9)
RESULTS["gradcam_note"] = (
    f"Patch-level proxy '{specialised['gradcam']['patch_level_task']}' (tiles labelled from the strokes), "
    f"because every image contains the structure and there is no image-level negative class: an awkward "
    f"fit, {vessel_tile_share:.0%} of the held-out tiles contain foreground"
)
RESULTS["gradcam_head"] = path2_report["gradcam_pseudo"] | {
    "long_side": side,
    "n_train": len(scrib_train),
    "tile_patches": TILE_PATCHES,
    "n_tiles": [int((tile_y == 1).sum()), int((tile_y == 0).sum())],
    "classifier_fit_acc": fit_acc,
    "held_vessel_tile_share": vessel_tile_share,
    "threshold": CAM_THRESHOLD,
    "pseudo_label_iou_fit": cam_quality["iou"],
    "train_sec": round(float(cls_sec + cam_head_sec), 2),
    "pseudo_sec": cam_sec,
    "infer_sec": cam_infer,
}
cam_truth = float(np.mean([it["mask"][region(it)].mean() for it in scrib_train]))
show_table(
    "Grad-CAM and SAM pseudo-labels, checked against the truth",
    [
        {"name": "Grad-CAM labels", "colour": approach_colours["gradcam_pseudo"], **cam_quality},
        {"name": "SAM labels (cut 0.5)", "colour": approach_colours["sam_pseudo"], **sam_quality},
    ],
    ["iou", "recall", "marked"],
    refs=[{"name": "really vessel (the ground truth)", "marked": cam_truth, "truth": True}],
    name_label="pseudo-labels",
    meta=f"{len(scrib_train)} training images · Grad-CAM maps made in {cam_sec} s · never used for training",
)
path2_table(
    path2_report,
    "The heads trained on the strokes and the pseudo-labels",
    meta=f"{len(spec_eval)} held-out images · Grad-CAM head trained in {cam_head_sec:.1f} s on {DEVICE}",
)
show_masks(
    [
        (f"{scrib_ex['name']}", scrib_ex["image"]),
        ("Grad-CAM map", prob_rgb(cam_probs[0])),
        (
            "Grad-CAM pseudo-label\nvs ground truth",
            error_map(scrib_ex["image"], cam_targets[0] == 1, scrib_ex),
        ),
        ("SAM pseudo-label\nvs ground truth", error_map(scrib_ex["image"], sam_targets[0] == 1, scrib_ex)),
    ],
    width=2.8,
    roles=["input", "map", "errors", "errors"],
    legend=[*prob_key, *error_key],
    note="Grad-CAM map: blue = strongest evidence for vessel; the pseudo-label keeps every pixel above "
    "half the peak.",
)

# %% [markdown]
# **Q16.** 🟡 Compare the two pseudo-labels of the same training image. What shape does the Grad-CAM
# evidence have, and where does it sit? Which of the two would you rather correct by hand, and why?
#
# <details><summary>What to look for</summary>
#
# - Shape: blobs one or a few patches wide, where the classifier found the most "vessel-like" patches,
#   usually the thick, dark vessels. It does not follow a line, because its resolution is the 16 px patch and
#   its question does not need a line. After the cut, the blobs become fat regions with the thin vessels
#   missing. SAM fails differently, as one big region with a careful edge.
# - Which is better: neither is the vessel map. A coarse but *local* guess can teach the head more than a
#   sharp but global one, so compare the held-out rows above before you pick.
# - Other tasks: where the class really is a blob (a defect, a nucleus) and some images contain none, the
#   same recipe fits much better.
#
# </details>

# %% [markdown]
# ### 6.4 Path 2, side by side
#
# This cell shows the scribble head of section 4 and the two pseudo-label heads in one table, draws their
# error maps on the held-out demo image and writes the annotation effort for section 8: the same strokes
# for all three. What to look for: $\color{#6A1B9A}{\blacksquare}$ violet is what each head got right,
# $\color{#E65100}{\blacksquare}$ orange is what it invented, $\color{#2E7D32}{\blacksquare}$ green is
# what it missed. Recall is the share of the true vessels a head finds; "calls vessel" is the share of the
# fundus it calls vessel, next to the share that really is (the "really vessel" row).

# %%
# @title Path 2 side by side: table and error maps { display-mode: "form" }
RESULTS["path2_effort_min"] = {
    k: round(effort_minutes(n_strokes=n_strokes), 2) for k in ("sam_pseudo", "gradcam_pseudo")
}
path2_table(
    path2_report,
    "The scribble head and the two pseudo-label heads",
    meta=f"{len(spec_eval)} held-out images · annotation for all three: {n_strokes} strokes on "
    f"{len(scrib_train)} images (≈ {effort_minutes(n_strokes=n_strokes):.1f} min)",
    note="The pseudo-labels add compute, not annotation.",
    refs=[{"name": "everything is foreground", "iou": RESULTS["specialised_everything_iou"], "marked": 1.0}],
)


def head_title(label: str, pred: np.ndarray, item: dict) -> str:
    """Panel title: the head, then its IoU, recall and the share of this fundus it calls vessel."""
    r = region(item)
    p, t = pred & r, item["mask"] & r
    return (
        f"{label}\nIoU {score(pred, item)['iou']:.2f} · recall {hidden_recall(p, t):.2f}\n"
        f"calls {100 * p[r].mean():.0f} % vessel (really {100 * t[r].mean():.0f} %)"
    )


show_masks(
    [
        (
            head_title(approach_names.get(k, k).split(":")[-1].strip(), p[k_demo], demo),
            error_map(demo["image"], p[k_demo], demo),
        )
        for k, p in path2_preds.items()
    ]
    + [("ground truth\n(every vessel, green)", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.8,
    roles=["errors"] * len(path2_preds) + ["truth"],
    legend=error_key,
    note="held-out demo image · recall = true vessels found · calls … vessel = share of the fundus it "
    "calls vessel",
)

# %% [markdown]
# <!-- callout: takeaway | Takeaway: Path 2 -->
# - **You are about to train on a teacher's guesses** → score them on a few fully labelled images first, and
#   keep the person's strokes stronger than the teacher.
# - **Its probabilities crowd around 0.5** → do not trust a hard cut. Tune the threshold on training images
#   and ignore the doubtful band instead of guessing.
# - **Every image has the structure, so there is no "no" class** → Grad-CAM is a poor fit. Use it when you
#   have real image-level labels (defects, tumours).
# - **You need more labels but cannot pay for strokes** → pseudo-labels add compute, not annotation, and
#   bring the teacher's bias with them.
