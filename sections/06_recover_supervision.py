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
# This helper cell adds a colour for each head of this path and defines how to spread SAM clicks along the
# strokes of one kind, vessel or background.

# %%
# @title Path 2 helpers: colours, clicks along the strokes { display-mode: "form" }
approach_colours |= {
    "sam_pseudo": "#0277BD",
    "gradcam_pseudo": "#827717",
}
approach_names |= {
    "sam_pseudo": "pseudo-labels: SAM from stroke clicks",
    "gradcam_pseudo": "pseudo-labels: Grad-CAM of a tile classifier",
}


def prob_rgb(prob: np.ndarray) -> np.ndarray:
    """A probability map as RGB on a fixed scale: purple = 1, white = 0.5 (unsure), orange = 0."""
    return (mpl.colormaps["PuOr"](np.clip(prob, 0, 1))[..., :3] * 255).astype(np.uint8)


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


# %% [markdown]
# ### 6.1 SAM fills in the gaps
#
# We take the scribbles of section 4 again: a few vessel strokes and a few background strokes on each of the
# same five training images.
#
# - Twenty clicks spread along the vessel strokes go in as "vessel", twenty along the background strokes as
#   "not vessel".
# - SAM's answer for every pixel is the pseudo-label.
# - Only the strokes go in. The ground truth appears in the figure so you can judge the guesses, and never
#   in training.

# %% [markdown]
# This helper cell defines `sam_pseudo_labels`: it asks SAM for a probability map on each training image (or
# loads our maps), prints how sure SAM is and draws one image.

# %%
# @title SAM pseudo-label helper: probability maps from the stroke clicks { display-mode: "form" }
def sam_pseudo_labels(run_sam: bool, n_clicks: int):
    """SAM's probability map for each training image from `n_clicks` + `n_clicks` clicks on the strokes
    (or our precomputed maps). Prints the numbers and draws one image. Returns the maps, the clicks, the
    seconds, SAM's quality at the 0.5 cut and the share of the fundus it marks."""
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
    quality = mean_scores([score(p >= 0.5, it) for p, it in zip(probs, scrib_train, strict=True)])
    marked = float(np.mean([(p >= 0.5)[region(it)].mean() for p, it in zip(probs, scrib_train, strict=True)]))
    # how sure SAM is, inside the fundus (no truth needed): the spread of its probabilities
    spread = np.quantile(
        np.concatenate([p[region(it)] for p, it in zip(probs, scrib_train, strict=True)]), [0.05, 0.5, 0.95]
    )
    print(
        f"SAM pseudo-labels on {len(scrib_train)} training images · {src} · {sec} s on {DEVICE}\n"
        f"SAM's probability inside the fundus · 5 % of the pixels below {spread[0]:.2f} · median "
        f"{spread[1]:.2f} · 95 % below {spread[2]:.2f}\n"
        f"diagnosis (truth used to score only): at 0.5 they mark {marked:.0%} of the fundus · IoU "
        f"{quality['iou']:.3f} · clDice {quality['cldice']:.3f}"
    )
    ex, (pos, neg) = scrib_train[0], clicks_used[0]
    show_clicks(
        [
            (f"{ex['name']} · vessel clicks", ex["image"], pos),
            ("background clicks", ex["image"], neg),
            ("SAM probability: purple 1, white 0.5, orange 0", prob_rgb(probs[0]), None),
            ("pseudo-label at 0.5 vs truth", error_map(ex["image"], probs[0] >= 0.5, ex), None),
        ],
        width=2.8,
    )
    return probs, clicks_used, sec, quality, marked


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Section 2 showed what SAM does with clicks on a vessel tree. Now it gets forty clicks per image, half of
# them saying "not here". Will its mask follow the vessels, or cover a region around them? Write one guess.

# %% [markdown]
# How to read the printout and the last panel of the next cell:
#
# - **Probability spread:** the 5 %, 50 % and 95 % points of SAM's probability over the fundus pixels. Near 0
#   or 1 means sure, near 0.5 means unsure.
# - **Marked:** the share of fundus pixels the pseudo-label calls vessel (section 2 printed how much is
#   really vessel).
# - **IoU, clDice:** overlap with the hidden truth, as in section 2. Diagnosis only.
# - **Colours:** violet = a real vessel found, orange = a vessel SAM invented, green = a vessel SAM left out.
#   Every orange and green pixel becomes a wrong training label if we take the guesses as they are.

# %% [markdown]
# This cell asks SAM for a probability map on each training image from the stroke clicks and shows one image
# (seconds per image on a T4, much slower on a CPU, where `RUN_SAM = False` loads our maps, made the same way
# with clicks drawn at random).

# %%
# @title Run SAM on the stroke clicks to get pseudo-labels (False = load our maps) { display-mode: "form" }
RUN_SAM = True  # @param {type:"boolean"}
N_PSEUDO_CLICKS = 20  # @param {type:"integer"}
pseudo_probs, pseudo_clicks, pseudo_sec, sam_quality, sam_marked = sam_pseudo_labels(RUN_SAM, N_PSEUDO_CLICKS)
ex = scrib_train[0]

# %% [markdown]
# **Q24.** 🟡 Look at the error map and the probability panel. Where does SAM's guess go wrong: around
# the thick vessels, along the thin ones, in the background between the strokes? Is SAM sure of itself (deep
# purple, deep orange) or unsure (pale)? What does that mean for a cut at exactly 0.5?
#
# <details><summary>What to look for</summary>
#
# SAM looks for *objects* that contain the positive clicks and avoid the negative ones.
#
# - Forty clicks scattered over a vessel tree do not describe one object. SAM settles for the object that
#   holds most of them, often the whole fundus. Its edge is careful, but the inside is a region (orange), not
#   a tree. Thin branches between the clicks can drop out (green).
# - If most of the probability panel is pale, the values sit just around 0.5 and the exact cut decides
#   almost everything: a little below and the whole fundus is "vessel", a little above and nothing is. The
#   binary mask hides that doubt, because it looks just as crisp as a confident one.
# - Whatever you see on your run, the threshold is your choice, not a property of SAM.
#
# </details>

# %% [markdown]
# ### 6.2 From a probability map to training targets
#
# A target is the label the head trains on: 1 vessel, 0 background, −1 ignored (the loss skips it). The cut
# turns SAM's probability into a target, and an unsure band lets us ignore the doubtful pixels instead of
# guessing.

# %% [markdown]
# This cell draws the rule once, for a cut of 0.5 and a band of 0.1.

# %%
# @title The cut and the unsure band, drawn { display-mode: "form" }
cut, band_demo = 0.5, 0.1
fig, ax = plt.subplots(figsize=(7.2, 1.5))
for lo, hi, colour, label in [
    (0, cut - band_demo, THEME["background"], "0 · background"),
    (cut - band_demo, cut + band_demo, "white", "−1 · ignored"),
    (cut + band_demo, 1, THEME["works"], "1 · vessel"),
]:
    ax.barh(0, hi - lo, left=lo, color=colour, edgecolor="black", height=0.5)
    text_colour = "black" if colour == "white" else "white"
    ax.text((lo + hi) / 2, 0, label, ha="center", va="center", fontsize=8, color=text_colour)
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xlabel("SAM's probability for a pixel")
ax.set_title(
    f"cut {cut}, band {band_demo}: below {cut - band_demo} → 0 · between → −1 · from {cut + band_demo} → 1",
    fontsize=9,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# Two rules sit on top of the cut: a person's stroke always wins over SAM, and pixels outside the fundus are
# always −1.

# %% [markdown]
# This helper cell holds our solution of the next exercise. Open it only if you are stuck.

# %%
# @title Reference solution and play example (open only if stuck) { display-mode: "form" }
def _ref_pseudo_targets(
    prob: np.ndarray, scribbles: np.ndarray, region: np.ndarray, threshold: float = 0.5, band: float = 0.0
) -> np.ndarray:
    """Reference: sure guesses become labels, doubtful ones are ignored, strokes and region come last."""
    targets = np.full(prob.shape, -1, np.int8)
    targets[prob >= threshold + band] = 1
    targets[prob < threshold - band] = 0
    targets[scribbles == 1] = 1
    targets[scribbles == 2] = 0
    targets[~np.asarray(region, bool)] = -1
    return targets


_play_prob = np.array([[0.95, 0.62, 0.45, 0.10], [0.55, 0.30, 0.71, 0.05], [0.20, 0.90, 0.40, 0.80]])
_play_scrib = np.zeros((3, 4), np.uint8)
_play_scrib[1, 1] = 1  # a vessel stroke where SAM says "not vessel"
_play_scrib[2, 1] = 2  # a background stroke where SAM says "vessel"
_play_region = np.ones((3, 4), bool)
_play_region[:, 3] = False  # the last column is outside the evaluation region


def _check_pseudo_targets(f):
    """What `use` compares: the targets of the play map without and with an unsure band."""
    return [np.asarray(f(_play_prob, _play_scrib, _play_region, 0.5, b)).astype(int) for b in (0.0, 0.2)]


def show_pseudo_play(fn) -> None:
    """Print the targets `fn` makes on the 3 x 4 play map (threshold 0.5, band 0.2)."""
    print("play example · targets with threshold 0.5 and band 0.2 (−1 = ignored):")
    print(fn(_play_prob, _play_scrib, _play_region, 0.5, 0.2))


# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 3 min) -->
# Fill `my_pseudo_targets` in the next cell.
#
# - **In:** SAM's `prob` map (floats in [0, 1]), the `scribbles` map (0 unlabelled, 1 vessel stroke, 2
#   background stroke), the boolean `region`, a `threshold` and an unsure `band`.
# - **Out:** int8 targets, 1 vessel, 0 background, −1 ignored.
# - **Rules:** `prob ≥ threshold + band` is vessel, `prob < threshold − band` is background, the pixels in
#   between are ignored. Strokes always win. Pixels outside `region` are always −1.
#
# The cell checks your version on a 3 × 4 play map, without and with a band, and continues with ours if you
# skip it.


# %%
# ✍️ YOUR CODE
def my_pseudo_targets(
    prob: np.ndarray, scribbles: np.ndarray, region: np.ndarray, threshold: float = 0.5, band: float = 0.0
) -> np.ndarray:
    """Targets (int8) from a pseudo-label probability map: 1 where prob >= threshold + band, 0 where
    prob < threshold - band, -1 in between. Strokes overrule the map (1 vessel stroke -> 1, 2 background
    stroke -> 0); outside `region` -> -1.
    e.g. threshold 0.5, band 0.1, no strokes: prob [[0.9, 0.55, 0.3, 0.05]] -> [[1, -1, 0, 0]]"""
    ...


pseudo_targets = use(
    my_pseudo_targets, reference=_ref_pseudo_targets, check=_check_pseudo_targets, name="pseudo_targets"
)
show_pseudo_play(pseudo_targets)

# %% [markdown]
# This helper cell defines `train_sam_head`: it turns SAM's maps into targets at the usual cut and trains the
# section-2 head on them, or loads our checkpoint.

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
# This cell scores the new head on the held-out images with the numbers of section 4 and prints it under the
# scribble head, which saw the same strokes but kept every other pixel unknown.

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


def print_path2(reports: dict) -> None:
    """One line per head with the numbers of section 4."""
    print(f"{'approach':<46}{COLUMNS}")
    for k, v in reports.items():
        name = approach_names.get(k, k)
        print(
            f"{name:<46}{v['iou']:>6.3f}{v['cldice']:>8.3f}{v['recall']:>8.3f}{v['precision']:>8.3f}"
            f"{v['marked']:>8.0%}"
        )


print_path2(path2_report)

# %% [markdown]
# This helper cell defines `tune_pseudo_labels`: it makes targets with your cut and band, trains the head on
# them and prints the diagnosis columns and the held-out line.

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
    printed under the head at 0.5."""
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
    print(f"{'targets on the training images':<46}{'ignored':>8}{'vessel → bg':>13}{'bg → vessel':>13}")
    for name, t in {"threshold 0.5, no band": sam_targets, "your setting": tuned_targets}.items():
        e = target_errors(t)
        print(f"{name:<46}{e['ignored']:>8.0%}{e['vessel_as_bg']:>13.0%}{e['bg_as_vessel']:>13.0%}")
    print(f"\nyour head: threshold {threshold}, band {band} · trained in {tuned_sec:.1f} s")
    print_path2({"threshold 0.5, no band": path2_report["sam_pseudo"], "your setting": tuned_report})


# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell, then move the cut.
#
# **Q25.** 🟡 Tune the pseudo-labels on the **training images**, never on the test images. Raise or lower
# `PSEUDO_THRESHOLD` and give the doubtful pixels an `UNSURE_BAND` to ignore, picking values from the spread
# printed in 6.1 and judging them by how many training targets are wrong (the two error columns). Then look
# at the held-out line once: does it move the same way? Is there a setting that beats 0.5 on your run?
#
# - `PSEUDO_THRESHOLD`: try 0.55, 0.6, 0.7 (or the median printed in 6.1).
# - `UNSURE_BAND`: try 0.03; pixels within 0.03 of the threshold are ignored.
#
# Both are sliders in the next cell: move one, then run it again.

# %% [markdown]
# This cell trains the same head on targets made with your setting and prints it under the head at 0.5, with
# three diagnosis columns from the hidden truth on the training images: **ignored** (the share of the fundus
# set to −1), **vessel → bg** (the share of true vessel pixels that got label 0) and **bg → vessel** (the
# share of true background pixels that got label 1).

# %%
# @title Q25 — tune the pseudo-label threshold and the unsure band { display-mode: "form" }
PSEUDO_THRESHOLD = 0.5  # @param {type:"slider", min:0.45, max:0.75, step:0.025}
UNSURE_BAND = 0.0  # @param {type:"slider", min:0, max:0.1, step:0.01}
tune_pseudo_labels(PSEUDO_THRESHOLD, UNSURE_BAND)

# %% [markdown]
# <!-- callout: question | 👥 Pair moment (optional, 4 min) -->
# One of you tries a low cut (`PSEUDO_THRESHOLD = 0.55`), the other a high one (`0.7`). Compare your **vessel
# → bg** and **bg → vessel** columns: which kind of wrong label did each of you buy, and what happened to the
# held-out line?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The threshold trades one kind of wrong label for the other.
#
# - **Low cut:** more of SAM's region becomes "vessel" (bg → vessel rises) and the head learns to mark more.
# - **High cut:** only SAM's surest pixels stay vessel and the rest becomes background (vessel → bg rises).
#   That is the section-3 mistake again, with SAM as the annotator who stopped early.
# - **Crowded probabilities:** when they crowd around one value, a step of 0.05 can flip most of the fundus
#   from one side to the other, so there may be no comfortable middle.
# - **The band:** it removes the doubtful pixels instead of guessing, so both error columns can fall at once.
#   The price is fewer labelled pixels (the "ignored" column).
#
# Whether the held-out numbers follow the training diagnosis depends on your run. In a real project you pick
# the setting on a small fully labelled check set, never on the test images.
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
# of the fundus (usually the optic disc) vessel, trains one head on the strokes alone and one with that blind
# spot, and shows both on the same held-out image (no ground truth, seconds per head).

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
print(
    f"blind spot ({blob_demo.mean():.1%} of the image) marked as vessel · head on strokes alone "
    f"{p_clean[blob_demo].mean():.0%} · head on the biased teacher {p_biased[blob_demo].mean():.0%}\n"
    f"fundus marked as vessel · strokes alone {p_clean[region(demo)].mean():.0%} · "
    f"biased teacher {p_biased[region(demo)].mean():.0%}"
)
show_masks(
    [
        ("the teacher's blind spot", overlay(demo["image"], blob_demo, THEME["fails"])),
        ("head on the strokes alone", overlay(demo["image"], p_clean, THEME["works"])),
        ("head on the biased teacher", overlay(demo["image"], p_biased, THEME["works"])),
    ],
    width=3.0,
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# Compare the blind-spot share in the printout and the violet area inside the orange blob. If the biased head
# marks the blob far more than the clean head, it copied the teacher's mistake onto an image it never saw
# labelled. It may also mark other bright patches, because it learnt "bright and round means vessel", not
# "this exact blob". If both heads mark about the same share on your run, the strokes alone already
# outweighed the blind spot. Try another `demo` image in section 2 and compare.
#
# </details>

# %% [markdown]
# ### 6.3 Grad-CAM: where a classifier looks
#
# The second colleague does not segment at all.
#
# 1. We train a small classifier to answer one yes/no question.
# 2. We ask it *where* in the image its "yes" came from. Grad-CAM turns that answer into a heat map.
# 3. A threshold turns the heat map into a pseudo-mask.

# %% [markdown]
# <!-- callout: pitfall | Pitfall: an awkward fit, said plainly -->
# Grad-CAM needs an image-level question with a real "no" class, such as "is there a defect in this image?".
#
# - Every retina image contains vessels, so there is no "no" at image level.
# - We fall back to a patch-level question: "does this 64 px tile contain a vessel?", with tile labels read
#   off the strokes.
# - Most tiles of a fundus crop do contain a vessel (the next cell counts them), and a map at patch
#   resolution cannot outline a line a few pixels wide, so it fits this task poorly.
# - We look at it anyway: on a task with real image-level labels (defects, tumours, "contains a cell") it is
#   often the cheapest supervision there is, and seeing it fail here shows what to check.

# %% [markdown]
# <!-- callout: theory | Theory: Grad-CAM -->
# Grad-CAM answers: which parts of the input made the classifier say "yes"?
#
# - **How:** each of the classifier's feature maps Aₖ gets a weight αₖ (how much the "yes" score grows when Aₖ
#   grows). The heat map is the weighted sum of the maps, keeping only the positive evidence.
# - **Pets:** for "is there a cat?" the map lights up the face, not the whole cat.
# - **Here:** one value per 16 px patch, because the features have that resolution.

# %% [markdown]
# <!-- callout: go-deeper | Grad-CAM formula, CAM and the package -->
# The formula: `αₖ = mean over (i, j) of ∂score/∂Aₖ[i, j]`, then `map = ReLU(Σₖ αₖ·Aₖ)`, rescaled to [0, 1].
#
# Our classifier ends in an average over patches. For such a classifier Grad-CAM reduces to the older CAM: the
# classifier's own evidence for "vessel", patch by patch. We use the `grad-cam` package.
#
# - Grad-CAM, Selvaraju et al., ICCV 2017: [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
# - CAM, Zhou et al., CVPR 2016: [arXiv:1512.04150](https://arxiv.org/abs/1512.04150)

# %% [markdown]
# This helper cell defines the tile classifier, the tiling, the tile labels from the strokes, its training and
# its Grad-CAM map on a whole image.

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
for it, f in zip(scrib_train, std_feats, strict=True):
    lab = tile_labels(it, tuple(f.shape[1:]), TILE_PATCHES)
    tile_x.append(to_tiles(f, TILE_PATCHES)[lab >= 0])
    tile_y.append(lab[lab >= 0])
tile_x, tile_y = torch.cat(tile_x), torch.cat(tile_y)
clf, cls_sec = train_tile_classifier(tile_x, tile_y, CLS_STEPS)
with torch.no_grad():
    fit_acc = float(((clf(tile_x)[:, 0] > 0).long() == tile_y).float().mean())
# evaluation only: the share of held-out tiles that truly contain a vessel
held_tiles = []
for it in spec_eval:
    grid = tuple(features_for(specialised, it, side).shape[1:])
    share = to_tiles(patch_share(it["mask"] & region(it), grid)[None], TILE_PATCHES).amax((1, 2, 3))
    inside = to_tiles(patch_share(region(it), grid)[None], TILE_PATCHES).mean((1, 2, 3)) > 0.5
    held_tiles.append(float((share[inside] > 0).float().mean()))
vessel_tile_share = float(np.mean(held_tiles))
print(
    f"tile classifier · {int((tile_y == 1).sum())} 'vessel' tiles, {int((tile_y == 0).sum())} 'no vessel' "
    f"tiles from the strokes · fit accuracy {fit_acc:.0%} · {cls_sec:.1f} s on {DEVICE}"
)
print(f"held-out truth (evaluation only): {vessel_tile_share:.0%} of the fundus tiles contain a vessel")

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The classifier only ever answered "vessel in this tile, yes or no". Will its heat map look like a vessel
# tree, like blobs, or like noise? Write one guess.

# %% [markdown]
# This cell cuts each training image's Grad-CAM map at half its peak into a pseudo-mask (strokes still win),
# trains the same head on it in seconds, scores it on the held-out images and shows one training image.

# %%
# @title Grad-CAM pseudo-labels: map, cut, head, score { display-mode: "form" }
CAM_THRESHOLD = 0.5  # share of the map's peak; Grad-CAM maps are rescaled to [0, 1] per image
t0 = time.monotonic()
cam_probs = []
for it, f in zip(scrib_train, std_feats, strict=True):
    cam = torch.from_numpy(cam_map(clf, f))[None, None]
    cam_probs.append(
        F.interpolate(cam, size=it["mask"].shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    )
cam_sec = round(time.monotonic() - t0, 1)
# the supervision: strokes, and the tile classifier's heat map (trained on stroke-derived tile labels)
cam_targets = [
    pseudo_targets(p, it["scribbles"], region(it), CAM_THRESHOLD, 0.0)
    for p, it in zip(cam_probs, scrib_train, strict=True)
]
cam_quality = mean_scores([score(t == 1, it) for t, it in zip(cam_targets, scrib_train, strict=True)])
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
cam_marked = float(
    np.mean([(t == 1)[region(it)].mean() for t, it in zip(cam_targets, scrib_train, strict=True)])
)
print(
    f"Grad-CAM pseudo-labels · {cam_sec} s · diagnosis (truth used to score only): they mark "
    f"{cam_marked:.0%} of the fundus · IoU {cam_quality['iou']:.3f} · clDice {cam_quality['cldice']:.3f} "
    f"on the training images (SAM at 0.5: {sam_marked:.0%}, IoU {sam_quality['iou']:.3f})"
)
print(f"Grad-CAM head trained in {cam_head_sec:.1f} s on {DEVICE}")
print_path2(path2_report)
show_masks(
    [
        (f"{ex['name']}", ex["image"]),
        ("Grad-CAM: purple = strongest evidence", prob_rgb(cam_probs[0])),
        ("Grad-CAM pseudo-label vs truth", error_map(ex["image"], cam_targets[0] == 1, ex)),
        ("SAM pseudo-label vs truth", error_map(ex["image"], sam_targets[0] == 1, ex)),
    ],
    width=2.8,
)

# %% [markdown]
# **Q26.** 🟡 Compare the two pseudo-labels of the same training image. What shape does the Grad-CAM
# evidence have, and where does it sit? Which of the two would you rather correct by hand, and why?
#
# <details><summary>What to look for</summary>
#
# - **Shape:** blobs one or a few patches wide, where the classifier found the most "vessel-like" patches:
#   usually the thick, dark vessels, sometimes dark background. It does not follow a line, because its
#   resolution is the 16 px patch and because its question ("is there a vessel somewhere in this tile?")
#   does not need a line.
# - **After the cut:** fat regions around the vessels the classifier found most convincing, with the thin
#   ones mostly missing. SAM's guess fails differently, as one big region with a careful edge.
# - **Which is better:** neither is the vessel map. A coarse but *local* guess can still teach the head more
#   than a sharp but global one, so compare the held-out lines above before you pick.
# - **Other tasks:** where the class really is a blob (a defect, a nucleus) and some images contain none,
#   the same recipe fits much better.
#
# </details>

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · how the heat map is made -->
# Two cells for the curious. The first runs the four steps of Grad-CAM by hand on one training image, and
# checks the result against the package. The second draws the three shapes a heat map could take (a vessel
# tree, blobs, noise), to compare with the real one.

# %%
# @title Grad-CAM step by step on one training image { display-mode: "form" }
cam_feat = std_feats[0]
with torch.enable_grad():
    act = clf.act(clf.hidden(cam_feat[None]))  # the feature maps A_k: [1, 32, h, w]
    yes_score = clf.out(act).mean((2, 3))[0, 0]  # the "vessel" score
    grads = torch.autograd.grad(yes_score, act)[0]  # how the score changes with each A_k pixel
maps_a = act[0].detach().cpu().numpy()
alpha = grads[0].mean((1, 2)).cpu().numpy()  # step 1: one weight per feature map
weighted = np.tensordot(alpha, maps_a, axes=1)  # step 2: sum of alpha_k * A_k
positive = np.maximum(weighted, 0)  # step 3: keep only the evidence FOR "vessel"
positive = positive - positive.min()
steps_heat = positive / (1e-7 + positive.max())  # step 4: rescale to [0, 1]
k_hi, k_lo = int(alpha.argmax()), int(alpha.argmin())
fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))
axes[0].imshow(ex["image"])
axes[0].set_title("training image", fontsize=9)
axes[1].imshow(maps_a[k_hi], cmap="Greys")
axes[1].set_title(f"A_k with the largest α ({alpha[k_hi]:+.2f})", fontsize=8)
axes[2].imshow(maps_a[k_lo], cmap="Greys")
axes[2].set_title(f"A_k with the smallest α ({alpha[k_lo]:+.2f})", fontsize=8)
sorted_alpha = np.sort(alpha)
axes[3].bar(
    range(len(alpha)), sorted_alpha, color=[THEME["works"] if a > 0 else THEME["fails"] for a in sorted_alpha]
)
axes[3].set_title("1 · weights α_k (one per map)", fontsize=8)
limit = np.abs(weighted).max() + 1e-9
axes[4].imshow(weighted, cmap="PuOr", vmin=-limit, vmax=limit)
axes[4].set_title("2 · Σ α_k·A_k (purple +, orange −)", fontsize=8)
axes[5].imshow(prob_rgb(steps_heat))
axes[5].set_title("3 + 4 · ReLU, rescale: the heat map", fontsize=8)
for a in axes:
    if a is not axes[3]:
        a.axis("off")
plt.tight_layout()
plt.show()
print(
    f"{(alpha > 0).sum()} of {len(alpha)} feature maps have a positive weight · "
    f"max difference to the grad-cam package: {np.abs(steps_heat - cam_map(clf, cam_feat)).max():.4f}"
)


# %%
# @title Sketch: three shapes a heat map could take { display-mode: "form" }
sketch_rng = np.random.default_rng(SEED)
tree = np.zeros((48, 48))
for x0, y0, x1, y1 in [(24, 47, 24, 18), (24, 30, 8, 8), (24, 26, 42, 6), (24, 38, 40, 30), (24, 40, 10, 32)]:
    for s in np.linspace(0, 1, 80):
        tree[int(y0 + s * (y1 - y0)), int(x0 + s * (x1 - x0))] = 1
tree = ndi.gaussian_filter(tree, 0.8)
yy, xx = np.mgrid[0:48, 0:48]
centres = [(14, 14), (32, 28), (18, 38)]
blobs = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5**2)) for cx, cy in centres)
noise = ndi.gaussian_filter(sketch_rng.random((48, 48)), 0.7)
shapes = [("vessel tree: thin lines", tree), ("blobs: a few patches", blobs), ("noise: no structure", noise)]
show_masks(
    [(title, prob_rgb(0.5 + 0.5 * (m - m.min()) / (m.max() - m.min()))) for title, m in shapes],
    width=2.0,
)

# %% [markdown]
# This helper cell defines `cam_sam_prompt`: it picks SAM clicks on each training image's heat map, asks SAM
# for a mask and scores the three kinds of pseudo-label on the training images (diagnosis only).

# %%
# @title Grad-CAM-as-prompt helper: heat map to SAM clicks to masks { display-mode: "form" }
def cam_sam_prompt(n_clicks: int) -> None:
    """Hottest `n_clicks` patches of each heat map = vessel clicks, coldest = background clicks; SAM turns
    them into masks. Prints the three pseudo-labels' quality and draws one image."""
    if not float(n_clicks).is_integer() or n_clicks < 1:
        raise SystemExit("N_CAM_CLICKS must be a whole number of at least 1.")
    n_clicks = int(n_clicks)
    masks, clicks_used = [], []
    for it, f in tqdm(list(zip(scrib_train, std_feats, strict=True)), desc="Grad-CAM, then SAM"):
        s = torch.from_numpy(cam_map(clf, f)).to(DEVICE)
        keep = patch_share(region(it), tuple(s.shape))
        pos = top_points(s, keep, n_clicks, it["mask"].shape, largest=True)
        neg = top_points(s, keep, n_clicks, it["mask"].shape, largest=False)
        prob = sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
        masks.append(prob >= 0.5)
        clicks_used.append(pos)
    quality = mean_scores([score(m, it) for m, it in zip(masks, scrib_train, strict=True)])
    RESULTS["gradcam_sam_prompt_iou_fit"] = quality["iou"]
    print("pseudo-labels on the training images (diagnosis only: scored against the hidden truth)")
    for name, q in {
        "SAM from stroke clicks (6.1, at 0.5)": sam_quality,
        f"Grad-CAM cut at {CAM_THRESHOLD}": cam_quality,
        f"SAM from {n_clicks} + {n_clicks} Grad-CAM clicks": quality,
    }.items():
        print(f"{name:<44} IoU {q['iou']:.3f} · clDice {q['cldice']:.3f}")
    show_clicks(
        [
            ("Grad-CAM clicks: vessel", ex["image"], clicks_used[0]),
            ("SAM from Grad-CAM clicks vs truth", error_map(ex["image"], masks[0], ex), clicks_used[0]),
        ],
        width=3.4,
    )


# %% [markdown]
# <!-- callout: question | Question (optional) -->
# Run the next cell to use the heat map as a SAM prompt instead of a mask.
#
# **Q27.** 🔴 Instead of cutting the heat map, let it pick SAM clicks: the hottest patches as "vessel", the
# coldest as "not vessel" (the trick of section 4.2, with Grad-CAM in place of the prototype). Do SAM's
# masks from these clicks look more like the SAM pseudo-labels of 6.1 or like the Grad-CAM blobs?
# `N_CAM_CLICKS` (clicks of each kind per image) is a slider in the next cell: try 2 and 10.

# %% [markdown]
# This cell picks SAM clicks on each training image's heat map, asks SAM for a mask and scores the three kinds
# of pseudo-label on the training images (diagnosis only, no training).

# %%
# @title Q27 — Grad-CAM as a SAM prompt: hottest patches = vessel clicks { display-mode: "form" }
N_CAM_CLICKS = 5  # @param {type:"slider", min:1, max:10, step:1}
cam_sam_prompt(N_CAM_CLICKS)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - The clicks land where the heat map is hottest, usually on or next to a thick vessel. SAM's mask then
#   looks like its masks of 6.1: sharp edges around whatever object contains the clicks, often a region
#   rather than a line.
# - The prompt route swaps Grad-CAM's coarse blobs for SAM's sharp edges, but it brings back SAM's idea of an
#   object. A click that lands on dark background drags the mask there.
# - Two weak signals combined are not automatically a strong one: their mistakes can add up as easily as
#   cancel out.
#
# </details>

# %% [markdown]
# ### 6.4 Path 2, side by side
#
# This cell puts the scribble head of section 4 and the SAM and Grad-CAM pseudo-label heads side by side
# (bars in the approach colours, error maps on the held-out demo image) and writes the annotation effort
# for section 8: the same strokes for all three.

# %%
# @title Path 2 side by side: table, bars and error maps { display-mode: "form" }
RESULTS["path2_effort_min"] = {
    k: round(effort_minutes(n_strokes=n_strokes), 2) for k in ("sam_pseudo", "gradcam_pseudo")
}
print_path2(path2_report)
print(
    f"{'everything is foreground':<46}{RESULTS['specialised_everything_iou']:>6.3f}\n"
    f"annotation for all three: {n_strokes} strokes on {len(scrib_train)} images "
    f"(≈ {effort_minutes(n_strokes=n_strokes):.1f} min); the pseudo-labels add compute, not annotation"
)
metrics = ["iou", "cldice", "recall", "precision"]
x, width = np.arange(len(metrics)), 0.8 / len(path2_report)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for n, (k, v) in enumerate(path2_report.items()):
    shift = (n - (len(path2_report) - 1) / 2) * width
    ax.bar(x + shift, [v[m] for m in metrics], width, color=approach_colours[k], label=approach_names[k])
ax.axhline(
    RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--", label='IoU of "everything"'
)
ax.set_xticks(x, ["IoU", "clDice", "recall", "precision"])
ax.set_ylim(0, 1)
ax.set_title("Path 2: same strokes, three ways to fill in the rest")
ax.legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout()
plt.show()
show_masks(
    [
        (
            f"{approach_names[k].split(':')[-1].strip()} · IoU {score(p[k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], p[k_demo], demo),
        )
        for k, p in path2_preds.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.8,
)

# %% [markdown]
# **Q28.** 🟡 All three heads cost the same strokes. You can add one more afternoon (about 4 hours) of work.
# Suppose a pilot (invented numbers) says:
#
# - **A, more strokes:** a second round on the same five images would lift the scribble head's held-out IoU
#   by about 0.04.
# - **B, a better teacher:** a SAM checkpoint fine-tuned on retinas would lift the pseudo-labels' IoU on the
#   training images from 0.45 to 0.65, and nobody has measured the head yet.
#
# Which would you pick? What does the table suggest, and what would you want to measure first?
#
# <details><summary>A possible answer</summary>
#
# - **Compare like with like:** A's +0.04 is on the head itself, B's +0.20 is on the pseudo-labels only. A
#   better teacher helps the head only as far as its mistakes stop being systematic.
# - **The cap:** a head trained on pseudo-labels is capped by how well the second model understands *your*
#   structures. SAM thinks in objects and Grad-CAM in tiles, and neither thinks in thin lines, so better
#   pseudo-labels need a better teacher for this domain, not a cleverer threshold.
# - **More strokes:** they add true information, but no extra coverage between them.
# - **How to choose:** if the pseudo-labels are close to the truth on a few fully labelled images, a better
#   teacher (or a cleaned-up pseudo-label) is the lever. If they are far, spend the afternoon on strokes.
# - **First measurement:** the diagnosis line of 6.1 on a handful of images you labelled completely.
#
# </details>

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
