# %% [markdown]
# ## 6. Path 2 · Recover supervision
#
# **Objective:** turn SAM pseudo-labels and Grad-CAM maps into training signal.
#
# Path 1 kept the label file and changed the loss. Path 2 does the opposite: it keeps the loss and fills in
# the labels. A second model looks at every pixel nobody painted and makes a guess, and the head trains on
# those guesses as if a person had drawn them. That is a **pseudo-label**. Think of handing your half-done
# colouring-in to a fast, confident colleague: the page comes back full, and every mistake they made with
# confidence is now in your training set. We start from the scribbles of section 4 (a few strokes per
# image, minutes of work), so this path costs no more annotation than rung 2 of the ladder. Two colleagues
# get the job: **SAM**, prompted with clicks taken from the strokes, and **Grad-CAM**, the heat map of a
# small classifier that only learnt whether a tile contains a vessel.
#
# <table width="100%" bgcolor="#EDE7F6"><tr><td><font color="#311B92">📘 <b>Theory: pseudo-labels.</b>
# Pseudo-labelling (Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method", ICML
# 2013 workshop) trains a model on another model's predictions for unlabelled data. It works when the
# guesses are more often right than wrong <i>and</i> their mistakes are not systematic: a model trained on
# its teacher's errors learns to repeat them, which the literature calls <i>confirmation bias</i>. The usual
# guard is a confidence threshold: keep only guesses the teacher is sure of and leave the rest unlabelled
# (FixMatch, Sohn et al., NeurIPS 2020, <a href="https://arxiv.org/abs/2001.07685">arXiv:2001.07685</a>).
# Here the teacher is a different model, not the head itself, and the person's strokes always overrule it.
# </font></td></tr></table>
#
# This helper cell adds a colour for each head of this path and defines how to spread SAM clicks along the
# strokes of one kind, vessel or background.

# %%
# @title Path 2 helpers: colours, clicks along the strokes { display-mode: "form" }
APPROACH_COLOURS |= {
    "sam_pseudo": "#0277BD",
    "gradcam_pseudo": "#827717",
}
APPROACH_NAMES |= {
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
# The scribbles of section 4 again: a few vessel strokes and a few background strokes on each of the same
# five training images. We turn them into SAM prompts, twenty clicks spread along the vessel strokes as
# "vessel" and twenty along the background strokes as "not vessel", and keep SAM's answer for every pixel.
# Only the strokes go in (L-005). The ground truth appears in the figure for you to judge the guesses,
# never in training.
#
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your prediction.</b> Section 2
# showed what SAM does with clicks on a vessel tree. With forty clicks per image, half of them saying "not
# here", will its mask follow the vessels, or cover a region around them? Write one guess.
# </font></td></tr></table>
#
# This cell asks SAM for a probability map on each training image from the stroke clicks (one SAM image
# encoding per image: seconds on a GPU, up to half a minute per image on a CPU) and shows one image. Short
# on time? Set `RUN_SAM = False` to load our maps, made with the same recipe and clicks drawn at random.

# %%
RUN_SAM = True  # False -> load our precomputed SAM maps for the training images
N_PSEUDO_CLICKS = 20  # clicks per kind (vessel / background) and image, taken from the strokes
if not isinstance(RUN_SAM, bool):
    raise SystemExit("RUN_SAM must be True or False.")

t0 = time.monotonic()
pseudo_probs, pseudo_clicks = [], []  # per training image: SAM's probability map, (vessel, background) clicks
if RUN_SAM:
    for it in scrib_train:
        pos, neg = stroke_points(it, 1, N_PSEUDO_CLICKS), stroke_points(it, 2, N_PSEUDO_CLICKS)
        prob = sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
        pseudo_probs.append(prob)
        pseudo_clicks.append((pos, neg))
    src = f"SAM 2.1 run here, {N_PSEUDO_CLICKS} + {N_PSEUDO_CLICKS} clicks per image"
else:
    ours = PRECOMPUTED / "pseudo_labels" / "specialised"
    if not (ours / "prompts.json").is_file():
        raise SystemExit(f"{ours} is missing. Run setup_check.ipynb first, or set RUN_SAM = True.")
    prompts = json.loads((ours / "prompts.json").read_text())["prompts"]
    for it in scrib_train:
        pseudo_probs.append(read_png(ours / f"{it['name']}.png") / 255)
        p = prompts[it["name"]]
        clicks = [
            [xy for xy, lab in zip(p["points_xy"], p["labels"], strict=True) if lab == k] for k in (1, 0)
        ]
        pseudo_clicks.append(tuple(clicks))
    src = "our precomputed maps (same recipe, clicks drawn at random from the strokes)"
pseudo_sec = round(time.monotonic() - t0, 1)

# diagnosis only: SAM's guesses at the usual 0.5 cut, scored against the hidden truth (never for training)
sam_quality = mean_scores([score(p >= 0.5, it) for p, it in zip(pseudo_probs, scrib_train, strict=True)])
sam_marked = float(
    np.mean([(p >= 0.5)[region(it)].mean() for p, it in zip(pseudo_probs, scrib_train, strict=True)])
)
# how sure SAM is, inside the fundus (no truth needed): the spread of its probabilities
spread = np.quantile(
    np.concatenate([p[region(it)] for p, it in zip(pseudo_probs, scrib_train, strict=True)]),
    [0.05, 0.5, 0.95],
)
print(
    f"SAM pseudo-labels on {len(scrib_train)} training images · {src} · {pseudo_sec} s on {DEVICE}\n"
    f"SAM's probability inside the fundus · 5 % of the pixels below {spread[0]:.2f} · median "
    f"{spread[1]:.2f} · 95 % below {spread[2]:.2f}\n"
    f"diagnosis (truth used to score only): at 0.5 they mark {sam_marked:.0%} of the fundus · IoU "
    f"{sam_quality['iou']:.3f} · clDice {sam_quality['cldice']:.3f}"
)
ex, (pos, neg) = scrib_train[0], pseudo_clicks[0]
show_clicks(
    [
        (f"{ex['name']} · vessel clicks", ex["image"], pos),
        ("background clicks", ex["image"], neg),
        ("SAM probability: purple 1, white 0.5, orange 0", prob_rgb(pseudo_probs[0]), None),
        ("pseudo-label at 0.5 vs truth", error_map(ex["image"], pseudo_probs[0] >= 0.5, ex), None),
    ],
    width=2.8,
)

# %% [markdown]
# In the last panel, violet = SAM's guess is a real vessel, orange = SAM invented a vessel, green = a vessel
# SAM left out. Every orange and green pixel becomes a wrong training label if we take the guesses as they
# are.
#
# **Q22.** 🟡 Where does SAM's guess go wrong: around the thick vessels, along the thin ones, in the
# background between the strokes? Now look at the probability panel and the printed spread: is SAM sure
# of itself (deep purple, deep orange) or unsure (pale)? What does that mean for a cut at exactly 0.5?
#
# <details><summary>What to look for</summary>
#
# SAM looks for <i>objects</i> that contain the positive clicks and avoid the negative ones. Forty clicks
# scattered over a vessel tree do not describe one object, so SAM settles for the object that contains
# most of them, often the whole fundus, and draws its edge with care while the inside stays a region
# (orange), not a tree. Thin branches between the clicks can drop out (green). The probability panel tells
# you how much SAM believes it. If most of it is pale, the probabilities sit just around 0.5, and the
# exact cut decides almost everything: a little below and the whole fundus is "vessel", a little above
# and nothing is. The binary mask hides that doubt: it looks just as crisp as a confident one. Whatever you
# see on your run, the threshold is a choice you make, not a property of SAM.
#
# </details>
#
# ### 6.2 From a probability map to training targets
#
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your turn (optional, 3 min).</b>
# Fill <code>my_pseudo_targets</code>. In: SAM's <code>prob</code> map (floats in [0, 1]), the
# <code>scribbles</code> map (0 unlabelled, 1 vessel stroke, 2 background stroke), the boolean
# <code>region</code>, a <code>threshold</code> and an unsure <code>band</code>. Out: int8 targets, 1 vessel,
# 0 background, −1 ignored. A pixel with <code>prob ≥ threshold + band</code> is vessel, one with
# <code>prob &lt; threshold − band</code> is background, and the doubtful ones in between are ignored. The
# person's strokes always win over SAM, and pixels outside <code>region</code> are always −1. The cell
# checks your version on a 3 × 4 play map, without and with a band, and continues with ours if you skip it.
# </font></td></tr></table>


# %%
# ✍️ YOUR CODE
def my_pseudo_targets(
    prob: np.ndarray, scribbles: np.ndarray, region: np.ndarray, threshold: float = 0.5, band: float = 0.0
) -> np.ndarray:
    """Targets (int8) from a pseudo-label probability map: 1 where prob >= threshold + band, 0 where
    prob < threshold - band, -1 in between. Strokes overrule the map (1 vessel stroke -> 1, 2 background
    stroke -> 0); outside `region` -> -1."""
    ...


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
pseudo_targets = use(
    my_pseudo_targets,
    reference=_ref_pseudo_targets,
    check=lambda f: [
        np.asarray(f(_play_prob, _play_scrib, _play_region, 0.5, b)).astype(int) for b in (0.0, 0.2)
    ],
    name="pseudo_targets",
)
print("play example · targets with threshold 0.5 and band 0.2 (−1 = ignored):")
print(pseudo_targets(_play_prob, _play_scrib, _play_region, 0.5, 0.2))

# %% [markdown]
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your prediction.</b> The
# scribble head of section 4 heard about the strokes only. This head hears about every pixel of the fundus,
# most of it in SAM's words. Will it mark more of the held-out fundus than the scribble head, or less? Will
# its IoU be higher? Write two guesses.</font></td></tr></table>
#
# This cell turns SAM's maps into targets at the usual cut (threshold 0.5, no band), trains the section-2
# head on them, and prints the seconds it took. Not done in a minute, or on a CPU and in a hurry? Set
# `TRAIN_YOURSELF = False` and run it again: it loads our "pseudo" checkpoint, trained the same way on all
# fit images.

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "pseudo" head instead of training
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

# the supervision of this path: the strokes, and SAM's guesses from clicks on the strokes (L-005)
sam_targets = [
    pseudo_targets(p, it["scribbles"], region(it), 0.5, 0.0)
    for p, it in zip(pseudo_probs, scrib_train, strict=True)
]
if TRAIN_YOURSELF:
    sam_head, sam_sec = train_head(scrib_feats, sam_targets, HEAD_STEPS)
    src = f"trained here on {len(scrib_train)} images, {HEAD_STEPS} steps, SAM pseudo-labels at 0.5"
else:
    sam_head, entry = load_head(SPECIALISED, "pseudo", side)
    sam_sec = entry["train_sec"]
    src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
print(f"SAM pseudo-label head @ {side} px · {src} · {sam_sec:.1f} s on {DEVICE}")
if sam_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the new head on the held-out images with the numbers of section 4 and prints it under
# the scribble head, which saw the same strokes but kept every other pixel unknown.

# %%
t0 = time.monotonic()
path2_preds = {
    "scribbles": ladder_preds["scribbles"],
    "sam_pseudo": [
        predict_head(sam_head, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
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
    "trained_here": TRAIN_YOURSELF,
    "train_sec": round(float(sam_sec), 2),
    "infer_sec": sam_infer,
}


def print_path2(reports: dict) -> None:
    """One line per head with the numbers of section 4."""
    print(f"{'approach':<46}{COLUMNS}")
    for k, v in reports.items():
        name = APPROACH_NAMES.get(k, k)
        print(
            f"{name:<46}{v['iou']:>6.3f}{v['cldice']:>8.3f}{v['recall']:>8.3f}{v['precision']:>8.3f}"
            f"{v['marked']:>8.0%}"
        )


print_path2(path2_report)

# %% [markdown]
# <table width="100%" bgcolor="#F3E5F5"><tr><td><font color="#4A148C">❓ <b>Question.</b> Run the next cell,
# then move the cut.</font></td></tr></table>
#
# **Q23.** 🟡 Tune the pseudo-labels. Raise or lower `PSEUDO_THRESHOLD` and give the doubtful pixels an
# `UNSURE_BAND` to ignore; pick your values from the spread printed in 6.1. How many training pixels does
# each setting get wrong (the diagnosis lines), and does the head on the held-out images move the same way?
# Is there a setting that beats 0.5 on your run?
#
# ```python
# PSEUDO_THRESHOLD = 0.5   # try 0.55, 0.6, 0.7 (or the median printed in 6.1)
# UNSURE_BAND = 0.0        # try 0.03: pixels within 0.03 of the threshold are ignored
# ```
#
# This cell turns SAM's maps into targets with your setting, measures on the training images how many of
# them disagree with the hidden truth (diagnosis only), trains the same head on them (seconds) and prints it
# next to the head at 0.5.

# %%
# Q23 — tune the pseudo-label threshold and the unsure band
PSEUDO_THRESHOLD = 0.5  # try 0.55, 0.6, 0.7 (or the median printed in 6.1)
UNSURE_BAND = 0.0  # try 0.03: pixels within 0.03 of the threshold are ignored
ok = all(isinstance(v, (int, float)) for v in (PSEUDO_THRESHOLD, UNSURE_BAND))
if not ok or UNSURE_BAND < 0 or PSEUDO_THRESHOLD - UNSURE_BAND <= 0 or PSEUDO_THRESHOLD + UNSURE_BAND > 1:
    raise SystemExit(
        "PSEUDO_THRESHOLD must lie between 0 and 1, and UNSURE_BAND ≥ 0 must keep both edges inside."
    )


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


tuned_targets = [
    pseudo_targets(p, it["scribbles"], region(it), PSEUDO_THRESHOLD, UNSURE_BAND)
    for p, it in zip(pseudo_probs, scrib_train, strict=True)
]
tuned_head, tuned_sec = train_head(scrib_feats, tuned_targets, HEAD_STEPS)
tuned_preds = [
    predict_head(tuned_head, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
]
tuned_report = rung_report(tuned_preds)
RESULTS["sam_pseudo_tuned"] = tuned_report | {"threshold": PSEUDO_THRESHOLD, "band": UNSURE_BAND}
print(f"{'targets on the training images':<46}{'ignored':>8}{'vessel → bg':>13}{'bg → vessel':>13}")
for name, t in {"threshold 0.5, no band": sam_targets, "your setting": tuned_targets}.items():
    e = target_errors(t)
    print(f"{name:<46}{e['ignored']:>8.0%}{e['vessel_as_bg']:>13.0%}{e['bg_as_vessel']:>13.0%}")
print(f"\nyour head: threshold {PSEUDO_THRESHOLD}, band {UNSURE_BAND} · trained in {tuned_sec:.1f} s")
print_path2({"threshold 0.5, no band": path2_report["sam_pseudo"], "your setting": tuned_report})

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The threshold trades one kind of wrong label for the other. A low cut turns more of SAM's region into
# "vessel" (bg → vessel rises), and the head learns to mark more; a high cut keeps only SAM's surest
# pixels and calls the rest background (vessel → bg rises): the section-3 mistake again, now with SAM as
# the annotator who stopped early. When SAM's probabilities crowd around one value, a step of 0.05 can
# flip most of the fundus from one side to the other, so there may be no comfortable middle. The band
# removes the doubtful pixels instead of guessing, so both error columns can fall at once, paid for with
# fewer labelled pixels (the "ignored" column). Whether the held-out numbers follow the training diagnosis
# depends on your run. In a real project you would pick the setting on a small fully labelled check set,
# never on the test images.
#
# </details>
#
# <table width="100%" bgcolor="#FDECEA"><tr><td><font color="#8B1A10">⚠️ <b>Pitfall: over-trusting SAM's
# confident masks.</b> A pseudo-label looks like a real label: crisp edges, full coverage, no "I don't
# know". The head cannot tell the difference, and it will learn SAM's habits (here, a region where you
# wanted a line) as faithfully as the vessels. Before training on pseudo-labels, score them on a few images
# you labelled completely, as the diagnosis line does here.</font></td></tr></table>
#
# ### 6.3 Grad-CAM: where a classifier looks
#
# The second colleague does not segment at all. We train a small classifier to answer one yes/no question,
# then ask it <i>where</i> in the image its "yes" came from. Grad-CAM turns that answer into a heat map,
# and a threshold turns the heat map into a pseudo-mask.
#
# <table width="100%" bgcolor="#FDECEA"><tr><td><font color="#8B1A10">⚠️ <b>Pitfall: an awkward fit, said
# plainly.</b> Grad-CAM needs an image-level question with a "no" class, such as "is there a defect in this
# image?". Every retina image contains vessels, so there is no "no" at image level. We fall back to a
# patch-level question, "does this 64 px tile contain a vessel?", with tile labels read off the strokes.
# Most tiles of a fundus crop do contain a vessel (the next cell counts them), and a heat map at tile or
# patch resolution cannot outline a line a few pixels wide. Our bake-off judged it a poor fit; we run it
# anyway because on a task with real image-level labels (defects, tumours, "contains a cell") it is often
# the cheapest supervision there is.</font></td></tr></table>
#
# <table width="100%" bgcolor="#EDE7F6"><tr><td><font color="#311B92">📘 <b>Theory: Grad-CAM.</b> Take the
# classifier's last feature maps <i>A<sub>k</sub></i> and the gradient of the class score with respect to
# them. Average each gradient map into one weight α<sub>k</sub>, sum α<sub>k</sub>·<i>A<sub>k</sub></i>, keep
# the positive part, and rescale to [0, 1]: that is the map (Selvaraju et al., ICCV 2017,
# <a href="https://arxiv.org/abs/1610.02391">arXiv:1610.02391</a>; we use the <code>grad-cam</code> package).
# Our classifier ends in an average over patches, and for such a classifier Grad-CAM reduces to the older
# CAM (Zhou et al., CVPR 2016, <a href="https://arxiv.org/abs/1512.04150">arXiv:1512.04150</a>): the
# classifier's own evidence for "vessel", patch by patch. The map is as coarse as the features it is
# computed on: one value per 16 px patch.</font></td></tr></table>
#
# This helper cell defines the tile classifier (a 3 × 3 convolution and a 1 × 1 convolution on the frozen
# DINOv3 features, averaged over the tile), cuts feature grids into tiles, reads tile labels off the strokes,
# trains the classifier and computes its Grad-CAM map on a whole image.

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
# This cell cuts the features of the five training images into tiles of 64 px of the crop, labels each tile
# from the strokes (a vessel stroke crosses it → "vessel"; only background strokes → "no vessel"; no stroke
# → unused), trains the classifier (seconds) and prints how many tiles of each kind it saw. The last line
# uses the held-out truth to show how rare a vessel-free tile really is.

# %%
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
for it in SPEC_EVAL:
    grid = tuple(features_for(SPECIALISED, it, side).shape[1:])
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
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your prediction.</b> The
# classifier only ever answered "vessel in this tile, yes or no". Will its heat map look like a vessel
# tree, like blobs, or like noise? Write one guess.</font></td></tr></table>
#
# This cell computes the Grad-CAM map of every training image, cuts it at half its peak into a pseudo-mask
# (with your `pseudo_targets`: strokes still win), trains the same head on it (seconds; there is no
# checkpoint for this one) and scores it on the held-out images. The figure shows one training image.

# %%
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
    predict_head(cam_head, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
]
cam_infer = round(time.monotonic() - t0, 1)
path2_report["gradcam_pseudo"] = rung_report(path2_preds["gradcam_pseudo"])
RESULTS["gradcam_iou"] = path2_report["gradcam_pseudo"]["iou"]  # held-out IoU of the head (SPEC §9)
RESULTS["gradcam_note"] = (
    f"Patch-level proxy '{SPECIALISED['gradcam']['patch_level_task']}' (tiles labelled from the strokes), "
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
# **Q24.** 🟡 Compare the two pseudo-labels of the same training image. What shape does the Grad-CAM
# evidence have, and where does it sit? Which of the two would you rather correct by hand, and why?
#
# <details><summary>What to look for</summary>
#
# The heat map is a set of blobs one or a few patches wide, sitting where the classifier found the most
# "vessel-like" patches: usually the thick, dark vessels, and sometimes dark background. It does not
# follow a line, because its resolution is the 16 px patch and because the question it learnt ("is there a
# vessel somewhere in this tile?") does not need a line. Cut at half its peak, it turns into fat regions
# around the vessels the classifier found most convincing: the thick ones, drawn several patches wide,
# with the thin ones mostly missing. SAM's guess fails differently, as one big region with a careful edge.
# Neither is the vessel map. A coarse but <i>local</i> guess can still teach the head more than a sharp
# but global one, so compare the held-out lines above before you pick. On a task where the class really
# is a blob (a defect, a nucleus) and some images contain none, the same recipe fits much better.
#
# </details>

# %% [markdown]
# <table width="100%" bgcolor="#F3E5F5"><tr><td><font color="#4A148C">❓ <b>Question (optional).</b> Run the
# next cell to use the heat map as a SAM prompt instead of a mask.</font></td></tr></table>
#
# **Q25.** 🔴 Instead of cutting the heat map, let it pick SAM clicks: the hottest patches as "vessel", the
# coldest as "not vessel" (the trick of section 4.2, with Grad-CAM in place of the prototype). Do SAM's
# masks from these clicks look more like the SAM pseudo-labels of 6.1 or like the Grad-CAM blobs?
#
# ```python
# N_CAM_CLICKS = 5   # try 2, 10
# ```
#
# This cell picks the clicks on each training image's heat map, asks SAM for a mask (one SAM image encoding
# per image) and scores the three kinds of pseudo-label on the training images: diagnosis only, no training.

# %%
# Q25 — Grad-CAM as a SAM prompt: hottest patches = vessel clicks, coldest = background clicks
N_CAM_CLICKS = 5  # try 2, 10
if not isinstance(N_CAM_CLICKS, int) or N_CAM_CLICKS < 1:
    raise SystemExit("N_CAM_CLICKS must be a whole number of at least 1.")
cam_sam_masks, cam_clicks = [], []
for it, f in zip(scrib_train, std_feats, strict=True):
    s = torch.from_numpy(cam_map(clf, f)).to(DEVICE)
    keep = patch_share(region(it), tuple(s.shape))
    pos = top_points(s, keep, N_CAM_CLICKS, it["mask"].shape, largest=True)
    neg = top_points(s, keep, N_CAM_CLICKS, it["mask"].shape, largest=False)
    cam_sam_masks.append(
        sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg)) >= 0.5
    )
    cam_clicks.append(pos)
cam_sam_quality = mean_scores([score(m, it) for m, it in zip(cam_sam_masks, scrib_train, strict=True)])
RESULTS["gradcam_sam_prompt_iou_fit"] = cam_sam_quality["iou"]
print("pseudo-labels on the training images (diagnosis only: scored against the hidden truth)")
for name, q in {
    "SAM from stroke clicks (6.1, at 0.5)": sam_quality,
    f"Grad-CAM cut at {CAM_THRESHOLD}": cam_quality,
    f"SAM from {N_CAM_CLICKS} + {N_CAM_CLICKS} Grad-CAM clicks": cam_sam_quality,
}.items():
    print(f"{name:<44} IoU {q['iou']:.3f} · clDice {q['cldice']:.3f}")
show_clicks(
    [
        ("Grad-CAM clicks: vessel", ex["image"], cam_clicks[0]),
        ("SAM from Grad-CAM clicks vs truth", error_map(ex["image"], cam_sam_masks[0], ex), cam_clicks[0]),
    ],
    width=3.4,
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The clicks land where the heat map is hottest, usually on or next to a thick vessel, so SAM's mask looks
# like SAM's masks of 6.1: sharp edges around whatever object contains the clicks, often a region rather
# than a line. The prompt route swaps Grad-CAM's coarse blobs for SAM's sharp edges, but it also brings
# back SAM's idea of an object, and a click that lands on dark background drags the mask there. Two weak
# signals combined are not automatically a strong one; their mistakes can add up as easily as cancel out.
#
# </details>
#
# ### 6.4 Path 2, side by side
#
# This cell puts the three heads that start from the same strokes next to each other: the scribble head of
# section 4 (unlabelled = unknown), the head on SAM's guesses, the head on Grad-CAM's guesses. Then a bar
# chart in the approach colours and the error maps on the held-out demo image (violet found, orange
# invented, green missed). It also writes the annotation effort for the comparison of section 8: the same
# strokes for all three.

# %%
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
    ax.bar(x + shift, [v[m] for m in metrics], width, color=APPROACH_COLOURS[k], label=APPROACH_NAMES[k])
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
            f"{APPROACH_NAMES[k].split(':')[-1].strip()} · IoU {score(p[k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], p[k_demo], demo),
        )
        for k, p in path2_preds.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.8,
)

# %% [markdown]
# **Q26.** 🟡 All three heads cost the same strokes. Which one would you train if you could add one more
# afternoon of work: more strokes, or a better second model (say, a SAM checkpoint fine-tuned on retinas)?
# What does the table suggest, and what would you want to measure first?
#
# <details><summary>A possible answer</summary>
#
# Pseudo-labels pass on the second model's habits, so a head trained on them is capped by how well that
# model understands <i>your</i> structures. Here SAM thinks in objects and Grad-CAM in tiles, and neither
# thinks in thin lines, so better pseudo-labels would need a better teacher for this domain, not a cleverer
# threshold. More strokes add true information but no extra coverage between them. Which pays more depends
# on the gap you see: if the pseudo-labels are close to the truth on a few fully labelled images, a better
# teacher (or a cleaned-up pseudo-label) is the lever; if they are far, spend the afternoon on strokes. In
# both cases the first measurement is the diagnosis line of 6.1 on a handful of images you labelled
# completely.
#
# </details>
#
# <table width="100%" bgcolor="#E8EAF6"><tr><td><font color="#1A237E">🔀 <b>Across approaches.</b> Path 1
# changed what the loss is told about unlabelled pixels; Path 2 asks another model to label them. The
# scribble head leaves the gaps unknown; the SAM head fills them with SAM's idea of an object; the
# Grad-CAM head fills them with a classifier's coarse evidence. Each adds information <i>and</i> the second
# model's bias, and the head cannot tell which is which. Section 7 drops the head altogether and asks SAM
# directly, with the same clicks.</font></td></tr></table>
