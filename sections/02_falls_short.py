# %% [markdown]
# ## 2. Where it falls short

# %% [markdown]
# <!-- callout: objective | Objective -->
# Run the same tools on the specialised task and probe SAM with escalating prompts.

# %% [markdown]
# Same three tools, same cells, new task. Section 1 gave you three numbers on pets: SAM with one click,
# SAM with one box, and a linear head on five full masks. Here they meet the specialised images, where the
# objects are thin, branching and low-contrast. Then we push SAM with more and more prompts, and give
# DINOv3 twice the resolution, to see which of the two limits the result.
#

# %% [markdown]
# <!-- callout: theory | Theory: why thin structures are hard -->
# DINOv3 describes the image in 16 × 16 px patches. A vessel 5 px wide covers a third of a
# patch, so its description is mostly "background with a dark line in it", and the head can only say
# "vessel" or "not" for the whole square. SAM sees finer detail, but it was trained on a billion masks of
# *objects*: compact things with a clear inside and outside. A vessel tree has
# no inside, and one click on it is ambiguous (this vessel? the whole tree? the eye?).

# %% [markdown]
# This helper cell defines what this section measures with: the evaluation region (the fundus, not the
# black border around it), skeletons for the thin-structure metric, clicks spread along a person's
# scribbles, and a figure that marks every click.

# %%
# @title Thin-structure helpers: region, skeleton, clicks, figure { display-mode: "form" }
from skimage.morphology import skeletonize


def region(item: dict) -> np.ndarray:
    """Where we score: the field of view if the dataset has one, else the whole image."""
    return item["fov"] if item.get("fov") is not None else np.ones_like(item["mask"])


def skeleton(mask: np.ndarray) -> np.ndarray:
    """The one-pixel-wide centre line of a boolean mask."""
    return skeletonize(np.asarray(mask, bool))


def scribble_points(item: dict, n: int) -> list[list[float]]:
    """n (x, y) clicks along the foreground scribbles: the section-1 click first, then spread out as far as
    possible from each other (farthest-point sampling), the way a person would click along the vessels."""
    ys, xs = np.nonzero(item["scribbles"] == 1)
    pts = np.stack([xs, ys], 1).astype(float)
    chosen = [np.array(click_point(item))]
    dist = np.linalg.norm(pts - chosen[0], axis=1)
    while len(chosen) < min(n, len(pts)):
        k = int(dist.argmax())
        chosen.append(pts[k])
        dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return [[float(x), float(y)] for x, y in chosen]


def score(pred: np.ndarray, item: dict) -> dict:
    """IoU and clDice of a boolean prediction, both inside the evaluation region."""
    r = region(item)
    return {"iou": iou(pred & r, item["mask"] & r), "cldice": cldice(pred & r, item["mask"] & r)}


def mean_scores(scores: list[dict]) -> dict:
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}


def show_clicks(panels: list[tuple[str, np.ndarray, list | None]], width: float = 2.6):
    """One row of panels (title, RGB image, clicks or None); clicks are drawn as teal stars."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    for ax, (title, img, pts) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        for x, y in pts or []:
            ax.plot(x, y, marker="*", markersize=10, color=THEME["scribble"], markeredgecolor="white")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# This cell loads the held-out specialised images, the evaluation set for the rest of the notebook, and
# prints how much of the fundus the structures cover: keep that number, it comes back as a baseline.

# %%
SPECIALISED["feat_side"] = 256 if SMOKE else SPECIALISED["feature_long_side"]
SPECIALISED["eval"] = SPECIALISED["held"] if (SMOKE or DEVICE == "cuda") else SPECIALISED["held"][:12]
SPEC_EVAL = [load_item(SPECIALISED, it) for it in SPECIALISED["eval"]]
fg_share = float(np.mean([it["mask"][region(it)].mean() for it in SPEC_EVAL]))
print(
    f"evaluation set: {len(SPEC_EVAL)} held-out images · {SPECIALISED['task']} · the foreground covers "
    f"{fg_share:.0%} of the evaluation region · DINOv3 features at {SPECIALISED['feat_side']} px long side"
)

# %% [markdown]
# ### 2.1 SAM: one click, one box, again

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# The next cell is the section-1 cell on a specialised image: one click on a vessel (a pixel of a
# person's scribble), then a box around all of them. Before you run it: what will SAM segment from one
# click on a thin line? And what is "the object" inside a box that covers the whole image?

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell and compare the panels with the pets in section 1.

# %% [markdown]
# **Q5.** 🟡 What did SAM segment from the click, and from the box: a vessel, part of the tree, a blob, the
# whole fundus? Try two other images by changing the index (on a CPU session there are 12).
#
# ```python
# demo = SPEC_EVAL[0]   # try 1, 2, ...
# ```
#
# This cell runs SAM twice on one held-out specialised image, once with the click and once with the box,
# and shows both masks in orange next to the ground truth.

# %%
# Q5 — SAM with one click and with one box on a held-out specialised image
demo = SPEC_EVAL[0]  # try 1, 2, ...
point, box = click_point(demo), box_prompt(demo)
t0 = time.monotonic()
sam_point = sam.probability(demo["image"], points=[point], labels=[1]) >= 0.5
sam_box = sam.probability(demo["image"], box=box) >= 0.5
print(f"{demo['name']}: two SAM calls in {time.monotonic() - t0:.1f} s on {DEVICE}")
show_masks(
    [
        ("SAM, one click", overlay(demo["image"], sam_point, THEME["fails"])),
        ("SAM, one box", overlay(demo["image"], sam_box, THEME["fails"])),
        ("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"])),
    ],
    prompts={"point": point, "box": box},
)

# %% [markdown]
# ### 2.2 A second number for thin structures: clDice

# %% [markdown]
# <!-- callout: theory | Theory: clDice -->
# IoU counts
# pixels, and on thin structures pixels mislead. Predict every vessel twice as wide as it is and IoU drops
# to about 0.5, although nothing was missed; miss all the capillaries and IoU barely moves, because they
# hold few pixels. **clDice** (Shit et al., CVPR 2021) compares *centre lines* instead. It takes
# the skeleton of each mask and asks two questions: which share of the predicted centre line lies inside
# the true vessels (*topology precision*), and which share of the true centre line the prediction
# covers (*topology sensitivity*). clDice is their harmonic mean: 1.0 when the tree is traced
# exactly, low when branches are missing or invented. We report it next to IoU for the specialised task.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 3 min) -->
# Fill `my_cldice`: two boolean masks in, one float out. `skeleton(mask)` gives the
# centre line. Topology precision = pixels of `skeleton(pred)` inside `truth`, divided
# by the pixels of `skeleton(pred)`; topology sensitivity is the same with the roles swapped.
# Return 0.0 when either skeleton is empty. The cell checks your version on two play masks, a good trace
# and a blob that covers everything, and continues with ours if you skip it.

# %%
# ✍️ YOUR CODE
def my_cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """clDice of two boolean masks: 2·tprec·tsens / (tprec + tsens); 0.0 when either skeleton is empty."""
    ...


def _ref_cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Reference: harmonic mean of topology precision and topology sensitivity."""
    sk_pred, sk_truth = skeleton(pred), skeleton(truth)
    if not sk_pred.any() or not sk_truth.any():
        return 0.0
    tprec = float((sk_pred & truth).sum() / sk_pred.sum())
    tsens = float((sk_truth & pred).sum() / sk_truth.sum())
    return 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0


_play_truth = np.zeros((12, 12), bool)
_play_truth[2:4, :] = True  # a horizontal vessel, 2 px wide
_play_truth[:, 9:11] = True  # crossed by a vertical one
_play_trace = np.zeros((12, 12), bool)
_play_trace[1:5, :] = True  # the horizontal vessel, drawn too wide; the vertical one missed
_play_blob = np.ones((12, 12), bool)  # "everything is vessel"
cldice = use(
    my_cldice,
    reference=_ref_cldice,
    check=lambda f: [round(float(f(p, _play_truth)), 4) for p in (_play_trace, _play_blob)],
    name="cldice",
)
print(
    f"play example · IoU / clDice of the wide trace: {iou(_play_trace, _play_truth):.2f} / "
    f"{cldice(_play_trace, _play_truth):.2f} · of the blob: {iou(_play_blob, _play_truth):.2f} / "
    f"{cldice(_play_blob, _play_truth):.2f}"
)

# %% [markdown]
# ### 2.3 Escalating prompts: 1 → 5 → 20 clicks, and a box
#
# If one click is ambiguous, maybe more clicks tell SAM what we mean. The next cell gives it 1, 5 and 20
# clicks spread along the scribbles a person drew, then the box, on every evaluation image. It also scores
# the laziest possible answer, a mask that says "everything is foreground": a prediction no better than
# that line has learnt nothing about the task.

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# Write down
# what you expect from 1 to 20 clicks: will IoU rise steadily, jump at some point, or stay flat? Will the
# box beat 20 clicks? And where will the grey "everything" line sit?

# %% [markdown]
# This cell runs the prompt ladder on every evaluation image (one SAM image encoding per image, then four
# cheap prompt decodings) and keeps the mean IoU, clDice and the share of the fundus SAM marks per setting.

# %%
LADDER = {"1 point": 1, "5 points": 5, "20 points": 20, "box": None}  # setting -> number of clicks
t0 = time.monotonic()
ladder_scores = {name: [] for name in LADDER}
ladder_cover = {name: [] for name in LADDER}
ladder_demo = {}  # the demo image's masks and clicks, for the figure below
sam_box_masks = []  # every image's box mask, for the gallery in 2.6
everything = []
for it in SPEC_EVAL:
    r = region(it)
    for name, n in LADDER.items():
        pts = scribble_points(it, n) if n else None
        if n:
            pred = sam.probability(it["image"], points=pts, labels=[1] * len(pts)) >= 0.5
        else:
            pred = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
        ladder_scores[name].append(score(pred, it))
        ladder_cover[name].append(float(pred[r].mean()))
        if it is demo:
            ladder_demo[name] = (pred, pts)
    sam_box_masks.append(pred)  # the last setting is the box
    everything.append(score(r.copy(), it))
ladder_mean = {name: mean_scores(s) for name, s in ladder_scores.items()}
everything_mean = mean_scores(everything)
spec_sec = round(time.monotonic() - t0, 1)

RESULTS["sam_prompt_ladder"] = {name: v["iou"] for name, v in ladder_mean.items()}
RESULTS["sam_prompt_ladder_cldice"] = {name: v["cldice"] for name, v in ladder_mean.items()}
RESULTS["specialised_zero_shot_iou"] = ladder_mean["box"]["iou"]  # same headline as pets: one box
RESULTS["specialised_zero_shot_point_iou"] = ladder_mean["1 point"]["iou"]
RESULTS["specialised_zero_shot_box_iou"] = ladder_mean["box"]["iou"]
RESULTS["specialised_everything_iou"] = everything_mean["iou"]
RESULTS["specialised_zero_shot_seconds"] = spec_sec

print(f"SAM prompt ladder on {len(SPEC_EVAL)} held-out images · {spec_sec} s on {DEVICE}")
print(f"{'setting':<12}{'IoU':>7}{'clDice':>8}{'marked':>8}")
for name, v in ladder_mean.items():
    print(f"{name:<12}{v['iou']:>7.3f}{v['cldice']:>8.3f}{np.mean(ladder_cover[name]):>8.0%}")
print(f"{'everything':<12}{everything_mean['iou']:>7.3f}{everything_mean['cldice']:>8.3f}{1:>8.0%}")

fig, ax = plt.subplots(figsize=(6.4, 3))
names = list(ladder_mean)
ax.bar(names, [ladder_mean[k]["iou"] for k in names], color=THEME["fails"], label="specialised, SAM")
ax.axhline(everything_mean["iou"], color=THEME["background"], ls="--", label='"everything is foreground"')
ax.axhline(RESULTS["pets_zero_shot_iou"], color=THEME["works"], ls=":", label="pets, SAM one box (section 1)")
ax.set_ylabel("mean IoU")
ax.set_ylim(0, 1)
ax.set_title("More prompts, same model")
ax.legend(fontsize=8, frameon=False, loc="upper left")
plt.tight_layout()
plt.show()

show_clicks(
    [(f"SAM, {k}", overlay(demo["image"], m, THEME["fails"]), p) for k, (m, p) in ladder_demo.items()]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]), None)]
)

# %% [markdown]
# **Q6.** 🟡 Compare the ladder with your prediction. Did more clicks help, and did the masks get closer to
# the vessels or just bigger (look at the "marked" column and the grey line)? What does that tell you about
# what SAM thinks an object is?
#
# <details><summary>What to look for</summary>
#
# Read the IoU next to the grey line, not on its own. A mask that marks most of the fundus scores exactly
# the "everything" baseline, whatever the prompt. If more clicks push the "marked" share up while IoU stays
# near the line, SAM is answering "the object that contains all these points", and for a vessel tree that
# object is the eye. If clicks do help on your images, look at where: usually along the thick vessels near
# the clicks, rarely the capillaries in between. SAM has no notion of "all the thin dark lines"; a prompt
# can only say <i>where</i>, not <i>what kind of thing</i>. Section 7 comes back to prompting with
# background clicks too.
#
# </details>

# %% [markdown]
# <!-- callout: pitfall | Pitfall: over-trusting SAM's confident masks -->
# SAM returns a crisp mask for every prompt, and its own confidence score is high for
# a mask of the whole eye, because the whole eye *is* a good object. Nothing in the output says "this
# is the wrong kind of thing". Always score against a baseline that knows nothing (the grey line), and
# look at the mask.

# %% [markdown]
# ### 2.4 What DINOv3 sees here
#
# This cell shows the same 3-colour PCA of the DINOv3 patch features as section 1.3, now on specialised
# images, so you can see whether the features separate the vessels from the background before any training.

# %%
fig_items = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(SPECIALISED, it, SPECIALISED["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(panels, width=2.6)

# %% [markdown]
# **Q7.** 🟡 Do the PCA colours follow the vessels, or something else (the bright optic disc, the dark
# border, the overall brightness)? Compare with the pets, where one direction separated animal from floor.
#
# <details><summary>What to look for</summary>
#
# The three main directions of variation describe what varies most across the image, and on a fundus that
# is rarely the vessels: it is the optic disc, the illumination falling off towards the edge, the border.
# Thick vessels may show as a faint trace of blocks; thin ones vanish inside the 16 px squares. That does
# not mean the features hold no vessel information (a trained head can still find a direction for it), but
# it is no longer the first thing the model "sees".
#
# </details>
#
# ### 2.5 The linear head, and the resolution it looks at
#
# The head from section 1.4, trained the same way on the full masks of five specialised images. This is
# the last time training sees a full mask: from section 3 on, only incomplete labels. We train it twice,
# on DINOv3 features computed at two resolutions. At a 512 px long side, each 16 px patch covers 32 × 32
# pixels of the 1024 px crop; at 1024 px, 16 × 16. (DINOv2 papers use 518 px: 37 patches of 14 px. DINOv3's
# patches are 16 px, so 512 is the same idea.) Double the side and you get four times the patches, and
# roughly four to sixteen times the compute.

# %% [markdown]
# <!-- callout: your-turn | Your prediction -->
# With
# *perfect* labels, will the head beat SAM here? And will doubling the resolution help IoU, clDice,
# both or neither? Write two guesses.

# %% [markdown]
# This cell trains one head per resolution on the five fit images and prints the seconds it took. Not done
# in a minute, or on a CPU and in a hurry? Set `TRAIN_YOURSELF = False` and run it again: it loads our
# checkpoints (trained the same way on all fit images).

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "full" heads instead of training
N_TRAIN_SPEC, HEAD_STEPS = 5, 30 if SMOKE else 200
SIDES = SPECIALISED["feature_long_sides"]  # [512, 1024]: the two resolutions we compare
SIDES = [s // 4 for s in SIDES] if SMOKE else SIDES  # smoke thumbnails are 256 px
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

spec_train = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:N_TRAIN_SPEC]]
spec_heads = {}  # long side -> (head tensors, training seconds)
for side in SIDES:
    if TRAIN_YOURSELF:
        feats = [features_for(SPECIALISED, it, side) for it in spec_train]
        labels = [np.where(region(it), it["mask"], -1).astype(np.int8) for it in spec_train]  # border ignored
        spec_heads[side] = train_head(feats, labels, HEAD_STEPS)
        src = f"trained here on {len(spec_train)} images, {HEAD_STEPS} steps"
    else:
        tensors, entry = load_head(SPECIALISED, "full", side)
        spec_heads[side] = (tensors, entry["train_sec"])
        src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
    print(f"linear head @ {side} px · {src} · {spec_heads[side][1]:.1f} s on {DEVICE}")
if sum(sec for _, sec in spec_heads.values()) > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoints next time.")

# %% [markdown]
# This cell scores both heads on the evaluation images with IoU and clDice, and zooms into the centre of the
# demo image so you can compare the two resolutions on the thinnest vessels.

# %%
t0 = time.monotonic()
head_preds, res_scores = {}, {}
for side, (tensors, _) in spec_heads.items():
    head_preds[side] = [
        predict_head(tensors, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
    ]
    res_scores[side] = mean_scores([score(p, it) for p, it in zip(head_preds[side], SPEC_EVAL, strict=True)])
lo, hi = min(SIDES), max(SIDES)
RESULTS["specialised_head_iou"] = res_scores[hi]["iou"]
RESULTS["specialised_head"] = {
    "iou": res_scores[hi]["iou"],
    "cldice": res_scores[hi]["cldice"],
    "long_side": hi,
    "n_train": len(spec_train),
    "train_sec": round(float(spec_heads[hi][1]), 2),
    "infer_sec": round(time.monotonic() - t0, 1),
}
RESULTS["resolution_effect"] = {f"{side} px": v for side, v in res_scores.items()}
for side, v in res_scores.items():
    print(f"linear head @ {side:>4} px · mean IoU {v['iou']:.3f} · clDice {v['cldice']:.3f}")
print(f"SAM one box · mean IoU {ladder_mean['box']['iou']:.3f} · clDice {ladder_mean['box']['cldice']:.3f}")

k = next(i for i, it in enumerate(SPEC_EVAL) if it is demo)
h, w = demo["mask"].shape
zoom = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))  # the central quarter
show_masks(
    [
        (
            f"head @ {s} px · IoU {score(head_preds[s][k], demo)['iou']:.2f}",
            overlay(demo["image"], head_preds[s][k], THEME["fails"])[zoom],
        )
        for s in SIDES
    ]
    +[("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"])[zoom])],
    width=3.0,
)

# %% [markdown]
# <!-- callout: pitfall | Pitfall: resolution is not free -->
# Self-attention compares every patch with every other, so doubling the side multiplies the
# patches by four and the attention cost by up to sixteen. At 1024 px, one specialised image is 4096 patches.
# On the full 3504 px fundus it would be 48 000: that is why we work on crops and why the features are
# precomputed. When the structure is thinner than a patch, check the resolution before anything else.

# %% [markdown]
# ### 2.6 The failure gallery
#
# Numbers average the failures away; the gallery shows them. The next cell sorts the evaluation images
# from worst to best (by the head's IoU at the higher resolution) and shows three at a time: the image, SAM
# with a box, the head, and the ground truth. The last row puts the same models on the whole fundus, shrunk
# to 1024 px, where the vessels are only 1–3 px wide.

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell, then page through the gallery.

# %% [markdown]
# **Q8.** 🟡 For the worst images, is the failure the same kind for SAM and for the head? Look for: missed
# thin vessels, vessels drawn too wide, the optic disc or the border marked as vessel, whole regions marked.
# Page with the start index:
#
# ```python
# GALLERY_START = 0   # 0 = the three worst; try 3, 6, ...
# ```
#
# This cell draws the failure gallery from the predictions you already have (new model calls only for the
# last row) and prints each image's IoU and clDice in the titles.

# %%
# Q8 — the failure gallery, worst images first
GALLERY_START = 0  # 0 = the three worst; try 3, 6, ...
if not isinstance(GALLERY_START, int) or not 0 <= GALLERY_START < len(SPEC_EVAL):
    raise SystemExit(f"GALLERY_START must be a whole number from 0 to {len(SPEC_EVAL) - 1}.")
gallery = [score(p, it) for p, it in zip(head_preds[hi], SPEC_EVAL, strict=True)]
order = np.argsort([s["iou"] for s in gallery])
for k in order[GALLERY_START : GALLERY_START + 3]:
    it, sam_s, head_s = SPEC_EVAL[k], ladder_scores["box"][k], gallery[k]
    show_masks(
        [
            (it["name"], it["image"]),
            (
                f"SAM box · IoU {sam_s['iou']:.2f} · clDice {sam_s['cldice']:.2f}",
                overlay(it["image"], sam_box_masks[k], THEME["fails"]),
            ),
            (
                f"head @ {hi} px · IoU {head_s['iou']:.2f} · clDice {head_s['cldice']:.2f}",
                overlay(it["image"], head_preds[hi][k], THEME["fails"]),
            ),
            ("ground truth", overlay(it["image"], it["mask"], THEME["truth"])),
        ],
        width=2.8,
    )

# the whole fundus of the demo image, shrunk so its long side is 1024 px (smoke: 256 px)
meta = next(m for m in SPECIALISED["eval"] if m["name"] == demo["name"])
root = SPECIALISED["dir"]
full = {
    "image": np.array(Image.open(root / meta["full"]["image"]).convert("RGB")),
    "mask": read_png(root / meta["full"]["mask"]) > 127,
    "fov": read_png(root / meta["full"]["fov"]) > 127,
}
full_sam = sam.probability(full["image"], box=box_prompt(full)) >= 0.5
full_feat = dino.features(full["image"], max(full["image"].shape[:2]))
full_head = predict_head(spec_heads[hi][0], full_feat, full["mask"].shape)
full_masks = {"SAM box": full_sam, f"head @ {hi} px": full_head}
full_s = {name: score(m, full) for name, m in full_masks.items()}
show_masks(
    [(f"whole fundus · {meta['name']}", full["image"])]
    + [
        (f"{name} · IoU {full_s[name]['iou']:.2f} · clDice {full_s[name]['cldice']:.2f}",
         overlay(full["image"], m, THEME["fails"]))
        for name, m in full_masks.items()
    ]
    + [("ground truth", overlay(full["image"], full["mask"], THEME["truth"]))],
    width=2.8,
)
RESULTS["specialised_full_view"] = full_s

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min: whose image fails worst? -->
# Each of you picks a different `GALLERY_START` or `demo` index and
# finds the worst mask you can. Show it to your neighbour and name the failure in three words ("misses thin
# vessels", "marks the disc", "the whole eye"). Is it the same failure for SAM and for the head?

# %% [markdown]
# <details><summary>What people usually find</summary>
#
# The two tools fail differently. SAM tends to fail <i>globally</i>: it marks a whole region, or one thick
# vessel, because it looks for an object. The head fails <i>locally</i>: it finds the thick vessels, draws
# them as wide as a patch, and loses the capillaries, because it can only label 16 px squares. On the
# whole fundus both get worse, since every vessel is now thinner than a patch.
#
# </details>
#
# ### 2.7 Pets vs the specialised task
#
# This cell puts the section-1 numbers next to this section's, the same three tools on both tasks, so the
# gap is one picture.

# %%
rows = {
    "SAM, one click": (RESULTS["pets_zero_shot_point_iou"], RESULTS["specialised_zero_shot_point_iou"]),
    "SAM, one box": (RESULTS["pets_zero_shot_box_iou"], RESULTS["specialised_zero_shot_box_iou"]),
    "linear head": (RESULTS["pets_head_iou"], RESULTS["specialised_head_iou"]),
}
print(f"{'tool':<16}{'pets IoU':>10}{'specialised IoU':>17}")
for name, (p, s) in rows.items():
    print(f"{name:<16}{p:>10.3f}{s:>17.3f}")
print(f"{'everything':<16}{'':>10}{RESULTS['specialised_everything_iou']:>17.3f}")

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.bar(x - 0.2, [v[0] for v in rows.values()], 0.4, color=THEME["works"], label=PETS["name"])
ax.bar(x + 0.2, [v[1] for v in rows.values()], 0.4, color=THEME["fails"], label=SPECIALISED["name"])
ax.axhline(RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--",
           label='specialised, "everything is foreground"')
ax.set_xticks(x, list(rows))
ax.set_ylabel("mean IoU on held-out images")
ax.set_ylim(0, 1)
ax.set_title("Same tools, two tasks")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: across | Across approaches -->
# Two
# different limits showed up. SAM's limit is its idea of an object: more prompts make the mask bigger, not
# thinner. The head's limit is the patch: even with perfect labels it can only answer per 16 px square, and
# resolution is one lever (look at how far it moved IoU and clDice, and whether they agreed). From here on
# the question changes from "does it work?" to "what information can we add, cheaply, to make it work?".
# Perfect masks cost 3–5 hours per fundus, so the next sections only use cheap, incomplete ones.

# %% [markdown]
# **Q9.** 🟡 You now have a small checklist for a new problem. Which two or three signs from this section
# would tell you, on your own data, that a foundation model is not good enough out of the box?
#
# <details><summary>One possible answer</summary>
#
# (1) Its score sits near a baseline that knows nothing ("everything is foreground"), so compare with one.
# (2) More or better prompts change the <i>size</i> of the mask but not its <i>shape</i>. (3) The objects
# are smaller than a feature patch, and a higher resolution moves the numbers: a sign the representation,
# not the labels, is the bottleneck. (4) A pixel metric and a structure metric (IoU vs clDice) disagree.
# And always: the mask looks crisp and confident, so look at it next to the truth.
#
# </details>

# %% [markdown]
# <!-- callout: your-turn | Try it on your image (optional, 2 min) -->
# Got a photo of something thin or specialised (a leaf's veins, a crack, a cable, a
# micrograph)? Upload it with the folder icon on the left, put its path in `MY_IMAGE` and one or
# more `(x, y)` clicks on the thing in `MY_CLICKS`, then run the cell. Left empty, it
# runs on the demo image. Section 9 does this properly, with scribbles and the whole ladder.

# %% [markdown]
# This cell runs SAM on your image (or the demo image) with your clicks and shows the mask, the quickest way
# to see whether your own problem looks more like the pets or like this section.

# %%
MY_IMAGE = ""  # e.g. "/content/my_photo.jpg"; empty = the demo image
MY_CLICKS = []  # e.g. [(120, 80), (300, 210)]; empty = the image centre
if not isinstance(MY_IMAGE, str) or not isinstance(MY_CLICKS, list):
    raise SystemExit('MY_IMAGE must be a path in quotes (or "") and MY_CLICKS a list of (x, y) pairs.')
if MY_IMAGE and not Path(MY_IMAGE).is_file():
    raise SystemExit(f"{MY_IMAGE} not found. Upload it with the folder icon on the left and copy its path.")
my_img = np.array(Image.open(MY_IMAGE).convert("RGB")) if MY_IMAGE else demo["image"]
my_h, my_w = my_img.shape[:2]
my_clicks = [[float(x), float(y)] for x, y in MY_CLICKS] or [[my_w / 2, my_h / 2]]
if not all(0 <= x < my_w and 0 <= y < my_h for x, y in my_clicks):
    raise SystemExit(f"every click must lie inside the image: 0 ≤ x < {my_w}, 0 ≤ y < {my_h}.")
my_mask = sam.probability(my_img, points=my_clicks, labels=[1] * len(my_clicks)) >= 0.5
show_clicks(
    [
        (MY_IMAGE or f"demo · {demo['name']}", my_img, my_clicks),
        (
            f"SAM, {len(my_clicks)} click(s) · marks {my_mask.mean():.0%}",
            overlay(my_img, my_mask, THEME["fails"]),
            my_clicks,
        ),
    ],
    width=3.2,
)
