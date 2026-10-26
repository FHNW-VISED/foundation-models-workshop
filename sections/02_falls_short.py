# %% [markdown]
# ## 2. Where it falls short

# %% [markdown]
# <!-- callout: objective | Objective -->
# Run the same tools on the specialised task and probe SAM with escalating prompts.

# %% [markdown]
# Same three tools, same cells, new task. Section 1 gave you three numbers on pets: SAM with one click,
# SAM with one box, and a linear head on five full masks.
#
# Here they meet retina images (a *fundus* is a photo of the back of the eye), where the objects are thin,
# branching and low-contrast. Then we push SAM with more and more prompts, and give DINOv3 twice the
# resolution, to see whether the limit is the prompt or the patch size.

# %% [markdown]
# This helper cell defines what this section measures with: the evaluation region (the fundus, not the
# black border), skeletons for the thin-structure metric, clicks spread along a person's scribbles, and a
# figure that marks every click.

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
    """One row of panels (title, RGB image, clicks or None[, box]); stars = clicks, frame = box."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    for ax, (title, img, pts, *extra) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        if extra and extra[0] is not None:  # an optional box (x0, y0, x1, y1)
            bx0, by0, bx1, by1 = extra[0]
            ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=2,
                                       ec=THEME["scribble"]))
        for x, y in pts or []:
            ax.plot(x, y, marker="*", markersize=10, color=THEME["scribble"], markeredgecolor="white")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# This cell loads the held-out retina images (1024 px crops of a 3504 × 2336 fundus), the evaluation set for
# the rest of the notebook, and prints how much of the fundus the vessels cover; keep that number, it comes
# back as a baseline.

# %%
# @title Load the held-out retina images { display-mode: "form" }
SPECIALISED["feat_side"] = 256 if SMOKE else SPECIALISED["feature_long_side"]
SPECIALISED["eval"] = SPECIALISED["held"] if (SMOKE or DEVICE == "cuda") else SPECIALISED["held"][:12]
SPEC_EVAL = [load_item(SPECIALISED, it) for it in SPECIALISED["eval"]]
fg_share = float(np.mean([it["mask"][region(it)].mean() for it in SPEC_EVAL]))
print(
    f"evaluation set: {len(SPEC_EVAL)} held-out images · {SPECIALISED['task']} · the foreground covers "
    f"{fg_share:.0%} of the evaluation region · DINOv3 features at {SPECIALISED['feat_side']} px long side"
)

# %% [markdown]
# Before any theory, look at the images: this cell shows one retina photo, its vessels, and a zoom with the
# 16 px grid DINOv3 sees, and measures how wide a vessel is next to one patch.

# %%
# @title Meet the retina images: vessels next to the DINOv3 patch grid { display-mode: "form" }
peek = SPEC_EVAL[0]
ph, pw = peek["mask"].shape
patch_px = 16 * max(ph, pw) / SPECIALISED["feat_side"]  # one DINOv3 patch, in image pixels
centre_line = skeleton(peek["mask"] & region(peek))
vessel_px = float(peek["mask"].sum() / max(centre_line.sum(), 1))  # area ÷ centre-line length = mean width
win = int(min(8 * patch_px, ph, pw))  # the zoom window: about 8 × 8 patches
y0, x0 = (ph - win) // 2, (pw - win) // 2
zoom = (slice(y0, y0 + win), slice(x0, x0 + win))
truth_img = overlay(peek["image"], peek["mask"], THEME["truth"])

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
axes[0].imshow(peek["image"])
axes[0].set_title("fundus photo: bright round patch = optic disc,\nblack ring = outside the view", fontsize=8)
axes[1].imshow(truth_img)
axes[1].set_title("vessels (ground truth, green)", fontsize=9)
axes[2].imshow(truth_img[zoom])
grid = np.arange(0, win + 1, patch_px)
axes[2].vlines(grid, 0, win, colors="white", linewidths=0.8)
axes[2].hlines(grid, 0, win, colors="white", linewidths=0.8)
axes[2].set_xlim(0, win)
axes[2].set_ylim(win, 0)
axes[2].set_title(f"zoom: each white square is one {patch_px:.0f} × {patch_px:.0f} px patch", fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
print(
    f"{peek['name']}: vessels are about {vessel_px:.1f} px wide on average, one patch is {patch_px:.0f} px, "
    f"so a typical vessel fills about {vessel_px / patch_px:.0%} of a patch's width"
)

# %% [markdown]
# <!-- callout: theory | Theory: why thin structures are hard -->
# Each tool has its own trouble with the vessels you just saw.
#
# - **DINOv3 looks at 16 × 16 px patches.** A vessel a few pixels wide fills a fraction of a patch (see the
#   zoom), so the patch is mostly "background with a dark line in it". The head can only say "vessel" or
#   "not" for the whole square.
# - **SAM learnt from a billion masks of *objects*:** compact things with a clear inside and outside, like a
#   pet. A vessel tree has no inside, and one click is ambiguous: this vessel? the whole tree? the eye?

# %% [markdown]
# ### 2.1 SAM: one click, one box, again

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Before you run the next cell (the section-1 cell on a retina image: one click on a vessel, then a box
# around all of them), guess: what will SAM segment from one click on a thin line? And what is "the object"
# inside a box that covers the whole fundus?
#
# Then run it and compare the panels with the pets in section 1.
#
# **Q5.** 🟡 What did SAM segment from the click, and from the box: a vessel, part of the tree, a blob, the
# whole fundus? Use the printed shares: the true vessels cover a small share of the fundus, so a mask that
# marks far more is not tracing vessels. Try two other images by changing the index (on a CPU session there
# are 12 held-out images).
#
# ```python
# demo = SPEC_EVAL[0]   # try 1, 2, ...
# ```

# %%
# Q5 — SAM with one click and with one box on a held-out specialised image
demo = SPEC_EVAL[0]  # try 1, 2, ...
point, box = click_point(demo), box_prompt(demo)
t0 = time.monotonic()
sam_point = sam.probability(demo["image"], points=[point], labels=[1]) >= 0.5
sam_box = sam.probability(demo["image"], box=box) >= 0.5
print(f"{demo['name']}: two SAM calls in {time.monotonic() - t0:.1f} s on {DEVICE}")
shares = {"click mask": sam_point, "box mask": sam_box, "true vessels": demo["mask"]}
marked = " · ".join(f"{name} {m[region(demo)].mean():.0%}" for name, m in shares.items())
print(f"share of the fundus marked · {marked}")
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
# **Why IoU misleads on thin vessels.** IoU counts pixels, and thick vessels hold most of them.
#
# - Draw every vessel twice as wide as it is: nothing is missed, yet IoU drops to about 0.5.
# - Miss all the capillaries: they hold few pixels, so IoU barely moves.
#
# **clDice** compares *centre lines* instead: `tprec` (how much of what you drew is real vessel) and `tsens`
# (how much of the real vessel you drew), combined by harmonic mean.
#
# 1. `tprec` = prediction centre line inside the truth ÷ prediction centre line (invented branches lower it).
# 2. `tsens` = true centre line inside the prediction ÷ true centre line (missed branches lower it).
# 3. clDice = 2 · tprec · tsens ÷ (tprec + tsens). Example: tprec 1.0 and tsens 0.25 give 0.40.

# %% [markdown]
# <!-- callout: go-deeper | Where clDice comes from -->
# clDice is from Shit et al., "clDice: a novel topology-preserving loss function for tubular structure
# segmentation", CVPR 2021.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 3 min) -->
# Fill `my_cldice`: two boolean masks in, one float out.
#
# - `skeleton(mask)` gives the centre line.
# - `tprec` = pixels of `skeleton(pred)` inside `truth`, divided by the pixels of `skeleton(pred)`.
# - `tsens` = the same with the roles swapped.
# - Return 0.0 when either skeleton is empty.
#
# The cell checks your version on two play masks (a good trace and a blob that covers everything) and
# continues with ours if you skip it. Stuck? Open the collapsed "Reference solution" cell above it.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Reference: harmonic mean of topology precision and topology sensitivity."""
    sk_pred, sk_truth = skeleton(pred), skeleton(truth)
    if not sk_pred.any() or not sk_truth.any():
        return 0.0
    tprec = float((sk_pred & truth).sum() / sk_pred.sum())
    tsens = float((sk_truth & pred).sum() / sk_truth.sum())
    return 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0


# %%
# ✍️ YOUR CODE
def my_cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """clDice of two boolean masks: 2·tprec·tsens / (tprec + tsens); 0.0 when either skeleton is empty.
    e.g. 0.71"""
    ...


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
# If one click is ambiguous, maybe more clicks tell SAM what we mean. Each step tells SAM something new:
#
# 1. **1 point**: one spot. SAM must guess: this vessel, the tree, or the eye?
# 2. **5 points**: five spots along different vessels. It can look for one thing that contains all five.
# 3. **20 points**: dense along the tree. More information, but still only *where*, never *what kind*.
# 4. **box**: a box around the whole fundus says "everything in here".

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Write down three guesses before you run the ladder.
#
# - Will IoU rise steadily from 1 to 20 clicks, jump at some point, or stay flat?
# - Will the box beat 20 clicks?
# - Where will the grey line sit? It scores a mask that says "everything is foreground": if vessels cover 12%
#   of the fundus, overlap 12% ÷ union 100% gives IoU 0.12. The share printed above is yours.

# %% [markdown]
# This cell runs the ladder on every evaluation image (one SAM image encoding, four cheap prompt decodings)
# and prints mean IoU, clDice and the share of the fundus marked per setting, where a result near the
# "everything" line means the prompt taught SAM nothing about vessels. The last figure draws the four
# prompts (teal stars and box) on the demo image next to SAM's mask for each.

# %%
# @title Run the SAM prompt ladder on every evaluation image { display-mode: "form" }
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
            ladder_demo[name] = (pred, pts, box_prompt(it) if not n else None)
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
    [(f"SAM, {k}", overlay(demo["image"], m, THEME["fails"]), p, b) for k, (m, p, b) in ladder_demo.items()]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]), None)]
)

# %% [markdown]
# **Q6.** 🟡 Compare the ladder with your prediction. Read the table row by row: IoU, clDice, and *marked*
# (the share of the fundus SAM marked). How to read a row: IoU 0.30 with 70% marked, against a grey line at
# 0.29, is a big mask that is barely better than marking everything.
#
# - Which rows beat the grey line by more than 0.05 IoU?
# - Did more clicks bring the masks closer to the vessels, or just make them bigger?
# - What does that say about what SAM thinks an object is?
#
# <details><summary>What to look for</summary>
#
# - Read the IoU next to the grey line, not on its own. A mask that marks most of the fundus scores the
#   "everything" baseline, whatever the prompt.
# - If more clicks push "marked" up while IoU stays near the line, SAM answers "the object that contains all
#   these points". For a vessel tree that object is the eye.
# - If clicks do help, look where: usually along the thick vessels near the clicks, rarely the capillaries
#   in between.
# - A prompt says *where*, not *what kind of thing*. Section 7 comes back to prompting.
#
# </details>

# %% [markdown]
# <!-- callout: pitfall | Pitfall: over-trusting SAM's confident masks -->
# SAM returns a crisp mask for every prompt. Its own confidence score is high for a mask of the whole eye,
# because the whole eye *is* a good object. Nothing in the output says "this is the wrong kind of thing".
#
# - Score against a baseline that knows nothing (the grey line).
# - Always look at the mask.

# %% [markdown]
# ### 2.4 🔴 Optional: What DINOv3 sees here
#
# Same 3-colour view as section 1.3 (a PCA of the 384 numbers per patch, painted red, green and blue); this
# cell shows it on retina images, to see whether the features separate vessels from background before any
# training.

# %%
# @title DINOv3 PCA colours on three retina images { display-mode: "form" }
fig_items = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(SPECIALISED, it, SPECIALISED["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(panels, width=2.6)

# %% [markdown]
# **Q7.** 🔴 Do the thick vessels share a colour at all, across the three images? Which of the optic disc
# (bright round patch), the dark border and the vessels stands out most from its surroundings? Compare with
# the pets, where one direction separated animal from floor.
#
# <details><summary>What to look for</summary>
#
# - The colours are the directions of largest variation. On a fundus that is rarely the vessels: it is the
#   optic disc, the illumination falling off towards the edge, the border.
# - Thick vessels may show as a faint trace of blocks; thin ones vanish inside the 16 px squares.
# - That does not mean the features hold no vessel information (a trained head can find a direction for it).
#   It is just not the first thing the model "sees".
#
# </details>

# %% [markdown]
# ### 2.5 The linear head, and the resolution it looks at
#
# The head from section 1.4, trained the same way on the full masks of five retina images (the last time
# training sees a full mask: a perfect vessel map costs an expert 3 to 5 hours per fundus). We train it
# twice, on DINOv3 features at two resolutions:
#
# - **512 px long side**: each 16 px patch covers 32 × 32 pixels of the 1024 px crop.
# - **1024 px long side**: each patch covers 16 × 16 pixels. Four times the patches, about 16 times the
#   attention cost (see the go-deeper below the chart).

# %% [markdown]
# <!-- callout: go-deeper | Why 512 and not 518? -->
# DINOv2 papers use 518 px: 37 patches of 14 px. DINOv3's patches are 16 px, so 512 px (32 patches) is the
# same idea.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# With *perfect* labels, write two guesses.
#
# - Will the head beat SAM's one-box IoU from the 2.3 table?
# - Will doubling the resolution add more than 0.05 to IoU, to clDice, to both or to neither?

# %% [markdown]
# This cell trains one head per resolution on the five fit images and prints the seconds it took; if it is
# slow, set `TRAIN_YOURSELF = False` and run it again to load our checkpoints (trained the same way on all
# fit images).

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
# This cell scores both heads on the evaluation images with IoU and clDice and zooms into the centre of the
# demo image to compare the two resolutions on the thinnest vessels.

# %%
# @title Score both heads and zoom into the thinnest vessels { display-mode: "form" }
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
# This cell puts the section-1 numbers next to this section's, the same three tools on both tasks, so the gap
# is one picture.

# %%
# @title Pets vs the specialised task, side by side { display-mode: "form" }
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
# <!-- callout: go-deeper | Resolution is not free: the attention arithmetic -->
# Four times the patches costs about 16 times the attention, so when the structure is thinner than a patch,
# check the resolution before anything else.
#
# Self-attention compares every patch with every other, so the cost grows with the *square* of the patches.
#
# 1. 512 px side: 32 × 32 = 1 024 patches, about 1 million patch pairs.
# 2. 1024 px side: 64 × 64 = 4 096 patches, about 17 million pairs: 4× the patches, about 16× the attention
#    cost.
# 3. The whole 3504 × 2336 fundus at 16 px: about 219 × 146 = 32 000 patches, about 1 billion pairs. That is
#    why we work on crops and precompute the features.

# %% [markdown]
# ### 2.6 The failure gallery
#
# Numbers average the failures away; the gallery shows them. The next cell sorts the evaluation images from
# worst to best (by the head's IoU at the higher resolution) and shows three at a time: the image, SAM with a
# box, the head, and the ground truth.

# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell: it draws the failure gallery from the predictions you already have, with each image's
# IoU and clDice in the titles.
#
# **Q8.** 🟡 Read the IoU and clDice in each title. For the worst image, is the failure the same kind for SAM
# and for the head? Pick from: missed thin vessels, vessels drawn too wide, the optic disc or the border
# marked as vessel, whole regions marked. Page with the start index:
#
# ```python
# GALLERY_START = 0   # 0 = the three worst; try 3, 6, ...
# ```

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

# %% [markdown]
# Secondary visual: this cell shows the typical (median) and the best image of the same ranking, to see what
# an IoU of, say, 0.3 versus 0.7 looks like next to the worst ones.

# %%
# @title For contrast: the typical and the best image of the ranking { display-mode: "form" }
for label, k in (("typical (median)", order[len(order) // 2]), ("best", order[-1])):
    it, sam_s, head_s = SPEC_EVAL[k], ladder_scores["box"][k], gallery[k]
    show_masks(
        [
            (f"{label} · {it['name']}", it["image"]),
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

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min: whose image fails worst? -->
# Each of you picks a different `GALLERY_START` (change that value only, not `demo`) and finds the worst
# mask you can.
#
# - Show it to your neighbour and name the failure in three words ("misses thin vessels", "marks the disc",
#   "the whole eye").
# - Is it the same failure for SAM and for the head?

# %% [markdown]
# <details><summary>What people usually find</summary>
#
# - SAM tends to fail *globally*: it marks a whole region or one thick vessel, because it seeks an object.
# - The head fails *locally*: it finds the thick vessels, draws them as wide as a patch, and loses the
#   capillaries, because it can only label 16 px squares.
# - On the whole fundus both get worse, since every vessel is now thinner than a patch (optional 2.9 below).
#
# </details>
#
# ### 2.7 A checklist for a new problem

# %% [markdown]
# **Q9.** 🟡 Use what is on screen as a checklist: the "everything is foreground" line in the ladder, the
# 512 versus 1024 px chart, and IoU next to clDice. A colleague segments cracks in concrete and reports:
#
# - SAM with one box: IoU 0.31.
# - "Everything is foreground": IoU 0.28.
# - Linear head at 512 px: IoU 0.35; at 1024 px: IoU 0.48.
# - IoU 0.48 but clDice 0.22.
#
# Which two or three signs from this section tell you the foundation model is not good enough out of the box?
#
# <details><summary>One possible answer</summary>
#
# - SAM's 0.31 sits near the 0.28 baseline that knows nothing: it learnt little about cracks.
# - Going from 512 to 1024 px moves the head by 0.13: the cracks are thinner than a patch, so the
#   representation, not the labels, is the bottleneck.
# - IoU 0.48 next to clDice 0.22: a pixel metric and a structure metric disagree, so the mask misses the
#   branches.
# - And always: the mask looks crisp and confident, so look at it next to the truth.
#
# </details>

# %% [markdown]
# ### 2.8 🔴 Optional: what if the model paints only the thick trunk?
#
# Back to the two numbers of 2.2, now on a real retina image: three fake predictions next to the truth, the
# playful one being a model that paints only the thick trunk: violet marks the true centre line the
# prediction covers, green the part it misses, grey is the prediction.

# %%
# @title Playful test: what if the model paints only the thick trunk? { display-mode: "form" }
from scipy import ndimage as ndi

sim = SPEC_EVAL[0]
sim_truth = sim["mask"] & region(sim)
sim_line = skeleton(sim_truth)
sim_width = float(sim_truth.sum() / max(sim_line.sum(), 1))  # mean vessel width in px
sim_dist = ndi.distance_transform_edt(sim_truth)  # distance of each vessel pixel to the vessel edge
sim_cut = np.percentile(sim_dist[sim_truth], 60)  # keep the 40 % thickest vessel pixels
fakes = {
    "trunk only": ndi.binary_dilation(sim_dist >= sim_cut, iterations=max(int(sim_cut), 1)) & sim_truth,
    "2× too wide": ndi.binary_dilation(sim_truth, iterations=max(int(round(sim_width / 2)), 1)) & region(sim),
    '"everything"': region(sim),
}
fat = max(1, max(sim_truth.shape) // 400)  # draw the 1 px centre line a few px thick so it is visible


def trace_panel(pred: np.ndarray) -> np.ndarray:
    """The prediction in grey; the true centre line violet where covered, green where missed."""
    img = overlay(sim["image"], pred, THEME["background"])
    img = overlay(img, ndi.binary_dilation(sim_line & pred, iterations=fat), THEME["works"], alpha=1.0)
    return overlay(img, ndi.binary_dilation(sim_line & ~pred, iterations=fat), THEME["truth"], alpha=1.0)


panels = [
    (
        "truth and its centre line",
        overlay(
            overlay(sim["image"], sim_truth, THEME["truth"]),
            ndi.binary_dilation(sim_line, iterations=fat),
            "#FFFFFF",
            alpha=1.0,
        ),
    )
]
for name, pred in fakes.items():
    found = float((sim_line & pred).sum() / max(sim_line.sum(), 1))
    panels.append(
        (
            f"{name} · IoU {iou(pred, sim_truth):.2f} · clDice {cldice(pred, sim_truth):.2f}\n"
            f"true centre line covered: {found:.0%}",
            trace_panel(pred),
        )
    )
show_masks(panels, width=3.0)

# %% [markdown]
# ### 2.9 🔴 Optional: the whole fundus
#
# The gallery shows 1024 px crops. This cell runs SAM with a box and the head on the whole fundus of the demo
# image shrunk to 1024 px (vessels only 1–3 px wide), next to the truth.

# %%
# @title The failure gallery on the whole fundus { display-mode: "form" }
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
# **Got a photo of something thin or specialised?** Section 9 runs SAM, the ladder and the head on it.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **Objects thinner than a 16 px patch** → raise the resolution first, and check how far the numbers move.
# - **Score close to the "everything is foreground" line** → the model learnt nothing about the task; always
#   compare with a baseline that knows nothing, and look at the mask next to the truth.
# - **More clicks make SAM's mask bigger, not thinner** → stop prompting: a prompt says *where*, not *what
#   kind of thing*.
# - **IoU and clDice disagree** → trust the structure metric on thin objects, and look at the masks.
