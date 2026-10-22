# %% [markdown]
# ## 1. Out of the box: pets

# %% [markdown]
# <!-- callout: objective | Objective -->
# See a foundation model segment pets almost perfectly, zero-shot (with no training on this task).

# %% [markdown]
# Here SAM, DINOv3 and a linear head (explained below the first figure) meet the easy task; keep your
# numbers, because section 2 runs the same three cells on the specialised task.

# %% [markdown]
# This helper cell defines how to turn a person's hint into a SAM prompt, and `show_masks()`, the figure used
# for every comparison in the notebook (prediction in violet, ground truth in green, prompts in teal).

# %%
# @title Prompts and figure helpers { display-mode: "form" }
from safetensors import safe_open
from safetensors.torch import load_file

_FEATURE_FILES: dict = {}


def click_point(item: dict) -> list[float]:
    """One (x, y) where a person would click: a pixel of the first foreground scribble stroke."""
    ys, xs = np.nonzero(item["scribbles"] == 1)
    rng = np.random.default_rng([SEED, len(ys)])
    k = int(rng.integers(len(ys)))
    return [float(xs[k]), float(ys[k])]


def box_prompt(item: dict) -> list[float]:
    """The box a person would draw around the object: the full mask's bounding box, as x0, y0, x1, y1."""
    ys, xs = np.nonzero(item["mask"])
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def show_masks(panels: list[tuple[str, np.ndarray]], prompts: dict | None = None, width: float = 3.2):
    """One row of RGB panels (title, image). `prompts` = {"point": [x, y], "box": [x0, y0, x1, y1]}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    axes = np.atleast_1d(axes)
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    if prompts:
        for ax in axes:
            if prompts.get("point") is not None:
                x, y = prompts["point"]
                ax.plot(x, y, marker="*", markersize=14, color=THEME["scribble"], markeredgecolor="white")
            if prompts.get("box") is not None:
                x0, y0, x1, y1 = prompts["box"]
                rect = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2)
                rect.set_edgecolor(THEME["scribble"])
                ax.add_patch(rect)
    plt.tight_layout()
    plt.show()


def features_for(dataset: dict, item: dict, long_side: int) -> torch.Tensor:
    """DINOv3 patch features [384, h, w]: from the precomputed file when it exists, else computed now."""
    prefix = "pets" if dataset is PETS else "specialised"
    path = PRECOMPUTED / "features" / f"{prefix}_{long_side}.safetensors"
    if path.is_file():
        if path not in _FEATURE_FILES:
            _FEATURE_FILES[path] = safe_open(path, framework="pt")
        store = _FEATURE_FILES[path]
        if item["name"] in store.keys():
            return store.get_tensor(item["name"]).to(DEVICE).float()
    return dino.features(item["image"], long_side)


def pca_rgb(feat: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Project the 384-dim patch features onto their 3 main directions and show them as red, green, blue."""
    d, h, w = feat.shape
    x = feat.reshape(d, h * w).T.cpu().numpy().astype(np.float64)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rgb = x @ vt[:3].T
    rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-9)
    img = torch.from_numpy(rgb.reshape(h, w, 3)).permute(2, 0, 1)[None].float()
    img = F.interpolate(img, size=shape, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return (img.numpy() * 255).astype(np.uint8)


# %% [markdown]
# This cell loads the held-out pets, the evaluation set every approach in this notebook is scored on (12 of
# them on a CPU, to keep cells short), and fixes the resolution DINOv3 works at.

# %%
# @title Load the held-out pets { display-mode: "form" }
PETS["feat_side"] = 256 if SMOKE else PETS["index"]["feature_long_side"]
PETS["eval"] = PETS["held"] if (SMOKE or DEVICE == "cuda") else PETS["held"][:12]
PETS_EVAL = [load_item(PETS, it) for it in PETS["eval"]]
print(f"evaluation set: {len(PETS_EVAL)} held-out pets · DINOv3 features at {PETS['feat_side']} px long side")

# %% [markdown]
# ### 1.1 SAM: one click, one box
#

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The next cell gives SAM a single click on a pet, then a box around it. Before you run it: will one
# click be enough for a clean mask of the whole animal, and what could go wrong (fur vs floor, the tail,
# a second animal)? Say it to your neighbour or write it down.
#
# Then run the next cell and look at the three panels.
#
# **Q2.** 🟡 Compare the one-click mask and the box mask with the ground truth.
#
# - Where do they differ, if at all?
# - Is the difference at the object's boundary, or is a whole part missing or added?
# - Try another pet: change the index in `demo = PETS_EVAL[0]` (try 1, 2, ...).

# %% [markdown]
# This cell runs SAM twice on one held-out pet, once with a single click and once with a box, and shows
# both masks next to the ground truth, so you can see what a promptable model gives for free.

# %%
# Q2 — SAM with one click and with one box on a held-out pet
demo = PETS_EVAL[0]  # try 1, 2, ...
point, box = click_point(demo), box_prompt(demo)
t0 = time.monotonic()
sam_point = sam.probability(demo["image"], points=[point], labels=[1]) >= 0.5
sam_box = sam.probability(demo["image"], box=box) >= 0.5
print(f"{demo['name']}: two SAM calls in {time.monotonic() - t0:.1f} s on {DEVICE}")
show_masks(
    [
        ("SAM, one click", overlay(demo["image"], sam_point, THEME["works"])),
        ("SAM, one box", overlay(demo["image"], sam_box, THEME["works"])),
        ("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"])),
    ],
    prompts={"point": point, "box": box},
)

# %% [markdown]
# <!-- callout: theory | Theory: the three tools -->
# - **SAM** (Segment Anything) is *promptable*. You give it an image and a hint (a click, a box on a pet);
#   it returns the mask of the object under the hint. Nothing is trained: it learnt "what an object is" from
#   a billion masks of everyday photos.
# - **DINOv3** is a *feature extractor*. It cuts the image into 16 × 16 px patches and returns 384 numbers
#   per patch, learnt without labels. Patches with similar numbers look alike to the model (two cat ears,
#   say).
# - A **linear head** is the tiny classifier that turns those features into a mask: the features segment
#   nothing on their own, and the head learns a new class from a handful of masks in seconds (section 1.4).

# %% [markdown]
# ### 1.2 One number: IoU
#
# Looking at masks is the honest way to judge a segmentation; one number lets us compare approaches on a
# hundred images. The whole workshop uses one metric, **IoU** (intersection over union):
#
# `IoU = |pred ∩ truth| / |pred ∪ truth|`
#
# - Example: the predicted mask and the true mask each cover 100 pixels and share 80. The union is 120
#   pixels, so IoU = 80 / 120 = 0.67.
# - 1.0 is a perfect match, 0.0 means no overlap. Section 2 adds one extra metric for thin structures.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min) -->
# Fill `my_iou`: two boolean arrays in, one float out.
#
# - Intersection: pixels where both are True.
# - Union: pixels where at least one is True.
# - Empty union: return 1.0 (two empty masks agree).
#
# The cell below checks your version on this 4 × 4 play example (1 = True) and continues with ours if you
# skip it:
#
# ```
# pred   truth
# 0000   0110
# 0111   0110
# 0111   0110
# 0000   0000
# ```
#
# Intersection 4, union 8, so the answer is 0.5.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Reference: |pred ∧ truth| / |pred ∨ truth|."""
    inter, union = float((pred & truth).sum()), float((pred | truth).sum())
    return inter / union if union else 1.0


_play_pred = np.zeros((4, 4), bool)
_play_pred[1:3, 1:4] = True  # 6 pixels
_play_truth = np.zeros((4, 4), bool)
_play_truth[0:3, 1:3] = True  # 6 pixels, 4 shared -> IoU = 4 / 8

# %%
# ✍️ YOUR CODE
def my_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two boolean masks of the same shape; 1.0 when both are empty.
    e.g. 0.62"""
    ...


iou = use(my_iou, reference=_ref_iou, check=lambda f: round(float(f(_play_pred, _play_truth)), 4), name="iou")

# %% [markdown]
# This cell scores SAM's one-click and one-box masks on every evaluation pet with the IoU you just
# defined and keeps the mean as this task's zero-shot number, for section 2 to compare.

# %%
# @title Score SAM on every held-out pet { display-mode: "form" }
t0 = time.monotonic()
zero_shot_all = {"point": [], "box": []}
for it in PETS_EVAL:
    pt = click_point(it)
    by_point = sam.probability(it["image"], points=[pt], labels=[1]) >= 0.5
    by_box = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
    zero_shot_all["point"].append(iou(by_point, it["mask"]))
    zero_shot_all["box"].append(iou(by_box, it["mask"]))
zero_shot = {k: float(np.mean(v)) for k, v in zero_shot_all.items()}
RESULTS["pets_zero_shot_iou"] = zero_shot["box"]  # the headline: SAM's best single prompt, one box
RESULTS["pets_zero_shot_point_iou"] = zero_shot["point"]
RESULTS["pets_zero_shot_box_iou"] = zero_shot["box"]
RESULTS["pets_zero_shot_seconds"] = round(time.monotonic() - t0, 1)
print(
    f"SAM zero-shot on {len(PETS_EVAL)} held-out pets · mean IoU one click {zero_shot['point']:.3f} · "
    f"one box {zero_shot['box']:.3f} · {RESULTS['pets_zero_shot_seconds']} s"
)

# %% [markdown]
# A mean hides the spread, so this cell shows the **worst, median and best** held-out pet by one-box IoU:
# violet where mask and truth agree, orange for a false alarm (extra mask), green for what SAM missed.

# %%
# @title Worst, median and best pet by IoU { display-mode: "form" }
order = np.argsort(zero_shot_all["box"])
panels = []
for label, pos in (("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])):
    it = PETS_EVAL[int(pos)]
    pred = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
    img = overlay(it["image"], pred & it["mask"], THEME["works"])
    img = overlay(img, pred & ~it["mask"], THEME["fails"])
    img = overlay(img, ~pred & it["mask"], THEME["truth"])
    panels.append((f"{label} · IoU {zero_shot_all['box'][int(pos)]:.2f}", img))
show_masks(panels)

# %% [markdown]
# ### 1.3 What DINOv3 sees
#
# This cell keeps the **3 main directions (a PCA)** of each patch's 384 numbers and shows them as red, green
# and blue, so the same colour means the model sees those patches as similar (both cat ears, say).

# %%
# @title Show the patch features as colours { display-mode: "form" }
fig_items = [load_item(PETS, it) for it in PETS["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(PETS, it, PETS["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(panels, width=2.6)

# %% [markdown]
# **Q3.** 🟡 Look at the PCA images above.
#
# - Which pixels share a colour: the whole animal, parts of it (head, fur, legs), or the floor too?
# - Does the colour follow the object's outline?
# - How fine is the grid behind it?
#
# <details><summary>What to look for</summary>
#
# The colours are not classes: they are the 3 directions along which the features vary most in this image.
# Typically the animal separates from the background in one direction and its parts (face vs body) in
# another. The blocky edges show the 16 px grid: the features know <i>what</i> is where, at patch resolution,
# not <i>exactly where</i> the boundary is. Whether that is enough depends on the size of what you want to
# segment.
#
# </details>

# %% [markdown]
# ### 1.4 A linear head on five pets

# %% [markdown]
# **How the head works.** Each patch gets one score from its 384 feature numbers `f`:
#
# `score = w · f + b` (384 weights + 1 bias = 385 parameters). Score above 0: the patch is "pet"; below 0:
# "background".
#
# The "1 × 1 convolution" is this same formula applied to every patch of the grid.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The next cells train a classifier with 385 parameters on the patch features of just **five** pets with
# full masks, then score it on the held-out ones. Will its IoU land above, near or below SAM's one-click
# number from 1.2? Write down a number before you run.

# %% [markdown]
# This helper cell defines the head used for every training step in the notebook (that 1 × 1 convolution on
# standardised patch features, with a balanced loss, where each class gets half the total weight, that
# ignores pixels labelled −1) plus `load_head()` for our precomputed checkpoints.

# %%
# @title Linear head: train, predict, load a checkpoint { display-mode: "form" }
def train_head(
    feats: list[torch.Tensor], labels: list[np.ndarray], steps: int, lr: float = 0.05, balanced: bool = True
) -> tuple[dict, float]:
    """Linear head (384 -> 1) on standardised patch features. Labels per pixel: 1 fg, 0 bg, -1 ignored.

    Returns the head's tensors (weight, bias, feat_mean, feat_std) and the training seconds.
    """
    torch.manual_seed(SEED)
    allf = torch.cat([f.flatten(1) for f in feats], 1)
    mu, sd = allf.mean(1), allf.std(1) + 1e-6
    feats = [(f - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1) for f in feats]
    labs = [torch.from_numpy(np.asarray(lab, dtype=np.int8)).to(DEVICE) for lab in labels]
    n_pos = sum(int((lab == 1).sum()) for lab in labs)
    n_neg = sum(int((lab == 0).sum()) for lab in labs)
    if balanced:
        w_pos, w_neg = 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)
    else:
        w_pos = w_neg = 1.0 / max(n_pos + n_neg, 1)
    head = torch.nn.Conv2d(feats[0].shape[0], 1, 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.monotonic()
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for f, lab in zip(feats, labs, strict=True):
            logit = F.interpolate(head(f[None]), size=lab.shape, mode="bilinear", align_corners=False)[0, 0]
            w = torch.where(lab == 1, w_pos, torch.where(lab == 0, w_neg, 0.0))
            bce = F.binary_cross_entropy_with_logits(logit, (lab == 1).float(), reduction="none")
            loss = loss + (bce * w).sum()
        loss.backward()
        opt.step()
    tensors = {
        "weight": head.weight.detach().cpu(),
        "bias": head.bias.detach().cpu(),
        "feat_mean": mu.cpu(),
        "feat_std": sd.cpu(),
    }
    return tensors, time.monotonic() - t0


@torch.no_grad()
def predict_head(tensors: dict, feat: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Apply a head to one feature grid; returns the boolean foreground mask at `shape` (logit > 0)."""
    dev = feat.device
    f = (feat - tensors["feat_mean"].to(dev).view(-1, 1, 1)) / tensors["feat_std"].to(dev).view(-1, 1, 1)
    logit = F.conv2d(f[None], tensors["weight"].to(dev), tensors["bias"].to(dev))
    logit = F.interpolate(logit, size=shape, mode="bilinear", align_corners=False)[0, 0]
    return (logit > 0).cpu().numpy()


def load_head(dataset: dict, regime: str, long_side: int | None = None) -> tuple[dict, dict]:
    """Our precomputed checkpoint for (dataset, regime): its tensors and its index entry."""
    long_side = long_side or dataset["feat_side"]
    key = "pets" if dataset is PETS else SPECIALISED["key"]
    wanted = (key, regime, long_side)
    heads = PRECOMPUTED_HEADS
    entry = next((h for h in heads if (h["dataset"], h["regime"], h["long_side"]) == wanted), None)
    if entry is None:
        raise SystemExit(f"no precomputed head for {wanted}. Run setup_check.ipynb first.")
    return load_file(PRECOMPUTED / entry["file"]), entry


# %% [markdown]
# This cell trains the head on the full masks of five fit pets (only sections 1 and 2 use full masks); set
# `TRAIN_YOURSELF = False` and run it again to load our checkpoint instead.

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed head "full5" instead of training
N_TRAIN_PETS, HEAD_STEPS = 5, 30 if SMOKE else 200
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

train_items = [load_item(PETS, it) for it in PETS["fit"][:N_TRAIN_PETS]]
train_feats = [features_for(PETS, it, PETS["feat_side"]) for it in train_items]
if TRAIN_YOURSELF:
    train_labels = [it["mask"].astype(np.int8) for it in train_items]  # full masks: allowed in section 1 only
    head_full5, head_sec = train_head(train_feats, train_labels, HEAD_STEPS)
    head_src = f"trained here on {len(train_items)} pets, {HEAD_STEPS} steps"
else:
    head_full5, entry = load_head(PETS, "full5")
    head_sec = entry["train_sec"]
    head_src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} pets)"
print(f"linear head · {head_src} · {head_sec:.1f} s on {DEVICE}")
if head_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the head on the same evaluation pets as SAM, keeps the mean IoU, and shows three held-out
# pets with the head's mask next to the ground truth.

# %%
# @title Score the head on the held-out pets { display-mode: "form" }
t0 = time.monotonic()
head_preds = [
    predict_head(head_full5, features_for(PETS, it, PETS["feat_side"]), it["mask"].shape) for it in PETS_EVAL
]
head_ious = [iou(pred, it["mask"]) for pred, it in zip(head_preds, PETS_EVAL, strict=True)]
RESULTS["pets_head_iou"] = float(np.mean(head_ious))
RESULTS["pets_head"] = {
    "iou": RESULTS["pets_head_iou"],
    "n_train": len(train_items),
    "train_sec": round(float(head_sec), 2),
    "infer_sec": round(time.monotonic() - t0, 1),
}
print(
    f"linear head on {len(PETS_EVAL)} held-out pets · mean IoU {RESULTS['pets_head_iou']:.3f} "
    f"(SAM one click: {RESULTS['pets_zero_shot_point_iou']:.3f}, "
    f"one box: {RESULTS['pets_zero_shot_box_iou']:.3f})"
)
panels = []
for it, pred, v in list(zip(PETS_EVAL, head_preds, head_ious, strict=True))[:3]:
    panels += [
        (f"head · IoU {v:.2f}", overlay(it["image"], pred, THEME["works"])),
        ("ground truth", overlay(it["image"], it["mask"], THEME["truth"])),
    ]
show_masks(panels, width=2.6)

# %% [markdown]
# This cell shows the raw score `w · f + b` per 16 × 16 patch of three held-out pets, where the mask above
# is only its sign: violet = pet-like (above 0), orange = background-like, white = undecided.

# %%
# @title Head score per patch, as a heatmap { display-mode: "form" }
cmap = mpl.colors.LinearSegmentedColormap.from_list("score", [THEME["fails"], "#FFFFFF", THEME["works"]])
panels = []
for it in PETS_EVAL[:3]:
    feat = features_for(PETS, it, PETS["feat_side"])
    h = {k: v.to(feat.device) for k, v in head_full5.items()}
    with torch.no_grad():
        z = (feat - h["feat_mean"].view(-1, 1, 1)) / h["feat_std"].view(-1, 1, 1)
        score = F.conv2d(z[None], h["weight"], h["bias"])[0, 0].cpu().numpy()
    rgb = (cmap(0.5 + 0.5 * score / (np.abs(score).max() + 1e-9))[..., :3] * 255).astype(np.uint8)
    blocks = np.repeat(np.repeat(rgb, 16, 0), 16, 1)  # one 16 px block per patch
    panels += [
        (it["name"], it["image"]),
        (f"score per patch · {score.min():.1f} to {score.max():.1f}", blocks),
    ]
show_masks(panels, width=2.6)

# %% [markdown]
# <!-- callout: across | Across approaches -->
# Two ways to the same mask, scored on the same hidden ground truth:
#
# - **SAM** needs *one hint per image* and no training.
# - **The head** needs *five full masks once*, then runs on every image with no hint.
#
# Keep both numbers: the comparison table in section 8 starts with them, and section 2 reruns both on the
# specialised task.

# %% [markdown]
# **Q4.** 🟡 Compare the two numbers you got with the prediction you wrote down.
#
# - Which approach would you pick for 10 000 pet photos, and which for 10 photos?
# - What does each one ask of a person?
#
# <details><summary>One possible answer</summary>
#
# For 10 photos, a click each is nothing, and SAM's masks come with sharp boundaries. For 10 000, a person
# cannot click on each one, while the head, once trained on a few masks, runs with no hint at all. Its
# boundaries follow the 16 px patch grid, so it loses a little IoU at the edges, which matters more as the
# objects get smaller. Hardware or seed may change the numbers; the trade-off stays.
#
# </details>

# %% [markdown]
# <!-- callout: question | 👥 Optional, 3 min -->
# Pick different indices in `demo = PETS_EVAL[0]` (Q2), so you and your neighbour see different pictures, and
# compare.
#
# - Did one click fail on either of yours?
# - If so, was the click on an ambiguous spot (fur against a similar floor, a collar, a second animal)?

# %% [markdown]
# This cell clicks on the floor instead of the pet and shows what SAM returns.

# %%
# @title Click on the floor, not the pet { display-mode: "form" }
it = PETS_EVAL[0]
floor_ys, floor_xs = np.nonzero(~it["mask"])
k = int(np.random.default_rng([SEED, len(floor_ys)]).integers(len(floor_ys)))
floor_pt = [float(floor_xs[k]), float(floor_ys[k])]
by_floor = sam.probability(it["image"], points=[floor_pt], labels=[1]) >= 0.5
floor_title = f"SAM, click on the floor · IoU {iou(by_floor, it['mask']):.2f}"
show_masks(
    [
        (floor_title, overlay(it["image"], by_floor, THEME["works"])),
        ("ground truth", overlay(it["image"], it["mask"], THEME["truth"])),
    ],
    prompts={"point": floor_pt},
)

# %% [markdown]
# <!-- callout: pitfall | Pitfall: a clean mask is not a correct one -->
# SAM returned a confident-looking mask for a click on the floor, so always compare a mask with the ground
# truth (section 2 shows why).

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - When you have a few photos and can click each one → SAM with a point or a box gives sharp masks, with no
#   training.
# - When you have thousands of photos and no one to click → train the linear head on a few full masks.
# - When a mask looks clean → still check it against the ground truth before trusting it.
# - When the head's edges look blocky (16 px grid) → expect lower IoU on small or thin objects; section 2
#   tests it.
