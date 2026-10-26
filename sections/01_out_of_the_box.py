# %% [markdown]
# ## 1. Out of the box: pets

# %% [markdown]
# <!-- callout: objective | Objective -->
# See a foundation model segment pets almost perfectly, zero-shot (with no training on this task).

# %% [markdown]
# Here SAM, DINOv3 and a linear head (explained below the first figure) meet the easy task; keep your
# numbers, because section 2 runs the same three cells on the specialised task.

# %% [markdown]
# This cell defines the prompt, figure and scoring helpers used all through the notebook.

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


def _key_handle(entry):
    """One legend handle for a colour-key entry: "found", or (key, label), or (text, colour)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    known = {
        "found": ("found (prediction and truth)", THEME["works"]),
        "invented": ("invented (prediction only)", THEME["fails"]),
        "missed": ("missed (truth only)", THEME["truth"]),
        "unscored": ("not scored (grey band)", "#E0E0E0"),
        "mask": ("the model's mask", THEME["works"]),
        "truth": ("ground truth", THEME["truth"]),
    }
    if isinstance(entry, str):
        entry = (entry, None)
    key, label = entry
    teal = THEME["scribble"]
    if key == "click":
        return Line2D([], [], marker="*", ls="", ms=11, color=teal, mec="white", label=label or "click")
    if key == "box":
        return Patch(fc="none", ec=teal, lw=2, label=label or "box")
    if key in known:
        default, colour = known[key]
        return Patch(fc=colour, ec="#888888", lw=0.5, label=label or default)
    return Patch(fc=label, ec="#888888", lw=0.5, label=key)  # (text, colour)


def add_key(fig, entries: list | None, note: str | None = None) -> None:
    """Lay out `fig`, then add a colour key and an optional one-line note under it (it grows a little).

    Entries: "found" | "invented" | "missed" | "unscored" | "mask" | "truth" | "click" | "box", or a pair
    (key, label) to reword one, or (text, colour) for any other colour. Call it last, instead of
    tight_layout()."""
    fig.tight_layout()
    handles = [_key_handle(e) for e in entries or []]
    width, height = fig.get_size_inches()
    cols = max(1, min(len(handles), int(width // 2.3)))
    rows = -(-len(handles) // cols) if handles else 0
    note_h, key_h = (0.3 if note else 0.0), 0.28 * rows
    extra = note_h + key_h + (0.1 if handles or note else 0.0)
    if not extra:
        return
    new_height = height + extra
    for ax in fig.axes:  # keep every axes the same size in inches, and make room underneath
        pos = ax.get_position(original=True)
        y0, h = (pos.y0 * height + extra) / new_height, pos.height * height / new_height
        ax.set_position([pos.x0, y0, pos.width, h])
    fig.set_size_inches(width, new_height)
    if note:
        fig.text(0.5, 0.1 / new_height, note, ha="center", va="bottom", fontsize=8, style="italic")
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=cols, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, (0.05 + note_h) / new_height))


def show_masks(
    panels: list[tuple[str, np.ndarray]],
    prompts: dict | list | None = None,
    width: float = 3.2,
    legend: list | None = None,
    note: str | None = None,
    ncols: int | None = None,
):
    """One row of RGB panels (title, image), or several rows of `ncols` panels each.

    `prompts` = {"point": [x, y], "box": [x0, y0, x1, y1]} drawn on every panel, or a list with one such dict
    (or None) per panel. `legend` = colour-key entries for add_key(); `note` = one italic line under it."""
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, width * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (title, img) in zip(axes, panels, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=8 if ncols > 3 else 9)
        ax.axis("off")
    per_panel = prompts if isinstance(prompts, list) else [prompts] * len(panels)
    for ax, p in zip(axes, per_panel, strict=False):
        if not p:
            continue
        if p.get("point") is not None:
            x, y = p["point"]
            ax.plot(x, y, marker="*", markersize=14, color=THEME["scribble"], markeredgecolor="white")
        if p.get("box") is not None:
            x0, y0, x1, y1 = p["box"]
            rect = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2)
            rect.set_edgecolor(THEME["scribble"])
            ax.add_patch(rect)
    add_key(fig, legend, note)
    plt.show()


def pet_iou(pred: np.ndarray, item: dict) -> float:
    """IoU of a boolean mask with the pet, counted only inside the scored area (the grey band is left out)."""
    r = region(item)
    p, t = pred & r, item["mask"] & r
    union = (p | t).sum()
    return float((p & t).sum() / union) if union else 1.0  # same formula as iou() in 1.2, usable before it


def truth_panel(item: dict) -> np.ndarray:
    """The photo with the ground truth in green; the not-scored band is drawn grey."""
    return overlay(item["image"], item["mask"], THEME["truth"], unscored=~region(item))


def error_panel(item: dict, pred: np.ndarray) -> np.ndarray:
    """Violet = pet found, orange = false alarm, green = missed, grey = not scored."""
    r = region(item)
    pred, truth = pred & r, item["mask"] & r
    img = overlay(item["image"], pred & truth, THEME["works"], unscored=~r)
    img = overlay(img, pred & ~truth, THEME["fails"])
    return overlay(img, truth & ~pred, THEME["truth"])


def features_for(dataset: dict, item: dict, long_side: int) -> torch.Tensor:
    """DINOv3 patch features [384, h, w]: from the precomputed file when it exists, else computed now."""
    prefix = "pets" if dataset is pets_task else "specialised"
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
# This cell loads the held-out pets that every approach in this notebook is scored on.

# %%
# @title Load the held-out pets { display-mode: "form" }
pets_task["feat_side"] = 256 if SMOKE else pets_task["index"]["feature_long_side"]
pets_task["eval"] = pets_task["held"] if (SMOKE or DEVICE == "cuda") else pets_task["held"][:12]
pets_eval = [load_item(pets_task, it) for it in pets_task["eval"]]
print(
    f"evaluation set: {len(pets_eval)} held-out pets · "
    f"DINOv3 features at {pets_task['feat_side']} px long side"
)

# %% [markdown]
# ### 1.1 SAM: one click, one box
#
# **What the $\color{#E0E0E0}{\blacksquare}$ grey band is.** The Oxford pet photos come with a ground truth
# drawn by annotators. Around every
# outline they added a "not sure" strip, about 10 to 20 pixels wide. A pixel in that strip is neither pet nor
# floor: the annotator said "unknown". We do not score it, so ground-truth pictures show it in
# $\color{#9E9E9E}{\blacksquare}$ grey and a mask
# is judged only on the rest. The difference between "unknown" and "background" is the idea behind the whole
# workshop.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The next cell lets you drive SAM yourself on the held-out pets. Before you start: will one click be enough
# for a clean mask of the whole animal, and what could go wrong (fur against floor, the tail, a second
# animal)? Say it to your neighbour or write it down.

# %% [markdown]
# The cell below shows a held-out pet and takes a click, a box or a word, then draws SAM's mask with its IoU
# against the ground truth (IoU = overlap ÷ union of the two masks: 1 is a perfect match, 0 no overlap; 1.2
# explains it). The ◀ ▶ buttons change pet. 👁 Show ground truth colours the result:
# $\color{#6A1B9A}{\blacksquare}$ violet found, $\color{#E65100}{\blacksquare}$ orange
# invented, $\color{#2E7D32}{\blacksquare}$ green missed, $\color{#E0E0E0}{\blacksquare}$ light grey not
# scored. Without Colab it shows one fixed click and one box instead.
#
# **Q2.** 🟢 Drive SAM on three or four pets.
#
# - Click on a pet, shift-click on the floor, then draw a box. What does each give you?
# - Add a second click on a part SAM missed. What changes?
# - Press 👁 Show ground truth. Does the mask differ from the truth at the outline, or is a whole part
#   missing or added?
# - Where does one click fail, and does a box fix it?
# - Optional: type `dog` or `cat` and press "Ask SAM 3" (only if SAM 3 is in your cache).
#
# Look at the floor click: SAM still returns a clean-looking mask. A clean mask is not a correct one, so
# always compare it with the ground truth (section 2 shows why).

# %%
# @title Q2 — Click, box or a word: drive SAM yourself { display-mode: "form" }
sam_playground(pets_eval, "Pets: click, box, or a word", text_prompt=True, default_text="dog")

# %% [markdown]
# <!-- diagram: s01-how-sam-works | How SAM turns a hint into a mask -->
# Read left to right. The big image encoder reads the photo once; each hint (a click, a box or, with SAM 3,
# a word) goes through a small prompt encoder; the mask decoder combines the two and returns a mask with a
# score.

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

# %% [markdown]
# <!-- diagram: s01-iou | IoU: pixels in both masks divided by pixels in either -->
# Count the pixels both masks mark (80), count the pixels either mask marks (120), then divide.

# %% [markdown]
# <details><summary>The same in words</summary>
#
# - Example: the predicted mask and the true mask each cover 100 pixels and share 80. The union is 120
#   pixels, so IoU = 80 / 120 = 0.67.
#
# </details>
#
# - 1.0 is a perfect match, 0.0 means no overlap. Section 2 adds one extra metric for thin structures.
# - On the pets, both masks are first cut to the scored area, so the $\color{#E0E0E0}{\blacksquare}$ grey
#   "not sure" band counts for nothing.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 1 min) -->
# IoU = intersection ÷ union. We wrote the intersection. Fill `my_union`: how many pixels are True in at
# least one of two boolean masks? The cell below checks your line on this 4 × 4 play example (1 = True) and
# continues with ours if you skip it:
#
# ```
# pred   truth
# 0000   0110
# 0111   0110
# 0111   0110
# 0000   0000
# ```
#
# The intersection is 4 and the union is 8, so the answer is 8 and the IoU is 0.5.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_union(pred: np.ndarray, truth: np.ndarray) -> int:
    """Reference: the number of pixels where pred or truth is True."""
    return int((pred | truth).sum())


_play_pred = np.zeros((4, 4), bool)
_play_pred[1:3, 1:4] = True  # 6 pixels
_play_truth = np.zeros((4, 4), bool)
_play_truth[0:3, 1:3] = True  # 6 pixels, 4 shared -> union 8, IoU = 4 / 8

# %%
# ✍️ YOUR CODE
def my_union(pred: np.ndarray, truth: np.ndarray) -> int:
    """Number of pixels that are True in pred, in truth, or in both.
    e.g. 8"""
    ...


union_pixels = use(
    my_union, reference=_ref_union, check=lambda f: int(f(_play_pred, _play_truth)), name="iou"
)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two boolean masks of the same shape; 1.0 when both are empty.
    e.g. 0.5"""
    inter, union = float((pred & truth).sum()), float(union_pixels(pred, truth))
    return inter / union if union else 1.0


# %% [markdown]
# %% [markdown]
# This cell scores SAM on every held-out pet with the IoU above and keeps the mean as the zero-shot number
# for section 2. The click is one pixel of a foreground scribble and the box is the tight box around the pet,
# both taken from the ground truth: what a person would give SAM.

# %%
# @title Score SAM on every held-out pet { display-mode: "form" }
t0 = time.monotonic()
zero_shot_all = {"point": [], "box": []}
for it in tqdm(pets_eval, desc="SAM on the held-out pets"):
    pt = click_point(it)
    by_point = sam.probability(it["image"], points=[pt], labels=[1]) >= 0.5
    by_box = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
    zero_shot_all["point"].append(pet_iou(by_point, it))
    zero_shot_all["box"].append(pet_iou(by_box, it))
zero_shot = {k: float(np.mean(v)) for k, v in zero_shot_all.items()}
RESULTS["pets_zero_shot_iou"] = zero_shot["box"]  # the headline: SAM's best single prompt, one box
RESULTS["pets_zero_shot_point_iou"] = zero_shot["point"]
RESULTS["pets_zero_shot_box_iou"] = zero_shot["box"]
RESULTS["pets_zero_shot_seconds"] = round(time.monotonic() - t0, 1)
print(
    f"SAM zero-shot on {len(pets_eval)} held-out pets · mean IoU one click {zero_shot['point']:.3f} · "
    f"one box {zero_shot['box']:.3f} · {RESULTS['pets_zero_shot_seconds']} s"
)

# %% [markdown]
# ### 1.3 What DINOv3 sees
#
# This cell keeps the **3 main directions (a PCA)** of each patch's 384 numbers and shows them as
# $\color{#E53935}{\blacksquare}$ red, $\color{#43A047}{\blacksquare}$ green
# and $\color{#1E88E5}{\blacksquare}$ blue, next to each photo. The same colour means the model sees those
# patches as similar (both cat
# ears, say). What to look for: does the animal get colours that differ from the floor, and do its parts
# (face, body) differ from each other? The blocks are the 16 px patches.

# %%
# @title Show the patch features as colours { display-mode: "form" }
fig_items = [load_item(pets_task, it) for it in pets_task["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(pets_task, it, pets_task["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(
    panels,
    width=2.6,
    note="Colours = the 3 main directions of the 384 numbers per patch (red, green, blue). They are not "
    "classes: similar colour = patches the model finds similar.",
)

# %% [markdown]
# **Q3.** 🟡 Look at the PCA images above.
#
# - Which pixels share a colour: the whole animal, parts of it (head, fur, legs), or the floor too?
# - Does the colour follow the object's outline?
# - How fine is the grid behind it?
#
# <details><summary>What to look for</summary>
#
# - The colours are not classes: they are the 3 directions along which the features vary most in this image.
# - Typically the animal separates from the floor in one direction and its parts (face, body) in another.
# - The blocky edges are the 16 px grid: the features know <i>what</i> is where, not <i>exactly where</i> the
#   boundary is. Whether that is enough depends on how big the thing you segment is.
#
# </details>

# %% [markdown]
# ### 1.4 A linear head on five pets

# %% [markdown]
# <!-- diagram: s01-patches-to-mask | From 16 px patches to a mask: DINOv3 and a linear head -->
# Follow the outlined square from the photo to the final mask: its 384 numbers become one score, then one
# value p for that square. The same steps run for every square.

# %% [markdown]
# <details><summary>The same in words</summary>
#
# **How the head works.** Each patch gets one score from its 384 feature numbers `f`:
#
# `score = w · f + b` (384 weights + 1 bias = 385 parameters), the same `w` and `b` for every patch (a
# "1 × 1 convolution"). Score above 0: the patch is "pet"; below 0: "background".
#
# A sigmoid `σ` squeezes the score into a probability `p = σ(score)` between 0 and 1 (score 2.2 gives
# p = 0.90, score 0 gives 0.5), so "score above 0" and "p above 0.5" say the same thing. The per-patch map is
# then upsampled to the image.
#
# </details>
#
# Training uses a **balanced loss**: pet pixels and background pixels each get half of the total weight, so
# a small pet is not drowned out by the floor.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The next cells train a classifier with 385 parameters on the patch features of just **five** pets with
# full masks, then score it on the held-out ones. Will its IoU land above, near or below SAM's one-click
# number from 1.2? Write down a number before you run.

# %% [markdown]
# This cell defines the linear head that every later training step reuses.

# %%
# @title Linear head: train, predict, load a checkpoint { display-mode: "form" }
def train_head(
    feats: list[torch.Tensor], labels: list[np.ndarray], steps: int, lr: float = 0.05, balanced: bool = True
) -> tuple[dict, float]:
    """Linear head (384 -> 1) on standardised patch features. Labels per pixel: 1 fg, 0 bg, -1 ignored.

    Pixels labelled -1 are skipped. `balanced=True`: each class gets half of the total loss weight.
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
    key = "pets" if dataset is pets_task else specialised["key"]
    wanted = (key, regime, long_side)
    heads = PRECOMPUTED_HEADS
    entry = next((h for h in heads if (h["dataset"], h["regime"], h["long_side"]) == wanted), None)
    if entry is None:
        raise SystemExit(f"no precomputed head for {wanted}. Run setup_check.ipynb first.")
    return load_file(PRECOMPUTED / entry["file"]), entry


def fit_pet_head(train_yourself: bool, n_pets: int, steps: int) -> tuple[list[dict], dict, float, str]:
    """Train the pet head on the full masks of the first `n_pets` fit pets, or load ours.

    The "not sure" band is labelled -1, so the loss skips it. Returns pets, head, seconds, source line.
    """
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_PET_HEAD must be True or False.")
    items = [load_item(pets_task, it) for it in pets_task["fit"][:n_pets]]
    feats = [features_for(pets_task, it, pets_task["feat_side"]) for it in items]
    if not train_yourself:
        head, entry = load_head(pets_task, "full5")
        src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} pets)"
        return items, head, entry["train_sec"], src
    labels = []
    for it in items:  # full masks: allowed in section 1 only
        lab = it["mask"].astype(np.int8)
        lab[~region(it)] = -1
        labels.append(lab)
    head, sec = train_head(feats, labels, steps)
    return items, head, sec, f"trained here on {len(items)} pets, {steps} steps"


# %% [markdown]
# This cell trains the head on the full masks of five pets, leaving out the
# $\color{#E0E0E0}{\blacksquare}$ grey band. Untick
# `TRAIN_PET_HEAD` to load our checkpoint instead.

# %%
# @title Train the linear head on five pets { display-mode: "form" }
TRAIN_PET_HEAD = True  # @param {type:"boolean"}
N_TRAIN_PETS = 5
train_items, head_full5, head_sec, head_src = fit_pet_head(TRAIN_PET_HEAD, N_TRAIN_PETS, HEAD_STEPS)
print(f"linear head · {head_src} · {head_sec:.1f} s on {DEVICE}")
if head_sec > 60:
    print("ℹ️ Slow session? Untick TRAIN_PET_HEAD above to load our checkpoint next time.")

# %% [markdown]
# This cell runs the head you just trained on one training pet and shows each step (patches, one probability
# per patch, the upsampled map, the final mask), then works through one patch by hand. The
# $\color{#00838F}{\blacksquare}$ teal square marks
# that patch; $\square$ white to $\color{#6A1B9A}{\blacksquare}$ violet is the probability p of "pet"
# ($\square$ white 0, $\color{#6A1B9A}{\blacksquare}$ violet 1). What to look for: the map in
# panel 2 is coarse (one value per patch), and panels 3 and 4 only smooth and threshold it.

# %%
# @title The linear head step by step on one pet { display-mode: "form" }
hx = train_items[0]
hfeat = features_for(pets_task, hx, pets_task["feat_side"])  # [384, h, w]: 384 numbers per patch
dev = hfeat.device
with torch.no_grad():
    hp = {k: v.to(dev) for k, v in head_full5.items()}
    hz = (hfeat - hp["feat_mean"].view(-1, 1, 1)) / hp["feat_std"].view(-1, 1, 1)
    hscore = F.conv2d(hz[None], hp["weight"], hp["bias"])[0, 0]  # w·f + b, one score per patch
    hprob = torch.sigmoid(hscore)
    hup = F.interpolate(hprob[None, None], size=hx["mask"].shape, mode="bilinear", align_corners=False)[0, 0]
gh, gw = hprob.shape
py, px = np.unravel_index(int(hprob.argmax()), hprob.shape)  # the most pet-like patch
f_vec, w_vec = hz[:, py, px], hp["weight"].reshape(-1)
f_head = ", ".join(f"{v:.2f}" for v in f_vec[:4].tolist())
dot, bias = float((w_vec * f_vec).sum()), float(hp["bias"])
total, p_patch = float(hscore[py, px]), float(hprob[py, px])
print(f"patch (row {py}, col {px}) · f = [{f_head}, … 384 numbers]")
print(f"w · f + b = {dot:.2f} + ({bias:.2f}) = {total:.2f} → p = σ({total:.2f}) = {p_patch:.2f}")

prob_cmap = mpl.colors.LinearSegmentedColormap.from_list("prob", ["#FFFFFF", THEME["works"]])
H, W = hx["mask"].shape
fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.4))
axes[0].imshow(hx["image"])
axes[0].vlines(np.arange(0, W + 1, W / gw), 0, H, colors="white", linewidths=0.4)
axes[0].hlines(np.arange(0, H + 1, H / gh), 0, W, colors="white", linewidths=0.4)
cell_box = plt.Rectangle((px * W / gw, py * H / gh), W / gw, H / gh, fill=False, ec=THEME["scribble"], lw=2)
axes[0].add_patch(cell_box)
axes[0].set_title("1 · image cut into patches\n(white lines = the grid)", fontsize=9)
axes[1].imshow(hprob.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1, interpolation="nearest")
axes[1].add_patch(plt.Rectangle((px - 0.5, py - 0.5), 1, 1, fill=False, ec=THEME["scribble"], lw=2))
axes[1].set_title("2 · one p per patch\n(white 0, violet 1)", fontsize=9)
axes[2].imshow(hup.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1)
axes[2].set_title("3 · upsampled\nto the image", fontsize=9)
axes[3].imshow(overlay(hx["image"], predict_head(head_full5, hfeat, hx["mask"].shape), THEME["works"]))
axes[3].set_title("4 · p > 0.5 → pet\n(violet = predicted pet)", fontsize=9)
for ax in axes:
    ax.axis("off")
add_key(
    fig,
    [("box", "patch worked through in the printout"), ("16 px patch grid (panel 1)", "#FFFFFF"),
     ("p near 1 = pet (panels 2-4)", THEME["works"]), ("p near 0 = background", "#F2F2F2")],
    note="Panel 4 is the head's mask: violet = pixels with p above 0.5.",
)
plt.show()

# %% [markdown]
# This cell runs the head on the held-out pets it never saw and scores it with the same IoU. The figure shows
# SAM from the box (row 1) and the head (row 2) on the same three pets: the worst, median and best by SAM's
# IoU. The $\color{#00838F}{\blacksquare}$ teal rectangle is the box given to SAM. What to look for: is the
# spread across pets as large for
# the head as for SAM, and are the head's edges blockier (16 px patches)?

# %%
# @title Score the head, and SAM next to it { display-mode: "form" }
t0 = time.monotonic()
head_preds = [
    predict_head(head_full5, features_for(pets_task, it, pets_task["feat_side"]), it["mask"].shape)
    for it in pets_eval
]
head_ious = [pet_iou(pred, it) for pred, it in zip(head_preds, pets_eval, strict=True)]
RESULTS["pets_head_iou"] = float(np.mean(head_ious))
RESULTS["pets_head"] = {
    "iou": RESULTS["pets_head_iou"],
    "n_train": len(train_items),
    "train_sec": round(float(head_sec), 2),
    "infer_sec": round(time.monotonic() - t0, 1),
}
n_eval = len(pets_eval)
print(f"Linear head: trained in 1.4 on {len(train_items)} pets with full masks, run on {n_eval} unseen pets.")
print(f"Score: per-pet IoU with the ground truth (grey band not scored), averaged over the {n_eval} pets.")
print(f"  linear head       mean IoU {RESULTS['pets_head_iou']:.3f}")
print(f"  SAM, one click    mean IoU {RESULTS['pets_zero_shot_point_iou']:.3f}  (the 1.2 masks)")
print(f"  SAM, one box      mean IoU {RESULTS['pets_zero_shot_box_iou']:.3f}  (the 1.2 masks)")
pet_order = np.argsort(zero_shot_all["box"])  # the same three pets in both rows
picked = [("worst", pet_order[0]), ("median", pet_order[n_eval // 2]), ("best", pet_order[-1])]
sam_row, head_row, boxes = [], [], []
for label, pos in picked:
    it = pets_eval[int(pos)]
    by_box = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
    sam_iou, head_iou = zero_shot_all["box"][int(pos)], head_ious[int(pos)]
    sam_row.append((f"{label} pet · SAM from the box\nIoU {sam_iou:.2f}", error_panel(it, by_box)))
    head_row.append((f"same pet · linear head\nIoU {head_iou:.2f}", error_panel(it, head_preds[int(pos)])))
    boxes.append({"box": box_prompt(it)})
show_masks(
    sam_row + head_row,
    prompts=boxes + [None] * len(head_row),
    ncols=3,
    legend=["found", "invented", "missed", "unscored", ("box", "box given to SAM")],
    note="Pets chosen by SAM's box IoU (worst, median, best); the head is shown on the same three.",
)

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
# - 10 photos: a click each is nothing, and SAM's masks have sharp boundaries.
# - 10 000 photos: nobody clicks them all, while the head, trained once on a few masks, needs no hint. Its
#   edges follow the 16 px grid, so it loses a little IoU, more as objects get smaller.
# - Hardware or seed may change the numbers; the trade-off stays.
#
# </details>

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - When you have a few photos and can click each one → SAM with a point or a box gives sharp masks, with no
#   training.
# - When you have thousands of photos and no one to click → train the linear head on a few full masks.
# - When a mask looks clean → still check it against the ground truth before trusting it.
# - When the head's edges look blocky (16 px grid) → expect lower IoU on small or thin objects; section 2
#   tests it.
