# %% [markdown]
# ## 1. Out of the box: pets

# %% [markdown]
# <!-- callout: objective | Objective -->
# See a foundation model segment pets almost perfectly, zero-shot (with no training on this task).

# %% [markdown]
# Here SAM, DINOv3 and a linear head (explained below the first figure) meet the easy task; keep your
# numbers, because section 2 runs the same three cells on the specialised task.

# %% [markdown]
# This helper cell defines how to turn a person's hint into a SAM prompt, `show_masks()`, the figure used for
# every comparison in the notebook (prediction in violet, ground truth in green, prompts in teal), and the
# small functions that score a mask only inside the scored area (the grey band is left out).

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


def pet_iou(pred: np.ndarray, item: dict) -> float:
    """IoU of a boolean mask with the pet, counted only inside the scored area (the grey band is left out)."""
    r = region(item)
    return iou(pred & r, item["mask"] & r)


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


def show_sam_demo(items: list[dict], index: int) -> None:
    """SAM with one click and with one box on items[index], next to the ground truth."""
    if not isinstance(index, int) or not 0 <= index < len(items):
        raise SystemExit(f"DEMO_INDEX must be a whole number from 0 to {len(items) - 1}.")
    pet = items[index]
    point, box = click_point(pet), box_prompt(pet)
    t0 = time.monotonic()
    by_point = sam.probability(pet["image"], points=[point], labels=[1]) >= 0.5
    by_box = sam.probability(pet["image"], box=box) >= 0.5
    print(f"{pet['name']}: two SAM calls in {time.monotonic() - t0:.1f} s on {DEVICE}")
    works = THEME["works"]
    show_masks(
        [
            ("SAM, one click", overlay(pet["image"], by_point, works)),
            ("SAM, one box", overlay(pet["image"], by_box, works)),
            ("ground truth (grey = not scored)", truth_panel(pet)),
        ],
        prompts={"point": point, "box": box},
    )


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
# This cell loads the held-out pets, the evaluation set every approach in this notebook is scored on (12 of
# them on a CPU, to keep cells short), and fixes the resolution DINOv3 works at.

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
# **What the grey band is.** The Oxford pet photos come with a ground truth drawn by annotators. Around every
# outline they added a "not sure" strip, about 10 to 20 pixels wide. A pixel in that strip is neither pet nor
# floor: the annotator said "unknown". We do not score it, so ground-truth pictures show it in grey and a mask
# is judged only on the rest. The difference between "unknown" and "background" is the idea behind the whole
# workshop.

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
# - Try another pet: move the `DEMO_INDEX` slider (0 to 11) and run the cell again.

# %% [markdown]
# This cell runs SAM twice on the held-out pet you pick, once with a single click and once with a box, and
# shows both masks next to the ground truth, so you can see what a promptable model gives for free.

# %%
# @title Q2 — SAM with one click and with one box on a held-out pet { display-mode: "form" }
DEMO_INDEX = 0  # @param {type:"slider", min:0, max:11, step:1}
show_sam_demo(pets_eval, DEMO_INDEX)

# %% [markdown]
# Now drive SAM yourself on the same held-out pets: the widget below takes a click, a box or a word and draws
# the mask with its IoU against the ground truth (IoU = overlap ÷ union of the two masks: 1 is a perfect
# match, 0 no overlap; 1.2 explains it). Without Colab it shows one fixed click and one box instead.
#
# **Q3.** 🟢 Click on a pet, then click on the floor, then draw a box around the pet.
#
# - What does each of the three give you?
# - What changes when you add a second click on the part SAM missed?
# - Type `dog` or `cat` in the text box and press "Ask SAM 3" (only if SAM 3 is in your cache).

# %%
# @title Q3 — Click, box or a word: drive SAM yourself { display-mode: "form" }
sam_playground(pets_eval, "Pets: click, box, or a word", text_prompt=True, default_text="dog")

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
# - On the pets, both masks are first cut to the scored area, so the grey "not sure" band counts for nothing.

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
# defined (inside the scored area) and keeps the mean as this task's zero-shot number, for section 2 to
# compare.

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
# A mean hides the spread, so this cell shows the **worst, median and best** held-out pet by one-box IoU:
# violet where mask and truth agree, orange for a false alarm (extra mask), green for what SAM missed, grey
# for the band that is not scored.

# %%
# @title Worst, median and best pet by IoU { display-mode: "form" }
order = np.argsort(zero_shot_all["box"])
panels = []
for label, pos in (("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])):
    it = pets_eval[int(pos)]
    pred = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
    panels.append((f"{label} · IoU {zero_shot_all['box'][int(pos)]:.2f}", error_panel(it, pred)))
show_masks(panels)

# %% [markdown]
# ### 1.3 What DINOv3 sees
#
# This cell keeps the **3 main directions (a PCA)** of each patch's 384 numbers and shows them as red, green
# and blue, so the same colour means the model sees those patches as similar (both cat ears, say).

# %%
# @title Show the patch features as colours { display-mode: "form" }
fig_items = [load_item(pets_task, it) for it in pets_task["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(pets_task, it, pets_task["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(panels, width=2.6)

# %% [markdown]
# **Q4.** 🟡 Look at the PCA images above.
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
# `score = w · f + b` (384 weights + 1 bias = 385 parameters), the same `w` and `b` for every patch (a
# "1 × 1 convolution"). Score above 0: the patch is "pet"; below 0: "background".
#
# A sigmoid `σ` squeezes the score into a probability `p = σ(score)` between 0 and 1 (score 2.2 gives
# p = 0.90, score 0 gives 0.5), so "score above 0" and "p above 0.5" say the same thing. The per-patch map is
# then upsampled to the image.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The next cells train a classifier with 385 parameters on the patch features of just **five** pets with
# full masks, then score it on the held-out ones. Will its IoU land above, near or below SAM's one-click
# number from 1.2? Write down a number before you run.

# %% [markdown]
# This helper cell defines the head used for every training step in the notebook (that 1 × 1 convolution on
# standardised patch features, with a balanced loss, where each class gets half the total weight, that
# ignores pixels labelled −1), `load_head()` for our precomputed checkpoints, and `fit_pet_head()`, which
# trains the pet head or loads ours.

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
# This cell trains the head on the full masks of five fit pets (only sections 1 and 2 use full masks); the
# grey "not sure" band is left out of its loss. Untick `TRAIN_PET_HEAD` and run it again to load our
# checkpoint instead.

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
# per patch, the upsampled map, the final mask), then works through one patch by hand.

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
fig, axes = plt.subplots(1, 4, figsize=(11.6, 3))
axes[0].imshow(hx["image"])
axes[0].vlines(np.arange(0, W + 1, W / gw), 0, H, colors="white", linewidths=0.4)
axes[0].hlines(np.arange(0, H + 1, H / gh), 0, W, colors="white", linewidths=0.4)
cell_box = plt.Rectangle((px * W / gw, py * H / gh), W / gw, H / gh, fill=False, ec=THEME["scribble"], lw=2)
axes[0].add_patch(cell_box)
axes[0].set_title("1 · image cut into patches", fontsize=9)
axes[1].imshow(hprob.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1, interpolation="nearest")
axes[1].add_patch(plt.Rectangle((px - 0.5, py - 0.5), 1, 1, fill=False, ec=THEME["scribble"], lw=2))
axes[1].set_title("2 · one p per patch (white 0, violet 1)", fontsize=9)
axes[2].imshow(hup.cpu().numpy(), cmap=prob_cmap, vmin=0, vmax=1)
axes[2].set_title("3 · upsampled to the image", fontsize=9)
axes[3].imshow(overlay(hx["image"], predict_head(head_full5, hfeat, hx["mask"].shape), THEME["works"]))
axes[3].set_title("4 · p > 0.5 → pet", fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# This cell runs the linear head you trained in 1.4 (5 pets, full masks) on the held-out pets it never saw.
# For each pet, IoU is the overlap of the head's mask and the ground truth divided by their union, with the
# grey band left out. It then averages over the pets and shows the worst, median and best one.

# %%
# @title Score the head on the held-out pets { display-mode: "form" }
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
panels, works = [], THEME["works"]
order = np.argsort(head_ious)
for label, pos in (("worst", order[0]), ("median", order[n_eval // 2]), ("best", order[-1])):
    it = pets_eval[int(pos)]
    panels += [
        (f"head, {label} · IoU {head_ious[int(pos)]:.2f}", overlay(it["image"], head_preds[int(pos)], works)),
        ("ground truth", truth_panel(it)),
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
# **Q5.** 🟡 Compare the two numbers you got with the prediction you wrote down.
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
# Pick different `DEMO_INDEX` values (Q2), so you and your neighbour see different pictures, and compare.
#
# - Did one click fail on either of yours?
# - If so, was the click on an ambiguous spot (fur against a similar floor, a collar, a second animal)?

# %% [markdown]
# This cell clicks on the floor instead of the pet and shows what SAM returns.

# %%
# @title Click on the floor, not the pet { display-mode: "form" }
it = pets_eval[0]
floor_ys, floor_xs = np.nonzero(region(it) & ~it["mask"])  # a floor pixel, not one in the grey band
k = int(np.random.default_rng([SEED, len(floor_ys)]).integers(len(floor_ys)))
floor_pt = [float(floor_xs[k]), float(floor_ys[k])]
by_floor = sam.probability(it["image"], points=[floor_pt], labels=[1]) >= 0.5
floor_title = f"SAM, click on the floor · IoU {pet_iou(by_floor, it):.2f}"
show_masks(
    [
        (floor_title, overlay(it["image"], by_floor, THEME["works"])),
        ("ground truth", truth_panel(it)),
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
