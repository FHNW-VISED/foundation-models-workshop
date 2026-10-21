# %% [markdown]
# ## 7. Path 3 · Skip specialist training
#
# Paths 1 and 2 trained a head once from strokes on five images, and that head then segmented every new image
# on its own. Path 3 trains nothing: the strokes a person draws on an image become SAM's prompts on *that
# same image*, and SAM's answer is the result.
#
# It is the difference between teaching an intern once and pointing at every image yourself: no training time
# and no training data, but the pointing never stops.
#
# Section 2 gave SAM clicks on the vessels and one big box. Here we add what it lacked: "not here" clicks from
# the background strokes, and many small boxes, one per stroke.

# %% [markdown]
# <!-- callout: objective | Objective -->
# Prompt the foundation model directly with scribble points and boxes, no training, and know when that beats
# training a head.

# %% [markdown]
# <!-- callout: theory | Theory: what a SAM prompt says -->
# Sections 2 and 6.1 showed that forty scattered clicks ask SAM for one object, and it answers with a region.
# What is new here:
#
# - Split the prompt into many small ones, each its own object (pets: one dog, one mask), and combine the
#   masks. SAM then answers many local questions instead of one global one.
# - Nothing is trained. The image encoder runs once per image, and every extra prompt costs only a fast
#   decoder pass.

# %% [markdown]
# <!-- callout: go-deeper | SAM papers -->
# - SAM, Kirillov et al., ICCV 2023: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
# - SAM 2, Ravi et al. 2024: [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)

# %% [markdown]
# This helper cell adds a colour for each direct approach and defines the prompts we take from the strokes:
# the separate strokes of an image, a click at the centre of a stroke, SAM on many (box, click) objects at
# once, the two direct recipes on a list of images, and a figure that draws boxes.

# %%
# @title Path 3 helpers: colours, strokes, SAM on many prompts, the two recipes { display-mode: "form" }
from matplotlib.patches import Rectangle

APPROACH_COLOURS |= {
    "sam_clicks": "#546E7A",
    "sam_boxes": "#880E4F",
}
APPROACH_NAMES |= {
    "sam_clicks": "direct: SAM from stroke clicks",
    "sam_boxes": "direct: SAM, one box + click per stroke",
}


def stroke_masks(item: dict, value: int = 1) -> list[np.ndarray]:
    """The separate strokes of one kind (1 vessel, 2 background) as boolean masks: the 8-connected pieces
    of the scribble map."""
    lab, n = ndi.label(item["scribbles"] == value, structure=np.ones((3, 3)))
    return [lab == k for k in range(1, n + 1)]


def stroke_click(stroke: np.ndarray) -> list[float]:
    """(x, y) of the stroke pixel nearest the stroke's centre: a click that is always on the stroke."""
    ys, xs = np.nonzero(stroke)
    k = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    return [float(xs[k]), float(ys[k])]


@torch.no_grad()
def sam_many(image: np.ndarray, boxes: list[list[float]], clicks: list[list[float]] | None = None, chunk=32):
    """SAM with one object per box (plus one positive click inside it, if given), decoded in chunks after a
    single image encoding. Returns the highest probability over the objects: a pixel is foreground as soon
    as one object covers it."""
    best = np.zeros(image.shape[:2], np.float32)
    for s in range(0, len(boxes), chunk):
        kw = {"input_boxes": [boxes[s : s + chunk]]}
        if clicks is not None:
            c = clicks[s : s + chunk]
            kw |= {"input_points": [[[p] for p in c]], "input_labels": [[[1]] * len(c)]}
        inputs = sam.proc(images=image, return_tensors="pt", **kw)
        if sam._cache is None or sam._cache[0] is not image:  # the same image cache as Sam.probability
            sam._cache = (image, sam.model.get_image_embeddings(inputs["pixel_values"].to(sam.device)))
        out = sam.model(
            image_embeddings=sam._cache[1],
            input_points=inputs["input_points"].to(sam.device) if clicks is not None else None,
            input_labels=inputs["input_labels"].to(sam.device) if clicks is not None else None,
            input_boxes=inputs["input_boxes"].to(sam.device),
            multimask_output=False,
        )
        logits = sam.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"], binarize=False)[
            0
        ]
        best = np.maximum(best, torch.sigmoid(logits[:, 0]).amax(0).numpy())
    return best


def direct_clicks(items: list[dict], n: int) -> tuple[list[np.ndarray], list[tuple]]:
    """Recipe 1: n vessel clicks and n background clicks spread along the strokes of each image, one SAM
    prompt per image. Returns the masks and each image's (vessel, background) clicks."""
    preds, clicks = [], []
    for it in items:
        pos, neg = stroke_points(it, 1, n), stroke_points(it, 2, n)
        preds.append(
            sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg)) >= 0.5
        )
        clicks.append((pos, neg))
    return preds, clicks


def direct_boxes(items: list[dict], pad: int, with_click: bool) -> tuple[list[np.ndarray], list[tuple]]:
    """Recipe 2: one box around every vessel stroke (grown by `pad` px; `stroke_box` is set in 7.2), plus a
    click at the stroke's centre if `with_click`; the masks of all strokes are combined. Returns the masks
    and each image's (boxes, clicks)."""
    preds, prompts = [], []
    for it in items:
        strokes = stroke_masks(it, 1)
        boxes = [[float(v) for v in stroke_box(s, pad)] for s in strokes]
        clicks = [stroke_click(s) for s in strokes] if with_click else None
        preds.append(
            sam_many(it["image"], boxes, clicks) >= 0.5 if boxes else np.zeros(it["mask"].shape, bool)
        )
        prompts.append((boxes, clicks))
    return preds, prompts


def show_boxes(panels: list[tuple[str, np.ndarray, list | None]], width: float = 3.2):
    """One row of panels (title, RGB image, boxes or None); boxes are drawn as teal rectangles."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    for ax, (title, img, boxes) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        for x0, y0, x1, y1 in boxes or []:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1, color=THEME["scribble"]))
    plt.tight_layout()
    plt.show()


# %% [markdown]
# This cell draws the two prompt types we take from a person's strokes on the demo image: points `(x, y,
# label)` with label 1 = "vessel" and 0 = "not vessel", and boxes `(x0, y0, x1, y1)`.

# %%
# @title Prompt types made from strokes: points and boxes { display-mode: "form" }
type_fg, type_bg = stroke_points(demo, 1, 8), stroke_points(demo, 2, 8)
type_boxes, type_centres = [], []
for stroke in stroke_masks(demo, 1):
    ys, xs = np.nonzero(stroke)
    h, w = stroke.shape
    type_boxes.append(
        [max(xs.min() - 4, 0), max(ys.min() - 4, 0), min(xs.max() + 4, w - 1), min(ys.max() + 4, h - 1)]
    )
    type_centres.append(stroke_click(stroke))
strokes_rgb = overlay(demo["image"], demo["scribbles"] == 1, THEME["scribble"])
strokes_rgb = overlay(strokes_rgb, demo["scribbles"] == 2, THEME["background"])
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, title, img in [
    (axes[0], "strokes: teal = vessel, grey = background", strokes_rgb),
    (axes[1], "points: ★ label 1, ✖ label 0", demo["image"]),
    (axes[2], "boxes: one per vessel stroke", demo["image"]),
    (axes[3], "box + centre click: one object each", demo["image"]),
]:
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
for x, y in type_fg:
    axes[1].plot(x, y, marker="*", markersize=9, color=THEME["scribble"], markeredgecolor="white")
for x, y in type_bg:
    axes[1].plot(x, y, marker="X", markersize=7, color=THEME["fails"], markeredgecolor="white")
for ax in (axes[2], axes[3]):
    for x0, y0, x1, y1 in type_boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1, color=THEME["scribble"]))
for x, y in type_centres:
    axes[3].plot(x, y, marker="*", markersize=7, color=THEME["scribble"], markeredgecolor="white")
plt.tight_layout()
plt.show()
print(f"{len(type_boxes)} vessel strokes on the demo image = {len(type_boxes)} possible boxes")

# %% [markdown]
# ### 7.1 Clicks on the vessels, and "not here"
#
# The news is **fairness**: the heads of sections 4 to 6 never saw a stroke on the held-out images, but Path 3
# gets strokes on every one of them. It gets more help per image, and pays for it at use time.
#
# The first recipe is section 6.1 without the head:
#
# - Twenty clicks along the vessel strokes and twenty along the background strokes of each *held-out* image.
# - SAM's mask is the answer. The strokes are prompts a person would draw at use time, and the ground truth
#   only scores the result.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Section 6 turned SAM's guesses into a trained head. Here SAM sees each held-out image's own clicks and
# nothing is trained. Will its IoU be higher or lower than that head's, and by roughly how much? Write one
# guess. The table below prints both.

# %% [markdown]
# This cell prompts SAM with the stroke clicks on every evaluation image, scores the masks against the hidden
# truth and shows the demo image (one SAM image encoding per image: seconds on a T4, much slower on a CPU).

# %%
# @title SAM from twenty vessel and twenty background clicks per image { display-mode: "form" }
N_DIRECT_CLICKS = 20  # clicks per kind (vessel / background) and image, taken from the strokes
t0 = time.monotonic()
path3_preds = {"zero_shot": ladder_preds["zero_shot"], "scribbles": ladder_preds["scribbles"]}
path3_preds["sam_clicks"], click_prompts = direct_clicks(SPEC_EVAL, N_DIRECT_CLICKS)
clicks_sec = round(time.monotonic() - t0, 1)
path3_report = {k: rung_report(v) for k, v in path3_preds.items()}
print(
    f"SAM from {N_DIRECT_CLICKS} + {N_DIRECT_CLICKS} stroke clicks on {len(SPEC_EVAL)} images · "
    f"{clicks_sec} s on {DEVICE}"
)
print_path2(path3_report | {"sam_pseudo": path2_report["sam_pseudo"]})
print(f"{'section 2: SAM, 20 vessel clicks only':<46}{RESULTS['sam_prompt_ladder']['20 points']:>6.3f}")
pos, neg = click_prompts[k_demo]
show_clicks(
    [
        (f"{demo['name']} · vessel clicks", demo["image"], pos),
        ("background clicks", demo["image"], neg),
        ("SAM from both vs truth", error_map(demo["image"], path3_preds["sam_clicks"][k_demo], demo), None),
    ],
    width=3.0,
)

# %% [markdown]
# In the last panel, violet = vessel found, orange = SAM marked background as vessel, green = a vessel SAM
# missed.
#
# **Q27.** 🟡 Did the "not here" clicks pull SAM's mask onto the vessels? Change the number of clicks per kind
# and run the next cell: does SAM do better with fewer clicks or with more? Why would that be, given that
# every prompt set describes one object?
#
# ```python
# N_CLICKS_TRY = 5   # try 1, 5, 50
# ```

# %% [markdown]
# This cell runs the same recipe with your number of clicks on every evaluation image and prints it under the
# default of 20.

# %%
# Q27 — fewer or more stroke clicks for SAM
N_CLICKS_TRY = 5  # try 1, 5, 50
if not isinstance(N_CLICKS_TRY, int) or N_CLICKS_TRY < 1:
    raise SystemExit("N_CLICKS_TRY must be a whole number of at least 1.")
try_preds, _ = direct_clicks(SPEC_EVAL, N_CLICKS_TRY)
RESULTS["sam_direct_clicks_try"] = rung_report(try_preds) | {"n_clicks": N_CLICKS_TRY}
print_path2(
    {
        f"{N_DIRECT_CLICKS} + {N_DIRECT_CLICKS} clicks (default)": path3_report["sam_clicks"],
        f"{N_CLICKS_TRY} + {N_CLICKS_TRY} clicks (yours)": RESULTS["sam_direct_clicks_try"],
    }
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# SAM looks for one object that contains every positive click and none of the negative ones.
#
# - A vessel tree is one connected structure, but it is a few pixels wide and spread over the whole crop. The
#   simplest object that contains clicks all over it is a large region around it.
# - The negative clicks, which also sit all over the crop, mostly cut holes in that region rather than
#   turning it into a line.
# - A few clicks can leave SAM a more local object to settle on. Many clicks push it towards "the whole
#   fundus minus a few holes". The marked column tells you which happened on your run.
# - More prompts in *one* set are not more information to SAM if they describe something it does not see as
#   one object.
#
# </details>

# %% [markdown]
# ### 7.2 One small box per stroke
#
# The second recipe splits the question:
#
# 1. Each vessel stroke gets its own box, a few pixels larger than the stroke, and a click at its centre.
# 2. SAM answers each box as a separate object.
# 3. The vessel map is the union of all the answers.
#
# One stroke is a short piece of one vessel, so each box asks a local question SAM can answer: "the thing
# along this line, inside this rectangle".

# %% [markdown]
# This helper cell holds our solution of the next exercise. Open it only if you are stuck.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_stroke_box(stroke: np.ndarray, pad: int) -> list[float]:
    """Reference: min/max of the stroke's columns and rows, padded, then clipped."""
    ys, xs = np.nonzero(stroke)
    h, w = stroke.shape
    return [
        float(max(xs.min() - pad, 0)),
        float(max(ys.min() - pad, 0)),
        float(min(xs.max() + pad, w - 1)),
        float(min(ys.max() + pad, h - 1)),
    ]


# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min) -->
# Fill `my_stroke_box` in the next cell.
#
# - **In:** one stroke as a boolean image (`True` on the stroke) and a margin `pad` in pixels.
# - **Out:** the box `[x0, y0, x1, y1]` around the stroke's pixels, grown by `pad` on every side and
#   clipped to the image (x from 0 to width − 1, y from 0 to height − 1).
# - **Careful:** `np.nonzero` returns rows (y) before columns (x).
#
# The cell checks your version on a 6 × 8 play stroke, without and with a margin, and continues with ours if
# you skip it.


# %%
# ✍️ YOUR CODE
def my_stroke_box(stroke: np.ndarray, pad: int) -> list[float]:
    """The box [x0, y0, x1, y1] around the True pixels of `stroke`, grown by `pad` pixels on every side and
    clipped to the image: 0 <= x <= width - 1, 0 <= y <= height - 1."""
    ...


_play_stroke = np.zeros((6, 8), bool)
_play_stroke[1, 1:4] = True  # a short diagonal stroke near the top-left corner
_play_stroke[2, 4] = _play_stroke[3, 5] = True
stroke_box = use(
    my_stroke_box,
    reference=_ref_stroke_box,
    check=lambda f: [[float(v) for v in f(_play_stroke, p)] for p in (0, 2)],
    name="stroke_box",
)
print(
    "play stroke · box with pad 0:", stroke_box(_play_stroke, 0), "· with pad 2:", stroke_box(_play_stroke, 2)
)

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The strokes cover only some of the vessels (section 4 dropped 70 % of the segments), and each box sits on a
# stroke. Two guesses:
#
# - Will this recipe mark more of the fundus than the clicks of 7.1, or less?
# - Will it beat simply thickening the strokes by a few pixels, with no model at all?

# %% [markdown]
# This cell asks SAM for one object per vessel stroke (a box with 4 px of margin plus a click at the
# centre) on every evaluation image and prints it next to the strokes thickened by 3 px, the "no model"
# reference.

# %%
# @title SAM with one box and one click per stroke, next to the thickened strokes { display-mode: "form" }
BOX_PAD = 1 if SMOKE else 4  # px around each stroke (smoke thumbnails are 4× smaller)
THICKEN = 1 if SMOKE else 3  # px: the "no model" reference, the vessel strokes thickened
t0 = time.monotonic()
path3_preds["sam_boxes"], box_prompts = direct_boxes(SPEC_EVAL, BOX_PAD, with_click=True)
boxes_sec = round(time.monotonic() - t0, 1)
path3_report["sam_boxes"] = rung_report(path3_preds["sam_boxes"])
strokes_alone = [ndi.binary_dilation(it["scribbles"] == 1, iterations=THICKEN) for it in SPEC_EVAL]
strokes_report = rung_report(strokes_alone)
n_boxes = [len(b) for b, _ in box_prompts]
print(
    f"SAM, one box + click per vessel stroke · {int(np.mean(n_boxes))} boxes per image on average · "
    f"{boxes_sec} s on {DEVICE}"
)
print_path2(path3_report | {f"no model: the strokes thickened by {THICKEN} px": strokes_report})
boxes, _ = box_prompts[k_demo]
show_boxes(
    [
        (f"{demo['name']} · {len(boxes)} stroke boxes", demo["image"], boxes),
        (
            "SAM, one object per box vs truth",
            error_map(demo["image"], path3_preds["sam_boxes"][k_demo], demo),
            None,
        ),
        ("strokes thickened vs truth", error_map(demo["image"], strokes_alone[k_demo], demo), None),
    ],
    width=3.0,
)

# %% [markdown]
# Is the demo image typical? This cell scores the box recipe on each evaluation image and shows the worst,
# median and best side by side.

# %%
# @title Box recipe: worst, typical and best evaluation image { display-mode: "form" }
box_iou = [score(p, it)["iou"] for p, it in zip(path3_preds["sam_boxes"], SPEC_EVAL, strict=True)]
rank_box = np.argsort(box_iou)
mid_box = rank_box[len(rank_box) // 2]
box_picks = [("worst", rank_box[0]), ("typical (median)", mid_box), ("best", rank_box[-1])]
show_masks(
    [
        (
            f"{label} · IoU {box_iou[k]:.2f}",
            error_map(SPEC_EVAL[k]["image"], path3_preds["sam_boxes"][k], SPEC_EVAL[k]),
        )
        for label, k in box_picks
    ],
    width=3.0,
)

# %% [markdown]
# **Q28.** 🟡 Compare the box recipe with the clicks of 7.1 and with the thickened strokes: which column
# moved, recall or precision? Then grow the boxes and drop the centre click in the next cell. What does a
# larger box let SAM find, and what does it cost?
#
# ```python
# BOX_PAD_TRY = 16      # try 0, 16, 32 (px of the 1024 px crop)
# WITH_CLICK_TRY = True  # try False: the box alone
# ```

# %% [markdown]
# This cell runs the box recipe with your margin and click setting on every evaluation image and prints it
# under the default.

# %%
# Q28 — larger or smaller boxes, with or without the centre click
BOX_PAD_TRY = 16  # try 0, 16, 32 (px of the 1024 px crop)
WITH_CLICK_TRY = True  # try False: the box alone
if not isinstance(BOX_PAD_TRY, int) or BOX_PAD_TRY < 0 or not isinstance(WITH_CLICK_TRY, bool):
    raise SystemExit("BOX_PAD_TRY must be a whole number ≥ 0 and WITH_CLICK_TRY True or False.")
pad_try = max(BOX_PAD_TRY // 4, 0) if SMOKE else BOX_PAD_TRY  # smoke thumbnails are 4× smaller
try_preds, _ = direct_boxes(SPEC_EVAL, pad_try, WITH_CLICK_TRY)
RESULTS["sam_direct_boxes_try"] = rung_report(try_preds) | {"pad": BOX_PAD_TRY, "with_click": WITH_CLICK_TRY}
print_path2(
    {
        f"pad {BOX_PAD} px, with click (default)": path3_report["sam_boxes"],
        f"pad {BOX_PAD_TRY} px, {'with' if WITH_CLICK_TRY else 'no'} click (yours)": RESULTS[
            "sam_direct_boxes_try"
        ],
    }
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - **Recall is capped.** Each box finds only what lies inside it, so recall is limited by how much of the
#   tree the strokes touch: the vessels a person stroked, plus whatever branches cross their boxes. That is
#   why the box recipe marks a small share of the fundus, unlike the clicks of 7.1.
# - **A larger box** reaches more branches (recall can rise) but also more background that SAM may take for
#   the object (precision falls). At some size it turns back into section 2's big box.
# - **The centre click** tells SAM which thing inside the box is meant. It helps most when the box holds more
#   than one candidate.
# - **The baseline:** if the thickened strokes come close to or beat SAM here, that is a real finding, not a
#   bug. On thin structures a cheap baseline built from the hints alone is the first thing to compare against.
#
# </details>

# %% [markdown]
# ### 7.3 Path 3, side by side
#
# This cell puts the two direct recipes next to the section-2 box, the scribble head and the SAM
# pseudo-label head (bars and error maps on the demo image), and writes the numbers section 8 compares: no
# training time, but annotation on *every* image you segment, at 2 s per stroke.

# %%
# @title Path 3 side by side: table, annotation cost, bars and error maps { display-mode: "form" }
path3_preds["sam_pseudo"] = path2_preds["sam_pseudo"]
path3_report["sam_pseudo"] = path2_report["sam_pseudo"]
eval_meta = SPECIALISED["eval"]
strokes_per_image = {
    "sam_clicks": float(np.mean([m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in eval_meta])),
    "sam_boxes": float(np.mean([m.get("n_fg_strokes", 0) for m in eval_meta])),  # vessel strokes only
}
effort = {k: effort_minutes(n_strokes=round(v * len(eval_meta))) for k, v in strokes_per_image.items()}
sec = {"sam_clicks": clicks_sec, "sam_boxes": boxes_sec}
result_key = {"sam_clicks": "sam_direct_clicks", "sam_boxes": "sam_direct_boxes"}
for k in ("sam_clicks", "sam_boxes"):
    per_image = [score(p, it)["iou"] for p, it in zip(path3_preds[k], SPEC_EVAL, strict=True)]
    RESULTS[result_key[k]] = path3_report[k] | {
        "iou_std": float(np.std(per_image)),
        "train_sec": 0.0,
        "infer_sec": sec[k],
        "n_images": len(SPEC_EVAL),
        "strokes_per_image": strokes_per_image[k],
        "effort_min_per_image": round(effort_minutes(n_strokes=1) * strokes_per_image[k], 2),
        "effort_min": round(effort[k], 2),
    }
RESULTS["sam_direct_boxes"] |= {"pad": BOX_PAD, "with_click": True}
RESULTS["sam_direct_clicks"] |= {"n_clicks": N_DIRECT_CLICKS}
RESULTS["sam_direct_iou"] = path3_report["sam_boxes"]["iou"]  # headline: points + boxes (SPEC §6, 07)
RESULTS["sam_direct_clicks_iou"] = path3_report["sam_clicks"]["iou"]
RESULTS["sam_direct_strokes_alone"] = strokes_report | {"thicken_px": THICKEN}
RESULTS["path3_effort_min"] = {k: round(v, 2) for k, v in effort.items()}

order = ["zero_shot", "scribbles", "sam_pseudo", "sam_clicks", "sam_boxes"]
print_path2({k: path3_report[k] for k in order})
print(
    f"{'no model: strokes thickened':<46}{strokes_report['iou']:>6.3f}\n"
    f"{'everything is foreground':<46}{RESULTS['specialised_everything_iou']:>6.3f}"
)
print(
    f"\nannotation · the heads: {n_strokes} strokes on {len(scrib_train)} training images, once "
    f"(≈ {effort_minutes(n_strokes=n_strokes):.1f} min)"
)
for k in ("sam_clicks", "sam_boxes"):
    print(
        f"annotation · {APPROACH_NAMES[k]}: ≈ {RESULTS[result_key[k]]['effort_min_per_image']:.1f} min on "
        f"every image you segment ({effort[k]:.0f} min for these {len(SPEC_EVAL)})"
    )
metrics = ["iou", "cldice", "recall", "precision"]
x, width = np.arange(len(metrics)), 0.8 / len(order)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for n, k in enumerate(order):
    shift = (n - (len(order) - 1) / 2) * width
    ax.bar(
        x + shift,
        [path3_report[k][m] for m in metrics],
        width,
        color=APPROACH_COLOURS[k],
        label=APPROACH_NAMES[k],
    )
ax.axhline(
    RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--", label='IoU of "everything"'
)
ax.set_xticks(x, ["IoU", "clDice", "recall", "precision"])
ax.set_ylim(0, 1)
ax.set_title("Path 3: SAM prompted directly, next to the trained heads")
ax.legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout()
plt.show()
show_masks(
    [
        (
            f"{APPROACH_NAMES[k].split(':')[-1].strip()} · "
            f"IoU {score(path3_preds[k][k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], path3_preds[k][k_demo], demo),
        )
        for k in order
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.6,
)

# %% [markdown]
# **Q29.** 🟡 Path 3 trains nothing, but its annotation never stops. Suppose you must segment 1,000 new fundus
# images next month. Using the annotation lines above (minutes per image for clicks and for boxes) and the
# table, when would you prompt SAM directly, and when would you rather train a head once? Is there a way to
# combine the two?
#
# <details><summary>A possible answer</summary>
#
# - **Cost:** the heads pay their strokes once, on a few training images, and then segment any number of new
#   images for free. Path 3 pays per image: at 1,000 images it costs a thousand times the per-image line,
#   hours or days of clicking.
# - **Direct prompting wins** when there are few images, when each one matters (a case to review, an image
#   nobody will see again), or when the task changes too often to retrain.
# - **A head wins** when the images keep coming.
# - **Combine them:** prompt SAM on a handful of images, correct its masks by hand, and train the head on
#   those (Path 2, with a person in the loop). Or let the head propose and a person fix only the worst images.
# - **Always** compare with the cheapest baseline built from the same hints, such as the thickened strokes.
#
# </details>

# %% [markdown]
# ### 7.4 🔴 Optional: SAM 3 from a word
#
# SAM 3 segments every instance of a typed noun phrase, with no click at all. Nothing later depends on it.

# %% [markdown]
# <!-- callout: go-deeper | About SAM 3 -->
# SAM 3 (Meta, 2025) adds *concept* prompts: type a short noun phrase and it segments every instance of it.
#
# - It is gated on Hugging Face: accept the terms on its model page with your account to use it.
# - The workshop never requires it. `setup_check.ipynb` caches it only if your account has access.
# - If it is not in your cache, the next cell says so and stops.
# - The cell loads it from your cache only (never from the internet), in fp16 on a GPU, runs your phrase on
#   every evaluation image and keeps the union of the instances it finds.

# %% [markdown]
# <!-- callout: question | Question (optional) -->
# Run the next cell if you have SAM 3, then try other words by editing `SAM3_TEXT` in its code (open the
# collapsed cell).
#
# **Q30.** 🔴 Does a word find the vessels better than your strokes did? Try a few phrasings
# (`"blood vessel"`, `"vessel"`, `"artery"`, `"dark line"`, `"retina"`). Which words work, and what does that
# tell you about the images SAM 3 learnt its concepts from?

# %%
# @title Q30 — 🔴 SAM 3 from a text prompt (optional, only if SAM 3 is in your cache) { display-mode: "form" }
SAM3_TEXT = "blood vessel"  # @param {type:"string"}
SAM3_ID, SAM3_LICENCE = "facebook/sam3", "SAM License (Meta)"
if not isinstance(SAM3_TEXT, str) or not SAM3_TEXT.strip():
    raise SystemExit('SAM3_TEXT must be a short phrase in quotes, e.g. "blood vessel".')
RESULTS["sam3_text_iou"] = None
sam3 = None
try:
    from transformers import Sam3Model, Sam3Processor

    sam3_dtype = torch.float16 if DEVICE == "cuda" else torch.float32  # fp16 on a T4, never bf16 (SPEC §5)
    sam3_src = str(model_folder(SAM3_ID) or SAM3_ID)  # plain files on Drive (setup_check), else the hub id
    sam3_proc = Sam3Processor.from_pretrained(sam3_src, local_files_only=True)
    sam3 = Sam3Model.from_pretrained(sam3_src, local_files_only=True, dtype=sam3_dtype).eval().to(DEVICE)
    print(f"SAM 3 · source {SAM3_ID} · licence {SAM3_LICENCE} · {str(sam3_dtype).split('.')[-1]} on {DEVICE}")
except Exception as err:  # noqa: BLE001 - not cached, no access, an older transformers: all mean "skip"
    RESULTS["sam3_note"] = f"SAM 3 not available in this session ({type(err).__name__}); it is optional"
    print(
        "ℹ️ SAM 3 is not in your cache, so this optional cell stops here. To try it later: request access on "
        f"huggingface.co/{SAM3_ID}, then run setup_check.ipynb with your HF token. Nothing else needs it."
    )
if sam3 is not None:
    try:
        t0, sam3_preds = time.monotonic(), []
        for it in SPEC_EVAL:
            inputs = sam3_proc(images=it["image"], text=SAM3_TEXT.strip(), return_tensors="pt").to(DEVICE)
            inputs["pixel_values"] = inputs["pixel_values"].to(sam3_dtype)
            with torch.no_grad():
                out = sam3(**inputs)
            found = sam3_proc.post_process_instance_segmentation(
                out, threshold=0.3, mask_threshold=0.5, target_sizes=[it["mask"].shape]
            )[0]
            masks = found["masks"].bool().cpu().numpy()
            sam3_preds.append(masks.any(0) if len(masks) else np.zeros(it["mask"].shape, bool))
        sam3_report = rung_report(sam3_preds)
        RESULTS["sam3_text_iou"] = sam3_report["iou"]
        RESULTS["sam3_text"] = sam3_report | {"text": SAM3_TEXT, "infer_sec": round(time.monotonic() - t0, 1)}
        print_path2(
            {
                f'SAM 3, "{SAM3_TEXT}"': sam3_report,
                "SAM 2.1, one box + click per stroke": path3_report["sam_boxes"],
            }
        )
        show_masks(
            [
                (f'SAM 3, "{SAM3_TEXT}" vs truth', error_map(demo["image"], sam3_preds[k_demo], demo)),
                (
                    "SAM 2.1 from stroke boxes vs truth",
                    error_map(demo["image"], path3_preds["sam_boxes"][k_demo], demo),
                ),
            ],
            width=3.2,
        )
    except Exception as err:  # noqa: BLE001 - an optional extension must never stop the notebook
        RESULTS["sam3_note"] = f"SAM 3 loaded but failed: {type(err).__name__}: {err}"
        print(f"⚠️ SAM 3 loaded but did not run here ({type(err).__name__}: {err}). It is optional; carry on.")

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - **Only known concepts work.** A concept prompt works for concepts SAM 3 met in training, named the way
#   its captions named them. "Blood vessel" in a fundus photograph is a rare, specialist phrase next to "dog"
#   or "car", so expect thick vessels, a blob, or nothing, and a lot of change with the wording.
# - **Appearance beats meaning.** Generic words like "dark line" sometimes do better than the correct medical
#   term, which hints that the model matches appearance more than meaning.
# - **It is free.** A word costs no annotation, so even a weak result is a free starting point for the
#   strokes of 7.2.
#
# </details>

# %% [markdown]
# Want to try this on a photo of your own? Section 9 lets you draw the strokes yourself; the optional
# collapsed cell below is the quick version (details in the box just below).

# %% [markdown]
# <!-- callout: go-deeper | Try it on your image (optional, 2 min) -->
# Direct prompting needs no training, so it is the quickest thing to try on a photo of your own. Open the
# collapsed cell below and edit its first lines.
#
# - Upload one with the folder icon on the left and put its path in `MY_IMAGE`.
# - Add `(x, y)` clicks on the thing (`MY_FG_CLICKS`) and on what you do not want (`MY_BG_CLICKS`).
# - Add small boxes `(x0, y0, x1, y1)`, one per piece of the thing (`MY_BOXES`).
# - Left empty, it runs on the demo image with prompts from its strokes. It asks SAM for one mask from all
#   your clicks together and one from your boxes, each box its own object.

# %%
# @title Try it on your image (optional): clicks and boxes on a photo { display-mode: "form" }
MY_IMAGE = ""  # e.g. "/content/my_photo.jpg"; empty = the demo image
MY_FG_CLICKS = []  # e.g. [(120, 80), (300, 210)] on the thing you want
MY_BG_CLICKS = []  # e.g. [(20, 20)] on what you do not want
MY_BOXES = []  # e.g. [(100, 60, 160, 120)], one small box per piece of the thing
lists = (MY_FG_CLICKS, MY_BG_CLICKS, MY_BOXES)
if not isinstance(MY_IMAGE, str) or not all(isinstance(v, list) for v in lists):
    raise SystemExit('MY_IMAGE must be a path in quotes (or "") and the clicks and boxes lists.')
if MY_IMAGE and not Path(MY_IMAGE).is_file():
    raise SystemExit(f"{MY_IMAGE} not found. Upload it with the folder icon on the left and copy its path.")
if MY_IMAGE and not (MY_FG_CLICKS or MY_BOXES):
    raise SystemExit("Add at least one (x, y) click to MY_FG_CLICKS or one (x0, y0, x1, y1) box to MY_BOXES.")
my_img = np.array(Image.open(MY_IMAGE).convert("RGB")) if MY_IMAGE else demo["image"]
my_h, my_w = my_img.shape[:2]
if MY_IMAGE:
    my_fg = [[float(x), float(y)] for x, y in MY_FG_CLICKS]
    my_bg = [[float(x), float(y)] for x, y in MY_BG_CLICKS]
    my_boxes = [[float(v) for v in b] for b in MY_BOXES]
else:
    my_fg, my_bg = stroke_points(demo, 1, 5), stroke_points(demo, 2, 5)
    my_boxes = box_prompts[k_demo][0]
if not all(0 <= x < my_w and 0 <= y < my_h for x, y in my_fg + my_bg):
    raise SystemExit(f"every click must lie inside the image: 0 ≤ x < {my_w}, 0 ≤ y < {my_h}.")
if not all(len(b) == 4 and 0 <= b[0] < b[2] < my_w and 0 <= b[1] < b[3] < my_h for b in my_boxes):
    raise SystemExit(
        f"every box must be (x0, y0, x1, y1) with 0 ≤ x0 < x1 < {my_w} and 0 ≤ y0 < y1 < {my_h}."
    )
panels = []
if my_fg:
    clicks_mask = (
        sam.probability(my_img, points=my_fg + my_bg, labels=[1] * len(my_fg) + [0] * len(my_bg)) >= 0.5
    )
    panels.append(
        (f"clicks: {len(my_fg)} on, {len(my_bg)} off", overlay(my_img, clicks_mask, THEME["fails"]), None)
    )
if my_boxes:
    boxes_mask = sam_many(my_img, my_boxes) >= 0.5
    panels.append(
        (f"{len(my_boxes)} boxes, one object each", overlay(my_img, boxes_mask, THEME["fails"]), my_boxes)
    )
show_boxes([(MY_IMAGE or f"demo · {demo['name']}", my_img, None)] + panels, width=3.0)

# %% [markdown]
# <!-- callout: takeaway | Takeaway: Path 3 -->
# - **Your structure is thin and spread over the image** → more clicks only make a bigger region: split the
#   prompt into one small box and a centre click per stroke.
# - **Few images, or each one matters** → prompt SAM directly and review every mask.
# - **Images keep coming** → train a head once (Path 1 or 2); direct prompting pays strokes on every image.
# - **SAM gives a clean-looking mask** → it looks finished whether or not it is right, so compare it with the
#   thickened strokes and with a careful manual mask on a few images.

# %% [markdown]
# <!-- callout: across | Across approaches -->
# Paths 1 and 2 pay annotation once and reuse a head; Path 3 trains nothing but pays on every image, and how
# you split the prompt mattered as much as how many hints you gave. Section 8 compares them all.
