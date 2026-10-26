# %% [markdown]
# ## 7. Path 3 · Skip specialist training
#
# Paths 1 and 2 trained a head once from strokes on five images, and that head then segmented every new image
# on its own. Path 3 trains nothing: the strokes a person draws on an image become SAM's prompts on *that
# same image*, and SAM's answer is the result.
#
# It is the difference between teaching an intern once and pointing at every image yourself: no training time
# and no training data, but the pointing never stops.

# %% [markdown]
# <!-- callout: objective | Objective -->
# Prompt the foundation model directly with scribble points and boxes, no training, and know when that beats
# training a head.


# %% [markdown]
# <!-- diagram: s00-map-path3 | You are here: Path 3 -->
# Where this section sits on the map of section 0.

# %% [markdown]
# <!-- diagram: s07-prompt-every-image | Train once, or prompt every image -->
# Paths 1 and 2 pay for strokes once and then run alone. Path 3 pays for strokes on every image.

# %% [markdown]
# <!-- callout: theory | Theory: what a SAM prompt says -->
# Section 2.3 showed that many scattered clicks make SAM answer with one big region. Two things are new here:
#
# - The prompt is split into many small ones, one box per stroke, each its own object. SAM answers many local
#   questions instead of one global one, and the masks are combined.
# - Nothing is trained. The image encoder runs once per image, and every extra prompt costs only a fast
#   decoder pass.

# %% [markdown]
# <!-- callout: go-deeper | SAM papers -->
# - SAM, Kirillov et al., ICCV 2023: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
# - SAM 2, Ravi et al. 2024: [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
# - SAM 3 (concept prompts, used in 7.4): Meta, 2025.

# %% [markdown]
# This helper cell defines the stroke prompts, SAM on many boxes, the two recipes and a box-drawing figure.

# %%
# @title Path 3 helpers: colours, strokes, SAM on many prompts, recipes, printout { display-mode: "form" }
from matplotlib.patches import Rectangle

approach_colours |= {
    "sam_clicks": "#546E7A",
    "sam_boxes": "#880E4F",
    "sam3_text": "#5D4037",
}
approach_names |= {
    "sam_clicks": "direct: SAM from stroke clicks",
    "sam_boxes": "direct: SAM, one box + click per stroke",
    "sam3_text": "direct: SAM 3 from a word",
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
    for it in tqdm(items, desc="SAM, stroke clicks", leave=False):
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
    for it in tqdm(items, desc="SAM, one box per stroke", leave=False):
        strokes = stroke_masks(it, 1)
        boxes = [[float(v) for v in stroke_box(s, pad)] for s in strokes]
        clicks = [stroke_click(s) for s in strokes] if with_click else None
        preds.append(
            sam_many(it["image"], boxes, clicks) >= 0.5 if boxes else np.zeros(it["mask"].shape, bool)
        )
        prompts.append((boxes, clicks))
    return preds, prompts


def show_boxes(
    panels: list[tuple], width: float = 3.2, legend: list | None = None, note: str | None = None,
    roles: list | None = None,
):
    """One row of panels (title, RGB image, boxes or None[, clicks]); boxes and clicks are drawn in teal.
    `legend` and `note` are the colour key of add_key() under the figure; `roles` one corner tag per panel."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    for ax, (title, img, boxes, *extra), role in zip(
        np.atleast_1d(axes), panels, roles or [None] * len(panels), strict=True
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        tag_panel(ax, role)
        for x0, y0, x1, y1 in boxes or []:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1, color=THEME["scribble"]))
        for x, y in extra[0] if extra and extra[0] else []:
            ax.plot(x, y, marker="*", markersize=7, color=THEME["scribble"], markeredgecolor="white")
    add_key(fig, legend, note)
    plt.show()


def path3_table(
    reports: dict, title: str, meta: str | None = None, refs: list | None = None, extra: dict | None = None
) -> None:
    """The standard table of this path on the held-out images: IoU, recall and "calls vessel" per approach,
    then the reference rows (`refs`, then the ground truth). `extra` = {column: {approach: value}}."""
    extra = extra or {}
    rows = [
        {
            "name": approach_names.get(k, k),
            "colour": approach_colours.get(k),
            **v,
            **{c: vals.get(k) for c, vals in extra.items()},
        }
        for k, v in reports.items()
    ]
    columns = ["iou", "recall", "marked"] + [
        ("annotation", "annotation", "what a person draws", "text")
    ] * bool("annotation" in extra)
    truth = {"name": "really vessel (the ground truth)", "marked": fg_share, "truth": True}
    show_table(title, rows, columns, refs=[*(refs or []), truth], meta=meta)


def try_boxes(box_pad: int, with_click: bool) -> None:
    """Run the box recipe with a margin of `box_pad` px (of the 1024 px crop) and the centre click on or off
    on every evaluation image and print it under the default. Stores the numbers in
    RESULTS["sam_direct_boxes_try"]."""
    if not isinstance(box_pad, int) or box_pad < 0 or not isinstance(with_click, bool):
        raise SystemExit("BOX_PAD_TRY must be a whole number ≥ 0 and WITH_CLICK_TRY True or False.")
    pad = max(box_pad // 4, 0) if SMOKE else box_pad  # smoke thumbnails are 4× smaller
    preds, _ = direct_boxes(spec_eval, pad, with_click)
    RESULTS["sam_direct_boxes_try"] = rung_report(preds) | {"pad": box_pad, "with_click": with_click}
    path3_table(
        {
            f"pad {BOX_PAD} px, with click (default)": path3_report["sam_boxes"],
            f"pad {box_pad} px, {'with' if with_click else 'no'} click (yours)": RESULTS[
                "sam_direct_boxes_try"
            ],
        },
        "Your box setting next to the default",
    )


# %% [markdown]
# ### 7.1 Clicks on the vessels, and "not here"
#
# The news is **fairness**: the heads of sections 4 to 6 never saw a stroke on the held-out images, but Path 3
# gets strokes on every one of them. It gets more help per image, and pays for it at use time.
#
# The first recipe is section 6.1 without the head: twenty clicks along the vessel strokes and twenty along
# the background strokes of each *held-out* image, and SAM's mask is the answer. The next cell prints that
# one row. *Marked* is the share of the fundus (the round, scored part of the photo) the mask covers.

# %%
# @title SAM from twenty vessel and twenty background clicks per image { display-mode: "form" }
N_DIRECT_CLICKS = 20  # clicks per kind (vessel / background) and image, taken from the strokes
t0 = time.monotonic()
path3_preds = {"zero_shot": ladder_preds["zero_shot"], "scribbles": ladder_preds["scribbles"]}
path3_preds["sam_clicks"], click_prompts = direct_clicks(spec_eval, N_DIRECT_CLICKS)
clicks_sec = round(time.monotonic() - t0, 1)
path3_report = {k: rung_report(v) for k, v in path3_preds.items()}
path3_table(
    {"sam_clicks": path3_report["sam_clicks"]},
    f"SAM from {N_DIRECT_CLICKS} vessel + {N_DIRECT_CLICKS} background clicks per image",
    meta=f"{len(spec_eval)} held-out images · {clicks_sec} s on {DEVICE}",
    refs=[
        {"name": "section 2: SAM, 20 vessel clicks only", "iou": RESULTS["sam_prompt_ladder"]["20 points"]},
        {"name": "everything is foreground", "iou": RESULTS["specialised_everything_iou"], "marked": 1.0},
    ],
)

# %% [markdown]
# **What to look for.** Read *calls vessel* first. A value near 100 % means the "not here" clicks only cut a
# few
# holes in one big region (section 2.3), and the IoU stays near the "everything" line.

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
# This helper cell defines `stroke_box`: the box around one stroke, grown by a margin.


# %%
# @title stroke_box: the box around one stroke { display-mode: "form" }
def stroke_box(stroke: np.ndarray, pad: int) -> list[float]:
    """The box [x0, y0, x1, y1] around the True pixels of `stroke`, grown by `pad` pixels on every side and
    clipped to the image: 0 <= x <= width - 1, 0 <= y <= height - 1. np.nonzero returns rows (y) first."""
    ys, xs = np.nonzero(stroke)
    h, w = stroke.shape
    return [
        float(max(xs.min() - pad, 0)),
        float(max(ys.min() - pad, 0)),
        float(min(xs.max() + pad, w - 1)),
        float(min(ys.max() + pad, h - 1)),
    ]


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The strokes cover only some of the vessels (section 4 dropped 70 % of the segments), and each box sits on a
# stroke. Two guesses:
#
# - Will this recipe mark more of the fundus than the clicks of 7.1, or less?
# - Will it beat simply thickening the strokes by a few pixels, with no model at all?

# %% [markdown]
# This cell asks SAM for one object per vessel stroke (a box with 4 px of margin plus a centre click) on every
# evaluation image, and prints it next to the strokes thickened by 3 px, the "no model" reference. The figure
# shows the demo image: SAM's input on the left, its answer in the middle, the thickened strokes on the right.

# %%
# @title SAM with one box and one click per stroke, next to the thickened strokes { display-mode: "form" }
BOX_PAD = 1 if SMOKE else 4  # px around each stroke (smoke thumbnails are 4× smaller)
THICKEN = 1 if SMOKE else 3  # px: the "no model" reference, the vessel strokes thickened
t0 = time.monotonic()
path3_preds["sam_boxes"], box_prompts = direct_boxes(spec_eval, BOX_PAD, with_click=True)
boxes_sec = round(time.monotonic() - t0, 1)
path3_report["sam_boxes"] = rung_report(path3_preds["sam_boxes"])
strokes_alone = [ndi.binary_dilation(it["scribbles"] == 1, iterations=THICKEN) for it in spec_eval]
strokes_report = rung_report(strokes_alone)
n_boxes = [len(b) for b, _ in box_prompts]
path3_table(
    {k: path3_report[k] for k in ("sam_clicks", "sam_boxes")},
    "SAM with one box and one centre click per vessel stroke",
    meta=f"{int(np.mean(n_boxes))} boxes per image on average · {boxes_sec} s on {DEVICE}",
    refs=[{"name": f"no model: the strokes thickened by {THICKEN} px", **strokes_report}],
)
boxes, centre_clicks = box_prompts[k_demo]
box_nums = image_numbers(path3_preds["sam_boxes"][k_demo], demo)
thick_nums = image_numbers(strokes_alone[k_demo], demo)
show_boxes(
    [
        (f"SAM's input: {len(boxes)} boxes\nand a centre click in each", demo["image"], boxes, centre_clicks),
        (
            f"SAM, one object per box\nIoU {box_nums['iou']:.2f} · calls {box_nums['marked']:.0%} vessel",
            error_map(demo["image"], path3_preds["sam_boxes"][k_demo], demo),
            None,
        ),
        (
            f"baseline: the strokes themselves, no SAM\nIoU {thick_nums['iou']:.2f} · "
            f"calls {thick_nums['marked']:.0%} vessel",
            error_map(demo["image"], strokes_alone[k_demo], demo),
            None,
        ),
    ],
    width=3.0,
    roles=["input", "errors", "errors"],
    legend=["box", "click", "found", "invented", "missed"],
    note="Left is what SAM is given; the other two panels compare with the true vessels.",
)

# %% [markdown]
# **What to look for.** $\color{#2E7D32}{\blacksquare}$ Green is missed vessels. Each box finds only what
# lies inside it, so expect a lot of
# $\color{#2E7D32}{\blacksquare}$ green between the boxes. Compare the middle and right panels: is SAM's
# answer better than the plain
# thickened strokes?

# %% [markdown]
# **Q17.** 🟡 Compare the box recipe with the clicks of 7.1 and with the thickened strokes: which column
# moved, recall or "calls vessel"? Then grow the boxes (try 0, 16 and 32 px of the 1024 px crop) and untick
# the
# centre click in the next cell. What does a larger box let SAM find, and what does it cost?

# %% [markdown]
# This cell reruns the box recipe with your margin and click setting and prints it under the default.

# %%
# @title Q17 — larger or smaller boxes, with or without the centre click { display-mode: "form" }
BOX_PAD_TRY = 16  # @param {type:"slider", min:0, max:48, step:8}
WITH_CLICK_TRY = True  # @param {type:"boolean"}
try_boxes(BOX_PAD_TRY, WITH_CLICK_TRY)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - **Recall is capped.** Each box finds only what lies inside it: the vessels a person stroked, plus the
#   branches that cross their boxes. A larger box reaches more branches (recall rises) but also more
#   background (*calls vessel* rises, and part of it is invented). At some size it becomes section 2's big
#   box.
# - **The centre click** tells SAM which thing inside the box is meant. It helps most when the box holds more
#   than one candidate.
# - **The baseline:** if the thickened strokes come close to or beat SAM, that is a real finding, not a bug.
#   On thin structures, compare against the cheapest baseline built from the hints alone.
#
# </details>

# %% [markdown]
# ### 7.3 Path 3, side by side
#
# This cell puts the two direct recipes next to the section-2 box, the scribble head and the SAM
# pseudo-label head, and writes the numbers section 8 compares. The figure shows the demo image under each
# approach: no training time, but annotation on *every* image you segment, at 2 s per stroke.

# %%
# @title Path 3 side by side: table, annotation cost and error maps { display-mode: "form" }
path3_preds["sam_pseudo"] = path2_preds["sam_pseudo"]
path3_report["sam_pseudo"] = path2_report["sam_pseudo"]
eval_meta = specialised["eval"]
strokes_per_image = {
    "sam_clicks": float(np.mean([m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in eval_meta])),
    "sam_boxes": float(np.mean([m.get("n_fg_strokes", 0) for m in eval_meta])),  # vessel strokes only
}
effort = {k: effort_minutes(n_strokes=round(v * len(eval_meta))) for k, v in strokes_per_image.items()}
sec = {"sam_clicks": clicks_sec, "sam_boxes": boxes_sec}
result_key = {"sam_clicks": "sam_direct_clicks", "sam_boxes": "sam_direct_boxes"}
for k in ("sam_clicks", "sam_boxes"):
    per_image = [score(p, it)["iou"] for p, it in zip(path3_preds[k], spec_eval, strict=True)]
    RESULTS[result_key[k]] = path3_report[k] | {
        "iou_std": float(np.std(per_image)),
        "train_sec": 0.0,
        "infer_sec": sec[k],
        "n_images": len(spec_eval),
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
once = f"{n_strokes} strokes once (≈ {effort_minutes(n_strokes=n_strokes):.0f} min)"
annotation = {"zero_shot": "1 box per image", "scribbles": once, "sam_pseudo": once} | {
    k: f"≈ {RESULTS[result_key[k]]['effort_min_per_image']:.1f} min on every image"
    for k in ("sam_clicks", "sam_boxes")
}
path3_table(
    {k: path3_report[k] for k in order},
    "Path 3 next to the trained heads",
    meta=f"{len(spec_eval)} held-out images",
    refs=[
        {"name": "no model: the strokes thickened", **strokes_report},
        {"name": "everything is foreground", "iou": RESULTS["specialised_everything_iou"], "marked": 1.0},
    ],
    extra={"annotation": annotation},
)
short3 = {
    "zero_shot": "zero-shot\n(one box)",
    "scribbles": "scribble head\n(trained once)",
    "sam_pseudo": "SAM pseudo-label\nhead (trained once)",
    "sam_clicks": "SAM from\nstroke clicks",
    "sam_boxes": "SAM, one box\nper stroke",
}
show_masks(
    [
        (
            f"{short3[k]}\nIoU {score(path3_preds[k][k_demo], demo)['iou']:.2f}",
            error_map(demo["image"], path3_preds[k][k_demo], demo),
        )
        for k in order
    ]
    + [("ground truth\n(the true vessels)", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=2.6,
    roles=["errors"] * len(order) + ["truth"],
    legend=["found", "invented", ("missed", "missed (truth only); the last panel shows every true vessel")],
    note="Same demo image under every approach; the last panel is the answer key.",
)

# %% [markdown]
# **What to look for.** The two trained heads (second and third panel) are not prompted on this image at use
# time. The two SAM recipes are, with strokes drawn on it. Look at $\color{#E65100}{\blacksquare}$ orange
# (invented) and $\color{#2E7D32}{\blacksquare}$ green (missed):
# which approach mostly invents, which mostly misses?

# %% [markdown]
# ### 7.4 SAM 3: a word instead of a stroke (needs SAM 3 access)
#
# SAM 3 (Meta, 2025) takes a short noun phrase such as "retinal vessels" and segments every instance of it,
# with no click at all. It is gated on Hugging Face, so it works only if your account has access and
# `setup_check.ipynb` cached it. If not, the next cell says so in one line and you can skip to the takeaway;
# nothing later depends on it.

# %% [markdown]
# <!-- callout: question | Question -->
# **Q18.** 🟡 Run the next cell, type "retinal vessels" and press "Ask SAM 3" on a few retina images. Then
# try other words ("blood vessel", "vessel", "dark line", "optic disc"). Which words find the vessels? Does
# the best word beat the stroke boxes of 7.2, which cost you strokes on every image? What does the answer
# tell you about the images SAM 3 learnt its words from?

# %% [markdown]
# This cell shows whether SAM 3 is available, then opens a playground on the retina images: type a word and
# press "Ask SAM 3", or click and drag to prompt SAM 2.1 instead. Tick "Show ground truth" to colour it.

# %%
# @title Q18 — Ask SAM 3 with a word on the retina (needs SAM 3 access) { display-mode: "form" }
s3 = load_sam3()  # from 00: loads SAM 3 once from your cache, never from the internet
if "error" in s3:
    RESULTS["sam3_note"] = "SAM 3 not available in this session"
    print(f"ℹ️ {s3['error']}")
    print("The playground below still takes clicks and boxes (SAM 2.1). Nothing else needs SAM 3.")
else:
    print(f"SAM 3 ready · {SAM3_ID} · SAM License (Meta) · {str(s3['dtype']).split('.')[-1]} on {DEVICE}")
sam_playground(
    spec_eval,
    "Retina: ask SAM 3 with a word, or click",
    text_prompt=True,
    default_text="retinal vessels",
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - **Only known concepts work.** A concept prompt works for words SAM 3 met in training, named the way its
#   captions named them. "Vessel" in a fundus photograph is a rare, specialist phrase next to "dog" or "car",
#   so expect thick vessels, a blob, or nothing, and a lot of change with the wording.
# - **Appearance can beat meaning.** A generic word like "dark line" sometimes does better than the correct
#   medical term.
# - **It is free.** A word costs no annotation, so even a weak answer is a free starting point for the strokes
#   of 7.2.
#
# </details>

# %% [markdown]
# <!-- callout: go-deeper | Optional: score a word on all evaluation images -->
# The playground shows one image at a time. The next cell scores one phrase on every evaluation image, with
# the same IoU, recall and "calls vessel" columns as above, and stores it in `RESULTS["sam3_text"]` so
# section 8 can add an optional row. It needs SAM 3 access and otherwise prints one line and stops.

# %%
# @title Optional: score one SAM 3 phrase on all images (needs SAM 3 access) { display-mode: "form" }
SAM3_TEXT = "retinal vessels"  # @param ["retinal vessels", "blood vessel", "dark line"] {allow-input: true}
if not isinstance(SAM3_TEXT, str) or not SAM3_TEXT.strip():
    raise SystemExit('SAM3_TEXT must be a short phrase in quotes, e.g. "retinal vessels".')
RESULTS["sam3_text_iou"] = None
if "error" in s3:
    print("ℹ️ SAM 3 is not available in this session, so this optional cell stops here.")
else:
    try:
        t0, sam3_preds = time.monotonic(), []
        for it in tqdm(spec_eval, desc="SAM 3, one phrase", leave=False):
            found = sam3_masks(it["image"], SAM3_TEXT)
            sam3_preds.append(found.any(0) if len(found) else np.zeros(it["mask"].shape, bool))
        sam3_report = rung_report(sam3_preds)
        RESULTS["sam3_text_iou"] = sam3_report["iou"]
        RESULTS["sam3_text"] = sam3_report | {"text": SAM3_TEXT, "infer_sec": round(time.monotonic() - t0, 1)}
        path3_table(
            {
                f'SAM 3, "{SAM3_TEXT}"': sam3_report,
                "SAM 2.1, one box + click per stroke": path3_report["sam_boxes"],
            },
            "SAM 3 from one phrase, next to SAM 2.1 from the strokes",
            meta=f"{len(spec_eval)} held-out images · {RESULTS['sam3_text']['infer_sec']} s",
        )
    except Exception as err:  # noqa: BLE001 - an optional extension must never stop the notebook
        RESULTS["sam3_note"] = f"SAM 3 loaded but failed: {type(err).__name__}: {err}"
        print(f"⚠️ SAM 3 loaded but did not run here ({type(err).__name__}: {err}). It is optional; carry on.")

# %% [markdown]
# <!-- diagram: s07-three-prompts | Three ways to prompt SAM -->
# What SAM is given in each recipe of this section, and how each one usually goes wrong.

# %% [markdown]
# Want to try this on a photo of your own? Section 9 lets you draw the strokes yourself.

# %% [markdown]
# <!-- callout: takeaway | Takeaway: Path 3 -->
# - **Your structure is thin and spread over the image** → more clicks only make a bigger region: split the
#   prompt into one small box and a centre click per stroke.
# - **Few images, or each one matters** → prompt SAM directly and review every mask.
# - **Images keep coming** → train a head once (Path 1 or 2); direct prompting pays strokes on every image.
# - **A word is free but narrow** → SAM 3 only knows the concepts it was trained on; check it against the
#   strokes.
