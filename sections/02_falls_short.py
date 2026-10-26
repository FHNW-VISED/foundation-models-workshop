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
# This helper cell defines the scores, clicks and panel titles used in this section.

# %%
# @title Thin-structure helpers: skeleton, score, clicks, figures { display-mode: "form" }
import hashlib

from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_SKEL_CACHE: dict = {}  # prediction skeletons by mask content: {(shape, hash): flat pixel indices}
_SKEL_CACHE_MAX = 300


def skeleton(mask: np.ndarray) -> np.ndarray:
    """The one-pixel-wide centre line of a boolean mask."""
    return skeletonize(np.asarray(mask, bool))


def blob_skeleton(mask: np.ndarray) -> np.ndarray:
    """Centre line of a mask that fills most of the image, found at half resolution. Thinning a near-solid
    1024 px mask takes ~0.8 s and its centre line is a short stub either way; at half size it takes ~0.1 s and
    gave the same clDice on our retina crops."""
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(skeleton(mask[::2, ::2]))
    out[ys * 2, xs * 2] = True
    return out & mask


def cached_skeleton(mask: np.ndarray) -> np.ndarray:
    """skeleton(mask), but a mask seen before is not skeletonised again, and a near-solid one goes faster."""
    mask = np.asarray(mask, bool)
    key = (mask.shape, hashlib.blake2b(np.packbits(mask), digest_size=16).digest())
    if key not in _SKEL_CACHE:
        if len(_SKEL_CACHE) >= _SKEL_CACHE_MAX:
            _SKEL_CACHE.pop(next(iter(_SKEL_CACHE)))  # drop the oldest
        sk = blob_skeleton(mask) if mask.mean() > 0.6 else skeleton(mask)
        _SKEL_CACHE[key] = np.flatnonzero(sk)
    out = np.zeros(mask.shape, bool)
    out.flat[_SKEL_CACHE[key]] = True
    return out


def truth_skeleton(item: dict) -> np.ndarray:
    """The true centre line inside the evaluation region, computed once per item and kept on it."""
    if "_skel" not in item:
        item["_skel"] = skeleton(item["mask"] & region(item))
    return item["_skel"]


def score(pred: np.ndarray, item: dict) -> dict:
    """IoU and clDice of a boolean prediction, both inside the evaluation region."""
    r = region(item)
    p, t = pred & r, item["mask"] & r
    return {"iou": iou(p, t), "cldice": cldice(p, t, cached_skeleton(p), truth_skeleton(item))}


def everything_score(item: dict) -> dict:
    """Score of "everything in the evaluation region is foreground", computed once per item."""
    if "_everything" not in item:
        item["_everything"] = score(region(item).copy(), item)
    return item["_everything"]


def mean_scores(scores: list[dict]) -> dict:
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}


def marked_pct(pred: np.ndarray, item: dict) -> str:
    """How much of the evaluation region a prediction marks, e.g. "marked 98 %"."""
    return f"marked {100 * pred[region(item)].mean():.0f} %"


def recall_of(pred: np.ndarray, item: dict) -> float:
    """Recall: the share of the true vessel pixels (inside the evaluation region) that the mask covers."""
    truth = item["mask"] & region(item)
    return float((pred & truth).sum() / truth.sum()) if truth.any() else 1.0


MARKED_NOTE = (
    "marked = share of the fundus (the photo inside the black ring) that the mask covers · "
    "recall = share of the true vessel pixels it finds"
)


def pred_title(label: str, pred: np.ndarray, item: dict) -> str:
    """Panel title on three short lines: what made the mask, IoU and recall, then the share marked."""
    r = region(item)
    return (
        f"{label}\nIoU {iou(pred & r, item['mask'] & r):.2f} · recall {recall_of(pred, item):.2f}\n"
        f"{marked_pct(pred, item)}"
    )


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


def show_clicks(
    panels: list[tuple[str, np.ndarray, list | None]],
    width: float = 2.6,
    legend: list | None = None,
    note: str | None = None,
):
    """One row of panels (title, RGB image, clicks or None[, box]); stars = clicks, frame = box.
    `legend` and `note` are the colour key of add_key() under the figure."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width))
    for ax, (title, img, pts, *extra) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if extra and extra[0] is not None:  # an optional box (x0, y0, x1, y1)
            bx0, by0, bx1, by1 = extra[0]
            ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=2,
                                       ec=THEME["scribble"]))
        for x, y in pts or []:
            ax.plot(x, y, marker="*", markersize=10, color=THEME["scribble"], markeredgecolor="white")
    add_key(fig, legend, note)
    plt.show()


def model_key(name: str) -> tuple:
    """Legend entry for a model's plain mask, drawn in violet like everywhere else in this notebook."""
    return (name, THEME["works"])


# %% [markdown]
# This cell loads the held-out retina images (1024 px crops of a 3504 × 2336 fundus), the evaluation set for
# the rest of the notebook. Keep the printed share of the fundus the vessels cover; it comes back as a
# baseline.

# %%
# @title Load the held-out retina images { display-mode: "form" }
specialised["feat_side"] = 256 if SMOKE else specialised["feature_long_side"]
specialised["eval"] = specialised["held"] if (SMOKE or DEVICE == "cuda") else specialised["held"][:12]
spec_eval = [load_item(specialised, it) for it in specialised["eval"]]
demo = spec_eval[0]  # the image the figures of sections 2 to 8 show
fg_share = float(np.mean([it["mask"][region(it)].mean() for it in spec_eval]))
print(
    f"evaluation set: {len(spec_eval)} held-out images · {specialised['task']} · the foreground covers "
    f"{fg_share:.0%} of the evaluation region · DINOv3 features at {specialised['feat_side']} px long side"
)

# %% [markdown]
# This cell shows one retina photo, its vessels (green) and a zoom with the 16 px grid DINOv3 sees (white
# squares), and measures a vessel's width against one patch. What to look for: in the zoom, a vessel is
# thinner than one white square, so no square can be "all vessel".

# %%
# @title Meet the retina images: vessels next to the DINOv3 patch grid { display-mode: "form" }
ph, pw = demo["mask"].shape
patch_px = 16 * max(ph, pw) / specialised["feat_side"]  # one DINOv3 patch, in image pixels
centre_line = skeleton(demo["mask"] & region(demo))
vessel_px = float(demo["mask"].sum() / max(centre_line.sum(), 1))  # area ÷ centre-line length = mean width
win = int(min(8 * patch_px, ph, pw))  # the zoom window: about 8 × 8 patches
y0, x0 = (ph - win) // 2, (pw - win) // 2
zoom = (slice(y0, y0 + win), slice(x0, x0 + win))
truth_img = overlay(demo["image"], demo["mask"], THEME["truth"])

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
axes[0].imshow(demo["image"])
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
add_key(fig, [("vessels (ground truth)", THEME["truth"]), ("16 px patch grid", "#FFFFFF")])
plt.show()
print(
    f"{demo['name']}: vessels are about {vessel_px:.1f} px wide on average, one patch is {patch_px:.0f} px, "
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
# ### 2.1 SAM on vessels: click, box, or a word

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Before you open the playground, guess. What will SAM mark from one click on a thin vessel? And from a box
# that covers all the vessels, which on a fundus is nearly the whole photo?

# %% [markdown]
# **Try it yourself.** Section 1's SAM could mark a pet. Now see whether you can make it mark only the
# vessels. What to look for: the violet mask is what SAM returned. Press "Show ground truth" to colour what
# it found (violet), invented (orange) and missed (green); the key under the picture names the colours.
#
# **Q5.** 🟡 Can you get SAM to mark only the vessels? Try these four and note after each what SAM marked: a
# vessel, the whole tree, a blob, or the whole fundus.
#
# 1. One click on a vessel.
# 2. Several clicks along one vessel.
# 3. A small box around a few vessels.
# 4. A box around the whole fundus.
#
# Then compare with the pets of section 1. If SAM 3 is in your cache, type `retinal vessels` in the text box
# and compare its mask with your clicks.

# %%
# @title Q5 — Try it: make SAM mark only the vessels { display-mode: "form" }
sam_playground(spec_eval, "Retina: click, box, or a word", text_prompt=True, default_text="retinal vessels")

# %% [markdown]
# ### 2.2 A second number for thin structures: clDice

# %% [markdown]
# <!-- callout: theory | Theory: clDice -->
# IoU counts pixels, and thick vessels hold most of them. So a prediction that draws every vessel twice as
# wide loses a lot of IoU although nothing is missed, and one that misses all the capillaries loses little
# IoU although a large part of the tree's length is gone.
#
# **clDice** compares *centre lines* (each vessel thinned to a one-pixel line) instead:
#
# 1. `tprec` = the part of the prediction's centre line that lies inside the true vessels. Invented
#    branches lower it.
# 2. `tsens` = the part of the true centre line that lies inside the prediction. Missed branches lower it.
# 3. clDice = 2 · tprec · tsens ÷ (tprec + tsens). Example: tprec 1.0 and tsens 0.25 give 0.40.
#
# The workshop is not about vessels, so this is the only place clDice gets this much room. The figure below
# shows all of it on a made-up vessel tree.

# %% [markdown]
# This collapsed cell is clDice as code (the three steps above); you do not need to read it.

# %%
# @title Reference: clDice (tprec, tsens, harmonic mean) { display-mode: "form" }
def cldice_parts(pred: np.ndarray, truth: np.ndarray, sk_pred=None, sk_truth=None) -> tuple:
    """tprec, tsens and the two centre lines of two boolean masks (skeletons are computed when not given)."""
    sk_pred = skeleton(pred) if sk_pred is None else sk_pred
    sk_truth = skeleton(truth) if sk_truth is None else sk_truth
    if not sk_pred.any() or not sk_truth.any():
        return 0.0, 0.0, sk_pred, sk_truth
    tprec = float((sk_pred & truth).sum() / sk_pred.sum())
    tsens = float((sk_truth & pred).sum() / sk_truth.sum())
    return tprec, tsens, sk_pred, sk_truth


def cldice(pred: np.ndarray, truth: np.ndarray, sk_pred=None, sk_truth=None) -> float:
    """clDice: harmonic mean of topology precision and topology sensitivity (0.0 when a skeleton is empty)."""
    tprec, tsens, _, _ = cldice_parts(pred, truth, sk_pred, sk_truth)
    return 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0


# %% [markdown]
# This collapsed cell builds the made-up vessel tree and the clDice figure.

# %%
# @title Helper: the made-up vessel tree and the clDice figure { display-mode: "form" }
TOY_CASES = ["too wide", "capillaries missed", "blob"]


def vessel_toy(scale: float = 1.0, min_radius: int = 0, size: int = 128) -> np.ndarray:
    """A made-up vessel tree: a thick trunk, two branches, six capillaries. `scale` widens every vessel;
    `min_radius` leaves out the vessels thinner than that (2 drops the capillaries)."""
    from skimage.draw import line

    tree = [  # (x, y) corners of each vessel and its radius in pixels
        ([(0, 25), (30, 32), (60, 40), (100, 36), (127, 48)], 6),  # trunk
        ([(60, 40), (95, 20), (115, 4)], 3),  # branch up
        ([(60, 40), (48, 80), (70, 110), (75, 127)], 3),  # branch down
        ([(30, 32), (20, 10), (25, 2)], 1),  # capillaries from here on
        ([(48, 80), (25, 90), (8, 110)], 1),
        ([(48, 80), (30, 60), (8, 55)], 1),
        ([(100, 36), (112, 65), (105, 95)], 1),
        ([(100, 36), (85, 60), (95, 85)], 1),
        ([(70, 110), (100, 115), (125, 105)], 1),
    ]
    out = np.zeros((size, size), bool)
    for corners, radius in tree:
        if radius < min_radius:
            continue
        centre = np.zeros((size, size), bool)
        for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:], strict=True):
            rr, cc = line(y0, x0, y1, x1)
            centre[rr, cc] = True
        out |= ndi.distance_transform_edt(~centre) <= radius * scale
    return out


def toy_prediction(case: str, size: int = 128) -> np.ndarray:
    """The prediction of one case: every vessel twice as wide, no capillaries, or one big disc."""
    if case == "too wide":
        return vessel_toy(scale=2.0)
    if case == "capillaries missed":
        return vessel_toy(min_radius=2)
    yy, xx = np.mgrid[:size, :size]
    return (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (0.7 * size) ** 2


def explain_cldice(case: str) -> None:
    """Draw the four steps of clDice on one case, and IoU next to clDice for all three cases."""
    if case not in TOY_CASES:
        raise SystemExit(f"CASE must be one of {TOY_CASES}.")
    truth = vessel_toy()

    def paint(img, mask, colour, alpha=1.0):
        img = img.copy()
        img[mask] = (1 - alpha) * img[mask] + alpha * hex_rgb(colour)
        return img

    def thick(mask):  # a 1 px line is drawn 3 px wide so it shows up
        return ndi.binary_dilation(mask, np.ones((3, 3), bool))

    blank = np.full(truth.shape + (3,), 0.97)
    results = {}
    for name in TOY_CASES:
        pred = toy_prediction(name)
        tprec, tsens, sk_pred, sk_truth = cldice_parts(pred, truth)
        both = (iou(pred, truth), cldice(pred, truth, sk_pred, sk_truth))
        results[name] = (pred, tprec, tsens, sk_pred, sk_truth, *both)
    pred, tprec, tsens, sk_pred, sk_truth, iou_c, cld_c = results[case]
    white = "#FFFFFF"
    n_in, n_all = int((sk_pred & truth).sum()), int(sk_pred.sum())
    m_in, m_all = int((sk_truth & pred).sum()), int(sk_truth.sum())

    fig, axes = plt.subplots(
        1, 5, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]}
    )
    img = paint(paint(blank, truth, THEME["truth"], 0.6), sk_truth, white)
    axes[0].imshow(img)
    axes[0].set_title("1 · truth + centre line", fontsize=9)
    img = paint(paint(blank, pred, THEME["background"], 0.85), sk_pred, white)
    axes[1].imshow(img)
    axes[1].set_title(f"2 · prediction ({case})\n+ centre line", fontsize=9)
    img = paint(blank, truth, THEME["truth"], 0.35)
    img = paint(paint(img, thick(sk_pred & ~truth), THEME["fails"]), thick(sk_pred & truth), THEME["works"])
    axes[2].imshow(img)
    axes[2].set_title(f"3 · tprec = {tprec:.2f}\n{n_in} of {n_all} counted", fontsize=9)
    img = paint(blank, pred, THEME["background"], 0.5)
    img = paint(paint(img, thick(sk_truth & ~pred), THEME["truth"]), thick(sk_truth & pred), THEME["works"])
    axes[3].imshow(img)
    axes[3].set_title(f"4 · tsens = {tsens:.2f}\n{m_in} of {m_all} counted", fontsize=9)
    for ax in axes[:4]:
        ax.axis("off")

    x = np.arange(len(TOY_CASES))
    vals = [(results[c][5], results[c][6]) for c in TOY_CASES]
    axes[4].bar(x - 0.2, [v[0] for v in vals], 0.4, color=THEME["background"], label="IoU")
    axes[4].bar(x + 0.2, [v[1] for v in vals], 0.4, color=THEME["works"], label="clDice")
    for xi, (a, b) in zip(x, vals, strict=True):
        axes[4].text(xi - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        axes[4].text(xi + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    axes[4].set_xticks(x, [c.replace(" ", "\n", 1) for c in TOY_CASES], fontsize=8)
    axes[4].get_xticklabels()[TOY_CASES.index(case)].set_fontweight("bold")
    axes[4].set_ylim(0, 1.15)
    axes[4].set_title("IoU vs clDice, all three cases\n(bold = the case drawn)", fontsize=9)
    axes[4].legend(fontsize=7, frameon=False, loc="upper right")
    add_key(
        fig,
        [
            ("truth", "green: truth, and truth line the prediction misses"),
            ("prediction (grey)", THEME["background"]),
            ("centre line (white)", "#FFFFFF"),
            ("counted (violet)", THEME["works"]),
            ("orange: prediction line outside the truth (invented)", THEME["fails"]),
        ],
    )
    plt.show()
    print(
        f"clDice = 2 · tprec · tsens ÷ (tprec + tsens) = 2 · {tprec:.2f} · {tsens:.2f} ÷ "
        f"({tprec:.2f} + {tsens:.2f}) = {cld_c:.2f}   (IoU is {iou_c:.2f})"
    )


# %% [markdown]
# This cell draws clDice on a made-up vessel tree (a thick trunk, two branches and six capillaries). Pick a
# case in the `CASE` field and the figure redraws. The four steps below match the four left panels, with the
# numbers of the default case, "too wide" (a prediction that draws every vessel twice as wide). The fifth
# panel puts IoU next to clDice for all three cases. A blob's centre line is a short stub in its middle: here
# the stub misses every vessel, so clDice is 0.
#
# 1. **Thin the true vessels to their centre line** (white line on the green truth, panel 1).
# 2. **Thin the prediction the same way** (white line on the grey prediction, panel 2).
# 3. **tprec**: of the prediction's centre line, how much lies inside the true vessels? Violet pixels count,
#    orange ones do not. 490 of 529 pixels = 0.93.
# 4. **tsens**: of the true centre line, how much is covered by the prediction? 534 of 534 = 1.00.
#    Then clDice = 2 · 0.93 · 1.00 ÷ (0.93 + 1.00) = 0.96, while IoU for the same masks is only 0.57.
#
# What to look for: in "too wide" the prediction is fat but its centre line still follows the truth, so
# clDice stays high and IoU drops. Try "capillaries missed": now tsens drops, because the thin branches are
# gone.

# %%
# @title See clDice on a made-up vessel tree { display-mode: "form", run: "auto" }
CASE = "too wide"  # @param ["too wide", "capillaries missed", "blob"]
explain_cldice(CASE)

# %% [markdown]
# <!-- callout: go-deeper | Where clDice comes from -->
# clDice is from Shit et al., "clDice: a novel topology-preserving loss function for tubular structure
# segmentation", CVPR 2021.

# %% [markdown]
# ### 2.3 Escalating prompts: 1 → 5 → 20 clicks, and a box
#
# If one click is ambiguous, maybe more clicks tell SAM what we mean. Each step tells SAM something new:
#
# 1. **1 point**: one spot. SAM must guess: this vessel, the tree, or the eye?
# 2. **5 points**: five spots along different vessels. SAM looks for one object containing all the clicks.
# 3. **20 points**: dense along the tree. More information, but still only *where*, never *what kind*.
# 4. **box**: a box around the whole fundus says "everything in here".

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Write down three guesses before you run the ladder.
#
# - Will IoU rise steadily from 1 to 20 clicks, jump at some point, or stay flat?
# - Will the box beat 20 clicks?
# - Where will the grey line sit? It scores a mask that says "everything is foreground": if vessels cover 12 %
#   of the fundus, overlap 12 % ÷ union 100 % gives IoU 0.12. The load cell printed your share.

# %% [markdown]
# This cell runs the ladder on every evaluation image and prints three numbers per setting. **IoU** is
# overlap ÷ union. **Recall** is the share of the true vessel pixels the mask finds. **Marked** is the share
# of the fundus the mask covers (100 % = the whole eye).
#
# Then two figures. The bars show mean IoU for each setting (orange), the dashed grey line is the score of
# marking everything, and the dotted violet line is SAM's one-box score on the pets. The pictures show the
# four prompts (teal stars and box) on the demo image, with SAM's mask in violet next to the green truth.
# What to look for: bars that do not rise above the grey dashed line, and masks that grow without getting
# thinner.

# %%
# @title Run the SAM prompt ladder on every evaluation image { display-mode: "form" }
LADDER = {"1 point": 1, "5 points": 5, "20 points": 20, "box": None}  # setting -> number of clicks
t0 = time.monotonic()
ladder_scores = {name: [] for name in LADDER}
ladder_cover = {name: [] for name in LADDER}
ladder_recall = {name: [] for name in LADDER}
ladder_demo = {}  # the demo image's masks and clicks, for the figure below
sam_box_masks = []  # every image's box mask, for the gallery in 2.5
everything = []
for it in tqdm(spec_eval, desc="SAM prompt ladder"):
    r = region(it)
    for name, n in LADDER.items():
        pts = scribble_points(it, n) if n else None
        if n:
            pred = sam.probability(it["image"], points=pts, labels=[1] * len(pts)) >= 0.5
        else:
            pred = sam.probability(it["image"], box=box_prompt(it)) >= 0.5
        ladder_scores[name].append(score(pred, it))
        ladder_cover[name].append(float(pred[r].mean()))
        ladder_recall[name].append(recall_of(pred, it))
        if it is demo:
            ladder_demo[name] = (pred, pts, box_prompt(it) if not n else None)
    sam_box_masks.append(pred)  # the last setting is the box
    everything.append(everything_score(it))
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

vessel_share = float(np.mean([it["mask"][region(it)].mean() for it in spec_eval]))
print(f"SAM prompt ladder on {len(spec_eval)} held-out images · {spec_sec} s on {DEVICE}")
print(
    "marked = share of the fundus the mask covers "
    f"(100 % = the whole eye; the vessels cover about {vessel_share:.0%})"
)
print("recall = share of the true vessel pixels the mask finds")
print(f"{'setting':<12}{'IoU':>7}{'recall':>8}{'marked':>8}")
for name, v in ladder_mean.items():
    print(f"{name:<12}{v['iou']:>7.3f}{np.mean(ladder_recall[name]):>8.3f}{np.mean(ladder_cover[name]):>8.0%}")
print(f"{'everything':<12}{everything_mean['iou']:>7.3f}{1:>8.3f}{1:>8.0%}")
box_marked = float(np.mean(ladder_cover["box"]))
if abs(ladder_mean["box"]["iou"] - everything_mean["iou"]) < 0.02:
    print(
        "'box' and 'everything' have the same IoU because SAM's box mask covers "
        f"{box_marked:.0%} of the fundus, so it scores like marking everything."
    )

fig, ax = plt.subplots(figsize=(6.4, 3))
names = list(ladder_mean)
ax.bar(names, [ladder_mean[k]["iou"] for k in names], color=THEME["fails"], label="retina, SAM")
ax.axhline(everything_mean["iou"], color=THEME["background"], ls="--", label='"everything is foreground"')
ax.axhline(RESULTS["pets_zero_shot_iou"], color=THEME["works"], ls=":", label="pets, SAM one box (section 1)")
ax.set_ylabel("mean IoU")
ax.set_ylim(0, 1)
ax.set_title("More prompts, same model: mean IoU on the retina")
ax.legend(fontsize=8, frameon=False, loc="center right")
plt.tight_layout()
plt.show()

show_clicks(
    [
        (pred_title(f"SAM, {k}", m, demo), overlay(demo["image"], m, THEME["works"]), p, b)
        for k, (m, p, b) in ladder_demo.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]), None)],
    legend=[model_key("SAM's mask"), ("truth", "vessels (ground truth)"), "click", "box"],
    note=MARKED_NOTE,
)

# %% [markdown]
# **Q6.** 🟡 Compare the ladder with your prediction. Read the table row by row. Example: IoU 0.30 with 70 %
# marked, against a grey line at 0.29, is a big mask that is barely better than marking everything.
#
# - Which rows beat the grey line by more than 0.05 IoU?
# - Did more clicks bring the masks closer to the vessels, or just make them bigger?
# - What does that say about what SAM thinks an object is?
#
# <details><summary>What to look for</summary>
#
# - Read IoU next to the grey line, not on its own. A mask that marks most of the fundus scores the
#   "everything" baseline, whatever the prompt.
# - If more clicks push "marked" up while IoU stays near the line, SAM is growing one big object around the
#   clicks, and for a vessel tree that object is the eye.
# - If clicks do help, it is along the thick vessels near the clicks, rarely the capillaries in between. A
#   prompt says *where*, not *what kind of thing* (section 7 comes back to it).
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
# ### 2.4 The linear head, and the resolution it looks at
#
# The head from section 1.4, trained the same way on the full masks of five retina images (the last time
# training sees a full mask: a perfect vessel map costs an expert 3 to 5 hours per fundus). We train it
# twice, on DINOv3 features at two resolutions:
#
# - **512 px long side**: each 16 px patch covers 32 × 32 pixels of the 1024 px crop.
# - **1024 px long side**: each patch covers 16 × 16 pixels. Four times the patches, and about 16 times the
#   cost, because attention compares every patch with every other.

# %% [markdown]
# This collapsed cell trains (or loads) one head per resolution on the five fit images.

# %%
# @title Helper: train one head per resolution, or load ours { display-mode: "form" }
N_TRAIN_SPEC = 5
sides = specialised["feature_long_sides"]  # [512, 1024]: the two resolutions we compare
sides = [side // 4 for side in sides] if SMOKE else sides  # smoke thumbnails are 256 px


def fit_spec_heads(train_yourself: bool) -> tuple[list[dict], dict]:
    """Train one linear head per resolution on the first N_TRAIN_SPEC fit images (or load ours).

    The border is labelled -1, so the loss skips it. Returns the fit images and {long side: (head, seconds)}.
    """
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_SPEC_HEADS must be True or False.")
    items = [load_item(specialised, it) for it in specialised["fit"][:N_TRAIN_SPEC]]
    heads = {}
    for side in sides:
        if train_yourself:
            feats = [features_for(specialised, it, side) for it in items]
            labels = [np.where(region(it), it["mask"], -1).astype(np.int8) for it in items]
            heads[side] = train_head(feats, labels, HEAD_STEPS)
            src = f"trained here on {len(items)} images, {HEAD_STEPS} steps"
        else:
            tensors, entry = load_head(specialised, "full", side)
            heads[side] = (tensors, entry["train_sec"])
            src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
        print(f"linear head @ {side} px · {src} · {heads[side][1]:.1f} s on {DEVICE}")
    if sum(sec for _, sec in heads.values()) > 60:
        print("ℹ️ Slow session? Untick TRAIN_SPEC_HEADS above to load our checkpoints next time.")
    return items, heads


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# With *perfect* labels, write two guesses.
#
# - Will the head beat SAM's one-box IoU from the 2.3 table?
# - Will doubling the resolution add more than 0.05 to IoU, to recall, to both or to neither?

# %% [markdown]
# This cell trains one head per resolution on the five fit images and prints the seconds it took. If it is
# slow, untick `TRAIN_SPEC_HEADS` and run it again to load our checkpoints.

# %%
# @title Train the two heads here (or load ours) { display-mode: "form" }
TRAIN_SPEC_HEADS = True  # @param {type:"boolean"}
spec_train, spec_heads = fit_spec_heads(TRAIN_SPEC_HEADS)

# %% [markdown]
# This cell scores both heads on the evaluation images and prints mean IoU, recall and marked per
# resolution, next to SAM's one box. What to look for: does doubling the resolution add more than 0.05 to
# IoU, and does recall rise with it?

# %%
# @title Score both heads on the evaluation images { display-mode: "form" }
t0 = time.monotonic()
spec_head_preds, res_scores, res_recall, res_marked = {}, {}, {}, {}
for side, (tensors, _) in spec_heads.items():
    spec_head_preds[side] = [
        predict_head(tensors, features_for(specialised, it, side), it["mask"].shape)
        for it in tqdm(spec_eval, desc=f"head @ {side} px")
    ]
    preds = spec_head_preds[side]
    res_scores[side] = mean_scores([score(p, it) for p, it in zip(preds, spec_eval, strict=True)])
    res_recall[side] = float(np.mean([recall_of(p, it) for p, it in zip(preds, spec_eval, strict=True)]))
    res_marked[side] = float(np.mean([p[region(it)].mean() for p, it in zip(preds, spec_eval, strict=True)]))
top_side = max(sides)  # the higher of the two resolutions
RESULTS["specialised_head_iou"] = res_scores[top_side]["iou"]
RESULTS["specialised_head"] = {
    "iou": res_scores[top_side]["iou"],
    "cldice": res_scores[top_side]["cldice"],
    "long_side": top_side,
    "n_train": len(spec_train),
    "train_sec": round(float(spec_heads[top_side][1]), 2),
    "infer_sec": round(time.monotonic() - t0, 1),
}
RESULTS["resolution_effect"] = {f"{side} px": v for side, v in res_scores.items()}
for side, v in res_scores.items():
    print(
        f"linear head @ {side:>4} px · mean IoU {v['iou']:.3f} · recall {res_recall[side]:.3f} · "
        f"marked {res_marked[side]:.0%}"
    )
print(
    f"SAM one box · mean IoU {ladder_mean['box']['iou']:.3f} · recall {np.mean(ladder_recall['box']):.3f} · "
    f"marked {np.mean(ladder_cover['box']):.0%}"
)
print("(marked = share of the fundus the mask covers; recall = share of the true vessel pixels it finds)")

# %% [markdown]
# This cell puts the section-1 pet numbers next to this section's vessel numbers, for the same two tools
# (SAM and a linear head). The tools are the same, only the task changed. Violet bars are the pets, orange
# bars the vessels, and the dashed grey line is the score of marking everything on the vessels. What to look
# for: every orange bar sits far below its violet neighbour and close to the dashed line.

# %%
# @title Pets vs vessels: the same two tools, high on pets and low on vessels { display-mode: "form" }
rows = {
    "SAM,\none click": (RESULTS["pets_zero_shot_point_iou"], RESULTS["specialised_zero_shot_point_iou"]),
    "SAM,\none box": (RESULTS["pets_zero_shot_box_iou"], RESULTS["specialised_zero_shot_box_iou"]),
    "linear head on\nDINOv3 features": (RESULTS["pets_head_iou"], RESULTS["specialised_head_iou"]),
}
print(f"{'tool':<32}{'pets IoU':>10}{'vessels IoU':>13}")
for name, (p, v) in rows.items():
    print(f"{name.replace(chr(10), ' '):<32}{p:>10.3f}{v:>13.3f}")
print(f"{'everything is foreground':<32}{'':>10}{RESULTS['specialised_everything_iou']:>13.3f}")

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(
    x - 0.2,
    [v[0] for v in rows.values()],
    0.4,
    color=THEME["works"],
    label=f"{pets_task['name']} (easy task)",
)
ax.bar(x + 0.2, [v[1] for v in rows.values()], 0.4, color=THEME["fails"],
       label=f"{specialised['name']} (specialised task)")
ax.axhline(RESULTS["specialised_everything_iou"], color=THEME["background"], ls="--",
           label='vessels, "everything is foreground"')
ax.set_xticks(x, list(rows), fontsize=8)
ax.set_ylabel("mean IoU on held-out images")
ax.set_ylim(0, 1)
ax.set_title("Same two tools, two tasks: pets vs vessels")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 The failure gallery
#
# Numbers average the failures away; the gallery shows them. It sorts the evaluation images from worst to
# best (by the head's IoU at the higher resolution) and shows three at a time. Each row is one image: the
# photo, SAM's mask from a box (violet), the head's mask (violet) and the ground truth (green). What to look
# for: the kind of mistake, whole regions marked for SAM, missing thin vessels for the head.

# %% [markdown]
# This collapsed cell ranks the evaluation images by the head's IoU and draws the gallery.

# %%
# @title Helper: the failure gallery, worst images first { display-mode: "form" }
def gallery_row(k: int, side: int) -> None:
    """One row for image k: the photo, SAM with a box, the head at `side` px, and the ground truth."""
    it = spec_eval[k]
    head_mask = spec_head_preds[side][k]
    show_masks(
        [
            (it["name"], it["image"]),
            (
                pred_title("SAM from a box", sam_box_masks[k], it),
                overlay(it["image"], sam_box_masks[k], THEME["works"]),
            ),
            (
                pred_title(f"linear head, features @ {side} px", head_mask, it),
                overlay(it["image"], head_mask, THEME["works"]),
            ),
            (
                f"ground truth\nvessels cover {100 * it['mask'][region(it)].mean():.0f} % of the fundus",
                overlay(it["image"], it["mask"], THEME["truth"]),
            ),
        ],
        width=2.8,
        legend=[model_key("the model's mask"), ("truth", "vessels (ground truth)")],
        note=MARKED_NOTE,
    )


def rank_images() -> tuple[list[dict], np.ndarray, int]:
    """Score the higher-resolution head on every evaluation image and rank the images, worst first.

    Returns the scores, the ranking (image indices) and the resolution used. The predictions are looked up by
    resolution (a key of `spec_head_preds`, such as 1024), never by image number."""
    side = max(spec_head_preds)
    scores = [score(p, it) for p, it in zip(spec_head_preds[side], spec_eval, strict=True)]
    return scores, np.argsort([s["iou"] for s in scores]), side


def show_gallery(start: int) -> None:
    """Draw the three images from rank `start` (0 = worst) of the ranking by the head's IoU."""
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(spec_eval):
        raise SystemExit(f"GALLERY_START must be a whole number from 0 to {len(spec_eval) - 1}.")
    scores, ranking, side = rank_images()
    for k in ranking[start : start + 3]:
        gallery_row(int(k), side)


# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min -->
# Run the next cell: it draws the failure gallery from the predictions you already have, three images at a
# time, worst first (ranked by the head's IoU).
#
# **Q7.** 🟡 Each of you pages to a different place with the `GALLERY_START` field (0 = the three worst; try
# 3, 6, ...; change only that value) and finds the worst mask you can. Show it to your neighbour and name the
# failure in three words. Pick from: missed thin vessels, vessels drawn too wide, the optic disc or the border
# marked as vessel, whole regions marked.
#
# - Is it the same failure for SAM and for the head?

# %%
# @title Q7 — The failure gallery, worst images first { display-mode: "form" }
GALLERY_START = 0  # @param {type:"integer"}
show_gallery(GALLERY_START)

# %% [markdown]
# <details><summary>What people usually find</summary>
#
# - SAM tends to fail *globally*: it marks a whole region or one thick vessel, because it looks for an object.
# - The head fails *locally*: it finds the thick vessels, draws them as wide as a patch, and loses the
#   capillaries, because it labels 16 px squares.
# - Neither fails by being unsure: both draw crisp masks. Always put the mask next to the truth.
#
# </details>


# %% [markdown]
# **Got a photo of something thin or specialised?** Section 9 runs SAM, the ladder and the head on it.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **Objects thinner than a 16 px patch** → raise the resolution first, and check how far the numbers move.
# - **Score close to the "everything is foreground" line** → the model learnt nothing about the task; always
#   compare with a baseline that knows nothing, and look at the mask next to the truth.
# - **More clicks make SAM's mask bigger, not thinner** → stop prompting: a prompt says *where*, not *what
#   kind of thing*.
# - **IoU and clDice disagree** (2.2) → trust the structure metric on thin objects, and look at the masks.
