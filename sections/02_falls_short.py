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
# This helper cell defines what this section measures with: skeletons for the thin-structure metric (cached,
# because clDice is slow on near-solid masks), clicks spread along a person's scribbles, and the figure
# helpers that title every prediction with how much of the fundus it marked.

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


def pred_title(label: str, pred: np.ndarray, item: dict, s: dict | None = None, with_cldice: bool = True):
    """Panel title: what made the mask, then IoU, clDice (from 2.2 on) and the share of the fundus marked."""
    if not with_cldice:  # before 2.2 introduces clDice
        r = region(item)
        return f"{label}\nIoU {iou(pred & r, item['mask'] & r):.2f} · {marked_pct(pred, item)}"
    s = s or score(pred, item)
    return f"{label}\nIoU {s['iou']:.2f} · clDice {s['cldice']:.2f} · {marked_pct(pred, item)}"


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


def show_clicks(panels: list[tuple[str, np.ndarray, list | None]], width: float = 2.6):
    """One row of panels (title, RGB image, clicks or None[, box]); stars = clicks, frame = box."""
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
    plt.tight_layout()
    plt.show()


# %% [markdown]
# This cell loads the held-out retina images (1024 px crops of a 3504 × 2336 fundus), the evaluation set for
# the rest of the notebook, and prints how much of the fundus the vessels cover; keep that number, it comes
# back as a baseline.

# %%
# @title Load the held-out retina images { display-mode: "form" }
specialised["feat_side"] = 256 if SMOKE else specialised["feature_long_side"]
specialised["eval"] = specialised["held"] if (SMOKE or DEVICE == "cuda") else specialised["held"][:12]
spec_eval = [load_item(specialised, it) for it in specialised["eval"]]
fg_share = float(np.mean([it["mask"][region(it)].mean() for it in spec_eval]))
print(
    f"evaluation set: {len(spec_eval)} held-out images · {specialised['task']} · the foreground covers "
    f"{fg_share:.0%} of the evaluation region · DINOv3 features at {specialised['feat_side']} px long side"
)

# %% [markdown]
# Before any theory, look at the images: this cell shows one retina photo, its vessels, and a zoom with the
# 16 px grid DINOv3 sees, and measures how wide a vessel is next to one patch.

# %%
# @title Meet the retina images: vessels next to the DINOv3 patch grid { display-mode: "form" }
peek = spec_eval[0]
ph, pw = peek["mask"].shape
patch_px = 16 * max(ph, pw) / specialised["feat_side"]  # one DINOv3 patch, in image pixels
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
# This helper cell defines what the first cell of 2.1 runs: SAM with one click and with one box on one
# held-out image, the share of the fundus each mask marks, and the three-panel figure.

# %%
# @title Helper: SAM with one click and one box on a retina image { display-mode: "form" }
def pick_demo(index: int) -> dict:
    """The held-out retina image with this index; a plain message when the index does not exist."""
    if not isinstance(index, int) or not 0 <= index < len(spec_eval):
        raise SystemExit(f"IMAGE must be a whole number from 0 to {len(spec_eval) - 1}.")
    return spec_eval[index]


def click_and_box(item: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM on one click and on one box, print the shares of the fundus they mark, draw the figure."""
    point, box = click_point(item), box_prompt(item)
    t0 = time.monotonic()
    by_click = sam.probability(item["image"], points=[point], labels=[1]) >= 0.5
    by_box = sam.probability(item["image"], box=box) >= 0.5
    print(f"{item['name']}: two SAM calls in {time.monotonic() - t0:.1f} s on {DEVICE}")
    shares = {"click mask": by_click, "box mask": by_box, "true vessels": item["mask"]}
    marked = " · ".join(f"{k} {m[region(item)].mean():.0%}" for k, m in shares.items())
    print(f"share of the fundus marked · {marked}")
    show_masks(
        [
            (
                pred_title("SAM, one click", by_click, item, with_cldice=False),
                overlay(item["image"], by_click, THEME["fails"]),
            ),
            (
                pred_title("SAM, one box", by_box, item, with_cldice=False),
                overlay(item["image"], by_box, THEME["fails"]),
            ),
            (f"ground truth\nvessels cover {100 * item['mask'][region(item)].mean():.0f} % of the fundus",
             overlay(item["image"], item["mask"], THEME["truth"])),
        ],
        prompts={"point": point, "box": box},
    )
    return by_click, by_box


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Before you run the next cell (the section-1 cell on a retina image: one click on a vessel, then a box
# around all of them), guess: what will SAM segment from one click on a thin line? And what is "the object"
# inside a box that covers the whole fundus?
#
# Then run it and compare the panels with the pets in section 1.
#
# **Q6.** 🟡 What did SAM segment from the click, and from the box: a vessel, part of the tree, a blob, the
# whole fundus? Use the printed shares: the true vessels cover a small share of the fundus, so a mask that
# marks far more is not tracing vessels. Try two other images with the `IMAGE` field (on a CPU session there
# are 12 held-out images).

# %%
# @title Q6 — SAM with one click and with one box on a held-out retina image { display-mode: "form" }
IMAGE = 0  # @param {type:"integer"}
demo = pick_demo(IMAGE)
sam_point, sam_box = click_and_box(demo)

# %% [markdown]
# **Try it yourself.** Section 1's SAM could mark a pet. Now see whether you can make it mark only the
# vessels: click on one vessel, click along several, or draw a small box around a few of them, and watch what
# SAM marks each time.
#
# **Q7.** 🟡 Can you get SAM to mark only the vessels? Try one click on a vessel, several clicks along it
# and a small box. What does SAM mark each time: the vessel, the tree, or something bigger?

# %%
# @title Q7 — Try it: make SAM mark only the vessels { display-mode: "form" }
sam_playground(spec_eval, "Retina: try to get only the vessels")

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
# This collapsed cell is the reference clDice, the three steps above as code. Later cells score masks with it,
# and the figure below uses it too; you do not need to read it.

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
# This collapsed cell builds the made-up vessel tree and the figure of the next cell, so that cell stays a
# single line.

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
    """Draw truth and prediction with their centre lines, what tprec and tsens count, and IoU vs clDice."""
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

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    img = paint(paint(blank, truth, THEME["truth"], 0.6), sk_truth, white)
    axes[0].imshow(img)
    axes[0].set_title("made-up truth (green)\nand its centre line (white)", fontsize=8)
    img = paint(paint(blank, pred, THEME["background"], 0.85), sk_pred, white)
    axes[1].imshow(img)
    marked = 100 * pred.mean()
    axes[1].set_title(
        f"prediction: {case}\nIoU {iou_c:.2f} · clDice {cld_c:.2f} · marked {marked:.0f} %", fontsize=8
    )
    img = paint(blank, truth, THEME["truth"], 0.35)
    img = paint(paint(img, thick(sk_pred & ~truth), THEME["fails"]), thick(sk_pred & truth), THEME["works"])
    axes[2].imshow(img)
    axes[2].set_title(
        f"tprec = {tprec:.2f}\n{n_in} of {n_all} prediction centre-line pixels\n"
        "inside the truth (violet = counted)", fontsize=8)
    img = paint(blank, pred, THEME["background"], 0.5)
    img = paint(paint(img, thick(sk_truth & ~pred), THEME["fails"]), thick(sk_truth & pred), THEME["works"])
    axes[3].imshow(img)
    axes[3].set_title(
        f"tsens = {tsens:.2f}\n{m_in} of {m_all} true centre-line pixels\n"
        "inside the prediction (violet = counted)", fontsize=8)
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
    axes[4].set_title("IoU vs clDice, all three cases", fontsize=8)
    axes[4].legend(fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# This cell draws clDice on a made-up vessel tree (a thick trunk, two branches and six capillaries). Pick a
# case in the `CASE` field and the figure redraws: truth and prediction, each with its centre line, then the
# centre-line pixels that `tprec` and `tsens` count, and IoU next to clDice for all three cases. A blob's
# centre line is a short stub in its middle: here the stub misses every vessel, so clDice is 0.

# %%
# @title See clDice on a made-up vessel tree { display-mode: "form", run: "auto" }
CASE = "too wide"  # @param ["too wide", "capillaries missed", "blob"]
explain_cldice(CASE)

# %% [markdown]
# <!-- callout: go-deeper | Where clDice comes from -->
# clDice is from Shit et al., "clDice: a novel topology-preserving loss function for tubular structure
# segmentation", CVPR 2021.

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

print(f"SAM prompt ladder on {len(spec_eval)} held-out images · {spec_sec} s on {DEVICE}")
print(f"{'setting':<12}{'IoU':>7}{'clDice':>8}{'marked':>8}")
for name, v in ladder_mean.items():
    print(f"{name:<12}{v['iou']:>7.3f}{v['cldice']:>8.3f}{np.mean(ladder_cover[name]):>8.0%}")
print(f"{'everything':<12}{everything_mean['iou']:>7.3f}{'—':>8}{1:>8.0%}")
print("(no clDice for 'everything': its centre line is a stub of a few pixels, so it scores 0 or 1 by luck)")

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
    [
        (pred_title(f"SAM, {k}", m, demo), overlay(demo["image"], m, THEME["fails"]), p, b)
        for k, (m, p, b) in ladder_demo.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]), None)]
)

# %% [markdown]
# **Q8.** 🟡 Compare the ladder with your prediction. Read the table row by row: IoU, clDice, and *marked*
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
fig_items = [load_item(specialised, it) for it in specialised["fit"][:3]]
panels = []
for it in fig_items:
    feat = features_for(specialised, it, specialised["feat_side"])
    title = f"DINOv3 PCA · {feat.shape[2]}×{feat.shape[1]} patches"
    panels += [(it["name"], it["image"]), (title, pca_rgb(feat, it["image"].shape[:2]))]
show_masks(panels, width=2.6)

# %% [markdown]
# **Q9.** 🔴 Do the thick vessels share a colour at all, across the three images? Which of the optic disc
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
# This collapsed cell defines how the two heads get trained (or loaded) on the five fit retina images, at the
# two resolutions.

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
# - Will doubling the resolution add more than 0.05 to IoU, to clDice, to both or to neither?

# %% [markdown]
# This cell trains one head per resolution on the five fit images and prints the seconds it took; if it is
# slow, untick `TRAIN_SPEC_HEADS` and run it again to load our checkpoints (trained the same way on all fit
# images).

# %%
# @title Train the two heads here (or load ours) { display-mode: "form" }
TRAIN_SPEC_HEADS = True  # @param {type:"boolean"}
spec_train, spec_heads = fit_spec_heads(TRAIN_SPEC_HEADS)

# %% [markdown]
# This cell scores both heads on the evaluation images with IoU and clDice and zooms into the centre of the
# demo image to compare the two resolutions on the thinnest vessels.

# %%
# @title Score both heads and zoom into the thinnest vessels { display-mode: "form" }
t0 = time.monotonic()
head_preds, res_scores = {}, {}
for side, (tensors, _) in spec_heads.items():
    head_preds[side] = [
        predict_head(tensors, features_for(specialised, it, side), it["mask"].shape)
        for it in tqdm(spec_eval, desc=f"head @ {side} px")
    ]
    res_scores[side] = mean_scores([score(p, it) for p, it in zip(head_preds[side], spec_eval, strict=True)])
lo, hi = min(sides), max(sides)
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

k = next(i for i, it in enumerate(spec_eval) if it is demo)
h, w = demo["mask"].shape
zoom = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))  # the central quarter
show_masks(
    [
        (
            pred_title(f"linear head on DINOv3\nfeatures @ {s} px (centre zoom)", head_preds[s][k], demo),
            overlay(demo["image"], head_preds[s][k], THEME["fails"])[zoom],
        )
        for s in sides
    ]
    + [("ground truth (centre zoom)", overlay(demo["image"], demo["mask"], THEME["truth"])[zoom])],
    width=3.0,
)

# %% [markdown]
# This cell puts the section-1 numbers on pets next to this section's numbers on retina vessels, for the same
# two tools: SAM (prompted with one click or one box) and a linear head on DINOv3 features. Both tools score
# high on pets and low on vessels. That gap is the point of section 2: the tools are the same, only the task
# changed.

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
# This collapsed cell defines the failure gallery: it ranks the evaluation images by the head's IoU, worst
# first, and draws three at a time.

# %%
# @title Helper: the failure gallery, worst images first { display-mode: "form" }
def gallery_row(label: str, k: int, scores: list[dict]) -> None:
    """One row for evaluation image k: the photo, SAM with a box, the head, and the ground truth."""
    it, sam_s, head_s = spec_eval[k], ladder_scores["box"][k], scores[k]
    show_masks(
        [
            (f"{label}{it['name']}", it["image"]),
            (
                pred_title("SAM, one box", sam_box_masks[k], it, sam_s),
                overlay(it["image"], sam_box_masks[k], THEME["fails"]),
            ),
            (
                pred_title(f"linear head on DINOv3\nfeatures @ {hi} px", head_preds[hi][k], it, head_s),
                overlay(it["image"], head_preds[hi][k], THEME["fails"]),
            ),
            (
                f"ground truth\nvessels cover {100 * it['mask'][region(it)].mean():.0f} % of the fundus",
                overlay(it["image"], it["mask"], THEME["truth"]),
            ),
        ],
        width=2.8,
    )


def show_gallery(start: int) -> tuple[list[dict], np.ndarray]:
    """Draw the three images from rank `start` (0 = worst). Returns every head score and the ranking."""
    if not isinstance(start, int) or not 0 <= start < len(spec_eval):
        raise SystemExit(f"GALLERY_START must be a whole number from 0 to {len(spec_eval) - 1}.")
    scores = [score(p, it) for p, it in zip(head_preds[hi], spec_eval, strict=True)]
    ranking = np.argsort([s["iou"] for s in scores])
    for k in ranking[start : start + 3]:
        gallery_row("", int(k), scores)
    return scores, ranking


# %% [markdown]
# <!-- callout: question | Question -->
# Run the next cell: it draws the failure gallery from the predictions you already have, with each image's
# IoU, clDice and marked share in the titles.
#
# **Q10.** 🟡 Read the IoU, clDice and *marked* share in each title. For the worst image, is the failure the
# same kind for SAM and for the head? Pick from: missed thin vessels, vessels drawn too wide, the optic disc
# or the border marked as vessel, whole regions marked. Page with the `GALLERY_START` field (0 = the three
# worst; try 3, 6, ...).

# %%
# @title Q10 — The failure gallery, worst images first { display-mode: "form" }
GALLERY_START = 0  # @param {type:"integer"}
gallery, order = show_gallery(GALLERY_START)

# %% [markdown]
# Secondary visual: this cell shows the typical (median) and the best image of the same ranking, to see what
# an IoU of, say, 0.3 versus 0.7 looks like next to the worst ones.

# %%
# @title For contrast: the typical and the best image of the ranking { display-mode: "form" }
for label, k in (("typical (median)", order[len(order) // 2]), ("best", order[-1])):
    gallery_row(f"{label} · ", int(k), gallery)

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
# - On the whole fundus both get worse, since every vessel is now thinner than a patch (optional 2.8 below).
#
# </details>
#
# ### 2.7 A checklist for a new problem

# %% [markdown]
# **Q11.** 🟡 Use what is on screen as a checklist: the "everything is foreground" line in the ladder, the
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
# ### 2.8 🔴 Optional: the whole fundus
#
# The gallery shows 1024 px crops. This cell runs SAM with a box and the head on the whole fundus of the demo
# image shrunk to 1024 px (vessels only 1–3 px wide), next to the truth.

# %%
# @title The failure gallery on the whole fundus { display-mode: "form" }
# the whole fundus of the demo image, shrunk so its long side is 1024 px (smoke: 256 px)
meta = next(m for m in specialised["eval"] if m["name"] == demo["name"])
root = specialised["dir"]
full = {
    "image": np.array(Image.open(root / meta["full"]["image"]).convert("RGB")),
    "mask": read_png(root / meta["full"]["mask"]) > 127,
    "fov": read_png(root / meta["full"]["fov"]) > 127,
}
full_sam = sam.probability(full["image"], box=box_prompt(full)) >= 0.5
full_feat = dino.features(full["image"], max(full["image"].shape[:2]))
full_head = predict_head(spec_heads[hi][0], full_feat, full["mask"].shape)
full_masks = {"SAM, one box": full_sam, f"linear head on DINOv3\nfeatures @ {hi} px": full_head}
full_s = {name: score(m, full) for name, m in full_masks.items()}
show_masks(
    [(f"whole fundus · {meta['name']}", full["image"])]
    + [
        (pred_title(name, m, full, full_s[name]), overlay(full["image"], m, THEME["fails"]))
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
