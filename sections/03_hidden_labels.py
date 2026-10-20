# %% [markdown]
# ## 3. Hide the labels: the naive baseline
#
# **Objective:** train a head with "missing = background" and watch it fail differently.
#
# A perfect vessel map of one fundus takes an expert 3 to 5 hours, so real label files are rarely complete.
# Picture an annotator who traced the big vessels, ran out of time halfway through the tree, and saved the
# file. Nothing in that file says "I stopped here": every pixel they did not paint is stored as background,
# exactly like the pixels that really are background. In this section we make such a file ourselves, by
# hiding about half of the vessels from the ground truth, train the section-2 head on it the obvious way,
# and score it on the vessels we hid. From here on the full masks are used for scoring only.
#
# <table width="100%" bgcolor="#EDE7F6"><tr><td><font color="#311B92">📘 <b>Theory: incomplete
# supervision.</b> The rest of the notebook trains on cheap labels made from the full ground truth, which we
# keep hidden for evaluation. <b>Missing labels</b> (this section and section 5): a share of the objects is
# fully labelled, the rest is silently left out. <b>Scribbles</b> (section 4): a few strokes on some
# objects and some background, everything else unlabelled. Both leave pixels nobody looked at. The question
# of this block is what a model should believe about those pixels. The obvious answer, and the one you get
# by passing the label file straight into a loss, is "background".</font></td></tr></table>
#
# This helper cell defines how we hide labels, the same way `scripts/prepare_data.py` built the cached
# files: cut the ground-truth skeleton into segments at its junctions (connected objects when the structures
# are not thin), give every foreground pixel to its nearest segment, then hide a random share of the
# segments.

# %%
# @title Missing labels: cut into segments, hide a share { display-mode: "form" }
import zlib

from scipy import ndimage as ndi


def label_units(mask: np.ndarray, thin: bool) -> np.ndarray:
    """Label map of the units we hide (0 = background): skeleton segments for thin structures or a single
    object, connected components otherwise. Every foreground pixel gets the label of its nearest segment."""
    lab, n = ndi.label(mask)
    if not thin and n >= 2:
        return lab
    sk = skeleton(mask)
    neighbours = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    seg, _ = ndi.label(sk & ~(sk & (neighbours > 2)), structure=np.ones((3, 3)))  # cut at junctions
    if seg.max() == 0:
        return np.where(mask, 1, 0)
    _, (iy, ix) = ndi.distance_transform_edt(seg == 0, return_indices=True)
    return np.where(mask, seg[iy, ix], 0)


def hide_units(mask: np.ndarray, fraction: float, rng: np.random.Generator, thin: bool) -> np.ndarray:
    """The 'missing labels' file: the ground truth with ~fraction of its units removed. Returns the visible
    mask; the hidden foreground is mask & ~visible."""
    lab = label_units(mask, thin)
    ids = np.arange(1, int(lab.max()) + 1)
    rng.shuffle(ids)
    k = int(round(len(ids) * fraction))
    hidden_ids = ids[: max(1, k)] if len(ids) >= 2 else ids[:0]
    return mask & ~np.isin(lab, hidden_ids)


def hide_rng(name: str) -> np.random.Generator:
    """One fixed random generator per image, so the same segments are hidden on every run (N-013)."""
    return np.random.default_rng([SEED, zlib.crc32(f"{name}/hide".encode())])


def part_recall(preds: list[np.ndarray], items: list[dict], part: str) -> float:
    """Mean over images of the share of item[part] ("mask", "visible", "hidden"), inside the evaluation
    region, that the prediction marks. Uses `hidden_recall` from 3.2."""
    pairs = zip(preds, items, strict=True)
    return float(np.mean([hidden_recall(p, it[part] & region(it)) for p, it in pairs]))


def two_colour(image: np.ndarray, good: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Overlay `good` in violet ("works") and `bad` in orange ("fails") on one image."""
    return overlay(overlay(image, good, THEME["works"]), bad, THEME["fails"])


# %% [markdown]
# ### 3.1 Make a half-finished label file
#
# This cell takes the five specialised training images of section 2.5, hides about half of each image's
# vessel segments (the share comes from the dataset config), and shows one of them: the full truth we keep
# for scoring, the labels training may use, and what the label file now wrongly calls background.

# %%
MISSING = SPECIALISED["missing_labels"]  # {"hidden_fraction": 0.5, "unit": ...}: never hard-coded here
miss_train = [load_item(SPECIALISED, it) for it in SPECIALISED["fit"][:N_TRAIN_SPEC]]
same_as_cache = []
for it in miss_train:
    rng = hide_rng(it["name"])
    visible = hide_units(it["mask"], MISSING["hidden_fraction"], rng, SPECIALISED["thin_structures"])
    same_as_cache.append(float((visible == it["visible"]).mean()))
    it["visible"], it["hidden"] = visible, it["mask"] & ~visible  # from now on: what training may see
hidden_share = float(np.mean([it["hidden"].sum() / max(it["mask"].sum(), 1) for it in miss_train]))
print(
    f"{len(miss_train)} training images · {hidden_share:.0%} of the vessel pixels hidden "
    f"(asked for {MISSING['hidden_fraction']:.0%} of the segments) · pixels identical to the cached labels "
    f"our checkpoints used: {np.mean(same_as_cache):.2%}"
)

ex = miss_train[0]
show_masks(
    [
        (ex["name"], ex["image"]),
        ("ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
        ("labels training sees: vessel", overlay(ex["image"], ex["visible"], THEME["scribble"])),
        ('hidden: the file says "background"', overlay(ex["image"], ex["hidden"], THEME["fails"])),
    ],
    width=2.8,
)

# %% [markdown]
# **Q10.** 🟡 Look at the third panel on its own. If a colleague handed you that label file, could you tell
# that half of it is missing? What would you need to know to tell?
#
# <details><summary>What to look for</summary>
#
# Usually not. The visible vessels are clean and correct, they simply stop, and a vessel tree that stops
# looks plausible. The file has two values, vessel and background, and no third value for "nobody looked
# here". To tell, you would need to know how the file was made (who annotated, for how long, what they were
# told to skip), which is exactly the information that gets lost. That is why we hid the labels on purpose:
# here we know which pixels are missing, so we can measure what the model does with them.
#
# </details>
#
# ### 3.2 A metric for the missing vessels: recall
#
# <table width="100%" bgcolor="#EDE7F6"><tr><td><font color="#311B92">📘 <b>Theory: recall and
# precision.</b> <b>Recall</b> is the share of the true vessel pixels that the prediction marks: it asks
# "of what is there, how much did you find?". <b>Precision</b> is the share of the marked pixels that are
# true vessels: "of what you marked, how much is right?". IoU mixes the two into one number. Here we want
# them apart, because the question is specific: of the vessels we hid, which the label file calls
# background, how many does the trained head still find? That is recall on the hidden foreground.
# </font></td></tr></table>
#
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your turn (optional, 1 min).</b>
# Fill <code>my_hidden_recall</code>: a boolean prediction and a boolean mask of hidden pixels in, one
# float out, the share of the hidden pixels the prediction marks. Return 0.0 when nothing is hidden. The
# cell checks your version on two play masks and continues with ours if you skip it.</font></td></tr></table>

# %%
# ✍️ YOUR CODE
def my_hidden_recall(pred: np.ndarray, hidden: np.ndarray) -> float:
    """Share of the `hidden` pixels that `pred` marks as foreground; 0.0 when `hidden` is empty."""
    ...


def _ref_hidden_recall(pred: np.ndarray, hidden: np.ndarray) -> float:
    """Reference: the prediction's mean over the hidden pixels."""
    hidden = np.asarray(hidden, bool)
    return float(np.asarray(pred, bool)[hidden].mean()) if hidden.any() else 0.0


_play_hidden = np.zeros((8, 8), bool)
_play_hidden[2, :] = True  # a hidden vessel, 8 px long
_play_pred = np.zeros((8, 8), bool)
_play_pred[1:4, :2] = True  # the prediction finds its first 2 px, plus some background
hidden_recall = use(
    my_hidden_recall,
    reference=_ref_hidden_recall,
    check=lambda f: [round(float(f(_play_pred, h)), 4) for h in (_play_hidden, np.zeros((8, 8), bool))],
    name="hidden_recall",
)
print(f"play example · recall on the hidden vessel: {hidden_recall(_play_pred, _play_hidden):.2f}")

# %% [markdown]
# ### 3.3 The naive baseline: missing = background
#
# The naive recipe is what you get by passing the label file straight to the textbook loss: every pixel
# not painted as vessel is a background target, and plain binary cross-entropy counts every pixel equally.
# Two things change from section 2.5, the labels and the loss weighting; 3.4 separates them. Everything else
# is the same: the same five images, the same DINOv3 features, the same linear head, the same steps.
#
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your prediction.</b> The head
# was told that half of the vessels in these five images are background. On those hidden vessels, will its
# recall be near the section-2 head's, about half of it, or close to zero? And on new images, will it mark
# more pixels than the section-2 head, or fewer? Write two guesses.</font></td></tr></table>
#
# This cell trains the naive head on the incomplete labels only (the fundus border is ignored, as in 2.5)
# and prints the seconds it took. Not done in a minute, or on a CPU and in a hurry? Set
# `TRAIN_YOURSELF = False` and run it again: it loads our checkpoint, trained the same way on all fit images.

# %%
TRAIN_YOURSELF = True  # False -> load our precomputed "hidden_bg" head instead of training
if not isinstance(TRAIN_YOURSELF, bool):
    raise SystemExit("TRAIN_YOURSELF must be True or False.")

side = SPECIALISED["feat_side"]
miss_feats = [features_for(SPECIALISED, it, side) for it in miss_train]
# missing = background: the label file as it is. The hidden ground truth never enters training (L-005).
naive_labels = [np.where(region(it), it["visible"], -1).astype(np.int8) for it in miss_train]
if TRAIN_YOURSELF:
    naive_head, naive_sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=False)
    src = f"trained here on {len(miss_train)} images, {HEAD_STEPS} steps, plain cross-entropy"
else:
    naive_head, entry = load_head(SPECIALISED, "hidden_bg", side)
    naive_sec = entry["train_sec"]
    src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
print(f"naive head @ {side} px · {src} · {naive_sec:.1f} s on {DEVICE}")
if naive_sec > 60:
    print("ℹ️ Slow session? Set TRAIN_YOURSELF = False above to load our checkpoint next time.")

# %% [markdown]
# This cell scores the naive head and the section-2 head (full masks) on three groups of vessel pixels: the
# ones we hid in the training images, the ones training saw, and every vessel of the held-out images. It
# prints recall per group plus IoU, clDice and precision on the held-out images, and plots the recalls.

# %%
full_head = spec_heads[side][0]  # section 2.5: same features, trained on full masks
HEADS = {"full masks (section 2)": full_head, "naive: missing = background": naive_head}
t0 = time.monotonic()
naive_report = {}
for name, tensors in HEADS.items():
    fit_preds = [
        predict_head(tensors, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
    ]
    held_preds = [
        predict_head(tensors, features_for(SPECIALISED, it, side), it["mask"].shape) for it in SPEC_EVAL
    ]
    held = mean_scores([score(p, it) for p, it in zip(held_preds, SPEC_EVAL, strict=True)])
    # precision = the share of the marked pixels that are true vessels
    held_pairs = zip(held_preds, SPEC_EVAL, strict=True)
    precision = [hidden_recall(it["mask"], p & region(it)) for p, it in held_pairs]
    naive_report[name] = {
        "recall_hidden_fit": part_recall(fit_preds, miss_train, "hidden"),
        "recall_visible_fit": part_recall(fit_preds, miss_train, "visible"),
        "recall_held": part_recall(held_preds, SPEC_EVAL, "mask"),
        "precision_held": float(np.mean(precision)),
        "iou_held": held["iou"],
        "cldice_held": held["cldice"],
        "fit_preds": fit_preds,
        "held_preds": held_preds,
    }
infer_sec = round(time.monotonic() - t0, 1)

naive = naive_report["naive: missing = background"]
RESULTS["naive_recall_hidden_fg"] = naive["recall_hidden_fit"]  # recall on the vessels hidden in training
RESULTS["naive_head"] = {k: v for k, v in naive.items() if not k.endswith("_preds")} | {
    "long_side": side,
    "n_train": len(miss_train),
    "hidden_share": hidden_share,
    "train_sec": round(float(naive_sec), 2),
    "infer_sec": infer_sec,
}

GROUPS = {
    "recall_hidden_fit": "hidden vessels\n(told: background)",
    "recall_visible_fit": "visible vessels\n(told: vessel)",
    "recall_held": "held-out images\n(all vessels)",
}
print(f"{'head':<30}{'recall: hidden':>15}{'visible':>9}{'held-out':>10}   held-out IoU / clDice / precision")
for name, r in naive_report.items():
    print(
        f"{name:<30}{r['recall_hidden_fit']:>15.3f}{r['recall_visible_fit']:>9.3f}{r['recall_held']:>10.3f}"
        f"   {r['iou_held']:.3f} / {r['cldice_held']:.3f} / {r['precision_held']:.3f}"
    )

x = np.arange(len(GROUPS))
fig, ax = plt.subplots(figsize=(6.4, 3))
for shift, name, colour in zip((-0.2, 0.2), HEADS, (THEME["works"], THEME["fails"]), strict=True):
    ax.bar(x + shift, [naive_report[name][g] for g in GROUPS], 0.4, color=colour, label=name)
ax.set_xticks(x, list(GROUPS.values()), fontsize=8)
ax.set_ylabel("recall (share of vessel pixels found)")
ax.set_ylim(0, 1)
ax.set_title("Which vessels does each head still find?")
ax.legend(fontsize=8, frameon=False, loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# <table width="100%" bgcolor="#F3E5F5"><tr><td><font color="#4A148C">❓ <b>Question.</b> Run the next
# cell: it shows where the naive head looks for vessels, first on a training image, then on the held-out
# demo image of section 2.</font></td></tr></table>
#
# **Q11.** 🟡 On the training image, which hidden vessels did the naive head miss (orange), and which did it
# still find (violet)? On the held-out image, compare it with the section-2 head: does it fail the same way
# (too wide, the disc, the whole eye) or differently? Try another training image:
#
# ```python
# FAIL_IMAGE = 0   # try 1, 2, ...
# ```
#
# This cell draws the failure figure from the predictions you already have: no new training, no model calls.

# %%
# Q11 — where the naive head fails: a training image, then the held-out demo image
FAIL_IMAGE = 0  # try 1, 2, ...
if not isinstance(FAIL_IMAGE, int) or not 0 <= FAIL_IMAGE < len(miss_train):
    raise SystemExit(f"FAIL_IMAGE must be a whole number from 0 to {len(miss_train) - 1}.")
it, pred = miss_train[FAIL_IMAGE], naive["fit_preds"][FAIL_IMAGE]
found, missed = it["hidden"] & pred, it["hidden"] & ~pred
show_masks(
    [
        ("labels training saw", overlay(it["image"], it["visible"], THEME["scribble"])),
        ("naive head", overlay(it["image"], pred, THEME["fails"])),
        (
            f"hidden vessels · found {hidden_recall(pred, it['hidden']):.0%}",
            two_colour(it["image"], found, missed),
        ),
    ],
    width=3.0,
)
k = next(i for i, e in enumerate(SPEC_EVAL) if e is demo)
show_masks(
    [
        (
            f"{name} · IoU {score(r['held_preds'][k], demo)['iou']:.2f}",
            overlay(demo["image"], r["held_preds"][k], THEME["fails"]),
        )
        for name, r in naive_report.items()
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"]))],
    width=3.0,
)

# %% [markdown]
# <details><summary>What to look for</summary>
#
# The naive head did what the labels told it. Half of the vessels it saw were labelled background, so a
# vessel-looking patch became, for this head, more likely background than not, and plain cross-entropy
# turns "less than 50 %" into "background". It keeps the clearest, thickest vessels and drops the rest: on
# the hidden segments, on new images, and often even on the visible segments it was shown. Compare the
# precision column: what it does mark is mostly right. The section-2 head failed by being too generous
# (vessels drawn as wide as a patch); this one fails by being too stingy. Now look at the held-out IoU of
# the two heads: it can be close, because IoU rewards the precision the naive head gained as much as it
# punishes the recall it lost. One number would have hidden the failure. Same features, same head: the only
# thing that changed is what we told it about the pixels nobody labelled.
#
# </details>
#
# <table width="100%" bgcolor="#FDECEA"><tr><td><font color="#8B1A10">⚠️ <b>Pitfall: a precise model looks
# trustworthy.</b> Every vessel the naive head marks is a real vessel, so a quick look at a few predictions,
# or a precision score, says "works". The failure is in what is <i>not</i> marked, and you can only measure
# that against labels that are complete. On your own data, keep a small, fully labelled evaluation set, even
# if training uses cheap labels.</font></td></tr></table>
#
# ### 3.4 🔴 Optional: is it the labels, or the loss?
#
# Plain cross-entropy is one of the two changes from section 2.5. The heads of sections 1 and 2 used a
# <i>balanced</i> loss, which gives the vessel pixels as much total weight as the background pixels.
#
# **Q12.** 🔴 Train the same naive labels with the balanced loss of section 2.5 and compare its recall on
# the hidden vessels with the plain head's. Does reweighting fix "missing = background"? What did it fix?
#
# ```python
# head, sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=True)
# ```
#
# This cell trains that balanced variant on the same incomplete labels (seconds) and prints its recall on
# the three groups next to the plain head's.

# %%
# Q12 — the same "missing = background" labels with the balanced loss of section 2.5
balanced_head, balanced_sec = train_head(miss_feats, naive_labels, HEAD_STEPS, balanced=True)
bal_fit = [
    predict_head(balanced_head, f, it["mask"].shape) for f, it in zip(miss_feats, miss_train, strict=True)
]
bal_hidden, bal_visible = (part_recall(bal_fit, miss_train, part) for part in ("hidden", "visible"))
RESULTS["naive_balanced_recall_hidden_fg"] = bal_hidden
print(f"balanced head · trained in {balanced_sec:.1f} s on {DEVICE}")
print(f"{'loss':<12}{'recall: hidden':>15}{'visible':>9}")
print(f"{'plain':<12}{naive['recall_hidden_fit']:>15.3f}{naive['recall_visible_fit']:>9.3f}")
print(f"{'balanced':<12}{bal_hidden:>15.3f}{bal_visible:>9.3f}")

# %% [markdown]
# <details><summary>What to look for</summary>
#
# Reweighting usually recovers a lot of the hidden vessels. It does not fix the labels, though: it fixes
# the <i>proportion</i>. Missing = background told the head that vessels are about half as common as they
# are, and the balanced loss stops caring about proportions, so the hidden vessels come back as a side
# effect. It works here for two reasons that need not hold on your data: we hid segments <i>at random</i>,
# so the hidden ones look like the visible ones, and a linear head is too simple to memorise which segment
# was hidden. If annotators systematically skip the thin capillaries, the head learns "thin = background",
# and no class weight undoes that. Section 5 fixes the cause instead: tell the loss which pixels are
# unknown.
#
# </details>
#
# <table width="100%" bgcolor="#E8EAF6"><tr><td><font color="#1A237E">🔀 <b>Across approaches.</b> Block 1
# in one line each. Pets: the foundation model works out of the box. Specialised task: SAM looks for objects
# and finds the eye; the head with perfect masks is limited by its 16 px patches. Incomplete labels, used
# naively: the head believes the label file and learns that vessels are background. Three different
# failures, and none of them is fixed by "a bigger model". Keep this section's recall on the hidden vessels:
# it is the baseline every approach after the break has to beat.</font></td></tr></table>
#
# ### ☕ Before the break
#
# <table width="100%" bgcolor="#F3E5F5"><tr><td><font color="#4A148C">❓ <b>Question for the coffee
# break.</b> No code, just think about it over coffee.</font></td></tr></table>
#
# **Q13.** 🟡 Perfect labels cost hours per image, and the cheap ones lie: the label file says "background"
# wherever nobody looked. Without buying more labels, what could you tell the training about a pixel nobody
# labelled? Think of at least two options before section 4.
#
# <details><summary>A hint, for after the break</summary>
#
# Three families of answers come back in block 2. (1) Say "unknown" instead of "background", and let the
# loss skip those pixels (sections 4 and 5). (2) Guess the missing labels with another model, for example
# SAM prompted from the labels you have, and train on the guesses (section 6). (3) Skip training and give
# the cheap labels to the foundation model directly as prompts (section 7).
#
# </details>
