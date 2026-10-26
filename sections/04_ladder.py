# %% [markdown]
# ## 4. The adaptation ladder
#
# In section 3 the head believed a label file that said "background" wherever nobody had looked. Now the ways
# out. The first is a ladder: give the foundation model a little more task information at each rung, and look
# at what changes. Higher is not automatically better and each rung fails in its own way, so we look at the
# masks, not only at the numbers.

# %% [markdown]
# <!-- callout: objective | Objective -->
# zero-shot → scribbles: see how predictions change when the model gets more task information. A third rung,
# one annotated image, waits in a go-deeper box at the end.

# %% [markdown]
# The next cell draws what the model receives at each rung, with the time that annotation costs.
#
# What to look for: the input is the only thing that changes, and one complete mask costs hours where a few
# strokes cost minutes.

# %%
# @title What each rung receives { display-mode: "form" }
def effort_minutes(n_strokes: int = 0, n_full_masks: int = 0) -> float:
    """Annotation minutes under the dataset's stated assumptions (seconds per stroke, hours per full mask)."""
    cost = specialised["annotation_effort"]
    return n_strokes * cost["seconds_per_stroke"] / 60 + n_full_masks * 60 * float(
        np.mean(cost["hours_per_full_mask"])
    )


_shot0 = load_item(specialised, specialised["fit"][0])
_scribs = [load_item(specialised, m) for m in specialised["fit"][:N_TRAIN_SPEC]]
_n_str = sum(m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in specialised["fit"][:N_TRAIN_SPEC])


def _cost(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.0f} h"


_ex = _scribs[0]
_thick = {v: ndi.binary_dilation(_ex["scribbles"] == v, iterations=2) for v in (1, 2)}
_strokes = overlay(_ex["image"], _thick[2], THEME["background"], 0.9)
_strokes = overlay(_strokes, _thick[1], THEME["scribble"], 0.9)
_c0, _c1 = _cost(effort_minutes(n_strokes=1)), _cost(effort_minutes(n_full_masks=1))
_c2 = _cost(effort_minutes(n_strokes=_n_str))
show_masks(
    [
        (f"Rung 0 · zero-shot\n1 image + 1 box · ≈ {_c0}", demo["image"]),
        (
            f"Rung 1 · one-shot (go-deeper)\n1 image + its complete mask · ≈ {_c1}",
            overlay(_shot0["image"], _shot0["mask"], THEME["scribble"]),
        ),
        (f"Rung 2 · scribbles\n{len(_scribs)} images + {_n_str} strokes · ≈ {_c2}", _strokes),
    ],
    prompts=[{"box": box_prompt(demo)}, None, None],
    width=3.4,
    legend=[
        ("teal: what the person draws (box, mask, vessel strokes)", THEME["scribble"]),
        ("grey: background strokes", THEME["background"]),
    ],
    note="The model gets only these marks and nothing else about the task.",
)

# %% [markdown]
# <!-- callout: theory | Theory: what each rung tells the model -->
# - **Rung 0 · zero-shot** (section 2): SAM gets the image and one box, and nothing about the task. It
#   returns its best guess of "the object": on the retina, the eye or one thick vessel.
# - **Rung 2 · scribbles:** strokes on some vessels and some background in a few images; everything else is
#   *unlabelled*. The section-2 head is trained with a loss that counts only the labelled pixels (4.2).
# - **Rung 1 · one-shot:** one image with a complete vessel mask. It is the go-deeper box at the end of the
#   section; its row joins the table when you run it.
# - **Cost:** one complete mask takes hours, one scribble seconds.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper: where the rungs come from -->
# - Partial cross-entropy on scribbles: Tang et al., ECCV 2018.
# - The recipe of the MIT "DINOv2 + scribbles" work, here on DINOv3.
# - Prototypes from one example: PerSAM, Zhang et al., ICLR 2024.

# %% [markdown]
# This helper defines the ladder's tools: one colour per approach, prototypes, error maps and a report.

# %%
# @title Ladder helpers: colours, prototypes, error maps, report { display-mode: "form" }
approach_colours = {  # one colour per approach (L-003); later sections add theirs and keep these
    "zero_shot": "#E65100",
    "one_shot": "#C2185B",
    "scribbles": "#3949AB",
}
approach_names = {
    "zero_shot": "zero-shot: SAM, one box",
    "one_shot": "one-shot: nearest prototype",
    "scribbles": "scribbles: partial cross-entropy",
}


def patch_share(mask: np.ndarray, grid: tuple[int, int]) -> torch.Tensor:
    """Share of each feature patch covered by a boolean pixel mask: [h, w] floats in [0, 1]."""
    m = torch.from_numpy(np.asarray(mask, np.float32))[None, None]
    return F.interpolate(m, size=grid, mode="area")[0, 0].to(DEVICE)


def prototypes(feat: torch.Tensor, w_fg: torch.Tensor, w_bg: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Foreground and background prototypes: the weighted means of the unit-length patch features."""
    f = F.normalize(feat.flatten(1), dim=0)  # [D, h*w]
    return tuple(F.normalize(f @ w.flatten().to(f), dim=0) for w in (w_fg, w_bg))


@torch.no_grad()
def prototype_score(feat: torch.Tensor, protos: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Per patch: cosine similarity to the foreground prototype minus to the background one ([h, w])."""
    return torch.einsum("dhw,d->hw", F.normalize(feat, dim=0), protos[0] - protos[1])


def score_to_mask(s: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Upsample a patch score map to the image and mark the pixels nearer the foreground prototype."""
    up = F.interpolate(s[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
    return (up > 0).cpu().numpy()


def heat_rgb(s: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """A score map as RGB: dark violet = like the foreground prototype, light = like the background."""
    up = F.interpolate(s[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    up = 0.5 + 0.5 * up / (np.abs(up).max() + 1e-9)
    return (mpl.colormaps["Purples"](up)[..., :3] * 255).astype(np.uint8)


def error_map(image: np.ndarray, pred: np.ndarray, item: dict) -> np.ndarray:
    """Inside the evaluation region: vessels found in violet, false alarms in orange, missed vessels in
    green (the ground truth the prediction did not cover); outside it, light grey (not scored)."""
    r = region(item)
    truth, p = item["mask"] & r, pred & r
    out = overlay(image, p & truth, THEME["works"], unscored=~r)
    out = overlay(out, p & ~truth, THEME["fails"])
    return overlay(out, truth & ~p, THEME["truth"])


def rung_title(label: str, pred: np.ndarray, item: dict) -> str:
    """Panel title: what made the mask, then IoU, recall and the share of the fundus marked."""
    r = region(item)
    truth, p = item["mask"] & r, pred & r
    numbers = f"IoU {iou(p, truth):.2f} · recall {hidden_recall(pred, truth):.2f} · marked {p[r].mean():.0%}"
    return f"{label}\n{numbers}"


def image_numbers(pred: np.ndarray, item: dict) -> dict:
    """IoU, recall, precision and marked share of one prediction on one image."""
    r = region(item)
    p = pred & r
    return {
        "iou": iou(p, item["mask"] & r),
        "recall": hidden_recall(pred, item["mask"] & r),
        "precision": hidden_recall(item["mask"], p),
        "marked": float(p[r].mean()),
    }


def rung_report(preds: list[np.ndarray]) -> dict:
    """Mean IoU, clDice, recall, precision and marked share of the evaluation region, over spec_eval."""
    pairs = tqdm(list(zip(preds, spec_eval, strict=True)), desc="scoring", leave=False)
    out = mean_scores([score(p, it) for p, it in pairs])
    out["recall"] = part_recall(preds, spec_eval, "mask")
    pairs = zip(preds, spec_eval, strict=True)
    out["precision"] = float(np.mean([hidden_recall(it["mask"], p & region(it)) for p, it in pairs]))
    out["marked"] = float(np.mean([p[region(it)].mean() for p, it in zip(preds, spec_eval, strict=True)]))
    return out


# %% [markdown]
# ### 4.1 Rung 0: zero-shot, from section 2
#
# Nothing new to run: the next cell measures section 2's SAM-with-one-box masks with the numbers every rung
# is compared on (the table in 4.3 explains each one).

# %%
# @title Score rung 0 { display-mode: "form" }
k_demo = next(i for i, it in enumerate(spec_eval) if it is demo)  # the section-2 demo image
ladder_preds = {"zero_shot": sam_box_masks}  # section 2.3: SAM, one box, per evaluation image
ladder_report = {"zero_shot": rung_report(sam_box_masks)}
v = ladder_report["zero_shot"]
print(
    f"zero-shot (SAM, one box) on {len(spec_eval)} held-out images · IoU {v['iou']:.3f} · recall "
    f"{v['recall']:.3f} · precision {v['precision']:.3f} · marked {v['marked']:.0%} of the fundus"
)

# %% [markdown]
# ### 4.2 Rung 2: scribbles
#
# The annotator from section 3 again, with a new instruction: "don't trace everything; draw a few strokes
# along some vessels and a few on clear background, in each of five images, then stop". Every other pixel is
# *unlabelled*, and this time the stroke map says so, so the loss can skip it.
#
# In the stroke map 0 = unlabelled, 1 = vessel stroke, 2 = background stroke. In the training targets
# −1 = ignored.

# %% [markdown]
# The next cell shows the strokes on one training image and, in the third picture, what the loss will see.
#
# What to look for: almost the whole picture is dark, so the loss ignores almost every pixel.

# %%
# @title Show the scribbles of one training image { display-mode: "form" }
scrib_train = [load_item(specialised, it) for it in specialised["fit"][:N_TRAIN_SPEC]]
labelled = float(np.mean([(it["scribbles"] > 0)[region(it)].mean() for it in scrib_train]))
scrib_meta = specialised["fit"][:N_TRAIN_SPEC]
n_strokes = sum(m.get("n_fg_strokes", 0) + m.get("n_bg_strokes", 0) for m in scrib_meta)
print(
    f"{len(scrib_train)} training images · {n_strokes} strokes · {labelled:.2%} of the fundus pixels "
    f"labelled · ≈ {effort_minutes(n_strokes=n_strokes):.1f} min at "
    f"{specialised['annotation_effort']['seconds_per_stroke']} s per stroke"
)
ex = scrib_train[0]
thick = {v: ndi.binary_dilation(ex["scribbles"] == v, iterations=2) for v in (1, 2)}
strokes = overlay(overlay(ex["image"], thick[2], THEME["background"], 0.9), thick[1], THEME["scribble"], 0.9)
counted = ndi.binary_dilation(ex["scribbles"] > 0, iterations=2)  # thickened for display only
loss_sees = np.where(counted[..., None], strokes, (ex["image"] * 0.25).astype(np.uint8))
show_masks(
    [
        (ex["name"], ex["image"]),
        ("scribbles: vessel · background", strokes),
        ("what the loss sees", loss_sees),
        ("ground truth · evaluation only", overlay(ex["image"], ex["mask"], THEME["truth"])),
    ],
    width=2.8,
    legend=[
        ("teal: vessel stroke", THEME["scribble"]),
        ("grey: background stroke", THEME["background"]),
        ("dark: pixels the loss ignores", "#404040"),
        "truth",
    ],
    note="Strokes are thickened for the figure only.",
)

# %% [markdown]
# <!-- callout: theory | Theory: partial cross-entropy -->
# The loss of section 2 adds a cross-entropy term for every pixel. The partial version adds it only over
# the set S of pixels a person labelled:
#
# `L = −(1/|S|) · Σ_{i∈S} [ y_i·log p_i + (1−y_i)·log(1−p_i) ]`
#
# - A vessel stroke says y = 1, a background stroke says y = 0.
# - An unlabelled pixel is not in S: it adds nothing. In code it is a target of `-1`, which `train_head`
#   already skips (it has skipped the fundus border that way since section 2).
# - In section 3 the unpainted pixels were targets of `0`: a thin vessel nobody painted counted as a
#   "background" error.

# %% [markdown]
# This helper turns a stroke map into training targets.

# %%
# @title scribble_targets: stroke map → training targets { display-mode: "form" }
def scribble_targets(scribbles: np.ndarray, region: np.ndarray, unlabelled: int = -1) -> np.ndarray:
    """Targets (int8) from a stroke map: 1 vessel stroke, 0 background stroke, `unlabelled` everywhere else
    (-1 = ignored, 0 = background); -1 outside `region`."""
    targets = np.full(scribbles.shape, unlabelled, np.int8)
    targets[scribbles == 2] = 0
    targets[scribbles == 1] = 1
    targets[~np.asarray(region, bool)] = -1
    return targets


# %% [markdown]
# This helper trains the section-2 head on the strokes, or loads our checkpoint.

# %%
# @title Scribble-head helper: train on the strokes, or load ours { display-mode: "form" }
def train_scribble_head(train_yourself: bool, unlabelled: int):
    """The scribble head of rung 2. Returns the head's tensors, the training seconds and the DINOv3
    features of the training images (section 6 trains on them too)."""
    if not isinstance(train_yourself, bool):
        raise SystemExit("TRAIN_SCRIBBLE_HEAD must be True or False.")
    if unlabelled not in (-1, 0):
        raise SystemExit("UNLABELLED_TARGET must be -1 (ignore) or 0 (background).")
    side = specialised["feat_side"]
    feats = [features_for(specialised, it, side) for it in scrib_train]
    # the rung's supervision: strokes only; unlabelled pixels and the border are ignored (-1) by default
    labels = [scribble_targets(it["scribbles"], region(it), unlabelled) for it in scrib_train]
    if unlabelled != -1 and not train_yourself:
        print("ℹ️ Our checkpoint ignores unlabelled pixels, so with 0 the head is trained here.")
        train_yourself = True
    if train_yourself:
        head, sec = train_head(feats, labels, HEAD_STEPS)
        what = "ignored" if unlabelled == -1 else "background"
        src = f"trained here on {len(scrib_train)} images, {HEAD_STEPS} steps, unlabelled = {what}"
    else:
        head, entry = load_head(specialised, "scribbles", side)
        sec = entry["train_sec"]
        src = f"our checkpoint {entry['file']} ({len(entry['train_names'])} images)"
    print(f"scribble head @ {side} px · {src} · {sec:.1f} s on {DEVICE}")
    if sec > 60:
        print("ℹ️ Slow session? Set TRAIN_SCRIBBLE_HEAD = False above to load our checkpoint next time.")
    return head, sec, feats


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The head will see labels on fewer than two pixels in a hundred (see the printout above), and no vessel
# will ever be called "background". Write two guesses:
#
# - Will it find more of the held-out vessels (recall) than SAM's box, or fewer?
# - Will it mark more of the fundus, or less?

# %% [markdown]
# <!-- callout: your-turn | Your turn: unlabelled pixels, ignored or background? -->
# - The next cell trains the section-2 head on the strokes of the five training images. `UNLABELLED_TARGET`
#   says what an unlabelled pixel counts as: **-1** ignored (the default), **0** background.
# - Predict first: what happens to the vessels nobody stroked if you pick 0? (Section 3 has the answer.)
# - Run it with -1 and read the table in 4.3. If you have time, pick 0, run the cells down to that table
#   again, and compare.
# - `TRAIN_SCRIBBLE_HEAD = False` loads our checkpoint, which used -1. With 0 the head is always trained here.

# %%
# @title Train the scribble head on the strokes { display-mode: "form" }
TRAIN_SCRIBBLE_HEAD = True  # @param {type:"boolean"}
UNLABELLED_TARGET = -1  # @param [-1, 0] {type:"raw"}
scrib_head, scrib_sec, scrib_feats = train_scribble_head(TRAIN_SCRIBBLE_HEAD, UNLABELLED_TARGET)

# %% [markdown]
# This cell scores the scribble head on the evaluation images with the same numbers as rung 0.

# %%
# @title Score the scribble head on the evaluation images { display-mode: "form" }
t0 = time.monotonic()
side = specialised["feat_side"]
ladder_preds["scribbles"] = [
    predict_head(scrib_head, features_for(specialised, it, side), it["mask"].shape) for it in spec_eval
]
ladder_report["scribbles"] = rung_report(ladder_preds["scribbles"])
scrib_infer = round(time.monotonic() - t0, 1)
v = ladder_report["scribbles"]
print(
    f"scribble head · IoU {v['iou']:.3f} · recall {v['recall']:.3f} · precision {v['precision']:.3f} · "
    f"marked {v['marked']:.0%} of the fundus · {scrib_infer} s on {DEVICE}"
)

# %% [markdown]
# ### 4.3 The ladder, side by side
#
# The next cell prints one row per rung with the annotation it needed, then draws each rung's error map on
# the demo image. The numbers:
#
# - **IoU:** overlap of the mask with the true vessels.
# - **Recall:** of the true vessel pixels, the share the mask finds.
# - **Precision:** of the pixels the mask marks, the share that really are vessel.
# - **Marked:** the share of the fundus (the scored area) the mask covers.
#
# What to look for: which rung misses vessels ($\color{#2E7D32}{\blacksquare}$ green) and which invents them
# ($\color{#E65100}{\blacksquare}$ orange).

# %%
# @title Table and error maps of the rungs { display-mode: "form" }
effort = {
    "zero_shot": effort_minutes(n_strokes=len(spec_eval)),  # one box per image you segment
    "one_shot": effort_minutes(n_full_masks=1),
    "scribbles": effort_minutes(n_strokes=n_strokes),
}
effort_text = {
    "zero_shot": f"1 box per image ({len(spec_eval)} here)",
    "one_shot": "1 complete mask",
    "scribbles": f"{n_strokes} strokes, {len(scrib_train)} images",
}
RESULTS["ladder"] = {k: ladder_report[k]["iou"] for k in ("zero_shot", "scribbles")}  # one_shot: go-deeper
RESULTS["ladder_detail"] = dict(ladder_report)
RESULTS["ladder_effort_min"] = {k: round(v, 2) for k, v in effort.items()}
RESULTS["scribbles_head"] = ladder_report["scribbles"] | {
    "long_side": side,
    "n_train": len(scrib_train),
    "n_strokes": n_strokes,
    "labelled_fraction": labelled,
    "train_sec": round(float(scrib_sec), 2),
    "infer_sec": scrib_infer,
}

print(f"{'approach':<36}{'IoU':>6}{'recall':>8}{'precis.':>8}{'marked':>8}   annotation (≈ minutes)")
for k, v in ladder_report.items():
    print(
        f"{approach_names[k]:<36}{v['iou']:>6.3f}{v['recall']:>8.3f}{v['precision']:>8.3f}"
        f"{v['marked']:>8.0%}   {effort_text[k]} (≈ {effort[k]:.1f})"
    )
print(f"{'everything is foreground':<36}{RESULTS['specialised_everything_iou']:>6.3f}")

show_masks(
    [
        (
            rung_title(approach_names[k].split(":")[0], ladder_preds[k][k_demo], demo),
            error_map(demo["image"], ladder_preds[k][k_demo], demo),
        )
        for k in ladder_preds
    ]
    + [("ground truth", overlay(demo["image"], demo["mask"], THEME["truth"], unscored=~region(demo)))],
    width=3.2,
    legend=["found", "invented", "missed", "unscored"],
    note="Each rung's mask on the demo image, compared with the ground truth.",
)

# %% [markdown]
# **Q9.** 🟡 Read the table and the two error maps:
#
# - Which rung misses more vessels ($\color{#2E7D32}{\blacksquare}$ green), and which invents more
#   ($\color{#E65100}{\blacksquare}$ orange)? Is the rung with the higher IoU
#   also the one with fewer of both?
# - You need a screening tool that must not miss a vessel, and a tool whose every mark must be right. Do you
#   pick the same rung for both?

# %% [markdown]
# <details><summary>What to look for</summary>
#
# - SAM's box tends to mark a region or the whole eye: high recall, many false alarms, an IoU near the
#   "everything is foreground" row.
# - The scribble head saw the fewest labelled pixels, yet from several images, and it was never told a vessel
#   was background. Its errors are not section 3's: compare its recall with the naive head's.
# - Which rung is best depends on whether a miss or a false alarm costs more, and on the minutes in the last
#   column. Your order may differ from your neighbour's.
#
# </details>

# %% [markdown]
# <!-- callout: go-deeper | Go deeper: rung 1, one fully annotated image -->
# Someone spends an afternoon tracing every vessel of **one** training image. That mask is all this rung
# may use. No training, just one mean per class:
#
# 1. Cut the annotated image into 16 px patches; DINOv3 gives one feature vector per patch.
# 2. Average the vessel patches into a vessel prototype, the background patches into a background prototype.
# 3. For a new image, score every patch: `cos(feature, vessel) − cos(feature, background)`.
# 4. Upsample the scores to pixels and mark those above 0.
#
# Before you run the next cell, guess: will the prototype mark more of the fundus than SAM's box, or less?
# Will it find the thin vessels, or only the thick ones?
#
# Then change `ONE_SHOT_IMAGE` (the annotated training image: try 1, 2, …) and run it again. Does the
# held-out IoU move much, and what does that say about relying on a single example?
#
# The prototype is as coarse as a 16 px patch, like the section-2 head, and it depends on *which* example
# you annotated: with one image, a lucky or unlucky choice is the whole training set.

# %% [markdown]
# This helper builds the prototypes from one annotated image and scores every evaluation image.

# %%
# @title One-shot helper: prototypes from one annotated image { display-mode: "form" }
def run_one_shot(image_index: int):
    """Rung 1 from the annotated training image `image_index`: prints the numbers and draws the maps.
    Returns the annotated item, the similarity map of every evaluation image and the scoring seconds."""
    if not isinstance(image_index, int) or not 0 <= image_index < len(specialised["fit"]):
        raise SystemExit(f"ONE_SHOT_IMAGE must be a whole number from 0 to {len(specialised['fit']) - 1}.")
    side = specialised["feat_side"]
    shot = load_item(specialised, specialised["fit"][image_index])
    shot_feat = features_for(specialised, shot, side)
    grid, r = tuple(shot_feat.shape[1:]), region(shot)
    # the rung's supervision: one complete mask of one *training* image (L-005)
    protos = prototypes(shot_feat, patch_share(shot["mask"] & r, grid), patch_share(~shot["mask"] & r, grid))
    t0 = time.monotonic()
    proto_scores = [prototype_score(features_for(specialised, it, side), protos) for it in spec_eval]
    ladder_preds["one_shot"] = [
        score_to_mask(s, it["mask"].shape) for s, it in zip(proto_scores, spec_eval, strict=True)
    ]
    ladder_report["one_shot"] = rung_report(ladder_preds["one_shot"])
    proto_sec = round(time.monotonic() - t0, 1)
    v = ladder_report["one_shot"]
    RESULTS.setdefault("ladder", {})["one_shot"] = v["iou"]
    RESULTS.setdefault("ladder_detail", {})["one_shot"] = v
    RESULTS["one_shot"] = v | {"image": shot["name"], "infer_sec": proto_sec}
    print(
        f"one-shot prototype from {shot['name']} · IoU {v['iou']:.3f} · recall {v['recall']:.3f} · precision "
        f"{v['precision']:.3f} · marked {v['marked']:.0%} of the fundus · {proto_sec} s on {DEVICE}"
    )
    print(f"for scale: the section-2 head on five complete masks · IoU {RESULTS['specialised_head_iou']:.3f}")
    show_masks(
        [
            (
                f"the one annotated image\n{shot['name']}",
                overlay(shot["image"], shot["mask"], THEME["scribble"]),
            ),
            ("similarity: vessel − background", heat_rgb(proto_scores[k_demo], demo["mask"].shape)),
            (
                rung_title("one-shot prototype", ladder_preds["one_shot"][k_demo], demo),
                error_map(demo["image"], ladder_preds["one_shot"][k_demo], demo),
            ),
        ],
        width=3.4,
        legend=[
            ("teal: the annotator's complete mask", THEME["scribble"]),
            "found",
            "invented",
            "missed",
            "unscored",
        ],
        note="Similarity: dark violet = looks like the annotated vessels, light = looks like the background.",
    )
    return shot, proto_scores, proto_sec


# %% [markdown]
# This cell runs the one-shot rung; it also runs under "Run all", so the table in section 8 has its row.

# %%
# @title One-shot: one annotated image, applied to every held-out image { display-mode: "form" }
ONE_SHOT_IMAGE = 0  # @param {type:"integer"}
shot, proto_scores, proto_sec = run_one_shot(ONE_SHOT_IMAGE)

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **Zero-shot IoU looks fine but the mask covers a whole region** → read the "marked" share and the error
#   map, not only IoU.
# - **You have cheap scribbles** → train with a loss that ignores unlabelled pixels. "Far from every stroke,
#   so surely background" is a guess that brings section 3's mistake back.
# - **You have one complete mask** → expect patch-wide vessels, and swap the annotated image to see how
#   much the result depends on your choice.
# - **Two rungs score alike** → compare their error maps and the minutes of annotation: they differ more
#   in how they fail than in one number. Keep the rung colours; they return in section 8.
