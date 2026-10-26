# %% [markdown]
# ## 10. Guiding principles & wrap-up
#
# A table is good for one decision; a rule is what you take to the next problem. This last section reads the
# numbers *your* session produced in sections 1 to 8 and writes them up as a handful of principles, each
# with its evidence.
#
# They are not general truths about foundation models. They are what this run, on this task, with these
# settings, showed. If your neighbour's session ran with other settings, some sentences may read
# differently, and that is fine.

# %% [markdown]
# <!-- callout: objective | Objective -->
# turn the results into rules you can reuse.

# %% [markdown]
# This cell draws the whole story in one picture: the pets one box got almost right, the same box on the
# specialised task, and the best approach of every path on the same hidden ground truth.
#
# What to look for: the gap between the two one-box bars (where it fell short), then how far each path climbs
# back towards the complete-masks bar at the bottom, and how far each bar sits above the dashed line (marking
# everything).

# %%
# @title The story in one picture { display-mode: "form" }
table = {r["approach"]: r for r in RESULTS["comparison_table"]}  # section 8's rows, by short name
everything_iou = RESULTS["specialised_everything_iou"]
PATHS = {  # the rows of section 8, grouped the way the notebook introduced them
    "ladder (section 4)": ("scribbles",),
    "path 1 (section 5)": ("naive", "masked", "pu"),
    "path 2 (section 6)": ("sam_pseudo", "gradcam_pseudo"),
    "path 3 (section 7)": ("sam_clicks", "sam_boxes"),
}
story = [("pets · one box (section 1)", RESULTS["pets_zero_shot_iou"], THEME["works"])]
story.append(
    (f"{specialised['name']} · one box (section 2)", table["zero_shot"]["iou"], approach_colours["zero_shot"])
)
for path, keys in PATHS.items():
    best = max(keys, key=lambda k: table[k]["iou"])
    story.append((f"best of {path}: {best}", table[best]["iou"], approach_colours[best]))
story.append(("complete masks (section 2)", table["full_masks"]["iou"], approach_colours["full_masks"]))

fig, ax = plt.subplots(figsize=(8, 3.4))
labels, values, colours = zip(*story[::-1], strict=True)
ax.barh(labels, values, color=colours)
for y, v in enumerate(values):
    ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color="#311B92")
ax.axvline(everything_iou, color=THEME["background"], ls="--", label='IoU of "everything" (specialised task)')
ax.set_xlim(0, 1.08)
ax.set_xlabel("mean IoU on the held-out images")
ax.set_title("Where it worked, where it fell short,\nhow far each path got back")
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
add_key(fig, None, "bar colour = the approach's colour in section 8 (violet = the pets) · number = mean IoU")
plt.show()

# %% [markdown]
# ### 10.1 The principles, from your numbers
#
# The next cell writes six rules as a numbered list. Each carries a number from your own session, which tells
# you how strongly to believe it, and says so where a lesson did not hold in your run.

# %%
# @title Principles from this run { display-mode: "form" }
import textwrap


def p_check_first() -> str:
    """Outcome 1: is the foundation model good enough out of the box?"""
    pets, spec = RESULTS["pets_zero_shot_iou"], RESULTS["specialised_zero_shot_iou"]
    rule = "Test the foundation model on your own task first, and read its score against marking everything."
    fact = (
        f"One box scored IoU {pets:.2f} on pets and {spec:.2f} on {specialised['name']}, where a mask over "
        f"the whole image scores {everything_iou:.2f}."
    )
    if spec < everything_iou + 0.1:
        return f"{rule} {fact} Out of the box it was close to doing nothing useful here."
    return f"{rule} {fact} It did part of the job out of the box; adapt only if the rest matters."


def p_resolution() -> str:
    """Section 2: the patch is 16 px; does a larger input help?"""
    res = sorted((int(k.split()[0]), v) for k, v in RESULTS["resolution_effect"].items())
    (lo, a), (hi, b) = res[0], res[-1]
    if specialised.get("thin_structures"):
        rule = "Thin structures defeat patch features: check the resolution first."
    else:
        rule = "Check the resolution first: a patch feature describes a 16 px block, not a pixel."
    fact = (
        f"The same head on complete masks scored IoU {a['iou']:.2f} at {lo} px and {b['iou']:.2f} at {hi} px "
        f"on pets the same kind of head scored {RESULTS['pets_head_iou']:.2f}."
    )
    if b["iou"] - a["iou"] < 0.05:
        return f"{rule} {fact} More pixels helped little, so the limit is the patch: plan for cheap labels."
    return f"{rule} {fact} More pixels helped, so try the larger input before anything more expensive."


def p_missing_is_unknown() -> str:
    """Sections 3 and 5: missing labels are unknown, not background."""
    naive, masked = RESULTS["naive_recall_hidden_fg"], RESULTS["masked_recall_hidden_fg"]
    rule = "An unlabelled pixel is unknown, not background: tell the loss so."
    fact = (
        f"On the same half-painted maps, the head that read the gaps as background found {naive:.0%} of the "
        f"hidden foreground, the masked loss {masked:.0%}."
    )
    if masked <= naive:
        return f"{rule} {fact} Here ignoring the gaps did not recover more; check what the strokes cover."
    pn, pm = table["naive"]["precision"], table["masked"]["precision"]
    if pm < pn:
        return f"{rule} {fact} The price was precision ({pn:.2f} → {pm:.2f}): fewer misses, more alarms."
    return f"{rule} {fact} And it kept its precision ({pn:.2f} → {pm:.2f})."


def p_cheapest_baseline() -> str:
    """Section 7: compare SAM with the hints alone."""
    alone = RESULTS["sam_direct_strokes_alone"]
    sam_rows = [k for k in ("zero_shot", "sam_clicks", "sam_boxes") if k in table]
    best = max(sam_rows, key=lambda k: table[k]["iou"])
    gain = table[best]["iou"] - alone["iou"]
    rule = "Compare the foundation model with the cheapest baseline built from the same hints."
    fact = (
        f"The strokes thickened by {alone['thicken_px']} px, no model at all, scored IoU {alone['iou']:.2f}; "
        f"the best SAM recipe, {best}, scored {table[best]['iou']:.2f}."
    )
    if gain <= 0:
        return f"{rule} {fact} The model added nothing the strokes did not already say."
    if gain < 0.05:
        return f"{rule} {fact} The model added only {gain:.2f}."
    return f"{rule} {fact} The model earned its place, by {gain:.2f}."


def p_one_number() -> str:
    """Section 8: no single column decides."""
    rows = RESULTS["comparison_table"]
    by_iou = max(rows, key=lambda r: r["iou"])
    by_recall = max(rows, key=lambda r: r["recall"])
    steady = min(rows, key=lambda r: r["reliability"])
    rule = "Choose by what a mistake costs, not by one number."
    fact = (
        f"Highest IoU: {by_iou['approach']} ({by_iou['iou']:.2f}); highest recall: {by_recall['approach']} "
        f"({by_recall['recall']:.2f}); smallest spread: {steady['approach']} ({steady['reliability']:.2f})."
    )
    leaders = {by_iou["approach"], by_recall["approach"], steady["approach"]}
    if len(leaders) == 1:
        return f"{rule} {fact} This time one approach led all three; check that before assuming it."
    lead = f"{len(leaders)} different approaches lead the three columns"
    return f"{rule} {fact} {lead}, so the right row depends on which mistake costs most."


def p_minutes_before_hours() -> str:
    """Section 8: annotation effort, once and per new image."""
    full = table["full_masks"]
    cheap_rows = [r for r in RESULTS["comparison_table"] if r["effort_min"] < full["effort_min"] / 10]
    cheap = max(cheap_rows or RESULTS["comparison_table"], key=lambda r: r["iou"])
    per_image = max(RESULTS["comparison_table"], key=lambda r: r["effort_min_per_new_image"])
    rule = "Spend minutes before hours: start with strokes, and buy complete masks only if the table says so."
    fact = (
        f"Complete masks cost {full['effort_min']:.0f} min of annotation for IoU {full['iou']:.2f}; "
        f"{cheap['approach']} needed {cheap['effort_min']:.1f} min for IoU {cheap['iou']:.2f}."
    )
    if cheap["iou"] >= full["iou"]:
        verdict = "The cheap labels matched or beat the expensive ones."
    else:
        ratio = full["effort_min"] / max(cheap["effort_min"], 0.01)
        verdict = f"The hours bought {full['iou'] - cheap['iou']:.2f} IoU for {ratio:.0f}× the effort."
    scale = (
        f"And count per new image: {per_image['approach']} costs "
        f"{per_image['effort_min_per_new_image']:.2f} min on every one, a trained head none."
    )
    return f"{rule} {fact} {verdict} {scale}"


principles = [
    p_check_first(),
    p_resolution(),
    p_missing_is_unknown(),
    p_cheapest_baseline(),
    p_one_number(),
    p_minutes_before_hours(),
]
RESULTS["principles"] = principles
show_list(
    "Principles from this session",
    principles,
    meta=f"{specialised['name']} · {len(spec_eval)} held-out images · {DEVICE}",
)

# %% [markdown]
# ### 10.2 Your framework: six questions for any task
#
# The six principles are the same six questions, asked in the order you would meet them on a new problem.
# Each one has a number you can measure in an afternoon.

# %% [markdown]
# <!-- diagram: s10-six-questions | Six questions for a new task -->
# Follow the arrows: each card is one question, the number that answers it, and what to do with the answer.
# The section sign points to where this notebook measured it.

# %% [markdown]
# <!-- callout: takeaway | Six questions to ask on any new task -->
# <details><summary>The same in words</summary>
#
# 1. **Is the out-of-the-box result above the "everything" line?** Run one box on about 20 images and compare
#    with marking everything. If not, plan to adapt (sections 1 and 2).
# 2. **Is the structure smaller than a 16 px patch?** Train the same head at two input sizes. If the larger
#    one barely helps, buy cheap labels, not a bigger model (section 2).
# 3. **What do the gaps in the labels mean?** Unlabelled is unknown, not background. Tell the loss, and
#    score recall on the hidden part (sections 3 and 5).
# 4. **Does the model beat the hints alone?** Thicken your strokes with no model and compare (section 7).
# 5. **Which mistake costs most?** A missed structure or a false alarm decides whether you rank by recall,
#    precision or IoU (section 8).
# 6. **How many minutes does every new image cost?** Strokes once is cheap, strokes on every image is not
#    (section 8).
#
# </details>
#
# Answer them in this order, write each answer as a number, and keep the table. That table is your
# experiment.

# %% [markdown]
# This helper cell defines `show_my_plan`, which prints your plan next to the principle you picked.


# %%
# @title show_my_plan: print your problem, your first step and the principle { display-mode: "form" }
def show_my_plan(problem: str, principle: int, first_step: str) -> None:
    """Print the problem and the first step, and the principle (1 to 6) they rest on. Checks nothing."""
    if not isinstance(principle, int) or not 0 <= principle <= len(principles):
        raise SystemExit(f"MY_PRINCIPLE must be a whole number from 0 to {len(principles)}.")
    if not (isinstance(problem, str) and isinstance(first_step, str)):
        raise SystemExit('MY_PROBLEM and MY_FIRST_STEP must be text in quotes, e.g. "my problem".')
    if problem or first_step:
        print(f"problem: {problem or '(not written)'}\nfirst step: {first_step or '(not written)'}")
        if principle:
            print(textwrap.fill(principles[principle - 1], width=100, initial_indent="because: "))
    else:
        print("nothing written; fine. Keep the question for the train home: what would you measure first?")


# %% [markdown]
# <!-- callout: your-turn | Your turn (2 min) -->
# **Q23.** 🟢 Think of one segmentation or detection problem of your own (work, research, or the images of
# section 9). Which of the six questions would you ask first, and what number would you measure to answer it?
#
# - Write the problem and your first step on Monday, as a rule with a number, in the form of the next cell
#   (for example "one box on 20 images, compared with the 'everything' line").
# - The cell prints them next to the principle you picked and checks nothing.

# %%
# @title Q23 — your problem and your first step (leave empty to skip) { display-mode: "form" }
MY_PROBLEM = ""  # @param {type:"string"}
# The number of the question or principle above that changes your plan (1 to 6), or 0 for none.
MY_PRINCIPLE = 0  # @param {type:"slider", min:0, max:6, step:1}
MY_FIRST_STEP = ""  # @param {type:"string"}
show_my_plan(MY_PROBLEM, MY_PRINCIPLE, MY_FIRST_STEP)

# %% [markdown]
# <details><summary>A possible answer</summary>
#
# Take cracks in drone photos of bridges. They are thin, like vessels, so principle 2 changes the plan.
#
# 1. Run one box and one feature head on ten images at two input sizes.
# 2. See whether the larger size moves IoU at all.
# 3. If it does not, the patch is the limit: spend the first afternoon on strokes for twenty images.
#
# The number to measure is the gap between the head and the "everything" line on a few images you label
# completely. What carries over is writing the step with the number that would prove it wrong.
#
# </details>

# %% [markdown]
# <!-- callout: question | Show of hands -->
# - Which principle would you have got wrong this morning, before section 1?
# - Which one would you most like to check on your own data next week?

# %% [markdown]
# ### 10.3 A specialist from scratch, and the state of the art
#
# The best IoU of the day is far from a finished vessel map, so it is fair to ask whether the methods are
# worth anything. Two checks. First, a specialist with no foundation model at all, trained here on the same
# five training images. Then the published state of the art.

# %% [markdown]
# <!-- callout: theory | Theory: a small U-Net -->
# A **U-Net** is the classic network for medical segmentation. It sees only pixels and learns everything from
# the masks you give it.
#
# - It shrinks the image step by step, to see the context around each pixel (is this a vessel or a shadow?).
# - It grows it back to full size, and copies the fine detail across from the way down ("skip connections").
# - So it decides **per pixel**, where every head of today decided per 16 px patch.
#
# Ours is small (three levels, about 120,000 numbers to learn) and starts from random weights.

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# The U-Net trains twice on the same five images: once on the complete masks (like the head of section 2.4),
# once on the strokes only (like the scribble head of section 4). Will it beat the head with the complete
# masks? And with the strokes only?

# %% [markdown]
# This helper defines the U-Net, its training loop (random 256 px pieces of the five images, flipped at
# random, the balanced loss of every head so far) and its prediction. The cut is chosen on the training
# images, never on the held-out ones.

# %%
# @title U-Net helpers: the network, training on pieces of the images, the cut { display-mode: "form" }
class SmallUNet(torch.nn.Module):
    """Three levels down (16, 32, 64 channels) and two back up, with skip connections."""

    def __init__(self, c: tuple[int, int, int] = (16, 32, 64)):
        super().__init__()

        def block(a: int, b: int) -> torch.nn.Sequential:
            conv = torch.nn.Conv2d
            return torch.nn.Sequential(conv(a, b, 3, padding=1), torch.nn.ReLU(), conv(b, b, 3, padding=1),
                                       torch.nn.ReLU())

        self.e1, self.e2, self.e3 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.u2, self.d2 = torch.nn.ConvTranspose2d(c[2], c[1], 2, 2), block(2 * c[1], c[1])
        self.u1, self.d1 = torch.nn.ConvTranspose2d(c[1], c[0], 2, 2), block(2 * c[0], c[0])
        self.out = torch.nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        return self.out(self.d1(torch.cat([self.u1(d2), e1], 1)))


def unet_input(item: dict) -> torch.Tensor:
    """The photo as a [3, H, W] tensor, centred around 0."""
    return (torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 255 - 0.5) / 0.25


def train_unet(items: list[dict], labels: list[np.ndarray], steps: int, piece: int = 256, batch: int = 8):
    """A fresh SmallUNet trained on random `piece` px pieces; labels 1 vessel, 0 background, -1 ignored.
    Returns the network and the training seconds."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    images = [unet_input(it) for it in items]
    labs = [torch.from_numpy(np.asarray(t, np.int8)) for t in labels]
    n_pos, n_neg = (sum(int((t == v).sum()) for t in labs) for v in (1, 0))
    net = SmallUNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.monotonic()
    for _ in tqdm(range(steps), desc="U-Net training", leave=False):
        xs, ts = [], []
        for _ in range(batch):
            k = int(rng.integers(len(images)))
            h, w = labs[k].shape
            size = min(piece, h, w) // 4 * 4
            y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
            img, t = images[k][:, y : y + size, x : x + size], labs[k][y : y + size, x : x + size]
            if rng.random() < 0.5:
                img, t = img.flip(-1), t.flip(-1)
            if rng.random() < 0.5:
                img, t = img.flip(-2), t.flip(-2)
            xs.append(img)
            ts.append(t)
        x, t = torch.stack(xs).to(DEVICE), torch.stack(ts).to(DEVICE)
        z = net(x)[:, 0]
        weight = torch.where(t == 1, 0.5 / max(n_pos, 1), torch.where(t == 0, 0.5 / max(n_neg, 1), 0.0))
        loss = (F.binary_cross_entropy_with_logits(z, (t == 1).float(), reduction="none") * weight).sum()
        opt.zero_grad()
        (loss * (n_pos + n_neg) / t.numel()).backward()
        opt.step()
        sched.step()
    return net.eval(), time.monotonic() - t0


@torch.no_grad()
def unet_probability(net: torch.nn.Module, item: dict) -> np.ndarray:
    """The U-Net's probability of vessel for every pixel of one image (sides cropped to a multiple of 4)."""
    x = unet_input(item)
    h, w = x.shape[1:]
    prob = np.zeros((h, w), np.float32)
    prob[: h // 4 * 4, : w // 4 * 4] = torch.sigmoid(net(x[None, :, : h // 4 * 4, : w // 4 * 4].to(DEVICE)))[
        0, 0
    ].cpu().numpy()
    return prob


def training_cut(probs: list[np.ndarray], items: list[dict]) -> float:
    """The probability cut with the best mean IoU on the training images (never the held-out images)."""
    cuts = np.linspace(0.05, 0.95, 19)
    scores = [np.mean([score(p >= c, it)["iou"] for p, it in zip(probs, items, strict=True)]) for c in cuts]
    return float(cuts[int(np.argmax(scores))])


approach_colours |= {"unet_full": "#37474F", "unet_strokes": "#90A4AE"}
approach_names |= {
    "unet_full": "U-Net from scratch · 5 complete masks",
    "unet_strokes": "U-Net from scratch · strokes only",
}

# %% [markdown]
# The next cell trains the U-Net twice, on the complete masks and on the strokes, and scores both on the
# held-out images next to the two heads they should be compared with. On a T4 each training takes one to two
# minutes (`UNET_STEPS` training steps); on a CPU untick `TRAIN_UNET` and the next cell shows our
# run's number instead. What to look for: the IoU of the two U-Nets against the two heads, and how much of the
# fundus each U-Net calls vessel.

# %%
# @title Train the U-Net on the complete masks, then on the strokes only { display-mode: "form" }
TRAIN_UNET = True  # @param {type:"boolean"}
UNET_STEPS = 1500  # @param {type:"slider", min:1000, max:3000, step:250}
unet_report = {}
if TRAIN_UNET:
    steps = 20 if SMOKE else UNET_STEPS
    unet_targets = {
        "unet_full": [np.where(region(it), it["mask"], -1).astype(np.int8) for it in scrib_train],
        # 4.2's targets: vessel and background strokes, every other pixel unknown (-1)
        "unet_strokes": [scribble_targets(it["scribbles"], region(it)) for it in scrib_train],
    }
    unet_preds = {}
    for k, targets in unet_targets.items():
        net, sec = train_unet(scrib_train, targets, steps)
        cut = training_cut([unet_probability(net, it) for it in scrib_train], scrib_train)
        unet_preds[k] = [unet_probability(net, it) >= cut for it in spec_eval]
        unet_report[k] = rung_report(unet_preds[k]) | {"cut": cut, "train_sec": round(sec, 1), "steps": steps}
    RESULTS["unet"] = unet_report
    full_head_preds = spec_head_preds[max(spec_head_preds)]  # section 2.4, the higher resolution
    show_table(
        "A small U-Net from scratch, on the same five training images",
        [
            {"name": approach_names[k], "colour": approach_colours[k], **v,
             "cut_text": f"{v['cut']:.2f}", "time": v["train_sec"]}
            for k, v in unet_report.items()
        ],
        ["iou", "recall", "marked", ("cut_text", "cut", "chosen on the training images", "text"),
         ("time", "training", "on this session's device", "sec")],
        refs=[
            {"name": "linear head on DINOv3 · 5 complete masks (2.4)", **rung_report(full_head_preds)},
            {"name": "scribble head on DINOv3 · strokes only (4.2)", **ladder_report["scribbles"]},
            {"name": "really vessel (the ground truth)", "marked": fg_share, "truth": True},
        ],
        meta=f"{len(spec_eval)} held-out images · {steps} training steps each on {DEVICE}",
    )
    show_masks(
        [
            (rung_title("U-Net · complete masks", unet_preds["unet_full"][k_demo], demo),
             error_map(demo["image"], unet_preds["unet_full"][k_demo], demo)),
            (rung_title("U-Net · strokes only", unet_preds["unet_strokes"][k_demo], demo),
             error_map(demo["image"], unet_preds["unet_strokes"][k_demo], demo)),
            (rung_title("linear head · complete masks", full_head_preds[k_demo], demo),
             error_map(demo["image"], full_head_preds[k_demo], demo)),
        ],
        width=3.2,
        roles=["errors"] * 3,
        legend=["found", "invented", "missed", "unscored"],
        note="The held-out demo image under the two U-Nets and the section-2 head.",
    )
else:
    print("ℹ️ U-Net skipped: the next cell shows our run's number (IoU 0.58 with the complete masks) instead.")

# %% [markdown]
# Now the published reference points, next to your session's bars:
#
# - **ESRVS** (Xu & Zhang, PRCV 2026, [arXiv:2607.24453](https://arxiv.org/abs/2607.24453)): DINOv3
#   features and pseudo-labels, the ingredients of section 6, with **one** labelled image and 31 unlabelled
#   ones, and a real decoder (Mask2Former) at 1024 px.
# - **The best published fully supervised models on HRF**, trained on 15 complete masks: Dice 0.81 to 0.82
#   (Little W-Net, Galdran et al. 2020, [arXiv:2009.01907](https://arxiv.org/abs/2009.01907), and Shin et
#   al., MedIA 2019, in its Table 2).
#
# The papers report Dice; the cell converts it to IoU (IoU = Dice ÷ (2 − Dice)). Their splits and image sizes
# differ from ours, so read them as the right order of magnitude, not as an exact ranking.
#
# What to look for: the gap between your best bar and the references, and which reference uses the same
# ingredients as the notebook.

# %%
# @title Your results next to the state of the art { display-mode: "form" }
def dice_to_iou(dice: float) -> float:
    """IoU from a Dice score of the same masks: IoU = Dice / (2 - Dice)."""
    return dice / (2 - dice)


sota_references = [  # (label, IoU, where it comes from): published, see the text above
    ("ESRVS · DINOv3 + pseudo-labels · 1 labelled image", dice_to_iou(0.7263), "arXiv:2607.24453, Table 3"),
    ("best published · 15 complete masks", dice_to_iou(0.8104), "Little W-Net, arXiv:2009.01907, Table 2"),
]
if not unet_report:  # the U-Net was skipped: our run's number, measured the same way (2026-10-26)
    sota_references.insert(0, ("U-Net from scratch · 5 complete masks", 0.581, "our run on these crops"))
sota_rows = [(label, value, colour, False) for label, value, colour in story[1:]]  # the specialised task
sota_rows += [(approach_names[k], v["iou"], approach_colours[k], False) for k, v in unet_report.items()]
sota_rows += [(label, value, "#BDBDBD", True) for label, value, _ in sota_references]
fig, ax = plt.subplots(figsize=(8, 4.2))
for y, (_label, value, colour, outside) in enumerate(sota_rows[::-1]):
    edge = "#616161" if outside else colour
    ax.barh(y, value, color=colour, hatch="//" if outside else None, edgecolor=edge)
    ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=8, color="#311B92")
ax.set_yticks(range(len(sota_rows)), [r[0] for r in sota_rows[::-1]], fontsize=8)
ax.axvline(everything_iou, color=THEME["background"], ls="--")
ax.set_xlim(0, 1)
ax.set_xlabel("mean IoU on held-out retina images")
ax.set_title("Your session (solid) and the reference points (hatched)")
add_key(
    fig,
    [
        ("solid: your session (colour = the approach, as in section 8)", "#455A64"),
        ("published, or our run if you skipped the U-Net (hatched)", "#BDBDBD"),
        ('IoU of marking everything (dashed line)', THEME["background"]),
    ],
)
plt.show()
show_table(
    "The reference points, measured outside this notebook",
    [{"name": label, "iou": value, "source": source} for label, value, source in sota_references],
    ["iou", ("source", "source", "where the number comes from", "text")],
    name_label="reference",
    note="The papers report Dice; it is converted to IoU here.",
)

# %% [markdown]
# <!-- callout: takeaway | Why it was not all for nothing -->
# - **The limit you met is the patch, not the idea.** Every head today decides per 16 px patch, and the
#   vessels are 4 to 20 px wide (section 2.4). The U-Net decides per pixel and, on our run, reached IoU 0.58
#   on the same five complete masks, about twice the full-mask head (0.28). Choosing the cut on the training
#   images, as the U-Net did, lifts that head to 0.33 on its own, so part of the gap is also the threshold.
# - **Complete masks are what the U-Net needs.** Those five masks cost 15 to 25 hours of annotation. Trained
#   on the strokes alone, the same U-Net marked almost the whole fundus (IoU 0.11 on our run), below the
#   scribble head on foundation-model features (0.17). Cheap labels need the foundation model.
# - **The same ingredients, at full strength, are close to the top.** ESRVS combines DINOv3 features,
#   pseudo-labels and one labelled image with a decoder that works at pixel level, and keeps about 97 % of
#   its fully supervised Dice on HRF.
# - **What you would change first on a real project:** a decoder that sees the pixels on top of the frozen
#   features, the threshold chosen on a few fully labelled images, and the unlabelled images of your archive
#   as pseudo-label material.

# %% [markdown]
# ### 10.4 What you can do now

# %% [markdown]
# At the end you can:
#
# 1. **Judge when a vision foundation model is good enough out of the box**, and recognise the signs that it
#    is not (sections 1 and 2).
# 2. **Adapt it with cheap, incomplete supervision**, and explain what each approach uses and assumes
#    (sections 3 to 7).
# 3. **Compare approaches** on accuracy, annotation effort, reliability and compute, and decide what to do
#    next on a new problem, including your own images (sections 8 and 9).

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · where to go next -->
# - **Label a small test set first**: a few fully annotated images turn every later decision into a
#   measurement, as the hidden ground truth did here.
# - **Let the model choose the next stroke**: draw where the approaches disagree (section 9), retrain, repeat.
# - **A light decoder instead of a linear head**: a small upsampling decoder on the same frozen features
#   comes before any fine-tuning.
# - **Sources**: [DINOv3](https://github.com/facebookresearch/dinov3) ·
#   [SAM 2](https://github.com/facebookresearch/sam2) ·
#   [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

# %% [markdown]
# <!-- callout: across | Across approaches, one last time -->
# The question on the first screen was what modern AI solves out of the box, when its limits start to
# matter, and what to do next.
#
# Your session's answer is in the six principles above, with numbers. The foundation model was a starting
# point, not an answer, and every way out paid for its gain in a different currency: minutes of strokes, a
# wrong assumption about the gaps, a second model's mistakes, or a person pointing at every image.
#
# On your next problem, measure first, spend minutes before hours, and keep the table. Thank you for taking
# part.

# %%
# @title Credits: models and datasets of this session { display-mode: "form" }
cfg = specialised
print(f"DINOv3 ViT-S/16 · {DINO_SOURCE} · {DINO_LICENCE}")
print(f"SAM 2.1 hiera-small · {SAM_ID} · {SAM_LICENCE}")
print(f"{pets_task['name']} · {pets_task['licence']}")
print(f"{cfg['name']} · {cfg['licence']} ({cfg['licence_url']})")
print(textwrap.fill(cfg["citation"], width=100, initial_indent="  ", subsequent_indent="  "))
RESULTS["credits"] = {
    "models": {DINO_SOURCE: DINO_LICENCE, SAM_ID: SAM_LICENCE},
    "datasets": {pets_task["name"]: pets_task["licence"], specialised["name"]: specialised["licence"]},
}

# %% [markdown]
# <!-- callout: go-deeper | Credits & licences -->
# - **DINOv3** ViT-S/16 (Meta), via the `timm` re-host
#   [timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m).
# - DINOv3 is licensed under the
#   [DINOv3 License](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), Copyright © Meta
#   Platforms, Inc. All Rights Reserved. It allows research and commercial use under Meta's conditions and
#   acceptable-use policy; keep this notice if you redistribute the weights or anything derived from them.
#   SAM 2.1 is Apache-2.0.
# - **SAM 2.1** hiera-small (Meta),
#   [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small). SAM 3, if you used
#   it, comes under Meta's own SAM 3 licence, which you accepted on Hugging Face.
# - **Oxford-IIIT Pet**: Parkhi, Vedaldi, Zisserman and Jawahar, *Cats and Dogs*, CVPR 2012,
#   [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The subset shipped with this workshop
#   is resized and shared under the same licence.
# - **pytorch-grad-cam** (Jacob Gildenblat and contributors), MIT.
# - The specialised dataset is printed by the credits cell above, read from the dataset config.
# - Every derived file (scribbles, hidden labels, pseudo-labels, features) carries its source's licence.
#   The full list is in `data/ATTRIBUTION.md`.
