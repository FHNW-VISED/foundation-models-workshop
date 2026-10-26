# %% [markdown]
# ## 9. Your own images
#
# Everything so far ran on our images, where a hidden ground truth told us who was right. This section is
# the dress rehearsal for your next problem.
#
# Pick one to three photos of your own (a leaf, a crack in a wall, cells, a satellite tile, a pet) and draw a
# few strokes on what you want and on what you do not want. Then run three rungs of the ladder on them.
#
# There is no ground truth here. You judge the masks the way you would on a real project: by eye, against
# your strokes, and by where the three approaches disagree.

# %% [markdown]
# <!-- callout: objective | Objective -->
# scribble on your own images and run the ladder.

# %% [markdown]
# <!-- callout: pitfall | Pitfall: privacy -->
# - Your images stay in your Colab session: they go to this session's `/content` folder, and nothing is
#   written to Drive.
# - The canvas runs in your browser and sends your strokes only to your own Colab kernel.
# - We never use a public share link (such as Gradio's `share=True`): that would put your images on a
#   public URL.
# - Even so, pick images you are allowed to process in a cloud notebook: no patient data, nothing under
#   an NDA.

# %% [markdown]
# This helper cell defines image picking, the drawing canvas and the stroke decoding (0 nothing, 1 the thing,
# 2 background, as in section 4).

# %%
# @title Your images: example, canvas, picking images, stroke decoding { display-mode: "form" }
import base64
import io
import uuid
from string import Template

from IPython.display import HTML, display

own_strokes: dict[int, np.ndarray] = {}  # image number -> strokes saved from the canvas (0 / 1 thing / 2 bg)
CANVAS_PX = 640  # long side of the canvas on screen; your strokes are scaled back to the image
BRUSH_PX = 6  # pen width on screen
_CANVAS = Template("""
<div><b>$title</b><br>
<div style="position:relative;width:${cw}px;height:${ch}px">
<img src="data:image/jpeg;base64,$b64" width="$cw" height="$ch" style="position:absolute;left:0;top:0">
<canvas id="fmc$uid" width="$cw" height="$ch"
 style="position:absolute;left:0;top:0;cursor:crosshair;touch-action:none"></canvas>
</div>
<button id="fmt$uid">🖍️ the thing</button> <button id="fmb$uid">🖍️ background</button>
<button id="fmx$uid">clear</button> <button id="fms$uid">💾 save strokes</button>
<span id="fmm$uid">pen: the thing</span></div>
<script>
(function () {
  var c = document.getElementById("fmc$uid"), ctx = c.getContext("2d");
  var msg = document.getElementById("fmm$uid"), pen = "$thing", down = false;
  ctx.lineWidth = $brush; ctx.lineCap = "round"; ctx.lineJoin = "round";
  function xy(e) { var r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  c.addEventListener("pointerdown", function (e) {
    down = true; c.setPointerCapture(e.pointerId);
    var p = xy(e); ctx.strokeStyle = pen; ctx.beginPath(); ctx.moveTo(p[0], p[1]);
    ctx.lineTo(p[0], p[1]); ctx.stroke();
  });
  c.addEventListener("pointermove", function (e) {
    if (!down) return; var p = xy(e); ctx.lineTo(p[0], p[1]); ctx.stroke();
  });
  c.addEventListener("pointerup", function () { down = false; });
  document.getElementById("fmt$uid").onclick = function () {
    pen = "$thing"; msg.textContent = "pen: the thing";
  };
  document.getElementById("fmb$uid").onclick = function () {
    pen = "$bg"; msg.textContent = "pen: background";
  };
  document.getElementById("fmx$uid").onclick = function () {
    ctx.clearRect(0, 0, c.width, c.height); msg.textContent = "cleared (press save to send the empty canvas)";
  };
  document.getElementById("fms$uid").onclick = function () {
    if (!(window.google && google.colab && google.colab.kernel)) {
      msg.textContent = "saving needs Colab: use the PNG fallback below"; return;
    }
    msg.textContent = "saving...";
    google.colab.kernel.invokeFunction("fm_workshop.save_strokes", [$k, c.toDataURL("image/png")], {})
      .then(function () { msg.textContent = "✅ saved, now run the cell below the canvases"; })
      .catch(function (err) { msg.textContent = "❌ " + err; });
  };
})();
</script>
""")


def shrink(arr: np.ndarray, max_side: int) -> np.ndarray:
    """Resize so the long side is at most `max_side`: bilinear for an RGB image, nearest for a 2-D map."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    s = max_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    resample = Image.BILINEAR if arr.ndim == 3 else Image.NEAREST
    return np.array(Image.fromarray(arr.astype(np.uint8)).resize(size, resample)).astype(arr.dtype)


def own_item(name: str, image: np.ndarray, source: str, auto=None, mask=None) -> dict:
    """One image of this section: `auto` = generated strokes (example only), `mask` = its ground truth
    (example only, for scoring, never for training: L-005)."""
    return {
        "name": name,
        "image": image,
        "source": source,
        "auto": auto,
        "mask": mask,
        "fov": np.ones(image.shape[:2], bool),  # no border to ignore: the whole image counts
        "scribbles": None,  # filled in 9.2: 0 nothing, 1 the thing, 2 background
    }


def shrink_strokes(strokes: np.ndarray, max_side: int) -> np.ndarray:
    """Like `shrink` for strokes, but thin strokes survive: a pixel keeps a stroke if its block has one"""
    if max(strokes.shape) <= max_side:
        return strokes
    out = np.zeros(shrink(strokes, max_side).shape, np.uint8)
    for v in (2, 1):  # the thing wins over the background where they meet
        layer = Image.fromarray(((strokes == v) * 255).astype(np.uint8))
        out[np.array(layer.resize((out.shape[1], out.shape[0]), Image.BOX)) > 0] = v
    return out


def example_items(max_side: int) -> list[dict]:
    """Our example: two held-out pets (compact) and one held-out retina image (thin vessels), with the strokes
    our scribble generator drew from their masks."""
    out = []
    for it in [load_item(pets_task, meta) for meta in pets_task["held"][:2]] + [spec_eval[0]]:
        item = own_item(
            it["name"],
            shrink(it["image"], max_side),
            "example",
            auto=shrink_strokes(it["scribbles"], max_side),
            mask=shrink(it["mask"], max_side),
        )
        if it.get("fov") is not None:  # the retina's black border is not scored
            item["fov"] = shrink(it["fov"], max_side)
        out.append(item)
    return out


def show_canvas(k: int, image: np.ndarray, title: str) -> None:
    """A drawing canvas over image k: two pens, clear, save. Saving calls `save_strokes` in this kernel."""
    small = shrink(image, CANVAS_PX)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format="JPEG", quality=85)
    s = CANVAS_PX / max(image.shape[:2])
    ch, cw = round(image.shape[0] * s), round(image.shape[1] * s)
    html = _CANVAS.substitute(
        title=title,
        k=k,
        cw=cw,
        ch=ch,
        b64=base64.b64encode(buf.getvalue()).decode(),
        uid=uuid.uuid4().hex[:8],
        thing=THEME["scribble"],
        bg=THEME["background"],
        brush=BRUSH_PX,
    )
    display(HTML(html))


def decode_strokes(png: bytes, shape: tuple[int, int]) -> np.ndarray:
    """A stroke map (0 nothing, 1 the thing, 2 background) at `shape` from a PNG: either the canvas layer
    (transparent, teal = thing, grey = background) or a copy of your image painted elsewhere with pure red
    (thing) and pure blue (background)."""
    img = Image.open(io.BytesIO(png)).convert("RGBA").resize((shape[1], shape[0]), Image.NEAREST)
    r, g, b, a = (np.asarray(img, int)[..., i] for i in range(4))
    red, blue = (r > 200) & (g < 60) & (b < 60), (b > 200) & (r < 60) & (g < 60)
    out = np.zeros(shape, np.uint8)
    if (a < 255).any():  # a transparent layer: every opaque pixel is a stroke
        on = a > 127
        out[on] = 2  # grey (or blue)
        out[on & (((r < 79) & (g > 79)) | red)] = 1  # teal (or red)
    else:  # a painted copy of the photo: only the pure pen colours count
        out[red], out[blue] = 1, 2
    return out


def strokes_png(strokes: np.ndarray) -> bytes:
    """The other way round, as the canvas sends it: a transparent PNG with teal and grey strokes."""
    rgba = np.zeros((*strokes.shape, 4), np.uint8)
    for v, colour in ((1, THEME["scribble"]), (2, THEME["background"])):
        rgba[strokes == v] = [*np.round(hex_rgb(colour) * 255).astype(int), 255]
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def save_strokes(k, data_url: str) -> None:
    """Called by a canvas's 💾 button: decode the strokes of image k and keep them in this session."""
    k = int(k)
    own_strokes[k] = decode_strokes(base64.b64decode(data_url.split(",", 1)[1]), own[k]["image"].shape[:2])


STROKE_KEY = [("thing stroke", THEME["scribble"]), ("background stroke", THEME["background"])]


def stroke_panel(it: dict) -> np.ndarray:
    """The image with its strokes, thickened to stay visible: the thing in teal, the background in grey."""
    grow = max(1, max(it["image"].shape[:2]) // 256)
    thick = {v: ndi.binary_dilation(it["scribbles"] == v, iterations=grow) for v in (1, 2)}
    drawn = overlay(it["image"], thick[2], THEME["background"], 0.9)
    return overlay(drawn, thick[1], THEME["scribble"], 0.9, dim=False)


def stroke_targets(strokes: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Training targets from strokes: 1 the thing, 0 background, -1 (ignored) everywhere you drew nothing.
    e.g. strokes [[1, 0, 2]] -> [[1, -1, 0]]"""
    targets = np.full(strokes.shape, -1, np.int8)
    targets[strokes == 2] = 0
    targets[strokes == 1] = 1
    targets[~np.asarray(region, bool)] = -1
    return targets


def show_strokes(items: list[dict]) -> None:
    """Each image next to its strokes."""
    panels = []
    for it in items:
        panels += [(it["name"], it["image"]), (f"strokes · {it['stroke_source']}", stroke_panel(it))]
    show_masks(panels, width=2.6, legend=STROKE_KEY, note="strokes are drawn thicker so you can see them")


def pick_images(ask_upload: bool, use_example: bool, my_images: list) -> list[dict]:
    """The images of this section: uploaded in Colab if `ask_upload`, else the files listed in `my_images`
    (up to 3, each shrunk to OWN_MAX_SIDE), else our example. Prints and shows them."""
    if (
        not isinstance(ask_upload, bool)
        or not isinstance(use_example, bool)
        or not isinstance(my_images, list)
    ):
        raise SystemExit(
            "ASK_UPLOAD, USE_EXAMPLE must be True or False; MY_IMAGES a list of paths in quotes."
        )
    missing = [p for p in my_images if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"not found: {missing}. Upload with the folder icon on the left and copy each path.")
    seed_everything()
    own_strokes.clear()  # a fresh start: strokes from an earlier run belong to earlier images
    own_paths = [Path(p) for p in my_images]
    if ask_upload and IN_COLAB and not use_example and not own_paths:  # opt-in: Run all never waits here
        from google.colab import files

        print("Pick 1 to 3 images (JPEG or PNG), or press Cancel to use our example.")
        upload_dir = Path("/content/own_images")  # this session's disk only; gone when the session ends
        upload_dir.mkdir(exist_ok=True)
        try:
            uploaded = files.upload()
        except Exception as err:  # noqa: BLE001 - a cancelled upload raises in some Colab versions
            print(f"(upload stopped: {type(err).__name__})")
            uploaded = {}
        for fname, data in uploaded.items():
            (upload_dir / fname).write_bytes(data)
            own_paths.append(upload_dir / fname)
    if len(own_paths) > 3:
        print(f"ℹ️ {len(own_paths)} images: keeping the first 3, so every step stays short.")
        own_paths = own_paths[:3]
    images = []
    for p in own_paths if not use_example else []:
        try:
            images.append(
                own_item(p.stem, shrink(np.array(Image.open(p).convert("RGB")), OWN_MAX_SIDE), "yours")
            )
        except Exception as err:  # noqa: BLE001 - any unreadable file gets the same plain message
            print(
                f"⚠️ {p.name}: could not read it as an image ({type(err).__name__}), skipped. Use JPEG or PNG."
            )
    if not images:
        images = example_items(OWN_MAX_SIDE)
        print("ℹ️ Using our example: two held-out pets and one retina image, with generated strokes.")
    sizes = ", ".join(f"{it['name']} {it['image'].shape[1]}×{it['image'].shape[0]}" for it in images)
    print(f"{len(images)} image(s) · {sizes} · {images[0]['source']}")
    show_masks([(it["name"], it["image"]) for it in images], width=3.0)
    return images


def stroke_effort(items: list[dict]) -> tuple[dict[int, int], float]:
    """Number of strokes per kind (1 thing, 2 background) over `items`, and the minutes they cost to draw."""
    n_strokes = {v: sum(len(stroke_masks(it, v)) for it in items) for v in (1, 2)}
    return n_strokes, effort_minutes(n_strokes=n_strokes[1] + n_strokes[2])


def collect_strokes(stroke_pngs: list) -> list[dict]:
    """Give every image of own its strokes (your PNGs, then the canvas, then the generated strokes of the
    example), print the counts and show them. Returns the images the ladder will run on: the ones with a
    thing stroke, or our example if there are none (or no background stroke at all)."""
    if not isinstance(stroke_pngs, list) or len(stroke_pngs) > len(own):
        raise SystemExit(
            f"MY_STROKE_PNGS must be a list of at most {len(own)} paths, one per image, in order."
        )
    for k, p in enumerate(stroke_pngs):
        if p and not Path(p).is_file():
            raise SystemExit(f"{p} not found. Upload it with the folder icon on the left and copy its path.")
        if p:
            own_strokes[k] = decode_strokes(Path(p).read_bytes(), own[k]["image"].shape[:2])
    for k, it in enumerate(own):
        if k in own_strokes:
            it["scribbles"], it["stroke_source"] = own_strokes[k], "drawn by you"
        elif it["auto"] is not None:  # the example, not redrawn: generated strokes through the canvas decoder
            it["scribbles"] = decode_strokes(strokes_png(it["auto"]), it["image"].shape[:2])
            it["stroke_source"] = "generated"
        else:
            it["scribbles"], it["stroke_source"] = np.zeros(it["image"].shape[:2], np.uint8), "none yet"
    no_thing = [it["name"] for it in own if not (it["scribbles"] == 1).any()]
    if no_thing:
        print(
            f"⚠️ no 'thing' stroke on {no_thing}: skipped. Draw one, press 💾 save, and run this cell again."
        )
    ladder = [it for it in own if (it["scribbles"] == 1).any()]
    if not any((it["scribbles"] == 2).any() for it in ladder):
        print(
            "⚠️ No background stroke on any image, and the head needs both kinds. Draw at least one background"
        )
        print("   stroke and save. Until then the ladder runs on our example, so the notebook keeps going.")
        ladder = []
    if not ladder:
        ladder = example_items(OWN_MAX_SIDE)
        for it in ladder:
            it["scribbles"] = decode_strokes(strokes_png(it["auto"]), it["image"].shape[:2])
            it["stroke_source"] = "generated"
    n_strokes, minutes = stroke_effort(ladder)
    print(
        f"{len(ladder)} image(s) · {n_strokes[1]} thing strokes, {n_strokes[2]} background strokes · "
        f"≈ {minutes:.1f} min at {specialised['annotation_effort']['seconds_per_stroke']} s per stroke"
    )
    show_strokes(ladder)
    return ladder


OWN_MAX_SIDE = 256 if SMOKE else 1024  # long side, in pixels, after shrinking


if IN_COLAB:
    from google.colab import output

    output.register_callback("fm_workshop.save_strokes", save_strokes)  # the canvas talks to this kernel only

# %% [markdown]
# ### 9.1 Pick one to three images

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min): choose your images -->
# By default the next cell uses our example: two held-out pets and one retina image, with strokes drawn by
# our scribble generator. To use your own images:
#
# 1. Tick `ASK_UPLOAD` in the next cell and run it.
# 2. Pick one to three JPEG or PNG files in the upload button that appears (press Cancel to keep the example).
#
# - Choose images that show the *same kind of thing*: the head is trained on all of them together.
# - Already uploaded them with the folder icon on the left? Put their paths in `MY_IMAGES` instead.
#
# Nothing here is mandatory.

# %% [markdown]
# This cell uploads your images if `ASK_UPLOAD` is ticked (in Colab), shrinks them to at most 1024 px, and
# shows them. Tick `USE_EXAMPLE` to keep our example.

# %%
# @title Choose your images { display-mode: "form" }
ASK_UPLOAD = False  # @param {type:"boolean"}
USE_EXAMPLE = False  # @param {type:"boolean"}
MY_IMAGES = []  # @param {type:"raw"}
own = pick_images(ASK_UPLOAD, USE_EXAMPLE, MY_IMAGES)

# %% [markdown]
# ### 9.2 Draw your strokes 🖍️

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 4 min): scribble -->
# Run the next cell: a canvas appears over each image.
#
# 1. With the **thing** pen (teal), draw two to five short strokes *on* what you want segmented, on
#    different parts of it.
# 2. With the **background** pen (grey), draw two to five strokes on what you do not want, especially near
#    the boundary and on anything that looks similar.
# 3. Press **💾 save strokes** under each canvas, then run the "collect strokes" cell further down.
#
# Strokes, not outlines: a few seconds per image, the annotation budget of section 4. On the example,
# drawing is optional: an image you do not save keeps its generated strokes.

# %% [markdown]
# This cell draws one canvas per image. The save button sends your strokes to this Colab session.

# %%
# @title Draw your strokes on a canvas { display-mode: "form" }
for k, it in enumerate(own):
    show_canvas(k, it["image"], f"image {k + 1} of {len(own)} · {it['name']}")
if not IN_COLAB:
    print("ℹ️ Not in Colab: the canvases show, but saving needs Colab. Use the PNG fallback below.")

# %% [markdown]
# <!-- callout: pitfall | Pitfall: no canvas, or the save button does nothing? -->
# Some browsers or extensions block the canvas. The fallback:
#
# 1. Open a copy of your image in any paint program (Preview, Paint, a phone's markup tool).
# 2. Draw the thing strokes in **pure red** (255, 0, 0) and the background strokes in **pure blue**
#    (0, 0, 255), and save it as PNG.
# 3. Upload it with the folder icon on the left and put its path in `MY_STROKE_PNGS` below, one per image,
#    in the order above (`""` skips one). Any size works.

# %% [markdown]
# This cell gathers the strokes of every image, counts them, and shows them next to the photos.
#
# What to look for: teal strokes on the thing, grey strokes on what you do not want. If an image shows none,
# the next cells use our generated strokes for it.

# %%
# @title Collect your strokes { display-mode: "form" }
MY_STROKE_PNGS = []  # @param {type:"raw"}
ladder_own = collect_strokes(MY_STROKE_PNGS)

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min): prompt SAM yourself -->
# Before the ladder runs, try SAM by hand on your images. Click on the thing, shift-click on what you do not
# want, drag a box, or type a few words and press "Ask SAM 3". This is the same canvas idea as above, with
# SAM answering at once.

# %%
# @title Prompt SAM on your images { display-mode: "form" }
sam_playground(own, "Your images: click, box, or a word", text_prompt=True)

# %% [markdown]
# ### 9.3 The ladder on your images
#
# Three rungs, the same recipes as before, nothing new to learn.
#
# - **Zero-shot**: SAM with one box, the box a person would draw around the thing (here: around your thing
#   strokes, grown by 5 % of the image).
# - **Your scribble head**: the section-4 head, a linear layer on DINOv3 patch features. It is trained with
#   the partial cross-entropy on your strokes only (every unlabelled pixel ignored), on all your images
#   together.
# - **SAM from your strokes**: ten clicks along your thing strokes and ten along your background strokes,
#   one SAM prompt per image (Path 3, section 7.1).

# %% [markdown]
# This helper cell defines the three rungs, a stroke check (no ground truth needed) and the panels.

# %%
# @title The ladder on your images: three rungs, scoring, panels { display-mode: "form" }
OWN_APPROACHES = ("zero_shot", "scribbles", "sam_clicks")  # the colours and names of sections 4 and 7


def strokes_box(item: dict, pad: float) -> list[float]:
    """Zero-shot's box: around every 'thing' stroke, grown by `pad` × the long side, inside the image."""
    ys, xs = np.nonzero(item["scribbles"] == 1)
    h, w = item["scribbles"].shape
    g = pad * max(h, w)
    x0, y0 = max(0.0, xs.min() - g), max(0.0, ys.min() - g)
    return [float(x0), float(y0), float(min(w - 1.0, xs.max() + g)), float(min(h - 1.0, ys.max() + g))]


def own_ladder(items: list[dict], n_clicks: int, box_pad: float, feat_side: int, steps: int):
    """The three rungs on `items`. Returns masks per approach (one per image), seconds per approach, and the
    seconds the head took to train."""
    preds, sec = {k: [] for k in OWN_APPROACHES}, dict.fromkeys(OWN_APPROACHES, 0.0)
    for it in tqdm(items, desc="SAM on your images", leave=False):  # the image is encoded once, then reused
        t0 = time.monotonic()
        preds["zero_shot"].append(sam.probability(it["image"], box=strokes_box(it, box_pad)) >= 0.5)
        t1 = time.monotonic()
        pos, neg = stroke_points(it, 1, n_clicks), stroke_points(it, 2, n_clicks)
        prob = sam.probability(it["image"], points=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
        preds["sam_clicks"].append(prob >= 0.5)
        sec["zero_shot"] += t1 - t0  # includes the image encoding
        sec["sam_clicks"] += time.monotonic() - t1
    t0 = time.monotonic()
    feats = [dino.features(it["image"], min(feat_side, max(it["image"].shape[:2]))) for it in items]
    # the rung's supervision: your strokes only; every unlabelled pixel is ignored (-1)
    labels = [stroke_targets(it["scribbles"], region(it)) for it in items]
    head, train_sec = train_head(feats, labels, steps)
    pairs = zip(feats, items, strict=True)
    preds["scribbles"] = [predict_head(head, f, it["image"].shape[:2]) for f, it in pairs]
    sec["scribbles"] = time.monotonic() - t0
    return preds, sec, train_sec


def stroke_agreement(pred: np.ndarray, item: dict) -> tuple[float, float]:
    """Share of the thing-stroke pixels the mask covers, and share of the background-stroke pixels it leaves
    out (nan if the image has no background stroke). Both are 1.0 for a mask that respects every stroke."""
    s = item["scribbles"]
    spared = 1.0 - float(pred[s == 2].mean()) if (s == 2).any() else float("nan")
    return float(pred[s == 1].mean()), spared


def score_own_ladder(items: list[dict], preds: dict, sec: dict) -> dict:
    """Per approach: the share of the image marked, the two stroke checks, the seconds, and (the example
    only, its ground truth scores and never trains: L-005) the IoU inside the evaluation region."""
    has_truth = all(it["mask"] is not None for it in items)
    report = {}
    for k in OWN_APPROACHES:
        agree = [stroke_agreement(p, it) for p, it in zip(preds[k], items, strict=True)]
        report[k] = {
            "marked": float(np.mean([p.mean() for p in preds[k]])),
            "thing_strokes_covered": float(np.mean([a[0] for a in agree])),
            "bg_strokes_spared": float(np.nanmean([a[1] for a in agree])),
            "seconds": round(sec[k], 1),
        }
        if has_truth:
            pairs = zip(preds[k], items, strict=True)
            report[k]["iou"] = float(
                np.mean([iou(p & region(it), it["mask"] & region(it)) for p, it in pairs])
            )
    return report


def show_own_ladder(items: list[dict], preds: dict, report: dict) -> None:
    """The table of the three rungs, then one row of panels per image: strokes, the three masks, and the
    ground truth when there is one."""
    has_truth = all(it["mask"] is not None for it in items)
    print(
        f"{'approach':<36}{'marks':>7}{'thing ✓':>9}{'bg ✓':>7}{'s':>6}"
        + (f"{'IoU':>7}" if has_truth else "")
    )
    for k in OWN_APPROACHES:
        v = report[k]
        print(
            f"{approach_names[k]:<36}{v['marked']:>7.0%}{v['thing_strokes_covered']:>9.0%}"
            f"{v['bg_strokes_spared']:>7.0%}{v['seconds']:>6.1f}" + (f"{v['iou']:>7.3f}" if has_truth else "")
        )
    print("marks = share of the image the mask covers")
    print("thing ✓ = share of your thing-stroke pixels inside the mask (100 % = all covered)")
    print("bg ✓ = share of your background-stroke pixels outside the mask (100 % = none marked)")
    key = [*STROKE_KEY, "mask", *(["truth"] if has_truth else [])]
    for n, it in enumerate(items):
        panels = [(f"{it['name']}\nyour strokes", stroke_panel(it))]
        for k in OWN_APPROACHES:
            r_n = region(it)
            tag = f"\nIoU {iou(preds[k][n] & r_n, it['mask'] & r_n):.2f}" if has_truth else ""
            shown = overlay(it["image"], preds[k][n], THEME["works"])
            panels.append((f"{approach_names[k].split(':')[0]}{tag}", shown))
        if has_truth:
            panels.append(("ground truth\n(scoring only)", overlay(it["image"], it["mask"], THEME["truth"])))
        show_masks(panels, width=2.6, legend=key)


def run_own_ladder(items: list[dict], n_clicks: int, box_pad: float) -> dict:
    """Q21: the three rungs on `items`, the table and the panels. Stores RESULTS["own_images"] and
    RESULTS["own_images_ok"]. Returns the masks per approach (one per image)."""
    if not isinstance(n_clicks, int) or not 1 <= n_clicks <= 50:
        raise SystemExit("N_OWN_CLICKS must be a whole number from 1 to 50.")
    if not isinstance(box_pad, (int, float)) or not 0 <= box_pad <= 0.5:
        raise SystemExit("OWN_BOX_PAD must be a number from 0 to 0.5 (a share of the image's long side).")
    seed_everything()
    preds, sec, train_sec = own_ladder(items, n_clicks, box_pad, 256 if SMOKE else 512, HEAD_STEPS)
    if train_sec > 60:
        print("ℹ️ Slow session? The head trained for over a minute: fewer or smaller images make it faster.")
    report = score_own_ladder(items, preds, sec)
    show_own_ladder(items, preds, report)
    n_strokes, minutes = stroke_effort(items)
    RESULTS["own_images"] = {
        "source": items[0]["source"],
        "n_images": len(items),
        "n_strokes": n_strokes[1] + n_strokes[2],
        "effort_min": round(minutes, 2),
        "head_train_sec": round(float(train_sec), 2),
        "approaches": report,
    }
    RESULTS["own_images_ok"] = all(len(preds[k]) == len(items) for k in OWN_APPROACHES)  # SPEC §9
    return preds


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction (1 min) -->
# Look at your images and strokes above. Is your thing an "object" in SAM's sense (compact, a clear inside
# and outside), or more like the vessels (thin, scattered, a texture)?
#
# - Write down which of the three rungs you expect to give the cleanest mask, and which the worst.
# - Then run the next cell (seconds on a T4, much slower on CPU). Each row is one image: your strokes, the
#   three masks in violet, and on the example the ground truth in green with the IoU.
# - What to look for: where the three violet masks differ, and which one respects your strokes.
#
# **Q21.** 🟡 Which rung gave the mask you would keep, image by image, and did it match your prediction?
# Where a rung failed, was it:
#
# - SAM's idea of an object (the box or the clicks grabbed the wrong thing, or too much)?
# - the 16 px patches of the head (blocky edges, small parts lost)?
# - your strokes (a part of the thing you never touched)?

# %%
# @title Q21 — the ladder on your images: box, scribble head, SAM clicks { display-mode: "form" }
# Clicks per kind (thing / background) and image, spread along your strokes.
N_OWN_CLICKS = 10  # @param {type:"slider", min:1, max:50, step:1}
# Zero-shot box: your thing strokes' bounding box, grown by this share of the long side.
OWN_BOX_PAD = 0.05  # @param {type:"slider", min:0, max:0.5, step:0.05}
own_preds = run_own_ladder(ladder_own, N_OWN_CLICKS, OWN_BOX_PAD)

# %% [markdown]
# <!-- callout: go-deeper | How to read the stroke checks and the masks -->
# - The two percentages are weak checks: SAM's clicks sit on your strokes and the head trained on them, so
#   both should respect them. A low number means the approach ignored what you drew.
# - The masks matter more. A compact thing makes the rungs agree, as on the pets. A thin or textural thing
#   brings section 2 back: a box grabs a region, the head draws at patch resolution.

# %% [markdown]
# ### 9.4 No ground truth: look where they disagree
#
# Without a ground truth, run two or three approaches and look at the pixels where they disagree. That is
# where the image is hard, and where one more stroke is worth the most.

# %% [markdown]
# <!-- callout: theory | Theory: the disagreement map -->
# The map is pink on every pixel that **some** of the three masks mark and **others** do not. A thin rim
# along the thing means the rungs agree on *what* it is and differ on the edge. Whole patches mean they
# disagree on what belongs to the thing.

# %% [markdown]
# This helper cell defines `disagreement`: marked by some masks, not by all.

# %%
# @title disagreement: marked by some masks, not by all { display-mode: "form" }
def disagreement(masks: list[np.ndarray]) -> np.ndarray:
    """True where at least one mask marks the pixel and at least one does not.
    e.g. three 1×4 masks [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0] -> [False, True, True, False]"""
    stack = np.stack([np.asarray(m, bool) for m in masks])
    return stack.any(0) & ~stack.all(0)


# %% [markdown]
# This cell computes the disagreement of the three rungs on each of your images and shows it in pink.
#
# What to look for: a thin pink rim, or whole pink parts, and whether the pink sits on your strokes.

# %%
# @title Disagreement map of the three rungs { display-mode: "form" }
own_disagree = [disagreement([own_preds[k][n] for k in OWN_APPROACHES]) for n in range(len(ladder_own))]
RESULTS["own_images"]["disagreement"] = float(np.mean([d.mean() for d in own_disagree]))
DISAGREE = "#D81B60"  # pink: the rungs disagree (orange is kept for "invented" in the error maps)
panels = []
for it, d in zip(ladder_own, own_disagree, strict=True):
    print(f"{it['name']}: the three rungs disagree on {d.mean():.0%} of the image")
    panels += [
        (f"{it['name']}\nyour strokes", stroke_panel(it)),
        (f"disagreement · {d.mean():.0%} of the image", overlay(it["image"], d, DISAGREE)),
    ]
show_masks(
    panels,
    width=2.8,
    legend=[*STROKE_KEY, ("some rungs mark it, others do not", DISAGREE)],
)

# %% [markdown]
# **Q22.** 🟡 Where is the pink?
#
# - Along the boundary of the thing?
# - On whole parts of it?
# - On something in the background that looks similar?
#
# If you could add only two more strokes per image, where would you put them, and with which pen? On the
# example, compare the retina's pink with the pets'. To test your answer, draw those strokes on the canvas
# of 9.2, save, and run the cells from the stroke collection again.
#
# <details><summary>What to look for</summary>
#
# - A thin pink rim: the rungs agree on *what* it is, and differ on the exact edge (SAM's sharp boundary
#   against the head's 16 px squares). More strokes will not fix that.
# - Whole pink parts: they disagree on what belongs to the thing, usually a part you never stroked. That is
#   where the next stroke pays.
# - After redrawing, SAM from clicks changes only near the new clicks. The head changes on every image,
#   because it learns one rule from all strokes: a trained head spreads annotation everywhere.
#
# </details>

# %% [markdown]
# <!-- callout: question | 👥 Pairs, 5 min: compare your images -->
# Show your neighbour your strokes, your three masks and your disagreement map.
#
# - Whose thing was easier for the foundation model, and what made it easier: its shape, its contrast, how
#   "photo-like" it is?
# - Did the same rung win on both your images?
# - Agree on one sign you would look for, on a new problem, that says "a box will do" versus "I need
#   strokes and a head".

# %% [markdown]
# <details><summary>What pairs usually find</summary>
#
# - Compact, everyday objects (a mug, a pet, a leaf on a plain background) make the rungs agree, and the
#   zero-shot box is often the mask you would keep.
# - Thin or textural things (cracks, cells, vessels, defects) make them disagree, and the trained head or
#   many small prompts start to matter.
# - The sign from section 2: no clear inside and outside, or smaller than a 16 px patch, means plan for
#   strokes.
#
# </details>

# %% [markdown]
# <!-- callout: takeaway | When you see X, do Y: on your own images -->
# - **When** your thing is compact and photo-like and the three rungs agree **→** the zero-shot box is
#   probably enough; stop there.
# - **When** the rungs disagree on whole parts of the thing **→** add a thing stroke on the missed part or
#   a background stroke on the false alarm, then rerun.
# - **When** only a thin rim disagrees **→** that is the 16 px patch against SAM's sharp edge; more strokes
#   will not fix it.
# - **When** there is no ground truth **→** use your strokes and the disagreement map as a sanity check,
#   then label a few images completely so the next decision is measured.
#
# Section 10 turns all of this into rules you can take to the next problem.
