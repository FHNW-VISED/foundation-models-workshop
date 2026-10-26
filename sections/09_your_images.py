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
# This helper cell defines the tools of this section: shrinking an image, our example images, a drawing
# canvas with two pens, and the code that turns a drawing (or a PNG you painted elsewhere) into a stroke
# map: 0 nothing, 1 the thing, 2 background, as in section 4.

# %%
# @title Your images: example, canvas, stroke decoding { display-mode: "form" }
import base64
import io
import uuid
from string import Template

from IPython.display import HTML, display

OWN_STROKES: dict[int, np.ndarray] = {}  # image number -> strokes saved from the canvas (0 / 1 thing / 2 bg)
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
    for it in [load_item(PETS, meta) for meta in PETS["held"][:2]] + [SPEC_EVAL[0]]:
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
    OWN_STROKES[k] = decode_strokes(base64.b64decode(data_url.split(",", 1)[1]), OWN[k]["image"].shape[:2])


def show_strokes(items: list[dict]) -> None:
    """Each image next to its strokes: the thing in teal, background in grey, thickened for the figure."""
    panels = []
    for it in items:
        grow = max(1, max(it["image"].shape[:2]) // 256)
        thick = {v: ndi.binary_dilation(it["scribbles"] == v, iterations=grow) for v in (1, 2)}
        drawn = overlay(it["image"], thick[2], THEME["background"], 0.9)
        drawn = overlay(drawn, thick[1], THEME["scribble"], 0.9)
        panels += [(it["name"], it["image"]), (f"strokes · {it['stroke_source']}", drawn)]
    show_masks(panels, width=2.6)


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
# This cell uploads your images only if `ASK_UPLOAD` is ticked (in Colab), shrinks each one to at most
# 1024 px on its long side, and shows them.

# %%
ASK_UPLOAD = False  # @param {type:"boolean"}
# ASK_UPLOAD: set True to upload your own images; False uses the bundled example images.
USE_EXAMPLE = False  # True -> use our example images even if you uploaded or listed your own
MY_IMAGES = []  # or paths uploaded with the folder icon, e.g. ["/content/leaf1.jpg", "/content/leaf2.jpg"]
OWN_MAX_SIDE = 256 if SMOKE else 1024  # long side, in pixels, after shrinking
if not isinstance(ASK_UPLOAD, bool) or not isinstance(USE_EXAMPLE, bool) or not isinstance(MY_IMAGES, list):
    raise SystemExit("ASK_UPLOAD, USE_EXAMPLE must be True or False; MY_IMAGES a list of paths in quotes.")
missing = [p for p in MY_IMAGES if not Path(p).is_file()]
if missing:
    raise SystemExit(f"not found: {missing}. Upload with the folder icon on the left and copy each path.")

seed_everything()
OWN_STROKES.clear()  # a fresh start: strokes from an earlier run belong to earlier images
own_paths = [Path(p) for p in MY_IMAGES]
if ASK_UPLOAD and IN_COLAB and not USE_EXAMPLE and not own_paths:  # opt-in: Run all never waits here
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

OWN = []
for p in own_paths if not USE_EXAMPLE else []:
    try:
        OWN.append(own_item(p.stem, shrink(np.array(Image.open(p).convert("RGB")), OWN_MAX_SIDE), "yours"))
    except Exception as err:  # noqa: BLE001 - any unreadable file gets the same plain message
        print(f"⚠️ {p.name}: could not read it as an image ({type(err).__name__}), skipped. JPEG or PNG work.")
if not OWN:
    OWN = example_items(OWN_MAX_SIDE)
    print("ℹ️ Using our example: two held-out pets and one retina image, with generated strokes.")
sizes = ", ".join(f"{it['name']} {it['image'].shape[1]}×{it['image'].shape[0]}" for it in OWN)
print(f"{len(OWN)} image(s) · {sizes} · {OWN[0]['source']}")
show_masks([(it["name"], it["image"]) for it in OWN], width=3.0)

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
# This cell draws one canvas per image; the save button sends your strokes, as a small PNG, to this Colab
# kernel, where they become a stroke map.

# %%
# @title Draw your strokes on a canvas { display-mode: "form" }
for k, it in enumerate(OWN):
    show_canvas(k, it["image"], f"image {k + 1} of {len(OWN)} · {it['name']}")
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
# This "collect strokes" cell gathers the strokes of every image (your PNGs, then the canvas, then the
# generated strokes for the example), counts them and shows them.

# %%
MY_STROKE_PNGS = []  # fallback, e.g. ["/content/leaf1_strokes.png", ""]: one path (or "") per image, in order
if not isinstance(MY_STROKE_PNGS, list) or len(MY_STROKE_PNGS) > len(OWN):
    raise SystemExit(f"MY_STROKE_PNGS must be a list of at most {len(OWN)} paths, one per image, in order.")
for k, p in enumerate(MY_STROKE_PNGS):
    if p and not Path(p).is_file():
        raise SystemExit(f"{p} not found. Upload it with the folder icon on the left and copy its path.")
    if p:
        OWN_STROKES[k] = decode_strokes(Path(p).read_bytes(), OWN[k]["image"].shape[:2])

for k, it in enumerate(OWN):
    if k in OWN_STROKES:
        it["scribbles"], it["stroke_source"] = OWN_STROKES[k], "drawn by you"
    elif it["auto"] is not None:  # the example, not redrawn: generated strokes through the canvas decoder
        it["scribbles"] = decode_strokes(strokes_png(it["auto"]), it["image"].shape[:2])
        it["stroke_source"] = "generated"
    else:
        it["scribbles"], it["stroke_source"] = np.zeros(it["image"].shape[:2], np.uint8), "none yet"
no_thing = [it["name"] for it in OWN if not (it["scribbles"] == 1).any()]
if no_thing:
    print(f"⚠️ no 'thing' stroke on {no_thing}: skipped. Draw one, press 💾 save, and run this cell again.")
LADDER_OWN = [it for it in OWN if (it["scribbles"] == 1).any()]
if not any((it["scribbles"] == 2).any() for it in LADDER_OWN):
    print("⚠️ No background stroke on any image, and the head needs both kinds. Draw at least one background")
    print("   stroke and save. Until then the ladder runs on our example, so the notebook keeps going.")
    LADDER_OWN = []
if not LADDER_OWN:
    LADDER_OWN = example_items(OWN_MAX_SIDE)
    for it in LADDER_OWN:
        it["scribbles"] = decode_strokes(strokes_png(it["auto"]), it["image"].shape[:2])
        it["stroke_source"] = "generated"

own_n_strokes = {v: sum(len(stroke_masks(it, v)) for it in LADDER_OWN) for v in (1, 2)}
own_effort = effort_minutes(n_strokes=own_n_strokes[1] + own_n_strokes[2])
print(
    f"{len(LADDER_OWN)} image(s) · {own_n_strokes[1]} thing strokes, {own_n_strokes[2]} background strokes · "
    f"≈ {own_effort:.1f} min at {SPECIALISED['annotation_effort']['seconds_per_stroke']} s per stroke"
)
show_strokes(LADDER_OWN)

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
# This helper cell defines the three rungs on a list of images, plus the only check we have without a ground
# truth: how much of your thing strokes a mask covers, and how much of your background strokes it spares.

# %%
# @title The ladder on your images: box, head on your strokes, SAM from clicks { display-mode: "form" }
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
    for it in items:  # SAM: the image is encoded once, then both prompts reuse it
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
    labels = [scribble_targets(it["scribbles"], region(it)) for it in items]
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


# %% [markdown]
# <!-- callout: question | 🤔 Your prediction (1 min) -->
# Look at your images and strokes above. Is your thing an "object" in SAM's sense (compact, a clear inside
# and outside), or more like the vessels (thin, scattered, a texture)?
#
# - Write down which of the three rungs you expect to give the cleanest mask, and which the worst.
# - Then run the next cell (seconds on a T4, much slower on CPU): it shows each image with the three masks
#   in the approach colours of sections 4 and 7, and the IoU on the example.
#
# **Q33.** 🟡 Which rung gave the mask you would keep, image by image, and did it match your prediction?
# Where a rung failed, was it:
#
# - SAM's idea of an object (the box or the clicks grabbed the wrong thing, or too much)?
# - the 16 px patches of the head (blocky edges, small parts lost)?
# - your strokes (a part of the thing you never touched)?

# %%
# Q33 — the ladder on your images: zero-shot SAM, your scribble head, SAM from your strokes
N_OWN_CLICKS = 10  # clicks per kind (thing / background) and image, spread along your strokes
OWN_BOX_PAD = 0.05  # zero-shot box: your thing strokes' bounding box, grown by this share of the long side
OWN_FEAT_SIDE = 256 if SMOKE else 512  # DINOv3 resolution for the head (long side, pixels)
if not isinstance(N_OWN_CLICKS, int) or not 1 <= N_OWN_CLICKS <= 50:
    raise SystemExit("N_OWN_CLICKS must be a whole number from 1 to 50.")
if not isinstance(OWN_BOX_PAD, (int, float)) or not 0 <= OWN_BOX_PAD <= 0.5:
    raise SystemExit("OWN_BOX_PAD must be a number from 0 to 0.5 (a share of the image's long side).")

seed_everything()
own_preds, own_sec, own_train_sec = own_ladder(
    LADDER_OWN, N_OWN_CLICKS, OWN_BOX_PAD, OWN_FEAT_SIDE, HEAD_STEPS
)
if own_train_sec > 60:
    print("ℹ️ Slow session? The head trained for over a minute: fewer or smaller images make it faster.")
has_truth = all(it["mask"] is not None for it in LADDER_OWN)
own_report = {}
print(f"{'approach':<36}{'marks':>7}{'thing ✓':>9}{'bg ✓':>7}{'s':>6}" + (f"{'IoU':>7}" if has_truth else ""))
for k in OWN_APPROACHES:
    agree = [stroke_agreement(p, it) for p, it in zip(own_preds[k], LADDER_OWN, strict=True)]
    own_report[k] = {
        "marked": float(np.mean([p.mean() for p in own_preds[k]])),
        "thing_strokes_covered": float(np.mean([a[0] for a in agree])),
        "bg_strokes_spared": float(np.nanmean([a[1] for a in agree])),
        "seconds": round(own_sec[k], 1),
    }
    if has_truth:  # the example only: its ground truth scores, it never trains (L-005)
        pairs = zip(own_preds[k], LADDER_OWN, strict=True)
        own_report[k]["iou"] = float(
            np.mean([iou(p & region(it), it["mask"] & region(it)) for p, it in pairs])
        )
    v = own_report[k]
    print(
        f"{APPROACH_NAMES[k]:<36}{v['marked']:>7.0%}{v['thing_strokes_covered']:>9.0%}"
        f"{v['bg_strokes_spared']:>7.0%}{v['seconds']:>6.1f}" + (f"{v['iou']:>7.3f}" if has_truth else "")
    )
print("thing ✓ = share of your thing-stroke pixels inside the mask (100 % = all covered)")
print("bg ✓ = share of your background-stroke pixels outside the mask (100 % = none marked)")
for n, it in enumerate(LADDER_OWN):
    panels = [(f"{it['name']} · strokes", overlay(it["image"], it["scribbles"] > 0, THEME["scribble"], 0.9))]
    for k in OWN_APPROACHES:
        r_n = region(it)
        tag = f" · IoU {iou(own_preds[k][n] & r_n, it['mask'] & r_n):.2f}" if has_truth else ""
        title = f"{APPROACH_NAMES[k].split(':')[0]}{tag}"
        panels.append((title, overlay(it["image"], own_preds[k][n], APPROACH_COLOURS[k])))
    if has_truth:
        panels.append(("ground truth · scoring only", overlay(it["image"], it["mask"], THEME["truth"])))
    show_masks(panels, width=2.6)

RESULTS["own_images"] = {
    "source": LADDER_OWN[0]["source"],
    "n_images": len(LADDER_OWN),
    "n_strokes": own_n_strokes[1] + own_n_strokes[2],
    "effort_min": round(own_effort, 2),
    "head_train_sec": round(float(own_train_sec), 2),
    "approaches": own_report,
}
RESULTS["own_images_ok"] = all(len(own_preds[k]) == len(LADDER_OWN) for k in OWN_APPROACHES)  # SPEC §9

# %% [markdown]
# <!-- callout: go-deeper | How to read the stroke checks and the masks -->
# - The two percentages are the only "score" you get on a real problem before anyone labels a test set, and
#   they are weak. SAM's clicks sit *on* your strokes and the head was trained on them, so both should
#   respect them. A low number means the approach ignored what you drew.
# - The masks matter more. For a compact, photo-like thing, expect the three rungs to agree, as on the
#   pets in section 1: the box alone is often enough.
# - For a thin, scattered or textural thing, expect section 2 again. SAM's box grabs a region, the clicks
#   give one blob or several, and the head draws the thing at patch resolution but in all the right places.
# - If your strokes covered only one part of the thing, both the head and the clicks tend to stop at that
#   part. Information you did not give is information no rung has.

# %% [markdown]
# ### 9.4 No ground truth: look where they disagree
#
# Without a ground truth, run two or three different approaches and look at the pixels where they
# disagree. That is where the image is hard, and where one more stroke is worth the most.

# %% [markdown]
# <!-- callout: theory | Theory: the disagreement map -->
# The map is `True` on every pixel that **some** of the masks mark and **others** do not. Three masks over
# four pixels:
#
# ```
# mask A      1 1 0 0
# mask B      1 0 1 0
# mask C      1 1 1 0
# disagree    0 1 1 0
# ```
#
# - **Low, a thin rim** along the thing: the rungs agree on *what* it is and differ only on the exact edge.
# - **High, whole patches**: they disagree on what belongs to the thing. That is where the next stroke pays.

# %% [markdown]
# <!-- callout: your-turn | Your turn (optional, 2 min): the disagreement map -->
# Fill `my_disagreement`.
#
# - **In:** a list of boolean masks of the same image.
# - **Out:** a boolean map, `True` where they do not all agree (some mark the pixel, some do not).
# - **Hint:** `np.stack` puts the masks on a new first axis; `.any(0)` and `.all(0)` reduce along it.
#
# The cell checks your version on the three play masks above and continues with ours if you skip it.

# %%
# @title Reference solution (open only if stuck) { display-mode: "form" }
def _ref_disagreement(masks: list[np.ndarray]) -> np.ndarray:
    """Reference: marked by some, not by all."""
    stack = np.stack([np.asarray(m, bool) for m in masks])
    return stack.any(0) & ~stack.all(0)


# %%
# ✍️ YOUR CODE
def my_disagreement(masks: list[np.ndarray]) -> np.ndarray:
    """Boolean map, same shape as each mask: True where the masks disagree (at least one marks the pixel and
    at least one does not).
    e.g. three 1×4 masks [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1] -> [False, True, False, True]"""
    ...


_play_masks = [np.array([[1, 1, 0, 0]], bool), np.array([[1, 0, 1, 0]], bool), np.array([[1, 1, 1, 0]], bool)]
disagreement = use(
    my_disagreement,
    reference=_ref_disagreement,
    check=lambda f: np.asarray(f(_play_masks)).astype(int),
    name="disagreement",
)
print("play example · three masks over four pixels → disagreement:", disagreement(_play_masks).astype(int))

# %% [markdown]
# This cell computes the disagreement map of the three rungs on each of your images, prints the share of
# the image it covers, and shows it in orange next to your strokes.

# %%
# @title Disagreement map of the three rungs { display-mode: "form" }
own_disagree = [disagreement([own_preds[k][n] for k in OWN_APPROACHES]) for n in range(len(LADDER_OWN))]
RESULTS["own_images"]["disagreement"] = float(np.mean([d.mean() for d in own_disagree]))
panels = []
for it, d in zip(LADDER_OWN, own_disagree, strict=True):
    print(f"{it['name']}: the three rungs disagree on {d.mean():.0%} of the image")
    panels += [
        (f"{it['name']} · strokes", overlay(it["image"], it["scribbles"] > 0, THEME["scribble"], 0.9)),
        (f"disagreement · {d.mean():.0%}", overlay(it["image"], d, THEME["fails"])),
    ]
show_masks(panels, width=2.8)

# %% [markdown]
# **Q34.** 🟡 Where is the orange?
#
# - Along the boundary of the thing?
# - On whole parts of it?
# - On something in the background that looks similar?
#
# If you could add only two more strokes per image, where would you put them, and with which pen? On the
# example, compare the retina's orange with the pets'.
#
# <details><summary>What to look for</summary>
#
# - A thin orange rim means the rungs agree on *what* it is and differ only on the exact edge: SAM's sharp
#   boundary against the head's 16 px squares. That is a resolution question, and more strokes will not
#   fix it.
# - Whole orange parts mean they disagree on what belongs to the thing: usually a part you never stroked,
#   or a background region that looks like it. That is exactly where the next stroke pays.
# - Put a thing stroke on a missed part, or a background stroke on a false alarm. This is the idea behind
#   interactive annotation and active learning: let the models show you where to look.
#
# </details>

# %% [markdown]
# **Q35.** 🔴 Test your answer to Q34.
#
# 1. Draw your two extra strokes per image on the canvas of 9.2. Draw over the old ones; they stay until
#    you press clear. On the example, a saved canvas replaces the generated strokes, so draw a full set.
# 2. Save, and run the cells from the stroke collection to here again.
# 3. Did the disagreement shrink where you drew? Did the head change elsewhere too?
#
# <details><summary>What to look for</summary>
#
# - SAM from clicks changes only on the image you drew on, near the new clicks.
# - The head changes on every image, because it learns one rule from all the strokes together. A
#   background stroke on a false alarm in one image can remove similar false alarms in another, or, if the
#   two regions really are the same kind of thing, remove true parts too.
# - That is the trade-off of section 8 in miniature: a trained head pays annotation once and spreads it
#   everywhere, for better and for worse; direct prompting stays local.
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
# - Compact, everyday objects (a mug, a pet, a car, a leaf on a plain background) make all three rungs
#   agree, and the zero-shot box is often already the mask you would keep.
# - Thin, repeated, textural or specialist things (cracks, cells, vessels, roads from above, defects) make
#   them disagree, and the trained head or many small prompts start to matter.
# - The sign to look for is the one from section 2: if the thing is not an "object" with a clear inside
#   and outside, or is much smaller than a 16 px patch, plan for strokes.
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
