# %% [markdown]
# # When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation

# %% [markdown]
# <!-- callout: banner | When Foundation Models Fall Short -->
# **From Off-the-Shelf AI to Problem-Specific Adaptation**
#
# AMLD 2027 · EPFL, 11 March 2027 &nbsp;·&nbsp; SDS 2027 · Zurich, 1 June 2027
#
# **The question:** what can modern AI solve out of the box, when do its limitations
# start to matter, and what should we do next?
#
# **At the end of this notebook you will be able to:**
#
# 1. Judge when a vision foundation model is good enough out of the box, and recognise
#    the signs that it is not.
# 2. Adapt it with cheap, incomplete supervision (scribbles, missing labels,
#    pseudo-labels, explanations, prompts), and explain what each approach uses and assumes.
# 3. Compare approaches on accuracy, annotation effort, reliability and compute, and
#    decide what to do next on a new problem, including your own images.
#
# Levels: 🟢 beginner · 🟡 intermediate (the default path) · 🔴 advanced, optional.
# &nbsp; ✍️ = you write a few lines &nbsp; 🖍️ = you draw &nbsp; 👥 = compare with your neighbour.
#
# **Run `setup_check.ipynb` first.** It was shared with you the week
# before: it puts the models, images and precomputed files in your Google Drive, so nothing is downloaded
# in the room. If you have not run it, do it now (about 5 minutes), then come back here.

# %% [markdown]
# <!-- callout: objective | Objective -->
# Load the cached models and data; see the question and outcomes.

# %% [markdown]
# **You need** basic ML (training, validation, evaluation) and some Python, no computer-vision background, and
# this notebook in Colab with **Runtime → Change runtime type → T4 GPU**. Every step also runs on a CPU, just
# slower, and each training cell offers a precomputed checkpoint.
#
# **The plan.** Block 1 (sections 1–3): a foundation model segments pets almost perfectly, then meets a
# specialised task and falls short. Block 2 (sections 4–10): the ways out, from scribbles to prompting the
# model directly, all compared on the same hidden ground truth, then on your own images.

# %% [markdown]
# ## 0. Setup & the question
#
# This cell detects Colab and the GPU, fixes the seed, creates the `RESULTS` dictionary every section writes
# into, loads the `tqdm` progress bars used by the long steps, and defines `use()`, so your own code never
# blocks the notebook. Run it once.

# %%
# @title Setup (run me first) { display-mode: "form" }
import ast
import inspect
import json
import os
import random
import sys
import time
import weakref
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm  # noqa: F401 - progress bars for the long loops of later sections

IN_COLAB = "google.colab" in sys.modules
SMOKE = os.environ.get("NB_SMOKE") == "1"  # automated test: tiny data, CPU, seconds. Never set this by hand.
SEED = 42

# Where the cache lives. setup_check.ipynb filled MyDrive/fm-workshop-cache/ in Colab; the smoke run uses the
# small fixtures committed in data/smoke/; a local checkout uses data/cache/ (scripts/prepare_data.py).
if IN_COLAB and not SMOKE:
    CACHE_DIR = Path("/content/drive/MyDrive/fm-workshop-cache")
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as err:  # noqa: BLE001 - one plain line, then a plain message below
        print(f"(Drive mount error: {type(err).__name__}: {err})")
    if not CACHE_DIR.is_dir():
        print(
            "⚠️ Google Drive is not mounted or the cache folder is missing. Run this cell again and click "
            "'Allow' in the pop-up. If the folder is still missing, run setup_check.ipynb first (5 minutes)."
        )
    if (CACHE_DIR / "models").is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"  # nothing is downloaded in the room (N-030)
    DATA_DIR, PRECOMPUTED = CACHE_DIR / "data", CACHE_DIR / "precomputed"
    SPECIALISED_JSON = CACHE_DIR / "specialised.json"
else:
    CACHE_DIR = Path("data/smoke" if SMOKE else "data/cache")
    DATA_DIR = CACHE_DIR if SMOKE else CACHE_DIR / "workshop"
    PRECOMPUTED = CACHE_DIR / "precomputed"
    SPECIALISED_JSON = Path("data/specialised.json")

# Model weights: plain files in <cache>/models/<org>--<name>/, written by setup_check.ipynb (no symlinks,
# so they survive Drive). `.fm-complete.json` is written last, once the folder is verified. No folder
# (smoke run, local checkout): the models load from the Hugging Face hub id instead.
MODELS_DIR = CACHE_DIR / "models"
MODEL_DONE = ".fm-complete.json"


def model_folder(repo_id: str) -> Path | None:
    """The cached plain-file folder of a Hugging Face model, or None when setup_check has not made one."""
    folder = MODELS_DIR / repo_id.replace("/", "--")
    return folder if (folder / MODEL_DONE).is_file() else None


def weights_file(folder: Path) -> Path:
    """The weight file inside a model folder (timm: model.safetensors)."""
    default = folder / "model.safetensors"
    return default if default.is_file() else sorted(folder.glob("*.safetensors"))[0]


if IN_COLAB:
    try:
        import pytorch_grad_cam  # noqa: F401 - the only package Colab lacks (SPEC §5); installed if missing
    except ImportError:
        get_ipython().run_line_magic("pip", "install -q grad-cam")  # noqa: F821 - %pip, inside a cell

import torch

if SMOKE:
    DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
    print(
        "ℹ️ No GPU in this session, so everything runs on the CPU: fine, but slower. For a GPU, use "
        "Runtime → Change runtime type → T4 GPU, then run this cell again. Every training cell also "
        "offers our precomputed checkpoint."
    )


def seed_everything(seed: int = SEED) -> None:
    """Fix every random source, so re-running a section from its first cell gives the same result (N-013)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()

RESULTS: dict = {"use_fallbacks": {}}  # every section writes its named numbers here (SPEC §9)


def _is_stub(fn) -> bool:
    """True when the function body is still the `...` placeholder (optionally after a docstring)."""
    try:
        tree = ast.parse(inspect.getsource(fn).strip() or "pass")
    except (OSError, TypeError, SyntaxError):
        return False
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    def const(n, kind):
        is_const = isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        return is_const and isinstance(n.value.value, kind)

    body = [n for n in node.body if not const(n, str)]  # drop the docstring
    return all(const(n, type(...)) for n in body)


def _same(a, b) -> bool:
    """Compare two results of any common type (numbers, arrays, tensors, dicts, sequences)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
        b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=1e-5))
    except (TypeError, ValueError):
        return a == b


def use(fn, reference, check=None, name: str | None = None):
    """Return the function the notebook continues with: yours if it matches the reference, else the reference.

    `check(f)` runs `f` on a small play example and returns what to compare (any numbers, arrays or dicts).
    Nothing here is mandatory: an untouched stub (`...` body) silently selects the reference (N-029).
    """
    name = name or fn.__name__
    if _is_stub(fn):
        RESULTS["use_fallbacks"][name] = True
        print(f"ℹ️ {name}: using the reference solution. Fill the stub above to use your own version.")
        return reference
    if check is None:
        RESULTS["use_fallbacks"][name] = False
        print(f"✅ {name}: using your version (no automatic check for this one).")
        return fn
    try:
        yours, theirs = check(fn), check(reference)
    except Exception as err:  # noqa: BLE001 - the participant's code may fail in any way; keep going
        RESULTS["use_fallbacks"][name] = True
        print(f"❌ {name}: your version raised {type(err).__name__}: {err}. Continuing with the reference.")
        return reference
    if _same(yours, theirs):
        RESULTS["use_fallbacks"][name] = False
        print(f"✅ {name}: matches the reference. Continuing with your version.")
        return fn
    RESULTS["use_fallbacks"][name] = True
    print(f"❌ {name}: {yours} vs reference {theirs}. Continuing with the reference.")
    return reference


if not SPECIALISED_JSON.is_file():
    raise SystemExit(f"{SPECIALISED_JSON} is missing. Run setup_check.ipynb first, then run this cell again.")
SPECIALISED: dict = json.loads(SPECIALISED_JSON.read_text())  # the specialised task; never hard-coded
SPECIALISED["dir"] = DATA_DIR / "specialised"
PETS: dict = {"key": "pets", "name": "Oxford-IIIT Pet", "task": "pet vs background"}
PETS |= {"licence": "CC BY-SA 4.0", "dir": DATA_DIR / "pets"}

print(f"Colab: {IN_COLAB} · smoke mode: {SMOKE} · device: {DEVICE} · seed: {SEED}")
print(f"cache: {CACHE_DIR} · specialised task: {SPECIALISED['name']} ({SPECIALISED['task']})")

# %% [markdown]
# <!-- callout: theory | Theory: how use() works -->
# Some cells ask you to fill a function whose body is `...` (the first is `my_iou` in section 1). The line
# after it, `iou = use(my_iou, reference=_ref_iou, ...)`, runs your version and ours on a small play example:
#
# - Stub untouched: the notebook continues with ours.
# - ✅ Yours matches: it continues with yours.
# - ❌ Yours differs or crashes: it prints both results and continues with ours.
#
# Writing the code is where the learning happens; skipping it never breaks anything.

# %% [markdown]
# ### The theme
#
# This cell sets the colours used in every figure: violet means "it works", orange "it fails, we adapt",
# green is the hidden ground truth and teal marks scribbles, the same in every plot. Masks are drawn on a
# grey copy of the photo, so every colour you see belongs to a mask and none to the photo.

# %%
# @title Theme and image helpers { display-mode: "form" }
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

THEME = {
    "works": "#6A1B9A",  # violet: the foundation model does the job
    "fails": "#E65100",  # orange: it falls short / we adapt
    "truth": "#2E7D32",  # green: hidden ground truth, evaluation only
    "scribble": "#00838F",  # teal: strokes a person drew
    "background": "#9E9E9E",  # grey overlay
    "banner": "#4A148C",
}
mpl.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.titlesize": 10,
        "axes.titlecolor": "#311B92",
        "axes.prop_cycle": mpl.cycler(color=[THEME[k] for k in ("works", "fails", "truth", "scribble")]),
    }
)


def hex_rgb(colour: str) -> np.ndarray:
    """'#RRGGBB' -> float array in [0, 1]."""
    return np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255


_OVERLAID = weakref.WeakValueDictionary()  # images overlay() made, so stacking a second mask never re-greys


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colour: str = THEME["truth"],
    alpha: float = 0.75,
    unscored: np.ndarray | None = None,
    dim: bool | None = None,
):
    """Draw a binary mask in one theme colour on a grey, dimmed copy of an RGB uint8 image.

    The photo is turned grey (0.35 to 0.85 brightness) so an orange or green mask stays visible on a red
    retina or a brown dog. `unscored` pixels (not evaluated) are painted light grey. Stacking a second mask
    on the result of a first call keeps the first mask's colour (`dim` is then False unless you set it).
    """
    if dim is None:
        dim = _OVERLAID.get(id(image)) is not image
    out = image.astype(float) / 255
    if dim:
        grey = out @ np.array([0.299, 0.587, 0.114])
        out = np.repeat((0.35 + 0.5 * grey)[..., None], 3, axis=2)
    if unscored is not None:
        out[unscored] = 0.3 * out[unscored] + 0.7 * 0.88
    out[mask] = (1 - alpha) * out[mask] + alpha * hex_rgb(colour)
    result = (out * 255).astype(np.uint8)
    _OVERLAID[id(result)] = result
    return result


def region(item: dict) -> np.ndarray:
    """Where we score: the field of view if the dataset has one, else the whole image."""
    return item["fov"] if item.get("fov") is not None else np.ones_like(item["mask"])


def read_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_item(dataset: dict, item: dict) -> dict:
    """Read one image with its full mask (evaluation only!), scribbles, visible labels and FOV."""
    root = dataset["dir"]
    out = {
        "name": item["name"],
        "split": item["split"],
        "image": np.array(Image.open(root / item["image"]).convert("RGB")),
        "mask": read_png(root / item["mask"]) > 127,
        "scribbles": read_png(root / item["scribbles"]),
        "visible": read_png(root / item["visible"]) > 127,
        "fov": read_png(root / item["fov"]) > 127 if "fov" in item else None,
    }
    out["hidden"] = out["mask"] & ~out["visible"]
    return out


def load_index(dataset: dict) -> dict:
    """Attach the dataset's index.json (facts + one entry per image) and return it."""
    index_file = dataset["dir"] / "index.json"
    if not index_file.is_file():
        raise SystemExit(f"{index_file} is missing. Run setup_check.ipynb first, then run this cell again.")
    dataset["index"] = json.loads(index_file.read_text())
    dataset["items"] = dataset["index"]["items"]
    dataset["fit"] = [it for it in dataset["items"] if it["split"] == "fit"]
    dataset["held"] = [it for it in dataset["items"] if it["split"] == "held"]
    return dataset["index"]

# %% [markdown]
# ### The data
#
# This cell reads the two datasets (pets, and the specialised task) and shows two images of each with the
# ground truth in green and its share of the pixels in the title; the ground truth is for evaluation only.
# Light grey marks pixels nobody scores: on the pets, the "not sure" band that the annotators drew around the
# outline of each animal.

# %%
# @title Load both datasets and show two examples of each { display-mode: "form" }
for ds in (PETS, SPECIALISED):
    idx = load_index(ds)
    print(
        f"{ds['name']}: {len(ds['items'])} images ({len(ds['fit'])} fit, {len(ds['held'])} held-out) · "
        f"{idx['task']}"
    )
PRECOMPUTED_HEADS = json.loads((PRECOMPUTED / "heads" / "index.json").read_text())["heads"]
print(f"precomputed: {len(PRECOMPUTED_HEADS)} head checkpoints in {PRECOMPUTED}")

fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
for k, ds in enumerate((PETS, SPECIALISED)):
    for j, item in enumerate(ds["fit"][:2]):
        it = load_item(ds, item)
        axes[k, 2 * j].imshow(it["image"])
        axes[k, 2 * j].set_title(f"{ds['name']}: {it['name']}")
        not_scored = ~region(it) if ds is PETS else None  # the pets' "not sure" band
        axes[k, 2 * j + 1].imshow(overlay(it["image"], it["mask"], unscored=not_scored))
        axes[k, 2 * j + 1].set_title(f"ground truth · {it['mask'].mean():.0%} of the pixels")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Look at the images above. Which task do you expect the foundation model to solve out of the box, and which
# one will give it trouble? Write one reason; section 2 will tell you whether you were right.
#
# **Q1.** 🟢 What is different about the specialised task that could matter for a model trained on everyday
# photos? Look at the size, shape and contrast of the objects, not the domain name.
#
# <details><summary>One possible answer</summary>
#
# Pets are big and clear; the specialised targets are thin, faint and small.
#
# </details>

# %% [markdown]
# ### The models
#
# This cell defines two wrappers: `dino.features(image, long_side)` gives a grid of image features and
# `sam.probability(image, points, labels)` gives a foreground map from point prompts.

# %%
# @title Model loaders { display-mode: "form" }
import timm
import torch.nn.functional as F

DINO_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_SOURCE = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_LICENCE = "DINOv3 License (Meta)"
SAM_ID = "facebook/sam2.1-hiera-small"  # SPEC §5: SAM 2.1 hiera-small, Apache-2.0, no account needed
SAM_LICENCE = "Apache-2.0"


class Dino:
    """Frozen DINOv3 ViT-S/16: an image in, a grid of 384-dim patch features out."""

    def __init__(self, device: str = DEVICE):
        t = time.monotonic()
        self.device = device
        folder = model_folder(DINO_SOURCE)  # plain files on Drive, else the hub id
        extra = {"pretrained_cfg_overlay": {"file": str(weights_file(folder))}} if folder else {}
        model = timm.create_model(DINO_ID, pretrained=True, num_classes=0, dynamic_img_size=True, **extra)
        self.model = model.eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        self.n_prefix = self.model.num_prefix_tokens
        self.dim = self.model.num_features
        dt = time.monotonic() - t
        print(f"DINOv3 ViT-S/16 · {DINO_SOURCE} · {DINO_LICENCE} · {dt:.0f} s")

    @torch.no_grad()
    def features(self, image: np.ndarray, long_side: int) -> torch.Tensor:
        """Patch features [D, h, w] for the image resized so its long side is `long_side` (multiple of 16)."""
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255
        h, w = image.shape[:2]
        s = long_side / max(h, w)
        H, W = max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16)
        x = F.interpolate(x.to(self.device), size=(H, W), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        f = self.model.forward_features(x)[0, self.n_prefix :]
        return f.reshape(H // 16, W // 16, -1).permute(2, 0, 1).contiguous().float()


class Sam:
    """SAM 2.1 (promptable segmentation): an image plus points or a box in, a foreground probability out."""

    def __init__(self, device: str = DEVICE):
        from transformers import Sam2Model, Sam2Processor

        t = time.monotonic()
        self.device = device
        src = str(model_folder(SAM_ID) or SAM_ID)  # plain files on Drive, else the hub id
        self.proc = Sam2Processor.from_pretrained(src)
        self.model = Sam2Model.from_pretrained(src).eval().to(device)
        self._cache: tuple[np.ndarray, torch.Tensor] | None = None  # (the last image, its embeddings)
        print(f"SAM 2.1 hiera-small · {SAM_ID} · {SAM_LICENCE} · {time.monotonic() - t:.0f} s")

    @torch.no_grad()
    def probability(
        self,
        image: np.ndarray,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        box: list[float] | None = None,
    ) -> np.ndarray:
        """Foreground probability (H x W float) for points (x, y; label 1 = fg, 0 = bg) and/or a box."""
        kw = {}
        if points:
            kw["input_points"], kw["input_labels"] = [[points]], [[labels]]
        if box is not None:
            kw["input_boxes"] = [[box]]
        inputs = self.proc(images=image, return_tensors="pt", **kw)
        if self._cache is None or self._cache[0] is not image:
            self._cache = (image, self.model.get_image_embeddings(inputs["pixel_values"].to(self.device)))
        out = self.model(
            image_embeddings=self._cache[1],
            input_points=inputs["input_points"].to(self.device) if points else None,
            input_labels=inputs["input_labels"].to(self.device) if points else None,
            input_boxes=inputs["input_boxes"].to(self.device) if box is not None else None,
            multimask_output=False,
        )
        masks = out.pred_masks.cpu()
        logits = self.proc.post_process_masks(masks, inputs["original_sizes"], binarize=False)[0]
        return torch.sigmoid(logits[0, 0]).numpy()


# %% [markdown]
# <!-- callout: pitfall | Pitfall: the first slow step -->
# Loading and the first call of a model are the slow steps: PyTorch compiles kernels the first time a model
# sees a new image size. On a T4 this takes seconds; on a CPU it is much slower. Later calls are faster.

# %% [markdown]
# This cell loads DINOv3 and SAM from the cache and prints one line each: name · source · licence · load time.

# %%
# @title Load DINOv3 and SAM { display-mode: "form" }
dino = Dino()
sam = Sam()
RESULTS["models"] = {"dino": DINO_SOURCE, "sam": SAM_ID, "device": DEVICE}

# %% [markdown]
# This cell defines the interactive SAM tool used in sections 1, 2 and 7: click a dot, drag a box, or (in
# section 7) type a word, and SAM answers with a mask you can compare with the ground truth. Run it once.

# %%
# @title Interactive SAM: click, box, or a word { display-mode: "form" }
import base64
import io
import uuid
from string import Template

from IPython.display import HTML, JSON, display

PLAY_PX = 560  # long side of the playground image on screen
SAM3_ID = "facebook/sam3"
_SAM3: dict = {}  # SAM 3 is loaded on the first text request and kept here


def load_sam3() -> dict:
    """SAM 3 from your cache (never the internet): {"model", "proc", "dtype"}, or {"error": a plain line}."""
    if _SAM3:
        return _SAM3
    try:
        from transformers import Sam3Model, Sam3Processor

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32  # fp16 on a T4, never bf16
        src = str(model_folder(SAM3_ID) or SAM3_ID)
        proc = Sam3Processor.from_pretrained(src, local_files_only=True)
        model = Sam3Model.from_pretrained(src, local_files_only=True, dtype=dtype).eval().to(DEVICE)
        _SAM3.update(model=model, proc=proc, dtype=dtype)
    except Exception as err:  # noqa: BLE001 - not cached, no access, an older transformers: all mean "no SAM 3"
        _SAM3["error"] = (
            f"SAM 3 is not in your cache ({type(err).__name__}). To use it, request access on "
            f"huggingface.co/{SAM3_ID}, then run setup_check.ipynb with your HF token."
        )
    return _SAM3


def sam3_masks(image: np.ndarray, text: str, threshold: float = 0.3) -> np.ndarray:
    """Every instance SAM 3 finds for a short noun phrase: bool array [n, H, W] (n may be 0)."""
    s3 = load_sam3()
    if "error" in s3:
        raise RuntimeError(s3["error"])
    inputs = s3["proc"](images=image, text=text.strip(), return_tensors="pt").to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(s3["dtype"])
    with torch.no_grad():
        out = s3["model"](**inputs)
    found = s3["proc"].post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=0.5, target_sizes=[image.shape[:2]]
    )[0]
    return found["masks"].bool().cpu().numpy()


def sam_playground(items: list[dict], title: str, text_prompt: bool = False, default_text: str = "") -> None:
    """Click, box or word prompts on `items` (dicts from load_item); SAM 2.1 answers, with IoU vs the truth.

    Left click = object dot, shift-click = background dot, drag = box. With `text_prompt=True` a text box asks
    SAM 3 (if it is in your cache). Outside Colab and in the smoke run it shows one fixed click and one fixed
    box on the first image instead, so nothing blocks. All the code lives in this function on purpose.
    """
    page = Template("""
<div style="font-family:sans-serif"><b>$title</b><br>
<div id="st$uid" style="position:relative;margin:4px 0">
<img id="im$uid" style="position:absolute;left:0;top:0">
<img id="mk$uid" style="position:absolute;left:0;top:0;display:none;pointer-events:none">
<canvas id="cv$uid" style="position:absolute;left:0;top:0;cursor:crosshair;touch-action:none"></canvas></div>
<div id="ctl$uid"><button id="pv$uid">◀</button> <button id="nx$uid">▶</button>
<button id="pen$uid">➕ object dot</button> <button id="clr$uid">clear</button> $ask
<span id="nm$uid"></span></div>
<div id="msg$uid" style="margin-top:4px;min-height:1.4em">Click = object dot · shift-click = background dot ·
drag = box.</div></div>
<script>
(function () {
  var D = $data, uid = "$uid", k = 0, pts = [], box = null, bg = false, seq = 0, start = null, cur = null;
  function el(s) { return document.getElementById(s + uid); }
  var im = el("im"), mk = el("mk"), cv = el("cv"), ctx = cv.getContext("2d"), msg = el("msg");
  function size(node, w, h) { node.style.width = w + "px"; node.style.height = h + "px"; }
  function xy(e) {
    var r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
  }
  function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height); ctx.lineWidth = 2;
    pts.forEach(function (p) {
      ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 6.3);
      ctx.fillStyle = p[2] ? "#fff" : "#000"; ctx.strokeStyle = p[2] ? "$fg" : "#fff";
      ctx.fill(); ctx.stroke();
    });
    var b = start ? [start[0], start[1], cur[0], cur[1]] : box;
    if (b) {
      ctx.setLineDash([6, 4]); ctx.strokeStyle = "#fff"; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.strokeStyle = "$fg"; ctx.lineDashOffset = 5; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.setLineDash([]); ctx.lineDashOffset = 0;
    }
  }
  function reset() {
    var d = D[k]; pts = []; box = null; seq++;
    im.src = "data:image/jpeg;base64," + d.b64; im.style.filter = ""; mk.style.display = "none";
    cv.width = d.w; cv.height = d.h; size(im, d.w, d.h); size(mk, d.w, d.h); size(cv, d.w, d.h);
    size(el("st"), d.w, d.h); el("nm").textContent = d.name + " (" + (k + 1) + "/" + D.length + ")";
    msg.textContent = "Click = object dot · shift-click = background dot · drag = box."; draw();
  }
  function run(text) {
    if (!(window.google && google.colab && google.colab.kernel)) {
      msg.textContent = "Needs a live Colab session."; return;
    }
    var mine = ++seq; msg.textContent = "SAM is thinking...";
    google.colab.kernel.invokeFunction("fm_workshop.sam_play_" + uid, [k, pts, box, text || ""], {})
      .then(function (r) {
        if (mine !== seq) return;
        var o = r.data["application/json"]; msg.textContent = o.line;
        if (o.png) {
          mk.src = o.png; mk.style.display = "block"; im.style.filter = "grayscale(1) brightness(.75)";
        }
        else { mk.style.display = "none"; im.style.filter = ""; }
      })
      .catch(function (err) { msg.textContent = "Error: " + err; });
  }
  cv.addEventListener("pointerdown", function (e) {
    start = cur = xy(e); cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", function (e) { if (start) { cur = xy(e); draw(); } });
  cv.addEventListener("pointerup", function (e) {
    if (!start) return;
    var s = start, c = xy(e); start = null;
    if (Math.hypot(c[0] - s[0], c[1] - s[1]) < 8) pts.push([s[0], s[1], (e.shiftKey || bg) ? 0 : 1]);
    else box = [Math.min(s[0], c[0]), Math.min(s[1], c[1]), Math.max(s[0], c[0]), Math.max(s[1], c[1])];
    draw(); run("");
  });
  el("pv").onclick = function () { k = (k + D.length - 1) % D.length; reset(); };
  el("nx").onclick = function () { k = (k + 1) % D.length; reset(); };
  el("pen").onclick = function () {
    bg = !bg; this.textContent = bg ? "➖ background dot" : "➕ object dot";
  };
  el("clr").onclick = reset;
  if (el("ask")) el("ask").onclick = function () {
    var t = el("txt").value.trim();
    if (t) run(t); else msg.textContent = "Type a word first, for example the object you want.";
  };
  reset();
})();
</script>
""")

    def score(item: dict, pred: np.ndarray) -> str:
        """'IoU vs ground truth 0.83 · marked 12 %' inside the scored region."""
        r = region(item)
        line = f"marked {(pred & r).sum() / max(1, r.sum()):.0%}"
        if item.get("mask") is not None:
            union = ((pred | item["mask"]) & r).sum()
            line = f"IoU vs ground truth {((pred & item['mask']) & r).sum() / max(1, union):.2f} · " + line
        return line

    def predict(item: dict, pts: list, box: list | None, text: str, scale: float) -> np.ndarray:
        """The binary mask SAM 3 (a word) or SAM 2.1 (dots and box, in image pixels) gives."""
        if text.strip():
            found = sam3_masks(item["image"], text)
            return found.any(0) if len(found) else np.zeros(item["image"].shape[:2], bool)
        prob = sam.probability(
            item["image"],
            points=[[x * scale, y * scale] for x, y, _ in pts] or None,
            labels=[int(label) for _, _, label in pts] or None,
            box=[v * scale for v in box] if box else None,
        )
        return prob > 0.5

    if not IN_COLAB or SMOKE:  # no browser to talk to: one fixed click and one fixed box, as a figure
        item = items[0]
        ys, xs = np.nonzero(item["mask"]) if item.get("mask") is not None else (np.array([]), np.array([]))
        if len(xs):
            near = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
            click = [float(xs[near]), float(ys[near])]
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        else:
            h, w = item["image"].shape[:2]
            click, box = [w / 2, h / 2], [w / 4, h / 4, 3 * w / 4, 3 * h / 4]
        panels = [
            ("click", sam.probability(item["image"], points=[click], labels=[1]) > 0.5),
            ("box", sam.probability(item["image"], box=box) > 0.5),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        axes[0].imshow(item["image"])
        axes[0].plot(*click, "o", color="white", mec=THEME["works"])
        wh = (box[2] - box[0], box[3] - box[1])
        axes[0].add_patch(plt.Rectangle(box[:2], *wh, fill=False, ec="white", ls="--"))
        axes[0].set_title(f"{title}: {item['name']}")
        for ax, (how, pred) in zip(axes[1:], panels, strict=True):
            ax.imshow(overlay(item["image"], pred, THEME["works"]))
            ax.set_title(f"SAM, one {how} · {score(item, pred)}", fontsize=8)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        word = ", or ask SAM 3 with a word" if text_prompt else ""
        print(f"In Colab this is a live canvas: click, shift-click, drag a box{word}.")
        return

    uid = uuid.uuid4().hex[:8]
    shown = []  # per image: the JPEG the canvas shows and its on-screen size
    for item in items:
        h, w = item["image"].shape[:2]
        s = PLAY_PX / max(h, w)
        size = (round(w * s), round(h * s))
        buf = io.BytesIO()
        Image.fromarray(item["image"]).resize(size).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        shown.append({"name": item["name"], "w": size[0], "h": size[1], "b64": b64})
    colour = np.round(hex_rgb(THEME["works"]) * 255).astype(np.uint8)

    def answer(k, pts, box, text) -> JSON:
        """Called by the canvas: run SAM for image k and send back a transparent mask PNG plus one line."""
        item, d = items[int(k)], shown[int(k)]
        if not text.strip() and not pts and box is None:
            return JSON({"png": "", "line": "Click, shift-click or drag first."})
        try:
            pred = predict(item, pts, box, text, item["image"].shape[1] / d["w"])
        except Exception as err:  # noqa: BLE001 - the widget shows one plain line, never a traceback
            line = f"SAM could not run: {type(err).__name__}: {err}"
            line = str(err) if isinstance(err, RuntimeError) else line
            return JSON({"png": "", "line": line})
        soft = Image.fromarray(pred.astype(np.uint8) * 255).resize((d["w"], d["h"]), Image.BILINEAR)
        alpha = np.asarray(soft)
        rgba = np.dstack([np.broadcast_to(colour, (d["h"], d["w"], 3)), (alpha * 0.7).astype(np.uint8)])
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        png = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        who = "SAM 3" if text.strip() else "SAM 2.1"
        line = f"{who} · {score(item, pred)}" if pred.any() else f"{who} found nothing."
        return JSON({"png": png, "line": line})

    from google.colab import output

    output.register_callback(f"fm_workshop.sam_play_{uid}", answer)
    ask = (
        f'<input id="txt{uid}" value="{default_text}" size="18"> <button id="ask{uid}">Ask SAM 3</button>'
        if text_prompt
        else ""
    )
    data = json.dumps(shown).replace("</", "<\\/")
    display(HTML(page.substitute(title=title, uid=uid, data=data, ask=ask, fg=THEME["works"])))

# %% [markdown]
# <!-- callout: note | If something goes wrong -->
# - "Google Drive is not mounted" or "... is missing" → run the setup cell again and click Allow; if the
#   folder is still missing, run `setup_check.ipynb` first.
# - "No GPU in this session" → Runtime → Change runtime type → T4 GPU, then run the setup cell again.
# - A cell is slow on a CPU → normal; training cells have `TRAIN_YOURSELF = False` to load our checkpoint.
# - A cell with a `...` stub → fill it or skip it; `use()` picks yours or ours and the notebook goes on.

# %% [markdown]
# <!-- callout: go-deeper | Recipe and sources -->
# - Feature extractor: **DINOv3** ViT-S/16 (Meta, 2025, via `timm`).
# - Promptable segmenter: **SAM 2.1** hiera-small (Meta).
# - Scribble idea: train a tiny classifier on frozen DINO patch features, using only the pixels a person
#   marked.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - When a cell needs data → `PETS` and `SPECIALISED` are loaded from the cache, with ground truth for
#   evaluation only.
# - When a cell needs a model → `dino` and `sam` are already loaded, nothing is downloaded.
# - When you skip or fail a `...` stub → `use()` continues with our version, nothing breaks.
# - When you read a figure → violet works, orange falls short, green is ground truth, teal is a scribble.
