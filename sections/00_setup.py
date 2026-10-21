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
# **What you need**
#
# - Basic ML: training, validation, objectives, evaluation. Some Python.
# - This notebook open in Google Colab with **Runtime → Change runtime type → T4 GPU**.
# - No computer-vision background. Every step also runs on a CPU, just slower, and each training cell offers a
#   precomputed checkpoint.

# %% [markdown]
# **The plan**
#
# - **Block 1 (sections 1–3):** a foundation model segments pets almost perfectly, then meets a specialised
#   task and falls short.
# - **Block 2 (sections 4–10):** the ways out, from scribbles to prompting the model directly, all compared
#   on the same hidden ground truth, then on your own images.

# %% [markdown]
# <!-- callout: go-deeper | Recipe and sources -->
# - Feature extractor: **DINOv3** ViT-S/16 (Meta, 2025, via `timm`).
# - Promptable segmenter: **SAM 2.1** hiera-small (Meta).
# - Scribble idea: train a tiny classifier on frozen DINO patch features, using only the pixels a person
#   marked (follows the MIT "DINOv2 + scribbles" recipe).

# %% [markdown]
# ## 0. Setup & the question
#
# The next cell detects where you are (Colab or not, GPU or not), fixes the random seed, creates the
# `RESULTS` dictionary every section writes into, and defines `use()`, which lets you write your own code
# without ever blocking the notebook. Run it once.

# %%
# @title Setup (run me first) { display-mode: "form" }
import ast
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

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
# This cell sets the colours used in every figure: cool violet means "it works", warm orange means "it
# fails, we adapt", green is the hidden ground truth and teal marks scribbles. A colour means the same
# thing in every plot, so you can read a figure without its legend.

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


def overlay(image: np.ndarray, mask: np.ndarray, colour: str = THEME["truth"], alpha: float = 0.55):
    """Blend a binary mask in one theme colour over an RGB uint8 image; returns an RGB uint8 image."""
    out = image.astype(float) / 255
    out[mask] = (1 - alpha) * out[mask] + alpha * hex_rgb(colour)
    return (out * 255).astype(np.uint8)


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
# This cell reads the two datasets from the cache, the simple task (pets) and the specialised one, and
# shows two images of each with the ground truth in green. The title gives the share of pixels the target
# covers. The ground truth is for evaluation only: from section 3 on, training sees only incomplete labels.

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
        axes[k, 2 * j + 1].imshow(overlay(it["image"], it["mask"]))
        axes[k, 2 * j + 1].set_title(f"ground truth · {it['mask'].mean():.0%} of the pixels")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: question | 🤔 Your prediction -->
# Look at the images above. Which task do you expect the foundation model to solve out of the box, and which
# one will give it trouble? Write one reason, then keep it: section 2 will tell you whether you were right.
#
# **Q1.** 🟢 What is different about the specialised task that could matter for a model trained on everyday
# photos? Look at the size, shape and contrast of the objects, not the domain name. Answer in your head or in
# a note.
#
# <details><summary>One possible answer</summary>
#
# Pets are big, compact shapes with clear edges; the specialised targets are thin, faint and cover a small
# share of the image.
#
# </details>

# %% [markdown]
# ### The models
#
# This cell defines two small wrappers around the foundation models:
#
# - `dino.features(image, long_side)` gives patch features.
# - `sam.probability(image, points, labels)` gives a foreground map from point prompts.

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
# The first time a model sees a new image size, PyTorch compiles kernels for it. The first call of the next
# section is slower than the ones after; on a CPU the first SAM call on a 1024 px image takes ~20 s.

# %% [markdown]
# This cell loads DINOv3 and SAM from the cache and prints one line each: name · source · licence · load time.

# %%
# @title Load DINOv3 and SAM { display-mode: "form" }
dino = Dino()
sam = Sam()
RESULTS["models"] = {"dino": DINO_SOURCE, "sam": SAM_ID, "device": DEVICE}

# %% [markdown]
# <!-- callout: note | If something goes wrong -->
# - "Google Drive is not mounted" → run the setup cell again and click Allow; if the folder is still missing,
#   run `setup_check.ipynb` first.
# - "No GPU in this session" → Runtime → Change runtime type → T4 GPU, then run the setup cell again.
# - A cell with a `...` stub → fill it or skip it; `use()` picks yours or ours and the notebook goes on.
