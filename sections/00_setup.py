# %% [markdown]
# # When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation
#
# <table width="100%" bgcolor="#4A148C"><tr><td>
# <font color="white" size="6"><b>When Foundation Models Fall Short</b></font><br>
# <font color="white" size="4">From Off-the-Shelf AI to Problem-Specific Adaptation</font><br>
# <font color="white">AMLD 2027 · EPFL, 11 March 2027 &nbsp;·&nbsp; SDS 2027 · Zurich, 1 June 2027</font>
# <br><br>
# <font color="white"><b>The question:</b> what can modern AI solve out of the box, when do its limitations
# start to matter, and what should we do next?</font>
# <br><br>
# <font color="white"><b>At the end of this notebook you will be able to:</b></font>
# <ol>
# <li><font color="white">Judge when a vision foundation model is good enough out of the box, and recognise
# the signs that it is not.</font></li>
# <li><font color="white">Adapt it with cheap, incomplete supervision (scribbles, missing labels,
# pseudo-labels, explanations, prompts), and explain what each approach uses and assumes.</font></li>
# <li><font color="white">Compare approaches on accuracy, annotation effort, reliability and compute, and
# decide what to do next on a new problem, including your own images.</font></li>
# </ol>
# <font color="white">Levels: 🟢 beginner · 🟡 intermediate (the default path) · 🔴 advanced, optional.
# &nbsp; ✍️ = you write a few lines &nbsp; 🖍️ = you draw &nbsp; 👥 = compare with your neighbour.</font>
# <br><br>
# <font color="white"><b>Run <code>setup_check.ipynb</code> first.</b> It was shared with you the week
# before: it puts the models, images and precomputed files in your Google Drive, so nothing is downloaded
# in the room. If you have not run it, do it now (about 5 minutes), then come back here.</font>
# </td></tr></table>
#
# **Objective:** load the cached models and data; see the question and outcomes.
#
# **What you need:** basic ML (training, validation, objectives, evaluation), some Python, this notebook open
# in Google Colab with **Runtime → Change runtime type → T4 GPU**. No computer-vision background is needed.
# Every step also runs on a CPU, just slower, and each training cell offers a precomputed checkpoint.
#
# **The plan.** Block 1 (sections 1–3): a foundation model segments pets almost perfectly, then meets a
# specialised task and falls short. Block 2 (sections 4–10): the ways out, from scribbles to prompting the
# model directly, all compared on the same hidden ground truth, then on your own images.
#
# <table width="100%" bgcolor="#FFF8E1"><tr><td><font color="#5D4037">🧾 <b>Recipe and sources.</b>
# The feature extractor is <b>DINOv3</b> ViT-S/16 (Meta, 2025, via <code>timm</code>) and the promptable
# segmenter is <b>SAM 2.1</b> hiera-small (Meta, Apache-2.0). The scribble idea follows the MIT
# "DINOv2 + scribbles" recipe: a tiny head on frozen patch features, trained with a loss that ignores
# unlabelled pixels. Datasets and licences are listed in section 10 and in <code>data/ATTRIBUTION.md</code>.
# </font></td></tr></table>

# %% [markdown]
# ## 0. Setup & the question
#
# The next cell detects where you are (Colab or not, GPU or not), fixes the random seed, creates the
# `RESULTS` dictionary every section writes into, and defines `use()`, the helper that lets you write your
# own code without ever blocking the notebook. Run it once.

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
    if (CACHE_DIR / "hf").is_dir():
        os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hf")  # model weights, before torch/timm are imported
        os.environ["HF_HUB_OFFLINE"] = "1"  # nothing is downloaded in the room (N-030)
    DATA_DIR, PRECOMPUTED = CACHE_DIR / "data", CACHE_DIR / "precomputed"
    SPECIALISED_JSON = CACHE_DIR / "specialised.json"
else:
    CACHE_DIR = Path("data/smoke" if SMOKE else "data/cache")
    DATA_DIR = CACHE_DIR if SMOKE else CACHE_DIR / "workshop"
    PRECOMPUTED = CACHE_DIR / "precomputed"
    SPECIALISED_JSON = Path("data/specialised.json")

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
# <table width="100%" bgcolor="#EDE7F6"><tr><td><font color="#311B92">📘 <b>Theory: how <code>use()</code>
# works.</b> Several cells later ask you to fill a function whose body is <code>...</code>. The line after
# it, <code>fn = use(my_fn, reference=_ref_fn, check=...)</code>, runs both versions on a small play
# example and prints ✅ <i>matches</i> or ❌ <i>yours vs reference</i>. Either way the notebook continues:
# with your version if it matches, with ours if the stub is untouched or wrong. Writing the code yourself is
# where the learning happens; skipping it never breaks anything.</font></td></tr></table>

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
# shows one image of each with its ground truth in green. The ground truth is for evaluation only: from
# section 3 on, training sees only incomplete labels.

# %%
for ds in (PETS, SPECIALISED):
    idx = load_index(ds)
    print(
        f"{ds['name']}: {len(ds['items'])} images ({len(ds['fit'])} fit, {len(ds['held'])} held-out) · "
        f"{idx['task']} · licence {idx['licence']}"
    )
PRECOMPUTED_HEADS = json.loads((PRECOMPUTED / "heads" / "index.json").read_text())["heads"]
print(f"precomputed: {len(PRECOMPUTED_HEADS)} head checkpoints in {PRECOMPUTED}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for k, ds in enumerate((PETS, SPECIALISED)):
    it = load_item(ds, ds["fit"][0])
    axes[2 * k].imshow(it["image"])
    axes[2 * k].set_title(f"{ds['name']}: {it['name']}")
    axes[2 * k + 1].imshow(overlay(it["image"], it["mask"]))
    axes[2 * k + 1].set_title(f"ground truth: {ds['index']['task']}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# <table width="100%" bgcolor="#FFE8D6"><tr><td><font color="#7A2E00">✏️ <b>Your prediction.</b> Look at the
# two images above. Which task do you expect the foundation model to solve out of the box, and which one
# will give it trouble? Write one reason, then keep it: section 2 will tell you whether you were right.
# </font></td></tr></table>
#
# <table width="100%" bgcolor="#F3E5F5"><tr><td><font color="#4A148C">❓ <b>Question.</b> Answer in your
# head or in a note; a worked answer is folded below.</font></td></tr></table>
#
# **Q1.** 🟢 What is different about the specialised task that could matter for a model trained on everyday
# photos? Think about what the objects look like (size, shape, contrast), not about the domain name.
#
# <details><summary>One possible answer</summary>
#
# Everyday photos show compact objects with clear boundaries, exactly what pets are. The specialised task
# has structures that are thin, branching and low-contrast, and they cover a small fraction of the image.
# A model that learnt "objects" from the web has seen few of those, and its patch features (16 px squares)
# are coarser than the structures themselves. Whether that matters is what the next two sections measure.
#
# </details>

# %% [markdown]
# ### The models
#
# This cell loads the two foundation models from the cache and prints where each one comes from and under
# which licence, then wraps them in two small helpers: `dino.features(image, long_side)` gives patch
# features, `sam.probability(image, points, labels)` gives a foreground map from point prompts.

# %%
# @title Model loaders { display-mode: "form" }
import timm
import torch.nn.functional as F

DINO_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_SOURCE = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_LICENCE = "DINOv3 License (Meta), see the notice below"
SAM_ID = "facebook/sam2.1-hiera-small"  # SPEC §5: SAM 2.1 hiera-small, Apache-2.0, no account needed
SAM_LICENCE = "Apache-2.0"


class Dino:
    """Frozen DINOv3 ViT-S/16: an image in, a grid of 384-dim patch features out."""

    def __init__(self, device: str = DEVICE):
        t = time.monotonic()
        self.device = device
        model = timm.create_model(DINO_ID, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.model = model.eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        self.n_prefix = self.model.num_prefix_tokens
        self.dim = self.model.num_features
        dt = time.monotonic() - t
        print(f"DINOv3 ViT-S/16 · source {DINO_SOURCE} · licence {DINO_LICENCE} · {dt:.0f} s")

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
        self.proc = Sam2Processor.from_pretrained(SAM_ID)
        self.model = Sam2Model.from_pretrained(SAM_ID).eval().to(device)
        self._cache: tuple[np.ndarray, torch.Tensor] | None = None  # (the last image, its embeddings)
        print(f"SAM 2.1 hiera-small · source {SAM_ID} · licence {SAM_LICENCE} · {time.monotonic() - t:.0f} s")

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


# %%
dino = Dino()
sam = Sam()
RESULTS["models"] = {"dino": DINO_SOURCE, "sam": SAM_ID, "device": DEVICE}

# %% [markdown]
# <table width="100%" bgcolor="#FFF8E1"><tr><td><font color="#5D4037">🧾 <b>DINOv3 licence notice.</b>
# DINOv3 is licensed under the <a href="https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m">DINOv3
# License</a>, Copyright © Meta Platforms, Inc. All Rights Reserved. It allows research and commercial use
# under Meta's conditions and acceptable-use policy; keep this notice if you redistribute the weights or
# anything derived from them. The <code>timm</code> re-host used here is open to everyone, no account needed.
# SAM 2.1 is Apache-2.0. Both are re-checked about four weeks before each event, since Meta can change the
# terms.</font></td></tr></table>
#
# <table width="100%" bgcolor="#FDECEA"><tr><td><font color="#8B1A10">⚠️ <b>Pitfall: the first slow step.</b>
# The first time a model sees a new image size, PyTorch compiles kernels for it: the first call of the next
# section is slower than the ones after. On the CPU the first SAM call on a 1024 px image takes ~20 s.
# </font></td></tr></table>
