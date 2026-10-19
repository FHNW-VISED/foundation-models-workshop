"""Bake-off (issue #1): which specialised dataset makes the workshop story work?

Compares three candidate specialised datasets (retina HRF, industrial KolektorSDD2, microscopy
DSB2018) against the simple task (Oxford-IIIT Pet) on the two arms the workshop uses:

  A. SAM 2.1 hiera-small prompted with points sampled from the ground truth and a GT box
  B. a frozen DINOv3 ViT-S/16 (timm) + linear 1x1 head on upsampled patch features, trained under
     the weak-supervision regimes the workshop teaches (full masks, auto-scribbles, hidden
     components as background vs ignore, SAM pseudo-labels, Grad-CAM pseudo-masks on KSDD2)

Usage:
    uv run python scripts/bakeoff.py --download          # only fetch the data into data/cache/
    uv run python scripts/bakeoff.py                     # everything; writes reports/bakeoff/*.json + PNGs
    uv run python scripts/bakeoff.py --datasets pets hrf # subset
    uv run python scripts/bakeoff.py --report            # only regenerate reports/bakeoff.md from the JSONs

Every number in reports/bakeoff.md comes from the JSON files this script writes. The prose verdicts
live in reports/bakeoff/verdicts.md and are spliced in unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "reports" / "bakeoff"
SEED = 42
N_FIT, N_HELD = 10, 10
WORK_LONG = 1024  # images and masks are stored (and evaluated) at this long side at most
FEAT_RES = (512, 1024)  # DINOv3 feature resolutions (long side; 518 is not a multiple of patch 16)
POINT_COUNTS = (1, 5, 20)
SAM_ID = "facebook/sam2.1-hiera-small"
SAM_FALLBACK_ID = "facebook/sam2.1-hiera-tiny"
DINO_ID = "vit_small_patch16_dinov3.lvd1689m"
TRAIN_STEPS = 300
MAX_PNG_BYTES = 300_000

URLS = {
    "hrf": ("https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip", "hrf_all.zip"),
    "ksdd2": ("https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip", "KolektorSDD2.zip"),
    "dsb": ("https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip", "dsb_stage1_train.zip"),
}


# ----------------------------------------------------------------------------- downloads
def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Resumable download with urllib (HTTP Range). Prints the size. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    total = int(head.headers.get("Content-Length") or 0)
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"  cached {dest.name}: {total / 1e6:.1f} MB")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    print(f"  downloading {dest.name}: {total / 1e6:.1f} MB (resume from {have / 1e6:.1f} MB)")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:
            f.seek(0)
            f.truncate()
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
    part.rename(dest)
    print(f"  done {dest.name}: {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    return dest


def unzip(zf: Path, into: Path, members_filter=None) -> Path:
    """Extract once (marker file), optionally only members passing members_filter(name)."""
    marker = into / ".extracted"
    if marker.exists():
        return into
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zf) as z:
        names = [n for n in z.namelist() if members_filter is None or members_filter(n)]
        print(f"  extracting {len(names)} members of {zf.name} -> {into}")
        for n in names:
            z.extract(n, into)
    marker.touch()
    return into


def fetch_all(which: list[str]) -> None:
    if "pets" in which:
        from torchvision.datasets import OxfordIIITPet

        print("Oxford-IIIT Pet (torchvision, CC BY-SA 4.0)")
        OxfordIIITPet(CACHE / "pets", split="trainval", target_types="segmentation", download=True)
    if "hrf" in which:
        print("HRF retina (CC BY 4.0)")
        url, name = URLS["hrf"]
        unzip(download(url, CACHE / name), CACHE / "hrf")
    if "dsb" in which:
        print("DSB2018 stage1_train (CC0)")
        url, name = URLS["dsb"]
        unzip(download(url, CACHE / name), CACHE / "dsb")
    if "ksdd2" in which:
        print("KolektorSDD2 (CC BY-NC-SA 4.0)")
        url, name = URLS["ksdd2"]
        unzip(download(url, CACHE / name), CACHE / "ksdd2", lambda n: "train/" in n)


# ----------------------------------------------------------------------------- data
@dataclass
class Sample:
    name: str
    image: np.ndarray  # H x W x 3 uint8
    mask: np.ndarray  # H x W bool
    fov: np.ndarray | None = None  # H x W bool (retina only); metrics are restricted to it


@dataclass
class Dataset:
    key: str
    title: str
    licence: str
    thin: bool  # vessels: skeleton points for SAM, clDice
    fit: list[Sample] = field(default_factory=list)
    held: list[Sample] = field(default_factory=list)
    native_size: str = ""
    cls: list[tuple[np.ndarray, int]] = field(default_factory=list)  # KSDD2 only: (image, defective)


def _resize(img: np.ndarray, long_side: int, is_mask: bool) -> np.ndarray:
    from PIL import Image

    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img
    s = long_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    if is_mask:
        return np.array(Image.fromarray(img.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))


def _split(samples: list[Sample], rng: random.Random) -> tuple[list[Sample], list[Sample]]:
    rng.shuffle(samples)
    return samples[:N_FIT], samples[N_FIT : N_FIT + N_HELD]


def load_pets(rng: random.Random) -> Dataset:
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(CACHE / "pets", split="trainval", target_types="segmentation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    samples = []
    for i in idx[: N_FIT + N_HELD]:
        img, tri = ds[i]
        img = np.array(img.convert("RGB"))
        tri = np.array(tri)
        mask = (tri == 1) | (tri == 3)  # 1 = pet, 2 = background, 3 = border -> pet
        samples.append(
            Sample(Path(ds._images[i]).stem, _resize(img, WORK_LONG, False), _resize(mask, WORK_LONG, True))
        )
    d = Dataset("pets", "Oxford-IIIT Pet (simple task)", "CC BY-SA 4.0", thin=False)
    d.native_size = f"~{img.shape[1]}x{img.shape[0]} (varies)"
    d.fit, d.held = _split(samples, rng)
    return d


def load_hrf(rng: random.Random) -> Dataset:
    from PIL import Image

    root = CACHE / "hrf"
    samples = []
    for p in sorted((root / "images").glob("*.*")):
        gt = next((root / "manual1").glob(p.stem + ".*"))
        fov = next((root / "mask").glob(p.stem + "_mask.*"))
        img = np.array(Image.open(p).convert("RGB"))
        mask = np.array(Image.open(gt).convert("L")) > 127
        fovm = np.array(Image.open(fov).convert("L")) > 127
        samples.append(
            Sample(
                p.stem,
                _resize(img, WORK_LONG, False),
                _resize(mask, WORK_LONG, True),
                _resize(fovm, WORK_LONG, True),
            )
        )
    d = Dataset("hrf", "HRF retina vessels (thin structures)", "CC BY 4.0", thin=True)
    d.native_size = f"{img.shape[1]}x{img.shape[0]}"
    d.fit, d.held = _split(samples, rng)
    return d


def load_hrfcrop(rng: random.Random) -> Dataset:
    """HRF at native resolution: one random 1024x1024 crop per image, inside the FOV, with >= 5 % vessels.
    Same 20 images as `hrf`, so this isolates the effect of resolution on thin structures."""
    from PIL import Image

    root = CACHE / "hrf"
    nrng = np.random.default_rng(SEED)
    samples = []
    for p in sorted((root / "images").glob("*.*")):
        gt = next((root / "manual1").glob(p.stem + ".*"))
        fov = next((root / "mask").glob(p.stem + "_mask.*"))
        img = np.array(Image.open(p).convert("RGB"))
        mask = np.array(Image.open(gt).convert("L")) > 127
        fovm = np.array(Image.open(fov).convert("L")) > 127
        h, w = mask.shape
        for _ in range(50):
            y, x = int(nrng.integers(0, h - WORK_LONG)), int(nrng.integers(0, w - WORK_LONG))
            sl = (slice(y, y + WORK_LONG), slice(x, x + WORK_LONG))
            if fovm[sl].mean() > 0.95 and mask[sl].mean() >= 0.05:
                break
        samples.append(Sample(p.stem, img[sl].copy(), mask[sl].copy(), fovm[sl].copy()))
    d = Dataset("hrfcrop", "HRF retina vessels, native-resolution 1024 px crops", "CC BY 4.0", thin=True)
    d.native_size = f"{w}x{h}, cropped to {WORK_LONG}x{WORK_LONG}"
    d.fit, d.held = _split(samples, rng)
    return d


def load_dsb(rng: random.Random) -> Dataset:
    from PIL import Image

    root = CACHE / "dsb"
    dirs = sorted(p for p in root.iterdir() if (p / "images").is_dir())
    rng.shuffle(dirs)
    samples = []
    for p in dirs[: N_FIT + N_HELD]:
        img = np.array(Image.open(next((p / "images").glob("*.png"))).convert("RGB"))
        masks = [np.array(Image.open(m).convert("L")) > 0 for m in (p / "masks").glob("*.png")]
        mask = np.any(np.stack(masks), 0) if masks else np.zeros(img.shape[:2], bool)
        samples.append(Sample(p.name[:8], _resize(img, WORK_LONG, False), _resize(mask, WORK_LONG, True)))
    d = Dataset("dsb", "DSB2018 nuclei (microscopy)", "CC0", thin=False)
    d.native_size = "256x256 to 1024x1024 (varies, mixed modalities)"
    d.fit, d.held = _split(samples, rng)
    return d


def load_ksdd2(rng: random.Random) -> Dataset:
    from PIL import Image

    root = CACHE / "ksdd2" / "train"
    gts = sorted(root.glob("*_GT.png"))
    rng.shuffle(gts)
    defective, clean = [], []
    for g in gts:
        img_p = root / g.name.replace("_GT.png", ".png")
        if not img_p.exists():
            continue
        m = np.array(Image.open(g).convert("L")) > 0
        if m.any() and len(defective) < N_FIT + N_HELD:
            img = np.array(Image.open(img_p).convert("RGB"))
            defective.append(
                Sample(g.stem.replace("_GT", ""), _resize(img, WORK_LONG, False), _resize(m, WORK_LONG, True))
            )
        elif not m.any() and len(clean) < 60:
            clean.append(img_p)
        if len(defective) == N_FIT + N_HELD and len(clean) == 60:
            break
    d = Dataset("ksdd2", "KolektorSDD2 surface defects (industrial)", "CC BY-NC-SA 4.0", thin=False)
    d.native_size = f"{img.shape[1]}x{img.shape[0]}"
    d.fit, d.held = _split(defective, rng)
    # image-level classification set for Grad-CAM: the 10 fit defective images + 60 clean ones
    d.cls = [(s.image, 1) for s in d.fit]
    d.cls += [(_resize(np.array(Image.open(p).convert("RGB")), WORK_LONG, False), 0) for p in clean]
    return d


LOADERS = {"pets": load_pets, "hrf": load_hrf, "hrfcrop": load_hrfcrop, "ksdd2": load_ksdd2, "dsb": load_dsb}


# ----------------------------------------------------------------------------- metrics
def _eval_region(s: Sample) -> np.ndarray:
    return s.fov if s.fov is not None else np.ones(s.mask.shape, bool)


def metrics(pred: np.ndarray, s: Sample) -> dict[str, float]:
    from skimage.morphology import skeletonize

    r = _eval_region(s)
    p, g = pred & r, s.mask & r
    inter = float((p & g).sum())
    union = float((p | g).sum())
    out = {
        "iou": inter / union if union else 1.0,
        "dice": 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0,
        "recall": inter / g.sum() if g.sum() else 1.0,
        "precision": inter / p.sum() if p.sum() else 0.0,
    }
    sp, sg = skeletonize(p), skeletonize(g)
    tprec = (sp & g).sum() / sp.sum() if sp.sum() else 0.0
    tsens = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    out["cldice"] = 2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) else 0.0
    return out


def mean_metrics(ms: list[dict[str, float]]) -> dict[str, float]:
    keys = ms[0].keys()
    return {k: float(np.mean([m[k] for m in ms])) for k in keys}


# ----------------------------------------------------------------------------- supervision generators
def skeleton_segments(mask: np.ndarray) -> np.ndarray:
    """Label map of skeleton segments: the GT skeleton cut at junction pixels (0 = not skeleton)."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    sk = skeletonize(mask)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    junction = sk & (nb > 2)
    lab, _ = ndi.label(sk & ~junction, structure=np.ones((3, 3)))
    return lab


def assign_mask_to_segments(mask: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Give every foreground pixel the label of its nearest skeleton segment."""
    from scipy import ndimage as ndi

    if seg.max() == 0:
        return np.where(mask, 1, 0)
    _, (iy, ix) = ndi.distance_transform_edt(seg == 0, return_indices=True)
    return np.where(mask, seg[iy, ix], 0)


def make_scribbles(mask: np.ndarray, rng: np.random.Generator, fov: np.ndarray | None) -> np.ndarray:
    """Auto-scribbles: int8 map, 1 = foreground stroke, 0 = background stroke, -1 = unlabelled.

    Foreground: GT skeleton segments with 70 % dropped. Background: 3-5 random polyline strokes
    that stay > 3 px away from any foreground."""
    from scipy import ndimage as ndi
    from skimage.draw import line

    seg = skeleton_segments(mask)
    n = seg.max()
    keep = rng.random(n + 1) < 0.3
    keep[0] = False
    if n and not keep[1:].any():
        keep[rng.integers(1, n + 1)] = True
    lab = np.full(mask.shape, -1, np.int8)
    lab[keep[seg]] = 1
    dist = ndi.distance_transform_edt(~mask)
    region = (dist > 8) & (fov if fov is not None else True)
    ys, xs = np.nonzero(region)
    h, w = mask.shape
    for _ in range(int(rng.integers(3, 6))):
        if len(ys) == 0:
            break
        k = rng.integers(len(ys))
        y, x = int(ys[k]), int(xs[k])
        for _ in range(3):  # 3 waypoints
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.integers(40, 120)
            y2 = int(np.clip(y + length * np.sin(ang), 0, h - 1))
            x2 = int(np.clip(x + length * np.cos(ang), 0, w - 1))
            rr, cc = line(y, x, y2, x2)
            ok = dist[rr, cc] > 3
            lab[rr[ok], cc[ok]] = np.where(lab[rr[ok], cc[ok]] == 1, 1, 0)
            y, x = y2, x2
    return lab


def hide_half(mask: np.ndarray, rng: np.random.Generator, thin: bool) -> tuple[np.ndarray, np.ndarray]:
    """Hide ~50 % of the foreground: components if there are >= 2, else skeleton segments (vessels, pets).

    Returns (visible_mask, hidden_mask)."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if thin or n < 2:
        seg = skeleton_segments(mask)
        lab = assign_mask_to_segments(mask, seg)
        n = int(lab.max())
    ids = np.arange(1, n + 1)
    rng.shuffle(ids)
    hidden_ids = ids[: max(1, n // 2)] if n >= 2 else ids[:0]
    hidden = np.isin(lab, hidden_ids)
    return mask & ~hidden, hidden


def sample_points(
    mask: np.ndarray, n: int, rng: np.random.Generator, thin: bool, fov: np.ndarray | None
) -> tuple[list[list[float]], list[int]]:
    """n positive + n negative prompt points. Positives: skeleton pixels (thin) or component
    centroids, topped up with interior pixels. Negatives: background > 5 px from the foreground."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    pos: list[tuple[int, int]] = []
    if thin:
        ys, xs = np.nonzero(skeletonize(mask))
        for k in rng.choice(len(ys), size=min(n, len(ys)), replace=False):
            pos.append((int(ys[k]), int(xs[k])))
    else:
        lab, ncomp = ndi.label(mask)
        sizes = ndi.sum(mask, lab, range(1, ncomp + 1))
        order = np.argsort(-np.asarray(sizes))[:n]
        dist = ndi.distance_transform_edt(mask)
        for i in order:
            comp = lab == i + 1
            cy, cx = ndi.center_of_mass(comp)
            if not comp[int(cy), int(cx)]:
                cy, cx = np.unravel_index(np.argmax(np.where(comp, dist, -1)), mask.shape)
            pos.append((int(cy), int(cx)))
        ys, xs = np.nonzero(dist > max(1.0, dist.max() * 0.3))
        while len(pos) < n and len(ys):
            k = rng.integers(len(ys))
            pos.append((int(ys[k]), int(xs[k])))
    far = ndi.distance_transform_edt(~mask) > 5
    if fov is not None:
        far &= fov
    ys, xs = np.nonzero(far)
    neg = [(int(ys[k]), int(xs[k])) for k in rng.choice(len(ys), size=min(n, len(ys)), replace=False)]
    pts = [[float(x), float(y)] for y, x in pos] + [[float(x), float(y)] for y, x in neg]
    return pts, [1] * len(pos) + [0] * len(neg)


def gt_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ----------------------------------------------------------------------------- arm A: SAM 2.1
class Sam:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.id = model_id
        self.device = device
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).eval().to(device)
        self.torch = torch

    def embed(self, image: np.ndarray):
        inputs = self.proc(images=image, return_tensors="pt")
        with self.torch.no_grad():
            emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        return emb, inputs["original_sizes"]

    def predict(self, image, emb, sizes, points=None, labels=None, box=None) -> np.ndarray:
        kw = {}
        if points is not None:
            inputs = self.proc(
                images=image, input_points=[[points]], input_labels=[[labels]], return_tensors="pt"
            )
            kw = {
                "input_points": inputs["input_points"].to(self.device),
                "input_labels": inputs["input_labels"].to(self.device),
            }
        if box is not None:
            inputs = self.proc(images=image, input_boxes=[[box]], return_tensors="pt")
            kw["input_boxes"] = inputs["input_boxes"].to(self.device)
        multimask = points is not None and len(points) == 1 and box is None
        with self.torch.no_grad():
            out = self.model(image_embeddings=emb, multimask_output=multimask, **kw)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), sizes)[0]
        best = int(out.iou_scores[0, 0].argmax()) if multimask else 0
        return masks[0, best].numpy().astype(bool)


def sam_sanity(sam: Sam, s: Sample, rng: np.random.Generator) -> float:
    pts, labs = sample_points(s.mask, 1, rng, False, None)
    emb, sizes = sam.embed(s.image)
    return metrics(sam.predict(s.image, emb, sizes, pts[:1], labs[:1]), s)["iou"]


def run_sam(sam: Sam, d: Dataset, rng: np.random.Generator) -> tuple[dict, dict[str, list[np.ndarray]]]:
    res: dict[str, list[dict]] = {f"{n}pt": [] for n in POINT_COUNTS}
    res["box"] = []
    preds: dict[str, list[np.ndarray]] = {k: [] for k in res}
    embed_t, prompt_t = [], []
    for s in d.held:
        t0 = time.time()
        emb, sizes = sam.embed(s.image)
        embed_t.append(time.time() - t0)
        for n in POINT_COUNTS:
            pts, labs = sample_points(s.mask, n, rng, d.thin, s.fov)
            t0 = time.time()
            m = sam.predict(s.image, emb, sizes, pts, labs)
            prompt_t.append(time.time() - t0)
            res[f"{n}pt"].append(metrics(m, s))
            preds[f"{n}pt"].append(m)
        m = sam.predict(s.image, emb, sizes, box=gt_box(s.mask))
        res["box"].append(metrics(m, s))
        preds["box"].append(m)
    out = {k: mean_metrics(v) for k, v in res.items()}
    out["embed_sec_per_image"] = float(np.mean(embed_t))
    out["prompt_sec"] = float(np.mean(prompt_t))
    return out, preds


# ----------------------------------------------------------------------------- arm B: DINOv3 + linear head
class Dino:
    def __init__(self, device: str):
        import timm
        import torch

        self.torch = torch
        self.device = device
        self.model = (
            timm.create_model(DINO_ID, pretrained=True, num_classes=0, dynamic_img_size=True)
            .eval()
            .to(device)
        )
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.n_prefix = self.model.num_prefix_tokens

    def features(self, image: np.ndarray, long_side: int, device: str | None = None):
        """Patch features (D, h, w) for the image resized to long_side (rounded to multiples of 16)."""
        import torch.nn.functional as F

        dev = device or self.device
        x = self.torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255
        h, w = image.shape[:2]
        s = long_side / max(h, w)
        H, W = max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(dev)
        model = self.model.to(dev)
        with self.torch.no_grad():
            f = model.forward_features(x)[0, self.n_prefix :]
        self.model.to(self.device)
        return f.reshape(H // 16, W // 16, -1).permute(2, 0, 1).contiguous().float()


PLAIN_BCE = ("full-plain", "hidden=bg-plain")  # regimes trained with unweighted BCE (the naive baseline)


def train_head(
    dino: Dino, feats: list, labels: list[np.ndarray], steps: int, lr: float = 0.05, balanced: bool = True
) -> tuple:
    """Linear head (384 -> 1) on standardised patch features, logits upsampled bilinearly to the label
    size (identical to a 1x1 head on bilinearly upsampled features). Balanced (or plain) BCE over
    labelled pixels; label -1 = ignore."""
    import torch
    import torch.nn.functional as F

    dev = dino.device
    mu = torch.cat([f.flatten(1) for f in feats], 1).mean(1).view(-1, 1, 1)
    sd = torch.cat([f.flatten(1) for f in feats], 1).std(1).view(-1, 1, 1) + 1e-6
    feats = [(f - mu) / sd for f in feats]
    labs = [torch.from_numpy(lab.astype(np.int8)).to(dev) for lab in labels]
    n_pos = sum(int((lab == 1).sum()) for lab in labs)
    n_neg = sum(int((lab == 0).sum()) for lab in labs)
    if balanced:
        w_pos, w_neg = 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)
    else:
        w_pos = w_neg = 1.0 / max(n_pos + n_neg, 1)
    head = torch.nn.Conv2d(feats[0].shape[0], 1, 1).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for f, lab in zip(feats, labs, strict=True):
            logit = F.interpolate(head(f[None]), size=lab.shape, mode="bilinear", align_corners=False)[0, 0]
            tgt = (lab == 1).float()
            w = torch.where(lab == 1, w_pos, torch.where(lab == 0, w_neg, 0.0))
            loss = loss + (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum()
        loss.backward()
        opt.step()
    train_sec = time.time() - t0

    def predict(f):
        with torch.no_grad():
            return head(((f - mu) / sd)[None])[0, 0]  # logits at patch resolution

    return predict, train_sec


def head_predict(predict, feat, shape) -> np.ndarray:
    import torch.nn.functional as F

    logit = predict(feat)
    logit = F.interpolate(logit[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
    return (logit > 0).cpu().numpy()


def labelled_fraction(labels: list[np.ndarray]) -> float:
    return float(np.mean([(lab >= 0).mean() for lab in labels]))


# ----------------------------------------------------------------------------- zero-label references
def frangi_otsu(s: Sample) -> np.ndarray:
    from skimage.filters import frangi, threshold_otsu

    g = s.image[..., 1].astype(float) / 255
    v = frangi(1 - g, sigmas=range(1, 4), black_ridges=False)
    r = _eval_region(s)
    thr = threshold_otsu(v[r])
    return (v > thr) & r


def gradcam_ksdd2(
    d: Dataset, device: str, rng: np.random.Generator
) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    """ResNet-18 on defective/clean image labels, Grad-CAM at layer4, thresholded into pseudo-masks."""
    import torch
    import torch.nn.functional as F
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from torchvision.models import ResNet18_Weights, resnet18

    def prep(img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        # fixed size so images that differ by a few pixels batch together (KSDD2: ~231x624)
        x = F.interpolate(x[None], size=(192, 512), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(512, 2)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    xs = [prep(img) for img, _ in d.cls]
    ys = torch.tensor([y for _, y in d.cls])
    t0 = time.time()
    net.train()
    for _ in range(12):
        order = rng.permutation(len(xs))
        for i in range(0, len(order), 8):
            idx = order[i : i + 8]
            xb = torch.cat([xs[j] for j in idx]).to(device)
            if rng.random() < 0.5:
                xb = xb.flip(-1)
            yb = ys[torch.as_tensor(idx)].to(device)
            w = torch.tensor([1.0, 3.0], device=device)  # 10 defective vs 60 clean
            loss = F.cross_entropy(net(xb), yb, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_sec = time.time() - t0
    net.eval()
    with torch.no_grad():
        acc_fit = float(
            np.mean([int(net(x.to(device)).argmax(1).item() == y) for x, y in zip(xs, ys, strict=True)])
        )
        held_pred = [int(net(prep(s.image).to(device)).argmax(1).item()) for s in d.held]
    held_recall = float(np.mean(held_pred))  # all held images are defective
    cam = GradCAM(model=net, target_layers=[net.layer4])

    def cam_mask(img: np.ndarray) -> np.ndarray:
        x = prep(img).to(device)
        m = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
        m = torch.from_numpy(m)[None, None]
        m = F.interpolate(m, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        return m >= 0.5 * m.max() if m.max() > 0 else np.zeros(img.shape[:2], bool)

    fit_masks = [cam_mask(s.image) for s in d.fit]
    held_masks = [cam_mask(s.image) for s in d.held]
    out = {
        "classifier_train_sec": train_sec,
        "classifier_fit_acc": acc_fit,
        "classifier_held_defect_recall": held_recall,
        "cam_pseudo_on_fit": mean_metrics([metrics(m, s) for m, s in zip(fit_masks, d.fit, strict=True)]),
        "cam_direct_on_held": mean_metrics([metrics(m, s) for m, s in zip(held_masks, d.held, strict=True)]),
    }
    return out, fit_masks, held_masks


# ----------------------------------------------------------------------------- per-dataset run
def run_dataset(key: str, device: str, sam: Sam, dino: Dino, steps: int) -> dict:
    rng = np.random.default_rng(SEED)
    prng = random.Random(SEED)
    t_all = time.time()
    d = LOADERS[key](prng)
    print(f"\n== {d.title}: {len(d.fit)} fit / {len(d.held)} held, native {d.native_size}")
    R: dict = {
        "key": key,
        "title": d.title,
        "licence": d.licence,
        "native_size": d.native_size,
        "work_size": f"{d.fit[0].image.shape[1]}x{d.fit[0].image.shape[0]}",
        "n_fit": len(d.fit),
        "n_held": len(d.held),
        "train_steps": steps,
        "fg_fraction": float(np.mean([s.mask[_eval_region(s)].mean() for s in d.fit + d.held])),
        "images_with_fg": float(np.mean([s.mask.any() for s in d.fit + d.held])),
        "components_per_image": float(np.mean([_n_components(s.mask) for s in d.fit + d.held])),
    }

    # ---- arm A
    print("  SAM 2.1 ...", end=" ", flush=True)
    R["sam"], sam_preds = run_sam(sam, d, rng)
    R["sam"]["checkpoint"] = sam.id
    print({k: round(v["iou"], 3) for k, v in R["sam"].items() if isinstance(v, dict)})

    # ---- supervision for the fit images
    scrib = [make_scribbles(s.mask, rng, s.fov) for s in d.fit]
    hidden = [hide_half(s.mask, rng, d.thin) for s in d.fit]
    regimes: dict[str, list[np.ndarray]] = {
        "full": [s.mask.astype(np.int8) for s in d.fit],
        "full-plain": [s.mask.astype(np.int8) for s in d.fit],
        "scribbles": scrib,
        "hidden=bg": [vis.astype(np.int8) for vis, _ in hidden],
        "hidden=bg-plain": [vis.astype(np.int8) for vis, _ in hidden],
        "hidden=ignore": [
            np.where(vis, 1, np.where(_dilate(hid, 3), -1, 0)).astype(np.int8) for vis, hid in hidden
        ],
    }
    # (iv) SAM pseudo-labels from the scribble points
    t0 = time.time()
    pseudo = []
    for s, sc in zip(d.fit, scrib, strict=True):
        fy, fx = np.nonzero(sc == 1)
        by, bx = np.nonzero(sc == 0)
        fi = rng.choice(len(fy), size=min(20, len(fy)), replace=False)
        bi = rng.choice(len(by), size=min(20, len(by)), replace=False)
        pts = [[float(fx[i]), float(fy[i])] for i in fi] + [[float(bx[i]), float(by[i])] for i in bi]
        labs = [1] * len(fi) + [0] * len(bi)
        emb, sizes = sam.embed(s.image)
        pseudo.append(sam.predict(s.image, emb, sizes, pts, labs))
    R["sam_pseudo_sec"] = time.time() - t0
    R["sam_pseudo_on_fit"] = mean_metrics([metrics(m, s) for m, s in zip(pseudo, d.fit, strict=True)])
    regimes["sam-pseudo"] = [m.astype(np.int8) for m in pseudo]
    R["supervision"] = {
        "scribbles_labelled_frac": labelled_fraction(scrib),
        "scribbles_fg_pixels_per_image": float(np.mean([(sc == 1).sum() for sc in scrib])),
        "hidden_fg_frac": float(
            np.mean([hid.sum() / max(s.mask.sum(), 1) for s, (_, hid) in zip(d.fit, hidden, strict=True)])
        ),
    }
    extra_fig: dict[str, list[np.ndarray]] = {}
    if key == "ksdd2":
        print("  Grad-CAM ...", end=" ", flush=True)
        R["gradcam"], cam_fit, cam_held = gradcam_ksdd2(d, device, rng)
        regimes["gradcam-pseudo"] = [m.astype(np.int8) for m in cam_fit]
        extra_fig["gradcam-direct"] = cam_held
        print(round(R["gradcam"]["cam_direct_on_held"]["iou"], 3))
    if d.thin:
        t0 = time.time()
        fr = [frangi_otsu(s) for s in d.held]
        R["frangi"] = mean_metrics([metrics(m, s) for m, s in zip(fr, d.held, strict=True)])
        R["frangi"]["sec_per_image"] = (time.time() - t0) / len(d.held)
        extra_fig["frangi"] = fr
        print(f"  Frangi+Otsu iou {R['frangi']['iou']:.3f} clDice {R['frangi']['cldice']:.3f}")

    # ---- arm B
    R["dino"] = {"feat_sec_per_image": {}, "cpu_feat_sec_per_image": {}, "regimes": {}}
    head_preds: dict[str, list[np.ndarray]] = {}
    for res in FEAT_RES:
        t0 = time.time()
        feats_fit = [dino.features(s.image, res) for s in d.fit]
        feats_held = [dino.features(s.image, res) for s in d.held]
        _sync(device)
        R["dino"]["feat_sec_per_image"][str(res)] = (time.time() - t0) / (len(d.fit) + len(d.held))
        t0 = time.time()
        dino.features(d.fit[0].image, res, device="cpu")
        R["dino"]["cpu_feat_sec_per_image"][str(res)] = time.time() - t0
        for name, labels in regimes.items():
            predict, train_sec = train_head(dino, feats_fit, labels, steps, balanced=name not in PLAIN_BCE)
            preds = [head_predict(predict, f, s.mask.shape) for f, s in zip(feats_held, d.held, strict=True)]
            m = mean_metrics([metrics(p, s) for p, s in zip(preds, d.held, strict=True)])
            m["train_sec"] = train_sec
            m["labelled_frac"] = labelled_fraction(labels)
            R["dino"]["regimes"][f"{res}/{name}"] = m
            head_preds[f"{res}/{name}"] = preds
            print(
                f"  DINOv3 {res:4d} {name:15s} iou {m['iou']:.3f} dice {m['dice']:.3f} "
                f"clDice {m['cldice']:.3f} ({train_sec:.0f} s)"
            )
        if res == FEAT_RES[0]:
            t0 = time.time()
            cpu_feats = [f.cpu() for f in feats_fit]
            cpu_dino = _CpuView(dino)
            train_head(cpu_dino, cpu_feats, regimes["full"], steps)
            R["dino"]["cpu_train_sec_full_512"] = time.time() - t0
    # how different are the weak-supervision predictions from each other? (mean pairwise IoU)
    names = [f"{FEAT_RES[-1]}/{n}" for n in regimes]
    pair = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair.append(
                float(np.mean([_iou(x, y) for x, y in zip(head_preds[a], head_preds[b], strict=True)]))
            )
    R["dino"]["mean_pairwise_pred_iou_1024"] = float(np.mean(pair))
    R["dino"]["min_pairwise_pred_iou_1024"] = float(np.min(pair))

    # ---- figures
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sam(d, sam_preds, OUT / f"{key}_1_sam.png")
    fig_heads(d, head_preds, extra_fig, OUT / f"{key}_2_heads.png")
    fig_supervision(d, scrib, hidden, pseudo, regimes.get("gradcam-pseudo"), OUT / f"{key}_3_supervision.png")
    R["total_sec"] = time.time() - t_all
    R["device"] = device
    (OUT / f"{key}.json").write_text(json.dumps(R, indent=1))
    print(f"  total {R['total_sec']:.0f} s")
    return R


class _CpuView:
    """The Dino object with device switched to CPU, to time a head training on CPU."""

    def __init__(self, dino: Dino):
        self.torch, self.device = dino.torch, "cpu"


def _n_components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi

    return int(ndi.label(mask)[1])


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_dilation(m, iterations=r)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------- figures
def _show(ax, img, mask=None, color=(0.9, 0.32, 0.0), title=""):
    ax.imshow(img)
    if mask is not None:
        over = np.zeros((*mask.shape, 4))
        over[mask] = (*color, 0.55)
        ax.imshow(over)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _disp(arr: np.ndarray, is_mask: bool, long_side: int = 384) -> np.ndarray:
    """Display copy: shrunk, and tall-and-narrow images (KSDD2) rotated to landscape."""
    out = _resize(arr, long_side, is_mask)
    return np.rot90(out) if out.shape[0] > 2 * out.shape[1] else out


def _small(s: Sample, long_side: int = 384):
    return _disp(s.image, False, long_side), _disp(s.mask, True, long_side)


def _savefig(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    for dpi in (80, 64, 50, 40):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if path.stat().st_size <= MAX_PNG_BYTES:
            break
    plt.close(fig)


def fig_sam(d: Dataset, preds: dict[str, list[np.ndarray]], path: Path) -> None:
    import matplotlib.pyplot as plt

    cols = ["image", "ground truth"] + list(preds)
    rows = [0, 1, 2]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows) * 0.8 + 0.4))
    for r, i in enumerate(rows):
        s = d.held[i]
        img, gt = _small(s)
        _show(axes[r, 0], img, title=s.name)
        _show(axes[r, 1], img, gt, (0.18, 0.49, 0.2), "GT")
        for c, k in enumerate(preds):
            m = _disp(preds[k][i], True)
            _show(axes[r, c + 2], img, m, title=f"SAM {k}: IoU {_iou(m, gt):.2f}")
    fig.suptitle(f"{d.title}: SAM 2.1 with GT-sampled prompts (held-out images)", fontsize=9)
    _savefig(fig, path)


def fig_heads(
    d: Dataset, preds: dict[str, list[np.ndarray]], extra: dict[str, list[np.ndarray]], path: Path
) -> None:
    import matplotlib.pyplot as plt

    keys = [k for k in preds if k.startswith(str(FEAT_RES[-1]))] + [f"{FEAT_RES[0]}/full"]
    cols = ["ground truth"] + keys + list(extra)
    rows = [0, 1, 2]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows) * 0.8 + 0.4))
    for r, i in enumerate(rows):
        s = d.held[i]
        img, gt = _small(s)
        _show(axes[r, 0], img, gt, (0.18, 0.49, 0.2), f"{s.name} GT")
        for c, k in enumerate(keys):
            m = _disp(preds[k][i], True)
            _show(axes[r, c + 1], img, m, (0.42, 0.11, 0.6), f"{k}\nIoU {_iou(m, gt):.2f}")
        for c, k in enumerate(extra):
            m = _disp(extra[k][i], True)
            _show(axes[r, c + 1 + len(keys)], img, m, (0.9, 0.32, 0.0), f"{k}\nIoU {_iou(m, gt):.2f}")
    fig.suptitle(f"{d.title}: DINOv3 linear head under each supervision regime (held-out)", fontsize=9)
    _savefig(fig, path)


def fig_supervision(d: Dataset, scrib, hidden, pseudo, cam, path: Path) -> None:
    import matplotlib.pyplot as plt

    cols = ["ground truth", "auto-scribbles", "hidden 50 %", "SAM pseudo-label"] + (
        ["Grad-CAM pseudo"] if cam else []
    )
    rows = [0, 1, 2]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.2 * len(cols), 2.2 * len(rows) * 0.8 + 0.4))
    for r, i in enumerate(rows):
        s = d.fit[i]
        img, gt = _small(s)
        _show(axes[r, 0], img, gt, (0.18, 0.49, 0.2), f"{s.name} GT")
        sc = scrib[i]
        fg = _disp(_dilate(sc == 1, 5), True)  # thickened for display only
        bg = _disp(_dilate(sc == 0, 5), True)
        _show(axes[r, 1], img, fg, (0.0, 0.52, 0.56), "scribbles (fg teal, bg grey)")
        over = np.zeros((*bg.shape, 4))
        over[bg] = (0.3, 0.3, 0.3, 0.8)
        axes[r, 1].imshow(over)
        vis, hid = hidden[i]
        _show(
            axes[r, 2], img, _disp(vis, True), (0.18, 0.49, 0.2), "visible (green) / hidden (orange)"
        )
        over = np.zeros((*bg.shape, 4))
        over[_disp(hid, True)] = (0.9, 0.32, 0.0, 0.55)
        axes[r, 2].imshow(over)
        m = _disp(pseudo[i], True)
        _show(axes[r, 3], img, m, (0.42, 0.11, 0.6), f"SAM from scribble pts\nIoU {_iou(m, gt):.2f}")
        if cam:
            m = _disp(cam[i].astype(bool), True)
            _show(axes[r, 4], img, m, (0.9, 0.32, 0.0), f"Grad-CAM >= 0.5 max\nIoU {_iou(m, gt):.2f}")
    fig.suptitle(f"{d.title}: the incomplete supervision generated for the fit images", fontsize=9)
    _savefig(fig, path)


# ----------------------------------------------------------------------------- report
def _f(x: float) -> str:
    return f"{x:.3f}"


def dataset_table(R: dict) -> str:
    thin = R["key"].startswith("hrf")
    cl = " | clDice" if thin else ""
    lines = [
        f"### {R['title']}",
        "",
        f"Licence {R['licence']} · native {R['native_size']} · evaluated at {R['work_size']} long side · "
        f"{R['n_fit']} fit / {R['n_held']} held-out · foreground {100 * R['fg_fraction']:.1f} % of pixels · "
        f"{R['components_per_image']:.1f} components/image · {100 * R['images_with_fg']:.0f} % of images "
        f"contain foreground · device {R['device']} · total {R['total_sec']:.0f} s",
        "",
        f"| Approach | Supervision used | IoU | Dice | Recall{cl} | Seconds |",
        f"|---|---|---|---|---|{'---|' if thin else ''}---|",
    ]

    def row(name, sup, m, secs):
        c = f" {_f(m['cldice'])} |" if thin else ""
        lines.append(f"| {name} | {sup} | {_f(m['iou'])} | {_f(m['dice'])} | {_f(m['recall'])} |{c} {secs} |")

    S = R["sam"]
    for n in POINT_COUNTS:
        row(
            f"SAM 2.1, {n}+{n} points",
            "GT-sampled points, no training",
            S[f"{n}pt"],
            f"{S['prompt_sec']:.2f}/prompt",
        )
    row("SAM 2.1, GT box", "GT box, no training", S["box"], f"{S['embed_sec_per_image']:.2f}/image embed")
    if "frangi" in R:
        row("Frangi + Otsu (1998 filter)", "none", R["frangi"], f"{R['frangi']['sec_per_image']:.1f}/image")
    if "gradcam" in R:
        G = R["gradcam"]
        row(
            "Grad-CAM mask, direct",
            "image-level labels (10 defective + 60 clean)",
            G["cam_direct_on_held"],
            f"{G['classifier_train_sec']:.0f} train",
        )
    D = R["dino"]
    hid = f"{100 * R['supervision']['hidden_fg_frac']:.0f} % of fg hidden"
    pseudo_iou = _f(R["sam_pseudo_on_fit"]["iou"])
    sup = {
        "full": "full masks",
        "full-plain": "full masks, plain (unweighted) BCE",
        "scribbles": f"auto-scribbles ({100 * R['supervision']['scribbles_labelled_frac']:.2f} % of pixels)",
        "hidden=bg": f"{hid}, missing = background",
        "hidden=bg-plain": f"{hid}, missing = background, plain BCE (naive baseline)",
        "hidden=ignore": f"{hid}, missing = ignore (oracle)",
        "sam-pseudo": f"SAM pseudo-labels from scribble points (IoU {pseudo_iou} vs GT)",
        "gradcam-pseudo": "Grad-CAM pseudo-masks on fit images"
        + (f" (IoU {_f(R['gradcam']['cam_pseudo_on_fit']['iou'])} vs GT)" if "gradcam" in R else ""),
    }
    for k, m in D["regimes"].items():
        res, name = k.split("/")
        row(f"DINOv3 head @{res}, {name}", sup[name], m, f"{m['train_sec']:.1f} train")
    fs, cs = D["feat_sec_per_image"], D["cpu_feat_sec_per_image"]
    lines += [
        "",
        f"Feature extraction per image on {R['device']}: {fs['512']:.2f} s @512, {fs['1024']:.2f} s @1024; "
        f"on CPU: {cs['512']:.2f} s @512, {cs['1024']:.2f} s @1024. "
        f"Head training on CPU (full masks, @512, 10 images, {R.get('train_steps', TRAIN_STEPS)} steps): "
        f"{D['cpu_train_sec_full_512']:.0f} s. "
        f"SAM pseudo-labelling of the 10 fit images: {R['sam_pseudo_sec']:.0f} s.",
        f"Disagreement between the 1024 heads' predictions: mean pairwise IoU "
        f"{_f(D['mean_pairwise_pred_iou_1024'])}, min {_f(D['min_pairwise_pred_iou_1024'])} "
        "(lower = more visibly different).",
    ]
    if "gradcam" in R:
        G = R["gradcam"]
        lines.append(
            f"Grad-CAM classifier: fit accuracy {_f(G['classifier_fit_acc'])}, held-out defect recall "
            f"{_f(G['classifier_held_defect_recall'])}; CAM pseudo-masks on the fit images IoU "
            f"{_f(G['cam_pseudo_on_fit']['iou'])}, recall {_f(G['cam_pseudo_on_fit']['recall'])}."
        )
    lines += [
        "",
        "Figures: "
        + ", ".join(
            f"`bakeoff/{R['key']}_{i}_{n}.png`" for i, n in ((1, "sam"), (2, "heads"), (3, "supervision"))
        ),
        "",
    ]
    return "\n".join(lines)


def write_report() -> None:
    order = ["pets", "hrf", "hrfcrop", "ksdd2", "dsb"]
    Rs = [json.loads((OUT / f"{k}.json").read_text()) for k in order if (OUT / f"{k}.json").exists()]
    verdicts = (
        (OUT / "verdicts.md").read_text()
        if (OUT / "verdicts.md").exists()
        else "_verdicts not written yet_\n"
    )
    dev = Rs[0]["device"] if Rs else "?"
    head = [
        "# Bake-off: which specialised dataset makes the story work? (issue #1)",
        "",
        f"Generated by `scripts/bakeoff.py` on {time.strftime('%Y-%m-%d')}, {platform.machine()} / {dev}. "
        "Every number below is read from `reports/bakeoff/<dataset>.json`, written by that script "
        f"(seed {SEED}, {N_FIT} fit / {N_HELD} held-out images per dataset; no estimates).",
        "",
        "**Arm A** = SAM 2.1 hiera-small (`transformers`), prompted with points sampled from the ground "
        "truth (skeleton points for vessels, component centroids otherwise; as many negatives as positives) "
        "or the GT box. "
        "**Arm B** = frozen DINOv3 ViT-S/16 (`timm`) patch features at 512 and 1024 long side (518 is not a "
        f"multiple of the 16 px patch) with a linear 1x1 head trained {TRAIN_STEPS} Adam steps with balanced "
        "BCE on the labelled pixels. Metrics are means over the 10 held-out images, computed at the "
        "evaluation resolution (inside the FOV for retina).",
        "",
        "## Summary: zero-shot gap and best weak-supervision rung",
        "",
        "| Dataset | SAM 1+1 pt IoU | SAM 20+20 pt IoU | SAM box IoU | Head full @1024 IoU | "
        "Head scribbles @1024 IoU | hidden=bg (plain BCE) recall | hidden=ignore recall | Total s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for R in Rs:
        g, S = R["dino"]["regimes"], R["sam"]
        head.append(
            f"| {R['title']} | {_f(S['1pt']['iou'])} | {_f(S['20pt']['iou'])} | {_f(S['box']['iou'])} | "
            f"{_f(g['1024/full']['iou'])} | {_f(g['1024/scribbles']['iou'])} | "
            f"{_f(g['1024/hidden=bg-plain']['recall'])} | {_f(g['1024/hidden=ignore']['recall'])} | "
            f"{R['total_sec']:.0f} |"
        )
    body = "\n".join(head) + "\n\n## Per-dataset tables\n\n" + "\n".join(dataset_table(R) for R in Rs)
    (ROOT / "reports" / "bakeoff.md").write_text(body + "\n## Verdicts\n\n" + verdicts)
    print(f"wrote {ROOT / 'reports' / 'bakeoff.md'}")


# ----------------------------------------------------------------------------- main
def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--download", action="store_true", help="only download/extract")
    ap.add_argument("--report", action="store_true", help="only rebuild reports/bakeoff.md from JSON")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS, help="head training steps")
    args = ap.parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.download:
        fetch_all(args.datasets)
        return 0
    if args.report:
        write_report()
        return 0
    import matplotlib
    import torch

    matplotlib.use("Agg")
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")
    fetch_all(args.datasets)
    sam = Sam(SAM_ID, device)
    rng = np.random.default_rng(SEED)
    pet = load_pets(random.Random(SEED)).held[0]
    sanity = sam_sanity(sam, pet, rng)
    print(f"SAM sanity (one point on a pet): IoU {sanity:.3f} with {SAM_ID}")
    if sanity < 0.3:
        print(f"  looks broken -> switching to {SAM_FALLBACK_ID}")
        sam = Sam(SAM_FALLBACK_ID, device)
        sanity = sam_sanity(sam, pet, rng)
        print(f"  SAM sanity with fallback: IoU {sanity:.3f}")
    dino = Dino(device)
    for key in args.datasets:
        R = run_dataset(key, device, sam, dino, args.steps)
        R["sam"]["sanity_iou_one_point_pet"] = sanity
        (OUT / f"{key}.json").write_text(json.dumps(R, indent=1))
    write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
