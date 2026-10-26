"""Prepare the workshop data (issue #2): subsets, incomplete-supervision variants and smoke fixtures.

Two datasets, both driven by config: the simple task (Oxford-IIIT Pet, constants below) and the
specialised task (`data/specialised.json`: name, source, licence, preprocessing, scribble generator,
metric extras). Swap the specialised dataset by editing that JSON and re-running this script.

Usage:
    uv run python scripts/prepare_data.py             # download (idempotent) + workshop cache + smoke
    uv run python scripts/prepare_data.py --download  # only fetch the raw archives into data/cache/
    uv run python scripts/prepare_data.py --smoke     # only rebuild data/smoke/ (committed, <= 2 MB)
    uv run python scripts/prepare_data.py --only-pets # rebuild pets only, specialised stays byte-identical

Layout (one folder per dataset, identical for the workshop cache and the smoke fixtures):

    data/cache/workshop/<pets|specialised>/    data/smoke/<pets|specialised>/
        index.json         dataset facts, encoding, one entry per image (name, split, file paths, stats)
        images/NAME.jpg    RGB image at the working size
        masks/NAME.png     full ground truth (0/255)         -> EVALUATION ONLY (SPEC L-005)
        fov/NAME.png       evaluation region (retina: inside the FOV; pets: all but the not-sure band)
        scribbles/NAME.png auto-scribbles: 0 = unlabelled, 1 = foreground stroke, 2 = background stroke
        visible/NAME.png   "missing labels": the ground truth with ~50 % of the foreground hidden (0/255)
        full/, full_masks/, full_fov/   (specialised, crop mode only) the whole image downsampled

Everything random is seeded per image from its name, so re-running gives byte-identical variants
and a dataset swap does not shuffle the pets.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
WORKSHOP = CACHE / "workshop"
SMOKE_DIR = DATA / "smoke"
SPECIALISED_JSON = DATA / "specialised.json"
SEED = 42
SMOKE_LONG_SIDE = 256
SMOKE_PER_DATASET = 4  # SPEC §4: 4 pets + 4 specialised images
SMOKE_BUDGET_BYTES = 2_000_000
JPEG_QUALITY = 92

PETS = {
    "key": "pets",
    "name": "Oxford-IIIT Pet",
    "task": "pet vs background",
    "licence": "CC BY-SA 4.0",
    "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    "citation": "Parkhi OM, Vedaldi A, Zisserman A, Jawahar CV. Cats and Dogs. CVPR 2012.",
    "n_images": 100,
    "split": {"fit": 40, "held": 60},
    "preprocessing": {"mode": "resize", "long_side": 1024},
    "feature_long_side": 512,
    "feature_long_sides": [512],
    "thin_structures": False,
    "eval_region": "trimap: everything except the not-sure band (class 3) around the outline",
    "scribble_generator": {
        "type": "skeleton_segments",
        "fg_keep_fraction": 0.3,
        "bg_strokes": [3, 5],
        "bg_margin_px": 8,
        "bg_stroke_len_px": [40, 120],
        "bg_waypoints": 3,
    },
    "missing_labels": {"hidden_fraction": 0.5, "unit": "component_or_skeleton_segment"},
    "metric_extras": [],
    "paths": {"workshop": "data/cache/workshop/pets", "smoke": "data/smoke/pets"},
}


def load_specialised() -> dict:
    return json.loads(SPECIALISED_JSON.read_text())


def rng_for(name: str, salt: str = "") -> np.random.Generator:
    """One deterministic generator per image (and purpose), independent of processing order."""
    return np.random.default_rng([SEED, zlib.crc32(f"{name}/{salt}".encode())])


# ----------------------------------------------------------------------------- downloads
def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Resumable download (HTTP Range), idempotent: a complete file is not fetched again."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    total = int(head.headers.get("Content-Length") or 0)
    if dest.exists() and (not total or dest.stat().st_size == total):
        print(f"  cached {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
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


def unzip(zf: Path, into: Path) -> Path:
    marker = into / ".extracted"
    if marker.exists():
        return into
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zf) as z:
        print(f"  extracting {len(z.namelist())} members of {zf.name} -> {into}")
        z.extractall(into)
    marker.touch()
    return into


def fetch(spec: dict) -> None:
    from torchvision.datasets import OxfordIIITPet

    print(f"{PETS['name']} (torchvision, {PETS['licence']})")
    OxfordIIITPet(CACHE / "pets", split="trainval", target_types="segmentation", download=True)
    print(f"{spec['name']} ({spec['licence']})")
    unzip(download(spec["source_url"], CACHE / spec["archive_name"]), CACHE / spec["loader"])


# ----------------------------------------------------------------------------- samples
@dataclass
class Sample:
    name: str
    image: np.ndarray  # H x W x 3 uint8
    mask: np.ndarray  # H x W bool, the full ground truth
    fov: np.ndarray | None = None  # evaluation region (retina)
    split: str = "held"
    full: dict = field(default_factory=dict)  # crop mode: {"image", "mask", "fov", "crop_xywh"}


def resize(arr: np.ndarray, long_side: int, is_mask: bool) -> np.ndarray:
    """Shrink to `long_side` (never enlarge); masks with nearest neighbour."""
    from PIL import Image

    h, w = arr.shape[:2]
    if max(h, w) <= long_side:
        return arr
    s = long_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    if is_mask:
        return np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return np.array(Image.fromarray(arr).resize(size, Image.BILINEAR))


def assign_splits(samples: list[Sample], split: dict[str, int]) -> list[Sample]:
    """Shuffle once with the fixed seed, then the first `fit` are fit, the rest held-out."""
    order = sorted(samples, key=lambda s: s.name)
    random.Random(SEED).shuffle(order)
    for i, s in enumerate(order):
        s.split = "fit" if i < split["fit"] else "held"
    return order


def load_pets(n: int | None = None) -> list[Sample]:
    """The first `n` of the seeded shuffle of trainval (the bake-off used the first 20 of the same order)."""
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(CACHE / "pets", split="trainval", target_types="segmentation")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    long = PETS["preprocessing"]["long_side"]
    samples = []
    for i in idx[: n or PETS["n_images"]]:
        img, tri = ds[i]
        img = np.array(img.convert("RGB"))
        tri = np.array(tri)
        # Trimap: 1 = pet, 2 = background, 3 = "not sure" band (~10-20 px along the outline).
        # Pet = class 1 only. The not-sure band is neither pet nor background: it is left out of the
        # evaluation region (stored as the item's fov PNG, exactly like the retina FOV).
        mask = tri == 1
        scored = tri != 3
        samples.append(
            Sample(
                Path(ds._images[i]).stem,
                resize(img, long, False),
                resize(mask, long, True),
                resize(scored, long, True),
            )
        )
    return assign_splits(samples, PETS["split"])


def load_hrf(spec: dict) -> list[Sample]:
    """HRF: every image with its manual vessel mask and FOV mask, cropped or resized per the config."""
    from PIL import Image

    root = CACHE / "hrf"
    pre = spec["preprocessing"]
    samples = []
    for p in sorted((root / "images").glob("*.*")):
        gt = next((root / "manual1").glob(p.stem + ".*"))
        fov = next((root / "mask").glob(p.stem + "_mask.*"))
        img = np.array(Image.open(p).convert("RGB"))
        mask = np.array(Image.open(gt).convert("L")) > 127
        fovm = np.array(Image.open(fov).convert("L")) > 127
        if pre["mode"] == "crop":
            s = crop_sample(p.stem, img, mask, fovm, pre)
        else:
            long = pre["size"]
            s = Sample(p.stem, resize(img, long, False), resize(mask, long, True), resize(fovm, long, True))
        samples.append(s)
    return assign_splits(samples, spec["split"])


def crop_sample(name: str, img: np.ndarray, mask: np.ndarray, fov: np.ndarray, pre: dict) -> Sample:
    """One seeded random square crop inside the FOV with enough foreground, plus the downsampled whole."""
    size = pre["size"]
    rng = rng_for(name, "crop")
    h, w = mask.shape
    best, best_fg = None, -1.0
    for _ in range(100):
        y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
        sl = (slice(y, y + size), slice(x, x + size))
        fg = float(mask[sl].mean())
        if fov[sl].mean() > pre["min_fov_fraction"] and fg > best_fg:
            best, best_fg = (y, x), fg
            if fg >= pre["min_fg_fraction"]:
                break
    if best is None:
        raise RuntimeError(f"{name}: no crop of {size} px fits inside the field of view")
    y, x = best
    sl = (slice(y, y + size), slice(x, x + size))
    long = pre["full_view_long_side"]
    full = {
        "image": resize(img, long, False),
        "mask": resize(mask, long, True),
        "fov": resize(fov, long, True),
        "crop_xywh": [x, y, size, size],
    }
    return Sample(name, img[sl].copy(), mask[sl].copy(), fov[sl].copy(), full=full)


LOADERS = {"hrf": load_hrf}


def load_specialised_samples(spec: dict) -> list[Sample]:
    return LOADERS[spec["loader"]](spec)


# ----------------------------------------------------------------------------- incomplete supervision
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


def make_scribbles(
    mask: np.ndarray, rng: np.random.Generator, fov: np.ndarray | None, cfg: dict, scale: float = 1.0
) -> tuple[np.ndarray, int, int]:
    """Auto-scribbles as a uint8 map: 0 = unlabelled, 1 = foreground stroke, 2 = background stroke.

    Foreground strokes: GT skeleton segments, (1 - fg_keep_fraction) of them dropped. Background
    strokes: random polylines in the FOV that stay away from the foreground. `scale` shrinks the
    pixel parameters for small (smoke) images. Returns (map, n_fg_strokes, n_bg_strokes)."""
    from scipy import ndimage as ndi
    from skimage.draw import line

    seg = skeleton_segments(mask)
    n = int(seg.max())
    keep = rng.random(n + 1) < cfg["fg_keep_fraction"]
    keep[0] = False
    if n and not keep[1:].any():
        keep[rng.integers(1, n + 1)] = True
    lab = np.zeros(mask.shape, np.uint8)
    lab[keep[seg]] = 1
    n_fg = int(keep[1:].sum())
    margin = max(2, round(cfg["bg_margin_px"] * scale))
    lo, hi = (max(5, round(v * scale)) for v in cfg["bg_stroke_len_px"])
    dist = ndi.distance_transform_edt(~mask)
    region = (dist > margin) & (fov if fov is not None else True)
    ys, xs = np.nonzero(region)
    h, w = mask.shape
    n_bg = 0
    for _ in range(int(rng.integers(cfg["bg_strokes"][0], cfg["bg_strokes"][1] + 1))):
        if len(ys) == 0:
            break
        k = rng.integers(len(ys))
        y, x = int(ys[k]), int(xs[k])
        drawn = False
        for _ in range(cfg["bg_waypoints"]):
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.integers(lo, hi + 1)
            y2 = int(np.clip(y + length * np.sin(ang), 0, h - 1))
            x2 = int(np.clip(x + length * np.cos(ang), 0, w - 1))
            rr, cc = line(y, x, y2, x2)
            ok = (dist[rr, cc] > max(1, margin // 2)) & (lab[rr, cc] != 1)
            if fov is not None:
                ok &= fov[rr, cc]
            lab[rr[ok], cc[ok]] = 2
            drawn |= bool(ok.any())
            y, x = y2, x2
        n_bg += int(drawn)
    return lab, n_fg, n_bg


def hide_half(mask: np.ndarray, rng: np.random.Generator, cfg: dict, thin: bool) -> np.ndarray:
    """The 'missing labels' variant: hide ~hidden_fraction of the foreground. Units are connected
    components when there are at least two and the structures are not thin, else skeleton segments
    (vessel segments; one big pet). Returns the visible mask (hidden = mask & ~visible)."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if thin or n < 2:
        lab = assign_mask_to_segments(mask, skeleton_segments(mask))
        n = int(lab.max())
    ids = np.arange(1, n + 1)
    rng.shuffle(ids)
    k = int(round(n * cfg["hidden_fraction"]))
    hidden_ids = ids[: max(1, k)] if n >= 2 else ids[:0]
    return mask & ~np.isin(lab, hidden_ids)


# ----------------------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, gt: np.ndarray, region: np.ndarray | None, extras: list[str]) -> dict:
    """IoU / Dice / recall / precision inside `region`, plus clDice when asked (thin structures)."""
    r = region if region is not None else np.ones(gt.shape, bool)
    p, g = pred & r, gt & r
    inter = float((p & g).sum())
    union = float((p | g).sum())
    out = {
        "iou": inter / union if union else 1.0,
        "dice": 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0,
        "recall": inter / g.sum() if g.sum() else 1.0,
        "precision": inter / p.sum() if p.sum() else 0.0,
    }
    if "cldice" in extras:
        from skimage.morphology import skeletonize

        sp, sg = skeletonize(p), skeletonize(g)
        tprec = (sp & g).sum() / sp.sum() if sp.sum() else 0.0
        tsens = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
        out["cldice"] = 2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) else 0.0
    return out


# ----------------------------------------------------------------------------- writing
def _png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255 if arr.dtype == bool else arr.astype(np.uint8)).save(path)


def _jpg(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=JPEG_QUALITY)


def write_dataset(samples: list[Sample], cfg: dict, out: Path, scale: float = 1.0) -> dict:
    """Write images, masks and the generated supervision; return the index that was written."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    thin = cfg["thin_structures"]
    items = []
    for s in samples:
        n = s.name
        _jpg(out / "images" / f"{n}.jpg", s.image)
        _png(out / "masks" / f"{n}.png", s.mask)
        gen = cfg["scribble_generator"]
        scrib, n_fg, n_bg = make_scribbles(s.mask, rng_for(n, "scribbles"), s.fov, gen, scale)
        _png(out / "scribbles" / f"{n}.png", scrib)
        visible = hide_half(s.mask, rng_for(n, "hide"), cfg["missing_labels"], thin)
        _png(out / "visible" / f"{n}.png", visible)
        item = {
            "name": n,
            "split": s.split,
            "size_hw": list(s.mask.shape),
            "image": f"images/{n}.jpg",
            "mask": f"masks/{n}.png",
            "scribbles": f"scribbles/{n}.png",
            "visible": f"visible/{n}.png",
            "fg_fraction": round(float(s.mask.mean()), 4),
            "n_fg_strokes": n_fg,
            "n_bg_strokes": n_bg,
            "scribbled_fraction": round(float((scrib > 0).mean()), 5),
            "hidden_fg_fraction": round(float(1 - visible.sum() / max(s.mask.sum(), 1)), 3),
        }
        if s.fov is not None:
            _png(out / "fov" / f"{n}.png", s.fov)
            item["fov"] = f"fov/{n}.png"
        if s.full:
            _jpg(out / "full" / f"{n}.jpg", s.full["image"])
            _png(out / "full_masks" / f"{n}.png", s.full["mask"])
            _png(out / "full_fov" / f"{n}.png", s.full["fov"])
            item["full"] = {
                "image": f"full/{n}.jpg",
                "mask": f"full_masks/{n}.png",
                "fov": f"full_fov/{n}.png",
                "size_hw": list(s.full["mask"].shape),
                "crop_xywh_native": s.full["crop_xywh"],
            }
        items.append(item)
    index = {
        "dataset": cfg["key"],
        "name": cfg["name"],
        "task": cfg["task"],
        "licence": cfg["licence"],
        "citation": cfg["citation"],
        "seed": SEED,
        "thin_structures": thin,
        "eval_region": cfg["eval_region"],
        "metric_extras": cfg["metric_extras"],
        "feature_long_side": cfg["feature_long_side"],
        "encoding": {
            "masks": "PNG, 255 = foreground. Ground truth: evaluation only (SPEC L-005).",
            "scribbles": "PNG, 0 = unlabelled, 1 = foreground stroke, 2 = background stroke",
            "visible": "PNG, 255 = labelled foreground; the rest of the true foreground is hidden "
            "(missing labels)",
            "fov": "PNG, 255 = inside the evaluation region (pets: 0 = the trimap's not-sure band)",
        },
        "n_fit": sum(s.split == "fit" for s in samples),
        "n_held": sum(s.split == "held" for s in samples),
        "items": items,
    }
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def shrink(s: Sample, long_side: int) -> Sample:
    """A smoke copy: image, mask and FOV at <= long_side (the supervision is regenerated at that size)."""
    full = {}
    if s.full:
        full = {k: (resize(v, long_side, k != "image") if k != "crop_xywh" else v) for k, v in s.full.items()}
    return Sample(
        s.name,
        resize(s.image, long_side, False),
        resize(s.mask, long_side, True),
        None if s.fov is None else resize(s.fov, long_side, True),
        s.split,
        full,
    )


def smoke_pick(samples: list[Sample]) -> list[Sample]:
    """2 fit + 2 held-out images, the first of each split, so a tiny head can train and evaluate."""
    fit = [s for s in samples if s.split == "fit"][: SMOKE_PER_DATASET // 2]
    held = [s for s in samples if s.split == "held"][: SMOKE_PER_DATASET - len(fit)]
    return fit + held


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def summarise(index: dict, out: Path) -> None:
    it = index["items"]

    def avg(key: str) -> float:
        return float(np.mean([i[key] for i in it]))

    print(
        f"  {index['name']}: {len(it)} images ({index['n_fit']} fit / {index['n_held']} held), "
        f"fg {100 * avg('fg_fraction'):.1f} %, scribbles cover {100 * avg('scribbled_fraction'):.2f} % "
        f"of pixels ({avg('n_fg_strokes'):.1f} fg + {avg('n_bg_strokes'):.1f} bg strokes), "
        f"{100 * avg('hidden_fg_fraction'):.0f} % of fg hidden -> "
        f"{out.relative_to(ROOT)} ({dir_bytes(out) / 1e6:.1f} MB)"
    )


# ----------------------------------------------------------------------------- main
def build_workshop(spec: dict, pets: list[Sample], special: list[Sample] | None) -> None:
    print("Workshop subsets")
    summarise(write_dataset(pets, PETS, WORKSHOP / "pets"), WORKSHOP / "pets")
    if special is not None:
        summarise(write_dataset(special, spec, WORKSHOP / "specialised"), WORKSHOP / "specialised")


def build_smoke(spec: dict, pets: list[Sample], special: list[Sample] | None) -> None:
    print(f"Smoke fixtures ({SMOKE_PER_DATASET} + {SMOKE_PER_DATASET} images at <= {SMOKE_LONG_SIDE} px)")
    todo = [(PETS, pets, "pets")] + ([(spec, special, "specialised")] if special is not None else [])
    for cfg, samples, sub in todo:
        small = [shrink(s, SMOKE_LONG_SIDE) for s in smoke_pick(samples)]
        scale = SMOKE_LONG_SIDE / max(max(s.mask.shape) for s in samples)
        summarise(write_dataset(small, cfg, SMOKE_DIR / sub, scale=scale), SMOKE_DIR / sub)
    total = dir_bytes(SMOKE_DIR)
    print(f"  data/smoke total: {total / 1e6:.2f} MB (budget {SMOKE_BUDGET_BYTES / 1e6:.0f} MB)")
    if total > SMOKE_BUDGET_BYTES:
        sys.exit("smoke fixtures exceed the 2 MB budget")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="only download/extract the raw data")
    ap.add_argument("--smoke", action="store_true", help="only rebuild data/smoke/")
    ap.add_argument("--only-pets", action="store_true", help="rebuild pets only; leave specialised untouched")
    args = ap.parse_args(argv)
    spec = load_specialised()
    fetch(spec)
    if args.download:
        return 0
    t0 = time.time()
    pets = load_pets()
    special = None if args.only_pets else load_specialised_samples(spec)
    if not args.smoke:
        build_workshop(spec, pets, special)
    build_smoke(spec, pets, special)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
