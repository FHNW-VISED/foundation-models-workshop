"""Precompute (issue #2): what the CPU fallback path loads instead of computing (SPEC §5, W-7).

For both workshop datasets written by scripts/prepare_data.py:
  1. DINOv3 ViT-S/16 patch features for every image, at each resolution the dataset config lists
  2. SAM 2.1 pseudo-labels prompted with points sampled from the auto-scribbles (fit images)
  3. a linear-head checkpoint for every training step in SPEC §6: full masks, scribbles,
     missing-as-background (the naive baseline), masked loss, SAM pseudo-labels

Usage:
    uv run python scripts/precompute.py           # -> data/cache/precomputed/  (hosted later, not in git)
    uv run python scripts/precompute.py --smoke   # -> data/smoke/precomputed/  (tiny heads, committed)
    uv run python scripts/precompute.py --steps 300 --lr 0.05 --device cpu

Outputs (sizes are printed at the end, and listed with sha256 in manifest.json):
    features/<dataset>_<long_side>.safetensors   key = image name, float16 [384, h, w] patch grid
    pseudo_labels/<dataset>/NAME.png             SAM probability x 255 (threshold it yourself); prompts.json
    heads/<dataset>_<regime>_<long_side>.safetensors   weight [1,384,1,1], bias [1], feat_mean/feat_std [384]
    heads/index.json                             every head with its config and held-out metrics
In smoke mode features are not stored: 8 images at 256 px take a second on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (  # noqa: E402
    PETS,
    ROOT,
    SEED,
    SMOKE_DIR,
    SMOKE_LONG_SIDE,
    WORKSHOP,
    load_specialised,
    metrics,
    rng_for,
)

PRECOMPUTED = ROOT / "data" / "cache" / "precomputed"
SMOKE_PRECOMPUTED = SMOKE_DIR / "precomputed"
DINO_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_LICENCE = "DINOv3 License (Meta), https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m"
SAM_ID = "facebook/sam2.1-hiera-small"
SAM_LICENCE = "Apache-2.0"
N_PROMPT_POINTS = 20  # positives and negatives sampled from the scribbles for the pseudo-labels
PSEUDO_THRESHOLD = 0.5
IGNORE_DILATION_PX = 3

# The training steps of SPEC §6. loss = "balanced" (classes reweighted to 0.5/0.5) or "plain" BCE.
# labels: 1 = foreground, 0 = background, -1 = ignored by the loss.
REGIMES = {
    "full": {"section": "01/08", "loss": "balanced", "supervision": "full masks (upper reference)"},
    "scribbles": {
        "section": "04",
        "loss": "balanced",
        "supervision": "auto-scribbles; unlabelled pixels ignored",
    },
    "hidden_bg": {
        "section": "03",
        "loss": "plain",
        "supervision": "missing labels, missing = background, plain BCE (the naive baseline)",
    },
    "hidden_bg_balanced": {
        "section": "03",
        "loss": "balanced",
        "supervision": "missing labels, missing = background, balanced BCE",
    },
    "masked": {
        "section": "05",
        "loss": "balanced",
        "supervision": f"missing labels, hidden pixels (dilated {IGNORE_DILATION_PX} px) ignored by the loss",
    },
    "pseudo": {
        "section": "06",
        "loss": "balanced",
        "supervision": f"SAM 2.1 pseudo-labels from scribble points, probability >= {PSEUDO_THRESHOLD}",
    },
}
PETS_EXTRA = {
    "full5": {"section": "01", "loss": "balanced", "supervision": "full masks of 5 pets", "n_train": 5}
}


# ----------------------------------------------------------------------------- data
def read_index(root: Path) -> dict:
    return json.loads((root / "index.json").read_text())


def read_item(root: Path, item: dict) -> dict:
    from PIL import Image

    def png(key):
        return np.array(Image.open(root / item[key]).convert("L"))

    out = {
        "name": item["name"],
        "split": item["split"],
        "image": np.array(Image.open(root / item["image"]).convert("RGB")),
        "mask": png("mask") > 127,
        "scribbles": png("scribbles"),
        "visible": png("visible") > 127,
        "fov": png("fov") > 127 if "fov" in item else None,
    }
    out["hidden"] = out["mask"] & ~out["visible"]
    return out


# ----------------------------------------------------------------------------- models
def pick_device(want: str | None) -> str:
    import torch

    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Dino:
    def __init__(self, device: str):
        import timm
        import torch

        self.torch, self.device = torch, device
        self.model = (
            timm.create_model(DINO_ID, pretrained=True, num_classes=0, dynamic_img_size=True)
            .eval()
            .to(device)
        )
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
        self.n_prefix = self.model.num_prefix_tokens

    def features(self, image: np.ndarray, long_side: int):
        """Patch features [D, h, w] for the image resized to long_side (rounded to multiples of 16)."""
        import torch.nn.functional as F

        x = self.torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255
        h, w = image.shape[:2]
        s = long_side / max(h, w)
        H, W = max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with self.torch.no_grad():
            f = self.model.forward_features(x)[0, self.n_prefix :]
        return f.reshape(H // 16, W // 16, -1).permute(2, 0, 1).contiguous().float()


class Sam:
    def __init__(self, device: str):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch, self.device = torch, device
        self.proc = Sam2Processor.from_pretrained(SAM_ID)
        self.model = Sam2Model.from_pretrained(SAM_ID).eval().to(device)

    def probability(self, image: np.ndarray, points: list[list[float]], labels: list[int]) -> np.ndarray:
        """Foreground probability map (H x W float) for point prompts; the single-mask output."""
        inputs = self.proc(
            images=image, input_points=[[points]], input_labels=[[labels]], return_tensors="pt"
        )
        with self.torch.no_grad():
            emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
            out = self.model(
                image_embeddings=emb,
                input_points=inputs["input_points"].to(self.device),
                input_labels=inputs["input_labels"].to(self.device),
                multimask_output=False,
            )
        logits = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"], binarize=False)[
            0
        ]
        return self.torch.sigmoid(logits[0, 0]).numpy()


# ----------------------------------------------------------------------------- steps
def extract_features(dino: Dino, items: list[dict], long_side: int, path: Path | None) -> dict:
    from safetensors.torch import save_file

    t0 = time.time()
    feats = {it["name"]: dino.features(it["image"], long_side) for it in items}
    sec = (time.time() - t0) / len(items)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": DINO_ID,
            "long_side": str(long_side),
            "patch": "16",
            "layout": "D,h,w",
            "dtype": "float16",
        }
        save_file({k: v.cpu().half().contiguous() for k, v in feats.items()}, path, metadata=meta)
    print(
        f"  features @{long_side}: {len(items)} images, {sec:.2f} s/image"
        + (f" -> {path.name}" if path else "")
    )
    return feats


def scribble_prompts(it: dict, n: int) -> tuple[list[list[float]], list[int]]:
    """n foreground + n background points sampled from the scribble strokes (no ground truth)."""
    rng = rng_for(it["name"], "prompts")
    pts, labs = [], []
    for value, label in ((1, 1), (2, 0)):
        ys, xs = np.nonzero(it["scribbles"] == value)
        for k in rng.choice(len(ys), size=min(n, len(ys)), replace=False):
            pts.append([float(xs[k]), float(ys[k])])
            labs.append(label)
    return pts, labs


def pseudo_labels(sam: Sam, items: list[dict], out: Path, extras: list[str]) -> dict[str, np.ndarray]:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    probs, prompts, ms = {}, {}, []
    for it in items:
        pts, labs = scribble_prompts(it, N_PROMPT_POINTS)
        p = sam.probability(it["image"], pts, labs)
        probs[it["name"]] = p
        prompts[it["name"]] = {"points_xy": pts, "labels": labs}
        Image.fromarray(np.round(p * 255).astype(np.uint8)).save(out / f"{it['name']}.png")
        ms.append(metrics(p >= PSEUDO_THRESHOLD, it["mask"], it["fov"], extras))
    (out / "prompts.json").write_text(
        json.dumps({"model": SAM_ID, "points_per_class": N_PROMPT_POINTS, "prompts": prompts}, indent=1)
    )
    m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    print(
        f"  SAM pseudo-labels: {len(items)} fit images in {time.time() - t0:.0f} s, "
        f"IoU {m['iou']:.3f} vs the hidden GT (recall {m['recall']:.3f}) -> {out.relative_to(ROOT)}"
    )
    return probs


def labels_for(regime: str, it: dict, prob: np.ndarray | None) -> np.ndarray:
    from scipy import ndimage as ndi

    if regime in ("full", "full5"):
        return it["mask"].astype(np.int8)
    if regime == "scribbles":
        sc = it["scribbles"]
        return np.where(sc == 1, 1, np.where(sc == 2, 0, -1)).astype(np.int8)
    if regime in ("hidden_bg", "hidden_bg_balanced"):
        return it["visible"].astype(np.int8)
    if regime == "masked":
        ignore = ndi.binary_dilation(it["hidden"], iterations=IGNORE_DILATION_PX) & ~it["visible"]
        return np.where(it["visible"], 1, np.where(ignore, -1, 0)).astype(np.int8)
    if regime == "pseudo":
        return (prob >= PSEUDO_THRESHOLD).astype(np.int8)
    raise KeyError(regime)


def train_head(feats: list, labels: list[np.ndarray], steps: int, lr: float, balanced: bool, device: str):
    """Linear head (384 -> 1) on standardised patch features; logits upsampled bilinearly to the
    label size; balanced or plain BCE over the labelled pixels (label -1 = ignore)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    allf = torch.cat([f.flatten(1) for f in feats], 1)
    mu, sd = allf.mean(1), allf.std(1) + 1e-6
    feats = [(f - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1) for f in feats]
    labs = [torch.from_numpy(lab).to(device) for lab in labels]
    n_pos = sum(int((lab == 1).sum()) for lab in labs)
    n_neg = sum(int((lab == 0).sum()) for lab in labs)
    if balanced:
        w_pos, w_neg = 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)
    else:
        w_pos = w_neg = 1.0 / max(n_pos + n_neg, 1)
    head = torch.nn.Conv2d(feats[0].shape[0], 1, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for f, lab in zip(feats, labs, strict=True):
            logit = F.interpolate(head(f[None]), size=lab.shape, mode="bilinear", align_corners=False)[0, 0]
            w = torch.where(lab == 1, w_pos, torch.where(lab == 0, w_neg, 0.0))
            loss = (
                loss
                + (F.binary_cross_entropy_with_logits(logit, (lab == 1).float(), reduction="none") * w).sum()
            )
        loss.backward()
        opt.step()
    tensors = {
        "weight": head.weight.detach().cpu().contiguous(),
        "bias": head.bias.detach().cpu().contiguous(),
        "feat_mean": mu.cpu().contiguous(),
        "feat_std": sd.cpu().contiguous(),
    }
    return tensors, time.time() - t0


def predict(tensors: dict, feat, shape: tuple[int, int]) -> np.ndarray:
    import torch.nn.functional as F

    dev = feat.device
    f = (feat - tensors["feat_mean"].to(dev).view(-1, 1, 1)) / tensors["feat_std"].to(dev).view(-1, 1, 1)
    logit = F.conv2d(f[None], tensors["weight"].to(dev), tensors["bias"].to(dev))
    logit = F.interpolate(logit, size=shape, mode="bilinear", align_corners=False)[0, 0]
    return (logit > 0).cpu().numpy()


def train_all_heads(
    cfg: dict,
    items: list[dict],
    feats_by_res: dict[int, dict],
    probs: dict,
    out: Path,
    steps: int,
    lr: float,
    device,
) -> list[dict]:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    fit = [it for it in items if it["split"] == "fit"]
    held = [it for it in items if it["split"] == "held"]
    regimes = dict(REGIMES, **(PETS_EXTRA if cfg["key"] == "pets" else {}))
    extras = cfg["metric_extras"]
    rows = []
    for res, feats in feats_by_res.items():
        for name, r in regimes.items():
            train = fit[: r["n_train"]] if "n_train" in r else fit
            labels = [labels_for(name, it, probs.get(it["name"])) for it in train]
            tensors, sec = train_head(
                [feats[it["name"]] for it in train], labels, steps, lr, r["loss"] == "balanced", device
            )
            ms = [
                metrics(predict(tensors, feats[it["name"]], it["mask"].shape), it["mask"], it["fov"], extras)
                for it in held
            ]
            m = {k: round(float(np.mean([x[k] for x in ms])), 4) for k in ms[0]}
            hid = [it for it in train if it["hidden"].any()]
            if name in ("hidden_bg", "hidden_bg_balanced", "masked") and hid:
                m["recall_on_hidden_fg_train"] = round(
                    float(
                        np.mean(
                            [
                                predict(tensors, feats[it["name"]], it["mask"].shape)[it["hidden"]].mean()
                                for it in hid
                            ]
                        )
                    ),
                    4,
                )
            row = {
                "file": f"heads/{cfg['key']}_{name}_{res}.safetensors",
                "dataset": cfg["key"],
                "regime": name,
                "section": r["section"],
                "supervision": r["supervision"],
                "loss": r["loss"],
                "long_side": res,
                "steps": steps,
                "lr": lr,
                "model": DINO_ID,
                "train_names": [it["name"] for it in train],
                "labelled_fraction": round(float(np.mean([(lab >= 0).mean() for lab in labels])), 5),
                "train_sec": round(sec, 2),
                "held_metrics": m,
            }
            meta = {k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in row.items()}
            save_file(tensors, out / f"{cfg['key']}_{name}_{res}.safetensors", metadata=meta)
            rows.append(row)
            print(
                f"  head {cfg['key']:11s} {name:18s} @{res:4d}: IoU {m['iou']:.3f} dice {m['dice']:.3f} "
                f"recall {m['recall']:.3f}"
                + (f" clDice {m['cldice']:.3f}" if "cldice" in m else "")
                + f" ({sec:.0f} s)"
            )
    return rows


# ----------------------------------------------------------------------------- bookkeeping
def manifest(out: Path) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    entries, total = [], 0
    for p in files:
        b = p.stat().st_size
        total += b
        entries.append(
            {
                "path": str(p.relative_to(out)),
                "bytes": b,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    (out / "manifest.json").write_text(
        json.dumps({"root": str(out.relative_to(ROOT)), "files": entries}, indent=1)
    )
    print(f"\nFiles in {out.relative_to(ROOT)}:")
    by_dir: dict[str, tuple[int, int]] = {}
    for e in entries:
        d = e["path"].split("/")[0]
        n, b = by_dir.get(d, (0, 0))
        by_dir[d] = (n + 1, b + e["bytes"])
    for e in entries:
        if e["bytes"] > 1_000_000:
            print(f"  {e['bytes'] / 1e6:8.1f} MB  {e['path']}")
    for d, (n, b) in by_dir.items():
        print(f"  {b / 1e6:8.1f} MB  {d}/ ({n} files)")
    print(f"  {total / 1e6:8.1f} MB  total ({len(entries)} files, sha256 in manifest.json)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny run on data/smoke -> data/smoke/precomputed")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    spec = load_specialised()
    data_root, out = (SMOKE_DIR, SMOKE_PRECOMPUTED) if args.smoke else (WORKSHOP, PRECOMPUTED)
    steps = min(args.steps, 30) if args.smoke else args.steps
    device = pick_device(args.device)
    t_all = time.time()
    print(f"device {device} · DINOv3 {DINO_ID} ({DINO_LICENCE}) · SAM {SAM_ID} ({SAM_LICENCE})")
    dino, sam = Dino(device), Sam(device)
    heads_index = []
    for cfg in (PETS, spec):
        sub = "pets" if cfg["key"] == "pets" else "specialised"
        root = data_root / sub
        if not root.exists():
            sys.exit(f"{root} missing: run scripts/prepare_data.py first")
        index = read_index(root)
        items = [read_item(root, it) for it in index["items"]]
        print(f"\n== {cfg['name']}: {len(items)} images from {root.relative_to(ROOT)}")
        resolutions = [SMOKE_LONG_SIDE] if args.smoke else cfg["feature_long_sides"]
        feats_by_res = {
            res: extract_features(
                dino, items, res, None if args.smoke else out / "features" / f"{sub}_{res}.safetensors"
            )
            for res in resolutions
        }
        fit = [it for it in items if it["split"] == "fit"]
        probs = pseudo_labels(sam, fit, out / "pseudo_labels" / sub, cfg["metric_extras"])
        heads_index += train_all_heads(cfg, items, feats_by_res, probs, out / "heads", steps, args.lr, device)
    (out / "heads" / "index.json").write_text(
        json.dumps({"seed": SEED, "model": DINO_ID, "sam": SAM_ID, "heads": heads_index}, indent=1)
    )
    manifest(out)
    print(f"done in {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
