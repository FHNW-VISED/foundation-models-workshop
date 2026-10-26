"""Shared loaders and metrics for the HRF tuning experiments (outside the notebook)."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

REPO = Path.home() / "Work/notebooks/foundation-models-workshop"
ROOT = REPO / "data/cache/workshop/specialised"
FEATS = REPO / "data/cache/precomputed/features/specialised_1024.safetensors"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path(__file__).parent  # results.jsonl is written next to this file



def gray(p):
    return np.array(Image.open(p).convert("L"))


def load(item):
    return {
        "name": item["name"],
        "split": item["split"],
        "image": np.array(Image.open(ROOT / item["image"]).convert("RGB")),
        "mask": gray(ROOT / item["mask"]) > 127,
        "scribbles": gray(ROOT / item["scribbles"]),
        "visible": gray(ROOT / item["visible"]) > 127,
        "fov": gray(ROOT / item["fov"]) > 127,
    }


def dataset():
    items = json.loads((ROOT / "index.json").read_text())["items"]
    fit = [load(i) for i in items if i["split"] == "fit"]
    held = [load(i) for i in items if i["split"] == "held"]
    return fit, held


_store = None


def dino_feat(item):
    global _store
    if _store is None:
        _store = safe_open(FEATS, framework="pt")
    return _store.get_tensor(item["name"]).float()


def metrics(preds, items):
    rows = []
    for p, it in zip(preds, items, strict=True):
        r = it["fov"]
        p, t = p & r, it["mask"] & r
        inter, union = (p & t).sum(), (p | t).sum()
        rows.append(
            {
                "iou": inter / max(union, 1),
                "dice": 2 * inter / max(p.sum() + t.sum(), 1),
                "recall": inter / max(t.sum(), 1),
                "precision": inter / max(p.sum(), 1),
                "marked": p[r].mean(),
            }
        )
    return {k: round(float(np.mean([x[k] for x in rows])), 3) for k in rows[0]}


def best_threshold(probs, items, grid=np.linspace(0.05, 0.95, 37)):
    """The probability cut that maximises mean IoU on these (training) images."""
    scores = [metrics([p >= g for p in probs], items)["iou"] for g in grid]
    return float(grid[int(np.argmax(scores))])


def log(name, res, extra=None):
    line = {"name": name, **res, **(extra or {})}
    print(json.dumps(line), flush=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


class Timer:
    def __enter__(self):
        self.t = time.monotonic()
        return self

    def __exit__(self, *a):
        self.s = round(time.monotonic() - self.t, 1)
