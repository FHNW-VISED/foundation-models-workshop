"""Does the SAM 3 workshop copy work for a participant? Downloads it the way the notebook does, loads it and
segments one pet image. Downloads ~3.4 GB into a temporary folder and deletes it afterwards.

    uv run python scripts/check_sam3_copy.py              # with your Hugging Face login (works while private)
    uv run python scripts/check_sam3_copy.py --anonymous  # as a participant (repo must be public)

Exit code 0 = downloaded, loaded and found at least one mask; 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Sam3Model, Sam3Processor

REPO = "simopippa/sam3-amld2027-workshop"  # must match SAM3_ID for "workshop" in sections/00_setup.py
FILES = ["*.json", "*.txt", "model.safetensors", "*.model"]  # the notebook's allow_patterns
IMAGE = Path(__file__).resolve().parent.parent / "data/smoke/pets/fov/boxer_173.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymous", action="store_true", help="no token, as a participant without an account")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    t0 = time.monotonic()
    try:
        folder = tmp / REPO.replace("/", "--")
        token = False if args.anonymous else None  # False = never send a token
        snapshot_download(REPO, local_dir=folder, allow_patterns=FILES, token=token)
        who = "anonymous" if args.anonymous else "your login"
        print(f"downloaded {REPO} ({who}) in {time.monotonic() - t0:.0f} s")
        proc = Sam3Processor.from_pretrained(str(folder), local_files_only=True)
        model = Sam3Model.from_pretrained(str(folder), local_files_only=True, dtype=torch.float32).eval()
        image = np.array(Image.open(IMAGE).convert("RGB"))
        inputs = proc(images=image, text="animal", return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        found = proc.post_process_instance_segmentation(
            out, threshold=0.3, mask_threshold=0.5, target_sizes=[image.shape[:2]]
        )[0]
        n = len(found["masks"])
        print(f'"animal" on {IMAGE.name}: {n} mask(s); total {time.monotonic() - t0:.0f} s')
        return 0 if n > 0 else 1
    except Exception as err:  # noqa: BLE001 - report any failure as a failed check
        print(f"FAILED: {type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
