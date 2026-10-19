# %% [markdown]
# <table width="100%" bgcolor="#4A148C"><tr><td>
# <font color="white" size="5"><b>Setup check · When Foundation Models Fall Short</b></font><br>
# <font color="white">From Off-the-Shelf AI to Problem-Specific Adaptation · AMLD 2027 · SDS 2027</font>
# <br><br>
# <font color="white">Run this notebook <b>once, in the week before the workshop</b>. It takes about 5 minutes
# on a normal connection and leaves everything the workshop needs (models, images, precomputed features,
# checkpoints) in a folder of your Google Drive. On the day, the main notebook loads from that folder and
# nothing is downloaded or installed in the room.</font>
# </td></tr></table>
#
# **Objective:** fill `MyDrive/fm-workshop-cache/` with everything the workshop needs and see ✅ everywhere.
#
# What you need: a Google account, this notebook open in **Google Colab**, and ~400 MB of free space on Drive.
# No GPU is needed for this check. Run every cell from top to bottom (**Runtime → Run all**), then read the
# summary at the bottom. If a line shows ❌, run the notebook again: every download picks up where it stopped.
# If it still fails, send the summary to the organisers.

# %% [markdown]
# ## 1. Where things go
#
# This cell detects where you are running and decides the cache folder; nothing is downloaded yet.

# %%
import os
import sys
import time
from pathlib import Path

T0 = time.monotonic()
IN_COLAB = "google.colab" in sys.modules

# The release that hosts the data and precomputed files. Decided later: leave this placeholder until then.
BASE_URL = "https://REPLACE-ME.example.org/fm-workshop/v1"

# FM_SETUP_SOURCE overrides BASE_URL with another URL or a local folder (used by the automated test).
SOURCE = os.environ.get("FM_SETUP_SOURCE", BASE_URL)
LOCAL_MODE = not SOURCE.startswith(("http://", "https://"))

if LOCAL_MODE:
    CACHE_DIR = Path(os.environ.get("FM_SETUP_CACHE", "build/setup-check-cache")).resolve()
    HF_CACHE_DIR = None  # the machine's own Hugging Face cache
else:
    CACHE_DIR = Path("/content/drive/MyDrive/fm-workshop-cache")  # the main notebook reuses this path
    HF_CACHE_DIR = CACHE_DIR / "hf"  # model weights, in Hugging Face's cache layout
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)  # must be set before torch/timm/transformers are imported

TRY_SAM3 = os.environ.get("FM_TRY_SAM3", "1" if IN_COLAB else "0") == "1"  # optional, never required

CHECKS: dict[str, tuple[str, str]] = {}  # item -> (icon, message); the summary at the end prints it


def record(item: str, ok: bool | None, message: str, icon: str | None = None) -> None:
    """Store one line of the summary and print it now. ok=None means informational (⚠️, or ⏭️ if given)."""
    icon = icon or ("✅" if ok else ("❌" if ok is False else "⚠️"))
    CHECKS[item] = (icon, message)
    print(f"{icon} {item}: {message}")


print(f"Colab: {IN_COLAB} · source: {SOURCE}\ncache folder: {CACHE_DIR}")

# %% [markdown]
# ## 2. Google Drive
#
# This cell mounts your Google Drive so the cache survives between Colab sessions; outside Colab it uses a
# local folder instead.

# %%
if IN_COLAB and not LOCAL_MODE:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as err:  # noqa: BLE001 - any failure here gets the same plain message
        print(f"(mount error: {type(err).__name__}: {err})")
    if not Path("/content/drive/MyDrive").is_dir():
        raise SystemExit(
            "Google Drive is not mounted. Run this cell again and, when the pop-up appears, choose your "
            "Google account and click 'Allow' on every permission. Without Drive the cache would vanish "
            "when this Colab session ends."
        )
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / ".write-test").write_text("ok")
    (CACHE_DIR / ".write-test").unlink()
    record("Cache folder", True, f"{CACHE_DIR} is writable")
except OSError as err:
    record("Cache folder", False, f"cannot write to {CACHE_DIR}: {err}")

# %% [markdown]
# ## 3. GPU and packages
#
# This cell checks the session's GPU and the packages the workshop imports; it only reports, it never stops.

# %%
import torch

if torch.cuda.is_available():
    record("GPU", True, f"{torch.cuda.get_device_name(0)} (this is what the workshop expects)")
else:
    record(
        "GPU",
        None,
        "no GPU in this session. The downloads do not need one. On the workshop day, before running the main "
        "notebook, choose Runtime → Change runtime type → T4 GPU. Everything also works on CPU, just slower.",
    )

if IN_COLAB:
    # %pip install -q grad-cam
    pass  # Colab forgets pip installs between sessions, so the main notebook installs it again on the day.

VERSIONS = {"torch": torch.__version__}
try:
    import pytorch_grad_cam  # noqa: F401 - import is the check
    import timm
    import transformers

    VERSIONS |= {"timm": timm.__version__, "transformers": transformers.__version__}
    record("Packages", True, " · ".join(f"{k} {v}" for k, v in VERSIONS.items()) + " · grad-cam")
except ImportError as err:
    record("Packages", False, f"{err}. In Colab, run this cell again; otherwise `uv sync` in the repo.")

# %% [markdown]
# ## 4. Data, features and checkpoints
#
# This cell downloads every file the release manifest lists into the cache folder, verifies each one with its
# SHA-256 checksum, and skips what is already there; it is the slow step (a few minutes on Drive).

# %%
# @title Download helpers { display-mode: "form" }
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str, dest: Path) -> None:
    """Copy or download one file via a .part file, so a half-finished download never passes as complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if LOCAL_MODE:
        shutil.copyfile(Path(SOURCE) / rel, part)
    else:
        with urllib.request.urlopen(f"{SOURCE}/{rel}", timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)


def read_manifest() -> list[dict]:
    """The release manifest: path, bytes, sha256 per file (same shape as precomputed/manifest.json)."""
    fetch("manifest.json", CACHE_DIR / "manifest.json")
    return json.loads((CACHE_DIR / "manifest.json").read_text())["files"]


def sync_files(files: list[dict]) -> dict[str, dict]:
    """Make CACHE_DIR hold every manifest file with the right checksum. Returns per-group stats."""
    groups: dict[str, dict] = defaultdict(lambda: {"n": 0, "bytes": 0, "cached": 0, "failed": [], "s": 0.0})
    for entry in files:
        rel, dest = entry["path"], CACHE_DIR / entry["path"]
        group = "/".join(rel.split("/")[:2]) if "/" in rel else rel
        g = groups[group]
        t = time.monotonic()
        g["n"] += 1
        g["bytes"] += entry["bytes"]
        if dest.is_file() and dest.stat().st_size == entry["bytes"] and sha256_of(dest) == entry["sha256"]:
            g["cached"] += 1
        else:
            for attempt in (1, 2):
                try:
                    fetch(rel, dest)
                    if sha256_of(dest) == entry["sha256"]:
                        break
                    dest.unlink()
                    print(f"  checksum mismatch for {rel} (attempt {attempt})")
                except Exception as err:  # noqa: BLE001 - reported, the loop continues with the next file
                    print(f"  {rel}: {type(err).__name__}: {err} (attempt {attempt})")
            else:
                g["failed"].append(rel)
        g["s"] += time.monotonic() - t
    return groups


# %%
if "REPLACE-ME" in SOURCE:
    record(
        "Data and precomputed files", False, "the release is not published yet (BASE_URL is a placeholder)"
    )
else:
    try:
        manifest = read_manifest()
    except Exception as err:  # noqa: BLE001 - one plain line instead of a traceback
        manifest = []
        record("Data and precomputed files", False, f"cannot read {SOURCE}/manifest.json: {err}")
    if manifest:
        stats = sync_files(manifest)
        for group, g in stats.items():
            ok = not g["failed"]
            note = f"{g['n']} files · {g['bytes'] / 2**20:.1f} MB · {g['s']:.1f} s"
            if g["cached"]:
                note += f" · {g['cached']} already cached"
            if not ok:
                note += f" · FAILED: {', '.join(g['failed'][:3])}{' …' if len(g['failed']) > 3 else ''}"
            record(group, ok, note)

# %% [markdown]
# ## 5. Models
#
# This cell downloads the two foundation models from Hugging Face into the cache (no account needed), verifies
# the weight files against the checksums Hugging Face publishes, and loads each one once on the CPU.

# %%
# @title Model helpers { display-mode: "form" }
from huggingface_hub import HfApi, snapshot_download

DINO_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_LICENCE = "DINOv3 License (Meta)"
SAM_ID = "facebook/sam2.1-hiera-small"
SAM_LICENCE = "Apache-2.0"
SAM3_ID = "facebook/sam3"


def verify_snapshot(repo_id: str, folder: Path, token: str | None) -> tuple[int, int]:
    """Re-hash the weight files of a downloaded snapshot against the sha256 the Hub publishes for them."""
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
        expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    except Exception:  # noqa: BLE001 - offline: the cache's blob names are those same sha256 values
        expected = {}
        for p in folder.rglob("*"):
            blob = Path(os.path.realpath(p)).name
            if p.is_file() and len(blob) == 64:
                expected[p.relative_to(folder).as_posix()] = blob
    checked, bad = 0, 0
    for name, sha in expected.items():
        if (folder / name).is_file():
            checked += 1
            bad += sha256_of(folder / name) != sha
    return checked, bad


def cache_model(item: str, repo_id: str, licence: str, load) -> None:
    t = time.monotonic()
    try:
        folder = Path(snapshot_download(repo_id, token=HF_TOKEN))
        checked, bad = verify_snapshot(repo_id, folder, HF_TOKEN)
        if bad:
            record(item, False, f"{bad} of {checked} weight files failed their checksum: run this cell again")
            return
        load()
        mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 2**20
        record(
            item, True, f"{repo_id} · {licence} · {mb:.0f} MB · loads on CPU · {time.monotonic() - t:.0f} s"
        )
    except Exception as err:  # noqa: BLE001 - one plain line instead of a traceback
        record(item, False, f"{repo_id}: {type(err).__name__}: {str(err).splitlines()[0][:160]}")


def load_dino() -> None:
    timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).eval()


def load_sam() -> None:
    from transformers import Sam2Model, Sam2Processor

    Sam2Processor.from_pretrained(SAM_ID)
    Sam2Model.from_pretrained(SAM_ID).eval()


# %%
HF_TOKEN = os.environ.get("HF_TOKEN")  # only matters for SAM 3; DINOv3 and SAM 2.1 need no account
if IN_COLAB and HF_TOKEN is None:
    try:
        from google.colab import userdata

        HF_TOKEN = userdata.get("HF_TOKEN")  # Colab's 🔑 Secrets panel, if you added one
    except Exception:  # noqa: BLE001 - no secret, no token: fine
        HF_TOKEN = None

cache_model("DINOv3 ViT-S/16", DINO_ID, DINO_LICENCE, load_dino)
cache_model("SAM 2.1 hiera-small", SAM_ID, SAM_LICENCE, load_sam)

# %% [markdown]
# ## 6. Optional: SAM 3
#
# SAM 3 is gated on Hugging Face: Meta approves access per account. The workshop never requires it. This
# cell caches it only if you already have access and a token in Colab's Secrets panel under `HF_TOKEN`;
# otherwise it says so and moves on.

# %%
if not TRY_SAM3:
    record("SAM 3 (optional)", None, "skipped", icon="⏭️")
elif HF_TOKEN is None:
    record(
        "SAM 3 (optional)",
        None,
        "no HF_TOKEN in Colab's Secrets panel, so not tried. SAM 2.1 is enough.",
        "⏭️",
    )
else:
    try:
        folder = Path(snapshot_download(SAM3_ID, token=HF_TOKEN))
        checked, bad = verify_snapshot(SAM3_ID, folder, HF_TOKEN)
        record("SAM 3 (optional)", not bad, f"{SAM3_ID} cached ({checked} weight files checked)")
    except Exception as err:  # noqa: BLE001 - optional: never a ❌
        msg = f"not available for this account ({type(err).__name__}). SAM 2.1 is enough."
        record("SAM 3 (optional)", None, msg, icon="⏭️")

# %% [markdown]
# ## 7. Summary
#
# This cell prints one line per item and writes the result into the cache, where the main notebook reads it.

# %%
TOTAL_S = time.monotonic() - T0
failed = [k for k, (icon, _) in CHECKS.items() if icon == "❌"]
print("\n".join(f"{icon} {item}: {msg}" for item, (icon, msg) in CHECKS.items()))
print(f"\nTotal time: {TOTAL_S:.0f} s · cache folder: {CACHE_DIR}")
if failed:
    print(
        f"❌ {len(failed)} item(s) failed: {', '.join(failed)}. Run the notebook again (Runtime → Run all)."
    )
else:
    print("✅ All set. You can close this notebook; on the workshop day open the main one.")
SETUP_RESULT = {
    "ok": not failed,
    "checks": {k: {"icon": i, "message": m} for k, (i, m) in CHECKS.items()},
    "total_s": round(TOTAL_S, 1),
    "hf_cache_dir": None if HF_CACHE_DIR is None else str(HF_CACHE_DIR),
    "versions": VERSIONS,
}
try:
    (CACHE_DIR / "setup-check.json").write_text(json.dumps(SETUP_RESULT, indent=1))
except OSError as err:
    print(f"(could not write setup-check.json: {err})")

# %% [markdown]
# <table width="100%" bgcolor="#FFF8E1"><tr><td><font color="#5D4037">🧾 <b>What the main notebook reuses.</b>
# <code>CACHE_DIR</code> = <code>MyDrive/fm-workshop-cache/</code>, holding <code>data/</code> (images, masks,
# scribbles), <code>precomputed/</code> (DINOv3 features, SAM pseudo-labels, head checkpoints),
# <code>hf/</code> (model weights, loaded with <code>HF_HUB_CACHE</code> pointing there),
# <code>specialised.json</code>, <code>manifest.json</code> and <code>setup-check.json</code>.
# DINOv3 weights come with Meta's DINOv3 License, SAM 2.1 with Apache-2.0.
# </font></td></tr></table>
