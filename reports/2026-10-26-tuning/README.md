# Small tuning runs on the retina task (2026-10-26)

Simone asked whether better results were possible with small, task-specific training runs. These runs
are outside the notebook: same data (HRF 1024 px crops), the same 5 training images the notebook uses
(`fit[:5]`), the same 25 held-out images, the same FOV-only scoring (mean over images). Mac M2 Pro, MPS.

`run.py <experiment> [steps]` runs one experiment. Every run is scored three ways: at the 0.5 cut, at the cut
that maximises IoU on the **training** images (legitimate), and at the best cut on the held-out images
(an oracle, reference only). `results.jsonl` has every run, including two early short test runs.

## The numbers that matter (cut chosen on the training images)

| what | labels | held-out IoU | Dice | train time |
|---|---|---|---|---|
| linear head on DINOv3 patches (= the notebook's section-2 head) | 5 complete masks | 0.332 (0.282 at cut 0.5) | 0.487 | 6 s |
| same, scribbles (= the notebook's scribble head) | 5 scribbled images | 0.183 (0.166 at 0.5) | 0.305 | 2 s |
| small decoder: DINOv3 + pixels, 200 steps | 5 complete masks | 0.463 | 0.628 | 14 s |
| same decoder, 1500 steps (overfits) | 5 complete masks | 0.201 | 0.325 | 97 s |
| same decoder, scribbles, 1500 steps | 5 scribbled images | 0.143 | 0.248 | 97 s |
| U-Net from scratch on the pixels, 3000 steps | 5 complete masks | **0.581** | **0.731** | 199 s |
| same U-Net, scribbles | 5 scribbled images | 0.108 | 0.193 | 204 s |
| same U-Net, 6000 steps | 20 complete masks | 0.622 | 0.765 | 412 s |

## What it says

- **The 16 px patch is the main limit.** A U-Net that works on the pixels doubles the full-mask head's IoU with
  the same five masks. With 20 masks it reaches Dice 0.77, near published supervised results (Dice 0.78–0.82,
  `../2026-10-26-hrf-state-of-the-art.md`), which validates the setup.
- **Cheap labels need the foundation model.** On scribbles the U-Net paints almost the whole fundus (IoU 0.11);
  the linear head on DINOv3 features does better (0.17). The scribbles have only 3–5 background strokes per
  image, far from the vessels, so a pixel model never learns where a vessel ends.
- **Choosing the cut on the training images** is a free +0.05 IoU for the notebook's heads.
- **A small decoder on DINOv3 + pixels helps early (0.46 after 14 s) but overfits** with longer training on five
  images; it would need early stopping on held-out training images before it could go into the notebook.

What went into the notebook: the state-of-the-art comparison in 10.3 (with the U-Net number as "our run"). Nothing
here changed a method on the learner's path; that is Simone's call.
