# data/

What is here, what is generated, and what is committed. Sources and licences: `ATTRIBUTION.md`.

| Path | In git | Made by | What |
|---|---|---|---|
| `specialised.json` | yes | hand | the specialised task's config (name, source, licence, preprocessing, scribble generator, metric extras). Everything else reads it. |
| `smoke/` | yes (≤ 2 MB) | `scripts/prepare_data.py`, `scripts/precompute.py --smoke` | 4 pets + 4 specialised images at ≤ 256 px with masks, scribbles, hidden-label variants, SAM pseudo-labels and tiny head checkpoints. The headless smoke run uses only these. |
| `cache/` | no | `scripts/prepare_data.py` | raw downloads, extracted archives, and `workshop/` (the subsets participants get) |
| `cache/precomputed/` | no | `scripts/precompute.py` | DINOv3 features, SAM pseudo-labels and a head checkpoint per training step; `manifest.json` has sizes and sha256 for the setup check |

## Dataset folder layout (same for `cache/workshop/<x>/` and `smoke/<x>/`, `<x>` = `pets` or `specialised`)

```
index.json          dataset facts + one entry per image: name, split (fit | held), paths, fg_fraction,
                    n_fg_strokes, n_bg_strokes, scribbled_fraction, hidden_fg_fraction, size_hw
images/NAME.jpg     RGB image
masks/NAME.png      full ground truth, 255 = foreground      -> evaluation only (SPEC L-005)
fov/NAME.png        evaluation region, 255 = inside           (retina only)
scribbles/NAME.png  0 = unlabelled, 1 = foreground stroke, 2 = background stroke
visible/NAME.png    "missing labels": 255 = labelled foreground; the hidden half of the true
                    foreground is 0 here (hidden = mask & ~visible)
full/, full_masks/, full_fov/   (specialised, crop mode) the whole image downsampled; index entry
                    "full" has its paths and the crop's native position (crop_xywh_native)
```

## Precomputed layout (`cache/precomputed/` and, without features, `smoke/precomputed/`)

```
features/<x>_<long_side>.safetensors   key = image name -> float16 [384, h, w] DINOv3 patch grid
pseudo_labels/<x>/NAME.png             SAM 2.1 foreground probability x 255 for the fit images,
                                       prompted with 20 + 20 points from the scribbles (prompts.json)
heads/<x>_<regime>_<long_side>.safetensors   weight [1,384,1,1], bias [1], feat_mean [384],
                                       feat_std [384]; metadata = config + held-out metrics
heads/index.json                       the same, as one list
manifest.json                          path, bytes, sha256 of every file
```

A head is applied as `conv1x1((feat - feat_mean) / feat_std)`, logits upsampled bilinearly to the
image, foreground = logit > 0. Regimes: `full`, `full5` (pets: 5 images), `scribbles`, `hidden_bg`
(naive baseline, plain BCE), `hidden_bg_balanced`, `masked` (hidden pixels ignored), `pseudo`.
