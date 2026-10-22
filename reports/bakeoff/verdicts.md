All numbers quoted here are copied from the tables above (held-out means, 10 images). Three questions per
dataset: (1) does the foundation model fall short? (2) do the weak-supervision methods give visibly
different predictions? (3) does it fit a T4 / CPU in workshop time? Plus: is Grad-CAM meaningful?

Hardware caveat: the run was on an Apple M-series GPU (`mps`), not a T4. The per-image CPU timings in each
table are measured, and a T4 is faster than this laptop's GPU for these models, so "fits on mps" implies
"fits on a T4". Only the CPU numbers are hardware-independent enough to plan with.

### Oxford-IIIT Pet, the simple task (reference row)

SAM 2.1 does what the story needs: 0.770 IoU with one positive and one negative point, 0.792 with the GT
box. The DINOv3 head reaches 0.789 from auto-scribbles covering 0.43 % of the pixels. Everything runs in
47 s in total. One caveat for section 01: with 300 steps and the balanced loss the full-mask head at 512
over-predicts a halo around the pet (0.619 IoU, recall 0.992); at 1024 it is 0.746, and a 5-step trial run
during development gave 0.836. The notebook's head needs a quick learning-rate / step sweep, not a
different design.

### HRF retina, whole fundus downsampled to 1024 px long side

1. **Falls short: yes, spectacularly.** Every SAM prompt returns the whole fundus disc: IoU 0.071 to
   0.093, recall 0.998 with the box, and 0.093 is exactly the vessel fraction (9.0 %). More points do not
   help (1 pt 0.071, 20 pt 0.091). The frozen DINOv3 head with *full* masks reaches only 0.229 IoU
   (clDice 0.483) at 1024 and 0.147 at 512, because vessels are 1 to 3 px wide against 16 px patches.
   The 1998 Frangi filter with no labels beats the full-mask head on IoU (0.277 vs 0.229), while the head
   wins on clDice (0.483 vs 0.388). Resolution matters visibly: 512 → 1024 lifts IoU from 0.147 to 0.229.
2. **Visibly different: yes.** Mean pairwise IoU between the 1024 heads is 0.255 (min 0.001). The naive
   baseline (missing = background, plain BCE) collapses to recall 0.008 and IoU 0.008, while missing =
   ignore keeps recall 0.789 and IoU 0.217: the "it learns that vessels are background" lesson is as
   extreme as it gets. Scribbles give 0.115, SAM pseudo-labels give 0.091 (they are the disc again).
   The weakness: the whole ladder lives between 0.09 and 0.23 IoU, so "predictions differ" is true but
   "any rung looks good" is not.
3. **Fits: yes.** 126 s for everything on mps; features 0.41 s per image on CPU at 1024; head training
   22 s on CPU for 10 images; SAM embedding 0.22 s per image.

### HRF retina, native-resolution 1024 × 1024 crops (added during the bake-off)

Same 20 images, one random crop each inside the FOV with ≥ 5 % vessels, vessels now 4 to 20 px wide.

1. **Falls short: yes.** SAM stays at 0.094 to 0.115 IoU (disc again). The full-mask head improves to
   0.335 IoU / clDice 0.563 at 1024 (0.274 at 512), and now beats Frangi (0.162 / 0.211), so the
   "2025 foundation model vs 1998 filter" comparison flips between the two views, which is itself a good
   discussion point.
2. **Visibly different: yes.** Mean pairwise IoU 0.242 (min 0.023). Naive baseline recall 0.158 vs
   0.668 for missing = ignore; scribbles 0.124 (over-predicts, recall 0.817); SAM pseudo 0.097.
3. **Fits: yes.** 166 s in total on mps, 0.66 s per image on CPU at 1024, head training 29 s on CPU.

### KolektorSDD2 surface defects

1. **Falls short: no, not with a prompt.** SAM with the GT box gives 0.684 IoU and 5+5 points 0.605,
   better than any trained head (full masks 0.396). The FM only "fails" when nobody tells it where the
   defect is, which is a different story from the workshop's. Defects are 2.7 % of the pixels and
   1.2 components per image, so a single well-placed prompt solves the task.
2. **Visibly different: yes, but noisy.** Mean pairwise IoU 0.248. Naive baseline recall 0.257 vs
   0.729 for missing = ignore. Scribbles 0.099 (recall 0.908: the head paints the whole dark region).
   Caveat: this dataset's head numbers moved by up to 0.09 IoU between two runs of the same seed on
   mps (full @512 was 0.265 in the first run, 0.176 in the rerun; SAM and the Frangi/DINO features are
   bit-identical, the head training on mps is not), because 10 images with tiny defects give the head
   very few positive patches. With 100 images this would settle; in a workshop it will look random.
3. **Fits: yes, easily.** 50 s in total; images are small (231 × 624).

Licence: CC BY-NC-SA 4.0, fine for a free workshop, not for anything commercial derived from it.

### DSB2018 nuclei

1. **Falls short: yes, in a teachable way.** One point gives 0.045 IoU because SAM segments *one*
   nucleus out of 41 per image, which is correct behaviour for the wrong task. 20+20 points reach 0.479,
   the GT box 0.112 (it covers the whole field and SAM returns the field). The trained head with full
   masks reaches 0.686 at 1024 (0.591 at 512), so there is real headroom for the ladder.
2. **Visibly different: yes, with the cleanest ladder.** Scribbles 0.536, SAM pseudo 0.426, naive
   baseline recall 0.688 vs 0.922 for missing = ignore, full masks 0.686. Mean pairwise IoU 0.533, so
   the differences are smaller than on retina but every rung is a recognisable nucleus mask (see
   `dsb_2_heads.png`). The mixed modalities (bright-field H&E, fluorescence) make some images fail
   outright and others succeed, which gives the "whose image fails worst?" moment for free.
3. **Fits: yes, easily.** 50 s in total; 0.69 s per image on CPU at 1024; the images are 256 to 1024 px.

### Is Grad-CAM meaningful?

- **KSDD2**: the only dataset with a natural image-level task (defective / clean), but with 10 defective
  training images the ResNet-18 fits them perfectly (accuracy 1.000) and recognises only 0.200 of the
  held-out defective images; its CAM masks score 0.072 IoU directly and 0.102 when used as pseudo-labels
  for the head. Meaningful in principle, not at the data size that fits a workshop cell. It would need the
  full 356 defective images and minutes of GPU training, i.e. a precomputed checkpoint.
- **HRF / HRF crops**: 100 % of images contain vessels, so there is no negative class at image level. The
  SPEC's fallback (patch-level "contains a vessel") is honest but awkward: almost every patch contains a
  vessel, and a CAM blob cannot outline a 2 px line.
- **DSB2018**: 100 % of images contain nuclei, same problem at image level. At patch level it is slightly
  less awkward than vessels because nuclei are blobs the size of a CAM blob, but it is still a proxy.
- **Pets**: 100 % of images contain a pet; irrelevant for the simple task anyway.

Recommendation for section 06: keep Grad-CAM as the *second* recovered-supervision method with the
patch-level proxy, say plainly that it is a poor fit on this data, and let the comparison table show it.

## Ranked recommendation

1. **HRF retina, as 1024 × 1024 native-resolution crops.** It is Simone's preference and it does the
   job the SPEC asks for: the sharpest "falls short" (SAM 0.10 vs 0.77 on pets), the sharpest naive-
   baseline lesson (recall 0.158 vs 0.668), a working clDice metric, the Frangi extension, and the
   resolution effect (0.274 → 0.335) all come from the same 20 images. The honest risk: absolute IoUs on
   the ladder stay below 0.35, so sections 04 to 08 must be written around "how the predictions differ"
   (which the SPEC already says) and not around "quality rises". Use the whole-fundus view
   (SAM returns the disc, Frangi beats the FM) as the section 02 failure gallery, and crops everywhere
   a head is trained.
2. **DSB2018 nuclei.** The safest ladder: every rung produces a recognisable mask and the numbers climb
   from 0.045 (one point) to 0.686 (full masks). CC0 licence, 83 MB. Choose it if, after section 03 is
   built on retina, the low absolute numbers turn out to confuse learners rather than teach them.
3. **KolektorSDD2.** SAM with a box already solves it (0.684), the head numbers are noisy at 10 images,
   Grad-CAM needs more data than the workshop can train on, and the licence is NC. Not recommended.

**Provisional choice for later issues: HRF retina crops** (`hrfcrop` in `scripts/bakeoff.py`), with the
loader, clDice and the skeleton-based scribble generator from this script as the starting point for the
`SPECIALISED` config. Simone can overrule by picking DSB2018 before issue #3 (hidden labels) starts.
