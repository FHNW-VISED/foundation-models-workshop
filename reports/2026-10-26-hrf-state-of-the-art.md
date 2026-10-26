# HRF vessel segmentation: where the field is (checked 2026-10-26)

Every number below was read from a document I opened (PDF text extracted or page fetched). "IoU*" = computed as Dice/(2-Dice), not reported by the paper. Dice and F1 are the same quantity for a binary mask.
Your reference point: IoU ~0.28 (= Dice ~0.44) on 1024x1024 native-resolution HRF crops, linear head, frozen DINOv3 ViT-S/16, 5 complete masks.

## 1. Supervised methods on HRF

| Method (year, venue) | Dice/F1 | IoU | Se | Acc | AUC | Split | Resolution | FOV-only? | Source |
|---|---|---|---|---|---|---|---|---|---|
| M2U-Net, 0.55M params (2019, CVPR Workshops per Galdran's table) | 0.7814 | 0.641* | n/r | 0.9635 | n/r | **First 5 of each class train (15), other 30 test** (the Orlando split) | **Full 2336x3504, no crop/resize**; batch 1 | not stated | https://arxiv.org/pdf/1811.07738 (Table 4, split in sec. 4.1) |
| Little W-Net (Galdran et al., preprint 2020; Sci Rep 2022) | 0.8104 (Little U-Net 0.8059) | 0.681* (0.675*) | n/r | n/r | 0.9824 | 15 of 45 for train+val (same split family as M2U-Net, "most restrictive found"), 30 test | **Train at 1024x1024 (downsampled); metrics at native resolution** | **Yes, explicitly** (threshold chosen on train set) | https://arxiv.org/pdf/2009.01907 (Table 2, sec. 2.5, 3.1). Published Sci Rep version not opened; numbers may differ. |
| Shin et al., graph-connectivity net (MedIA 2019), quoted second-hand | 0.8151 | 0.688* | n/r | n/r | 0.9838 | not stated in the source I opened | not stated | not stated | same table in https://arxiv.org/pdf/2009.01907 (Table 2). Original paper not opened. |
| Neuron-programming U-Net, 0.84M params (arXiv Nov 2024) | 0.8123 | 0.684* | 0.8195 | 0.9709 | 0.9885 | **First 5 of each class train, remaining 30 test** | **Downsampled x4 for training**; test resolution not stated | not stated | https://arxiv.org/html/2411.11110 (Table 6) |
| FSG-Net (BMC Med Imaging 2025; arXiv Jan 2025) | 0.8157 | 0.689* (paper's "mIoU" 0.8309 looks like a 2-class mean, not vessel IoU) | 0.8362 | 0.9711 | 0.9874 | 1:1: first half (about 22 images) train, rest validation/test | Padded to a multiple of 32, "no resizing" at inference; but Table 1 lists a padded training shape of 1344 for HRF, so the HRF resolution is unclear to me | not stated | https://arxiv.org/pdf/2501.18921 (Table 2, Table 1) |
| Plain 1.9M-param U-Net, resolution study (arXiv Apr 2026, single author) | **0.8078 at native 3504 px**; 0.8019 at 1752; 0.7918 at 876; 0.7725 at 512 | 0.678*, 0.669*, 0.656*, 0.629* | 0.8211 (native) | n/r | n/r | 5-fold CV over 45 images (36 train per fold, some used for validation) | native 3504x2336 down to 512x512 | not stated | https://arxiv.org/pdf/2604.02977 (Table 3). Thin-vessel (half-width <3 px) sensitivity at native: 0.585, best 0.650 at 876 px. |
| SAM2 + LoRA, fine-tuned (arXiv Oct 2025) | 0.78 | 0.639* | n/r | n/r | 0.98 | 70/30-style split, random 25% test, one model trained on 11 pooled datasets | resized to 1000x1000 | not stated | https://arxiv.org/pdf/2510.10288 (Table I). Paper says it lost to SE-U-Net (HRF Dice 0.84, quoted there) and MFA U-Net (0.81). |

Warning, do not use: TransUNext (arXiv Nov 2024, https://arxiv.org/pdf/2411.02724) prints HRF F1 0.8976 / Se 0.8857, but the text says 38 train / 7 test while its Table 1 says 34/4/7, and the HRF row sits in a table labelled as a DRIVE ablation. Internally inconsistent; I would not cite it.

Reading the table: the credible full-data ceiling on HRF is Dice ~0.80-0.82 (IoU ~0.67-0.69). The numbers move by about 0.01 depending on resolution and split, and Galdran et al. say outright that evaluation protocols across the literature are inconsistent (their sec. 2.5). Even for one method, the sources disagree: Yan et al. 2018 is 0.7814 in the Galdran table but 0.7212 in the M2U-Net table (https://arxiv.org/pdf/2009.01907 vs https://arxiv.org/pdf/1811.07738).

## 2. Human second observer

- HRF: I found no second annotation. The dataset page lists one binary "gold standard" per image made by "a group of experts and clinicians", number of annotators not stated, plus FOV masks (https://www5.cs.fau.de/research/data/fundus-images/). The Budai paper also gives no annotator count (https://pmc.ncbi.nlm.nih.gov/articles/PMC3876700). Absence of evidence rather than a positive statement that none exists, but no HRF second-observer score appears in any paper I opened.
- **Reference from DRIVE (not HRF):** 2nd human annotator Dice **0.7881** (IoU* 0.650), Acc 0.9472. On CHASE_DB1 it is 0.7686 (IoU* 0.624). Source: https://arxiv.org/pdf/1811.07738 (Table 2 and 3). Note that trained networks (0.81-0.83 on DRIVE) beat this human number, so annotation noise is not the ceiling you should compare against.

## 3. Foundation models / label-efficient

| Setting | Dice | IoU | Protocol | Source |
|---|---|---|---|---|
| **ESRVS, 1 labelled image + unlabelled pool**, DINOv3-adapted prototypes + pseudo-labels, U-Net backbone (arXiv Jul 2026) | **0.6347** | 0.465* | HRF: 45 images, 32 train, 13 test; the 1 labelled image is picked by a centroid rule, **not random**; the other 31 train images are used unlabelled; whole image at 1024x1024 (HRF is downsampled ~3.4x in width; DINOv3 stage at 2048x2048); clDice 0.644 | https://arxiv.org/pdf/2607.24453 (Tables 1-3) |
| Same, Mask2Former backbone | 0.7263 (random reference image: 0.7209) | 0.570* (0.564*) | same | same, Table 3 and appendix C |
| Same split, fully supervised on all 32 labelled images | U-Net 0.7687, Mask2Former 0.7523 | 0.624*, 0.603* | same | same, Table 3 |
| Semi-supervised baselines on HRF, 6 labelled + 26 unlabelled (20%) | Mean Teacher 0.504, ICT 0.575, CPS 0.203, CTBCT 0.368, URPC 0.368, KnowSAM 0.556 | 0.404* (ICT) and 0.385* (KnowSAM) for the best two | same | same, Table 2 |
| Zero-shot RetSAM (retina-pretrained SAM-style model, "linear inference" = no weight updates) | 0.779 | **0.640 (reported)** | HRF test = 20% random split (70/10/20) of 45 images, input 640x640; **pretraining data are public fundus images, so possible overlap with HRF images was not ruled out by anything I read** | https://arxiv.org/pdf/2602.07012 (Table 7) |
| RetSAM fine-tuned on full HRF train split | 0.799 | 0.667 | same | same |
| Baselines trained on the same HRF split: SAM2 0.725 (JAC 0.570), SAM3 0.717 (0.559), SAM2-UNet 0.699 (0.538), MedSAM2 0.547 (0.379), Med-SAM-Adapter 0.544 (0.375), **RETFound 0.576 (0.405)**, MedSAM 0.159 (0.086) | see left | reported | same, full-train-set fine-tuning | same, Table 7 |
| Label-fraction curve (1%, 5% of labels) | FIVES only, shown as a figure; I did not extract values, none for HRF | | | same, Fig. 2 and text |
| Cross-dataset, zero HRF labels: Little W-Net trained on DRIVE, tested on HRF | 0.7039 | 0.543* | native-resolution test, FOV-only | https://arxiv.org/pdf/2009.01907 (Table 3) |

Searched and found nothing: a published frozen DINOv2/DINOv3 linear-probe result on HRF or DRIVE with numbers. DINO-MVR (https://arxiv.org/pdf/2605.07221) and SegDINO (https://arxiv.org/pdf/2509.00833) use frozen DINOv3 but contain no retinal data (grep for DRIVE/HRF/vessel/fundus returned nothing). I also could not confirm the claim "DINOv2 cannot isolate vessels in fundus" that a search snippet attributed to arXiv 2304.13425; its abstract page does not say it. No 5-image HRF result was found anywhere; the closest are 1 image (ESRVS) and 15 images (M2U-Net / Galdran split).

## 4. Classical / unsupervised baselines on HRF

| Method | Dice | IoU | Se | Sp | Acc | Source / protocol |
|---|---|---|---|---|---|---|
| Budai et al. 2013 (Int J Biomed Imaging), multiscale Frangi-style, tuned on a small subset, no training labels | n/r | n/r | 0.669 | 0.985 | 0.961 (healthy 0.961, glaucoma 0.965, DR 0.955) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3876700 (Table 4); full-resolution 3504x2336; metrics vs the manual gold standard, FOV handling not stated |
| Frangi et al. (as run in the Budai paper) | n/r | n/r | 0.622 | 0.982 | 0.954 | same page, comparison table |
| Odstrcilik et al. matched filter (as run in Budai) | n/r | n/r | 0.774 | 0.966 | 0.949 | same page; also FAU page gives per-group (healthy Se 0.786, Acc 0.954) https://www5.cs.fau.de/research/data/fundus-images/ |
| Odstrcilik et al., Dice | 0.7324 | 0.578* | | | 0.9494 | https://arxiv.org/pdf/1811.07738 (Table 4) |
| Annunziata et al., unsupervised | 0.7578 | 0.610* | | | 0.9581 | same Table 4 |
| Orlando et al. 2017 (FCRF, supervised, first-5 split) | 0.7158 | 0.558* | | | n/r | same Table 4; 0.7158 also in https://arxiv.org/pdf/2009.01907 |

Takeaway: classical filters sit at Dice ~0.73-0.76 on HRF. A linear head with Dice ~0.44 is below hand-designed filters; the 2019 first-5-image-per-class trained networks are at 0.78.

## 5. Protocol caveats for comparing with your 0.28 IoU

1. **Scale of the image.** Nearly every number above either trains on a downsampled HRF (1024x1024 whole image in Galdran and ESRVS, x4 in Neuron-programming) or evaluates it at native size after training at lower size. Your run uses native-resolution 1024 crops, where vessels are several pixels wide but a ViT-S/16 patch is 16 px, so the logit grid is 64x64 per crop and thin vessels are sub-patch. The resolution study (https://arxiv.org/pdf/2604.02977) shows that even a conv U-Net loses thin-vessel sensitivity at native scale (0.585 vs 0.650 at x4 downsample), so part of the gap is your readout stride and not DINOv3 itself. This is my inference, not a result any paper reports.
2. **Test set and FOV.** Galdran computes metrics at native size inside the FOV; the others do not say. Counting the black border as background raises accuracy and slightly raises Dice. Make sure your IoU is over FOV pixels on all test images, not crops that are mostly vessel-bearing.
3. **Averaging.** Per-image mean Dice and pooled-pixel Dice differ, and papers do not say which. Your IoU averaged over crops is a third variant. Threshold is also tuned differently (Galdran: on train set; most others fixed 0.5).
4. **Training data.** First 5 per class (15 images, complete masks) is the standard "restrictive" HRF split. Your 5 images are a harder, 3x smaller budget, so the right comparison rows are ESRVS 1-label (Dice 0.635 with unlabelled images, 0.72 with a Mask2Former head) and the 15-image rows, not the 36-image CV row.
5. **Heads differ.** All rows are full decoders or fine-tuned encoders. Nothing I found reports a frozen encoder with a linear or 1x1 head on HRF. ESRVS's own appendix and DINO-MVR's ablation (a linear probe was the largest drop vs a small MLP readout, though on Kvasir-SEG, not retina) suggest a linear head is a deliberately weak readout.
6. **Preprint status.** Resolution study, ESRVS, Neuron-programming, SAM2LoRA, RetSAM, Little W-Net preprint are arXiv versions; I did not find peer-reviewed confirmation of their numbers (FSG-Net is BMC Med Imaging 2025, M2U-Net CVPR-W 2019, per the sources above).
