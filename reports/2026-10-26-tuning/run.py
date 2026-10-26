"""HRF tuning experiments: linear head (notebook) vs threshold, a light decoder, a U-Net from scratch.

usage: python run.py <experiment> [steps]
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import DEV, Timer, best_threshold, dataset, dino_feat, log, metrics

torch.manual_seed(42)
np.random.seed(42)
fit_all, held = dataset()
fit5 = fit_all[:5]


def targets_full(it):
    t = it["mask"].astype(np.int8)
    t[~it["fov"]] = -1
    return t


def targets_scrib(it):
    t = np.full(it["mask"].shape, -1, np.int8)
    t[it["scribbles"] == 2] = 0
    t[it["scribbles"] == 1] = 1
    t[~it["fov"]] = -1
    return t


# ---------------------------------------------------------------- linear head, as in the notebook
def linear(items, labels, steps=200):
    feats = [dino_feat(it).to(DEV) for it in items]
    allf = torch.cat([f.flatten(1) for f in feats], 1)
    mu, sd = allf.mean(1), allf.std(1) + 1e-6
    feats = [(f - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1) for f in feats]
    labs = [torch.from_numpy(t).to(DEV) for t in labels]
    npos = sum(int((t == 1).sum()) for t in labs)
    nneg = sum(int((t == 0).sum()) for t in labs)
    head = nn.Conv2d(384, 1, 1).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0
        for f, t in zip(feats, labs):
            z = F.interpolate(head(f[None]), size=t.shape, mode="bilinear", align_corners=False)[0, 0]
            w = torch.where(t == 1, 0.5 / npos, torch.where(t == 0, 0.5 / nneg, 0.0))
            loss = loss + (F.binary_cross_entropy_with_logits(z, (t == 1).float(), reduction="none") * w).sum()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def prob(it):
        f = (dino_feat(it).to(DEV) - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1)
        z = F.interpolate(head(f[None]), size=it["mask"].shape, mode="bilinear", align_corners=False)[0, 0]
        return torch.sigmoid(z).cpu().numpy()

    return prob


# ---------------------------------------------------------------- light decoder on DINO + pixels
class Decoder(nn.Module):
    """DINOv3 patch features projected to 32 channels, upsampled to pixels, joined with the RGB pixels,
    then three small convolutions: the patch features say "vessel region", the pixels say where the edge is."""

    def __init__(self, use_dino=True, use_rgb=True, ch=32):
        super().__init__()
        self.use_dino, self.use_rgb = use_dino, use_rgb
        cin = (ch if use_dino else 0) + (3 if use_rgb else 0)
        self.proj = nn.Conv2d(384, ch, 1) if use_dino else None
        self.body = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, grid, rgb):
        parts = []
        if self.use_dino:
            # grid carries one extra patch of context on each side; trim it after upsampling
            up = F.interpolate(self.proj(grid), scale_factor=16, mode="bilinear", align_corners=False)
            parts.append(up[..., 16:-16, 16:-16])
        if self.use_rgb:
            parts.append(rgb)
        return self.body(torch.cat(parts, 1))


class UNet(nn.Module):
    """A small U-Net on the RGB pixels only: the specialist without a foundation model."""

    def __init__(self, c=(16, 32, 64)):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

        self.e1, self.e2, self.e3 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(2 * c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(2 * c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, grid, rgb):
        e1 = self.e1(rgb)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def prep(it):
    rgb = torch.from_numpy(it["image"]).permute(2, 0, 1).float() / 255
    rgb = (rgb - 0.5) / 0.25
    g = dino_feat(it)  # [384, 64, 64] for a 1024 crop
    g = F.pad(g[None], (1, 1, 1, 1), mode="replicate")[0]  # one patch of context on every side
    return rgb, g


def train_conv(model, items, labels, steps, crop_cells=16, batch=8, lr=2e-3, mu=None, sd=None):
    data = [prep(it) for it in items]
    if mu is None:
        allg = torch.cat([g[:, 1:-1, 1:-1].flatten(1) for _, g in data], 1)
        mu, sd = allg.mean(1), allg.std(1) + 1e-6
    data = [(rgb, (g - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1)) for rgb, g in data]
    labs = [torch.from_numpy(t) for t in labels]
    npos = sum(int((t == 1).sum()) for t in labs)
    nneg = sum(int((t == 0).sum()) for t in labs)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    cpx = crop_cells * 16
    for _ in range(steps):
        rgbs, grids, ts = [], [], []
        for _ in range(batch):
            k = np.random.randint(len(data))
            rgb, g = data[k]
            gh, gw = g.shape[1] - 2, g.shape[2] - 2
            y, x = np.random.randint(0, gh - crop_cells + 1), np.random.randint(0, gw - crop_cells + 1)
            r = rgb[:, y * 16 : y * 16 + cpx, x * 16 : x * 16 + cpx]
            gg = g[:, y : y + crop_cells + 2, x : x + crop_cells + 2]
            t = labs[k][y * 16 : y * 16 + cpx, x * 16 : x * 16 + cpx]
            if np.random.rand() < 0.5:
                r, gg, t = r.flip(-1), gg.flip(-1), t.flip(-1)
            if np.random.rand() < 0.5:
                r, gg, t = r.flip(-2), gg.flip(-2), t.flip(-2)
            rgbs.append(r)
            grids.append(gg)
            ts.append(t)
        rgb, g, t = torch.stack(rgbs).to(DEV), torch.stack(grids).to(DEV), torch.stack(ts).to(DEV)
        z = model(g, rgb)[:, 0]
        w = torch.where(t == 1, 0.5 / npos, torch.where(t == 0, 0.5 / nneg, 0.0)) * (npos + nneg) / batch
        loss = (F.binary_cross_entropy_with_logits(z, (t == 1).float(), reduction="none") * w).sum() / (cpx * cpx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    @torch.no_grad()
    def prob(it):
        rgb, g = prep(it)
        g = (g - mu.view(-1, 1, 1)) / sd.view(-1, 1, 1)
        return torch.sigmoid(model.eval()(g[None].to(DEV), rgb[None].to(DEV))[0, 0]).cpu().numpy()

    return prob


def evaluate(name, prob, train_items, extra=None):
    """Held-out numbers at the 0.5 cut, and at the cut that maximises IoU on the training images."""
    p_held = [prob(it) for it in held]
    p_fit = [prob(it) for it in train_items]
    thr = best_threshold(p_fit, train_items)
    log(name + " @0.5", metrics([p >= 0.5 for p in p_held], held), extra)
    log(name + f" @train-cut {thr:.2f}", metrics([p >= thr for p in p_held], held), extra)
    oracle = best_threshold(p_held, held)
    log(name + f" @oracle-cut {oracle:.2f} (held, for reference only)", metrics([p >= oracle for p in p_held], held), extra)
    return p_held


if __name__ == "__main__":
    exp = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    full5 = [targets_full(it) for it in fit5]
    scrib5 = [targets_scrib(it) for it in fit5]
    with Timer() as t:
        if exp == "linear":
            prob = linear(fit5, full5)
            items, lab = fit5, "linear head · 5 full masks"
        elif exp == "linear_scrib":
            prob = linear(fit5, scrib5)
            items, lab = fit5, "linear head · 5 scribbled images"
        elif exp == "dec_full":
            prob = train_conv(Decoder(), fit5, full5, steps)
            items, lab = fit5, "decoder DINO+RGB · 5 full masks"
        elif exp == "dec_scrib":
            prob = train_conv(Decoder(), fit5, scrib5, steps)
            items, lab = fit5, "decoder DINO+RGB · 5 scribbled images"
        elif exp == "dec_dino_only_full":
            prob = train_conv(Decoder(use_rgb=False), fit5, full5, steps)
            items, lab = fit5, "decoder DINO only · 5 full masks"
        elif exp == "rgb_only_full":
            prob = train_conv(Decoder(use_dino=False), fit5, full5, steps)
            items, lab = fit5, "3-conv on RGB only · 5 full masks"
        elif exp == "unet_full":
            prob = train_conv(UNet(), fit5, full5, steps)
            items, lab = fit5, "U-Net from scratch · 5 full masks"
        elif exp == "unet_scrib":
            prob = train_conv(UNet(), fit5, scrib5, steps)
            items, lab = fit5, "U-Net from scratch · 5 scribbled images"
        elif exp == "dec_full20":
            full20 = [targets_full(it) for it in fit_all]
            prob = train_conv(Decoder(), fit_all, full20, steps)
            items, lab = fit_all, "decoder DINO+RGB · 20 full masks"
        elif exp == "unet_full20":
            full20 = [targets_full(it) for it in fit_all]
            prob = train_conv(UNet(), fit_all, full20, steps)
            items, lab = fit_all, "U-Net from scratch · 20 full masks"
        else:
            raise SystemExit(exp)
    evaluate(lab, prob, items, {"train_sec": t.s, "steps": steps, "device": DEV})
