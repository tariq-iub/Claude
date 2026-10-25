"""Approach D: deep semantic segmentation (U-Net / DeepLabV3+) with PyTorch.

Optional dependency: ``pip install cucorr[deep]`` (torch, segmentation-models-pytorch).
If ``segmentation_models_pytorch`` is absent a small built-in U-Net is used so the code
path stays testable on CPU. Uncertainty is estimated with Monte-Carlo dropout.

Supervision: ``approach_d.supervision`` must be ``reference`` (human masks) or ``pseudo``
(masks produced by another approach). Pseudo-supervised results must be reported as
provisional; they are flagged in ``describe()``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..schema import NUM_CLASSES, UNCERTAIN
from .base import Segmenter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH = False


def require_torch():
    if not TORCH:
        raise ImportError("Approach D needs PyTorch: pip install 'cucorr[deep]'")


if TORCH:
    class TinyUNet(nn.Module):
        """Minimal 3-level U-Net with dropout (CPU-friendly fallback)."""

        def __init__(self, n_classes: int = NUM_CLASSES, width: int = 16, p_drop: float = 0.2):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                     nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
            w = width
            self.e1, self.e2, self.e3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
            self.drop = nn.Dropout2d(p_drop)
            self.head = nn.Conv2d(w, n_classes, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            e3 = self.drop(self.e3(F.max_pool2d(e2, 2)))
            d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[2:]), e2], 1))
            d1 = self.d1(torch.cat([F.interpolate(self.drop(d2), size=e1.shape[2:]), e1], 1))
            return self.head(d1)

    def dice_loss(logits, target, ignore=UNCERTAIN, eps=1.0):
        valid = (target != ignore)
        t = torch.where(valid, target, torch.zeros_like(target))
        p = logits.softmax(1) * valid[:, None]
        oh = F.one_hot(t, NUM_CLASSES).permute(0, 3, 1, 2).float() * valid[:, None]
        inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        present = oh.sum((0, 2, 3)) > 0
        return 1 - ((2 * inter + eps) / (den + eps))[present].mean()


def build_model(cfg_d: dict):
    require_torch()
    try:
        import segmentation_models_pytorch as smp
        weights = "imagenet" if cfg_d.get("pretrained") else None
        arch = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[cfg_d["architecture"]]
        return arch(encoder_name=cfg_d["encoder"], encoder_weights=weights, classes=NUM_CLASSES), "smp"
    except ImportError:
        return TinyUNet(), "tiny_unet"


def augment(img: np.ndarray, m: np.ndarray, rng: np.random.Generator):
    """Physically plausible augmentation: flips/rot90, exposure, white-balance gain, gamma."""
    k = rng.integers(4)
    img, m = np.rot90(img, k), np.rot90(m, k)
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    f = img.astype(np.float32) / 255.0
    f = f * rng.uniform(0.8, 1.2) * rng.uniform(0.93, 1.07, size=3)
    f = np.clip(f, 0, 1) ** rng.uniform(0.85, 1.15)
    return np.ascontiguousarray(f), np.ascontiguousarray(m)


class DeepSegmenter(Segmenter):
    name = "approach_d_deep"

    def __init__(self, cfg: dict, checkpoint_dir: str | Path | None = None):
        require_torch()
        super().__init__(cfg)
        self.p = cfg["approach_d"]
        self.seed = cfg.get("seed", 0)
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.backend = build_model(self.p)
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history: list[dict] = []

    def _tiles(self, img, m, rng):
        t = self.p["tile_size"]
        h, w = m.shape
        if h <= t and w <= t:
            return img, m
        y, x = rng.integers(0, max(h - t, 0) + 1), rng.integers(0, max(w - t, 0) + 1)
        return img[y:y + t, x:x + t], m[y:y + t, x:x + t]

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
            val: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None) -> "DeepSegmenter":
        rng = np.random.default_rng(self.seed)
        imgs = [self.preprocess(i) for i in images]
        counts = np.bincount(np.concatenate([m[m != UNCERTAIN].ravel() for m in masks]), minlength=NUM_CLASSES)
        wts = torch.tensor(np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0),
                           dtype=torch.float32, device=self.device)
        wts = wts / wts[wts > 0].mean()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.p["lr"], weight_decay=1e-4)
        use_amp = self.p.get("mixed_precision") and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best, bad, best_state = np.inf, 0, None
        for epoch in range(self.p["epochs"]):
            self.model.train()
            order = rng.permutation(len(imgs))
            tot = 0.0
            for s in range(0, len(order), self.p["batch_size"]):
                bx, by = [], []
                for i in order[s:s + self.p["batch_size"]]:
                    a, b = augment(*self._tiles(imgs[i], masks[i], rng), rng)
                    bx.append(a); by.append(b)
                hh = min(x.shape[0] for x in bx) // 32 * 32 or min(x.shape[0] for x in bx)
                ww = min(x.shape[1] for x in bx) // 32 * 32 or min(x.shape[1] for x in bx)
                x = torch.tensor(np.stack([b[:hh, :ww] for b in bx])).permute(0, 3, 1, 2).to(self.device)
                y = torch.tensor(np.stack([b[:hh, :ww] for b in by]).astype(np.int64)).to(self.device)
                with torch.autocast(self.device.type, enabled=use_amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y, weight=wts, ignore_index=UNCERTAIN) \
                        + self.p["dice_weight"] * dice_loss(logits, y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += float(loss)
            vloss = self._val_loss(val) if val else tot
            self.history.append({"epoch": epoch, "train_loss": tot, "val_loss": vloss, "t": time.time()})
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.checkpoint_dir:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": best_state, "cfg": self.p, "epoch": epoch},
                               self.checkpoint_dir / f"{self.name}_best.pt")
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def _val_loss(self, val) -> float:
        self.model.eval()
        tot = 0.0
        with torch.no_grad():
            for im, m in zip(*val):
                lg = self._logits(self.preprocess(im))
                y = torch.tensor(m.astype(np.int64))[None].to(self.device)
                tot += float(F.cross_entropy(lg, y, ignore_index=UNCERTAIN))
        return tot

    def _logits(self, img: np.ndarray):
        h, w = img.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        x = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        x = F.pad(x, (0, pw, 0, ph), mode="reflect").to(self.device)
        return self.model(x)[..., :h, :w]

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        img = self.preprocess(image)
        n = max(1, self.p.get("mc_dropout_samples", 1))
        self.model.eval()
        if n > 1:  # MC dropout: enable dropout layers only
            for mod in self.model.modules():
                if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
                    mod.train()
        with torch.no_grad():
            ps = torch.stack([self._logits(img).softmax(1)[0] for _ in range(n)])
        self.last_mc_std = ps.std(0).mean(0).cpu().numpy() if n > 1 else None
        return ps.mean(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def describe(self) -> dict:
        d = super().describe()
        d.update(backend=self.backend, supervision=self.p.get("supervision"),
                 provisional=self.p.get("supervision") == "pseudo", device=str(self.device),
                 n_params=int(sum(p.numel() for p in self.model.parameters())))
        return d
