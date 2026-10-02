"""Export per-sample predictions (npz) consumed by qualitative figures (7, 8, 14, 15)."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .datasets import collate
from .train import TrainConfig, select_device


@torch.no_grad()
def export_predictions(models: Dict[str, torch.nn.Module], ds, outdir: str = "results/predictions", n: int = 12, cfg: TrainConfig | None = None,
                       data_origin: str = "UNSPECIFIED") -> list[str]:
    """For each sample write rgb, gt mask, per-method argmax + max-prob + entropy, pit prob, S_hat luminance (if available)."""
    cfg = cfg or TrainConfig(device="cpu")
    dev = select_device(cfg.device)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i in range(min(n, len(ds))):
        b = collate([ds[i]])
        rec = {"rgb": b["rgb"][0].numpy(), "gt": b["mask"][0].numpy(), "data_origin": np.array(data_origin)}
        if "stack" in b:
            rec["stack"], rec["angles"] = b["stack"][0].numpy(), b["angles"][0].numpy()
        if "S_gt" in b:
            rec["S_gt"] = b["S_gt"][0].numpy()
        for name, m in models.items():
            m.to(dev).eval()
            o = m(b["rgb"].to(dev), b["normals_prior"].to(dev) if cfg.use_normals_prior else None)
            p = torch.softmax(o["seg_logits"], 1)[0].cpu()
            rec[f"pred__{name}"] = p.argmax(0).numpy()
            rec[f"conf__{name}"] = p.max(0).values.numpy()
            rec[f"entropy__{name}"] = (-(p * p.clamp_min(1e-8).log()).sum(0)).numpy()
            if "pit_logit" in o:
                rec[f"pit__{name}"] = torch.sigmoid(o["pit_logit"])[0, 0].cpu().numpy()
            if "S_hat" in o:
                rec[f"S_hat__{name}"] = o["S_hat"][0].cpu().numpy()
        path = os.path.join(outdir, f"sample_{i:03d}.npz")
        np.savez_compressed(path, **rec)
        paths.append(path)
    return paths
