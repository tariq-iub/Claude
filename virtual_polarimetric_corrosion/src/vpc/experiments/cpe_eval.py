"""Does ambiguity show up as uncertainty? Compares CPE spread of virtual analyzer images with actual error vs reference Stokes
(RQ6) and tests whether corrosion evidence is stable across plausible latent states (CPE stability)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy import stats as sps

from ..models.vp_corrosion_net import VPCorrosionNet


@torch.no_grad()
def cpe_error_correlation(model: VPCorrosionNet, batch: Dict, K: int = 16, angle_deg: float = 45.0) -> Dict[str, float]:
    """Spearman correlation between per-pixel CPE std of I_theta and |I_theta_hat(mean) - I_theta_ref| (synthetic: exact reference
    from S_gt; real: measured analyzer image). A positive correlation supports calibrated ambiguity."""
    model.eval()
    out = model(batch["rgb"])
    cpe = model.cpe(out, K=K, angles_deg=(angle_deg,))
    std = cpe["var"].sqrt()[:, 0].mean(1)                                    # (B,H,W) mean over colour
    S = batch["S_gt"]
    ref = 0.5 * (S[:, :, 0] + S[:, :, 1] * np.cos(np.radians(2 * angle_deg)) + S[:, :, 2] * np.sin(np.radians(2 * angle_deg))).mean(1)
    err = (cpe["mean"][:, 0].mean(1) - ref).abs()
    rho = sps.spearmanr(std.flatten().cpu().numpy(), err.flatten().cpu().numpy())[0]
    return {"spearman_std_vs_abs_error": float(rho), "mean_std": float(std.mean()), "mean_abs_err": float(err.mean())}
