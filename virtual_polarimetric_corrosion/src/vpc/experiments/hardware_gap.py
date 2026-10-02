"""Hardware-to-software gap experiment: physical analyzer vs virtual analyzer vs distilled virtual analyzer.

Configuration A: teacher on real analyzer stack.   B: RGB student, no distillation.   C: RGB student, distilled from A.
If the data are synthetic the report is tagged ``SYNTHETIC_SMOKE`` and must not be quoted as a result about real cartridges.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from ..corrosion.segmentation import NUM_CLASSES
from ..models.teacher import PolarimetricTeacher, measured_stokes
from ..models.vp_corrosion_net import NetConfig, VPCorrosionNet
from ..polarization.angle_search import optimise_analyzer
from ..polarization.virtual_analyzer import VirtualAnalyzer
from .datasets import SyntheticCorrosionDataset, collate
from .metrics import (glare_suppression_ratio, highlight_area_reduction, image_similarity, polarimetric_agreement,
                      edge_preservation, texture_preservation)
from .train import TrainConfig, evaluate, select_device, train


@torch.no_grad()
def polarization_fidelity(model: VPCorrosionNet, ds, tcfg: TrainConfig, n_max: int = 32) -> List[Dict]:
    """Virtual (estimated) vs physical (measured) analyzer images and polarimetric features, per sample, luminance channel."""
    dev = select_device(tcfg.device)
    model.to(dev).eval()
    va = VirtualAnalyzer()
    rows = []
    for i in range(min(len(ds), n_max)):
        b = collate([ds[i]])
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["rgb"], b["normals_prior"] if tcfg.use_normals_prior else None)
        angles = b["angles"][0].cpu().tolist()
        phys = b["stack"][0]                                                    # (A,3,H,W) measured analyzer images
        S_meas = measured_stokes(b["stack"], angles)[0]                         # (3,3,H,W)
        S_hat = out["S_hat"][0]
        w = torch.tensor([0.2126729, 0.7151522, 0.0721750], device=dev).view(3, 1, 1, 1)
        SL_meas, SL_hat = (S_meas * w).sum(0).cpu().numpy(), (S_hat * w).sum(0).cpu().numpy()
        virt = va(S_hat.unsqueeze(0), angles)[0]                                # (A,3,H,W)
        # compare in the common exposure scale: measured stack is e*I_theta, estimates are in linear-image units
        for ai, a in enumerate(angles):
            pl, vl = (phys[ai] * w[:, 0]).sum(0).clamp(0, 1).cpu().numpy(), (virt[ai] * w[:, 0]).sum(0).clamp(0, 1).cpu().numpy()
            rows.append({"sample": i, "angle": a, **{f"img_{k}": v for k, v in image_similarity(vl, pl).items()}})
        pa = polarimetric_agreement(SL_hat, SL_meas)
        rgb_lum = (b["lin"][0] * w[:, 0]).sum(0).clamp(0, 1).cpu().numpy()
        phys_lum = (phys * w[:, 0:1].view(1, 3, 1, 1)).sum(1).clamp(0, 1).cpu().numpy()
        virt_lum = (virt * w[:, 0:1].view(1, 3, 1, 1)).sum(1).clamp(0, 1).cpu().numpy()
        mask = (b["mask"][0] > 1).cpu().numpy()
        best_phys = phys_lum[np.argmin([(p >= 0.95).sum() for p in phys_lum])]
        theta_v = optimise_analyzer(virt_lum, angles, mask=mask, weights=(0.5, 1, 0.5, 1, 0.5)).theta_star_deg
        vbest = va(S_hat.unsqueeze(0), theta_v)[0]
        vbest = (vbest * w[:, 0:1].view(1, 3, 1, 1)[0]).sum(0).clamp(0, 1).cpu().numpy() if vbest.dim() == 3 else vbest
        rows.append({"sample": i, "angle": "summary", **{f"pol_{k}": v for k, v in pa.items()},
                     "glare_supp_phys": glare_suppression_ratio(rgb_lum, best_phys), "glare_supp_virt": glare_suppression_ratio(rgb_lum, vbest),
                     "hl_area_red_phys": highlight_area_reduction(rgb_lum, best_phys), "hl_area_red_virt": highlight_area_reduction(rgb_lum, vbest),
                     "theta_star_virtual_deg": theta_v})
    return rows


def run_hardware_gap(train_ds, test_ds, tcfg: TrainConfig, student_cfg: NetConfig | None = None, seeds=(0,), val_ds=None) -> List[dict]:
    """Train A (teacher), B (student), C (distilled student) and report segmentation, polarization fidelity and latency."""
    tag = "SYNTHETIC_SMOKE" if isinstance(test_ds, SyntheticCorrosionDataset) else "REAL"
    student_cfg = student_cfg or NetConfig()
    rows = []
    for seed in seeds:
        tc = TrainConfig(**{**tcfg.__dict__, "seed": seed, "use_normals_prior": student_cfg.cartridge_prior})
        A = PolarimetricTeacher(angles_deg=train_ds[0]["angles"].tolist())
        train(A, train_ds, tc, val_ds, train_teacher=True)
        evA = evaluate(A, test_ds, tc, stack_input=True)
        B = VPCorrosionNet(student_cfg); train(B, train_ds, tc, val_ds)
        C = VPCorrosionNet(student_cfg); train(C, train_ds, tc, val_ds, teacher=A)
        for name, m, ev in (("A_physical_teacher", A, evA), ("B_rgb_virtual", B, evaluate(B, test_ds, tc)), ("C_distilled_virtual", C, evaluate(C, test_ds, tc))):
            row = {"config": name, "seed": seed, "data": tag, **{k: ev["summary"][k] for k in ("miou", "mdice", "mf1")}, "severity_mae": ev["severity"].get("mae", float("nan"))}
            x = (torch.rand(1, *train_ds[0]["stack"].shape) if name.startswith("A") else torch.rand(1, 3, *train_ds[0]["rgb"].shape[-2:]))
            m.eval(); m.to("cpu")
            with torch.no_grad():
                m(x) if name.startswith("A") else m(x)
                t = time.perf_counter(); [m(x) for _ in range(5)]; row["latency_ms_cpu"] = (time.perf_counter() - t) / 5 * 1000
            if not name.startswith("A"):
                pf = polarization_fidelity(m, test_ds, tc, n_max=8)
                summ = [r for r in pf if r["angle"] == "summary"]
                for k in ("pol_dolp_mae", "pol_aolp_wrapped_err_deg", "glare_supp_phys", "glare_supp_virt"):
                    row[k] = float(np.nanmean([r[k] for r in summ]))
            rows.append(row)
    return rows
