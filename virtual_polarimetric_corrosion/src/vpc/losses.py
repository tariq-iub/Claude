"""Physics-constrained objective:
L_total = l_seg L_seg + l_cls L_cls + l_rec L_rec + l_pol L_pol + l_per L_periodic + l_edge L_edge + l_phys L_phys + l_unc L_unc
(+ l_distill L_distill, + l_pit L_pit). Terms whose targets are unavailable are skipped (returned as 0), so the same
function serves synthetic pre-training, real-data fine-tuning (no polarization GT), and teacher/student distillation.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .corrosion.segmentation import boundary_loss, seg_loss
from .corrosion.severity import ordinal_loss
from .optics.stokes import wrapped_angular_diff

Tensor = torch.Tensor

DEFAULT_WEIGHTS = dict(seg=1.0, cls=0.3, rec=0.5, pol=0.5, per=0.1, edge=0.2, phys=0.2, unc=0.1, pit=0.5, distill=0.5)


def circular_polarization_loss(S_hat: Tensor, S_ref: Tensor, eps: float = 1e-6) -> Dict[str, Tensor]:
    """Compare Stokes fields (B,C,3,H,W): normalised-Stokes L1 + AoLP loss 1-cos(2 dphi) weighted by the reference DoLP
    (AoLP is undefined where the light is unpolarized) -> circular, period-pi safe."""
    s0h, s0r = S_hat[:, :, 0], S_ref[:, :, 0]
    l_int = F.l1_loss(s0h, s0r)
    pol_h = S_hat[:, :, 1:] / s0h.unsqueeze(2).clamp_min(eps)
    pol_r = S_ref[:, :, 1:] / s0r.unsqueeze(2).clamp_min(eps)
    l_pol = F.l1_loss(pol_h, pol_r)
    mh = torch.sqrt(S_hat[:, :, 1] ** 2 + S_hat[:, :, 2] ** 2 + eps ** 2)
    mr = torch.sqrt(S_ref[:, :, 1] ** 2 + S_ref[:, :, 2] ** 2 + eps ** 2)
    cos_d = (S_hat[:, :, 1] * S_ref[:, :, 1] + S_hat[:, :, 2] * S_ref[:, :, 2]) / (mh * mr)   # = cos(2 dphi)
    w = (mr / s0r.clamp_min(eps)).clamp(0, 1).detach()
    l_ang = ((1 - cos_d) * w).sum() / w.sum().clamp_min(eps)
    return {"stokes_int": l_int, "stokes_pol": l_pol, "aolp": l_ang, "total": l_int + l_pol + l_ang}


def periodic_loss(response_fn: Callable[[Tensor], Tensor], thetas_deg: Tensor, orth_weight: float = 1.0, total: Optional[Tensor] = None) -> Tensor:
    """L_periodic: I(theta) ~= I(theta + 180 deg) (and I(t)+I(t+90) ~= S0-equivalent total when ``total`` is given).
    ``response_fn(theta_deg)`` must return analyzer images. Identically zero for the closed-form Stokes analyzer (it is
    a *test*, see tests); it becomes a real constraint for direct/unconstrained analyzer-response heads."""
    a = response_fn(thetas_deg)
    b = response_fn(thetas_deg + 180.0)
    loss = F.mse_loss(a, b)
    if total is not None:
        c = response_fn(thetas_deg + 90.0)
        loss = loss + orth_weight * F.mse_loss(a + c, total.expand_as(a))
    return loss


def physics_constraint_loss(out: Dict[str, Tensor]) -> Tensor:
    """L_phys: realizability |(S1,S2)| <= S0, non-negative analyzer intensities, unit normals, bounded energy
    (specular + diffuse not exceeding the linear image)."""
    terms = []
    if "S_hat" in out:
        S = out["S_hat"]
        pol = torch.sqrt(S[:, :, 1] ** 2 + S[:, :, 2] ** 2 + 1e-12)
        terms.append(F.relu(pol - S[:, :, 0]).pow(2).mean())
        if "stack" in out:
            terms.append(F.relu(-out["stack"]).pow(2).mean())
    if "latent" in out:
        z = out["latent"]
        terms.append((z["n"].norm(dim=1) - 1).pow(2).mean())
        if "lin" in out:
            terms.append(F.relu(z["D"] + z["S"] - out["lin"] * 1.25 - 0.02).pow(2).mean())
        # smoothness prior on roughness/eta/k proxies (they are weakly identifiable): total variation
        for k in ("r", "eta", "k"):
            terms.append(0.1 * (z[k][..., 1:, :] - z[k][..., :-1, :]).abs().mean())
    return sum(terms) if terms else torch.zeros(())


def reconstruction_loss(out: Dict[str, Tensor]) -> Tensor:
    """L_rec: I_lin ~= D + S (latent components must explain the observation; also gradient consistency)."""
    if "lin_rec" not in out:
        return torch.zeros(())
    r, t = out["lin_rec"], out["lin"]
    gx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    gy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    return F.l1_loss(r, t) + 0.5 * (F.l1_loss(gx(r), gx(t)) + F.l1_loss(gy(r), gy(t)))


def uncertainty_loss(logits: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """L_unc: heteroscedastic (aleatoric) cross-entropy: 0.5 exp(-s) CE + 0.5 s (Kendall & Gal 2017)."""
    ce = F.cross_entropy(logits, target, reduction="none").unsqueeze(1)
    lv = logvar.clamp(-6, 4)
    return (0.5 * torch.exp(-lv) * ce + 0.5 * lv).mean()


def distillation_loss(student: Dict[str, Tensor], teacher: Dict[str, Tensor], T: float = 2.0, w_S: float = 1.0,
                      w_feat: float = 1.0, w_kd: float = 1.0) -> Dict[str, Tensor]:
    """L_distill: (i) student S_hat -> teacher *measured* Stokes (circular-safe), (ii) projected-feature MSE,
    (iii) temperature-scaled KL on segmentation logits."""
    parts: Dict[str, Tensor] = {}
    if student.get("S_hat") is not None and "S_meas" in teacher:
        parts["S"] = w_S * circular_polarization_loss(student["S_hat"], teacher["S_meas"].detach())["total"]
    if "feat_proj" in student and "feat" in teacher:
        parts["feat"] = w_feat * F.mse_loss(student["feat_proj"], teacher["feat"].detach())
    if "seg_logits" in student and "seg_logits" in teacher:
        ps = F.log_softmax(student["seg_logits"] / T, 1)
        pt = F.softmax(teacher["seg_logits"].detach() / T, 1)
        parts["kd"] = w_kd * (T ** 2) * F.kl_div(ps, pt, reduction="batchmean") / (student["seg_logits"].shape[-1] * student["seg_logits"].shape[-2])
    parts["total"] = sum(parts.values()) if parts else torch.zeros(())
    return parts


def total_loss(out: Dict[str, Tensor], batch: Dict[str, Tensor], weights: Optional[dict] = None, class_weights: Optional[Tensor] = None,
               teacher_out: Optional[Dict[str, Tensor]] = None) -> tuple[Tensor, Dict[str, float]]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    dev = out["seg_logits"].device
    z = torch.zeros((), device=dev)
    comp: Dict[str, Tensor] = {}
    comp["seg"] = seg_loss(out["seg_logits"], batch["mask"], class_weights)
    if "pit_mask" in batch and "pit_logit" in out:
        comp["pit"] = F.binary_cross_entropy_with_logits(out["pit_logit"], batch["pit_mask"].float().unsqueeze(1),
                                                         pos_weight=torch.tensor(5.0, device=dev))
    if "severity_level" in batch and "severity_ordinal" in out:
        comp["cls"] = ordinal_loss(out["severity_ordinal"], batch["severity_level"]) + F.smooth_l1_loss(out["severity"], batch["severity"])
    comp["edge"] = boundary_loss(out["seg_logits"], batch["mask"])
    if "seg_logvar" in out:
        comp["unc"] = uncertainty_loss(out["seg_logits"], out["seg_logvar"], batch["mask"])
    if "lin_rec" in out:
        comp["rec"] = reconstruction_loss(out).to(dev)
        comp["phys"] = physics_constraint_loss(out).to(dev)
    if "S_gt" in batch and "S_hat" in out:
        comp["pol"] = circular_polarization_loss(out["S_hat"], batch["S_gt"])["total"]
    if "stack" in out and "angles" in out:
        pass
    if teacher_out is not None:
        comp["distill"] = distillation_loss(out, teacher_out)["total"]
    loss = sum(w.get(k, 0.0) * v for k, v in comp.items())
    return loss, {k: float(v.detach()) for k, v in comp.items()}
