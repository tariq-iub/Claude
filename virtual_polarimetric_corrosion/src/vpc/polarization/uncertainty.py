"""Uncertainty tools: heteroscedastic NLL, MC dropout, deep-ensemble aggregation, Counterfactual Polarimetric Ensemble (CPE)."""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

Tensor = torch.Tensor


def heteroscedastic_nll(mean: Tensor, logvar: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Gaussian NLL with predicted log-variance (Kendall & Gal 2017 form, constants dropped)."""
    nll = 0.5 * (torch.exp(-logvar) * (mean - target) ** 2 + logvar)
    return nll.mean() if reduction == "mean" else nll


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: Tensor, n_samples: int = 10, key: str = "seg_logits") -> Dict[str, Tensor]:
    """Monte-Carlo dropout (epistemic). Returns mean softmax prob, predictive entropy, mutual information (BALD)."""
    was = model.training
    model.eval()
    enable_dropout(model)
    probs = torch.stack([torch.softmax(model(x)[key], 1) for _ in range(n_samples)])  # (T,B,K,H,W)
    model.train(was)
    mean = probs.mean(0)
    ent = -(mean * mean.clamp_min(1e-8).log()).sum(1)
    exp_ent = -(probs * probs.clamp_min(1e-8).log()).sum(2).mean(0)
    return {"prob": mean, "entropy": ent, "mutual_info": (ent - exp_ent).clamp_min(0), "var": probs.var(0).sum(1)}


@torch.no_grad()
def ensemble_predict(models, x: Tensor, key: str = "seg_logits") -> Dict[str, Tensor]:
    probs = torch.stack([torch.softmax(m.eval()(x)[key], 1) for m in models])
    mean = probs.mean(0)
    ent = -(mean * mean.clamp_min(1e-8).log()).sum(1)
    exp_ent = -(probs * probs.clamp_min(1e-8).log()).sum(2).mean(0)
    return {"prob": mean, "entropy": ent, "mutual_info": (ent - exp_ent).clamp_min(0)}


def summarize_samples(samples: Tensor, level: float = 0.95) -> Dict[str, Tensor]:
    """samples (K,...) -> mean, variance, lower/upper credible bounds (empirical quantiles)."""
    q = (1 - level) / 2
    return {
        "mean": samples.mean(0), "var": samples.var(0, unbiased=False),
        "lo": torch.quantile(samples, q, dim=0), "hi": torch.quantile(samples, 1 - q, dim=0),
    }


@torch.no_grad()
def counterfactual_polarimetric_ensemble(latent_sampler: Callable[[], Tensor], analyzer, angles_deg,
                                         K: int = 16, level: float = 0.95) -> Dict[str, Tensor]:
    """Counterfactual Polarimetric Ensemble (CPE).

    ``latent_sampler()`` must return one physically-plausible Stokes field (B,C,3,H,W) drawn from the latent
    distribution (see LatentOpticsHead.sample_stokes). Each draw produces analyzer images I_theta^(k); we report
    mean, variance and credible interval across draws, plus DoLP_hat / AoLP stability (circular variance).
    """
    stacks, dolps, c2, s2 = [], [], [], []
    for _ in range(K):
        S = latent_sampler()
        stacks.append(analyzer(S, angles_deg))
        mag = torch.sqrt(S[:, :, 1] ** 2 + S[:, :, 2] ** 2 + 1e-12)
        dolps.append((mag / S[:, :, 0].clamp_min(1e-6)).clamp(0, 1))
        c2.append(S[:, :, 1] / mag)
        s2.append(S[:, :, 2] / mag)
    out = summarize_samples(torch.stack(stacks), level)
    out["dolp"] = summarize_samples(torch.stack(dolps), level)
    R = torch.sqrt(torch.stack(c2).mean(0) ** 2 + torch.stack(s2).mean(0) ** 2)
    out["aolp_circ_var"] = 1.0 - R  # 0: all draws agree on AoLP; 1: uniformly spread
    return out
