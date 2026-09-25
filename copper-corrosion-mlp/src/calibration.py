"""Post-hoc temperature scaling, fit on the validation partition only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(val_logits: np.ndarray, y_val: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> float:
    logits = torch.tensor(val_logits, dtype=torch.float32)
    y = torch.tensor(y_val, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = F.cross_entropy(logits / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
