"""Training loop with early stopping, best-validation-checkpoint restoration."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score

from .model import build_model
from .losses import WeightedCrossEntropy, FocalLoss, compute_class_weights


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_macro_f1: list = field(default_factory=list)
    best_epoch: int = -1
    stopped_epoch: int = -1
    stopping_reason: str = ""


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(X_train, y_train, X_val, y_val, in_dim, n_classes, cfg, seed=42, device="cpu"):
    set_seed(seed)
    model = build_model(in_dim, n_classes, cfg["model"]).to(device)

    class_weights = None
    if cfg["loss"]["class_weights"] != "none":
        class_weights = compute_class_weights(y_train, n_classes, cfg["loss"]["class_weights"],
                                               beta=cfg["loss"].get("effective_beta", 0.999))
    if cfg["loss"]["type"] == "focal":
        criterion = FocalLoss(class_weights, gamma=cfg["loss"].get("focal_gamma", 2.0))
    else:
        criterion = WeightedCrossEntropy(class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                             weight_decay=cfg["training"]["weight_decay"])
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    batch_size = min(cfg["training"]["batch_size"], len(X_train))

    scheduler_type = cfg["training"].get("scheduler", "plateau")
    if scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(3, patience // 3))
    elif scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    else:
        sched = None

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)

    n = len(Xtr)
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    hist = TrainHistory()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = criterion(val_logits, yva).item()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        hist.train_loss.append(epoch_loss)
        hist.val_loss.append(val_loss)
        hist.val_macro_f1.append(val_f1)

        if scheduler_type == "plateau":
            sched.step(val_loss)
        elif sched is not None:
            sched.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            hist.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                hist.stopped_epoch = epoch
                hist.stopping_reason = f"no val_loss improvement for {patience} epochs"
                break
    else:
        hist.stopped_epoch = max_epochs - 1
        hist.stopping_reason = "max_epochs reached"

    model.load_state_dict(best_state)  # restore best-validation model, never last epoch
    return model, hist, class_weights
