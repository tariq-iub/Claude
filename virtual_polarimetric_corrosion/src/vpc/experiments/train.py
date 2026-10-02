"""Generic trainer / evaluator shared by synthetic pre-training, real fine-tuning, teacher training and distillation.

CPU-friendly; auto GPU->CPU fallback. Reproducibility: explicit seeds, deterministic algorithms where available,
config + git hash written next to results.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..corrosion.segmentation import CLASS_NAMES, CORRODED, NUM_CLASSES
from ..corrosion.severity import area_to_level, decode_ordinal
from ..losses import total_loss
from .datasets import collate
from .metrics import SegMetrics, severity_regression


def select_device(prefer: str = "cuda") -> torch.device:
    """GPU if requested and usable, otherwise CPU (explicit fallback, never a crash)."""
    if prefer == "cuda" and torch.cuda.is_available():
        try:
            torch.zeros(1).to("cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    loss_weights: dict = field(default_factory=dict)
    use_normals_prior: bool = False
    num_workers: int = 0
    grad_clip: float = 5.0
    log_every: int = 0
    class_weights: Optional[List[float]] = None


def _to(batch: Dict, dev) -> Dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train(model: torch.nn.Module, train_ds, cfg: TrainConfig, val_ds=None, teacher: Optional[torch.nn.Module] = None,
          train_teacher: bool = False) -> Dict:
    """Train ``model`` (RGB student/baseline). If ``teacher`` is given (and not itself trained here) its outputs on the
    hardware stack supervise the student (L_distill). If ``train_teacher`` the model consumes batch['stack'] instead of rgb."""
    set_seed(cfg.seed)
    dev = select_device(cfg.device)
    model.to(dev)
    if teacher is not None:
        teacher.to(dev).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * ((len(train_ds) + cfg.batch_size - 1) // cfg.batch_size)))
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers, generator=g)
    cw = torch.tensor(cfg.class_weights, dtype=torch.float32, device=dev) if cfg.class_weights else None
    history = []
    for ep in range(cfg.epochs):
        model.train()
        t0, agg, nb = time.time(), {}, 0
        for batch in dl:
            batch = _to(batch, dev)
            t_out = None
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(batch["stack"])
            if train_teacher:
                out = model(batch["stack"])
            else:
                out = model(batch["rgb"], batch["normals_prior"] if cfg.use_normals_prior else None)
                out.setdefault("lin", batch["lin"])
            loss, parts = total_loss(out, batch, cfg.loss_weights, cw, t_out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
            nb += 1
        rec = {"epoch": ep, "time_s": time.time() - t0, **{k: v / max(nb, 1) for k, v in agg.items()}}
        if val_ds is not None:
            rec.update({f"val_{k}": v for k, v in evaluate(model, val_ds, cfg, stack_input=train_teacher)["summary"].items() if k in ("miou", "mf1")})
        history.append(rec)
        if cfg.log_every and (ep % cfg.log_every == 0):
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
    return {"history": history, "device": str(dev), "git": git_hash(), "config": asdict(cfg)}


@torch.no_grad()
def evaluate(model: torch.nn.Module, ds, cfg: TrainConfig, stack_input: bool = False, collect_probs: bool = False) -> Dict:
    dev = select_device(cfg.device)
    model.to(dev).eval()
    dl = DataLoader(ds, cfg.batch_size, shuffle=False, collate_fn=collate)
    sm = SegMetrics(NUM_CLASSES, CLASS_NAMES)
    sev_t, sev_p, probs, targets, groups = [], [], [], [], []
    for batch in dl:
        batch = _to(batch, dev)
        out = model(batch["stack"]) if stack_input else model(batch["rgb"], batch["normals_prior"] if cfg.use_normals_prior else None)
        pred = out["seg_logits"].argmax(1)
        sm.update(pred.cpu().numpy(), batch["mask"].cpu().numpy())
        if "severity" in out and "severity" in batch:
            sev_t += batch["severity"].cpu().tolist(); sev_p += out["severity"].cpu().tolist()
        if collect_probs:
            probs.append(torch.softmax(out["seg_logits"], 1).cpu()); targets.append(batch["mask"].cpu())
        groups += batch["group_id"].cpu().tolist()
    res = {"summary": sm.summary(), "per_class": sm.per_class(), "severity": severity_regression(sev_t, sev_p) if sev_t else {}}
    if collect_probs:
        res["probs"], res["targets"] = torch.cat(probs), torch.cat(targets)
    return res


def save_run(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def conv(o):
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if torch.is_tensor(o):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=conv)
