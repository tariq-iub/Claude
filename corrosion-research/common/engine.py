"""Generic train/eval loop reused by every architecture's train.py / infer.py."""
from __future__ import annotations

import time
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SyntheticCorrosionDataset, NUM_CLASSES, set_all_seeds
from .losses import CombinedSegLoss
from .metrics import summarize


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cpu: bool = False) -> torch.device:
    if not prefer_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(height=128, width=128, batch_size=4, train_len=200, val_len=40, seed=0, num_workers=0):
    train_ds = SyntheticCorrosionDataset(length=train_len, height=height, width=width, seed=seed)
    val_ds = SyntheticCorrosionDataset(length=val_len, height=height, width=width, seed=seed + 999, illumination_aug=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train(
    model_fn,
    out_dir: str,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 3e-3,
    height: int = 96,
    width: int = 96,
    seed: int = 0,
    prefer_cpu: bool = False,
    amp: bool = True,
    grad_accum_steps: int = 1,
    extra_forward_kwargs: dict | None = None,
):
    """
    model_fn: callable() -> nn.Module taking rgb (B,3,H,W) and returning logits (B,C,H,W)
    (uncertainty-producing models may return (logits, extra) tuples; handled below).
    """
    os.makedirs(out_dir, exist_ok=True)
    set_all_seeds(seed)
    device = get_device(prefer_cpu)
    model = model_fn().to(device)
    n_params = count_params(model)
    print(f"[engine] device={device} params={n_params:,} ({n_params/1e6:.4f} M)")

    train_loader, val_loader = make_loaders(height, width, batch_size, seed=seed)
    criterion = CombinedSegLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad()
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                out = model(img)
                logits = out[0] if isinstance(out, tuple) else out
                loss, parts = criterion(logits, mask)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            running += loss.item() * grad_accum_steps
        sched.step()
        avg = running / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": avg, "val_miou": val_metrics["miou"], "val_mf1": val_metrics["mf1"]})
        print(f"epoch {epoch+1}/{epochs} loss={avg:.4f} val_mIoU={val_metrics['miou']:.4f} val_F1={val_metrics['mf1']:.4f}")

    elapsed = time.time() - t0
    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"history": history, "params": n_params, "train_seconds": elapsed}, f, indent=2)
    print(f"[engine] training done in {elapsed:.1f}s, checkpoint -> {ckpt_path}")
    return model, history


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_targets.append(mask.cpu())
        all_probs.append(probs.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    probs = torch.cat(all_probs)
    return summarize(preds, targets, probs, NUM_CLASSES)


@torch.no_grad()
def benchmark_speed(model, device, height=128, width=128, n_warmup=3, n_iters=10):
    model.eval().to(device)
    x = torch.randn(1, 3, height, width, device=device).clamp(0, 1)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(n_iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iters
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    return {"latency_ms": dt * 1000, "fps": 1.0 / dt, "peak_vram_mb": peak_mem}
