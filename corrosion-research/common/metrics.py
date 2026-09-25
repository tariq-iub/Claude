"""Segmentation metrics: per-class IoU, Dice, P/R/F1, Boundary IoU/F1, Hausdorff, ECE."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    valid = target != IGNORE_INDEX
    pred, target = pred[valid], target[valid]
    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm.float()


@torch.no_grad()
def iou_dice_from_cm(cm: torch.Tensor, eps=1e-6):
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / cm.sum().clamp(min=1)
    return {
        "iou_per_class": iou, "miou": iou.mean().item(),
        "dice_per_class": dice, "mdice": dice.mean().item(),
        "precision": precision, "recall": recall, "f1": f1, "mf1": f1.mean().item(),
        "accuracy": acc.item(),
    }


def _boundary_map(mask: np.ndarray, dilation: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    b = np.zeros_like(mask, dtype=bool)
    for c in np.unique(mask):
        if c == IGNORE_INDEX:
            continue
        m = mask == c
        b |= binary_dilation(m, iterations=dilation) & ~binary_erosion(m, iterations=dilation)
    return b


@torch.no_grad()
def boundary_iou_f1(pred: torch.Tensor, target: torch.Tensor, dilation: int = 2):
    p = pred.cpu().numpy()
    t = target.cpu().numpy()
    ious, f1s = [], []
    for b in range(p.shape[0]):
        pb, tb = _boundary_map(p[b], dilation), _boundary_map(t[b], dilation)
        inter = (pb & tb).sum()
        union = (pb | tb).sum()
        iou = inter / union if union > 0 else 1.0
        prec = inter / pb.sum() if pb.sum() > 0 else 1.0
        rec = inter / tb.sum() if tb.sum() > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        ious.append(iou)
        f1s.append(f1)
    return float(np.mean(ious)), float(np.mean(f1s))


@torch.no_grad()
def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor):
    """Symmetric Hausdorff distance between boundary point sets, averaged over batch."""
    from scipy.spatial.distance import directed_hausdorff
    p = pred.cpu().numpy()
    t = target.cpu().numpy()
    vals = []
    for b in range(p.shape[0]):
        pb, tb = _boundary_map(p[b]), _boundary_map(t[b])
        ppts, tpts = np.argwhere(pb), np.argwhere(tb)
        if len(ppts) == 0 or len(tpts) == 0:
            continue
        d = max(directed_hausdorff(ppts, tpts)[0], directed_hausdorff(tpts, ppts)[0])
        vals.append(d)
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    valid = target != IGNORE_INDEX
    conf, pred = probs.max(dim=1)
    conf, pred, target = conf[valid], pred[valid], target[valid]
    correct = (pred == target).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.float().mean()) * (correct[m].mean() - conf[m].mean()).abs()
    return ece.item()


def summarize(pred: torch.Tensor, target: torch.Tensor, probs: torch.Tensor, num_classes: int):
    cm = confusion_matrix(pred.flatten(), target.flatten(), num_classes)
    out = iou_dice_from_cm(cm)
    b_iou, b_f1 = boundary_iou_f1(pred, target)
    out["boundary_iou"] = b_iou
    out["boundary_f1"] = b_f1
    out["hausdorff"] = hausdorff_distance(pred, target)
    out["ece"] = expected_calibration_error(probs, target)
    return out
