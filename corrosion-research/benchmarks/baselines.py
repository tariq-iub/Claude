"""
Classical and lightweight-deep baselines used as the comparison set required
by the research protocol: GMM, K-Means, SLIC-based classification, classical
CIELAB+ΔE00 nearest-centroid classification, and a tiny U-Net.

These are intentionally simple, dependency-light implementations (scikit-
learn / scikit-image + a small PyTorch U-Net) so the whole benchmark suite
runs on a CPU-only, 2GB-VRAM-class machine.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from common.colorspace import rgb_to_lab, deltaE2000
from common.dataset import NUM_CLASSES, CLASS_NAMES


# --------------------------------------------------------------------------- #
# Classical, non-learned baselines (operate per-image, no training loop)
# --------------------------------------------------------------------------- #

def kmeans_segment(rgb_hwc: np.ndarray, n_clusters: int = NUM_CLASSES) -> np.ndarray:
    from sklearn.cluster import KMeans
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    h, w, _ = lab.shape
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit(lab.reshape(-1, 3))
    return km.labels_.reshape(h, w)


def gmm_segment(rgb_hwc: np.ndarray, n_components: int = NUM_CLASSES) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    h, w, _ = lab.shape
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(lab.reshape(-1, 3))
    return gmm.predict(lab.reshape(-1, 3)).reshape(h, w)


def dbscan_segment(rgb_hwc: np.ndarray, eps: float = 5.0, min_samples: int = 20) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    h, w, _ = lab.shape
    # subsample for tractability, then nearest-assign the rest
    flat = lab.reshape(-1, 3)
    idx = np.random.RandomState(0).choice(len(flat), size=min(2000, len(flat)), replace=False)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(flat[idx])
    labels_sub = db.labels_
    labels_sub[labels_sub < 0] = labels_sub.max() + 1 if labels_sub.max() >= 0 else 0
    # nearest-centroid propagate to all pixels
    centroids = np.array([flat[idx][labels_sub == c].mean(0) for c in np.unique(labels_sub)])
    d = np.linalg.norm(flat[:, None, :] - centroids[None, :, :], axis=2)
    return d.argmin(1).reshape(h, w)


def slic_classify(rgb_hwc: np.ndarray, class_centroids_lab: np.ndarray, n_segments: int = 200) -> np.ndarray:
    """SLIC superpixels + nearest-Lab-centroid classification per superpixel."""
    from skimage.segmentation import slic
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    segments = slic(rgb_hwc, n_segments=n_segments, compactness=10, start_label=0, channel_axis=2)
    out = np.zeros(segments.shape, dtype=np.int64)
    for s in np.unique(segments):
        mask = segments == s
        mean_lab = lab[mask].mean(0)
        d = np.linalg.norm(class_centroids_lab - mean_lab[None, :], axis=1)
        out[mask] = d.argmin()
    return out


def lab_deltae_classify(rgb_hwc: np.ndarray, class_centroids_lab: np.ndarray) -> np.ndarray:
    """Classical per-pixel CIEDE2000 nearest-centroid classification."""
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0))  # (1,3,H,W)
    _, _, h, w = lab.shape
    lab_flat = lab.permute(0, 2, 3, 1).reshape(-1, 3)  # (N,3)
    centroids = torch.from_numpy(class_centroids_lab).float()  # (C,3)
    dists = torch.stack([deltaE2000(lab_flat, centroids[c].unsqueeze(0).expand_as(lab_flat)) for c in range(len(centroids))], dim=1)
    return dists.argmin(1).reshape(h, w).numpy()


def lut3d_classify(rgb_hwc: np.ndarray, class_centroids_rgb: np.ndarray) -> np.ndarray:
    """Nearest-centroid classification in raw 3D RGB colour space (the '3D colour LUT' baseline)."""
    h, w, _ = rgb_hwc.shape
    flat = rgb_hwc.reshape(-1, 3)
    d = np.linalg.norm(flat[:, None, :] - class_centroids_rgb[None, :, :], axis=2)
    return d.argmin(1).reshape(h, w)


def voxel5d_classify(rgb_hwc: np.ndarray, class_centroids_labxy: np.ndarray) -> np.ndarray:
    """Nearest-centroid classification in 5D Lab+xy voxel space."""
    h, w, _ = rgb_hwc.shape
    lab = rgb_to_lab(torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
    yy, xx = np.mgrid[0:h, 0:w]
    xy_norm = np.stack([xx / w, yy / h], axis=-1) * 100.0  # scale to Lab-comparable range
    feat = np.concatenate([lab, xy_norm], axis=-1).reshape(-1, 5)
    d = np.linalg.norm(feat[:, None, :] - class_centroids_labxy[None, :, :], axis=2)
    return d.argmin(1).reshape(h, w)


def estimate_class_centroids(images: list[np.ndarray], masks: list[np.ndarray], num_classes: int = NUM_CLASSES, space: str = "lab"):
    """Estimate per-class colour centroids from a small labelled reference set (used by all nearest-centroid baselines)."""
    sums = np.zeros((num_classes, 3 if space != "labxy" else 5))
    counts = np.zeros(num_classes)
    for img, mask in zip(images, masks):
        if space in ("lab", "labxy"):
            lab = rgb_to_lab(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
            feat = lab
        else:
            feat = img
        if space == "labxy":
            h, w, _ = feat.shape
            yy, xx = np.mgrid[0:h, 0:w]
            xy_norm = np.stack([xx / w, yy / h], axis=-1) * 100.0
            feat = np.concatenate([feat, xy_norm], axis=-1)
        for c in range(num_classes):
            m = mask == c
            if m.sum() > 0:
                sums[c] += feat[m].sum(0)
                counts[c] += m.sum()
    counts = np.clip(counts, 1, None)
    return sums / counts[:, None]


# --------------------------------------------------------------------------- #
# Tiny U-Net baseline (the canonical deep-learning reference point)
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """A parameter-light U-Net (~0.3M params) used as the standard encoder-decoder baseline."""

    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_tiny_unet():
    return TinyUNet()
