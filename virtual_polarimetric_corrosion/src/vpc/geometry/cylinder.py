"""Cylindrical-cartridge geometry prior (orthographic camera, axis in/near the image plane).

Camera frame: x right, y up, z toward the camera (right-handed); image rows increase downward so y = -row.
For a cylinder of radius R with axis direction a = (cos al cos el, sin al cos el, sin el), the surface normal at
signed lateral offset s = d/R (d = signed distance from the axis in the image plane) is
    n = s * e1 + sqrt(1 - s^2) * e2,   e1, e2 orthonormal and perpendicular to a, e2 pointing toward the camera.
Perspective and finite-length end effects (rim, bullet ogive) are ignored: documented limitation.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

Tensor = torch.Tensor


def fit_cylinder_from_mask(mask: np.ndarray) -> Dict[str, float]:
    """Estimate axis angle (radians, image frame x right / y up), centre (col,row) and radius proxy (px) from a
    binary silhouette via PCA; radius = median half-width perpendicular to the axis."""
    rows, cols = np.nonzero(mask)
    if rows.size < 10:
        raise ValueError("silhouette too small")
    pts = np.stack([cols.astype(np.float64), -rows.astype(np.float64)], 1)  # x, y(up)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, V = np.linalg.eigh(cov)
    a = V[:, np.argmax(w)]
    alpha = math.atan2(a[1], a[0]) % math.pi
    t = np.array([-a[1], a[0]])
    d = (pts - c) @ t
    # radius: half of robust lateral extent
    R = 0.5 * (np.percentile(d, 99.5) - np.percentile(d, 0.5))
    lateral_center = 0.5 * (np.percentile(d, 99.5) + np.percentile(d, 0.5))
    c = c + lateral_center * t
    return {"alpha": float(alpha), "cx": float(c[0]), "cy_up": float(c[1]), "radius": float(R)}


def cylinder_normals(h: int, w: int, alpha: float, cx: float, cy_up: float, radius: float,
                     elevation: float = 0.0, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Return (normals (3,H,W), lateral s (H,W) in [-1,1] clipped; pixels with |s|>1 get s=+-1)."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
    x, y = xs - cx, -ys - cy_up
    ca, sa = math.cos(alpha), math.sin(alpha)
    tx, ty = -sa, ca
    s = ((x * tx + y * ty) / max(radius, 1e-6)).clamp(-1, 1)
    ce, se = math.cos(elevation), math.sin(elevation)
    a = torch.tensor([ca * ce, sa * ce, se], dtype=dtype)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    e1 = torch.cross(z, a, dim=0)
    e1 = e1 / e1.norm().clamp_min(1e-8)  # lateral in-plane direction (perpendicular to axis and view)
    e2 = torch.cross(a, e1, dim=0)
    e2 = e2 * torch.sign(e2[2] + 1e-12)  # toward the camera
    n = s.unsqueeze(0) * e1.view(3, 1, 1) + torch.sqrt((1 - s ** 2).clamp_min(0)).unsqueeze(0) * e2.view(3, 1, 1)
    return n / n.norm(dim=0, keepdim=True).clamp_min(1e-8), s


def cylinder_incidence_angle(normals: Tensor, view: Tensor | None = None) -> Tensor:
    v = torch.tensor([0.0, 0.0, 1.0], dtype=normals.dtype) if view is None else view
    return torch.acos((normals * v.view(3, 1, 1)).sum(0).clamp(-1, 1))


def expected_highlight_lateral_position(light_dir: np.ndarray, alpha: float) -> float:
    """Lateral offset s* in [-1,1] where a mirror-like facet (n = h) lies for a distant light at light_dir
    (3,) and view z: s* = (h . e1) / |h_perp_axis|. Useful as a geometric prior for highlight location."""
    l = np.asarray(light_dir, float)
    l = l / np.linalg.norm(l)
    h = l + np.array([0, 0, 1.0])
    h = h / np.linalg.norm(h)
    a = np.array([math.cos(alpha), math.sin(alpha), 0.0])
    h_perp = h - np.dot(h, a) * a
    t = np.array([-math.sin(alpha), math.cos(alpha), 0.0])
    return float(np.clip(np.dot(h_perp, t) / max(np.linalg.norm(h_perp), 1e-9), -1, 1))
