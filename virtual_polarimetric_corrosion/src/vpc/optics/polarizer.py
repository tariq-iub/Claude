"""Hardware-emulating optical chain: source polarizer -> reflecting surface -> analyzer -> sensor.

Used (i) to document and test the virtual analyzer against an explicit Mueller product, and
(ii) by the synthetic renderer to generate analyzer images for known ground truth.
"""
from __future__ import annotations

from typing import Optional

import torch

from .mueller import (analyzer_intensity_from_mueller, apply_mueller, linear_polarizer_mueller,
                      rotate_mueller)

Tensor = torch.Tensor


def unpolarized_light(s0: Tensor) -> Tensor:
    z = torch.zeros_like(s0)
    return torch.stack([s0, z, z, z], dim=-1)


def polarized_light(s0: Tensor, theta: Tensor) -> Tensor:
    """100 % linearly polarized light at angle theta."""
    z = torch.zeros_like(s0)
    return torch.stack([s0, s0 * torch.cos(2 * theta), s0 * torch.sin(2 * theta), z], dim=-1)


def optical_chain_intensity(
    S_light: Tensor,
    M_surface_plane: Tensor,
    plane_azimuth: Tensor,
    analyzer_theta: Tensor,
    source_theta: Optional[Tensor] = None,
    analyzer_t: tuple = (1.0, 0.0),
    source_t: tuple = (1.0, 0.0),
) -> Tensor:
    """Intensity at the sensor for: light S_light (...,4) -> optional source polarizer (angle source_theta)
    -> surface Mueller (defined in the plane-of-incidence frame, rotated into the image frame by
    ``plane_azimuth``) -> analyzer at ``analyzer_theta``.
    """
    S = S_light
    if source_theta is not None:
        S = apply_mueller(linear_polarizer_mueller(source_theta, *source_t).to(S), S)
    M_img = rotate_mueller(M_surface_plane, plane_azimuth)
    S = apply_mueller(M_img.to(S), S)
    return analyzer_intensity_from_mueller(S, analyzer_theta, *analyzer_t)
