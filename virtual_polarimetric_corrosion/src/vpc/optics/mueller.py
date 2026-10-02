"""Jones and Mueller calculus for the virtual optical chain.

Established physics: Mueller matrices act on Stokes vectors and represent partially polarized light;
Jones calculus (E = [Ex, Ey]^T) only represents fully polarized, coherent light, so it is used here
solely as a *derivation / verification* tool (jones_to_mueller) and for deterministic elements.

Stokes definition: S0 = Ixx+Iyy, S1 = Ixx-Iyy, S2 = 2Re(Ex Ey*), S3 = -2Im(Ex Ey*) (sign of S3 is
convention dependent and unused in linear polarimetry).
"""
from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


def _pauli(dtype=torch.complex64):
    s0 = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
    s1 = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    s2 = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    s3 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    return torch.stack([s0, s1, s2, s3])


def jones_to_mueller(J: Tensor) -> Tensor:
    """Convert Jones matrices (...,2,2) (complex) to Mueller matrices (...,4,4): M_ij = 1/2 Tr(sigma_i J sigma_j J^H)."""
    sig = _pauli(J.dtype).to(J.device)
    Jh = J.conj().transpose(-1, -2)
    # (i,j,...) = 1/2 Tr(sig_i J sig_j Jh)
    t = torch.einsum("iab,...bc,jcd,...da->...ij", sig, J, sig, Jh)
    return 0.5 * t.real


def rotation_mueller(theta: Tensor) -> Tensor:
    """R(theta): re-expresses Stokes in a reference frame rotated by +theta. Shape (...,4,4)."""
    theta = torch.as_tensor(theta, dtype=torch.float32) if not isinstance(theta, Tensor) else theta
    c, s = torch.cos(2 * theta), torch.sin(2 * theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    rows = [
        torch.stack([o, z, z, z], -1),
        torch.stack([z, c, s, z], -1),
        torch.stack([z, -s, c, z], -1),
        torch.stack([z, z, z, o], -1),
    ]
    return torch.stack(rows, -2)


def rotate_mueller(M: Tensor, theta: Tensor) -> Tensor:
    """Express a Mueller matrix defined in a frame rotated by theta in the original frame: R(-theta) M R(theta)."""
    return rotation_mueller(-torch.as_tensor(theta)).to(M) @ M @ rotation_mueller(theta).to(M)


def linear_polarizer_mueller(theta: Tensor, t_max: float = 1.0, t_min: float = 0.0) -> Tensor:
    """Linear diattenuator / polarizer with transmission axis at theta (radians).

    Ideal polarizer (t_max=1, t_min=0): M_P = 1/2 [[1,c,s,0],[c,c^2,cs,0],[s,cs,s^2,0],[0,0,0,0]]
    with c=cos2theta, s=sin2theta. Extinction ratio ER = t_max/t_min models real polarizers.
    """
    theta = torch.as_tensor(theta, dtype=torch.float32) if not isinstance(theta, Tensor) else theta
    c, s = torch.cos(2 * theta), torch.sin(2 * theta)
    a = 0.5 * (t_max + t_min)
    b = 0.5 * (t_max - t_min)
    q = math.sqrt(t_max * t_min)
    z = torch.zeros_like(c)
    r0 = torch.stack([a * torch.ones_like(c), b * c, b * s, z], -1)
    r1 = torch.stack([b * c, a * c ** 2 + q * s ** 2, (a - q) * c * s, z], -1)
    r2 = torch.stack([b * s, (a - q) * c * s, a * s ** 2 + q * c ** 2, z], -1)
    r3 = torch.stack([z, z, z, q * torch.ones_like(c)], -1)
    return torch.stack([r0, r1, r2, r3], -2)


def retarder_mueller(delta: float, theta: Tensor) -> Tensor:
    """Linear retarder (fast axis theta, retardance delta). Quarter-wave: delta=pi/2."""
    theta = torch.as_tensor(theta, dtype=torch.float32) if not isinstance(theta, Tensor) else theta
    c, s = torch.cos(2 * theta), torch.sin(2 * theta)
    cd, sd = math.cos(delta), math.sin(delta)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    rows = [
        torch.stack([o, z, z, z], -1),
        torch.stack([z, c ** 2 + s ** 2 * cd, c * s * (1 - cd), -s * sd], -1),
        torch.stack([z, c * s * (1 - cd), s ** 2 + c ** 2 * cd, c * sd], -1),
        torch.stack([z, s * sd, -c * sd, cd * o], -1),
    ]
    return torch.stack(rows, -2)


def depolarizer_mueller(p: float = 0.0) -> Tensor:
    """Isotropic depolarizer retaining fraction p of the polarized part."""
    return torch.diag(torch.tensor([1.0, p, p, p]))


def fresnel_mueller(Rs: Tensor, Rp: Tensor, delta: Tensor) -> Tensor:
    """Mueller matrix of a specular reflection in the plane-of-incidence frame (x = s axis).
    Built from the Jones matrix diag(r_s, -r_p) = diag(sqrt(Rs), sqrt(Rp) e^{i delta}) (global phase dropped)."""
    a = torch.sqrt(Rs.clamp_min(0)).to(torch.complex64)
    b = (torch.sqrt(Rp.clamp_min(0)) * torch.exp(1j * delta.to(torch.complex64) * 1.0)).to(torch.complex64)
    z = torch.zeros_like(a)
    J = torch.stack([torch.stack([a, z], -1), torch.stack([z, b], -1)], -2)
    return jones_to_mueller(J)


def apply_mueller(M: Tensor, S: Tensor) -> Tensor:
    """S_out = M S_in. M (...,4,4), S (...,4)."""
    return torch.einsum("...ij,...j->...i", M, S)


def analyzer_intensity_from_mueller(S_in4: Tensor, theta: Tensor, t_max: float = 1.0, t_min: float = 0.0) -> Tensor:
    """I_theta = [M_P(theta) S_in]_0 using the full Mueller product (reference implementation).
    S_in4: (...,4); theta broadcastable to S_in4.shape[:-1]."""
    M = linear_polarizer_mueller(theta, t_max, t_min).to(S_in4)
    return apply_mueller(M, S_in4)[..., 0]
