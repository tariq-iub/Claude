import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import polarization as P  # noqa: E402

ANG = [0.0, 45.0, 90.0, 135.0]


def _stack(S):  # Malus: I_theta = 1/2 (S0 + S1 cos2t + S2 sin2t), replicated over 3 colours
    th = torch.tensor(ANG) * math.pi / 180
    I = 0.5 * (S[0] + S[1] * torch.cos(2 * th)[:, None, None] + S[2] * torch.sin(2 * th)[:, None, None])
    return I[:, None].repeat(1, 3, 1, 1)


def _field(h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    s0 = 0.5 + torch.rand(h, w, generator=g)
    rho = 0.6 * torch.rand(h, w, generator=g)
    phi = math.pi * torch.rand(h, w, generator=g)
    return torch.stack([s0, s0 * rho * torch.cos(2 * phi), s0 * rho * torch.sin(2 * phi)])


def test_round_trip_stokes():
    S = _field()
    assert torch.allclose(P.stokes_from_stack(_stack(S), ANG), S, atol=1e-5)


@pytest.mark.parametrize("k,flip", [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (3, True)])
def test_augmentation_matches_physical_rotation(k, flip):
    """Rotating/flipping the scene and re-measuring must equal augmenting the Stokes map.
    Physical model: after a spatial transform the analyzer axis is expressed in the new frame, so the measured stack at
    angle t of the transformed scene equals the original stack at t - k*90 (rotation) and -t (flip), evaluated at moved pixels."""
    S = _field()
    ref = P.augment_stokes(S, k, flip)
    # build the transformed scene's Stokes directly from the angle transformation of each pixel's (rho, phi)
    S2 = S
    if flip:
        S2 = torch.flip(S2, [-1]); phi = 0.5 * torch.atan2(S2[2], S2[1]); rho = torch.hypot(S2[1], S2[2]); phi = -phi
        S2 = torch.stack([S2[0], rho * torch.cos(2 * phi), rho * torch.sin(2 * phi)])
    if k:
        S2 = torch.rot90(S2, k, dims=(-2, -1)); phi = 0.5 * torch.atan2(S2[2], S2[1]) + k * math.pi / 2; rho = torch.hypot(S2[1], S2[2])
        S2 = torch.stack([S2[0], rho * torch.cos(2 * phi), rho * torch.sin(2 * phi)])
    assert torch.allclose(ref, S2, atol=1e-5)


def test_dolp_and_s0_invariant_aolp_not():
    S = _field()
    A = P.augment_stokes(S, 1, False)
    assert torch.allclose(P.pol_features(torch.rot90(S, 1, dims=(-2, -1)), "dolp"), P.pol_features(A, "dolp"), atol=1e-5)
    f0 = P.pol_features(torch.rot90(S, 1, dims=(-2, -1)), "stokes")
    f1 = P.pol_features(A, "stokes")
    assert torch.allclose(f1[:2], -f0[:2], atol=1e-5)  # 90 deg rotation flips the sign of (s1, s2)


def test_modes_shapes_and_none_is_empty():
    S = _field()
    for m in ("none", "dolp", "stokes", "stokes_aolp"):
        assert P.pol_features(S, m).shape[0] == P.pol_channels(m)
    with pytest.raises(ValueError):
        P.pol_channels("bogus")


def test_fusion_wrapper_zero_init_is_identity_and_p0_is_base():
    base = torch.nn.Conv2d(3, 6, 1)
    x = torch.rand(2, 3 + 3, 8, 8)
    w = P.PolFusionWrapper(base, 6, 3)
    assert torch.allclose(w(x), base(x[:, :3]))
    w0 = P.PolFusionWrapper(base, 6, 0)
    assert torch.allclose(w0(x[:, :3]), base(x[:, :3]))
    w(x).sum().backward()
    assert w.head[-1].weight.grad is not None


def test_folder_dataset_pol_end_to_end(tmp_path):
    from PIL import Image
    from common.dataset import FolderCorrosionDataset
    for d in ("images", "masks", "pol"):
        (tmp_path / d).mkdir()
    for i in range(3):
        Image.fromarray((np.random.rand(8, 8, 3) * 255).astype(np.uint8)).save(tmp_path / "images" / f"s{i}.png")
        Image.fromarray(np.random.randint(0, 5, (8, 8)).astype(np.uint8)).save(tmp_path / "masks" / f"s{i}.png")
        np.savez(tmp_path / "pol" / f"s{i}.npz", stack=_stack(_field(seed=i)).numpy(), angles=np.array(ANG))
    plain = FolderCorrosionDataset(str(tmp_path))
    assert plain[0][0].shape == (3, 8, 8)  # default behaviour unchanged
    ds = FolderCorrosionDataset(str(tmp_path), pol_mode="stokes", augment=True, seed=1)
    x, m = ds[0]
    assert x.shape == (6, 8, 8) and m.shape == (8, 8)
    sh = FolderCorrosionDataset(str(tmp_path), pol_mode="dolp", pol_shuffle=True)
    unsh = FolderCorrosionDataset(str(tmp_path), pol_mode="dolp")
    assert not torch.allclose(sh[0][0][3], unsh[0][0][3])
