"""RGB-only student for hardware-to-software distillation = VPCorrosionNet + distillation-target interface."""
from __future__ import annotations

from typing import Dict

import torch

from .vp_corrosion_net import NetConfig, VPCorrosionNet


class StudentVPCorrosionNet(VPCorrosionNet):
    """Same network as VPCorrosionNet; adds ``distill_outputs`` that expose exactly the quantities supervised by the
    teacher: S_hat (vs measured S), feature projection (vs teacher decoder features), logits (vs teacher logits).
    Deployment uses only RGB; the teacher/hardware are training-time only."""

    def distill_outputs(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"S": out.get("S_hat"), "feat": out["feat_proj"], "seg_logits": out["seg_logits"]}
