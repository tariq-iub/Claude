"""Run inference with a trained CPEN checkpoint on a single image.

Usage:
    python infer.py --checkpoint runs/seed0/model.pt --image path/to/img.png --out pred.png
If --image is omitted, runs on one synthetic sample and saves input/GT/pred side by side.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F

from common.dataset import SyntheticCorrosionDataset, CLASS_NAMES
from common.engine import get_device, benchmark_speed
from model import build_model

_PALETTE = np.array(
    [[200, 140, 60], [140, 55, 30], [30, 22, 20], [190, 195, 180], [50, 140, 75], [0, 0, 0]], dtype=np.uint8
)


def colorize(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "runs", "seed0", "model.pt"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "pred.png"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = get_device(args.cpu)
    model = build_model().to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: no checkpoint found, running with random weights (smoke test only)")
    model.eval()

    if args.image is None:
        ds = SyntheticCorrosionDataset(length=1, height=128, width=128, seed=1234)
        img, mask = ds[0]
    else:
        from PIL import Image
        arr = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)
        mask = None

    with torch.no_grad():
        out = model(img.unsqueeze(0).to(device))
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1)[0].cpu().numpy()

    speed = benchmark_speed(model, device, height=img.shape[-2], width=img.shape[-1])
    print(f"inference speed: {speed}")

    from PIL import Image
    pred_rgb = colorize(pred)
    Image.fromarray(pred_rgb).save(args.out)
    print(f"saved prediction -> {args.out}")
    print("class legend:", {i: n for i, n in enumerate(CLASS_NAMES)})


if __name__ == "__main__":
    main()
