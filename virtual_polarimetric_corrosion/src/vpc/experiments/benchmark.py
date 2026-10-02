"""Edge benchmarking: params, model size, MACs (hook-based), latency/FPS (CPU / GPU with CPU fallback), RAM, VRAM, ONNX Runtime (optional)."""
from __future__ import annotations

import os
import resource
import statistics
import tempfile
import time
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .train import select_device


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / 1e6
    os.unlink(f.name)
    return size


@torch.no_grad()
def count_macs(model: nn.Module, example: Sequence[torch.Tensor]) -> int:
    """Multiply-accumulates of Conv2d / ConvTranspose2d / Linear layers via forward hooks (excludes element-wise ops,
    interpolation and the analytic physics layers, which are O(pixels) and reported separately as 'non-conv ops not counted')."""
    macs = [0]
    hooks = []

    def conv_hook(m, i, o):
        macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    model(*example)
    model.train(was)
    for h in hooks:
        h.remove()
    return int(macs[0])


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3   # kB on Linux -> MB (peak)


@torch.no_grad()
def benchmark_latency(model: nn.Module, input_shape=(1, 3, 128, 128), device: str = "cpu", warmup: int = 3, iters: int = 20,
                      threads: Optional[int] = None) -> Dict[str, float]:
    """device in {'cpu','cuda'}; 'cuda' silently falls back to CPU (reported in 'device'). Reports mean/median/p95 ms, FPS, RAM, VRAM."""
    if threads:
        torch.set_num_threads(threads)
    dev = select_device(device) if device == "cuda" else torch.device("cpu")
    model = model.to(dev).eval()
    x = torch.rand(*input_shape, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"device": dev.type, "threads": torch.get_num_threads(), "batch": input_shape[0], "mean_ms": statistics.mean(ts),
            "median_ms": statistics.median(ts), "p95_ms": ts[min(len(ts) - 1, int(0.95 * len(ts)))],
            "fps": 1000.0 * input_shape[0] / statistics.mean(ts), "ram_mb": _rss_mb(),
            "vram_mb": torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0.0}


def onnx_benchmark(model: nn.Module, input_shape=(1, 3, 128, 128), iters: int = 20) -> Dict[str, float]:
    """Optional: export to ONNX and time ONNX Runtime on CPU. Requires onnx + onnxruntime; returns {} with a reason otherwise.
    Note: the dict outputs of VPCorrosionNet are wrapped to a tuple (seg_logits, pit_logit, severity)."""
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        return {"skipped": f"onnxruntime unavailable: {e}"}

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(x)
            return o["seg_logits"], o["pit_logit"], o["severity"]

    w = Wrap(model.cpu().eval())
    x = torch.rand(*input_shape)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.onnx")
        try:
            torch.onnx.export(w, x, p, opset_version=17, input_names=["rgb"])
        except Exception as e:  # pragma: no cover
            return {"skipped": f"export failed: {e}"}
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        xn = x.numpy()
        for _ in range(3):
            sess.run(None, {"rgb": xn})
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); sess.run(None, {"rgb": xn}); ts.append((time.perf_counter() - t) * 1000)
        return {"ort_mean_ms": statistics.mean(ts), "ort_fps": 1000.0 * input_shape[0] / statistics.mean(ts), "onnx_mb": os.path.getsize(p) / 1e6}


def full_report(model: nn.Module, name: str, size: int = 128, devices=("cpu", "cuda")) -> list[Dict]:
    ex = torch.rand(1, 3, size, size)
    base = {"model": name, "params": count_params(model), "size_mb": model_size_mb(model), "macs_G": count_macs(model, (ex,)) / 1e9, "input": size}
    return [{**base, **benchmark_latency(model, (1, 3, size, size), d)} for d in devices]
