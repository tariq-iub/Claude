#!/usr/bin/env python3
"""Edge benchmark (params, size, MACs, latency/FPS, RAM/VRAM; GPU->CPU fallback; optional ONNX Runtime). Appends to results/runtime.csv, memory.csv."""
import _common  # noqa
import argparse
from vpc.experiments.benchmark import full_report, onnx_benchmark
from vpc.experiments.tables import append_rows
from vpc.models.baseline import build_baseline
from vpc.models.vp_corrosion_net import NetConfig, VPCorrosionNet

ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=128); ap.add_argument("--onnx", action="store_true")
a = ap.parse_args()
models = {"vp_corrosion_net": VPCorrosionNet(NetConfig()), "mobile_rgb": build_baseline("mobile_rgb"), "unet_lite": build_baseline("unet_lite")}
rt, mem = [], []
for n, m in models.items():
    for r in full_report(m, n, a.size):
        rt.append({k: r[k] for k in ("device", "threads", "batch", "input", "mean_ms", "median_ms", "p95_ms", "fps")} | {"model": n})
        mem.append({"model": n, "params": r["params"], "size_mb": r["size_mb"], "macs_G": r["macs_G"], "ram_mb": r["ram_mb"], "vram_mb": r["vram_mb"], "device": r["device"]})
    if a.onnx:
        print(n, onnx_benchmark(m, (1, 3, a.size, a.size)))
print(append_rows("runtime", rt, "results"), append_rows("memory", mem, "results"))
