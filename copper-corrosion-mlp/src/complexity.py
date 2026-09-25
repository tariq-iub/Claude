"""Parameter count, model size, CPU latency/throughput benchmarking."""
from __future__ import annotations

import io
import time

import numpy as np
import torch

from .model import count_parameters


def model_size_bytes(model) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def benchmark_cpu_latency(model, in_dim: int, batch_sizes=(1, 32, 256), n_warmup=5, n_runs=30) -> dict:
    model.eval()
    torch.set_num_threads(1)
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, in_dim)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - t0)
        times = np.array(times)
        results[bs] = {
            "mean_ms": float(times.mean() * 1000), "median_ms": float(np.median(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000), "std_ms": float(times.std() * 1000),
            "throughput_samples_per_s": float(bs / times.mean()),
        }
    return results


def complexity_report(model, in_dim: int) -> dict:
    n_params = count_parameters(model)
    size_bytes = model_size_bytes(model)
    latency = benchmark_cpu_latency(model, in_dim)
    macs = _approx_macs(model, in_dim)
    return {
        "trainable_parameters": n_params,
        "fp32_size_bytes": size_bytes,
        "fp32_size_kb": round(size_bytes / 1024, 2),
        "approx_macs": macs,
        "cpu_latency": latency,
    }


def _approx_macs(model, in_dim: int) -> int:
    macs = 0
    prev = in_dim
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            macs += m.in_features * m.out_features
    return macs
