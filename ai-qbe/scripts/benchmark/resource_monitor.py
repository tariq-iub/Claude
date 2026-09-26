"""Background sampler for GPU/RAM utilization during a benchmark run.

Uses `nvidia-smi --query-gpu` (if present) for VRAM/GPU utilization and
`psutil` for system RAM. Both are optional: on a machine without an
NVIDIA GPU (e.g. this codebase's own CI/dev sandbox) GPU sampling is
skipped rather than failing the run, and the resulting report explicitly
records that VRAM was not measured -- never a guessed number.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class ResourceSample:
    timestamp: float
    ram_used_mb: float | None
    gpu_util_percent: float | None
    gpu_mem_used_mb: float | None


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self._samples: list[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        try:
            import psutil  # noqa: F401

            self._has_psutil = True
        except ImportError:
            self._has_psutil = False

    def _sample_once(self) -> ResourceSample:
        ram_used_mb = None
        if self._has_psutil:
            import psutil

            ram_used_mb = psutil.virtual_memory().used / (1024 * 1024)

        gpu_util = None
        gpu_mem = None
        if self._has_nvidia_smi:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                    text=True,
                )
                first_line = out.strip().splitlines()[0]
                util_str, mem_str = [v.strip() for v in first_line.split(",")]
                gpu_util = float(util_str)
                gpu_mem = float(mem_str)
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass

        return ResourceSample(time.time(), ram_used_mb, gpu_util, gpu_mem)

    def _run(self):
        while not self._stop_event.is_set():
            self._samples.append(self._sample_once())
            self._stop_event.wait(self.interval_seconds)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict:
        if not self._samples:
            return {
                "measured": False,
                "reason": "no_samples_collected",
                "gpu_monitoring_available": self._has_nvidia_smi,
            }
        gpu_utils = [s.gpu_util_percent for s in self._samples if s.gpu_util_percent is not None]
        gpu_mems = [s.gpu_mem_used_mb for s in self._samples if s.gpu_mem_used_mb is not None]
        rams = [s.ram_used_mb for s in self._samples if s.ram_used_mb is not None]

        return {
            "measured": True,
            "sample_count": len(self._samples),
            "gpu_monitoring_available": self._has_nvidia_smi,
            "peak_gpu_mem_used_mb": max(gpu_mems) if gpu_mems else None,
            "avg_gpu_util_percent": sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            "peak_ram_used_mb": max(rams) if rams else None,
        }
