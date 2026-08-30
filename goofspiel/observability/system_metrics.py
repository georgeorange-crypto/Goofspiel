"""System and GPU metric probes."""

from __future__ import annotations

import subprocess
from typing import Any


def collect_system_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    try:
        import psutil

        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        metrics["memory_available"] = int(psutil.virtual_memory().available)
    except Exception as exc:
        metrics["psutil_error"] = repr(exc)
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        metrics["gpus"] = []
        for line in proc.stdout.splitlines():
            index, name, util, mem_used, mem_total = [part.strip() for part in line.split(",")]
            metrics["gpus"].append(
                {
                    "index": int(index),
                    "name": name,
                    "utilization_gpu": float(util),
                    "memory_used_mb": float(mem_used),
                    "memory_total_mb": float(mem_total),
                }
            )
    except Exception as exc:
        metrics["gpu_probe_error"] = repr(exc)
    return metrics
