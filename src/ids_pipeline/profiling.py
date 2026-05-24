"""Lightweight timing and resource profiling."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterator


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:  # pragma: no cover - optional dependency/runtime
        return None


def _gpu_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:  # pragma: no cover - optional dependency/runtime
        return None
    return None


@dataclass
class Profiler:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_ram_mb: float | None = None

    @contextlib.contextmanager
    def track(self, name: str) -> Iterator[None]:
        before_ram = _rss_mb()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            after_ram = _rss_mb()
            candidates = [v for v in [self.peak_ram_mb, before_ram, after_ram] if v is not None]
            self.peak_ram_mb = max(candidates) if candidates else None
            self.records[name] = {
                "seconds": elapsed,
                "rss_before_mb": before_ram,
                "rss_after_mb": after_ram,
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.records,
            "peak_ram_mb": self.peak_ram_mb,
            "gpu_vram_max_mb": _gpu_vram_mb(),
        }
