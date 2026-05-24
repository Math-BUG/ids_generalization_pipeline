"""Backend selection helpers.

CPU uses pandas + scikit-learn. GPU uses RAPIDS cuML/CuPy when available.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def requested_backend(config_backend: str) -> str:
    return os.environ.get("IDS_COMPUTE_BACKEND", config_backend).lower()


def resolve_backend(config_backend: str) -> str:
    backend = requested_backend(config_backend)
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("compute_backend must be one of: cpu, gpu, auto")
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        require_gpu_backend()
        return "gpu"
    if gpu_backend_available():
        LOGGER.info("RAPIDS cuML/CuPy detected. Using GPU backend.")
        return "gpu"
    LOGGER.info("RAPIDS cuML/CuPy not detected. Falling back to CPU backend.")
    return "cpu"


def gpu_backend_available() -> bool:
    try:
        import cupy  # noqa: F401
        import cuml  # noqa: F401
    except Exception:
        return False
    return True


def require_gpu_backend() -> None:
    try:
        import cupy  # noqa: F401
        import cuml  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "GPU backend requested, but RAPIDS cuML/CuPy is not importable. "
            "Install a RAPIDS build compatible with the cluster CUDA version, "
            "or set compute_backend: cpu/auto."
        ) from exc


def to_gpu_array(X: Any):
    import cupy as cp

    if hasattr(X, "toarray"):
        X = X.toarray()
    return cp.asarray(X, dtype=cp.float32)


def to_numpy_array(X: Any) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    try:
        import cupy as cp

        if isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
    except Exception:
        pass
    if hasattr(X, "get"):
        return X.get()
    return np.asarray(X)
