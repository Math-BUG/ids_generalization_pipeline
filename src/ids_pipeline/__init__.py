"""IoT/NIDS generalization experiment pipeline.

The cluster nodes may expose many CPU cores. If OpenBLAS sees all of them it can
try to allocate too much thread metadata before numpy/scipy work starts. These
defaults keep direct CLI runs conservative while still allowing Slurm scripts or
users to override the values explicitly before launching Python.
"""

from __future__ import annotations

import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "8")

__version__ = "0.1.0"
