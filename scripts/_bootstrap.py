"""
_bootstrap.py

Import FIRST in every script (before numpy/scipy/sklearn) to:
  1. cap BLAS/OpenMP thread pools — this is a shared host where the default
     64-thread OpenBLAS pool hits RLIMIT_NPROC and crashes on import;
  2. put the repo root on sys.path so `import hts_tape_env` works when a script
     is run directly (python scripts/foo.py).
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
