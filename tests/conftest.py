"""
conftest.py

Caps BLAS/OpenMP thread pools before numpy/scipy/torch are imported anywhere
in the test session -- this is a shared host where the default thread pool
hits RLIMIT_NPROC and crashes (see scripts/_bootstrap.py, same fix).
Also puts the repo root on sys.path so `import hts_tape_env` works when
pytest is invoked from any directory.
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
