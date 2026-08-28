#!/usr/bin/env python
"""
benchmark_all_envs.py

Run all 5 conductor benchmarks sequentially and consolidate the
cross_conductor_benchmark.csv.  Each benchmark is run as a subprocess
so failures in one don't block the others.

Usage:
    # Quick smoke test
    python scripts/benchmark_all_envs.py --scale quick

    # Production run
    python scripts/benchmark_all_envs.py --scale production
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

BENCHMARKS = [
    {
        "name": "Bi-2223 no-stab",
        "script": "train_eval_bi2223_nostab.py",
        "args_quick": ["--n-samples", "200", "--n-eval-points", "10", "--seq-epochs", "10"],
        "args_prod": ["--n-samples", "2000", "--n-eval-points", "30", "--seq-epochs", "200"],
    },
    {
        "name": "Bi-2223 Ag-stab",
        "script": "train_eval_bi2223_agstab.py",
        "args_quick": ["--n-samples", "200", "--seq-episodes", "30", "--seq-epochs", "10"],
        "args_prod": ["--n-samples", "2000", "--seq-episodes", "500", "--seq-epochs", "200"],
    },
    {
        "name": "REBCO no-redist",
        "script": "train_eval_rebco_nostab.py",
        "args_quick": ["--n-samples", "200", "--n-episodes", "30", "--seq-epochs", "10"],
        "args_prod": ["--n-samples", "2000", "--n-episodes", "500", "--seq-epochs", "200"],
    },
    {
        "name": "REBCO Cu/SS redist",
        "script": "train_eval_rebco_stab.py",
        "args_quick": ["--n-samples", "200", "--n-episodes", "30", "--seq-epochs", "10"],
        "args_prod": ["--n-samples", "2000", "--n-episodes", "500", "--seq-epochs", "200"],
    },
    {
        "name": "REBCO 1D-FD",
        "script": "train_eval_rebco_fd1d.py",
        "args_quick": ["--n-samples", "50", "--n-episodes", "10", "--seq-epochs", "5",
                        "--nx", "50", "--backend", "jax"],
        "args_prod": ["--n-samples", "2000", "--n-episodes", "500", "--seq-epochs", "200",
                       "--nx", "100", "--backend", "jax"],
    },
]


def run_benchmark(bench: dict, scale: str, out_dir: str, seed: int,
                   adaptive: bool = False, blend: float = 0.5):
    script = SCRIPTS_DIR / bench["script"]
    args = bench[f"args_{scale}"]
    cmd = [
        sys.executable, str(script),
        *args,
        "--seed", str(seed),
        "--out-dir", out_dir,
    ]
    if adaptive:
        cmd.extend(["--adaptive", "--blend", str(blend)])

    print(f"\n{'='*60}")
    print(f"  {bench['name']}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    env = dict(__import__("os").environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SCRIPTS_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{bench['name']}] {status} in {elapsed/60:.1f} min\n")
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="quick", choices=["quick", "production"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--only", type=str, default=None,
                    help="Run only benchmarks whose name contains this substring")
    ap.add_argument("--adaptive", action="store_true",
                    help="Pass --adaptive to each benchmark script")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Pass --blend to each benchmark script")
    args = ap.parse_args()

    results = {}
    for bench in BENCHMARKS:
        if args.only and args.only.lower() not in bench["name"].lower():
            print(f"Skipping {bench['name']} (--only filter)")
            continue
        ok = run_benchmark(bench, args.scale, args.out_dir, args.seed,
                           adaptive=args.adaptive, blend=args.blend)
        results[bench["name"]] = ok

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()

    n_fail = sum(1 for v in results.values() if not v)
    if n_fail:
        print(f"{n_fail} benchmark(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
