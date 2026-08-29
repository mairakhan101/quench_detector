#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N fd1d_debug
#PBS -j oe

# Quick FD1d smoke test on debug queue.
#
# Submit:  qsub scripts/polaris/fd1d_debug.sh

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
OUTDIR=/eagle/fanc_ldrd/magnets_fem/fd1d_debug

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

echo "=== FD1d debug run on $(hostname) ==="
echo "Date: $(date)"

$PYTHON scripts/train_eval_rebco_fd1d.py \
    --n-samples 200 \
    --n-episodes 50 \
    --seq-epochs 20 \
    --nx 50 \
    --backend jax \
    --seed 42 \
    --out-dir "$OUTDIR"

echo "=== FD1d debug done: $(date) ==="
