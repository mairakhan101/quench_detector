#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N fd1d_prod
#PBS -j oe

# FD1d production training on debug queue (1hr, 1 node).
# Reduced episodes to fit in debug walltime; resubmit for full run.
#
# Submit:  qsub scripts/polaris/fd1d_prod.sh

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
OUTDIR=/eagle/fanc_ldrd/magnets_fem/fd1d_prod

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

echo "=== FD1d production run on $(hostname) ==="
echo "Date: $(date)"

$PYTHON scripts/train_eval_rebco_fd1d.py \
    --n-samples 2000 \
    --n-episodes 500 \
    --seq-epochs 200 \
    --nx 100 \
    --backend jax \
    --seed 42 \
    --out-dir "$OUTDIR"

echo "=== FD1d production done: $(date) ==="
