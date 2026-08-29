#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q prod
#PBS -l select=2:system=polaris
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -N fd1d_prod
#PBS -j oe

# Production FD1d training: 2000 samples, 500 episodes, 400 epochs, nx=100, JAX backend.
# Single node — FD solver is CPU-bound, no pbsdsh needed.
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
    --n-episodes 1000 \
    --seq-epochs 400 \
    --nx 100 \
    --backend jax \
    --seed 42 \
    --out-dir "$OUTDIR"

echo "=== FD1d production done: $(date) ==="
