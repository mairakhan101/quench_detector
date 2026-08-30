#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N train_controller
#PBS -j oe

# Train supervised + RL current controllers for one conductor.
#
# Set CONDUCTOR and SURROGATE env vars before submit:
#   CONDUCTOR=rebco_nostab SURROGATE=/eagle/fanc_ldrd/magnets_fem/retrain_sinusoidal/rebco_nostab_seq_gru.pt \
#     qsub scripts/polaris/train_controller.sh
#
# Or run interactively via tmux (see run_controllers_tmux.sh).

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export DISABLE_JAX=1

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
OUTDIR=/eagle/fanc_ldrd/magnets_fem/controllers

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

COND="${CONDUCTOR:-rebco_nostab}"
SURR="${SURROGATE:-/eagle/fanc_ldrd/magnets_fem/retrain_sinusoidal/${COND}_seq_gru.pt}"

echo "=== Controller training: $COND ==="
echo "  Surrogate: $SURR"
echo "  Date: $(date)"

# ── 1. Supervised controller ──
echo ""
echo "--- Supervised controller ---"
$PYTHON scripts/train_supervised_controller.py \
    --surrogate-checkpoint "$SURR" \
    --conductor "$COND" \
    --n-episodes 3000 \
    --epochs 200 \
    --hidden 64 \
    --eval-episodes 500 \
    --seed 42 \
    --out-dir "$OUTDIR/$COND"

# ── 2. RL controller (PPO) ──
echo ""
echo "--- RL controller (PPO) ---"
$PYTHON scripts/train_rl_controller.py \
    --surrogate-checkpoint "$SURR" \
    --conductor "$COND" \
    --total-timesteps 200000 \
    --n-steps 2048 \
    --hidden 64 \
    --eval-episodes 500 \
    --seed 42 \
    --out-dir "$OUTDIR/$COND"

echo ""
echo "=== $COND controller training done: $(date) ==="
