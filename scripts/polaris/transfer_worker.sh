#!/bin/bash
# Worker script invoked by pbsdsh — one instance per node.
# Reads PBS_NODENUM to determine which (source, target) pair to run.
#
# Args passed via environment:
#   TRANSFER_OUT   — output directory

set -euo pipefail

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
SEED=42
OUT="${TRANSFER_OUT:-/eagle/fanc_ldrd/magnets_fem/transfer_learning}"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# 10 pairs: 6 cross-conductor + 4 baselines
PAIRS=(
    "rebco_nostab:bi2223_nostab"
    "bi2223_nostab:rebco_nostab"
    "rebco_nostab:rebco_stab"
    "rebco_stab:rebco_nostab"
    "bi2223_nostab:rebco_stab"
    "rebco_stab:bi2223_nostab"
    "rebco_nostab:rebco_nostab"
    "rebco_stab:rebco_stab"
    "bi2223_nostab:bi2223_nostab"
    "bi2223_agstab:bi2223_agstab"
)

RANK=${PBS_NODENUM:-0}
PAIR=${PAIRS[$RANK]}

echo "=== Node $RANK: transfer $PAIR on $(hostname) ==="
echo "Date: $(date)"
mkdir -p "$OUT"

$PYTHON scripts/transfer_learning.py \
    --scale production \
    --pairs "$PAIR" \
    --archs gru transformer ssm diffusion \
    --seed $SEED \
    --out-dir "$OUT"

echo "=== Node $RANK done ($PAIR): $(date) ==="
