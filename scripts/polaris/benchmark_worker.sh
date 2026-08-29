#!/bin/bash
# Worker script invoked by pbsdsh — one instance per node.
# Reads PBS_NODENUM to determine which (conductor, sampling_mode) combo to run.

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
SEED=42
OUT_UNI=/eagle/fanc_ldrd/magnets_fem/benchmark_prod_uniform
OUT_ADA=/eagle/fanc_ldrd/magnets_fem/benchmark_prod_adaptive
PYTHON=$HOME/.conda/envs/quench_surr/bin/python

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TASKS=(
  "Bi-2223 no-stab|uniform|$OUT_UNI"
  "Bi-2223 Ag-stab|uniform|$OUT_UNI"
  "REBCO no-redist|uniform|$OUT_UNI"
  "REBCO Cu/SS redist|uniform|$OUT_UNI"
  "REBCO 1D-FD|uniform|$OUT_UNI"
  "Bi-2223 no-stab|adaptive|$OUT_ADA"
  "Bi-2223 Ag-stab|adaptive|$OUT_ADA"
  "REBCO no-redist|adaptive|$OUT_ADA"
  "REBCO Cu/SS redist|adaptive|$OUT_ADA"
  "REBCO 1D-FD|adaptive|$OUT_ADA"
)

RANK=${PBS_NODENUM:-0}
IFS='|' read -r CONDUCTOR MODE OUTDIR <<< "${TASKS[$RANK]}"

echo "=== Node $RANK: $CONDUCTOR ($MODE) on $(hostname) ==="
echo "Date: $(date)"
mkdir -p "$OUTDIR"

if [ "$MODE" = "adaptive" ]; then
  $PYTHON scripts/benchmark_all_envs.py --scale production --only "$CONDUCTOR" --out-dir "$OUTDIR" --seed $SEED --adaptive --blend 0.5
else
  $PYTHON scripts/benchmark_all_envs.py --scale production --only "$CONDUCTOR" --out-dir "$OUTDIR" --seed $SEED
fi

echo "=== Node $RANK done: $(date) ==="
