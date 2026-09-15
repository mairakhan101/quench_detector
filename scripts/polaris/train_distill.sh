#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N train_distill
#PBS -j oe

# Teacher-student distillation for one conductor.
# Teacher sees 5 features (4 channels + lambda), student sees 4.
#
# Set CONDUCTOR before submit:
#   CONDUCTOR=rebco_nostab qsub scripts/polaris/train_distill.sh
#   CONDUCTOR=rebco_stab   qsub scripts/polaris/train_distill.sh
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
OUTDIR=/eagle/fanc_ldrd/magnets_fem/distill

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

COND="${CONDUCTOR:-rebco_nostab}"
EPOCHS=400
SEED=42

echo "=== Distillation: $COND ==="
echo "  Date: $(date)"

$PYTHON scripts/train_distill.py \
    --conductor "$COND" \
    --teacher-arch gru --student-arch gru \
    --teacher-epochs $EPOCHS --student-epochs $EPOCHS \
    --n-episodes 3000 \
    --alpha 0.5 --temperature 2.0 \
    --seed $SEED \
    --out-dir "$OUTDIR/$COND"

echo ""
echo "=== $COND distillation done: $(date) ==="
