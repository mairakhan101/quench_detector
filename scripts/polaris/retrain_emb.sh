#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N retrain_emb
#PBS -j oe

# Retrain Transformer + SSM with sinusoidal positional embeddings.
# Runs ONE conductor per submission (debug queue, 1hr, 1 node).
#
# Set CONDUCTOR env var before submit:
#   CONDUCTOR=bi2223_nostab qsub scripts/polaris/retrain_emb.sh
#   CONDUCTOR=bi2223_agstab qsub scripts/polaris/retrain_emb.sh
#   CONDUCTOR=rebco_nostab  qsub scripts/polaris/retrain_emb.sh
#   CONDUCTOR=rebco_stab    qsub scripts/polaris/retrain_emb.sh
#   CONDUCTOR=rebco_fd1d    qsub scripts/polaris/retrain_emb.sh
#
# Or submit all at once:
#   for c in bi2223_nostab bi2223_agstab rebco_nostab rebco_stab rebco_fd1d; do
#     CONDUCTOR=$c qsub scripts/polaris/retrain_emb.sh
#   done

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
OUTDIR=/eagle/fanc_ldrd/magnets_fem/retrain_sinusoidal

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

MODEL_KWARGS='{"pos_emb": "sinusoidal"}'
EPOCHS=400
SEED=42
COND="${CONDUCTOR:-rebco_nostab}"

echo "=== Retrain $COND with sinusoidal embeddings ==="
echo "Date: $(date)"

case "$COND" in
  bi2223_nostab)
    $PYTHON scripts/train_eval_bi2223_nostab.py \
        --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
        --out-dir "$OUTDIR" --archs transformer ssm \
        --model-kwargs "$MODEL_KWARGS"
    ;;
  bi2223_agstab)
    $PYTHON scripts/train_eval_bi2223_agstab.py \
        --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
        --out-dir "$OUTDIR" --archs transformer ssm \
        --model-kwargs "$MODEL_KWARGS"
    ;;
  rebco_nostab)
    $PYTHON scripts/train_eval_rebco_nostab.py \
        --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
        --out-dir "$OUTDIR" --archs transformer ssm \
        --model-kwargs "$MODEL_KWARGS"
    ;;
  rebco_stab)
    $PYTHON scripts/train_eval_rebco_stab.py \
        --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
        --out-dir "$OUTDIR" --archs transformer ssm \
        --model-kwargs "$MODEL_KWARGS"
    ;;
  rebco_fd1d)
    $PYTHON scripts/train_eval_rebco_fd1d.py \
        --n-samples 2000 --n-episodes 1000 --seq-epochs $EPOCHS \
        --nx 100 --backend jax --seed $SEED \
        --out-dir "$OUTDIR" --archs transformer ssm \
        --model-kwargs "$MODEL_KWARGS"
    ;;
  *)
    echo "Unknown conductor: $COND"
    exit 1
    ;;
esac

echo "=== $COND retrain done: $(date) ==="
