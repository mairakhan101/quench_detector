#!/bin/bash
#PBS -A fanc_ldrd
#PBS -q prod
#PBS -l select=1:system=polaris
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -N retrain_emb
#PBS -j oe

# Retrain Transformer + SSM with sinusoidal positional embeddings across
# all 5 conductors. 400 epochs, doubled episodes where applicable.
# Single node, sequential — each conductor takes ~20-30 min for transformer+ssm.
#
# Submit:  qsub scripts/polaris/retrain_emb.sh

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

echo "=== Retrain Transformer + SSM with sinusoidal embeddings ==="
echo "Date: $(date)"
echo "Model kwargs: $MODEL_KWARGS"

# Bi-2223 no-stab (--n-samples 8000 for more training data)
echo ""
echo "=== bi2223_nostab ==="
$PYTHON scripts/train_eval_bi2223_nostab.py \
    --n-samples 8000 \
    --seq-epochs $EPOCHS --seed $SEED --out-dir "$OUTDIR" \
    --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

# Bi-2223 Ag-stab
echo ""
echo "=== bi2223_agstab ==="
$PYTHON scripts/train_eval_bi2223_agstab.py \
    --n-samples 8000 \
    --seq-epochs $EPOCHS --seed $SEED --out-dir "$OUTDIR" \
    --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

# REBCO no-stab (default is 1500 episodes, bump to 3000)
echo ""
echo "=== rebco_nostab ==="
$PYTHON scripts/train_eval_rebco_nostab.py \
    --n-episodes 3000 \
    --seq-epochs $EPOCHS --seed $SEED --out-dir "$OUTDIR" \
    --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

# REBCO stab
echo ""
echo "=== rebco_stab ==="
$PYTHON scripts/train_eval_rebco_stab.py \
    --n-episodes 3000 \
    --seq-epochs $EPOCHS --seed $SEED --out-dir "$OUTDIR" \
    --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

# REBCO FD1d (bump from 500 to 1000 episodes)
echo ""
echo "=== rebco_fd1d ==="
$PYTHON scripts/train_eval_rebco_fd1d.py \
    --n-samples 2000 \
    --n-episodes 1000 \
    --seq-epochs $EPOCHS --nx 100 --backend jax --seed $SEED --out-dir "$OUTDIR" \
    --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=== All retraining complete: $(date) ==="
