#!/bin/bash
# Run FD1d + sinusoidal retrain sequentially in tmux.
#
# Usage:
#   tmux new -s train
#   bash scripts/polaris/run_all_tmux.sh 2>&1 | tee run_all.log
#
# Detach: Ctrl-b d    Reattach: tmux attach -t train

set -euo pipefail

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export JAX_PLATFORMS=cpu

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
OUTDIR=/eagle/fanc_ldrd/magnets_fem/retrain_sinusoidal
FD1D_OUT=/eagle/fanc_ldrd/magnets_fem/fd1d_prod

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$OUTDIR" "$FD1D_OUT"

MODEL_KWARGS='{"pos_emb": "sinusoidal"}'
EPOCHS=400
SEED=42

echo "=========================================="
echo "  Started: $(date)"
echo "=========================================="

# ── 1. FD1d production ──
echo ""
echo "=== [1/6] rebco_fd1d production ==="
$PYTHON scripts/train_eval_rebco_fd1d.py \
    --n-samples 2000 --n-episodes 1000 --seq-epochs $EPOCHS \
    --nx 100 --backend jax --seed $SEED --out-dir "$FD1D_OUT"

# ── 2-6. Retrain Transformer + SSM with sinusoidal embeddings ──
echo ""
echo "=== [2/6] bi2223_nostab ==="
$PYTHON scripts/train_eval_bi2223_nostab.py \
    --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
    --out-dir "$OUTDIR" --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=== [3/6] bi2223_agstab ==="
$PYTHON scripts/train_eval_bi2223_agstab.py \
    --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
    --out-dir "$OUTDIR" --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=== [4/6] rebco_nostab ==="
$PYTHON scripts/train_eval_rebco_nostab.py \
    --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
    --out-dir "$OUTDIR" --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=== [5/6] rebco_stab ==="
$PYTHON scripts/train_eval_rebco_stab.py \
    --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
    --out-dir "$OUTDIR" --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=== [6/6] rebco_fd1d sinusoidal ==="
$PYTHON scripts/train_eval_rebco_fd1d.py \
    --n-samples 2000 --n-episodes 1000 --seq-epochs $EPOCHS \
    --nx 100 --backend jax --seed $SEED \
    --out-dir "$OUTDIR" --archs transformer ssm \
    --model-kwargs "$MODEL_KWARGS"

echo ""
echo "=========================================="
echo "  All done: $(date)"
echo "=========================================="
