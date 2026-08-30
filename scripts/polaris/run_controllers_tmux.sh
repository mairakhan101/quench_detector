#!/bin/bash
# Run distillation + controller training sequentially in tmux.
#
# Primary metric: integrated current before degradation (∫I·dt [A·s])
# — total useful transport current delivered before thermal runaway.
#
# Usage:
#   tmux new -s ctrl
#   bash scripts/polaris/run_controllers_tmux.sh 2>&1 | tee run_controllers.log
#
# Detach: Ctrl-b d    Reattach: tmux attach -t ctrl

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export DISABLE_JAX=1

REPO=$HOME/fanc/magnets_rl/quench_detector
PYTHON=$HOME/.conda/envs/quench_surr/bin/python
DISTILL_OUT=/eagle/fanc_ldrd/magnets_fem/distill
CTRL_OUT=/eagle/fanc_ldrd/magnets_fem/controllers
SURR_DIR=/eagle/fanc_ldrd/magnets_fem/retrain_sinusoidal

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$DISTILL_OUT" "$CTRL_OUT"

EPOCHS=400
SEED=42

echo "=========================================="
echo "  Distillation + Controller Training"
echo "  Started: $(date)"
echo "  Metric: integrated current ∫I·dt [A·s]"
echo "=========================================="

# ── 1-2. Distillation (REBCO only — need lambda in rollout) ──

for COND in rebco_nostab rebco_stab; do
    echo ""
    echo "=== Distillation: $COND ==="
    $PYTHON scripts/train_distill.py \
        --conductor "$COND" \
        --teacher-arch gru --student-arch gru \
        --teacher-epochs $EPOCHS --student-epochs $EPOCHS \
        --n-episodes 3000 \
        --alpha 0.5 --temperature 2.0 \
        --seed $SEED \
        --out-dir "$DISTILL_OUT/$COND"
done

# ── 3-4. Supervised + RL controllers ──

for COND in rebco_nostab rebco_stab; do
    SURR="$SURR_DIR/${COND}_seq_gru.pt"
    if [ ! -f "$SURR" ]; then
        echo "WARNING: surrogate not found at $SURR — skipping $COND controllers"
        continue
    fi

    echo ""
    echo "=== Supervised controller: $COND ==="
    $PYTHON scripts/train_supervised_controller.py \
        --surrogate-checkpoint "$SURR" \
        --conductor "$COND" \
        --n-episodes 3000 \
        --epochs 200 \
        --hidden 64 \
        --eval-episodes 500 \
        --seed $SEED \
        --out-dir "$CTRL_OUT/$COND"

    echo ""
    echo "=== RL controller (PPO): $COND ==="
    $PYTHON scripts/train_rl_controller.py \
        --surrogate-checkpoint "$SURR" \
        --conductor "$COND" \
        --total-timesteps 200000 \
        --n-steps 2048 \
        --hidden 64 \
        --eval-episodes 500 \
        --seed $SEED \
        --out-dir "$CTRL_OUT/$COND"
done

echo ""
echo "=========================================="
echo "  All done: $(date)"
echo "=========================================="
