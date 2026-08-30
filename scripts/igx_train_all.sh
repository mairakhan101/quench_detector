#!/bin/bash
# ============================================================================
# igx_train_all.sh — Run all training jobs on IGX (GPU-accelerated)
#
# This script runs three training campaigns on the 4 lumped-model conductors:
#   1. Sinusoidal embedding retrain (Transformer + SSM, 400 epochs)
#   2. Cross-family transfer learning (all 12 pairs + 4 baselines, production)
#   3. Controllers (supervised + PPO, for all 4 conductors)
#
# Usage:
#   # Run everything:
#   bash scripts/igx_train_all.sh
#
#   # Run only one stage:
#   bash scripts/igx_train_all.sh retrain
#   bash scripts/igx_train_all.sh transfer
#   bash scripts/igx_train_all.sh controllers
#
# Output goes to models/igx_<campaign>/
# ============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"
STAGE="${1:-all}"
SEED=42

CONDUCTORS=(rebco_nostab rebco_stab bi2223_nostab bi2223_agstab)

echo "================================================================"
echo "  IGX Training Pipeline"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  Python:  $($PYTHON --version 2>&1)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo "  Stage:   $STAGE"
echo "================================================================"

# ── 1. Sinusoidal Embedding Retrain ────────────────────────────────────────
run_retrain() {
    echo ""
    echo "================================================================"
    echo "  STAGE 1: Sinusoidal Embedding Retrain (Transformer + SSM)"
    echo "================================================================"

    OUTDIR="models/igx_retrain_sinusoidal"
    mkdir -p "$OUTDIR"

    MODEL_KWARGS='{"pos_emb": "sinusoidal"}'
    EPOCHS=400

    for COND in "${CONDUCTORS[@]}"; do
        echo ""
        echo "--- $COND (Transformer + SSM, $EPOCHS epochs) ---"
        t0=$SECONDS

        case "$COND" in
            bi2223_nostab)
                $PYTHON scripts/train_eval_bi2223_nostab.py \
                    --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
                    --out-dir "$OUTDIR" --archs transformer ssm \
                    --model-kwargs "$MODEL_KWARGS" \
                    2>&1 | tee "$OUTDIR/${COND}.log"
                ;;
            bi2223_agstab)
                $PYTHON scripts/train_eval_bi2223_agstab.py \
                    --n-samples 8000 --seq-epochs $EPOCHS --seed $SEED \
                    --out-dir "$OUTDIR" --archs transformer ssm \
                    --model-kwargs "$MODEL_KWARGS" \
                    2>&1 | tee "$OUTDIR/${COND}.log"
                ;;
            rebco_nostab)
                $PYTHON scripts/train_eval_rebco_nostab.py \
                    --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
                    --out-dir "$OUTDIR" --archs transformer ssm \
                    --model-kwargs "$MODEL_KWARGS" \
                    2>&1 | tee "$OUTDIR/${COND}.log"
                ;;
            rebco_stab)
                $PYTHON scripts/train_eval_rebco_stab.py \
                    --n-episodes 3000 --seq-epochs $EPOCHS --seed $SEED \
                    --out-dir "$OUTDIR" --archs transformer ssm \
                    --model-kwargs "$MODEL_KWARGS" \
                    2>&1 | tee "$OUTDIR/${COND}.log"
                ;;
        esac

        elapsed=$(( SECONDS - t0 ))
        echo "  $COND done in $((elapsed / 60))m $((elapsed % 60))s"
    done
}

# ── 2. Cross-Family Transfer Learning ─────────────────────────────────────
run_transfer() {
    echo ""
    echo "================================================================"
    echo "  STAGE 2: Cross-Family Transfer Learning (Production)"
    echo "================================================================"

    OUTDIR="models/igx_transfer_learning"
    mkdir -p "$OUTDIR"

    # Full cross-family: all 12 directional pairs + 4 baselines
    # Includes Bi-2223 ↔ REBCO (the new results) plus within-family
    $PYTHON scripts/transfer_learning.py \
        --scale production \
        --archs gru transformer ssm \
        --pairs \
            rebco_nostab:bi2223_nostab \
            bi2223_nostab:rebco_nostab \
            rebco_nostab:bi2223_agstab \
            bi2223_agstab:rebco_nostab \
            rebco_stab:bi2223_nostab \
            bi2223_nostab:rebco_stab \
            rebco_stab:bi2223_agstab \
            bi2223_agstab:rebco_stab \
            rebco_nostab:rebco_stab \
            rebco_stab:rebco_nostab \
            bi2223_nostab:bi2223_agstab \
            bi2223_agstab:bi2223_nostab \
        --seed $SEED \
        --out-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/transfer.log"
}

# ── 3. Controller Training ────────────────────────────────────────────────
run_controllers() {
    echo ""
    echo "================================================================"
    echo "  STAGE 3: Controller Training (Supervised + PPO)"
    echo "================================================================"

    OUTDIR="models/igx_controllers"
    mkdir -p "$OUTDIR"

    # Use the best GRU checkpoint for each conductor.
    # Prefer adaptive if available, else uniform production.
    for COND in "${CONDUCTORS[@]}"; do
        echo ""
        echo "--- $COND controllers ---"

        # Find the best GRU surrogate checkpoint
        SURR=""
        for d in models/igx_retrain_sinusoidal \
                 magnets_fem/benchmark_prod_adaptive \
                 models/benchmark_prod_uniform \
                 magnets_fem/benchmark_prod_uniform; do
            if [[ -f "$d/${COND}_seq_gru.pt" ]]; then
                SURR="$d/${COND}_seq_gru.pt"
                break
            fi
        done

        if [[ -z "$SURR" ]]; then
            echo "  WARNING: No GRU checkpoint found for $COND, skipping controllers"
            continue
        fi
        echo "  Using surrogate: $SURR"

        # Supervised controller
        echo ""
        echo "  --- Supervised controller ---"
        $PYTHON scripts/train_supervised_controller.py \
            --surrogate-checkpoint "$SURR" \
            --conductor "$COND" \
            --n-episodes 3000 \
            --epochs 200 \
            --hidden 64 \
            --eval-episodes 500 \
            --seed $SEED \
            --out-dir "$OUTDIR/$COND" \
            2>&1 | tee "$OUTDIR/${COND}_supervised.log"

        # RL controller (PPO)
        echo ""
        echo "  --- RL controller (PPO) ---"
        $PYTHON scripts/train_rl_controller.py \
            --surrogate-checkpoint "$SURR" \
            --conductor "$COND" \
            --total-timesteps 200000 \
            --n-steps 2048 \
            --hidden 64 \
            --eval-episodes 500 \
            --seed $SEED \
            --out-dir "$OUTDIR/$COND" \
            2>&1 | tee "$OUTDIR/${COND}_rl.log"
    done
}

# ── Dispatch ──────────────────────────────────────────────────────────────
case "$STAGE" in
    retrain)     run_retrain ;;
    transfer)    run_transfer ;;
    controllers) run_controllers ;;
    all)
        run_retrain
        run_transfer
        run_controllers
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [all|retrain|transfer|controllers]"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  All done: $(date)"
echo "================================================================"
