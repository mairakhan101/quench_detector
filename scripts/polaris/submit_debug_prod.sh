#!/bin/bash
# Submit production work as debug-queue chunks on Polaris.
# Each chunk runs on 1 node × 1 hr with full production parameters.
#
# Usage:
#   bash scripts/polaris/submit_debug_prod.sh --benchmark       # 10 chunks
#   bash scripts/polaris/submit_debug_prod.sh --noise           # 30 uniform chunks
#   bash scripts/polaris/submit_debug_prod.sh --noise-adaptive  # 30 adaptive chunks
#   bash scripts/polaris/submit_debug_prod.sh --transfer        # 10 chunks
#   bash scripts/polaris/submit_debug_prod.sh --all             # everything (80 chunks)
#   bash scripts/polaris/submit_debug_prod.sh --priority        # highest-value subset
#
# Options:
#   --dry-run    Show qsub commands without submitting
#   --range N-M  Only submit task IDs N through M (inclusive)
#
# Note: debug queue runs 1 job at a time, so chunks execute sequentially.
# Each takes ~30-60 min. Plan accordingly — "--priority" gives you the
# most value in a few hours of debug queue time.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false
RANGE_LO=""
RANGE_HI=""
JOB_TYPES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)      JOB_TYPES+=("benchmark"); shift ;;
        --noise)          JOB_TYPES+=("noise"); shift ;;
        --noise-adaptive) JOB_TYPES+=("noise_adaptive"); shift ;;
        --transfer)       JOB_TYPES+=("transfer"); shift ;;
        --all)            JOB_TYPES=("benchmark" "noise" "noise_adaptive" "transfer"); shift ;;
        --priority)       JOB_TYPES+=("priority"); shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --range)          IFS='-' read -r RANGE_LO RANGE_HI <<< "$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ${#JOB_TYPES[@]} -eq 0 ]; then
    echo "Usage: $0 {--benchmark|--noise|--noise-adaptive|--transfer|--all|--priority} [--dry-run] [--range N-M]"
    echo ""
    echo "Job types and chunk counts:"
    echo "  --benchmark       10 chunks (5 conductors × 2 sampling modes)"
    echo "  --noise           30 chunks (5 conductors × 6 noise levels, uniform)"
    echo "  --noise-adaptive  30 chunks (same, adaptive sampling)"
    echo "  --transfer        10 chunks (10 cross-conductor pairs)"
    echo "  --all             80 chunks total"
    echo "  --priority        ~12 highest-value chunks across job types"
    echo ""
    echo "Debug queue: 1 node, 1 hr, 1 running job at a time."
    echo "Each chunk ≈ 30-60 min. Plan: 'priority' ≈ 8-12 hr of debug time."
    exit 1
fi

submit() {
    local script="$1" task_id="$2" extra_vars="${3:-}"
    local vars="TASK_ID=$task_id"
    [ -n "$extra_vars" ] && vars="$vars,$extra_vars"

    if $DRY_RUN; then
        echo "  [dry-run] qsub -v \"$vars\" $script"
    else
        local jobid
        jobid=$(qsub -v "$vars" "$script")
        echo "  $jobid  (TASK_ID=$task_id)"
    fi
}

in_range() {
    local id=$1
    if [ -n "$RANGE_LO" ] && [ "$id" -lt "$RANGE_LO" ]; then return 1; fi
    if [ -n "$RANGE_HI" ] && [ "$id" -gt "$RANGE_HI" ]; then return 1; fi
    return 0
}

N_SUBMITTED=0

for JOB_TYPE in "${JOB_TYPES[@]}"; do
    case "$JOB_TYPE" in

    benchmark)
        echo ""
        echo "=== Benchmark (production params, debug queue) ==="
        echo "10 chunks: 5 conductors × {uniform, adaptive}"
        CONDUCTORS=("Bi-2223 no-stab" "Bi-2223 Ag-stab" "REBCO no-redist" "REBCO Cu/SS redist" "REBCO 1D-FD")
        for i in $(seq 0 9); do
            in_range "$i" || continue
            cond_idx=$((i % 5))
            mode="uniform"; [ "$i" -ge 5 ] && mode="adaptive"
            echo "  chunk $i: ${CONDUCTORS[$cond_idx]} ($mode)"
            submit "$SCRIPTS_DIR/run_benchmark_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
        ;;

    noise)
        echo ""
        echo "=== Noise benchmark UNIFORM (production params, debug queue) ==="
        echo "30 chunks: 5 conductors × 6 noise levels × 4 archs"
        CONDUCTORS=(bi2223_nostab bi2223_agstab rebco_nostab rebco_stab rebco_fd1d)
        NOISES=(noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)
        for i in $(seq 0 29); do
            in_range "$i" || continue
            ci=$((i / 6)); ni=$((i % 6))
            echo "  chunk $i: ${CONDUCTORS[$ci]} / ${NOISES[$ni]}"
            submit "$SCRIPTS_DIR/run_noise_bench_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
        ;;

    noise_adaptive)
        echo ""
        echo "=== Noise benchmark ADAPTIVE (production params, debug queue) ==="
        echo "30 chunks: 5 conductors × 6 noise levels × 4 archs"
        CONDUCTORS=(bi2223_nostab bi2223_agstab rebco_nostab rebco_stab rebco_fd1d)
        NOISES=(noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)
        for i in $(seq 0 29); do
            in_range "$i" || continue
            ci=$((i / 6)); ni=$((i % 6))
            echo "  chunk $i: ${CONDUCTORS[$ci]} / ${NOISES[$ni]} (adaptive)"
            submit "$SCRIPTS_DIR/run_noise_bench_debug_chunk.pbs" "$i" "NOISE_BENCH_MODE=adaptive"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
        ;;

    transfer)
        echo ""
        echo "=== Transfer learning (production params, debug queue) ==="
        echo "10 chunks: 10 cross-conductor pairs"
        PAIRS=(
            "rebco_nostab→bi2223_nostab"
            "bi2223_nostab→rebco_nostab"
            "rebco_nostab→rebco_stab"
            "rebco_stab→rebco_nostab"
            "bi2223_nostab→rebco_stab"
            "rebco_stab→bi2223_nostab"
            "rebco_nostab→rebco_nostab (baseline)"
            "rebco_stab→rebco_stab (baseline)"
            "bi2223_nostab→bi2223_nostab (baseline)"
            "bi2223_agstab→bi2223_agstab (baseline)"
        )
        for i in $(seq 0 9); do
            in_range "$i" || continue
            echo "  chunk $i: ${PAIRS[$i]}"
            submit "$SCRIPTS_DIR/run_transfer_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
        ;;

    priority)
        echo ""
        echo "=== Priority subset (highest-value chunks) ==="
        echo "Targets: benchmark noiseless baselines + key noise combos + top transfer pairs"
        echo ""

        echo "--- Benchmark: all 5 conductors, uniform (most important) ---"
        CONDUCTORS=("Bi-2223 no-stab" "Bi-2223 Ag-stab" "REBCO no-redist" "REBCO Cu/SS redist" "REBCO 1D-FD")
        for i in $(seq 0 4); do
            echo "  bench $i: ${CONDUCTORS[$i]} (uniform)"
            submit "$SCRIPTS_DIR/run_benchmark_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done

        echo ""
        echo "--- Noise: noiseless + realistic for rebco_nostab & bi2223_nostab ---"
        # rebco_nostab noiseless=12, realistic=17; bi2223_nostab noiseless=0, realistic=5
        NOISE_CHUNKS=(0 5 12 17)
        NOISE_LABELS=("bi2223_nostab/noiseless" "bi2223_nostab/realistic" "rebco_nostab/noiseless" "rebco_nostab/realistic")
        for idx in "${!NOISE_CHUNKS[@]}"; do
            i=${NOISE_CHUNKS[$idx]}
            echo "  noise $i: ${NOISE_LABELS[$idx]}"
            submit "$SCRIPTS_DIR/run_noise_bench_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done

        echo ""
        echo "--- Transfer: top 3 cross-conductor pairs ---"
        XFER_CHUNKS=(0 1 2)
        XFER_LABELS=("rebco_nostab→bi2223_nostab" "bi2223_nostab→rebco_nostab" "rebco_nostab→rebco_stab")
        for idx in "${!XFER_CHUNKS[@]}"; do
            i=${XFER_CHUNKS[$idx]}
            echo "  xfer $i: ${XFER_LABELS[$idx]}"
            submit "$SCRIPTS_DIR/run_transfer_debug_chunk.pbs" "$i"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
        ;;

    esac
done

echo ""
echo "=== Submitted $N_SUBMITTED debug-queue chunks ==="
echo "Each: 1 node × 1 hr. Debug queue runs 1 at a time."
echo "Estimated wall time: ~${N_SUBMITTED} hr (sequential)."
echo ""
echo "Monitor:  qstat -u \$USER"
echo "Results:  /eagle/fanc_ldrd/magnets_fem/"
