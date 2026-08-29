#!/bin/bash
# Submit debug-queue chunks one at a time, waiting for each to finish
# before submitting the next. Hands-off serial execution.
#
# Usage:
#   bash scripts/polaris/submit_debug_serial.sh --priority
#   bash scripts/polaris/submit_debug_serial.sh --benchmark
#   bash scripts/polaris/submit_debug_serial.sh --noise
#   bash scripts/polaris/submit_debug_serial.sh --transfer
#   bash scripts/polaris/submit_debug_serial.sh --all
#   bash scripts/polaris/submit_debug_serial.sh --custom 'bench:0 bench:1 noise:12 xfer:0'
#
# Run in a tmux/screen session so it survives logout:
#   tmux new -s debug_jobs
#   bash scripts/polaris/submit_debug_serial.sh --priority
#   # Ctrl-b d to detach, tmux attach -t debug_jobs to reattach

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
POLL_INTERVAL=60

wait_for_job() {
    local jobid="$1"
    local short_id="${jobid%%.*}"
    echo "  Waiting for $short_id ..."
    while true; do
        state=$(qstat -f "$jobid" 2>/dev/null | grep "job_state" | awk '{print $3}' || echo "gone")
        case "$state" in
            F|gone|"") break ;;
            R) printf "\r  %s running (%s)  " "$short_id" "$(date +%H:%M)" ;;
            Q) printf "\r  %s queued (%s)   " "$short_id" "$(date +%H:%M)" ;;
        esac
        sleep "$POLL_INTERVAL"
    done
    echo ""
    echo "  $short_id finished at $(date +%H:%M:%S)"
}

submit_and_wait() {
    local script="$1" task_id="$2" label="$3" extra_vars="${4:-}"
    local vars="TASK_ID=$task_id"
    [ -n "$extra_vars" ] && vars="$vars,$extra_vars"

    echo ""
    echo "[$(date +%H:%M:%S)] Submitting: $label (TASK_ID=$task_id)"
    local jobid
    jobid=$(qsub -v "$vars" "$script")
    echo "  Job: ${jobid%%.*}"
    wait_for_job "$jobid"
}

build_queue() {
    local mode="$1"
    case "$mode" in
    priority)
        for i in $(seq 0 4); do
            CONDUCTORS=("Bi-2223 no-stab" "Bi-2223 Ag-stab" "REBCO no-redist" "REBCO Cu/SS redist" "REBCO 1D-FD")
            echo "bench:$i:${CONDUCTORS[$i]} uniform"
        done
        echo "noise:0:bi2223_nostab/noiseless"
        echo "noise:5:bi2223_nostab/realistic"
        echo "noise:12:rebco_nostab/noiseless"
        echo "noise:17:rebco_nostab/realistic"
        echo "xfer:0:rebco_nostab→bi2223_nostab"
        echo "xfer:1:bi2223_nostab→rebco_nostab"
        echo "xfer:2:rebco_nostab→rebco_stab"
        ;;
    benchmark)
        CONDUCTORS=("Bi-2223 no-stab" "Bi-2223 Ag-stab" "REBCO no-redist" "REBCO Cu/SS redist" "REBCO 1D-FD")
        for i in $(seq 0 4); do echo "bench:$i:${CONDUCTORS[$i]} uniform"; done
        for i in $(seq 5 9); do
            ci=$((i - 5))
            echo "bench:$i:${CONDUCTORS[$ci]} adaptive"
        done
        ;;
    noise)
        CONDUCTORS=(bi2223_nostab bi2223_agstab rebco_nostab rebco_stab rebco_fd1d)
        NOISES=(noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)
        for i in $(seq 0 29); do
            ci=$((i / 6)); ni=$((i % 6))
            echo "noise:$i:${CONDUCTORS[$ci]}/${NOISES[$ni]}"
        done
        ;;
    noise-adaptive)
        CONDUCTORS=(bi2223_nostab bi2223_agstab rebco_nostab rebco_stab rebco_fd1d)
        NOISES=(noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)
        for i in $(seq 0 29); do
            ci=$((i / 6)); ni=$((i % 6))
            echo "noise_ada:$i:${CONDUCTORS[$ci]}/${NOISES[$ni]} adaptive"
        done
        ;;
    transfer)
        PAIRS=("rebco_nostab→bi2223_nostab" "bi2223_nostab→rebco_nostab"
               "rebco_nostab→rebco_stab" "rebco_stab→rebco_nostab"
               "bi2223_nostab→rebco_stab" "rebco_stab→bi2223_nostab"
               "rebco_nostab→rebco_nostab" "rebco_stab→rebco_stab"
               "bi2223_nostab→bi2223_nostab" "bi2223_agstab→bi2223_agstab")
        for i in $(seq 0 9); do echo "xfer:$i:${PAIRS[$i]}"; done
        ;;
    all)
        build_queue benchmark
        build_queue noise
        build_queue noise-adaptive
        build_queue transfer
        ;;
    esac
}

parse_custom() {
    for token in $1; do
        echo "$token"
    done
}

MODE="${1:---priority}"
MODE="${MODE#--}"

if [ "$MODE" = "custom" ]; then
    QUEUE_ITEMS=$(parse_custom "$2")
else
    QUEUE_ITEMS=$(build_queue "$MODE")
fi

TOTAL=$(echo "$QUEUE_ITEMS" | wc -l)
echo "=== Debug serial runner ==="
echo "Mode: $MODE | Chunks: $TOTAL | ~${TOTAL} hr estimated"
echo "Started: $(date)"
echo "========================================="

COUNT=0
echo "$QUEUE_ITEMS" | while IFS=: read -r job_type task_id label; do
    COUNT=$((COUNT + 1))
    progress="[$COUNT/$TOTAL]"

    case "$job_type" in
        bench)
            submit_and_wait "$SCRIPTS_DIR/run_benchmark_debug_chunk.pbs" "$task_id" "$progress $label"
            ;;
        noise)
            submit_and_wait "$SCRIPTS_DIR/run_noise_bench_debug_chunk.pbs" "$task_id" "$progress $label"
            ;;
        noise_ada)
            submit_and_wait "$SCRIPTS_DIR/run_noise_bench_debug_chunk.pbs" "$task_id" "$progress $label" "NOISE_BENCH_MODE=adaptive"
            ;;
        xfer)
            submit_and_wait "$SCRIPTS_DIR/run_transfer_debug_chunk.pbs" "$task_id" "$progress $label"
            ;;
        *)
            echo "Unknown job type: $job_type (skipping)"
            ;;
    esac
done

echo ""
echo "========================================="
echo "All $TOTAL chunks finished at $(date)"
echo "Results: /eagle/fanc_ldrd/magnets_fem/"
