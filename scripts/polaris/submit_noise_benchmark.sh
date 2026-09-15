#!/bin/bash
# Submit the full noise+diffusion benchmark matrix on Polaris.
#
# Usage:
#   bash scripts/polaris/submit_noise_benchmark.sh [--debug-only]
#
# Jobs:
#   1. Debug smoke test (1 node, 1 hr) — always submitted first
#   2. Uniform production (30 nodes, 3 hr) — submitted after debug
#   3. Adaptive production (30 nodes, 3 hr) — submitted after debug
#
# Estimated cost:
#   Debug:    1 node × 1 hr = 4 GPU-hr (A100 × 4)
#   Uniform:  30 nodes × 3 hr = 360 GPU-hr
#   Adaptive: 30 nodes × 3 hr = 360 GPU-hr
#   Total: ~724 GPU-hr out of 5,000 allocation (14.5%)
#
# Matrix coverage:
#   4 archs (gru, transformer, ssm, diffusion)
#   × 6 noise (noiseless, gauss_0.5x, 1x, 2x, 5x, realistic)
#   × 5 conductors
#   × 2 sampling (uniform, adaptive)
#   = 240 total training runs

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
DEBUG_ONLY=false

if [[ "${1:-}" == "--debug-only" ]]; then
    DEBUG_ONLY=true
fi

echo "=== Noise+Diffusion Benchmark Submission ==="
echo ""

# Always submit debug first
echo "Submitting debug smoke test..."
DEBUG_JOB=$(qsub "$SCRIPTS_DIR/run_noise_bench_debug.pbs")
echo "  Debug job: $DEBUG_JOB"

if $DEBUG_ONLY; then
    echo ""
    echo "Debug-only mode. Check results before submitting production:"
    echo "  qstat $DEBUG_JOB"
    echo "  ls /eagle/fanc_ldrd/magnets_fem/noise_bench_debug/"
    echo ""
    echo "Then submit production:"
    echo "  qsub $SCRIPTS_DIR/run_noise_bench_prod.pbs"
    echo "  qsub $SCRIPTS_DIR/run_noise_bench_prod_adaptive.pbs"
    exit 0
fi

echo ""
echo "Submitting production jobs (will queue behind debug)..."

UNIFORM_JOB=$(qsub "$SCRIPTS_DIR/run_noise_bench_prod.pbs")
echo "  Uniform job: $UNIFORM_JOB"

ADAPTIVE_JOB=$(qsub "$SCRIPTS_DIR/run_noise_bench_prod_adaptive.pbs")
echo "  Adaptive job: $ADAPTIVE_JOB"

echo ""
echo "=== Submitted ==="
echo "  Debug:    $DEBUG_JOB  (1 node, ~1 hr)"
echo "  Uniform:  $UNIFORM_JOB  (30 nodes, ~3 hr)"
echo "  Adaptive: $ADAPTIVE_JOB  (30 nodes, ~3 hr)"
echo ""
echo "Estimated GPU-hours: ~724 / 5,000 (14.5%)"
echo ""
echo "Monitor: qstat -u \$USER"
echo "Results: /eagle/fanc_ldrd/magnets_fem/noise_bench_{debug,uniform,adaptive}/"
