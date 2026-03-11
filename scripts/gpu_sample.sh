#!/usr/bin/env bash
# GPU utilization sampler — wraps a command and captures GPU metrics.
#
# Usage:
#   bash scripts/gpu_sample.sh [--gpu N] [--interval S] [--out FILE] -- CMD [ARGS...]
#
# Produces:
#   FILE (default /tmp/gpu_sample.csv): timestamp,gpu,util%,mem_used_MiB,mem_total_MiB
#   Prints summary (peak util, peak mem) to stdout after CMD exits.
#
# Requires: nvitop (preferred) or nvidia-smi dmon
set -euo pipefail

GPU_ID=0
INTERVAL=1
OUT="/tmp/gpu_sample.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU_ID="$2";  shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --out)      OUT="$2";     shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: gpu_sample.sh [opts] -- CMD [ARGS...]"; exit 1
fi

echo "timestamp,gpu,util_pct,mem_used_MiB,mem_total_MiB" > "${OUT}"

# Background sampler using nvidia-smi (nvitop has no CSV mode)
# Note: nvidia-smi query exits with rc=255 on this server if some GPU handles
#       are unavailable; we ignore that via || true and only query --id=${GPU_ID}
sample_loop() {
    while true; do
        ts=$(date +%s)
        row=$(nvidia-smi --id="${GPU_ID}" \
            --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null || echo ",,")
        util=$(echo "${row}" | awk -F',' '{print $1}' | tr -d ' ')
        mem_used=$(echo "${row}" | awk -F',' '{print $2}' | tr -d ' ')
        mem_total=$(echo "${row}" | awk -F',' '{print $3}' | tr -d ' ')
        echo "${ts},${GPU_ID},${util},${mem_used},${mem_total}" >> "${OUT}"
        sleep "${INTERVAL}"
    done
}

sample_loop &
SAMPLER_PID=$!
trap "kill ${SAMPLER_PID} 2>/dev/null || true" EXIT

# Run the wrapped command
CUDA_VISIBLE_DEVICES="${GPU_ID}" "$@"
CMD_RC=$?

kill "${SAMPLER_PID}" 2>/dev/null || true
trap - EXIT
sleep 0.2  # let sampler flush last row

# Summary
echo ""
echo "=== GPU sample summary (GPU ${GPU_ID}) ==="
awk -F',' 'NR>1 && $3!="" {
    if ($3+0 > max_u) max_u=$3+0;
    if ($4+0 > max_m) max_m=$4+0;
    sum_u+=$3+0; sum_m+=$4+0; n++
} END {
    if (n>0) printf "  samples      : %d\n  peak_util%%  : %d\n  peak_mem_MiB : %d\n  avg_util%%   : %.1f\n  avg_mem_MiB  : %.1f\n",
        n, max_u, max_m, sum_u/n, sum_m/n
    else print "  (no samples collected)"
}' "${OUT}"
echo "  csv          : ${OUT}"

exit "${CMD_RC}"
