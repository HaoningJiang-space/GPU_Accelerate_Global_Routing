#!/usr/bin/env bash
# B2 subgraph contraction study.
# Two sub-experiments:
#
#  A. mempool_tile_rank  – vary --max-hpwl (10 20 30 50) with --max-nets=500
#     Shows how problem size grows as we admit longer-range nets.
#
#  B. ariane133_51       – vary --max-nets (5 10 20 50) with no HPWL cap
#     Shows how n_vars scales with net count on a medium benchmark;
#     first N nets share a small region so bbox grows slowly.
#
# Oversized combinations (exceeding the builder's 10M variable limit) are
# detected from builder stderr messages and recorded as SKIPPED_OVERSIZE.
#
# Outputs results/subgraph_study_<timestamp>.csv.
#
# Usage: bash scripts/run_subgraph_study.sh [--gpu N]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="${REPO}/results"
CSV="${OUTDIR}/subgraph_study_${TS}.csv"
BIN="${REPO}/router_lp/router_lp"
CFG="${REPO}/router_lp/config/cupdlpx_routing_best.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -f "${BIN}" ]] || bash "${REPO}/scripts/build_router_lp.sh"
[[ -f "${CFG}" ]] || CFG="${REPO}/router_lp/config/cupdlpx_routing_default.yaml"
mkdir -p "${OUTDIR}"

TMPLOG="/tmp/subgraph_log.txt"
TMPOUT="/tmp/subgraph_out.out"

echo "experiment,benchmark,max_hpwl,max_nets,n_vars,n_cons,nnz,n_nets_in,routed,disconnected,solve_time_s,round_time_s,total_time_s,status" | tee "${CSV}"

run_point() {
    local exp="$1" bench_label="$2" cap="$3" net="$4" hpwl="$5" maxnets="$6"

    local RC=0
    timeout 90 "${BIN}" \
        -cap "${cap}" -net "${net}" -out "${TMPOUT}" \
        --gpu "${GPU_ID}" \
        --max-nets "${maxnets}" --max-hpwl "${hpwl}" \
        --config "${CFG}" \
        > "${TMPLOG}" 2>&1 || RC=$?

    local nvars ncons nnz nnets routed disconn stime rtime ttime status
    nvars=$(grep  "^  n_vars"        "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    ncons=$(grep  "^  n_cons"        "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    nnz=$(grep    "^  nnz"           "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    nnets=$(grep  "^  n_nets_in"     "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    routed=$(grep "^  routed_nets"   "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    disconn=$(grep "^  disconnected" "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' '   || true)
    stime=$(grep  "^  solve_time"    "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' s'  || true)
    rtime=$(grep  "^  round_time"    "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' s'  || true)
    ttime=$(grep  "^  total_time"    "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' s'  || true)

    if   [[ "${RC}" -eq 124 ]]; then status="TIMEOUT"
    elif grep -qF "Pre-flight reject" "${TMPLOG}" 2>/dev/null; then status="SKIPPED_OVERSIZE"
    elif grep -qF "Overflow guard"    "${TMPLOG}" 2>/dev/null; then status="SKIPPED_OVERSIZE"
    elif grep -qF "No nets after"     "${TMPLOG}" 2>/dev/null; then status="NO_NETS"
    elif grep -qF "solver did not reach OPTIMAL" "${TMPLOG}" 2>/dev/null; then status="SOLVER_LIMIT"
    elif [[ "${RC}" -ne 0   ]]; then status="INFRA_ERROR"
    elif [[ "${disconn:-1}" -eq 0 && "${routed:-0}" -ge 1 ]]; then status="OK"
    else status="PARTIAL"
    fi

    echo "${exp},${bench_label},${hpwl},${maxnets},${nvars:-?},${ncons:-?},${nnz:-?},${nnets:-?},${routed:-0},${disconn:-?},${stime:-?},${rtime:-?},${ttime:-?},${status}" \
        | tee -a "${CSV}"

    if [[ "${status}" == "TIMEOUT" ]]; then
        tail -4 "${TMPLOG}" | sed 's/^/  [log] /'
    fi
}

# ── Experiment A: mempool_tile_rank, vary max-hpwl ────────────────────────────
CAP_A="${REPO}/InstantGR/benchmarks/mempool_tile_rank.cap"
NET_A="${REPO}/InstantGR/benchmarks/mempool_tile_rank.net"
if [[ -f "${CAP_A}" && -f "${NET_A}" ]]; then
    echo "# Experiment A: mempool_tile_rank – varying max_hpwl (max_nets=500)"
    for hpwl in 10 20 30 50 100; do
        run_point "A-hpwl" "mempool_tile_rank" "${CAP_A}" "${NET_A}" "${hpwl}" 500
    done
else
    echo "# Experiment A: mempool_tile_rank not found, skipping"
fi

# ── Experiment B: ariane133_51, vary max-nets ─────────────────────────────────
CAP_B="${REPO}/benchmarks/ispd2024/ariane133_51.cap"
NET_B="${REPO}/benchmarks/ispd2024/ariane133_51.net"
if [[ -f "${CAP_B}" && -f "${NET_B}" ]]; then
    echo "# Experiment B: ariane133_51 – varying max_nets (max_hpwl=10000)"
    for maxnets in 5 10 20 50; do
        run_point "B-nets" "ariane133_51" "${CAP_B}" "${NET_B}" 10000 "${maxnets}"
    done
else
    echo "# Experiment B: ariane133_51 not found, skipping"
fi

echo ""
echo "=== Subgraph study complete: ${CSV} ==="
echo ""
echo "--- Problem size vs quality (all runs) ---"
awk -F',' '
NR==1{printf "%-8s %-20s %-8s %-8s %-10s %-10s %-8s %-8s %-13s %-15s\n",
    "exp","benchmark","hpwl","nets","n_vars","n_cons","routed","disc","solve_time_s","status"; next}
/^#/{next}
{printf "%-8s %-20s %-8s %-8s %-10s %-10s %-8s %-8s %-13s %-15s\n",
    $1,$2,$3,$4,$5,$6,$9,$10,$11,$14}' "${CSV}"

