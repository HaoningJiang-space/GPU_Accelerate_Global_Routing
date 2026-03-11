#!/usr/bin/env bash
# B2 solver parameter sweep on a small benchmark.
# Sweeps eps_optimal_relative, time_sec_limit, iteration_limit.
# Outputs results/param_sweep_<timestamp>.csv and prints Pareto summary.
#
# Usage: bash scripts/run_param_sweep.sh [--gpu N] [--bench DIR] [--max-nets N] [--max-hpwl N]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
BENCH="${REPO}/InstantGR/benchmarks"
MAX_NETS=500
MAX_HPWL=30
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="${REPO}/results"
CSV="${OUTDIR}/param_sweep_${TS}.csv"
BIN="${REPO}/router_lp/router_lp"
YAML_TMPL="${REPO}/router_lp/config/cupdlpx_routing_default.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU_ID="$2";   shift 2 ;;
        --bench)    BENCH="$2";    shift 2 ;;
        --max-nets) MAX_NETS="$2"; shift 2 ;;
        --max-hpwl) MAX_HPWL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

CAP="${BENCH}/mempool_tile_rank.cap"
NET="${BENCH}/mempool_tile_rank.net"
[[ -f "${CAP}" && -f "${NET}" ]] || { echo "Benchmark not found: ${BENCH}"; exit 1; }
[[ -f "${BIN}" ]] || bash "${REPO}/scripts/build_router_lp.sh"
mkdir -p "${OUTDIR}"

echo "eps_opt,time_limit,iter_limit,routed,disconnected,solve_time_s,iterations,obj,n_vars,n_cons,status" | tee "${CSV}"

# Parameter grid
EPS_OPT_VALUES=(1e-3 1e-4 1e-6)
TIME_LIMITS=(5 15 60)
ITER_LIMITS=(500 2000 10000)

TMPYAML="/tmp/sweep_params.yaml"
TMPOUT="/tmp/sweep_out.out"
TMPLOG="/tmp/sweep_log.txt"

for eps in "${EPS_OPT_VALUES[@]}"; do
for tlim in "${TIME_LIMITS[@]}"; do
for ilim in "${ITER_LIMITS[@]}"; do

    # Write temp YAML
    cat > "${TMPYAML}" << YAML
eps_optimal_relative: ${eps}
eps_feasible_relative: ${eps}
time_sec_limit: ${tlim}
iteration_limit: ${ilim}
feasibility_polishing: true
verbose: false
YAML

    # Run router_lp with timeout slightly above tlim
    timeout $((tlim + 10)) "${BIN}" \
        -cap "${CAP}" -net "${NET}" -out "${TMPOUT}" \
        --gpu "${GPU_ID}" \
        --max-nets "${MAX_NETS}" --max-hpwl "${MAX_HPWL}" \
        --config "${TMPYAML}" \
        > "${TMPLOG}" 2>&1 || true

    # Extract metrics from summary block
    routed=$(grep  "^  routed_nets"  "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')
    disconn=$(grep "^  disconnected" "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')
    stime=$(grep   "^  solve_time"   "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' s')
    iters=$(grep   "^  iterations"   "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')
    obj=$(grep     "^  obj_value"    "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')
    nvars=$(grep   "^  n_vars"       "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')
    ncons=$(grep   "^  n_cons"       "${TMPLOG}" | awk -F': ' '{print $2}' | tr -d ' ')

    # Determine status label
    if grep -qF "CUDA_VISIBLE_DEVICES" "${TMPLOG}" && grep -qF "routed_nets" "${TMPLOG}"; then
        if [[ "${disconn:-1}" -eq 0 && "${routed:-0}" -ge 1 ]]; then
            status="OK"
        else
            status="PARTIAL"
        fi
    else
        status="FAILED"
    fi

    row="${eps},${tlim},${ilim},${routed:-0},${disconn:-?},${stime:-?},${iters:-?},${obj:-?},${nvars:-?},${ncons:-?},${status}"
    echo "${row}" | tee -a "${CSV}"

done
done
done

echo ""
echo "=== Param sweep complete: ${CSV} ==="
echo ""
echo "--- Pareto: OK rows sorted by solve_time ---"
awk -F',' 'NR==1{print; next} $11=="OK"{print}' "${CSV}" | \
    sort -t',' -k6 -n | \
    awk -F',' 'NR==1{printf "%-10s %-10s %-10s %-8s %-8s %-14s %-10s %-12s\n","eps_opt","t_lim","i_lim","routed","disc","solve_time_s","iters","obj"; next}
    {printf "%-10s %-10s %-10s %-8s %-8s %-14s %-10s %-12s\n",$1,$2,$3,$4,$5,$6,$7,$8}'
