#!/usr/bin/env bash
# run_experiment_matrix.sh
# Runs router_lp in multiple modes across tiny-tier benchmarks, collects metrics,
# writes a timestamped CSV, and prints a formatted summary table.
#
# Usage: bash scripts/run_experiment_matrix.sh [--tier TIER]
#   --tier TIER  benchmark tier to run (default: tiny)
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIER="${TIER:-tiny}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER="$2"; shift 2 ;;
        *) echo "[error] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"
CSV="${RESULTS_DIR}/experiment_matrix_${TIMESTAMP}.csv"

ROUTER="router_lp/router_lp"
EVALUATOR="InstantGR/run/evaluator"
MANIFEST="benchmarks/local_suite_manifest.yaml"
ROUTE_TMP_DIR="/tmp/router_lp_exp_${TIMESTAMP}"
mkdir -p "$ROUTE_TMP_DIR"

# Mode definitions: name -> args appended to the base router invocation
declare -A MODES
MODES["lag_cpu"]="--mode lagrangian --lag-iters 50 --max-nets 50"
MODES["lag_gpu"]="--mode lagrangian --lag-gpu --lag-iters 50 --max-nets 50"
MODES["lag_gpu_polish"]="--mode lagrangian --lag-gpu --lag-iters 50 --max-nets 50 --lag-polish"

# Detect which modes are actually supported by testing the binary's help output
SUPPORTED_MODES=()
if [[ -f "$ROUTER" && -x "$ROUTER" ]]; then
    for mode_name in "lag_cpu" "lag_gpu" "lag_gpu_polish"; do
        # Split the mode args into an array for safe passing to the binary
        read -ra mode_args <<< "${MODES[$mode_name]}"
        # Run with intentionally missing required args; "Unknown option" means unsupported
        test_out=$("$ROUTER" "${mode_args[@]}" 2>&1 || true)
        if echo "$test_out" | grep -q "Unknown option"; then
            echo "[info] Mode '${mode_name}' skipped: unsupported flag detected"
        else
            SUPPORTED_MODES+=("$mode_name")
        fi
    done
else
    echo "[warn] router_lp binary not found at $ROUTER — routing will be skipped"
fi

echo "[info] Supported modes: ${SUPPORTED_MODES[*]:-none}"

# ---------------------------------------------------------------------------
# Step 1: Build evaluator if absent
# ---------------------------------------------------------------------------
if [[ ! -f "$EVALUATOR" ]]; then
    echo "[build] Evaluator not found, building..."
    bash scripts/build_evaluator.sh
fi
HAVE_EVALUATOR=false
if [[ -f "$EVALUATOR" && -x "$EVALUATOR" ]]; then
    HAVE_EVALUATOR=true
    echo "[info] Evaluator found: $EVALUATOR"
else
    echo "[warn] Evaluator not available — evaluator metrics will be N/A"
fi

# ---------------------------------------------------------------------------
# Step 2: Parse benchmark manifest for the requested tier
# ---------------------------------------------------------------------------
if [[ ! -f "$MANIFEST" ]]; then
    echo "[error] Manifest not found: $MANIFEST" >&2; exit 1
fi

# Returns lines of: name|cap_path|net_path|out_path
export TIER  # make visible to the Python subprocess below
BENCH_LIST=$(python3 - <<'PYEOF'
import yaml, sys, os

manifest_path = "benchmarks/local_suite_manifest.yaml"
tier = os.environ.get("TIER", "tiny")

with open(manifest_path) as f:
    m = yaml.safe_load(f)

cases = m.get("cases", [])
for c in cases:
    if c.get("tier") == tier:
        print(f"{c['name']}|{c['cap']}|{c['net']}|{c['out']}")
PYEOF
)

if [[ -z "$BENCH_LIST" ]]; then
    echo "[warn] No benchmarks found for tier '${TIER}'" >&2
fi

# ---------------------------------------------------------------------------
# Step 3: Write CSV header
# ---------------------------------------------------------------------------
echo "benchmark,mode,routed_nets,disconnected,max_viol,avg_viol,total_time_s,wl_cost,via_cost,overflow_cost,open_nets,incompleted_nets" > "$CSV"
echo "[info] CSV: $CSV"

# ---------------------------------------------------------------------------
# Helper: parse router_lp stdout fields
# ---------------------------------------------------------------------------
parse_router_field() {
    local log="$1" field="$2"
    # Matches lines like "  routed_nets  : 123" or "  total_time   : 4.56s"
    grep -oP "(?<=:)\s*\K[0-9]+\.?[0-9]*(?=s?\s*$)" <(grep "${field}" "$log" 2>/dev/null | tail -1) 2>/dev/null || echo "N/A"
}

# Helper: parse evaluator stdout fields
parse_eval_field() {
    local log="$1" field="$2"
    grep -oP "(?<=:)\s*\K[0-9]+\.?[0-9]*" <(grep "${field}" "$log" 2>/dev/null | tail -1) 2>/dev/null || echo "N/A"
}

# ---------------------------------------------------------------------------
# Step 4: Loop over benchmarks and modes
# ---------------------------------------------------------------------------
SUMMARY_ROWS=()  # accumulate rows for the terminal table

while IFS='|' read -r bench_name cap_path net_path out_path; do
    [[ -z "$bench_name" ]] && continue

    echo ""
    echo "=== Benchmark: ${bench_name} (tier=${TIER}) ==="

    # Validate benchmark files
    if [[ ! -f "$cap_path" ]]; then
        echo "[warn] .cap not found: $cap_path — skipping ${bench_name}"
        continue
    fi
    if [[ ! -f "$net_path" ]]; then
        echo "[warn] .net not found: $net_path — skipping ${bench_name}"
        continue
    fi

    for mode_name in "${SUPPORTED_MODES[@]:-}"; do
        [[ -z "$mode_name" ]] && continue

        echo "  -- mode: ${mode_name}"
        read -ra mode_args <<< "${MODES[$mode_name]}"

        out_file="${ROUTE_TMP_DIR}/${bench_name}_${mode_name}.out"
        router_log="${ROUTE_TMP_DIR}/${bench_name}_${mode_name}.router.log"

        # Run router
        routed="N/A"; disconnected="N/A"; max_viol="N/A"; avg_viol="N/A"; total_time="N/A"
        if [[ -f "$ROUTER" && -x "$ROUTER" ]]; then
            set +e
            "$ROUTER" -cap "$cap_path" -net "$net_path" -out "$out_file" \
                "${mode_args[@]}" > "$router_log" 2>&1
            router_exit=$?
            set -e

            if [[ $router_exit -ne 0 ]]; then
                echo "    [warn] router_lp exited with code ${router_exit} (may indicate unresolved violations)"
            fi
            # Always attempt to parse metrics from the log (non-zero exit is normal when
            # capacity violations remain after the iteration limit is reached)
            if [[ -s "$router_log" ]]; then
                routed=$(parse_router_field "$router_log" "routed_nets")
                disconnected=$(parse_router_field "$router_log" "disconnected")
                max_viol=$(parse_router_field "$router_log" "max_viol")
                avg_viol=$(parse_router_field "$router_log" "avg_viol")
                total_time=$(parse_router_field "$router_log" "total_time")
                echo "    routed=${routed} disconnected=${disconnected} max_viol=${max_viol} avg_viol=${avg_viol} time=${total_time}s"
            else
                echo "    [warn] router_lp produced no output — all metrics N/A"
            fi
        fi

        # Run evaluator (only if route output was produced)
        wl_cost="N/A"; via_cost="N/A"; overflow_cost="N/A"; open_nets="N/A"; incompleted="N/A"
        if $HAVE_EVALUATOR && [[ -f "$out_file" ]]; then
            eval_log="${ROUTE_TMP_DIR}/${bench_name}_${mode_name}.eval.log"
            set +e
            "$EVALUATOR" "$cap_path" "$net_path" "$out_file" > "$eval_log" 2>&1
            eval_exit=$?
            set -e
            if [[ $eval_exit -ne 0 ]]; then
                echo "    [warn] evaluator exited with code ${eval_exit}"
            else
                wl_cost=$(parse_eval_field "$eval_log" "Wire cost")
                via_cost=$(parse_eval_field "$eval_log" "Via cost")
                overflow_cost=$(parse_eval_field "$eval_log" "Overflow cost")
                open_nets=$(parse_eval_field "$eval_log" "Number of open nets")
                incompleted=$(parse_eval_field "$eval_log" "Number of incompleted nets")
                echo "    wl=${wl_cost} via=${via_cost} overflow=${overflow_cost} open=${open_nets} incomplete=${incompleted}"
            fi
        fi

        # Write CSV row
        echo "${bench_name},${mode_name},${routed},${disconnected},${max_viol},${avg_viol},${total_time},${wl_cost},${via_cost},${overflow_cost},${open_nets},${incompleted}" >> "$CSV"
        SUMMARY_ROWS+=("${bench_name}|${mode_name}|${routed}|${disconnected}|${max_viol}|${avg_viol}|${total_time}|${wl_cost}|${via_cost}|${overflow_cost}|${open_nets}|${incompleted}")
    done
done <<< "$BENCH_LIST"

# ---------------------------------------------------------------------------
# Step 5: Print formatted summary table
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo " Experiment Matrix Summary  (tier=${TIER})  — ${TIMESTAMP}"
echo "=================================================================="
printf "%-26s %-18s %7s %6s %8s %8s %8s %10s %10s %13s %6s %6s\n" \
    "benchmark" "mode" "routed" "disc" "max_viol" "avg_viol" "time_s" \
    "wl_cost" "via_cost" "overflow_cost" "open" "incompl"
echo "----------------------------------------------------------------------------------------------------------------------------------"
for row in "${SUMMARY_ROWS[@]:-}"; do
    [[ -z "$row" ]] && continue
    IFS='|' read -r r_bench r_mode r_routed r_disc r_maxv r_avgv r_time \
        r_wl r_via r_ovf r_open r_inc <<< "$row"
    printf "%-26s %-18s %7s %6s %8s %8s %8s %10s %10s %13s %6s %6s\n" \
        "$r_bench" "$r_mode" "$r_routed" "$r_disc" "$r_maxv" "$r_avgv" "$r_time" \
        "$r_wl" "$r_via" "$r_ovf" "$r_open" "$r_inc"
done
echo "=================================================================="
echo ""
echo "[done] Results written to: $CSV"
echo "[done] Temporary routing outputs: $ROUTE_TMP_DIR"
