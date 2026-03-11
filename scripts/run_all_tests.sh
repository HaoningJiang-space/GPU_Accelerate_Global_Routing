#!/usr/bin/env bash
# Unified test runner: unit tests + PoC + B1 smoke.
# Usage: bash scripts/run_all_tests.sh [--gpu N]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

PASS=0; FAIL=0
stage() { echo; echo "──────────────────────────────────────────"; echo "[stage] $*"; }
ok()    { echo "[  OK  ] $*"; PASS=$((PASS+1)); }
fail()  { echo "[ FAIL ] $*"; FAIL=$((FAIL+1)); }

# Capture full output then check (avoids SIGPIPE from grep -q with pipefail)
capture_and_check() {
    local log="$1"; local pattern="$2"; local cmd=("${@:3}")
    "${cmd[@]}" > "${log}" 2>&1 || true
    grep -F "${pattern}" "${log}" > /dev/null
}

# ── Stage 1: Unit tests ───────────────────────────────────────────────────────
stage "Unit tests (test_file_reader)"
cd "${REPO}/router_lp"
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    tests/test_file_reader.cpp src/file_reader.cpp \
    -I include -o /tmp/rlp_test_file_reader
/tmp/rlp_test_file_reader > /tmp/unit_out.txt 2>&1
if grep -F "tests passed" /tmp/unit_out.txt > /dev/null; then
    result=$(grep "tests passed" /tmp/unit_out.txt)
    ok "test_file_reader: ${result}"
else
    cat /tmp/unit_out.txt
    fail "test_file_reader failed"
fi

# ── Stage 2: PoC smoke ────────────────────────────────────────────────────────
stage "PoC smoke (min_route_lp_poc)"
cd "${REPO}"
bash scripts/run_min_lp_poc.sh --gpu "${GPU_ID}" > /tmp/poc_out.txt 2>&1 || true
if grep -F "[PoC] Status: PASS" /tmp/poc_out.txt > /dev/null; then
    ok "min_route_lp_poc: PASS"
else
    cat /tmp/poc_out.txt
    fail "min_route_lp_poc: DID NOT PASS"
fi

# ── Stage 3: B1 smoke (small benchmark) ──────────────────────────────────────
stage "B1 smoke (mempool_tile_rank, --max-nets 500 --max-hpwl 30)"
BENCH="${REPO}/InstantGR/benchmarks"
CAP="${BENCH}/mempool_tile_rank.cap"
NET="${BENCH}/mempool_tile_rank.net"
OUT="/tmp/b1_smoke_out.out"

if [[ ! -f "${CAP}" || ! -f "${NET}" ]]; then
    echo "[stage] Benchmark not found at ${BENCH}, skipping B1 smoke"
else
    BIN="${REPO}/router_lp/router_lp"
    [[ -f "${BIN}" ]] || bash "${REPO}/scripts/build_router_lp.sh"

    B1_RC=0
    timeout 120 "${BIN}" \
        -cap "${CAP}" -net "${NET}" -out "${OUT}" \
        --max-nets 500 --max-hpwl 30 \
        --gpu "${GPU_ID}" --config "${REPO}/router_lp/config/cupdlpx_routing_default.yaml" \
        > /tmp/b1_smoke.txt 2>&1 || B1_RC=$?

    routed=$(grep "^  routed_nets" /tmp/b1_smoke.txt | awk -F': ' '{print $2}' | tr -d ' ' || true)
    disconn=$(grep "^  disconnected" /tmp/b1_smoke.txt | awk -F': ' '{print $2}' | tr -d ' ' || true)
    solve_t=$(grep "^  solve_time" /tmp/b1_smoke.txt | awk -F': ' '{print $2}' | tr -d ' ' || true)
    out_size=$(wc -c < "${OUT}" 2>/dev/null || echo 0)

    b1_ok=1
    [[ "${B1_RC}" -eq 0 ]]         || { fail "B1 smoke: exit code ${B1_RC} (non-zero)"; b1_ok=0; }
    [[ "${routed:-0}" -ge 1 ]]     || { fail "B1 smoke: routed_nets=${routed:-0} (need >= 1)"; b1_ok=0; }
    [[ "${disconn:-1}" -eq 0 ]]    || { fail "B1 smoke: disconnected=${disconn} (need 0)"; b1_ok=0; }
    [[ "${out_size:-0}" -gt 0 ]]   || { fail "B1 smoke: output file empty or missing"; b1_ok=0; }

    if [[ "${b1_ok}" -eq 1 ]]; then
        ok "B1 smoke: routed=${routed} disconnected=${disconn} solve_time=${solve_t} out=${out_size}B"
    else
        echo "--- B1 smoke log ---"
        cat /tmp/b1_smoke.txt
        echo "--------------------"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════"
echo "  Test results: ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════════"
[[ "${FAIL}" -eq 0 ]] && exit 0 || exit 1
