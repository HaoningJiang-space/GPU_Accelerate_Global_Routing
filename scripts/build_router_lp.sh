#!/usr/bin/env bash
# Build router_lp main binary.
# Usage: bash scripts/build_router_lp.sh [--debug]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CUPDLPX_BUILD="${CUPDLPX_BUILD:-/tmp/cupdlpx_build}"
OUT="${REPO}/router_lp/router_lp"

OPT="-O2"
[[ "${1:-}" == "--debug" ]] && OPT="-O0 -g"

SRCS=(
    "${REPO}/router_lp/main.cpp"
    "${REPO}/router_lp/src/file_reader.cpp"
    "${REPO}/router_lp/src/routing_lp_builder.cpp"
    "${REPO}/router_lp/src/solve_with_cupdlpx.cpp"
    "${REPO}/router_lp/src/solution_rounding.cpp"
    "${REPO}/router_lp/src/windowed_routing.cpp"
    "${REPO}/router_lp/src/adaptive_routing.cpp"
    "${REPO}/router_lp/src/lagrangian_router.cpp"
)

echo "[build] Compiling router_lp..."
g++ -std=c++17 ${OPT} -Wall -Wextra -Werror \
    "${SRCS[@]}" \
    -I "${REPO}/router_lp/include" \
    -I "${REPO}/cuPDLPx/include" \
    -I "${CUPDLPX_BUILD}/_deps/pslp-src/include/PSLP" \
    -L "${CUPDLPX_BUILD}" -lcupdlpx \
    -Wl,-rpath,"${CUPDLPX_BUILD}" \
    -o "${OUT}"

echo "[build] OK → ${OUT}"
