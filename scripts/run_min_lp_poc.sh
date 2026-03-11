#!/usr/bin/env bash
# scripts/run_min_lp_poc.sh
# Compile and run the Phase B0 minimum routing LP PoC.
#
# Usage:
#   bash scripts/run_min_lp_poc.sh [--gpu <id>]
#
# Requires cuPDLPx built at /tmp/cupdlpx_build (or set CUPDLPX_BUILD_DIR)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUPDLPX_INC="$REPO_ROOT/cuPDLPx/include"
CUPDLPX_BUILD="${CUPDLPX_BUILD_DIR:-/tmp/cupdlpx_build}"
PSLP_INC="$CUPDLPX_BUILD/_deps/pslp-src/include/PSLP"
SRC="$REPO_ROOT/router_lp/examples/min_route_lp_with_cupdlpx.cpp"
BIN="$REPO_ROOT/router_lp/examples/min_route_lp_poc"
GPU_ID="${GPU_ID:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    *)     echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[b0-poc] Compiling..."
g++ -std=c++17 -O2 \
    "$SRC" \
    -I"$CUPDLPX_INC" \
    -I"$PSLP_INC" \
    -L"$CUPDLPX_BUILD" -lcupdlpx \
    -Wl,-rpath,"$CUPDLPX_BUILD" \
    -o "$BIN"
echo "[b0-poc] Compile OK"

echo "[b0-poc] Running on GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$BIN"
echo "[b0-poc] Exit code: $?"
