#!/usr/bin/env bash
# Usage: scripts/eval_one.sh <cap_file> <net_file> <mode> [extra args...]
# Outputs: prints evaluator metrics to stdout
set -e
CAP=$1; NET=$2; MODE=$3; shift 3
OUT=/tmp/eval_one_$(basename $CAP .cap).out

./router_lp/router_lp -cap "$CAP" -net "$NET" -out "$OUT" --mode "$MODE" "$@"

if [ ! -f InstantGR/run/evaluator ]; then
    bash scripts/build_evaluator.sh
fi
echo "=== Evaluator results ==="
InstantGR/run/evaluator "$CAP" "$NET" "$OUT"
