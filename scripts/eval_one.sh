#!/usr/bin/env bash
# Usage: scripts/eval_one.sh <cap_file> <net_file> <mode> [extra args...]
# Outputs: prints evaluator metrics to stdout
#
# When --max-nets N is passed, only a subset of nets is routed.  The script
# automatically builds a matching subset .net so the evaluator scores only
# the nets that were actually attempted, avoiding inflated "incompleted" counts.
set -euo pipefail
CAP=$1; NET=$2; MODE=$3; shift 3
STEM=$(basename "$CAP" .cap)
OUT=/tmp/eval_one_${STEM}.out
SUBSET_NET=/tmp/eval_one_${STEM}_subset.net

# router_lp exits non-zero when violations remain but still writes a valid .out
./router_lp/router_lp -cap "$CAP" -net "$NET" -out "$OUT" --mode "$MODE" "$@" || true

if [ ! -f "$OUT" ]; then
    echo "[eval_one] router_lp produced no output file, aborting" >&2
    exit 1
fi

# Check whether this is a partial run (--max-nets used).
# Count unique net names in .out vs total nets in .net.
N_ROUTED=$(awk '/^\(/{next} /^\)/{next} !/^[[:space:]]/{count++} END{print count+0}' "$OUT")
N_TOTAL=$(grep -c '^(' "$NET" || true)
if [ "$N_ROUTED" -lt "$N_TOTAL" ]; then
    echo "[eval_one] WARNING: partial run ($N_ROUTED routed / $N_TOTAL total nets)"
    echo "[eval_one] Building subset .net so evaluator only scores attempted nets..."

    # Extract routed net names (lines that are not '(', ')', or segment lines)
    awk '
      /^\(/ { in_seg=1; next }
      /^\)/ { in_seg=0; next }
      !in_seg && NF>0 { print }
    ' "$OUT" > /tmp/eval_one_routed_names.txt

    # Build subset .net: copy only blocks whose net name is in routed set
    python3 - "$NET" /tmp/eval_one_routed_names.txt "$SUBSET_NET" <<'PYEOF'
import sys
net_file, names_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
with open(names_file) as f:
    routed = set(l.strip() for l in f if l.strip())
with open(net_file) as f, open(out_file, 'w') as g:
    block = []
    name = None
    for line in f:
        s = line.rstrip()
        if not block and s and not s.startswith('(') and not s.startswith('['):
            name = s
            block = [line]
        elif block:
            block.append(line)
            if s == ')':
                if name in routed:
                    g.writelines(block)
                block = []
                name = None
PYEOF
    EVAL_NET="$SUBSET_NET"
else
    EVAL_NET="$NET"
fi

if [ ! -f InstantGR/run/evaluator ]; then
    bash scripts/build_evaluator.sh
fi
echo "=== Evaluator results (nets evaluated: $N_ROUTED) ==="
InstantGR/run/evaluator "$CAP" "$EVAL_NET" "$OUT"
