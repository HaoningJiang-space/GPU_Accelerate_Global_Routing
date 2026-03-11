#!/usr/bin/env bash
# scripts/run_instantgr_baseline.sh
# Run InstantGR on all cases in local_suite_manifest.yaml and record results.
#
# Usage:
#   bash scripts/run_instantgr_baseline.sh [--gpu <id>] [--arch <sm_XX>] [--recompile]
#
# Outputs:
#   results/baseline/<case>/route.out   routing solution
#   results/baseline/<case>/runtime.txt wall-clock time in seconds
#   results/baseline/metrics.csv        aggregated metrics (appended by eval script)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/benchmarks/local_suite_manifest.yaml"
INSTANTGR_SRC="$REPO_ROOT/InstantGR/src"
INSTANTGR_BIN="$REPO_ROOT/InstantGR/run/InstantGR"
EVALUATOR_BIN="$REPO_ROOT/InstantGR/run/evaluator"

GPU_ID="${GPU_ID:-1}"
ARCH="${ARCH:-sm_86}"
RECOMPILE=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)    GPU_ID="$2"; shift 2 ;;
    --arch)   ARCH="$2";   shift 2 ;;
    --recompile) RECOMPILE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[baseline] GPU=$GPU_ID  arch=$ARCH  recompile=$RECOMPILE"

# ── Compile InstantGR if needed ───────────────────────────────────────────────
if [[ ! -f "$INSTANTGR_BIN" || "$RECOMPILE" -eq 1 ]]; then
  echo "[baseline] Compiling InstantGR (arch=$ARCH)..."
  cd "$INSTANTGR_SRC"
  CUDA_VISIBLE_DEVICES="$GPU_ID" nvcc main.cpp -o ../run/InstantGR \
    -std=c++17 -x cu -O3 -arch="$ARCH"
  echo "[baseline] Compile done."
  cd "$REPO_ROOT"
fi

# ── Parse manifest with Python (avoids yq dependency) ────────────────────────
CASES_JSON=$(python3 - <<'PY'
import yaml, json, sys
with open("benchmarks/local_suite_manifest.yaml") as f:
    data = yaml.safe_load(f)
print(json.dumps(data["cases"]))
PY
)

N=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$CASES_JSON")
echo "[baseline] Running $N case(s) from manifest..."

# ── Run each case ─────────────────────────────────────────────────────────────
python3 - "$CASES_JSON" "$REPO_ROOT" "$INSTANTGR_BIN" "$EVALUATOR_BIN" "$GPU_ID" <<'PY'
import json, os, subprocess, sys, time

cases      = json.loads(sys.argv[1])
root       = sys.argv[2]
router_bin = sys.argv[3]
eval_bin   = sys.argv[4]
gpu_id     = sys.argv[5]
env        = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}

for c in cases:
    name   = c["name"]
    cap    = os.path.join(root, c["cap"])
    net    = os.path.join(root, c["net"])
    out    = os.path.join(root, c["out"])

    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"\n[baseline] ── case: {name} ──────────────────────")
    # InstantGR must be run from its own run/ dir (reads POWV9.dat/POST9.dat from cwd)
    run_dir = os.path.dirname(router_bin)
    t0 = time.perf_counter()
    result = subprocess.run(
        [router_bin, "-cap", cap, "-net", net, "-out", out],
        env=env, capture_output=True, text=True, cwd=run_dir
    )
    runtime = time.perf_counter() - t0

    # Save runtime
    rt_file = os.path.join(os.path.dirname(out), "runtime.txt")
    with open(rt_file, "w") as f:
        f.write(f"{runtime:.3f}\n")

    if result.returncode != 0:
        print(f"[baseline] ERROR: router exited {result.returncode}")
        print(result.stderr[-2000:])
        continue

    print(f"[baseline] router done in {runtime:.1f}s")

    # Evaluate
    eval_result = subprocess.run(
        [eval_bin, cap, net, out],
        capture_output=True, text=True
    )
    print(eval_result.stdout[-3000:])

    # Parse FOR_STAT and other summary lines
    wl = via = of = total = None
    opens = incompleted = None
    for line in eval_result.stdout.splitlines():
        if line.startswith("FOR_STAT"):
            parts = line.split()
            wl, via, of, total = parts[1], parts[2], parts[3], parts[4]
        if "Number of open nets" in line:
            opens = line.split(":")[-1].strip()
        if "Number of incompleted" in line:
            incompleted = line.split(":")[-1].strip()

    # Append to metrics.csv
    csv_path = os.path.join(root, "results/baseline/metrics.csv")
    write_header = not os.path.exists(csv_path)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(csv_path, "a") as f:
        if write_header:
            f.write("case,tier,wirelength_cost,via_cost,overflow_cost,total_cost,"
                    "open_nets,incompleted_nets,runtime_s,timestamp\n")
        f.write(f"{name},{c.get('tier','?')},{wl},{via},{of},{total},"
                f"{opens},{incompleted},{runtime:.3f},{timestamp}\n")
    print(f"[baseline] metrics appended → results/baseline/metrics.csv")
PY

echo ""
echo "[baseline] Done. See results/baseline/metrics.csv"
