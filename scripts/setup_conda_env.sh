#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-envs/gagr_dev.yml}"
ENV_NAME="${2:-gagr-dev}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: environment file '$ENV_FILE' not found." >&2
  exit 1
fi

echo "[1/4] Creating/updating conda env: $ENV_NAME"
conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune || conda env create -f "$ENV_FILE"

echo "[2/4] Activating env: $ENV_NAME"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[3/4] Verifying Python packages"
python - <<'PY'
import numpy, scipy, pandas, yaml, matplotlib, networkx
print('Python and core packages OK')
PY

echo "[4/4] Notes"
echo "- Ensure system CUDA + NVCC are installed (recommended CUDA >= 12.4)."
echo "- InstantGR compilation still uses system nvcc, not conda nvcc by default."
