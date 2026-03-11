#!/usr/bin/env bash
set -euo pipefail

HOST_ALIAS="${HOST_ALIAS:-research23.saas.hku.hk}"
REMOTE_DIR="${REMOTE_DIR:-/home/ynwang/jhn/EDA/GPU_Accelerate_Global_Routing}"

usage() {
  cat <<'USAGE'
Usage:
  ./connect_gpu_routing_remote.sh                # interactive login and cd to workspace
  ./connect_gpu_routing_remote.sh --cmd "pwd"   # run one remote command in workspace

Env overrides:
  HOST_ALIAS   SSH host alias (default: research23.saas.hku.hk)
  REMOTE_DIR   remote workspace path
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--cmd" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Error: --cmd requires a command string" >&2
    exit 1
  fi
  ssh -t "$HOST_ALIAS" "cd '$REMOTE_DIR' && $2"
  exit 0
fi

ssh -t "$HOST_ALIAS" "cd '$REMOTE_DIR' && exec \\$SHELL -l"
