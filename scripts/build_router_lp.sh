#!/usr/bin/env bash
# Build router_lp main binary.
# Usage: bash scripts/build_router_lp.sh [--debug]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CUPDLPX_BUILD="${CUPDLPX_BUILD:-/tmp/cupdlpx_build}"
OUT="${REPO}/router_lp/router_lp"
CUDA_ARCH="${CUDA_ARCH:-86}"   # RTX 3090 = sm_86; override with CUDA_ARCH=80 etc.

OPT="-O2"
NVCC_OPT="-O2"
[[ "${1:-}" == "--debug" ]] && { OPT="-O0 -g"; NVCC_OPT="-O0 -g"; }

SRCS=(
    "${REPO}/router_lp/main.cpp"
    "${REPO}/router_lp/src/file_reader.cpp"
    "${REPO}/router_lp/src/routing_lp_builder.cpp"
    "${REPO}/router_lp/src/solve_with_cupdlpx.cpp"
    "${REPO}/router_lp/src/solution_rounding.cpp"
    "${REPO}/router_lp/src/windowed_routing.cpp"
    "${REPO}/router_lp/src/adaptive_routing.cpp"
    "${REPO}/router_lp/src/lagrangian_router.cpp"
    "${REPO}/router_lp/src/repair_router.cpp"
    "${REPO}/router_lp/src/hotspot_polish.cpp"
)

GPU_OBJ="/tmp/lagrangian_gpu_sm${CUDA_ARCH}.o"

# Step 1: compile CUDA kernel with nvcc
echo "[build] Compiling CUDA kernel (sm_${CUDA_ARCH})..."
nvcc -std=c++17 ${NVCC_OPT} \
    -arch=sm_${CUDA_ARCH} \
    -I "${REPO}/router_lp/include" \
    -c "${REPO}/router_lp/src/lagrangian_gpu.cu" \
    -o "${GPU_OBJ}"

# Step 2: compile and link C++ sources with g++, adding GPU object + cudart
echo "[build] Compiling router_lp..."
g++ -std=c++17 ${OPT} -Wall -Wextra -Werror -fopenmp \
    "${SRCS[@]}" \
    "${GPU_OBJ}" \
    -I "${REPO}/router_lp/include" \
    -I "${REPO}/cuPDLPx/include" \
    -I "${CUPDLPX_BUILD}/_deps/pslp-src/include/PSLP" \
    -L "${CUPDLPX_BUILD}" -lcupdlpx \
    -L /usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,"${CUPDLPX_BUILD}" \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -o "${OUT}"

echo "[build] OK -> ${OUT}"
