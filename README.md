# GPU-Accelerated Global Routing via Lagrangian Decomposition

A GPU-accelerated global router (`router_lp/`) that formulates routing as a
Linear Program (LP) and solves it with **cuPDLPx** (GPU-accelerated first-order
LP solver) or via **Lagrangian decomposition** with parallel per-net shortest-path
on GPU.  InstantGR is used as a baseline reference only and is never linked into
the new router.

---

## Architecture

```
GPU_Accelerate_Global_Routing/
├── router_lp/               ← new routing prototype (main deliverable)
│   ├── main.cpp             ← entry point; --mode dispatch
│   ├── include/             ← all public headers
│   │   ├── routing_types.hpp          GridInfo, TwoNet, RoutingSegment
│   │   ├── routing_lp_problem.hpp     LP IR: RoutingLPProblem / Solution
│   │   ├── routing_lp_builder.hpp     build_routing_lp() config
│   │   ├── solve_with_cupdlpx.hpp     cuPDLPx C-API adapter
│   │   ├── solution_rounding.hpp      BFS flow decomposition
│   │   ├── windowed_routing.hpp       tiled-window LP mode
│   │   ├── adaptive_routing.hpp       centroid-bin batching mode
│   │   ├── lagrangian_router.hpp      CPU Dijkstra Lagrangian mode
│   │   └── lagrangian_gpu.hpp         GPU BF Lagrangian mode
│   ├── src/
│   │   ├── file_reader.cpp            .cap/.net parser
│   │   ├── routing_lp_builder.cpp     LP matrix construction (CSR)
│   │   ├── solve_with_cupdlpx.cpp     cuPDLPx C-API wrapper
│   │   ├── solution_rounding.cpp      fractional → discrete path recovery
│   │   ├── windowed_routing.cpp       tiled LP solver
│   │   ├── adaptive_routing.cpp       adaptive per-net bbox solver
│   │   ├── lagrangian_router.cpp      CPU sequential Dijkstra + subgradient
│   │   └── lagrangian_gpu.cu          GPU batched BF + subgradient (CUDA)
│   └── config/
│       ├── cupdlpx_routing_default.yaml
│       └── cupdlpx_routing_best.yaml  (B2 sweep winner)
├── cuPDLPx/                 ← GPU LP solver (restarted Halpern PDHG)
├── InstantGR/               ← baseline heuristic router (reference only)
├── benchmarks/ispd2024/     ← ISPD 2024 routing benchmarks
├── scripts/                 ← build, test, sweep, baseline scripts
├── results/                 ← solver outputs and sweep CSVs
└── docs/                    ← metrics_definition.md, C API docs
```

---

## Routing Modes

| `--mode`      | Backend         | Scale | Notes |
|---------------|-----------------|-------|-------|
| `pure_lp`     | cuPDLPx GPU LP  | Small subgraphs (< ~10M vars) | Joint-bbox LP; pre-flight size check |
| `windowed`    | cuPDLPx GPU LP  | Medium | Tiled WX×WY windows; nets outside window = unassigned |
| `adaptive`    | cuPDLPx GPU LP  | Medium | Centroid-bin batching + per-net fallback; 0 unassigned |
| `lagrangian`  | CPU Dijkstra    | Large | Subgradient, per-net Dijkstra; float costs |
| `lagrangian --lag-gpu` | GPU BF (CUDA) | Large | All-nets BF in parallel; int costs; `atomicMin` |

### LP Formulation (`pure_lp` / `windowed` / `adaptive`)

```
Variables:  x_{n,e,d} ∈ [0,1]   (net n, directed edge (e,d))
Capacity:   Σ_n (x_{n,e,0} + x_{n,e,1}) ≤ cap(e)  ∀e
Flow cons.: source sends 1, sink receives 1, intermediate nodes conserved
Objective:  min Σ_{n,e} (x_{n,e,0} + x_{n,e,1}) × wire_cost_{n,e}
Solved by:  cuPDLPx (restarted Halpern PDHG, GPU)
```

### Lagrangian Decomposition (`lagrangian`)

```
Relax capacity constraints with multipliers λ_e ≥ 0:
  min  Σ_{n,e} x_{n,e} × (c_{n,e} + λ_e)          <- decomposes per net
  s.t. flow conservation per net

Each net → independent shortest path with augmented cost (c_e + λ_e).
Multiplier update: λ_e ← max(0, λ_e + α_t × (usage_e − cap_e))
Step schedule:     α_t = α_0 / t^decay    (default: sqrt schedule)
```

**GPU backend (`--lag-gpu`)**: all per-net BF SSSPs run concurrently — one CUDA
block per 2-pin net, 256 threads cooperate on that net's bbox subgraph.
Integer costs (`float × 1000 → int32`) enable native `atomicMin`. Lambda
arrays (L×X×Y float) stay on GPU throughout; host transfer only at init and
final path extraction.

---

## Build

### Prerequisites

- GCC ≥ 9, CUDA 12.4+, CMake ≥ 3.20, ninja
- Conda environment: `bash scripts/setup_conda_env.sh && conda activate gagr_dev`

### Build cuPDLPx (once)

```bash
pip install -e cuPDLPx/ --no-build-isolation
```

### Build `router_lp`

```bash
# Default (sm_86 = RTX 3090)
bash scripts/build_router_lp.sh

# Override GPU architecture
CUDA_ARCH=80 bash scripts/build_router_lp.sh   # A100
CUDA_ARCH=86 bash scripts/build_router_lp.sh   # RTX 3090

# Debug build
bash scripts/build_router_lp.sh --debug
```

Output: `router_lp/router_lp`

---

## Usage

```
router_lp -cap <file.cap> -net <file.net> -out <file.out>
          [--mode pure_lp|windowed|adaptive|lagrangian]
          [--max-nets N]       cap multi-pin net count (0 = unlimited)
          [--max-hpwl N]       per-net HPWL filter; same 3D definition across all modes
          [--window-x N]       windowed mode tile width  (default 40)
          [--window-y N]       windowed mode tile height (default 40)
          [--margin N]         bbox expansion for adaptive/lagrangian (default 5)
          [--batch-grid N]     centroid-bin resolution for adaptive (default 50)
          [--lag-iters N]      Lagrangian iterations (default 50)
          [--lag-step F]       initial step size α_0 (default 0.2)
          [--lag-decay F]      step decay exponent (default 0.7)
          [--lag-beta F]       forward-dispersion coefficient (default 0.5)
          [--lag-ema F]        EMA momentum for prev_use smoothing (default 0.3)
          [--lag-gpu]          use GPU parallel BF backend (--mode lagrangian only)
          [--gpu N]            set CUDA_VISIBLE_DEVICES=N
          [--config yaml]      cuPDLPx param file (default: config/cupdlpx_routing_default.yaml)
          [--threshold T]      LP rounding threshold (default 0.1)
          [--no-via]           exclude via edges
          [--inexact-tol T]    accept ITER_LIMIT solutions with primal residual < T (default 1e-3)
```

### Quick examples

```bash
BM=benchmarks/ispd2024/mempool_tile_rank

# Smoke test (small LP, 3 nets)
./router_lp/router_lp -cap $BM.cap -net $BM.net -out /tmp/out.txt \
  --mode pure_lp --max-nets 500 --max-hpwl 30

# Lagrangian, CPU Dijkstra
./router_lp/router_lp -cap benchmarks/ispd2024/ariane133_51.cap \
  -net benchmarks/ispd2024/ariane133_51.net -out /tmp/out.txt \
  --mode lagrangian --max-nets 50 --lag-iters 50 --margin 5

# Lagrangian, GPU parallel BF (~2x faster)
./router_lp/router_lp -cap benchmarks/ispd2024/ariane133_51.cap \
  -net benchmarks/ispd2024/ariane133_51.net -out /tmp/out.txt \
  --mode lagrangian --lag-gpu --max-nets 50 --lag-iters 50 --margin 5

# Windowed LP
./router_lp/router_lp -cap $BM.cap -net $BM.net -out /tmp/out.txt \
  --mode windowed --window-x 40 --window-y 40 --max-nets 200
```

---

## Test & Evaluation

```bash
# Full regression (unit tests + PoC smoke + B1 smoke), ~30s
bash scripts/run_all_tests.sh

# Run on remote GPU server
./connect_gpu_routing_remote.sh --cmd "bash scripts/run_all_tests.sh"

# Evaluate routing quality vs. InstantGR baseline
bash scripts/run_instantgr_baseline.sh
python scripts/eval_routing_result.py results/baseline/mempool_tile_rank/route.out \
  benchmarks/ispd2024/mempool_tile_rank.cap benchmarks/ispd2024/mempool_tile_rank.net

# Subgraph scaling study (sweep bbox sizes + net counts)
bash scripts/run_subgraph_study.sh
```

---

## Benchmark Results

All runs on RTX 3090 (24 GB, sm_86), ariane133_51 benchmark.

### Lagrangian CPU vs GPU — quality parity (`--max-nets 50`, 100 iterations)

After quality fixes (see below), GPU Lagrangian matches CPU Dijkstra exactly:

| Backend | Routed | Disconnected | max_viol | total time |
|---------|--------|-------------|----------|------------|
| CPU Dijkstra | 128/128 | 0 | 5 | ~34 s |
| GPU BF (`--lag-gpu`) | 128/128 | 0 | 5 | ~12 s |

**~3× speedup** with identical solution quality.

### Root cause of the original GPU quality gap (max_viol=104 vs CPU=5)

The gap was caused by **integer BF exact-cost ties → multi-net path clustering → usage stacking**:

1. **Integer quantization** (`COST_SCALE=1000`): many distinct float costs become equal integers.
   Multiple nets independently pick the same minimum-cost path because all ties look identical.

2. **Path clustering**: when 10+ nets share exactly the same cheapest edge sequence, their
   usage vectors all point at the same set of edges.  After the first λ update those edges get
   penalized together, but BF on the next iteration still breaks ties the same way → oscillation.

3. **Subgradient oscillation**: the fixed bias amplifies the oscillation — max_viol bounces
   between ~80 and ~120 instead of converging.

### Three-stage fix (`lagrangian_gpu.cu`)

| Stage | Technique | Effect |
|-------|-----------|--------|
| 1 | **Deterministic eps perturbation** `(ni*7+ei)%11` added to each edge weight | Breaks integer ties: distinct nets now get distinct costs on the same edge |
| 2 | **Forward dispersion term** `β × use_prev/cap` in BF edge cost | Discourages already-congested edges during path selection; reduces initial clustering |
| 3 | **EMA smoothing** `prev_use ← γ·prev_use + (1-γ)·use` (γ=0.3) | Damps 2-period oscillation caused by dispersion alternating between two congestion states |

All three stages share the same formula in `bf_iters_kernel`, `bf_iters_small_kernel`,
`trace_and_use_kernel`, and `lag_gpu_extract_routes` — consistency is required for
the predecessor-match condition `dist[v]+w == d_cur` to hold.

### Bbox bucketing + shared memory optimization

Nets with small bounding boxes (`n_local ≤ 2048 nodes`) use `bf_iters_small_kernel`:
their `dist[]` array is loaded into `__shared__` memory before BF passes begin, and
`atomicMin` operates on shared instead of global memory (~100× lower latency).
Large nets continue using `bf_iters_kernel` with global atomics.

At init, `lag_gpu_init` partitions nets into `d_nets_small` / `d_nets_large` and stores
`max_small_n_local` for the dynamic shared memory allocation.  Each net also stores its
exact BF diameter (`max_bf_iters = bx+by+L`) so blocks run only the required passes
instead of the global maximum.

### Mode comparison on mempool_tile_rank (`--max-nets 500 --max-hpwl 30`)

| Mode | Routed | Disconnected | solve_time |
|------|--------|-------------|------------|
| `pure_lp` | 3 | 0 | 0.42 s |

---

## GPU Backend Design (`lagrangian_gpu.cu`)

```
Per Lagrangian iteration:
  1. reset_int_kernel        -- d_dist[:] = INF
  2. init_src_kernel         -- d_dist[src_li] = 0 per net
  3a. bf_iters_small_kernel  -- small nets (n_local<=2048): dist[] in __shared__,
     └─ one block/net, 256 threads, atomicMin(int) on shared mem; writeback at end
  3b. bf_iters_kernel        -- large nets: same but atomicMin on global d_dist
     └─ each block loops net.max_bf_iters (= bx+by+L) times, not global max
     └─ cost = (base + λ + β*use_prev/cap)*1000 + eps(ni,ei)   [eps breaks int ties]
  4. reset_float_kernel      -- d_use[:] = 0
  5. trace_and_use_kernel    -- snk→src walk, atomicAdd usage, one thread/net
     └─ cost formula must match bf_iters exactly for predecessor matching
  6. update_lam_kernel       -- λ_e ← max(0, λ_e + α*(use-cap))  elementwise
     └─ EMA: use_prev ← γ·use_prev + (1-γ)·use  (γ=0.3, damps oscillation)
  Final (if β>0): one extra clean BF with β=0 for consistent CPU path extraction

Memory on GPU (ariane, --max-nets 100):
  d_dist:    ~39 MB  (packed per-net int arrays)
  λ/use/cap: ~230 MB (3 × 3 arrays × L×X×Y float)
  Total:     ~270 MB  (fits in 24 GB RTX 3090)

Host↔Device transfers:
  Init:       static cap/layer data uploaded once (~120 MB)
  Per 10 iters: 5 arrays downloaded for violation stats (~200 MB total)
  End:        dist + lambda downloaded for CPU path reconstruction
```

---

## Algorithm Reference

- **Pathfinding-based global routing** (DAC 2023): Lagrangian relaxation
  of capacity constraints → per-net independent shortest path subproblems.
- **cuPDLPx**: Restarted Halpern PDHG for LP on GPU (first-order method,
  no matrix factorization).  C API: `create_lp_problem` / `solve_lp_problem`.
- **FLUTE**: Multi-pin → two-pin net decomposition via Steiner minimum tree.

---

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| A | InstantGR baseline harness | Done |
| B0 | cuPDLPx Python PoC | Done |
| B1 | File reader, LP builder, cuPDLPx adapter, rounding | Done |
| B2 | Overflow guards, param sweep, subgraph study | Done |
| B3 | Windowed + adaptive routing modes | Done |
| B3+ | Lagrangian decomposition (CPU Dijkstra) | Done |
| B3+ | Lagrangian GPU parallel BF (`--lag-gpu`) | Done |
| v1  | Inner BF loop (50 kernel launches vs 12800) | Done |
| v1  | GPU quality parity (eps + dispersion + EMA → max_viol matches CPU) | Done |
| v2  | Bbox bucketing + shared memory for small nets (n_local ≤ 8192) | Done |
| v2  | `--lag-iters 0` GPU/CPU semantic parity | Done |
| v3  | Multi-pin output reassembly (orig_name grouping, evaluator-compatible) | Done |
| v3  | Disconnected net repair (Dijkstra λ=0 fallback, `--no-repair`) | Done |
| v3  | cuPDLPx hotspot polish (`--lag-polish`): bbox filter + LP merge | Done |
| v3  | Paper experiment matrix script (`scripts/run_experiment_matrix.sh`) | Done |
| v4  | CUDA Graph batch submission | Planned |

---

## Remote GPU Server

```bash
./connect_gpu_routing_remote.sh              # interactive SSH
./connect_gpu_routing_remote.sh --cmd "..."  # run single command
```

Target: `research23.saas.hku.hk` via proxy `research22.saas.hku.hk`.
