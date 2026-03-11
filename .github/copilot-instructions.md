# Copilot Instructions


## Language and Format Requirements

**Language Rules:**
- Use Chinese for all interactions and discussions with the user
- Professional terms may be kept in English when appropriate
- All printed content in code should be in English
- Add appropriate Chinese and English comments in generated code

**Format Rules:**
- No emojis allowed in any files or code (except markdown files)
- Use clear, professional formatting

## Work Principles

You must follow these three-tier principles based on task complexity:

**How to Determine Task Tier:**
- **Tier 1**: Changes ≤ 5 lines of code, single file, clear requirements
- **Tier 2**: Changes > 5 lines of code OR multiple files OR unclear technical approach
- **Tier 3**: Affects core architecture OR multiple modules OR new feature addition

**Tier 1 - Simple Tasks (≤ 5 lines):**
- When given a clear, single, simple task requiring ≤ 5 lines of code changes
- Execute code modifications directly
- Provide code with appropriate comments

**Tier 2 - Complex Tasks (> 5 lines):**
- When the task requires > 5 lines of code OR involves multiple components OR when a technical solution is proposed
- DO NOT generate code immediately
- Instead, discuss the specific technical approach with the user
- Refine the user's proposal and identify unclear technical details
- Generate a complete engineering plan only after clarification
- Your engineering plan must include:
  - Custom function names with input/output specifications
  - Data flow between different functions
  - File architecture updates if needed
- Use thinking/megathink/ultrathink modes based on difficulty
- Only generate code after the user approves the plan

**Tier 3 - Project-Wide Changes:**
- For requirements affecting the entire project rather than single modules/scripts
- Follow Tier 2 requirements PLUS:
- Generate a task documentation markdown file to track and record work
- Synchronously update the task documentation during coding
- Use thinking/megathink/ultrathink modes based on difficulty

**Response Guidelines:**
- Use clear, concise language in Chinese for discussions
- Focus on delivering the specific output requested (code/discussion/plan)
- Avoid unnecessary explanatory details unless asked

## Coding Principles

Follow "Progressive Complexity" principle: start with the simplest working solution, add complexity only when necessary.

**Core Guidelines:**

1. **Simplicity First (KISS)**
   - Use clear, intuitive variable and function names
   - Prefer built-in language features over custom implementations
   - Each function should focus on a single task
   - Avoid nesting beyond 3 levels of conditions or loops

2. **Implement Only What's Needed (YAGNI)**
   - Implement only currently required functionality
   - Don't add "might be useful" parameters or configurations
   - Refuse "just in case" code branches
   - When requirements are unclear, stop code generation and ask for specifics

3. **Structured Design (SOLID-Lite)**
   - Each module/class handles one clear functional domain
   - Pass dependencies through parameters, not hard-coding
   - Consider abstraction only with 3+ similar use cases or explicit user request
   - Keep interfaces minimal, expose only necessary methods

4. **Fail-Fast (Early Error Detection)**
   - Detect and report errors immediately when they occur
   - Do not attempt to continue execution with invalid states
   - Provide clear error messages indicating what went wrong
   - Function parameters should minimize default values
     - Core functions: No default values for required parameters
     - Utility functions: Reasonable defaults for UI/plotting parameters are acceptable
     - Use type hints (Optional[T]) to clearly mark optional parameters

**Code Review Checklist:**
- Can the same functionality be achieved with less code?
- Are there unused code segments or parameters?
- Does each function do only one thing?
- Are dependencies clear and minimized?
- Are errors detected and reported immediately? (Fail-Fast)
- Do required parameters lack default values?
- Are type hints used to clearly mark optional parameters?

## Handling Uncertainty

When facing unclear requirements or technical questions:

**Decision Making Process:**
1. **Never guess** - Do not assume requirements or make arbitrary technical decisions
2. **Ask clarifying questions** - Use clear, specific language to identify ambiguities
3. **Propose alternatives** - When multiple approaches exist, present 2-3 options with trade-offs
4. **Consult documentation** - Use MCP context7 for API references rather than relying on knowledge base
5. **Fail explicitly** - If a task cannot be completed, explain why clearly and what information is needed

**When to Stop and Ask:**
- Requirements are ambiguous or contradictory
- Multiple valid technical approaches exist without clear best choice
- Missing critical information (file paths, configuration values, etc.)
- User's request conflicts with existing code patterns or architecture

## Development Guidelines

### API Consultation
When consulting APIs, use the MCP extension context7 rather than your knowledge base. If this MCP service is unavailable, explicitly mention this in your response.


## Project Overview

This project builds a new GPU global routing prototype (`router_lp/`) using **cuPDLPx** (GPU-accelerated first-order LP solver, restarted Halpern PDHG) as the core optimization backend. **InstantGR** serves only as a baseline and algorithmic reference — its source code is never called or linked into the new prototype. You can refer 

Goal: reformulate global routing as an LP (continuous relaxation of ILP with edge-flow variables, flow conservation, and capacity constraints), solve on GPU via cuPDLPx, then recover a discrete routing solution.

**Current status:** Design phase. `router_lp/` not yet created. Implementation roadmap: `IMPLEMENTATION_PLAN_GPU_ROUTING_CUPDLPX.md`. Technical analysis: `gpu_global_routing_cupdlpx_analysis.md` (both in Chinese) , please refer them frequently.

## Build Commands

### InstantGR (CUDA/C++)

Build from `InstantGR/src/`:
```bash
nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_80
```
Adjust `-arch` for your GPU (e.g., `sm_86` for RTX 3090, `sm_80` for A100).

Run routing on a benchmark:
```bash
cd InstantGR/run
./InstantGR ../benchmarks/mempool_tile_rank.cap ../benchmarks/mempool_tile_rank.net
```

Evaluate routing quality:
```bash
# Compile evaluator
nvcc evaluator.cpp -o evaluator -std=c++17
./evaluator <cap_file> <net_file> <route_output>
```

### cuPDLPx (CUDA/C/Python)

Build Python bindings (from `cuPDLPx/`):
```bash
pip install -e . --no-build-isolation
```
Requires CMake ≥ 3.20, CUDA 12.4+, ninja.

Run all tests:
```bash
cd cuPDLPx && pytest
```

Run a single test:
```bash
cd cuPDLPx && pytest tests/test_<name>.py::test_<function>
```

### Environment Setup

```bash
bash scripts/setup_conda_env.sh
conda activate gagr_dev
```
Environment spec: `envs/gagr_dev.yml` (Python 3.11, CMake ≥ 3.20, numpy/scipy/networkx/pytest).

### Remote GPU Server

```bash
./connect_gpu_routing_remote.sh              # interactive login
./connect_gpu_routing_remote.sh --cmd "..."  # run a single command
```
Target: `research23.saas.hku.hk` via proxy `research22.saas.hku.hk`, workspace `/home/ynwang/jhn/EDA/GPU_Accelerate_Global_Routing`.

## Architecture

### Component Map

```
InstantGR/          — GPU global router; baseline reference ONLY (never linked)
cuPDLPx/            — GPU LP solver (C/CUDA + Python bindings); core backend
router_lp/          — new routing prototype (to be created); owns all new code
  include/          —   RoutingLPProblem, RoutingLPSolution IR
  src/              —   LP construction, cuPDLPx adapter, solution rounding
  config/           —   cupdlpx_routing_default.yaml
  examples/         —   min_route_lp_with_cupdlpx.cpp (PoC)
  main.cpp          —   entry point with --mode switch
envs/               — conda environment spec
scripts/            — environment setup helpers
results/baseline/   — InstantGR baseline outputs (per case)
benchmarks/         — routing test cases + local_suite_manifest.yaml
docs/               — metrics_definition.md, C API docs
```

**Critical constraint:** Do not modify `InstantGR/` source code. Do not call or link `InstantGR` code from `router_lp/`. Use only InstantGR's output files for baseline comparison.

### InstantGR Internals (`InstantGR/src/`)

All major modules are **header-only** (`.hpp`) to allow single-pass NVCC compilation:

| File | Role |
|------|------|
| `main.cpp` | Entry point; reads `.cap`/`.net` files |
| `database.hpp` | CPU-side net/layer/pin data structures |
| `database_cuda.hpp` | GPU-side database; RSMT/Steiner tree construction via FLUTE |
| `graph.hpp` | CUDA kernels: edge cost, demand accumulation, overflow detection |
| `Lshape_route.hpp` | Phase 1: initial L-shaped routing for all nets |
| `Lshape_route_detour.hpp` | Phase 2: detour-based rip-up & reroute |
| `flute.hpp` | FLUTE library (multi-pin → two-pin Steiner decomposition) |

**Three-phase routing flow:**
1. Decompose multi-pin nets into two-pin segments via FLUTE RSMT on GPU
2. Initial L-shaped routing (greedy)
3. Congestion detection → detour rip-up & reroute for overflow nets

**Cost model:**
- Wire cost: edge length × layer unit cost
- Via cost: exponential penalty for layer transitions
- Overflow cost: `exp(0.5 × (demand − capacity))`

### cuPDLPx Internals (`cuPDLPx/src/`)

Restarted Halpern PDHG for LP, four-stage pipeline:

| Stage | File | Where |
|-------|------|--------|
| Presolve | `presolve.c` | CPU |
| Rescaling (Ruiz + Pock-Chambolle) | `utils.cu` | GPU |
| PDHG iterations | `solver.cu` | GPU |
| Feasibility polishing | `feasibility_polish.cu` | GPU |

Sparse matrices use CSR/CSC/COO formats. Python API lives in `cuPDLPx/python/cupdlpx/`.

C API entry points: `create_lp_problem()` / `solve_lp_problem()` — see `cuPDLPx/docs/C_API.md`.

### Planned `router_lp/` Architecture

New prototype entry point (`router_lp/main.cpp`) with three modes:
- `--mode pure_lp` — LP construction → cuPDLPx solve → rounding → repair
- `--mode lagrangian_path` — Pathfinding-style: per-net shortest path + Lagrange multiplier updates
- `--mode hybrid` — pricing/path pool + cuPDLPx master LP

**LP problem structure:**
- Variables: edge-flow `f_{n,e} ∈ [0,1]` (continuous relaxation)
- Constraints: flow conservation (source/sink/intermediate), edge capacity
- Objective: `wire_cost + λ × overflow_slack`

Key files to create:
- `router_lp/include/`: `RoutingLPProblem` (variable/constraint indexing, CSR builder), `RoutingLPSolution` (x, dual, status)
- `router_lp/src/to_cupdlpx.cpp`: IR → cuPDLPx CSR + bounds
- `router_lp/src/solve_with_cupdlpx.cpp`: calls `create_lp_problem` / `solve_lp_problem`
- `router_lp/src/solution_rounding.cpp`: fractional → integer routing recovery
- `router_lp/src/cupdlpx_param_loader.cpp`: loads YAML config

**Fallback chain:** `pure_lp` timeout/failure → `lagrangian_path` or `repair-only`. Always ensure the pipeline completes.

**cuPDLPx default parameters for routing** (`router_lp/config/cupdlpx_routing_default.yaml`):
```
presolve: true
termination_evaluation_frequency: 200
eps_opt: 1e-4
eps_feas: 1e-4
time_limit: 30–120s  # scale with subgraph size
feasibility_polishing: true  # post-processing only
```

**Hotspot subgraph strategy:** Apply LP only to overflow hotspot subgraphs first (not the full routing graph) to avoid variable/constraint explosion. Borrow the congestion view concept from InstantGR but implement data structures independently.

## Git Conventions

Branch naming: `codex/phase-<x>-<description>` (e.g., `codex/phase-a-baseline-harness`).

Commit type prefixes: `feat:` / `refactor:` / `test:` / `docs:` — one type per commit, no mixed changes.

Commit template:
```
<type>(<scope>): <summary>

What:
- ...

Why:
- ...

Validation:
- command: ...
- result: ...
```

Pre-commit checklist: compiles, at least one small case runs end-to-end, key metrics logged (CSV or log file), fallback switch present.

## Key Conventions

### CUDA Patterns (Both Components)

- Prefer `atomicAdd` for concurrent demand updates on GPU
- Use dynamic shared memory for prefix sums / reductions
- Batch-process nets to maximize GPU occupancy
- Kernels follow `__global__ void kernel_name(args)` naming with matching host wrappers

### InstantGR Conventions

- All implementation in `.hpp` headers; `main.cpp` is a thin driver
- GPU data structures live in `database_cuda.hpp`; CPU structures in `database.hpp`
- Routing files (`.cap`, `.net`) are custom text formats; see `benchmarks/` for examples
- CUDA architecture must be set explicitly at compile time (`-arch=sm_XX`)

### cuPDLPx Conventions

- C code follows clang-format style (`.clang-format` in `cuPDLPx/`)
- CI checks: `check-clang-format.yml` (formatting), `build.yml` (build + tests), `spelling.yml`
- 80% test coverage minimum enforced by `pytest.ini`
- Internal headers live in `cuPDLPx/internal/`; public API only in `cuPDLPx/include/`
- Python bindings are in `cuPDLPx/python_bindings/_core_bindings.cpp`

### Formal Problem Definition (from reference papers)

**Grid graph:** Undirected G(V,E). Nodes = G-Cells, edges = adjacent cell connections. Each edge e has `cap(e)`. Directed version D(V,E): each undirected edge e → two directed edges ê+ (left/bottom→right/top) and ê- (reverse).

**ILP (Pathfinding DAC'23, also RUPlace DAC'25):**
```
Variables:  x_{n,ê} ∈ {0,1}   (net n uses directed edge ê)
            (LP relaxation: x_{n,ê} ∈ [0,1])

Capacity:   Σ_n (x_{n,ê+} + x_{n,ê-}) ≤ cap(e),  ∀e ∈ E

Flow cons.: Σ_{ê ∈ E+(so(n))} x_{n,ê} = 1          (source sends out 1)
            Σ_{ê ∈ E-(so(n))} x_{n,ê} = 0           (source receives 0)
            Σ_{ê ∈ E-(si(n))} x_{n,ê} = 1            (sink receives 1)
            Σ_{ê ∈ E+(si(n))} x_{n,ê} = 0            (sink sends out 0)
            Σ_{ê ∈ E-(v)} x_{n,ê} = Σ_{ê ∈ E+(v)} x_{n,ê},  ∀v ∉ {so(n),si(n)}

Objective:  min Σ_{n,e} (x_{n,ê+} + x_{n,ê-}) × c_{n,e}
```
Multi-pin nets are first decomposed into 2-pin nets via FLUTE (same as InstantGR).

**Lagrangian decomposition (used in `lagrangian_path` mode):**
- Relax capacity constraint into objective with multipliers μ_e ≥ 0
- New edge cost: `C_{n,e} = c_{n,e} + μ_e`
- Problem decomposes per net → independent shortest-path for each net
- Update multipliers via gradient ascent: `μ_e^{k+1} = max(0, μ_e^k + α × ∇f(μ_e^k))`
- Gradient: `∇f(μ_e) = Σ_n (x_{n,ê+} + x_{n,ê-}) - cap(e)` (demand minus capacity)
- Solve each net with direction-aware weighted A\* (DAWA\*)

**`pure_lp` mode:** feed LP relaxation directly to cuPDLPx (no Lagrangian decomposition). Suitable for small subgraphs. Variable indexing: directed edge `(e, dir)` for net `n` → var index `n * 2 * |E_sub| + e * 2 + dir`.
