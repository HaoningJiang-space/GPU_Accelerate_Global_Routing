// router_lp/include/lagrangian_gpu.hpp
// GPU-parallel Lagrangian routing via batched frontier Bellman-Ford.
//
// Algorithm:
//   All n_nets Bellman-Ford SSSP computations run concurrently on GPU:
//     one CUDA block per net, threads cooperate on that net's bbox subgraph.
//   Edge costs: integer-scaled (float * COST_SCALE → int32) for atomicMin.
//   Lambda arrays (L*X*Y float each) live on GPU throughout all iterations.
//   Usage arrays updated on GPU by trace_and_use_kernel (one thread per net).
//   Lambda subgradient update runs as a GPU elementwise kernel.
//   Host-device transfer occurs only at: init (static data), final iter (dist+lam).
//
// Speedup rationale:
//   CPU: n_nets sequential Dijkstras × max_iters  (serialized)
//   GPU: all n_nets BF computations in parallel × max_iters  (~100-600x faster)
//
// Limitations / v1 known issues:
//   - No CUDA Graph (planned v2): each iteration launches O(diameter) kernels.
//   - No size bucketing: uniform 256 threads/block regardless of bbox size.
//   - Memory: O(sum(n_local) + 3*L*X*Y) floats/ints on GPU.
//     For ariane --max-nets 100 this is ~350 MB; fits in 24 GB RTX 3090.
//
// Public API: run_lagrangian_routing_gpu()
//   Drop-in replacement for run_lagrangian_routing() when --lag-gpu is set.

#pragma once
#include "routing_types.hpp"
#include "lagrangian_router.hpp"
#include <vector>

namespace rlp {

// Entry point: same signature as CPU version.
std::vector<NetRoute> run_lagrangian_routing_gpu(
    const std::vector<TwoNet>& twonets,
    const GridInfo&            grid,
    const LagrangianConfig&    cfg,
    LagrangianStats&           stats);

} // namespace rlp
