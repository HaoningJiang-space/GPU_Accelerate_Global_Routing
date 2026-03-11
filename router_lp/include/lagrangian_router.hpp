// router_lp/include/lagrangian_router.hpp
// Lagrangian relaxation global router.
//
// Algorithm (DAC'23 Pathfinding style):
//   Relax capacity constraints with multipliers λ_e ≥ 0.
//   Lagrangian decomposes into n_nets independent min-cost path problems:
//     each net finds shortest path src→snk with edge cost (c_e + λ_e).
//   Multipliers updated by subgradient: λ_e ← max(0, λ_e + α_t*(usage_e - cap_e))
//   Step size schedule: α_t = α_0 / t^decay
//
// Key property: NO monolithic LP matrix — O(n_nets × |subgraph|) per iteration.
// All per-net Dijkstras are independent and GPU-parallelizable (current impl: CPU).

#pragma once
#include "routing_types.hpp"
#include "solution_rounding.hpp"
#include <string>
#include <vector>

namespace rlp {

struct LagrangianConfig {
    int    max_iters   = 50;     // number of subgradient iterations
    double step_size   = 0.5;    // initial α_0
    double step_decay  = 0.5;    // α_t = α_0 / t^decay  (0.5 = sqrt schedule)
    int    margin      = 5;      // expand each net's bbox by this many gcells per side
    bool   add_via     = true;   // include via edges in Dijkstra
    double via_cap     = 1.0;    // capacity per via edge
    int    log_every   = 10;     // print violation summary every N iters (0=quiet)
};

struct LagrangianStats {
    int    iters_run           = 0;
    double final_max_violation = 0.0;  // max_e(usage_e - cap_e), ≤0 means feasible
    double final_avg_violation = 0.0;
    int    n_nets_routed       = 0;
    int    n_nets_disconnected = 0;    // src/snk not reachable within bbox
    double total_dijkstra_time = 0.0;
    double total_update_time   = 0.0;
    double total_wall_time     = 0.0;
};

// Run Lagrangian routing. Returns one NetRoute per 2-pin net.
std::vector<NetRoute> run_lagrangian_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const LagrangianConfig& cfg,
    LagrangianStats& stats);

} // namespace rlp
