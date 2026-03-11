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
    int    max_iters        = 50;    // number of subgradient iterations
    double step_size        = 0.2;   // initial α_0  (was 0.5; lower = less oscillation)
    double step_decay       = 0.7;   // α_t = α_0 / t^decay  (higher = faster decay)
    int    margin           = 5;     // expand each net's bbox by this many gcells per side
    bool   add_via          = true;  // include via edges in Dijkstra / BF
    double via_cap          = 1.0;   // capacity per via edge
    int    log_every        = 10;    // print violation summary every N iters (0=quiet)
    // Forward dispersion: add beta*(prev_use/cap) to BF edge cost to discourage
    // routing through already-congested edges in the NEXT shortest-path solve.
    // 0.0 disables (backward-compatible); 0.5 is a good starting value.
    double beta_dispersion  = 0.5;
    // EMA momentum for smoothing prev_use across iterations.
    // Prevents 2-period oscillation caused by dispersion alternating congestion states.
    // 0.0 = plain copy (prev_use = current_use); 0.7 = recommended.
    double ema_momentum     = 0.3;
    // L-shape warm-start: run a quick O(1)-per-net L-shape pass before the
    // Lagrangian iterations to warm-start λ multipliers.  This reduces the
    // number of iterations required to converge.
    // init_alpha scale: λ_e = max(0, demand_e - cap_e) * step_size * lshape_alpha.
    bool   lshape_warmstart = true;
    double lshape_alpha     = 0.5;   // multiplier scale for warm-start λ init
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
