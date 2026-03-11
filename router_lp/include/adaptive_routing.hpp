// router_lp/include/adaptive_routing.hpp
// Adaptive windowed LP routing.
//
// Strategy (option 1+2 combined):
//   1. Group 2-pin nets by their centroid bin (batch_grid_x × batch_grid_y cells).
//   2. For each bin, attempt a joint LP over the union bbox + margin.
//      If the builder pre-flight rejects the batch (too large), fall back to
//      individual per-net solves (each net in its own bbox + margin).
//   3. Every net gets an explicit solve attempt — zero unassigned nets.
//
// The margin expansion ensures nets near tile boundaries have enough subgraph
// context to find a route, equivalent to overlapping-window coverage.

#pragma once
#include "routing_types.hpp"
#include "routing_lp_builder.hpp"
#include "solve_with_cupdlpx.hpp"
#include "solution_rounding.hpp"
#include <string>
#include <vector>

namespace rlp {

struct AdaptiveRoutingConfig {
    int    margin         = 5;    // expand each net/batch bbox by this many gcells per side
    int    batch_grid_x   = 50;   // coarse bin width for grouping nets by centroid
    int    batch_grid_y   = 50;   // coarse bin height for grouping nets by centroid
    int    max_hpwl       = 100000; // per-LP HPWL filter (default: no filter)
    bool   add_via_edges  = true;
    double threshold      = 0.1;
    std::string config_path;
    double inexact_residual_tol = 1e-3;  // accept ITER_LIMIT if primal_res < this
};

struct AdaptiveRoutingStats {
    int n_nets_total       = 0;
    int n_batches          = 0;   // total LP solves attempted (batch + individual)
    int n_batches_solved   = 0;
    int n_batches_inexact  = 0;
    int n_batches_skipped  = 0;   // oversize even as single-net
    int n_nets_routed      = 0;
    int n_nets_disconnected= 0;
    double total_build_time = 0.0;
    double total_solve_time = 0.0;
    double total_round_time = 0.0;
    double total_wall_time  = 0.0;
};

// Run adaptive LP routing.  Returns one NetRoute per input 2-pin net.
std::vector<NetRoute> run_adaptive_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const AdaptiveRoutingConfig& cfg,
    AdaptiveRoutingStats& stats);

} // namespace rlp
