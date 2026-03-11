#pragma once
// router_lp/include/routing_lp_builder.hpp
// Converts a set of TwoNets + GridInfo into a RoutingLPProblem.
//
// Algorithm:
//   1. Compute bounding box of all nets to form subgraph.
//   2. Enumerate edges in subgraph (H, V, Via edges).
//   3. Build variable indexing: var = net * 2 * n_edges + edge * 2 + dir
//   4. Build capacity constraints (one per undirected subgraph edge).
//   5. Build flow conservation constraints (one per (net, subgraph node)).
//   6. Set objective: sum of (x_{n,e,fwd} + x_{n,e,bwd}) * wire_cost(e) for each net,edge.
//      Via edges add unit_via_cost instead of wire cost.

#include "routing_types.hpp"
#include "routing_lp_problem.hpp"
#include <vector>

namespace rlp {

struct LPBuilderConfig {
    int   max_hpwl  = 40;   // skip nets with HPWL > this (gcell units, sum of x+y+z span)
    bool  add_via_edges  = true;  // include layer-change (via) edges
};

// Build the LP problem from a set of 2-pin nets and grid info.
// Returns false if no valid nets or subgraph is empty.
bool build_routing_lp(const std::vector<TwoNet>& nets,
                      const GridInfo& grid,
                      const LPBuilderConfig& cfg,
                      RoutingLPProblem& prob);

} // namespace rlp
