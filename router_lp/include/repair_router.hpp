#pragma once
// router_lp/include/repair_router.hpp
// Re-route disconnected 2-pin nets using Dijkstra with zero lambda (no capacity
// penalty). Acts as a greedy fallback that guarantees connectivity at the cost
// of potentially adding demand to already-congested edges.

#include "routing_types.hpp"
#include "solution_rounding.hpp"
#include <vector>

namespace rlp {

// Re-route disconnected nets with lambda=0 (ignores capacity, greedy fallback).
// Modifies routes in-place: replaces empty NetRoute entries with found paths.
// Returns the number of nets successfully repaired.
int repair_disconnected_nets(
    std::vector<NetRoute>& routes,
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    bool add_via = true);

} // namespace rlp
