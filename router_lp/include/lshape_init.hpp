// router_lp/include/lshape_init.hpp
// L-shape warm-start for Lagrangian routing.
//
// For each 2-pin net, tries both L-shape orientations (H-first and V-first),
// selects the lower-cost route, and accumulates edge demand.  The resulting
// demand imbalance is used to warm-start the Lagrangian multipliers λ before
// the iterative subgradient loop begins, dramatically reducing the number of
// iterations needed to reach a good solution.
//
// Layer 0 (pin-access layer) is skipped for H/V routing, matching the
// evaluator convention.

#pragma once
#include "routing_types.hpp"
#include "solution_rounding.hpp"
#include <vector>

namespace rlp {

// A segment in an L-shape route: one horizontal or vertical run.
struct LShapeSeg {
    int x, y, l;   // starting gcell
    EdgeDir dir;   // H or V
    int len;       // number of edges (always ≥ 1)
};

// Warm-start the λ multipliers from a quick L-shape pass over all 2-pin nets.
//
// Algorithm:
//   For each net: try H-first L (row first, then column) and V-first L
//   (column first, then row) on the preferred layer pair of src/snk.
//   Select the orientation whose route has lower (wire_cost + overflow_penalty).
//   Accumulate demand on H/V edges.
//   After all nets: λ_e ← max(0, (demand_e − cap_e)) × init_alpha.
//
// Parameters:
//   twonets        — 2-pin nets to process
//   grid           — grid topology, capacities, costs
//   init_alpha     — multiplier scale factor (e.g. 0.5 × initial step size)
//   lam_h/v       — output: warm-started λ arrays (indexed l*X*Y + x*Y + y)
//                    Must be pre-allocated and pre-zeroed by caller.
void lshape_warmstart(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    float init_alpha,
    std::vector<float>& lam_h,
    std::vector<float>& lam_v);

} // namespace rlp
