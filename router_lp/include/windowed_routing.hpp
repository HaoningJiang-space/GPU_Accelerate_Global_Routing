// router_lp/include/windowed_routing.hpp
// Windowed LP routing: tile the GR grid into WX×WY windows, solve one LP
// per window independently, then aggregate routes.
//
// Each window LP is bounded to n_nets_window × n_edges_window × 2 variables,
// keeping problem sizes manageable without changing the LP formulation.
//
// Nets whose src and snk both fall inside a window are assigned to that window.
// Nets spanning multiple windows are left unrouted (marked as disconnected).

#pragma once
#include "routing_types.hpp"
#include "routing_lp_builder.hpp"
#include "solve_with_cupdlpx.hpp"
#include "solution_rounding.hpp"
#include <string>
#include <vector>

namespace rlp {

struct WindowedRoutingConfig {
    int    window_x    = 40;    // window width  in GCells
    int    window_y    = 40;    // window height in GCells
    int    max_hpwl    = 10000; // per-window HPWL filter (default: no filter)
    bool   add_via_edges = true;
    double threshold   = 0.1;   // rounding threshold
    std::string config_path;    // cuPDLPx YAML param file
    // Round even on ITER_LIMIT if absolute_primal_residual < this tolerance.
    double inexact_residual_tol = 1e-3;
};

struct WindowedRoutingStats {
    int n_windows_total     = 0;
    int n_windows_solved    = 0;  // reached OPTIMAL
    int n_windows_inexact   = 0;  // rounded from ITER_LIMIT/TIME_LIMIT with small residual
    int n_windows_skipped   = 0;  // oversize or hard solver failure
    int n_nets_total        = 0;
    int n_nets_routed       = 0;
    int n_nets_disconnected = 0;
    int n_nets_unassigned   = 0;  // nets that span multiple windows
    double total_build_time  = 0.0;
    double total_solve_time  = 0.0;
    double total_round_time  = 0.0;
    double total_wall_time   = 0.0;
};

// Run windowed LP routing.
// Returns routes for all 2-pin nets assigned to windows (one NetRoute per net).
// Stats are populated with per-window and aggregate counts.
std::vector<NetRoute> run_windowed_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const WindowedRoutingConfig& cfg,
    WindowedRoutingStats& stats);

} // namespace rlp
