#pragma once
// router_lp/include/hotspot_polish.hpp
// cuPDLPx hotspot polishing pipeline: identify overflowed edges, extract the
// subset of nets touching those edges, build a local LP, solve with cuPDLPx,
// and merge improved routes back into the global solution.

#include "routing_types.hpp"
#include "solution_rounding.hpp"
#include <climits>
#include <string>
#include <vector>

namespace rlp {

struct HotspotPolishConfig {
    double overflow_threshold  = 0.0;   // polish edges where (demand - cap) > threshold
    int    max_nets_to_polish  = 200;   // cap hotspot LP size (pre-bbox-filter)
    int    hotspot_bbox_margin = 10;    // expand overflowed-edge bbox by this many gcells
    int    max_hpwl_polish     = INT_MAX; // HPWL cap for LP build; INT_MAX = no filter
    double lp_time_limit_s     = 30.0;
    double inexact_tol         = 1e-3;
    bool   add_via             = true;  // must match --no-via CLI flag
    std::string cupdlpx_config_path;    // path to YAML config (empty = use defaults)
};

struct HotspotPolishStats {
    int    n_overflowed_edges = 0;
    int    n_hotspot_nets     = 0;
    int    n_nets_improved    = 0;
    double viol_before        = 0.0; // max(demand - cap) before polishing
    double viol_after         = 0.0; // max(demand - cap) after polishing
    double polish_time_s      = 0.0;
};

// Main entry point: identify overflow hotspots, build LP, solve, merge routes.
// routes:   in-out, updated for nets that improved under the LP
// twonets:  2-pin nets corresponding to routes (same indexing)
// grid:     routing grid with capacity info
HotspotPolishStats run_hotspot_polish(
    std::vector<NetRoute>& routes,
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const HotspotPolishConfig& cfg);

// Compute max and average edge overflow (demand - cap) from a set of routes.
// max_viol <= 0.0 means fully feasible.
void compute_route_violations(
    const std::vector<NetRoute>& routes,
    const GridInfo& grid,
    double& max_viol_out,
    double& avg_viol_out);

} // namespace rlp
