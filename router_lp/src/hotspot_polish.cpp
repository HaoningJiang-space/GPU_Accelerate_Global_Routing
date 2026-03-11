// router_lp/src/hotspot_polish.cpp
// cuPDLPx hotspot polishing: find overflowed edges, extract hotspot nets,
// build local LP, solve with cuPDLPx, merge improved routes back.

#include "../include/hotspot_polish.hpp"
#include "../include/routing_lp_builder.hpp"
#include "../include/solve_with_cupdlpx.hpp"
#include "../include/solution_rounding.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace rlp {

// ── Flat edge-index helpers (match lagrangian_router convention) ──────────────
// H-edge (l,x,y): l*X*Y + x*Y + y  (edge goes to (x+1,y,l))
// V-edge (l,x,y): l*X*Y + x*Y + y  (edge goes to (x,y+1,l))
// Via    (l,x,y): l*X*Y + x*Y + y  (edge goes to (x,y,l+1))
// Separate arrays for H, V, Via so indices don't collide.

static inline int eidx(int l, int x, int y, int X, int Y) {
    return l * X * Y + x * Y + y;
}

// ── Step 1: Compute per-edge demand from current routes ───────────────────────

struct DemandArrays {
    std::vector<double> h;   // [L*X*Y], H-edge demand
    std::vector<double> v;   // [L*X*Y], V-edge demand
    std::vector<double> via; // [L*X*Y], via demand
};

static DemandArrays compute_demand(
    const std::vector<NetRoute>& routes,
    const GridInfo& grid)
{
    const int L = grid.L, X = grid.X, Y = grid.Y;
    DemandArrays d;
    d.h  .assign(L * X * Y, 0.0);
    d.v  .assign(L * X * Y, 0.0);
    d.via.assign(L * X * Y, 0.0);

    for (const auto& route : routes) {
        for (const auto& seg : route.segments) {
            // Normalise: ensure (x1,y1,z1) ≤ (x2,y2,z2) by component
            int x1 = seg.x1, y1 = seg.y1, z1 = seg.z1;
            int x2 = seg.x2, y2 = seg.y2, z2 = seg.z2;
            if (x1 == x2 && y1 == y2) {
                // Via segment: l from min(z1,z2) to max(z1,z2)-1
                int lmin = std::min(z1, z2);
                int lmax = std::max(z1, z2);
                for (int l = lmin; l < lmax; ++l) {
                    if (l >= 0 && l < L - 1 && x1 >= 0 && x1 < X && y1 >= 0 && y1 < Y)
                        d.via[eidx(l, x1, y1, X, Y)] += 1.0;
                }
            } else if (y1 == y2 && z1 == z2) {
                // H segment
                int xmin = std::min(x1, x2);
                int xmax = std::max(x1, x2);
                for (int x = xmin; x < xmax; ++x) {
                    int l = z1;
                    if (l >= 0 && l < L && x >= 0 && x < X - 1 && y1 >= 0 && y1 < Y)
                        d.h[eidx(l, x, y1, X, Y)] += 1.0;
                }
            } else if (x1 == x2 && z1 == z2) {
                // V segment
                int ymin = std::min(y1, y2);
                int ymax = std::max(y1, y2);
                for (int y = ymin; y < ymax; ++y) {
                    int l = z1;
                    if (l >= 0 && l < L && x1 >= 0 && x1 < X && y >= 0 && y < Y - 1)
                        d.v[eidx(l, x1, y, X, Y)] += 1.0;
                }
            }
            // Diagonal or same-cell segments are not expected; skip silently.
        }
    }
    return d;
}

// ── Step 2: Compute max violation from demand arrays ─────────────────────────

static double max_violation(const DemandArrays& d, const GridInfo& grid) {
    const int L = grid.L, X = grid.X, Y = grid.Y;
    double mv = 0.0;
    for (int l = 0; l < L; ++l) {
        if (grid.layer_dir[l] == 0) { // H layer
            for (int x = 0; x < X - 1; ++x)
                for (int y = 0; y < Y; ++y)
                    mv = std::max(mv, d.h[eidx(l,x,y,X,Y)] - grid.cap[l][x][y]);
        } else { // V layer
            for (int x = 0; x < X; ++x)
                for (int y = 0; y < Y - 1; ++y)
                    mv = std::max(mv, d.v[eidx(l,x,y,X,Y)] - grid.cap[l][x][y]);
        }
    }
    return mv;
}

// ── Step 3: Identify overflowed edges and the nets touching them ──────────────
//
// Also computes the tight bbox of all overflowed edge grid-cells so callers can
// restrict the LP subgraph to the congested region (avoiding a full-grid bbox
// that would explode n_vars and fail the pre-flight size check).

struct OverflowBbox {
    int xmin = INT_MAX, xmax = INT_MIN;
    int ymin = INT_MAX, ymax = INT_MIN;
    bool empty() const { return xmin > xmax; }
};

static std::vector<int> find_hotspot_nets(
    const std::vector<NetRoute>& routes,
    const DemandArrays& d,
    const GridInfo& grid,
    double overflow_threshold,
    int& n_overflowed_out,
    int max_nets,
    OverflowBbox& ovf_bbox)
{
    const int L = grid.L, X = grid.X, Y = grid.Y;

    std::set<long long> overflowed_h, overflowed_v;
    n_overflowed_out = 0;
    ovf_bbox = OverflowBbox{}; // reset to default (INT_MAX/INT_MIN)

    for (int l = 0; l < L; ++l) {
        if (grid.layer_dir[l] == 0) {
            for (int x = 0; x < X - 1; ++x)
                for (int y = 0; y < Y; ++y) {
                    int idx = eidx(l,x,y,X,Y);
                    if (d.h[idx] > grid.cap[l][x][y] + overflow_threshold) {
                        overflowed_h.insert(idx);
                        ++n_overflowed_out;
                        ovf_bbox.xmin = std::min(ovf_bbox.xmin, x);
                        ovf_bbox.xmax = std::max(ovf_bbox.xmax, x + 1);
                        ovf_bbox.ymin = std::min(ovf_bbox.ymin, y);
                        ovf_bbox.ymax = std::max(ovf_bbox.ymax, y);
                    }
                }
        } else {
            for (int x = 0; x < X; ++x)
                for (int y = 0; y < Y - 1; ++y) {
                    int idx = eidx(l,x,y,X,Y);
                    if (d.v[idx] > grid.cap[l][x][y] + overflow_threshold) {
                        overflowed_v.insert(idx);
                        ++n_overflowed_out;
                        ovf_bbox.xmin = std::min(ovf_bbox.xmin, x);
                        ovf_bbox.xmax = std::max(ovf_bbox.xmax, x);
                        ovf_bbox.ymin = std::min(ovf_bbox.ymin, y);
                        ovf_bbox.ymax = std::max(ovf_bbox.ymax, y + 1);
                    }
                }
        }
    }

    if (overflowed_h.empty() && overflowed_v.empty()) return {};

    // For each net, count how many overflowed edges it touches
    std::vector<std::pair<int,int>> net_overlap; // (overlap_count, net_index)
    net_overlap.reserve(routes.size());

    for (int i = 0; i < (int)routes.size(); ++i) {
        int overlap = 0;
        for (const auto& seg : routes[i].segments) {
            int x1 = seg.x1, y1 = seg.y1, z1 = seg.z1;
            int x2 = seg.x2, y2 = seg.y2; //, z2 = seg.z2;
            if (y1 == y2 && z1 == seg.z2) {
                // H segment
                int xmin = std::min(x1, x2), xmax = std::max(x1, x2);
                for (int x = xmin; x < xmax; ++x) {
                    if (!overflowed_h.empty() &&
                        overflowed_h.count(eidx(z1, x, y1, X, Y)))
                        ++overlap;
                }
            } else if (x1 == x2 && z1 == seg.z2) {
                // V segment
                int ymin = std::min(y1, y2), ymax = std::max(y1, y2);
                for (int y = ymin; y < ymax; ++y) {
                    if (!overflowed_v.empty() &&
                        overflowed_v.count(eidx(z1, x1, y, X, Y)))
                        ++overlap;
                }
            }
        }
        if (overlap > 0)
            net_overlap.push_back({overlap, i});
    }

    // Sort descending by overlap count
    std::sort(net_overlap.begin(), net_overlap.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    // Take up to max_nets
    if ((int)net_overlap.size() > max_nets)
        net_overlap.resize(max_nets);

    std::vector<int> result;
    result.reserve(net_overlap.size());
    for (auto& [cnt, idx] : net_overlap) result.push_back(idx);
    return result;
}

// ── Main entry point ──────────────────────────────────────────────────────────

HotspotPolishStats run_hotspot_polish(
    std::vector<NetRoute>& routes,
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const HotspotPolishConfig& cfg)
{
    auto t0 = std::chrono::steady_clock::now();
    HotspotPolishStats stats;

    if (routes.size() != twonets.size()) {
        std::cerr << "[polish] routes/twonets size mismatch\n";
        return stats;
    }

    // Step 1: compute current demand
    DemandArrays d = compute_demand(routes, grid);
    stats.viol_before = max_violation(d, grid);

    // Step 2+3: find overflowed edges and hotspot nets
    OverflowBbox ovf_bbox;
    std::vector<int> hotspot_indices = find_hotspot_nets(
        routes, d, grid, cfg.overflow_threshold,
        stats.n_overflowed_edges, cfg.max_nets_to_polish, ovf_bbox);

    stats.n_hotspot_nets = (int)hotspot_indices.size(); // pre-bbox-filter count

    if (hotspot_indices.empty()) {
        stats.viol_after = stats.viol_before;
        stats.polish_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return stats;
    }

    // Step 4: build LP for hotspot nets
    // Only include nets whose src AND snk both lie within the overflow bbox +
    // margin.  This prevents the LP from spanning the full grid when congested
    // nets happen to have large bounding boxes (which would explode n_vars).
    const int margin = cfg.hotspot_bbox_margin;
    const int bx0 = std::max(0, ovf_bbox.xmin - margin);
    const int bx1 = std::min(grid.X - 1, ovf_bbox.xmax + margin);
    const int by0 = std::max(0, ovf_bbox.ymin - margin);
    const int by1 = std::min(grid.Y - 1, ovf_bbox.ymax + margin);

    // kept_indices[j] == routes[] index for hotspot_twonets[j]; must stay in sync.
    std::vector<TwoNet> hotspot_twonets;
    std::vector<int>    kept_indices;
    hotspot_twonets.reserve(hotspot_indices.size());
    kept_indices.reserve(hotspot_indices.size());
    for (int idx : hotspot_indices) {
        const TwoNet& tn = twonets[idx];
        bool src_in = tn.src.loc.x >= bx0 && tn.src.loc.x <= bx1 &&
                      tn.src.loc.y >= by0 && tn.src.loc.y <= by1;
        bool snk_in = tn.snk.loc.x >= bx0 && tn.snk.loc.x <= bx1 &&
                      tn.snk.loc.y >= by0 && tn.snk.loc.y <= by1;
        if (src_in && snk_in) {
            hotspot_twonets.push_back(tn);
            kept_indices.push_back(idx);
        }
    }

    if (hotspot_twonets.empty()) {
        std::cerr << "[polish] All hotspot nets lie outside overflow bbox+margin, skipping LP\n";
        stats.viol_after = stats.viol_before;
        stats.polish_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return stats;
    }

    LPBuilderConfig bcfg;
    bcfg.add_via_edges = cfg.add_via; // honour --no-via flag
    bcfg.max_hpwl      = cfg.max_hpwl_polish; // INT_MAX by default = truly no HPWL filter
    bcfg.margin        = 0;

    // Update stats to reflect the post-bbox-filter count that actually enters LP.
    stats.n_hotspot_nets = (int)kept_indices.size();

    RoutingLPProblem prob;
    if (!build_routing_lp(hotspot_twonets, grid, bcfg, prob)) {
        std::cerr << "[polish] build_routing_lp failed for hotspot nets\n";
        stats.viol_after = stats.viol_before;
        stats.polish_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return stats;
    }

    std::cout << "[polish] LP: n_vars=" << prob.n_vars
              << " n_cons=" << prob.n_cons
              << " n_nets=" << prob.n_nets << "\n";

    // Step 5: solve LP with cuPDLPx, passing time_limit_s / inexact_tol overrides
    RoutingLPSolution sol;
    bool solved = solve_routing_lp(prob, cfg.cupdlpx_config_path, sol,
                                   cfg.lp_time_limit_s, cfg.inexact_tol);

    if (!solved) {
        std::cerr << "[polish] cuPDLPx did not return OPTIMAL "
                     "(status=" << (int)sol.status << "), keeping original routes\n";
        stats.viol_after = stats.viol_before;
        stats.polish_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        return stats;
    }

    // Step 6: extract routes from LP solution
    std::vector<NetRoute> lp_routes = round_lp_solution(prob, sol, /*threshold=*/0.1);

    // Step 7: merge back — only replace if LP found a non-empty path.
    // kept_indices[j] is the routes[] index for lp_routes[j].
    for (int j = 0; j < (int)kept_indices.size() && j < (int)lp_routes.size(); ++j) {
        if (!lp_routes[j].segments.empty()) {
            int orig_idx = kept_indices[j];
            // Preserve orig_name from the original route
            lp_routes[j].orig_name = routes[orig_idx].orig_name;
            routes[orig_idx] = std::move(lp_routes[j]);
            ++stats.n_nets_improved;
        }
    }

    // Step 8: recompute demand and max violation after merge
    DemandArrays d_after = compute_demand(routes, grid);
    stats.viol_after = max_violation(d_after, grid);

    stats.polish_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return stats;
}

// ── Public violation helper ───────────────────────────────────────────────────

void compute_route_violations(
    const std::vector<NetRoute>& routes,
    const GridInfo& grid,
    double& max_viol_out,
    double& avg_viol_out)
{
    DemandArrays d = compute_demand(routes, grid);
    const int L = grid.L, X = grid.X, Y = grid.Y;
    double max_v = 0.0, sum_v = 0.0;
    int n_over = 0;
    for (int l = 0; l < L; ++l) {
        if (grid.layer_dir[l] == 0) {
            for (int x = 0; x < X - 1; ++x)
                for (int y = 0; y < Y; ++y) {
                    double v = d.h[eidx(l,x,y,X,Y)] - grid.cap[l][x][y];
                    if (v > 0) { max_v = std::max(max_v, v); sum_v += v; ++n_over; }
                }
        } else {
            for (int x = 0; x < X; ++x)
                for (int y = 0; y < Y - 1; ++y) {
                    double v = d.v[eidx(l,x,y,X,Y)] - grid.cap[l][x][y];
                    if (v > 0) { max_v = std::max(max_v, v); sum_v += v; ++n_over; }
                }
        }
    }
    max_viol_out = max_v;
    avg_viol_out = (n_over > 0) ? sum_v / n_over : 0.0;
}

} // namespace rlp
