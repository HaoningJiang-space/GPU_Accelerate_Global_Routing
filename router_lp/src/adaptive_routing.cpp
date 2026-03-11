// router_lp/src/adaptive_routing.cpp
// Adaptive windowed LP routing — see include/adaptive_routing.hpp.

#include "../include/adaptive_routing.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <utility>

namespace rlp {

// ── helpers ──────────────────────────────────────────────────────────────────

// Attempt one LP solve+round for a set of nets; populate result_routes (one
// entry per net in `batch`).  Returns true if solve+round was attempted.
// can_round is set when rounding was performed (OPTIMAL or near-feasible).
static bool solve_batch(
    const std::vector<TwoNet>& batch,
    const GridInfo& grid,
    const AdaptiveRoutingConfig& cfg,
    AdaptiveRoutingStats& stats,
    std::vector<NetRoute>& result_routes)
{
    LPBuilderConfig bcfg;
    bcfg.max_hpwl      = cfg.max_hpwl;
    bcfg.add_via_edges = cfg.add_via_edges;
    bcfg.margin        = cfg.margin;

    RoutingLPProblem prob;
    auto t_b0 = std::chrono::steady_clock::now();
    bool built = build_routing_lp(batch, grid, bcfg, prob);
    stats.total_build_time +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_b0).count();

    if (!built) return false;   // pre-flight rejected → caller falls back

    ++stats.n_batches;

    RoutingLPSolution sol;
    auto t_s0 = std::chrono::steady_clock::now();
    bool solved = solve_routing_lp(prob, cfg.config_path, sol);
    stats.total_solve_time +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s0).count();

    bool can_round = solved;
    if (!can_round) {
        bool limit_hit = (sol.status == RoutingLPSolution::Status::ITER_LIMIT ||
                          sol.status == RoutingLPSolution::Status::TIME_LIMIT);
        if (limit_hit && sol.primal_residual >= 0.0 &&
            sol.primal_residual < cfg.inexact_residual_tol)
            can_round = true;
    }

    if (!can_round) {
        ++stats.n_batches_skipped;
        for (const auto& tn : batch) {
            NetRoute r; r.name = tn.name; r.orig_name = tn.orig_name;
            result_routes.push_back(std::move(r));
        }
        stats.n_nets_disconnected += (int)batch.size();
        return true;
    }

    if (solved) ++stats.n_batches_solved;
    else        ++stats.n_batches_inexact;

    auto t_r0 = std::chrono::steady_clock::now();
    auto routes = round_lp_solution(prob, sol, cfg.threshold);
    stats.total_round_time +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_r0).count();

    for (auto& r : routes) {
        if (r.segments.empty()) ++stats.n_nets_disconnected;
        else                    ++stats.n_nets_routed;
        result_routes.push_back(std::move(r));
    }
    return true;
}

// ── main ─────────────────────────────────────────────────────────────────────

std::vector<NetRoute> run_adaptive_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const AdaptiveRoutingConfig& cfg,
    AdaptiveRoutingStats& stats)
{
    auto wall0 = std::chrono::steady_clock::now();
    stats.n_nets_total = (int)twonets.size();

    const int BGX = std::max(1, cfg.batch_grid_x);
    const int BGY = std::max(1, cfg.batch_grid_y);

    // ── Group nets into spatial bins by centroid ───────────────────────────────
    std::map<std::pair<int,int>, std::vector<int>> bins;
    for (int ni = 0; ni < (int)twonets.size(); ++ni) {
        const TwoNet& t = twonets[ni];
        int cx = (t.src.loc.x + t.snk.loc.x) / 2;
        int cy = (t.src.loc.y + t.snk.loc.y) / 2;
        bins[{cx / BGX, cy / BGY}].push_back(ni);
    }

    std::cout << "[adaptive] " << twonets.size() << " nets in "
              << bins.size() << " centroid bins"
              << "  batch_grid=" << BGX << "x" << BGY
              << "  margin=" << cfg.margin << "\n";

    // ── Process each bin ──────────────────────────────────────────────────────
    // result_routes[ni] will be set for every net index ni
    std::vector<NetRoute> all_routes(twonets.size());
    std::vector<bool>     filled(twonets.size(), false);

    for (auto& [key, indices] : bins) {
        // Collect TwoNets for this batch
        std::vector<TwoNet> batch;
        batch.reserve(indices.size());
        for (int ni : indices) batch.push_back(twonets[ni]);

        // Try joint batch solve first
        std::vector<NetRoute> batch_routes;
        bool ok = solve_batch(batch, grid, cfg, stats, batch_routes);

        if (!ok) {
            // Pre-flight rejected joint batch — fall back to individual solves
            for (int k = 0; k < (int)batch.size(); ++k) {
                std::vector<NetRoute> single_routes;
                bool single_ok = solve_batch({batch[k]}, grid, cfg, stats, single_routes);
                if (!single_ok) {
                    // Even single-net solve rejected (net span > MAX_VARS limit)
                    ++stats.n_batches_skipped;
                    NetRoute r; r.name = batch[k].name; r.orig_name = batch[k].orig_name;
                    single_routes.push_back(std::move(r));
                    ++stats.n_nets_disconnected;
                }
                all_routes[indices[k]] = std::move(single_routes[0]);
                filled[indices[k]] = true;
            }
        } else {
            // Place results back into indexed positions
            for (int k = 0; k < (int)indices.size(); ++k) {
                all_routes[indices[k]] = std::move(batch_routes[k]);
                filled[indices[k]] = true;
            }
        }
    }

    // Sanity: every net must have a route entry
    for (int ni = 0; ni < (int)twonets.size(); ++ni) {
        if (!filled[ni]) {
            all_routes[ni].name      = twonets[ni].name;
            all_routes[ni].orig_name = twonets[ni].orig_name;
            ++stats.n_nets_disconnected;
        }
    }

    stats.total_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();

    std::cout << "[adaptive] Done:"
              << " batches=" << stats.n_batches
              << " solved=" << stats.n_batches_solved
              << " inexact=" << stats.n_batches_inexact
              << " skipped=" << stats.n_batches_skipped
              << "  nets routed=" << stats.n_nets_routed
              << " disconnected=" << stats.n_nets_disconnected << "\n";

    return all_routes;
}

} // namespace rlp
