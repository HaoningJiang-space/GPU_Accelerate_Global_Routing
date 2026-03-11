// router_lp/src/windowed_routing.cpp
// Windowed LP routing implementation.
// See include/windowed_routing.hpp for design notes.

#include "../include/windowed_routing.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace rlp {

std::vector<NetRoute> run_windowed_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const WindowedRoutingConfig& cfg,
    WindowedRoutingStats& stats)
{
    auto wall0 = std::chrono::steady_clock::now();

    const int WX = cfg.window_x;
    const int WY = cfg.window_y;
    const int nWX = (grid.X + WX - 1) / WX;
    const int nWY = (grid.Y + WY - 1) / WY;

    stats.n_nets_total    = (int)twonets.size();
    stats.n_windows_total = nWX * nWY;

    // ── Assign each 2-pin net to the window containing BOTH endpoints ─────────
    // bucket[wi*nWY + wj] = list of net indices assigned to window (wi,wj)
    std::vector<std::vector<int>> buckets(nWX * nWY);
    int n_unassigned = 0;

    for (int ni = 0; ni < (int)twonets.size(); ++ni) {
        const TwoNet& t = twonets[ni];
        int wsx = t.src.loc.x / WX, wsy = t.src.loc.y / WY;
        int wdx = t.snk.loc.x / WX, wdy = t.snk.loc.y / WY;

        if (wsx == wdx && wsy == wdy)
            buckets[wsx * nWY + wsy].push_back(ni);
        else
            ++n_unassigned;
    }

    stats.n_nets_unassigned = n_unassigned;
    int n_nonempty = 0;
    for (const auto& b : buckets) if (!b.empty()) ++n_nonempty;

    if (n_unassigned > 0)
        std::cout << "[windowed] " << n_unassigned << "/" << twonets.size()
                  << " nets span multiple windows (unrouted)\n";

    std::cout << "[windowed] Grid " << grid.X << "x" << grid.Y
              << "  window=" << WX << "x" << WY
              << "  tiles=" << nWX << "x" << nWY
              << "  non-empty=" << n_nonempty << "\n";

    // ── Process each non-empty window ─────────────────────────────────────────
    std::vector<NetRoute> all_routes;
    all_routes.reserve(twonets.size());

    for (int wi = 0; wi < nWX; ++wi) {
        for (int wj = 0; wj < nWY; ++wj) {
            const auto& bucket = buckets[wi * nWY + wj];
            if (bucket.empty()) continue;

            // Gather TwoNets for this window
            std::vector<TwoNet> window_nets;
            window_nets.reserve(bucket.size());
            for (int ni : bucket) window_nets.push_back(twonets[ni]);

            // Helper: push empty (disconnected) routes for all nets in this window
            auto push_disconnected = [&]() {
                for (const auto& tn : window_nets) {
                    NetRoute r;
                    r.name = tn.name;
                    all_routes.push_back(std::move(r));
                }
                stats.n_nets_disconnected += (int)window_nets.size();
            };

            // ── Build LP ──────────────────────────────────────────────────────
            LPBuilderConfig bcfg;
            bcfg.max_hpwl      = cfg.max_hpwl;
            bcfg.add_via_edges = cfg.add_via_edges;

            RoutingLPProblem prob;
            auto t_b0 = std::chrono::steady_clock::now();
            bool built = build_routing_lp(window_nets, grid, bcfg, prob);
            stats.total_build_time +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_b0).count();

            if (!built) {
                ++stats.n_windows_skipped;
                push_disconnected();
                continue;
            }

            // ── Solve ─────────────────────────────────────────────────────────
            RoutingLPSolution sol;
            auto t_s0 = std::chrono::steady_clock::now();
            bool solved = solve_routing_lp(prob, cfg.config_path, sol);
            stats.total_solve_time +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s0).count();

            // Accept ITER_LIMIT / TIME_LIMIT if primal residual is small enough
            bool can_round = solved;
            if (!can_round) {
                bool limit_hit = (sol.status == RoutingLPSolution::Status::ITER_LIMIT ||
                                  sol.status == RoutingLPSolution::Status::TIME_LIMIT);
                if (limit_hit && sol.primal_residual >= 0.0 &&
                    sol.primal_residual < cfg.inexact_residual_tol) {
                    can_round = true;
                }
            }

            if (!can_round) {
                ++stats.n_windows_skipped;
                push_disconnected();
                continue;
            }

            if (solved) ++stats.n_windows_solved;
            else        ++stats.n_windows_inexact;

            // ── Round ─────────────────────────────────────────────────────────
            auto t_r0 = std::chrono::steady_clock::now();
            auto window_routes = round_lp_solution(prob, sol, cfg.threshold);
            stats.total_round_time +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_r0).count();

            for (auto& r : window_routes) {
                if (r.segments.empty()) ++stats.n_nets_disconnected;
                else                    ++stats.n_nets_routed;
                all_routes.push_back(std::move(r));
            }
        }
    }

    stats.total_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();

    std::cout << "[windowed] Done:"
              << " windows=" << stats.n_windows_total
              << " solved=" << stats.n_windows_solved
              << " inexact=" << stats.n_windows_inexact
              << " skipped=" << stats.n_windows_skipped
              << "  nets routed=" << stats.n_nets_routed
              << " disconnected=" << stats.n_nets_disconnected
              << " unassigned=" << stats.n_nets_unassigned << "\n";

    return all_routes;
}

} // namespace rlp
