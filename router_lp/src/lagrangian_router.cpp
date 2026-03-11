// router_lp/src/lagrangian_router.cpp
// Lagrangian relaxation global router.
// See include/lagrangian_router.hpp for algorithm description.

#include "../include/lagrangian_router.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

namespace rlp {

// ── Flat index helpers ────────────────────────────────────────────────────────

// H-edge from (x,y,l) → (x+1,y,l): lambda_h[l*X*Y + x*Y + y]
// V-edge from (x,y,l) → (x,y+1,l): lambda_v[l*X*Y + x*Y + y]
// Via from  (x,y,l) → (x,y,l+1):   lambda_via[l*X*Y + x*Y + y]

static inline int edge_idx(int l, int x, int y, int X, int Y) {
    return l * X * Y + x * Y + y;
}

// ── Per-net Dijkstra ──────────────────────────────────────────────────────────

// A path segment: edge direction + tail gcell
struct PathSeg {
    int l, x, y;
    EdgeDir dir;  // H, V, or Via
};

// Run Dijkstra for one net on its bbox subgraph with augmented edge costs.
// Returns path as list of PathSeg (empty = disconnected).
static std::vector<PathSeg> dijkstra_net(
    const TwoNet& net,
    const GridInfo& grid,
    const LagrangianConfig& cfg,
    const std::vector<float>& lam_h,
    const std::vector<float>& lam_v,
    const std::vector<float>& lam_via)
{
    const int X = grid.X, Y = grid.Y;

    // Compute bbox + margin
    int x0 = std::min(net.src.loc.x, net.snk.loc.x);
    int x1 = std::max(net.src.loc.x, net.snk.loc.x);
    int y0 = std::min(net.src.loc.y, net.snk.loc.y);
    int y1 = std::max(net.src.loc.y, net.snk.loc.y);
    int l0 = std::min(net.src.loc.l, net.snk.loc.l);
    int l1 = std::max(net.src.loc.l, net.snk.loc.l);

    // Expand by margin (include all layers to allow via routing)
    x0 = std::max(0,        x0 - cfg.margin);
    x1 = std::min(X - 1,   x1 + cfg.margin);
    y0 = std::max(0,        y0 - cfg.margin);
    y1 = std::min(Y - 1,   y1 + cfg.margin);
    l0 = 0;
    l1 = grid.L - 1;

    int bx = x1 - x0 + 1;
    int by = y1 - y0 + 1;
    int bl = l1 - l0 + 1;  // = grid.L
    int n_local = bx * by * bl;

    // Local flat index within bbox: (x-x0)*by*bl + (y-y0)*bl + (l-l0)
    auto lid = [&](int x, int y, int l) -> int {
        return (x - x0) * by * bl + (y - y0) * bl + (l - l0);
    };

    int src_l = lid(net.src.loc.x, net.src.loc.y, net.src.loc.l);
    int snk_l = lid(net.snk.loc.x, net.snk.loc.y, net.snk.loc.l);

    // Dijkstra
    std::vector<float> dist(n_local, 1e30f);
    std::vector<int>   prev(n_local, -1);
    std::vector<PathSeg> prev_seg(n_local);

    using PQ = std::priority_queue<std::pair<float,int>,
                                   std::vector<std::pair<float,int>>,
                                   std::greater<>>;
    PQ pq;
    dist[src_l] = 0.0f;
    pq.push({0.0f, src_l});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u] + 1e-9f) continue;
        if (u == snk_l) break;

        // Decode local index → (x, y, l)
        int llu = u % bl + l0;
        int tmp = u / bl;
        int yy  = tmp % by + y0;
        int xx  = tmp / by + x0;

        // ── H-edges (layer preferred direction = 0) ───────────────────────────
        if (grid.layer_dir[llu] == 0) {
            float base = (float)(grid.unit_wire_cost * grid.layer_short_cost[llu]);
            // Forward: (xx,yy,llu) → (xx+1,yy,llu)
            if (xx + 1 <= x1) {
                int ei = edge_idx(llu, xx, yy, X, Y);
                float w = base + lam_h[ei];
                int v = lid(xx + 1, yy, llu);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu, xx, yy, EdgeDir::H};
                    pq.push({dist[v], v});
                }
            }
            // Backward: (xx,yy,llu) ← (xx-1,yy,llu) — use edge at (xx-1,yy,llu)
            if (xx - 1 >= x0) {
                int ei = edge_idx(llu, xx - 1, yy, X, Y);
                float w = base + lam_h[ei];
                int v = lid(xx - 1, yy, llu);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu, xx - 1, yy, EdgeDir::H};
                    pq.push({dist[v], v});
                }
            }
        }

        // ── V-edges (layer preferred direction = 1) ───────────────────────────
        if (grid.layer_dir[llu] == 1) {
            float base = (float)(grid.unit_wire_cost * grid.layer_short_cost[llu]);
            if (yy + 1 <= y1) {
                int ei = edge_idx(llu, xx, yy, X, Y);
                float w = base + lam_v[ei];
                int v = lid(xx, yy + 1, llu);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu, xx, yy, EdgeDir::V};
                    pq.push({dist[v], v});
                }
            }
            if (yy - 1 >= y0) {
                int ei = edge_idx(llu, xx, yy - 1, X, Y);
                float w = base + lam_v[ei];
                int v = lid(xx, yy - 1, llu);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu, xx, yy - 1, EdgeDir::V};
                    pq.push({dist[v], v});
                }
            }
        }

        // ── Via edges (up/down between layers) ────────────────────────────────
        if (cfg.add_via) {
            float via_cost = (float)grid.unit_via_cost;
            // Up: (xx,yy,llu) → (xx,yy,llu+1)
            if (llu + 1 <= l1) {
                int ei = edge_idx(llu, xx, yy, X, Y);
                float w = via_cost + lam_via[ei];
                int v = lid(xx, yy, llu + 1);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu, xx, yy, EdgeDir::Via};
                    pq.push({dist[v], v});
                }
            }
            // Down: (xx,yy,llu) → (xx,yy,llu-1)
            if (llu - 1 >= l0) {
                int ei = edge_idx(llu - 1, xx, yy, X, Y);
                float w = via_cost + lam_via[ei];
                int v = lid(xx, yy, llu - 1);
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    prev_seg[v] = {llu - 1, xx, yy, EdgeDir::Via};
                    pq.push({dist[v], v});
                }
            }
        }
    }

    if (dist[snk_l] > 1e29f) return {};  // unreachable

    // Reconstruct path
    std::vector<PathSeg> path;
    int cur = snk_l;
    while (prev[cur] != -1) {
        path.push_back(prev_seg[cur]);
        cur = prev[cur];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ── Convert path to NetRoute ──────────────────────────────────────────────────

static NetRoute path_to_route(const TwoNet& net,
                               const std::vector<PathSeg>& path)
{
    NetRoute r;
    r.name = net.name;
    for (const auto& seg : path) {
        RoutingSegment s;
        s.x1 = seg.x; s.y1 = seg.y; s.z1 = seg.l;
        if (seg.dir == EdgeDir::H) {
            s.x2 = seg.x + 1; s.y2 = seg.y; s.z2 = seg.l;
        } else if (seg.dir == EdgeDir::V) {
            s.x2 = seg.x; s.y2 = seg.y + 1; s.z2 = seg.l;
        } else {  // Via
            s.x2 = seg.x; s.y2 = seg.y; s.z2 = seg.l + 1;
        }
        r.segments.push_back(s);
    }
    return r;
}

// ── Main Lagrangian loop ──────────────────────────────────────────────────────

std::vector<NetRoute> run_lagrangian_routing(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const LagrangianConfig& cfg,
    LagrangianStats& stats)
{
    auto wall0 = std::chrono::steady_clock::now();

    const int X = grid.X, Y = grid.Y, L = grid.L;
    const int XY = X * Y;
    const int LXY = L * XY;

    // ── Allocate global multipliers and usage counters ────────────────────────
    std::vector<float> lam_h  (LXY, 0.0f);
    std::vector<float> lam_v  (LXY, 0.0f);
    std::vector<float> lam_via(LXY, 0.0f);
    std::vector<float> use_h  (LXY, 0.0f);
    std::vector<float> use_v  (LXY, 0.0f);
    std::vector<float> use_via(LXY, 0.0f);

    int n_nets = (int)twonets.size();
    // Store current iteration paths for each net
    std::vector<std::vector<PathSeg>> paths(n_nets);

    std::cout << "[lagrangian] " << n_nets << " nets, grid "
              << L << "x" << X << "x" << Y
              << "  iters=" << cfg.max_iters
              << "  step=" << cfg.step_size
              << "  decay=" << cfg.step_decay
              << "  margin=" << cfg.margin << "\n";

    // ── Subgradient iterations ────────────────────────────────────────────────
    for (int iter = 1; iter <= cfg.max_iters; ++iter) {

        // ── Step 1: Per-net shortest path ─────────────────────────────────────
        auto t_dijk0 = std::chrono::steady_clock::now();
        for (int ni = 0; ni < n_nets; ++ni) {
            paths[ni] = dijkstra_net(twonets[ni], grid, cfg,
                                     lam_h, lam_v, lam_via);
        }
        stats.total_dijkstra_time +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_dijk0).count();

        // ── Step 2: Aggregate edge usage ──────────────────────────────────────
        auto t_upd0 = std::chrono::steady_clock::now();
        std::fill(use_h.begin(),   use_h.end(),   0.0f);
        std::fill(use_v.begin(),   use_v.end(),   0.0f);
        std::fill(use_via.begin(), use_via.end(), 0.0f);

        for (int ni = 0; ni < n_nets; ++ni) {
            for (const auto& seg : paths[ni]) {
                int idx = edge_idx(seg.l, seg.x, seg.y, X, Y);
                if      (seg.dir == EdgeDir::H)   use_h  [idx] += 1.0f;
                else if (seg.dir == EdgeDir::V)   use_v  [idx] += 1.0f;
                else                              use_via[idx] += 1.0f;
            }
        }

        // ── Step 3: Compute violation and step size ───────────────────────────
        double alpha = cfg.step_size / std::pow((double)iter, cfg.step_decay);

        double max_viol = 0.0, sum_viol = 0.0;
        int n_overload = 0;

        // ── Step 4: Update multipliers ────────────────────────────────────────
        // H-edges: cap = grid.cap[l][x][y] for H-preferred layers
        for (int l = 0; l < L; ++l) {
            if (grid.layer_dir[l] != 0) continue;
            for (int x = 0; x < X - 1; ++x) {
                for (int y = 0; y < Y; ++y) {
                    int idx = edge_idx(l, x, y, X, Y);
                    float cap = (float)grid.cap[l][x][y];
                    float viol = use_h[idx] - cap;
                    if (viol > (float)max_viol) max_viol = viol;
                    if (viol > 0) { sum_viol += viol; ++n_overload; }
                    lam_h[idx] = std::max(0.0f,
                                          lam_h[idx] + (float)(alpha * viol));
                }
            }
        }
        // V-edges
        for (int l = 0; l < L; ++l) {
            if (grid.layer_dir[l] != 1) continue;
            for (int x = 0; x < X; ++x) {
                for (int y = 0; y < Y - 1; ++y) {
                    int idx = edge_idx(l, x, y, X, Y);
                    float cap = (float)grid.cap[l][x][y];
                    float viol = use_v[idx] - cap;
                    if (viol > (float)max_viol) max_viol = viol;
                    if (viol > 0) { sum_viol += viol; ++n_overload; }
                    lam_v[idx] = std::max(0.0f,
                                          lam_v[idx] + (float)(alpha * viol));
                }
            }
        }
        // Via edges (use fixed cap = cfg.via_cap; no λ update for vias)

        stats.total_update_time +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_upd0).count();

        if (cfg.log_every > 0 && iter % cfg.log_every == 0) {
            std::cout << "[lagrangian] iter=" << iter
                      << "  alpha=" << alpha
                      << "  max_viol=" << max_viol
                      << "  overload_edges=" << n_overload << "\n";
        }

        stats.final_max_violation = max_viol;
        stats.final_avg_violation = (n_overload > 0) ? sum_viol / n_overload : 0.0;
    }

    stats.iters_run = cfg.max_iters;

    // ── Extract final routes ──────────────────────────────────────────────────
    std::vector<NetRoute> routes;
    routes.reserve(n_nets);
    for (int ni = 0; ni < n_nets; ++ni) {
        if (paths[ni].empty()) {
            ++stats.n_nets_disconnected;
            NetRoute r; r.name = twonets[ni].name;
            routes.push_back(std::move(r));
        } else {
            ++stats.n_nets_routed;
            routes.push_back(path_to_route(twonets[ni], paths[ni]));
        }
    }

    stats.total_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();

    std::cout << "[lagrangian] Done:"
              << " iters=" << stats.iters_run
              << "  max_viol=" << stats.final_max_violation
              << "  routed=" << stats.n_nets_routed
              << " disconnected=" << stats.n_nets_disconnected
              << "  dijk_time=" << stats.total_dijkstra_time << "s"
              << "  update_time=" << stats.total_update_time << "s"
              << "  total=" << stats.total_wall_time << "s\n";

    return routes;
}

} // namespace rlp
