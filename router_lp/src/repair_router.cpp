// router_lp/src/repair_router.cpp
// Repair pass: for each disconnected 2-pin net, compute shortest path on the
// routing grid with all lambda=0 (unconstrained). Uses Dijkstra with float costs.
// This guarantees 0 disconnected nets for any routable benchmark (modulo
// unreachable pin pairs due to geometry), at the cost of potentially adding
// capacity violations on already-congested edges.

#include "../include/repair_router.hpp"
#include <algorithm>
#include <queue>
#include <vector>
#include <limits>
#include <iostream>

namespace rlp {

// Expand bbox by margin and clamp to grid bounds
static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Repair a single disconnected net by running Dijkstra with zero lambda.
// Returns true if a path was found and the route was updated.
static bool repair_one_net(
    NetRoute& route,
    const TwoNet& net,
    const GridInfo& grid,
    bool add_via)
{
    const int X = grid.X, Y = grid.Y, L = grid.L;
    constexpr int MARGIN = 10;

    // Bounding box around src/snk, expanded by MARGIN
    int x0 = clamp(std::min(net.src.loc.x, net.snk.loc.x) - MARGIN, 0, X - 1);
    int x1 = clamp(std::max(net.src.loc.x, net.snk.loc.x) + MARGIN, 0, X - 1);
    int y0 = clamp(std::min(net.src.loc.y, net.snk.loc.y) - MARGIN, 0, Y - 1);
    int y1 = clamp(std::max(net.src.loc.y, net.snk.loc.y) + MARGIN, 0, Y - 1);
    // Allow all layers so via-routing can change layers freely
    const int l0 = 0;
    const int l1 = L - 1;

    int bx = x1 - x0 + 1;
    int by = y1 - y0 + 1;
    int bl = l1 - l0 + 1;
    int n_local = bx * by * bl;

    // Local flat index: (x-x0)*by*bl + (y-y0)*bl + (l-l0)
    auto lid = [&](int x, int y, int l) -> int {
        return (x - x0) * by * bl + (y - y0) * bl + (l - l0);
    };

    int src_l = lid(net.src.loc.x, net.src.loc.y, net.src.loc.l);
    int snk_l = lid(net.snk.loc.x, net.snk.loc.y, net.snk.loc.l);

    // Dijkstra structures
    constexpr float INF = 1e30f;
    std::vector<float> dist(n_local, INF);
    std::vector<int>   prev(n_local, -1);

    // prev_seg encodes the edge taken: packed as (l*X*Y + x*Y + y)*4 + dir
    // We store (x,y,l,dir) from the predecessor so we can rebuild segments
    struct PrevSeg { int x, y, l; int dir; }; // dir: 0=H, 1=V, 2=Via
    std::vector<PrevSeg> prev_seg(n_local, {-1,-1,-1,-1});

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

        // ── H-edges (H-preferred layer: dir==0) ───────────────────────────────
        if (grid.layer_dir[llu] == 0) {
            float base = (float)(grid.unit_wire_cost * grid.layer_short_cost[llu]);
            // Forward: (xx,yy,llu) → (xx+1,yy,llu)
            if (xx + 1 <= x1) {
                int v = lid(xx + 1, yy, llu);
                float nd = d + base;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx, yy, llu, 0}; // H from (xx,yy,llu)
                    pq.push({nd, v});
                }
            }
            // Backward: (xx,yy,llu) → (xx-1,yy,llu)
            if (xx - 1 >= x0) {
                int v = lid(xx - 1, yy, llu);
                float nd = d + base;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx - 1, yy, llu, 0}; // H from (xx-1,yy,llu)
                    pq.push({nd, v});
                }
            }
        }

        // ── V-edges (V-preferred layer: dir==1) ───────────────────────────────
        if (grid.layer_dir[llu] == 1) {
            float base = (float)(grid.unit_wire_cost * grid.layer_short_cost[llu]);
            // Forward: (xx,yy,llu) → (xx,yy+1,llu)
            if (yy + 1 <= y1) {
                int v = lid(xx, yy + 1, llu);
                float nd = d + base;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx, yy, llu, 1}; // V from (xx,yy,llu)
                    pq.push({nd, v});
                }
            }
            // Backward: (xx,yy,llu) → (xx,yy-1,llu)
            if (yy - 1 >= y0) {
                int v = lid(xx, yy - 1, llu);
                float nd = d + base;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx, yy - 1, llu, 1}; // V from (xx,yy-1,llu)
                    pq.push({nd, v});
                }
            }
        }

        // ── Via-edges (between layers) ────────────────────────────────────────
        if (add_via) {
            float via_cost = (float)grid.unit_via_cost;
            // Up: (xx,yy,llu) → (xx,yy,llu+1)
            if (llu + 1 <= l1) {
                int v = lid(xx, yy, llu + 1);
                float nd = d + via_cost;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx, yy, llu, 2}; // Via up from llu
                    pq.push({nd, v});
                }
            }
            // Down: (xx,yy,llu) → (xx,yy,llu-1)
            if (llu - 1 >= l0) {
                int v = lid(xx, yy, llu - 1);
                float nd = d + via_cost;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    prev[v] = u;
                    prev_seg[v] = {xx, yy, llu - 1, 2}; // Via up from llu-1
                    pq.push({nd, v});
                }
            }
        }
    }

    if (dist[snk_l] >= INF / 2.0f) {
        // Unreachable (no path within bbox, or add_via=false blocks layer change)
        return false;
    }

    // Trace back from snk to src and build segments
    std::vector<RoutingSegment> segs;
    int cur = snk_l;
    while (cur != src_l) {
        const PrevSeg& ps = prev_seg[cur];
        RoutingSegment s;
        s.x1 = ps.x; s.y1 = ps.y; s.z1 = ps.l;
        if (ps.dir == 0) {        // H
            s.x2 = ps.x + 1; s.y2 = ps.y;     s.z2 = ps.l;
        } else if (ps.dir == 1) { // V
            s.x2 = ps.x;     s.y2 = ps.y + 1; s.z2 = ps.l;
        } else {                  // Via (up)
            s.x2 = ps.x;     s.y2 = ps.y;     s.z2 = ps.l + 1;
        }
        segs.push_back(s);
        cur = prev[cur];
    }
    // Reverse so segments run src→snk
    std::reverse(segs.begin(), segs.end());

    // Write back (preserve orig_name)
    route.segments = std::move(segs);
    return true;
}

int repair_disconnected_nets(
    std::vector<NetRoute>& routes,
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    bool add_via)
{
    if (routes.size() != twonets.size()) {
        std::cerr << "[repair] routes/twonets size mismatch: "
                  << routes.size() << " vs " << twonets.size() << "\n";
        return 0;
    }

    int repaired = 0;
    for (int i = 0; i < (int)routes.size(); ++i) {
        if (!routes[i].segments.empty()) continue; // already connected
        if (repair_one_net(routes[i], twonets[i], grid, add_via)) {
            ++repaired;
        }
    }
    return repaired;
}

} // namespace rlp
