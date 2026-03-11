// router_lp/src/solution_rounding.cpp
// B4: Round fractional LP solution to paths and write .out file.
//
// Algorithm: BFS-based single-path extraction per 2-pin net.
//   For each 2-pin net n:
//   1. Build directed adjacency from the LP solution (fwd/bwd arcs above threshold).
//   2. BFS from src to snk — strict visited set, O(V+E), no cycles, no backtracking.
//   3. Emit the single path found. For 2-pin nets one connected path is sufficient.
//
//   Replaces the previous backtracking DFS which could loop indefinitely on cyclic
//   residual graphs (visited[u]=false on backtrack + positive residuals = infinite loop).

#include "../include/solution_rounding.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>

namespace rlp {

// ── Arc structure ─────────────────────────────────────────────────────────────

struct Arc {
    int to;       // destination subgraph node
    int edge_idx; // index into prob.edges
    double flow;  // flow value (copy, not pointer — adjacency rebuilt per net)
    bool fwd;     // direction flag (for segment orientation)
};

// BFS path from src to snk. Strict visited set — guaranteed O(V+E) termination.
// Prefers heavier arcs at each node (sort cands desc before enqueuing).
// Returns sequence of (edge_idx, fwd) for the path, empty if unreachable.
struct PathStep { int edge_idx; bool fwd; };

static std::vector<PathStep> bfs_path(
    int src, int snk,
    const std::vector<std::vector<Arc>>& adj,
    double threshold)
{
    int n = (int)adj.size();
    std::vector<bool> visited(n, false);
    std::vector<int>  parent(n, -1);
    std::vector<int>  parent_eidx(n, -1);
    std::vector<bool> parent_fwd(n, false);

    std::queue<int> q;
    q.push(src);
    visited[src] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == snk) break;

        // Collect and sort unvisited neighbours by flow (desc) for greedy quality
        std::vector<const Arc*> cands;
        for (const auto& a : adj[u])
            if (!visited[a.to] && a.flow > threshold)
                cands.push_back(&a);
        std::sort(cands.begin(), cands.end(),
                  [](const Arc* x, const Arc* y){ return x->flow > y->flow; });

        for (const Arc* a : cands) {
            if (visited[a->to]) continue;
            visited[a->to]    = true;
            parent[a->to]     = u;
            parent_eidx[a->to] = a->edge_idx;
            parent_fwd[a->to]  = a->fwd;
            q.push(a->to);
        }
    }

    if (!visited[snk]) return {};

    std::vector<PathStep> path;
    for (int v = snk; v != src; v = parent[v])
        path.push_back({parent_eidx[v], parent_fwd[v]});
    std::reverse(path.begin(), path.end());
    return path;
}

static std::vector<RoutingSegment> extract_net_path(
    int n,
    const RoutingLPProblem& prob,
    const std::vector<double>& sol_x,
    double threshold)
{
    int n_edges = prob.n_edges;
    int n_nodes = prob.n_nodes_sub;
    int src_sub = prob.net_src_sub[n];
    int snk_sub = prob.net_snk_sub[n];

    if (src_sub < 0 || snk_sub < 0) {
        std::cerr << "[rounding] net " << prob.nets[n].name
                  << " src or snk not in subgraph\n";
        return {};
    }

    int base = n * n_edges * 2;

    // Build adjacency list (value-copy of flows — no pointer aliasing)
    std::vector<std::vector<Arc>> adj(n_nodes);
    for (int e = 0; e < n_edges; ++e) {
        int t = prob.edge_tail_sub[e];
        int h = prob.edge_head_sub[e];
        double fflow = sol_x[base + e * 2 + 0];
        double bflow = sol_x[base + e * 2 + 1];
        if (fflow > threshold) adj[t].push_back({h, e, fflow, true});
        if (bflow > threshold) adj[h].push_back({t, e, bflow, false});
    }

    auto path = bfs_path(src_sub, snk_sub, adj, threshold);
    if (path.empty()) return {};

    const auto& edges = prob.edges;
    std::vector<RoutingSegment> segs;
    segs.reserve(path.size());
    for (const auto& step : path) {
        const Edge& e = edges[step.edge_idx];
        RoutingSegment s;
        if (step.fwd) {
            s.x1 = e.xl; s.y1 = e.yl; s.z1 = e.ll;
            s.x2 = e.xh; s.y2 = e.yh; s.z2 = e.lh;
        } else {
            s.x1 = e.xh; s.y1 = e.yh; s.z1 = e.lh;
            s.x2 = e.xl; s.y2 = e.yl; s.z2 = e.ll;
        }
        segs.push_back(s);
    }
    return segs;
}

// ── Public API ────────────────────────────────────────────────────────────────

std::vector<NetRoute> round_lp_solution(const RoutingLPProblem& prob,
                                         const RoutingLPSolution& sol,
                                         double threshold)
{
    std::vector<NetRoute> routes;
    if (sol.x.empty()) return routes;

    for (int n = 0; n < prob.n_nets; ++n) {
        NetRoute nr;
        nr.name      = prob.nets[n].name;
        nr.orig_name = prob.nets[n].orig_name;
        nr.segments = extract_net_path(n, prob, sol.x, threshold);
        routes.push_back(std::move(nr));
    }

    int routed = 0;
    for (const auto& r : routes) if (!r.segments.empty()) ++routed;
    std::cout << "[rounding] " << routed << "/" << prob.n_nets << " nets have segments\n";
    return routes;
}

bool write_out_file(const std::string& path, const std::vector<NetRoute>& routes) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[rounding] Cannot open output file: " << path << "\n";
        return false;
    }

    // Group 2-pin subroutes by original multi-pin net name.
    // Use insertion-order tracking to preserve output order.
    // If orig_name is empty (LP mode without decomposition), fall back to name.
    std::vector<std::string> order;
    std::unordered_map<std::string, std::vector<const NetRoute*>> groups;
    for (const auto& r : routes) {
        const std::string& key = r.orig_name.empty() ? r.name : r.orig_name;
        if (groups.find(key) == groups.end()) order.push_back(key);
        groups[key].push_back(&r);
    }

    int written = 0;
    for (const auto& key : order) {
        const auto& group = groups[key];
        // Collect all segments from all subroutes
        bool any_segment = false;
        for (const NetRoute* r : group)
            if (!r->segments.empty()) { any_segment = true; break; }
        if (!any_segment) continue;  // skip fully disconnected nets

        f << key << "\n(\n";
        for (const NetRoute* r : group)
            for (const auto& s : r->segments)
                f << s.x1 << " " << s.y1 << " " << s.z1 << " "
                  << s.x2 << " " << s.y2 << " " << s.z2 << "\n";
        f << ")\n";
        ++written;
    }
    std::cout << "[rounding] Wrote " << written << " nets to " << path << "\n";
    return true;
}

} // namespace rlp
