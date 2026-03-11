// router_lp/src/solution_rounding.cpp
// B4: Round fractional LP solution to paths and write .out file.
//
// Algorithm: flow decomposition DFS.
//   For each 2-pin net n:
//   1. Build a residual directed adjacency list from the LP solution.
//      Forward arc  (tail→head) if x[n,e,0] > threshold.
//      Backward arc (head→tail) if x[n,e,1] > threshold.
//   2. Repeatedly trace a path from src to snk by greedy DFS (highest flow first).
//      After tracing, subtract the bottleneck flow and emit segments.
//   3. Repeat until the total flow out of src < threshold.
//   This guarantees connected src→snk paths for each net.

#include "../include/solution_rounding.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>

namespace rlp {

// ── Flow decomposition ───────────────────────────────────────────────────────

struct Arc {
    int to;       // destination subgraph node
    int edge_idx; // index into prob.edges
    double* flow; // pointer into mutable flow array (fwd or bwd)
    bool fwd;     // direction flag (for segment orientation)
};

// Greedy DFS path from src_sub to snk_sub.
// Returns path as sequence of arcs. Returns empty if no path.
static std::vector<Arc*> dfs_path(
    int src, int snk,
    std::vector<std::vector<Arc>>& adj,
    double threshold)
{
    int n_nodes = (int)adj.size();
    std::vector<int> parent(n_nodes, -1);
    std::vector<Arc*> parent_arc(n_nodes, nullptr);
    std::vector<bool> visited(n_nodes, false);

    // Iterative DFS; at each node pick the unvisited neighbor with highest flow
    std::stack<int> stk;
    stk.push(src);
    visited[src] = true;

    while (!stk.empty()) {
        int u = stk.top();
        if (u == snk) break;

        // Find best unvisited outgoing arc
        Arc* best = nullptr;
        double best_f = threshold;
        for (auto& a : adj[u]) {
            if (!visited[a.to] && *a.flow > best_f) {
                best_f = *a.flow;
                best = &a;
            }
        }
        if (!best) {
            // Dead end — backtrack
            stk.pop();
            if (!stk.empty()) visited[u] = false;  // allow revisiting from other branches
            continue;
        }
        parent[best->to]     = u;
        parent_arc[best->to] = best;
        visited[best->to]    = true;
        stk.push(best->to);
    }

    if (!visited[snk]) return {};

    // Reconstruct path from snk back to src
    std::vector<Arc*> path;
    for (int v = snk; v != src; v = parent[v])
        path.push_back(parent_arc[v]);
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

    // Mutable local copy of this net's flows [n_edges * 2]
    int base = n * n_edges * 2;
    std::vector<double> flow(sol_x.begin() + base, sol_x.begin() + base + n_edges * 2);

    // Build adjacency list (arcs with pointers into 'flow')
    std::vector<std::vector<Arc>> adj(n_nodes);
    for (int e = 0; e < n_edges; ++e) {
        int t = prob.edge_tail_sub[e];
        int h = prob.edge_head_sub[e];
        // forward arc
        adj[t].push_back({h, e, &flow[e * 2 + 0], true});
        // backward arc
        adj[h].push_back({t, e, &flow[e * 2 + 1], false});
    }

    const auto& edges = prob.edges;
    std::vector<RoutingSegment> segs;

    // Flow decomposition: trace paths until net outflow is exhausted
    for (int iter = 0; iter < n_edges + 1; ++iter) {
        // Check remaining outflow from src
        double out = 0.0;
        for (auto& a : adj[src_sub]) out += *a.flow;
        if (out <= threshold) break;

        auto path = dfs_path(src_sub, snk_sub, adj, threshold);
        if (path.empty()) break;

        // Bottleneck flow along path
        double bot = 1e9;
        for (Arc* a : path) bot = std::min(bot, *a->flow);
        bot = std::max(bot, threshold);

        // Subtract flow and collect segments
        for (Arc* a : path) {
            *a->flow -= bot;
            const Edge& e = edges[a->edge_idx];
            RoutingSegment s;
            if (a->fwd) {
                s.x1 = e.xl; s.y1 = e.yl; s.z1 = e.ll;
                s.x2 = e.xh; s.y2 = e.yh; s.z2 = e.lh;
            } else {
                s.x1 = e.xh; s.y1 = e.yh; s.z1 = e.lh;
                s.x2 = e.xl; s.y2 = e.yl; s.z2 = e.ll;
            }
            segs.push_back(s);
        }
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
        nr.name     = prob.nets[n].name;
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
    for (const auto& r : routes) {
        f << r.name << "\n(\n";
        for (const auto& s : r.segments)
            f << s.x1 << " " << s.y1 << " " << s.z1 << " "
              << s.x2 << " " << s.y2 << " " << s.z2 << "\n";
        f << ")\n";
    }
    std::cout << "[rounding] Wrote " << routes.size() << " nets to " << path << "\n";
    return true;
}

} // namespace rlp
