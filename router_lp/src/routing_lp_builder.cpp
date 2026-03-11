// router_lp/src/routing_lp_builder.cpp
// Builds the LP from TwoNets + GridInfo.

#include "../include/routing_lp_builder.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <climits>
#include <cmath>
#include <tuple>
#include <functional>

namespace rlp {

// ── helpers ───────────────────────────────────────────────────────────────────

static inline int hpwl(const TwoNet& n) {
    return std::abs(n.src.loc.x - n.snk.loc.x) +
           std::abs(n.src.loc.y - n.snk.loc.y) +
           std::abs(n.src.loc.l - n.snk.loc.l);
}

// Canonical edge key: (min_l,min_x,min_y,dir)  so both endpoints give same key
using EdgeKey = std::tuple<int,int,int,int>; // (l, x, y, dir:0=H/1=V/2=Via)

static EdgeKey edge_key(int l, int x, int y, int dir) {
    return {l, x, y, dir};
}

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        auto h = [](size_t a, size_t b) { return a ^ (b * 2654435761ULL + 0x9e3779b97f4a7c15ULL + (a<<6) + (a>>2)); };
        return h(h(h(std::get<0>(k), std::get<1>(k)), std::get<2>(k)), std::get<3>(k));
    }
};

// ── edge enumeration for a bounding box ───────────────────────────────────────

// Given bounding box [l0..l1][x0..x1][y0..y1], enumerate all edges inside.
// H-edge (l,x,y)→(l,x+1,y): only if grid.layer_dir[l]==0 (H layer)
// V-edge (l,x,y)→(l,x,y+1): only if grid.layer_dir[l]==1 (V layer)
// Via-edge (l,x,y)→(l+1,x,y): always (if add_via_edges)
static void enumerate_edges(const GridInfo& grid,
                             int l0, int l1, int x0, int x1, int y0, int y1,
                             bool add_via, std::vector<Edge>& edges,
                             std::unordered_map<EdgeKey,int,EdgeKeyHash>& key_to_idx)
{
    int id = (int)edges.size();
    for (int l = l0; l <= l1; ++l) {
        // Layer 0 is the pin-access layer (not a routing layer).
        // The evaluator never sets a direction for layer 0 (NVR_DIR_NONE),
        // so any wire segment on layer 0 triggers an assertion failure.
        // Only via edges (l=0 → l=1) are allowed to touch layer 0.
        const bool routing_layer = (l > 0);
        for (int x = x0; x <= x1; ++x) {
            for (int y = y0; y <= y1; ++y) {
                // H-edge: only valid on H-direction routing layers (dir==0, l>0)
                if (routing_layer && x+1 <= x1 && grid.layer_dir[l] == 0) {
                    auto k = edge_key(l, x, y, 0);
                    if (key_to_idx.find(k) == key_to_idx.end()) {
                        double cap  = grid.cap[l][x][y];
                        double cost = grid.unit_wire_cost;
                        edges.push_back({id, EdgeDir::H, x, y, l, x+1, y, l, cap, cost});
                        key_to_idx[k] = id++;
                    }
                }
                // V-edge: only valid on V-direction routing layers (dir==1, l>0)
                if (routing_layer && y+1 <= y1 && grid.layer_dir[l] == 1) {
                    auto k = edge_key(l, x, y, 1);
                    if (key_to_idx.find(k) == key_to_idx.end()) {
                        double cap  = grid.cap[l][x][y];
                        double cost = grid.unit_wire_cost;
                        edges.push_back({id, EdgeDir::V, x, y, l, x, y+1, l, cap, cost});
                        key_to_idx[k] = id++;
                    }
                }
                // Via-edge
                if (add_via && l+1 <= l1) {
                    auto k = edge_key(l, x, y, 2);
                    if (key_to_idx.find(k) == key_to_idx.end()) {
                        // via capacity: limited by the smaller of the two layer caps
                        // (simplified: use a large default cap for vias)
                        double cap = 1000.0;
                        double cost = grid.unit_via_cost;
                        edges.push_back({id, EdgeDir::Via, x, y, l, x, y, l+1, cap, cost});
                        key_to_idx[k] = id++;
                    }
                }
            }
        }
    }
}

// ── COO accumulator for building CSR ─────────────────────────────────────────

struct COOMatrix {
    int n_rows = 0, n_cols = 0;
    std::vector<int>    row_idx;
    std::vector<int>    col_idx;
    std::vector<double> val;

    void add(int r, int c, double v) {
        row_idx.push_back(r);
        col_idx.push_back(c);
        val.push_back(v);
    }

    // Convert to CSR
    void to_csr(std::vector<int>& csr_row, std::vector<int>& csr_col,
                std::vector<double>& csr_val) const {
        csr_row.assign(n_rows + 1, 0);
        // Count per row
        for (int r : row_idx) csr_row[r + 1]++;
        // Prefix sum
        for (int i = 1; i <= n_rows; ++i) csr_row[i] += csr_row[i-1];
        int nnz = (int)row_idx.size();
        csr_col.resize(nnz);
        csr_val.resize(nnz);
        std::vector<int> pos(csr_row.begin(), csr_row.begin() + n_rows);
        for (int i = 0; i < nnz; ++i) {
            int r = row_idx[i];
            int p = pos[r]++;
            csr_col[p] = col_idx[i];
            csr_val[p] = val[i];
        }
    }
};

// ── main builder ──────────────────────────────────────────────────────────────

bool build_routing_lp(const std::vector<TwoNet>& all_nets,
                      const GridInfo& grid,
                      const LPBuilderConfig& cfg,
                      RoutingLPProblem& prob)
{
    // 1. Filter nets by HPWL
    std::vector<TwoNet> nets;
    for (const auto& n : all_nets) {
        if (hpwl(n) <= cfg.max_hpwl) nets.push_back(n);
    }
    if (nets.empty()) {
        std::cerr << "[lp_builder] No nets after HPWL filter (max=" << cfg.max_hpwl << ")\n";
        return false;
    }

    // 2. Compute global bounding box over all accepted nets
    int l0 = grid.L, l1 = 0, x0 = grid.X, x1 = 0, y0 = grid.Y, y1 = 0;
    for (const auto& n : nets) {
        for (const GCell* g : {&n.src.loc, &n.snk.loc}) {
            l0 = std::min(l0, g->l); l1 = std::max(l1, g->l);
            x0 = std::min(x0, g->x); x1 = std::max(x1, g->x);
            y0 = std::min(y0, g->y); y1 = std::max(y1, g->y);
        }
    }
    // Clamp to grid (apply margin before clamping)
    l0 = std::max(0,        l0 - cfg.margin); l1 = std::min(grid.L-1, l1 + cfg.margin);
    x0 = std::max(0,        x0 - cfg.margin); x1 = std::min(grid.X-1, x1 + cfg.margin);
    y0 = std::max(0,        y0 - cfg.margin); y1 = std::min(grid.Y-1, y1 + cfg.margin);

    std::cout << "[lp_builder] " << nets.size() << " nets, bounding box "
              << "l=[" << l0 << "," << l1 << "] x=[" << x0 << "," << x1
              << "] y=[" << y0 << "," << y1 << "]\n";

    // 3. Pre-flight size estimate before expensive enumeration.
    //    Use layer_dir to count H- and V-preferred edges separately —
    //    each layer contributes edges only in its preferred direction,
    //    so this estimate is tight rather than a 2× worst-case overcount.
    {
        int64_t bx = x1 - x0 + 1, by = y1 - y0 + 1, bl = l1 - l0 + 1;
        int64_t h_edges = 0, v_edges = 0;
        for (int l = l0; l <= l1; ++l) {
            if (grid.layer_dir[l] == 0) h_edges += (bx - 1) * by;   // H-layer
            else                         v_edges += bx * (by - 1);   // V-layer
        }
        int64_t via_edges = cfg.add_via_edges ? bx * by * (bl > 1 ? bl - 1 : 0) : 0;
        int64_t est_edges = h_edges + v_edges + via_edges;
        int64_t n_nets64  = (int64_t)nets.size();
        int64_t est_vars  = n_nets64 * est_edges * 2LL;
        constexpr int64_t MAX_VARS_EST = 10'000'000LL;
        if (est_vars > MAX_VARS_EST) {
            std::cerr << "[lp_builder] Pre-flight reject: est_n_vars=" << est_vars
                      << " > limit=" << MAX_VARS_EST
                      << "\n  bbox l=[" << l0 << "," << l1 << "]"
                      << " x=[" << x0 << "," << x1 << "]"
                      << " y=[" << y0 << "," << y1 << "]"
                      << "  n_nets=" << n_nets64
                      << "  est_edges=" << est_edges
                      << " (h=" << h_edges << " v=" << v_edges << " via=" << via_edges << ")"
                      << "\n  Reduce --max-nets or --max-hpwl to shrink the joint bounding box.\n";
            return false;
        }
    }

    // 3b. Now enumerate edges (safe: pre-flight passed)
    std::vector<Edge> edges;
    std::unordered_map<EdgeKey,int,EdgeKeyHash> key_to_idx;
    enumerate_edges(grid, l0, l1, x0, x1, y0, y1, cfg.add_via_edges, edges, key_to_idx);

    int n_nets  = (int)nets.size();
    int n_edges = (int)edges.size();
    if (n_edges == 0) {
        std::cerr << "[lp_builder] No subgraph edges\n";
        return false;
    }

    // 4. Build node map (subgraph node index)
    auto node_flat = [&](int x, int y, int l) { return node_id(x, y, l, grid.X, grid.Y); };

    std::unordered_map<int,int> node_map_raw;
    for (const auto& e : edges) {
        int a = node_flat(e.xl, e.yl, e.ll);
        int b = node_flat(e.xh, e.yh, e.lh);
        if (node_map_raw.find(a) == node_map_raw.end())
            node_map_raw[a] = (int)node_map_raw.size();
        if (node_map_raw.find(b) == node_map_raw.end())
            node_map_raw[b] = (int)node_map_raw.size();
    }
    int n_nodes_sub = (int)node_map_raw.size();

    // ── Comprehensive overflow guard ─────────────────────────────────────────
    // All size multiplications done in int64_t; abort if any exceeds limits.
    constexpr int64_t MAX_VARS = 10'000'000LL;        // LP variables
    constexpr int64_t MAX_CONS = 20'000'000LL;        // LP constraints
    constexpr int64_t MAX_NNZ  = 100'000'000LL;       // CSR non-zeros
    constexpr int64_t MAX_INT  = (int64_t)INT_MAX;

    int64_t n_nets64       = n_nets;
    int64_t n_edges64      = n_edges;
    int64_t n_nodes64      = n_nodes_sub;
    int64_t n_vars64       = n_nets64 * n_edges64 * 2LL;
    int64_t n_flow_cons64  = n_nets64 * n_nodes64;
    int64_t n_cons64       = n_edges64 + n_flow_cons64;
    // NNZ upper bound: cap rows (2*n_nets per edge) + flow rows (4 entries per edge per net)
    int64_t nnz64          = n_edges64 * n_nets64 * 2LL   // capacity block
                           + n_edges64 * n_nets64 * 4LL;  // flow block (2 nodes × 2 dirs)

    auto check = [&](const char* name, int64_t val, int64_t limit) -> bool {
        if (val > limit) {
            std::cerr << "[lp_builder] Overflow guard: " << name << "=" << val
                      << " > limit=" << limit
                      << "  (n_nets=" << n_nets64
                      << " n_edges=" << n_edges64
                      << " n_nodes_sub=" << n_nodes64 << ")\n"
                      << "  Reduce --max-nets or --max-hpwl.\n";
            return false;
        }
        return true;
    };
    if (!check("n_vars",      n_vars64,      MAX_VARS)) return false;
    if (!check("n_cons",      n_cons64,      MAX_CONS)) return false;
    if (!check("nnz",         nnz64,         MAX_NNZ))  return false;
    if (!check("n_vars/INT",  n_vars64,      MAX_INT))  return false;
    if (!check("n_cons/INT",  n_cons64,      MAX_INT))  return false;
    if (!check("nnz/INT",     nnz64,         MAX_INT))  return false;

    // Also guard total_nodes used for node_map below
    int64_t total_nodes64 = (int64_t)grid.L * grid.X * grid.Y;
    if (!check("total_nodes/INT", total_nodes64, MAX_INT)) return false;

    int n_vars = (int)n_vars64;
    int n_cons = (int)n_cons64;

    // var_idx: safe because n_vars <= MAX_VARS <= INT_MAX
    auto var_idx = [&](int n, int e, int d) -> int {
        return n * n_edges * 2 + e * 2 + d;
    };

    // 5. Constraint layout:
    //    rows [0 .. n_edges-1]                   : capacity
    //    rows [n_edges .. n_edges+n_nets*n_nodes_sub-1]: flow conservation

    COOMatrix A;
    A.n_rows = n_cons;
    A.n_cols = n_vars;

    std::vector<double> con_lb(n_cons), con_ub(n_cons);
    std::vector<double> obj_c(n_vars, 0.0);
    std::vector<double> var_lb(n_vars, 0.0), var_ub(n_vars, 1.0);

    // 6. Capacity constraints
    for (int e = 0; e < n_edges; ++e) {
        con_lb[e] = 0.0;
        con_ub[e] = edges[e].cap;
        for (int n = 0; n < n_nets; ++n) {
            A.add(e, var_idx(n, e, 0), 1.0);   // fwd
            A.add(e, var_idx(n, e, 1), 1.0);   // bwd
        }
    }

    // 7. Flow conservation constraints
    // For each (net, subgraph_node):
    //   Σ_{e: tail→*} x_{n,e,fwd} + Σ_{e: *→tail} x_{n,e,bwd} (outgoing in directed graph)
    //   - Σ_{e: *→head} x_{n,e,fwd} - Σ_{e: tail→*} x_{n,e,bwd}  (incoming in directed graph)
    //   = rhs (1 if source, -1 if sink, 0 otherwise)
    //
    // Interpretation: signed flow. For directed edge fwd (u→v):
    //   contributes +1 to out-degree of u, -1 to in-degree of v... simplified:
    // Actually we use: net flow out of node = demand
    //   demand(source) = +1, demand(sink) = -1, intermediate = 0
    // Flow balance: Σ_{e: head at v} x_fwd - Σ_{e: tail at v} x_fwd
    //            + Σ_{e: tail at v} x_bwd - Σ_{e: head at v} x_bwd = -demand(v)
    // i.e. (in_fwd - out_fwd) + (out_bwd - in_bwd) = -demand
    // Equivalently: out_fwd - in_fwd + in_bwd - out_bwd = demand

    // Pre-build node adjacency for edges
    // edge_tail_subidx[e] = subgraph node index of tail
    // edge_head_subidx[e] = subgraph node index of head
    std::vector<int> tail_sub(n_edges), head_sub(n_edges);
    for (int e = 0; e < n_edges; ++e) {
        tail_sub[e] = node_map_raw.at(node_flat(edges[e].xl, edges[e].yl, edges[e].ll));
        head_sub[e] = node_map_raw.at(node_flat(edges[e].xh, edges[e].yh, edges[e].lh));
    }

    auto flow_con_row = [&](int n, int v_sub) -> int {
        return n_edges + n * n_nodes_sub + v_sub;
    };

    // Initialise rhs = 0 for all flow constraints
    for (int i = n_edges; i < n_cons; ++i) {
        con_lb[i] = con_ub[i] = 0.0;  // will overwrite for src/snk
    }

    // Add edge contributions to flow conservation
    for (int e = 0; e < n_edges; ++e) {
        int t = tail_sub[e];
        int h = head_sub[e];
        for (int n = 0; n < n_nets; ++n) {
            int r_t = flow_con_row(n, t);
            int r_h = flow_con_row(n, h);
            // fwd (tail→head): +1 at tail (leaves tail), -1 at head (enters head)
            A.add(r_t, var_idx(n, e, 0), +1.0);
            A.add(r_h, var_idx(n, e, 0), -1.0);
            // bwd (head→tail): -1 at tail (enters tail from bwd), +1 at head (leaves head)
            A.add(r_t, var_idx(n, e, 1), -1.0);
            A.add(r_h, var_idx(n, e, 1), +1.0);
        }
    }

    // Set rhs for source and sink
    for (int n = 0; n < n_nets; ++n) {
        int src_flat = node_flat(nets[n].src.loc.x, nets[n].src.loc.y, nets[n].src.loc.l);
        int snk_flat = node_flat(nets[n].snk.loc.x, nets[n].snk.loc.y, nets[n].snk.loc.l);

        if (node_map_raw.find(src_flat) == node_map_raw.end() ||
            node_map_raw.find(snk_flat) == node_map_raw.end()) {
            std::cerr << "[lp_builder] net " << nets[n].name
                      << " pin outside subgraph, skipping flow constraints\n";
            continue;
        }

        int src_sub = node_map_raw.at(src_flat);
        int snk_sub = node_map_raw.at(snk_flat);

        // net flow out of source = +1
        con_lb[flow_con_row(n, src_sub)] = con_ub[flow_con_row(n, src_sub)] = 1.0;
        // net flow out of sink = -1 (net flow into sink = +1)
        con_lb[flow_con_row(n, snk_sub)] = con_ub[flow_con_row(n, snk_sub)] = -1.0;
    }

    // 8. Objective: cost of using each edge per net
    for (int n = 0; n < n_nets; ++n) {
        for (int e = 0; e < n_edges; ++e) {
            double c = edges[e].wire_cost;
            obj_c[var_idx(n, e, 0)] = c;
            obj_c[var_idx(n, e, 1)] = c;
        }
    }

    // 9. Assemble RoutingLPProblem
    prob.n_nets  = n_nets;
    prob.n_edges = n_edges;
    prob.n_vars  = n_vars;
    prob.n_cons  = n_cons;
    prob.obj_c   = std::move(obj_c);
    prob.con_lb  = std::move(con_lb);
    prob.con_ub  = std::move(con_ub);
    prob.var_lb  = std::move(var_lb);
    prob.var_ub  = std::move(var_ub);
    prob.edges   = std::move(edges);
    prob.nets    = nets;
    prob.n_nodes_sub = n_nodes_sub;
    prob.grid_X  = grid.X;
    prob.grid_Y  = grid.Y;

    // Store per-edge sub-indices for solution_rounding path extraction
    prob.edge_tail_sub = std::move(tail_sub);
    prob.edge_head_sub = std::move(head_sub);

    // Store per-net src/snk sub-indices
    prob.net_src_sub.resize(n_nets, -1);
    prob.net_snk_sub.resize(n_nets, -1);
    for (int n = 0; n < n_nets; ++n) {
        int sf = node_flat(nets[n].src.loc.x, nets[n].src.loc.y, nets[n].src.loc.l);
        int tf = node_flat(nets[n].snk.loc.x, nets[n].snk.loc.y, nets[n].snk.loc.l);
        auto sit = node_map_raw.find(sf);
        auto tit = node_map_raw.find(tf);
        if (sit != node_map_raw.end()) prob.net_src_sub[n] = sit->second;
        if (tit != node_map_raw.end()) prob.net_snk_sub[n] = tit->second;
    }

    // Build full node_map vector
    int total_nodes = grid.L * grid.X * grid.Y;
    prob.node_map.assign(total_nodes, -1);
    for (auto& kv : node_map_raw) prob.node_map[kv.first] = kv.second;

    A.to_csr(prob.csr_row, prob.csr_col, prob.csr_val);

    std::cout << "[lp_builder] n_vars=" << n_vars << " n_cons=" << n_cons
              << " nnz=" << A.val.size() << "\n";
    return true;
}

} // namespace rlp
