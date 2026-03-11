#pragma once
// router_lp/include/routing_types.hpp
// Basic types shared across router_lp modules.
// Independent of InstantGR — do NOT include InstantGR headers here.

#include <string>
#include <vector>

namespace rlp {

// ── Grid ─────────────────────────────────────────────────────────────────────

struct GridInfo {
    int L, X, Y;                          // num_layers, grid_x, grid_y
    double unit_wire_cost;
    double unit_via_cost;
    std::vector<double> layer_short_cost; // [L]
    std::vector<int> layer_dir;           // [L]: 0=H, 1=V
    // capacity[l][x][y] = capacity of edge from (x,y,l) to (x+1,y,l) (H) or (x,y+1,l) (V)
    std::vector<std::vector<std::vector<double>>> cap; // [L][X][Y]
};

// ── Net / Pin ────────────────────────────────────────────────────────────────

struct GCell {
    int x, y, l;
};

// A single pin = one chosen access point (gcell)
struct Pin {
    GCell loc;
};

// A 2-pin net after FLUTE decomposition (or directly 2-pin from file)
struct TwoNet {
    std::string name;      // decomposed 2-pin name: "orig_name_N"
    std::string orig_name; // original multi-pin net name (for output grouping)
    Pin src, snk;
};

// ── Edge ────────────────────────────────────────────────────────────────────
// Undirected edge in the routing grid.
// H-edge (l,x,y): node (x,y,l) — (x+1,y,l),  dir=0, tail=(x,y,l), head=(x+1,y,l)
// V-edge (l,x,y): node (x,y,l) — (x,y+1,l),  dir=1, tail=(x,y,l), head=(x,y+1,l)
// Via-edge (x,y,l): node (x,y,l) — (x,y,l+1), dir=2

enum class EdgeDir { H = 0, V = 1, Via = 2 };

struct Edge {
    int id;
    EdgeDir dir;
    int xl, yl, ll;   // tail gcell (x,y,l)
    int xh, yh, lh;   // head gcell
    double cap;
    double wire_cost;  // unit cost × length
};

// ── Node flat index ──────────────────────────────────────────────────────────
// node_id(x, y, l, X, Y) = l*X*Y + x*Y + y
inline int node_id(int x, int y, int l, int X, int Y) {
    return l * X * Y + x * Y + y;
}

} // namespace rlp
