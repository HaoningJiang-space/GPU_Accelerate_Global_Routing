// router_lp/src/lshape_init.cpp
// L-shape warm-start implementation.
// See include/lshape_init.hpp for algorithm description.

#include "../include/lshape_init.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

namespace rlp {

namespace {

// Flat edge index: same convention as lagrangian_router.cpp
// H-edge (l,x,y): l*X*Y + x*Y + y
// V-edge (l,x,y): l*X*Y + x*Y + y
inline int edge_idx(int l, int x, int y, int X, int Y) {
    return l * X * Y + x * Y + y;
}

// Returns the best routing layer index for a horizontal run from x0..x1 at
// row y.  Prefers the first H-preferred layer (dir==0) that is above layer 0
// (layer 0 is pin-access only, not routable as wire).
// Falls back to any layer > 0 if no H layer found.
static int best_h_layer(const GridInfo& grid) {
    for (int l = 1; l < grid.L; ++l)
        if (grid.layer_dir[l] == 0) return l;
    return (grid.L > 1) ? 1 : -1;  // -1 means no usable layer
}

static int best_v_layer(const GridInfo& grid) {
    for (int l = 1; l < grid.L; ++l)
        if (grid.layer_dir[l] == 1) return l;
    return (grid.L > 1) ? 1 : -1;
}

// Cost of routing a horizontal segment on layer l from x_lo to x_hi (exclusive)
// at row y, given current demand.
// Returns sum of (wire_cost_per_edge + overflow_penalty_per_edge).
// overflow_penalty = max(0, demand − cap) as a cheap congestion proxy.
static float seg_cost_h(int l, int x_lo, int x_hi, int y,
                         const GridInfo& grid,
                         const std::vector<float>& demand_h) {
    const int X = grid.X, Y = grid.Y;
    float cost = 0.0f;
    for (int x = x_lo; x < x_hi; ++x) {
        int idx = edge_idx(l, x, y, X, Y);
        float cap  = (float)grid.cap[l][x][y];
        float dem  = demand_h[idx];
        cost += (float)grid.unit_wire_cost + std::max(0.0f, dem - cap);
    }
    return cost;
}

static float seg_cost_v(int l, int x, int y_lo, int y_hi,
                         const GridInfo& grid,
                         const std::vector<float>& demand_v) {
    const int X = grid.X, Y = grid.Y;
    float cost = 0.0f;
    for (int y = y_lo; y < y_hi; ++y) {
        int idx = edge_idx(l, x, y, X, Y);
        float cap  = (float)grid.cap[l][x][y];
        float dem  = demand_v[idx];
        cost += (float)grid.unit_wire_cost + std::max(0.0f, dem - cap);
    }
    return cost;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────

void lshape_warmstart(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    float init_alpha,
    std::vector<float>& lam_h,
    std::vector<float>& lam_v)
{
    const int L = grid.L, X = grid.X, Y = grid.Y;
    const int LXY = L * X * Y;

    const int hl = best_h_layer(grid);
    const int vl = best_v_layer(grid);
    if (hl < 0 || vl < 0) {
        std::cerr << "[lshape_warmstart] No usable routing layers, skipping.\n";
        return;
    }

    // Shared demand accumulation arrays (one per edge class).
    // These are updated in parallel with atomic increments.
    std::vector<float> demand_h(LXY, 0.0f);
    std::vector<float> demand_v(LXY, 0.0f);

    const int n_nets = (int)twonets.size();

    // --- Pass 1: L-shape assignment (parallel, read demand_h/v from prev iter)
    // Each net independently picks H-first or V-first based on local wire cost.
    // Demand is accumulated with atomic increments.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int ni = 0; ni < n_nets; ++ni) {
        const TwoNet& tn = twonets[ni];
        int sx = tn.src.loc.x, sy = tn.src.loc.y;
        int tx = tn.snk.loc.x, ty = tn.snk.loc.y;

        // Canonical order: ensure x/y ranges go lo→hi
        int x_lo = std::min(sx, tx), x_hi = std::max(sx, tx);
        int y_lo = std::min(sy, ty), y_hi = std::max(sy, ty);

        if (x_lo == x_hi && y_lo == y_hi) continue; // same gcell, skip

        // Option A: H-first on hl (row sy), then V on vl from sy to ty at x=tx
        float cost_hv = seg_cost_h(hl, x_lo, x_hi, sy, grid, demand_h)
                      + seg_cost_v(vl, tx,    y_lo, y_hi, grid, demand_v);

        // Option B: V-first on vl (col sx), then H on hl from sx to tx at y=ty
        float cost_vh = seg_cost_v(vl, sx,    y_lo, y_hi, grid, demand_v)
                      + seg_cost_h(hl, x_lo, x_hi, ty, grid, demand_h);

        // Pick the lower-cost orientation and accumulate demand atomically.
        if (cost_hv <= cost_vh) {
            // H-first: horizontal run at y=sy on layer hl
            for (int x = x_lo; x < x_hi; ++x) {
                int idx = edge_idx(hl, x, sy, X, Y);
                #pragma omp atomic
                demand_h[idx] += 1.0f;
            }
            // then vertical run at x=tx on layer vl
            for (int y = y_lo; y < y_hi; ++y) {
                int idx = edge_idx(vl, tx, y, X, Y);
                #pragma omp atomic
                demand_v[idx] += 1.0f;
            }
        } else {
            // V-first: vertical run at x=sx on layer vl
            for (int y = y_lo; y < y_hi; ++y) {
                int idx = edge_idx(vl, sx, y, X, Y);
                #pragma omp atomic
                demand_v[idx] += 1.0f;
            }
            // then horizontal run at y=ty on layer hl
            for (int x = x_lo; x < x_hi; ++x) {
                int idx = edge_idx(hl, x, ty, X, Y);
                #pragma omp atomic
                demand_h[idx] += 1.0f;
            }
        }
    }

    // --- Pass 2: warm-start λ from demand imbalance
    // λ_e = max(0, demand_e - cap_e) * init_alpha
    int n_init_h = 0, n_init_v = 0;
    #pragma omp parallel for schedule(static) reduction(+:n_init_h)
    for (int l = 1; l < L; ++l) {
        if (grid.layer_dir[l] != 0) continue;
        for (int x = 0; x < X - 1; ++x) {
            for (int y = 0; y < Y; ++y) {
                int idx = edge_idx(l, x, y, X, Y);
                float viol = demand_h[idx] - (float)grid.cap[l][x][y];
                if (viol > 0.0f) {
                    lam_h[idx] = viol * init_alpha;
                    ++n_init_h;
                }
            }
        }
    }
    #pragma omp parallel for schedule(static) reduction(+:n_init_v)
    for (int l = 1; l < L; ++l) {
        if (grid.layer_dir[l] != 1) continue;
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y - 1; ++y) {
                int idx = edge_idx(l, x, y, X, Y);
                float viol = demand_v[idx] - (float)grid.cap[l][x][y];
                if (viol > 0.0f) {
                    lam_v[idx] = viol * init_alpha;
                    ++n_init_v;
                }
            }
        }
    }

    std::cout << "[lshape_warmstart] warmed " << n_nets << " nets"
              << "  init_alpha=" << init_alpha
              << "  hot_h=" << n_init_h
              << "  hot_v=" << n_init_v << "\n";
}

} // namespace rlp
