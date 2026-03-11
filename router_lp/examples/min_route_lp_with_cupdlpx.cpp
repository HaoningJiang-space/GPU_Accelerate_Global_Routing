/*
 * router_lp/examples/min_route_lp_with_cupdlpx.cpp
 *
 * Phase B0: Minimal routing LP PoC using cuPDLPx C API.
 * Uses the CORRECT directed-edge double-variable formulation from Pathfinding DAC'23.
 *
 * 1-layer 2×2 grid, 2 nets:
 *
 *   Nodes:  0=(0,0)  1=(1,0)  2=(0,1)  3=(1,1)
 *   Undirected edges (oriented tail→head):
 *     e0 (h0): 0→1    e1 (h1): 2→3
 *     e2 (v0): 0→2    e3 (v1): 1→3
 *
 *   Net 0: src=node0, snk=node3  (diagonal)
 *   Net 1: src=node2, snk=node1  (anti-diagonal)
 *
 * Variables: x[n][e][d]  d∈{0=fwd,1=bwd}  →  16 variables total
 *   var_idx(n,e,d) = n*n_edges*2 + e*2 + d
 *
 * Constraints:
 *   Capacity (4):    Σ_n (x[n][e][0]+x[n][e][1]) ≤ 1  per undirected edge
 *   Flow cons (8):   net outflow = demand  (4 nodes × 2 nets)
 *   Total rows: 12
 *     demand = +1 source, -1 sink, 0 intermediate
 *     Flow conservation: Σ_{e:tail=v} x[n,e,0] - Σ_{e:tail=v} x[n,e,1]
 *                      + Σ_{e:head=v} x[n,e,1] - Σ_{e:head=v} x[n,e,0] = demand(n,v)
 *
 * Feasibility check after solving:
 *   1. All x[n][e][d] in [0,1]
 *   2. Capacity constraints satisfied
 *   3. Flow conservation satisfied (within tolerance)
 */

#include "cupdlpx.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

static const char *termination_str(termination_reason_t r) {
    switch (r) {
    case TERMINATION_REASON_OPTIMAL:             return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:   return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:     return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:          return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:     return "ITERATION_LIMIT";
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS: return "FEAS_POLISH_SUCCESS";
    default:                                     return "UNSPECIFIED";
    }
}

int main() {
    printf("=== Min Route LP PoC (directed-edge formulation) ===\n\n");

    // ── Dimensions ────────────────────────────────────────────────────────
    const int n_nets  = 2;
    const int n_edges = 4;   // undirected
    const int n_nodes = 4;
    const int n_vars  = n_nets * n_edges * 2;  // 16
    const int n_cap   = n_edges;               //  4  capacity rows
    const int n_flow  = n_nets * n_nodes;      //  8  flow conservation rows
    const int n_con   = n_cap + n_flow;        // 12

    auto vi = [&](int n, int e, int d) { return n * n_edges * 2 + e * 2 + d; };

    // ── Objective: minimize total routed edges ────────────────────────────
    std::vector<double> obj(n_vars, 1.0);
    std::vector<double> var_lb(n_vars, 0.0);
    std::vector<double> var_ub(n_vars, 1.0);

    // ── Build constraint matrix (COO) ─────────────────────────────────────
    // Edge incidence (tail→head):
    //   e0: 0→1   e1: 2→3   e2: 0→2   e3: 1→3
    //
    // For each undirected edge e with tail t and head h:
    //   x[n,e,0] (fwd, t→h): +1 at t, -1 at h in flow conservation
    //   x[n,e,1] (bwd, h→t): -1 at t, +1 at h in flow conservation
    //
    // Constraint rows:
    //   [0..3]  capacity:       Σ_n (x[n,e,0]+x[n,e,1]) ≤ 1
    //   [4..11] flow cons:      row = 4 + n*n_nodes + v

    // Edge topology: tail[e], head[e]
    const int tail[] = {0, 2, 0, 1};
    const int head[] = {1, 3, 2, 3};

    // Net demands: demand[n][v]  (+1 source, -1 sink, 0 intermediate)
    // Net 0: src=0, snk=3  →  demand[0] = {+1, 0, 0, -1}
    // Net 1: src=2, snk=1  →  demand[1] = {0, -1, +1, 0}
    const double demand[2][4] = {{+1,0,0,-1},{0,-1,+1,0}};

    std::vector<int>    row_ind, col_ind;
    std::vector<double> vals;
    auto add = [&](int r, int c, double v) {
        row_ind.push_back(r); col_ind.push_back(c); vals.push_back(v);
    };

    // Capacity constraints
    for (int e = 0; e < n_edges; ++e)
        for (int n = 0; n < n_nets; ++n) {
            add(e, vi(n,e,0), +1.0);
            add(e, vi(n,e,1), +1.0);
        }

    // Flow conservation constraints
    for (int e = 0; e < n_edges; ++e) {
        int t = tail[e], h = head[e];
        for (int n = 0; n < n_nets; ++n) {
            int row_t = n_cap + n * n_nodes + t;
            int row_h = n_cap + n * n_nodes + h;
            // fwd (t→h): +1 leaves t, -1 enters h
            add(row_t, vi(n,e,0), +1.0);
            add(row_h, vi(n,e,0), -1.0);
            // bwd (h→t): +1 leaves h, -1 enters t
            add(row_h, vi(n,e,1), +1.0);
            add(row_t, vi(n,e,1), -1.0);
        }
    }

    // ── Constraint bounds ─────────────────────────────────────────────────
    std::vector<double> con_lb(n_con), con_ub(n_con);
    for (int e = 0; e < n_cap; ++e) { con_lb[e] = 0.0; con_ub[e] = 1.0; }
    for (int n = 0; n < n_nets; ++n)
        for (int v = 0; v < n_nodes; ++v) {
            int r = n_cap + n * n_nodes + v;
            con_lb[r] = con_ub[r] = demand[n][v];
        }

    // ── Solve ─────────────────────────────────────────────────────────────
    matrix_desc_t A;
    A.m = n_con; A.n = n_vars; A.fmt = matrix_coo;
    A.data.coo.nnz     = (int)vals.size();
    A.data.coo.row_ind = row_ind.data();
    A.data.coo.col_ind = col_ind.data();
    A.data.coo.vals    = vals.data();

    double obj_const = 0.0;
    lp_problem_t *prob = create_lp_problem(
        obj.data(), &A,
        con_lb.data(), con_ub.data(),
        var_lb.data(), var_ub.data(),
        &obj_const);

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.termination_criteria.eps_optimal_relative  = 1e-4;
    params.termination_criteria.eps_feasible_relative = 1e-4;
    params.termination_criteria.time_sec_limit        = 30.0;
    params.feasibility_polishing = true;
    params.verbose = false;

    cupdlpx_result_t *res = solve_lp_problem(prob, &params);

    printf("Termination : %s\n", termination_str(res->termination_reason));
    printf("Primal obj  : %.6f\n", res->primal_objective_value);
    printf("Primal resid: %.2e\n", res->relative_primal_residual);
    printf("Dual resid  : %.2e\n", res->relative_dual_residual);
    printf("Iterations  : %d\n",   res->total_count);
    printf("Solve time  : %.4f s\n\n", res->cumulative_time_sec);

    // Print solution
    const char *enames[] = {"h0","h1","v0","v1"};
    for (int n = 0; n < n_nets; ++n)
        for (int e = 0; e < n_edges; ++e)
            printf("  x[%d][%s][fwd]=%.4f  x[%d][%s][bwd]=%.4f\n",
                   n, enames[e], res->primal_solution[vi(n,e,0)],
                   n, enames[e], res->primal_solution[vi(n,e,1)]);

    // ── Feasibility verification ──────────────────────────────────────────
    const double tol = 1e-3;
    int n_viol = 0;

    // 1. Variable bounds
    for (int j = 0; j < n_vars; ++j) {
        double x = res->primal_solution[j];
        if (x < -tol || x > 1.0 + tol) {
            printf("[FEAS] var %d = %.6f violates [0,1]\n", j, x);
            ++n_viol;
        }
    }

    // 2. Capacity constraints
    for (int e = 0; e < n_edges; ++e) {
        double sum = 0.0;
        for (int n = 0; n < n_nets; ++n)
            sum += res->primal_solution[vi(n,e,0)] + res->primal_solution[vi(n,e,1)];
        if (sum > 1.0 + tol) {
            printf("[FEAS] cap constraint e%d: sum=%.6f > 1\n", e, sum);
            ++n_viol;
        }
    }

    // 3. Flow conservation
    for (int n = 0; n < n_nets; ++n) {
        for (int v = 0; v < n_nodes; ++v) {
            double netflow = 0.0;
            for (int e = 0; e < n_edges; ++e) {
                if (tail[e] == v) netflow += res->primal_solution[vi(n,e,0)]
                                           - res->primal_solution[vi(n,e,1)];
                if (head[e] == v) netflow -= res->primal_solution[vi(n,e,0)]
                                           - res->primal_solution[vi(n,e,1)];
            }
            if (std::abs(netflow - demand[n][v]) > tol) {
                printf("[FEAS] flow cons net%d node%d: netflow=%.6f  demand=%.1f\n",
                       n, v, netflow, demand[n][v]);
                ++n_viol;
            }
        }
    }

    bool solver_ok = (res->termination_reason == TERMINATION_REASON_OPTIMAL ||
                      res->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
    bool feas_ok   = (n_viol == 0);
    bool pass      = solver_ok && feas_ok;

    printf("\nFeasibility violations: %d\n", n_viol);
    printf("[PoC] Status: %s\n", pass ? "PASS" : "FAIL");

    cupdlpx_result_free(res);
    lp_problem_free(prob);
    return pass ? 0 : 1;
}
