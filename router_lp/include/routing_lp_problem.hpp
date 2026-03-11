#pragma once
// router_lp/include/routing_lp_problem.hpp
// LP formulation in CSR format, ready for cuPDLPx.
//
// Variable ordering (pure_lp mode):
//   For each (net_idx n, subgraph-edge_idx e, direction d ∈ {0=forward,1=backward}):
//     var_idx = n * 2 * n_edges + e * 2 + d
//
// Constraints ordering:
//   [0 .. n_edges-1]                capacity constraints
//   [n_edges .. n_edges + n_flow-1]  flow conservation constraints
//
// Flow conservation:
//   source node of net n:   Σ_{ê ∈ E+(v)} x_{n,ê,fwd} - Σ_{ê ∈ E-(v)} x_{n,ê,fwd} = 1
//                           (equivalently: out-in = 1 using directed forward convention)
//   sink node:              in-out = 1
//   intermediate node:      in-out = 0
//
// We use the directed edge approach from Pathfinding DAC'23.
// Each undirected edge e in the subgraph gives:
//   x_{n,e,0} = flow in forward direction (tail→head)
//   x_{n,e,1} = flow in backward direction (head→tail)
// Both in [0,1].

#include "routing_types.hpp"
#include <vector>
#include <string>

namespace rlp {

struct RoutingLPProblem {
    int n_nets;
    int n_edges;         // number of *undirected* subgraph edges
    int n_vars;          // = n_nets * n_edges * 2
    int n_cons;          // capacity(n_edges) + flow_conservation(n_nets * n_nodes_sub)

    // Objective: c vector [n_vars]
    std::vector<double> obj_c;

    // Constraint matrix in CSR format
    std::vector<int>    csr_row;  // [n_cons+1]
    std::vector<int>    csr_col;  // [nnz]
    std::vector<double> csr_val;  // [nnz]

    // Constraint bounds: lb[i] ≤ A*x ≤ ub[i]
    std::vector<double> con_lb;   // [n_cons]
    std::vector<double> con_ub;   // [n_cons]

    // Variable bounds
    std::vector<double> var_lb;   // [n_vars], all 0
    std::vector<double> var_ub;   // [n_vars], all 1

    // Metadata for debugging / solution decoding
    std::vector<Edge>   edges;    // subgraph edges
    std::vector<TwoNet> nets;
    int n_nodes_sub;              // number of distinct nodes in subgraph
    std::vector<int>    node_map; // global node_id → subgraph node index (-1 if absent)

    // Grid dimensions (needed by solution_rounding to compute flat node ids)
    int grid_X = 0, grid_Y = 0;

    // Subgraph node index for each edge's tail and head (length = n_edges each)
    std::vector<int> edge_tail_sub;  // edge_tail_sub[e] = subgraph node index of tail
    std::vector<int> edge_head_sub;  // edge_head_sub[e] = subgraph node index of head

    // Source and sink subgraph node indices for each net (length = n_nets each)
    std::vector<int> net_src_sub;    // net_src_sub[n] = subgraph index of net n source
    std::vector<int> net_snk_sub;    // net_snk_sub[n] = subgraph index of net n sink
};

struct RoutingLPSolution {
    enum class Status { OPTIMAL, FEAS, INFEASIBLE, TIME_LIMIT, ITER_LIMIT, UNKNOWN };
    Status status;
    double obj_value;
    std::vector<double> x;  // primal solution [n_vars]
    double solve_time_s;
    int iterations;

    // True only when solver found an optimal or feasibility-polished solution.
    // TIME_LIMIT / ITER_LIMIT are NOT considered "solved" for downstream use.
    bool is_solved() const {
        return status == Status::OPTIMAL || status == Status::FEAS;
    }
};

} // namespace rlp
