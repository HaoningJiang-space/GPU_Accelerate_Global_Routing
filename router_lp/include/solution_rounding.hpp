#pragma once
// router_lp/include/solution_rounding.hpp
// B4: Extract routing paths from fractional LP solution and write .out file.
//
// Strategy:
//   For each 2-pin net, trace the max-flow path from src to snk.
//   We iteratively select the highest-flow edge leaving the current node,
//   subtract that flow, and advance until we reach the sink.
//   Repeat until total outflow from src < threshold.
//
// Output format:
//   net_name
//   (
//   x1 y1 z1 x2 y2 z2
//   ...
//   )

#include "routing_lp_problem.hpp"
#include <string>
#include <vector>

namespace rlp {

struct RoutingSegment {
    int x1, y1, z1;  // gcell 1 (0-indexed)
    int x2, y2, z2;  // gcell 2
};

struct NetRoute {
    std::string name;      // 2-pin decomposed name
    std::string orig_name; // original multi-pin net name
    std::vector<RoutingSegment> segments;
};

// Round fractional solution to integer routes.
// threshold: x_{n,e} < threshold is treated as zero.
std::vector<NetRoute> round_lp_solution(const RoutingLPProblem& prob,
                                         const RoutingLPSolution& sol,
                                         double threshold = 0.1);

// Write routes to .out file.
bool write_out_file(const std::string& path,
                    const std::vector<NetRoute>& routes);

} // namespace rlp
