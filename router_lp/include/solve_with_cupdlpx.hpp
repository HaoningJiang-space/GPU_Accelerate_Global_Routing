#pragma once
// router_lp/include/solve_with_cupdlpx.hpp
// Adapter: RoutingLPProblem → cuPDLPx C API → RoutingLPSolution

#include "routing_lp_problem.hpp"
#include <string>

namespace rlp {

// Load solver params from YAML config file (optional).
// If config_path is empty or file missing, uses cuPDLPx defaults.
// time_limit_s and inexact_tol, when >= 0, override the YAML values.
// Returns false if cuPDLPx returns INFEASIBLE or an error occurs.
bool solve_routing_lp(const RoutingLPProblem& prob,
                      const std::string& config_path,
                      RoutingLPSolution& sol,
                      double time_limit_s = -1.0,
                      double inexact_tol  = -1.0);

} // namespace rlp
