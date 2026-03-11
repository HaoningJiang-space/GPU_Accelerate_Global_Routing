// router_lp/src/solve_with_cupdlpx.cpp
// B3: Adapter — RoutingLPProblem → cuPDLPx → RoutingLPSolution

#include "../include/solve_with_cupdlpx.hpp"
#include "../../cuPDLPx/include/cupdlpx.h"

#include <iostream>
#include <chrono>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

namespace rlp {

// ── YAML param loader (minimal, no external dependency) ──────────────────────
// Reads key: value pairs from the YAML config file.

static bool load_yaml_params(const std::string& path, pdhg_parameters_t& params) {
    std::ifstream f(path);
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        // Strip comments
        auto sharp = line.find('#');
        if (sharp != std::string::npos) line = line.substr(0, sharp);
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        // Trim leading and trailing whitespace
        auto trim = [](std::string& s) {
            auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) { s.clear(); return; }
            s.erase(0, first);
            auto last = s.find_last_not_of(" \t\r\n");
            if (last != std::string::npos) s.erase(last + 1);
        };
        trim(key); trim(val);
        if (val.empty()) continue;

        try {
            if (key == "eps_optimal_relative")
                params.termination_criteria.eps_optimal_relative = std::stod(val);
            else if (key == "eps_feasible_relative")
                params.termination_criteria.eps_feasible_relative = std::stod(val);
            else if (key == "time_sec_limit")
                params.termination_criteria.time_sec_limit = std::stod(val);
            else if (key == "iteration_limit")
                params.termination_criteria.iteration_limit = std::stoi(val);
            else if (key == "feasibility_polishing")
                params.feasibility_polishing = (val == "true" || val == "1");
            else if (key == "presolve")
                params.presolve = (val == "true" || val == "1");
            else if (key == "verbose")
                params.verbose = (val == "true" || val == "1");
            else if (key == "termination_evaluation_frequency")
                params.termination_evaluation_frequency = std::stoi(val);
        } catch (...) {
            std::cerr << "[cupdlpx_adapter] Warning: could not parse '" << key
                      << "' = '" << val << "'\n";
        }
    }
    return true;
}

// ── main solve ────────────────────────────────────────────────────────────────

bool solve_routing_lp(const RoutingLPProblem& prob,
                      const std::string& config_path,
                      RoutingLPSolution& sol)
{
    // 1. Set parameters
    pdhg_parameters_t params;
    set_default_parameters(&params);

    if (!config_path.empty()) {
        if (load_yaml_params(config_path, params)) {
            std::cout << "[cupdlpx_adapter] Loaded params from " << config_path << "\n";
        } else {
            std::cout << "[cupdlpx_adapter] Could not load " << config_path
                      << ", using defaults\n";
        }
    }

    // 2. Log problem size and active GPU device
    {
        const char* dev = getenv("CUDA_VISIBLE_DEVICES");
        std::cout << "[cupdlpx_adapter] GPU device: CUDA_VISIBLE_DEVICES="
                  << (dev ? dev : "(unset/default)") << "\n"
                  << "[cupdlpx_adapter] Problem: n_vars=" << prob.n_vars
                  << " n_cons=" << prob.n_cons
                  << " nnz=" << prob.csr_col.size()
                  << " n_nets=" << prob.n_nets
                  << " n_edges=" << prob.n_edges << "\n";
    }

    // 3. Build matrix descriptor (CSR)
    matrix_desc_t A_desc;
    A_desc.m   = prob.n_cons;
    A_desc.n   = prob.n_vars;
    A_desc.fmt = matrix_csr;
    A_desc.data.csr.nnz     = (int)prob.csr_col.size();
    A_desc.data.csr.row_ptr = prob.csr_row.data();
    A_desc.data.csr.col_ind = prob.csr_col.data();
    A_desc.data.csr.vals    = prob.csr_val.data();

    // 3. Create LP problem
    double obj_const = 0.0;
    lp_problem_t* lp = create_lp_problem(
        prob.obj_c.data(),
        &A_desc,
        prob.con_lb.data(),
        prob.con_ub.data(),
        prob.var_lb.data(),
        prob.var_ub.data(),
        &obj_const
    );
    if (!lp) {
        std::cerr << "[cupdlpx_adapter] create_lp_problem failed\n";
        return false;
    }

    // 4. Solve
    auto t0 = std::chrono::steady_clock::now();
    cupdlpx_result_t* res = solve_lp_problem(lp, &params);
    auto t1 = std::chrono::steady_clock::now();

    if (!res) {
        std::cerr << "[cupdlpx_adapter] solve_lp_problem returned null\n";
        lp_problem_free(lp);
        return false;
    }

    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // 5. Parse result
    sol.solve_time_s    = elapsed;
    sol.iterations      = res->total_count;
    sol.obj_value       = res->primal_objective_value;
    sol.primal_residual = res->absolute_primal_residual;

    switch (res->termination_reason) {
        case TERMINATION_REASON_OPTIMAL:
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
            sol.status = RoutingLPSolution::Status::OPTIMAL;
            break;
        case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        case TERMINATION_REASON_DUAL_INFEASIBLE:
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
            sol.status = RoutingLPSolution::Status::INFEASIBLE;
            break;
        case TERMINATION_REASON_TIME_LIMIT:
            sol.status = RoutingLPSolution::Status::TIME_LIMIT;
            break;
        case TERMINATION_REASON_ITERATION_LIMIT:
            sol.status = RoutingLPSolution::Status::ITER_LIMIT;
            break;
        default:
            sol.status = RoutingLPSolution::Status::UNKNOWN;
            break;
    }

    std::cout << "[cupdlpx_adapter] termination="
              << (int)res->termination_reason
              << " iters=" << res->total_count
              << " obj=" << res->primal_objective_value
              << " primal_res=" << res->absolute_primal_residual
              << " time=" << elapsed << "s\n";

    if (res->primal_solution && prob.n_vars > 0) {
        sol.x.assign(res->primal_solution,
                     res->primal_solution + prob.n_vars);
    }

    cupdlpx_result_free(res);
    lp_problem_free(lp);

    return sol.is_solved();
}

} // namespace rlp
