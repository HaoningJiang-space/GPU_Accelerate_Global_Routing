// router_lp/main.cpp — GPU-accelerated LP global router
// Pipeline: read_cap → read_net → decompose → build_lp → solve(GPU) → round → write_out
//
// Usage:
//   router_lp -cap <file.cap> -net <file.net> -out <file.out>
//             [--mode pure_lp]
//             [--max-nets N]    (default 0 = unlimited)
//             [--max-hpwl N]    (default 40, gcell units)
//             [--gpu N]         (sets CUDA_VISIBLE_DEVICES before solve)
//             [--config <yaml>] (solver params; default: config/cupdlpx_routing_default.yaml)
//             [--threshold T]   (rounding threshold; default 0.1)
//             [--no-via]        (skip via edges in LP)

#include "include/file_reader.hpp"
#include "include/routing_lp_builder.hpp"
#include "include/solve_with_cupdlpx.hpp"
#include "include/solution_rounding.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " -cap <cap> -net <net> -out <out>"
           " [--mode pure_lp] [--max-nets N] [--max-hpwl N]"
           " [--gpu N] [--config yaml] [--threshold T] [--no-via]\n";
}

int main(int argc, char* argv[]) {
    // ── Parse arguments ───────────────────────────────────────────────────────
    std::string cap_path, net_path, out_path;
    std::string config_path = "config/cupdlpx_routing_default.yaml";
    int    max_nets   = 0;
    int    max_hpwl   = 40;
    int    gpu_id     = -1;   // -1 = not set, respect CUDA_VISIBLE_DEVICES from env
    double threshold  = 0.1;
    bool   add_via    = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "-cap")        cap_path    = next();
        else if (a == "-net")        net_path    = next();
        else if (a == "-out")        out_path    = next();
        else if (a == "--config")    config_path = next();
        else if (a == "--max-nets")  max_nets    = std::stoi(next());
        else if (a == "--max-hpwl")  max_hpwl    = std::stoi(next());
        else if (a == "--gpu")       gpu_id      = std::stoi(next());
        else if (a == "--threshold") threshold   = std::stod(next());
        else if (a == "--no-via")    add_via     = false;
        else if (a == "--mode")      next();  // only pure_lp for now
        else { std::cerr << "Unknown option: " << a << "\n"; usage(argv[0]); return 1; }
    }

    if (cap_path.empty() || net_path.empty() || out_path.empty()) {
        usage(argv[0]); return 1;
    }

    // GPU device: only override CUDA_VISIBLE_DEVICES when --gpu was explicitly given.
    // Otherwise inherit the caller's environment (allows script/CI-level pinning).
    std::string gpu_label;
    if (gpu_id >= 0) {
        std::string cuda_dev = std::to_string(gpu_id);
        setenv("CUDA_VISIBLE_DEVICES", cuda_dev.c_str(), 1);
        gpu_label = cuda_dev;
    } else {
        const char* env_dev = getenv("CUDA_VISIBLE_DEVICES");
        gpu_label = (env_dev && env_dev[0]) ? env_dev : "(default)";
    }
    std::cout << "[main] CUDA_VISIBLE_DEVICES=" << gpu_label
              << "  cap=" << cap_path
              << "  net=" << net_path << "\n";

    auto wall0 = std::chrono::steady_clock::now();

    // ── B1: Read cap ──────────────────────────────────────────────────────────
    rlp::GridInfo grid;
    if (!rlp::read_cap_file(cap_path, grid)) {
        std::cerr << "[main] Failed to read cap file\n"; return 1;
    }
    std::cout << "[main] Grid: L=" << grid.L << " X=" << grid.X << " Y=" << grid.Y << "\n";

    // ── B1: Read nets ─────────────────────────────────────────────────────────
    std::vector<rlp::MultiPinNet> mpnets;
    if (!rlp::read_net_file(net_path, grid, mpnets)) {
        std::cerr << "[main] Failed to read net file\n"; return 1;
    }
    std::cout << "[main] Multi-pin nets: " << mpnets.size() << "\n";

    // Limit number of nets if requested
    if (max_nets > 0 && (int)mpnets.size() > max_nets) {
        mpnets.resize(max_nets);
        std::cout << "[main] Capped to " << max_nets << " nets\n";
    }

    // ── B1: Decompose to 2-pin ────────────────────────────────────────────────
    std::vector<rlp::TwoNet> twonets;
    for (const auto& mp : mpnets) {
        auto decomp = rlp::decompose_to_2pin(mp);
        for (auto& tn : decomp) twonets.push_back(std::move(tn));
    }
    std::cout << "[main] 2-pin nets: " << twonets.size() << "\n";

    // ── B2: Build LP ──────────────────────────────────────────────────────────
    rlp::LPBuilderConfig bcfg;
    bcfg.max_hpwl     = max_hpwl;
    bcfg.add_via_edges = add_via;

    rlp::RoutingLPProblem prob;
    auto t_build0 = std::chrono::steady_clock::now();
    if (!rlp::build_routing_lp(twonets, grid, bcfg, prob)) {
        std::cerr << "[main] build_routing_lp failed (no valid nets?)\n"; return 1;
    }
    auto t_build1 = std::chrono::steady_clock::now();
    double build_time = std::chrono::duration<double>(t_build1 - t_build0).count();

    std::cout << "[main] LP problem: n_vars=" << prob.n_vars
              << " n_cons=" << prob.n_cons
              << " nnz=" << prob.csr_col.size()
              << " n_nets=" << prob.n_nets
              << " n_edges=" << prob.n_edges
              << " n_nodes=" << prob.n_nodes_sub
              << "  build_time=" << build_time << "s\n";

    // ── B3: Solve on GPU ──────────────────────────────────────────────────────
    rlp::RoutingLPSolution sol;
    auto t_solve0 = std::chrono::steady_clock::now();
    bool solved = rlp::solve_routing_lp(prob, config_path, sol);
    auto t_solve1 = std::chrono::steady_clock::now();
    double solve_time = std::chrono::duration<double>(t_solve1 - t_solve0).count();

    std::cout << "[main] Solve: status=" << (int)sol.status
              << " iterations=" << sol.iterations
              << " obj=" << sol.obj_value
              << " solve_time=" << solve_time << "s\n";

    if (!solved) {
        std::cerr << "[main] WARNING: solver did not reach OPTIMAL"
                     " (status=" << (int)sol.status << "). "
                     "Skipping rounding.\n";
        return 1;
    }

    // ── B4: Round & write ─────────────────────────────────────────────────────
    auto t_round0 = std::chrono::steady_clock::now();
    auto routes = rlp::round_lp_solution(prob, sol, threshold);
    auto t_round1 = std::chrono::steady_clock::now();
    double round_time = std::chrono::duration<double>(t_round1 - t_round0).count();

    int routed = 0, disconnected = 0;
    for (const auto& r : routes) {
        if (r.segments.empty()) ++disconnected;
        else ++routed;
    }
    std::cout << "[main] Rounding: routed=" << routed
              << " disconnected=" << disconnected
              << " round_time=" << round_time << "s\n";

    if (!rlp::write_out_file(out_path, routes)) {
        std::cerr << "[main] Failed to write output\n"; return 1;
    }

    double wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n=== router_lp summary ===\n"
              << "  n_vars       : " << prob.n_vars    << "\n"
              << "  n_cons       : " << prob.n_cons    << "\n"
              << "  nnz          : " << prob.csr_col.size() << "\n"
              << "  n_nets_in    : " << prob.n_nets    << "\n"
              << "  routed_nets  : " << routed         << "\n"
              << "  disconnected : " << disconnected   << "\n"
              << "  obj_value    : " << sol.obj_value  << "\n"
              << "  iterations   : " << sol.iterations << "\n"
              << "  build_time   : " << build_time     << "s\n"
              << "  solve_time   : " << solve_time     << "s\n"
              << "  round_time   : " << round_time     << "s\n"
              << "  total_time   : " << wall_time      << "s\n"
              << "  output       : " << out_path       << "\n";

    return 0;
}
