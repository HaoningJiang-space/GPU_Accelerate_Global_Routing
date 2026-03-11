// router_lp/main.cpp — GPU-accelerated LP global router
// Pipeline: read_cap → read_net → decompose → build_lp → solve(GPU) → round → write_out
//
// Usage:
//   router_lp -cap <file.cap> -net <file.net> -out <file.out>
//             [--mode pure_lp|windowed]
//             [--max-nets N]    (default 0 = unlimited)
//             [--max-hpwl N]    (default 40, gcell units; pure_lp mode)
//             [--window-x N]    (default 40; windowed mode)
//             [--window-y N]    (default 40; windowed mode)
//             [--gpu N]         (sets CUDA_VISIBLE_DEVICES before solve)
//             [--config <yaml>] (solver params; default: config/cupdlpx_routing_default.yaml)
//             [--threshold T]   (rounding threshold; default 0.1)
//             [--no-via]        (skip via edges in LP)
//             [--inexact-tol T] (primal residual tolerance for inexact rounding; default 1e-3)

#include "include/file_reader.hpp"
#include "include/routing_lp_builder.hpp"
#include "include/solve_with_cupdlpx.hpp"
#include "include/solution_rounding.hpp"
#include "include/windowed_routing.hpp"
#include "include/adaptive_routing.hpp"
#include "include/lagrangian_router.hpp"
#include "include/lagrangian_gpu.hpp"
#include "include/repair_router.hpp"
#include "include/hotspot_polish.hpp"

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
           " [--mode pure_lp|windowed|adaptive|lagrangian] [--max-nets N] [--max-hpwl N]"
           " [--window-x N] [--window-y N] [--margin N] [--batch-grid N]"
           " [--lag-iters N] [--lag-step F] [--lag-decay F] [--lag-beta F] [--lag-ema F]"
           " [--lag-gpu]"
           " [--no-repair]"
           " [--lag-polish] [--lag-polish-max N]"
           " [--gpu N] [--config yaml] [--threshold T] [--no-via]"
           " [--inexact-tol T]\n";
}

int main(int argc, char* argv[]) {
    // ── Parse arguments ───────────────────────────────────────────────────────
    std::string cap_path, net_path, out_path;
    std::string config_path = "config/cupdlpx_routing_default.yaml";
    std::string mode      = "pure_lp";
    int    max_nets       = 0;
    int    max_hpwl       = 40;
    bool   max_hpwl_set   = false;
    int    window_x       = 40;
    int    window_y       = 40;
    int    margin         = 5;
    int    batch_grid     = 50;
    int    lag_iters      = 50;
    double lag_step       = 0.2;   // default matches LagrangianConfig::step_size
    double lag_decay      = 0.7;   // default matches LagrangianConfig::step_decay
    double lag_beta       = 0.5;   // forward dispersion beta (0=disabled)
    double lag_ema        = 0.3;   // EMA momentum for prev_use smoothing
    int    gpu_id         = -1;   // -1 = not set, respect CUDA_VISIBLE_DEVICES from env
    double threshold      = 0.1;
    double inexact_tol    = 1e-3;
    bool   add_via        = true;
    bool   lag_gpu        = false;  // use GPU-parallel BF Lagrangian
    bool   no_repair      = false;  // disable repair pass after lagrangian
    bool   lag_polish     = false;  // enable cuPDLPx hotspot polishing
    int    lag_polish_max = 200;    // max hotspot nets for polishing LP

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "-cap")           cap_path    = next();
        else if (a == "-net")           net_path    = next();
        else if (a == "-out")           out_path    = next();
        else if (a == "--config")       config_path = next();
        else if (a == "--max-nets")     max_nets    = std::stoi(next());
        else if (a == "--max-hpwl")     { max_hpwl = std::stoi(next()); max_hpwl_set = true; }
        else if (a == "--window-x")     window_x    = std::stoi(next());
        else if (a == "--window-y")     window_y    = std::stoi(next());
        else if (a == "--margin")       margin      = std::stoi(next());
        else if (a == "--batch-grid")   batch_grid  = std::stoi(next());
        else if (a == "--lag-iters")    lag_iters   = std::stoi(next());
        else if (a == "--lag-step")     lag_step    = std::stod(next());
        else if (a == "--lag-decay")    lag_decay   = std::stod(next());
        else if (a == "--lag-beta")     lag_beta    = std::stod(next());
        else if (a == "--lag-ema")      lag_ema     = std::stod(next());
        else if (a == "--gpu")          gpu_id      = std::stoi(next());
        else if (a == "--threshold")    threshold   = std::stod(next());
        else if (a == "--inexact-tol")  inexact_tol = std::stod(next());
        else if (a == "--no-via")       add_via     = false;
        else if (a == "--lag-gpu")      lag_gpu     = true;
        else if (a == "--no-repair")    no_repair   = true;
        else if (a == "--lag-polish")   lag_polish  = true;
        else if (a == "--lag-polish-max") lag_polish_max = std::stoi(next());
        else if (a == "--mode")         mode        = next();
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
    std::cout << "[main] 2-pin nets: " << twonets.size() << "  mode=" << mode << "\n";

    // ── Validate mode ─────────────────────────────────────────────────────────
    static const char* const VALID_MODES[] = {
        "pure_lp", "windowed", "adaptive", "lagrangian", nullptr
    };
    bool mode_ok = false;
    for (const char* const* m = VALID_MODES; *m; ++m)
        if (mode == *m) { mode_ok = true; break; }
    if (!mode_ok) {
        std::cerr << "[main] Unknown mode: '" << mode << "'. "
                     "Valid: pure_lp, windowed, adaptive, lagrangian\n";
        usage(argv[0]); return 1;
    }

    // ── Per-net HPWL filter for non-LP modes ──────────────────────────────────
    // pure_lp: max_hpwl is the joint-bbox limit handled inside build_routing_lp.
    // windowed / adaptive / lagrangian: filter individual 2-pin nets by HPWL
    // only when --max-hpwl was explicitly provided (preserving existing defaults).
    if (mode != "pure_lp" && max_hpwl_set) {
        // Use the same 3D HPWL definition as the LP builder: |dx|+|dy|+|dl|.
        // --max-hpwl 0 filters all nets with HPWL > 0 (consistent with pure_lp).
        auto rm = std::remove_if(twonets.begin(), twonets.end(),
            [&](const rlp::TwoNet& n) {
                int h = std::abs(n.src.loc.x - n.snk.loc.x)
                      + std::abs(n.src.loc.y - n.snk.loc.y)
                      + std::abs(n.src.loc.l - n.snk.loc.l);
                return h > max_hpwl;
            });
        int removed = (int)(twonets.end() - rm);
        twonets.erase(rm, twonets.end());
        std::cout << "[main] HPWL filter (per-net, max=" << max_hpwl
                  << "): kept " << twonets.size()
                  << " removed " << removed << "\n";
    }

    // ── Windowed mode ─────────────────────────────────────────────────────────
    if (mode == "windowed") {
        if (window_x <= 0 || window_y <= 0) {
            std::cerr << "[main] --window-x and --window-y must be > 0"
                         " (got " << window_x << "x" << window_y << ")\n";
            return 1;
        }
        rlp::WindowedRoutingConfig wcfg;
        wcfg.window_x          = window_x;
        wcfg.window_y          = window_y;
        wcfg.add_via_edges     = add_via;
        wcfg.threshold         = threshold;
        wcfg.config_path       = config_path;
        wcfg.inexact_residual_tol = inexact_tol;

        rlp::WindowedRoutingStats wstats;
        auto routes = rlp::run_windowed_routing(twonets, grid, wcfg, wstats);

        if (!rlp::write_out_file(out_path, routes)) {
            std::cerr << "[main] Failed to write output\n"; return 1;
        }

        double wall_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();

        std::cout << "\n=== router_lp summary ===\n"
                  << "  mode         : windowed\n"
                  << "  window_xy    : " << window_x << "x" << window_y << "\n"
                  << "  n_vars       : N/A (per-window)\n"
                  << "  n_cons       : N/A (per-window)\n"
                  << "  nnz          : N/A (per-window)\n"
                  << "  n_nets_in    : " << wstats.n_nets_total     << "\n"
                  << "  routed_nets  : " << wstats.n_nets_routed    << "\n"
                  << "  disconnected : " << wstats.n_nets_disconnected << "\n"
                  << "  unassigned   : " << wstats.n_nets_unassigned << "\n"
                  << "  obj_value    : N/A\n"
                  << "  iterations   : N/A\n"
                  << "  build_time   : " << wstats.total_build_time  << "s\n"
                  << "  solve_time   : " << wstats.total_solve_time  << "s\n"
                  << "  round_time   : " << wstats.total_round_time  << "s\n"
                  << "  total_time   : " << wall_time                << "s\n"
                  << "  output       : " << out_path                 << "\n";

        bool success = (wstats.n_nets_routed > 0 && wstats.n_nets_disconnected == 0
                        && wstats.n_nets_unassigned == 0);
        return success ? 0 : 1;
    }

    // ── Adaptive mode ─────────────────────────────────────────────────────────
    if (mode == "adaptive") {
        rlp::AdaptiveRoutingConfig acfg;
        acfg.margin            = margin;
        acfg.batch_grid_x      = batch_grid;
        acfg.batch_grid_y      = batch_grid;
        acfg.add_via_edges     = add_via;
        acfg.threshold         = threshold;
        acfg.config_path       = config_path;
        acfg.inexact_residual_tol = inexact_tol;

        rlp::AdaptiveRoutingStats astats;
        auto routes = rlp::run_adaptive_routing(twonets, grid, acfg, astats);

        if (!rlp::write_out_file(out_path, routes)) {
            std::cerr << "[main] Failed to write output\n"; return 1;
        }

        double wall_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();

        std::cout << "\n=== router_lp summary ===\n"
                  << "  mode         : adaptive\n"
                  << "  batch_grid   : " << batch_grid << "x" << batch_grid << "\n"
                  << "  margin       : " << margin << "\n"
                  << "  n_vars       : N/A (per-batch)\n"
                  << "  n_cons       : N/A (per-batch)\n"
                  << "  nnz          : N/A (per-batch)\n"
                  << "  n_nets_in    : " << astats.n_nets_total       << "\n"
                  << "  routed_nets  : " << astats.n_nets_routed      << "\n"
                  << "  disconnected : " << astats.n_nets_disconnected << "\n"
                  << "  obj_value    : N/A\n"
                  << "  iterations   : N/A\n"
                  << "  build_time   : " << astats.total_build_time   << "s\n"
                  << "  solve_time   : " << astats.total_solve_time   << "s\n"
                  << "  round_time   : " << astats.total_round_time   << "s\n"
                  << "  total_time   : " << wall_time                 << "s\n"
                  << "  output       : " << out_path                  << "\n";

        bool success = (astats.n_nets_routed > 0 && astats.n_nets_disconnected == 0);
        return success ? 0 : 1;
    }

    // ── Lagrangian mode ───────────────────────────────────────────────────────
    if (mode == "lagrangian") {
        rlp::LagrangianConfig lcfg;
        lcfg.max_iters        = lag_iters;
        lcfg.step_size        = lag_step;
        lcfg.step_decay       = lag_decay;
        lcfg.margin           = margin;
        lcfg.add_via          = add_via;
        lcfg.log_every        = 10;
        lcfg.beta_dispersion  = lag_beta;
        lcfg.ema_momentum     = lag_ema;

        rlp::LagrangianStats lstats;
        auto routes = lag_gpu
            ? rlp::run_lagrangian_routing_gpu(twonets, grid, lcfg, lstats)
            : rlp::run_lagrangian_routing    (twonets, grid, lcfg, lstats);

        // ── Repair pass: re-route disconnected nets with λ=0 ─────────────────
        if (!no_repair) {
            int repaired = rlp::repair_disconnected_nets(
                routes, twonets, grid, /*add_via=*/add_via);
            if (repaired > 0)
                std::cout << "[repair] Repaired " << repaired
                          << " disconnected nets\n";
            lstats.n_nets_disconnected -= repaired;
            lstats.n_nets_routed       += repaired;
        }

        // ── Hotspot polishing with cuPDLPx ────────────────────────────────────
        if (lag_polish) {
            rlp::HotspotPolishConfig polish_cfg;
            polish_cfg.overflow_threshold  = 0.0;
            polish_cfg.max_nets_to_polish  = lag_polish_max;
            polish_cfg.lp_time_limit_s     = 30.0;
            polish_cfg.inexact_tol         = inexact_tol;
            polish_cfg.add_via             = add_via;  // honour --no-via in LP build
            polish_cfg.cupdlpx_config_path = config_path;

            auto pstats = rlp::run_hotspot_polish(routes, twonets, grid, polish_cfg);
            std::cout << "[polish] overflowed_edges=" << pstats.n_overflowed_edges
                      << "  hotspot_nets="  << pstats.n_hotspot_nets
                      << "  improved="      << pstats.n_nets_improved
                      << "  viol_before="   << pstats.viol_before
                      << "  viol_after="    << pstats.viol_after
                      << "  time="          << pstats.polish_time_s << "s\n";
        }

        if (!rlp::write_out_file(out_path, routes)) {
            std::cerr << "[main] Failed to write output\n"; return 1;
        }

        // Recompute violation stats after repair and optional polish so that
        // the summary reflects the final output, not the raw Lagrangian result.
        double post_max_viol = lstats.final_max_violation;
        double post_avg_viol = lstats.final_avg_violation;
        if (!no_repair || lag_polish) {
            rlp::compute_route_violations(routes, grid, post_max_viol, post_avg_viol);
        }

        double wall_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();

        std::cout << "\n=== router_lp summary ===\n"
                  << "  mode         : lagrangian"
                  << (lag_gpu ? " (GPU-BF)" : " (CPU-Dijkstra)") << "\n"
                  << "  lag_iters    : " << lstats.iters_run           << "\n"
                  << "  lag_step     : " << lag_step                   << "\n"
                  << "  lag_decay    : " << lag_decay                  << "\n"
                  << "  margin       : " << margin                     << "\n"
                  << "  n_vars       : N/A (per-net Dijkstra)\n"
                  << "  n_cons       : N/A\n"
                  << "  nnz          : N/A\n"
                  << "  n_nets_in    : " << lstats.n_nets_routed + lstats.n_nets_disconnected << "\n"
                  << "  routed_nets  : " << lstats.n_nets_routed       << "\n"
                  << "  disconnected : " << lstats.n_nets_disconnected  << "\n"
                  << "  max_viol     : " << post_max_viol              << "\n"
                  << "  avg_viol     : " << post_avg_viol              << "\n"
                  << "  obj_value    : N/A\n"
                  << "  iterations   : " << lstats.iters_run           << "\n"
                  << "  build_time   : 0s\n"
                  << "  solve_time   : " << lstats.total_dijkstra_time  << "s\n"
                  << "  round_time   : 0s\n"
                  << "  total_time   : " << wall_time                  << "s\n"
                  << "  output       : " << out_path                   << "\n";

        bool success = (lstats.n_nets_routed > 0 && lstats.n_nets_disconnected == 0
                        && post_max_viol <= 0.0);
        return success ? 0 : 1;
    }

    // ── B2: Build LP (pure_lp mode) ───────────────────────────────────────────
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

    int routed = 0, disconnected = 0;
    double round_time = 0.0;

    if (!solved) {
        std::cerr << "[main] WARNING: solver did not reach OPTIMAL"
                     " (status=" << (int)sol.status << "). "
                     "Skipping rounding.\n";
        // Still emit summary so param-sweep scripts can harvest obj/iters/n_vars.
        double wall_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();
        std::cout << "\n=== router_lp summary ===\n"
                  << "  n_vars       : " << prob.n_vars          << "\n"
                  << "  n_cons       : " << prob.n_cons          << "\n"
                  << "  nnz          : " << prob.csr_col.size()  << "\n"
                  << "  n_nets_in    : " << prob.n_nets          << "\n"
                  << "  routed_nets  : " << 0                    << "\n"
                  << "  disconnected : " << prob.n_nets          << "\n"
                  << "  obj_value    : " << sol.obj_value        << "\n"
                  << "  iterations   : " << sol.iterations       << "\n"
                  << "  build_time   : " << build_time           << "s\n"
                  << "  solve_time   : " << solve_time           << "s\n"
                  << "  round_time   : " << 0.0                  << "s\n"
                  << "  total_time   : " << wall_time            << "s\n"
                  << "  output       : " << out_path             << "\n";
        return 1;
    }

    // ── B4: Round & write ─────────────────────────────────────────────────────
    auto t_round0 = std::chrono::steady_clock::now();
    auto routes = rlp::round_lp_solution(prob, sol, threshold);
    auto t_round1 = std::chrono::steady_clock::now();
    round_time = std::chrono::duration<double>(t_round1 - t_round0).count();

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
