// router_lp/src/lagrangian_gpu.cu
// GPU-parallel Lagrangian routing: batched frontier Bellman-Ford SSSP.
//
// Design:
//   One CUDA block per 2-pin net; threads in the block cooperatively relax
//   all edges of that net's bbox subgraph.  A fixed number of BF passes
//   (= bbox diameter = bx+by+L) guarantees convergence for non-negative costs.
//
//   Local node index within bbox: lid = (lx)*by*L + (ly)*L + ll
//     where lx = gx - x0, ly = gy - y0, ll = layer
//   Global edge index:  ei = ll * gX * gY + gx * gY + gy
//     (tail of the undirected edge in the layer preferred direction)
//
//   Integer costs: float_cost * COST_SCALE (=1000) -> int32.
//   atomicMin(int*) is natively supported; avoids CAS loops on float.
//   NaN/Inf safety: lambda is always >= 0 by construction (fmaxf clip).

#include "../include/lagrangian_gpu.hpp"
#include "../include/solution_rounding.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace rlp {

static constexpr int COST_SCALE = 1000;
static constexpr int INF_DIST   = 0x3fffffff;

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "[cuda] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// Per-net bbox descriptor uploaded to GPU once.
struct NetGPU {
    int src_li, snk_li;   // local node indices
    int x0, y0;           // global bbox origin
    int bx, by, L;        // bbox dimensions (L = full layer count)
    int n_local;          // = bx * by * L
    int dist_offset;      // start in d_dist[]
};

// ---- Kernels ----------------------------------------------------------------

__global__ void reset_int_kernel(int* arr, int n, int val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void reset_float_kernel(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0.0f;
}

// Set dist[src_li] = 0 for every net.
__global__ void init_src_kernel(const NetGPU* __restrict__ nets, int* __restrict__ d_dist) {
    int ni = blockIdx.x;
    d_dist[nets[ni].dist_offset + nets[ni].src_li] = 0;
}

// One BF relaxation pass: one block per net, threads iterate over local nodes.
// Repeated max_bf_iters times to guarantee convergence (diameter of bbox graph).
__global__ void bf_pass_kernel(
    const NetGPU* __restrict__ nets,
    int*          __restrict__ d_dist,
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY, bool add_via)
{
    int ni = blockIdx.x;
    const NetGPU& net = nets[ni];
    int* dist = d_dist + net.dist_offset;
    const int bx = net.bx, by = net.by, L = net.L;
    const int n_local = net.n_local;

    for (int li = (int)threadIdx.x; li < n_local; li += (int)blockDim.x) {
        int d_u = dist[li];
        if (d_u >= INF_DIST) continue;

        int ll  = li % L;
        int tmp = li / L;
        int ly  = tmp % by;
        int lx  = tmp / by;
        int gx  = lx + net.x0;
        int gy  = ly + net.y0;

        float base_wire = unit_wire_cost * d_layer_sc[ll];
        float base_via  = unit_via_cost;
        int   dir       = d_layer_dir[ll];

        if (dir == 0) {
            // H forward: (gx,gy,ll) -> (gx+1,gy,ll), edge ei=(ll,gx,gy)
            if (lx + 1 < bx) {
                int ei = ll * gX * gY + gx * gY + gy;
                int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
                atomicMin(dist + (lx+1)*by*L + ly*L + ll, d_u + w);
            }
            // H backward: (gx,gy,ll) -> (gx-1,gy,ll), edge ei=(ll,gx-1,gy)
            if (lx > 0) {
                int ei = ll * gX * gY + (gx-1) * gY + gy;
                int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
                atomicMin(dist + (lx-1)*by*L + ly*L + ll, d_u + w);
            }
        }
        if (dir == 1) {
            // V forward: (gx,gy,ll) -> (gx,gy+1,ll), edge ei=(ll,gx,gy)
            if (ly + 1 < by) {
                int ei = ll * gX * gY + gx * gY + gy;
                int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
                atomicMin(dist + lx*by*L + (ly+1)*L + ll, d_u + w);
            }
            // V backward: (gx,gy,ll) -> (gx,gy-1,ll), edge ei=(ll,gx,gy-1)
            if (ly > 0) {
                int ei = ll * gX * gY + gx * gY + (gy-1);
                int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
                atomicMin(dist + lx*by*L + (ly-1)*L + ll, d_u + w);
            }
        }
        // Via up: (gx,gy,ll) -> (gx,gy,ll+1), via edge ei=(ll,gx,gy)
        if (add_via && ll + 1 < L) {
            int ei = ll * gX * gY + gx * gY + gy;
            int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
            atomicMin(dist + lx*by*L + ly*L + (ll+1), d_u + w);
        }
        // Via down: (gx,gy,ll) -> (gx,gy,ll-1), via edge ei=(ll-1,gx,gy)
        if (add_via && ll > 0) {
            int ei = (ll-1) * gX * gY + gx * gY + gy;
            int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
            atomicMin(dist + lx*by*L + ly*L + (ll-1), d_u + w);
        }
    }
}

// Inner-loop BF kernel: runs all max_bf_iters passes in a single kernel launch.
// One block per net; threads cooperate on that net's bbox dist[] in global memory.
// __syncthreads() replaces the per-pass cudaDeviceSynchronize():
//   - each net's dist[] is private to its block, so no inter-block dependency exists.
//   - __syncthreads() ensures all atomicMin writes of pass k are visible before pass k+1.
__global__ void bf_iters_kernel(
    const NetGPU* __restrict__ nets,
    int*          __restrict__ d_dist,
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY, bool add_via, int max_bf_iters)
{
    int ni = blockIdx.x;
    const NetGPU& net = nets[ni];
    int* dist = d_dist + net.dist_offset;
    const int bx = net.bx, by = net.by, L = net.L;
    const int n_local = net.n_local;

    for (int pass = 0; pass < max_bf_iters; ++pass) {
        for (int li = (int)threadIdx.x; li < n_local; li += (int)blockDim.x) {
            int d_u = dist[li];
            if (d_u >= INF_DIST) continue;

            int ll  = li % L;
            int tmp = li / L;
            int ly  = tmp % by;
            int lx  = tmp / by;
            int gx  = lx + net.x0;
            int gy  = ly + net.y0;

            float base_wire = unit_wire_cost * d_layer_sc[ll];
            float base_via  = unit_via_cost;
            int   dir       = d_layer_dir[ll];

            if (dir == 0) {
                if (lx + 1 < bx) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
                    atomicMin(dist + (lx+1)*by*L + ly*L + ll, d_u + w);
                }
                if (lx > 0) {
                    int ei = ll * gX * gY + (gx-1) * gY + gy;
                    int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
                    atomicMin(dist + (lx-1)*by*L + ly*L + ll, d_u + w);
                }
            }
            if (dir == 1) {
                if (ly + 1 < by) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
                    atomicMin(dist + lx*by*L + (ly+1)*L + ll, d_u + w);
                }
                if (ly > 0) {
                    int ei = ll * gX * gY + gx * gY + (gy-1);
                    int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
                    atomicMin(dist + lx*by*L + (ly-1)*L + ll, d_u + w);
                }
            }
            if (add_via && ll + 1 < L) {
                int ei = ll * gX * gY + gx * gY + gy;
                int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
                atomicMin(dist + lx*by*L + ly*L + (ll+1), d_u + w);
            }
            if (add_via && ll > 0) {
                int ei = (ll-1) * gX * gY + gx * gY + gy;
                int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
                atomicMin(dist + lx*by*L + ly*L + (ll-1), d_u + w);
            }
        }
        // Barrier: all threads in this block must finish atomicMin writes before
        // the next pass reads dist[].  Safe because dist[] is private to this block.
        __syncthreads();
    }
}

// Trace path snk->src per net (one thread per net), accumulate edge usage.
// Exact integer equality dist[v]+w == dist[cur] is valid because both BF and
// trace use the same lambda values and the same integer truncation formula.
__global__ void trace_and_use_kernel(
    const NetGPU* __restrict__ nets,
    const int*    __restrict__ d_dist,
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY,
    float* __restrict__ d_use_h,
    float* __restrict__ d_use_v,
    float* __restrict__ d_use_via,
    int n_nets, bool add_via)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n_nets) return;

    const NetGPU& net  = nets[ni];
    const int*    dist = d_dist + net.dist_offset;
    const int bx = net.bx, by = net.by, L = net.L;
    const int x0 = net.x0, y0 = net.y0;

    int cur = net.snk_li;
    if (dist[cur] >= INF_DIST) return;

    int max_steps = bx + by + L + 10;
    for (int step = 0; step < max_steps && cur != net.src_li; ++step) {
        int ll  = cur % L;
        int tmp = cur / L;
        int ly  = tmp % by;
        int lx  = tmp / by;
        int gx  = lx + x0;
        int gy  = ly + y0;

        int   d_cur     = dist[cur];
        float base_wire = unit_wire_cost * d_layer_sc[ll];
        float base_via  = unit_via_cost;
        int   dir       = d_layer_dir[ll];
        bool  moved     = false;

        // Check each possible predecessor in the same order as bf_pass
        if (!moved && dir == 0 && lx+1 < bx) {
            int v  = (lx+1)*by*L + ly*L + ll;
            int ei = ll*gX*gY + gx*gY + gy;
            int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_h + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved && dir == 0 && lx > 0) {
            int v  = (lx-1)*by*L + ly*L + ll;
            int ei = ll*gX*gY + (gx-1)*gY + gy;
            int w  = (int)((base_wire + fmaxf(0.0f, d_lam_h[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_h + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved && dir == 1 && ly+1 < by) {
            int v  = lx*by*L + (ly+1)*L + ll;
            int ei = ll*gX*gY + gx*gY + gy;
            int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_v + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved && dir == 1 && ly > 0) {
            int v  = lx*by*L + (ly-1)*L + ll;
            int ei = ll*gX*gY + gx*gY + (gy-1);
            int w  = (int)((base_wire + fmaxf(0.0f, d_lam_v[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_v + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved && add_via && ll+1 < L) {
            int v  = lx*by*L + ly*L + (ll+1);
            int ei = ll*gX*gY + gx*gY + gy;
            int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_via + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved && add_via && ll > 0) {
            int v  = lx*by*L + ly*L + (ll-1);
            int ei = (ll-1)*gX*gY + gx*gY + gy;
            int w  = (int)((base_via + fmaxf(0.0f, d_lam_via[ei])) * COST_SCALE);
            if (dist[v] != INF_DIST && dist[v] + w == d_cur) {
                atomicAdd(d_use_via + ei, 1.0f); cur = v; moved = true;
            }
        }
        if (!moved) break;
    }
}

// Subgradient update: lambda_e <- max(0, lambda_e + alpha*(use_e - cap_e))
__global__ void update_lam_kernel(
    float* __restrict__ d_lam, const float* __restrict__ d_use,
    const float* __restrict__ d_cap, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_lam[i] = fmaxf(0.0f, d_lam[i] + alpha * (d_use[i] - d_cap[i]));
}

__global__ void update_lam_via_kernel(
    float* __restrict__ d_lam_via, const float* __restrict__ d_use_via,
    float via_cap, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_lam_via[i] = fmaxf(0.0f, d_lam_via[i] + alpha * (d_use_via[i] - via_cap));
}

// ---- Host context -----------------------------------------------------------

struct LagGPUCtx {
    int gX, gY, gL, LXY;
    int n_nets, total_dist_nodes, max_bf_iters;
    std::vector<NetGPU> h_nets;

    NetGPU* d_nets      = nullptr;
    int*    d_dist      = nullptr;
    float*  d_lam_h     = nullptr;
    float*  d_lam_v     = nullptr;
    float*  d_lam_via   = nullptr;
    float*  d_use_h     = nullptr;
    float*  d_use_v     = nullptr;
    float*  d_use_via   = nullptr;
    float*  d_cap_h     = nullptr;
    float*  d_cap_v     = nullptr;
    int*    d_layer_dir = nullptr;
    float*  d_layer_sc  = nullptr;

    float  unit_wire_cost, unit_via_cost;
    double step_size, step_decay, via_cap;
    bool   add_via;

    double t_bf = 0, t_trace = 0, t_lam = 0;
    double final_max_viol = 0.0, final_avg_viol = 0.0;
    int    iters_run = 0;
};

static LagGPUCtx* lag_gpu_init(
    const std::vector<TwoNet>& twonets,
    const GridInfo& grid,
    const LagrangianConfig& cfg)
{
    auto* ctx = new LagGPUCtx();
    ctx->gX = grid.X; ctx->gY = grid.Y; ctx->gL = grid.L;
    ctx->LXY = grid.L * grid.X * grid.Y;
    ctx->n_nets = (int)twonets.size();
    ctx->unit_wire_cost = (float)grid.unit_wire_cost;
    ctx->unit_via_cost  = (float)grid.unit_via_cost;
    ctx->step_size  = cfg.step_size;
    ctx->step_decay = cfg.step_decay;
    ctx->via_cap    = cfg.via_cap;
    ctx->add_via    = cfg.add_via;

    const int X = grid.X, Y = grid.Y, L = grid.L;

    ctx->h_nets.resize(ctx->n_nets);
    int dist_offset = 0;
    int max_diam    = 0;

    for (int ni = 0; ni < ctx->n_nets; ++ni) {
        const TwoNet& tn = twonets[ni];
        int x0 = std::max(0,   std::min(tn.src.loc.x, tn.snk.loc.x) - cfg.margin);
        int x1 = std::min(X-1, std::max(tn.src.loc.x, tn.snk.loc.x) + cfg.margin);
        int y0 = std::max(0,   std::min(tn.src.loc.y, tn.snk.loc.y) - cfg.margin);
        int y1 = std::min(Y-1, std::max(tn.src.loc.y, tn.snk.loc.y) + cfg.margin);
        int bx = x1 - x0 + 1, by = y1 - y0 + 1;

        NetGPU& ng    = ctx->h_nets[ni];
        ng.x0 = x0;  ng.y0 = y0;
        ng.bx = bx;  ng.by = by;  ng.L = L;
        ng.n_local    = bx * by * L;
        ng.dist_offset = dist_offset;
        ng.src_li = (tn.src.loc.x - x0)*by*L + (tn.src.loc.y - y0)*L + tn.src.loc.l;
        ng.snk_li = (tn.snk.loc.x - x0)*by*L + (tn.snk.loc.y - y0)*L + tn.snk.loc.l;

        dist_offset += ng.n_local;
        int diam = bx + by + L;
        if (diam > max_diam) max_diam = diam;
    }
    ctx->total_dist_nodes = dist_offset;
    ctx->max_bf_iters     = max_diam;

    long long gpu_mb = ((long long)ctx->total_dist_nodes * 4
                       + (long long)ctx->LXY * 8 * 4) / 1024 / 1024;
    std::cout << "[lag_gpu] " << ctx->n_nets << " nets"
              << "  total_dist_nodes=" << ctx->total_dist_nodes
              << "  max_bf_iters="     << max_diam
              << "  est_gpu_MB="       << gpu_mb << "\n";

    CUDA_CHECK(cudaMalloc(&ctx->d_nets,      ctx->n_nets * sizeof(NetGPU)));
    CUDA_CHECK(cudaMalloc(&ctx->d_dist,      ctx->total_dist_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_h,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_v,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_via,   ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_h,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_v,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_via,   ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_cap_h,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_cap_v,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_layer_dir, L * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_layer_sc,  L * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(ctx->d_nets, ctx->h_nets.data(),
                          ctx->n_nets * sizeof(NetGPU), cudaMemcpyHostToDevice));

    std::vector<int>   h_ldir(L);
    std::vector<float> h_lsc(L);
    for (int l = 0; l < L; ++l) {
        h_ldir[l] = grid.layer_dir[l];
        h_lsc[l]  = (float)grid.layer_short_cost[l];
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_layer_dir, h_ldir.data(), L*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_layer_sc,  h_lsc.data(),  L*sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_cap_h(ctx->LXY, 0.0f), h_cap_v(ctx->LXY, 0.0f);
    for (int l = 0; l < L; ++l) {
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y) {
                int idx = l*X*Y + x*Y + y;
                float c = (float)grid.cap[l][x][y];
                if (grid.layer_dir[l] == 0) h_cap_h[idx] = c;
                else                        h_cap_v[idx] = c;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_cap_h, h_cap_h.data(), ctx->LXY*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_cap_v, h_cap_v.data(), ctx->LXY*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(ctx->d_lam_h,   0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_lam_v,   0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_lam_via, 0, ctx->LXY * sizeof(float)));

    return ctx;
}

static void lag_gpu_run(LagGPUCtx* ctx, int max_iters, int log_every)
{
    const int n_nets = ctx->n_nets;
    const int LXY    = ctx->LXY;
    const int gX = ctx->gX, gY = ctx->gY;

    int grid_upd   = (LXY + 255) / 256;
    int grid_reset = (ctx->total_dist_nodes + 255) / 256;
    int grid_trace = (n_nets + 127) / 128;

    for (int iter = 1; iter <= max_iters; ++iter) {
        float alpha = (float)(ctx->step_size / std::pow((double)iter, ctx->step_decay));

        // Step 1: Reset dist, set source = 0
        auto t0 = std::chrono::steady_clock::now();
        reset_int_kernel<<<grid_reset, 256>>>(ctx->d_dist, ctx->total_dist_nodes, INF_DIST);
        init_src_kernel<<<n_nets, 1>>>(ctx->d_nets, ctx->d_dist);

        // Step 2: Single kernel launch runs all BF passes internally.
        // Each net's block loops max_bf_iters times with __syncthreads() between passes,
        // replacing the previous host-side loop of max_bf_iters separate kernel launches.
        bf_iters_kernel<<<n_nets, 256>>>(
            ctx->d_nets, ctx->d_dist,
            ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
            ctx->d_layer_dir, ctx->d_layer_sc,
            ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
            ctx->add_via, ctx->max_bf_iters);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        ctx->t_bf += std::chrono::duration<double>(t1 - t0).count();

        // Step 3: Reset usage + trace paths
        reset_float_kernel<<<grid_upd, 256>>>(ctx->d_use_h,   LXY);
        reset_float_kernel<<<grid_upd, 256>>>(ctx->d_use_v,   LXY);
        reset_float_kernel<<<grid_upd, 256>>>(ctx->d_use_via, LXY);
        trace_and_use_kernel<<<grid_trace, 128>>>(
            ctx->d_nets, ctx->d_dist,
            ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
            ctx->d_layer_dir, ctx->d_layer_sc,
            ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
            ctx->d_use_h, ctx->d_use_v, ctx->d_use_via, n_nets, ctx->add_via);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        ctx->t_trace += std::chrono::duration<double>(t2 - t1).count();

        // Step 4: Update lambda — skipped on the last iteration so that the
        // final d_dist and d_lam_* remain consistent for path extraction.
        if (iter < max_iters) {
            update_lam_kernel<<<grid_upd, 256>>>(ctx->d_lam_h,   ctx->d_use_h,   ctx->d_cap_h, alpha, LXY);
            update_lam_kernel<<<grid_upd, 256>>>(ctx->d_lam_v,   ctx->d_use_v,   ctx->d_cap_v, alpha, LXY);
            if (ctx->add_via) {
                update_lam_via_kernel<<<grid_upd, 256>>>(
                    ctx->d_lam_via, ctx->d_use_via, (float)ctx->via_cap, alpha, LXY);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        auto t3 = std::chrono::steady_clock::now();
        ctx->t_lam  += std::chrono::duration<double>(t3 - t2).count();
        ctx->iters_run = iter;

        // Violation report: always on the last iteration, also on log checkpoints.
        bool is_last = (iter == max_iters);
        if (is_last || (log_every > 0 && iter % log_every == 0)) {
            std::vector<float> h_use_h(LXY), h_use_v(LXY), h_use_via(LXY);
            std::vector<float> h_cap_h(LXY), h_cap_v(LXY);
            CUDA_CHECK(cudaMemcpy(h_use_h.data(),   ctx->d_use_h,   LXY*4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_use_v.data(),   ctx->d_use_v,   LXY*4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_use_via.data(), ctx->d_use_via, LXY*4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_cap_h.data(),   ctx->d_cap_h,   LXY*4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_cap_v.data(),   ctx->d_cap_v,   LXY*4, cudaMemcpyDeviceToHost));
            double max_v = 0.0, sum_v = 0.0; int n_ov = 0;
            for (int i = 0; i < LXY; ++i) {
                double v = std::max({(double)(h_use_h[i]-h_cap_h[i]),
                                     (double)(h_use_v[i]-h_cap_v[i]),
                                     (double)(h_use_via[i]-(float)ctx->via_cap)});
                if (v > max_v) max_v = v;
                if (v > 0) { sum_v += v; ++n_ov; }
            }
            ctx->final_max_viol = max_v;
            ctx->final_avg_viol = n_ov > 0 ? sum_v / n_ov : 0.0;
            std::cout << "[lag_gpu] iter=" << iter
                      << "  alpha=" << alpha
                      << "  max_viol=" << max_v
                      << "  overload_edges=" << n_ov << "\n";
        }
    }
}

static std::vector<NetRoute> lag_gpu_extract_routes(
    LagGPUCtx* ctx, const std::vector<TwoNet>& twonets)
{
    const int gX = ctx->gX, gY = ctx->gY;

    std::vector<int>   h_dist(ctx->total_dist_nodes);
    std::vector<float> h_lam_h(ctx->LXY), h_lam_v(ctx->LXY), h_lam_via(ctx->LXY);
    std::vector<int>   h_ldir(ctx->gL);
    std::vector<float> h_lsc(ctx->gL);

    CUDA_CHECK(cudaMemcpy(h_dist.data(),    ctx->d_dist,      ctx->total_dist_nodes*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lam_h.data(),   ctx->d_lam_h,     ctx->LXY*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lam_v.data(),   ctx->d_lam_v,     ctx->LXY*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lam_via.data(), ctx->d_lam_via,   ctx->LXY*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ldir.data(),    ctx->d_layer_dir, ctx->gL*4,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lsc.data(),     ctx->d_layer_sc,  ctx->gL*4,  cudaMemcpyDeviceToHost));

    std::vector<NetRoute> routes;
    routes.reserve(ctx->n_nets);

    for (int ni = 0; ni < ctx->n_nets; ++ni) {
        const NetGPU& ng  = ctx->h_nets[ni];
        const int*    dist = h_dist.data() + ng.dist_offset;
        const int bx = ng.bx, by = ng.by, L = ng.L;
        const int x0 = ng.x0, y0 = ng.y0;

        NetRoute r;
        r.name = twonets[ni].name;

        if (dist[ng.snk_li] >= INF_DIST) {
            routes.push_back(std::move(r));
            continue;
        }

        int cur       = ng.snk_li;
        int max_steps = bx + by + L + 10;

        for (int step = 0; step < max_steps && cur != ng.src_li; ++step) {
            int ll  = cur % L;
            int tmp = cur / L;
            int ly  = tmp % by;
            int lx  = tmp / by;
            int gx  = lx + x0;
            int gy  = ly + y0;
            int d_cur     = dist[cur];
            float base_wire = ctx->unit_wire_cost * h_lsc[ll];
            float base_via  = ctx->unit_via_cost;
            int   dir       = h_ldir[ll];
            bool  moved     = false;

            // try_pred: check if predecessor v is the correct prev-hop
            // Emits segment (sx1,sy1,sz1)->(sx2,sy2,sz2) if matched.
            auto try_pred = [&](int v, int w,
                                int sx1, int sy1, int sz1,
                                int sx2, int sy2, int sz2) {
                if (moved || v < 0 || v >= ng.n_local) return;
                if (dist[v] == INF_DIST || dist[v] + w != d_cur) return;
                RoutingSegment seg;
                seg.x1=sx1; seg.y1=sy1; seg.z1=sz1;
                seg.x2=sx2; seg.y2=sy2; seg.z2=sz2;
                r.segments.push_back(seg);
                cur = v; moved = true;
            };

            if (dir == 0) {
                if (lx+1 < bx) {
                    int v=(lx+1)*by*L+ly*L+ll, ei=ll*gX*gY+gx*gY+gy;
                    int w=(int)((base_wire+fmaxf(0.0f,h_lam_h[ei]))*COST_SCALE);
                    try_pred(v, w, gx, gy, ll, gx+1, gy, ll);
                }
                if (lx > 0) {
                    int v=(lx-1)*by*L+ly*L+ll, ei=ll*gX*gY+(gx-1)*gY+gy;
                    int w=(int)((base_wire+fmaxf(0.0f,h_lam_h[ei]))*COST_SCALE);
                    try_pred(v, w, gx-1, gy, ll, gx, gy, ll);
                }
            }
            if (dir == 1) {
                if (ly+1 < by) {
                    int v=lx*by*L+(ly+1)*L+ll, ei=ll*gX*gY+gx*gY+gy;
                    int w=(int)((base_wire+fmaxf(0.0f,h_lam_v[ei]))*COST_SCALE);
                    try_pred(v, w, gx, gy, ll, gx, gy+1, ll);
                }
                if (ly > 0) {
                    int v=lx*by*L+(ly-1)*L+ll, ei=ll*gX*gY+gx*gY+(gy-1);
                    int w=(int)((base_wire+fmaxf(0.0f,h_lam_v[ei]))*COST_SCALE);
                    try_pred(v, w, gx, gy-1, ll, gx, gy, ll);
                }
            }
            if (ll+1 < L) {
                int v=lx*by*L+ly*L+(ll+1), ei=ll*gX*gY+gx*gY+gy;
                int w=(int)((base_via+fmaxf(0.0f,h_lam_via[ei]))*COST_SCALE);
                try_pred(v, w, gx, gy, ll, gx, gy, ll+1);
            }
            if (ll > 0) {
                int v=lx*by*L+ly*L+(ll-1), ei=(ll-1)*gX*gY+gx*gY+gy;
                int w=(int)((base_via+fmaxf(0.0f,h_lam_via[ei]))*COST_SCALE);
                try_pred(v, w, gx, gy, ll-1, gx, gy, ll);
            }
            if (!moved) break;
        }

        routes.push_back(std::move(r));
    }
    return routes;
}

static void lag_gpu_free(LagGPUCtx* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_nets);     cudaFree(ctx->d_dist);
    cudaFree(ctx->d_lam_h);   cudaFree(ctx->d_lam_v);   cudaFree(ctx->d_lam_via);
    cudaFree(ctx->d_use_h);   cudaFree(ctx->d_use_v);   cudaFree(ctx->d_use_via);
    cudaFree(ctx->d_cap_h);   cudaFree(ctx->d_cap_v);
    cudaFree(ctx->d_layer_dir); cudaFree(ctx->d_layer_sc);
    delete ctx;
}

// ---- Public entry point -----------------------------------------------------

std::vector<NetRoute> run_lagrangian_routing_gpu(
    const std::vector<TwoNet>& twonets,
    const GridInfo&            grid,
    const LagrangianConfig&    cfg,
    LagrangianStats&           stats)
{
    auto wall0 = std::chrono::steady_clock::now();

    std::cout << "[lag_gpu] Initializing...\n";
    LagGPUCtx* ctx = lag_gpu_init(twonets, grid, cfg);

    std::cout << "[lag_gpu] Running " << cfg.max_iters << " iterations...\n";
    lag_gpu_run(ctx, cfg.max_iters, cfg.log_every);

    std::cout << "[lag_gpu] Extracting routes...\n";
    auto routes = lag_gpu_extract_routes(ctx, twonets);

    stats.iters_run           = ctx->iters_run;
    stats.final_max_violation = ctx->final_max_viol;
    stats.final_avg_violation = ctx->final_avg_viol;
    stats.total_dijkstra_time = ctx->t_bf;
    stats.total_update_time   = ctx->t_trace + ctx->t_lam;
    for (const auto& r : routes) {
        if (r.segments.empty()) ++stats.n_nets_disconnected;
        else                    ++stats.n_nets_routed;
    }
    stats.total_wall_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall0).count();

    std::cout << "[lag_gpu] Done:"
              << " iters="        << stats.iters_run
              << "  bf_time="     << ctx->t_bf    << "s"
              << "  trace_time="  << ctx->t_trace << "s"
              << "  lam_time="    << ctx->t_lam   << "s"
              << "  routed="      << stats.n_nets_routed
              << " disconnected=" << stats.n_nets_disconnected
              << "  total="       << stats.total_wall_time << "s\n";

    lag_gpu_free(ctx);
    return routes;
}

} // namespace rlp
