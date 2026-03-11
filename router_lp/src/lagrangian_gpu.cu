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
#include <cfloat>
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

// Deterministic per-(net,edge) perturbation in {0..10}.
// Added to integer edge cost so that ties in float -> int rounding are broken
// differently for each net, preventing inter-net path clustering.
// Must be used identically in bf_iters_kernel, trace_and_use_kernel, and
// lag_gpu_extract_routes to keep dist[v]+w == d_cur checks consistent.
__host__ __device__ inline int edge_eps(int ni, int ei) {
    return (ni * 7 + ei) % 11;
}

// Small-net threshold: nets with n_local <= SMALL_THRESH have their dist[]
// held in shared memory during BF, saving global-memory atomicMin latency.
// RTX 3090 (sm_86): 48 KB dynamic shared mem per block (opt-in via
// cudaFuncSetAttribute).  12288 ints × 4 B = 48 KB.
// Covers margin=5 nets up to bx*by*L ≤ 12288 (e.g. 35×35×10 or ~110×1×10).
// cudaFuncSetAttribute must be called once before launch (see lag_gpu_init).
static constexpr int SMALL_THRESH     = 12288;
static constexpr int SMALL_SHMEM_BYTES = SMALL_THRESH * (int)sizeof(int); // 49152 B

// Per-net bbox descriptor uploaded to GPU once.
struct NetGPU {
    int src_li, snk_li;   // local node indices
    int x0, y0;           // global bbox origin
    int bx, by, L;        // bbox dimensions (L = full layer count)
    int n_local;          // = bx * by * L
    int dist_offset;      // start in d_dist[]
    int max_bf_iters;     // = bx + by + L (BF diameter for this net)
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

// EMA update: ema[i] = momentum * ema[i] + (1 - momentum) * cur[i].
// Used to smooth the prev_use dispersion signal and prevent 2-period oscillation.
__global__ void ema_update_kernel(float* __restrict__ ema, const float* __restrict__ cur,
                                  float momentum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ema[i] = momentum * ema[i] + (1.0f - momentum) * cur[i];
}

// Set dist[src_li] = 0 for every net.
__global__ void init_src_kernel(const NetGPU* __restrict__ nets, int* __restrict__ d_dist) {
    int ni = blockIdx.x;
    d_dist[nets[ni].dist_offset + nets[ni].src_li] = 0;
}

// Inner-loop BF kernel: runs all BF passes in a single kernel launch (large nets).
// dist[] lives in global memory; one block per net.
// Edge cost = base + lambda + eps(ni,ei) + beta*(prev_use/cap).
// Each block runs net.max_bf_iters passes (exact BF diameter for that net).
__global__ void bf_iters_kernel(
    const NetGPU* __restrict__ nets,
    int*          __restrict__ d_dist,
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const float*  __restrict__ d_use_prev_h,
    const float*  __restrict__ d_use_prev_v,
    const float*  __restrict__ d_use_prev_via,
    const float*  __restrict__ d_cap_h,
    const float*  __restrict__ d_cap_v,
    float via_cap,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY, bool add_via, float beta)
{
    int ni = blockIdx.x;
    const NetGPU& net = nets[ni];
    int* dist = d_dist + net.dist_offset;
    const int bx = net.bx, by = net.by, L = net.L;
    const int n_local = net.n_local;

    // Macro: compute augmented integer edge cost (lambda + dispersion + eps).
    // cap_val > 0 is guaranteed by grid construction; clamp to avoid div-by-zero.
#define EDGE_W_H(ei_)  ((int)((base_wire + fmaxf(0.0f, d_lam_h[ei_]) \
    + beta * fminf(d_use_prev_h[ei_] / fmaxf(d_cap_h[ei_], 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))
#define EDGE_W_V(ei_)  ((int)((base_wire + fmaxf(0.0f, d_lam_v[ei_]) \
    + beta * fminf(d_use_prev_v[ei_] / fmaxf(d_cap_v[ei_], 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))
#define EDGE_W_VIA(ei_) ((int)((base_via + fmaxf(0.0f, d_lam_via[ei_]) \
    + beta * fminf(d_use_prev_via[ei_] / fmaxf(via_cap, 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))

    for (int pass = 0; pass < net.max_bf_iters; ++pass) {
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

            if (dir == 0 && ll > 0) {
                if (lx + 1 < bx) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    atomicMin(dist + (lx+1)*by*L + ly*L + ll, d_u + EDGE_W_H(ei));
                }
                if (lx > 0) {
                    int ei = ll * gX * gY + (gx-1) * gY + gy;
                    atomicMin(dist + (lx-1)*by*L + ly*L + ll, d_u + EDGE_W_H(ei));
                }
            }
            if (dir == 1 && ll > 0) {
                if (ly + 1 < by) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    atomicMin(dist + lx*by*L + (ly+1)*L + ll, d_u + EDGE_W_V(ei));
                }
                if (ly > 0) {
                    int ei = ll * gX * gY + gx * gY + (gy-1);
                    atomicMin(dist + lx*by*L + (ly-1)*L + ll, d_u + EDGE_W_V(ei));
                }
            }
            if (add_via && ll + 1 < L) {
                int ei = ll * gX * gY + gx * gY + gy;
                atomicMin(dist + lx*by*L + ly*L + (ll+1), d_u + EDGE_W_VIA(ei));
            }
            if (add_via && ll > 0) {
                int ei = (ll-1) * gX * gY + gx * gY + gy;
                atomicMin(dist + lx*by*L + ly*L + (ll-1), d_u + EDGE_W_VIA(ei));
            }
        }
        __syncthreads();
    }
#undef EDGE_W_H
#undef EDGE_W_V
#undef EDGE_W_VIA
}

// Small-net BF kernel: dist[] held in shared memory for the duration of all passes.
// Shared atomicMin is ~100x faster than global atomicMin, saving significant
// latency for nets with small bboxes (n_local <= SMALL_THRESH).
//
// Launch: bf_iters_small_kernel<<<n_small, 256, max_small_n_local*sizeof(int)>>>
//   max_small_n_local must be <= SMALL_THRESH; cudaFuncSetAttribute must be
//   called to raise the dynamic shared memory limit to SMALL_SHMEM_BYTES.
// Each block handles one net from d_nets_small (the small-net sub-array).
// Edge costs and global arrays (lambda, cap, use_prev) are the same as the
// large kernel; only dist[] differs (shared vs global).
__global__ void bf_iters_small_kernel(
    const NetGPU* __restrict__ nets,
    int*          __restrict__ d_dist,       // global (for load/store only)
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const float*  __restrict__ d_use_prev_h,
    const float*  __restrict__ d_use_prev_v,
    const float*  __restrict__ d_use_prev_via,
    const float*  __restrict__ d_cap_h,
    const float*  __restrict__ d_cap_v,
    float via_cap,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY, bool add_via, float beta)
{
    extern __shared__ int smem[];     // dist[] for this net, size = n_local ints

    int ni = blockIdx.x;
    const NetGPU& net    = nets[ni];
    int*          gdist  = d_dist + net.dist_offset;  // global pointer
    const int bx = net.bx, by = net.by, L = net.L;
    const int n_local = net.n_local;

    // Load global dist → shared memory
    for (int li = (int)threadIdx.x; li < n_local; li += (int)blockDim.x)
        smem[li] = gdist[li];
    __syncthreads();

#define EDGE_W_H(ei_)  ((int)((base_wire + fmaxf(0.0f, d_lam_h[ei_]) \
    + beta * fminf(d_use_prev_h[ei_] / fmaxf(d_cap_h[ei_], 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))
#define EDGE_W_V(ei_)  ((int)((base_wire + fmaxf(0.0f, d_lam_v[ei_]) \
    + beta * fminf(d_use_prev_v[ei_] / fmaxf(d_cap_v[ei_], 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))
#define EDGE_W_VIA(ei_) ((int)((base_via + fmaxf(0.0f, d_lam_via[ei_]) \
    + beta * fminf(d_use_prev_via[ei_] / fmaxf(via_cap, 1e-6f), 8.0f)) \
    * COST_SCALE) + edge_eps(ni, (ei_)))

    for (int pass = 0; pass < net.max_bf_iters; ++pass) {
        for (int li = (int)threadIdx.x; li < n_local; li += (int)blockDim.x) {
            int d_u = smem[li];
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

            if (dir == 0 && ll > 0) {
                if (lx + 1 < bx) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    atomicMin(smem + (lx+1)*by*L + ly*L + ll, d_u + EDGE_W_H(ei));
                }
                if (lx > 0) {
                    int ei = ll * gX * gY + (gx-1) * gY + gy;
                    atomicMin(smem + (lx-1)*by*L + ly*L + ll, d_u + EDGE_W_H(ei));
                }
            }
            if (dir == 1 && ll > 0) {
                if (ly + 1 < by) {
                    int ei = ll * gX * gY + gx * gY + gy;
                    atomicMin(smem + lx*by*L + (ly+1)*L + ll, d_u + EDGE_W_V(ei));
                }
                if (ly > 0) {
                    int ei = ll * gX * gY + gx * gY + (gy-1);
                    atomicMin(smem + lx*by*L + (ly-1)*L + ll, d_u + EDGE_W_V(ei));
                }
            }
            if (add_via && ll + 1 < L) {
                int ei = ll * gX * gY + gx * gY + gy;
                atomicMin(smem + lx*by*L + ly*L + (ll+1), d_u + EDGE_W_VIA(ei));
            }
            if (add_via && ll > 0) {
                int ei = (ll-1) * gX * gY + gx * gY + gy;
                atomicMin(smem + lx*by*L + ly*L + (ll-1), d_u + EDGE_W_VIA(ei));
            }
        }
        __syncthreads();
    }
#undef EDGE_W_H
#undef EDGE_W_V
#undef EDGE_W_VIA

    // Write shared memory back to global dist[]
    for (int li = (int)threadIdx.x; li < n_local; li += (int)blockDim.x)
        gdist[li] = smem[li];
}


// Tie-breaking: among all equal-cost predecessors, select the one with the
// smallest lambda (less congested), then smallest edge_id (deterministic).
// Lambda is read-only during tracing — no race condition.
// The edge-weight formula MUST match bf_iters_kernel exactly (same eps +
// dispersion term) so that dist[v]+w==d_cur holds along the optimal path.
__global__ void trace_and_use_kernel(
    const NetGPU* __restrict__ nets,
    const int*    __restrict__ d_dist,
    const float*  __restrict__ d_lam_h,
    const float*  __restrict__ d_lam_v,
    const float*  __restrict__ d_lam_via,
    const float*  __restrict__ d_use_prev_h,
    const float*  __restrict__ d_use_prev_v,
    const float*  __restrict__ d_use_prev_via,
    const float*  __restrict__ d_cap_h,
    const float*  __restrict__ d_cap_v,
    float via_cap,
    const int*    __restrict__ d_layer_dir,
    const float*  __restrict__ d_layer_sc,
    float unit_wire_cost, float unit_via_cost,
    int gX, int gY,
    float* __restrict__ d_use_h,
    float* __restrict__ d_use_v,
    float* __restrict__ d_use_via,
    int n_nets, bool add_via, float beta)
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

        // Best predecessor: primary key = min lambda (congestion proxy),
        // secondary key = min edge_id (deterministic). type: 0=H 1=V 2=via.
        int   best_v = -1, best_ei = -1, best_type = -1;
        float best_lam = FLT_MAX;

#define CHECK_PRED_H(v_, ei_) {                                             \
    int v__=(v_), ei__=(ei_);                                               \
    int w__=(int)((base_wire+fmaxf(0.0f,d_lam_h[ei__])                     \
        + beta * fminf(d_use_prev_h[ei__] / fmaxf(d_cap_h[ei__], 1e-6f), 8.0f)) \
        * COST_SCALE) + edge_eps(ni, ei__);                                 \
    if (dist[v__]!=INF_DIST && dist[v__]+w__==d_cur) {                     \
        float lam__=d_lam_h[ei__];                                          \
        if (lam__<best_lam||(lam__==best_lam&&ei__<best_ei)) {             \
            best_v=v__; best_ei=ei__; best_type=0; best_lam=lam__; } } }

#define CHECK_PRED_V(v_, ei_) {                                             \
    int v__=(v_), ei__=(ei_);                                               \
    int w__=(int)((base_wire+fmaxf(0.0f,d_lam_v[ei__])                     \
        + beta * fminf(d_use_prev_v[ei__] / fmaxf(d_cap_v[ei__], 1e-6f), 8.0f)) \
        * COST_SCALE) + edge_eps(ni, ei__);                                 \
    if (dist[v__]!=INF_DIST && dist[v__]+w__==d_cur) {                     \
        float lam__=d_lam_v[ei__];                                          \
        if (lam__<best_lam||(lam__==best_lam&&ei__<best_ei)) {             \
            best_v=v__; best_ei=ei__; best_type=1; best_lam=lam__; } } }

#define CHECK_PRED_VIA(v_, ei_) {                                           \
    int v__=(v_), ei__=(ei_);                                               \
    int w__=(int)((base_via+fmaxf(0.0f,d_lam_via[ei__])                    \
        + beta * fminf(d_use_prev_via[ei__] / fmaxf(via_cap, 1e-6f), 8.0f)) \
        * COST_SCALE) + edge_eps(ni, ei__);                                 \
    if (dist[v__]!=INF_DIST && dist[v__]+w__==d_cur) {                     \
        float lam__=d_lam_via[ei__];                                        \
        if (lam__<best_lam||(lam__==best_lam&&ei__<best_ei)) {             \
            best_v=v__; best_ei=ei__; best_type=2; best_lam=lam__; } } }

        if (dir == 0 && ll > 0) {
            if (lx+1 < bx) CHECK_PRED_H((lx+1)*by*L+ly*L+ll, ll*gX*gY+gx*gY+gy)
            if (lx > 0)    CHECK_PRED_H((lx-1)*by*L+ly*L+ll,  ll*gX*gY+(gx-1)*gY+gy)
        }
        if (dir == 1 && ll > 0) {
            if (ly+1 < by) CHECK_PRED_V(lx*by*L+(ly+1)*L+ll, ll*gX*gY+gx*gY+gy)
            if (ly > 0)    CHECK_PRED_V(lx*by*L+(ly-1)*L+ll,  ll*gX*gY+gx*gY+(gy-1))
        }
        if (add_via) {
            if (ll+1 < L) CHECK_PRED_VIA(lx*by*L+ly*L+(ll+1), ll*gX*gY+gx*gY+gy)
            if (ll > 0)   CHECK_PRED_VIA(lx*by*L+ly*L+(ll-1),  (ll-1)*gX*gY+gx*gY+gy)
        }
#undef CHECK_PRED_H
#undef CHECK_PRED_V
#undef CHECK_PRED_VIA

        if (best_v < 0) break;
        if      (best_type == 0) atomicAdd(d_use_h   + best_ei, 1.0f);
        else if (best_type == 1) atomicAdd(d_use_v   + best_ei, 1.0f);
        else                     atomicAdd(d_use_via + best_ei, 1.0f);
        cur = best_v;
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

    NetGPU* d_nets          = nullptr;   // all nets (for trace, init)
    NetGPU* d_nets_small    = nullptr;   // n_local <= SMALL_THRESH
    NetGPU* d_nets_large    = nullptr;   // n_local >  SMALL_THRESH
    int     n_small         = 0;
    int     n_large         = 0;
    int     max_small_n_local = 0;       // max n_local among small nets (smem size)
    int*    d_dist          = nullptr;
    float*  d_lam_h         = nullptr;
    float*  d_lam_v         = nullptr;
    float*  d_lam_via       = nullptr;
    float*  d_use_h         = nullptr;
    float*  d_use_v         = nullptr;
    float*  d_use_via       = nullptr;
    float*  d_use_prev_h    = nullptr;
    float*  d_use_prev_v    = nullptr;
    float*  d_use_prev_via  = nullptr;
    float*  d_cap_h         = nullptr;
    float*  d_cap_v         = nullptr;
    int*    d_layer_dir     = nullptr;
    float*  d_layer_sc      = nullptr;

    float  unit_wire_cost, unit_via_cost;
    double step_size, step_decay, via_cap;
    float  beta;
    float  ema_momentum;
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
    ctx->beta           = (float)cfg.beta_dispersion;
    ctx->ema_momentum   = (float)cfg.ema_momentum;

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
        ng.max_bf_iters = diam;   // per-net exact BF diameter
        if (diam > max_diam) max_diam = diam;
    }
    ctx->total_dist_nodes = dist_offset;
    ctx->max_bf_iters     = max_diam;

    // Partition nets into small (n_local <= SMALL_THRESH) and large subsets.
    std::vector<NetGPU> h_nets_small, h_nets_large;
    for (const NetGPU& ng : ctx->h_nets) {
        if (ng.n_local <= SMALL_THRESH) {
            h_nets_small.push_back(ng);
            if (ng.n_local > ctx->max_small_n_local)
                ctx->max_small_n_local = ng.n_local;
        } else {
            h_nets_large.push_back(ng);
        }
    }
    ctx->n_small = (int)h_nets_small.size();
    ctx->n_large = (int)h_nets_large.size();

    long long gpu_mb = ((long long)ctx->total_dist_nodes * 4
                       + (long long)ctx->LXY * 8 * 4) / 1024 / 1024;
    std::cout << "[lag_gpu] " << ctx->n_nets << " nets"
              << "  small=" << ctx->n_small << " large=" << ctx->n_large
              << "  max_small_n_local=" << ctx->max_small_n_local
              << "  total_dist_nodes=" << ctx->total_dist_nodes
              << "  max_bf_iters="     << max_diam
              << "  est_gpu_MB="       << gpu_mb << "\n";

    CUDA_CHECK(cudaMalloc(&ctx->d_nets,      ctx->n_nets * sizeof(NetGPU)));
    if (ctx->n_small > 0) {
        // Allow bf_iters_small_kernel to use up to SMALL_SHMEM_BYTES of dynamic
        // shared memory (48 KB on sm_86; default limit is 48 KB but must be
        // declared explicitly when exceeding 32 KB to avoid silent truncation).
        CUDA_CHECK(cudaFuncSetAttribute(
            bf_iters_small_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMALL_SHMEM_BYTES));
        CUDA_CHECK(cudaMalloc(&ctx->d_nets_small, ctx->n_small * sizeof(NetGPU)));
        CUDA_CHECK(cudaMemcpy(ctx->d_nets_small, h_nets_small.data(),
                              ctx->n_small * sizeof(NetGPU), cudaMemcpyHostToDevice));
    }
    if (ctx->n_large > 0) {
        CUDA_CHECK(cudaMalloc(&ctx->d_nets_large, ctx->n_large * sizeof(NetGPU)));
        CUDA_CHECK(cudaMemcpy(ctx->d_nets_large, h_nets_large.data(),
                              ctx->n_large * sizeof(NetGPU), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&ctx->d_dist,      ctx->total_dist_nodes * sizeof(int)));
    // Initialize d_dist to INF_DIST so that when max_iters=0 (no BF runs),
    // extract_routes sees all sinks as unreachable and reports disconnected —
    // matching CPU behaviour for --lag-iters 0.
    CUDA_CHECK(cudaMemset(ctx->d_dist, 0x7f, ctx->total_dist_nodes * sizeof(int)));
    // 0x7f7f7f7f = 2139062143, well above any realistic path cost, treated as INF.
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_h,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_v,     ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_lam_via,   ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_h,         ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_v,         ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_via,       ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_prev_h,    ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_prev_v,    ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_use_prev_via,  ctx->LXY * sizeof(float)));
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

    CUDA_CHECK(cudaMemset(ctx->d_lam_h,       0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_lam_v,       0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_lam_via,     0, ctx->LXY * sizeof(float)));
    // prev_use starts at zero (no history before iteration 1)
    CUDA_CHECK(cudaMemset(ctx->d_use_prev_h,   0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_use_prev_v,   0, ctx->LXY * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx->d_use_prev_via, 0, ctx->LXY * sizeof(float)));

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

        // Step 2: BF per net — small nets use shared-memory kernel, large use global.
        // Each net's block reads max_bf_iters from NetGPU (exact per-net diameter),
        // replacing the previous single global max_bf_iters broadcast.
        if (ctx->n_small > 0)
            bf_iters_small_kernel<<<ctx->n_small, 256, ctx->max_small_n_local * sizeof(int)>>>(
                ctx->d_nets_small, ctx->d_dist,
                ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
                ctx->d_use_prev_h, ctx->d_use_prev_v, ctx->d_use_prev_via,
                ctx->d_cap_h, ctx->d_cap_v, (float)ctx->via_cap,
                ctx->d_layer_dir, ctx->d_layer_sc,
                ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
                ctx->add_via, ctx->beta);
        if (ctx->n_large > 0)
            bf_iters_kernel<<<ctx->n_large, 256>>>(
                ctx->d_nets_large, ctx->d_dist,
                ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
                ctx->d_use_prev_h, ctx->d_use_prev_v, ctx->d_use_prev_via,
                ctx->d_cap_h, ctx->d_cap_v, (float)ctx->via_cap,
                ctx->d_layer_dir, ctx->d_layer_sc,
                ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
                ctx->add_via, ctx->beta);
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
            ctx->d_use_prev_h, ctx->d_use_prev_v, ctx->d_use_prev_via,
            ctx->d_cap_h, ctx->d_cap_v, (float)ctx->via_cap,
            ctx->d_layer_dir, ctx->d_layer_sc,
            ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
            ctx->d_use_h, ctx->d_use_v, ctx->d_use_via,
            n_nets, ctx->add_via, ctx->beta);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        ctx->t_trace += std::chrono::duration<double>(t2 - t1).count();

        // Step 4: Update lambda — skipped on the last iteration so that the
        // final d_dist and d_lam_* remain consistent for path extraction.
        if (iter < max_iters) {
            // EMA-smooth prev_use: damps 2-period oscillation caused by dispersion
            // alternating between two congestion states.  With ema_momentum=0 this
            // degenerates to a plain copy (prev_use = current_use).
            if (ctx->beta != 0.0f) {
                ema_update_kernel<<<grid_upd, 256>>>(ctx->d_use_prev_h,   ctx->d_use_h,   ctx->ema_momentum, LXY);
                ema_update_kernel<<<grid_upd, 256>>>(ctx->d_use_prev_v,   ctx->d_use_v,   ctx->ema_momentum, LXY);
                ema_update_kernel<<<grid_upd, 256>>>(ctx->d_use_prev_via, ctx->d_use_via, ctx->ema_momentum, LXY);
            }
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

            // Diagnostic: max lambda and max finite dist (overflow / quantization check)
            std::vector<float> h_lam_h_d(LXY), h_lam_v_d(LXY);
            CUDA_CHECK(cudaMemcpy(h_lam_h_d.data(), ctx->d_lam_h, LXY*4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lam_v_d.data(), ctx->d_lam_v, LXY*4, cudaMemcpyDeviceToHost));
            float max_lam = 0.0f;
            for (int i = 0; i < LXY; ++i) {
                max_lam = std::max(max_lam, std::max(h_lam_h_d[i], h_lam_v_d[i]));
            }
            std::vector<int> h_dist_d(ctx->total_dist_nodes);
            CUDA_CHECK(cudaMemcpy(h_dist_d.data(), ctx->d_dist,
                                  ctx->total_dist_nodes*4, cudaMemcpyDeviceToHost));
            int max_finite_dist = 0, n_near_inf = 0;
            for (int v : h_dist_d) {
                if (v < INF_DIST) max_finite_dist = std::max(max_finite_dist, v);
                else              ++n_near_inf;
            }

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
                      << "  overload_edges=" << n_ov
                      << "  max_lam=" << max_lam
                      << "  max_finite_dist=" << max_finite_dist
                      << "  INF_nodes=" << n_near_inf << "\n";
        }
    }

    // Final clean BF: recompute dist with the converged lambda but beta=0.
    // Skipped entirely when max_iters=0 (no iterations ran, lambda=0, dist
    // should remain all-INF so extraction reports all nets disconnected —
    // matching CPU behaviour for --lag-iters 0).
    if (max_iters > 0 && ctx->beta != 0.0f) {
        reset_int_kernel<<<(ctx->total_dist_nodes+255)/256, 256>>>(
            ctx->d_dist, ctx->total_dist_nodes, INF_DIST);
        init_src_kernel<<<n_nets, 1>>>(ctx->d_nets, ctx->d_dist);
        if (ctx->n_small > 0)
            bf_iters_small_kernel<<<ctx->n_small, 256, ctx->max_small_n_local * sizeof(int)>>>(
                ctx->d_nets_small, ctx->d_dist,
                ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
                ctx->d_use_prev_h, ctx->d_use_prev_v, ctx->d_use_prev_via,
                ctx->d_cap_h, ctx->d_cap_v, (float)ctx->via_cap,
                ctx->d_layer_dir, ctx->d_layer_sc,
                ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
                ctx->add_via, /*beta=*/0.0f);
        if (ctx->n_large > 0)
            bf_iters_kernel<<<ctx->n_large, 256>>>(
                ctx->d_nets_large, ctx->d_dist,
                ctx->d_lam_h, ctx->d_lam_v, ctx->d_lam_via,
                ctx->d_use_prev_h, ctx->d_use_prev_v, ctx->d_use_prev_via,
                ctx->d_cap_h, ctx->d_cap_v, (float)ctx->via_cap,
                ctx->d_layer_dir, ctx->d_layer_sc,
                ctx->unit_wire_cost, ctx->unit_via_cost, gX, gY,
                ctx->add_via, /*beta=*/0.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
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
        r.name      = twonets[ni].name;
        r.orig_name = twonets[ni].orig_name;

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

            // Best-match predecessor: primary = min lambda, secondary = min ei.
            // Mirrors the tie-break in trace_and_use_kernel for full consistency.
            struct Cand {
                int v, ei, type; // type: 0=H 1=V 2=via
                int sx1,sy1,sz1, sx2,sy2,sz2;
                float lam;
            };
            Cand best{-1,-1,-1,0,0,0,0,0,0,FLT_MAX};

            auto consider = [&](int v, int ei, float lam, int type,
                                int sx1,int sy1,int sz1,int sx2,int sy2,int sz2,
                                float cost_base, const std::vector<float>& lam_arr) {
                if (v < 0 || v >= ng.n_local) return;
                // Must mirror bf_iters_kernel exactly: same COST_SCALE and edge_eps.
                int w = (int)((cost_base + fmaxf(0.0f, lam_arr[ei])) * COST_SCALE)
                        + edge_eps(ni, ei);
                if (dist[v] == INF_DIST || dist[v] + w != d_cur) return;
                if (lam < best.lam || (lam == best.lam && ei < best.ei))
                    best = {v, ei, type, sx1,sy1,sz1,sx2,sy2,sz2, lam};
            };

            if (dir == 0 && ll > 0) {
                if (lx+1 < bx) { int v=(lx+1)*by*L+ly*L+ll, ei=ll*gX*gY+gx*gY+gy;
                    consider(v,ei,h_lam_h[ei],0, gx,gy,ll, gx+1,gy,ll, base_wire,h_lam_h); }
                if (lx > 0)    { int v=(lx-1)*by*L+ly*L+ll, ei=ll*gX*gY+(gx-1)*gY+gy;
                    consider(v,ei,h_lam_h[ei],0, gx-1,gy,ll, gx,gy,ll, base_wire,h_lam_h); }
            }
            if (dir == 1 && ll > 0) {
                if (ly+1 < by) { int v=lx*by*L+(ly+1)*L+ll, ei=ll*gX*gY+gx*gY+gy;
                    consider(v,ei,h_lam_v[ei],1, gx,gy,ll, gx,gy+1,ll, base_wire,h_lam_v); }
                if (ly > 0)    { int v=lx*by*L+(ly-1)*L+ll, ei=ll*gX*gY+gx*gY+(gy-1);
                    consider(v,ei,h_lam_v[ei],1, gx,gy-1,ll, gx,gy,ll, base_wire,h_lam_v); }
            }
            if (ctx->add_via) {
                if (ll+1 < L) { int v=lx*by*L+ly*L+(ll+1), ei=ll*gX*gY+gx*gY+gy;
                    consider(v,ei,h_lam_via[ei],2, gx,gy,ll, gx,gy,ll+1, base_via,h_lam_via); }
                if (ll > 0)   { int v=lx*by*L+ly*L+(ll-1), ei=(ll-1)*gX*gY+gx*gY+gy;
                    consider(v,ei,h_lam_via[ei],2, gx,gy,ll-1, gx,gy,ll, base_via,h_lam_via); }
            }

            if (best.v < 0) break;
            RoutingSegment seg;
            seg.x1=best.sx1; seg.y1=best.sy1; seg.z1=best.sz1;
            seg.x2=best.sx2; seg.y2=best.sy2; seg.z2=best.sz2;
            r.segments.push_back(seg);
            cur = best.v;
        }

        routes.push_back(std::move(r));
    }
    return routes;
}

static void lag_gpu_free(LagGPUCtx* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_nets);
    if (ctx->d_nets_small) cudaFree(ctx->d_nets_small);
    if (ctx->d_nets_large) cudaFree(ctx->d_nets_large);
    cudaFree(ctx->d_dist);
    cudaFree(ctx->d_lam_h);   cudaFree(ctx->d_lam_v);   cudaFree(ctx->d_lam_via);
    cudaFree(ctx->d_use_h);   cudaFree(ctx->d_use_v);   cudaFree(ctx->d_use_via);
    cudaFree(ctx->d_use_prev_h); cudaFree(ctx->d_use_prev_v); cudaFree(ctx->d_use_prev_via);
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
