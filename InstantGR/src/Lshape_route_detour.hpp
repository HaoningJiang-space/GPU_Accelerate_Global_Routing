#include "graph.hpp"
#include <omp.h>
#include <atomic>
#include <unistd.h>
#include <queue>
#define INF_LAYER 20
#define MAX_LAYER 10
#define MIN_ROUTE_LAYER 1
#define MAX_DEPTH 5000
__managed__ double *cost_edges;
__managed__ int *best_change;
__managed__ int edge_cnt;

namespace Lshape_route_detour {

//declaration
void Lshape_route_detour(vector<int> &nets2route);
__managed__ int *macroBorder;
int *macroBorder_cpu;
int cntCongested = 0;
int totalEdgeNum = 0;

__managed__ int *node_cnt_sum, *nodes, *par_nodes, *from, *layer_range;
int node_cnt_estimate;
int parent_cnt_estimate;
__managed__ int *child_num; 
__managed__ int *child_num_sum; 
__managed__ int *in_degree;
__managed__ int *currentChildIDX;

__managed__ int *par_num;
__managed__ int *par_num_sum;
__managed__ int *locks;
__managed__ double *childCosts;
__managed__ int *childCosts_road;

__managed__ int *best_path;

__managed__ int *layer_output;
__managed__ double *costs;
__managed__ int *fixed_layers;
__managed__ int *node_net_idx;
__managed__ int *node_net_idx2;

__managed__ int *lock_gpu;

__managed__ int *node_depth;
__managed__ int *net_depth;

__managed__ int *batch_depth;
__managed__ int *depth_node;
__managed__ int *depth_node_cnt;

bool *congestionView_cpu;
float *congestionView_xsum_cpu;
float *congestionView_ysum_cpu;
int *node_cnt_sum_cpu, *node_depth_cpu, *net_depth_cpu, *batch_depth_cnt_cpu, *depth_node_cnt_cpu, *depth_node_cpu,
    *nodes_cpu, *node_net_idx_cpu, *node_net_idx2_cpu, *child_num_cpu, *child_num_sum_cpu, *par_num_cpu, *par_num_sum_cpu,
    *par_nodes_cpu, *currentChildIDX_cpu, *depthID2nodeSequence;

__device__ void atomicMinDouble(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__global__ void init_min_child_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts[index] = INF;
}

__global__ void init_road(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts_road[index] = 200000000;
}

__global__ void init_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>limit)
    {
        return;
    }
    costs[index] = INF;
}

__global__ void Lshape_route_node_cuda(int shift, int end_shift) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_idx = depth_node[node_sequence];
    int parent_num_cur = par_num_sum[node_idx+1]-par_num_sum[node_idx];
    int fixed_layer_low = 1 + nodes[node_idx] / X / Y;
    int x = nodes[node_idx] / Y % X, y = nodes[node_idx] % Y;
    int fixed_layer_high = fixed_layer_low==10?0:fixed_layer_low;
    int cur_child_num = child_num_sum[node_idx+1]-child_num_sum[node_idx];

    int *cur_best_path = best_path + child_num_sum[node_idx] * MAX_LAYER;
    double *cur_childCosts = childCosts + child_num_sum[node_idx] * MAX_LAYER;
    int *cur_childCosts_road = childCosts_road + child_num_sum[node_idx] * MAX_LAYER;
    double minChildCosts[6];
    int bestPaths[6];
    for (int lowLayerIndex = MIN_ROUTE_LAYER; lowLayerIndex <= fixed_layer_low; lowLayerIndex++) {
        for(int cid=0; cid<cur_child_num; cid++)
        {
            minChildCosts[cid] = INF;
        }
        double via_cost = 0;
        for (int layerIndex = lowLayerIndex; layerIndex < (L+1); layerIndex++) {
            if(layerIndex>lowLayerIndex)
            {   
                // min value of lowLayerIndex is 1 
                via_cost += vcost[IDX(layerIndex - 2, x, y)];
            }
            // int min_layer = 10;
            for (int childIndex = 0; childIndex < cur_child_num; childIndex++) {
                double cur_child_cost = cur_childCosts[childIndex * MAX_LAYER + layerIndex];
                if (cur_child_cost < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = cur_child_cost;
                    bestPaths[childIndex] = cur_childCosts_road[childIndex * MAX_LAYER + layerIndex] * MAX_LAYER + layerIndex;
                }
            }
            if (layerIndex >= fixed_layer_high) {
                double cost = via_cost;
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cost += minChildCosts[childIndex];
                }
                if (cost<INF && cost < costs[node_idx*MAX_LAYER+layerIndex]) {
                    costs[node_idx*MAX_LAYER+layerIndex] = cost;
                    for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                    {
                        cur_best_path[childIndex * MAX_LAYER + layerIndex] = bestPaths[childIndex];
                    }
                }
            }
        }
        for (int layerIndex = (L+1) - 2; layerIndex >= lowLayerIndex; layerIndex--) {//
            if (costs[node_idx*MAX_LAYER+layerIndex + 1] < costs[node_idx*MAX_LAYER+layerIndex]) {
                costs[node_idx*MAX_LAYER+layerIndex] = costs[node_idx*MAX_LAYER+layerIndex + 1];
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cur_best_path[childIndex * MAX_LAYER + layerIndex] = cur_best_path[childIndex * MAX_LAYER + layerIndex+1];
                }
            }
        }
    }
    int node_x = nodes[node_idx] / Y % X, node_y = nodes[node_idx] % Y; 

    for(int par_id = 0; par_id < parent_num_cur; par_id++)
    {
        int parent_IDX = par_nodes[par_num_sum[node_idx] + par_id];
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_idx] + par_id]%10;
        double *parent_childCosts = childCosts + child_num_sum[parent_IDX] * MAX_LAYER;
        int *parent_childCosts_road = childCosts_road + child_num_sum[parent_IDX] * MAX_LAYER;
        int px = nodes[parent_IDX] / Y % X, py = nodes[parent_IDX] % Y; 
        assert(px==node_x||py==node_y);
        for(int layer = MIN_ROUTE_LAYER; layer<MAX_LAYER; layer++)
        {
            if((layer & 1 ^ DIR) == 1 && node_y != py) continue;
            if((layer & 1 ^ DIR) == 0 && node_x != px) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + layer;
            double cost = costs[node_idx * MAX_LAYER + layer] + graph::wire_segment_cost(layer-1, min(node_x, px), max(node_x, px), min(node_y, py), max(node_y, py));
            int shift_modify = child_num_sum[parent_IDX] * MAX_LAYER + index_;
            atomicMinDouble(&parent_childCosts[index_], cost);
            if(parent_childCosts[index_]==cost)
            {
                parent_childCosts_road[index_] = node_idx;
            }
        }
    }
}

__global__ void get_routing_tree_cuda(int shift, int end_shift, int depth, int stamp) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_id = depth_node[node_sequence];
    int net_id = node_net_idx[node_id];
    int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int *cur_best_path = best_path + child_num_sum[node_id] * MAX_LAYER;

    int l = nodes[node_id] / Y / X;
    int cur_x = nodes[node_id] / Y % X, cur_y = nodes[node_id] % Y;
    if(par_num_sum[node_id+1]-par_num_sum[node_id]==0)
    {
        int min_layer = 0;
        double min_cost = costs[node_id * MAX_LAYER];
        for(int layer = 1; layer < MAX_LAYER; layer++)
        {
            if(costs[node_id * MAX_LAYER + layer] < min_cost)
            {
                min_cost = costs[ node_id * MAX_LAYER + layer];
                min_layer = layer;
            }
        }
        layer_output[node_id] = min_layer;
        net_routes[0] = 1;
    } else{
        int par_layer = -1;
        int par_idx = -1;
        int par_sequence = -1;
        for(int par_id = 0; par_id< par_num_sum[node_id+1]-par_num_sum[node_id]; par_id++)
        {
            int par_node = par_nodes[par_num_sum[node_id]+par_id];
            if(layer_output[par_node]>=0)
            {
                par_idx = par_node;
                par_sequence = par_id;
                par_layer = layer_output[par_node];
                int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
                int *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
                int path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
                int child_idx = path / MAX_LAYER;
                if(child_idx == node_id)
                {
                    layer_output[node_id] = path % MAX_LAYER;
                    int px = nodes[par_idx] / Y % X, py = nodes[par_idx] % Y;
                    assert(px==cur_x||py==cur_y); 
                    if(px==cur_x && cur_y!=py)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, px, px, min(py,cur_y), max(py,cur_y), stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, px, min(py,cur_y));
                        net_routes[idd1+1] = IDX(layer_output[node_id] - 1, px, max(py,cur_y));
                    }
                    else if(py==cur_y && cur_x != px)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, min(px,cur_x), max(px,cur_x), py, py, stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, min(px,cur_x), py);
                        net_routes[idd1+1] = IDX(layer_output[node_id] - 1, max(px,cur_x), py);
                    }
                    break;
                }else{
                    layer_output[node_id] = -1;
                }
            }
        }
        if(par_layer==-1)
        {
            layer_output[node_id] = -1;
            return;
        }
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
        int *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
        int path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
        int child_idx = path / MAX_LAYER;
        if( child_idx != node_id)
        {
            layer_output[node_id] = -1;
            return;
        }
    }
    int num_child = child_num_sum[node_id+1] - child_num_sum[node_id];
    int minl = l+1;
    int maxl = (l+1)==MAX_LAYER?1:minl;
    minl = min(minl,layer_output[node_id]);
    maxl = max(maxl,layer_output[node_id]);
    assert(num_child>=0);
    if(num_child>0)
    {
        for(int child_id=0; child_id<num_child; child_id++)
        {
            int layer_of_child = cur_best_path[child_id * MAX_LAYER + layer_output[node_id]] % MAX_LAYER;
            assert(layer_of_child!=0);
            minl = min(layer_of_child, minl);
            maxl = max(layer_of_child, maxl);
        }
    }
    if(minl<maxl)
    {
        int idd1 = atomicAdd(net_routes,2);
        net_routes[idd1] = IDX(minl - 1, cur_x, cur_y);
        net_routes[idd1+1] = IDX(maxl - 1, cur_x, cur_y);
    }
}

void process_net(int thread_idx, vector<int> &nets2route, int thread_num, std::atomic<int>& currentNetId) {
    while (true) {
        int netId = currentNetId.fetch_add(1);
        if (netId >= nets2route.size()) {
            break;
        }
        nets[nets2route[netId]].generate_detours(move(congestionView_cpu), move(congestionView_xsum_cpu), move(congestionView_ysum_cpu), false);
    }
}

void multithreaded_processing(vector<int> &nets2route) {
    std::vector<std::thread> threads;
    int max_threads = 8;
    threads.reserve(max_threads);
    std::atomic<int> currentNetId(0);
    for (int i = 0; i < max_threads; ++i) {
        threads.emplace_back(process_net, i, std::ref(nets2route), max_threads, std::ref(currentNetId));
    }
    for (auto& t : threads) {
        t.join();
    }
}

void Lshape_route_detour_wrap(vector<int> &nets2route)
{
    double DAG_start_time = elapsed_time();
    if (nets2route.size() == 0)
    {
        return;
    }
    sort(nets2route.begin(), nets2route.end(), [](int l, int r)
         { return nets[l].hpwl > nets[r].hpwl; });
    congestionView_cpu = new bool[X * Y * sizeof(bool)];
    congestionView_xsum_cpu = new float[X * Y * sizeof(float)];
    congestionView_ysum_cpu = new float[X * Y * sizeof(float)];
    cudaMemcpy(congestionView_cpu, congestion, X * Y * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_xsum_cpu, congestion_xsum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_ysum_cpu, congestion_ysum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nets2route.size(); i++)
    {
        if (nets[nets2route[i]].rsmt.size() < 1)
        {
            nets[nets2route[i]].construct_rsmt();
        }
    }
    multithreaded_processing(nets2route);
    if(LOG) printf("[%5.1f] Generating batches Starts\n", elapsed_time());
    auto batches = generate_batches_rsmt(nets2route, 300000);
    if(LOG) printf("[%5.1f] Generating batches Ends\n", elapsed_time());
    int net_cnt_estimate = 0;
    int node_num_max = 0;
    int par_num_max = 0;
    for (int ii = 0; ii < batches.size(); ii++)
    {
        int tmp = 0;
        int tmp2 = 0;
        if (batches[ii].size() > net_cnt_estimate)
        {
            net_cnt_estimate = batches[ii].size();
        }
        for (int j = 0; j < batches[ii].size(); j++)
        {
            auto &graph_x = nets[batches[ii][j]].rsmt;
            tmp += nets[batches[ii][j]].node_index_cnt;
            tmp2 += nets[batches[ii][j]].par_num_sum_cpu[nets[batches[ii][j]].node_index_cnt];
        }
        if (tmp > node_num_max)
        {
            node_num_max = tmp;
        }
        if (tmp2 > par_num_max)
        {
            par_num_max = tmp2;
        }
    }

    net_cnt_estimate += 5;
    node_cnt_estimate = node_num_max + 10;
    parent_cnt_estimate = par_num_max + 10;
    if(LOG) printf("[%5.1f] Lshape_route Starts\n", elapsed_time());
    cudaMalloc(&node_cnt_sum, net_cnt_estimate * sizeof(int));
    cudaMalloc(&nodes, node_cnt_estimate * sizeof(int));
    cudaMalloc(&net_depth, net_cnt_estimate * sizeof(int));
    cudaMalloc(&batch_depth, (batches.size() + 1) * sizeof(int));
    cudaMalloc(&child_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx2, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_depth, node_cnt_estimate * sizeof(int));
    cudaMalloc(&depth_node, node_cnt_estimate * sizeof(int));
    cudaMalloc(&layer_range, node_cnt_estimate * sizeof(int));
    cudaMalloc(&costs, node_cnt_estimate * MAX_LAYER * sizeof(double));
    cudaMalloc(&locks, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMemset(locks, 0, sizeof(int) * parent_cnt_estimate * MAX_LAYER);
    cudaMalloc(&layer_output, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_nodes, parent_cnt_estimate * sizeof(int));
    cudaMalloc(&cost_edges, parent_cnt_estimate * 81 * sizeof(double));
    cudaMalloc(&best_change, parent_cnt_estimate * 81 * sizeof(int));
    cudaMalloc(&best_path, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMalloc(&childCosts, parent_cnt_estimate * MAX_LAYER * sizeof(double));
    cudaMalloc(&childCosts_road, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMalloc(&currentChildIDX, parent_cnt_estimate * sizeof(int));
    ///////////////////////////////////  cpu arrays init starts  ////////////////////////////////////////
    node_cnt_sum_cpu = new int[net_cnt_estimate]();
    int reserve_node_num = node_cnt_estimate;
    int biggest_depth = MAX_DEPTH;
    node_depth_cpu = new int[reserve_node_num]();
    net_depth_cpu = new int[net_cnt_estimate]();
    batch_depth_cnt_cpu = new int[batches.size() + 1]();
    depth_node_cnt_cpu = new int[biggest_depth * (batches.size() + 1)]();
    depth_node_cpu = new int[reserve_node_num]();
    nodes_cpu = new int[reserve_node_num]();
    node_net_idx_cpu = new int[reserve_node_num]();
    node_net_idx2_cpu = new int[reserve_node_num]();
    child_num_cpu = new int[reserve_node_num]();
    child_num_sum_cpu = new int[reserve_node_num]();
    par_num_cpu = new int[reserve_node_num]();
    par_num_sum_cpu = new int[reserve_node_num]();
    par_nodes_cpu = new int[parent_cnt_estimate]();
    currentChildIDX_cpu = new int[parent_cnt_estimate]();
    depthID2nodeSequence = new int[batches.size() * MAX_DEPTH];
    ///////////////////////////////////  cpu arrays init ends  ////////////////////////////////////////

    for (int ii = batches.size() - 1; ii >= 0; ii--)
    {
        Lshape_route_detour(batches[ii]);
    }

    cudaFree(node_cnt_sum);
    cudaFree(par_nodes);
    cudaFree(nodes);
    cudaFree(from);
    cudaFree(layer_range);
    cudaFree(in_degree);
    cudaFree(currentChildIDX);
    cudaFree(par_num_sum);
    cudaFree(locks);
    cudaFree(layer_output);
    cudaFree(fixed_layers);
    cudaFree(node_net_idx);
    cudaFree(node_net_idx2);
    cudaFree(lock_gpu);
    cudaFree(node_depth);
    cudaFree(net_depth);
    cudaFree(batch_depth);
    cudaFree(depth_node);
    cudaFree(depth_node_cnt);
    cudaFree(childCosts);
    cudaFree(childCosts_road);
    cudaFree(best_path);
    delete[] congestionView_cpu;
    delete[] node_cnt_sum_cpu;
    delete[] node_depth_cpu;
    delete[] net_depth_cpu;
    delete[] batch_depth_cnt_cpu;
    delete[] depth_node_cnt_cpu;
    delete[] depth_node_cpu;
    delete[] nodes_cpu;
    delete[] node_net_idx_cpu;
    delete[] node_net_idx2_cpu;
    delete[] child_num_cpu;
    delete[] child_num_sum_cpu;
    delete[] par_num_cpu;
    delete[] par_num_sum_cpu;
    delete[] par_nodes_cpu;
    delete[] currentChildIDX_cpu;
    if(LOG) printf("[%5.1f] Lshape_route Ends\n", elapsed_time());
    DAG_time = elapsed_time() - DAG_start_time;
}

void Lshape_route_detour(vector<int> &nets2route) {
    vector<vector<int>> batches;
    batches.push_back(nets2route);
    vector<int> batch_cnt_sum(batches.size() + 1, 0);
    for(int i = 0; i < batches.size(); i++) {
        batch_cnt_sum[i + 1] = batch_cnt_sum[i] + batches[i].size();
        for(int j = 0; j < batches[i].size(); j++)
        {
            int net_id = batch_cnt_sum[i] + j;
            nets2route[net_id] = batches[i][j];
        }
    }
    int net_cnt = nets2route.size();
    int node_cnt = 0;
    int par_cnt = 0;
    for(auto net_id : nets2route)
    {
        node_cnt += nets[net_id].node_index_cnt;
        par_cnt += nets[net_id].par_num_sum_cpu[nets[net_id].node_index_cnt];
    }
    ////////////////////////////////////////////// cpu array memset starts //////////////////////////////////////////////////
    memset(node_cnt_sum_cpu, 0, (net_cnt + 1) * sizeof(int));
    int batch_reserve_node = node_cnt + 10;
    int reserve_node_num = min(node_cnt_estimate, batch_reserve_node);//to be optimized
    int reserve_par_num = min(parent_cnt_estimate, par_cnt+10);
    memset(net_depth_cpu, 0, net_cnt * sizeof(int));
    memset(batch_depth_cnt_cpu, 0, (batches.size()+1) * sizeof(int));
    memset(depth_node_cnt_cpu, 0, MAX_DEPTH*(batches.size()+1) * sizeof(int));
    memset(child_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(depthID2nodeSequence, 0, batches.size()*MAX_DEPTH * sizeof(int));
    ////////////////////////////////////////////// cpu array memset ends //////////////////////////////////////////////////
    for(int b_id=0; b_id<batches.size(); b_id++)
    for(int j=0; j< batches[b_id].size(); j++){
        int net_idx = batches[b_id][j];
        auto &graph_x = nets[net_idx].rsmt;
        int select_root = nets[net_idx].select_root;
        int id = batch_cnt_sum[b_id] + j;
        int net_num_nodes = nets[net_idx].node_index_cnt;
        node_cnt_sum_cpu[id+1] = net_num_nodes;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        { 
            int node_id = node_cnt_sum_cpu[id] + n_i;
            nodes_cpu[node_id] = nets[net_idx].nodes_cpu[n_i];
            child_num_cpu[node_id] = nets[net_idx].child_num_cpu[n_i];
            child_num_sum_cpu[node_id+1] = child_num_sum_cpu[node_id] + child_num_cpu[node_id];
            node_depth_cpu[node_id] = nets[net_idx].node_depth_cpu[n_i];
            int depth = node_depth_cpu[node_id];
            batch_depth_cnt_cpu[b_id+1] = max(batch_depth_cnt_cpu[b_id+1], node_depth_cpu[node_id]+1);
            net_depth_cpu[id] = max(net_depth_cpu[id], depth);
            par_num_cpu[node_id] = nets[net_idx].par_num_cpu[n_i];
            par_num_sum_cpu[node_id+1] = par_num_sum_cpu[node_id] + par_num_cpu[node_id];
            node_net_idx_cpu[node_id] = net_idx;
            node_net_idx2_cpu[node_id] = id;
        }
        int par_num_total = nets[net_idx].par_num_sum_cpu[net_num_nodes];
        int node_start = node_cnt_sum_cpu[id];
        int pid_total = 0;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        {
            int node_id = node_cnt_sum_cpu[id] + n_i;
            for(int n_pid = 0; n_pid <  par_num_cpu[node_id]; n_pid++)
            {
                currentChildIDX_cpu[par_num_sum_cpu[node_start] + pid_total] 
                    = nets[net_idx].currentChildIDX_cpu[pid_total] + node_id * 10;
                par_nodes_cpu[par_num_sum_cpu[node_start] + pid_total] = nets[net_idx].par_nodes_cpu[pid_total] + node_cnt_sum_cpu[id];
                pid_total++;
            }
        }
        for(int i=node_cnt_sum_cpu[id];i<node_cnt_sum_cpu[id]+node_cnt_sum_cpu[id + 1];i++)
        {
            int depth_node_i = node_depth_cpu[i];
            depth_node_cnt_cpu[b_id*MAX_DEPTH+depth_node_i+1]++;
        }     
        node_cnt_sum_cpu[id + 1] += node_cnt_sum_cpu[id];
    }
    for(int bid=1; bid <= batches.size(); bid++)
    {
        batch_depth_cnt_cpu[bid]+=batch_depth_cnt_cpu[bid-1];
    }
    for(int bid=0; bid < batches.size(); bid++)
    {
        assert(batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]<MAX_DEPTH);
        for(int d = 0; d< batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depth_node_cnt_cpu[bid*MAX_DEPTH+d+1]+=depth_node_cnt_cpu[bid*MAX_DEPTH+d];
        }
        for(int d = 0; d<= batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depthID2nodeSequence[batch_depth_cnt_cpu[bid]+d] = node_cnt_sum_cpu[batch_cnt_sum[bid]] + depth_node_cnt_cpu[bid*MAX_DEPTH+d];
            depth_node_cnt_cpu[bid*MAX_DEPTH+d] += node_cnt_sum_cpu[batch_cnt_sum[bid]];
        }
        
        for(int node_id = node_cnt_sum_cpu[batch_cnt_sum[bid]]; node_id < node_cnt_sum_cpu[batch_cnt_sum[bid+1]]; node_id++)
        {
            int depth = node_depth_cpu[node_id];
            depth_node_cpu[depth_node_cnt_cpu[bid*MAX_DEPTH+depth]++] = node_id;
        }
    }
    int node_total = node_cnt_sum_cpu[net_cnt];
    for(int node_id = 1; node_id <= node_total; node_id++)
    {
        child_num_sum_cpu[node_id] = child_num_cpu[node_id-1];
        child_num_sum_cpu[node_id] += child_num_sum_cpu[node_id - 1];
    }
    cudaMemcpy(net_ids, nets2route.data(), net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(layer_output, 0xFF, reserve_node_num * sizeof(int));
    cudaMemcpy(node_cnt_sum, node_cnt_sum_cpu, (net_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nodes, nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(net_depth, net_depth_cpu, net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_depth, batch_depth_cnt_cpu, (batches.size()+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(child_num_sum, child_num_sum_cpu, (node_cnt_sum_cpu[net_cnt]+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_num_sum, par_num_sum_cpu, (node_total+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_net_idx, node_net_idx_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_depth, node_depth_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_node, depth_node_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_nodes, par_nodes_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(currentChildIDX, currentChildIDX_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    init_costs<<<node_total, 10>>>(node_total*MAX_LAYER);
    init_min_child_costs<<<(child_num_sum_cpu[node_total]+1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total]+1) * MAX_LAYER);
    init_road<<<(child_num_sum_cpu[node_total]+1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total]+1) * MAX_LAYER);
    
    for(int i = 0; i < batches.size(); i++) {
        int total_node_num = batch_cnt_sum[i+1] - batch_cnt_sum[i];
        int net_offset = batch_cnt_sum[i];
        int next_net_offset = batch_cnt_sum[i+1];

        graph::commit_wire_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, ++global_timestamp, -1);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp, -1);     
        global_timestamp++;
        graph::update_cost();
        graph::compute_presum<<<all_track_cnt, THREAD_NUM, sizeof(double) * XY>>> ();
        cudaDeviceSynchronize();
        int cur_batch_depth = batch_depth_cnt_cpu[i+1] - batch_depth_cnt_cpu[i];
        for(int d = cur_batch_depth - 1; d >= 0; d--)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d+1];
            Lshape_route_node_cuda<<<BLOCK_NUM(end_shift-shift+1), 512>>> (shift, end_shift);
        }
        for(int d = 0; d < cur_batch_depth; d++)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d+1];
            get_routing_tree_cuda<<<BLOCK_NUM(end_shift-shift+1), 512>>> (shift, end_shift, d, global_timestamp);
            cudaDeviceSynchronize();
        }
        graph::batch_wire_update(global_timestamp);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp);
    }
}

}