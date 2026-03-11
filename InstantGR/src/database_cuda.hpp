#pragma once
#include "database.hpp"
#include <queue>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>

namespace cudb {

__managed__ int *rip_up_list;
__managed__ bool *congestion;
__managed__ float *congestion_xsum;
__managed__ float *congestion_ysum;
std::mutex mtx;

__managed__ unsigned long long ROUTE_PER_PIN = 27;

__managed__ int DIR, L, X, Y, XY, NET_NUM, PIN_NUM, *x_edge_len, *y_edge_len, *pin_acc_num, *pins;//DIR: direction of layer 0 in cuda database
__managed__ double unit_length_wire_cost, unit_via_cost, *unit_length_short_costs, *layer_min_len;
__managed__ float *vcost, *wcost, *capacity, *demand;
__managed__ double total_wirelength = 0, total_overflow_cost, layer_overflow_cost[10];
__managed__ int total_via_count = 0;

__managed__ bool *is_of_net;
__managed__ int *of_edge_sum;
__managed__ int *routes, *timestamp, *pre_demand;
__managed__ int *last, all_track_cnt, *idx2track, *track_net, *track_pos, *track_xy;
__managed__ double *presum;
__managed__ int *net_ids;
int *net_ids_cpu;
int global_timestamp = 0;
vector<vector<int>> net_x_cpu, net_y_cpu;
vector<int> pin_cnt_sum_cpu;

#define IDX(l, x, y) ((l) * X * Y + (x) * Y + (y))
#define THREAD_NUM 512
#define BLOCK_NUM(n) ((n) / THREAD_NUM + 1)
#define BLOCK_CNT(tot, thread_cnt) ((tot) / (thread_cnt) + ((tot) % (thread_cnt) > 0))
#define INF 1e22


void build_cuda_database();

struct net {
    void construct_rsmt();
    void generate_detours(bool *congestionView_cpu, float *congestionView_xsum_cpu, float *congestionView_ysum_cpu,
                                  bool construct_segments = true, bool display = false);
    void calc_hpwl();

    int minx, maxx, miny, maxy, hpwl, original_net_id;
    vector<int> pins;
    vector<vector<int>> rsmt;
    vector<pair<int, int>> rsmt_h_segments, rsmt_v_segments;
    vector<int> par_num_cpu;
    vector<int> par_num_sum_cpu;
    vector<int> currentChildIDX_cpu;
    vector<int> par_nodes_cpu;
    vector<int> child_num_cpu;
    vector<int> node_depth_cpu;
    vector<int> nodes_cpu;
    vector<int> points;
    int node_index_cnt = 0;
    int MAX_LAYER=10;
    int select_root = 0;

    flute::Tree tree;
    unordered_map<int, int> layer;
};
vector<net> nets;


void net::calc_hpwl() {
    minx = X, maxx = 0, miny = Y, maxy = 0;
    for(auto p : pins) {
        int x = p / Y % X, y = p % Y;
        minx = min(minx, x);
        maxx = max(maxx, x);
        miny = min(miny, y);
        maxy = max(maxy, y);
    }
    hpwl = maxx - minx + maxy - miny;
}


vector<vector<int>> generate_batches_rsmt(vector<int> &nets2route, int MAX_BATCH_SIZE = 1000000) {    
    auto _time = elapsed_time();
    vector<vector<int>> batches;
    vector<vector<bool>> batch_vis;

    auto has_conflict = [&] (int net_id, int batch_id) {
        for(auto p : nets[net_id].points) if(batch_vis[batch_id][p]) return true;
        return false;
    };

    auto mark_3x3 = [&] (int pos, int batch_id) {
        int _x = pos / Y, _y = pos % Y;
        for(int x = _x - 1; x <= _x + 1; x++) if(0 <= x && x < X)
            for(int y = _y - 1; y < _y + 1; y++) if(0 <= y && y < Y)
                if(x == _x || y == _y) batch_vis[batch_id][x * Y + y] = 1;
    };


    long long segment_len = 0, segment_cnt = 0, failed = 0;
    for(auto net_id : nets2route) {
        int batch_id = -1;
        for(int i = 0; i < batches.size(); i++) if(batches[i].size() < MAX_BATCH_SIZE)
        {
            if(!has_conflict(net_id, i)) { batch_id = i; break; }
            else failed++;
        }
        if(batch_id == -1) {
            batch_id = batches.size();
            batches.emplace_back(vector<int> ());
            batch_vis.emplace_back(vector<bool> (X * Y, 0));
        }
        batches[batch_id].emplace_back(net_id);
        for(auto seg : nets[net_id].rsmt_h_segments) {
            segment_len += seg.second / Y - seg.first / Y;
            segment_cnt++;
            for(auto p = seg.first; p <= seg.second; p += Y) batch_vis[batch_id][p] = 1;
        }
        for(auto seg : nets[net_id].rsmt_v_segments) {
            for(auto p = seg.first; p <= seg.second; p += 1) batch_vis[batch_id][p] = 1;
        }
        for(auto p : nets[net_id].points) mark_3x3(p, batch_id);
    }
    _time = elapsed_time() - _time;
    if(LOG) cout << setw(40) << "Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << setw(20) << "Time" << endl;
    if(LOG) cout << setw(40) << "Generation" << setw(20) << nets2route.size() << setw(20) << batches.size() << setw(20) << setprecision(2) << _time << endl;
    return move(batches);
}

double TOT_RSMT_LENGTH = 0;
vector<vector<int>> my_flute(unordered_set<int> &pos) {
    const int MAX_DEGREE = 10000;
    vector<int> x;
    vector<int> y;
    int cnt = 0;
    vector<int> nodes, parent;
    vector<tuple<int, int, int>> edges;
    for(auto e : pos) {
        x.push_back(db::dr_x[e / Y]);
        y.push_back(db::dr_y[e % Y]);
        cnt++;
    }
    auto tree = flute::flute(cnt, x.data(), y.data(), 3);
    for(int i = 0; i < cnt * 2 - 2; i++) {
        flute::Branch &branch = tree.branch[i];
        nodes.emplace_back(db::dr2gr_x[branch.x] * Y + db::dr2gr_y[branch.y]);
    }    
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    parent.resize(nodes.size());
    for(int i = 0; i < nodes.size(); i++) parent[i] = i;
    edges.reserve(cnt * 2);
    for(int i = 0; i < cnt * 2 - 2; i++) if(tree.branch[i].n < cnt * 2 - 2) {
        Branch &branch1 = tree.branch[i], &branch2 = tree.branch[branch1.n];
        int u, v;
        u = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch1.x] * Y + db::dr2gr_y[branch1.y]) - nodes.begin();
        v = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch2.x] * Y + db::dr2gr_y[branch2.y]) - nodes.begin();
        if(u == v) continue;
        edges.emplace_back(make_tuple(abs(branch1.x - branch2.x) + abs(branch1.y - branch2.y), u, v));
            
    }
    sort(edges.begin(), edges.end());
    function<int(int)> find_parent = [&] (int x) { return x == parent[x] ? x : parent[x] = find_parent(parent[x]); };
    vector<vector<int>> graph(nodes.size());
    for(auto edge : edges) {
        int u = get<1> (edge), v = get<2> (edge), par_u = find_parent(u), par_v = find_parent(v);
        if(par_u != par_v) {
            graph[u].emplace_back(v);
            graph[v].emplace_back(u);
            TOT_RSMT_LENGTH += get<0> (edge);
            parent[par_u] = par_v;
        }
    }
    int tot_degree = 0;
    for(int i = 0; i < nodes.size(); i++) tot_degree += graph[i].size();
    assert(tot_degree == 2 * (nodes.size() - 1));
    graph.emplace_back(move(nodes));
    return move(graph);
}

void net::construct_rsmt() {
    unordered_map<int, int> layer;
    unordered_set<int> pos, nodes;
    for(int i = 0; i < pins.size(); i++) {
        int pos2D = pins[i] % (X * Y);
        pos.insert(pos2D);
        layer[pos2D] = pins[i] / X / Y;
    }
    assert(pos.size() == pins.size());
    
    rsmt = my_flute(pos);
    rsmt_h_segments.clear();
    rsmt_v_segments.clear();
    rsmt_h_segments.reserve(rsmt.size());
    rsmt_v_segments.reserve(rsmt.size());
    points = rsmt.back();
    for(int i = 0; i < rsmt.back().size(); i++) {
        int xi = rsmt.back()[i] / Y, yi = rsmt.back()[i] % Y;
        for(auto j : rsmt[i]) if(j < i) {
            int xj = rsmt.back()[j] / Y, yj = rsmt.back()[j] % Y;
            int minx = min(xi, xj), maxx = max(xi, xj), miny = min(yi, yj), maxy = max(yi, yj);
            if(xi != xj && yi != yj) {
                rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
                rsmt_h_segments.emplace_back(minx * Y + maxy, maxx * Y + maxy);
                rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
                rsmt_v_segments.emplace_back(maxx * Y + miny, maxx * Y + maxy);
                points.emplace_back(xi * Y + yj);
                points.emplace_back(xj * Y + yi);
            } else if(xi != xj) {
                rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
            } else if(yi != yj) {
                rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
            } else {
                cerr << "error" << endl;
            }
        }
    }
    for(auto &e : rsmt.back()) 
        e += (layer.count(e) ? layer[e] : L) * X * Y;
}


typedef unsigned int BITSET_TYPE;
const int BITSET_LEN = 32;

int select_root_net(vector<vector<int>> rsmt)
{
    queue<pair<int,int>> list;
    int visited[rsmt.size()-1];
    int select = -1;
    for(int i=0; i < rsmt.size()-1; i++)
    {
        visited[i]=0;
        if(rsmt[i].size()==1)
        {
            list.push(make_pair(i, -1));
        }
    }
    while(!list.empty())
    {
        pair<int,int> front_element = list.front();
        list.pop();
        select = front_element.first;
        if(visited[select]) continue;
        visited[select]=1;
        int fa = front_element.second;
        for(int j = 0; j< rsmt[select].size(); j++)
        {
            if(rsmt[select][j]!=fa)
            {
                list.push(make_pair(rsmt[select][j], select));
            }
        }
    }
    return select;
}

void net::generate_detours(bool *congestionView_cpu, float *congestionView_xsum_cpu, float *congestionView_ysum_cpu,
                             bool construct_segments, bool display) {
    
    auto graph_x = rsmt;
    int node_estimate = (graph_x.size()-1)*10;

    par_num_cpu.clear();
    par_num_sum_cpu.clear();
    currentChildIDX_cpu.clear();
    par_nodes_cpu.clear();
    child_num_cpu.clear();
    node_depth_cpu.clear();
    nodes_cpu.clear();
    node_index_cnt = 0;

    par_num_cpu.reserve(node_estimate);
    par_num_sum_cpu.reserve(node_estimate);
    currentChildIDX_cpu.reserve(node_estimate);
    par_nodes_cpu.reserve(node_estimate);
    child_num_cpu.reserve(node_estimate);
    node_depth_cpu.reserve(node_estimate);
    nodes_cpu.reserve(node_estimate);

    par_num_cpu.emplace_back(0);
    par_num_sum_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    par_nodes_cpu.emplace_back(0);
    child_num_cpu.emplace_back(0);
    node_depth_cpu.emplace_back(0);
    nodes_cpu.emplace_back(0);
    
    int depth_max = 0;
    select_root = select_root_net(graph_x);
    vector<int> congestionRegionID[2];
    vector<vector<pair<int, int> >> congestionRanges;
    congestionRanges.resize(2);
    vector<vector<vector<int>>> stems;
    stems.resize(2);
    congestionRegionID[0].resize(graph_x.size());
    congestionRegionID[1].resize(graph_x.size());
    for(int g_i = 0; g_i < graph_x.size(); g_i++)
    {
        congestionRegionID[0][g_i] = -1;
        congestionRegionID[1][g_i] = -1;
    }
    congestionRanges[0].resize(graph_x.size());
    congestionRanges[1].resize(graph_x.size());
    stems[0].resize(graph_x.size());
    stems[1].resize(graph_x.size());
    function<int(int, int)> getRegionID = [&] (int x, int direction) {
        if(x==-1) return -1;
        if(congestionRegionID[direction][x] == -1) return -1;
        if(congestionRegionID[direction][x] != x)
        {
            if(congestionRegionID[direction][x] == x)
            {
                assert(0);
            }
            int ans = getRegionID(congestionRegionID[direction][x], direction);
            congestionRegionID[direction][x] = ans;
            return ans;
        }else{
            return congestionRegionID[direction][x];
        }
    };
    for(int x = 0; x < graph_x.back().size(); x++)
    {
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur / Y % X, cury = position_cur % Y;
        for(int dir=0; dir<2; dir++)
        {
            if(curl<MAX_LAYER-1)
            {
                int stem_pos = dir?curx:cury;
                stems[dir][x].push_back(stem_pos);
            }
            int trunk_pos = dir?cury:curx;
            congestionRanges[dir][x] = make_pair(trunk_pos, trunk_pos);
        }
     
    }
    function<void(int, int)> markCongestion = [&] (int x, int par) {
        int position_cur = graph_x.back()[x];
        int curx = position_cur / Y % X, cury = position_cur % Y;
        for(auto e : graph_x[x]) if(e != par)
        {
            int ex = graph_x.back()[e] / Y % X;
            int ey = graph_x.back()[e] % Y;
            int dir=-1;
            int congestion = -1;
            if(ex == curx)
            {
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 1;
                }
            }
            else if(ey == cury)
            {
                bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0;
                if(is_congestion_x)
                {
                    dir = 0;
                    congestion = 1;
                }
            }
            else{
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0 || 
                                        (congestionView_ysum_cpu[ex*Y+max(ey, cury)]-congestionView_ysum_cpu[ex*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 2;
                }
                if(congestion==-1)
                {

                    bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0 || 
                                        (congestionView_xsum_cpu[max(ex, curx)*Y+ey]-congestionView_xsum_cpu[min(ex, curx)*Y+ey])>0;
                    if(is_congestion_x)
                    {
                        dir = 0;
                        congestion = 2;
                    }
                }
            }
            if(congestion == 1)
            {
                int target_region = -1;
                if(x!=select_root)
                {
                    if(congestionRegionID[dir][x]==-1)
                    {
                        congestionRegionID[dir][x] = x;
                    }
                    int region_x = getRegionID(x, dir);
                    target_region = region_x;
                }else{
                    if(congestionRegionID[dir][e]==-1)
                    {
                        congestionRegionID[dir][e] = e;
                    }
                    int region_e = getRegionID(e, dir);
                    target_region = region_e;
                }
                
                congestionRegionID[dir][e] = target_region;
                if(x!=select_root)
                for(auto pos: stems[dir][e])
                {
                    stems[dir][target_region].push_back(pos);
                }
                congestionRanges[dir][target_region].first = min(congestionRanges[dir][target_region].first, congestionRanges[dir][e].first);
                congestionRanges[dir][target_region].second = max(congestionRanges[dir][target_region].second, congestionRanges[dir][e].second);
            }
            else if(congestion == 2)
            {
                if(x!=select_root&&congestionRegionID[dir][x]==-1)
                {
                    congestionRegionID[dir][x] = x;
                    
                }
                if(congestionRegionID[dir][e]==-1)
                {
                    congestionRegionID[dir][e] = e;
                }
            }
            markCongestion(e, x);
        }
        for(int dir=0; dir<2; dir++)
        {
            int x_region = getRegionID(x, dir);
            for(auto e : graph_x[x]) if(e != par)
            {
                int ex = graph_x.back()[e] / Y % X;
                int ey = graph_x.back()[e] % Y;
                int e_region = getRegionID(e, dir);
                if(x_region!=e_region)
                {
                    if(x_region>=0)
                    {
                        stems[dir][x_region].push_back(dir?ex:ey);
                        congestionRanges[dir][x_region].first = min(congestionRanges[dir][x_region].first, dir?ey:ex);
                        congestionRanges[dir][x_region].second = max(congestionRanges[dir][x_region].second, dir?ey:ex);
                    }
                    if(e_region>=0)
                    {
                        stems[dir][e_region].push_back(dir?curx:cury);
                        congestionRanges[dir][e_region].first = min(congestionRanges[dir][e_region].first, dir?cury:curx);
                        congestionRanges[dir][e_region].second = max(congestionRanges[dir][e_region].second, dir?cury:curx);
                    }
                }
            }
        }
    };
    markCongestion(select_root, -1);

    function<int(int, int, int, int)> create_node = [&] (int l, int x, int y, int num_child) {
        int node_idx_insert = node_index_cnt++;
        par_num_cpu.emplace_back(0);
        par_num_sum_cpu.emplace_back(0);
        par_nodes_cpu.emplace_back(0);
        child_num_cpu.emplace_back(0);
        node_depth_cpu.emplace_back(0);
        nodes_cpu.emplace_back(0);
        
        par_num_sum_cpu[node_idx_insert+1] = 0;
        node_depth_cpu[node_idx_insert] = 0;
        child_num_cpu[node_idx_insert] = num_child;
        nodes_cpu[node_idx_insert] = l * X * Y + x * Y + y;
        return node_idx_insert;
    };

    function<int(int, int, int)> connect_node = [&] (int par_node_index, int cur_index, int cur_child_id) {
        int node_idx_insert = cur_index;
        int position_cur = nodes_cpu[cur_index];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int position_par = nodes_cpu[par_node_index];
        int parx = position_par/ Y % X, pary = position_par % Y;
        if(construct_segments)
        {
            if(curx==parx)
            {
                rsmt_v_segments.emplace_back(make_pair(curx*Y+min(cury, pary), curx*Y+max(cury, pary)));
            }
            if(cury==pary)
            {
                rsmt_h_segments.emplace_back(make_pair(min(curx, parx)+cury*X, max(curx, parx)+cury*X));
            }
        }        
        points.emplace_back(curx*Y+cury);
        points.emplace_back(parx*Y+pary);
        assert(curx==parx||cury==pary);
        node_depth_cpu[node_idx_insert] = max(node_depth_cpu[node_idx_insert], node_depth_cpu[par_node_index] + 1);
        int depth = node_depth_cpu[node_idx_insert];

        int position = par_num_sum_cpu[node_idx_insert]+par_num_cpu[node_idx_insert]++;
        par_nodes_cpu.emplace_back(0);
        currentChildIDX_cpu.emplace_back(0);
        par_nodes_cpu[position] = par_node_index;
        depth_max = max(depth_max, depth+1);
        assert(cur_child_id<child_num_cpu[par_node_index]);
        currentChildIDX_cpu[position]=cur_child_id;
        return node_idx_insert;
    };
    float ratio = 0.15;
    int num_tracks = 9;
    function<int(int, int, int)> calc_displace = [&] (int query_pos, int dir, int region_id) {
        int ans = 0;
        for(auto pos: stems[dir][region_id])
        {
            ans+=abs(pos-query_pos);
        }
        return ans;
    };
    function<vector<int>(int, int)> get_mirror_places = [&] (int graph_node_id, int dir) {
        assert(graph_node_id<graph_x.back().size());
        int position_cur = graph_x.back()[graph_node_id];
        assert(congestionRegionID[dir][graph_node_id]>=0);
        int congestion_region = congestionRegionID[dir][graph_node_id];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int trunk_len = congestionRanges[dir][congestion_region].second - congestionRanges[dir][congestion_region].first;
        int max_displace = ratio*float(trunk_len);
        int origional_pos = dir?curx:cury;
        int origional_displacement = calc_displace(origional_pos, dir, congestion_region);
        int init_low = origional_pos;
        int init_high = origional_pos;
        int bound = dir?X:Y;
        while (init_low - 1 >= 0 && calc_displace(init_low - 1, dir, congestion_region) - origional_displacement <= max_displace) init_low--;
        while (init_high + 1 < bound && calc_displace(init_high - 1, dir, congestion_region) - origional_displacement <= max_displace) init_high++;
        int step = 1;
        while ((origional_pos - init_low) / (step + 1) + (init_high - origional_pos) / (step + 1) >= num_tracks) step++;
        init_low = origional_pos - (origional_pos - init_low) / step * step;
        init_high = origional_pos + (init_high - origional_pos) / step * step;
        vector<int> shifts;
        for (double pos = init_low; pos <= init_high; pos += step) {
            int shiftAmount = (pos - origional_pos); 
            if(shiftAmount==0) continue;
            shifts.push_back(pos);
            int min_trunk = congestionRanges[dir][congestion_region].first;
            int max_trunk = congestionRanges[dir][congestion_region].second;
        }
        std::vector<int> indices(shifts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::vector<int> new_shifts;
        for (int index : indices) {
            new_shifts.push_back(shifts[index]);
        }
    
        shifts = new_shifts;
        return shifts;
    };

    function<void(int, int, int, int, int, vector<vector<int>>, vector<vector<int>>)> dfs_detours = [&] (int x, int par, int par_node_idx, int child_idx, int depth, vector<vector<int>> mirrors, vector<vector<int>> old_mirror_places) {
        if(mirrors.size()==0)
        {
            mirrors.resize(2);
        }
        int size = graph_x.back().size() - 1;
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur/ Y % X, cury = position_cur % Y;
        int node_idx = -1;
        if(x==select_root)
        {
            node_idx = create_node(curl,curx,cury, graph_x[x].size());
            par_num_sum_cpu[node_idx+1] += par_num_cpu[node_idx];
            par_num_sum_cpu[node_idx+1] += par_num_sum_cpu[node_idx];
        }
        vector<vector<int>> new_mirrors;
        vector<vector<int>> mirror_places;
        new_mirrors.resize(2);
        mirror_places.resize(2);
        if(old_mirror_places.size()==2)
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(congestionRegionID[dir][x], dir);
            int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
            if(region_id >= 0)
            {
                assert(region_id<graph_x.back().size());
                if(getRegionID(congestionRegionID[dir][par], dir)==par)
                {
                    depth+=2;
                }
                if(x==region_id)
                {
                    mirror_places[dir] = get_mirror_places(x, dir);
                }else{
                    mirror_places[dir] = old_mirror_places[dir];
                }
                if(region_id!=x&&mirror_places[dir].size()!=mirrors[dir].size())
                {
                    assert(0);
                }
                assert(region_id==x||mirror_places[dir].size()==mirrors[dir].size());
                for(int m_i = 0; m_i < mirror_places[dir].size(); m_i++)
                {
                    int new_x = dir?mirror_places[dir][m_i]:curx;
                    int new_y = dir?cury:mirror_places[dir][m_i];
                    int new_mirror=-1;
                    if(region_id!=x)
                    {
                        int pre_idx = mirrors[dir][m_i];
                        new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        assert( mirrors[dir].size()==mirror_places[dir].size());
                        connect_node(pre_idx, new_mirror, child_idx);
                    }
                    else{
                        if(x!=select_root)
                        {
                            int position_par = nodes_cpu[par_node_idx];
                            int parx = position_par/ Y % X, pary = position_par % Y;
                            if(new_x!=parx&&new_y!=pary)
                            {
                                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                    int midx = pathIndex ? new_x : parx;
                                    int midy = pathIndex ? pary : new_y;
                                    int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                    connect_node(par_node_idx, node_insert, child_idx);
                                    par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert];
                                    par_num_sum_cpu[node_insert + 1] += par_num_sum_cpu[node_insert];
                                }
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(new_mirror-1, new_mirror, 0);
                                connect_node(new_mirror-2, new_mirror, 0);
                            }
                            else{
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(par_node_idx, new_mirror, child_idx);
                            }
                        }else{
                            new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        } 
                    }
                    assert(new_mirror>0);
                    new_mirrors[dir].push_back(new_mirror);
                    par_num_sum_cpu[new_mirror + 1] += par_num_cpu[new_mirror];
                    par_num_sum_cpu[new_mirror + 1] += par_num_sum_cpu[new_mirror];
                }
            }
        }
        if(par_node_idx == -1){}
        else {
            int px = nodes_cpu[par_node_idx] / Y % X, py = nodes_cpu[par_node_idx] % Y;
            vector<int> pre_node_idxs;
            vector<int> pre_node_idxs_direct;
            if(px != curx && py != cury)
            {
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int midx = pathIndex ? curx : px;
                    int midy = pathIndex ? py : cury;
                    int pre_node = create_node(MAX_LAYER-1, midx, midy, 1);
                    connect_node(par_node_idx, pre_node, child_idx);
                    par_num_sum_cpu[pre_node + 1] += par_num_cpu[pre_node];
                    par_num_sum_cpu[pre_node + 1] += par_num_sum_cpu[pre_node];                    
                    pre_node_idxs.push_back(pre_node);
                }
                
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int length_edge = pathIndex ? (max(py, cury) - min(py, cury)) : (max(px, curx) - min(px, curx));
                    int max_z_shape = min(10, length_edge);
                    for(int dispace_id = 1; dispace_id < max_z_shape; dispace_id++)
                    {
                        int midx1 = pathIndex ? px : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy1 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : py;
                        int pre_node1 = create_node(MAX_LAYER-1, midx1, midy1, 1);
                        connect_node(par_node_idx, pre_node1, child_idx);
                        par_num_sum_cpu[pre_node1 + 1] += par_num_cpu[pre_node1];
                        par_num_sum_cpu[pre_node1 + 1] += par_num_sum_cpu[pre_node1];
    
                        int midx2 = pathIndex ? curx : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy2 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : cury;
                        int pre_node2 = create_node(MAX_LAYER-1, midx2, midy2, 1);
                        connect_node(pre_node1, pre_node2, 0);
                        par_num_sum_cpu[pre_node2 + 1] += par_num_cpu[pre_node2];
                        par_num_sum_cpu[pre_node2 + 1] += par_num_sum_cpu[pre_node2];                   
                        pre_node_idxs.push_back(pre_node2);
                    }
                }
            }
            for(int dir = 0; dir<2; dir++)
            {
                int region_id = getRegionID(congestionRegionID[dir][x], dir);
                int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
                if(par_region_id>=0&&region_id!=par_region_id)
                {
                    for(auto node_par_mirror: mirrors[dir])
                    {
                        int position_par = nodes_cpu[node_par_mirror];
                        int parx = position_par/ Y % X, pary = position_par % Y;
                        if(parx!=curx&&pary!=cury)
                        {
                            for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                int midx = pathIndex ? curx : parx;
                                int midy = pathIndex ? pary : cury;
                                int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                connect_node(node_par_mirror, node_insert, child_idx);
                                par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert] + par_num_sum_cpu[node_insert];
                                pre_node_idxs.push_back(node_insert);
                            }
                        }else{
                            pre_node_idxs_direct.push_back(node_par_mirror);
                        }
                    }
                }
            }
            int connect_parent = par_node_idx;
            int connect_child_idx = child_idx;

            node_idx = create_node(curl, curx, cury, graph_x[x].size() - 1);
            
            if(px == curx || py == cury){
                connect_node(connect_parent, node_idx, connect_child_idx);
            }
            if(x!=select_root)
            {
                for(auto pre_node_idx: pre_node_idxs)
                {
                    connect_node(pre_node_idx, node_idx, 0);
                }
                for(auto pre_node_idx: pre_node_idxs_direct)
                {
                    connect_node(pre_node_idx, node_idx, child_idx);
                }
            }
        }
        depth_max = max(depth_max, node_depth_cpu[node_idx]+1);
        if(x!=select_root)
        {
            par_num_sum_cpu[node_idx+1] += (par_num_sum_cpu[node_idx] + par_num_cpu[node_idx]);
        }
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(x, dir);
            int par_region_id = getRegionID(par, dir);
            if(region_id<0) continue;
            if(true)
            {
                int pos2 = dir?cury:curx;
                assert(region_id>=0||region_id<graph_x.back().size());
                int is_tail = congestionRanges[dir][region_id].second==pos2 || congestionRanges[dir][region_id].first==pos2;
                int is_pin = curl < MAX_LAYER - 1;
                assert(x!=select_root);
                if(x!=select_root&&is_pin)
                {
                    for(auto mirror_id: new_mirrors[dir])
                    {
                        int node_duplicate = create_node(curl, curx, cury, 0);
                        child_num_cpu[mirror_id] = graph_x[x].size();// connect in advance
                        connect_node(mirror_id, node_duplicate, graph_x[x].size()-1);
                        par_num_sum_cpu[node_duplicate + 1] += par_num_cpu[node_duplicate];
                        par_num_sum_cpu[node_duplicate + 1] += par_num_sum_cpu[node_duplicate];
                    }
                }
            }
        }
        int idx = 0;
        for(auto e : graph_x[x]) if(e != par)
        {
            depth_max = max(depth_max, depth+1);
            dfs_detours(e, x, node_idx, idx++, depth+1, new_mirrors, mirror_places);
        }
    };
    if(construct_segments)
    {
        rsmt_h_segments.clear();
        rsmt_v_segments.clear();
        rsmt_h_segments.reserve(rsmt.size()*(num_tracks*2));
        rsmt_v_segments.reserve(rsmt.size()*(num_tracks*2));
    }
    points.clear();
    points.reserve(rsmt.size()*(num_tracks*4));
    dfs_detours(select_root, -1, -1, -1, 0, {}, {});
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end());
    points.erase(last, points.end());
    if(construct_segments)
    {
        for(int ii=0; ii<rsmt_h_segments.size();ii++)
        {
            int xpos = rsmt_h_segments[ii].first%X;
            int ypos = rsmt_h_segments[ii].first/X;
            rsmt_h_segments[ii].first = xpos*Y+ypos;
            xpos = rsmt_h_segments[ii].second%X;
            ypos = rsmt_h_segments[ii].second/X;
            rsmt_h_segments[ii].second = xpos*Y+ypos;
        }
    }
}



void build_cuda_database() {


    const int MAX_LEN_INT = 1700000000, MAX_LEN_DOUBLE = 1700000000;
    static int *temp_int = new int[MAX_LEN_INT];
    static double *temp_double = new double[MAX_LEN_DOUBLE];
    static float *temp_float = new float[MAX_LEN_DOUBLE];

    
    L = db::L - 1;
    X = db::X;
    Y = db::Y;
    XY = max(X, Y);
    

    nets.reserve(db::nets.size());

    int MAX_PIN_SIZE = 2000;
    int net_break_count = 0, max_pin_cnt = 1;
    for(int db_net_id = 0; db_net_id < db::nets.size(); db_net_id++) {
        auto &db_net = db::nets[db_net_id];
        if(db_net.pins.size() == 1) continue;
        max_pin_cnt = max(max_pin_cnt, (int) db_net.pins.size());
        if(db_net.pins.size() <= MAX_PIN_SIZE) {

            net new_net;
            new_net.pins = db_net.pins;
            new_net.original_net_id = db_net_id;
            for(auto &p : new_net.pins) 
                if(p >= X * Y) p -= X * Y;
            db_net.subnets.emplace_back(nets.size());
            nets.emplace_back(move(new_net));
            db_net.unfinished_subnet_count = db_net.subnets.size();
            continue;
        }
        net_break_count++;
        vector<int> pins = db_net.pins, sz(db_net.pins.size(), 1), par(db_net.pins.size());
        vector<tuple<int, int, int>> edges;
        edges.reserve(db_net.pins.size() * db_net.pins.size() / 2);

        function<int(int)> find_par = [&] (int x) { return x == par[x] ? x : par[x] = find_par(par[x]); };

        for(int i = 0; i < db_net.pins.size(); i++) {
            par[i] = i;
            if(pins[i] >= X * Y) pins[i] -= X * Y;
            for(int j = 0; j < i; j++) {
                int x0 = db_net.pins[i] / Y % X, y0 = db_net.pins[i] % Y;
                int x1 = db_net.pins[j] / Y % X, y1 = db_net.pins[j] % Y;
                edges.emplace_back(make_tuple(j, i, abs(x0 - x1) + abs(y0 - y1)));
            }
        }
        sort(edges.begin(), edges.end(), [&] (tuple<int, int, int> l, tuple<int, int, int> r) {
            return get<2> (l) < get<2> (r);
        });
        for(auto e : edges) {
            int u = find_par(get<0> (e)), v = find_par(get<1> (e));
            if(u == v || sz[u] + sz[v] > MAX_PIN_SIZE) continue;
            if(sz[u] > sz[v]) swap(u, v);
            par[u] = v;
            sz[v] += sz[u];
        }
        vector<vector<int>> new_pins(pins.size());
        for(int i = 0; i < pins.size(); i++)
            new_pins[find_par(i)].emplace_back(pins[i]);
        for(auto e : edges) {
            int u = get<0> (e), v = get<1> (e);
            int par_u = find_par(u), par_v = find_par(v);
            if(par_u == par_v) continue;
            if(sz[par_u] > sz[par_v]) {
                swap(u, v);
                swap(par_u, par_v);
            }
            sz[par_u]++;
            new_pins[par_u].emplace_back(pins[v]);
            par[par_v] = par_u;
            db_net.subnets.emplace_back(nets.size());
            nets.emplace_back(net());
            nets.back().pins = move(new_pins[par_v]);
            nets.back().original_net_id = db_net_id;
        }
        for(int i = 0; i < pins.size(); i++) if(find_par(i) == i) {
            db_net.subnets.emplace_back(nets.size());
            nets.emplace_back(net());
            nets.back().pins = move(new_pins[i]);
            nets.back().original_net_id = db_net_id;
        }
        db_net.unfinished_subnet_count = db_net.subnets.size();
    }
    pin_cnt_sum_cpu.resize(1 + nets.size(), 0);
    for(int i = 0; i < nets.size(); i++) {
        nets[i].calc_hpwl();
        pin_cnt_sum_cpu[i + 1] = pin_cnt_sum_cpu[i] + nets[i].pins.size();
    }
    printf("    MAX PINS: %d\n", max_pin_cnt);
    printf("    Broken Nets: %d\n", net_break_count);

    
    NET_NUM = nets.size();


    DIR = db::layers[1].dir;
    unit_length_wire_cost = db::unit_length_wire_cost;
    unit_via_cost = db::unit_via_cost;

    assert(X - 1 <= MAX_LEN_INT);
    for(int i = 0; i < X - 1; i++) temp_int[i] = db::x_edge_len[i];
    cudaMalloc(&x_edge_len, (X - 1) * sizeof(int));
    cudaMemcpy(x_edge_len, temp_int, (X - 1) * sizeof(int), cudaMemcpyHostToDevice);


    assert(Y - 1 <= MAX_LEN_INT);
    for(int i = 0; i < Y - 1; i++) temp_int[i] = db::y_edge_len[i];
    cudaMalloc(&y_edge_len, (Y - 1) * sizeof(int));
    cudaMemcpy(y_edge_len, temp_int, (Y - 1) * sizeof(int), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::unit_length_short_costs[i + 1];
    cudaMalloc(&unit_length_short_costs, L * sizeof(double));
    cudaMemcpy(unit_length_short_costs, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::layers[i + 1].min_len;
    cudaMalloc(&layer_min_len, L * sizeof(double));
    cudaMemcpy(layer_min_len, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);


    assert(L * X * Y <= MAX_LEN_DOUBLE);
    for(int l = 0; l < L; l++)
        for(int x = 0; x < X; x++)
            for(int y = 0; y < Y; y++)
                temp_float[IDX(l, x, y)] = db::capacity[l + 1][x][y];
    cudaMalloc(&capacity, L * X * Y * sizeof(float));
    cudaMemcpy(capacity, temp_float, L * X * Y * sizeof(float), cudaMemcpyHostToDevice);
    

    cudaMalloc(&pin_acc_num, (1 + NET_NUM) * sizeof(int));
    cudaMemcpy(pin_acc_num, pin_cnt_sum_cpu.data(), (1 + NET_NUM) * sizeof(int), cudaMemcpyHostToDevice);
    PIN_NUM = pin_cnt_sum_cpu.back();

    if(LOG) cerr << "PIN_NUM " << PIN_NUM << endl;

    
    assert(PIN_NUM <= MAX_LEN_INT);

    for(auto &dbnet : db::nets)
        for(auto pin : dbnet.pins) 
            if(pin < X * Y) total_via_count++;

    for(int i = 0, pin_id = 0; i < NET_NUM; i++)
        for(auto pin : nets[i].pins) temp_int[pin_id++] = pin;
    cudaMalloc(&pins, PIN_NUM * sizeof(int));
    cudaMemcpy(pins, temp_int, PIN_NUM * sizeof(int), cudaMemcpyHostToDevice);


    net_x_cpu.resize(NET_NUM);
    net_y_cpu.resize(NET_NUM);
    

    all_track_cnt = 0;
    for(int l = 0; l < L; l++) all_track_cnt += (l & 1 ^ DIR) ? X : Y;
    assert(all_track_cnt <= MAX_LEN_INT);
    for(int l = 0, cnt = 0; l < L; l++) {
        if((l & 1 ^ DIR) == 0) for(int y = 0; y < Y; y++) temp_int[cnt++] = l * XY + y;
        if((l & 1 ^ DIR) == 1) for(int x = 0; x < X; x++) temp_int[cnt++] = l * XY + x;
        if(l + 1 == L) assert(cnt == all_track_cnt);
    }
    cudaMalloc(&idx2track, all_track_cnt * sizeof(int));
    cudaMemcpy(idx2track, temp_int, sizeof(int) * all_track_cnt, cudaMemcpyHostToDevice);


    cudaMalloc(&congestion, X * Y * sizeof(bool));
    cudaMalloc(&congestion_xsum, X * Y * sizeof(float));
    cudaMalloc(&congestion_ysum, X * Y * sizeof(float));


    cudaMallocManaged(&routes, (ROUTE_PER_PIN * PIN_NUM) * sizeof(int));

    cudaMalloc(&wcost, L * X * Y * sizeof(float));
    cudaMalloc(&vcost, L * X * Y * sizeof(float));
    cudaMalloc(&presum, L * X * Y * sizeof(double));
    cudaMalloc(&demand, L * X * Y * sizeof(float));
    cudaMalloc(&pre_demand, L * X * Y * sizeof(int));



    cudaMalloc(&net_ids, NET_NUM * sizeof(int));
    cudaMalloc(&is_of_net, NET_NUM * sizeof(bool));
    cudaMalloc(&of_edge_sum, L * X * Y * sizeof(int));
    cudaMalloc(&last, L * X * Y * sizeof(int));
    cudaMalloc(&timestamp, L * X * Y * sizeof(int));


    

    cudaMemset(demand, 0, sizeof(float) * L * X * Y);
    cudaMemset(timestamp, 0, sizeof(int) * L * X * Y);
    cudaMemset(pre_demand, 0, sizeof(int) * L * X * Y);

    net_ids_cpu = new int[NET_NUM];
}

}

using namespace cudb;
