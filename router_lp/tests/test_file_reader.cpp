// router_lp/tests/test_file_reader.cpp
// Unit tests for file_reader: .cap parser and .net parser (multi-AP groups).
// Build (from router_lp/):
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror
//       tests/test_file_reader.cpp src/file_reader.cpp
//       -I include -o /tmp/test_file_reader && /tmp/test_file_reader

#include "../include/file_reader.hpp"
#include "../include/routing_types.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string tmp_path(const char* name) {
    return std::string("/tmp/rlp_test_") + name;
}
static void write_file(const std::string& path, const char* content) {
    std::ofstream f(path);
    assert(f);
    f << content;
}

static int g_run = 0, g_pass = 0;

static void check_impl(bool ok, const char* expr, const char* file, int line) {
    ++g_run;
    if (ok) { ++g_pass; }
    else { std::cerr << "FAIL  " << expr << "  (" << file << ":" << line << ")\n"; }
}
#define CHECK(e) check_impl((e), #e, __FILE__, __LINE__)

// ── .cap tests ────────────────────────────────────────────────────────────────

static void test_cap_basic() {
    write_file(tmp_path("basic.cap"),
        "2 3 3\n"
        "1.0 2.0 0.5 0.5\n"
        "10 10\n"
        "10 10\n"
        "M1 0 0.0\n"
        "3 3 3\n"
        "3 3 3\n"
        "3 3 3\n"
        "M2 1 0.0\n"
        "5 5 5\n"
        "5 5 5\n"
        "5 5 5\n");

    rlp::GridInfo grid;
    CHECK(rlp::read_cap_file(tmp_path("basic.cap"), grid));
    CHECK(grid.L == 2);
    CHECK(grid.X == 3);
    CHECK(grid.Y == 3);
    CHECK(grid.layer_dir[0] == 0);
    CHECK(grid.layer_dir[1] == 1);
    CHECK((int)grid.cap[0][0][0] == 3);
    CHECK((int)grid.cap[1][1][2] == 5);
}

// ── .net tests ────────────────────────────────────────────────────────────────

static rlp::GridInfo make_grid() {
    rlp::GridInfo g;
    g.L = 3; g.X = 10; g.Y = 10;
    g.layer_dir.assign(3, 0);
    g.cap.assign(3, std::vector<std::vector<double>>(10, std::vector<double>(10, 1.0)));
    return g;
}

// Test 1: single AP per group → that AP selected
static void test_net_single_ap() {
    write_file(tmp_path("single_ap.net"),
        "net_a\n(\n[(0, 1, 2)]\n[(0, 5, 6)]\n)\n");

    rlp::GridInfo grid = make_grid();
    std::vector<rlp::MultiPinNet> nets;
    CHECK(rlp::read_net_file(tmp_path("single_ap.net"), grid, nets));
    CHECK(nets.size() == 1u);
    CHECK(nets[0].pins.size() == 2u);
    CHECK(nets[0].pins[0].x == 1 && nets[0].pins[0].y == 2);
    CHECK(nets[0].pins[1].x == 5 && nets[0].pins[1].y == 6);
}

// Test 2: multiple APs — bbox centre from ALL points selects correct AP
//
//  Group 0: (0,0), (1,0), (2,0)   Group 1: (8,9), (9,9)
//  All points: xmin=0 xmax=9 ymin=0 ymax=9  => centre=(4.5, 4.5)
//  Group 0: dist2 x=0→40.5, x=1→32.5, x=2→26.5  => best=(2,0)
//  Group 1: dist2 x=8→32.5, x=9→40.5             => best=(8,9)
static void test_net_multi_ap_bbox() {
    write_file(tmp_path("multi_ap.net"),
        "net_b\n(\n"
        "[(0, 0, 0), (0, 1, 0), (0, 2, 0)]\n"
        "[(0, 8, 9), (0, 9, 9)]\n"
        ")\n");

    rlp::GridInfo grid = make_grid();
    std::vector<rlp::MultiPinNet> nets;
    CHECK(rlp::read_net_file(tmp_path("multi_ap.net"), grid, nets));
    CHECK(nets.size() == 1u);
    CHECK(nets[0].pins.size() == 2u);
    CHECK(nets[0].pins[0].x == 2 && nets[0].pins[0].y == 0);  // closest to (4.5,4.5)
    CHECK(nets[0].pins[1].x == 8 && nets[0].pins[1].y == 9);
}

// Test 3: pin group with all OOB APs => net has < 2 groups => skipped
static void test_net_oob_skipped() {
    write_file(tmp_path("oob.net"),
        "net_c\n(\n[(0, 1, 1)]\n[(99, 1, 1)]\n)\n");

    rlp::GridInfo grid = make_grid();
    std::vector<rlp::MultiPinNet> nets;
    rlp::read_net_file(tmp_path("oob.net"), grid, nets);
    CHECK(nets.empty());
}

// Test 4: single pin group => net skipped
static void test_net_one_group_skipped() {
    write_file(tmp_path("onepin.net"),
        "net_d\n(\n[(0, 3, 4)]\n)\n");

    rlp::GridInfo grid = make_grid();
    std::vector<rlp::MultiPinNet> nets;
    rlp::read_net_file(tmp_path("onepin.net"), grid, nets);
    CHECK(nets.empty());
}

// Test 5: multiple nets; only valid (>=2 groups) loaded
static void test_net_multiple() {
    write_file(tmp_path("multi.net"),
        "net_ok1\n(\n[(0,1,1)]\n[(0,5,5)]\n)\n"
        "net_skip\n(\n[(0,2,2)]\n)\n"
        "net_ok2\n(\n[(1,0,0),(1,1,0)]\n[(1,9,9)]\n)\n");

    rlp::GridInfo grid = make_grid();
    std::vector<rlp::MultiPinNet> nets;
    CHECK(rlp::read_net_file(tmp_path("multi.net"), grid, nets));
    CHECK(nets.size() == 2u);
    CHECK(nets[0].name == "net_ok1");
    CHECK(nets[1].name == "net_ok2");
}

// ── decompose_to_2pin tests ───────────────────────────────────────────────────

static void test_decompose_star() {
    rlp::MultiPinNet mnet;
    mnet.name = "n";
    mnet.pins = {{1,1,0}, {3,3,0}, {5,5,0}, {7,7,0}};
    auto twonets = rlp::decompose_to_2pin(mnet);
    CHECK(twonets.size() == 3u);
    for (const auto& tn : twonets) {
        CHECK(tn.src.loc.x == 1 && tn.src.loc.y == 1);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== test_file_reader ===\n";
    test_cap_basic();
    test_net_single_ap();
    test_net_multi_ap_bbox();
    test_net_oob_skipped();
    test_net_one_group_skipped();
    test_net_multiple();
    test_decompose_star();
    std::cout << g_pass << "/" << g_run << " tests passed\n";
    return (g_pass == g_run) ? 0 : 1;
}
