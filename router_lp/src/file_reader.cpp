// router_lp/src/file_reader.cpp
// Reads .cap and .net files. Independent of InstantGR.

#include "../include/file_reader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rlp {

// ── .cap reader ──────────────────────────────────────────────────────────────

bool read_cap_file(const std::string& path, GridInfo& grid) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[file_reader] Cannot open cap file: " << path << "\n";
        return false;
    }

    f >> grid.L >> grid.X >> grid.Y;
    f >> grid.unit_wire_cost >> grid.unit_via_cost;

    grid.layer_short_cost.resize(grid.L);
    for (int i = 0; i < grid.L; ++i) f >> grid.layer_short_cost[i];

    // x-edge lengths (X-1 values) and y-edge lengths (Y-1 values)
    // (used to compute actual wire lengths; we store but don't fully use for now)
    std::vector<double> xlen(grid.X - 1), ylen(grid.Y - 1);
    for (int i = 0; i < grid.X - 1; ++i) f >> xlen[i];
    for (int i = 0; i < grid.Y - 1; ++i) f >> ylen[i];

    grid.layer_dir.resize(grid.L);
    grid.cap.assign(grid.L, std::vector<std::vector<double>>(
        grid.X, std::vector<double>(grid.Y, 0.0)));

    for (int l = 0; l < grid.L; ++l) {
        std::string name;
        int dir;
        double min_len;
        if (!(f >> name >> dir >> min_len)) {
            std::cerr << "[file_reader] Failed to read layer " << l << " header\n";
            return false;
        }
        grid.layer_dir[l] = dir;  // 0 = H, 1 = V (integer, as in actual .cap files)

        for (int y = 0; y < grid.Y; ++y)
            for (int x = 0; x < grid.X; ++x)
                f >> grid.cap[l][x][y];
    }

    if (f.fail() && !f.eof()) {
        std::cerr << "[file_reader] cap parse error\n";
        return false;
    }
    return true;
}

// ── .net reader ──────────────────────────────────────────────────────────────
// Format per net:
//   net_name
//   (
//   [(l, x, y), (l, x, y), ...]   ← pin group 1 (one access point chosen)
//   [(l, x, y)]                   ← pin group 2
//   ...
//   )
//
// Parsing strategy: scan character-by-character.
//   '[' → start of pin group, read (l,x,y) triplets until ']'
//   ')' → end of net
// Commas, parentheses inside pin groups are treated as non-numeric separators.

// Read the next (l, x, y) triplet from buffer[pos], skipping non-digits.
// Returns false when a ']' or ')' is encountered before three integers.
static bool read_triplet(const std::vector<char>& buf, size_t& pos,
                         int& l, int& x, int& y)
{
    auto skip_non_digit = [&](char stop) -> bool {
        while (pos < buf.size()) {
            if (std::isdigit(buf[pos])) return true;
            if (buf[pos] == stop) return false;
            ++pos;
        }
        return false;
    };
    auto read_int = [&]() -> int {
        int v = 0;
        while (pos < buf.size() && std::isdigit(buf[pos]))
            v = v * 10 + (buf[pos++] - '0');
        return v;
    };

    if (!skip_non_digit(']')) return false;
    l = read_int();
    if (!skip_non_digit(']')) return false;
    x = read_int();
    if (!skip_non_digit(']')) return false;
    y = read_int();
    return true;
}

bool read_net_file(const std::string& path, const GridInfo& grid,
                   std::vector<MultiPinNet>& nets)
{
    std::ifstream f(path, std::ios::ate);
    if (!f) {
        std::cerr << "[file_reader] Cannot open net file: " << path << "\n";
        return false;
    }
    size_t fsize = f.tellg();
    f.seekg(0);
    std::vector<char> buf(fsize + 1);
    f.read(buf.data(), fsize);
    buf[fsize] = '\0';

    size_t pos = 0;
    int n_ok = 0, n_skip = 0;

    auto skip_ws = [&]() {
        while (pos < fsize && (buf[pos] == ' ' || buf[pos] == '\t' ||
                                buf[pos] == '\r' || buf[pos] == '\n'))
            ++pos;
    };
    auto read_token = [&](std::string& tok) {
        skip_ws();
        tok.clear();
        while (pos < fsize && buf[pos] != ' ' && buf[pos] != '\t' &&
               buf[pos] != '\r' && buf[pos] != '\n')
            tok += buf[pos++];
    };

    while (pos < fsize) {
        // Read net name
        std::string name;
        read_token(name);
        if (name.empty()) break;

        // Skip to '('
        while (pos < fsize && buf[pos] != '(') ++pos;
        ++pos;  // consume '('

        MultiPinNet mnet;
        mnet.name = name;

        // Collect all valid access points per pin group, then select the one
        // closest to the net's bounding-box centre for better routability.
        std::vector<std::vector<GCell>> groups;

        // Read pin groups until ')'
        while (pos < fsize) {
            // Skip whitespace/newlines to find '[' or ')'
            skip_ws();
            if (pos >= fsize) break;
            if (buf[pos] == ')') { ++pos; break; }
            if (buf[pos] != '[') { ++pos; continue; }
            ++pos;  // consume '['

            // Read ALL (l,x,y) triplets in this pin group
            std::vector<GCell> group_pts;
            while (pos < fsize && buf[pos] != ']') {
                int l, x, y;
                if (read_triplet(buf, pos, l, x, y)) {
                    if (l >= 0 && l < grid.L && x >= 0 && x < grid.X &&
                        y >= 0 && y < grid.Y) {
                        group_pts.push_back({x, y, l});
                    } else {
                        std::cerr << "[file_reader] out-of-bounds access point "
                                  << "(" << l << "," << x << "," << y
                                  << ") in net " << name << ", skipping\n";
                    }
                    // Skip to next ',' or ']'
                    while (pos < fsize && buf[pos] != ',' && buf[pos] != ']')
                        ++pos;
                } else {
                    break;
                }
            }
            if (pos < fsize && buf[pos] == ']') ++pos;  // consume ']'
            if (!group_pts.empty()) groups.push_back(std::move(group_pts));
        }

        if (groups.size() < 2) { ++n_skip; continue; }

        // Compute bounding-box centre across all groups' first access points
        // (used as a proxy for net centre; avoids chicken-and-egg with final selection)
        double cx = 0, cy = 0;
        for (const auto& g : groups) { cx += g[0].x; cy += g[0].y; }
        cx /= (double)groups.size();
        cy /= (double)groups.size();

        // For each group, pick the access point closest to (cx, cy)
        for (const auto& g : groups) {
            const GCell* best = &g[0];
            double best_d = (g[0].x - cx) * (g[0].x - cx) + (g[0].y - cy) * (g[0].y - cy);
            for (int i = 1; i < (int)g.size(); ++i) {
                double d = (g[i].x - cx) * (g[i].x - cx) + (g[i].y - cy) * (g[i].y - cy);
                if (d < best_d) { best_d = d; best = &g[i]; }
            }
            mnet.pins.push_back(*best);
        }
        nets.push_back(std::move(mnet));
        ++n_ok;
    }

    std::cout << "[file_reader] read_net: " << n_ok << " nets loaded, "
              << n_skip << " skipped (< 2 pins)\n";
    return n_ok > 0;
}

// ── 2-pin decomposition ──────────────────────────────────────────────────────
// Star decomposition: pick pin[0] as centre, connect to all others.

std::vector<TwoNet> decompose_to_2pin(const MultiPinNet& mnet) {
    std::vector<TwoNet> result;
    if (mnet.pins.size() < 2) return result;

    // Use pin[0] as the root (can improve with HPWL centre selection later)
    const GCell& root = mnet.pins[0];
    for (int i = 1; i < (int)mnet.pins.size(); ++i) {
        TwoNet tn;
        tn.name = mnet.name + "_" + std::to_string(i);
        tn.src  = Pin{root};
        tn.snk  = Pin{mnet.pins[i]};
        result.push_back(std::move(tn));
    }
    return result;
}

} // namespace rlp
