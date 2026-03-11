#pragma once
// router_lp/include/file_reader.hpp
// Independent .cap / .net reader — does NOT depend on InstantGR.
//
// .cap format:
//   L X Y
//   unit_wire_cost  unit_via_cost  short_cost_1 ... short_cost_L
//   [x_edge_lengths: (X-1) values, one per line or all on one line]
//   [y_edge_lengths: (Y-1) values]
//   [For each layer l=0..L-1:]
//     name  dir  min_len
//     cap[0][0] cap[0][1] ... cap[0][X-1]
//     cap[1][0] ...
//     ...
//     cap[Y-1][0] ...
//
// .net format (InstantGR):
//   net_name
//   (
//   [(l, x, y), (l, x, y), ...]   <- pin group: one or more (layer,x,y) access points
//   [(l, x, y)]
//   ...
//   )
//   ...
//   The reader picks the first access point in each pin group as the representative pin.

#include "routing_types.hpp"
#include <string>
#include <vector>

namespace rlp {

// Read grid info and per-layer capacities from a .cap file.
// Returns true on success.
bool read_cap_file(const std::string& path, GridInfo& grid);

// Read nets (raw, multi-pin) from a .net file.
// Each returned entry has one chosen access point per pin group (first in group).
// Returns vector of multi-pin nets as TwoNet lists (caller handles decomposition).
// For simplicity this reader returns raw multi-pin nets; use decompose_to_2pin() after.
struct MultiPinNet {
    std::string name;
    std::vector<GCell> pins;  // one access point per pin group (first access)
};

bool read_net_file(const std::string& path, const GridInfo& grid,
                   std::vector<MultiPinNet>& nets);

// Decompose multi-pin net into 2-pin nets via star decomposition (pin[0] as root).
// NOTE: This is a simple star decomposition, not a full HPWL-MST.
//       For N pins it generates N-1 two-pin nets, all sharing pin[0] as one endpoint.
//       A proper MST-based decomposition can be added later for better routability.
std::vector<TwoNet> decompose_to_2pin(const MultiPinNet& mnet);

} // namespace rlp
