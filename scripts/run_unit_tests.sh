#!/usr/bin/env bash
# Run all router_lp unit tests.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/router_lp"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    tests/test_file_reader.cpp src/file_reader.cpp \
    -I include -o /tmp/rlp_test_file_reader

echo "[unit] Running test_file_reader..."
/tmp/rlp_test_file_reader
echo "[unit] All done."
