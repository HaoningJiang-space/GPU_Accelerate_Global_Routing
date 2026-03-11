#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "[build] Compiling evaluator..."
g++ -O2 -std=c++17 -o InstantGR/run/evaluator InstantGR/run/evaluator.cpp
echo "[build] OK -> InstantGR/run/evaluator"
