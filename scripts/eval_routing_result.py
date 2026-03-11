#!/usr/bin/env python3
"""
scripts/eval_routing_result.py
Evaluate a single routing result and append metrics to a CSV.

Usage:
    python scripts/eval_routing_result.py \\
        --cap <cap_file> --net <net_file> --out <route_out> \\
        [--case <name>] [--tier <tiny|small|medium>] \\
        [--runtime <seconds>] [--csv <path>]

The evaluator binary is auto-located at InstantGR/run/evaluator.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = REPO_ROOT / "InstantGR" / "run" / "evaluator"
DEFAULT_CSV = REPO_ROOT / "results" / "baseline" / "metrics.csv"

CSV_FIELDS = [
    "case", "tier",
    "wirelength_cost", "via_cost", "overflow_cost", "total_cost",
    "open_nets", "incompleted_nets", "runtime_s", "timestamp",
]


def parse_evaluator_output(stdout: str) -> dict:
    metrics = {
        "wirelength_cost": None, "via_cost": None,
        "overflow_cost": None,   "total_cost": None,
        "open_nets": None,       "incompleted_nets": None,
    }
    for line in stdout.splitlines():
        if line.startswith("FOR_STAT"):
            parts = line.split()
            metrics["wirelength_cost"] = float(parts[1])
            metrics["via_cost"]        = float(parts[2])
            metrics["overflow_cost"]   = float(parts[3])
            metrics["total_cost"]      = float(parts[4])
        elif "Number of open nets" in line:
            metrics["open_nets"] = int(line.split(":")[-1].strip())
        elif "Number of incompleted" in line:
            metrics["incompleted_nets"] = int(line.split(":")[-1].strip())
    return metrics


def run_evaluator(cap: str, net: str, out: str) -> str:
    if not EVALUATOR.exists():
        sys.exit(
            f"Evaluator not found at {EVALUATOR}.\n"
            "Compile with: g++ -o InstantGR/run/evaluator InstantGR/run/evaluator.cpp -O3 -std=c++17"
        )
    result = subprocess.run(
        [str(EVALUATOR), cap, net, out],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f"Evaluator exited with code {result.returncode}")
    return result.stdout


def append_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a routing result and record metrics.")
    parser.add_argument("--cap",     required=True, help=".cap resource file")
    parser.add_argument("--net",     required=True, help=".net netlist file")
    parser.add_argument("--out",     required=True, help="routing solution file")
    parser.add_argument("--case",    default=None,  help="case name (default: basename of --out)")
    parser.add_argument("--tier",    default="?",   help="tiny|small|medium")
    parser.add_argument("--runtime", type=float, default=None, help="routing runtime in seconds")
    parser.add_argument("--csv",     default=str(DEFAULT_CSV), help="output CSV path")
    args = parser.parse_args()

    case_name = args.case or Path(args.out).stem

    print(f"[eval] Running evaluator on {case_name}...")
    stdout = run_evaluator(args.cap, args.net, args.out)
    print(stdout)

    metrics = parse_evaluator_output(stdout)
    if metrics["total_cost"] is None:
        sys.exit("ERROR: could not parse FOR_STAT from evaluator output.")

    row = {
        "case":             case_name,
        "tier":             args.tier,
        "wirelength_cost":  metrics["wirelength_cost"],
        "via_cost":         metrics["via_cost"],
        "overflow_cost":    metrics["overflow_cost"],
        "total_cost":       metrics["total_cost"],
        "open_nets":        metrics["open_nets"],
        "incompleted_nets": metrics["incompleted_nets"],
        "runtime_s":        args.runtime,
        "timestamp":        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    append_csv(Path(args.csv), row)
    print(f"[eval] Metrics appended → {args.csv}")
    print(f"[eval] total_cost={metrics['total_cost']:.4f}  "
          f"wl={metrics['wirelength_cost']:.4f}  "
          f"via={metrics['via_cost']:.4f}  "
          f"overflow={metrics['overflow_cost']:.4f}")


if __name__ == "__main__":
    main()
