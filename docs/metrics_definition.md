# Routing Metrics Definition

All metrics are extracted by `InstantGR/run/evaluator` and parsed by `scripts/eval_routing_result.py`.

## Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| `wirelength_cost` | Sum of wire segment lengths weighted by per-layer unit length cost | `FOR_STAT` field 1 |
| `via_cost` | Total via penalty, weighted by `unit_via_cost` from the `.cap` file | `FOR_STAT` field 2 |
| `overflow_cost` | Sum of per-edge overflow penalties; edge overflow = `max(0, demand - capacity)` | `FOR_STAT` field 3 |
| `total_cost` | `wirelength_cost + via_cost + overflow_cost` | `FOR_STAT` field 4 |
| `open_nets` | Number of nets with at least one disconnected pin pair | evaluator stdout |
| `incompleted_nets` | Number of nets that could not be fully routed | evaluator stdout |
| `runtime_s` | Wall-clock time for routing (excludes evaluator) | measured by run script |

## Evaluator Output Format

The evaluator prints a machine-readable summary line:
```
FOR_STAT <wirelength_cost> <via_cost> <overflow_cost> <total_cost>
```
All values are floating-point.

## CSV Schema

`results/baseline/metrics.csv` columns:
```
case, tier, wirelength_cost, via_cost, overflow_cost, total_cost, open_nets, incompleted_nets, runtime_s, timestamp
```
