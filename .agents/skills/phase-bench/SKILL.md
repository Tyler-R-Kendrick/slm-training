---
name: phase-bench
description: Performance benchmarking and generation profiling — decode/telemetry benches, acceleration microbenches, Cactus export benches, and profile_generate hot-path analysis. Use when measuring latency, throughput, or decode performance.
---

# Bench & profile phase

Perf guardrail measurements. Consumers: perf matrix + runtime docs.

## Prerequisites

- A checkpoint (bench_cactus / profile_generate) or train dir (bench_telemetry,
  bench_accel).

## Commands

```bash
# Telemetry decode bench (train + generate probes)
python -m scripts.bench_telemetry --train-dir outputs/data/train/v1 --out <json>

# Acceleration / microbench comparison
python -m scripts.bench_accel --train-dir outputs/data/train/v1 [--microbench] [--skip-hf]

# Cactus export bench
python -m scripts.bench_cactus --checkpoint outputs/runs/<id>/last.pt [--with-design-md]

# Generation hot-path profile
python -m scripts.profile_generate --checkpoint outputs/runs/<id>/last.pt \
  [--quant] [--compile] [--maskgit]
```

## Key flags

`--device`, `--repeats`, `--gen-rounds` / `--gen-prompts`, `--out`;
profile: `--multitoken`, `--lookahead`, `--no-incremental`.

## Outputs

JSON results (pass `--out`) destined for `docs/design/` perf scoreboards.

## Gates & invariants

- Perf guardrails are gates: regressions need a documented decision, not a
  quiet re-baseline.

## Close out

- Iron law: results land in `docs/design/perf-experiment-matrix.md` /
  `runtime-performance.md` with recipe metadata
  (`documenting-experiment-results`).
- Checks: `pytest -q tests/test_runtime`.
