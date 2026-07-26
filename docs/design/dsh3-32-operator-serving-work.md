# DSH3-32 operator-serving work

SLM-407 evaluates whether the operator path lowers real serving work at matched
quality. The canonical owner is
`harnesses/experiments/operator_systems_benchmark.py`; it records one raw row
per request and never drops fallback, timeout, or failure rows.

## Comparable denominator

The primary metric is **target-model forward equivalents**:

```text
target_model_forwards + calibrated_non_target_forward_equivalents
```

Each row also retains the raw legal-set build/actions, dry runs, executor,
validator, materialization, cache hit/miss, batch identity, prefix sharing,
CPU/device/wall time, and peak memory. Aggregate p50/p95 covers model forwards,
legal actions, dry runs, executor calls, wall time, and throughput across the
same full request denominator rather than model scoring alone. Every compared arm must
share target model revision, device, precision, batch size, cache mode, context
fingerprint, and meaningful-parse metric.

The required arms are X22, full generation, serialized operator, typed operator,
and compiler-forced. Crossover output is restricted to observed strata and
never interpolates a threshold. An efficiency claim requires lower total
forward-equivalent work *and* lower wall time at matched quality; otherwise the
contract rejects the claim.

## Local preflight (2026-07-26)

[`dsh3-32-operator-serving-work-20260726-local/report.json`](dsh3-32-operator-serving-work-20260726-local/report.json)
records the bounded local preflight against SLM-403's five-head typed-policy
control report. That control matrix rejected the trained policy hypothesis: all
five heads made 0/2 enabled-versus-zero held-out choice changes. No compatible
measurements exist yet for all five serving arms, so this run records
`unavailable`, not an efficiency result. It is local CPU evidence only, has no
checkpoint, no remote/HF workload, no human-rating gate, and no ship claim.

The next measured row must attach raw observations from the current compatible
target model for every required arm; it may not reuse this preflight as a speed
claim.
