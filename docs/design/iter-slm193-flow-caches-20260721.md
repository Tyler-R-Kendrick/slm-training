# SLM-193 (FFE3-02): bit-exact flow caches (slm193-flow-caches-20260721-f1bb3bc3)

Matrix set: `slm193_flow_caches`

Version: `ffe3-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

Exact state fingerprints recur across decode attempts and evaluation records, so bit-exact content-addressed caches reduce deterministic solver/bridge wall time by at least 2x while preserving identical outputs and certificates.

## Falsifier

Cache hit rates stay below 20% on warm repeated requests, or lookup/serialization overhead offsets the saved work on warm p50/p95, or cached results diverge from fresh computation.

## Arms

| arm_name | total_ms | hit_rate | n_entries | bytes_stored | speedup |
| --- | --- | --- | --- | --- | --- |
| exact_closure_cold | 1.049 | 0.00 | 6 | 4364 | 1.00 |
| exact_closure_warm | 1.301 | 0.67 | 6 | 4364 | 0.81 |
| exact_closure_cross_request | 1.212 | 0.50 | 12 | 8728 | 1.00 |
| bridge_plan_cold | 7.600 | 0.00 | 1 | 9031 | 1.00 |
| bridge_plan_warm | 0.015 | 1.00 | 1 | 24 | 1.00 |
| disk_restart | 1.523 | 1.00 | 6 | 0 | 1.00 |
| version_invalidation | 0.553 | 0.00 | 12 | 8728 | 1.00 |

## Disposition

**cache_wired**

Bit-exact content-addressed cache layer wired for exact closure and bridge plans; measured cold/warm/restart/invalidation arms on CPU-only fixtures.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The cache hit rates and speedups are measured over deterministic CPU-only operations with synthetic support provider signals. Production caching requires real verifier replay contracts and process-restart provenance before any ship claim.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- The toy support provider is synthetic (payload['ok'] flag); real support queries require a verifier and problem expander.
- Disk cache restart test uses a temporary directory under outputs/; production restart provenance requires a replay-safe certificate contract.
- Only the HERO fixture and one toy finite-domain state are exercised; production hit rates depend on actual state recurrence.

## Reproducibility

```bash
python -m scripts.bench_flow_caches --describe
python -m scripts.bench_flow_caches --fixture
```
