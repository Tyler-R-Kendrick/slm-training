# Continuous autotrain: 2026-08-03 cycle 2, session n8vwtq (non-positive, eval timeout)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Base commit:** `f877fc65` (this session's cycle-1 docs commit on `main`)

| Arm | Loss | Checkpoint SHA256 | Ship eval |
| --- | ---: | --- | --- |
| control | 14.3902 | `6abf57d4...db3512b` | timed out mid-decode (0/3 documents) |
| component-plan | 19.9160 | `20e573b1...f0f8a8e741` | timed out mid-decode (0/3 documents) |

**Verdict: non-positive (soft timeout, not a model result).** Both arms
trained to completion and produced checkpoints **byte-identical** to the
`j48f8u` session's independently-reproduced cycle 2 (same hypothesis,
positive 3 prior times: #1369, #1376, #1378). Honest ship-gate evaluation did
not finish: `compiler_decode_mode=tree` measures ~35.7s per smoke document on
this host's CPU (`decode_progress.json`: `compiler_ms_mean=34716`,
`total_ms_mean=35714` for the first document alone), so 3 documents exceed
the per-experiment wall budget mid-decode for both arms
(`decode_progress.status=interrupted`, `processed_record_n=0/3`).

This is a **host-speed-dependent soft timeout**, not a model or harness
defect. Per `continuous.md`: "Soft failures... timeouts... never stop the
loop." `MAX_RUN_MINUTES` is intentionally **not** weakened to force a pass.
No stacked PR for this cycle — docs-only local commit.

## Why this isn't scored as negative

The training-side result (byte-identical checkpoints to 3 prior independent
positive reproductions) already corroborates the `component-plan` hypothesis
on the training path; only this session's decode/eval stage is
environment-bound. `retry_measurement` is queued to replay the identical
frozen arm. If the timeout recurs across the configured
`max_consecutive_frozen_replays` bound, the driver will emit a
`repair_harness` action against `model_build` for eval wall-budget scaling
by decode mode — that repair (if warranted) is deferred to the next
iteration, not invented speculatively here.

## Next priorities

1. `retry_measurement` — replay the identical frozen `c2` arms.
2. The `component-plan` positive finding is already independently reproduced
   3 times; this cycle's byte-identical checkpoints are further (unscored)
   training-side corroboration only.
3. Do not conflate a host-speed eval timeout with a null or negative model
   result.

Machine evidence:
[`continuous-openui-local-n8vwtq-c2-results.json`](continuous-openui-local-n8vwtq-c2-results.json).
