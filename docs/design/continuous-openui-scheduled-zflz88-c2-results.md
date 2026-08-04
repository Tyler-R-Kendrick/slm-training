# Continuous autotrain: 2026-08-04 (session zflz88, scheduled) cycle 2 — decode-timeout infra block on component-plan screen

**Loop:** `continuous-openui-scheduled-zflz88`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-486913c8-c2`
**Integration commit:** `cba9a620` (post-c1-docs)

**Verdict:** measurement incomplete on both arms — not a model result, not a
harness regression.

| Arm | Params | decode_timeout_count | effective timeout | compiler_ms_mean (partial) |
| --- | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 3/3 | 24.0s | 23,565.3 ms |
| component-plan | 1,755,764 | 3/3 | 24.0s | 23,184.3 ms |

Every smoke record on both arms hit the decode wall clock before finishing;
`primary_metric` (`smoke.structural_similarity`) is unavailable for either
arm, so there is no attribution and no delta to classify.

## Diagnosis: environment-speed-bound, not a code regression

Both arms are size-matched at 1,755,764 params — larger than cycle 1's
1,608,962-param arms, which completed cleanly at the same 24.0s effective
per-record budget. That budget is computed by
`_effective_record_decode_timeout()`
(`src/slm_training/harnesses/model_build/eval_runner.py`) as a fair share of
the fixed `MAX_RUN_MINUTES=3` (180s) cumulative wall cap — it is not a raw
constant to retune.

The identical `component-plan` hypothesis, same recipe, same commit lineage,
has independently **completed and shown a positive structural_similarity
win** in multiple prior sessions:
[`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md),
[`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md).
That rules out a new regression in the component-plan code path — this
container's available CPU headroom today was not enough to fit both train
(20 steps) and eval (3 smoke records) for the larger arm inside the fixed
180s wall cap.

`MAX_RUN_MINUTES` is a non-negotiable hard cap (`AGENTS.md`) and is not
raised to compensate. No code change was made.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure`). No stack
layer opens for this cycle.

## Handoff action disposition

The driver queued a typed `repair_harness` action
(`harness_family=model_build`,
`frozen_manifest_sha256=e6e1caf222dc5fdde26e83c7f13d2bf509251204dc633f77736f90bcd6672c4c`).
Acknowledged **blocked** (not completed) with this diagnosis as evidence —
forcing a harness change without evidence of an actual bug would violate
`improve-openui-harnesses`' "change the shared owner only when evidence
requires it" contract. `retry_measurement` for the identical frozen arm
stays queued for a future cycle/session with more CPU headroom.

## Next priorities

1. Retry the identical frozen `c2-{control,component-plan}` arms in a future
   cycle; do not derive a new hypothesis from an incomplete measurement.
2. If decode timeouts on size-matched (>1.6M param) arms reproduce again
   under the same code in a future session, escalate to
   `improve-openui-harnesses` for a genuine decode-budget fix rather than
   attributing it to environment noise a third time.
3. Keep the c1 matched control (1,608,962 params) as the baseline for any
   smaller-capacity hypothesis screens.

Machine evidence:
[`continuous-openui-scheduled-zflz88-c2-results.json`](continuous-openui-scheduled-zflz88-c2-results.json).
