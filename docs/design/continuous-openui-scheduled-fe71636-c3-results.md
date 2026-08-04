# Continuous autotrain: 2026-08-04 (scheduled loop `fe71636`) cycle 3 — frozen replay still decode-timeout incomplete, partial repair applied

**Loop:** `continuous-openui-scheduled-fe71636`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-3d42338c-c3`
**Integration commit:** `fb6d460b` (cycle 2's AgentV harness-repair + docs commits merged)
**Replay of:** [`continuous-loop-20260804-continuous-openui-schedu-3d42338c-c2`](continuous-openui-scheduled-fe71636-c2-results.md)

**Verdict:** with the AgentV bootstrap repair in place, both arms now run the
full eval pipeline and produce a real `gates.json` verdict instead of a
`RuntimeError` — but every one of the 3 smoke records on **both** arms hit
the decode timeout, so `structural_similarity` and the other quality metrics
are still `None`/unavailable.

| Arm | Params | compiler_ms_mean | decode_timeout_count | Status |
| --- | ---: | ---: | ---: | --- |
| control | 1,608,962 | 23080.2 | 3/3 | incomplete |
| canvas | 1,608,962 | 23127.6 | 3/3 | incomplete |

## Root cause: an unnecessarily tight config, not a wall-budget shortfall

`scripts/run_autotrain_continuous._fit_screening_decode_timeout_seconds`
already computes a wall-fit ceiling — `(arm_wall_seconds - min_train_floor -
eval_overhead) / smoke_n = (70-20-8)/3 = 14.0s` for the nominal 70s arm
share — and takes `fitted = min(configured, ceiling)`. The **configured**
value (`8.0`) was the binding constraint (`clamp_bound=0.0` in this cycle's
`thrash_timing.json`), not the wall: the 8s-per-record (24s combined) chunk
budget left more than half of the ~51.7s of eval wall actually granted this
cycle unused.

## Repair (`model_build` family, partial)

Commit [`3c30288`](https://github.com/Tyler-R-Kendrick/slm-training/commit/3c3028822f358d5a702d9ee9bceb7968be4e662c)
raises `screening_decode_timeout_seconds` `8 → 12` in `policy.v1.json`
(`v4 → v5`), staying under the 14.0s wall-fit ceiling and inside
`test_climb_policy_measurement_helpers`'s pre-existing declared-safe
`[1.0, 12.0]` range — no test boundary changed. The runtime fair-share clamp
in `eval_runner._effective_record_decode_timeout` still bounds the actual
granted time to whatever wall remains at eval time regardless of this config
value, so raising it cannot cause an overrun; it only stops discarding
already-granted budget. Component `harness.autoresearch.experiment_campaign`
bumped `v178 → v179`.

**This is honestly a partial mitigation, not a fix.** 12s/record is still
well short of the observed ~23.1-23.2s/record cost on this host. A replay
under the new budget may still be incomplete — that is expected and will be
recorded honestly rather than declared resolved.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure`,
`fixture_insufficient_n` on both arms — primary metric never resolved). No
new stack layer for the training cycle itself; the decode-timeout repair is
tracked code delivered as its own commit, but is explicitly **not** claimed
as a proven executable unblock yet (replay-proof still pending).

## Next priorities

1. Replay the frozen `control`/`canvas` arm once more under the raised 12s
   budget (rank 1, confidence 0.9, `retry_measurement`). If still incomplete,
   the remaining gap is a genuine host CPU-throughput/fixture-cost mismatch —
   the next lever is a smaller `screening_smoke_n`, fewer training steps to
   free more eval wall, or accepting this fixture doesn't fit
   `MAX_RUN_MINUTES=3` on a CPU-only sandbox, not another blind timeout bump.
