# Continuous autotrain: 2026-08-04 (session babec8) cycle 2 — decode timeout, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `d7f494a0` (branch `claude/great-dirac-babec8`, scheduled
`autotrain` loop task)

**Verdict:** Both the size-matched `control` and the `component-plan`
quality-hypothesis candidate (the rank-1 priority carried over from cycle 1)
reproduce **byte-identical training checkpoints** to session `j48f8u`'s
confirmed `component-plan` structural-similarity win, but eval hit an
internal AgentV decode timeout on all 3 smoke records for both arms
(`compiler_ms_mean` ~23.2s/record). **No quality metric is available for
either arm** — the comparison is inconclusive, not a model result.
Infrastructure/measurement signal, not a model regression.

| Arm | Params | Seed | Checkpoint SHA | Exit | decode_timeout_count | incomplete_document_n |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| control | 1,755,764 | 100002 | `6abf57d4...db3512b` | 2 | 3 | 3 |
| component-plan | 1,755,764 | 100002 | `20e573b1...f0f8a8e741` | 2 | 3 | 3 |

## Diagnosis

This is at least the third independent session (after `j48f8u`'s confirmed
win and a `krjpdg`-session-adjacent occurrence documented on PR #1401) to hit
this exact decode timeout on this exact 1,755,764-param control/component-plan
pair, always the same seed and the same checkpoint SHAs. The driver's typed
handoff routes this to a `repair_harness` action against the `model_build`
harness family (frozen manifest
`ca1716f1ea70539f26f175efd34f90a36f9833f8c1aae166cf123012091c2ce3`), paired
with a `retry_measurement` action to replay the identical frozen arms once
repaired.

[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)
already investigated a related dual-arm decode timeout and explicitly found
that a drafted auto-retire-on-symmetric-timeout routing fix was **correctly
reverted** for violating a deliberate test contract
(`tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`),
and recommended a **dedicated** `improve-openui-harnesses` session with room
to profile compiler-tree decode cost before attempting any further fix. This
session follows that recommendation: it does **not** attempt a speculative
repair. The `repair_harness` action is left unacknowledged/deferred, which
per the driver's receipt contract blocks this session's automatic cycle 3 —
so this is where this session's autotrain loop stops for now, not a rule
violation.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `primary_metric_unavailable`).
Per `sdlc` autotrain-iteration-delivery, no stacked PR layer is opened for
this cycle — local commit and docs only (this doc still ships via a PR per
this repo's established practice of landing required documentation, per the
repo-wide GitHub-integration instruction to always open a PR for a pushed
branch).

## Next priorities (ranked)

1. **New, this session:** a dedicated `improve-openui-harnesses` profiling
   session to calibrate `_effective_record_decode_timeout`'s fair-share
   budget (`src/slm_training/harnesses/model_build/eval_runner.py:1187`) —
   or the upstream `evaluation_wall_seconds` allocation under the
   `MAX_RUN_MINUTES` hard cap — for ~1.7-1.8M-param compiler-tree decode on
   this container class, since this exact timeout has now reproduced across
   at least 3 independent sessions (confidence 0.85).
2. Once repaired, replay the exact frozen `control`/`component-plan` pair
   (seed 100002, 1,755,764 params, frozen manifest `ca1716f1...2091c2ce3`)
   before testing any new hypothesis (confidence 0.95).

Machine evidence:
[`continuous-openui-local-babec8-c2-results.json`](continuous-openui-local-babec8-c2-results.json).
