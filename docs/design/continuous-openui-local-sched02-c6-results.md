# Continuous autotrain: 2026-08-05 (scheduled session sched02) cycle 6 — frozen replay proves the recovery fix, reproduces the open decode-timeout blocker (screening, inconclusive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c6`
**Integration commit:** `a745dc64` (this session's `_recover_incomplete_handoff_feedback` fix,
[`autotrain-frozen-replay-model-build-area-recovery-fix-20260805.md`](autotrain-frozen-replay-model-build-area-recovery-fix-20260805.md))

**Verdict:** the harness fix works — the `retry_measurement` action queued by
cycle 2 was consumed cleanly this time (cycles c3/c4/c5 had hard-crashed on
`ValueError: latest hypothesis matrix has no terminal feedback` before the
fix landed). The replayed frozen `component-plan`/`control` pair itself
still fails, reproducing the same class of dual-arm decode timeout as
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md)
Blocker 1 — this is now the **third** independent seed
(`100005`, `100002` in cycle 2, `100002` again here) hitting the same
symptom class.

| Arm | Seed | Exit | Detail |
| --- | ---: | ---: | --- |
| component-plan | 100002 | 2 | `incomplete_document_n=3`, `decode_timeout_count=3`, `compiler_ms_mean=34679.2` |
| control | 100002 | 2 | `incomplete_document_n=3`, `decode_timeout_count=3` |

`climb_state=inconclusive`, `ship_state=blocked`, `measurement_complete=false`.

## New evidence for the open Blocker 1

`compiler_ms_mean=34679.2` (≈34.7s/record) on the component-plan arm vs. the
configured `screening_decode_timeout_seconds=12` in
`src/slm_training/resources/experiments/autotrain_climb/policy.v1.json`
(raised 8→12 on 2026-08-04, version-history v179, after an earlier session
measured `compiler_ms_mean ~23.1-23.2s/record` blowing the *then*-8s budget).
Compiler/decode cost on this sandbox now runs **~3x** the configured budget,
not the ~2x previously measured — consistent with the v179 history note's
own open caveat: "root cause (fixture compile+decode cost vs. nominal
per-record wall budget on CPU-only hosts) stays open." This session's
sandbox reports 4 vCPU / 15 GiB RAM (`nproc`, `free -h`), i.e. this is not a
resource-starved container; the compiler-search cost itself is the
bottleneck, not host contention.

## Driver's queued next action

The driver's `cycle_handoff.json` now requires, in order:

1. `repair_harness` (`owner=improve-openui-harnesses`,
   `harness_family=model_build`) — "AgentV finalized every record
   disposition and reported an internal decode timeout; repair canonical
   model-build runtime before replaying the frozen arm."
2. `retry_measurement` of the identical frozen pair (`frozen_manifest_sha256
   =fc7f74cc6b9176e0d0be5f33e2114de07de1aeba9ffe0b93b3e2413fc1427dfb`),
   gated behind (1).
3. `document`.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `harness_failure`, both arms).
No stack layer for this cycle's model result. The `_recover_incomplete_handoff_feedback`
harness fix itself already shipped as its own commit
(`a745dc6`) with a passing regression test — that fix is the positive,
reviewable delta from this session (see
[`autotrain-frozen-replay-model-build-area-recovery-fix-20260805.md`](autotrain-frozen-replay-model-build-area-recovery-fix-20260805.md)),
proven here by the fact that cycle 6 ran at all instead of hard-crashing a
4th consecutive time.

## Next priorities

1. **Do not** speculatively raise `screening_decode_timeout_seconds` again
   (8→12 already happened once and only partially mitigated a smaller gap).
   Route to a dedicated `improve-openui-harnesses` profiling session that
   measures where the ~34.7s/record compiler-search cost is actually spent
   (search width/backtrack/stagnation-patience knobs, MaskGIT decode steps,
   or true CPU decode throughput) before changing any budget or routing
   knob.
2. Until that profiling lands, do not keep re-issuing `retry_measurement`
   for this exact frozen pair — three independent seeds have now reproduced
   the same class of failure; further blind retries are not new information.
3. The `component-plan` quality hypothesis itself remains unconfirmed at
   fresh seed (still queued behind Blocker 1, as in
   [`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)).

Machine evidence: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c6/`
(gitignored; `sdlc_delivery.json`, `cycle_handoff.json` cited above).
