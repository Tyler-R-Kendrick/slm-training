# Continuous autotrain: 2026-08-04 (session io35qa) cycle 4 — dual-arm decode timeout recurs on a new hypothesis (Blocker 1, further evidence)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `136699ab` (this session's cycle-3 docs commit, on top
of `main` tip `eba6db30`)

## What happened

Both arms (`c20260804-continuous-openui-local-8c0b60dd-c4-component-edge` and
`...-control`) trained to completion but AgentV's ship-gate evaluation
finalized every record inside a typed decode timeout
(`decode_timeout_count=3`, `smoke:incomplete_document_n=3`) —
`measurement_incomplete` on both arms, `primary_metric_unavailable`.

This is the **same failure class** as the already-documented, still-open
**Blocker 1** in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md) /
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md) —
but on a **different hypothesis pair** (`component-edge` vs `control`, this
session's cycle 4) than the original `-confirm` arm at seed `100005`. Both
occurrences are dual-arm (both control and candidate time out together),
which is further evidence toward the CPU / wall-budget headroom explanation
those docs left undetermined, rather than something specific to one seed or
one hypothesis's decode graph.

## Per prior explicit guidance: no speculative fix attempted here

The c5/c6 follow-up doc is explicit: *"Do not keep re-attempting
automatically"* and *"needs dedicated `improve-openui-harnesses` attention in
a dedicated session with room to profile compiler-tree decode."* This cycle
does not attempt a routing or timeout-threshold change (the c5/c6 doc also
records that an auto-retire-on-symmetric-timeout fix was tried once and
correctly reverted for violating
`test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`).
This cycle only adds a second, independent data point that the pattern is
not confined to seed `100005`.

## SDLC Phase A

**Non-positive / infrastructure block for this arm.** `repair_harness`
action acknowledged **blocked** (not completed) — the real blocker is the
undetermined CPU/wall-budget decode-timeout root cause already tracked
above; this is not new information warranting a fresh repair attempt inside
a general continuous cycle. Per the loop's repeated-blocker rule, this is
this session's first occurrence, not three consecutive identical failures,
so the loop continues with a different, non-decode-timeout-affected
hypothesis next rather than stopping.

## New diagnostic: this occurrence is budget-bound, not (only) flaky

`_fit_screening_decode_timeout_seconds`
(`scripts/run_autotrain_continuous.py:261`) *clamps down* the configured
screening decode timeout to fit `MAX_RUN_MINUTES` (`= 3` in
`src/slm_training/levers.py`); it never raises the timeout for a
slower-to-decode hypothesis. This cycle's own telemetry shows
`component-edge`'s `compiler_ms_mean` at **~23.3-23.4s/record** — roughly
**3x** `component-plan`'s ~8.7-9.0s in cycle 3, and well past the ~8s clamp
this wall budget derives. The function's own docstring already names the
fix category: *"if this clamp always binds, either the arm share model or
the thrash recipe (steps/n) needs recalibration — not silent wall++."*
Raising the timeout would just mask the real signal (this hypothesis is
structurally more expensive to decode) and was correctly not attempted here.

Recalibrating the arm share model / thrash recipe for decode-heavy
hypotheses like `component-edge` is a real, scoped harness change, but one
that needs its own investigation into fairness across arms (a blanket
timeout bump would silently advantage slow arms over the `MAX_RUN_MINUTES`
cap) plus a regression test — out of scope to improvise inside a general
continuous cycle, consistent with why this blocker (in its seed-`100005`
form) was left for a dedicated session twice already.

## Next priorities

1. Route a **dedicated** `improve-openui-harnesses` session to recalibrate
   the screening arm share model / thrash recipe for decode-heavy hypotheses
   (`component-edge` measured here; profile the seed-`100005` `-confirm` arm
   too) so per-arm decode budget scales with measured compile cost instead of
   a single fixed clamp — with a regression test pinning the new allocation.
2. Do not retry the identical `component-edge`/`control` frozen arm pair
   speculatively, and do not raise `--decode-timeout-seconds` as a point fix;
   continue with a fresh hypothesis this loop once the predecessor gate is
   repaired.

Machine evidence:
[`continuous-openui-local-io35qa-c4-results.json`](continuous-openui-local-io35qa-c4-results.json).
