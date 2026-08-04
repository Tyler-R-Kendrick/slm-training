# Continuous autotrain: 2026-08-04 (session gd6j83) cycle 2 — symmetric dual-arm decode timeout (infra finding, not a fix)

**Verdict:** infrastructure failure, not scoreable. Both the `control` and
`component-plan` `wf_smoke_v2` arms (1,755,764 params, seed 100002) finalized
every record's disposition inside `evaluate_model.py --ship-gates`, but AgentV
reported a decode timeout on **all 3/3** smoke records on **both** arms
(`incomplete_document_n=3`, `completed_document_n=0`, effective decode timeout
24.0s/doc against a configured `decode_timeout_seconds=8.0`,
`evaluation_wall_seconds≈52.8s`). No smoke metrics, no scoreboard result, no
ship-gate outcome exists for either arm — this is not evidence about the
`component-plan` model hypothesis.

`compiler_ms_mean` jumped from ~3,900-4,300ms (cycle 1, seed 100001, same
`wf_smoke_v2` control recipe) to ~23,100ms (cycle 2, seed 100002) on **both**
arms — a ~5-6x slowdown that is symmetric across control and candidate, so it
is not attributable to the `component-plan` knob change itself.

## Same failure class as a previously documented, still-unresolved blocker

This matches the seed-dependent compiler-tree decode pathology already
recorded and explicitly left open in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)
/
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md):
a size-matched pair decoded quickly at one seed and then both arms timed out
symmetrically at a different seed, with root cause (seed-dependent compiler
search blowup vs. sandbox CPU/wall-budget headroom) undetermined. That doc is
explicit that the previously-drafted "auto-retire on symmetric dual-arm
timeout" routing fix was correctly reverted (it violates
`tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`),
so this cycle does not attempt a similar shortcut and does not keep
re-attempting the same seed automatically.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `harness_failure` on both arms;
`primary_metric_unavailable`). Per `sdlc` autotrain-iteration-delivery, no
stack layer opens for this cycle.

## Next steps (routed, not attempted here)

The driver's typed handoff names a `repair_harness` action against harness
family `model_build` for frozen manifest
`becbf08df082ca96a0f5b686cbd81d21130ca5be60637b95ac8601945f2adf7e`, followed by
a queued `retry_measurement` on the identical frozen arm once repaired. Per
the loop law (`references/continuous.md` rule 3), this repair routes through
`improve-openui-harnesses` rather than a speculative knob change in the
continuous driver: profile `strict_compiler_tree` decode under the seed-100002
fixture records, determine why compiler search cost jumps ~5-6x symmetrically
across arms, and either fix the underlying compiler-search cost or raise
`decode_timeout_seconds` with evidence that the extra wall time is legitimate
compile cost rather than a runaway search. Add a regression test before
replaying the frozen arm.

Machine evidence:
[`continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.json`](continuous-openui-local-gd6j83-c2-dual-arm-decode-timeout.json).
