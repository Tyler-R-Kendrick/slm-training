# Autotrain c4 (continuous-openui-20260803): dual-arm decode timeout, matches cross-loop precedent

**Verdict:** infrastructure failure, not scoreable, repair still required —
**not attempted here.** The fresh size-matched `component-edge` screening
pair (1,766,990 params) finalized all 3/3 smoke records inside the typed
24.0s decode timeout on **both** the control and candidate arms
(`decode_timeout_document_count=3` each). Both arms trained successfully;
only evaluation is affected.

This is the identical dual-arm (both matched arms, not just one) timeout
signature independently documented the same day in a different loop lineage:
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md)
and its
[follow-up](autotrain-cycle-c5-c6-replay-blocked-follow-up.md). That session
ruled out a defect in the per-record decode-timeout allocator, drafted then
**correctly reverted** an auto-retire routing change because it would have
violated the deliberate
`tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`
contract, and left root cause open between (a) seed-dependent decode
pathology and (b) sandbox CPU throughput vs. wall-budget headroom around
1.7-1.8M params.

This cycle adds one data point: the same symmetric failure reproduced in a
**different** loop, campaign, and param count (1,766,990 here vs. 1,755,760
there), with no shared seed. That weighs toward hypothesis (b) — sandbox
compute headroom at this size band — over a seed-specific artifact, though it
doesn't rule (a) out on its own.

Per the existing guardrail test and the prior session's explicit
recommendation, **this session did not attempt a speculative fix.** Writing
a throwaway `repair_harness` commit just to unblock automatic replay would
mask a real, still-unexplained infrastructure signal — exactly what the
guardrail test exists to prevent. Both checkpoints
(`eec4db4a...dc21f` control, `7ccbad86...398051` component-edge) are local,
explicit no-sync, never reusable, promoted, or ship.

Next: this campaign's frozen arm should **not** be auto-retried further
without the dedicated compiler-tree decode profiling both blocker docs call
for. This continuous-loop session is stopping new-cycle advancement on
`continuous-openui-20260803` here, having delivered one genuine harness
repair (AgentV preflight, [PR #1363](https://github.com/Tyler-R-Kendrick/slm-training/pull/1363))
and three fully-documented model cycles (c1-c3) beforehand.

Machine evidence:
[`continuous-openui-20260803-c4-dual-arm-decode-timeout.json`](continuous-openui-20260803-c4-dual-arm-decode-timeout.json).
