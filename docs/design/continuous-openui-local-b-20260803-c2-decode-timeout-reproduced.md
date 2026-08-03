# Autotrain continuous-openui-local-b c2: decode timeout reproduced independently

**Status:** blocked on `repair_harness`, same as
[`continuous-openui-local` c2](continuous-openui-local-20260803-c2-decode-timeout-budget-diagnosed.md).
Confirms the blocker is reproducible, not a one-off fluke of a single loop
lineage.

## What happened

An independent loop lineage (`continuous-openui-local-b`, no shared campaign
state with `continuous-openui-local`) hit the **identical** symmetric decode
timeout at its own cycle index 2: both `control` and `component-plan` arms
recorded `decode_timeout_count=3` (`completed_document_n=0/3` each) under the
governed `screening_decode_timeout_seconds=24`. Cycle index 1 in both
lineages (`continuous-openui-local` and `continuous-openui-local-b`) used
seed `100001` and always completed cleanly; cycle index 2 in both lineages
used seed `100002` and always timed out symmetrically. This is now
reproduced **twice, independently**, tying the timeout to cycle-index-derived
seed `100002` for this recipe (`wf_smoke_v2`, 20 steps, 1.6-1.76M params,
`strict_compiler_tree` policy) rather than to any one lineage's history or to
the `component-plan` hypothesis specifically (the plain control times out
identically).

Combined with the budget-sweep diagnostic already on record (24s -> 90s
resolves the same seed's records to 3/3 complete;
[`continuous-openui-local-20260803-c2-decode-timeout-budget-diagnosed.md`](continuous-openui-local-20260803-c2-decode-timeout-budget-diagnosed.md)),
this cycle does not add a new root cause, only independent reproduction. Per
that doc's recommendation, this is not resurrected as a same-cycle policy
edit; it needs the dedicated preregistered meta-campaign for
`screening_decode_timeout_seconds` referenced there.

## Decision

Stop spawning additional fresh-lineage cycles against this recipe this
session: two independent lineages both stall at the identical cycle-2 seed,
so a third lineage would reproduce the same wall without new information —
exactly the repeated-hard-blocker condition that the continuous-mode loop law
uses to stop and report rather than keep retrying blindly.
`repair_harness` is acked `blocked` with this commit as evidence.

Machine evidence:
[`continuous-openui-local-b-20260803-c2-decode-timeout-reproduced.json`](continuous-openui-local-b-20260803-c2-decode-timeout-reproduced.json).
