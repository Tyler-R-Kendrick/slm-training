# E733 lexer root-identity reachability audit

**Date:** 2026-07-22  
**Decision:** invalidate the run; reject lexer root-identity configurations before artifacts  
**Evidence:** [`iter-e733-lexer-root-identity-reachability-20260722.json`](iter-e733-lexer-root-identity-reachability-20260722.json)

E733 tested a proposed lexer-native root-reference identity head on the same
141-record symbol-only corpus used by E731. The local CPU train completed all
140 steps in 78.98 seconds under `max_wall_minutes=2`. The auxiliary objective
was active during training, but that did not establish decode reachability.

The matched strict compiler-tree smoke arms expose the defect. Identity weights
0 and 1 are prediction- and metric-identical, and the enabled treatment records
zero identity applications. The compiler grammar exposes root bind references
in canonical declaration order, so there is no multi-identity branch for this
head to rank. This is an invalid capability declaration, not useful negative
model evidence. The checkpoint is rejected and must never be synced, promoted,
served, resumed, or used as a parent.

The final harness retains root-reference identity as a choice-tokenizer-only
capability. Both lexer identity loss and decode weights now fail during config
construction, before the run directory exists. Regression tests cover the
exact lexer + compiler-tree combination. Checkpoint loading also rejects a
serving/eval/resume checkpoint whose config enables a root head but whose state
omits that head; missing auxiliary heads remain allowed only for explicit
warm-start loading.

This closes the proposed lexer identity path. The next model arm must target a
real compiler branch exposed by the canonical grammar, with an integration test
that proves nonzero applications before spending a training cycle.
