# Autotrain c4 (continuous-openui-local, session 60a9ge): decode timeout, not a model result

**Verdict:** infrastructure failure, not scoreable. Both the `control` and
`component-edge` `wf_smoke_v2` arms trained to completion (both produced
checkpoints, 1,766,990 trainable params each) but their honest-ship-gate
evaluation hit `decode_timeout_count=3` (all 3 smoke documents timed out per
arm) — `compiler_ms_mean` was 23.2–23.8 seconds per document, well past the
`decode_timeout_seconds=8.0` smoke default.

Both arms time out identically (control: 23202.2ms mean; component-edge:
23812.8ms mean), so this is **not** attributable to the `component-edge`
knob under test — it is a CPU-throughput/wall-budget mismatch for this
session's sandbox against the `wf_smoke_v2` / `e938_role_safe_all_targets_v2`
recipe's default timeout (tuned for a faster host). AgentV finalized every
record disposition; there were no execution errors, only timeouts.

No code change was required: training itself succeeded, and only the
decode-timing side of evaluation was affected. Per the loop's soft-failure
rule, a timeout never stops the loop — the correct response is to replay the
identical frozen arm, and if the timeout recurs consistently for this
recipe/host pairing, raise `decode_timeout_seconds` as a knob (not a harness
code repair) in a future cycle.

Checkpoints exist for both arms but produced no scoreboard, so neither is
promotable, ship-eligible, or reusable as model evidence. Lean is
`not_applicable:screening`.

## SDLC Phase A

**Non-positive** (`harness_failure` / `measurement_incomplete` on both arms;
`primary_metric_unavailable`). Per `sdlc` autotrain-iteration-delivery, no
stacked PR layer is opened for this cycle — local commit and docs only.
`checkpoint_documentation_required=true` is satisfied by the roster row in
[`docs/MODEL_CARD.md`](../MODEL_CARD.md) (never promote/sync/ship).

Next: replay the identical frozen `component-edge` arm (`retry_measurement`).

Machine evidence:
[`continuous-openui-local-60a9ge-c4-results.json`](continuous-openui-local-60a9ge-c4-results.json).
